//! tokenizer.rs — gpt2 BPE + `smollm` pre-tokenizer for the pinned SmolLM2 row.
//!
//! Tokenizer runtime owner is `faber-runtime` (need `30905943`; decision record (e)).
//! This module implements EXACT parity with the pinned llama.cpp 10150
//! (`dee2a846b`) comparator for the sole admitted row (SmolLM2-360M-Instruct
//! Q4_K_M), reproducing `evidence/contract-tokenize-probes.txt` probes P1–P11 and
//! the four workload prompt token-id lists (`gi0-workloads.md` §3).
//!
//! ## Algorithm (mirrors llama.cpp `dee2a846b` exactly)
//!
//! `src/llama-vocab.cpp` (`llm_tokenizer_bpe` + `llm_tokenizer_bpe_session`) and
//! `src/unicode.cpp` (`unicode_regex_split` + `unicode_regex_split_custom_gpt2`)
//! are the reference. The pipeline per `encode(text, parse_special)`:
//!
//! 1. **Special-token partition** (only when `parse_special`): split the raw text
//!    on the special-token texts (byte-level `find`, longest-text-first),
//!    producing `Token(id)` / raw-text fragments. Control and unknown specials are
//!    *not* partitioned when `parse_special == false`; user-defined ones still are.
//!    The pinned row's 17 specials are all control, so `parse_special == false`
//!    means no partition at all.
//! 2. **`smollm` pre-tokenizer**: two sequential regex passes over codepoints —
//!    `\p{N}` (each number codepoint becomes its own fragment) then the GPT2
//!    custom splitter `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
//!    ?[^\s\p{L}\p{N}]+|\s+(?!\S)` (implemented directly over the codepoint flag
//!    table below, identical to `unicode_regex_split_custom_gpt2`).
//! 3. **GPT2 byte encoding**: every byte of every fragment maps through the
//!    byte-to-display table (`unicode_byte_to_utf8_map`); space becomes `Ġ`
//!    (U+0120), newline becomes `Ċ` (U+010A), etc.
//! 4. **BPE merge loop** (per fragment): symbols start as the fragment's UTF-8
//!    chars; the bigram with the smallest merge rank (ties → smallest left index)
//!    is merged first; ranks come from `tokenizer.ggml.merges` in order. A symbol
//!    whose display text is not a vocab token falls back to per-raw-byte token
//!    lookups exactly like llama.cpp.
//!
//! ## Fail-closed posture (GI1-3 outcome)
//!
//! Construction rejects: `model != "gpt2"`, `pre != "smollm"`, vocab count ≠ 49152,
//! merges count ≠ 48900, token-type count ≠ vocab count, `scores` present,
//! malformed merge strings, byte-boundary violations (literal spaces in byte-level
//! tokens), empty or oversized token strings, special ids outside `[0, 49152)`,
//! `add_bos_token == true`, `add_space_prefix == true`. Explicit ceilings bound
//! token/merge string bytes, the special cache, the encode input, and the encode
//! output; offset/length arithmetic is checked.
//!
//! The row boundary holds: this tokenizer is the pinned gpt2/smollm row only
//! (decision (b); CTO report `2f90eafd` §A.1). BOS never auto-prepends
//! (`add_bos_token = false`); BOS/EOS enter sequences only through `<|im_start|>` /
//! `<|im_end|>` template text or explicit special tokens.
//!
//! ## Data provenance
//!
//! - Vocab (49152 tokens), token types, merges (48900): taken from the GI1-1
//!   admission (`crate::gguf::GgufAdmission`) — never re-discovered.
//! - Codepoint flags (`CPT_RANGES`, `WHITESPACE_SET`): generated from llama.cpp
//!   10150 `src/unicode-data.cpp` (`unicode_ranges_flags` +
//!   `unicode_set_whitespace`) — the exact data the pinned comparator executes.
//! - `BYTE_TO_DISPLAY`: GPT2 byte-to-unicode table (`unicode_byte_to_utf8_map`).

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use crate::gguf::{GgufAdmission, MetadataValue};

// ---------------------------------------------------------------------------
// Pinned-row constants (gi0-model-contract v1.0.0 §4/§5/§9; §10 intake checklist)
// ---------------------------------------------------------------------------

/// `tokenizer.ggml.model` for the pinned row.
pub const TOKENIZER_MODEL: &str = "gpt2";
/// `tokenizer.ggml.pre` for the pinned row.
pub const TOKENIZER_PRE: &str = "smollm";
/// `tokenizer.ggml.tokens` count.
pub const EXPECTED_VOCAB_SIZE: usize = 49_152;
/// `tokenizer.ggml.merges` count.
pub const EXPECTED_MERGES: usize = 48_900;
/// `tokenizer.ggml.bos_token_id`.
pub const BOS_TOKEN_ID: u32 = 1;
/// `tokenizer.ggml.eos_token_id` (== PAD).
pub const EOS_TOKEN_ID: u32 = 2;
/// `tokenizer.ggml.padding_token_id` (== EOS).
pub const PAD_TOKEN_ID: u32 = 2;
/// `tokenizer.ggml.unknown_token_id`.
pub const UNK_TOKEN_ID: u32 = 0;
/// `tokenizer.ggml.add_bos_token`.
pub const ADD_BOS_TOKEN: bool = false;
/// `tokenizer.ggml.add_space_prefix`.
pub const ADD_SPACE_PREFIX: bool = false;

// ---- Explicit ceilings (GI1-3 outcome) ----

/// Per-token text byte ceiling (mirrors `gguf::MAX_STRING_BYTES`).
pub const MAX_TOKEN_STRING_BYTES: usize = 4096;
/// Per-merge string byte ceiling (two token texts + separator).
pub const MAX_MERGE_STRING_BYTES: usize = 2 * MAX_TOKEN_STRING_BYTES + 1;
/// Special-token cache ceiling (the pinned row has 17).
pub const MAX_SPECIAL_TOKENS: usize = 1024;
/// Encode input byte ceiling.
pub const MAX_ENCODE_INPUT_BYTES: usize = 1 << 20;
/// Encode output token ceiling.
pub const MAX_ENCODE_OUTPUT_TOKENS: usize = 1 << 20;

// ---- tokenizer.ggml.token_type values (llama.cpp `llama_token_type`, llama.h) ----

pub(crate) const TOKEN_TYPE_UNDEFINED: i32 = 0;
pub(crate) const TOKEN_TYPE_NORMAL: i32 = 1;
pub(crate) const TOKEN_TYPE_UNKNOWN: i32 = 2;
pub(crate) const TOKEN_TYPE_CONTROL: i32 = 3;
pub(crate) const TOKEN_TYPE_USER_DEFINED: i32 = 4;
pub(crate) const TOKEN_TYPE_UNUSED: i32 = 5;
pub(crate) const TOKEN_TYPE_BYTE: i32 = 6;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Typed, fail-closed tokenizer errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// `tokenizer.ggml.model` is not `gpt2`.
    UnsupportedModel { model: String },
    /// `tokenizer.ggml.pre` is not `smollm`.
    UnsupportedPreTokenizer { pre: String },
    /// `tokenizer.ggml.tokens` count differs from the pinned 49152.
    VocabSizeMismatch { expected: usize, actual: usize },
    /// `tokenizer.ggml.token_type` count differs from the vocab count.
    TokenTypeCountMismatch { expected: usize, actual: usize },
    /// `tokenizer.ggml.merges` count differs from the pinned 48900.
    MergesCountMismatch { expected: usize, actual: usize },
    /// `tokenizer.ggml.scores` is present; the pinned row carries none.
    ScoresPresent,
    /// A merge string has no `" "` separator at position ≥ 1 or an empty side.
    MalformedMergeString { index: usize, merge: String },
    /// A token text violates the byte-level BPE boundary: empty, contains a
    /// literal space, or exceeds the string ceiling.
    ByteBoundaryViolation { token_id: usize, token: String },
    /// A token id from the facts or the vocab is outside `[0, vocab_size)`.
    TokenIdOutOfRange { id: u32, vocab_size: usize },
    /// `add_bos_token` contradicts the pinned row (`false`).
    AddBosTokenNotSupported { value: bool },
    /// `add_space_prefix` contradicts the pinned row (`false`).
    AddSpacePrefixNotSupported { value: bool },
    /// Encode input exceeds `MAX_ENCODE_INPUT_BYTES`.
    InputTooLarge { bytes: usize, ceiling: usize },
    /// Encode output would exceed `MAX_ENCODE_OUTPUT_TOKENS`.
    OutputTooLarge { tokens: usize, ceiling: usize },
    /// Checked offset/length arithmetic overflowed.
    ArithmeticOverflow { what: &'static str },
    /// A required tokenizer KV is missing from the admission.
    MissingMetadataKey { key: &'static str },
    /// A tokenizer KV has an unexpected type.
    WrongMetadataValueType { key: &'static str },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizerError::UnsupportedModel { model } => {
                write!(f, "tokenizer model {model:?} is not the pinned \"gpt2\"")
            }
            TokenizerError::UnsupportedPreTokenizer { pre } => {
                write!(f, "tokenizer pre {pre:?} is not the pinned \"smollm\"")
            }
            TokenizerError::VocabSizeMismatch { expected, actual } => {
                write!(f, "vocab count {actual} != expected {expected}")
            }
            TokenizerError::TokenTypeCountMismatch { expected, actual } => {
                write!(f, "token_type count {actual} != vocab count {expected}")
            }
            TokenizerError::MergesCountMismatch { expected, actual } => {
                write!(f, "merges count {actual} != expected {expected}")
            }
            TokenizerError::ScoresPresent => {
                write!(f, "tokenizer.ggml.scores is present; the pinned row has no scores")
            }
            TokenizerError::MalformedMergeString { index, merge } => {
                write!(f, "malformed merge string at index {index}: {merge:?}")
            }
            TokenizerError::ByteBoundaryViolation { token_id, token } => {
                write!(
                    f,
                    "byte-boundary violation at token {token_id}: {token:?} (empty, literal space, or overlong)"
                )
            }
            TokenizerError::TokenIdOutOfRange { id, vocab_size } => {
                write!(f, "token id {id} out of [0, {vocab_size})")
            }
            TokenizerError::AddBosTokenNotSupported { value } => {
                write!(f, "add_bos_token = {value}; the pinned row is add_bos_token = false")
            }
            TokenizerError::AddSpacePrefixNotSupported { value } => {
                write!(f, "add_space_prefix = {value}; the pinned row is add_space_prefix = false")
            }
            TokenizerError::InputTooLarge { bytes, ceiling } => {
                write!(f, "encode input {bytes} bytes exceeds ceiling {ceiling}")
            }
            TokenizerError::OutputTooLarge { tokens, ceiling } => {
                write!(f, "encode output {tokens} tokens exceeds ceiling {ceiling}")
            }
            TokenizerError::ArithmeticOverflow { what } => {
                write!(f, "checked arithmetic overflow in {what}")
            }
            TokenizerError::MissingMetadataKey { key } => {
                write!(f, "admission missing tokenizer metadata key {key}")
            }
            TokenizerError::WrongMetadataValueType { key } => {
                write!(f, "admission tokenizer metadata key {key} has the wrong value type")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

// ---------------------------------------------------------------------------
// Facts + tokenizer
// ---------------------------------------------------------------------------

/// The frozen tokenizer fact set from the GI1-1 admission
/// (gi0-model-contract v1.0.0 §4/§5/§9, fields 27–38).
#[derive(Debug, Clone)]
pub struct TokenizerFacts {
    /// `tokenizer.ggml.model` — must be `gpt2`.
    pub model: String,
    /// `tokenizer.ggml.pre` — must be `smollm`.
    pub pre: String,
    /// `tokenizer.ggml.tokens` (byte-level display strings), length 49152.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.token_type`, length 49152 (llama.cpp `llama_token_type`).
    pub token_types: Vec<i32>,
    /// `tokenizer.ggml.merges`, length 48900.
    pub merges: Vec<String>,
    /// `tokenizer.ggml.scores` present (must be false for the pinned row).
    pub scores_present: bool,
    /// `tokenizer.ggml.bos_token_id`.
    pub bos_token_id: u32,
    /// `tokenizer.ggml.eos_token_id`.
    pub eos_token_id: u32,
    /// `tokenizer.ggml.padding_token_id`.
    pub pad_token_id: u32,
    /// `tokenizer.ggml.unknown_token_id`.
    pub unk_token_id: u32,
    /// `tokenizer.ggml.add_bos_token` (must be false).
    pub add_bos_token: bool,
    /// `tokenizer.ggml.add_space_prefix` (must be false).
    pub add_space_prefix: bool,
}

/// Which special-cache bucket a special token belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpecialKind {
    Control,
    Unknown,
    UserDefined,
}

/// One cached special token: text, id, and partition kind.
#[derive(Debug, Clone)]
struct Special {
    text: String,
    id: u32,
    kind: SpecialKind,
}

/// gpt2 BPE + `smollm` pre-tokenizer for the pinned row.
#[derive(Debug, Clone)]
pub struct Gpt2BpeTokenizer {
    vocab_size: usize,
    /// token text (display bytes) → id.
    token_to_id: HashMap<Box<[u8]>, u32>,
    /// BPE merge pair → rank (index in `tokenizer.ggml.merges`; first wins).
    bpe_ranks: HashMap<(Box<[u8]>, Box<[u8]>), u32>,
    /// Specials sorted by text byte length descending (llama.cpp cache order).
    specials: Vec<Special>,
    /// raw-byte fallback token (llama.cpp `text_to_token(std::string(1, byte))`).
    byte_fallback: [u32; 256],
    bos: u32,
    eos: u32,
    pad: u32,
    unk: u32,
    add_bos: bool,
    add_space_prefix: bool,
    /// End-of-generation ids (llama.cpp `special_eog_ids` text rule).
    eog: Vec<u32>,
}

impl Gpt2BpeTokenizer {
    /// Construct the tokenizer from the frozen facts, fail-closed.
    ///
    /// Every contract fact is validated here (model/pre names, exact counts,
    /// scores absence, merge well-formedness, byte-level token boundaries,
    /// id ranges, `add_bos`/`add_space_prefix` = false) before any allocation
    /// sized by an input count.
    ///
    /// # Errors
    ///
    /// Returns the first typed `TokenizerError` that the facts contradict.
    pub fn new(facts: TokenizerFacts) -> Result<Self, TokenizerError> {
        if facts.model != TOKENIZER_MODEL {
            return Err(TokenizerError::UnsupportedModel { model: facts.model });
        }
        if facts.pre != TOKENIZER_PRE {
            return Err(TokenizerError::UnsupportedPreTokenizer { pre: facts.pre });
        }
        if facts.tokens.len() != EXPECTED_VOCAB_SIZE {
            return Err(TokenizerError::VocabSizeMismatch {
                expected: EXPECTED_VOCAB_SIZE,
                actual: facts.tokens.len(),
            });
        }
        if facts.token_types.len() != facts.tokens.len() {
            return Err(TokenizerError::TokenTypeCountMismatch {
                expected: facts.tokens.len(),
                actual: facts.token_types.len(),
            });
        }
        if facts.merges.len() != EXPECTED_MERGES {
            return Err(TokenizerError::MergesCountMismatch {
                expected: EXPECTED_MERGES,
                actual: facts.merges.len(),
            });
        }
        if facts.scores_present {
            return Err(TokenizerError::ScoresPresent);
        }
        if facts.add_bos_token != ADD_BOS_TOKEN {
            return Err(TokenizerError::AddBosTokenNotSupported {
                value: facts.add_bos_token,
            });
        }
        if facts.add_space_prefix != ADD_SPACE_PREFIX {
            return Err(TokenizerError::AddSpacePrefixNotSupported {
                value: facts.add_space_prefix,
            });
        }

        let vocab_size = facts.tokens.len();

        // 1. Token map + boundaries. Token id == array index, so every id is
        //    in [0, vocab_size) by construction; the facts' special ids are
        //    range-checked below.
        let mut token_to_id = HashMap::with_capacity(vocab_size);
        let mut byte_fallback = [u32::MAX; 256];
        for (id, text) in facts.tokens.iter().enumerate() {
            let bytes = text.as_bytes();
            if bytes.is_empty() || bytes.len() > MAX_TOKEN_STRING_BYTES || bytes.contains(&b' ') {
                return Err(TokenizerError::ByteBoundaryViolation {
                    token_id: id,
                    token: text.clone(),
                });
            }
            token_to_id.entry(bytes.to_vec().into_boxed_slice()).or_insert(id as u32);
            // llama.cpp's per-byte fallback looks up the single raw byte as a
            // string; only ASCII token texts can be a single (valid UTF-8) byte.
            if bytes.len() == 1 && bytes[0] < 0x80 {
                byte_fallback[bytes[0] as usize] = id as u32;
            }
        }

        // 2. Merge ranks — parse `word.find(' ', 1)`; first occurrence wins
        //    (llama.cpp `bpe_ranks.emplace`).
        let mut bpe_ranks = HashMap::with_capacity(facts.merges.len());
        for (index, merge) in facts.merges.iter().enumerate() {
            let bytes = merge.as_bytes();
            if bytes.len() > MAX_MERGE_STRING_BYTES {
                return Err(TokenizerError::MalformedMergeString {
                    index,
                    merge: merge.clone(),
                });
            }
            let Some(sep) = bytes.iter().position(|&b| b == b' ').filter(|&p| p >= 1) else {
                return Err(TokenizerError::MalformedMergeString {
                    index,
                    merge: merge.clone(),
                });
            };
            let left = &bytes[..sep];
            let right = &bytes[sep + 1..];
            if left.is_empty() || right.is_empty() {
                return Err(TokenizerError::MalformedMergeString {
                    index,
                    merge: merge.clone(),
                });
            }
            bpe_ranks
                .entry((left.to_vec().into_boxed_slice(), right.to_vec().into_boxed_slice()))
                .or_insert(index as u32);
        }

        // 3. Special ids range check.
        for id in [
            facts.bos_token_id,
            facts.eos_token_id,
            facts.pad_token_id,
            facts.unk_token_id,
        ] {
            if (id as usize) >= vocab_size {
                return Err(TokenizerError::TokenIdOutOfRange { id, vocab_size });
            }
        }

        // 4. Special cache (llama.cpp `cache_special_tokens`): control +
        //    user-defined + unknown, sorted by text byte length descending.
        let mut specials: Vec<Special> = Vec::new();
        for (id, (text, ttype)) in facts.tokens.iter().zip(facts.token_types.iter()).enumerate() {
            let kind = match ttype {
                &TOKEN_TYPE_CONTROL => SpecialKind::Control,
                &TOKEN_TYPE_UNKNOWN => SpecialKind::Unknown,
                &TOKEN_TYPE_USER_DEFINED => SpecialKind::UserDefined,
                // All remaining llama.cpp token types are never special-cached.
                &TOKEN_TYPE_NORMAL
                | &TOKEN_TYPE_UNDEFINED
                | &TOKEN_TYPE_UNUSED
                | &TOKEN_TYPE_BYTE
                | _ => continue,
            };
            specials.push(Special {
                text: text.clone(),
                id: id as u32,
                kind,
            });
        }
        if specials.len() > MAX_SPECIAL_TOKENS {
            return Err(TokenizerError::ArithmeticOverflow {
                what: "special token cache ceiling",
            });
        }
        specials.sort_by(|a, b| {
            b.text
                .len()
                .cmp(&a.text.len())
                .then_with(|| a.id.cmp(&b.id))
        });

        // 5. EOG ids (llama.cpp `special_eog_ids` text rule). FIM ids are NULL
        //    for the pinned row; eos/eot/eom ids are appended defensively.
        let eog_list: &[&str] = &[
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|flush|>",
            "<|calls|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "</s>",
            "<|eom_id|>",
            "<EOT>",
            "_<EOT>",
            "[EOT]",
            "[EOS]",
            "<|end_of_text|>",
            "<end_of_utterance>",
            "<eos>",
            "<turn|>",
            "<|tool_response>",
            "<｜end▁of▁sentence｜>",
            "[e~[",
        ];
        let mut eog = Vec::new();
        for (text, id) in facts.tokens.iter().zip(0..vocab_size as u32) {
            if eog_list.contains(&text.as_str()) {
                eog.push(id);
            }
        }
        // EOS enters the EOG set unconditionally (llama.cpp sanity check); EOT
        // and EOM ids are NULL for the pinned row, and BOS is never EOG.
        if !eog.contains(&facts.eos_token_id) {
            eog.push(facts.eos_token_id);
        }

        Ok(Self {
            vocab_size,
            token_to_id,
            bpe_ranks,
            specials,
            byte_fallback,
            bos: facts.bos_token_id,
            eos: facts.eos_token_id,
            pad: facts.pad_token_id,
            unk: facts.unk_token_id,
            add_bos: facts.add_bos_token,
            add_space_prefix: facts.add_space_prefix,
            eog,
        })
    }

    /// Intake the tokenizer facts from the GI1-1 admission (the bounded input;
    /// the admission has already verified the 40-field contract).
    ///
    /// # Errors
    ///
    /// Returns a typed `TokenizerError` if any tokenizer KV is missing, has the
    /// wrong type, or contradicts the pinned row.
    pub fn from_admission(admission: &GgufAdmission) -> Result<Self, TokenizerError> {
        let mut model = None;
        let mut pre = None;
        let mut tokens = None;
        let mut token_types = None;
        let mut merges = None;
        let mut scores_present = false;
        let mut bos = None;
        let mut eos = None;
        let mut pad = None;
        let mut unk = None;
        let mut add_bos = None;
        let mut add_space_prefix = None;

        for kv in &admission.metadata {
            match kv.key.as_str() {
                "tokenizer.ggml.model" => model = Some(kv.value.clone()),
                "tokenizer.ggml.pre" => pre = Some(kv.value.clone()),
                "tokenizer.ggml.tokens" => tokens = Some(kv.value.clone()),
                "tokenizer.ggml.token_type" => token_types = Some(kv.value.clone()),
                "tokenizer.ggml.merges" => merges = Some(kv.value.clone()),
                "tokenizer.ggml.scores" => scores_present = true,
                "tokenizer.ggml.bos_token_id" => bos = Some(kv.value.clone()),
                "tokenizer.ggml.eos_token_id" => eos = Some(kv.value.clone()),
                "tokenizer.ggml.padding_token_id" => pad = Some(kv.value.clone()),
                "tokenizer.ggml.unknown_token_id" => unk = Some(kv.value.clone()),
                "tokenizer.ggml.add_bos_token" => add_bos = Some(kv.value.clone()),
                "tokenizer.ggml.add_space_prefix" => add_space_prefix = Some(kv.value.clone()),
                _ => {}
            }
        }

        let str_of = |v: &MetadataValue, key: &'static str| -> Result<String, TokenizerError> {
            match v {
                MetadataValue::String(s) => Ok(s.clone()),
                _ => Err(TokenizerError::WrongMetadataValueType { key }),
            }
        };
        let u32_of = |v: &MetadataValue, key: &'static str| -> Result<u32, TokenizerError> {
            match v {
                MetadataValue::Uint32(n) => Ok(*n),
                _ => Err(TokenizerError::WrongMetadataValueType { key }),
            }
        };
        let bool_of = |v: &MetadataValue, key: &'static str| -> Result<bool, TokenizerError> {
            match v {
                MetadataValue::Bool(b) => Ok(*b),
                _ => Err(TokenizerError::WrongMetadataValueType { key }),
            }
        };
        let strings_of =
            |v: &MetadataValue, key: &'static str| -> Result<Vec<String>, TokenizerError> {
                match v {
                    MetadataValue::StringArray(ss) => Ok(ss.clone()),
                    _ => Err(TokenizerError::WrongMetadataValueType { key }),
                }
            };
        let int32s_of =
            |v: &MetadataValue, key: &'static str| -> Result<Vec<i32>, TokenizerError> {
                match v {
                    MetadataValue::Int32Array(ns) => Ok(ns.clone()),
                    _ => Err(TokenizerError::WrongMetadataValueType { key }),
                }
            };

        let facts = TokenizerFacts {
            model: str_of(
                model.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.model",
                })?,
                "tokenizer.ggml.model",
            )?,
            pre: str_of(
                pre.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.pre",
                })?,
                "tokenizer.ggml.pre",
            )?,
            tokens: strings_of(
                tokens.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.tokens",
                })?,
                "tokenizer.ggml.tokens",
            )?,
            token_types: int32s_of(
                token_types
                    .as_ref()
                    .ok_or(TokenizerError::MissingMetadataKey {
                        key: "tokenizer.ggml.token_type",
                    })?,
                "tokenizer.ggml.token_type",
            )?,
            merges: strings_of(
                merges.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.merges",
                })?,
                "tokenizer.ggml.merges",
            )?,
            scores_present,
            bos_token_id: u32_of(
                bos.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.bos_token_id",
                })?,
                "tokenizer.ggml.bos_token_id",
            )?,
            eos_token_id: u32_of(
                eos.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.eos_token_id",
                })?,
                "tokenizer.ggml.eos_token_id",
            )?,
            pad_token_id: u32_of(
                pad.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.padding_token_id",
                })?,
                "tokenizer.ggml.padding_token_id",
            )?,
            unk_token_id: u32_of(
                unk.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.unknown_token_id",
                })?,
                "tokenizer.ggml.unknown_token_id",
            )?,
            add_bos_token: bool_of(
                add_bos.as_ref().ok_or(TokenizerError::MissingMetadataKey {
                    key: "tokenizer.ggml.add_bos_token",
                })?,
                "tokenizer.ggml.add_bos_token",
            )?,
            add_space_prefix: bool_of(
                add_space_prefix
                    .as_ref()
                    .ok_or(TokenizerError::MissingMetadataKey {
                        key: "tokenizer.ggml.add_space_prefix",
                    })?,
                "tokenizer.ggml.add_space_prefix",
            )?,
        };
        Self::new(facts)
    }

    /// Encode `text` to token ids, reproducing the pinned llama.cpp fixture.
    ///
    /// - BOS-free by default (`add_bos_token = false`): no token is prepended.
    /// - `parse_special == true`: special-token texts become their single ids
    ///   (`<|im_start|>` → 1, `<|im_end|>` → 2, `<|endoftext|>` → 0).
    /// - `parse_special == false`: the pinned row's control specials are not
    ///   partitioned and are BPE-encoded as literal text.
    ///
    /// # Errors
    ///
    /// `InputTooLarge` beyond the byte ceiling, `OutputTooLarge` beyond the
    /// token ceiling, or `ArithmeticOverflow` on a checked offset/length.
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, TokenizerError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        if text.len() > MAX_ENCODE_INPUT_BYTES {
            return Err(TokenizerError::InputTooLarge {
                bytes: text.len(),
                ceiling: MAX_ENCODE_INPUT_BYTES,
            });
        }

        let mut output: Vec<u32> = Vec::new();
        let mut fragments: Vec<Fragment> = vec![Fragment::Text { start: 0, len: text.len() }];

        // 1. Special-token partition (llama.cpp `tokenizer_st_partition`).
        for special in &self.specials {
            if !parse_special && matches!(special.kind, SpecialKind::Control | SpecialKind::Unknown)
            {
                // Control + unknown specials are ignored when specials are not
                // parsed; user-defined tokens are still partitioned.
                continue;
            }
            self.partition_fragments(&mut fragments, text, special);
        }

        // 2. Per-fragment encode.
        for fragment in fragments {
            match fragment {
                Fragment::Token(id) => {
                    if (id as usize) >= self.vocab_size {
                        return Err(TokenizerError::TokenIdOutOfRange {
                            id,
                            vocab_size: self.vocab_size,
                        });
                    }
                    if output.len() >= MAX_ENCODE_OUTPUT_TOKENS {
                        return Err(TokenizerError::OutputTooLarge {
                            tokens: MAX_ENCODE_OUTPUT_TOKENS,
                            ceiling: MAX_ENCODE_OUTPUT_TOKENS,
                        });
                    }
                    output.push(id);
                }
                Fragment::Text { start, len } => {
                    let fragment_text = text.get(start..start + len).ok_or(
                        TokenizerError::ArithmeticOverflow { what: "fragment slice" },
                    )?;
                    let spans = pre_tokenize_smollm(fragment_text);
                    for (b_start, b_len) in spans {
                        let word = fragment_text.get(b_start..b_start + b_len).ok_or(
                            TokenizerError::ArithmeticOverflow { what: "word slice" },
                        )?;
                        let display = byte_encode(word);
                        self.bpe_encode_word(&display, &mut output)?;
                    }
                }
            }
        }

        Ok(output)
    }

    /// llama.cpp `tokenizer_st_partition` for one special over the fragment list.
    fn partition_fragments(&self, fragments: &mut Vec<Fragment>, raw: &str, special: &Special) {
        let pattern = special.text.as_bytes();
        let raw_bytes = raw.as_bytes();
        let mut i = 0;
        while i < fragments.len() {
            match fragments[i] {
                Fragment::Token(_) => i += 1,
                Fragment::Text { start, len } => {
                    let mut base = start;
                    let mut base_len = len;
                    let mut insert_at = i;
                    let mut total = 0usize;
                    loop {
                        let end = base + base_len;
                        let Some(m) = find_bytes(raw_bytes, pattern, base, end) else {
                            break;
                        };
                        let mut replacement: Vec<Fragment> = Vec::with_capacity(3);
                        if m > base {
                            replacement.push(Fragment::Text {
                                start: base,
                                len: m - base,
                            });
                        }
                        replacement.push(Fragment::Token(special.id));
                        let right_start = m + pattern.len();
                        let right_len = end - right_start;
                        if right_len > 0 {
                            replacement.push(Fragment::Text {
                                start: right_start,
                                len: right_len,
                            });
                        }
                        let repl_len = replacement.len();
                        fragments.splice(insert_at..=insert_at, replacement);
                        total += repl_len;
                        if right_len == 0 {
                            break;
                        }
                        // The right part is the last inserted element; llama.cpp
                        // continues scanning it (`it` points at that node).
                        insert_at += repl_len - 1;
                        base = right_start;
                        base_len = right_len;
                    }
                    // llama.cpp's outer loop advances past the last inserted
                    // node of the split.
                    i += if total == 0 { 1 } else { total };
                }
            }
        }
    }

    /// BPE-encode one byte-encoded (display) fragment, appending ids.
    fn bpe_encode_word(
        &self,
        display: &str,
        output: &mut Vec<u32>,
    ) -> Result<(), TokenizerError> {
        let n_chars = display.chars().count();
        if n_chars == 0 {
            return Ok(());
        }

        // Symbols: one per UTF-8 char of the display fragment.
        let mut symbols: Vec<Symbol> = Vec::with_capacity(n_chars);
        {
            let mut byte_cursor = 0usize;
            for (index, c) in display.chars().enumerate() {
                let len = c.len_utf8();
                symbols.push(Symbol {
                    prev: index as isize - 1,
                    next: if index + 1 == n_chars { -1 } else { index as isize + 1 },
                    start: byte_cursor,
                    n: len,
                });
                byte_cursor = checked_add(byte_cursor, len, "symbol byte cursor")?;
            }
        }

        let mut queue: BinaryHeap<QueueEntry> = BinaryHeap::new();
        let add_bigram = |queue: &mut BinaryHeap<QueueEntry>,
                              symbols: &[Symbol],
                              ranks: &HashMap<(Box<[u8]>, Box<[u8]>), u32>,
                              left: isize,
                              right: isize| {
            if left < 0 || right < 0 {
                return;
            }
            let l = left as usize;
            let r = right as usize;
            let ltext = &display[symbols[l].start..symbols[l].start + symbols[l].n];
            let rtext = &display[symbols[r].start..symbols[r].start + symbols[r].n];
            if let Some(&rank) =
                ranks.get(&(ltext.as_bytes().into(), rtext.as_bytes().into()))
            {
                let mut text = String::with_capacity(ltext.len() + rtext.len());
                text.push_str(ltext);
                text.push_str(rtext);
                queue.push(QueueEntry { rank, left: l, right: r, text });
            }
        };

        for i in 1..n_chars {
            add_bigram(&mut queue, &symbols, &self.bpe_ranks, i as isize - 1, i as isize);
        }

        while let Some(entry) = queue.pop() {
            let left_symbol = &symbols[entry.left];
            let right_symbol = &symbols[entry.right];
            if left_symbol.n == 0 || right_symbol.n == 0 {
                continue;
            }
            let ltext = &display[left_symbol.start..left_symbol.start + left_symbol.n];
            let rtext = &display[right_symbol.start..right_symbol.start + right_symbol.n];
            if ltext.len() + rtext.len() != entry.text.len()
                || ltext != &entry.text[..ltext.len()]
                || rtext != &entry.text[ltext.len()..]
            {
                // Outdated bigram (a neighbour already merged).
                continue;
            }

            // Merge the right symbol into the left one.
            let left_n = checked_add(symbols[entry.left].n, symbols[entry.right].n, "merge len")?;
            let next_of_right = symbols[entry.right].next;
            symbols[entry.left].n = left_n;
            symbols[entry.left].next = next_of_right;
            symbols[entry.right].n = 0;
            if next_of_right >= 0 {
                symbols[next_of_right as usize].prev = entry.left as isize;
            }

            let prev_of_left = symbols[entry.left].prev;
            add_bigram(&mut queue, &symbols, &self.bpe_ranks, prev_of_left, entry.left as isize);
            add_bigram(&mut queue, &symbols, &self.bpe_ranks, entry.left as isize, next_of_right);
        }

        // Collect alive symbols in chain order (llama.cpp `symbols_final` walk).
        let mut idx: isize = 0;
        loop {
            let symbol = &symbols[idx as usize];
            if symbol.n > 0 {
                let text = &display[symbol.start..symbol.start + symbol.n];
                match self.token_to_id.get(text.as_bytes()) {
                    Some(&id) => {
                        if output.len() >= MAX_ENCODE_OUTPUT_TOKENS {
                            return Err(TokenizerError::OutputTooLarge {
                                tokens: MAX_ENCODE_OUTPUT_TOKENS,
                                ceiling: MAX_ENCODE_OUTPUT_TOKENS,
                            });
                        }
                        output.push(id);
                    }
                    None => {
                        // llama.cpp byte fallback: emit tokens for each raw byte
                        // of the display string, dropping unmapped bytes.
                        for &byte in text.as_bytes() {
                            let id = self.byte_fallback[byte as usize];
                            if id != u32::MAX {
                                if output.len() >= MAX_ENCODE_OUTPUT_TOKENS {
                                    return Err(TokenizerError::OutputTooLarge {
                                        tokens: MAX_ENCODE_OUTPUT_TOKENS,
                                        ceiling: MAX_ENCODE_OUTPUT_TOKENS,
                                    });
                                }
                                output.push(id);
                            }
                        }
                    }
                }
            }
            if symbol.next < 0 {
                break;
            }
            idx = symbol.next;
        }

        Ok(())
    }

    /// End-of-generation token ids for the pinned row: `{0, 2}`.
    pub fn eog_tokens(&self) -> &[u32] {
        &self.eog
    }

    /// `tokenizer.ggml.tokens` count (49152).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// `tokenizer.ggml.bos_token_id` (1).
    #[must_use]
    pub fn bos_token_id(&self) -> u32 {
        self.bos
    }

    /// `tokenizer.ggml.eos_token_id` (2).
    #[must_use]
    pub fn eos_token_id(&self) -> u32 {
        self.eos
    }

    /// `tokenizer.ggml.padding_token_id` (2).
    #[must_use]
    pub fn pad_token_id(&self) -> u32 {
        self.pad
    }

    /// `tokenizer.ggml.unknown_token_id` (0).
    #[must_use]
    pub fn unk_token_id(&self) -> u32 {
        self.unk
    }

    /// `tokenizer.ggml.add_bos_token` (false — encode is BOS-free).
    #[must_use]
    pub fn add_bos_token(&self) -> bool {
        self.add_bos
    }

    /// `tokenizer.ggml.add_space_prefix` (false).
    #[must_use]
    pub fn add_space_prefix(&self) -> bool {
        self.add_space_prefix
    }

    /// The 17 cached specials (text → id) in llama.cpp cache order.
    pub fn special_tokens(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.specials.iter().map(|s| (s.text.as_str(), s.id))
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// A post-partition fragment: a raw-text byte span or a special-token id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fragment {
    Token(u32),
    Text { start: usize, len: usize },
}

/// A live BPE symbol: a byte range of the display fragment with chain links.
#[derive(Debug, Clone, Copy)]
struct Symbol {
    prev: isize,
    next: isize,
    /// Byte offset into the display string.
    start: usize,
    /// Byte length.
    n: usize,
}

/// A candidate BPE merge. `Ord` is reversed so the heap pops the smallest
/// (rank, left) first — llama.cpp `llm_bigram_bpe::comparator`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct QueueEntry {
    rank: u32,
    left: usize,
    right: usize,
    text: String,
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .rank
            .cmp(&self.rank)
            .then_with(|| other.left.cmp(&self.left))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn checked_add(a: usize, b: usize, what: &'static str) -> Result<usize, TokenizerError> {
    a.checked_add(b)
        .ok_or(TokenizerError::ArithmeticOverflow { what })
}

/// Find `pattern` in `haystack[from..end]`; returns the match start byte index.
fn find_bytes(haystack: &[u8], pattern: &[u8], from: usize, end: usize) -> Option<usize> {
    if pattern.is_empty() || from + pattern.len() > end {
        return None;
    }
    haystack[from..end]
        .windows(pattern.len())
        .position(|w| w == pattern)
        .map(|p| from + p)
}

/// `unicode_regex_split` for the `smollm` pre pair — returns byte spans.
fn pre_tokenize_smollm(text: &str) -> Vec<(usize, usize)> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();

    // Step 1: `\p{N}` — each number codepoint becomes its own fragment
    // (llama.cpp STL collapsed-regex path, equivalent per-digit split).
    let mut spans: Vec<(usize, usize)> = Vec::new();
    let mut cur = 0usize;
    for (i, &c) in chars.iter().enumerate() {
        if cpt_is_number(c) {
            if i > cur {
                spans.push((cur, i));
            }
            spans.push((i, i + 1));
            cur = i + 1;
        }
    }
    if cur < n {
        spans.push((cur, n));
    }

    // Step 2: gpt2 custom splitter over each span
    // (`unicode_regex_split_custom_gpt2`).
    let mut char_spans: Vec<(usize, usize)> = Vec::new();
    for &(s, e) in &spans {
        split_gpt2_span(&chars, s, e, &mut char_spans);
    }

    // char spans → byte spans.
    let mut byte_of: Vec<usize> = Vec::with_capacity(n + 1);
    let mut acc = 0usize;
    byte_of.push(0);
    for c in &chars {
        acc += c.len_utf8();
        byte_of.push(acc);
    }
    char_spans
        .into_iter()
        .map(|(s, e)| (byte_of[s], byte_of[e] - byte_of[s]))
        .collect()
}

/// The GPT2 custom splitter (`unicode_regex_split_custom_gpt2`) over one span
/// of codepoints, appending (char_start, char_end) fragments.
fn split_gpt2_span(chars: &[char], ini: usize, end: usize, out: &mut Vec<(usize, usize)>) {
    let get_cpt = |pos: usize| -> Option<char> {
        if ini <= pos && pos < end {
            Some(chars[pos])
        } else {
            None
        }
    };
    let get_flags = |pos: usize| -> u16 {
        if ini <= pos && pos < end {
            cpt_flags(chars[pos])
        } else {
            0
        }
    };

    let mut prev_end = ini;
    let mut add_token = |pos: usize, prev_end: &mut usize| {
        if pos > *prev_end {
            out.push((*prev_end, pos));
            *prev_end = pos;
        }
    };

    let mut pos = ini;
    while pos < end {
        let cpt = get_cpt(pos);
        let flags = get_flags(pos);

        // `'s|'t|'re|'ve|'m|'ll|'d` (lowercase only, like llama.cpp).
        if cpt == Some('\'') && pos + 1 < end {
            let next = get_cpt(pos + 1);
            if matches!(next, Some('s') | Some('t') | Some('m') | Some('d')) {
                pos += 2;
                add_token(pos, &mut prev_end);
                continue;
            }
            if pos + 2 < end {
                let next2 = get_cpt(pos + 2);
                if matches!(
                    (next, next2),
                    (Some('r'), Some('e'))
                        | (Some('v'), Some('e'))
                        | (Some('l'), Some('l'))
                ) {
                    pos += 3;
                    add_token(pos, &mut prev_end);
                    continue;
                }
            }
        }

        let next_flags = get_flags(pos + 1);
        let flags2 = if cpt == Some(' ') { next_flags } else { flags };

        // ` ?\p{L}+`
        if flags2 & CPT_LETTER != 0 {
            pos += usize::from(cpt == Some(' '));
            let mut f2 = flags2;
            while f2 & CPT_LETTER != 0 {
                pos += 1;
                f2 = get_flags(pos);
            }
            add_token(pos, &mut prev_end);
            continue;
        }

        // ` ?\p{N}+`
        if flags2 & CPT_NUMBER != 0 {
            pos += usize::from(cpt == Some(' '));
            let mut f2 = flags2;
            while f2 & CPT_NUMBER != 0 {
                pos += 1;
                f2 = get_flags(pos);
            }
            add_token(pos, &mut prev_end);
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+` — note the out-of-span flags are 0 (llama.cpp
        // `unicode_cpt_flags{}` sentinel), which terminates the run.
        if flags2 & (CPT_WHITESPACE | CPT_LETTER | CPT_NUMBER) == 0 && flags2 != 0 {
            pos += usize::from(cpt == Some(' '));
            let mut f2 = flags2;
            while f2 & (CPT_WHITESPACE | CPT_LETTER | CPT_NUMBER) == 0 && f2 != 0 {
                pos += 1;
                f2 = get_flags(pos);
            }
            add_token(pos, &mut prev_end);
            continue;
        }

        // Whitespace runs.
        let mut num_ws = 0usize;
        while get_flags(pos + num_ws) & CPT_WHITESPACE != 0 {
            num_ws += 1;
        }

        // `\s+(?!\S)` — run of ≥2 whitespace followed by more text inside the
        // span: emit all but the last, then let the last attach to the next
        // word via the space-prefix branches.
        if num_ws > 1 && get_cpt(pos + num_ws).is_some() {
            pos += num_ws - 1;
            add_token(pos, &mut prev_end);
            continue;
        }

        // `\s+`
        if num_ws > 0 {
            pos += num_ws;
            add_token(pos, &mut prev_end);
            continue;
        }

        // No match: consume a single codepoint (llama.cpp `_add_token(++pos)`).
        pos += 1;
        add_token(pos, &mut prev_end);
    }
}

/// GPT2 byte encoding (`unicode_byte_encoding_process` + `unicode_byte_to_utf8_map`).
fn byte_encode(word: &str) -> String {
    let mut out = String::with_capacity(word.len() * 2);
    for &byte in word.as_bytes() {
        out.push(BYTE_TO_DISPLAY[byte as usize]);
    }
    out
}

// ---------------------------------------------------------------------------
// Codepoint flags (llama.cpp 10150 unicode-data.cpp; see module docs)
// ---------------------------------------------------------------------------

const CPT_NUMBER: u16 = 0x0002;
const CPT_LETTER: u16 = 0x0004;
const CPT_WHITESPACE: u16 = 0x0100;

fn cpt_flags(c: char) -> u16 {
    let cp = c as u32;
    // `unicode_cpt_flags_array`: every codepoint defaults to UNDEFINED (0x0001)
    // and is overridden by its containing range; WHITESPACE is OR-ed from
    // `unicode_set_whitespace`.
    let idx = CPT_RANGES.partition_point(|&(start, _)| start <= cp) - 1;
    let mut flags = CPT_RANGES[idx].1;
    if WHITESPACE_SET.binary_search(&cp).is_ok() {
        flags |= CPT_WHITESPACE;
    }
    flags
}

fn cpt_is_number(c: char) -> bool {
    cpt_flags(c) & CPT_NUMBER != 0
}

// ---------------------------------------------------------------------------
// Embedded data: Unicode codepoint flags (llama.cpp 10150 dee2a846b
// src/unicode-data.cpp `unicode_ranges_flags` + `unicode_set_whitespace` —
// the exact data the pinned comparator executes) and the GPT2 byte-to-display
// table (`unicode_byte_to_utf8_map`).
// ---------------------------------------------------------------------------

#[rustfmt::skip]
pub(crate) const CPT_RANGES: &[(u32, u16)] = &[
    (0x000000, 0x0080),
    (0x000020, 0x0008),
    (0x000021, 0x0020),
    (0x000024, 0x0040),
    (0x000025, 0x0020),
    (0x00002B, 0x0040),
    (0x00002C, 0x0020),
    (0x000030, 0x0002),
    (0x00003A, 0x0020),
    (0x00003C, 0x0040),
    (0x00003F, 0x0020),
    (0x000041, 0x0004),
    (0x00005B, 0x0020),
    (0x00005E, 0x0040),
    (0x00005F, 0x0020),
    (0x000060, 0x0040),
    (0x000061, 0x0004),
    (0x00007B, 0x0020),
    (0x00007C, 0x0040),
    (0x00007D, 0x0020),
    (0x00007E, 0x0040),
    (0x00007F, 0x0080),
    (0x0000A0, 0x0008),
    (0x0000A1, 0x0020),
    (0x0000A2, 0x0040),
    (0x0000A7, 0x0020),
    (0x0000A8, 0x0040),
    (0x0000AA, 0x0004),
    (0x0000AB, 0x0020),
    (0x0000AC, 0x0040),
    (0x0000AD, 0x0080),
    (0x0000AE, 0x0040),
    (0x0000B2, 0x0002),
    (0x0000B4, 0x0040),
    (0x0000B5, 0x0004),
    (0x0000B6, 0x0020),
    (0x0000B8, 0x0040),
    (0x0000B9, 0x0002),
    (0x0000BA, 0x0004),
    (0x0000BB, 0x0020),
    (0x0000BC, 0x0002),
    (0x0000BF, 0x0020),
    (0x0000C0, 0x0004),
    (0x0000D7, 0x0040),
    (0x0000D8, 0x0004),
    (0x0000F7, 0x0040),
    (0x0000F8, 0x0004),
    (0x0002C2, 0x0040),
    (0x0002C6, 0x0004),
    (0x0002D2, 0x0040),
    (0x0002E0, 0x0004),
    (0x0002E5, 0x0040),
    (0x0002EC, 0x0004),
    (0x0002ED, 0x0040),
    (0x0002EE, 0x0004),
    (0x0002EF, 0x0040),
    (0x000300, 0x0010),
    (0x000370, 0x0004),
    (0x000375, 0x0040),
    (0x000376, 0x0004),
    (0x000378, 0x0001),
    (0x00037A, 0x0004),
    (0x00037E, 0x0020),
    (0x00037F, 0x0004),
    (0x000380, 0x0001),
    (0x000384, 0x0040),
    (0x000386, 0x0004),
    (0x000387, 0x0020),
    (0x000388, 0x0004),
    (0x00038B, 0x0001),
    (0x00038C, 0x0004),
    (0x00038D, 0x0001),
    (0x00038E, 0x0004),
    (0x0003A2, 0x0001),
    (0x0003A3, 0x0004),
    (0x0003F6, 0x0040),
    (0x0003F7, 0x0004),
    (0x000482, 0x0040),
    (0x000483, 0x0010),
    (0x00048A, 0x0004),
    (0x000530, 0x0001),
    (0x000531, 0x0004),
    (0x000557, 0x0001),
    (0x000559, 0x0004),
    (0x00055A, 0x0020),
    (0x000560, 0x0004),
    (0x000589, 0x0020),
    (0x00058B, 0x0001),
    (0x00058D, 0x0040),
    (0x000590, 0x0001),
    (0x000591, 0x0010),
    (0x0005BE, 0x0020),
    (0x0005BF, 0x0010),
    (0x0005C0, 0x0020),
    (0x0005C1, 0x0010),
    (0x0005C3, 0x0020),
    (0x0005C4, 0x0010),
    (0x0005C6, 0x0020),
    (0x0005C7, 0x0010),
    (0x0005C8, 0x0001),
    (0x0005D0, 0x0004),
    (0x0005EB, 0x0001),
    (0x0005EF, 0x0004),
    (0x0005F3, 0x0020),
    (0x0005F5, 0x0001),
    (0x000600, 0x0080),
    (0x000606, 0x0040),
    (0x000609, 0x0020),
    (0x00060B, 0x0040),
    (0x00060C, 0x0020),
    (0x00060E, 0x0040),
    (0x000610, 0x0010),
    (0x00061B, 0x0020),
    (0x00061C, 0x0080),
    (0x00061D, 0x0020),
    (0x000620, 0x0004),
    (0x00064B, 0x0010),
    (0x000660, 0x0002),
    (0x00066A, 0x0020),
    (0x00066E, 0x0004),
    (0x000670, 0x0010),
    (0x000671, 0x0004),
    (0x0006D4, 0x0020),
    (0x0006D5, 0x0004),
    (0x0006D6, 0x0010),
    (0x0006DD, 0x0080),
    (0x0006DE, 0x0040),
    (0x0006DF, 0x0010),
    (0x0006E5, 0x0004),
    (0x0006E7, 0x0010),
    (0x0006E9, 0x0040),
    (0x0006EA, 0x0010),
    (0x0006EE, 0x0004),
    (0x0006F0, 0x0002),
    (0x0006FA, 0x0004),
    (0x0006FD, 0x0040),
    (0x0006FF, 0x0004),
    (0x000700, 0x0020),
    (0x00070E, 0x0001),
    (0x00070F, 0x0080),
    (0x000710, 0x0004),
    (0x000711, 0x0010),
    (0x000712, 0x0004),
    (0x000730, 0x0010),
    (0x00074B, 0x0001),
    (0x00074D, 0x0004),
    (0x0007A6, 0x0010),
    (0x0007B1, 0x0004),
    (0x0007B2, 0x0001),
    (0x0007C0, 0x0002),
    (0x0007CA, 0x0004),
    (0x0007EB, 0x0010),
    (0x0007F4, 0x0004),
    (0x0007F6, 0x0040),
    (0x0007F7, 0x0020),
    (0x0007FA, 0x0004),
    (0x0007FB, 0x0001),
    (0x0007FD, 0x0010),
    (0x0007FE, 0x0040),
    (0x000800, 0x0004),
    (0x000816, 0x0010),
    (0x00081A, 0x0004),
    (0x00081B, 0x0010),
    (0x000824, 0x0004),
    (0x000825, 0x0010),
    (0x000828, 0x0004),
    (0x000829, 0x0010),
    (0x00082E, 0x0001),
    (0x000830, 0x0020),
    (0x00083F, 0x0001),
    (0x000840, 0x0004),
    (0x000859, 0x0010),
    (0x00085C, 0x0001),
    (0x00085E, 0x0020),
    (0x00085F, 0x0001),
    (0x000860, 0x0004),
    (0x00086B, 0x0001),
    (0x000870, 0x0004),
    (0x000888, 0x0040),
    (0x000889, 0x0004),
    (0x00088F, 0x0001),
    (0x000890, 0x0080),
    (0x000892, 0x0001),
    (0x000898, 0x0010),
    (0x0008A0, 0x0004),
    (0x0008CA, 0x0010),
    (0x0008E2, 0x0080),
    (0x0008E3, 0x0010),
    (0x000904, 0x0004),
    (0x00093A, 0x0010),
    (0x00093D, 0x0004),
    (0x00093E, 0x0010),
    (0x000950, 0x0004),
    (0x000951, 0x0010),
    (0x000958, 0x0004),
    (0x000962, 0x0010),
    (0x000964, 0x0020),
    (0x000966, 0x0002),
    (0x000970, 0x0020),
    (0x000971, 0x0004),
    (0x000981, 0x0010),
    (0x000984, 0x0001),
    (0x000985, 0x0004),
    (0x00098D, 0x0001),
    (0x00098F, 0x0004),
    (0x000991, 0x0001),
    (0x000993, 0x0004),
    (0x0009A9, 0x0001),
    (0x0009AA, 0x0004),
    (0x0009B1, 0x0001),
    (0x0009B2, 0x0004),
    (0x0009B3, 0x0001),
    (0x0009B6, 0x0004),
    (0x0009BA, 0x0001),
    (0x0009BC, 0x0010),
    (0x0009BD, 0x0004),
    (0x0009BE, 0x0010),
    (0x0009C5, 0x0001),
    (0x0009C7, 0x0010),
    (0x0009C9, 0x0001),
    (0x0009CB, 0x0010),
    (0x0009CE, 0x0004),
    (0x0009CF, 0x0001),
    (0x0009D7, 0x0010),
    (0x0009D8, 0x0001),
    (0x0009DC, 0x0004),
    (0x0009DE, 0x0001),
    (0x0009DF, 0x0004),
    (0x0009E2, 0x0010),
    (0x0009E4, 0x0001),
    (0x0009E6, 0x0002),
    (0x0009F0, 0x0004),
    (0x0009F2, 0x0040),
    (0x0009F4, 0x0002),
    (0x0009FA, 0x0040),
    (0x0009FC, 0x0004),
    (0x0009FD, 0x0020),
    (0x0009FE, 0x0010),
    (0x0009FF, 0x0001),
    (0x000A01, 0x0010),
    (0x000A04, 0x0001),
    (0x000A05, 0x0004),
    (0x000A0B, 0x0001),
    (0x000A0F, 0x0004),
    (0x000A11, 0x0001),
    (0x000A13, 0x0004),
    (0x000A29, 0x0001),
    (0x000A2A, 0x0004),
    (0x000A31, 0x0001),
    (0x000A32, 0x0004),
    (0x000A34, 0x0001),
    (0x000A35, 0x0004),
    (0x000A37, 0x0001),
    (0x000A38, 0x0004),
    (0x000A3A, 0x0001),
    (0x000A3C, 0x0010),
    (0x000A3D, 0x0001),
    (0x000A3E, 0x0010),
    (0x000A43, 0x0001),
    (0x000A47, 0x0010),
    (0x000A49, 0x0001),
    (0x000A4B, 0x0010),
    (0x000A4E, 0x0001),
    (0x000A51, 0x0010),
    (0x000A52, 0x0001),
    (0x000A59, 0x0004),
    (0x000A5D, 0x0001),
    (0x000A5E, 0x0004),
    (0x000A5F, 0x0001),
    (0x000A66, 0x0002),
    (0x000A70, 0x0010),
    (0x000A72, 0x0004),
    (0x000A75, 0x0010),
    (0x000A76, 0x0020),
    (0x000A77, 0x0001),
    (0x000A81, 0x0010),
    (0x000A84, 0x0001),
    (0x000A85, 0x0004),
    (0x000A8E, 0x0001),
    (0x000A8F, 0x0004),
    (0x000A92, 0x0001),
    (0x000A93, 0x0004),
    (0x000AA9, 0x0001),
    (0x000AAA, 0x0004),
    (0x000AB1, 0x0001),
    (0x000AB2, 0x0004),
    (0x000AB4, 0x0001),
    (0x000AB5, 0x0004),
    (0x000ABA, 0x0001),
    (0x000ABC, 0x0010),
    (0x000ABD, 0x0004),
    (0x000ABE, 0x0010),
    (0x000AC6, 0x0001),
    (0x000AC7, 0x0010),
    (0x000ACA, 0x0001),
    (0x000ACB, 0x0010),
    (0x000ACE, 0x0001),
    (0x000AD0, 0x0004),
    (0x000AD1, 0x0001),
    (0x000AE0, 0x0004),
    (0x000AE2, 0x0010),
    (0x000AE4, 0x0001),
    (0x000AE6, 0x0002),
    (0x000AF0, 0x0020),
    (0x000AF1, 0x0040),
    (0x000AF2, 0x0001),
    (0x000AF9, 0x0004),
    (0x000AFA, 0x0010),
    (0x000B00, 0x0001),
    (0x000B01, 0x0010),
    (0x000B04, 0x0001),
    (0x000B05, 0x0004),
    (0x000B0D, 0x0001),
    (0x000B0F, 0x0004),
    (0x000B11, 0x0001),
    (0x000B13, 0x0004),
    (0x000B29, 0x0001),
    (0x000B2A, 0x0004),
    (0x000B31, 0x0001),
    (0x000B32, 0x0004),
    (0x000B34, 0x0001),
    (0x000B35, 0x0004),
    (0x000B3A, 0x0001),
    (0x000B3C, 0x0010),
    (0x000B3D, 0x0004),
    (0x000B3E, 0x0010),
    (0x000B45, 0x0001),
    (0x000B47, 0x0010),
    (0x000B49, 0x0001),
    (0x000B4B, 0x0010),
    (0x000B4E, 0x0001),
    (0x000B55, 0x0010),
    (0x000B58, 0x0001),
    (0x000B5C, 0x0004),
    (0x000B5E, 0x0001),
    (0x000B5F, 0x0004),
    (0x000B62, 0x0010),
    (0x000B64, 0x0001),
    (0x000B66, 0x0002),
    (0x000B70, 0x0040),
    (0x000B71, 0x0004),
    (0x000B72, 0x0002),
    (0x000B78, 0x0001),
    (0x000B82, 0x0010),
    (0x000B83, 0x0004),
    (0x000B84, 0x0001),
    (0x000B85, 0x0004),
    (0x000B8B, 0x0001),
    (0x000B8E, 0x0004),
    (0x000B91, 0x0001),
    (0x000B92, 0x0004),
    (0x000B96, 0x0001),
    (0x000B99, 0x0004),
    (0x000B9B, 0x0001),
    (0x000B9C, 0x0004),
    (0x000B9D, 0x0001),
    (0x000B9E, 0x0004),
    (0x000BA0, 0x0001),
    (0x000BA3, 0x0004),
    (0x000BA5, 0x0001),
    (0x000BA8, 0x0004),
    (0x000BAB, 0x0001),
    (0x000BAE, 0x0004),
    (0x000BBA, 0x0001),
    (0x000BBE, 0x0010),
    (0x000BC3, 0x0001),
    (0x000BC6, 0x0010),
    (0x000BC9, 0x0001),
    (0x000BCA, 0x0010),
    (0x000BCE, 0x0001),
    (0x000BD0, 0x0004),
    (0x000BD1, 0x0001),
    (0x000BD7, 0x0010),
    (0x000BD8, 0x0001),
    (0x000BE6, 0x0002),
    (0x000BF3, 0x0040),
    (0x000BFB, 0x0001),
    (0x000C00, 0x0010),
    (0x000C05, 0x0004),
    (0x000C0D, 0x0001),
    (0x000C0E, 0x0004),
    (0x000C11, 0x0001),
    (0x000C12, 0x0004),
    (0x000C29, 0x0001),
    (0x000C2A, 0x0004),
    (0x000C3A, 0x0001),
    (0x000C3C, 0x0010),
    (0x000C3D, 0x0004),
    (0x000C3E, 0x0010),
    (0x000C45, 0x0001),
    (0x000C46, 0x0010),
    (0x000C49, 0x0001),
    (0x000C4A, 0x0010),
    (0x000C4E, 0x0001),
    (0x000C55, 0x0010),
    (0x000C57, 0x0001),
    (0x000C58, 0x0004),
    (0x000C5B, 0x0001),
    (0x000C5D, 0x0004),
    (0x000C5E, 0x0001),
    (0x000C60, 0x0004),
    (0x000C62, 0x0010),
    (0x000C64, 0x0001),
    (0x000C66, 0x0002),
    (0x000C70, 0x0001),
    (0x000C77, 0x0020),
    (0x000C78, 0x0002),
    (0x000C7F, 0x0040),
    (0x000C80, 0x0004),
    (0x000C81, 0x0010),
    (0x000C84, 0x0020),
    (0x000C85, 0x0004),
    (0x000C8D, 0x0001),
    (0x000C8E, 0x0004),
    (0x000C91, 0x0001),
    (0x000C92, 0x0004),
    (0x000CA9, 0x0001),
    (0x000CAA, 0x0004),
    (0x000CB4, 0x0001),
    (0x000CB5, 0x0004),
    (0x000CBA, 0x0001),
    (0x000CBC, 0x0010),
    (0x000CBD, 0x0004),
    (0x000CBE, 0x0010),
    (0x000CC5, 0x0001),
    (0x000CC6, 0x0010),
    (0x000CC9, 0x0001),
    (0x000CCA, 0x0010),
    (0x000CCE, 0x0001),
    (0x000CD5, 0x0010),
    (0x000CD7, 0x0001),
    (0x000CDD, 0x0004),
    (0x000CDF, 0x0001),
    (0x000CE0, 0x0004),
    (0x000CE2, 0x0010),
    (0x000CE4, 0x0001),
    (0x000CE6, 0x0002),
    (0x000CF0, 0x0001),
    (0x000CF1, 0x0004),
    (0x000CF3, 0x0010),
    (0x000CF4, 0x0001),
    (0x000D00, 0x0010),
    (0x000D04, 0x0004),
    (0x000D0D, 0x0001),
    (0x000D0E, 0x0004),
    (0x000D11, 0x0001),
    (0x000D12, 0x0004),
    (0x000D3B, 0x0010),
    (0x000D3D, 0x0004),
    (0x000D3E, 0x0010),
    (0x000D45, 0x0001),
    (0x000D46, 0x0010),
    (0x000D49, 0x0001),
    (0x000D4A, 0x0010),
    (0x000D4E, 0x0004),
    (0x000D4F, 0x0040),
    (0x000D50, 0x0001),
    (0x000D54, 0x0004),
    (0x000D57, 0x0010),
    (0x000D58, 0x0002),
    (0x000D5F, 0x0004),
    (0x000D62, 0x0010),
    (0x000D64, 0x0001),
    (0x000D66, 0x0002),
    (0x000D79, 0x0040),
    (0x000D7A, 0x0004),
    (0x000D80, 0x0001),
    (0x000D81, 0x0010),
    (0x000D84, 0x0001),
    (0x000D85, 0x0004),
    (0x000D97, 0x0001),
    (0x000D9A, 0x0004),
    (0x000DB2, 0x0001),
    (0x000DB3, 0x0004),
    (0x000DBC, 0x0001),
    (0x000DBD, 0x0004),
    (0x000DBE, 0x0001),
    (0x000DC0, 0x0004),
    (0x000DC7, 0x0001),
    (0x000DCA, 0x0010),
    (0x000DCB, 0x0001),
    (0x000DCF, 0x0010),
    (0x000DD5, 0x0001),
    (0x000DD6, 0x0010),
    (0x000DD7, 0x0001),
    (0x000DD8, 0x0010),
    (0x000DE0, 0x0001),
    (0x000DE6, 0x0002),
    (0x000DF0, 0x0001),
    (0x000DF2, 0x0010),
    (0x000DF4, 0x0020),
    (0x000DF5, 0x0001),
    (0x000E01, 0x0004),
    (0x000E31, 0x0010),
    (0x000E32, 0x0004),
    (0x000E34, 0x0010),
    (0x000E3B, 0x0001),
    (0x000E3F, 0x0040),
    (0x000E40, 0x0004),
    (0x000E47, 0x0010),
    (0x000E4F, 0x0020),
    (0x000E50, 0x0002),
    (0x000E5A, 0x0020),
    (0x000E5C, 0x0001),
    (0x000E81, 0x0004),
    (0x000E83, 0x0001),
    (0x000E84, 0x0004),
    (0x000E85, 0x0001),
    (0x000E86, 0x0004),
    (0x000E8B, 0x0001),
    (0x000E8C, 0x0004),
    (0x000EA4, 0x0001),
    (0x000EA5, 0x0004),
    (0x000EA6, 0x0001),
    (0x000EA7, 0x0004),
    (0x000EB1, 0x0010),
    (0x000EB2, 0x0004),
    (0x000EB4, 0x0010),
    (0x000EBD, 0x0004),
    (0x000EBE, 0x0001),
    (0x000EC0, 0x0004),
    (0x000EC5, 0x0001),
    (0x000EC6, 0x0004),
    (0x000EC7, 0x0001),
    (0x000EC8, 0x0010),
    (0x000ECF, 0x0001),
    (0x000ED0, 0x0002),
    (0x000EDA, 0x0001),
    (0x000EDC, 0x0004),
    (0x000EE0, 0x0001),
    (0x000F00, 0x0004),
    (0x000F01, 0x0040),
    (0x000F04, 0x0020),
    (0x000F13, 0x0040),
    (0x000F14, 0x0020),
    (0x000F15, 0x0040),
    (0x000F18, 0x0010),
    (0x000F1A, 0x0040),
    (0x000F20, 0x0002),
    (0x000F34, 0x0040),
    (0x000F35, 0x0010),
    (0x000F36, 0x0040),
    (0x000F37, 0x0010),
    (0x000F38, 0x0040),
    (0x000F39, 0x0010),
    (0x000F3A, 0x0020),
    (0x000F3E, 0x0010),
    (0x000F40, 0x0004),
    (0x000F48, 0x0001),
    (0x000F49, 0x0004),
    (0x000F6D, 0x0001),
    (0x000F71, 0x0010),
    (0x000F85, 0x0020),
    (0x000F86, 0x0010),
    (0x000F88, 0x0004),
    (0x000F8D, 0x0010),
    (0x000F98, 0x0001),
    (0x000F99, 0x0010),
    (0x000FBD, 0x0001),
    (0x000FBE, 0x0040),
    (0x000FC6, 0x0010),
    (0x000FC7, 0x0040),
    (0x000FCD, 0x0001),
    (0x000FCE, 0x0040),
    (0x000FD0, 0x0020),
    (0x000FD5, 0x0040),
    (0x000FD9, 0x0020),
    (0x000FDB, 0x0001),
    (0x001000, 0x0004),
    (0x00102B, 0x0010),
    (0x00103F, 0x0004),
    (0x001040, 0x0002),
    (0x00104A, 0x0020),
    (0x001050, 0x0004),
    (0x001056, 0x0010),
    (0x00105A, 0x0004),
    (0x00105E, 0x0010),
    (0x001061, 0x0004),
    (0x001062, 0x0010),
    (0x001065, 0x0004),
    (0x001067, 0x0010),
    (0x00106E, 0x0004),
    (0x001071, 0x0010),
    (0x001075, 0x0004),
    (0x001082, 0x0010),
    (0x00108E, 0x0004),
    (0x00108F, 0x0010),
    (0x001090, 0x0002),
    (0x00109A, 0x0010),
    (0x00109E, 0x0040),
    (0x0010A0, 0x0004),
    (0x0010C6, 0x0001),
    (0x0010C7, 0x0004),
    (0x0010C8, 0x0001),
    (0x0010CD, 0x0004),
    (0x0010CE, 0x0001),
    (0x0010D0, 0x0004),
    (0x0010FB, 0x0020),
    (0x0010FC, 0x0004),
    (0x001249, 0x0001),
    (0x00124A, 0x0004),
    (0x00124E, 0x0001),
    (0x001250, 0x0004),
    (0x001257, 0x0001),
    (0x001258, 0x0004),
    (0x001259, 0x0001),
    (0x00125A, 0x0004),
    (0x00125E, 0x0001),
    (0x001260, 0x0004),
    (0x001289, 0x0001),
    (0x00128A, 0x0004),
    (0x00128E, 0x0001),
    (0x001290, 0x0004),
    (0x0012B1, 0x0001),
    (0x0012B2, 0x0004),
    (0x0012B6, 0x0001),
    (0x0012B8, 0x0004),
    (0x0012BF, 0x0001),
    (0x0012C0, 0x0004),
    (0x0012C1, 0x0001),
    (0x0012C2, 0x0004),
    (0x0012C6, 0x0001),
    (0x0012C8, 0x0004),
    (0x0012D7, 0x0001),
    (0x0012D8, 0x0004),
    (0x001311, 0x0001),
    (0x001312, 0x0004),
    (0x001316, 0x0001),
    (0x001318, 0x0004),
    (0x00135B, 0x0001),
    (0x00135D, 0x0010),
    (0x001360, 0x0020),
    (0x001369, 0x0002),
    (0x00137D, 0x0001),
    (0x001380, 0x0004),
    (0x001390, 0x0040),
    (0x00139A, 0x0001),
    (0x0013A0, 0x0004),
    (0x0013F6, 0x0001),
    (0x0013F8, 0x0004),
    (0x0013FE, 0x0001),
    (0x001400, 0x0020),
    (0x001401, 0x0004),
    (0x00166D, 0x0040),
    (0x00166E, 0x0020),
    (0x00166F, 0x0004),
    (0x001680, 0x0008),
    (0x001681, 0x0004),
    (0x00169B, 0x0020),
    (0x00169D, 0x0001),
    (0x0016A0, 0x0004),
    (0x0016EB, 0x0020),
    (0x0016EE, 0x0002),
    (0x0016F1, 0x0004),
    (0x0016F9, 0x0001),
    (0x001700, 0x0004),
    (0x001712, 0x0010),
    (0x001716, 0x0001),
    (0x00171F, 0x0004),
    (0x001732, 0x0010),
    (0x001735, 0x0020),
    (0x001737, 0x0001),
    (0x001740, 0x0004),
    (0x001752, 0x0010),
    (0x001754, 0x0001),
    (0x001760, 0x0004),
    (0x00176D, 0x0001),
    (0x00176E, 0x0004),
    (0x001771, 0x0001),
    (0x001772, 0x0010),
    (0x001774, 0x0001),
    (0x001780, 0x0004),
    (0x0017B4, 0x0010),
    (0x0017D4, 0x0020),
    (0x0017D7, 0x0004),
    (0x0017D8, 0x0020),
    (0x0017DB, 0x0040),
    (0x0017DC, 0x0004),
    (0x0017DD, 0x0010),
    (0x0017DE, 0x0001),
    (0x0017E0, 0x0002),
    (0x0017EA, 0x0001),
    (0x0017F0, 0x0002),
    (0x0017FA, 0x0001),
    (0x001800, 0x0020),
    (0x00180B, 0x0010),
    (0x00180E, 0x0080),
    (0x00180F, 0x0010),
    (0x001810, 0x0002),
    (0x00181A, 0x0001),
    (0x001820, 0x0004),
    (0x001879, 0x0001),
    (0x001880, 0x0004),
    (0x001885, 0x0010),
    (0x001887, 0x0004),
    (0x0018A9, 0x0010),
    (0x0018AA, 0x0004),
    (0x0018AB, 0x0001),
    (0x0018B0, 0x0004),
    (0x0018F6, 0x0001),
    (0x001900, 0x0004),
    (0x00191F, 0x0001),
    (0x001920, 0x0010),
    (0x00192C, 0x0001),
    (0x001930, 0x0010),
    (0x00193C, 0x0001),
    (0x001940, 0x0040),
    (0x001941, 0x0001),
    (0x001944, 0x0020),
    (0x001946, 0x0002),
    (0x001950, 0x0004),
    (0x00196E, 0x0001),
    (0x001970, 0x0004),
    (0x001975, 0x0001),
    (0x001980, 0x0004),
    (0x0019AC, 0x0001),
    (0x0019B0, 0x0004),
    (0x0019CA, 0x0001),
    (0x0019D0, 0x0002),
    (0x0019DB, 0x0001),
    (0x0019DE, 0x0040),
    (0x001A00, 0x0004),
    (0x001A17, 0x0010),
    (0x001A1C, 0x0001),
    (0x001A1E, 0x0020),
    (0x001A20, 0x0004),
    (0x001A55, 0x0010),
    (0x001A5F, 0x0001),
    (0x001A60, 0x0010),
    (0x001A7D, 0x0001),
    (0x001A7F, 0x0010),
    (0x001A80, 0x0002),
    (0x001A8A, 0x0001),
    (0x001A90, 0x0002),
    (0x001A9A, 0x0001),
    (0x001AA0, 0x0020),
    (0x001AA7, 0x0004),
    (0x001AA8, 0x0020),
    (0x001AAE, 0x0001),
    (0x001AB0, 0x0010),
    (0x001ACF, 0x0001),
    (0x001B00, 0x0010),
    (0x001B05, 0x0004),
    (0x001B34, 0x0010),
    (0x001B45, 0x0004),
    (0x001B4D, 0x0001),
    (0x001B50, 0x0002),
    (0x001B5A, 0x0020),
    (0x001B61, 0x0040),
    (0x001B6B, 0x0010),
    (0x001B74, 0x0040),
    (0x001B7D, 0x0020),
    (0x001B7F, 0x0001),
    (0x001B80, 0x0010),
    (0x001B83, 0x0004),
    (0x001BA1, 0x0010),
    (0x001BAE, 0x0004),
    (0x001BB0, 0x0002),
    (0x001BBA, 0x0004),
    (0x001BE6, 0x0010),
    (0x001BF4, 0x0001),
    (0x001BFC, 0x0020),
    (0x001C00, 0x0004),
    (0x001C24, 0x0010),
    (0x001C38, 0x0001),
    (0x001C3B, 0x0020),
    (0x001C40, 0x0002),
    (0x001C4A, 0x0001),
    (0x001C4D, 0x0004),
    (0x001C50, 0x0002),
    (0x001C5A, 0x0004),
    (0x001C7E, 0x0020),
    (0x001C80, 0x0004),
    (0x001C89, 0x0001),
    (0x001C90, 0x0004),
    (0x001CBB, 0x0001),
    (0x001CBD, 0x0004),
    (0x001CC0, 0x0020),
    (0x001CC8, 0x0001),
    (0x001CD0, 0x0010),
    (0x001CD3, 0x0020),
    (0x001CD4, 0x0010),
    (0x001CE9, 0x0004),
    (0x001CED, 0x0010),
    (0x001CEE, 0x0004),
    (0x001CF4, 0x0010),
    (0x001CF5, 0x0004),
    (0x001CF7, 0x0010),
    (0x001CFA, 0x0004),
    (0x001CFB, 0x0001),
    (0x001D00, 0x0004),
    (0x001DC0, 0x0010),
    (0x001E00, 0x0004),
    (0x001F16, 0x0001),
    (0x001F18, 0x0004),
    (0x001F1E, 0x0001),
    (0x001F20, 0x0004),
    (0x001F46, 0x0001),
    (0x001F48, 0x0004),
    (0x001F4E, 0x0001),
    (0x001F50, 0x0004),
    (0x001F58, 0x0001),
    (0x001F59, 0x0004),
    (0x001F5A, 0x0001),
    (0x001F5B, 0x0004),
    (0x001F5C, 0x0001),
    (0x001F5D, 0x0004),
    (0x001F5E, 0x0001),
    (0x001F5F, 0x0004),
    (0x001F7E, 0x0001),
    (0x001F80, 0x0004),
    (0x001FB5, 0x0001),
    (0x001FB6, 0x0004),
    (0x001FBD, 0x0040),
    (0x001FBE, 0x0004),
    (0x001FBF, 0x0040),
    (0x001FC2, 0x0004),
    (0x001FC5, 0x0001),
    (0x001FC6, 0x0004),
    (0x001FCD, 0x0040),
    (0x001FD0, 0x0004),
    (0x001FD4, 0x0001),
    (0x001FD6, 0x0004),
    (0x001FDC, 0x0001),
    (0x001FDD, 0x0040),
    (0x001FE0, 0x0004),
    (0x001FED, 0x0040),
    (0x001FF0, 0x0001),
    (0x001FF2, 0x0004),
    (0x001FF5, 0x0001),
    (0x001FF6, 0x0004),
    (0x001FFD, 0x0040),
    (0x001FFF, 0x0001),
    (0x002000, 0x0008),
    (0x00200B, 0x0080),
    (0x002010, 0x0020),
    (0x002028, 0x0008),
    (0x00202A, 0x0080),
    (0x00202F, 0x0008),
    (0x002030, 0x0020),
    (0x002044, 0x0040),
    (0x002045, 0x0020),
    (0x002052, 0x0040),
    (0x002053, 0x0020),
    (0x00205F, 0x0008),
    (0x002060, 0x0080),
    (0x002065, 0x0001),
    (0x002066, 0x0080),
    (0x002070, 0x0002),
    (0x002071, 0x0004),
    (0x002072, 0x0001),
    (0x002074, 0x0002),
    (0x00207A, 0x0040),
    (0x00207D, 0x0020),
    (0x00207F, 0x0004),
    (0x002080, 0x0002),
    (0x00208A, 0x0040),
    (0x00208D, 0x0020),
    (0x00208F, 0x0001),
    (0x002090, 0x0004),
    (0x00209D, 0x0001),
    (0x0020A0, 0x0040),
    (0x0020C1, 0x0001),
    (0x0020D0, 0x0010),
    (0x0020F1, 0x0001),
    (0x002100, 0x0040),
    (0x002102, 0x0004),
    (0x002103, 0x0040),
    (0x002107, 0x0004),
    (0x002108, 0x0040),
    (0x00210A, 0x0004),
    (0x002114, 0x0040),
    (0x002115, 0x0004),
    (0x002116, 0x0040),
    (0x002119, 0x0004),
    (0x00211E, 0x0040),
    (0x002124, 0x0004),
    (0x002125, 0x0040),
    (0x002126, 0x0004),
    (0x002127, 0x0040),
    (0x002128, 0x0004),
    (0x002129, 0x0040),
    (0x00212A, 0x0004),
    (0x00212E, 0x0040),
    (0x00212F, 0x0004),
    (0x00213A, 0x0040),
    (0x00213C, 0x0004),
    (0x002140, 0x0040),
    (0x002145, 0x0004),
    (0x00214A, 0x0040),
    (0x00214E, 0x0004),
    (0x00214F, 0x0040),
    (0x002150, 0x0002),
    (0x002183, 0x0004),
    (0x002185, 0x0002),
    (0x00218A, 0x0040),
    (0x00218C, 0x0001),
    (0x002190, 0x0040),
    (0x002308, 0x0020),
    (0x00230C, 0x0040),
    (0x002329, 0x0020),
    (0x00232B, 0x0040),
    (0x002427, 0x0001),
    (0x002440, 0x0040),
    (0x00244B, 0x0001),
    (0x002460, 0x0002),
    (0x00249C, 0x0040),
    (0x0024EA, 0x0002),
    (0x002500, 0x0040),
    (0x002768, 0x0020),
    (0x002776, 0x0002),
    (0x002794, 0x0040),
    (0x0027C5, 0x0020),
    (0x0027C7, 0x0040),
    (0x0027E6, 0x0020),
    (0x0027F0, 0x0040),
    (0x002983, 0x0020),
    (0x002999, 0x0040),
    (0x0029D8, 0x0020),
    (0x0029DC, 0x0040),
    (0x0029FC, 0x0020),
    (0x0029FE, 0x0040),
    (0x002B74, 0x0001),
    (0x002B76, 0x0040),
    (0x002B96, 0x0001),
    (0x002B97, 0x0040),
    (0x002C00, 0x0004),
    (0x002CE5, 0x0040),
    (0x002CEB, 0x0004),
    (0x002CEF, 0x0010),
    (0x002CF2, 0x0004),
    (0x002CF4, 0x0001),
    (0x002CF9, 0x0020),
    (0x002CFD, 0x0002),
    (0x002CFE, 0x0020),
    (0x002D00, 0x0004),
    (0x002D26, 0x0001),
    (0x002D27, 0x0004),
    (0x002D28, 0x0001),
    (0x002D2D, 0x0004),
    (0x002D2E, 0x0001),
    (0x002D30, 0x0004),
    (0x002D68, 0x0001),
    (0x002D6F, 0x0004),
    (0x002D70, 0x0020),
    (0x002D71, 0x0001),
    (0x002D7F, 0x0010),
    (0x002D80, 0x0004),
    (0x002D97, 0x0001),
    (0x002DA0, 0x0004),
    (0x002DA7, 0x0001),
    (0x002DA8, 0x0004),
    (0x002DAF, 0x0001),
    (0x002DB0, 0x0004),
    (0x002DB7, 0x0001),
    (0x002DB8, 0x0004),
    (0x002DBF, 0x0001),
    (0x002DC0, 0x0004),
    (0x002DC7, 0x0001),
    (0x002DC8, 0x0004),
    (0x002DCF, 0x0001),
    (0x002DD0, 0x0004),
    (0x002DD7, 0x0001),
    (0x002DD8, 0x0004),
    (0x002DDF, 0x0001),
    (0x002DE0, 0x0010),
    (0x002E00, 0x0020),
    (0x002E2F, 0x0004),
    (0x002E30, 0x0020),
    (0x002E50, 0x0040),
    (0x002E52, 0x0020),
    (0x002E5E, 0x0001),
    (0x002E80, 0x0040),
    (0x002E9A, 0x0001),
    (0x002E9B, 0x0040),
    (0x002EF4, 0x0001),
    (0x002F00, 0x0040),
    (0x002FD6, 0x0001),
    (0x002FF0, 0x0040),
    (0x003000, 0x0008),
    (0x003001, 0x0020),
    (0x003004, 0x0040),
    (0x003005, 0x0004),
    (0x003007, 0x0002),
    (0x003008, 0x0020),
    (0x003012, 0x0040),
    (0x003014, 0x0020),
    (0x003020, 0x0040),
    (0x003021, 0x0002),
    (0x00302A, 0x0010),
    (0x003030, 0x0020),
    (0x003031, 0x0004),
    (0x003036, 0x0040),
    (0x003038, 0x0002),
    (0x00303B, 0x0004),
    (0x00303D, 0x0020),
    (0x00303E, 0x0040),
    (0x003040, 0x0001),
    (0x003041, 0x0004),
    (0x003097, 0x0001),
    (0x003099, 0x0010),
    (0x00309B, 0x0040),
    (0x00309D, 0x0004),
    (0x0030A0, 0x0020),
    (0x0030A1, 0x0004),
    (0x0030FB, 0x0020),
    (0x0030FC, 0x0004),
    (0x003100, 0x0001),
    (0x003105, 0x0004),
    (0x003130, 0x0001),
    (0x003131, 0x0004),
    (0x00318F, 0x0001),
    (0x003190, 0x0040),
    (0x003192, 0x0002),
    (0x003196, 0x0040),
    (0x0031A0, 0x0004),
    (0x0031C0, 0x0040),
    (0x0031E4, 0x0001),
    (0x0031EF, 0x0040),
    (0x0031F0, 0x0004),
    (0x003200, 0x0040),
    (0x00321F, 0x0001),
    (0x003220, 0x0002),
    (0x00322A, 0x0040),
    (0x003248, 0x0002),
    (0x003250, 0x0040),
    (0x003251, 0x0002),
    (0x003260, 0x0040),
    (0x003280, 0x0002),
    (0x00328A, 0x0040),
    (0x0032B1, 0x0002),
    (0x0032C0, 0x0040),
    (0x003400, 0x0004),
    (0x004DC0, 0x0040),
    (0x004E00, 0x0004),
    (0x00A48D, 0x0001),
    (0x00A490, 0x0040),
    (0x00A4C7, 0x0001),
    (0x00A4D0, 0x0004),
    (0x00A4FE, 0x0020),
    (0x00A500, 0x0004),
    (0x00A60D, 0x0020),
    (0x00A610, 0x0004),
    (0x00A620, 0x0002),
    (0x00A62A, 0x0004),
    (0x00A62C, 0x0001),
    (0x00A640, 0x0004),
    (0x00A66F, 0x0010),
    (0x00A673, 0x0020),
    (0x00A674, 0x0010),
    (0x00A67E, 0x0020),
    (0x00A67F, 0x0004),
    (0x00A69E, 0x0010),
    (0x00A6A0, 0x0004),
    (0x00A6E6, 0x0002),
    (0x00A6F0, 0x0010),
    (0x00A6F2, 0x0020),
    (0x00A6F8, 0x0001),
    (0x00A700, 0x0040),
    (0x00A717, 0x0004),
    (0x00A720, 0x0040),
    (0x00A722, 0x0004),
    (0x00A789, 0x0040),
    (0x00A78B, 0x0004),
    (0x00A7CB, 0x0001),
    (0x00A7D0, 0x0004),
    (0x00A7D2, 0x0001),
    (0x00A7D3, 0x0004),
    (0x00A7D4, 0x0001),
    (0x00A7D5, 0x0004),
    (0x00A7DA, 0x0001),
    (0x00A7F2, 0x0004),
    (0x00A802, 0x0010),
    (0x00A803, 0x0004),
    (0x00A806, 0x0010),
    (0x00A807, 0x0004),
    (0x00A80B, 0x0010),
    (0x00A80C, 0x0004),
    (0x00A823, 0x0010),
    (0x00A828, 0x0040),
    (0x00A82C, 0x0010),
    (0x00A82D, 0x0001),
    (0x00A830, 0x0002),
    (0x00A836, 0x0040),
    (0x00A83A, 0x0001),
    (0x00A840, 0x0004),
    (0x00A874, 0x0020),
    (0x00A878, 0x0001),
    (0x00A880, 0x0010),
    (0x00A882, 0x0004),
    (0x00A8B4, 0x0010),
    (0x00A8C6, 0x0001),
    (0x00A8CE, 0x0020),
    (0x00A8D0, 0x0002),
    (0x00A8DA, 0x0001),
    (0x00A8E0, 0x0010),
    (0x00A8F2, 0x0004),
    (0x00A8F8, 0x0020),
    (0x00A8FB, 0x0004),
    (0x00A8FC, 0x0020),
    (0x00A8FD, 0x0004),
    (0x00A8FF, 0x0010),
    (0x00A900, 0x0002),
    (0x00A90A, 0x0004),
    (0x00A926, 0x0010),
    (0x00A92E, 0x0020),
    (0x00A930, 0x0004),
    (0x00A947, 0x0010),
    (0x00A954, 0x0001),
    (0x00A95F, 0x0020),
    (0x00A960, 0x0004),
    (0x00A97D, 0x0001),
    (0x00A980, 0x0010),
    (0x00A984, 0x0004),
    (0x00A9B3, 0x0010),
    (0x00A9C1, 0x0020),
    (0x00A9CE, 0x0001),
    (0x00A9CF, 0x0004),
    (0x00A9D0, 0x0002),
    (0x00A9DA, 0x0001),
    (0x00A9DE, 0x0020),
    (0x00A9E0, 0x0004),
    (0x00A9E5, 0x0010),
    (0x00A9E6, 0x0004),
    (0x00A9F0, 0x0002),
    (0x00A9FA, 0x0004),
    (0x00A9FF, 0x0001),
    (0x00AA00, 0x0004),
    (0x00AA29, 0x0010),
    (0x00AA37, 0x0001),
    (0x00AA40, 0x0004),
    (0x00AA43, 0x0010),
    (0x00AA44, 0x0004),
    (0x00AA4C, 0x0010),
    (0x00AA4E, 0x0001),
    (0x00AA50, 0x0002),
    (0x00AA5A, 0x0001),
    (0x00AA5C, 0x0020),
    (0x00AA60, 0x0004),
    (0x00AA77, 0x0040),
    (0x00AA7A, 0x0004),
    (0x00AA7B, 0x0010),
    (0x00AA7E, 0x0004),
    (0x00AAB0, 0x0010),
    (0x00AAB1, 0x0004),
    (0x00AAB2, 0x0010),
    (0x00AAB5, 0x0004),
    (0x00AAB7, 0x0010),
    (0x00AAB9, 0x0004),
    (0x00AABE, 0x0010),
    (0x00AAC0, 0x0004),
    (0x00AAC1, 0x0010),
    (0x00AAC2, 0x0004),
    (0x00AAC3, 0x0001),
    (0x00AADB, 0x0004),
    (0x00AADE, 0x0020),
    (0x00AAE0, 0x0004),
    (0x00AAEB, 0x0010),
    (0x00AAF0, 0x0020),
    (0x00AAF2, 0x0004),
    (0x00AAF5, 0x0010),
    (0x00AAF7, 0x0001),
    (0x00AB01, 0x0004),
    (0x00AB07, 0x0001),
    (0x00AB09, 0x0004),
    (0x00AB0F, 0x0001),
    (0x00AB11, 0x0004),
    (0x00AB17, 0x0001),
    (0x00AB20, 0x0004),
    (0x00AB27, 0x0001),
    (0x00AB28, 0x0004),
    (0x00AB2F, 0x0001),
    (0x00AB30, 0x0004),
    (0x00AB5B, 0x0040),
    (0x00AB5C, 0x0004),
    (0x00AB6A, 0x0040),
    (0x00AB6C, 0x0001),
    (0x00AB70, 0x0004),
    (0x00ABE3, 0x0010),
    (0x00ABEB, 0x0020),
    (0x00ABEC, 0x0010),
    (0x00ABEE, 0x0001),
    (0x00ABF0, 0x0002),
    (0x00ABFA, 0x0001),
    (0x00AC00, 0x0004),
    (0x00D7A4, 0x0001),
    (0x00D7B0, 0x0004),
    (0x00D7C7, 0x0001),
    (0x00D7CB, 0x0004),
    (0x00D7FC, 0x0001),
    (0x00D800, 0x0080),
    (0x00F900, 0x0004),
    (0x00FA6E, 0x0001),
    (0x00FA70, 0x0004),
    (0x00FADA, 0x0001),
    (0x00FB00, 0x0004),
    (0x00FB07, 0x0001),
    (0x00FB13, 0x0004),
    (0x00FB18, 0x0001),
    (0x00FB1D, 0x0004),
    (0x00FB1E, 0x0010),
    (0x00FB1F, 0x0004),
    (0x00FB29, 0x0040),
    (0x00FB2A, 0x0004),
    (0x00FB37, 0x0001),
    (0x00FB38, 0x0004),
    (0x00FB3D, 0x0001),
    (0x00FB3E, 0x0004),
    (0x00FB3F, 0x0001),
    (0x00FB40, 0x0004),
    (0x00FB42, 0x0001),
    (0x00FB43, 0x0004),
    (0x00FB45, 0x0001),
    (0x00FB46, 0x0004),
    (0x00FBB2, 0x0040),
    (0x00FBC3, 0x0001),
    (0x00FBD3, 0x0004),
    (0x00FD3E, 0x0020),
    (0x00FD40, 0x0040),
    (0x00FD50, 0x0004),
    (0x00FD90, 0x0001),
    (0x00FD92, 0x0004),
    (0x00FDC8, 0x0001),
    (0x00FDCF, 0x0040),
    (0x00FDD0, 0x0001),
    (0x00FDF0, 0x0004),
    (0x00FDFC, 0x0040),
    (0x00FE00, 0x0010),
    (0x00FE10, 0x0020),
    (0x00FE1A, 0x0001),
    (0x00FE20, 0x0010),
    (0x00FE30, 0x0020),
    (0x00FE53, 0x0001),
    (0x00FE54, 0x0020),
    (0x00FE62, 0x0040),
    (0x00FE63, 0x0020),
    (0x00FE64, 0x0040),
    (0x00FE67, 0x0001),
    (0x00FE68, 0x0020),
    (0x00FE69, 0x0040),
    (0x00FE6A, 0x0020),
    (0x00FE6C, 0x0001),
    (0x00FE70, 0x0004),
    (0x00FE75, 0x0001),
    (0x00FE76, 0x0004),
    (0x00FEFD, 0x0001),
    (0x00FEFF, 0x0080),
    (0x00FF00, 0x0001),
    (0x00FF01, 0x0020),
    (0x00FF04, 0x0040),
    (0x00FF05, 0x0020),
    (0x00FF0B, 0x0040),
    (0x00FF0C, 0x0020),
    (0x00FF10, 0x0002),
    (0x00FF1A, 0x0020),
    (0x00FF1C, 0x0040),
    (0x00FF1F, 0x0020),
    (0x00FF21, 0x0004),
    (0x00FF3B, 0x0020),
    (0x00FF3E, 0x0040),
    (0x00FF3F, 0x0020),
    (0x00FF40, 0x0040),
    (0x00FF41, 0x0004),
    (0x00FF5B, 0x0020),
    (0x00FF5C, 0x0040),
    (0x00FF5D, 0x0020),
    (0x00FF5E, 0x0040),
    (0x00FF5F, 0x0020),
    (0x00FF66, 0x0004),
    (0x00FFBF, 0x0001),
    (0x00FFC2, 0x0004),
    (0x00FFC8, 0x0001),
    (0x00FFCA, 0x0004),
    (0x00FFD0, 0x0001),
    (0x00FFD2, 0x0004),
    (0x00FFD8, 0x0001),
    (0x00FFDA, 0x0004),
    (0x00FFDD, 0x0001),
    (0x00FFE0, 0x0040),
    (0x00FFE7, 0x0001),
    (0x00FFE8, 0x0040),
    (0x00FFEF, 0x0001),
    (0x00FFF9, 0x0080),
    (0x00FFFC, 0x0040),
    (0x00FFFE, 0x0001),
    (0x010000, 0x0004),
    (0x01000C, 0x0001),
    (0x01000D, 0x0004),
    (0x010027, 0x0001),
    (0x010028, 0x0004),
    (0x01003B, 0x0001),
    (0x01003C, 0x0004),
    (0x01003E, 0x0001),
    (0x01003F, 0x0004),
    (0x01004E, 0x0001),
    (0x010050, 0x0004),
    (0x01005E, 0x0001),
    (0x010080, 0x0004),
    (0x0100FB, 0x0001),
    (0x010100, 0x0020),
    (0x010103, 0x0001),
    (0x010107, 0x0002),
    (0x010134, 0x0001),
    (0x010137, 0x0040),
    (0x010140, 0x0002),
    (0x010179, 0x0040),
    (0x01018A, 0x0002),
    (0x01018C, 0x0040),
    (0x01018F, 0x0001),
    (0x010190, 0x0040),
    (0x01019D, 0x0001),
    (0x0101A0, 0x0040),
    (0x0101A1, 0x0001),
    (0x0101D0, 0x0040),
    (0x0101FD, 0x0010),
    (0x0101FE, 0x0001),
    (0x010280, 0x0004),
    (0x01029D, 0x0001),
    (0x0102A0, 0x0004),
    (0x0102D1, 0x0001),
    (0x0102E0, 0x0010),
    (0x0102E1, 0x0002),
    (0x0102FC, 0x0001),
    (0x010300, 0x0004),
    (0x010320, 0x0002),
    (0x010324, 0x0001),
    (0x01032D, 0x0004),
    (0x010341, 0x0002),
    (0x010342, 0x0004),
    (0x01034A, 0x0002),
    (0x01034B, 0x0001),
    (0x010350, 0x0004),
    (0x010376, 0x0010),
    (0x01037B, 0x0001),
    (0x010380, 0x0004),
    (0x01039E, 0x0001),
    (0x01039F, 0x0020),
    (0x0103A0, 0x0004),
    (0x0103C4, 0x0001),
    (0x0103C8, 0x0004),
    (0x0103D0, 0x0020),
    (0x0103D1, 0x0002),
    (0x0103D6, 0x0001),
    (0x010400, 0x0004),
    (0x01049E, 0x0001),
    (0x0104A0, 0x0002),
    (0x0104AA, 0x0001),
    (0x0104B0, 0x0004),
    (0x0104D4, 0x0001),
    (0x0104D8, 0x0004),
    (0x0104FC, 0x0001),
    (0x010500, 0x0004),
    (0x010528, 0x0001),
    (0x010530, 0x0004),
    (0x010564, 0x0001),
    (0x01056F, 0x0020),
    (0x010570, 0x0004),
    (0x01057B, 0x0001),
    (0x01057C, 0x0004),
    (0x01058B, 0x0001),
    (0x01058C, 0x0004),
    (0x010593, 0x0001),
    (0x010594, 0x0004),
    (0x010596, 0x0001),
    (0x010597, 0x0004),
    (0x0105A2, 0x0001),
    (0x0105A3, 0x0004),
    (0x0105B2, 0x0001),
    (0x0105B3, 0x0004),
    (0x0105BA, 0x0001),
    (0x0105BB, 0x0004),
    (0x0105BD, 0x0001),
    (0x010600, 0x0004),
    (0x010737, 0x0001),
    (0x010740, 0x0004),
    (0x010756, 0x0001),
    (0x010760, 0x0004),
    (0x010768, 0x0001),
    (0x010780, 0x0004),
    (0x010786, 0x0001),
    (0x010787, 0x0004),
    (0x0107B1, 0x0001),
    (0x0107B2, 0x0004),
    (0x0107BB, 0x0001),
    (0x010800, 0x0004),
    (0x010806, 0x0001),
    (0x010808, 0x0004),
    (0x010809, 0x0001),
    (0x01080A, 0x0004),
    (0x010836, 0x0001),
    (0x010837, 0x0004),
    (0x010839, 0x0001),
    (0x01083C, 0x0004),
    (0x01083D, 0x0001),
    (0x01083F, 0x0004),
    (0x010856, 0x0001),
    (0x010857, 0x0020),
    (0x010858, 0x0002),
    (0x010860, 0x0004),
    (0x010877, 0x0040),
    (0x010879, 0x0002),
    (0x010880, 0x0004),
    (0x01089F, 0x0001),
    (0x0108A7, 0x0002),
    (0x0108B0, 0x0001),
    (0x0108E0, 0x0004),
    (0x0108F3, 0x0001),
    (0x0108F4, 0x0004),
    (0x0108F6, 0x0001),
    (0x0108FB, 0x0002),
    (0x010900, 0x0004),
    (0x010916, 0x0002),
    (0x01091C, 0x0001),
    (0x01091F, 0x0020),
    (0x010920, 0x0004),
    (0x01093A, 0x0001),
    (0x01093F, 0x0020),
    (0x010940, 0x0001),
    (0x010980, 0x0004),
    (0x0109B8, 0x0001),
    (0x0109BC, 0x0002),
    (0x0109BE, 0x0004),
    (0x0109C0, 0x0002),
    (0x0109D0, 0x0001),
    (0x0109D2, 0x0002),
    (0x010A00, 0x0004),
    (0x010A01, 0x0010),
    (0x010A04, 0x0001),
    (0x010A05, 0x0010),
    (0x010A07, 0x0001),
    (0x010A0C, 0x0010),
    (0x010A10, 0x0004),
    (0x010A14, 0x0001),
    (0x010A15, 0x0004),
    (0x010A18, 0x0001),
    (0x010A19, 0x0004),
    (0x010A36, 0x0001),
    (0x010A38, 0x0010),
    (0x010A3B, 0x0001),
    (0x010A3F, 0x0010),
    (0x010A40, 0x0002),
    (0x010A49, 0x0001),
    (0x010A50, 0x0020),
    (0x010A59, 0x0001),
    (0x010A60, 0x0004),
    (0x010A7D, 0x0002),
    (0x010A7F, 0x0020),
    (0x010A80, 0x0004),
    (0x010A9D, 0x0002),
    (0x010AA0, 0x0001),
    (0x010AC0, 0x0004),
    (0x010AC8, 0x0040),
    (0x010AC9, 0x0004),
    (0x010AE5, 0x0010),
    (0x010AE7, 0x0001),
    (0x010AEB, 0x0002),
    (0x010AF0, 0x0020),
    (0x010AF7, 0x0001),
    (0x010B00, 0x0004),
    (0x010B36, 0x0001),
    (0x010B39, 0x0020),
    (0x010B40, 0x0004),
    (0x010B56, 0x0001),
    (0x010B58, 0x0002),
    (0x010B60, 0x0004),
    (0x010B73, 0x0001),
    (0x010B78, 0x0002),
    (0x010B80, 0x0004),
    (0x010B92, 0x0001),
    (0x010B99, 0x0020),
    (0x010B9D, 0x0001),
    (0x010BA9, 0x0002),
    (0x010BB0, 0x0001),
    (0x010C00, 0x0004),
    (0x010C49, 0x0001),
    (0x010C80, 0x0004),
    (0x010CB3, 0x0001),
    (0x010CC0, 0x0004),
    (0x010CF3, 0x0001),
    (0x010CFA, 0x0002),
    (0x010D00, 0x0004),
    (0x010D24, 0x0010),
    (0x010D28, 0x0001),
    (0x010D30, 0x0002),
    (0x010D3A, 0x0001),
    (0x010E60, 0x0002),
    (0x010E7F, 0x0001),
    (0x010E80, 0x0004),
    (0x010EAA, 0x0001),
    (0x010EAB, 0x0010),
    (0x010EAD, 0x0020),
    (0x010EAE, 0x0001),
    (0x010EB0, 0x0004),
    (0x010EB2, 0x0001),
    (0x010EFD, 0x0010),
    (0x010F00, 0x0004),
    (0x010F1D, 0x0002),
    (0x010F27, 0x0004),
    (0x010F28, 0x0001),
    (0x010F30, 0x0004),
    (0x010F46, 0x0010),
    (0x010F51, 0x0002),
    (0x010F55, 0x0020),
    (0x010F5A, 0x0001),
    (0x010F70, 0x0004),
    (0x010F82, 0x0010),
    (0x010F86, 0x0020),
    (0x010F8A, 0x0001),
    (0x010FB0, 0x0004),
    (0x010FC5, 0x0002),
    (0x010FCC, 0x0001),
    (0x010FE0, 0x0004),
    (0x010FF7, 0x0001),
    (0x011000, 0x0010),
    (0x011003, 0x0004),
    (0x011038, 0x0010),
    (0x011047, 0x0020),
    (0x01104E, 0x0001),
    (0x011052, 0x0002),
    (0x011070, 0x0010),
    (0x011071, 0x0004),
    (0x011073, 0x0010),
    (0x011075, 0x0004),
    (0x011076, 0x0001),
    (0x01107F, 0x0010),
    (0x011083, 0x0004),
    (0x0110B0, 0x0010),
    (0x0110BB, 0x0020),
    (0x0110BD, 0x0080),
    (0x0110BE, 0x0020),
    (0x0110C2, 0x0010),
    (0x0110C3, 0x0001),
    (0x0110CD, 0x0080),
    (0x0110CE, 0x0001),
    (0x0110D0, 0x0004),
    (0x0110E9, 0x0001),
    (0x0110F0, 0x0002),
    (0x0110FA, 0x0001),
    (0x011100, 0x0010),
    (0x011103, 0x0004),
    (0x011127, 0x0010),
    (0x011135, 0x0001),
    (0x011136, 0x0002),
    (0x011140, 0x0020),
    (0x011144, 0x0004),
    (0x011145, 0x0010),
    (0x011147, 0x0004),
    (0x011148, 0x0001),
    (0x011150, 0x0004),
    (0x011173, 0x0010),
    (0x011174, 0x0020),
    (0x011176, 0x0004),
    (0x011177, 0x0001),
    (0x011180, 0x0010),
    (0x011183, 0x0004),
    (0x0111B3, 0x0010),
    (0x0111C1, 0x0004),
    (0x0111C5, 0x0020),
    (0x0111C9, 0x0010),
    (0x0111CD, 0x0020),
    (0x0111CE, 0x0010),
    (0x0111D0, 0x0002),
    (0x0111DA, 0x0004),
    (0x0111DB, 0x0020),
    (0x0111DC, 0x0004),
    (0x0111DD, 0x0020),
    (0x0111E0, 0x0001),
    (0x0111E1, 0x0002),
    (0x0111F5, 0x0001),
    (0x011200, 0x0004),
    (0x011212, 0x0001),
    (0x011213, 0x0004),
    (0x01122C, 0x0010),
    (0x011238, 0x0020),
    (0x01123E, 0x0010),
    (0x01123F, 0x0004),
    (0x011241, 0x0010),
    (0x011242, 0x0001),
    (0x011280, 0x0004),
    (0x011287, 0x0001),
    (0x011288, 0x0004),
    (0x011289, 0x0001),
    (0x01128A, 0x0004),
    (0x01128E, 0x0001),
    (0x01128F, 0x0004),
    (0x01129E, 0x0001),
    (0x01129F, 0x0004),
    (0x0112A9, 0x0020),
    (0x0112AA, 0x0001),
    (0x0112B0, 0x0004),
    (0x0112DF, 0x0010),
    (0x0112EB, 0x0001),
    (0x0112F0, 0x0002),
    (0x0112FA, 0x0001),
    (0x011300, 0x0010),
    (0x011304, 0x0001),
    (0x011305, 0x0004),
    (0x01130D, 0x0001),
    (0x01130F, 0x0004),
    (0x011311, 0x0001),
    (0x011313, 0x0004),
    (0x011329, 0x0001),
    (0x01132A, 0x0004),
    (0x011331, 0x0001),
    (0x011332, 0x0004),
    (0x011334, 0x0001),
    (0x011335, 0x0004),
    (0x01133A, 0x0001),
    (0x01133B, 0x0010),
    (0x01133D, 0x0004),
    (0x01133E, 0x0010),
    (0x011345, 0x0001),
    (0x011347, 0x0010),
    (0x011349, 0x0001),
    (0x01134B, 0x0010),
    (0x01134E, 0x0001),
    (0x011350, 0x0004),
    (0x011351, 0x0001),
    (0x011357, 0x0010),
    (0x011358, 0x0001),
    (0x01135D, 0x0004),
    (0x011362, 0x0010),
    (0x011364, 0x0001),
    (0x011366, 0x0010),
    (0x01136D, 0x0001),
    (0x011370, 0x0010),
    (0x011375, 0x0001),
    (0x011400, 0x0004),
    (0x011435, 0x0010),
    (0x011447, 0x0004),
    (0x01144B, 0x0020),
    (0x011450, 0x0002),
    (0x01145A, 0x0020),
    (0x01145C, 0x0001),
    (0x01145D, 0x0020),
    (0x01145E, 0x0010),
    (0x01145F, 0x0004),
    (0x011462, 0x0001),
    (0x011480, 0x0004),
    (0x0114B0, 0x0010),
    (0x0114C4, 0x0004),
    (0x0114C6, 0x0020),
    (0x0114C7, 0x0004),
    (0x0114C8, 0x0001),
    (0x0114D0, 0x0002),
    (0x0114DA, 0x0001),
    (0x011580, 0x0004),
    (0x0115AF, 0x0010),
    (0x0115B6, 0x0001),
    (0x0115B8, 0x0010),
    (0x0115C1, 0x0020),
    (0x0115D8, 0x0004),
    (0x0115DC, 0x0010),
    (0x0115DE, 0x0001),
    (0x011600, 0x0004),
    (0x011630, 0x0010),
    (0x011641, 0x0020),
    (0x011644, 0x0004),
    (0x011645, 0x0001),
    (0x011650, 0x0002),
    (0x01165A, 0x0001),
    (0x011660, 0x0020),
    (0x01166D, 0x0001),
    (0x011680, 0x0004),
    (0x0116AB, 0x0010),
    (0x0116B8, 0x0004),
    (0x0116B9, 0x0020),
    (0x0116BA, 0x0001),
    (0x0116C0, 0x0002),
    (0x0116CA, 0x0001),
    (0x011700, 0x0004),
    (0x01171B, 0x0001),
    (0x01171D, 0x0010),
    (0x01172C, 0x0001),
    (0x011730, 0x0002),
    (0x01173C, 0x0020),
    (0x01173F, 0x0040),
    (0x011740, 0x0004),
    (0x011747, 0x0001),
    (0x011800, 0x0004),
    (0x01182C, 0x0010),
    (0x01183B, 0x0020),
    (0x01183C, 0x0001),
    (0x0118A0, 0x0004),
    (0x0118E0, 0x0002),
    (0x0118F3, 0x0001),
    (0x0118FF, 0x0004),
    (0x011907, 0x0001),
    (0x011909, 0x0004),
    (0x01190A, 0x0001),
    (0x01190C, 0x0004),
    (0x011914, 0x0001),
    (0x011915, 0x0004),
    (0x011917, 0x0001),
    (0x011918, 0x0004),
    (0x011930, 0x0010),
    (0x011936, 0x0001),
    (0x011937, 0x0010),
    (0x011939, 0x0001),
    (0x01193B, 0x0010),
    (0x01193F, 0x0004),
    (0x011940, 0x0010),
    (0x011941, 0x0004),
    (0x011942, 0x0010),
    (0x011944, 0x0020),
    (0x011947, 0x0001),
    (0x011950, 0x0002),
    (0x01195A, 0x0001),
    (0x0119A0, 0x0004),
    (0x0119A8, 0x0001),
    (0x0119AA, 0x0004),
    (0x0119D1, 0x0010),
    (0x0119D8, 0x0001),
    (0x0119DA, 0x0010),
    (0x0119E1, 0x0004),
    (0x0119E2, 0x0020),
    (0x0119E3, 0x0004),
    (0x0119E4, 0x0010),
    (0x0119E5, 0x0001),
    (0x011A00, 0x0004),
    (0x011A01, 0x0010),
    (0x011A0B, 0x0004),
    (0x011A33, 0x0010),
    (0x011A3A, 0x0004),
    (0x011A3B, 0x0010),
    (0x011A3F, 0x0020),
    (0x011A47, 0x0010),
    (0x011A48, 0x0001),
    (0x011A50, 0x0004),
    (0x011A51, 0x0010),
    (0x011A5C, 0x0004),
    (0x011A8A, 0x0010),
    (0x011A9A, 0x0020),
    (0x011A9D, 0x0004),
    (0x011A9E, 0x0020),
    (0x011AA3, 0x0001),
    (0x011AB0, 0x0004),
    (0x011AF9, 0x0001),
    (0x011B00, 0x0020),
    (0x011B0A, 0x0001),
    (0x011C00, 0x0004),
    (0x011C09, 0x0001),
    (0x011C0A, 0x0004),
    (0x011C2F, 0x0010),
    (0x011C37, 0x0001),
    (0x011C38, 0x0010),
    (0x011C40, 0x0004),
    (0x011C41, 0x0020),
    (0x011C46, 0x0001),
    (0x011C50, 0x0002),
    (0x011C6D, 0x0001),
    (0x011C70, 0x0020),
    (0x011C72, 0x0004),
    (0x011C90, 0x0001),
    (0x011C92, 0x0010),
    (0x011CA8, 0x0001),
    (0x011CA9, 0x0010),
    (0x011CB7, 0x0001),
    (0x011D00, 0x0004),
    (0x011D07, 0x0001),
    (0x011D08, 0x0004),
    (0x011D0A, 0x0001),
    (0x011D0B, 0x0004),
    (0x011D31, 0x0010),
    (0x011D37, 0x0001),
    (0x011D3A, 0x0010),
    (0x011D3B, 0x0001),
    (0x011D3C, 0x0010),
    (0x011D3E, 0x0001),
    (0x011D3F, 0x0010),
    (0x011D46, 0x0004),
    (0x011D47, 0x0010),
    (0x011D48, 0x0001),
    (0x011D50, 0x0002),
    (0x011D5A, 0x0001),
    (0x011D60, 0x0004),
    (0x011D66, 0x0001),
    (0x011D67, 0x0004),
    (0x011D69, 0x0001),
    (0x011D6A, 0x0004),
    (0x011D8A, 0x0010),
    (0x011D8F, 0x0001),
    (0x011D90, 0x0010),
    (0x011D92, 0x0001),
    (0x011D93, 0x0010),
    (0x011D98, 0x0004),
    (0x011D99, 0x0001),
    (0x011DA0, 0x0002),
    (0x011DAA, 0x0001),
    (0x011EE0, 0x0004),
    (0x011EF3, 0x0010),
    (0x011EF7, 0x0020),
    (0x011EF9, 0x0001),
    (0x011F00, 0x0010),
    (0x011F02, 0x0004),
    (0x011F03, 0x0010),
    (0x011F04, 0x0004),
    (0x011F11, 0x0001),
    (0x011F12, 0x0004),
    (0x011F34, 0x0010),
    (0x011F3B, 0x0001),
    (0x011F3E, 0x0010),
    (0x011F43, 0x0020),
    (0x011F50, 0x0002),
    (0x011F5A, 0x0001),
    (0x011FB0, 0x0004),
    (0x011FB1, 0x0001),
    (0x011FC0, 0x0002),
    (0x011FD5, 0x0040),
    (0x011FF2, 0x0001),
    (0x011FFF, 0x0020),
    (0x012000, 0x0004),
    (0x01239A, 0x0001),
    (0x012400, 0x0002),
    (0x01246F, 0x0001),
    (0x012470, 0x0020),
    (0x012475, 0x0001),
    (0x012480, 0x0004),
    (0x012544, 0x0001),
    (0x012F90, 0x0004),
    (0x012FF1, 0x0020),
    (0x012FF3, 0x0001),
    (0x013000, 0x0004),
    (0x013430, 0x0080),
    (0x013440, 0x0010),
    (0x013441, 0x0004),
    (0x013447, 0x0010),
    (0x013456, 0x0001),
    (0x014400, 0x0004),
    (0x014647, 0x0001),
    (0x016800, 0x0004),
    (0x016A39, 0x0001),
    (0x016A40, 0x0004),
    (0x016A5F, 0x0001),
    (0x016A60, 0x0002),
    (0x016A6A, 0x0001),
    (0x016A6E, 0x0020),
    (0x016A70, 0x0004),
    (0x016ABF, 0x0001),
    (0x016AC0, 0x0002),
    (0x016ACA, 0x0001),
    (0x016AD0, 0x0004),
    (0x016AEE, 0x0001),
    (0x016AF0, 0x0010),
    (0x016AF5, 0x0020),
    (0x016AF6, 0x0001),
    (0x016B00, 0x0004),
    (0x016B30, 0x0010),
    (0x016B37, 0x0020),
    (0x016B3C, 0x0040),
    (0x016B40, 0x0004),
    (0x016B44, 0x0020),
    (0x016B45, 0x0040),
    (0x016B46, 0x0001),
    (0x016B50, 0x0002),
    (0x016B5A, 0x0001),
    (0x016B5B, 0x0002),
    (0x016B62, 0x0001),
    (0x016B63, 0x0004),
    (0x016B78, 0x0001),
    (0x016B7D, 0x0004),
    (0x016B90, 0x0001),
    (0x016E40, 0x0004),
    (0x016E80, 0x0002),
    (0x016E97, 0x0020),
    (0x016E9B, 0x0001),
    (0x016F00, 0x0004),
    (0x016F4B, 0x0001),
    (0x016F4F, 0x0010),
    (0x016F50, 0x0004),
    (0x016F51, 0x0010),
    (0x016F88, 0x0001),
    (0x016F8F, 0x0010),
    (0x016F93, 0x0004),
    (0x016FA0, 0x0001),
    (0x016FE0, 0x0004),
    (0x016FE2, 0x0020),
    (0x016FE3, 0x0004),
    (0x016FE4, 0x0010),
    (0x016FE5, 0x0001),
    (0x016FF0, 0x0010),
    (0x016FF2, 0x0001),
    (0x017000, 0x0004),
    (0x0187F8, 0x0001),
    (0x018800, 0x0004),
    (0x018CD6, 0x0001),
    (0x018D00, 0x0004),
    (0x018D09, 0x0001),
    (0x01AFF0, 0x0004),
    (0x01AFF4, 0x0001),
    (0x01AFF5, 0x0004),
    (0x01AFFC, 0x0001),
    (0x01AFFD, 0x0004),
    (0x01AFFF, 0x0001),
    (0x01B000, 0x0004),
    (0x01B123, 0x0001),
    (0x01B132, 0x0004),
    (0x01B133, 0x0001),
    (0x01B150, 0x0004),
    (0x01B153, 0x0001),
    (0x01B155, 0x0004),
    (0x01B156, 0x0001),
    (0x01B164, 0x0004),
    (0x01B168, 0x0001),
    (0x01B170, 0x0004),
    (0x01B2FC, 0x0001),
    (0x01BC00, 0x0004),
    (0x01BC6B, 0x0001),
    (0x01BC70, 0x0004),
    (0x01BC7D, 0x0001),
    (0x01BC80, 0x0004),
    (0x01BC89, 0x0001),
    (0x01BC90, 0x0004),
    (0x01BC9A, 0x0001),
    (0x01BC9C, 0x0040),
    (0x01BC9D, 0x0010),
    (0x01BC9F, 0x0020),
    (0x01BCA0, 0x0080),
    (0x01BCA4, 0x0001),
    (0x01CF00, 0x0010),
    (0x01CF2E, 0x0001),
    (0x01CF30, 0x0010),
    (0x01CF47, 0x0001),
    (0x01CF50, 0x0040),
    (0x01CFC4, 0x0001),
    (0x01D000, 0x0040),
    (0x01D0F6, 0x0001),
    (0x01D100, 0x0040),
    (0x01D127, 0x0001),
    (0x01D129, 0x0040),
    (0x01D165, 0x0010),
    (0x01D16A, 0x0040),
    (0x01D16D, 0x0010),
    (0x01D173, 0x0080),
    (0x01D17B, 0x0010),
    (0x01D183, 0x0040),
    (0x01D185, 0x0010),
    (0x01D18C, 0x0040),
    (0x01D1AA, 0x0010),
    (0x01D1AE, 0x0040),
    (0x01D1EB, 0x0001),
    (0x01D200, 0x0040),
    (0x01D242, 0x0010),
    (0x01D245, 0x0040),
    (0x01D246, 0x0001),
    (0x01D2C0, 0x0002),
    (0x01D2D4, 0x0001),
    (0x01D2E0, 0x0002),
    (0x01D2F4, 0x0001),
    (0x01D300, 0x0040),
    (0x01D357, 0x0001),
    (0x01D360, 0x0002),
    (0x01D379, 0x0001),
    (0x01D400, 0x0004),
    (0x01D455, 0x0001),
    (0x01D456, 0x0004),
    (0x01D49D, 0x0001),
    (0x01D49E, 0x0004),
    (0x01D4A0, 0x0001),
    (0x01D4A2, 0x0004),
    (0x01D4A3, 0x0001),
    (0x01D4A5, 0x0004),
    (0x01D4A7, 0x0001),
    (0x01D4A9, 0x0004),
    (0x01D4AD, 0x0001),
    (0x01D4AE, 0x0004),
    (0x01D4BA, 0x0001),
    (0x01D4BB, 0x0004),
    (0x01D4BC, 0x0001),
    (0x01D4BD, 0x0004),
    (0x01D4C4, 0x0001),
    (0x01D4C5, 0x0004),
    (0x01D506, 0x0001),
    (0x01D507, 0x0004),
    (0x01D50B, 0x0001),
    (0x01D50D, 0x0004),
    (0x01D515, 0x0001),
    (0x01D516, 0x0004),
    (0x01D51D, 0x0001),
    (0x01D51E, 0x0004),
    (0x01D53A, 0x0001),
    (0x01D53B, 0x0004),
    (0x01D53F, 0x0001),
    (0x01D540, 0x0004),
    (0x01D545, 0x0001),
    (0x01D546, 0x0004),
    (0x01D547, 0x0001),
    (0x01D54A, 0x0004),
    (0x01D551, 0x0001),
    (0x01D552, 0x0004),
    (0x01D6A6, 0x0001),
    (0x01D6A8, 0x0004),
    (0x01D6C1, 0x0040),
    (0x01D6C2, 0x0004),
    (0x01D6DB, 0x0040),
    (0x01D6DC, 0x0004),
    (0x01D6FB, 0x0040),
    (0x01D6FC, 0x0004),
    (0x01D715, 0x0040),
    (0x01D716, 0x0004),
    (0x01D735, 0x0040),
    (0x01D736, 0x0004),
    (0x01D74F, 0x0040),
    (0x01D750, 0x0004),
    (0x01D76F, 0x0040),
    (0x01D770, 0x0004),
    (0x01D789, 0x0040),
    (0x01D78A, 0x0004),
    (0x01D7A9, 0x0040),
    (0x01D7AA, 0x0004),
    (0x01D7C3, 0x0040),
    (0x01D7C4, 0x0004),
    (0x01D7CC, 0x0001),
    (0x01D7CE, 0x0002),
    (0x01D800, 0x0040),
    (0x01DA00, 0x0010),
    (0x01DA37, 0x0040),
    (0x01DA3B, 0x0010),
    (0x01DA6D, 0x0040),
    (0x01DA75, 0x0010),
    (0x01DA76, 0x0040),
    (0x01DA84, 0x0010),
    (0x01DA85, 0x0040),
    (0x01DA87, 0x0020),
    (0x01DA8C, 0x0001),
    (0x01DA9B, 0x0010),
    (0x01DAA0, 0x0001),
    (0x01DAA1, 0x0010),
    (0x01DAB0, 0x0001),
    (0x01DF00, 0x0004),
    (0x01DF1F, 0x0001),
    (0x01DF25, 0x0004),
    (0x01DF2B, 0x0001),
    (0x01E000, 0x0010),
    (0x01E007, 0x0001),
    (0x01E008, 0x0010),
    (0x01E019, 0x0001),
    (0x01E01B, 0x0010),
    (0x01E022, 0x0001),
    (0x01E023, 0x0010),
    (0x01E025, 0x0001),
    (0x01E026, 0x0010),
    (0x01E02B, 0x0001),
    (0x01E030, 0x0004),
    (0x01E06E, 0x0001),
    (0x01E08F, 0x0010),
    (0x01E090, 0x0001),
    (0x01E100, 0x0004),
    (0x01E12D, 0x0001),
    (0x01E130, 0x0010),
    (0x01E137, 0x0004),
    (0x01E13E, 0x0001),
    (0x01E140, 0x0002),
    (0x01E14A, 0x0001),
    (0x01E14E, 0x0004),
    (0x01E14F, 0x0040),
    (0x01E150, 0x0001),
    (0x01E290, 0x0004),
    (0x01E2AE, 0x0010),
    (0x01E2AF, 0x0001),
    (0x01E2C0, 0x0004),
    (0x01E2EC, 0x0010),
    (0x01E2F0, 0x0002),
    (0x01E2FA, 0x0001),
    (0x01E2FF, 0x0040),
    (0x01E300, 0x0001),
    (0x01E4D0, 0x0004),
    (0x01E4EC, 0x0010),
    (0x01E4F0, 0x0002),
    (0x01E4FA, 0x0001),
    (0x01E7E0, 0x0004),
    (0x01E7E7, 0x0001),
    (0x01E7E8, 0x0004),
    (0x01E7EC, 0x0001),
    (0x01E7ED, 0x0004),
    (0x01E7EF, 0x0001),
    (0x01E7F0, 0x0004),
    (0x01E7FF, 0x0001),
    (0x01E800, 0x0004),
    (0x01E8C5, 0x0001),
    (0x01E8C7, 0x0002),
    (0x01E8D0, 0x0010),
    (0x01E8D7, 0x0001),
    (0x01E900, 0x0004),
    (0x01E944, 0x0010),
    (0x01E94B, 0x0004),
    (0x01E94C, 0x0001),
    (0x01E950, 0x0002),
    (0x01E95A, 0x0001),
    (0x01E95E, 0x0020),
    (0x01E960, 0x0001),
    (0x01EC71, 0x0002),
    (0x01ECAC, 0x0040),
    (0x01ECAD, 0x0002),
    (0x01ECB0, 0x0040),
    (0x01ECB1, 0x0002),
    (0x01ECB5, 0x0001),
    (0x01ED01, 0x0002),
    (0x01ED2E, 0x0040),
    (0x01ED2F, 0x0002),
    (0x01ED3E, 0x0001),
    (0x01EE00, 0x0004),
    (0x01EE04, 0x0001),
    (0x01EE05, 0x0004),
    (0x01EE20, 0x0001),
    (0x01EE21, 0x0004),
    (0x01EE23, 0x0001),
    (0x01EE24, 0x0004),
    (0x01EE25, 0x0001),
    (0x01EE27, 0x0004),
    (0x01EE28, 0x0001),
    (0x01EE29, 0x0004),
    (0x01EE33, 0x0001),
    (0x01EE34, 0x0004),
    (0x01EE38, 0x0001),
    (0x01EE39, 0x0004),
    (0x01EE3A, 0x0001),
    (0x01EE3B, 0x0004),
    (0x01EE3C, 0x0001),
    (0x01EE42, 0x0004),
    (0x01EE43, 0x0001),
    (0x01EE47, 0x0004),
    (0x01EE48, 0x0001),
    (0x01EE49, 0x0004),
    (0x01EE4A, 0x0001),
    (0x01EE4B, 0x0004),
    (0x01EE4C, 0x0001),
    (0x01EE4D, 0x0004),
    (0x01EE50, 0x0001),
    (0x01EE51, 0x0004),
    (0x01EE53, 0x0001),
    (0x01EE54, 0x0004),
    (0x01EE55, 0x0001),
    (0x01EE57, 0x0004),
    (0x01EE58, 0x0001),
    (0x01EE59, 0x0004),
    (0x01EE5A, 0x0001),
    (0x01EE5B, 0x0004),
    (0x01EE5C, 0x0001),
    (0x01EE5D, 0x0004),
    (0x01EE5E, 0x0001),
    (0x01EE5F, 0x0004),
    (0x01EE60, 0x0001),
    (0x01EE61, 0x0004),
    (0x01EE63, 0x0001),
    (0x01EE64, 0x0004),
    (0x01EE65, 0x0001),
    (0x01EE67, 0x0004),
    (0x01EE6B, 0x0001),
    (0x01EE6C, 0x0004),
    (0x01EE73, 0x0001),
    (0x01EE74, 0x0004),
    (0x01EE78, 0x0001),
    (0x01EE79, 0x0004),
    (0x01EE7D, 0x0001),
    (0x01EE7E, 0x0004),
    (0x01EE7F, 0x0001),
    (0x01EE80, 0x0004),
    (0x01EE8A, 0x0001),
    (0x01EE8B, 0x0004),
    (0x01EE9C, 0x0001),
    (0x01EEA1, 0x0004),
    (0x01EEA4, 0x0001),
    (0x01EEA5, 0x0004),
    (0x01EEAA, 0x0001),
    (0x01EEAB, 0x0004),
    (0x01EEBC, 0x0001),
    (0x01EEF0, 0x0040),
    (0x01EEF2, 0x0001),
    (0x01F000, 0x0040),
    (0x01F02C, 0x0001),
    (0x01F030, 0x0040),
    (0x01F094, 0x0001),
    (0x01F0A0, 0x0040),
    (0x01F0AF, 0x0001),
    (0x01F0B1, 0x0040),
    (0x01F0C0, 0x0001),
    (0x01F0C1, 0x0040),
    (0x01F0D0, 0x0001),
    (0x01F0D1, 0x0040),
    (0x01F0F6, 0x0001),
    (0x01F100, 0x0002),
    (0x01F10D, 0x0040),
    (0x01F1AE, 0x0001),
    (0x01F1E6, 0x0040),
    (0x01F203, 0x0001),
    (0x01F210, 0x0040),
    (0x01F23C, 0x0001),
    (0x01F240, 0x0040),
    (0x01F249, 0x0001),
    (0x01F250, 0x0040),
    (0x01F252, 0x0001),
    (0x01F260, 0x0040),
    (0x01F266, 0x0001),
    (0x01F300, 0x0040),
    (0x01F6D8, 0x0001),
    (0x01F6DC, 0x0040),
    (0x01F6ED, 0x0001),
    (0x01F6F0, 0x0040),
    (0x01F6FD, 0x0001),
    (0x01F700, 0x0040),
    (0x01F777, 0x0001),
    (0x01F77B, 0x0040),
    (0x01F7DA, 0x0001),
    (0x01F7E0, 0x0040),
    (0x01F7EC, 0x0001),
    (0x01F7F0, 0x0040),
    (0x01F7F1, 0x0001),
    (0x01F800, 0x0040),
    (0x01F80C, 0x0001),
    (0x01F810, 0x0040),
    (0x01F848, 0x0001),
    (0x01F850, 0x0040),
    (0x01F85A, 0x0001),
    (0x01F860, 0x0040),
    (0x01F888, 0x0001),
    (0x01F890, 0x0040),
    (0x01F8AE, 0x0001),
    (0x01F8B0, 0x0040),
    (0x01F8B2, 0x0001),
    (0x01F900, 0x0040),
    (0x01FA54, 0x0001),
    (0x01FA60, 0x0040),
    (0x01FA6E, 0x0001),
    (0x01FA70, 0x0040),
    (0x01FA7D, 0x0001),
    (0x01FA80, 0x0040),
    (0x01FA89, 0x0001),
    (0x01FA90, 0x0040),
    (0x01FABE, 0x0001),
    (0x01FABF, 0x0040),
    (0x01FAC6, 0x0001),
    (0x01FACE, 0x0040),
    (0x01FADC, 0x0001),
    (0x01FAE0, 0x0040),
    (0x01FAE9, 0x0001),
    (0x01FAF0, 0x0040),
    (0x01FAF9, 0x0001),
    (0x01FB00, 0x0040),
    (0x01FB93, 0x0001),
    (0x01FB94, 0x0040),
    (0x01FBCB, 0x0001),
    (0x01FBF0, 0x0002),
    (0x01FBFA, 0x0001),
    (0x020000, 0x0004),
    (0x02A6E0, 0x0001),
    (0x02A700, 0x0004),
    (0x02B73A, 0x0001),
    (0x02B740, 0x0004),
    (0x02B81E, 0x0001),
    (0x02B820, 0x0004),
    (0x02CEA2, 0x0001),
    (0x02CEB0, 0x0004),
    (0x02EBE1, 0x0001),
    (0x02EBF0, 0x0004),
    (0x02EE5E, 0x0001),
    (0x02F800, 0x0004),
    (0x02FA1E, 0x0001),
    (0x030000, 0x0004),
    (0x03134B, 0x0001),
    (0x031350, 0x0004),
    (0x0323B0, 0x0001),
    (0x0E0001, 0x0080),
    (0x0E0002, 0x0001),
    (0x0E0020, 0x0080),
    (0x0E0080, 0x0001),
    (0x0E0100, 0x0010),
    (0x0E01F0, 0x0001),
    (0x0F0000, 0x0080),
    (0x0FFFFE, 0x0001),
    (0x100000, 0x0080),
    (0x10FFFE, 0x0001),
    (0x110000, 0x0000),
];

#[rustfmt::skip]
pub(crate) const WHITESPACE_SET: &[u32] = &[
    0x000009,
    0x00000A,
    0x00000B,
    0x00000C,
    0x00000D,
    0x000020,
    0x000085,
    0x0000A0,
    0x001680,
    0x002000,
    0x002001,
    0x002002,
    0x002003,
    0x002004,
    0x002005,
    0x002006,
    0x002007,
    0x002008,
    0x002009,
    0x00200A,
    0x002028,
    0x002029,
    0x00202F,
    0x00205F,
    0x003000,
];

#[rustfmt::skip]
/// GPT2 byte-to-display table (`unicode_byte_to_utf8_map`, llama.cpp 10150).
pub(crate) const BYTE_TO_DISPLAY: [char; 256] = [
    '\u{0100}', '\u{0101}', '\u{0102}', '\u{0103}', '\u{0104}', '\u{0105}', '\u{0106}', '\u{0107}',
    '\u{0108}', '\u{0109}', '\u{010a}', '\u{010b}', '\u{010c}', '\u{010d}', '\u{010e}', '\u{010f}',
    '\u{0110}', '\u{0111}', '\u{0112}', '\u{0113}', '\u{0114}', '\u{0115}', '\u{0116}', '\u{0117}',
    '\u{0118}', '\u{0119}', '\u{011a}', '\u{011b}', '\u{011c}', '\u{011d}', '\u{011e}', '\u{011f}',
    '\u{0120}', '\u{0021}', '\u{0022}', '\u{0023}', '\u{0024}', '\u{0025}', '\u{0026}', '\u{0027}',
    '\u{0028}', '\u{0029}', '\u{002a}', '\u{002b}', '\u{002c}', '\u{002d}', '\u{002e}', '\u{002f}',
    '\u{0030}', '\u{0031}', '\u{0032}', '\u{0033}', '\u{0034}', '\u{0035}', '\u{0036}', '\u{0037}',
    '\u{0038}', '\u{0039}', '\u{003a}', '\u{003b}', '\u{003c}', '\u{003d}', '\u{003e}', '\u{003f}',
    '\u{0040}', '\u{0041}', '\u{0042}', '\u{0043}', '\u{0044}', '\u{0045}', '\u{0046}', '\u{0047}',
    '\u{0048}', '\u{0049}', '\u{004a}', '\u{004b}', '\u{004c}', '\u{004d}', '\u{004e}', '\u{004f}',
    '\u{0050}', '\u{0051}', '\u{0052}', '\u{0053}', '\u{0054}', '\u{0055}', '\u{0056}', '\u{0057}',
    '\u{0058}', '\u{0059}', '\u{005a}', '\u{005b}', '\u{005c}', '\u{005d}', '\u{005e}', '\u{005f}',
    '\u{0060}', '\u{0061}', '\u{0062}', '\u{0063}', '\u{0064}', '\u{0065}', '\u{0066}', '\u{0067}',
    '\u{0068}', '\u{0069}', '\u{006a}', '\u{006b}', '\u{006c}', '\u{006d}', '\u{006e}', '\u{006f}',
    '\u{0070}', '\u{0071}', '\u{0072}', '\u{0073}', '\u{0074}', '\u{0075}', '\u{0076}', '\u{0077}',
    '\u{0078}', '\u{0079}', '\u{007a}', '\u{007b}', '\u{007c}', '\u{007d}', '\u{007e}', '\u{0121}',
    '\u{0122}', '\u{0123}', '\u{0124}', '\u{0125}', '\u{0126}', '\u{0127}', '\u{0128}', '\u{0129}',
    '\u{012a}', '\u{012b}', '\u{012c}', '\u{012d}', '\u{012e}', '\u{012f}', '\u{0130}', '\u{0131}',
    '\u{0132}', '\u{0133}', '\u{0134}', '\u{0135}', '\u{0136}', '\u{0137}', '\u{0138}', '\u{0139}',
    '\u{013a}', '\u{013b}', '\u{013c}', '\u{013d}', '\u{013e}', '\u{013f}', '\u{0140}', '\u{0141}',
    '\u{0142}', '\u{00a1}', '\u{00a2}', '\u{00a3}', '\u{00a4}', '\u{00a5}', '\u{00a6}', '\u{00a7}',
    '\u{00a8}', '\u{00a9}', '\u{00aa}', '\u{00ab}', '\u{00ac}', '\u{0143}', '\u{00ae}', '\u{00af}',
    '\u{00b0}', '\u{00b1}', '\u{00b2}', '\u{00b3}', '\u{00b4}', '\u{00b5}', '\u{00b6}', '\u{00b7}',
    '\u{00b8}', '\u{00b9}', '\u{00ba}', '\u{00bb}', '\u{00bc}', '\u{00bd}', '\u{00be}', '\u{00bf}',
    '\u{00c0}', '\u{00c1}', '\u{00c2}', '\u{00c3}', '\u{00c4}', '\u{00c5}', '\u{00c6}', '\u{00c7}',
    '\u{00c8}', '\u{00c9}', '\u{00ca}', '\u{00cb}', '\u{00cc}', '\u{00cd}', '\u{00ce}', '\u{00cf}',
    '\u{00d0}', '\u{00d1}', '\u{00d2}', '\u{00d3}', '\u{00d4}', '\u{00d5}', '\u{00d6}', '\u{00d7}',
    '\u{00d8}', '\u{00d9}', '\u{00da}', '\u{00db}', '\u{00dc}', '\u{00dd}', '\u{00de}', '\u{00df}',
    '\u{00e0}', '\u{00e1}', '\u{00e2}', '\u{00e3}', '\u{00e4}', '\u{00e5}', '\u{00e6}', '\u{00e7}',
    '\u{00e8}', '\u{00e9}', '\u{00ea}', '\u{00eb}', '\u{00ec}', '\u{00ed}', '\u{00ee}', '\u{00ef}',
    '\u{00f0}', '\u{00f1}', '\u{00f2}', '\u{00f3}', '\u{00f4}', '\u{00f5}', '\u{00f6}', '\u{00f7}',
    '\u{00f8}', '\u{00f9}', '\u{00fa}', '\u{00fb}', '\u{00fc}', '\u{00fd}', '\u{00fe}', '\u{00ff}',
];
