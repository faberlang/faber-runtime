//! tokenizer — gpt2 BPE + `smollm` pre-tokenizer for the pinned SmolLM2 row.
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
//! ## Layout
//!
//! The stage logic is split into two private sub-modules, keeping this file to
//! the pinned-row facts, fail-closed construction, and encode orchestration:
//!
//! - `pretoken` — steps 2–3: the `smollm` regex splitter, GPT2 byte encoding,
//!   and the embedded Unicode flag + byte-to-display tables.
//! - `bpe` — step 4: the merge loop (`bpe_encode_word`) with its symbol /
//!   merge-queue data model; a pure `display -> tokens` function over the
//!   tokenizer's vocab maps (`TokenIdMap` / `RankMap`).
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
//! - Codepoint flags (`CPT_RANGES`, `WHITESPACE_SET`) and `BYTE_TO_DISPLAY`
//!   live in `pretoken`; they are generated from llama.cpp 10150
//!   `src/unicode-data.cpp` (`unicode_ranges_flags` + `unicode_set_whitespace`)
//!   and the GPT2 byte-to-unicode map (`unicode_byte_to_utf8_map`) — the exact
//!   data the pinned comparator executes.

use std::collections::HashMap;

use crate::gguf::{GgufAdmission, MetadataValue};

mod bpe;
mod pretoken;
use bpe::bpe_encode_word;
use pretoken::{byte_encode, pre_tokenize_smollm};

/// Token text (display bytes) → id.
pub(super) type TokenIdMap = HashMap<Box<[u8]>, u32>;
/// BPE merge pair → rank (index in `tokenizer.ggml.merges`; first wins).
pub(super) type RankMap = HashMap<(Box<[u8]>, Box<[u8]>), u32>;

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
                write!(
                    f,
                    "tokenizer.ggml.scores is present; the pinned row has no scores"
                )
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
                write!(
                    f,
                    "add_bos_token = {value}; the pinned row is add_bos_token = false"
                )
            }
            TokenizerError::AddSpacePrefixNotSupported { value } => {
                write!(
                    f,
                    "add_space_prefix = {value}; the pinned row is add_space_prefix = false"
                )
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
                write!(
                    f,
                    "admission tokenizer metadata key {key} has the wrong value type"
                )
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
    token_to_id: TokenIdMap,
    /// BPE merge pair → rank (index in `tokenizer.ggml.merges`; first wins).
    bpe_ranks: RankMap,
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
            token_to_id
                .entry(bytes.to_vec().into_boxed_slice())
                .or_insert(id as u32);
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
                .entry((
                    left.to_vec().into_boxed_slice(),
                    right.to_vec().into_boxed_slice(),
                ))
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
        for (id, (text, ttype)) in facts
            .tokens
            .iter()
            .zip(facts.token_types.iter())
            .enumerate()
        {
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
        let mut fragments: Vec<Fragment> = vec![Fragment::Text {
            start: 0,
            len: text.len(),
        }];

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
                    let fragment_text =
                        text.get(start..start + len)
                            .ok_or(TokenizerError::ArithmeticOverflow {
                                what: "fragment slice",
                            })?;
                    let spans = pre_tokenize_smollm(fragment_text);
                    for (b_start, b_len) in spans {
                        let word = fragment_text
                            .get(b_start..b_start + b_len)
                            .ok_or(TokenizerError::ArithmeticOverflow { what: "word slice" })?;
                        let display = byte_encode(word);
                        bpe_encode_word(
                            &display,
                            &self.token_to_id,
                            &self.bpe_ranks,
                            &self.byte_fallback,
                            &mut output,
                        )?;
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
//
// The `smollm` pre-tokenizer and the GPT2 BPE merge loop live in the private
// `pretoken` / `bpe` sub-modules; this file keeps only the partition types.
// ---------------------------------------------------------------------------

/// A post-partition fragment: a raw-text byte span or a special-token id.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fragment {
    Token(u32),
    Text { start: usize, len: usize },
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
