//! GPT2 BPE merge stage for the pinned gpt2 row.
//!
//! Mirrors llama.cpp 10150 `llm_tokenizer_bpe` + `llm_tokenizer_bpe_session`
//! (reference details in the parent module docs). Given one GPT2 byte-encoded
//! (display) fragment, symbols start as the fragment's UTF-8 chars; the bigram
//! with the smallest merge rank (ties → smallest left index) is merged first.
//! Ranks come from `tokenizer.ggml.merges` in order. A symbol whose display
//! text is not a vocab token falls back to per-raw-byte token lookups exactly
//! like llama.cpp.
//!
//! The tokenizer's vocab maps and merge ranks are passed in rather than taken
//! by `&self`, keeping the merge loop a pure `display -> tokens` function.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::{RankMap, TokenIdMap, TokenizerError, MAX_ENCODE_OUTPUT_TOKENS};

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

/// BPE-encode one byte-encoded (display) fragment, appending ids.
pub(super) fn bpe_encode_word(
    display: &str,
    token_to_id: &TokenIdMap,
    bpe_ranks: &RankMap,
    byte_fallback: &[u32; 256],
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
                next: if index + 1 == n_chars {
                    -1
                } else {
                    index as isize + 1
                },
                start: byte_cursor,
                n: len,
            });
            byte_cursor = checked_add(byte_cursor, len, "symbol byte cursor")?;
        }
    }

    let mut queue: BinaryHeap<QueueEntry> = BinaryHeap::new();
    let add_bigram = |queue: &mut BinaryHeap<QueueEntry>,
                      symbols: &[Symbol],
                      ranks: &RankMap,
                      left: isize,
                      right: isize| {
        if left < 0 || right < 0 {
            return;
        }
        let l = left as usize;
        let r = right as usize;
        let ltext = &display[symbols[l].start..symbols[l].start + symbols[l].n];
        let rtext = &display[symbols[r].start..symbols[r].start + symbols[r].n];
        if let Some(&rank) = ranks.get(&(ltext.as_bytes().into(), rtext.as_bytes().into())) {
            let mut text = String::with_capacity(ltext.len() + rtext.len());
            text.push_str(ltext);
            text.push_str(rtext);
            queue.push(QueueEntry {
                rank,
                left: l,
                right: r,
                text,
            });
        }
    };

    for i in 1..n_chars {
        add_bigram(
            &mut queue,
            &symbols,
            bpe_ranks,
            i as isize - 1,
            i as isize,
        );
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
        add_bigram(
            &mut queue,
            &symbols,
            bpe_ranks,
            prev_of_left,
            entry.left as isize,
        );
        add_bigram(
            &mut queue,
            &symbols,
            bpe_ranks,
            entry.left as isize,
            next_of_right,
        );
    }

    // Collect alive symbols in chain order (llama.cpp `symbols_final` walk).
    let mut idx: isize = 0;
    loop {
        let symbol = &symbols[idx as usize];
        if symbol.n > 0 {
            let text = &display[symbol.start..symbol.start + symbol.n];
            match token_to_id.get(text.as_bytes()) {
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
                        let id = byte_fallback[byte as usize];
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
