//! tokenizer_test.rs — probe-parity + fail-closed tests for `tokenizer.rs`.
//!
//! Every probe vector from `evidence/contract-tokenize-probes.txt` (P1–P11) and
//! the four workload prompt token-id lists (`gi0-workloads.md` §3) is embedded
//! below as a test fixture. The tokenizer is built from the GI1-1 admission of
//! the pinned SmolLM2 row (the vocab/merges are never re-discovered here); the
//! row is local-only (model contract §8), so these tests need the pinned file.

use std::sync::OnceLock;

use crate::tokenizer::*;

/// Pinned model row (gi0-model-contract v1.0.0 §1; local-only per §8).
const PINNED_MODEL_PATH: &str =
    "/Users/ianzepp/ai/models/SmolLM2-360M-Instruct-Q4_K_M.gguf";

/// Build the tokenizer from the pinned row's admission (cached per process).
fn pinned_tokenizer() -> &'static Gpt2BpeTokenizer {
    static TOK: OnceLock<Gpt2BpeTokenizer> = OnceLock::new();
    TOK.get_or_init(|| {
        let admission = crate::gguf::admit_file(PINNED_MODEL_PATH).expect(
            "pinned SmolLM2 GGUF must be present for probe-parity tests \
             (gi0-model-contract v1.0.0: local-only row)",
        );
        Gpt2BpeTokenizer::from_admission(&admission)
            .expect("admission intake of the pinned tokenizer facts must succeed")
    })
}

/// A minimal but valid fact set for fail-closed negatives (counts exact).
fn synthetic_facts() -> TokenizerFacts {
    let tokens: Vec<String> = (0..EXPECTED_VOCAB_SIZE).map(|i| format!("tok{i}")).collect();
    let token_types: Vec<i32> = vec![crate::tokenizer::TOKEN_TYPE_NORMAL; EXPECTED_VOCAB_SIZE];
    let merges: Vec<String> = (0..EXPECTED_MERGES).map(|_| "a b".to_string()).collect();
    TokenizerFacts {
        model: TOKENIZER_MODEL.to_string(),
        pre: TOKENIZER_PRE.to_string(),
        tokens,
        token_types,
        merges,
        scores_present: false,
        bos_token_id: BOS_TOKEN_ID,
        eos_token_id: EOS_TOKEN_ID,
        pad_token_id: PAD_TOKEN_ID,
        unk_token_id: UNK_TOKEN_ID,
        add_bos_token: ADD_BOS_TOKEN,
        add_space_prefix: ADD_SPACE_PREFIX,
    }
}

// ---------------------------------------------------------------------------
// Probes P1–P11 (evidence/contract-tokenize-probes.txt)
// ---------------------------------------------------------------------------

#[test]
fn probe_p1_hello_world_parse_special() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("Hello world", true).unwrap(), vec![19556, 905]);
}

#[test]
fn probe_p2_hello_world_no_bos() {
    // P1 == P2: the tokenizer never auto-prepends BOS (add_bos_token = false).
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("Hello world", false).unwrap(), vec![19556, 905]);
}

#[test]
fn probe_p3_hello_world_count() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("Hello world", true).unwrap().len(), 2);
}

#[test]
fn probe_p4_empty_prompt() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("", true).unwrap(), Vec::<u32>::new());
    assert_eq!(tok.encode("", false).unwrap(), Vec::<u32>::new());
}

#[test]
fn probe_p5_im_start_special() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("<|im_start|>", true).unwrap(), vec![1]);
}

#[test]
fn probe_p6_im_start_literal() {
    // --no-parse-special: the control special is BPE-encoded as literal text.
    let tok = pinned_tokenizer();
    assert_eq!(
        tok.encode("<|im_start|>", false).unwrap(),
        vec![44, 108, 306, 79, 3738, 108, 46]
    );
}

#[test]
fn probe_p7_im_end_special() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("<|im_end|>", true).unwrap(), vec![2]);
}

#[test]
fn probe_p8_endoftext_special() {
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("<|endoftext|>", true).unwrap(), vec![0]);
}

#[test]
fn probe_p9_chat_template_prefix() {
    let tok = pinned_tokenizer();
    let prompt = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n";
    let expected: Vec<u32> = vec![
        1, 9690, 198, 2683, 359, 253, 5356, 5646, 11173, 3365, 3511, 308, 34519, 28, 7018, 411,
        407, 19712, 8182, 2, 198, 1, 4093, 198, 26843, 665, 2, 198, 1, 520, 9531, 198,
    ];
    let ids = tok.encode(prompt, true).unwrap();
    assert_eq!(ids.len(), 32);
    assert_eq!(ids, expected);
}

#[test]
fn probe_p10_lazy_dog() {
    let tok = pinned_tokenizer();
    assert_eq!(
        tok.encode("The quick brown fox jumps over the lazy dog", true).unwrap(),
        vec![504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767]
    );
}

#[test]
fn probe_p11_leading_spaces() {
    // add_space_prefix = false: no implicit leading-space merge.
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("  leading spaces", true).unwrap(), vec![216, 2899, 5600]);
}

// ---------------------------------------------------------------------------
// Workload prompt token-id lists (gi0-workloads.md §3)
// ---------------------------------------------------------------------------

#[test]
fn workload_correctness_prompt() {
    let tok = pinned_tokenizer();
    let prompt = "The quick brown fox jumps over the lazy dog";
    assert_eq!(prompt.len(), 43);
    let ids = tok.encode(prompt, true).unwrap();
    assert_eq!(ids.len(), 9);
    assert_eq!(ids, vec![504, 2365, 6354, 16438, 27003, 690, 260, 23790, 2767]);
}

#[test]
fn workload_short_prompt() {
    let tok = pinned_tokenizer();
    let prompt = "Write a haiku about the ocean.\n";
    assert_eq!(prompt.len(), 31);
    let ids = tok.encode(prompt, true).unwrap();
    assert_eq!(ids.len(), 9);
    assert_eq!(ids, vec![19161, 253, 421, 30614, 563, 260, 5065, 30, 198]);
}

#[test]
fn workload_normal_prompt() {
    let tok = pinned_tokenizer();
    let prompt = "\
The Radix compiler reads Faber source and lowers it through the grammar, the semantic analysis, and the code generator into the GGUF runtime format. The runtime executes quantized neural network models on Apple Silicon using the Metal backend. Metal dispatch works well for matrix operations and for the fused attention kernel. The model weights stay in the Q4_K_M block layout, which the Metal shaders decode at load time. Inference runs through the Llama server process, which exposes an HTTP completion endpoint and a Prometheus metrics surface. The prompt phase and the decode phase are measured separately so the headline throughput number never mixes load time or prefill with generation. A short context workload exercises the prefill path with a modest number of tokens, while a normal workload exercises a full batch of attention heads over a paragraph of text. Every benchmark run is executed against the pinned comparator revision so that results are reproducible, and every workload file is hash-pinned before any measurement is taken.\
";
    assert_eq!(prompt.len(), 1047);
    let ids = tok.encode(prompt, true).unwrap();
    assert_eq!(ids.len(), 202);
    let expected: Vec<u32> = vec![
        504, 7080, 1088, 25316, 12927, 18411, 259, 2257, 284, 24208, 357, 738, 260, 10560,
    28, 260, 23864, 2318, 28, 284, 260, 2909, 12914, 618, 260, 452, 55, 69,
    54, 29709, 4624, 30, 378, 29709, 45438, 3324, 1005, 8844, 3082, 2859, 335, 10910,
    32688, 1015, 260, 24672, 25817, 30, 24672, 37736, 1806, 876, 327, 6736, 4261, 284,
    327, 260, 30167, 2674, 11498, 30, 378, 1743, 10379, 2951, 281, 260, 1606, 36,
    79, 59, 79, 61, 3608, 10220, 28, 527, 260, 24672, 7555, 366, 26420, 418,
    3509, 655, 30, 5883, 2095, 7313, 738, 260, 450, 4130, 81, 6064, 980, 28,
    527, 28263, 354, 17108, 11423, 22223, 284, 253, 10193, 41880, 12793, 2376, 30, 378,
    6011, 5239, 284, 260, 26420, 5239, 359, 6090, 13624, 588, 260, 32045, 34349, 1230,
    2093, 28602, 3509, 655, 355, 4478, 388, 351, 4686, 30, 330, 1890, 2468, 29278,
    6379, 260, 4478, 388, 2050, 351, 253, 15665, 1230, 282, 17837, 28, 979, 253,
    2955, 29278, 6379, 253, 2073, 7717, 282, 2674, 8648, 690, 253, 8510, 282, 1694,
    30, 5081, 26211, 1658, 314, 11834, 1523, 260, 8268, 3924, 4207, 1508, 14204, 588,
    338, 1844, 359, 41088, 28, 284, 897, 29278, 2301, 314, 15345, 29, 9955, 3924,
    1092, 750, 7585, 314, 2473, 30
    ];
    assert_eq!(ids, expected);
}

#[test]
fn workload_context_prompt() {
    let tok = pinned_tokenizer();
    let prompt = "\
Faber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\nFaber is a systems language for building AI applications. The compiler pipeline is split into distinct stages: lexing, parsing, name resolution, type checking, lowering to MIR, and finally code generation. Each stage operates on an immutable intermediate representation and reports diagnostics with precise source locations. For inference workloads, Faber programs call into the inference runtime, which loads quantized model files and dispatches tensor operations to the most capable backend available. On Apple Silicon the Metal backend is preferred because it keeps large matrices and fused attention kernels resident in unified memory. The runtime exposes a small, typed API surface so that application code does not depend on any particular backend implementation. Quantized weights follow the GGUF block layout, and the runtime decodes each block format during the matrix multiply. This design keeps the critical path short: a prompt is embedded into tokens, the tokens are passed to the model, and the sampled output is streamed back to the caller. None of the intermediate steps requires copying data across process boundaries, because the whole pipeline runs in one process with a shared context. This paragraph is deliberately neutral so that repeated use in a long context workload does not bias the model toward any particular topic. It exists only to provide a deterministic amount of context text for the benchmark, and it is assembled from stable ASCII bytes that reproduce exactly.\n\
";
    assert_eq!(prompt.len(), 11991);
    let ids = tok.encode(prompt, true).unwrap();
    assert_eq!(ids.len(), 2175);
    let expected: Vec<u32> = vec![
        54, 369, 259, 314, 253, 1734, 1789, 327, 2194, 5646, 3253, 30, 378, 25316,
    13281, 314, 7074, 618, 4073, 5933, 42, 19839, 274, 28, 34868, 28, 1462, 6558,
    28, 1502, 11160, 28, 16067, 288, 372, 5810, 28, 284, 5087, 2909, 4686, 30,
    3768, 3632, 13662, 335, 354, 44771, 14019, 6133, 284, 4631, 28713, 351, 8212, 2257,
    6332, 30, 1068, 23630, 746, 12936, 28, 18411, 259, 2774, 946, 618, 260, 23630,
    29709, 28, 527, 13101, 3324, 1005, 1743, 4577, 284, 27795, 2476, 13104, 4261, 288,
    260, 768, 5181, 25817, 1770, 30, 1985, 10910, 32688, 260, 24672, 25817, 314, 8996,
    975, 357, 8211, 1507, 18771, 284, 30167, 2674, 32528, 7878, 281, 19775, 3500, 30,
    378, 29709, 28263, 253, 1165, 28, 30473, 12077, 2376, 588, 338, 3279, 2909, 1072,
    441, 3749, 335, 750, 1542, 25817, 6230, 30, 14980, 1005, 10379, 1066, 260, 452,
    55, 69, 54, 3608, 10220, 28, 284, 260, 29709, 988, 3237, 971, 3608, 4624,
    981, 260, 6736, 17109, 30, 669, 1157, 8211, 260, 2609, 2050, 1890, 42, 253,
    6011, 314, 10717, 618, 17837, 28, 260, 17837, 359, 4180, 288, 260, 1743, 28,
    284, 260, 23146, 3124, 314, 2198, 2520, 1056, 288, 260, 38448, 30, 1943, 282,
    260, 14019, 3301, 3073, 23124, 940, 1699, 980, 6177, 28, 975, 260, 2444, 13281,
    7313, 281, 582, 980, 351, 253, 3600, 2468, 30, 669, 8510, 314, 18519, 9174,
    588, 338, 6514, 722, 281, 253, 986, 2468, 29278, 1072, 441, 8542, 260, 1743,
    1731, 750, 1542, 4234, 30, 657, 5961, 805, 288, 1538, 253, 43524, 1902, 282,
    2468, 1694, 327, 260, 26211, 28, 284, 357, 314, 16551, 429, 7094, 39799, 15590,
    338, 15121, 3869, 30, 198, 198, 54, 369, 259, 314, 253, 1734, 1789, 327,
    2194, 5646, 3253, 30, 378, 25316, 13281, 314, 7074, 618, 4073, 5933, 42, 19839,
    274, 28, 34868, 28, 1462, 6558, 28, 1502, 11160, 28, 16067, 288, 372, 5810,
    28, 284, 5087, 2909, 4686, 30, 3768, 3632, 13662, 335, 354, 44771, 14019, 6133,
    284, 4631, 28713, 351, 8212, 2257, 6332, 30, 1068, 23630, 746, 12936, 28, 18411,
    259, 2774, 946, 618, 260, 23630, 29709, 28, 527, 13101, 3324, 1005, 1743, 4577,
    284, 27795, 2476, 13104, 4261, 288, 260, 768, 5181, 25817, 1770, 30, 1985, 10910,
    32688, 260, 24672, 25817, 314, 8996, 975, 357, 8211, 1507, 18771, 284, 30167, 2674,
    32528, 7878, 281, 19775, 3500, 30, 378, 29709, 28263, 253, 1165, 28, 30473, 12077,
    2376, 588, 338, 3279, 2909, 1072, 441, 3749, 335, 750, 1542, 25817, 6230, 30,
    14980, 1005, 10379, 1066, 260, 452, 55, 69, 54, 3608, 10220, 28, 284, 260,
    29709, 988, 3237, 971, 3608, 4624, 981, 260, 6736, 17109, 30, 669, 1157, 8211,
    260, 2609, 2050, 1890, 42, 253, 6011, 314, 10717, 618, 17837, 28, 260, 17837,
    359, 4180, 288, 260, 1743, 28, 284, 260, 23146, 3124, 314, 2198, 2520, 1056,
    288, 260, 38448, 30, 1943, 282, 260, 14019, 3301, 3073, 23124, 940, 1699, 980,
    6177, 28, 975, 260, 2444, 13281, 7313, 281, 582, 980, 351, 253, 3600, 2468,
    30, 669, 8510, 314, 18519, 9174, 588, 338, 6514, 722, 281, 253, 986, 2468,
    29278, 1072, 441, 8542, 260, 1743, 1731, 750, 1542, 4234, 30, 657, 5961, 805,
    288, 1538, 253, 43524, 1902, 282, 2468, 1694, 327, 260, 26211, 28, 284, 357,
    314, 16551, 429, 7094, 39799, 15590, 338, 15121, 3869, 30, 198, 198, 54, 369,
    259, 314, 253, 1734, 1789, 327, 2194, 5646, 3253, 30, 378, 25316, 13281, 314,
    7074, 618, 4073, 5933, 42, 19839, 274, 28, 34868, 28, 1462, 6558, 28, 1502,
    11160, 28, 16067, 288, 372, 5810, 28, 284, 5087, 2909, 4686, 30, 3768, 3632,
    13662, 335, 354, 44771, 14019, 6133, 284, 4631, 28713, 351, 8212, 2257, 6332, 30,
    1068, 23630, 746, 12936, 28, 18411, 259, 2774, 946, 618, 260, 23630, 29709, 28,
    527, 13101, 3324, 1005, 1743, 4577, 284, 27795, 2476, 13104, 4261, 288, 260, 768,
    5181, 25817, 1770, 30, 1985, 10910, 32688, 260, 24672, 25817, 314, 8996, 975, 357,
    8211, 1507, 18771, 284, 30167, 2674, 32528, 7878, 281, 19775, 3500, 30, 378, 29709,
    28263, 253, 1165, 28, 30473, 12077, 2376, 588, 338, 3279, 2909, 1072, 441, 3749,
    335, 750, 1542, 25817, 6230, 30, 14980, 1005, 10379, 1066, 260, 452, 55, 69,
    54, 3608, 10220, 28, 284, 260, 29709, 988, 3237, 971, 3608, 4624, 981, 260,
    6736, 17109, 30, 669, 1157, 8211, 260, 2609, 2050, 1890, 42, 253, 6011, 314,
    10717, 618, 17837, 28, 260, 17837, 359, 4180, 288, 260, 1743, 28, 284, 260,
    23146, 3124, 314, 2198, 2520, 1056, 288, 260, 38448, 30, 1943, 282, 260, 14019,
    3301, 3073, 23124, 940, 1699, 980, 6177, 28, 975, 260, 2444, 13281, 7313, 281,
    582, 980, 351, 253, 3600, 2468, 30, 669, 8510, 314, 18519, 9174, 588, 338,
    6514, 722, 281, 253, 986, 2468, 29278, 1072, 441, 8542, 260, 1743, 1731, 750,
    1542, 4234, 30, 657, 5961, 805, 288, 1538, 253, 43524, 1902, 282, 2468, 1694,
    327, 260, 26211, 28, 284, 357, 314, 16551, 429, 7094, 39799, 15590, 338, 15121,
    3869, 30, 198, 198, 54, 369, 259, 314, 253, 1734, 1789, 327, 2194, 5646,
    3253, 30, 378, 25316, 13281, 314, 7074, 618, 4073, 5933, 42, 19839, 274, 28,
    34868, 28, 1462, 6558, 28, 1502, 11160, 28, 16067, 288, 372, 5810, 28, 284,
    5087, 2909, 4686, 30, 3768, 3632, 13662, 335, 354, 44771, 14019, 6133, 284, 4631,
    28713, 351, 8212, 2257, 6332, 30, 1068, 23630, 746, 12936, 28, 18411, 259, 2774,
    946, 618, 260, 23630, 29709, 28, 527, 13101, 3324, 1005, 1743, 4577, 284, 27795,
    2476, 13104, 4261, 288, 260, 768, 5181, 25817, 1770, 30, 1985, 10910, 32688, 260,
    24672, 25817, 314, 8996, 975, 357, 8211, 1507, 18771, 284, 30167, 2674, 32528, 7878,
    281, 19775, 3500, 30, 378, 29709, 28263, 253, 1165, 28, 30473, 12077, 2376, 588,
    338, 3279, 2909, 1072, 441, 3749, 335, 750, 1542, 25817, 6230, 30, 14980, 1005,
    10379, 1066, 260, 452, 55, 69, 54, 3608, 10220, 28, 284, 260, 29709, 988,
    3237, 971, 3608, 4624, 981, 260, 6736, 17109, 30, 669, 1157, 8211, 260, 2609,
    2050, 1890, 42, 253, 6011, 314, 10717, 618, 17837, 28, 260, 17837, 359, 4180,
    288, 260, 1743, 28, 284, 260, 23146, 3124, 314, 2198, 2520, 1056, 288, 260,
    38448, 30, 1943, 282, 260, 14019, 3301, 3073, 23124, 940, 1699, 980, 6177, 28,
    975, 260, 2444, 13281, 7313, 281, 582, 980, 351, 253, 3600, 2468, 30, 669,
    8510, 314, 18519, 9174, 588, 338, 6514, 722, 281, 253, 986, 2468, 29278, 1072,
    441, 8542, 260, 1743, 1731, 750, 1542, 4234, 30, 657, 5961, 805, 288, 1538,
    253, 43524, 1902, 282, 2468, 1694, 327, 260, 26211, 28, 284, 357, 314, 16551,
    429, 7094, 39799, 15590, 338, 15121, 3869, 30, 198, 198, 54, 369, 259, 314,
    253, 1734, 1789, 327, 2194, 5646, 3253, 30, 378, 25316, 13281, 314, 7074, 618,
    4073, 5933, 42, 19839, 274, 28, 34868, 28, 1462, 6558, 28, 1502, 11160, 28,
    16067, 288, 372, 5810, 28, 284, 5087, 2909, 4686, 30, 3768, 3632, 13662, 335,
    354, 44771, 14019, 6133, 284, 4631, 28713, 351, 8212, 2257, 6332, 30, 1068, 23630,
    746, 12936, 28, 18411, 259, 2774, 946, 618, 260, 23630, 29709, 28, 527, 13101,
    3324, 1005, 1743, 4577, 284, 27795, 2476, 13104, 4261, 288, 260, 768, 5181, 25817,
    1770, 30, 1985, 10910, 32688, 260, 24672, 25817, 314, 8996, 975, 357, 8211, 1507,
    18771, 284, 30167, 2674, 32528, 7878, 281, 19775, 3500, 30, 378, 29709, 28263, 253,
    1165, 28, 30473, 12077, 2376, 588, 338, 3279, 2909, 1072, 441, 3749, 335, 750,
    1542, 25817, 6230, 30, 14980, 1005, 10379, 1066, 260, 452, 55, 69, 54, 3608,
    10220, 28, 284, 260, 29709, 988, 3237, 971, 3608, 4624, 981, 260, 6736, 17109,
    30, 669, 1157, 8211, 260, 2609, 2050, 1890, 42, 253, 6011, 314, 10717, 618,
    17837, 28, 260, 17837, 359, 4180, 288, 260, 1743, 28, 284, 260, 23146, 3124,
    314, 2198, 2520, 1056, 288, 260, 38448, 30, 1943, 282, 260, 14019, 3301, 3073,
    23124, 940, 1699, 980, 6177, 28, 975, 260, 2444, 13281, 7313, 281, 582, 980,
    351, 253, 3600, 2468, 30, 669, 8510, 314, 18519, 9174, 588, 338, 6514, 722,
    281, 253, 986, 2468, 29278, 1072, 441, 8542, 260, 1743, 1731, 750, 1542, 4234,
    30, 657, 5961, 805, 288, 1538, 253, 43524, 1902, 282, 2468, 1694, 327, 260,
    26211, 28, 284, 357, 314, 16551, 429, 7094, 39799, 15590, 338, 15121, 3869, 30,
    198, 198, 54, 369, 259, 314, 253, 1734, 1789, 327, 2194, 5646, 3253, 30,
    378, 25316, 13281, 314, 7074, 618, 4073, 5933, 42, 19839, 274, 28, 34868, 28,
    1462, 6558, 28, 1502, 11160, 28, 16067, 288, 372, 5810, 28, 284, 5087, 2909,
    4686, 30, 3768, 3632, 13662, 335, 354, 44771, 14019, 6133, 284, 4631, 28713, 351,
    8212, 2257, 6332, 30, 1068, 23630, 746, 12936, 28, 18411, 259, 2774, 946, 618,
    260, 23630, 29709, 28, 527, 13101, 3324, 1005, 1743, 4577, 284, 27795, 2476, 13104,
    4261, 288, 260, 768, 5181, 25817, 1770, 30, 1985, 10910, 32688, 260, 24672, 25817,
    314, 8996, 975, 357, 8211, 1507, 18771, 284, 30167, 2674, 32528, 7878, 281, 19775,
    3500, 30, 378, 29709, 28263, 253, 1165, 28, 30473, 12077, 2376, 588, 338, 3279,
    2909, 1072, 441, 3749, 335, 750, 1542, 25817, 6230, 30, 14980, 1005, 10379, 1066,
    260, 452, 55, 69, 54, 3608, 10220, 28, 284, 260, 29709, 988, 3237, 971,
    3608, 4624, 981, 260, 6736, 17109, 30, 669, 1157, 8211, 260, 2609, 2050, 1890,
    42, 253, 6011, 314, 10717, 618, 17837, 28, 260, 17837, 359, 4180, 288, 260,
    1743, 28, 284, 260, 23146, 3124, 314, 2198, 2520, 1056, 288, 260, 38448, 30,
    1943, 282, 260, 14019, 3301, 3073, 23124, 940, 1699, 980, 6177, 28, 975, 260,
    2444, 13281, 7313, 281, 582, 980, 351, 253, 3600, 2468, 30, 669, 8510, 314,
    18519, 9174, 588, 338, 6514, 722, 281, 253, 986, 2468, 29278, 1072, 441, 8542,
    260, 1743, 1731, 750, 1542, 4234, 30, 657, 5961, 805, 288, 1538, 253, 43524,
    1902, 282, 2468, 1694, 327, 260, 26211, 28, 284, 357, 314, 16551, 429, 7094,
    39799, 15590, 338, 15121, 3869, 30, 198, 198, 54, 369, 259, 314, 253, 1734,
    1789, 327, 2194, 5646, 3253, 30, 378, 25316, 13281, 314, 7074, 618, 4073, 5933,
    42, 19839, 274, 28, 34868, 28, 1462, 6558, 28, 1502, 11160, 28, 16067, 288,
    372, 5810, 28, 284, 5087, 2909, 4686, 30, 3768, 3632, 13662, 335, 354, 44771,
    14019, 6133, 284, 4631, 28713, 351, 8212, 2257, 6332, 30, 1068, 23630, 746, 12936,
    28, 18411, 259, 2774, 946, 618, 260, 23630, 29709, 28, 527, 13101, 3324, 1005,
    1743, 4577, 284, 27795, 2476, 13104, 4261, 288, 260, 768, 5181, 25817, 1770, 30,
    1985, 10910, 32688, 260, 24672, 25817, 314, 8996, 975, 357, 8211, 1507, 18771, 284,
    30167, 2674, 32528, 7878, 281, 19775, 3500, 30, 378, 29709, 28263, 253, 1165, 28,
    30473, 12077, 2376, 588, 338, 3279, 2909, 1072, 441, 3749, 335, 750, 1542, 25817,
    6230, 30, 14980, 1005, 10379, 1066, 260, 452, 55, 69, 54, 3608, 10220, 28,
    284, 260, 29709, 988, 3237, 971, 3608, 4624, 981, 260, 6736, 17109, 30, 669,
    1157, 8211, 260, 2609, 2050, 1890, 42, 253, 6011, 314, 10717, 618, 17837, 28,
    260, 17837, 359, 4180, 288, 260, 1743, 28, 284, 260, 23146, 3124, 314, 2198,
    2520, 1056, 288, 260, 38448, 30, 1943, 282, 260, 14019, 3301, 3073, 23124, 940,
    1699, 980, 6177, 28, 975, 260, 2444, 13281, 7313, 281, 582, 980, 351, 253,
    3600, 2468, 30, 669, 8510, 314, 18519, 9174, 588, 338, 6514, 722, 281, 253,
    986, 2468, 29278, 1072, 441, 8542, 260, 1743, 1731, 750, 1542, 4234, 30, 657,
    5961, 805, 288, 1538, 253, 43524, 1902, 282, 2468, 1694, 327, 260, 26211, 28,
    284, 357, 314, 16551, 429, 7094, 39799, 15590, 338, 15121, 3869, 30, 198, 198,
    54, 369, 259, 314, 253, 1734, 1789, 327, 2194, 5646, 3253, 30, 378, 25316,
    13281, 314, 7074, 618, 4073, 5933, 42, 19839, 274, 28, 34868, 28, 1462, 6558,
    28, 1502, 11160, 28, 16067, 288, 372, 5810, 28, 284, 5087, 2909, 4686, 30,
    3768, 3632, 13662, 335, 354, 44771, 14019, 6133, 284, 4631, 28713, 351, 8212, 2257,
    6332, 30, 1068, 23630, 746, 12936, 28, 18411, 259, 2774, 946, 618, 260, 23630,
    29709, 28, 527, 13101, 3324, 1005, 1743, 4577, 284, 27795, 2476, 13104, 4261, 288,
    260, 768, 5181, 25817, 1770, 30, 1985, 10910, 32688, 260, 24672, 25817, 314, 8996,
    975, 357, 8211, 1507, 18771, 284, 30167, 2674, 32528, 7878, 281, 19775, 3500, 30,
    378, 29709, 28263, 253, 1165, 28, 30473, 12077, 2376, 588, 338, 3279, 2909, 1072,
    441, 3749, 335, 750, 1542, 25817, 6230, 30, 14980, 1005, 10379, 1066, 260, 452,
    55, 69, 54, 3608, 10220, 28, 284, 260, 29709, 988, 3237, 971, 3608, 4624,
    981, 260, 6736, 17109, 30, 669, 1157, 8211, 260, 2609, 2050, 1890, 42, 253,
    6011, 314, 10717, 618, 17837, 28, 260, 17837, 359, 4180, 288, 260, 1743, 28,
    284, 260, 23146, 3124, 314, 2198, 2520, 1056, 288, 260, 38448, 30, 1943, 282,
    260, 14019, 3301, 3073, 23124, 940, 1699, 980, 6177, 28, 975, 260, 2444, 13281,
    7313, 281, 582, 980, 351, 253, 3600, 2468, 30, 669, 8510, 314, 18519, 9174,
    588, 338, 6514, 722, 281, 253, 986, 2468, 29278, 1072, 441, 8542, 260, 1743,
    1731, 750, 1542, 4234, 30, 657, 5961, 805, 288, 1538, 253, 43524, 1902, 282,
    2468, 1694, 327, 260, 26211, 28, 284, 357, 314, 16551, 429, 7094, 39799, 15590,
    338, 15121, 3869, 30, 198
    ];
    assert_eq!(ids, expected);
}

// ---------------------------------------------------------------------------
// Behavioral contract (BOS-free, space-prefix-free, specials, EOG)
// ---------------------------------------------------------------------------

#[test]
fn behavior_bos_free_and_space_prefix_free() {
    let tok = pinned_tokenizer();
    assert!(!tok.add_bos_token());
    assert!(!tok.add_space_prefix());
    assert_eq!(tok.bos_token_id(), BOS_TOKEN_ID);
    assert_eq!(tok.eos_token_id(), EOS_TOKEN_ID);
    assert_eq!(tok.pad_token_id(), PAD_TOKEN_ID);
    assert_eq!(tok.unk_token_id(), UNK_TOKEN_ID);
    assert_eq!(tok.vocab_size(), EXPECTED_VOCAB_SIZE);
    // Empty prompt stays empty; a leading space never merges into the first
    // token beyond the pinned ` ?\p{L}+` behavior (P11).
    assert_eq!(tok.encode("", true).unwrap(), Vec::<u32>::new());
}

#[test]
fn behavior_special_parse_on_off() {
    // P5 vs P6: specials parse to single ids only when enabled.
    let tok = pinned_tokenizer();
    assert_eq!(tok.encode("<|im_start|>", true).unwrap(), vec![1]);
    let literal = tok.encode("<|im_start|>", false).unwrap();
    assert_eq!(literal.len(), 7);
    assert_ne!(literal, vec![1]);
}

#[test]
fn behavior_specials_cache() {
    let tok = pinned_tokenizer();
    let specials: Vec<(String, u32)> = tok.special_tokens().map(|(t, id)| (t.to_string(), id)).collect();
    assert_eq!(specials.len(), 17);
    // ids 0..=16 are the 17 control specials.
    let mut ids: Vec<u32> = specials.iter().map(|(_, id)| *id).collect();
    ids.sort_unstable();
    assert_eq!(ids, (0..=16).collect::<Vec<u32>>());
    // Cache order is text byte length descending (llama.cpp cache order).
    let lens: Vec<usize> = specials.iter().map(|(t, _)| t.len()).collect();
    let mut sorted = lens.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    assert_eq!(lens, sorted);
}

#[test]
fn behavior_eog_set() {
    let tok = pinned_tokenizer();
    let mut eog = tok.eog_tokens().to_vec();
    eog.sort_unstable();
    assert_eq!(eog, vec![0, 2]);
}

#[test]
fn behavior_deterministic() {
    let tok = pinned_tokenizer();
    let a = tok.encode("The quick brown fox jumps over the lazy dog", true).unwrap();
    let b = tok.encode("The quick brown fox jumps over the lazy dog", true).unwrap();
    assert_eq!(a, b);
}

#[test]
fn behavior_all_ids_in_range() {
    let tok = pinned_tokenizer();
    for text in [
        "Hello world",
        "<|im_start|>",
        "The quick brown fox jumps over the lazy dog",
        "  leading spaces",
    ] {
        for parse_special in [true, false] {
            let ids = tok.encode(text, parse_special).unwrap();
            assert!(ids.iter().all(|&id| (id as usize) < tok.vocab_size()));
        }
    }
}

// ---------------------------------------------------------------------------
// Fail-closed negatives
// ---------------------------------------------------------------------------

#[test]
fn negative_wrong_model() {
    let mut f = synthetic_facts();
    f.model = "llama".to_string();
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::UnsupportedModel { .. })));
}

#[test]
fn negative_wrong_pre() {
    let mut f = synthetic_facts();
    f.pre = "gpt2".to_string();
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::UnsupportedPreTokenizer { .. })));
}

#[test]
fn negative_vocab_count() {
    let mut f = synthetic_facts();
    f.tokens = vec!["a".to_string(); 10];
    f.token_types = vec![1; 10];
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::VocabSizeMismatch { .. })));
}

#[test]
fn negative_merges_count() {
    let mut f = synthetic_facts();
    f.merges = vec!["a b".to_string(); 100];
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::MergesCountMismatch { .. })));
}

#[test]
fn negative_token_type_count() {
    let mut f = synthetic_facts();
    f.token_types = vec![1; EXPECTED_VOCAB_SIZE - 1];
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::TokenTypeCountMismatch { .. })));
}

#[test]
fn negative_scores_present() {
    let mut f = synthetic_facts();
    f.scores_present = true;
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::ScoresPresent)));
}

#[test]
fn negative_malformed_merge() {
    let mut f = synthetic_facts();
    f.merges[0] = "no-separator".to_string();
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::MalformedMergeString { .. })));
}

#[test]
fn negative_byte_boundary_space_token() {
    let mut f = synthetic_facts();
    f.tokens[0] = "has space".to_string();
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::ByteBoundaryViolation { .. })));
}

#[test]
fn negative_byte_boundary_empty_token() {
    let mut f = synthetic_facts();
    f.tokens[0] = String::new();
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::ByteBoundaryViolation { .. })));
}

#[test]
fn negative_token_id_out_of_range() {
    let mut f = synthetic_facts();
    f.bos_token_id = EXPECTED_VOCAB_SIZE as u32;
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::TokenIdOutOfRange { .. })));
}

#[test]
fn negative_add_bos_true() {
    let mut f = synthetic_facts();
    f.add_bos_token = true;
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::AddBosTokenNotSupported { .. })));
}

#[test]
fn negative_add_space_prefix_true() {
    let mut f = synthetic_facts();
    f.add_space_prefix = true;
    assert!(matches!(Gpt2BpeTokenizer::new(f), Err(TokenizerError::AddSpacePrefixNotSupported { .. })));
}

#[test]
fn negative_encode_input_ceiling() {
    let tok = pinned_tokenizer();
    let huge = "a".repeat(MAX_ENCODE_INPUT_BYTES + 1);
    assert_eq!(
        tok.encode(&huge, true),
        Err(TokenizerError::InputTooLarge {
            bytes: MAX_ENCODE_INPUT_BYTES + 1,
            ceiling: MAX_ENCODE_INPUT_BYTES
        })
    );
}
