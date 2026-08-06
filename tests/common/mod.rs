//! Shared stdout-parsing helpers for faber-runtime integration tests.
//!
//! The compiler-generated exempla print their training traces in two shapes:
//!
//! - bare `nota` output — scalar lines and bracketed lists (`[3.14, 1.23]`)
//!   from non-device routes (e.g. `linear-regression` has no `[device]`
//!   section), read by [`parse_f32_values`];
//! - the S5-U5 RepeatingStep device route's per-step lines
//!   (`device:   step N: [loss]`, printed by
//!   `faber/src/package/device/run.rs`), read by [`parse_step_loss_trace`]
//!   and used by the MLP and BERT-tiny loss-trace e2e tests.
//!
//! Each integration-test crate compiles this module into its own crate and
//! uses only the helpers it needs, so allow dead code here.
#![allow(dead_code)]

/// Extract all f32 values from stdout, handling `nota` output formats.
///
/// Handles both bare f32 lines (single `nota` of a scalar) and bracketed
/// list format: `[3.14, 1.23, ...]` (output from `nota lista<f32>`).
pub fn parse_f32_values(stdout: &str) -> Vec<f32> {
    let mut values = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        // Try bare f32 parse first.
        if let Ok(v) = trimmed.parse::<f32>() {
            values.push(v);
            continue;
        }
        // Bracketed list: [val, val, ...]
        values.extend(parse_bracketed_f32_list(trimmed));
    }
    values
}

/// Parse a bracketed f32 list: `[3.14, 1.23, ...]`. Returns an empty vec for
/// any other input shape.
pub fn parse_bracketed_f32_list(line: &str) -> Vec<f32> {
    let line = line.trim();
    if !(line.starts_with('[') && line.ends_with(']')) {
        return Vec::new();
    }
    let inner = &line[1..line.len() - 1];
    let mut values = Vec::new();
    for part in inner.split(',') {
        let part = part.trim();
        if !part.is_empty() {
            if let Ok(v) = part.parse::<f32>() {
                values.push(v);
            }
        }
    }
    values
}

/// Extract the per-step loss trace from the device route's training report.
///
/// The RepeatingStep route (S5-U5, `faber/src/package/device/run.rs`) prints
/// the loss trace as one line per step:
///
/// ```text
/// device: training: 100 step(s) on ONE session; per-step observation (loss) trace:
/// device:   step 0: [1.5764482]
/// device:   step 1: [1.3989581]
/// ...
/// ```
///
/// Values are placed at their explicit step index — not by stdout position —
/// because the route prints the final loss observation buffer line *before*
/// the trace, so a naive line-order scan would misorder the first value.
pub fn parse_step_loss_trace(stdout: &str) -> Vec<f32> {
    let mut trace: Vec<f32> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        let Some(after_step) = line
            .strip_prefix("device:")
            .and_then(|rest| rest.trim().strip_prefix("step "))
        else {
            continue;
        };
        let digit_len = after_step
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .count();
        if digit_len == 0 {
            continue;
        }
        let Ok(step) = after_step[..digit_len].parse::<usize>() else {
            continue;
        };
        let Some(list) = after_step[digit_len..].strip_prefix(':') else {
            continue;
        };
        let values = parse_bracketed_f32_list(list);
        if values.is_empty() {
            continue;
        }
        if trace.len() <= step {
            trace.resize(step + values.len(), 0.0);
        }
        for (offset, &value) in values.iter().enumerate() {
            trace[step + offset] = value;
        }
    }
    trace
}
