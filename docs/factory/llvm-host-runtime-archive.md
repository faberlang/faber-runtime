# LLVM Host Runtime Archive — Artifact Metadata

**Status**: metadata record (llvm-host-parity campaign, Stage 9 release-prep)
**Date**: 2026-08-06

## Artifact identity

| Field | Value |
| --- | --- |
| Runtime name | `faber-host-llvm` |
| Crate | `faber-runtime/hosts/llvm` (crate-type `rlib` + `staticlib`) |
| Version | `0.1.0` (`[package] version` in `hosts/llvm/Cargo.toml`) |
| Staticlib artifact | `target/release/libfaber_host_llvm.a` (Unix) / `faber_host_llvm.lib` (Windows) |
| Build command | `cargo build --release --manifest-path hosts/llvm/Cargo.toml` |
| Producer | Faber `faber build/run --target llvm-host` (built on first use) |

## Purpose

This is the versioned host runtime that native LLVM host executables link
against: the `__faber_rt_v1_*` symbol surface (process argv, CLI descriptor
decode, exit policy, carriers, collection/advanced runtime) that the
MIR-to-LLVM emitter's entry modules declare and call.

## How the product records it

`faber build --target llvm-host` records the archive identity and path in the
inspectable artifact layout:

- `target/faber-llvm/{debug|release}/link-manifest.toml` → `[link]
  runtime_archive = <path>`;
- `target/faber-llvm/{debug|release}/runtime/identity.toml` → `runtime_name`,
  `runtime_version`, `archive`.

## Distribution status

The archive is built from source for the local-toolchain product gate. A
prebuilt binary distribution is deferred to the Stage 10 release decision;
the link manifest's archive identity is the contract point for substituting a
distributed archive without code changes (see
`radix/docs/factory/llvm-host-parity/stage-9-release-prep-note.md`).
