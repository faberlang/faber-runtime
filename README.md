# faber-runtime

Public Rust runtime types for Faber-generated code (`use faber::…`).

Generated packages from `faber build` depend on this crate for `Valor`, tensors,
frames, and related carriers. Runtime/compiler-shared contracts live in
`radix-runtime-contract`; the Rust crate name exposed to generated code remains
`faber`.

## Package vs crate name

| Cargo package | Rust crate name (`use`) |
| ------------- | ----------------------- |
| `faber-runtime` | `faber` |

```toml
faber = { package = "faber-runtime", path = "…" }
# or after publish:
# faber = { package = "faber-runtime", version = "0.1" }
```

## Local layout

```text
faberlang/
  faber-runtime/   this repo
  faber/           public CLI (path-deps here for generated crates)
  radix/           private compiler (path-deps here)
  cista/           package manager
  norma/           stdlib source
  triga/           optional graphics and geometry library
```

## Build

```bash
cargo check --workspace
cargo build --release
```

Use `cargo test` or `cargo nextest run` for full validation after targeted
mechanical checks pass.
