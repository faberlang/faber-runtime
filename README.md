# faber-runtime

Public Rust runtime types for Faber-generated code (`use faber::…`).

This crate is independent of the private Radix compiler. Generated packages from
`faber build` depend on it for `Valor`, tensors, frames, and related carriers.

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
cargo test
cargo build --release
```
