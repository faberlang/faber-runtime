//! Packaged device-handle runtime types (differentiable-GPU campaign S1-4).
//!
//! The frozen host-ownership contract (architecture record §5, N1.5) assigns
//! **faber-runtime** the packaged runtime types for generated binaries —
//! `Valor`, frames, and device-handle carriers — with the invariant that
//! **control frames never carry tensor payload bytes**. This module is that
//! packaged surface for native device execution:
//!
//! - [`DeviceBackend`] — the two admitted native backends (Metal, CUDA).
//! - [`DeviceSelection`] — the product-level selection request
//!   (`auto | metal | cuda`), mirroring the frozen FMIR `device.selection`
//!   surface (S1-2, `FmirDeviceSelection`).
//! - [`DeviceHandleKind`] / [`DeviceHandle`] — opaque host-owned handle
//!   carriers. A handle names `(backend, kind, id)` and nothing else; it
//!   **never** carries tensor payload bytes. Valor-frame integration
//!   ([`Valor::from(DeviceHandle)`](Valor)) lowers a handle to a control
//!   frame whose fields are scalar identifiers only, so an in-band tensor or
//!   module blob can never ride a handle control frame.
//!
//! The hosts `faber-host-macos-arm64` crate consumes these types for its
//! composite host and device sessions; generated binaries and the CLI package
//! layer consume them through the same `faber::` re-export.

use crate::valor::Valor;
use std::collections::BTreeMap;

/// A native device backend admitted by the product host.
///
/// The accepted machines are Apple Metal (M5 Max, burgus) and NVIDIA CUDA
/// (RTX 5070, pharos); these are the only backends the campaign productizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceBackend {
    /// Apple Metal (MSL modules; macOS-only).
    Metal,
    /// NVIDIA CUDA Driver API (PTX modules).
    Cuda,
}

impl DeviceBackend {
    /// Stable diagnostic spelling (`"metal"` / `"cuda"`). Used in control
    /// frames and structured error diagnostics.
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// Parse a backend from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }
}

impl std::fmt::Display for DeviceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.spelling())
    }
}

/// Product-level device selection request, mirroring the frozen FMIR
/// `device.selection` surface (S1-2 `FmirDeviceSelection: auto|metal|cuda`).
///
/// The selection is a runtime/product concern (A6): the package stays a
/// portable program representation and never hard-binds a backend at compile
/// time. `Auto` resolves against host capability probes at host construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceSelection {
    /// Resolve against host capability probes; fail closed when the machine
    /// admits zero or more than one backend.
    Auto,
    /// Select Metal explicitly; never silently falls back.
    Metal,
    /// Select CUDA explicitly; never silently falls back.
    Cuda,
}

impl DeviceSelection {
    /// Stable diagnostic spelling (`"auto"` / `"metal"` / `"cuda"`).
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }

    /// Parse a selection from its stable spelling.
    #[must_use]
    pub fn from_spelling(spelling: &str) -> Option<Self> {
        match spelling {
            "auto" => Some(Self::Auto),
            "metal" => Some(Self::Metal),
            "cuda" => Some(Self::Cuda),
            _ => None,
        }
    }

    /// The backend an explicit selection names; `Auto` is `None`.
    #[must_use]
    pub fn backend(self) -> Option<DeviceBackend> {
        match self {
            Self::Auto => None,
            Self::Metal => Some(DeviceBackend::Metal),
            Self::Cuda => Some(DeviceBackend::Cuda),
        }
    }
}

impl std::fmt::Display for DeviceSelection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.spelling())
    }
}

/// What kind of device object an opaque handle names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceHandleKind {
    /// A compiled module (MSL or PTX image loaded by the driver).
    Module,
    /// A device buffer of the given byte length.
    Buffer {
        /// Allocated byte length on the device.
        len_bytes: u64,
    },
}

impl DeviceHandleKind {
    /// Stable diagnostic spelling (`"module"` / `"buffer"`).
    #[must_use]
    pub fn spelling(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Buffer { .. } => "buffer",
        }
    }
}

/// Opaque host-owned handle identity carried across control boundaries.
///
/// A handle is a **carrier, not a payload**: it names the backend, the kind,
/// and the session-local opaque id, and nothing else. Tensor bytes, module
/// text, and shapes never travel inside a handle — they live in the owning
/// host session's registry. Valor-frame integration preserves this invariant:
/// the control frame for a handle is scalar identifiers only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceHandle {
    /// The backend that owns the underlying device object.
    pub backend: DeviceBackend,
    /// What kind of device object the id names.
    pub kind: DeviceHandleKind,
    /// Session-local opaque id (allocated by the owning session's registry).
    pub id: u64,
}

impl DeviceHandle {
    /// The byte length of a buffer handle; `None` for modules.
    #[must_use]
    pub fn len_bytes(self) -> Option<u64> {
        match self.kind {
            DeviceHandleKind::Module => None,
            DeviceHandleKind::Buffer { len_bytes } => Some(len_bytes),
        }
    }
}

impl From<DeviceHandle> for Valor {
    fn from(handle: DeviceHandle) -> Self {
        Valor::from(&handle)
    }
}

impl From<&DeviceHandle> for Valor {
    /// Control-frame representation of a handle: scalar identifiers only, no
    /// payload bytes (a control frame for a handle can never be a tensor).
    fn from(handle: &DeviceHandle) -> Self {
        let mut fields = BTreeMap::new();
        fields.insert(
            "device_backend".to_owned(),
            Valor::Textus(handle.backend.spelling().to_owned()),
        );
        fields.insert(
            "device_kind".to_owned(),
            Valor::Textus(handle.kind.spelling().to_owned()),
        );
        // Session-local opaque ids are small; the existing host control-frame
        // precedent already carries them as Numerus (`id.0 as i64`).
        #[allow(clippy::cast_possible_wrap)]
        fields.insert("device_id".to_owned(), Valor::Numerus(handle.id as i64));
        if let Some(len_bytes) = handle.len_bytes() {
            #[allow(clippy::cast_possible_wrap)]
            fields.insert("len_bytes".to_owned(), Valor::Numerus(len_bytes as i64));
        }
        Valor::Tabula(fields)
    }
}

impl crate::valor::FromValor for DeviceHandle {
    /// Extract a handle from its control-frame representation. Rejects any
    /// frame that is not a scalar-identifier tabula (a frame carrying a
    /// `Octeti` payload is not a handle control frame).
    fn from_valor(value: &Valor) -> Option<Self> {
        let Valor::Tabula(fields) = value else {
            return None;
        };
        if fields
            .values()
            .any(|field| matches!(field, Valor::Octeti(_)))
        {
            return None;
        }
        let backend_spelling = String::from_valor(fields.get("device_backend")?)?;
        let backend = DeviceBackend::from_spelling(&backend_spelling)?;
        let id = u64::try_from(i64::from_valor(fields.get("device_id")?)?).ok()?;
        let kind = String::from_valor(fields.get("device_kind")?)?;
        match kind.as_str() {
            "module" => Some(Self {
                backend,
                kind: DeviceHandleKind::Module,
                id,
            }),
            "buffer" => {
                let len_bytes = u64::try_from(i64::from_valor(fields.get("len_bytes")?)?).ok()?;
                Some(Self {
                    backend,
                    kind: DeviceHandleKind::Buffer { len_bytes },
                    id,
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
#[path = "device_test.rs"]
mod tests;
