//! Packaged device-handle runtime type tests (campaign S1-4).

use crate::device::{DeviceBackend, DeviceHandle, DeviceHandleKind, DeviceSelection};
use crate::valor::{FromValor, Valor};

#[test]
fn backend_spellings_round_trip() {
    assert_eq!(DeviceBackend::Metal.spelling(), "metal");
    assert_eq!(DeviceBackend::Cuda.spelling(), "cuda");
    assert_eq!(
        DeviceBackend::from_spelling("metal"),
        Some(DeviceBackend::Metal)
    );
    assert_eq!(
        DeviceBackend::from_spelling("cuda"),
        Some(DeviceBackend::Cuda)
    );
    assert_eq!(DeviceBackend::from_spelling("wgsl"), None);
}

#[test]
fn selection_spellings_and_backend_resolution() {
    assert_eq!(DeviceSelection::Auto.spelling(), "auto");
    assert_eq!(DeviceSelection::Metal.spelling(), "metal");
    assert_eq!(DeviceSelection::Cuda.spelling(), "cuda");
    assert_eq!(
        DeviceSelection::from_spelling("auto"),
        Some(DeviceSelection::Auto)
    );
    assert_eq!(
        DeviceSelection::from_spelling("metal"),
        Some(DeviceSelection::Metal)
    );

    assert_eq!(DeviceSelection::Auto.backend(), None);
    assert_eq!(DeviceSelection::Metal.backend(), Some(DeviceBackend::Metal));
    assert_eq!(DeviceSelection::Cuda.backend(), Some(DeviceBackend::Cuda));
}

#[test]
fn module_handle_control_frame_carries_identifiers_only() {
    let handle = DeviceHandle {
        backend: DeviceBackend::Metal,
        kind: DeviceHandleKind::Module,
        id: 3,
    };
    let frame = Valor::from(handle);
    match &frame {
        Valor::Tabula(fields) => {
            assert_eq!(
                fields.get("device_backend"),
                Some(&Valor::Textus("metal".to_owned()))
            );
            assert_eq!(
                fields.get("device_kind"),
                Some(&Valor::Textus("module".to_owned()))
            );
            assert_eq!(fields.get("device_id"), Some(&Valor::Numerus(3)));
            assert!(!fields.contains_key("len_bytes"));
        }
        other => panic!("expected tabula handle control frame: {other:?}"),
    }
    // Invariant: control frames never carry tensor payload bytes.
    assert!(no_octeti(&frame));
    assert_eq!(DeviceHandle::from_valor(&frame), Some(handle));
}

#[test]
fn buffer_handle_round_trips_with_len_bytes() {
    let handle = DeviceHandle {
        backend: DeviceBackend::Cuda,
        kind: DeviceHandleKind::Buffer { len_bytes: 4096 },
        id: 7,
    };
    assert_eq!(handle.len_bytes(), Some(4096));
    let frame = Valor::from(&handle);
    match &frame {
        Valor::Tabula(fields) => {
            assert_eq!(
                fields.get("device_backend"),
                Some(&Valor::Textus("cuda".to_owned()))
            );
            assert_eq!(
                fields.get("device_kind"),
                Some(&Valor::Textus("buffer".to_owned()))
            );
            assert_eq!(fields.get("device_id"), Some(&Valor::Numerus(7)));
            assert_eq!(fields.get("len_bytes"), Some(&Valor::Numerus(4096)));
        }
        other => panic!("expected tabula handle control frame: {other:?}"),
    }
    assert!(no_octeti(&frame));
    assert_eq!(DeviceHandle::from_valor(&frame), Some(handle));
}

#[test]
fn payload_carrying_frame_is_rejected_as_handle() {
    // A frame that carries an octeti payload is not a handle control frame:
    // extraction must reject it rather than silently dropping the payload.
    let tainted = Valor::Tabula(
        [
            (
                "device_backend".to_owned(),
                Valor::Textus("cuda".to_owned()),
            ),
            ("device_kind".to_owned(), Valor::Textus("buffer".to_owned())),
            ("device_id".to_owned(), Valor::Numerus(1)),
            ("len_bytes".to_owned(), Valor::Numerus(4)),
            ("payload".to_owned(), Valor::Octeti(vec![1, 2, 3, 4])),
        ]
        .into(),
    );
    assert_eq!(DeviceHandle::from_valor(&tainted), None);
}

#[test]
fn malformed_control_frames_extract_nothing() {
    assert_eq!(DeviceHandle::from_valor(&Valor::Nihil), None);
    assert_eq!(DeviceHandle::from_valor(&Valor::Numerus(1)), None);
    // Missing len_bytes for a buffer handle.
    let no_len = Valor::Tabula(
        [
            (
                "device_backend".to_owned(),
                Valor::Textus("cuda".to_owned()),
            ),
            ("device_kind".to_owned(), Valor::Textus("buffer".to_owned())),
            ("device_id".to_owned(), Valor::Numerus(1)),
        ]
        .into(),
    );
    assert_eq!(DeviceHandle::from_valor(&no_len), None);
}

fn no_octeti(value: &Valor) -> bool {
    match value {
        Valor::Octeti(_) => false,
        Valor::Lista(items) => items.iter().all(no_octeti),
        Valor::Tabula(fields) => fields.values().all(no_octeti),
        _ => true,
    }
}
