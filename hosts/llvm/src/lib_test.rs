use super::*;
use std::ffi::{c_void, CStr};

#[test]
fn init_write_and_shutdown_round_trip() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);
    assert!(!context.is_null());
    let status =
        unsafe { __faber_rt_v1_write_nota_text(context, FaberRtSliceV1::from_static(b"")) };
    assert_eq!(status, STATUS_OK);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn invalid_slice_fails_closed() {
    let status = unsafe {
        __faber_rt_v1_write_nota_text(
            ptr::dangling_mut(),
            FaberRtSliceV1 {
                data: ptr::null(),
                len: 1,
            },
        )
    };
    assert_eq!(status, STATUS_INVALID_ARGUMENT);
}

#[test]
fn diagnostic_family_reports_scalar_and_opaque_dispositions() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);

    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i64(context, 42) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i1(context, 1) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_f32(context, 1.25) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_f64(context, 2.5) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i8(context, -8) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_i32(context, -32) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_nota_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_mone_i64(context, -64) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_rt_v1_diagnostic_vide_i64(context, 64) },
        STATUS_OK
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn assertion_family_returns_handled_statuses() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);

    assert_eq!(unsafe { __faber_rt_v1_assert(context, 1) }, STATUS_OK);
    assert_eq!(unsafe { __faber_rt_v1_assert(context, 0) }, STATUS_PANIC);
    assert_eq!(
        unsafe { __faber_rt_v1_assert_message(context, 1, FaberRtSliceV1::from_static(b"unused")) },
        STATUS_OK
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_assert_message(
                context,
                0,
                FaberRtSliceV1::from_static(b"assertion failed"),
            )
        },
        STATUS_PANIC
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_format_family_renders_and_owns_text() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);

    let one = unsafe {
        __faber_rt_v1_format_i64(context, FaberRtSliceV1::from_static("n=§".as_bytes()), 42)
    };
    let reordered = unsafe {
        __faber_rt_v1_format_i64_i64(
            context,
            FaberRtSliceV1::from_static("§1/§0/§9".as_bytes()),
            3,
            7,
        )
    };
    let float = unsafe {
        __faber_rt_v1_format_f64(context, FaberRtSliceV1::from_static("x=§".as_bytes()), 1.5)
    };
    let three = unsafe {
        __faber_rt_v1_format_i64_i64_i64(
            context,
            FaberRtSliceV1::from_static("§/§/§".as_bytes()),
            1,
            2,
            3,
        )
    };
    let invalid =
        unsafe { __faber_rt_v1_format_i64(context, FaberRtSliceV1::from_static(&[0xff]), 42) };
    let paired = unsafe {
        __faber_rt_v1_format_text_text(
            context,
            FaberRtSliceV1::from_static("§ + §".as_bytes()),
            one.value.cast(),
            float.value.cast(),
        )
    };
    let single = unsafe {
        __faber_rt_v1_format_text(
            context,
            FaberRtSliceV1::from_static("[§]".as_bytes()),
            one.value.cast(),
        )
    };
    let mixed = unsafe {
        __faber_rt_v1_format_text_i64(
            context,
            FaberRtSliceV1::from_static("§:§".as_bytes()),
            one.value.cast(),
            9,
        )
    };
    let mut length = -1;
    let length_status =
        unsafe { __faber_rt_v1_text_length(context, paired.value.cast(), &mut length) };

    assert_eq!(one.status, STATUS_OK);
    assert_eq!(reordered.status, STATUS_OK);
    assert_eq!(float.status, STATUS_OK);
    assert_eq!(three.status, STATUS_OK);
    assert_eq!(paired.status, STATUS_OK);
    assert_eq!(single.status, STATUS_OK);
    assert_eq!(mixed.status, STATUS_OK);
    assert_eq!(length_status, STATUS_OK);
    assert_eq!(length, 12);
    assert_eq!(
        invalid,
        FaberRtPtrResultV1::failure(STATUS_INVALID_ARGUMENT)
    );
    assert_eq!(unsafe { &*one.value.cast::<RuntimeText>() }._value, "n=42");
    assert_eq!(
        unsafe { &*reordered.value.cast::<RuntimeText>() }._value,
        "7/3/§9"
    );
    assert_eq!(
        unsafe { &*float.value.cast::<RuntimeText>() }._value,
        "x=1.5"
    );
    assert_eq!(
        unsafe { &*paired.value.cast::<RuntimeText>() }._value,
        "n=42 + x=1.5"
    );
    assert_eq!(
        unsafe { &*single.value.cast::<RuntimeText>() }._value,
        "[n=42]"
    );
    assert_eq!(
        unsafe { &*mixed.value.cast::<RuntimeText>() }._value,
        "n=42:9"
    );
    assert_eq!(
        unsafe { &*three.value.cast::<RuntimeText>() }._value,
        "1/2/3"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_query_and_transformation_family_preserves_unicode_semantics() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let text = FaberRtSliceV1::from_static("  Rōma/AVĒ  ".as_bytes());
    let empty = FaberRtSliceV1::from_static(b"");
    let roma = FaberRtSliceV1::from_static("Rōma".as_bytes());
    let slash = FaberRtSliceV1::from_static(b"/");
    let ave = FaberRtSliceV1::from_static("AVĒ".as_bytes());
    let mut answer = 0;

    assert_eq!(
        unsafe { __faber_rt_v1_text_is_empty(context, &empty, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_text_contains(context, &text, &roma, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_text_starts_with(context, &text, &empty, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_text_ends_with(context, &text, &empty, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);

    let trimmed = unsafe { __faber_rt_v1_text_trim(context, &text) };
    let lower = unsafe { __faber_rt_v1_text_lowercase(context, trimmed.value.cast()) };
    let upper = unsafe { __faber_rt_v1_text_uppercase(context, lower.value.cast()) };
    let sliced = unsafe { __faber_rt_v1_text_slice(context, trimmed.value.cast(), 1, 5) };
    let replaced =
        unsafe { __faber_rt_v1_text_replace(context, trimmed.value.cast(), &ave, &roma) };
    let split = unsafe { __faber_rt_v1_text_split(context, trimmed.value.cast(), &slash) };
    for result in [trimmed, lower, upper, sliced, replaced, split] {
        assert_eq!(result.status, STATUS_OK);
    }
    assert_eq!(
        unsafe { &*trimmed.value.cast::<RuntimeText>() }._value,
        "Rōma/AVĒ"
    );
    assert_eq!(
        unsafe { &*lower.value.cast::<RuntimeText>() }._value,
        "rōma/avē"
    );
    assert_eq!(
        unsafe { &*upper.value.cast::<RuntimeText>() }._value,
        "RŌMA/AVĒ"
    );
    assert_eq!(
        unsafe { &*sliced.value.cast::<RuntimeText>() }._value,
        "ōma/"
    );
    assert_eq!(
        unsafe { &*replaced.value.cast::<RuntimeText>() }._value,
        "Rōma/Rōma"
    );
    let split = unsafe { &*split.value.cast::<RuntimeArray>() };
    assert_eq!(split.kind, VALUE_KIND_PTR);
    assert_eq!(split.values.len(), 2);
    let parts = split
        .values
        .iter()
        .map(|value| match value {
            array::RuntimeValue::Ptr(value) => {
                unsafe { &*value.cast::<RuntimeText>() }._value.as_str()
            }
            _ => panic!("split produced non-text carrier"),
        })
        .collect::<Vec<_>>();
    assert_eq!(parts, ["Rōma", "AVĒ"]);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn text_scalar_conversion_family_honors_width_radix_recovery_status() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let hex = FaberRtSliceV1::from_static(b"ff");
    let negative = FaberRtSliceV1::from_static(b"-8");
    let decimal = FaberRtSliceV1::from_static(b"1.25");
    let invalid = FaberRtSliceV1::from_static(b"invalid");
    let empty = FaberRtSliceV1::from_static(b"");
    let mut i32_value = 0i32;
    let mut i8_value = 0i8;
    let mut i64_value = 0i64;
    let mut f64_value = 0.0f64;
    let mut truthy = 1u8;

    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &hex,
                16,
                VALUE_KIND_I32,
                std::ptr::from_mut(&mut i32_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(i32_value, 255);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &negative,
                10,
                VALUE_KIND_I8,
                std::ptr::from_mut(&mut i8_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(i8_value, -8);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_float(
                context,
                &decimal,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut f64_value).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(f64_value, 1.25);
    assert_eq!(
        unsafe {
            __faber_rt_v1_text_parse_integer(
                context,
                &invalid,
                10,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut i64_value).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_text_truthy(context, &empty, &mut truthy) },
        STATUS_OK
    );
    assert_eq!(truthy, 0);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn typed_map_and_set_family_preserves_value_semantics() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let map = unsafe { __faber_rt_v1_map_new(context, VALUE_KIND_TEXT, VALUE_KIND_I64) };
    assert_eq!(map.status, STATUS_OK);
    let first_key = FaberRtSliceV1::from_static("aelia".as_bytes());
    let equal_key = FaberRtSliceV1::from_static("aelia".as_bytes());
    let missing_key = FaberRtSliceV1::from_static("balbus".as_bytes());
    let first_handle = std::ptr::from_ref(&first_key).cast_mut().cast::<c_void>();
    let equal_handle = std::ptr::from_ref(&equal_key).cast_mut().cast::<c_void>();
    let missing_handle = std::ptr::from_ref(&missing_key).cast_mut().cast::<c_void>();
    let value = 95i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_put(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&first_handle).cast(),
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let mut answer = 0u8;
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_contains(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&equal_handle).cast(),
                &mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(
        answer, 1,
        "distinct text descriptors compare by UTF-8 value"
    );
    let present = unsafe {
        __faber_rt_v1_map_option(
            context,
            map.value,
            VALUE_KIND_TEXT,
            std::ptr::from_ref(&equal_handle).cast(),
            VALUE_KIND_I64,
        )
    };
    let missing = unsafe {
        __faber_rt_v1_map_option(
            context,
            map.value,
            VALUE_KIND_TEXT,
            std::ptr::from_ref(&missing_handle).cast(),
            VALUE_KIND_I64,
        )
    };
    assert!(unsafe { &*present.value.cast::<RuntimeOption>() }
        .value
        .is_some());
    assert!(unsafe { &*missing.value.cast::<RuntimeOption>() }
        .value
        .is_none());
    let mut length = 0i64;
    assert_eq!(
        unsafe { __faber_rt_v1_map_length(context, map.value, &mut length) },
        STATUS_OK
    );
    assert_eq!(length, 1);
    let keys = unsafe { __faber_rt_v1_map_keys(context, map.value) };
    let values = unsafe { __faber_rt_v1_map_values(context, map.value) };
    assert_eq!(
        unsafe { &*keys.value.cast::<RuntimeArray>() }.kind,
        VALUE_KIND_TEXT
    );
    assert_eq!(
        unsafe { &*values.value.cast::<RuntimeArray>() }.kind,
        VALUE_KIND_I64
    );
    assert_eq!(
        unsafe { &*keys.value.cast::<RuntimeArray>() }.values.len(),
        1
    );
    assert_eq!(
        unsafe { &*values.value.cast::<RuntimeArray>() }
            .values
            .len(),
        1
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_map_delete(
                context,
                map.value,
                VALUE_KIND_TEXT,
                std::ptr::from_ref(&equal_handle).cast(),
                &mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_map_is_empty(context, map.value, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 1);

    let left = unsafe { __faber_rt_v1_set_new(context, VALUE_KIND_I64) };
    let right = unsafe { __faber_rt_v1_set_new(context, VALUE_KIND_I64) };
    for (set, values) in [
        (left.value, &[1i64, 2, 3][..]),
        (right.value, &[2i64, 4][..]),
    ] {
        for value in values {
            assert_eq!(
                unsafe {
                    __faber_rt_v1_set_add(
                        context,
                        set,
                        VALUE_KIND_I64,
                        std::ptr::from_ref(value).cast(),
                    )
                },
                STATUS_OK
            );
        }
    }
    let union = unsafe { __faber_rt_v1_set_union(context, left.value, right.value) };
    let intersection = unsafe { __faber_rt_v1_set_intersection(context, left.value, right.value) };
    let difference = unsafe { __faber_rt_v1_set_difference(context, left.value, right.value) };
    let symmetric =
        unsafe { __faber_rt_v1_set_symmetric_difference(context, left.value, right.value) };
    assert_eq!(
        unsafe { &*union.value.cast::<RuntimeSet>() }.values.len(),
        4
    );
    assert_eq!(
        unsafe { &*intersection.value.cast::<RuntimeSet>() }
            .values
            .len(),
        1
    );
    assert_eq!(
        unsafe { &*difference.value.cast::<RuntimeSet>() }
            .values
            .len(),
        2
    );
    assert_eq!(
        unsafe { &*symmetric.value.cast::<RuntimeSet>() }
            .values
            .len(),
        3
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_is_subset(context, intersection.value, union.value, &mut answer)
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_is_superset(context, union.value, intersection.value, &mut answer)
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    let two = 2i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_contains(
                context,
                left.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&two).cast(),
                &mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(answer, 1);
    assert_eq!(
        unsafe {
            __faber_rt_v1_set_delete(
                context,
                left.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&two).cast(),
                &mut answer,
            )
        },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_set_length(context, left.value, &mut length) },
        STATUS_OK
    );
    assert_eq!(length, 2);
    assert_eq!(
        unsafe { __faber_rt_v1_set_is_empty(context, left.value, &mut answer) },
        STATUS_OK
    );
    assert_eq!(answer, 0);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_text_conversion_family_owns_canonical_values() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);

    let integer = unsafe { __faber_rt_v1_text_i64(context, -42) };
    let float = unsafe { __faber_rt_v1_text_f64(context, 3.25) };
    let boolean = unsafe { __faber_rt_v1_text_i1(context, 1) };

    assert_eq!(integer.status, STATUS_OK);
    assert_eq!(float.status, STATUS_OK);
    assert_eq!(boolean.status, STATUS_OK);
    assert_eq!(
        unsafe { &*integer.value.cast::<RuntimeText>() }._value,
        "-42"
    );
    assert_eq!(
        unsafe { &*float.value.cast::<RuntimeText>() }._value,
        "3.25"
    );
    assert_eq!(
        unsafe { &*boolean.value.cast::<RuntimeText>() }._value,
        "true"
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn scalar_valor_conversion_family_owns_typed_values() {
    let mut context = ptr::null_mut();
    let status = unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) };
    assert_eq!(status, STATUS_OK);

    let integer = unsafe { __faber_rt_v1_valor_i64(context, -42) };
    let float = unsafe { __faber_rt_v1_valor_f64(context, 3.25) };
    let boolean = unsafe { __faber_rt_v1_valor_i1(context, 1) };

    assert_eq!(
        unsafe { &*integer.value.cast::<Valor>() },
        &Valor::Numerus(-42)
    );
    assert_eq!(
        unsafe { &*float.value.cast::<Valor>() },
        &Valor::Fractus(3.25)
    );
    assert_eq!(
        unsafe { &*boolean.value.cast::<Valor>() },
        &Valor::Bivalens(true)
    );

    let text = FaberRtSliceV1::from_static(b"salve");
    let boxed_text = unsafe { __faber_rt_v1_valor_text(context, &text) };
    let boxed_ascii = unsafe { __faber_rt_v1_valor_ascii(context, c"roma".as_ptr()) };
    let boxed_nihil = unsafe { __faber_rt_v1_valor_nihil(context) };
    assert_eq!(
        unsafe { &*boxed_text.value.cast::<Valor>() },
        &Valor::Textus("salve".into())
    );
    assert_eq!(
        unsafe { &*boxed_ascii.value.cast::<Valor>() },
        &Valor::Textus("roma".into())
    );
    assert_eq!(
        unsafe { &*boxed_nihil.value.cast::<Valor>() },
        &Valor::Nihil
    );

    let mut integer_out = 0;
    let mut float_out = 0.0;
    let mut boolean_out = 0;
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_i64(context, integer.value.cast(), &mut integer_out) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_f64(context, integer.value.cast(), &mut float_out) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_i1(context, boolean.value.cast(), &mut boolean_out) },
        STATUS_OK
    );
    assert_eq!((integer_out, float_out, boolean_out), (-42, -42.0, 1));

    let extracted_text = unsafe { __faber_rt_v1_valor_get_text(context, boxed_text.value.cast()) };
    let descriptor = unsafe { &*extracted_text.value.cast::<FaberRtSliceV1>() };
    assert_eq!(
        unsafe { std::slice::from_raw_parts(descriptor.data, descriptor.len as usize) },
        b"salve"
    );
    let extracted_ascii =
        unsafe { __faber_rt_v1_valor_get_ascii(context, boxed_ascii.value.cast()) };
    assert_eq!(
        unsafe { CStr::from_ptr(extracted_ascii.value.cast()) }.to_bytes(),
        b"roma"
    );
    assert_eq!(
        unsafe { __faber_rt_v1_valor_get_nihil(context, boxed_nihil.value.cast()) },
        STATUS_OK
    );

    let mismatch =
        unsafe { __faber_rt_v1_valor_get_i64(context, boxed_text.value.cast(), &mut integer_out) };
    assert_eq!(mismatch, STATUS_INVALID_ARGUMENT);
    let foreign =
        unsafe { __faber_rt_v1_valor_get_i64(context, ptr::dangling::<Valor>(), &mut integer_out) };
    assert_eq!(foreign, STATUS_INVALID_ARGUMENT);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_family_round_trips_every_value_kind_and_spreads() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );

    let i1 = 1_u8;
    let i8_value = -8_i8;
    let i16_value = -16_i16;
    let i32_value = -32_i32;
    let i64_value = -64_i64;
    let u8_value = 8_u8;
    let u16_value = 16_u16;
    let u32_value = 32_u32;
    let u64_value = 64_u64;
    let f16_value = 0x3c00_u16;
    let f32_value = 3.25_f32;
    let f64_value = 6.5_f64;
    let pointer_value = context.cast::<std::ffi::c_void>();
    let cases = [
        (VALUE_KIND_I1, std::ptr::from_ref(&i1).cast()),
        (VALUE_KIND_I8, std::ptr::from_ref(&i8_value).cast()),
        (VALUE_KIND_I16, std::ptr::from_ref(&i16_value).cast()),
        (VALUE_KIND_I32, std::ptr::from_ref(&i32_value).cast()),
        (VALUE_KIND_I64, std::ptr::from_ref(&i64_value).cast()),
        (VALUE_KIND_U8, std::ptr::from_ref(&u8_value).cast()),
        (VALUE_KIND_U16, std::ptr::from_ref(&u16_value).cast()),
        (VALUE_KIND_U32, std::ptr::from_ref(&u32_value).cast()),
        (VALUE_KIND_U64, std::ptr::from_ref(&u64_value).cast()),
        (VALUE_KIND_F16, std::ptr::from_ref(&f16_value).cast()),
        (VALUE_KIND_F32, std::ptr::from_ref(&f32_value).cast()),
        (VALUE_KIND_F64, std::ptr::from_ref(&f64_value).cast()),
        (VALUE_KIND_PTR, std::ptr::from_ref(&pointer_value).cast()),
    ];

    for (kind, input) in cases {
        let array = unsafe { __faber_rt_v1_array_new(context, kind) };
        assert_eq!(array.status, STATUS_OK);
        assert_eq!(
            unsafe { __faber_rt_v1_array_push(context, array.value, kind, input) },
            STATUS_OK
        );

        let mut length = -1_i64;
        assert_eq!(
            unsafe { __faber_rt_v1_array_length(context, array.value, &mut length) },
            STATUS_OK
        );
        assert_eq!(length, 1);

        let mut output = 0_u64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array.value,
                    0,
                    kind,
                    std::ptr::from_mut(&mut output).cast(),
                )
            },
            STATUS_OK
        );
        let width = match kind {
            VALUE_KIND_I1 | VALUE_KIND_I8 | VALUE_KIND_U8 => 1,
            VALUE_KIND_I16 | VALUE_KIND_U16 | VALUE_KIND_F16 => 2,
            VALUE_KIND_I32 | VALUE_KIND_U32 | VALUE_KIND_F32 => 4,
            VALUE_KIND_I64 | VALUE_KIND_U64 | VALUE_KIND_F64 | VALUE_KIND_PTR => 8,
            _ => unreachable!(),
        };
        assert_eq!(&output.to_ne_bytes()[..width], unsafe {
            std::slice::from_raw_parts(input.cast::<u8>(), width)
        });
    }

    let source = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let target = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let first = 1_i64;
    let second = 2_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                source.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&first).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_extend(context, target.value, source.value) },
        STATUS_OK
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_set(
                context,
                target.value,
                0,
                VALUE_KIND_I64,
                std::ptr::from_ref(&second).cast(),
            )
        },
        STATUS_OK
    );
    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                target.value,
                0,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, second);

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_family_rejects_foreign_handles_kinds_and_bounds() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let value = 1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );

    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                context.cast(),
                0,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                -1,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                1,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                array.value,
                0,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array.value, ptr::null_mut()) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_push(context, array.value, VALUE_KIND_I64, ptr::null()) },
        STATUS_INVALID_ARGUMENT
    );
    let mut aligned = [0_u64; 2];
    let misaligned = unsafe { aligned.as_mut_ptr().cast::<u8>().add(1).cast() };
    assert_eq!(
        unsafe { __faber_rt_v1_array_push(context, array.value, VALUE_KIND_I64, misaligned) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_get(context, array.value, 0, VALUE_KIND_I64, misaligned) },
        STATUS_INVALID_ARGUMENT
    );
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array.value, misaligned.cast()) },
        STATUS_INVALID_ARGUMENT
    );

    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_value_preserving_methods_clone_query_reverse_and_range() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [1_i64, 2, 3, 4, 5] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    let mut output = 0_u8;
    let three = 3_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_contains(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&three).cast(),
                &mut output,
            )
        },
        STATUS_OK
    );
    assert_eq!(output, 1);
    assert_eq!(
        unsafe { __faber_rt_v1_array_is_empty(context, array.value, &mut output) },
        STATUS_OK
    );
    assert_eq!(output, 0);

    let clone = unsafe { __faber_rt_v1_array_clone(context, array.value) };
    assert_eq!(clone.status, STATUS_OK);
    assert_eq!(
        unsafe { __faber_rt_v1_array_reverse(context, clone.value) },
        STATUS_OK
    );
    assert_array_i64(context, clone.value, &[5, 4, 3, 2, 1]);
    assert_array_i64(context, array.value, &[1, 2, 3, 4, 5]);

    for (mode, first, second, expected) in [
        (ARRAY_RANGE_SLICE, 1, 4, &[2_i64, 3, 4][..]),
        (ARRAY_RANGE_TAKE, 2, 0, &[1_i64, 2][..]),
        (ARRAY_RANGE_TAKE_LAST, 2, 0, &[4_i64, 5][..]),
        (ARRAY_RANGE_DROP_FIRST, 2, 0, &[3_i64, 4, 5][..]),
    ] {
        let result =
            unsafe { __faber_rt_v1_array_range(context, array.value, mode, first, second) };
        assert_eq!(result.status, STATUS_OK);
        assert_array_i64(context, result.value, expected);
    }
    for (mode, first, second) in [
        (ARRAY_RANGE_TAKE, -1, 0),
        (ARRAY_RANGE_SLICE, 0, -1),
        (99, 0, 0),
    ] {
        let result =
            unsafe { __faber_rt_v1_array_range(context, array.value, mode, first, second) };
        assert_eq!(result.status, STATUS_INVALID_ARGUMENT);
        assert!(result.value.is_null());
    }

    unsafe { __faber_rt_v1_shutdown(context) };
}

fn assert_array_i64(
    context: *mut FaberRtContextV1,
    array: *mut std::ffi::c_void,
    expected: &[i64],
) {
    let mut length = -1_i64;
    assert_eq!(
        unsafe { __faber_rt_v1_array_length(context, array, &mut length) },
        STATUS_OK
    );
    assert_eq!(usize::try_from(length), Ok(expected.len()));
    for (index, expected) in expected.iter().enumerate() {
        let mut actual = 0_i64;
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_get(
                    context,
                    array,
                    index as i64,
                    VALUE_KIND_I64,
                    std::ptr::from_mut(&mut actual).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(&actual, expected);
    }
}

#[test]
fn array_option_methods_cover_access_empty_and_removal() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    for value in [10_i64, 20, 30] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    array.value,
                    VALUE_KIND_I64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }

    for (mode, index, expected) in [
        (ARRAY_OPTION_INDEX, 1, Some(20_i64)),
        (ARRAY_OPTION_FIRST, 0, Some(10)),
        (ARRAY_OPTION_LAST, 0, Some(30)),
        (ARRAY_OPTION_INDEX, -1, None),
        (ARRAY_OPTION_INDEX, 99, None),
        (ARRAY_OPTION_REMOVE_FIRST, 0, Some(10)),
        (ARRAY_OPTION_REMOVE_LAST, 0, Some(30)),
    ] {
        let result = unsafe { __faber_rt_v1_array_option(context, array.value, mode, index) };
        assert_eq!(result.status, STATUS_OK);
        let option = unsafe { &*result.value.cast::<RuntimeOption>() };
        assert_eq!(option.kind, VALUE_KIND_I64);
        assert_eq!(option_i64(option), expected);
    }
    assert_array_i64(context, array.value, &[20]);

    let invalid = unsafe { __faber_rt_v1_array_option(context, array.value, 99, 0) };
    assert_eq!(invalid.status, STATUS_INVALID_ARGUMENT);
    assert!(invalid.value.is_null());
    unsafe { __faber_rt_v1_shutdown(context) };
}

fn option_i64(option: &RuntimeOption) -> Option<i64> {
    match option.value {
        Some(array::RuntimeValue::I64(value)) => Some(value),
        None => None,
        _ => panic!("unexpected runtime option kind"),
    }
}

#[test]
fn option_family_produces_queries_unwraps_and_coalesces_shared_handles() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );
    let value = 42_i64;
    let fallback = 9_i64;
    let none = unsafe { __faber_rt_v1_option_none(context, VALUE_KIND_I64) };
    let some = unsafe {
        __faber_rt_v1_option_some(context, VALUE_KIND_I64, std::ptr::from_ref(&value).cast())
    };
    assert_eq!(none.status, STATUS_OK);
    assert_eq!(some.status, STATUS_OK);

    for (option, expected) in [(none.value, 0_u8), (some.value, 1_u8)] {
        let mut present = 99_u8;
        assert_eq!(
            unsafe {
                __faber_rt_v1_option_is_present(
                    context,
                    option,
                    VALUE_KIND_I64,
                    std::ptr::from_mut(&mut present).cast(),
                )
            },
            STATUS_OK
        );
        assert_eq!(present, expected);
    }

    let mut output = 0_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                some.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, value);
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get_or(
                context,
                none.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&fallback).cast(),
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, fallback);
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                none.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_INVALID_ARGUMENT
    );

    let array = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_push(
                context,
                array.value,
                VALUE_KIND_I64,
                std::ptr::from_ref(&value).cast(),
            )
        },
        STATUS_OK
    );
    let endpoint =
        unsafe { __faber_rt_v1_array_option(context, array.value, ARRAY_OPTION_FIRST, 0) };
    assert_eq!(
        unsafe {
            __faber_rt_v1_option_get(
                context,
                endpoint.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut output).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(output, value);
    unsafe { __faber_rt_v1_shutdown(context) };
}

#[test]
fn array_numeric_family_preserves_signedness_orders_and_sums() {
    let mut context = ptr::null_mut();
    assert_eq!(
        unsafe { __faber_rt_v1_init(0, ptr::null(), &mut context) },
        STATUS_OK
    );

    let unsigned = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_U32) };
    for value in [u32::MAX, 1] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    unsigned.value,
                    VALUE_KIND_U32,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, unsigned.value) },
        STATUS_OK
    );
    let mut first = 0_u32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_get(
                context,
                unsigned.value,
                0,
                VALUE_KIND_U32,
                std::ptr::from_mut(&mut first).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(first, 1);
    let mut unsigned_sum = 0_u32;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                unsigned.value,
                VALUE_KIND_U32,
                std::ptr::from_mut(&mut unsigned_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(unsigned_sum, 0);

    let floats = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_F64) };
    for value in [3.5_f64, -1.0, 2.0] {
        assert_eq!(
            unsafe {
                __faber_rt_v1_array_push(
                    context,
                    floats.value,
                    VALUE_KIND_F64,
                    std::ptr::from_ref(&value).cast(),
                )
            },
            STATUS_OK
        );
    }
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, floats.value) },
        STATUS_OK
    );
    let mut float_sum = 0.0_f64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                floats.value,
                VALUE_KIND_F64,
                std::ptr::from_mut(&mut float_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(float_sum, 4.5);

    let empty = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_I64) };
    let mut empty_sum = -1_i64;
    assert_eq!(
        unsafe {
            __faber_rt_v1_array_sum(
                context,
                empty.value,
                VALUE_KIND_I64,
                std::ptr::from_mut(&mut empty_sum).cast(),
            )
        },
        STATUS_OK
    );
    assert_eq!(empty_sum, 0);

    let unsupported = unsafe { __faber_rt_v1_array_new(context, VALUE_KIND_PTR) };
    assert_eq!(
        unsafe { __faber_rt_v1_array_sort(context, unsupported.value) },
        STATUS_INVALID_ARGUMENT
    );
    unsafe { __faber_rt_v1_shutdown(context) };
}
