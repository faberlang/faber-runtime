use super::*;

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

    assert_eq!(one.status, STATUS_OK);
    assert_eq!(reordered.status, STATUS_OK);
    assert_eq!(float.status, STATUS_OK);
    assert_eq!(three.status, STATUS_OK);
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
        unsafe { &*three.value.cast::<RuntimeText>() }._value,
        "1/2/3"
    );

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
    let i32_value = -32_i32;
    let i64_value = -64_i64;
    let f32_value = 3.25_f32;
    let f64_value = 6.5_f64;
    let pointer_value = context.cast::<std::ffi::c_void>();
    let cases = [
        (VALUE_KIND_I1, std::ptr::from_ref(&i1).cast()),
        (VALUE_KIND_I8, std::ptr::from_ref(&i8_value).cast()),
        (VALUE_KIND_I32, std::ptr::from_ref(&i32_value).cast()),
        (VALUE_KIND_I64, std::ptr::from_ref(&i64_value).cast()),
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
            VALUE_KIND_I1 | VALUE_KIND_I8 => 1,
            VALUE_KIND_I32 | VALUE_KIND_F32 => 4,
            VALUE_KIND_I64 | VALUE_KIND_F64 | VALUE_KIND_PTR => 8,
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
