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
        unsafe { __faber_runtime_diagnostic_nota_1_i64(context, 42) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_runtime_diagnostic_nota_1_i1(context, 1) },
        STATUS_OK
    );
    assert_eq!(
        unsafe { __faber_runtime_diagnostic_nota_1_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
    );
    assert_eq!(
        unsafe { __faber_runtime_diagnostic_mone_1_ptr(context, ptr::null()) },
        STATUS_UNSUPPORTED
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
