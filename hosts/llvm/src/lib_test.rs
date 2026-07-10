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
