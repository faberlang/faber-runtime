use crate::frame::{self, FrameStatus, Scrinium};
use crate::Valor;

#[test]
fn runtime_echo_returns_opener_then_done() {
    let mut sermo = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut sermo, Valor::Textus("salve".into()));

    let item = frame::sermo_recv(&mut sermo).expect("echo item frame");
    assert_eq!(item.status, FrameStatus::Item);
    assert_eq!(
        item.parent_id.as_deref(),
        Some(sermo.conversation_id().as_str())
    );
    assert_eq!(item.call, "runtime:echo");
    assert_eq!(item.data, Valor::Textus("salve".into()));
    assert_eq!(item.from.as_deref(), Some("faber-runtime"));

    let done = frame::sermo_recv(&mut sermo).expect("echo terminal frame");
    assert_eq!(done.status, FrameStatus::Done);
    assert!(sermo.incoming_drained());
    assert!(frame::sermo_recv(&mut sermo).is_none());
}

// ---- `sermo ↦ T` materializers ----

#[test]
fn sermo_materialize_vacuum_drains_to_terminal() {
    let mut sermo = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut sermo, Valor::Textus("salve".into()));
    assert!(!sermo.incoming_drained());
    frame::sermo_materialize_vacuum(&mut sermo);
    assert!(sermo.incoming_drained());
}

#[test]
fn sermo_materialize_textus_concatenates_string_frames() {
    let mut sermo = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut sermo, Valor::Textus("salve, munde".into()));
    let out = frame::sermo_materialize_textus(&mut sermo);
    assert_eq!(out, "salve, munde");
}

#[test]
fn try_sermo_materialize_textus_rejects_non_text_frames() {
    let mut sermo = frame::sermo_open("test:skip-frames");
    sermo.push_incoming(Scrinium {
        id: "t1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:skip-frames".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("alpha".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "n1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:skip-frames".into(),
        status: FrameStatus::Item,
        data: Valor::Numerus(42),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:skip-frames".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let err =
        frame::try_sermo_materialize_textus(&mut sermo).expect_err("non-text frame must fail");
    assert_eq!(err.issue, "frame_textus_payload_not_textus");
    assert!(sermo.incoming_drained());
}

#[test]
fn sermo_materialize_octeti_concatenates_bytes() {
    let mut sermo = frame::sermo_open("test:bytes");
    sermo.push_incoming(Scrinium {
        id: "b1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Item,
        data: Valor::Lista(vec![Valor::Numerus(1), Valor::Numerus(2)]),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "b2".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Item,
        data: Valor::Lista(vec![Valor::Numerus(3)]),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let out = frame::sermo_materialize_octeti(&mut sermo);
    assert_eq!(out, vec![1u8, 2, 3]);
}

#[test]
fn sermo_materialize_octeti_accepts_dense_byte_payload() {
    let mut sermo = frame::sermo_open("test:bytes");
    sermo.push_incoming(Scrinium {
        id: "b1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Byte,
        data: Valor::Octeti(vec![1, 2, 3, 4]),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let out = frame::sermo_materialize_octeti(&mut sermo);
    assert_eq!(out, vec![1u8, 2, 3, 4]);
}

#[test]
fn try_sermo_materialize_octeti_rejects_out_of_range_bytes() {
    let mut sermo = frame::sermo_open("test:bytes");
    sermo.push_incoming(Scrinium {
        id: "b1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Item,
        data: Valor::Lista(vec![Valor::Numerus(300)]),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:bytes".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let err = frame::try_sermo_materialize_octeti(&mut sermo).expect_err("invalid byte must fail");
    assert_eq!(err.issue, "frame_octeti_byte_out_of_range");
}

#[test]
fn sermo_materialize_valor_returns_first_content_frame() {
    let mut sermo = frame::sermo_open("test:multiple");
    sermo.push_incoming(Scrinium {
        id: "c1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:multiple".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("first".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "c2".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:multiple".into(),
        status: FrameStatus::Item,
        data: Valor::Numerus(42),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:multiple".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let out = frame::sermo_materialize_valor(&mut sermo);
    assert_eq!(out, Valor::Textus("first".into()));
}

#[test]
fn sermo_materialize_valor_returns_nihil_when_no_content() {
    let mut sermo = frame::sermo_open("test:empty");
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:empty".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let out = frame::sermo_materialize_valor(&mut sermo);
    assert_eq!(out, Valor::Nihil);
}

#[test]
fn sermo_materialize_lista_collects_extractable_frames() {
    let mut sermo = frame::sermo_open("test:lines");
    sermo.push_incoming(Scrinium {
        id: "l1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:lines".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("one".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "l2".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:lines".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("two".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:lines".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let out: Vec<String> = frame::sermo_materialize_lista(&mut sermo);
    assert_eq!(out, vec!["one".to_string(), "two".to_string()]);
}

#[test]
fn try_sermo_materialize_lista_rejects_unextractable_frame() {
    let mut sermo = frame::sermo_open("test:lines");
    sermo.push_incoming(Scrinium {
        id: "l1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:lines".into(),
        status: FrameStatus::Item,
        data: Valor::Numerus(1),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:lines".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let err =
        frame::try_sermo_materialize_lista::<String>(&mut sermo).expect_err("bad item must fail");
    assert_eq!(err.issue, "frame_lista_payload_element_type_mismatch");
}

#[test]
fn sermo_materialize_scalar_single_frame_succeeds() {
    let mut sermo = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut sermo, Valor::Numerus(7));
    let out: i64 = frame::sermo_materialize_scalar(&mut sermo);
    assert_eq!(out, 7);
}

#[test]
fn tempus_nunc_route_materializes_instans() {
    let mut sermo = frame::sermo_open("tempus:nunc");
    let out = frame::sermo_materialize_instans(&mut sermo, crate::InstansPraecisio::Nanosecunda);

    assert_eq!(out.praecisio(), crate::InstansPraecisio::Nanosecunda);
}

#[test]
fn solum_lege_route_materializes_scalar_target_shape() {
    let stem = frame::next_frame_id();
    let text_path = std::env::temp_dir().join(format!("{stem}.txt"));
    let bin_path = std::env::temp_dir().join(format!("{stem}.bin"));
    std::fs::write(&text_path, "prima\nsecunda\n").expect("write text fixture");
    std::fs::write(&bin_path, [1u8, 2, 3]).expect("write byte fixture");

    let text_path = text_path.to_string_lossy().into_owned();
    let bin_path = bin_path.to_string_lossy().into_owned();

    let mut text_sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut text_sermo, Valor::Textus(text_path.clone()));
    let text: String = frame::sermo_materialize_scalar(&mut text_sermo);
    assert_eq!(text, "prima\nsecunda\n");

    let mut lines_sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut lines_sermo, Valor::Textus(text_path.clone()));
    let lines: Vec<String> = frame::sermo_materialize_scalar(&mut lines_sermo);
    assert_eq!(lines, vec!["prima".to_owned(), "secunda".to_owned()]);

    let mut bytes_sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut bytes_sermo, Valor::Textus(bin_path.clone()));
    let bytes: Vec<u8> = frame::sermo_materialize_scalar(&mut bytes_sermo);
    assert_eq!(bytes, vec![1, 2, 3]);

    let _ = std::fs::remove_file(text_path);
    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn solum_partem_route_materializes_dense_bounded_byte_range() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    std::fs::write(&path, [10u8, 11, 12, 13, 14]).expect("write byte range fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:partem");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Numerus(1),
            Valor::Numerus(3),
        ]),
    );
    let chunk = frame::sermo_recv(&mut sermo).expect("byte frame");
    assert_eq!(chunk.status, FrameStatus::Byte);
    assert_eq!(chunk.data, Valor::Octeti(vec![11, 12, 13]));
    let done = frame::sermo_recv(&mut sermo).expect("done frame");
    assert_eq!(done.status, FrameStatus::Done);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_partem_route_materializes_large_range_without_valor_list() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    let mut data = vec![42u8; 2 * 1024 * 1024];
    data[0] = 7;
    let last = data.len() - 1;
    data[last] = 9;
    std::fs::write(&path, &data).expect("write large range fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:partem");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Numerus(0),
            Valor::Numerus(data.len() as i64),
        ]),
    );
    let chunk = frame::sermo_recv(&mut sermo).expect("large byte frame");
    assert_eq!(chunk.status, FrameStatus::Byte);
    let Valor::Octeti(bytes) = chunk.data else {
        panic!("solum:partem must return dense octeti");
    };
    assert_eq!(bytes.len(), data.len());
    assert_eq!(bytes[0], 7);
    assert_eq!(bytes[bytes.len() - 1], 9);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_partem_route_materializes_octeti() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    std::fs::write(&path, [20u8, 21, 22, 23, 24]).expect("write octeti range fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:partem");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Numerus(2),
            Valor::Numerus(2),
        ]),
    );
    let bytes = frame::sermo_materialize_octeti(&mut sermo);
    assert_eq!(bytes, vec![22, 23]);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_mensura_route_materializes_file_size() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    std::fs::write(&path, [30u8, 31, 32, 33]).expect("write size fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:mensura");
    frame::sermo_set_opener(&mut sermo, Valor::Textus(path.clone()));
    let size: i64 = frame::sermo_materialize_scalar(&mut sermo);
    assert_eq!(size, 4);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_inveni_route_materializes_pattern_offset() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    std::fs::write(&path, b"prefix-general.file_type-suffix").expect("write search fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:inveni");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Textus("general.file_type".to_owned()),
            Valor::Numerus(0),
            Valor::Numerus(64),
        ]),
    );
    let offset: i64 = frame::sermo_materialize_scalar(&mut sermo);
    assert_eq!(offset, 7);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_exstat_route_materializes_bool() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    std::fs::write(&path, "present").expect("write existence fixture");
    let missing = path.with_extension("missing");
    let path = path.to_string_lossy().into_owned();
    let missing = missing.to_string_lossy().into_owned();

    let mut present_sermo = frame::sermo_open("solum:exstat");
    frame::sermo_set_opener(&mut present_sermo, Valor::Textus(path.clone()));
    assert!(frame::sermo_materialize_scalar::<bool>(&mut present_sermo));

    let mut missing_sermo = frame::sermo_open("solum:exstat");
    frame::sermo_set_opener(&mut missing_sermo, Valor::Textus(missing));
    assert!(!frame::sermo_materialize_scalar::<bool>(&mut missing_sermo));

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_path_bool_routes_materialize_bool() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    let dir = std::env::temp_dir().join(format!("{}.dir", frame::next_frame_id()));
    std::fs::write(&path, "present").expect("write path bool fixture");
    std::fs::create_dir(&dir).expect("create path bool directory");

    let file_path = path.to_string_lossy().into_owned();
    let dir_path = dir.to_string_lossy().into_owned();

    let mut regular_sermo = frame::sermo_open("solum:regularene");
    frame::sermo_set_opener(&mut regular_sermo, Valor::Textus(file_path.clone()));
    assert!(frame::sermo_materialize_scalar::<bool>(&mut regular_sermo));

    let mut dir_regular_sermo = frame::sermo_open("solum:regularene");
    frame::sermo_set_opener(&mut dir_regular_sermo, Valor::Textus(dir_path.clone()));
    assert!(!frame::sermo_materialize_scalar::<bool>(
        &mut dir_regular_sermo
    ));

    let mut dir_sermo = frame::sermo_open("solum:directoriumne");
    frame::sermo_set_opener(&mut dir_sermo, Valor::Textus(dir_path));
    assert!(frame::sermo_materialize_scalar::<bool>(&mut dir_sermo));

    let mut readable_sermo = frame::sermo_open("solum:legibilene");
    frame::sermo_set_opener(&mut readable_sermo, Valor::Textus(file_path.clone()));
    assert!(frame::sermo_materialize_scalar::<bool>(&mut readable_sermo));

    let _ = std::fs::remove_file(file_path);
    let _ = std::fs::remove_dir(dir);
}

#[test]
fn solum_scribe_route_materializes_vacuum_after_writing_file() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    let path = path.to_string_lossy().into_owned();
    let mut sermo = frame::sermo_open("solum:scribe");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Textus("salve".to_owned()),
        ]),
    );

    frame::sermo_materialize_vacuum(&mut sermo);

    assert_eq!(
        std::fs::read_to_string(&path).expect("read written file"),
        "salve"
    );
    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_dele_route_materializes_vacuum_after_removing_file() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    std::fs::write(&path, "stale").expect("write temp file");
    let path = path.to_string_lossy().into_owned();
    let mut sermo = frame::sermo_open("solum:dele");
    frame::sermo_set_opener(&mut sermo, Valor::Textus(path.clone()));

    frame::sermo_materialize_vacuum(&mut sermo);

    assert!(!std::path::Path::new(&path).exists());
}

#[test]
fn processus_exsequi_route_materializes_stdout() {
    let mut sermo = frame::sermo_open("processus:exsequi");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Textus("printf runtime-process-ok".into()),
    );

    let output = frame::sermo_materialize_textus(&mut sermo);

    assert_eq!(output, "runtime-process-ok");
}

#[test]
fn processus_captura_route_materializes_status_stdout_and_stderr() {
    let mut sermo = frame::sermo_open("processus:captura");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus("sh".into()),
            Valor::Textus("-c".into()),
            Valor::Textus("printf out; printf err >&2; exit 7".into()),
        ]),
    );

    let output = frame::sermo_materialize_valor(&mut sermo);

    let Valor::Tabula(fields) = output else {
        panic!("expected processus:captura to return a tabula");
    };
    assert_eq!(fields.get("status"), Some(&Valor::Numerus(7)));
    assert_eq!(fields.get("stdout"), Some(&Valor::Textus("out".into())));
    assert_eq!(fields.get("stderr"), Some(&Valor::Textus("err".into())));
}

#[test]
fn solum_parens_route_materializes_parent_path() {
    let mut sermo = frame::sermo_open("solum:parens");
    frame::sermo_set_opener(&mut sermo, Valor::Textus("/tmp/faber/path.txt".into()));

    let output = frame::sermo_materialize_textus(&mut sermo);

    assert_eq!(output, "/tmp/faber");
}

#[test]
fn try_sermo_materialize_scalar_returns_error_for_bad_payload() {
    let mut sermo = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut sermo, Valor::Textus("not a number".into()));
    let err =
        frame::try_sermo_materialize_scalar::<i64>(&mut sermo).expect_err("bad scalar must fail");
    assert_eq!(err.issue, "frame_scalar_payload_target_type_mismatch");
}

#[test]
fn try_sermo_materialize_vacuum_fails_on_error_terminal() {
    let mut sermo = frame::sermo_open("test:error");
    sermo.push_incoming(Scrinium {
        id: "err".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:error".into(),
        status: FrameStatus::Error,
        data: Valor::Textus("boom".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    let err =
        frame::try_sermo_materialize_vacuum(&mut sermo).expect_err("error terminal must fail");
    assert_eq!(err.issue, "frame_materialization_terminal_error");
}

#[test]
#[should_panic(expected = "no content frame")]
fn sermo_materialize_scalar_zero_content_frames_panics() {
    let mut sermo = frame::sermo_open("test:empty");
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:empty".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let _: i64 = frame::sermo_materialize_scalar(&mut sermo);
}

#[test]
#[should_panic(expected = "more than one content frame")]
fn sermo_materialize_scalar_multiple_content_frames_panics() {
    let mut sermo = frame::sermo_open("test:many");
    sermo.push_incoming(Scrinium {
        id: "c1".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:many".into(),
        status: FrameStatus::Item,
        data: Valor::Numerus(1),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "c2".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:many".into(),
        status: FrameStatus::Item,
        data: Valor::Numerus(2),
        created_ms: 0,
        from: None,
        trace: None,
    });
    sermo.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:many".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    let _: i64 = frame::sermo_materialize_scalar(&mut sermo);
}
