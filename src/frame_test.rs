use crate::frame::{
    self, Cancellation, DispatchError, FrameStatus, HostDispatch, ResponseSender, Scrinium,
    SermoRequest,
};
use crate::Valor;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::task::{Context, Poll, Wake, Waker};

#[derive(Default)]
struct CountingWake {
    count: AtomicUsize,
}

impl Wake for CountingWake {
    fn wake(self: Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.count.fetch_add(1, Ordering::SeqCst);
    }
}

fn test_waker(wake: &Arc<CountingWake>) -> Waker {
    Waker::from(wake.clone())
}

fn block_on<F: Future>(future: F) -> F::Output {
    let wake = Arc::new(CountingWake::default());
    let waker = test_waker(&wake);
    let mut cx = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match Future::poll(Pin::as_mut(&mut future), &mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}

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

struct InlineDispatch;

impl HostDispatch for InlineDispatch {
    fn start(
        &self,
        request: SermoRequest,
        responses: ResponseSender,
        _cancellation: Cancellation,
    ) -> Result<(), DispatchError> {
        std::thread::spawn(move || {
            responses.item(request.opener).expect("inline item");
            responses.done().expect("inline done");
        });
        Ok(())
    }
}

#[test]
fn explicit_dispatcher_does_not_use_global_installation() {
    let mut sermo = frame::sermo_open_with_dispatch("custom:echo", Arc::new(InlineDispatch));
    frame::sermo_set_opener(&mut sermo, Valor::Textus("isolated".into()));

    let item = frame::sermo_recv(&mut sermo).expect("explicit item");
    assert_eq!(item.status, FrameStatus::Item);
    assert_eq!(item.data, Valor::Textus("isolated".into()));
    assert_eq!(
        frame::sermo_recv(&mut sermo).expect("explicit done").status,
        FrameStatus::Done
    );
}

#[test]
fn sermo_recv_async_registers_runtime_neutral_wake() {
    let (mut sermo, _sender, _cancellation) = frame::test_response_sender("test:manual-wake");
    let wake = Arc::new(CountingWake::default());
    let waker = test_waker(&wake);
    let mut cx = Context::from_waker(&waker);
    {
        let mut future = Box::pin(frame::sermo_recv_async(&mut sermo));
        assert!(matches!(
            Future::poll(Pin::as_mut(&mut future), &mut cx),
            Poll::Pending
        ));
    }

    sermo.push_incoming(Scrinium {
        id: "manual".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "test:manual-wake".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("awakened".into()),
        created_ms: 0,
        from: Some("test".into()),
        trace: None,
    });
    assert_eq!(wake.count.load(Ordering::SeqCst), 1);

    let frame = block_on(frame::sermo_recv_async(&mut sermo)).expect("manual frame");
    assert_eq!(frame.data, Valor::Textus("awakened".into()));
}

#[test]
fn dropping_pending_async_receive_cancels_runtime_response() {
    let mut sermo = frame::sermo_open("tempus:dormiet");
    frame::sermo_set_opener(&mut sermo, Valor::Numerus(25));
    let wake = Arc::new(CountingWake::default());
    let waker = test_waker(&wake);
    let mut cx = Context::from_waker(&waker);

    {
        let mut future = Box::pin(frame::sermo_recv_async(&mut sermo));
        assert!(matches!(
            Future::poll(Pin::as_mut(&mut future), &mut cx),
            Poll::Pending
        ));
    }

    let terminal = frame::sermo_recv(&mut sermo).expect("cancel terminal");
    assert_eq!(terminal.status, FrameStatus::Cancel);
    assert!(sermo.incoming_drained());
}

#[test]
fn unsupported_route_resolves_to_error_terminal() {
    let mut sermo = frame::sermo_open("missing:route");
    frame::sermo_set_opener(&mut sermo, Valor::Nihil);

    let frame = frame::sermo_recv(&mut sermo).expect("unsupported route terminal");

    assert_eq!(frame.status, FrameStatus::Error);
    assert_eq!(frame.call, "missing:route");
    assert!(
        matches!(frame.data, Valor::Textus(message) if message.contains("unsupported ad route"))
    );
}

#[test]
fn response_sender_enforces_one_terminal_frame() {
    let (_sermo, sender, _cancellation) = frame::test_response_sender("test:sender-terminal");

    sender.done().expect("first terminal succeeds");
    let err = sender
        .error("late error")
        .expect_err("second terminal must fail");
    assert_eq!(err.issue, "frame_response_terminal_already_sent");
    let err = sender
        .item(Valor::Textus("late".into()))
        .expect_err("content after terminal must fail");
    assert_eq!(err.issue, "frame_response_after_terminal");
}

#[test]
fn response_sender_keeps_terminal_last_across_concurrent_clones() {
    for _ in 0..200 {
        let (mut sermo, sender, _cancellation) =
            frame::test_response_sender("test:sender-concurrent-terminal");
        let content_sender = sender.clone();
        let barrier = Arc::new(Barrier::new(3));
        let content_barrier = Arc::clone(&barrier);
        let content = std::thread::spawn(move || {
            content_barrier.wait();
            content_sender.item(Valor::Textus("item".into()))
        });
        let terminal_barrier = Arc::clone(&barrier);
        let terminal = std::thread::spawn(move || {
            terminal_barrier.wait();
            sender.done()
        });

        barrier.wait();
        let _ = content.join().expect("content producer");
        let _ = terminal.join().expect("terminal producer");

        let mut statuses = Vec::new();
        while let Some(frame) = frame::sermo_recv(&mut sermo) {
            statuses.push(frame.status);
            if frame.status.is_terminal() {
                break;
            }
        }
        assert!(statuses.last().is_some_and(|status| status.is_terminal()));
    }
}

#[test]
fn dropped_last_response_sender_enqueues_producer_dropped_error() {
    let (mut sermo, sender, _cancellation) = frame::test_response_sender("test:producer-drop");

    drop(sender);
    let frame = frame::sermo_recv(&mut sermo).expect("producer drop terminal");

    assert_eq!(frame.status, FrameStatus::Error);
    assert!(matches!(frame.data, Valor::Textus(message) if message.contains("producer dropped")));
}

#[test]
fn response_sender_suppresses_content_after_cancellation() {
    let (_sermo, sender, cancellation) = frame::test_response_sender("test:cancelled-response");

    cancellation.cancel();
    let err = sender
        .item(Valor::Textus("late".into()))
        .expect_err("content after cancellation must fail");
    assert_eq!(err.issue, "frame_response_cancelled");
    sender.cancel().expect("cancel terminal still succeeds");
}

#[test]
fn async_receive_poll_does_not_sleep_for_timer_route() {
    let mut sermo = frame::sermo_open("tempus:dormiet");
    frame::sermo_set_opener(&mut sermo, Valor::Numerus(75));
    let wake = Arc::new(CountingWake::default());
    let waker = test_waker(&wake);
    let mut cx = Context::from_waker(&waker);
    let started = std::time::Instant::now();

    let mut future = Box::pin(frame::sermo_recv_async(&mut sermo));
    let polled = Future::poll(Pin::as_mut(&mut future), &mut cx);

    assert!(matches!(polled, Poll::Pending));
    assert!(
        started.elapsed() < std::time::Duration::from_millis(25),
        "pending async receive poll must not run the timer route synchronously"
    );
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
fn solum_carpe_route_materializes_line_lista() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    std::fs::write(&path, "alpha\nbeta\ngamma\n").expect("write carpe fixture");
    let path_s = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:carpe");
    frame::sermo_set_opener(&mut sermo, Valor::Textus(path_s));
    let lines: Vec<String> = frame::sermo_materialize_lista(&mut sermo);
    assert_eq!(
        lines,
        vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()]
    );

    let _ = std::fs::remove_file(path);
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

    // Contract: codegen uses try_sermo_materialize_lista (one Item per line), same as carpe.
    let mut lines_sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut lines_sermo, Valor::Textus(text_path.clone()));
    let lines: Vec<String> = frame::sermo_materialize_lista(&mut lines_sermo);
    assert_eq!(lines, vec!["prima".to_owned(), "secunda".to_owned()]);

    let mut bytes_sermo = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut bytes_sermo, Valor::Textus(bin_path.clone()));
    let bytes: Vec<u8> = frame::sermo_materialize_scalar(&mut bytes_sermo);
    assert_eq!(bytes, vec![1, 2, 3]);

    // Generic monomorph path (provider lege<T>): auto picks lista for Vec<String>.
    let mut auto_lines = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut auto_lines, Valor::Textus(text_path.clone()));
    let auto: Vec<String> = frame::try_sermo_materialize_auto(&mut auto_lines).expect("auto lista");
    assert_eq!(auto, vec!["prima".to_owned(), "secunda".to_owned()]);

    let mut auto_text = frame::sermo_open("solum:lege");
    frame::sermo_set_opener(&mut auto_text, Valor::Textus(text_path.clone()));
    let auto_s: String = frame::try_sermo_materialize_auto(&mut auto_text).expect("auto text");
    assert_eq!(auto_s, "prima\nsecunda\n");

    let _ = std::fs::remove_file(text_path);
    let _ = std::fs::remove_file(bin_path);
}

#[test]
fn solum_inveni_empty_pattern_is_found_at_start() {
    let path = std::env::temp_dir().join(format!("{}.bin", frame::next_frame_id()));
    std::fs::write(&path, b"payload").expect("write search fixture");
    let path = path.to_string_lossy().into_owned();

    let mut sermo = frame::sermo_open("solum:inveni");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Textus(String::new()),
            Valor::Numerus(3),
            Valor::Numerus(8),
        ]),
    );
    let offset: i64 = frame::sermo_materialize_scalar(&mut sermo);
    assert_eq!(offset, 3);

    let _ = std::fs::remove_file(path);
}

#[test]
fn solum_dele_missing_path_is_success() {
    let missing = std::env::temp_dir().join(format!("{}.missing", frame::next_frame_id()));
    let missing = missing.to_string_lossy().into_owned();
    assert!(!std::path::Path::new(&missing).exists());

    let mut sermo = frame::sermo_open("solum:dele");
    frame::sermo_set_opener(&mut sermo, Valor::Textus(missing));
    frame::sermo_materialize_vacuum(&mut sermo);
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
fn externally_supplied_incoming_frames_suppress_runtime_fallback() {
    let path = std::env::temp_dir().join(format!("{}.txt", frame::next_frame_id()));
    let path = path.to_string_lossy().into_owned();
    let mut sermo = frame::sermo_open("solum:appone");
    frame::sermo_set_opener(
        &mut sermo,
        Valor::Lista(vec![
            Valor::Textus(path.clone()),
            Valor::Textus("salve".to_owned()),
        ]),
    );
    sermo.push_incoming(Scrinium {
        id: "host-done".into(),
        parent_id: Some(sermo.conversation_id()),
        call: "solum:appone".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: Some("host".into()),
        trace: None,
    });

    frame::sermo_materialize_vacuum(&mut sermo);

    assert!(!std::path::Path::new(&path).exists());
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

#[test]
fn async_materializer_twins_mirror_sync_materializers() {
    let mut vacuum = frame::sermo_open("test:vacuum-async");
    vacuum.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(vacuum.conversation_id()),
        call: "test:vacuum-async".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    block_on(frame::sermo_materialize_vacuum_async(&mut vacuum));
    assert!(vacuum.incoming_drained());

    let mut textus = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut textus, Valor::Textus("salve".into()));
    assert_eq!(
        block_on(frame::sermo_materialize_textus_async(&mut textus)),
        "salve"
    );

    let mut octeti = frame::sermo_open("test:octeti-async");
    octeti.push_incoming(Scrinium {
        id: "bytes".into(),
        parent_id: Some(octeti.conversation_id()),
        call: "test:octeti-async".into(),
        status: FrameStatus::Byte,
        data: Valor::Octeti(vec![1, 2, 3]),
        created_ms: 0,
        from: None,
        trace: None,
    });
    octeti.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(octeti.conversation_id()),
        call: "test:octeti-async".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    assert_eq!(
        block_on(frame::sermo_materialize_octeti_async(&mut octeti)),
        vec![1, 2, 3]
    );

    let mut valor = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut valor, Valor::Numerus(7));
    assert_eq!(
        block_on(frame::sermo_materialize_valor_async(&mut valor)),
        Valor::Numerus(7)
    );

    let mut lista = frame::sermo_open("test:lista-async");
    lista.push_incoming(Scrinium {
        id: "one".into(),
        parent_id: Some(lista.conversation_id()),
        call: "test:lista-async".into(),
        status: FrameStatus::Item,
        data: Valor::Textus("one".into()),
        created_ms: 0,
        from: None,
        trace: None,
    });
    lista.push_incoming(Scrinium {
        id: "done".into(),
        parent_id: Some(lista.conversation_id()),
        call: "test:lista-async".into(),
        status: FrameStatus::Done,
        data: Valor::Nihil,
        created_ms: 0,
        from: None,
        trace: None,
    });
    assert_eq!(
        block_on(frame::sermo_materialize_lista_async::<String>(&mut lista)),
        vec!["one".to_owned()]
    );

    let mut scalar = frame::sermo_open("runtime:echo");
    frame::sermo_set_opener(&mut scalar, Valor::Numerus(9));
    assert_eq!(
        block_on(frame::sermo_materialize_scalar_async::<i64>(&mut scalar)),
        9
    );

    let mut instans = frame::sermo_open("tempus:nunc");
    let materialized = block_on(frame::sermo_materialize_instans_async(
        &mut instans,
        crate::InstansPraecisio::Nanosecunda,
    ));
    assert_eq!(
        materialized.praecisio(),
        crate::InstansPraecisio::Nanosecunda
    );
}
