use faber::{
    builtin_route_frames, install_host_dispatch, Cancellation, DispatchError, FrameStatus,
    HostDispatch, ResponseSender, SermoRequest,
};
use std::panic::{self, AssertUnwindSafe};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;

const DEFAULT_WORKERS: usize = 4;
const DEFAULT_QUEUE_CAPACITY: usize = 64;

type JobReceiver = Arc<Mutex<Receiver<NativeJob>>>;

struct NativeJob {
    request: SermoRequest,
    responses: ResponseSender,
    cancellation: Cancellation,
}

#[derive(Clone)]
pub struct NativeHost {
    queue: SyncSender<NativeJob>,
}

impl NativeHost {
    pub fn new() -> Self {
        Self::with_limits(DEFAULT_WORKERS, DEFAULT_QUEUE_CAPACITY)
    }

    pub fn with_limits(workers: usize, queue_capacity: usize) -> Self {
        assert!(workers > 0, "native host requires at least one worker");
        let (queue, receiver) = sync_channel(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        for index in 0..workers {
            spawn_worker(index, Arc::clone(&receiver));
        }
        Self { queue }
    }
}

impl Default for NativeHost {
    fn default() -> Self {
        Self::new()
    }
}

impl HostDispatch for NativeHost {
    fn start(
        &self,
        request: SermoRequest,
        responses: ResponseSender,
        cancellation: Cancellation,
    ) -> Result<(), DispatchError> {
        if !supports_native_route(&request.route) {
            return Err(DispatchError::new(
                "native_host_unsupported_route",
                format!("unsupported native host route `{}`", request.route),
            ));
        }
        let job = NativeJob {
            request,
            responses,
            cancellation,
        };
        match self.queue.try_send(job) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(DispatchError::new(
                "native_host_queue_saturated",
                "native host worker queue is saturated",
            )),
            Err(TrySendError::Disconnected(_)) => Err(DispatchError::new(
                "native_host_shutdown",
                "native host worker queue is shut down",
            )),
        }
    }
}

pub fn install() -> Result<(), DispatchError> {
    install_host_dispatch(Arc::new(NativeHost::new()))
}

fn spawn_worker(index: usize, receiver: JobReceiver) {
    let name = format!("faber-native-host-{index}");
    thread::Builder::new()
        .name(name)
        .spawn(move || run_worker(receiver))
        .expect("spawn native host worker");
}

fn run_worker(receiver: JobReceiver) {
    loop {
        let job = {
            let receiver = receiver
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            receiver.recv()
        };
        let Ok(job) = job else {
            return;
        };
        run_job(job);
    }
}

fn run_job(job: NativeJob) {
    if job.cancellation.is_cancelled() {
        let _ = job.responses.cancel();
        return;
    }
    let responses = job.responses.clone();
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let frames = builtin_route_frames(job.request);
        send_frames(frames, job.responses, &job.cancellation);
    }));
    if result.is_err() {
        let _ = responses.error("native host worker panicked");
    }
}

fn send_frames(
    frames: Vec<(FrameStatus, faber::Valor)>,
    responses: ResponseSender,
    cancellation: &Cancellation,
) {
    for (status, data) in frames {
        if cancellation.is_cancelled() && !status.is_terminal() {
            let _ = responses.cancel();
            return;
        }
        if cancellation.is_cancelled() && status == FrameStatus::Done {
            let _ = responses.cancel();
            return;
        }
        let _ = responses.send(status, data);
    }
}

fn supports_native_route(route: &str) -> bool {
    route.starts_with("tempus:")
        || route.starts_with("solum:")
        || route.starts_with("processus:")
        || route.starts_with("runtime:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use faber::frame;
    use faber::{FrameStatus, HostDispatch, Valor};
    use std::time::{Duration, Instant};

    #[test]
    fn timer_route_hands_off_without_blocking_start() {
        let host = NativeHost::with_limits(1, 2);
        let (mut sermo, responses, cancellation) = frame::test_response_sender("tempus:dormiet");
        frame::sermo_set_opener(&mut sermo, Valor::Numerus(40));
        let request = sermo.first_outgoing().expect("request frame");
        let start = Instant::now();

        host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "tempus:dormiet".to_owned(),
                opener: request.data,
                target: None,
            },
            responses,
            cancellation,
        )
        .expect("dispatch");

        assert!(
            start.elapsed() < Duration::from_millis(20),
            "start must not wait for timer work"
        );
        let terminal = frame::sermo_recv(&mut sermo).expect("terminal");
        assert_eq!(terminal.status, FrameStatus::Done);
    }

    #[test]
    fn filesystem_route_materializes_on_worker() {
        let host = NativeHost::with_limits(1, 2);
        let dir = std::env::temp_dir().join(format!("faber-native-host-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("read.txt");
        std::fs::write(&path, "native file").expect("write fixture");
        let (mut sermo, responses, cancellation) = frame::test_response_sender("solum:lege");
        let opener = Valor::Textus(path.to_string_lossy().into_owned());
        frame::sermo_set_opener(&mut sermo, opener.clone());

        host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "solum:lege".to_owned(),
                opener,
                target: Some("alloc::string::String"),
            },
            responses,
            cancellation,
        )
        .expect("dispatch");

        let item = frame::sermo_recv(&mut sermo).expect("item");
        assert_eq!(item.data, Valor::Textus("native file".to_owned()));
        let terminal = frame::sermo_recv(&mut sermo).expect("terminal");
        assert_eq!(terminal.status, FrameStatus::Done);
    }

    #[test]
    fn process_route_materializes_stdout_on_worker() {
        let host = NativeHost::with_limits(1, 2);
        let (mut sermo, responses, cancellation) = frame::test_response_sender("processus:exsequi");
        let opener = Valor::Textus("printf native-process-ok".to_owned());
        frame::sermo_set_opener(&mut sermo, opener.clone());

        host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "processus:exsequi".to_owned(),
                opener,
                target: None,
            },
            responses,
            cancellation,
        )
        .expect("dispatch");

        let item = frame::sermo_recv(&mut sermo).expect("item");
        assert_eq!(item.data, Valor::Textus("native-process-ok".to_owned()));
        let terminal = frame::sermo_recv(&mut sermo).expect("terminal");
        assert_eq!(terminal.status, FrameStatus::Done);
    }

    #[test]
    fn saturated_queue_rejects_without_accepting_work() {
        let host = NativeHost::with_limits(1, 1);
        let mut held = Vec::new();
        for _ in 0..3 {
            let (mut sermo, responses, cancellation) =
                frame::test_response_sender("tempus:dormiet");
            frame::sermo_set_opener(&mut sermo, Valor::Numerus(200));
            let result = host.start(
                SermoRequest {
                    conversation_id: sermo.conversation_id(),
                    route: "tempus:dormiet".to_owned(),
                    opener: Valor::Numerus(200),
                    target: None,
                },
                responses,
                cancellation,
            );
            held.push((sermo, result));
        }

        assert!(
            held.iter().any(|(_, result)| {
                result
                    .as_ref()
                    .is_err_and(|error| error.issue == "native_host_queue_saturated")
            }),
            "one request should saturate the bounded queue"
        );
    }

    #[test]
    fn cancelled_job_sends_cancel_terminal_without_content() {
        let host = NativeHost::with_limits(1, 2);
        let (mut sermo, responses, cancellation) = frame::test_response_sender("solum:lege");
        cancellation.cancel();

        host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "solum:lege".to_owned(),
                opener: Valor::Textus("unused".to_owned()),
                target: Some("alloc::string::String"),
            },
            responses,
            cancellation,
        )
        .expect("dispatch");

        let terminal = frame::sermo_recv(&mut sermo).expect("terminal");
        assert_eq!(terminal.status, FrameStatus::Cancel);
    }

    #[test]
    fn cancellation_during_timer_suppresses_late_done() {
        let host = NativeHost::with_limits(1, 2);
        let (mut sermo, responses, cancellation) = frame::test_response_sender("tempus:dormiet");
        frame::sermo_set_opener(&mut sermo, Valor::Numerus(40));

        host.start(
            SermoRequest {
                conversation_id: sermo.conversation_id(),
                route: "tempus:dormiet".to_owned(),
                opener: Valor::Numerus(40),
                target: None,
            },
            responses,
            cancellation.clone(),
        )
        .expect("dispatch");
        cancellation.cancel();

        let terminal = frame::sermo_recv(&mut sermo).expect("terminal");
        assert_eq!(terminal.status, FrameStatus::Cancel);
    }
}
