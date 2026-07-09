//! In-process frame conversation types for expression `ad` and directional views.

use crate::{Instans, InstansPraecisio, Valor};
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::rc::Rc;
use std::thread;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

/// Frame lifecycle status for stream `ad` conversations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameStatus {
    Request,
    Item,
    Byte,
    Bulk,
    Done,
    Error,
    Cancel,
}

impl FrameStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Done | Self::Error | Self::Cancel)
    }

    pub fn is_content(self) -> bool {
        matches!(self, Self::Item | Self::Byte | Self::Bulk)
    }
}

/// Opaque frame record carried on a `Sermo` handle.
#[derive(Clone, Debug, PartialEq)]
pub struct Scrinium {
    pub id: String,
    pub parent_id: Option<String>,
    pub call: String,
    pub status: FrameStatus,
    pub data: Valor,
    pub created_ms: i64,
    pub from: Option<String>,
    pub trace: Option<Valor>,
}

#[derive(Debug)]
struct SermoInner {
    conversation_id: String,
    route: String,
    outgoing: Vec<Scrinium>,
    incoming: VecDeque<Scrinium>,
    runtime_response_generated: bool,
    incoming_drained: bool,
    /// Terminal `status` observed on the inbound direction (`done`, `error`, or `cancel`).
    incoming_terminal: Option<FrameStatus>,
    detached: bool,
    meus_closed: bool,
}

/// In-flight `ad` conversation handle.
#[derive(Clone, Debug)]
pub struct Sermo {
    inner: Rc<RefCell<SermoInner>>,
}

/// Caller-to-gateway live outbound half-stream view.
pub struct Meus<T> {
    inner: Rc<RefCell<SermoInner>>,
    _marker: PhantomData<T>,
}

/// Gateway-to-caller live inbound half-stream view.
pub struct Tuus<T> {
    inner: Rc<RefCell<SermoInner>>,
    _marker: PhantomData<T>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrameError {
    pub issue: &'static str,
    pub message: String,
}

impl FrameError {
    fn new(issue: &'static str, message: impl Into<String>) -> Self {
        Self {
            issue,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for FrameError {}

impl<T> std::fmt::Debug for Meus<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Meus")
            .field("conversation_id", &self.inner.borrow().conversation_id)
            .finish_non_exhaustive()
    }
}

impl<T> std::fmt::Debug for Tuus<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tuus")
            .field("conversation_id", &self.inner.borrow().conversation_id)
            .finish_non_exhaustive()
    }
}

/// Generated Rust status enums implement this trait instead of emitting shim fns.
pub trait IntoFrameStatus {
    fn into_frame_status(self) -> FrameStatus;
}

impl IntoFrameStatus for FrameStatus {
    fn into_frame_status(self) -> FrameStatus {
        self
    }
}

/// Generated Rust `scrinium` structs implement this trait instead of emitting shim fns.
pub trait IntoScrinium {
    fn into_scrinium(self) -> Scrinium;
}

impl IntoScrinium for Scrinium {
    fn into_scrinium(self) -> Scrinium {
        self
    }
}

pub fn frame_status_from_user<T: IntoFrameStatus>(value: T) -> FrameStatus {
    value.into_frame_status()
}

pub fn scrinium_from_user<T: IntoScrinium>(frame: T) -> Scrinium {
    frame.into_scrinium()
}

pub fn next_frame_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("frame-{}", NEXT.fetch_add(1, Ordering::Relaxed))
}

impl Sermo {
    pub fn conversation_id(&self) -> String {
        self.inner.borrow().conversation_id.clone()
    }

    pub fn route(&self) -> String {
        self.inner.borrow().route.clone()
    }

    pub fn incoming_drained(&self) -> bool {
        self.inner.borrow().incoming_drained
    }

    pub fn push_incoming(&mut self, frame: Scrinium) {
        let mut inner = self.inner.borrow_mut();
        inner.runtime_response_generated = true;
        inner.incoming.push_back(frame);
    }

    pub fn first_outgoing(&self) -> Option<Scrinium> {
        self.inner.borrow().outgoing.first().cloned()
    }
}

pub fn sermo_set_opener(sermo: &mut Sermo, data: Valor) {
    if let Some(request) = sermo.inner.borrow_mut().outgoing.first_mut() {
        if request.status == FrameStatus::Request {
            request.data = data;
        }
    }
}

pub fn sermo_open(route: &str) -> Sermo {
    let conversation_id = next_frame_id();
    Sermo {
        inner: Rc::new(RefCell::new(SermoInner {
            conversation_id: conversation_id.clone(),
            route: route.to_owned(),
            outgoing: vec![Scrinium {
                id: conversation_id,
                parent_id: None,
                call: route.to_owned(),
                status: FrameStatus::Request,
                data: Valor::Nihil,
                created_ms: now_millis(),
                from: None,
                trace: None,
            }],
            incoming: VecDeque::new(),
            runtime_response_generated: false,
            incoming_drained: false,
            incoming_terminal: None,
            detached: false,
            meus_closed: false,
        })),
    }
}

pub fn sermo_meus<T>(sermo: &Sermo) -> Meus<T> {
    Meus {
        inner: sermo.inner.clone(),
        _marker: PhantomData,
    }
}

pub fn sermo_tuus<T>(sermo: &Sermo) -> Tuus<T> {
    Tuus {
        inner: sermo.inner.clone(),
        _marker: PhantomData,
    }
}

pub fn meus_da<T>(meus: &Meus<T>, data: Valor) -> Result<(), FrameError> {
    let mut inner = meus.inner.borrow_mut();
    if inner.meus_closed {
        return Err(FrameError::new(
            "frame_meus_half_stream_closed",
            "meus half-stream is closed",
        ));
    }
    let conversation_id = inner.conversation_id.clone();
    let route = inner.route.clone();
    inner.outgoing.push(Scrinium {
        id: next_frame_id(),
        parent_id: Some(conversation_id),
        call: route,
        status: FrameStatus::Item,
        data,
        created_ms: now_millis(),
        from: None,
        trace: None,
    });
    Ok(())
}

pub fn meus_fini<T>(meus: &Meus<T>) -> FrameStatus {
    let mut inner = meus.inner.borrow_mut();
    if !inner.meus_closed {
        let conversation_id = inner.conversation_id.clone();
        let route = inner.route.clone();
        inner.outgoing.push(Scrinium {
            id: next_frame_id(),
            parent_id: Some(conversation_id),
            call: route,
            status: FrameStatus::Done,
            data: Valor::Nihil,
            created_ms: now_millis(),
            from: None,
            trace: None,
        });
        inner.meus_closed = true;
    }
    FrameStatus::Done
}

pub fn tuus_accipe<T>(tuus: &Tuus<T>) -> Option<Scrinium> {
    let mut inner = tuus.inner.borrow_mut();
    recv_content_frame(&mut inner)
}

/// Lazy inbound content-frame iterator; shares the queue with `tuus_accipe`.
pub struct TuusCursor<T> {
    inner: Rc<RefCell<SermoInner>>,
    _marker: PhantomData<T>,
}

impl<T> Iterator for TuusCursor<T> {
    type Item = Scrinium;

    fn next(&mut self) -> Option<Scrinium> {
        recv_content_frame(&mut self.inner.borrow_mut())
    }
}

pub fn tuus_cursor<T>(tuus: &Tuus<T>) -> TuusCursor<T> {
    TuusCursor {
        inner: tuus.inner.clone(),
        _marker: PhantomData,
    }
}

pub fn tuus_fini<T>(tuus: &Tuus<T>) -> FrameStatus {
    let mut inner = tuus.inner.borrow_mut();
    if inner.incoming_drained {
        return inner.incoming_terminal.unwrap_or(FrameStatus::Done);
    }
    ensure_runtime_response_inner(&mut inner);
    while let Some(frame) = inner.incoming.pop_front() {
        if frame.status.is_terminal() {
            record_incoming_terminal(&mut inner, frame.status);
            return frame.status;
        }
    }
    record_incoming_terminal(&mut inner, FrameStatus::Done);
    FrameStatus::Done
}

pub fn tuus_as_sermo<T>(tuus: &Tuus<T>) -> Sermo {
    Sermo {
        inner: tuus.inner.clone(),
    }
}

fn record_incoming_terminal(inner: &mut SermoInner, status: FrameStatus) {
    inner.incoming_terminal = Some(status);
    inner.incoming_drained = true;
}

fn recv_content_frame(inner: &mut SermoInner) -> Option<Scrinium> {
    if inner.detached || inner.incoming_drained {
        return None;
    }
    ensure_runtime_response_inner(inner);
    let frame = inner.incoming.pop_front()?;
    if frame.status.is_terminal() {
        record_incoming_terminal(inner, frame.status);
        return None;
    }
    Some(frame)
}

fn drain_incoming_to_terminal(sermo: &mut Sermo) {
    while let Some(frame) = sermo_recv(sermo) {
        if frame.status.is_terminal() {
            break;
        }
    }
    let mut inner = sermo.inner.borrow_mut();
    if !inner.incoming_drained {
        record_incoming_terminal(&mut inner, FrameStatus::Done);
    }
}

/// Drain inbound content frames into a raw frame list for internal materialization.
pub fn sermo_tuus_frames(mut sermo: Sermo) -> Vec<Scrinium> {
    let mut frames = Vec::new();
    while let Some(frame) = sermo_recv(&mut sermo) {
        if frame.status.is_terminal() {
            break;
        }
        frames.push(frame);
    }
    let mut inner = sermo.inner.borrow_mut();
    if inner.incoming_terminal.is_none() {
        record_incoming_terminal(&mut inner, FrameStatus::Done);
    }
    frames
}

pub fn sermo_recv(sermo: &mut Sermo) -> Option<Scrinium> {
    let mut inner = sermo.inner.borrow_mut();
    if inner.detached {
        return None;
    }
    ensure_runtime_response_inner(&mut inner);
    let frame = inner.incoming.pop_front()?;
    if frame.status.is_terminal() {
        record_incoming_terminal(&mut inner, frame.status);
    }
    Some(frame)
}

fn ensure_runtime_response_inner(inner: &mut SermoInner) {
    if inner.runtime_response_generated {
        return;
    }

    match inner.route.as_str() {
        "runtime:echo" => {
            let data = inner
                .outgoing
                .first()
                .map(|request| request.data.clone())
                .unwrap_or(Valor::Nihil);
            push_runtime_item_done(inner, data);
        }
        "tempus:nunc" => {
            let now = epoch_nanos();
            let instans = Instans::from_nanos(now, InstansPraecisio::Nanosecunda);
            push_runtime_item_done(inner, Valor::Instans(instans.to_rfc3339()));
        }
        "tempus:monotonicum" => {
            push_runtime_item_done(inner, Valor::Numerus(monotonic_nanos()));
        }
        "tempus:activum" => {
            push_runtime_item_done(inner, Valor::Numerus(process_active_millis()));
        }
        "tempus:dormiet" | "tempus:expectet" => {
            let ms = request_numerus(inner).unwrap_or(0).max(0);
            thread::sleep(Duration::from_millis(ms as u64));
            push_runtime_done(inner);
        }
        "solum:scribe" | "solum:scribet" | "solum:appone" | "solum:apponet" => {
            dispatch_solum_write_text(inner);
        }
        "solum:dele" | "solum:delet" => {
            dispatch_solum_delete(inner);
        }
        "solum:parens" => {
            dispatch_solum_parens(inner);
        }
        "solum:partem" => {
            try_generate_solum_partem_response::<Vec<u8>>(inner);
        }
        "processus:exsequi" | "processus:exsequetur" => {
            dispatch_processus_exsequi(inner);
        }
        "processus:captura" => {
            dispatch_processus_captura(inner);
        }
        _ => {}
    }
}

fn push_runtime_item_done(inner: &mut SermoInner, data: Valor) {
    inner.runtime_response_generated = true;
    push_runtime_frame(inner, FrameStatus::Item, data);
    push_runtime_frame(inner, FrameStatus::Done, Valor::Nihil);
}

fn push_runtime_bytes_done(inner: &mut SermoInner, bytes: Vec<u8>) {
    inner.runtime_response_generated = true;
    push_runtime_frame(inner, FrameStatus::Byte, Valor::Octeti(bytes));
    push_runtime_frame(inner, FrameStatus::Done, Valor::Nihil);
}

fn push_runtime_done(inner: &mut SermoInner) {
    inner.runtime_response_generated = true;
    push_runtime_frame(inner, FrameStatus::Done, Valor::Nihil);
}

fn push_runtime_error(inner: &mut SermoInner, message: impl Into<String>) {
    inner.runtime_response_generated = true;
    push_runtime_frame(inner, FrameStatus::Error, Valor::Textus(message.into()));
}

fn push_runtime_frame(inner: &mut SermoInner, status: FrameStatus, data: Valor) {
    inner.incoming.push_back(Scrinium {
        id: next_frame_id(),
        parent_id: Some(inner.conversation_id.clone()),
        call: inner.route.clone(),
        status,
        data,
        created_ms: now_millis(),
        from: Some("faber-runtime".into()),
        trace: None,
    });
}

fn request_data(inner: &SermoInner) -> Valor {
    inner
        .outgoing
        .first()
        .map(|request| request.data.clone())
        .unwrap_or(Valor::Nihil)
}

fn request_text(inner: &SermoInner) -> Option<String> {
    match request_data(inner) {
        Valor::Textus(text) => Some(text),
        _ => None,
    }
}

fn request_text_list(inner: &SermoInner) -> Option<Vec<String>> {
    let Valor::Lista(items) = request_data(inner) else {
        return None;
    };
    items
        .into_iter()
        .map(|item| match item {
            Valor::Textus(value) => Some(value),
            _ => None,
        })
        .collect()
}

fn request_numerus(inner: &SermoInner) -> Option<i64> {
    match request_data(inner) {
        Valor::Numerus(n) => Some(n),
        _ => None,
    }
}

fn request_text_pair(inner: &SermoInner) -> Result<(String, String), String> {
    let Valor::Lista(items) = request_data(inner) else {
        return Err("route opener must be [textus, textus]".to_owned());
    };
    let mut iter = items.into_iter();
    let (Some(Valor::Textus(path)), Some(Valor::Textus(data)), None) =
        (iter.next(), iter.next(), iter.next())
    else {
        return Err("route opener must be [textus, textus]".to_owned());
    };
    Ok((path, data))
}

fn request_text_range(inner: &SermoInner) -> Result<(String, i64, i64), String> {
    let Valor::Lista(items) = request_data(inner) else {
        return Err("route opener must be [textus, numerus, numerus]".to_owned());
    };
    let mut iter = items.into_iter();
    let (
        Some(Valor::Textus(path)),
        Some(Valor::Numerus(start)),
        Some(Valor::Numerus(length)),
        None,
    ) = (iter.next(), iter.next(), iter.next(), iter.next())
    else {
        return Err("route opener must be [textus, numerus, numerus]".to_owned());
    };
    Ok((path, start, length))
}

fn request_text_pattern_range(inner: &SermoInner) -> Result<(String, String, i64, i64), String> {
    let Valor::Lista(items) = request_data(inner) else {
        return Err("route opener must be [textus, textus, numerus, numerus]".to_owned());
    };
    let mut iter = items.into_iter();
    let (
        Some(Valor::Textus(path)),
        Some(Valor::Textus(pattern)),
        Some(Valor::Numerus(start)),
        Some(Valor::Numerus(length)),
        None,
    ) = (
        iter.next(),
        iter.next(),
        iter.next(),
        iter.next(),
        iter.next(),
    )
    else {
        return Err("route opener must be [textus, textus, numerus, numerus]".to_owned());
    };
    Ok((path, pattern, start, length))
}

fn dispatch_solum_write_text(inner: &mut SermoInner) {
    let route = inner.route.clone();
    let result = request_text_pair(inner).and_then(|(path, data)| {
        if matches!(route.as_str(), "solum:appone" | "solum:apponet") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|err| format!("failed to open file for append: {err}"))?;
            file.write_all(data.as_bytes())
                .map_err(|err| format!("failed to append to file: {err}"))
        } else {
            std::fs::write(&path, data).map_err(|err| format!("failed to write file: {err}"))
        }
    });

    match result {
        Ok(()) => push_runtime_done(inner),
        Err(message) => push_runtime_error(inner, message),
    }
}

fn dispatch_solum_parens(inner: &mut SermoInner) {
    let Some(path) = request_text(inner) else {
        push_runtime_error(inner, "solum:parens opener must be textus");
        return;
    };
    let parent = std::path::Path::new(&path)
        .parent()
        .map(|parent| parent.to_string_lossy().into_owned())
        .unwrap_or_default();
    push_runtime_item_done(inner, Valor::Textus(parent));
}

fn dispatch_solum_delete(inner: &mut SermoInner) {
    let Some(path) = request_text(inner) else {
        push_runtime_error(inner, "solum:dele opener must be textus");
        return;
    };
    match std::fs::remove_file(&path) {
        Ok(()) => push_runtime_done(inner),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => push_runtime_done(inner),
        Err(err) => push_runtime_error(inner, format!("solum.dele failed for {path}: {err}")),
    }
}

fn dispatch_processus_exsequi(inner: &mut SermoInner) {
    let Some(command) = request_text(inner) else {
        push_runtime_error(inner, "processus:exsequi opener must be textus");
        return;
    };
    let result = std::process::Command::new("sh")
        .arg("-c")
        .arg(&command)
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .map_err(|err| format!("processus.exsequi failed: {err}"));

    match result {
        Ok(stdout) => push_runtime_item_done(inner, Valor::Textus(stdout)),
        Err(message) => push_runtime_error(inner, message),
    }
}

fn dispatch_processus_captura(inner: &mut SermoInner) {
    let Some(args) = request_text_list(inner) else {
        push_runtime_error(inner, "processus:captura opener must be lista<textus>");
        return;
    };
    let Some((program, program_args)) = args.split_first() else {
        push_runtime_error(inner, "processus:captura requires a non-empty args list");
        return;
    };
    let result = std::process::Command::new(program)
        .args(program_args)
        .output()
        .map_err(|err| format!("processus.captura failed: {err}"));

    match result {
        Ok(output) => {
            let mut fields = BTreeMap::new();
            fields.insert(
                "status".to_owned(),
                Valor::Numerus(output.status.code().unwrap_or(-1) as i64),
            );
            fields.insert(
                "stdout".to_owned(),
                Valor::Textus(String::from_utf8_lossy(&output.stdout).into_owned()),
            );
            fields.insert(
                "stderr".to_owned(),
                Valor::Textus(String::from_utf8_lossy(&output.stderr).into_owned()),
            );
            push_runtime_item_done(inner, Valor::Tabula(fields));
        }
        Err(message) => push_runtime_error(inner, message),
    }
}

fn ensure_scalar_runtime_response<T>(sermo: &mut Sermo)
where
    T: crate::FromValor,
{
    let mut inner = sermo.inner.borrow_mut();
    if inner.runtime_response_generated {
        return;
    }
    if try_generate_solum_lege_response::<T>(&mut inner) {
        return;
    }
    if try_generate_solum_partem_response::<T>(&mut inner) {
        return;
    }
    if try_generate_solum_mensura_response::<T>(&mut inner) {
        return;
    }
    if try_generate_solum_inveni_response::<T>(&mut inner) {
        return;
    }
    if try_generate_solum_path_bool_response::<T>(&mut inner) {
        return;
    }
    ensure_runtime_response_inner(&mut inner);
}

fn try_generate_solum_lege_response<T>(inner: &mut SermoInner) -> bool
where
    T: crate::FromValor,
{
    if inner.route != "solum:lege" {
        return false;
    }

    let Some(path) = request_text(inner) else {
        push_runtime_error(inner, "solum:lege opener must be textus");
        return true;
    };

    let target = std::any::type_name::<T>();
    if target == std::any::type_name::<String>() {
        match std::fs::read_to_string(&path) {
            Ok(text) => push_runtime_item_done(inner, Valor::Textus(text)),
            Err(err) => push_runtime_error(inner, format!("failed to read file: {err}")),
        }
        return true;
    }

    if target == std::any::type_name::<Vec<String>>() {
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                let lines = text
                    .lines()
                    .map(|line| Valor::Textus(line.to_owned()))
                    .collect();
                push_runtime_item_done(inner, Valor::Lista(lines));
            }
            Err(err) => push_runtime_error(inner, format!("failed to read file: {err}")),
        }
        return true;
    }

    if target == std::any::type_name::<Vec<u8>>() {
        match std::fs::read(&path) {
            Ok(bytes) => {
                let values = bytes
                    .into_iter()
                    .map(|byte| Valor::Numerus(i64::from(byte)))
                    .collect();
                push_runtime_item_done(inner, Valor::Lista(values));
            }
            Err(err) => push_runtime_error(inner, format!("failed to read file: {err}")),
        }
        return true;
    }

    push_runtime_error(
        inner,
        format!("solum:lege target `{target}` is not supported"),
    );
    true
}

fn try_generate_solum_partem_response<T>(inner: &mut SermoInner) -> bool
where
    T: crate::FromValor,
{
    if inner.route != "solum:partem" {
        return false;
    }

    let Ok((path, start, length)) = request_text_range(inner) else {
        push_runtime_error(
            inner,
            "solum:partem opener must be [textus, numerus, numerus]",
        );
        return true;
    };

    let target = std::any::type_name::<T>();
    if target != std::any::type_name::<Vec<u8>>() {
        push_runtime_error(
            inner,
            format!("solum:partem target `{target}` is not supported"),
        );
        return true;
    }

    let Ok(start) = u64::try_from(start) else {
        push_runtime_error(inner, "solum:partem start must be non-negative");
        return true;
    };
    let Ok(length) = usize::try_from(length) else {
        push_runtime_error(inner, "solum:partem length must be non-negative");
        return true;
    };

    match std::fs::File::open(&path) {
        Ok(mut file) => {
            if let Err(err) = file.seek(SeekFrom::Start(start)) {
                push_runtime_error(inner, format!("failed to seek file: {err}"));
                return true;
            }
            let mut bytes = vec![0_u8; length];
            match file.read(&mut bytes) {
                Ok(count) => {
                    bytes.truncate(count);
                    push_runtime_bytes_done(inner, bytes);
                }
                Err(err) => push_runtime_error(inner, format!("failed to read file: {err}")),
            }
        }
        Err(err) => push_runtime_error(inner, format!("failed to open file: {err}")),
    }
    true
}

fn try_generate_solum_mensura_response<T>(inner: &mut SermoInner) -> bool
where
    T: crate::FromValor,
{
    if inner.route != "solum:mensura" {
        return false;
    }

    let Some(path) = request_text(inner) else {
        push_runtime_error(inner, "solum:mensura opener must be textus");
        return true;
    };

    let target = std::any::type_name::<T>();
    if target != std::any::type_name::<i64>() {
        push_runtime_error(
            inner,
            format!("solum:mensura target `{target}` is not supported"),
        );
        return true;
    }

    match std::fs::metadata(path) {
        Ok(metadata) => match i64::try_from(metadata.len()) {
            Ok(size) => push_runtime_item_done(inner, Valor::Numerus(size)),
            Err(_) => push_runtime_error(inner, "file size exceeds numerus range"),
        },
        Err(err) => push_runtime_error(inner, format!("failed to read file metadata: {err}")),
    }
    true
}

fn try_generate_solum_inveni_response<T>(inner: &mut SermoInner) -> bool
where
    T: crate::FromValor,
{
    if inner.route != "solum:inveni" {
        return false;
    }

    let Ok((path, pattern, start, length)) = request_text_pattern_range(inner) else {
        push_runtime_error(
            inner,
            "solum:inveni opener must be [textus, textus, numerus, numerus]",
        );
        return true;
    };

    let target = std::any::type_name::<T>();
    if target != std::any::type_name::<i64>() {
        push_runtime_error(
            inner,
            format!("solum:inveni target `{target}` is not supported"),
        );
        return true;
    }
    let Ok(start) = u64::try_from(start) else {
        push_runtime_error(inner, "solum:inveni start must be non-negative");
        return true;
    };
    let Ok(length) = usize::try_from(length) else {
        push_runtime_error(inner, "solum:inveni length must be non-negative");
        return true;
    };
    let pattern = pattern.into_bytes();
    if pattern.is_empty() {
        push_runtime_item_done(inner, Valor::Numerus(-1));
        return true;
    }

    match std::fs::File::open(&path) {
        Ok(mut file) => {
            if let Err(err) = file.seek(SeekFrom::Start(start)) {
                push_runtime_error(inner, format!("failed to seek file: {err}"));
                return true;
            }
            let mut bytes = vec![0_u8; length];
            match file.read(&mut bytes) {
                Ok(count) => {
                    bytes.truncate(count);
                    let found = bytes
                        .windows(pattern.len())
                        .position(|window| window == pattern.as_slice())
                        .and_then(|offset| i64::try_from(start.saturating_add(offset as u64)).ok())
                        .unwrap_or(-1);
                    push_runtime_item_done(inner, Valor::Numerus(found));
                }
                Err(err) => push_runtime_error(inner, format!("failed to read file: {err}")),
            }
        }
        Err(err) => push_runtime_error(inner, format!("failed to open file: {err}")),
    }
    true
}

fn try_generate_solum_path_bool_response<T>(inner: &mut SermoInner) -> bool
where
    T: crate::FromValor,
{
    if !matches!(
        inner.route.as_str(),
        "solum:exstat" | "solum:directoriumne" | "solum:regularene" | "solum:legibilene"
    ) {
        return false;
    }

    let Some(path) = request_text(inner) else {
        push_runtime_error(inner, format!("{} opener must be textus", inner.route));
        return true;
    };

    let target = std::any::type_name::<T>();
    if target == std::any::type_name::<bool>() {
        let path = std::path::Path::new(&path);
        let result = match inner.route.as_str() {
            "solum:exstat" => path.exists(),
            "solum:directoriumne" => path.is_dir(),
            "solum:regularene" => path.is_file(),
            "solum:legibilene" => path.is_file() && std::fs::File::open(path).is_ok(),
            _ => false,
        };
        push_runtime_item_done(inner, Valor::Bivalens(result));
        return true;
    }

    push_runtime_error(
        inner,
        format!("{} target `{target}` is not supported", inner.route),
    );
    true
}

fn epoch_nanos() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_nanos().min(i64::MAX as u128) as i64
        })
}

fn monotonic_nanos() -> i64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_nanos().min(i64::MAX as u128) as i64
}

fn process_active_millis() -> i64 {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_millis().min(i64::MAX as u128) as i64
}

pub fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

// ---- `sermo ↦ T` materializers --------------------------------------------

fn terminal_error(frame: &Scrinium) -> Option<FrameError> {
    match frame.status {
        FrameStatus::Error => Some(FrameError::new(
            "frame_materialization_terminal_error",
            format!("sermo materialization terminal error: {:?}", frame.data),
        )),
        FrameStatus::Cancel => Some(FrameError::new(
            "frame_materialization_cancelled",
            "sermo materialization cancelled",
        )),
        _ => None,
    }
}

fn drain_remaining_then_err<T>(sermo: &mut Sermo, error: FrameError) -> Result<T, FrameError> {
    drain_incoming_to_terminal(sermo);
    Err(error)
}

pub fn sermo_materialize_vacuum(sermo: &mut Sermo) {
    try_sermo_materialize_vacuum(sermo).expect("sermo ↦ vacuum materialization failed");
}

pub fn try_sermo_materialize_vacuum(sermo: &mut Sermo) -> Result<(), FrameError> {
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
    }
    Ok(())
}

pub fn sermo_materialize_textus(sermo: &mut Sermo) -> String {
    try_sermo_materialize_textus(sermo).expect("sermo ↦ textus materialization failed")
}

pub fn try_sermo_materialize_textus(sermo: &mut Sermo) -> Result<String, FrameError> {
    let mut out = String::new();
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        let Valor::Textus(s) = &frame.data else {
            return drain_remaining_then_err(
                sermo,
                FrameError::new(
                    "frame_textus_payload_not_textus",
                    "sermo ↦ textus: content frame payload was not textus",
                ),
            );
        };
        out.push_str(s);
    }
    Ok(out)
}

pub fn sermo_materialize_octeti(sermo: &mut Sermo) -> Vec<u8> {
    try_sermo_materialize_octeti(sermo).expect("sermo ↦ octeti materialization failed")
}

pub fn try_sermo_materialize_octeti(sermo: &mut Sermo) -> Result<Vec<u8>, FrameError> {
    {
        let mut inner = sermo.inner.borrow_mut();
        if !inner.runtime_response_generated
            && !try_generate_solum_lege_response::<Vec<u8>>(&mut inner)
        {
            try_generate_solum_partem_response::<Vec<u8>>(&mut inner);
        }
    }
    let mut out = Vec::new();
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        match &frame.data {
            Valor::Octeti(bytes) => out.extend_from_slice(bytes),
            Valor::Lista(bytes) => {
                for v in bytes {
                    let Valor::Numerus(n) = v else {
                        return drain_remaining_then_err(
                            sermo,
                            FrameError::new(
                                "frame_octeti_byte_not_numerus",
                                "sermo ↦ octeti: byte payload contained a non-numerus value",
                            ),
                        );
                    };
                    let Ok(byte) = u8::try_from(*n) else {
                        return drain_remaining_then_err(
                            sermo,
                            FrameError::new(
                                "frame_octeti_byte_out_of_range",
                                "sermo ↦ octeti: byte payload value was outside 0..255",
                            ),
                        );
                    };
                    out.push(byte);
                }
            }
            _ => {
                return drain_remaining_then_err(
                    sermo,
                    FrameError::new(
                        "frame_octeti_payload_not_bytes",
                        "sermo ↦ octeti: content frame payload was not octeti or byte lista",
                    ),
                );
            }
        }
    }
    Ok(out)
}

pub fn sermo_materialize_valor(sermo: &mut Sermo) -> Valor {
    try_sermo_materialize_valor(sermo).expect("sermo ↦ valor materialization failed")
}

pub fn try_sermo_materialize_valor(sermo: &mut Sermo) -> Result<Valor, FrameError> {
    let mut captured: Option<Valor> = None;
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        if captured.is_none() {
            captured = Some(frame.data);
        }
    }
    Ok(captured.unwrap_or(Valor::Nihil))
}

pub fn sermo_materialize_lista<T>(sermo: &mut Sermo) -> Vec<T>
where
    T: crate::FromValor,
{
    try_sermo_materialize_lista(sermo).expect("sermo ↦ lista<T> materialization failed")
}

pub fn try_sermo_materialize_lista<T>(sermo: &mut Sermo) -> Result<Vec<T>, FrameError>
where
    T: crate::FromValor,
{
    let mut out = Vec::new();
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        let Some(v) = T::from_valor(&frame.data) else {
            return drain_remaining_then_err(
                sermo,
                FrameError::new(
                    "frame_lista_payload_element_type_mismatch",
                    "sermo ↦ lista<T>: content frame payload did not match element type",
                ),
            );
        };
        out.push(v);
    }
    Ok(out)
}

pub fn sermo_materialize_scalar<T>(sermo: &mut Sermo) -> T
where
    T: crate::FromValor,
{
    try_sermo_materialize_scalar(sermo).expect("sermo ↦ T scalar materialization failed")
}

pub fn sermo_materialize_instans(sermo: &mut Sermo, precision: InstansPraecisio) -> Instans {
    try_sermo_materialize_instans(sermo, precision).expect("sermo ↦ instans materialization failed")
}

pub fn try_sermo_materialize_instans(
    sermo: &mut Sermo,
    precision: InstansPraecisio,
) -> Result<Instans, FrameError> {
    {
        let mut inner = sermo.inner.borrow_mut();
        ensure_runtime_response_inner(&mut inner);
    }
    let mut extracted: Option<Instans> = None;
    let mut content_count = 0u32;
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        content_count += 1;
        if extracted.is_none() {
            extracted = Instans::try_from_valor(&frame.data, precision);
        }
    }
    if content_count == 0 {
        return Err(FrameError::new(
            "frame_instans_no_content_frame",
            "sermo ↦ instans: no content frame before terminal",
        ));
    }
    if content_count > 1 {
        return Err(FrameError::new(
            "frame_instans_multiple_content_frames",
            format!(
                "sermo ↦ instans: more than one content frame (found {})",
                content_count
            ),
        ));
    }
    extracted.ok_or_else(|| {
        FrameError::new(
            "frame_instans_payload_target_type_mismatch",
            "sermo ↦ instans: content frame payload did not match target type",
        )
    })
}

pub fn try_sermo_materialize_scalar<T>(sermo: &mut Sermo) -> Result<T, FrameError>
where
    T: crate::FromValor,
{
    ensure_scalar_runtime_response::<T>(sermo);
    let mut extracted: Option<T> = None;
    let mut content_count = 0u32;
    while let Some(frame) = sermo_recv(sermo) {
        if let Some(message) = terminal_error(&frame) {
            return Err(message);
        }
        if frame.status.is_terminal() {
            break;
        }
        content_count += 1;
        if extracted.is_none() {
            extracted = T::from_valor(&frame.data);
        }
    }
    if content_count == 0 {
        return Err(FrameError::new(
            "frame_scalar_no_content_frame",
            "sermo ↦ T scalar: no content frame before terminal",
        ));
    }
    if content_count > 1 {
        return Err(FrameError::new(
            "frame_scalar_multiple_content_frames",
            format!(
                "sermo ↦ T scalar: more than one content frame (found {})",
                content_count
            ),
        ));
    }
    extracted.ok_or_else(|| {
        FrameError::new(
            "frame_scalar_payload_target_type_mismatch",
            "sermo ↦ T scalar: content frame payload did not match target type",
        )
    })
}
