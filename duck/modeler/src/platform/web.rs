//! Emscripten platform: the modeler on a canvas, driven by
//! `emscripten_set_main_loop`.

mod egui_input;

use std::cell::RefCell;
use std::ffi::{CString, c_char, c_int};
use std::path::{Path, PathBuf};

use anyhow::Result;
use duck_engine_viewer::emscripten_support;
use duck_engine_viewer::event::{DeviceEvent, Event};
use duck_engine_viewer::{GpuOptions, WindowSurface};

use super::FileFilter;
use crate::{FilePurpose, ViewerState};
use egui_input::EguiInput;

/// CSS selector of the canvas the modeler renders into.
const CANVAS: &str = "#canvas";

unsafe extern "C" {
    /// `fps <= 0` means `requestAnimationFrame`; a non-zero
    /// `simulate_infinite_loop` never returns.
    fn emscripten_set_main_loop(
        func: extern "C" fn(),
        fps: c_int,
        simulate_infinite_loop: c_int,
    );
}

// `web/library_host.js`.
unsafe extern "C" {
    fn duck_pick_file(accept: *const c_char, purpose: c_int);
    fn duck_take_picked_file(buf: *mut c_char, cap: usize, purpose: *mut c_int) -> usize;
    fn duck_download_file(path: *const c_char);
}

/// The canvas and the egui input accumulator built from its DOM events.
pub(crate) struct Host {
    egui_input: EguiInput,
    /// Surface size in physical pixels, tracked from resize events.
    size: (u32, u32),
}

impl Host {
    /// Create the canvas surface and the egui integration over it.
    pub(crate) async fn new(_egui_ctx: egui::Context) -> (Self, WindowSurface<'static>) {
        // Size the canvas backing store to its CSS box before creating the
        // surface, so the first frame is not stretched.
        let (width, height) = emscripten_support::canvas_size().unwrap_or((1200, 1000));
        let surface =
            WindowSurface::from_canvas_selector(CANVAS, width, height, GpuOptions::default())
                .await;

        (Self { egui_input: EguiInput::new(), size: (width, height) }, surface)
    }

    pub(crate) fn surface_size(&self) -> (u32, u32) {
        self.size
    }

    pub(crate) fn take_egui_input(&mut self) -> egui::RawInput {
        self.egui_input.take(self.size)
    }

    pub(crate) fn handle_platform_output(&mut self, _output: egui::PlatformOutput) {
        // Cursor icon, clipboard, and IME all need DOM access that emscripten's
        // HTML5 API does not wrap; `web/library_host.js` is the way in.
    }

    pub(crate) fn request_redraw(&self) {
        // The main loop already runs on `requestAnimationFrame`.
    }

    /// Open the browser's file picker for `purpose`. The pick arrives some
    /// frames later, through [`Self::receive_opened_file`].
    ///
    /// Browsers open a picker only shortly after a user gesture. The click or
    /// key press that asked for this happened within the last frame, well
    /// inside that window.
    pub(crate) fn open_file(&mut self, filter: &FileFilter, purpose: FilePurpose) {
        let accept: Vec<String> = filter.extensions.iter().map(|e| format!(".{e}")).collect();
        let accept = CString::new(accept.join(",")).expect("extensions hold no NUL");
        unsafe { duck_pick_file(accept.as_ptr(), purpose_tag(purpose)) };
    }

    /// Hand the oldest file picked and not yet received, if any, to `read`
    /// with what it was opened for, then drop its in-memory copy.
    pub(crate) fn receive_opened_file<R>(
        &mut self,
        read: impl FnOnce(FilePurpose, &Path) -> R,
    ) -> Option<R> {
        let mut buf = [0u8; 4096];
        let mut tag = 0;
        let len = unsafe { duck_take_picked_file(buf.as_mut_ptr().cast(), buf.len(), &mut tag) };
        if len == 0 {
            return None;
        }
        let path = PathBuf::from(String::from_utf8_lossy(&buf[..len]).into_owned());
        let result = tagged_purpose(tag).map(|purpose| read(purpose, &path));
        let _ = std::fs::remove_file(&path);
        result
    }

    /// Write `name` with `write` and hand it to the browser as a download.
    /// There is no save dialog, so `filters` go unused and the file keeps
    /// `name`.
    pub(crate) fn save_file(
        &self,
        name: &str,
        _filters: &[FileFilter],
        write: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let path = format!("/tmp/{name}");
        write(Path::new(&path))?;
        let path = CString::new(path).expect("file name holds no NUL");
        unsafe { duck_download_file(path.as_ptr()) };
        Ok(())
    }
}

/// `purpose` as the tag the picker carries; [`tagged_purpose`] reads it back.
fn purpose_tag(purpose: FilePurpose) -> c_int {
    match purpose {
        FilePurpose::CadImport => 0,
        FilePurpose::Environment => 1,
    }
}

/// The purpose [`purpose_tag`] turned into `tag`.
fn tagged_purpose(tag: c_int) -> Option<FilePurpose> {
    match tag {
        0 => Some(FilePurpose::CadImport),
        1 => Some(FilePurpose::Environment),
        _ => None,
    }
}

thread_local! {
    static STATE: RefCell<Option<ViewerState<'static>>> = const { RefCell::new(None) };
}

extern "C" fn frame() {
    STATE.with(|state| {
        let mut state = state.borrow_mut();
        let Some(state) = state.as_mut() else { return };

        for event in emscripten_support::drain_events() {
            // egui sees every event first, exactly as `egui_winit` does on
            // native, so `route_input` can ask whether egui wants it.
            state.host.egui_input.on_event(&event);

            match &event {
                Event::Device(DeviceEvent::Resized((width, height))) => {
                    state.host.size = (*width, *height);
                    state.resize_surface(*width, *height);
                }
                Event::Device(DeviceEvent::CursorMoved { position }) => {
                    state.set_cursor(position.0 as f32, position.1 as f32);
                }
                // Relative motion drives drags, and viewport routing has no arm
                // for it. winit delivers it on a separate channel that skips
                // routing entirely, so do the same here or drags never start.
                Event::Device(DeviceEvent::MouseMotion { .. }) => {
                    if !state.egui_wants_pointer() {
                        state.viewer_handle_event(&event);
                    }
                    continue;
                }
                _ => {}
            }

            state.route_input(event);
        }

        state.handle_redraw();
    });
}

pub(crate) fn run() {
    // Emscripten routes stdout/stderr to the browser console.
    env_logger::builder().filter_level(log::LevelFilter::Info).init();

    emscripten_support::register_input(CANVAS);

    // Emscripten's EGL instance is static, so adapter and device requests
    // resolve without yielding to the event loop; `block_on` cannot deadlock
    // here the way it would under wasm-bindgen.
    let mut state = pollster::block_on(async {
        let egui_ctx = egui::Context::default();
        let (host, surface) = Host::new(egui_ctx.clone()).await;
        ViewerState::new(egui_ctx, surface, host)
    });
    state.set_default_scene();

    STATE.with(|cell| *cell.borrow_mut() = Some(state));

    log::info!("modeler initialised, entering main loop");
    unsafe { emscripten_set_main_loop(frame, 0, 0) };
}
