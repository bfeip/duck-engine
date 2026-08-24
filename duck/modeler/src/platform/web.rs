//! Emscripten platform: the modeler on a canvas, driven by
//! `emscripten_set_main_loop`.

mod egui_input;

use std::cell::RefCell;
use std::ffi::c_int;

use duck_engine_viewer::emscripten_support;
use duck_engine_viewer::WindowSurface;

use crate::ViewerState;
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
        let surface = WindowSurface::from_canvas_selector(CANVAS, width, height).await;

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
        // HTML5 API does not wrap; a `--js-library` shim is the way in.
    }

    pub(crate) fn request_redraw(&self) {
        // The main loop already runs on `requestAnimationFrame`.
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

            if let duck_engine_viewer::event::Event::Device(
                duck_engine_viewer::event::DeviceEvent::Resized((width, height)),
            ) = &event
            {
                state.host.size = (*width, *height);
                state.resize_surface(*width, *height);
            }
            if let duck_engine_viewer::event::Event::Device(
                duck_engine_viewer::event::DeviceEvent::CursorMoved { position },
            ) = &event
            {
                state.set_cursor(position.0 as f32, position.1 as f32);
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
