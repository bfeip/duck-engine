//! Native platform: a winit window driven by `egui_winit`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::event::Event;
use duck_engine_viewer::winit_support;
use duck_engine_viewer::{GpuOptions, WindowSurface};

use super::FileFilter;
use crate::ViewerState;

/// The winit window plus the egui integration that feeds off it.
pub(crate) struct Host {
    // Drops before `window`: the clipboard is built from the window's raw
    // display handle, and on Wayland runs a worker thread on that connection.
    egui_winit: egui_winit::State,
    window: Arc<Window>,
    /// The file picked in `open_file`, until `receive_opened_file` takes it.
    opened: Option<PathBuf>,
}

impl Host {
    /// Create the window and its GPU surface, and wire up egui's winit
    /// integration against them.
    pub(crate) async fn new(
        event_loop: &ActiveEventLoop,
        egui_ctx: egui::Context,
    ) -> (Self, WindowSurface<'static>) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Modeler")
                        .with_inner_size(winit::dpi::LogicalSize::new(1200, 1000)),
                )
                .expect("Failed to create window"),
        );

        let size = window.inner_size();
        let surface = WindowSurface::new(Arc::clone(&window), size.width, size.height, GpuOptions::default())
            .await;

        let egui_winit = egui_winit::State::new(
            egui_ctx,
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        (Self { egui_winit, window, opened: None }, surface)
    }

    /// Surface size in physical pixels.
    pub(crate) fn surface_size(&self) -> (u32, u32) {
        let size = self.window.inner_size();
        (size.width, size.height)
    }

    pub(crate) fn take_egui_input(&mut self) -> egui::RawInput {
        self.egui_winit.take_egui_input(&self.window)
    }

    pub(crate) fn handle_platform_output(&mut self, output: egui::PlatformOutput) {
        self.egui_winit.handle_platform_output(&self.window, output);
    }

    pub(crate) fn request_redraw(&self) {
        self.window.request_redraw();
    }

    /// Ask the user for a file to open. The dialog blocks, so the file is
    /// ready for [`Self::receive_opened_file`] in the same frame.
    pub(crate) fn open_file(&mut self, filter: &FileFilter) {
        self.opened = rfd::FileDialog::new().add_filter(filter.name, filter.extensions).pick_file();
    }

    /// Hand the file opened since the last call, if any, to `read`.
    pub(crate) fn receive_opened_file(&mut self, read: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
        self.opened.take().map_or(Ok(()), |path| read(&path))
    }

    /// Ask the user where to save `name` and write it there with `write`. A
    /// path without one of `filters`' extensions gets the first filter's.
    pub(crate) fn save_file(
        &self,
        name: &str,
        filters: &[FileFilter],
        write: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let dialog = filters.iter().fold(rfd::FileDialog::new().set_file_name(name), |dialog, f| {
            dialog.add_filter(f.name, f.extensions)
        });
        let Some(mut path) = dialog.save_file() else {
            return Ok(());
        };
        let known = path.extension().and_then(|e| e.to_str()).is_some_and(|ext| {
            filters.iter().flat_map(|f| f.extensions).any(|e| e.eq_ignore_ascii_case(ext))
        });
        if !known && let Some(ext) = filters.first().and_then(|f| f.extensions.first()) {
            path.set_extension(ext);
        }
        write(&path)
    }

    /// Let egui see a window event, then normalize it for the viewer.
    fn on_window_event(&mut self, event: &WindowEvent) -> Option<Event> {
        let _ = self.egui_winit.on_window_event(&self.window, event);
        winit_support::convert_window_event(event.clone())
    }
}

struct App {
    state: Option<ViewerState<'static>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let mut state = pollster::block_on(async {
                let egui_ctx = egui::Context::default();
                let (host, surface) = Host::new(event_loop, egui_ctx.clone()).await;
                ViewerState::new(egui_ctx, surface, host)
            });
            state.set_default_scene();
            state.host.request_redraw();
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    if state.handle_redraw() {
                        event_loop.exit();
                    }
                }
            }
            _ => {
                let Some(state) = self.state.as_mut() else { return };

                // The surface tracks the whole window; the offscreen viewer is
                // sized from the central panel each frame, not from the window.
                if let WindowEvent::Resized(size) = &event {
                    state.resize_surface(size.width, size.height);
                }
                if let WindowEvent::CursorMoved { position, .. } = &event {
                    state.set_cursor(position.x as f32, position.y as f32);
                }

                if let Some(event) = state.host.on_window_event(&event) {
                    state.route_input(event);
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let Some(state) = self.state.as_mut() else { return };
        if state.egui_wants_pointer() {
            // Do not respond to device events that egui is also consuming.
            return;
        }
        if let Some(event) = winit_support::convert_device_event(event) {
            state.viewer_handle_event(&event);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        // Release the window, surface and egui state while the event loop is still alive.
        // `run_app` consumes the loop, so anything left here drops after it.
        self.state = None;
    }
}

pub(crate) fn run() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let mut app = App { state: None };
    event_loop.run_app(&mut app).unwrap();
}
