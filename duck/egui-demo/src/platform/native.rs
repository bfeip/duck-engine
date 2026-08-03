//! Native bootstrap: blocking wgpu init and the desktop entry point.

use std::path::PathBuf;
use std::sync::Arc;

use winit::dpi::LogicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::Window;

use crate::{App, UserEvent, ViewerState, ui};

pub(crate) fn run() {
    env_logger::init();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();

    let mut app = App {
        state: None,
        ui: ui::UiState::default(),
        workflow_index: 0,
        pending_hdr_path: None,
        pending_scene_load_path: None,
    };

    event_loop.run_app(&mut app).unwrap();
}

/// Create the window and viewer state synchronously, then queue the default
/// scene and environment from disk.
pub(crate) fn resume(app: &mut App, event_loop: &ActiveEventLoop) {
    if app.state.is_some() {
        return;
    }

    let window_attrs = Window::default_attributes()
        .with_title("Duck Engine - egui Example")
        .with_inner_size(LogicalSize::new(1600, 800));
    let window = Arc::new(
        event_loop
            .create_window(window_attrs)
            .expect("Failed to create window"),
    );

    let state = pollster::block_on(ViewerState::from_window(window, None));
    state.window.request_redraw();
    app.state = Some(state);

    let default_scene = PathBuf::from(default_scene_path!());
    if default_scene.exists() {
        app.pending_scene_load_path = Some(default_scene);
    } else {
        log::warn!("Default scene not found: {}", default_scene.display());
    }

    let default_env = PathBuf::from(default_environment_path!());
    if default_env.exists() {
        app.pending_hdr_path = Some(default_env);
    } else {
        log::warn!("Default environment not found: {}", default_env.display());
    }
}
