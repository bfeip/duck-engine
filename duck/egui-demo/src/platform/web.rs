//! Web bootstrap: canvas creation, async wgpu init via the event-loop proxy,
//! and the wasm entry point.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;
use winit::dpi::PhysicalSize;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::web::{EventLoopExtWebSys, WindowAttributesExtWebSys};
use winit::window::Window;

use crate::{App, UserEvent, ViewerState, ui};

pub(crate) fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let proxy = event_loop.create_proxy();

    let app = App {
        state: None,
        ui: ui::UiState::default(),
        workflow_index: 0,
        proxy,
        pending_scene_bytes: Rc::new(RefCell::new(None)),
        pending_hdr_bytes: Rc::new(RefCell::new(None)),
    };

    event_loop.spawn_app(app);
}

/// On web, wgpu init is async and cannot block. Create the canvas
/// synchronously, build the viewer state off the event loop, then deliver it
/// back through the proxy as a [`UserEvent::Initialized`].
pub(crate) fn resume(app: &mut App, event_loop: &ActiveEventLoop) {
    if app.state.is_some() {
        return;
    }

    let size = web_canvas_size();
    let window_attrs = Window::default_attributes()
        .with_title("Duck Engine - egui Example")
        .with_append(true)
        .with_inner_size(size);
    let window = Arc::new(
        event_loop
            .create_window(window_attrs)
            .expect("Failed to create window"),
    );

    // Queue the embedded default scene and environment so they load once the
    // viewer is ready.
    let default_scene = include_bytes!(default_scene_path!());
    *app.pending_scene_bytes.borrow_mut() = Some(default_scene.to_vec());

    let default_env = include_bytes!(default_environment_path!());
    *app.pending_hdr_bytes.borrow_mut() = Some(default_env.to_vec());

    // winit's web backend doesn't emit `Resized` for browser-window resizes, so
    // listen for them ourselves and feed the new size back through the proxy.
    register_resize_listener(app.proxy.clone());

    let proxy = app.proxy.clone();
    wasm_bindgen_futures::spawn_local(async move {
        let state = ViewerState::from_window(window, Some(size)).await;
        let _ = proxy.send_event(UserEvent::Initialized(state));
    });
}

/// Forward browser `resize` events into the winit event loop as
/// [`UserEvent::Resized`], carrying the new canvas size in physical pixels.
fn register_resize_listener(proxy: winit::event_loop::EventLoopProxy<UserEvent>) {
    let win = web_sys::window().expect("no window");
    let closure = Closure::<dyn FnMut()>::new(move || {
        let _ = proxy.send_event(UserEvent::Resized(web_canvas_size()));
    });
    win.add_event_listener_with_callback("resize", closure.as_ref().unchecked_ref())
        .expect("failed to register resize listener");
    // Keep the closure alive for the lifetime of the app.
    closure.forget();
}

/// Size the canvas to the browser window (in physical pixels).
fn web_canvas_size() -> PhysicalSize<u32> {
    let win = web_sys::window().expect("no window");
    let dpr = win.device_pixel_ratio();
    let width = win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(1280.0);
    let height = win.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(720.0);
    PhysicalSize::new((width * dpr) as u32, (height * dpr) as u32)
}
