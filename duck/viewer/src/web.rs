//! Simplified JavaScript-facing API via `wasm-bindgen`.
//!
//! [`WebViewer`] exposes a subset of the native [`Viewer`] API that is compatible
//! with `wasm-bindgen`'s type constraints. It covers scene loading, input forwarding,
//! and the render loop — enough for embedding a 3D viewer in a JS/TS web app.
//!
//! This wrapper exists because [`Viewer`] cannot be directly exported to JavaScript:
//! it carries a lifetime parameter (`Viewer<'a>`) for GPU surface ownership, uses
//! `cgmath` math types throughout, and relies on trait-object callbacks for its event
//! system — none of which are compatible with `wasm-bindgen`.
//!
//! For full-featured web applications that need access to the complete engine API
//! (scene graph manipulation, camera control, selection, operators, etc.), write the
//! application in Rust using [`Viewer`] directly and compile to WebAssembly. Use
//! `#[wasm_bindgen]` at your own application boundary to expose only the JS interop
//! your app requires.

use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::event::{DeviceEvent, Event};
use crate::input::{ElementState, Key, KeyEvent, MouseButton, MouseScrollDelta, NamedKey, PhysicalKey, TouchPhase};
use crate::import_export::LoadHandle;
use crate::viewer::SurfacedViewer;

use std::sync::{Arc, Mutex};

#[wasm_bindgen]
pub enum LoadStatus {
    NoLoad = -1,
    InProgress = 0,
    Success = 1,
    Error = 2,
}

#[wasm_bindgen]
pub struct WebViewer {
    viewer: SurfacedViewer<'static>,
    pending_load: Option<LoadHandle>,
}

#[wasm_bindgen]
impl WebViewer {
    /// Create a new WebViewer from an HTML canvas element.
    ///
    /// Initializes the wgpu rendering pipeline on the given canvas.
    /// Call this once, then use `update_and_render()` in a requestAnimationFrame loop.
    ///
    /// Usage from JS: `const viewer = await WebViewer.create(canvas);`
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WebViewer, JsValue> {
        console_error_panic_hook::set_once();
        console_log::init_with_level(log::Level::Info).ok();

        let viewer = SurfacedViewer::from_canvas(canvas).await;
        Ok(WebViewer {
            viewer,
            pending_load: None,
        })
    }

    /// Call once per frame from requestAnimationFrame.
    /// Updates and renders the scene.
    pub fn update_and_render(&mut self) {
        self.viewer.update();
        if let Err(e) = self.viewer.render() {
            log::error!("Render error: {}", e);
        }
    }

    /// Notify the viewer that the canvas was resized.
    /// Pass the canvas pixel dimensions (CSS size * devicePixelRatio).
    pub fn resize(&mut self, width: u32, height: u32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::Resized((width, height))));
    }

    /// Forward a mousemove event.
    /// - `x`, `y`: cursor position in canvas pixel coordinates
    /// - `dx`, `dy`: movement delta (from MouseEvent.movementX/Y)
    pub fn on_mouse_move(&mut self, x: f64, y: f64, dx: f64, dy: f64) {
        self.viewer
            .handle_event(&Event::Device(DeviceEvent::CursorMoved { position: (x, y) }));
        self.viewer
            .handle_event(&Event::Device(DeviceEvent::MouseMotion { delta: (dx, dy) }));
    }

    /// Forward a mousedown event. `button` is the DOM MouseEvent.button value.
    pub fn on_mouse_down(&mut self, button: i32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::MouseInput {
            state: ElementState::Pressed,
            button: dom_button(button),
        }));
    }

    /// Forward a mouseup event. `button` is the DOM MouseEvent.button value.
    pub fn on_mouse_up(&mut self, button: i32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::MouseInput {
            state: ElementState::Released,
            button: dom_button(button),
        }));
    }

    /// Forward a wheel event. `delta_x` and `delta_y` are in pixels.
    pub fn on_wheel(&mut self, delta_x: f32, delta_y: f32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::MouseWheel {
            delta: MouseScrollDelta::PixelDelta(delta_x, -delta_y),
        }));
    }

    /// Forward a keydown event.
    /// - `key`: the KeyboardEvent.key string (e.g. "w", "Escape")
    /// - `code`: the KeyboardEvent.keyCode numeric value
    /// - `repeat`: whether this is a repeat event
    pub fn on_key_down(&mut self, key: &str, code: u32, repeat: bool) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::KeyboardInput {
            event: make_key_event(key, code, ElementState::Pressed, repeat),
            is_synthetic: false,
        }));
    }

    /// Forward a keyup event.
    pub fn on_key_up(&mut self, key: &str, code: u32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::KeyboardInput {
            event: make_key_event(key, code, ElementState::Released, false),
            is_synthetic: false,
        }));
    }

    /// Forward a touchstart event for a single touch point.
    /// - `id`: the Touch.identifier value
    /// - `x`, `y`: touch position in canvas pixel coordinates
    pub fn on_touch_start(&mut self, id: i32, x: f64, y: f64) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::Touch {
            id,
            phase: TouchPhase::Started,
            position: (x, y),
        }));
    }

    /// Forward a touchmove event for a single touch point.
    pub fn on_touch_move(&mut self, id: i32, x: f64, y: f64) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::Touch {
            id,
            phase: TouchPhase::Moved,
            position: (x, y),
        }));
    }

    /// Forward a touchend event for a single touch point.
    pub fn on_touch_end(&mut self, id: i32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::Touch {
            id,
            phase: TouchPhase::Ended,
            position: (0.0, 0.0),
        }));
    }

    /// Forward a touchcancel event for a single touch point.
    pub fn on_touch_cancel(&mut self, id: i32) {
        self.viewer.handle_event(&Event::Device(DeviceEvent::Touch {
            id,
            phase: TouchPhase::Cancelled,
            position: (0.0, 0.0),
        }));
    }

    /// Begin an async scene load. Format is auto-detected from magic bytes.
    /// Cancels any in-flight load. Poll with `poll_load()` each frame.
    pub fn start_load(&mut self, data: &[u8]) {
        use crate::import_export::{LoadOptions, SceneSource};
        let aspect = self.viewer.camera().aspect;
        self.pending_load = Some(crate::import_export::load_async(
            SceneSource::Bytes(data.to_vec()),
            LoadOptions { aspect },
        ));
    }

    /// Poll the in-flight load.
    pub fn poll_load(&mut self) -> LoadStatus {
        let Some(ref handle) = self.pending_load else {
            return LoadStatus::NoLoad;
        };

        if !handle.is_done() {
            return LoadStatus::InProgress;
        }

        let handle = self.pending_load.take().unwrap();
        match handle.try_get() {
            Some(Ok(result)) => {
                self.set_loaded_scene(result.scene, result.camera);
                LoadStatus::Success
            }
            Some(Err(e)) => {
                log::error!("Load failed: {}", e);
                LoadStatus::Error
            }
            None => {
                log::error!("Load handle done but no result available");
                LoadStatus::Error
            }
        }
    }

    /// Current load progress percentage (0-100). Returns 0 if no load active or progress is indeterminate.
    pub fn load_progress_pct(&self) -> u8 {
        self.pending_load
            .as_ref()
            .and_then(|h| h.progress().snapshot().progress)
            .map(|f| (f * 100.0) as u8)
            .unwrap_or(0)
    }

    /// Human-readable description of the current load operation. Returns an empty string if no load active.
    pub fn load_description(&self) -> String {
        self.pending_load
            .as_ref()
            .map(|h| h.progress().snapshot().description)
            .unwrap_or_default()
    }

    /// Returns true if the active operation has indeterminate progress (show a spinner, not a bar).
    pub fn load_progress_is_indeterminate(&self) -> bool {
        self.pending_load
            .as_ref()
            .map(|h| h.progress().snapshot().progress.is_none())
            .unwrap_or(false)
    }

    /// Returns true if there is item-level stage progress to display.
    pub fn load_has_stage_progress(&self) -> bool {
        self.pending_load
            .as_ref()
            .and_then(|h| h.progress().snapshot().stage)
            .is_some()
    }

    /// Number of completed items in the active stage. Returns 0 if no stage progress or no active load.
    pub fn load_stage_current(&self) -> u32 {
        self.pending_load
            .as_ref()
            .and_then(|h| h.progress().snapshot().stage)
            .map(|(done, _)| done)
            .unwrap_or(0)
    }

    /// Total number of items in the active stage. Returns 0 if no stage progress or no active load.
    pub fn load_stage_total(&self) -> u32 {
        self.pending_load
            .as_ref()
            .and_then(|h| h.progress().snapshot().stage)
            .map(|(_, total)| total)
            .unwrap_or(0)
    }

    /// Load a scene synchronously from raw bytes. Format (glTF or duck) is auto-detected.
    /// The scene and camera will be set automatically.
    pub fn load(&mut self, data: &[u8]) -> Result<(), JsValue> {
        use crate::import_export::{LoadOptions, SceneSource};
        let aspect = self.viewer.camera().aspect;
        let result = crate::import_export::load_sync(
            SceneSource::Bytes(data.to_vec()),
            LoadOptions { aspect },
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

        self.set_loaded_scene(result.scene, result.camera);
        Ok(())
    }

    /// Install a freshly loaded scene, applying its camera if present or
    /// otherwise framing the scene bounds.
    fn set_loaded_scene(
        &mut self,
        scene: crate::scene::Scene,
        camera: Option<crate::scene::PositionedCamera>,
    ) {
        let bounds = scene.bounding().bounds;
        self.viewer.set_scene(Arc::new(Mutex::new(scene)));
        if let Some(camera) = camera {
            self.viewer.set_camera(camera);
        } else if let Some(bounds) = bounds {
            self.viewer.with_camera_mut(|c| c.fit_to_bounds(&bounds));
        }
    }

    /// Load a glTF/glb model from raw bytes.
    /// The scene and camera will be set automatically.
    pub fn load_gltf(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.load(data)
    }

    /// Load a scene from .duck format bytes.
    /// The scene and camera will be set automatically.
    pub fn load_scene(&mut self, data: &[u8]) -> Result<(), JsValue> {
        self.load(data)
    }

    /// Clear the current scene.
    pub fn clear_scene(&mut self) {
        self.viewer.clear_scene();
    }

    /// Get the number of root nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.viewer.scene().lock().unwrap().root_nodes().len()
    }

    /// Get the number of meshes in the scene.
    pub fn mesh_count(&self) -> usize {
        self.viewer.scene().lock().unwrap().mesh_count()
    }
}

/// Convert a DOM MouseEvent.button value to our MouseButton type.
fn dom_button(button: i32) -> MouseButton {
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        n => MouseButton::Other(n as u16),
    }
}

/// Convert a DOM KeyboardEvent.key string to our Key type.
fn dom_key(key: &str) -> Key {
    match key {
        "Escape" => Key::Named(NamedKey::Escape),
        "Enter" => Key::Named(NamedKey::Enter),
        "Tab" => Key::Named(NamedKey::Tab),
        "Backspace" => Key::Named(NamedKey::Backspace),
        "Delete" => Key::Named(NamedKey::Delete),
        " " => Key::Named(NamedKey::Space),
        "ArrowLeft" => Key::Named(NamedKey::ArrowLeft),
        "ArrowRight" => Key::Named(NamedKey::ArrowRight),
        "ArrowUp" => Key::Named(NamedKey::ArrowUp),
        "ArrowDown" => Key::Named(NamedKey::ArrowDown),
        "Home" => Key::Named(NamedKey::Home),
        "End" => Key::Named(NamedKey::End),
        "PageUp" => Key::Named(NamedKey::PageUp),
        "PageDown" => Key::Named(NamedKey::PageDown),
        "Control" => Key::Named(NamedKey::Control),
        "Alt" => Key::Named(NamedKey::Alt),
        "Shift" => Key::Named(NamedKey::Shift),
        "Meta" => Key::Named(NamedKey::Super),
        "F1" => Key::Named(NamedKey::F1),
        "F2" => Key::Named(NamedKey::F2),
        "F3" => Key::Named(NamedKey::F3),
        "F4" => Key::Named(NamedKey::F4),
        "F5" => Key::Named(NamedKey::F5),
        "F6" => Key::Named(NamedKey::F6),
        "F7" => Key::Named(NamedKey::F7),
        "F8" => Key::Named(NamedKey::F8),
        "F9" => Key::Named(NamedKey::F9),
        "F10" => Key::Named(NamedKey::F10),
        "F11" => Key::Named(NamedKey::F11),
        "F12" => Key::Named(NamedKey::F12),
        s if s.len() == 1 => {
            let c = s.chars().next().unwrap();
            Key::Character(c)
        }
        _ => Key::Unidentified,
    }
}

/// Build a KeyEvent from DOM keyboard event properties.
fn make_key_event(key: &str, code: u32, state: ElementState, repeat: bool) -> KeyEvent {
    KeyEvent {
        physical_key: PhysicalKey::Code(code),
        logical_key: dom_key(key),
        state,
        repeat,
    }
}
