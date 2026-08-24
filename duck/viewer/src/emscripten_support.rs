//! Canvas and input integration for the emscripten target.
//!
//! Counterpart to [`winit_support`](crate::winit_support). It supplies the two
//! things winit provides elsewhere: the surface target that
//! [`WindowSurface`](crate::WindowSurface) needs (there is no
//! `wgpu::SurfaceTarget::Canvas` on emscripten), and a translation of the
//! browser's DOM events into [`Event`].
//!
//! Register the callbacks once with [`register_input`], then call
//! [`drain_events`] each frame and feed the result to the viewer.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::{CString, c_char, c_int, c_void};

use wgpu::rwh::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WebDisplayHandle, WebWindowHandle, WindowHandle,
};

use crate::event::{DeviceEvent, Event};
use crate::input::{ElementState, Key, KeyEvent, MouseButton, MouseScrollDelta, NamedKey, PhysicalKey};

// ============================================================================
// Surface target
// ============================================================================

/// A canvas addressed by CSS selector, usable as a wgpu surface target.
///
/// wgpu's emscripten GLES backend reinterprets `WebWindowHandle::id` as a
/// `const char*` and passes it to `eglCreateWindowSurface`, which resolves it as
/// a CSS selector. The selector is leaked so the pointer stays valid for as long
/// as any surface derived from it.
#[derive(Clone, Copy)]
pub struct CanvasSelector {
    id: u32,
}

impl CanvasSelector {
    /// Wrap a CSS selector naming a canvas element, e.g. `"#canvas"`.
    ///
    /// # Panics
    ///
    /// If the selector contains an interior nul byte.
    pub fn new(selector: &str) -> Self {
        Self { id: leak_selector(selector) as usize as u32 }
    }
}

impl HasWindowHandle for CanvasSelector {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let raw = RawWindowHandle::Web(WebWindowHandle::new(self.id));
        // SAFETY: `id` points at a leaked C string, so it outlives the handle.
        Ok(unsafe { WindowHandle::borrow_raw(raw) })
    }
}

impl HasDisplayHandle for CanvasSelector {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        let raw = RawDisplayHandle::Web(WebDisplayHandle::new());
        // SAFETY: the web display handle carries no pointer.
        Ok(unsafe { DisplayHandle::borrow_raw(raw) })
    }
}

/// Copy a selector to the heap as a C string and leak it. Emscripten's callback
/// registration keeps the target string, so it must outlive registration.
fn leak_selector(selector: &str) -> *const c_char {
    let selector = CString::new(selector).expect("canvas selector contains a nul byte");
    Box::leak(selector.into_boxed_c_str()).as_ptr()
}

// ============================================================================
// Emscripten HTML5 ABI
// ============================================================================

/// Struct layouts mirror `emscripten/html5.h`. The size assertions below guard
/// against silent drift if the emsdk layout changes; verified against emsdk
/// 5.0.7.
mod ffi {
    use super::{c_char, c_int, c_void};

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MouseEvent {
        pub timestamp: f64,
        pub screen_x: c_int,
        pub screen_y: c_int,
        pub client_x: c_int,
        pub client_y: c_int,
        pub ctrl_key: bool,
        pub shift_key: bool,
        pub alt_key: bool,
        pub meta_key: bool,
        pub button: u16,
        pub buttons: u16,
        pub movement_x: c_int,
        pub movement_y: c_int,
        /// Position relative to the event target, i.e. canvas-local.
        pub target_x: c_int,
        pub target_y: c_int,
        /// Deprecated upstream; present only to keep the layout faithful.
        pub canvas_x: c_int,
        pub canvas_y: c_int,
        pub padding: c_int,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct WheelEvent {
        pub mouse: MouseEvent,
        pub delta_x: f64,
        pub delta_y: f64,
        pub delta_z: f64,
        pub delta_mode: u32,
    }

    pub const SHORT_STRING_LEN: usize = 32;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct KeyboardEvent {
        pub timestamp: f64,
        pub location: u32,
        pub ctrl_key: bool,
        pub shift_key: bool,
        pub alt_key: bool,
        pub meta_key: bool,
        pub repeat: bool,
        pub char_code: u32,
        pub key_code: u32,
        pub which: u32,
        pub key: [c_char; SHORT_STRING_LEN],
        pub code: [c_char; SHORT_STRING_LEN],
        pub char_value: [c_char; SHORT_STRING_LEN],
        pub locale: [c_char; SHORT_STRING_LEN],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct UiEvent {
        pub detail: c_int,
        pub document_body_client_width: c_int,
        pub document_body_client_height: c_int,
        pub window_inner_width: c_int,
        pub window_inner_height: c_int,
        pub window_outer_width: c_int,
        pub window_outer_height: c_int,
        pub scroll_top: c_int,
        pub scroll_left: c_int,
    }

    const _: () = assert!(size_of::<MouseEvent>() == 64);
    const _: () = assert!(size_of::<WheelEvent>() == 96);
    const _: () = assert!(size_of::<KeyboardEvent>() == 160);
    const _: () = assert!(size_of::<UiEvent>() == 36);

    pub type MouseCallback = extern "C" fn(c_int, *const MouseEvent, *mut c_void) -> bool;
    pub type WheelCallback = extern "C" fn(c_int, *const WheelEvent, *mut c_void) -> bool;
    pub type KeyCallback = extern "C" fn(c_int, *const KeyboardEvent, *mut c_void) -> bool;
    pub type UiCallback = extern "C" fn(c_int, *const UiEvent, *mut c_void) -> bool;

    /// `EMSCRIPTEN_EVENT_TARGET_WINDOW` — a sentinel, not a real pointer.
    pub const TARGET_WINDOW: *const c_char = 2 as *const c_char;
    /// `EM_CALLBACK_THREAD_CONTEXT_CALLING_THREAD`.
    pub const THREAD_CALLING: *mut c_void = 2 as *mut c_void;

    // The public `emscripten_set_*_callback` names are macros; the linkable
    // symbols are the `_on_thread` forms.
    unsafe extern "C" {
        pub fn emscripten_set_mousedown_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: MouseCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_mouseup_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: MouseCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_mousemove_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: MouseCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_wheel_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: WheelCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_keydown_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: KeyCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_keyup_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: KeyCallback, thread: *mut c_void,
        ) -> c_int;
        pub fn emscripten_set_resize_callback_on_thread(
            target: *const c_char, user_data: *mut c_void, use_capture: bool,
            callback: UiCallback, thread: *mut c_void,
        ) -> c_int;

        pub fn emscripten_get_element_css_size(
            target: *const c_char, width: *mut f64, height: *mut f64,
        ) -> c_int;
        pub fn emscripten_set_canvas_element_size(
            target: *const c_char, width: c_int, height: c_int,
        ) -> c_int;
    }
}

// ============================================================================
// Event queue
// ============================================================================

thread_local! {
    /// Emscripten delivers DOM events on the main thread between frames, so a
    /// plain thread-local queue is enough; the frame loop drains it.
    static EVENTS: RefCell<VecDeque<Event>> = RefCell::new(VecDeque::new());
    /// The canvas the input callbacks are registered on, for size queries.
    static CANVAS: RefCell<Option<*const c_char>> = const { RefCell::new(None) };
}

fn push(event: Event) {
    EVENTS.with(|q| q.borrow_mut().push_back(event));
}

/// Take every event queued since the last call.
///
/// Call once per frame and hand the events to
/// [`Viewer::handle_event`](crate::Viewer::handle_event).
pub fn drain_events() -> Vec<Event> {
    EVENTS.with(|q| q.borrow_mut().drain(..).collect())
}

// ============================================================================
// Conversions
// ============================================================================

fn mouse_button(button: u16) -> MouseButton {
    // DOM button numbering: 0 primary, 1 auxiliary, 2 secondary, 3 back,
    // 4 forward.
    match button {
        0 => MouseButton::Left,
        1 => MouseButton::Middle,
        2 => MouseButton::Right,
        3 => MouseButton::Back,
        4 => MouseButton::Forward,
        other => MouseButton::Other(other),
    }
}

/// Read one of the fixed-size UTF-8 fields of a keyboard event.
fn short_string(bytes: &[c_char; ffi::SHORT_STRING_LEN]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let bytes: Vec<u8> = bytes[..end].iter().map(|&b| b as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// Map a DOM `KeyboardEvent.key` value onto our logical key.
fn logical_key(key: &str) -> Key {
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
        // The DOM reports both Windows/Command keys as "Meta".
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
        other => {
            // A single-character `key` is the typed character; anything longer
            // is a named key we do not model.
            let mut chars = other.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) => Key::Character(c),
                _ => Key::Unidentified,
            }
        }
    }
}

// ============================================================================
// Callbacks
// ============================================================================

extern "C" fn on_mouse_down(_kind: c_int, event: *const ffi::MouseEvent, _: *mut c_void) -> bool {
    let event = unsafe { &*event };
    push(Event::Device(DeviceEvent::MouseInput {
        state: ElementState::Pressed,
        button: mouse_button(event.button),
    }));
    true
}

extern "C" fn on_mouse_up(_kind: c_int, event: *const ffi::MouseEvent, _: *mut c_void) -> bool {
    let event = unsafe { &*event };
    push(Event::Device(DeviceEvent::MouseInput {
        state: ElementState::Released,
        button: mouse_button(event.button),
    }));
    true
}

extern "C" fn on_mouse_move(_kind: c_int, event: *const ffi::MouseEvent, _: *mut c_void) -> bool {
    let event = unsafe { &*event };
    // Both are needed: operators track absolute position for picking and
    // relative motion for drags.
    push(Event::Device(DeviceEvent::CursorMoved {
        position: (event.target_x as f64, event.target_y as f64),
    }));
    push(Event::Device(DeviceEvent::MouseMotion {
        delta: (event.movement_x as f64, event.movement_y as f64),
    }));
    true
}

extern "C" fn on_wheel(_kind: c_int, event: *const ffi::WheelEvent, _: *mut c_void) -> bool {
    let event = unsafe { &*event };
    // DOM wheel deltas grow downward/rightward, the opposite of the winit
    // convention the operators are written against.
    let delta = match event.delta_mode {
        // DOM_DELTA_LINE
        1 => MouseScrollDelta::LineDelta(-event.delta_x as f32, -event.delta_y as f32),
        // DOM_DELTA_PIXEL (0) and DOM_DELTA_PAGE (2); page-mode wheels are
        // vanishingly rare and pixels are the better approximation.
        _ => MouseScrollDelta::PixelDelta(-event.delta_x as f32, -event.delta_y as f32),
    };
    push(Event::Device(DeviceEvent::MouseWheel { delta }));
    true
}

fn push_key(event: &ffi::KeyboardEvent, state: ElementState) {
    push(Event::Device(DeviceEvent::KeyboardInput {
        event: KeyEvent {
            // Nothing in the engine reads the physical key; everything
            // dispatches on the logical one.
            physical_key: PhysicalKey::Unidentified,
            logical_key: logical_key(&short_string(&event.key)),
            state,
            repeat: event.repeat,
        },
        is_synthetic: false,
    }));
}

extern "C" fn on_key_down(_kind: c_int, event: *const ffi::KeyboardEvent, _: *mut c_void) -> bool {
    push_key(unsafe { &*event }, ElementState::Pressed);
    // Returning false leaves browser shortcuts (reload, devtools) working.
    false
}

extern "C" fn on_key_up(_kind: c_int, event: *const ffi::KeyboardEvent, _: *mut c_void) -> bool {
    push_key(unsafe { &*event }, ElementState::Released);
    false
}

extern "C" fn on_resize(_kind: c_int, _event: *const ffi::UiEvent, _: *mut c_void) -> bool {
    // The UI event reports the window, not the canvas. Resize the canvas to its
    // own CSS box and report that, so the viewer always matches what is drawn.
    if let Some((width, height)) = canvas_size() {
        push(Event::Device(DeviceEvent::Resized((width, height))));
    }
    false
}

// ============================================================================
// Registration
// ============================================================================

/// Current canvas backing-store size in physical pixels, resizing the canvas to
/// match its CSS box first.
///
/// Returns `None` before [`register_input`] has run.
pub fn canvas_size() -> Option<(u32, u32)> {
    let canvas = CANVAS.with(|c| *c.borrow())?;

    let (mut css_width, mut css_height) = (0.0f64, 0.0f64);
    if unsafe { ffi::emscripten_get_element_css_size(canvas, &mut css_width, &mut css_height) } != 0
    {
        return None;
    }

    let width = (css_width.max(1.0)) as u32;
    let height = (css_height.max(1.0)) as u32;
    unsafe { ffi::emscripten_set_canvas_element_size(canvas, width as c_int, height as c_int) };
    Some((width, height))
}

/// Register DOM event callbacks for `canvas_selector`.
///
/// Pointer events are bound to the canvas so they carry canvas-local
/// coordinates; keyboard and resize are bound to the window so they arrive
/// without the canvas holding focus. Call once, before the frame loop; after
/// this, poll [`drain_events`] each frame.
pub fn register_input(canvas_selector: &str) {
    let canvas = leak_selector(canvas_selector);
    CANVAS.with(|c| *c.borrow_mut() = Some(canvas));

    unsafe {
        ffi::emscripten_set_mousedown_callback_on_thread(
            canvas, std::ptr::null_mut(), false, on_mouse_down, ffi::THREAD_CALLING,
        );
        // Pointer events must stay on the canvas: `target_x`/`target_y` are
        // relative to the event target, so binding these to the window would
        // report window coordinates. The cost is that a drag leaving the canvas
        // stops updating until it returns — fixing that needs DOM pointer
        // capture, which emscripten's HTML5 API does not wrap.
        ffi::emscripten_set_mouseup_callback_on_thread(
            canvas, std::ptr::null_mut(), false, on_mouse_up, ffi::THREAD_CALLING,
        );
        ffi::emscripten_set_mousemove_callback_on_thread(
            canvas, std::ptr::null_mut(), false, on_mouse_move, ffi::THREAD_CALLING,
        );
        ffi::emscripten_set_wheel_callback_on_thread(
            canvas, std::ptr::null_mut(), false, on_wheel, ffi::THREAD_CALLING,
        );
        ffi::emscripten_set_keydown_callback_on_thread(
            ffi::TARGET_WINDOW, std::ptr::null_mut(), false, on_key_down, ffi::THREAD_CALLING,
        );
        ffi::emscripten_set_keyup_callback_on_thread(
            ffi::TARGET_WINDOW, std::ptr::null_mut(), false, on_key_up, ffi::THREAD_CALLING,
        );
        ffi::emscripten_set_resize_callback_on_thread(
            ffi::TARGET_WINDOW, std::ptr::null_mut(), false, on_resize, ffi::THREAD_CALLING,
        );
    }
}
