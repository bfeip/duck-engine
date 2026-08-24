//! Canvas integration for the emscripten target.
//!
//! Counterpart to [`winit_support`](crate::winit_support): it supplies the
//! surface target that [`WindowSurface`](crate::WindowSurface) needs, since
//! `wgpu::SurfaceTarget::Canvas` does not exist on emscripten.

use std::ffi::CString;

use wgpu::rwh::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle,
    RawWindowHandle, WebDisplayHandle, WebWindowHandle, WindowHandle,
};

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
        let selector = CString::new(selector).expect("canvas selector contains a nul byte");
        let ptr = Box::leak(selector.into_boxed_c_str()).as_ptr();
        Self { id: ptr as usize as u32 }
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
