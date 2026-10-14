//! Platform bootstrap: surface creation, the entry point, the egui platform
//! integration, and file transfer.
//!
//! Each platform module exposes the same two items. [`Host`] owns whatever the
//! platform uses to talk to the windowing system — on native a winit window,
//! `egui_winit::State` and file dialogs, on the web a canvas, a hand-built egui
//! input accumulator and the browser's file picker and downloads — and `run()`
//! is the body of `main`.
//!
//! Everything above the host ([`ViewerState`](crate::ViewerState), the tools,
//! the document) is shared: platform events are normalized to
//! [`duck_engine_viewer::event::Event`] before they reach it.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{Host, run};

#[cfg(target_os = "emscripten")]
mod web;
#[cfg(target_os = "emscripten")]
pub(crate) use web::{Host, run};

/// A kind of file the user may pick, by extension (without the dot).
pub(crate) struct FileFilter {
    /// Shown by native dialogs; the browser's picker has nowhere to show it.
    #[cfg_attr(target_os = "emscripten", allow(dead_code))]
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}
