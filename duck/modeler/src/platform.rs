//! Platform bootstrap: surface creation, the entry point, and the egui
//! platform integration.
//!
//! Each platform module exposes the same two items. [`Host`] owns whatever the
//! platform uses to talk to the windowing system — on native a winit window and
//! `egui_winit::State`, on the web a canvas and a hand-built egui input
//! accumulator — and `run()` is the body of `main`.
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
