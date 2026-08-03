//! Platform bootstrap: event-loop creation, the entry point, and per-frame
//! window/viewer-state initialization.
//!
//! `run()` is the body of `main`; `resume()` is called from
//! `ApplicationHandler::resumed`. Native builds the viewer synchronously;
//! web creates the canvas, then builds the viewer asynchronously and delivers
//! it back through the event-loop proxy.

// The startup assets are macros rather than consts: web embeds them with
// `include_bytes!`, which needs a literal path.
/// Scene loaded at startup.
macro_rules! default_scene_path {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/1987_mazda_rx-7_fc.glb")
    };
}

/// Environment map applied at startup.
macro_rules! default_environment_path {
    () => {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/the_sky_is_on_fire_4k.hdr")
    };
}

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{resume, run};

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub(crate) use web::{resume, run};
