// Re-export renderer crate
pub use duck_engine_renderer as renderer;

// Re-export scene crate as `scene` module
pub use duck_engine_scene as scene;

// Re-export import-export crate
pub use duck_engine_import_export as import_export;

// Re-export common subsystems from scene crate
pub use duck_engine_scene::common;
pub use duck_engine_scene::geom_query;

// Core modules
pub mod bindings;
pub mod event;
pub mod gizmo;
pub mod input;
pub mod operator;
pub mod selection;
mod scene_scale;
mod viewer;

pub use viewer::{OffscreenViewer, SurfacedViewer, Viewer, WindowSurface};

#[cfg(target_arch = "wasm32")]
pub mod web;

// Optional integrations
#[cfg(feature = "winit-support")]
pub mod winit_support;

#[cfg(feature = "streaming")]
pub mod streaming;
