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
pub mod input;
pub mod operator;
pub mod selection;
mod compositor;
mod scene_scale;
mod view;
mod viewer;

pub use view::{Corner, PixelRect, View, ViewId, ViewLayout};
pub use viewer::{OffscreenViewer, SurfacedViewer, ViewMut, Viewer, WindowSurface};

// Optional integrations
#[cfg(feature = "winit-support")]
pub mod winit_support;
