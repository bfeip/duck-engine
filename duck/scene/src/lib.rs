
pub use duck_engine_common as common;

#[cfg(feature = "cad")]
pub mod cad;
pub mod camera;
pub mod environment;
pub mod geom_query;
pub mod light;
pub mod prelude;
pub mod resource;
pub mod view;

mod data;
mod scene_handle;

pub use data::{BoundingResult, SceneData, SceneProperties};
pub use scene_handle::{Scene, SceneGuard};

/// Default generation counter value for newly created resources.
/// Starts at 1 so initial change detection triggers on first use.
pub(crate) fn initial_generation() -> u64 {
    1
}
