
pub use duck_engine_common as common;

#[cfg(feature = "cad")]
pub mod cad;
pub mod geom_query;
pub mod prelude;
pub mod resource;

mod camera;
mod data;
mod environment;
mod light;
mod scene_handle;
mod view;

pub use camera::{CameraProjection, PositionedCamera};
pub use data::{BoundingResult, SceneData, SceneProperties};
pub use environment::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};
pub use light::{Light, LightType, MAX_LIGHTS};
pub use scene_handle::{Scene, SceneGuard};
pub use view::{View, ViewId};

/// Default generation counter value for newly created resources.
/// Starts at 1 so initial change detection triggers on first use.
pub(crate) fn initial_generation() -> u64 {
    1
}
