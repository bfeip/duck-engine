//! The crate's common types in one flat namespace.
//!
//! Convenience for consumers that touch many scene types in one file:
//! `use duck_engine_scene::prelude::*`. Code importing only a name or two
//! should prefer the explicit path. [`geom_query`](crate::geom_query) and
//! `cad` are not included.

pub use crate::resource::*;
pub use crate::{
    BoundingResult, CameraProjection, EnvironmentMap, EnvironmentMapId, EnvironmentSource, Light,
    LightType, MAX_LIGHTS, PositionedCamera, SavedView, SavedViewId, Scene, SceneData, SceneGuard,
    SceneProperties,
};
