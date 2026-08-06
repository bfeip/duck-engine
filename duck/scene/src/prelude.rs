//! The crate's common types in one flat namespace.
//!
//! Convenience for consumers that touch many scene types in one file:
//! `use duck_engine_scene::prelude::*`. Code importing only a name or two
//! should prefer the explicit module path.
//!
//! [`geom_query`](crate::geom_query) and `cad` are deliberately absent — they
//! are already namespaced and used sparingly.

pub use crate::camera::{CameraProjection, PositionedCamera};
pub use crate::environment::{EnvironmentMap, EnvironmentMapId, EnvironmentSource};
pub use crate::light::{Light, LightType, MAX_LIGHTS};
pub use crate::resource::*;
pub use crate::view::{View, ViewId};
pub use crate::{BoundingResult, Scene, SceneData, SceneGuard, SceneProperties};
