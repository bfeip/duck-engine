//! Geometry queries against a scene: ray and volume picking.
//!
//! # Picking a point on screen
//!
//! The typical flow turns a cursor position into scene hits:
//!
//! 1. Build a world-space ray with
//!    [`PositionedCamera::ray_from_screen_point`](crate::PositionedCamera::ray_from_screen_point).
//! 2. Describe what to pick with a [`RayPickQuery`]: [`RayPickQuery::faces`],
//!    [`lines`](RayPickQuery::lines), [`points`](RayPickQuery::points) (the
//!    latter two take a world-space tolerance), [`all`](RayPickQuery::all), or
//!    [`for_kinds`](RayPickQuery::for_kinds) for an arbitrary combination.
//! 3. Call [`pick_all_from_ray`]. Results come back sorted nearest-first as
//!    [`RayPickResult`]s: the node and instance hit, the distance, the hit
//!    point, and per-primitive detail in [`RayHit`].
//!
//! [`pick_all_from_volume`] is the volume analogue — every instance touching a
//! convex volume (a rubber-band selection frustum, say) — returning
//! [`VolumePickResult`]s.
//!
//! # Screen-space geometry
//!
//! Scenes can contain screen-space nodes (constant pixel size or
//! camera-facing, see [`DisplayBehavior`](crate::resource::DisplayBehavior)),
//! which are drawn somewhere other than their authored world transform. To
//! pick those where they appear on screen, use the `_with_view` variants
//! ([`pick_all_from_ray_with_view`], [`pick_all_with_view`]) and pass a
//! [`PickView`] describing the camera and viewport.
//!
//! # Lower layers
//!
//! The scene-walking drivers above are built on the [`PickQuery`] trait
//! ([`pick_all`] runs any implementation), and the narrow-phase tests are
//! plain per-mesh functions in local mesh space ([`intersect_ray`],
//! [`intersect_ray_nearest`], [`intersect_volume`], …) for callers that manage
//! their own traversal. Meshes accelerate these queries transparently with a
//! cached [`MeshSpatialIndex`].

mod mesh_intersection;
mod pick_query;
mod ray_picking;
mod spatial_index;
mod volume_picking;

pub use mesh_intersection::{
    intersect_ray, intersect_ray_nearest, intersect_ray_with_lines, intersect_ray_with_points,
    intersect_volume, LineMeshHit, MeshVolumeHit, PointMeshHit, TriangleMeshHit,
};
pub use pick_query::{pick_all, pick_all_with_view, PickQuery, PickView};
pub use spatial_index::MeshSpatialIndex;
pub use ray_picking::{
    pick_all_from_ray, pick_all_from_ray_with_view, RayHit, RayPickQuery, RayPickResult,
};
pub use volume_picking::{pick_all_from_volume, VolumePickQuery, VolumePickResult};
