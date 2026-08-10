//! GPU-free scene description for duck-engine.
//!
//! This crate defines what a scene *is*: a tree of nodes carrying mesh
//! instances, cameras, and lights, backed by shared resources — meshes,
//! materials, textures, and image-based-lighting environment maps. It also
//! answers geometric questions about a scene ([`geom_query`]: ray and volume
//! picking). It contains no GPU types; rendering is a separate concern built
//! on the change-tracking described below.
//!
//! # The scene container
//!
//! [`SceneData`] owns everything in a scene. [`Scene`] wraps one in a cheap,
//! cloneable handle: clones share the same scene, and a deep copy goes through
//! `SceneData: Clone`. There are two ways in:
//!
//! - **Handle methods.** [`Scene`] exposes the common operations directly
//!   ([`Scene::add_mesh`], [`Scene::set_node_transform`], …). Each call locks
//!   internally and is a complete critical section.
//! - **A guard.** [`Scene::lock`] returns a [`SceneGuard`] deref-ing to
//!   [`SceneData`], for compound work: importing a file, walking the tree,
//!   rendering a frame.
//!
//! The lock is not reentrant. Never call a `Scene` method while holding that
//! scene's guard. In debug builds this panics instead of deadlocking.
//!
//! # Resources, ids, and handles
//!
//! Resources live in [`resource`]. Each is addressed by a typed id
//! ([`resource::Id`], aliased per kind: [`resource::MeshId`],
//! [`resource::NodeId`], …), so ids of different kinds cannot be mixed up at
//! compile time.
//!
//! Adding a resource returns a refcounted, owning [`resource::Handle`]
//! ([`resource::MeshHandle`], [`resource::NodeHandle`], …). The scene's own
//! references are handles too: an [`Instance`](resource::Instance) owns its
//! mesh and material handles, materials own their texture handles, a node owns
//! its children and its payload instance, and the scene owns its root nodes.
//! When a resource's last handle drops, the resource is removed automatically
//! the next time the scene is locked or unlocked — along with anything only it
//! was keeping alive.
//!
//! Removal comes in two strengths:
//!
//! - [`SceneData::remove_node`] **detaches**: the tree gives up its ownership.
//!   With no other handles the subtree goes away; if you still hold a
//!   [`NodeHandle`](resource::NodeHandle) the node survives detached — in the
//!   scene, but not rendered or traversed — and
//!   [`SceneData::reparent_node`] reattaches it.
//! - [`SceneData::remove_mesh`] and the other per-resource removers
//!   **force-remove** even while handles exist; those handles become inert.
//!
//! Consumers that mirror scene resources into derived state can drain
//! [`SceneData::take_removals`] to learn what disappeared.
//!
//! # The scene graph
//!
//! A [`Node`](resource::Node) has a local
//! [`Transform`](common::Transform), an optional parent, children, and a
//! [`NodePayload`](resource::NodePayload) saying what it is: nothing (a
//! grouping node), an [`Instance`](resource::Instance) pairing a mesh with up
//! to three materials (face, line, point), a camera, or a light. Cameras and
//! lights take their pose from the node's world transform;
//! [`SceneData::set_active_camera`] chooses the node that drives rendering.
//!
//! Node [`Visibility`](resource::Visibility) is what you author; what actually
//! shows is the inherited
//! [`EffectiveVisibility`](resource::EffectiveVisibility).
//! [`NodeFlags`](resource::NodeFlags) opt a node out of picking or bounds
//! contribution, and [`DisplayBehavior`](resource::DisplayBehavior) — also
//! inherited — requests screen-space rendering (constant pixel size,
//! camera-facing, overlay layer).
//!
//! # Change tracking
//!
//! There are no dirty flags. Every mutable resource carries a `u64`
//! generation, bumped on each mutation and exposed as `generation()`; the node
//! tree as a whole is covered by [`SceneData::node_generation`]. A consumer
//! that mirrors scene state — a renderer keeping GPU copies, say — remembers
//! the last generation it saw per resource and re-reads whatever now reports a
//! newer one. Generations start above zero, so the first comparison always
//! reports a change. Removals are observed separately, through
//! [`SceneData::take_removals`].
//!
//! # Geometry queries
//!
//! [`geom_query`] resolves rays and volumes against the scene: closest-first
//! ray picking of faces, lines, and points; volume picking for rubber-band
//! selection; and per-mesh intersection primitives underneath. See the module
//! docs for the typical cursor-to-hit flow.
//!
//! # Serialization
//!
//! With the `serde` feature, resources serialize; a handle serializes as its
//! id alone. Deserialized data therefore holds unbound handles — call
//! [`SceneData::rebind`] to make ownership live before using it.
//!
//! # Example
//!
//! ```
//! use duck_engine_scene::{Scene, SceneData};
//! use duck_engine_scene::common::{RgbaColor, Transform};
//! use duck_engine_scene::resource::{FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType};
//!
//! // Build scene data, then wrap it in a shareable handle.
//! let mut data = SceneData::new();
//! let mesh = data.add_mesh(Mesh::sphere(1.0, 48, 24, PrimitiveType::TriangleList));
//! let material =
//!     data.add_face_material(FaceMaterial::new().with_base_color_factor(RgbaColor::RED));
//! data.add_instance_node(
//!     None, // parent; None creates a root node
//!     Instance::new(mesh).with_face_material(material),
//!     Some("sphere".to_string()),
//!     Transform::IDENTITY,
//!     NodeFlags::NONE,
//! )
//! .unwrap();
//! let scene = Scene::new(data);
//!
//! // Single operations lock internally; compound work takes a guard.
//! let bounds = scene.bounding();
//! let guard = scene.lock();
//! for node in guard.nodes() { /* ... */ }
//! ```

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
