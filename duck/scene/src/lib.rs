
pub use duck_engine_common as common;

// Scene submodules
mod id;
#[cfg(feature = "cad")]
pub mod cad;
mod camera;
mod display;
mod environment;
pub mod geom_query;
mod handle;
mod instance;
mod coordinate_space;
mod data;
mod light;
mod material;
mod mesh;
mod node;
mod resource_handle;
mod texture;
mod view;

// ID types
pub use id::{GenericId, Id};
pub use environment::EnvironmentMapId;
pub use instance::InstanceId;
pub use material::{FaceMaterialId, LineMaterialId, PointMaterialId};
pub use mesh::MeshId;
pub use node::NodeId;
pub use texture::TextureId;

pub use camera::{CameraProjection, PositionedCamera};
pub use data::{BoundingResult, SceneData, SceneProperties};
pub use handle::{Scene, SceneGuard};
pub use resource_handle::{
    FaceMaterialHandle, Handle, InstanceHandle, LineMaterialHandle, MeshHandle, NodeHandle,
    PointMaterialHandle, ResourceKind, SceneResource, TextureHandle, WeakHandle,
};
pub use display::{DisplayBehavior, RenderLayer};
pub use coordinate_space::CoordinateSpace;
pub use view::{View, ViewId};
pub use instance::Instance;
pub use light::{Light, LightType, MAX_LIGHTS};
pub use material::{
    AlphaMode, FaceMaterial, LineMaterial, MaterialFlags, MaterialProperties, PointMaterial,
    DEFAULT_METALLIC, DEFAULT_ROUGHNESS,
};
pub use mesh::{Mesh, MeshDescriptor, MeshIndex, MeshPrimitive, ObjMesh, PrimitiveType, SubGeometryElement, SubGeometryKind, SubMeshRange, Topology, Vertex};
pub use node::{CustomNodePayload, EffectiveVisibility, Node, NodePayload, Visibility, NodeFlags};
pub use texture::{Texture, TextureFormat};
pub use environment::{EnvironmentMap, EnvironmentSource};

/// Default generation counter value for newly created resources.
/// Starts at 1 so initial change detection triggers on first use.
pub(crate) fn initial_generation() -> u64 {
    1
}
