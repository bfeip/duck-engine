//! The scene's owned resources.
//!
//! Everything here is addressed by a typed [`Id`], stored in a [`SceneData`]
//! collection, and owned by a refcounted [`Handle`] — the seven types
//! implementing [`SceneResource`], plus the id and handle machinery itself.
//!
//! Membership is that rule, not a taxonomy: an [`EnvironmentMap`] carries an id
//! but is not refcounted, so it stays outside this module.
//!
//! [`SceneData`]: crate::SceneData
//! [`EnvironmentMap`]: crate::EnvironmentMap

mod display;
mod handle;
mod id;
mod instance;
mod material;
mod mesh;
mod node;
mod texture;

pub use display::{DisplayBehavior, RenderLayer};
pub use handle::{
    FaceMaterialHandle, Handle, InstanceHandle, LineMaterialHandle, MeshHandle, NodeHandle,
    PointMaterialHandle, ResourceKind, SceneResource, TextureHandle, WeakHandle,
};
pub use id::{GenericId, Id};
pub use instance::{Instance, InstanceId};
pub use material::{
    AlphaMode, DEFAULT_METALLIC, DEFAULT_ROUGHNESS, FaceMaterial, FaceMaterialId, LineMaterial,
    LineMaterialId, MaterialFlags, MaterialProperties, PointMaterial, PointMaterialId,
};
pub use mesh::{
    Mesh, MeshDescriptor, MeshId, MeshIndex, MeshPrimitive, ObjMesh, PrimitiveType,
    SubGeometryElement, SubGeometryKind, SubMeshRange, Topology, Vertex,
};
pub use node::{
    CustomNodePayload, EffectiveVisibility, Node, NodeFlags, NodeId, NodePayload, Visibility,
};
pub use texture::{Texture, TextureFormat, TextureId};

// Handle plumbing for `SceneData`; not part of the public surface.
pub(crate) use handle::{HandleCore, SceneBind};
