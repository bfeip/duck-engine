//! The scene's owned resources.
//!
//! Everything here is addressed by a typed [`Id`], stored in a [`SceneData`]
//! collection, and owned by a refcounted [`Handle`] — the seven types
//! implementing [`SceneResource`], plus the id and handle machinery itself.
//!
//! ([`EnvironmentMap`] carries an id but is not refcounted, so it lives at the
//! crate root instead.)
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
    PointMaterialHandle, ResourceKind, SceneResource, TextureHandle,
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
    EffectiveVisibility, Node, NodeFlags, NodeId, NodePayload, Visibility,
};
pub use texture::{Texture, TextureFormat, TextureId};

// Handle plumbing for `SceneData`; not part of the public surface.
pub(crate) use handle::{HandleCore, SceneBind};
