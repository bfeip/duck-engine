use super::material::{FaceMaterialId, LineMaterialId, PointMaterialId};
use super::mesh::MeshId;
use crate::resource_handle::{
    FaceMaterialHandle, LineMaterialHandle, MeshHandle, PointMaterialHandle,
};

/// Unique identifier for a mesh instance
pub type InstanceId = crate::Id<Instance>;

/// An instance referencing a mesh and up to one material per primitive kind.
///
/// A mesh may contain triangle, line, and/or point primitives; each kind is
/// shaded by its own optional material slot. A slot left `None` means that
/// primitive kind is not drawn for this instance.
///
/// The references are owning handles: the mesh and materials stay in the scene
/// while this instance exists.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Instance {
    pub id: InstanceId,
    mesh: MeshHandle,
    face_material: Option<FaceMaterialHandle>,
    line_material: Option<LineMaterialHandle>,
    point_material: Option<PointMaterialHandle>,
}

impl Instance {
    /// Creates a new instance referencing the given mesh, with no materials assigned.
    ///
    /// Assign materials with [`Instance::with_face_material`] and friends.
    pub fn new(mesh: MeshHandle) -> Self {
        Self {
            id: InstanceId::new(),
            mesh,
            face_material: None,
            line_material: None,
            point_material: None,
        }
    }

    /// Sets the face material (chainable).
    pub fn with_face_material(mut self, material: FaceMaterialHandle) -> Self {
        self.face_material = Some(material);
        self
    }

    /// Sets the line material (chainable).
    pub fn with_line_material(mut self, material: LineMaterialHandle) -> Self {
        self.line_material = Some(material);
        self
    }

    /// Sets the point material (chainable).
    pub fn with_point_material(mut self, material: PointMaterialHandle) -> Self {
        self.point_material = Some(material);
        self
    }

    /// Returns the mesh ID referenced by this instance.
    pub fn mesh(&self) -> MeshId {
        self.mesh.id()
    }

    /// Returns the owning handle of the referenced mesh.
    pub fn mesh_handle(&self) -> &MeshHandle {
        &self.mesh
    }

    /// Returns the face material ID, if assigned.
    pub fn face_material(&self) -> Option<FaceMaterialId> {
        self.face_material.as_ref().map(FaceMaterialHandle::id)
    }

    /// Returns the owning handle of the face material, if assigned.
    pub fn face_material_handle(&self) -> Option<&FaceMaterialHandle> {
        self.face_material.as_ref()
    }

    /// Returns the line material ID, if assigned.
    pub fn line_material(&self) -> Option<LineMaterialId> {
        self.line_material.as_ref().map(LineMaterialHandle::id)
    }

    /// Returns the owning handle of the line material, if assigned.
    pub fn line_material_handle(&self) -> Option<&LineMaterialHandle> {
        self.line_material.as_ref()
    }

    /// Returns the point material ID, if assigned.
    pub fn point_material(&self) -> Option<PointMaterialId> {
        self.point_material.as_ref().map(PointMaterialHandle::id)
    }

    /// Returns the owning handle of the point material, if assigned.
    pub fn point_material_handle(&self) -> Option<&PointMaterialHandle> {
        self.point_material.as_ref()
    }

    /// Replaces the referenced mesh.
    pub fn set_mesh(&mut self, mesh: MeshHandle) {
        self.mesh = mesh;
    }

    /// Sets or clears the face material.
    pub fn set_face_material(&mut self, material: Option<FaceMaterialHandle>) {
        self.face_material = material;
    }

    /// Sets or clears the line material.
    pub fn set_line_material(&mut self, material: Option<LineMaterialHandle>) {
        self.line_material = material;
    }

    /// Sets or clears the point material.
    pub fn set_point_material(&mut self, material: Option<PointMaterialHandle>) {
        self.point_material = material;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resource_handle::Handle;

    fn mesh_handle() -> MeshHandle {
        Handle::unbound(MeshId::new())
    }

    fn face_handle() -> FaceMaterialHandle {
        Handle::unbound(FaceMaterialId::new())
    }

    #[test]
    fn test_instance_new() {
        let mesh = mesh_handle();
        let mat = face_handle();
        let instance = Instance::new(mesh.clone()).with_face_material(mat.clone());

        assert_eq!(instance.mesh(), mesh.id());
        assert_eq!(instance.face_material(), Some(mat.id()));
    }

    #[test]
    fn test_instance_id_unique() {
        let instance1 = Instance::new(mesh_handle());
        let instance2 = Instance::new(mesh_handle());
        let instance3 = Instance::new(mesh_handle());

        assert_ne!(instance1.id, instance2.id);
        assert_ne!(instance1.id, instance3.id);
        assert_ne!(instance2.id, instance3.id);
    }

    #[test]
    fn test_instance_mesh_reference() {
        let mesh = mesh_handle();
        let instance = Instance::new(mesh.clone());
        assert_eq!(instance.mesh(), mesh.id());
    }

    #[test]
    fn test_instance_material_slots() {
        let face = face_handle();
        let line: LineMaterialHandle = Handle::unbound(LineMaterialId::new());
        let instance = Instance::new(mesh_handle())
            .with_face_material(face.clone())
            .with_line_material(line.clone());
        assert_eq!(instance.face_material(), Some(face.id()));
        assert_eq!(instance.line_material(), Some(line.id()));
        assert_eq!(instance.point_material(), None);
    }

    #[test]
    fn test_instance_same_mesh_different_material() {
        let mesh = mesh_handle();
        let instance1 = Instance::new(mesh.clone()).with_face_material(face_handle());
        let instance2 = Instance::new(mesh).with_face_material(face_handle());

        assert_eq!(instance1.mesh(), instance2.mesh());
        assert_ne!(instance1.face_material(), instance2.face_material());
    }

    #[test]
    fn test_instance_same_material_different_mesh() {
        let mat = face_handle();
        let instance1 = Instance::new(mesh_handle()).with_face_material(mat.clone());
        let instance2 = Instance::new(mesh_handle()).with_face_material(mat);

        assert_ne!(instance1.mesh(), instance2.mesh());
        assert_eq!(instance1.face_material(), instance2.face_material());
    }
}
