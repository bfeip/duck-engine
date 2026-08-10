//! The owning scene container.
//!
//! [`SceneData`] holds every mesh, material, texture, instance, node and light in
//! a scene, and owns the ids they are addressed by. [`Scene`](crate::Scene) is a
//! shared handle to one; most code goes through that rather than holding a
//! `SceneData` directly.

use duck_engine_common::{Deg, Matrix4, Point3, Quaternion, Rotation3, SquareMatrix, Vector3};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::common::{self, Aabb};
use crate::{
    CameraProjection, EnvironmentMap, EnvironmentMapId, initial_generation, Light, PositionedCamera,
};
use crate::resource::{
    EffectiveVisibility, FaceMaterial, FaceMaterialHandle, FaceMaterialId,
    GenericId, Handle, HandleCore, Id, Instance, InstanceHandle, InstanceId, LineMaterial,
    LineMaterialHandle, LineMaterialId, Mesh, MeshDescriptor, MeshHandle, MeshId, Node, NodeFlags,
    NodeHandle, NodeId, NodePayload, PointMaterial, PointMaterialHandle, PointMaterialId,
    ResourceKind, SceneBind, Texture, TextureHandle, TextureId, Visibility,
};

/// Scene-level properties that affect shader generation.
///
/// Unlike MaterialProperties which describe individual materials,
/// SceneProperties describes scene-wide rendering features.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct SceneProperties {
    /// Whether image-based lighting is active (environment map present)
    pub has_ibl: bool,
}

/// The result of a bounding box computation on an incomplete scene tree.
///
/// `incomplete` being true means at least one node, instance, or mesh referenced
/// in the subtree was not present, as happens while a scene is still being built
/// up. In that case the result is not cached, so the next call will retry and may
/// return a more complete value.
#[derive(Debug, Clone)]
pub struct BoundingResult {
    /// The computed bounds, or `None` if no geometry was reachable.
    pub bounds: Option<Aabb>,
    /// True when the subtree was not fully populated at the time of computation.
    pub incomplete: bool,
}

/// The scene container holding all meshes, materials, textures, instances, nodes, and lights.
///
/// SceneData provides device-free APIs for creating and managing scene objects.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::{common, SceneData};
/// use duck_engine_scene::common::RgbaColor;
/// use duck_engine_scene::resource::{
///     FaceMaterial, Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, Vertex,
/// };
///
/// let mut scene = SceneData::new();
///
/// // Add a mesh (no device needed)
/// let vertices = vec![
///     Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.5, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
/// ];
/// let primitives = vec![MeshPrimitive {
///     primitive_type: PrimitiveType::TriangleList,
///     indices: vec![0, 1, 2],
/// }];
/// let mesh = Mesh::from_raw(vertices, primitives);
/// let mesh = scene.add_mesh(mesh);
///
/// // Add a face material (no device needed)
/// let material = FaceMaterial::new().with_base_color_factor(RgbaColor::RED);
/// let material = scene.add_face_material(material);
///
/// // Create an instance node
/// let instance = Instance::new(mesh).with_face_material(material);
/// let node = scene.add_instance_node(
///     None,
///     instance,
///     None,
///     common::Transform::IDENTITY,
///     NodeFlags::NONE,
/// );
/// ```
pub struct SceneData {
    meshes: HashMap<MeshId, Mesh>,
    instances: HashMap<InstanceId, Instance>,

    // Scene tree
    nodes: HashMap<NodeId, Node>,
    /// Owning references: root nodes are kept alive by this list.
    root_nodes: Vec<NodeHandle>,

    face_materials: HashMap<FaceMaterialId, FaceMaterial>,
    line_materials: HashMap<LineMaterialId, LineMaterial>,
    point_materials: HashMap<PointMaterialId, PointMaterial>,
    textures: HashMap<TextureId, Texture>,

    // Environment maps for IBL
    environment_maps: HashMap<EnvironmentMapId, EnvironmentMap>,
    /// The currently active environment map for IBL lighting.
    active_environment_map: Option<EnvironmentMapId>,

    /// The node (with a Camera payload) that drives rendering when no explicit camera is passed.
    active_camera: Option<NodeId>,

    /// Generation counter that increments on any node add, remove, or mutation.
    /// Used by the renderer to detect when scene data need re-collection.
    node_generation: u64,

    /// Shared with every handle minted from this scene; carries the pending
    /// drop queue. See [`crate::resource`].
    bind: Arc<SceneBind>,
    /// Canonical live handle core per resource, so re-minting a handle for an
    /// id shares the refcount of every existing handle for it.
    slots: HashMap<GenericId, std::sync::Weak<HandleCore>>,
    /// Resources removed since the last [`take_removals`](Self::take_removals);
    /// the renderer drains this to evict GPU resources.
    removals: Vec<(ResourceKind, GenericId)>,
}

impl SceneData {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            instances: HashMap::new(),

            nodes: HashMap::new(),
            root_nodes: Vec::new(),

            face_materials: HashMap::new(),
            line_materials: HashMap::new(),
            point_materials: HashMap::new(),
            textures: HashMap::new(),

            environment_maps: HashMap::new(),
            active_environment_map: None,

            active_camera: None,

            node_generation: initial_generation(),

            bind: Arc::new(SceneBind::new()),
            slots: HashMap::new(),
            removals: Vec::new(),
        }
    }

    // ========== Handles ==========

    /// The bind shared with handles; `Scene::new` points it at the lock.
    pub(crate) fn bind(&self) -> &Arc<SceneBind> {
        &self.bind
    }

    /// The canonical core for `id`, minting one if none is live.
    fn mint_core(&mut self, id: GenericId, kind: ResourceKind) -> Arc<HandleCore> {
        if let Some(weak) = self.slots.get(&id)
            && let Some(core) = weak.upgrade()
        {
            return core;
        }
        let core = Arc::new(HandleCore { id, kind, bind: Arc::downgrade(&self.bind) });
        self.slots.insert(id, Arc::downgrade(&core));
        core
    }

    fn mint<K>(&mut self, id: Id<K>, kind: ResourceKind) -> Handle<K> {
        Handle::from_core(self.mint_core(id.erased(), kind))
    }

    /// An owning handle for an existing mesh.
    pub fn mesh_handle(&mut self, id: MeshId) -> Option<MeshHandle> {
        self.meshes.contains_key(&id).then(|| self.mint(id, ResourceKind::Mesh))
    }

    /// An owning handle for an existing instance.
    pub fn instance_handle(&mut self, id: InstanceId) -> Option<InstanceHandle> {
        self.instances.contains_key(&id).then(|| self.mint(id, ResourceKind::Instance))
    }

    /// An owning handle for an existing node.
    pub fn node_handle(&mut self, id: NodeId) -> Option<NodeHandle> {
        self.nodes.contains_key(&id).then(|| self.mint(id, ResourceKind::Node))
    }

    /// An owning handle for an existing face material.
    pub fn face_material_handle(&mut self, id: FaceMaterialId) -> Option<FaceMaterialHandle> {
        self.face_materials.contains_key(&id).then(|| self.mint(id, ResourceKind::FaceMaterial))
    }

    /// An owning handle for an existing line material.
    pub fn line_material_handle(&mut self, id: LineMaterialId) -> Option<LineMaterialHandle> {
        self.line_materials.contains_key(&id).then(|| self.mint(id, ResourceKind::LineMaterial))
    }

    /// An owning handle for an existing point material.
    pub fn point_material_handle(&mut self, id: PointMaterialId) -> Option<PointMaterialHandle> {
        self.point_materials.contains_key(&id).then(|| self.mint(id, ResourceKind::PointMaterial))
    }

    /// An owning handle for an existing texture.
    pub fn texture_handle(&mut self, id: TextureId) -> Option<TextureHandle> {
        self.textures.contains_key(&id).then(|| self.mint(id, ResourceKind::Texture))
    }

    /// Removes resources whose last handle has dropped, cascading through
    /// the references they owned.
    ///
    /// Runs automatically when the scene lock is taken or released; call it
    /// directly on a standalone `SceneData`.
    pub fn reap(&mut self) {
        let bind = self.bind.clone();
        loop {
            let batch = std::mem::take(
                &mut *bind.dropped.lock().unwrap_or_else(|e| e.into_inner()),
            );
            if batch.is_empty() {
                break;
            }
            for (kind, id) in batch {
                // Revived: a new handle was minted after the drop enqueued.
                if self.slots.get(&id).is_some_and(|w| w.strong_count() > 0) {
                    continue;
                }
                let removed = match kind {
                    ResourceKind::Mesh => self.meshes.remove(&id.cast()).is_some(),
                    ResourceKind::Instance => self.instances.remove(&id.cast()).is_some(),
                    ResourceKind::FaceMaterial => self.face_materials.remove(&id.cast()).is_some(),
                    ResourceKind::LineMaterial => self.line_materials.remove(&id.cast()).is_some(),
                    ResourceKind::PointMaterial => {
                        self.point_materials.remove(&id.cast()).is_some()
                    }
                    ResourceKind::Texture => self.textures.remove(&id.cast()).is_some(),
                    ResourceKind::Node => match self.nodes.remove(&id.cast()) {
                        Some(node) => {
                            // Children still alive through other handles are now
                            // detached roots.
                            for child in node.child_handles() {
                                if let Some(c) = self.nodes.get_mut(&child.id()) {
                                    c.set_parent_unchecked(None);
                                }
                            }
                            if self.active_camera == Some(id.cast()) {
                                self.active_camera = None;
                            }
                            self.node_generation += 1;
                            true
                            // Dropping `node` here releases its children and
                            // payload; the next loop iteration reaps them.
                        }
                        None => false,
                    },
                };
                if removed {
                    self.slots.remove(&id);
                    self.removals.push((kind, id));
                }
            }
        }
    }

    /// Drains the log of resources removed since the last call.
    ///
    /// Renderers use this to evict cached GPU resources. The log has a single
    /// consumer: draining it hands the removals to the caller.
    pub fn take_removals(&mut self) -> Vec<(ResourceKind, GenericId)> {
        std::mem::take(&mut self.removals)
    }

    // ========== Mesh API ==========

    /// Adds a mesh to the scene, returning its owning handle.
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshHandle {
        let id = mesh.id;
        self.meshes.insert(id, mesh);
        self.mint(id, ResourceKind::Mesh)
    }

    /// Creates and adds a mesh from a descriptor.
    pub fn add_mesh_from_descriptor(&mut self, descriptor: MeshDescriptor) -> anyhow::Result<MeshHandle> {
        let mesh = Mesh::from_descriptor(descriptor)?;
        Ok(self.add_mesh(mesh))
    }

    /// Gets a reference to a mesh by ID.
    pub fn get_mesh(&self, id: MeshId) -> Option<&Mesh> {
        self.meshes.get(&id)
    }

    /// Gets a mutable reference to a mesh by ID.
    pub fn get_mesh_mut(&mut self, id: MeshId) -> Option<&mut Mesh> {
        self.meshes.get_mut(&id)
    }

    /// Returns an iterator over all meshes.
    pub fn meshes(&self) -> impl Iterator<Item = &Mesh> {
        self.meshes.values()
    }

    /// Returns a mutable iterator over all meshes.
    pub fn meshes_mut(&mut self) -> impl Iterator<Item = &mut Mesh> {
        self.meshes.values_mut()
    }

    /// Returns the number of meshes in the scene.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Force-removes a mesh from the scene by ID, even while handles for it
    /// exist (they become inert).
    pub fn remove_mesh(&mut self, id: MeshId) {
        if self.meshes.remove(&id).is_some() {
            self.slots.remove(&id.erased());
            self.removals.push((ResourceKind::Mesh, id.erased()));
        }
    }

    // ========== Face material API ==========

    /// Adds a face material to the scene, returning its owning handle.
    pub fn add_face_material(&mut self, material: FaceMaterial) -> FaceMaterialHandle {
        let id = material.id;
        self.face_materials.insert(id, material);
        self.mint(id, ResourceKind::FaceMaterial)
    }

    /// Gets a reference to a face material by ID.
    pub fn get_face_material(&self, id: FaceMaterialId) -> Option<&FaceMaterial> {
        self.face_materials.get(&id)
    }

    /// Gets a mutable reference to a face material by ID.
    pub fn get_face_material_mut(&mut self, id: FaceMaterialId) -> Option<&mut FaceMaterial> {
        self.face_materials.get_mut(&id)
    }

    /// Returns an iterator over all face materials.
    pub fn face_materials(&self) -> impl Iterator<Item = &FaceMaterial> {
        self.face_materials.values()
    }

    /// Returns a mutable iterator over all face materials.
    pub fn face_materials_mut(&mut self) -> impl Iterator<Item = &mut FaceMaterial> {
        self.face_materials.values_mut()
    }

    /// Returns the number of face materials in the scene.
    pub fn face_material_count(&self) -> usize {
        self.face_materials.len()
    }

    /// Force-removes a face material from the scene by ID, even while handles
    /// for it exist (they become inert).
    pub fn remove_face_material(&mut self, id: FaceMaterialId) {
        if self.face_materials.remove(&id).is_some() {
            self.slots.remove(&id.erased());
            self.removals.push((ResourceKind::FaceMaterial, id.erased()));
        }
    }

    // ========== Line material API ==========

    /// Adds a line material to the scene, returning its owning handle.
    pub fn add_line_material(&mut self, material: LineMaterial) -> LineMaterialHandle {
        let id = material.id;
        self.line_materials.insert(id, material);
        self.mint(id, ResourceKind::LineMaterial)
    }

    /// Gets a reference to a line material by ID.
    pub fn get_line_material(&self, id: LineMaterialId) -> Option<&LineMaterial> {
        self.line_materials.get(&id)
    }

    /// Gets a mutable reference to a line material by ID.
    pub fn get_line_material_mut(&mut self, id: LineMaterialId) -> Option<&mut LineMaterial> {
        self.line_materials.get_mut(&id)
    }

    /// Returns an iterator over all line materials.
    pub fn line_materials(&self) -> impl Iterator<Item = &LineMaterial> {
        self.line_materials.values()
    }

    /// Returns a mutable iterator over all line materials.
    pub fn line_materials_mut(&mut self) -> impl Iterator<Item = &mut LineMaterial> {
        self.line_materials.values_mut()
    }

    /// Returns the number of line materials in the scene.
    pub fn line_material_count(&self) -> usize {
        self.line_materials.len()
    }

    /// Force-removes a line material from the scene by ID, even while handles
    /// for it exist (they become inert).
    pub fn remove_line_material(&mut self, id: LineMaterialId) {
        if self.line_materials.remove(&id).is_some() {
            self.slots.remove(&id.erased());
            self.removals.push((ResourceKind::LineMaterial, id.erased()));
        }
    }

    // ========== Point material API ==========

    /// Adds a point material to the scene, returning its owning handle.
    pub fn add_point_material(&mut self, material: PointMaterial) -> PointMaterialHandle {
        let id = material.id;
        self.point_materials.insert(id, material);
        self.mint(id, ResourceKind::PointMaterial)
    }

    /// Gets a reference to a point material by ID.
    pub fn get_point_material(&self, id: PointMaterialId) -> Option<&PointMaterial> {
        self.point_materials.get(&id)
    }

    /// Gets a mutable reference to a point material by ID.
    pub fn get_point_material_mut(&mut self, id: PointMaterialId) -> Option<&mut PointMaterial> {
        self.point_materials.get_mut(&id)
    }

    /// Returns an iterator over all point materials.
    pub fn point_materials(&self) -> impl Iterator<Item = &PointMaterial> {
        self.point_materials.values()
    }

    /// Returns a mutable iterator over all point materials.
    pub fn point_materials_mut(&mut self) -> impl Iterator<Item = &mut PointMaterial> {
        self.point_materials.values_mut()
    }

    /// Returns the number of point materials in the scene.
    pub fn point_material_count(&self) -> usize {
        self.point_materials.len()
    }

    /// Force-removes a point material from the scene by ID, even while handles
    /// for it exist (they become inert).
    pub fn remove_point_material(&mut self, id: PointMaterialId) {
        if self.point_materials.remove(&id).is_some() {
            self.slots.remove(&id.erased());
            self.removals.push((ResourceKind::PointMaterial, id.erased()));
        }
    }

    // ========== Texture API ==========

    /// Adds a texture to the scene, returning its owning handle.
    pub fn add_texture(&mut self, texture: Texture) -> TextureHandle {
        let id = texture.id;
        self.textures.insert(id, texture);
        self.mint(id, ResourceKind::Texture)
    }

    /// Creates and adds a texture from an image.
    pub fn add_texture_from_image(&mut self, image: DynamicImage) -> TextureHandle {
        self.add_texture(Texture::from_image(image))
    }

    /// Creates and adds a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded lazily when first needed.
    pub fn add_texture_from_path(&mut self, path: impl AsRef<Path>) -> TextureHandle {
        self.add_texture(Texture::from_path(path.as_ref()))
    }

    /// Gets a reference to a texture by ID.
    pub fn get_texture(&self, id: TextureId) -> Option<&Texture> {
        self.textures.get(&id)
    }

    /// Gets a mutable reference to a texture by ID.
    pub fn get_texture_mut(&mut self, id: TextureId) -> Option<&mut Texture> {
        self.textures.get_mut(&id)
    }

    /// Returns an iterator over all textures.
    pub fn textures(&self) -> impl Iterator<Item = &Texture> {
        self.textures.values()
    }

    /// Returns a mutable iterator over all textures.
    pub fn textures_mut(&mut self) -> impl Iterator<Item = &mut Texture> {
        self.textures.values_mut()
    }

    /// Returns the number of textures in the scene.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    // ========== Environment Map API (IBL) ==========

    /// Creates and adds an environment map from an equirectangular HDR file path.
    ///
    /// The HDR file will be loaded and processed into IBL maps when first rendered.
    pub fn add_environment_map_from_hdr_path(
        &mut self,
        path: impl Into<std::path::PathBuf>,
    ) -> EnvironmentMapId {
        let env_map = EnvironmentMap::from_hdr_path(path);
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
        id
    }

    /// Creates and adds an environment map from in-memory HDR data.
    ///
    /// The HDR data will be processed into IBL maps when first rendered.
    pub fn add_environment_map_from_hdr_data(
        &mut self,
        data: Vec<u8>,
    ) -> EnvironmentMapId {
        let env_map = EnvironmentMap::from_hdr_data(data);
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
        id
    }

    /// Adds an environment map to the scene using its existing ID.
    pub fn add_environment_map(&mut self, env_map: EnvironmentMap) -> EnvironmentMapId {
        let id = env_map.id;
        self.environment_maps.insert(id, env_map);
        id
    }

    /// Gets a reference to an environment map by ID.
    pub fn get_environment_map(&self, id: EnvironmentMapId) -> Option<&EnvironmentMap> {
        self.environment_maps.get(&id)
    }

    /// Gets a mutable reference to an environment map by ID.
    pub fn get_environment_map_mut(&mut self, id: EnvironmentMapId) -> Option<&mut EnvironmentMap> {
        self.environment_maps.get_mut(&id)
    }

    /// Returns an iterator over all environment maps.
    pub fn environment_maps(&self) -> impl Iterator<Item = &EnvironmentMap> {
        self.environment_maps.values()
    }

    /// Returns a mutable iterator over all environment maps.
    pub fn environment_maps_mut(&mut self) -> impl Iterator<Item = &mut EnvironmentMap> {
        self.environment_maps.values_mut()
    }

    /// Returns the number of environment maps in the scene.
    pub fn environment_map_count(&self) -> usize {
        self.environment_maps.len()
    }

    /// Returns true if the scene has any environment maps.
    pub fn has_environment_maps(&self) -> bool {
        !self.environment_maps.is_empty()
    }

    /// Sets the active environment map for IBL lighting.
    ///
    /// Pass `None` to disable IBL lighting.
    pub fn set_active_environment_map(&mut self, id: Option<EnvironmentMapId>) {
        self.active_environment_map = id;
    }

    /// Gets the currently active environment map ID, if any.
    pub fn active_environment_map(&self) -> Option<EnvironmentMapId> {
        self.active_environment_map
    }

    // ========== Instance API ==========

    /// Adds an instance to the scene, returning its owning handle.
    pub fn add_instance(&mut self, instance: Instance) -> InstanceHandle {
        let id = instance.id;
        self.instances.insert(id, instance);
        self.mint(id, ResourceKind::Instance)
    }

    /// Gets a reference to an instance by ID.
    pub fn get_instance(&self, id: InstanceId) -> Option<&Instance> {
        self.instances.get(&id)
    }

    /// Gets a mutable reference to an instance by ID.
    pub fn get_instance_mut(&mut self, id: InstanceId) -> Option<&mut Instance> {
        self.instances.get_mut(&id)
    }

    /// Returns an iterator over all instances.
    pub fn instances(&self) -> impl Iterator<Item = &Instance> {
        self.instances.values()
    }

    /// Returns a mutable iterator over all instances.
    pub fn instances_mut(&mut self) -> impl Iterator<Item = &mut Instance> {
        self.instances.values_mut()
    }

    /// Returns the number of instances in the scene.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Force-removes an instance from the scene by ID, even while handles for
    /// it exist (they become inert).
    pub fn remove_instance(&mut self, id: InstanceId) {
        if self.instances.remove(&id).is_some() {
            self.slots.remove(&id.erased());
            self.removals.push((ResourceKind::Instance, id.erased()));
        }
    }


    // ========== Active Camera API ==========

    /// Returns the node ID of the active camera, if any.
    pub fn active_camera(&self) -> Option<NodeId> {
        self.active_camera
    }

    /// Returns the [`CameraProjection`] from the active camera node, or `None` if there is no
    /// active camera or the node does not carry a `NodePayload::Camera` payload.
    pub fn active_camera_data(&self) -> Option<&CameraProjection> {
        let node = self.get_node(self.active_camera?)?;
        match node.payload() {
            NodePayload::Camera(cam) => Some(cam),
            _ => None,
        }
    }

    /// Constructs a [`PositionedCamera`] from the active camera node's world transform,
    /// projection payload, and the given viewport aspect ratio.
    ///
    /// Returns `None` if there is no active camera or the node lacks a Camera payload.
    pub fn active_camera_positioned(&self, aspect: f32) -> Option<PositionedCamera> {
        self.positioned_camera_for_node(self.active_camera?, aspect)
    }

    /// Constructs a [`PositionedCamera`] from any node carrying a `NodePayload::Camera`
    /// payload, using its world transform and the given viewport aspect ratio.
    ///
    /// Returns `None` if the node does not exist, lacks a Camera payload, or its
    /// world transform cannot be resolved (incomplete tree).
    pub fn positioned_camera_for_node(&self, node_id: NodeId, aspect: f32) -> Option<PositionedCamera> {
        let node = self.get_node(node_id)?;
        let proj = match node.payload() {
            NodePayload::Camera(cam) => cam.clone(),
            _ => return None,
        };
        let world_transform = self.nodes_transform(node_id)?;
        Some(proj.into_positioned(world_transform, aspect))
    }

    /// Sets the active camera node. Pass `None` to clear.
    pub fn set_active_camera(&mut self, node_id: Option<NodeId>) {
        self.active_camera = node_id;
    }

    // ========== Node Generation ==========

    /// Returns the current node generation counter.
    ///
    /// Increments on every node add, remove, or payload mutation. The renderer
    /// uses this to detect when lights (and other node data) need re-collection.
    pub fn node_generation(&self) -> u64 {
        self.node_generation
    }

    // ========== Light Node Helpers ==========

    /// Returns true if any attached node carries a `Light` payload.
    ///
    /// Detached nodes (kept alive by a handle after
    /// [`remove_node`](Self::remove_node)) are not rendered, so their lights
    /// don't count.
    pub fn has_light_nodes(&self) -> bool {
        self.nodes
            .values()
            .any(|n| matches!(n.payload(), NodePayload::Light(_)) && self.is_node_attached(n.id))
    }

    /// Returns the number of attached nodes with a [`NodePayload::Light`] payload.
    pub fn light_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| matches!(n.payload(), NodePayload::Light(_)) && self.is_node_attached(n.id))
            .count()
    }

    /// Adds a default key + fill directional light pair as children of `camera_node_id`.
    /// The key light comes from the upper-left corner; the fill from the lower-right corner.
    /// Both are expressed in camera space (node direction = its -Z axis).
    pub fn set_default_light_nodes(&mut self, camera_node_id: NodeId) {
        let white = crate::common::RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
        // Upper-left: yaw +45° (toward right), then pitch -45° (downward).
        let key_rotation = Quaternion::from_angle_x(Deg(-45.0)) * Quaternion::from_angle_y(Deg(45.0));
        let key_transform = common::Transform::new(
            Point3::new(0.0, 0.0, 0.0),
            key_rotation,
            Vector3::new(1.0, 1.0, 1.0),
        );
        let key = self
            .add_node(Some(camera_node_id), Some("DefaultKeyLight".to_string()), key_transform, NodeFlags::NONE)
            .expect("Failed to create key light node");
        self.nodes.get_mut(&key.id()).unwrap().set_payload(NodePayload::Light(Light::directional(white, 1.0)));
        self.node_generation += 1;
        // Lower-right: yaw -45° (toward left), then pitch +45° (upward) — opposite corner.
        let fill_rotation = Quaternion::from_angle_x(Deg(45.0)) * Quaternion::from_angle_y(Deg(-45.0));
        let fill_transform = common::Transform::new(
            Point3::new(0.0, 0.0, 0.0),
            fill_rotation,
            Vector3::new(1.0, 1.0, 1.0),
        );
        let fill = self
            .add_node(Some(camera_node_id), Some("DefaultFillLight".to_string()), fill_transform, NodeFlags::NONE)
            .expect("Failed to create fill light node");
        self.nodes.get_mut(&fill.id()).unwrap().set_payload(NodePayload::Light(Light::directional(white, 0.3)));
        self.node_generation += 1;
    }

    // ========== Node API ==========

    /// Gets a reference to a node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Returns an iterator over all nodes.
    pub fn nodes(&self) -> impl Iterator<Item = &Node> {
        self.nodes.values()
    }

    /// Returns the number of nodes in the scene.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if a node with the given ID exists in the scene.
    pub fn has_node(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// Iterates the root node IDs.
    pub fn root_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.root_nodes.iter().map(NodeHandle::id)
    }

    /// Adds a new node to the scene tree, returning its owning handle.
    ///
    /// The tree itself owns attached nodes, so the returned handle may be
    /// dropped freely; keep it for direct access or to own the node across a
    /// [`remove_node`](Self::remove_node).
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        name: Option<String>,
        transform: common::Transform,
        flags: NodeFlags
    ) -> anyhow::Result<NodeHandle> {
        // Validate parent exists if specified
        if let Some(parent_id) = parent
            && !self.nodes.contains_key(&parent_id) {
                anyhow::bail!("Parent node with ID {} not found in scene", parent_id);
            }

        let mut node = Node::new(name, transform, flags);
        let id = node.id;
        node.set_parent_unchecked(parent);
        self.nodes.insert(id, node);
        let handle = self.mint(id, ResourceKind::Node);

        if let Some(parent_id) = parent {
            // Safe to unwrap since we validated parent exists above
            self.nodes.get_mut(&parent_id).unwrap().add_child_unchecked(handle.clone());
        } else {
            self.root_nodes.push(handle.clone());
        }

        self.node_generation += 1;
        Ok(handle)
    }

    /// Adds a new node with an instance attached.
    ///
    /// This is a convenience method that creates both an instance and a node
    /// in one call.
    pub fn add_instance_node(
        &mut self,
        parent: Option<NodeId>,
        instance: Instance,
        name: Option<String>,
        transform: common::Transform,
        flags: NodeFlags
    ) -> anyhow::Result<NodeHandle> {
        // Create the instance
        let instance = self.add_instance(instance);

        // Create the node (validates parent exists)
        let node = self.add_node(parent, name, transform, flags)?;

        // Attach instance to node
        // Safe to unwrap since we just created the node above
        self.nodes.get_mut(&node.id()).unwrap().set_payload(NodePayload::Instance(instance));

        Ok(node)
    }

    /// Adds a node with default transform (identity).
    pub fn add_default_node(&mut self, parent: Option<NodeId>, name: Option<String>) -> anyhow::Result<NodeHandle> {
        self.add_node(parent, name, common::Transform::IDENTITY, NodeFlags::NONE)
    }

    /// Inserts a pre-built node using its existing ID, returning its owning
    /// handle.
    ///
    /// The node is attached under its recorded parent when that parent exists,
    /// or to the roots when it has none. With a recorded parent that is absent
    /// from the scene, the node stays detached: the returned handle is then its
    /// only owner.
    pub fn insert_node(&mut self, node: Node) -> NodeHandle {
        let id = node.id;
        let parent = node.parent();
        self.nodes.insert(id, node);
        let handle = self.mint(id, ResourceKind::Node);
        match parent {
            None => self.root_nodes.push(handle.clone()),
            Some(parent_id) => {
                if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                    parent_node.add_child_unchecked(handle.clone());
                }
            }
        }
        self.node_generation += 1;
        handle
    }

    /// Detaches a node's subtree from the scene tree.
    ///
    /// The tree's ownership of the node is released: unless other handles for
    /// it exist, the node, its descendants, and the resources they exclusively
    /// own are removed. A node kept alive by an outstanding handle survives
    /// detached — still in the scene but not rendered or traversed — and can be
    /// reattached with [`reparent_node`](Self::reparent_node).
    pub fn remove_node(&mut self, node_id: NodeId) {
        let Some(node) = self.nodes.get(&node_id) else {
            return;
        };
        match node.parent() {
            Some(parent_id) => {
                if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                    parent_node.remove_child_unchecked(node_id);
                }
                if let Some(node) = self.nodes.get_mut(&node_id) {
                    node.set_parent_unchecked(None);
                }
                self.invalidate_ancestor_bounds(parent_id);
            }
            None => {
                self.root_nodes.retain(|h| h.id() != node_id);
            }
        }
        self.node_generation += 1;
        self.reap();
    }

    /// Moves a node under a new parent, or to the roots with `None`.
    ///
    /// Also reattaches a node that was detached by
    /// [`remove_node`](Self::remove_node) while a handle kept it alive.
    ///
    /// # Errors
    ///
    /// Fails if either node is missing or the move would create a cycle.
    pub fn reparent_node(&mut self, node_id: NodeId, new_parent: Option<NodeId>) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.nodes.contains_key(&node_id),
            "Node with ID {} not found in scene",
            node_id
        );
        if let Some(parent_id) = new_parent {
            anyhow::ensure!(
                self.nodes.contains_key(&parent_id),
                "Parent node with ID {} not found in scene",
                parent_id
            );
            // The new parent must not be the node itself or one of its
            // descendants; climbing from the parent to the root must not pass
            // through the node.
            let mut current = Some(parent_id);
            while let Some(id) = current {
                anyhow::ensure!(
                    id != node_id,
                    "Reparenting node {} under {} would create a cycle",
                    node_id,
                    parent_id
                );
                current = self.nodes.get(&id).and_then(|n| n.parent());
            }
        }

        // Take the tree's owning handle (or mint one for a detached node).
        let old_parent = self.nodes.get(&node_id).and_then(|n| n.parent());
        let handle = match old_parent {
            Some(parent_id) => self
                .nodes
                .get_mut(&parent_id)
                .and_then(|p| p.take_child_unchecked(node_id)),
            None => self
                .root_nodes
                .iter()
                .position(|h| h.id() == node_id)
                .map(|pos| self.root_nodes.remove(pos)),
        }
        .unwrap_or_else(|| self.mint(node_id, ResourceKind::Node));
        if let Some(parent_id) = old_parent {
            self.invalidate_ancestor_bounds(parent_id);
        }

        self.nodes.get_mut(&node_id).unwrap().set_parent_unchecked(new_parent);
        match new_parent {
            Some(parent_id) => {
                self.nodes.get_mut(&parent_id).unwrap().add_child_unchecked(handle);
                self.invalidate_ancestor_bounds(parent_id);
            }
            None => self.root_nodes.push(handle),
        }
        self.invalidate_subtree_transforms(node_id);
        self.node_generation += 1;
        Ok(())
    }

    /// Returns true if the node exists and its subtree hangs off the scene
    /// roots (as opposed to being detached but kept alive by a handle).
    pub fn is_node_attached(&self, node_id: NodeId) -> bool {
        let mut current = node_id;
        loop {
            let Some(node) = self.nodes.get(&current) else {
                return false;
            };
            match node.parent() {
                Some(parent_id) => current = parent_id,
                None => return self.root_nodes.iter().any(|h| h.id() == current),
            }
        }
    }

    /// Clears all nodes from the scene, resetting it to an empty state.
    ///
    /// Forceful: resources are removed even while handles for them exist;
    /// those handles become inert.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root_nodes.clear();
        self.instances.clear();
        self.meshes.clear();
        self.face_materials.clear();
        self.line_materials.clear();
        self.point_materials.clear();
        self.textures.clear();
        self.environment_maps.clear();
        self.active_environment_map = None;
        self.active_camera = None;
        self.node_generation += 1;
        self.slots.clear();
        self.removals.clear();
        self.bind.dropped.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Invalidates cached bounds for a node and all its ancestors.
    ///
    /// Walks up the parent chain from the given node to the root,
    /// clearing cached bounds on each node. This should be called
    /// when a subtree's bounds change (e.g., after removing nodes).
    fn invalidate_ancestor_bounds(&self, node_id: NodeId) {
        let mut current_id = Some(node_id);

        while let Some(id) = current_id {
            let Some(node) = self.get_node(id) else {
                break;
            };

            node.mark_bounds_dirty();

            // Move to parent
            current_id = node.parent();
        }
    }

    fn invalidate_subtree_transforms(&self, node_id: NodeId) {
        let Some(node) = self.get_node(node_id) else {
            return;
        };
        node.mark_transform_dirty();
        node.mark_bounds_dirty();
        let children: Vec<NodeId> = node.children().collect();
        for child_id in children {
            self.invalidate_subtree_transforms(child_id);
        }
    }

    // ========== Node Transform Mutation API ==========

    /// Sets the payload of a node and invalidates ancestor bounds.
    /// Does nothing if the node does not exist.
    pub fn set_node_payload(&mut self, node_id: NodeId, payload: NodePayload) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_payload(payload);
        self.node_generation += 1;
        self.invalidate_ancestor_bounds(node_id);
    }

    /// Sets the transform of a node and invalidates dependent caches.
    /// Does nothing if the node does not exist.
    pub fn set_node_transform(&mut self, node_id: NodeId, transform: common::Transform) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_transform(transform);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the position of a node and invalidates ancestor bounds.
    /// Does nothing if the node does not exist.
    pub fn set_node_position(&mut self, node_id: NodeId, position: Point3) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_position(position);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the rotation of a node and invalidates ancestor bounds.
    /// Does nothing if the node does not exist.
    pub fn set_node_rotation(&mut self, node_id: NodeId, rotation: Quaternion) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_rotation(rotation);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the scale of a node and invalidates ancestor bounds.
    /// Does nothing if the node does not exist.
    pub fn set_node_scale(&mut self, node_id: NodeId, scale: Vector3) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_scale(scale);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets a node's render-presentation behavior (layer, screen-space flags).
    ///
    /// The behavior inherits down the subtree and is resolved by the renderer,
    /// so no cache invalidation beyond bumping the node generation is needed.
    /// Does nothing if the node does not exist.
    pub fn set_node_display(&mut self, node_id: NodeId, display: crate::resource::DisplayBehavior) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_display(display);
        self.node_generation += 1;
    }

    /// Sets a node's name. Does nothing if the node does not exist.
    pub fn set_node_name(&mut self, node_id: NodeId, name: Option<String>) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.name = name;
        self.node_generation += 1;
    }

    /// Sets a node's behavior flags and invalidates dependent bounds.
    /// Does nothing if the node does not exist.
    pub fn set_node_flags(&mut self, node_id: NodeId, flags: NodeFlags) {
        let Some(node) = self.nodes.get_mut(&node_id) else { return };
        node.set_flags(flags);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    // ========== Visibility API ==========

    /// Sets the visibility of a node and propagates invisibility to descendants.
    ///
    /// When setting to Invisible, all descendants are also marked invisible.
    /// When setting to Visible, only this node is changed.
    pub fn set_node_visibility(&mut self, node_id: NodeId, visibility: Visibility) {
        match visibility {
            Visibility::Visible => {
                let Some(node) = self.nodes.get_mut(&node_id) else { return };
                node.set_visibility(Visibility::Visible);
                self.invalidate_ancestor_effective_visibility(node_id);
            }
            Visibility::Invisible => {
                self.set_subtree_visibility_recursive(node_id, Visibility::Invisible);
                if let Some(node) = self.get_node(node_id)
                    && let Some(parent_id) = node.parent() {
                        self.invalidate_ancestor_effective_visibility(parent_id);
                    }
            }
        }
    }

    /// Recursively sets visibility for a node and all descendants.
    fn set_subtree_visibility_recursive(&mut self, node_id: NodeId, visibility: Visibility) {
        let Some(node) = self.nodes.get_mut(&node_id) else {
            return;
        };
        node.set_visibility(visibility);
        let children: Vec<NodeId> = node.children().collect();

        for child_id in children {
            self.set_subtree_visibility_recursive(child_id, visibility);
        }
    }

    /// Invalidates effective visibility cache for ancestors.
    fn invalidate_ancestor_effective_visibility(&self, node_id: NodeId) {
        let mut current_id = Some(node_id);
        while let Some(id) = current_id {
            let Some(node) = self.get_node(id) else {
                break;
            };
            node.mark_visibility_dirty();
            current_id = node.parent();
        }
    }

    /// Gets the effective visibility of a node with caching.
    /// A node that does not exist is `Invisible`.
    pub fn node_effective_visibility(
        &self,
        node_id: NodeId,
    ) -> EffectiveVisibility {
        let Some(node) = self.get_node(node_id) else {
            return EffectiveVisibility::Invisible;
        };

        if let Some(cached) = node.cached_effective_visibility() {
            return cached;
        }

        let effective = self.compute_effective_visibility_recursive(node_id);
        node.set_cached_effective_visibility(effective);
        effective
    }

    /// Recursively computes effective visibility.
    fn compute_effective_visibility_recursive(
        &self,
        node_id: NodeId,
    ) -> EffectiveVisibility {
        let Some(node) = self.get_node(node_id) else {
            return EffectiveVisibility::Invisible;
        };

        if node.visibility() == Visibility::Invisible {
            return EffectiveVisibility::Invisible;
        }

        let mut all_visible = true;
        for child_id in node.children() {
            let child_effective = self.node_effective_visibility(child_id);
            if child_effective != EffectiveVisibility::Visible {
                all_visible = false;
                break;
            }
        }

        if all_visible {
            EffectiveVisibility::Visible
        } else {
            EffectiveVisibility::Mixed
        }
    }

    // ========== Transform & Bounding Box API ==========

    /// Gets the world transform of a node.
    ///
    /// This returns the cached transform if valid, otherwise computes it by
    /// walking from the root to the node, computing and caching transforms
    /// along the way.
    ///
    /// Returns `None` if the node itself or any ancestor is missing from the
    /// scene, as happens while a scene is still being built up.
    pub fn nodes_transform(&self, node_id: NodeId) -> Option<Matrix4> {
        let node = self.get_node(node_id)?;

        // If cached and valid, return it
        if let Some(cached) = node.cached_world_transform() {
            return Some(cached);
        }

        // Need to compute: build path from root to node.
        // If any ancestor is missing, return None.
        let mut path = Vec::new();
        let mut current_id = node_id;

        loop {
            path.push(current_id);
            let current = self.get_node(current_id)?;
            if let Some(parent_id) = current.parent() {
                current_id = parent_id;
            } else {
                break; // Reached root
            }
        }

        // Reverse to get root-to-node path
        path.reverse();

        // Walk down the path, computing and caching transforms
        let mut world_transform = Matrix4::identity();

        for &id in &path {
            let node = self.get_node(id)?;

            if let Some(cached) = node.cached_world_transform() {
                world_transform = cached;
            } else {
                let local_transform = node.compute_local_transform();
                world_transform = world_transform * local_transform;
                node.set_cached_world_transform(world_transform);
            }
        }

        Some(world_transform)
    }

    /// Gets the world-space bounding box of the entire scene.
    ///
    /// Merges the bounding results of all root nodes. `incomplete` is true if any
    /// part of the scene tree was missing at the time of computation.
    pub fn bounding(&self) -> BoundingResult {
        let mut merged_bounds: Option<Aabb> = None;
        let mut incomplete = false;

        for root_id in self.root_nodes.iter().map(NodeHandle::id) {
            let result = self.nodes_bounding(root_id);
            if result.incomplete {
                incomplete = true;
            }
            if let Some(b) = result.bounds {
                merged_bounds = Some(match merged_bounds {
                    Some(existing) => existing.merge(&b),
                    None => b,
                });
            }
        }

        BoundingResult { bounds: merged_bounds, incomplete }
    }

    /// Gets the world-space bounding box of a node and its subtree.
    ///
    /// This returns the cached bounds if valid, otherwise recursively computes
    /// them bottom-up for the entire subtree rooted at this node.
    ///
    /// The bounds include both the node's instance (if any) and all descendants.
    /// Check `BoundingResult::incomplete` to know if missing resources prevented
    /// a full computation.
    pub fn nodes_bounding(&self, node_id: NodeId) -> BoundingResult {
        let Some(node) = self.get_node(node_id) else {
            return BoundingResult { bounds: None, incomplete: true };
        };

        if node.flags().contains(NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING) {
            return BoundingResult { bounds: None, incomplete: false };
        }

        // Cached value was only ever written for a complete subtree.
        if !node.bounds_dirty() {
            return BoundingResult { bounds: node.cached_bounds(), incomplete: false };
        }

        let mut incomplete = false;
        let mut merged_bounds: Option<Aabb> = None;

        for child_id in node.children() {
            if self.get_node(child_id).is_none() {
                // Child listed in the tree but not yet in the scene.
                incomplete = true;
                continue;
            }
            let child = self.nodes_bounding(child_id);
            if child.incomplete {
                incomplete = true;
            }
            if let Some(b) = child.bounds {
                merged_bounds = Some(match merged_bounds {
                    Some(existing) => existing.merge(&b),
                    None => b,
                });
            }
        }

        let bounds = match node.payload() {
            NodePayload::Instance(instance) => {
                let Some(world_transform) = self.nodes_transform(node_id) else {
                    incomplete = true;
                    return BoundingResult { bounds: merged_bounds, incomplete };
                };
                let Some(instance) = self.get_instance(instance.id()) else {
                    incomplete = true;
                    return BoundingResult { bounds: merged_bounds, incomplete };
                };
                let Some(mesh) = self.get_mesh(instance.mesh()) else {
                    incomplete = true;
                    return BoundingResult { bounds: merged_bounds, incomplete };
                };
                let world_bounds = mesh.bounding().map(|b| b.transform(&world_transform));
                match (world_bounds, merged_bounds) {
                    (Some(wb), Some(cb)) => Some(wb.merge(&cb)),
                    (Some(wb), None) => Some(wb),
                    (None, cb) => cb,
                }
            }
            _ => merged_bounds,
        };

        // Only cache when the subtree is fully populated.
        if !incomplete {
            node.set_cached_bounds(bounds);
        }

        BoundingResult { bounds, incomplete }
    }

    /// Returns `true` if the scene contains no dangling ID references.
    ///
    /// Specifically checks that:
    /// - Every node ID listed in any node's `children` array exists in the scene.
    /// - Every `Instance` payload references an instance that exists.
    /// - Every instance's mesh and material IDs exist.
    ///
    /// A scene under construction — mid-import, or between the steps of an
    /// incremental rebuild — is expected to fail this; use it to assert that a
    /// scene has settled.
    pub fn is_complete(&self) -> bool {
        for node in self.nodes.values() {
            for child_id in node.children() {
                if !self.nodes.contains_key(&child_id) {
                    return false;
                }
            }
            if let NodePayload::Instance(instance) = node.payload() {
                let Some(instance) = self.instances.get(&instance.id()) else {
                    return false;
                };
                if !self.meshes.contains_key(&instance.mesh()) {
                    return false;
                }
                if let Some(id) = instance.face_material()
                    && !self.face_materials.contains_key(&id)
                {
                    return false;
                }
                if let Some(id) = instance.line_material()
                    && !self.line_materials.contains_key(&id)
                {
                    return false;
                }
                if let Some(id) = instance.point_material()
                    && !self.point_materials.contains_key(&id)
                {
                    return false;
                }
            }
        }
        true
    }

    /// Rebuilds handle ownership from the tree structure.
    ///
    /// For scene data whose handles do not reflect ownership — a clone (its
    /// handles share refcounts with the original) or deserialized data (its
    /// handles are unbound): every stored handle is replaced with a canonical
    /// one on a fresh bind, and resources nothing owns are dropped. Existing
    /// external handles into this data are disconnected.
    pub fn rebind(&mut self) {
        use std::collections::HashSet;

        self.bind = Arc::new(SceneBind::new());
        self.slots.clear();
        self.removals.clear();

        // Mark: walk the ownership edges from the roots.
        let mut live_nodes: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<NodeId> = self.root_nodes.iter().map(NodeHandle::id).collect();
        while let Some(id) = stack.pop() {
            if self.nodes.contains_key(&id) && live_nodes.insert(id) {
                stack.extend(self.nodes[&id].children());
            }
        }
        let mut live_instances: HashSet<InstanceId> = HashSet::new();
        for id in &live_nodes {
            if let NodePayload::Instance(h) = self.nodes[id].payload() {
                live_instances.insert(h.id());
            }
        }
        let mut live_meshes: HashSet<MeshId> = HashSet::new();
        let mut live_face: HashSet<FaceMaterialId> = HashSet::new();
        let mut live_line: HashSet<LineMaterialId> = HashSet::new();
        let mut live_point: HashSet<PointMaterialId> = HashSet::new();
        for id in &live_instances {
            if let Some(inst) = self.instances.get(id) {
                live_meshes.insert(inst.mesh());
                live_face.extend(inst.face_material());
                live_line.extend(inst.line_material());
                live_point.extend(inst.point_material());
            }
        }
        let mut live_textures: HashSet<TextureId> = HashSet::new();
        for id in &live_face {
            if let Some(m) = self.face_materials.get(id) {
                live_textures.extend(m.base_color_texture());
                live_textures.extend(m.normal_texture());
                live_textures.extend(m.metallic_roughness_texture());
            }
        }
        for id in &live_line {
            if let Some(m) = self.line_materials.get(id) {
                live_textures.extend(m.base_color_texture());
            }
        }
        for id in &live_point {
            if let Some(m) = self.point_materials.get(id) {
                live_textures.extend(m.base_color_texture());
            }
        }

        // Sweep what nothing owns.
        self.nodes.retain(|id, _| live_nodes.contains(id));
        self.instances.retain(|id, _| live_instances.contains(id));
        self.meshes.retain(|id, _| live_meshes.contains(id));
        self.face_materials.retain(|id, _| live_face.contains(id));
        self.line_materials.retain(|id, _| live_line.contains(id));
        self.point_materials.retain(|id, _| live_point.contains(id));
        self.textures.retain(|id, _| live_textures.contains(id));
        if let Some(id) = self.active_camera
            && !self.nodes.contains_key(&id)
        {
            self.active_camera = None;
        }

        // Re-point every stored handle at a canonical core on the fresh bind.
        let bind = self.bind.clone();
        let mut cores: HashMap<GenericId, Arc<HandleCore>> = HashMap::new();
        fn canon<K>(
            cores: &mut HashMap<GenericId, Arc<HandleCore>>,
            bind: &Arc<SceneBind>,
            id: Id<K>,
            kind: ResourceKind,
        ) -> Handle<K> {
            let core = cores
                .entry(id.erased())
                .or_insert_with(|| {
                    Arc::new(HandleCore { id: id.erased(), kind, bind: Arc::downgrade(bind) })
                })
                .clone();
            Handle::from_core(core)
        }

        for inst in self.instances.values_mut() {
            inst.set_mesh(canon(&mut cores, &bind, inst.mesh(), ResourceKind::Mesh));
            if let Some(id) = inst.face_material() {
                inst.set_face_material(Some(canon(&mut cores, &bind, id, ResourceKind::FaceMaterial)));
            }
            if let Some(id) = inst.line_material() {
                inst.set_line_material(Some(canon(&mut cores, &bind, id, ResourceKind::LineMaterial)));
            }
            if let Some(id) = inst.point_material() {
                inst.set_point_material(Some(canon(&mut cores, &bind, id, ResourceKind::PointMaterial)));
            }
        }
        for mat in self.face_materials.values_mut() {
            if let Some(id) = mat.base_color_texture() {
                mat.set_base_color_texture(canon(&mut cores, &bind, id, ResourceKind::Texture));
            }
            if let Some(id) = mat.normal_texture() {
                mat.set_normal_texture(canon(&mut cores, &bind, id, ResourceKind::Texture));
            }
            if let Some(id) = mat.metallic_roughness_texture() {
                mat.set_metallic_roughness_texture(canon(&mut cores, &bind, id, ResourceKind::Texture));
            }
        }
        for mat in self.line_materials.values_mut() {
            if let Some(id) = mat.base_color_texture() {
                mat.set_base_color_texture(canon(&mut cores, &bind, id, ResourceKind::Texture));
            }
        }
        for mat in self.point_materials.values_mut() {
            if let Some(id) = mat.base_color_texture() {
                mat.set_base_color_texture(canon(&mut cores, &bind, id, ResourceKind::Texture));
            }
        }
        for node in self.nodes.values_mut() {
            let instance_id = match node.payload() {
                NodePayload::Instance(h) => Some(h.id()),
                _ => None,
            };
            if let Some(id) = instance_id {
                node.set_payload(NodePayload::Instance(canon(
                    &mut cores,
                    &bind,
                    id,
                    ResourceKind::Instance,
                )));
            }
            let children: Vec<NodeHandle> = node
                .children()
                .map(|id| canon(&mut cores, &bind, id, ResourceKind::Node))
                .collect();
            node.set_children_unchecked(children);
        }
        let root_ids: Vec<NodeId> = self.root_nodes.iter().map(NodeHandle::id).collect();
        self.root_nodes = root_ids
            .into_iter()
            .map(|id| canon(&mut cores, &bind, id, ResourceKind::Node))
            .collect();

        self.slots = cores.iter().map(|(id, core)| (*id, Arc::downgrade(core))).collect();
    }
}

impl Clone for SceneData {
    /// Clones the scene contents into an independent scene.
    ///
    /// Handle ownership is rebuilt from the tree via
    /// [`rebind`](SceneData::rebind), so the clone shares no refcounts with the
    /// original; resources kept alive in the original only by external handles
    /// are not carried over.
    fn clone(&self) -> Self {
        let mut clone = Self {
            meshes: self.meshes.clone(),
            instances: self.instances.clone(),
            nodes: self.nodes.clone(),
            root_nodes: self.root_nodes.clone(),
            face_materials: self.face_materials.clone(),
            line_materials: self.line_materials.clone(),
            point_materials: self.point_materials.clone(),
            textures: self.textures.clone(),
            environment_maps: self.environment_maps.clone(),
            active_environment_map: self.active_environment_map,
            active_camera: self.active_camera,
            node_generation: self.node_generation,
            bind: Arc::new(SceneBind::new()),
            slots: HashMap::new(),
            removals: Vec::new(),
        };
        clone.rebind();
        clone
    }
}

#[cfg(test)]
mod scene_tests;
