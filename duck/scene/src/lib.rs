
pub use duck_engine_common as common;

// Scene submodules
mod id;
#[cfg(feature = "cad")]
pub mod cad;
mod camera;
mod display;
mod environment;
pub mod geom_query;
mod instance;
mod coordinate_space;
mod light;
mod material;
mod mesh;
mod node;
mod sub_view;
mod texture;
mod view;

use duck_engine_common::{Deg, Matrix4, Point3, Quaternion, Rotation3, SquareMatrix, Vector3};
use image::DynamicImage;
use std::collections::HashMap;
use std::path::Path;

// ID types
pub use id::{GenericId, Id};
pub use environment::EnvironmentMapId;
pub use instance::InstanceId;
pub use material::{FaceMaterialId, LineMaterialId, PointMaterialId};
pub use mesh::MeshId;
pub use node::NodeId;
pub use texture::TextureId;

pub use camera::{CameraProjection, PositionedCamera};
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
pub use sub_view::{SubView, SubViewId, ViewportRect};
pub use texture::{Texture, TextureFormat};
pub use environment::{EnvironmentMap, EnvironmentSource};

use crate::{common::Aabb};

/// Default generation counter value for newly created resources.
/// Starts at 1 so initial change detection triggers on first use.
pub(crate) fn initial_generation() -> u64 {
    1
}

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
/// Scene provides device-free APIs for creating and managing scene objects.
///
/// # Examples
///
/// ```
/// use duck_engine_scene::{
///     common, FaceMaterial, Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, Scene, Vertex,
/// };
/// use duck_engine_scene::common::RgbaColor;
///
/// let mut scene = Scene::new();
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
/// let mesh_id = scene.add_mesh(mesh);
///
/// // Add a face material (no device needed)
/// let material = FaceMaterial::new().with_base_color_factor(RgbaColor::RED);
/// let mat_id = scene.add_face_material(material);
///
/// // Create an instance node
/// let instance = Instance::new(mesh_id).with_face_material(mat_id);
/// let node_id = scene.add_instance_node(
///     None,
///     instance,
///     None,
///     common::Transform::IDENTITY,
///     NodeFlags::NONE,
/// );
/// ```
#[derive(Clone)]
pub struct Scene {
    meshes: HashMap<MeshId, Mesh>,
    instances: HashMap<InstanceId, Instance>,

    // Scene tree
    nodes: HashMap<NodeId, Node>,
    root_nodes: Vec<NodeId>,

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

    /// Sub-views: sub-regions of the surface that render a subtree through their own camera.
    sub_views: HashMap<SubViewId, SubView>,

    /// Generation counter that increments on any node add, remove, or mutation.
    /// Used by the renderer to detect when scene data need re-collection.
    node_generation: u64,
}

impl Scene {
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

            sub_views: HashMap::new(),

            node_generation: initial_generation(),
        }
    }

    // ========== Mesh API ==========

    /// Adds a mesh to the scene.
    pub fn add_mesh(&mut self, mesh: Mesh) -> MeshId {
        let id = mesh.id;
        self.meshes.insert(id, mesh);
        id
    }

    /// Creates and adds a mesh from a descriptor.
    pub fn add_mesh_from_descriptor(&mut self, descriptor: MeshDescriptor) -> anyhow::Result<MeshId> {
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

    /// Removes a mesh from the scene by ID.
    pub fn remove_mesh(&mut self, id: MeshId) {
        self.meshes.remove(&id);
    }

    /// Returns true if no instance references this mesh, i.e. removing it would
    /// orphan nothing.
    pub fn is_mesh_orphaned(&self, id: MeshId) -> bool {
        !self.instances.values().any(|inst| inst.mesh() == id)
    }

    // ========== Face material API ==========

    /// Adds a face material to the scene.
    pub fn add_face_material(&mut self, material: FaceMaterial) -> FaceMaterialId {
        let id = material.id;
        self.face_materials.insert(id, material);
        id
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

    /// Removes a face material from the scene by ID.
    pub fn remove_face_material(&mut self, id: FaceMaterialId) {
        self.face_materials.remove(&id);
    }

    /// Returns true if no instance references this face material.
    pub fn is_face_material_orphaned(&self, id: FaceMaterialId) -> bool {
        !self.instances.values().any(|inst| inst.face_material() == Some(id))
    }

    // ========== Line material API ==========

    /// Adds a line material to the scene.
    pub fn add_line_material(&mut self, material: LineMaterial) -> LineMaterialId {
        let id = material.id;
        self.line_materials.insert(id, material);
        id
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

    /// Removes a line material from the scene by ID.
    pub fn remove_line_material(&mut self, id: LineMaterialId) {
        self.line_materials.remove(&id);
    }

    /// Returns true if no instance references this line material.
    pub fn is_line_material_orphaned(&self, id: LineMaterialId) -> bool {
        !self.instances.values().any(|inst| inst.line_material() == Some(id))
    }

    // ========== Point material API ==========

    /// Adds a point material to the scene.
    pub fn add_point_material(&mut self, material: PointMaterial) -> PointMaterialId {
        let id = material.id;
        self.point_materials.insert(id, material);
        id
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

    /// Removes a point material from the scene by ID.
    pub fn remove_point_material(&mut self, id: PointMaterialId) {
        self.point_materials.remove(&id);
    }

    /// Returns true if no instance references this point material.
    pub fn is_point_material_orphaned(&self, id: PointMaterialId) -> bool {
        !self.instances.values().any(|inst| inst.point_material() == Some(id))
    }

    // ========== Texture API ==========

    /// Adds a texture to the scene.
    pub fn add_texture(&mut self, texture: Texture) -> TextureId {
        let id = texture.id;
        self.textures.insert(id, texture);
        id
    }

    /// Creates and adds a texture from an image.
    pub fn add_texture_from_image(&mut self, image: DynamicImage) -> TextureId {
        self.add_texture(Texture::from_image(image))
    }

    /// Creates and adds a texture from a file path.
    ///
    /// The image is not loaded immediately - it will be loaded lazily when first needed.
    pub fn add_texture_from_path(&mut self, path: impl AsRef<Path>) -> TextureId {
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

    /// Returns true if no material references this texture.
    pub fn is_texture_orphaned(&self, id: TextureId) -> bool {
        let in_face = self.face_materials.values().any(|m| {
            m.base_color_texture() == Some(id)
                || m.normal_texture() == Some(id)
                || m.metallic_roughness_texture() == Some(id)
        });
        let in_line = self.line_materials.values().any(|m| m.base_color_texture() == Some(id));
        let in_point = self.point_materials.values().any(|m| m.base_color_texture() == Some(id));
        !(in_face || in_line || in_point)
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

    /// Adds an instance to the scene, binding a mesh to a material.
    pub fn add_instance(&mut self, instance: Instance) -> InstanceId {
        let id = instance.id;
        self.instances.insert(id, instance);
        id
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

    /// Removes an instance from the scene by ID.
    pub fn remove_instance(&mut self, id: InstanceId) {
        self.instances.remove(&id);
    }

    /// Returns true if no node references this instance.
    pub fn is_instance_orphaned(&self, id: InstanceId) -> bool {
        !self
            .nodes
            .values()
            .any(|node| matches!(node.payload(), NodePayload::Instance(i) if *i == id))
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

    // ========== Sub-View API ==========

    /// Adds a sub-view that renders the subtree rooted at `root` through the camera
    /// node `camera`, confined to the region `rect` (a proportion of the surface).
    /// Returns the new [`SubViewId`].
    pub fn add_sub_view(&mut self, rect: ViewportRect, root: NodeId, camera: NodeId) -> SubViewId {
        let sub_view = SubView::new(rect, root, camera);
        let id = sub_view.id;
        self.sub_views.insert(id, sub_view);
        id
    }

    /// Gets a reference to a sub-view by ID.
    pub fn get_sub_view(&self, id: SubViewId) -> Option<&SubView> {
        self.sub_views.get(&id)
    }

    /// Returns an iterator over all sub-views.
    pub fn sub_views(&self) -> impl Iterator<Item = &SubView> {
        self.sub_views.values()
    }

    /// Returns the number of sub-views in the scene.
    pub fn sub_view_count(&self) -> usize {
        self.sub_views.len()
    }

    /// Updates the viewport rectangle of an existing sub-view. No-op if absent.
    pub fn set_sub_view_rect(&mut self, id: SubViewId, rect: ViewportRect) {
        if let Some(sv) = self.sub_views.get_mut(&id) {
            sv.rect = rect;
        }
    }

    /// Removes a sub-view by ID.
    pub fn remove_sub_view(&mut self, id: SubViewId) {
        self.sub_views.remove(&id);
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

    /// Returns true if any node in the scene carries a `Light` payload.
    pub fn has_light_nodes(&self) -> bool {
        self.nodes.values().any(|n| matches!(n.payload(), NodePayload::Light(_)))
    }

    /// Returns the number of nodes with a [`NodePayload::Light`] payload.
    pub fn light_count(&self) -> usize {
        self.nodes.values().filter(|n| matches!(n.payload(), NodePayload::Light(_))).count()
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
        let key_id = self
            .add_node(Some(camera_node_id), Some("DefaultKeyLight".to_string()), key_transform, NodeFlags::NONE)
            .expect("Failed to create key light node");
        self.nodes.get_mut(&key_id).unwrap().set_payload(NodePayload::Light(Light::directional(white, 1.0)));
        self.node_generation += 1;
        // Lower-right: yaw -45° (toward left), then pitch +45° (upward) — opposite corner.
        let fill_rotation = Quaternion::from_angle_x(Deg(45.0)) * Quaternion::from_angle_y(Deg(-45.0));
        let fill_transform = common::Transform::new(
            Point3::new(0.0, 0.0, 0.0),
            fill_rotation,
            Vector3::new(1.0, 1.0, 1.0),
        );
        let fill_id = self
            .add_node(Some(camera_node_id), Some("DefaultFillLight".to_string()), fill_transform, NodeFlags::NONE)
            .expect("Failed to create fill light node");
        self.nodes.get_mut(&fill_id).unwrap().set_payload(NodePayload::Light(Light::directional(white, 0.3)));
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

    /// Returns a slice of root node IDs.
    pub fn root_nodes(&self) -> &[NodeId] {
        &self.root_nodes
    }

    /// Adds a new node to the scene tree.
    pub fn add_node(
        &mut self,
        parent: Option<NodeId>,
        name: Option<String>,
        transform: common::Transform,
        flags: NodeFlags
    ) -> anyhow::Result<NodeId> {
        // Validate parent exists if specified
        if let Some(parent_id) = parent
            && !self.nodes.contains_key(&parent_id) {
                anyhow::bail!("Parent node with ID {} not found in scene", parent_id);
            }

        let mut node = Node::new(name, transform, flags);
        let id = node.id;

        // Set up parent-child relationship
        if let Some(parent_id) = parent {
            node.set_parent_unchecked(Some(parent_id));
            // Safe to unwrap since we validated parent exists above
            self.nodes.get_mut(&parent_id).unwrap().add_child_unchecked(id);
        } else {
            // No parent, so this is a root node
            self.root_nodes.push(id);
        }

        self.nodes.insert(id, node);
        self.node_generation += 1;
        Ok(id)
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
    ) -> anyhow::Result<NodeId> {
        // Create the instance
        let instance_id = self.add_instance(instance);

        // Create the node (validates parent exists)
        let node_id = self.add_node(parent, name, transform, flags)?;

        // Attach instance to node
        // Safe to unwrap since we just created the node above
        self.nodes.get_mut(&node_id).unwrap().set_payload(NodePayload::Instance(instance_id));

        Ok(node_id)
    }

    /// Adds a node with default transform (identity).
    pub fn add_default_node(&mut self, parent: Option<NodeId>, name: Option<String>) -> anyhow::Result<NodeId> {
        self.add_node(parent, name, common::Transform::IDENTITY, NodeFlags::NONE)
    }

    /// Inserts a pre-built node using its existing ID.
    ///
    /// Automatically appends to `root_nodes` when the node has no parent.
    /// Parent/child links are the caller's responsibility (e.g. the node carries
    /// its children list from deserialization).
    pub fn insert_node(&mut self, node: Node) {
        if node.parent().is_none() {
            self.root_nodes.push(node.id);
        }
        self.nodes.insert(node.id, node);
        self.node_generation += 1;
    }

    /// Removes a node and all its children from the scene tree.
    ///
    /// This recursively removes all descendant nodes and cleans up
    /// parent-child relationships. Cached bounds in all ancestor nodes
    /// are invalidated since the removed subtree affects their bounds.
    pub fn remove_node(&mut self, node_id: NodeId) {
        // Store the parent before removal so we can invalidate ancestors
        let parent = self.nodes.get(&node_id).and_then(|node| node.parent());

        // Perform the recursive removal
        self.remove_node_recursive(node_id);

        // Invalidate cached bounds for all ancestors
        if let Some(parent_id) = parent {
            self.invalidate_ancestor_bounds(parent_id);
        }
    }

    /// Removes a node along with the resources it owns.
    /// 
    /// Removes this Node's instance, that instance's mesh, and its
    /// face/line/point materials — but only those left orphaned afterwards,
    /// so resources still shared by other instances survive.
    ///
    /// Use this (rather than [`remove_node`](Self::remove_node), which only unlinks
    /// the node) to fully discard self-contained geometry such as a transient
    /// preview, avoiding orphaned meshes and materials.
    pub fn cleanup_node(&mut self, node_id: NodeId) {
        let instance_id = match self.get_node(node_id).map(|n| n.payload()) {
            Some(NodePayload::Instance(id)) => Some(*id),
            _ => None,
        };

        self.remove_node(node_id);

        let Some(instance_id) = instance_id else { return };
        let Some((mesh_id, face_mat, line_mat, point_mat)) = self
            .get_instance(instance_id)
            .map(|i| (i.mesh(), i.face_material(), i.line_material(), i.point_material()))
        else {
            return;
        };

        if self.is_instance_orphaned(instance_id) {
            self.remove_instance(instance_id);
        }
        if self.is_mesh_orphaned(mesh_id) {
            self.remove_mesh(mesh_id);
        }
        if let Some(id) = face_mat {
            if self.is_face_material_orphaned(id) {
                self.remove_face_material(id);
            }
        }
        if let Some(id) = line_mat {
            if self.is_line_material_orphaned(id) {
                self.remove_line_material(id);
            }
        }
        if let Some(id) = point_mat {
            if self.is_point_material_orphaned(id) {
                self.remove_point_material(id);
            }
        }
    }

    /// Recursive helper for removing a node and all its children.
    ///
    /// This does NOT invalidate ancestor bounds. The caller is responsible
    /// for invalidating bounds after the entire removal is complete.
    fn remove_node_recursive(&mut self, node_id: NodeId) {
        // Get the node to find its parent and children
        let Some(node) = self.nodes.get(&node_id) else {
            return; // Node doesn't exist
        };

        let parent = node.parent();
        let children: Vec<NodeId> = node.children().to_vec();

        // Recursively remove all children first
        for child_id in children {
            self.remove_node_recursive(child_id);
        }

        // Remove this node from its parent's children list
        if let Some(parent_id) = parent {
            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                parent_node.remove_child_unchecked(node_id);
            }
        } else {
            // This is a root node, remove from root_nodes list
            self.root_nodes.retain(|&id| id != node_id);
        }

        // Finally, remove the node itself
        self.nodes.remove(&node_id);
        self.node_generation += 1;
    }

    /// Clears all nodes from the scene, resetting it to an empty state.
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
        self.sub_views.clear();
        self.node_generation += 1;
    }

    /// Removes orphaned resources not referenced by any node in the scene.
    ///
    /// Walks the scene tree to find all referenced instances, meshes, materials,
    /// and textures, then removes any that are unreferenced. The default material
    /// is always retained.
    pub fn cleanup(&mut self) {
        use std::collections::HashSet;

        // Collect all instance IDs referenced by nodes
        let referenced_instances: HashSet<InstanceId> = self
            .nodes
            .values()
            .filter_map(|node| match node.payload() {
                NodePayload::Instance(id) => Some(*id),
                _ => None,
            })
            .collect();

        // Collect mesh and material IDs referenced by retained instances
        let mut referenced_meshes = HashSet::new();
        let mut referenced_face_materials = HashSet::new();
        let mut referenced_line_materials = HashSet::new();
        let mut referenced_point_materials = HashSet::new();
        for &inst_id in &referenced_instances {
            if let Some(inst) = self.instances.get(&inst_id) {
                referenced_meshes.insert(inst.mesh());
                if let Some(id) = inst.face_material() {
                    referenced_face_materials.insert(id);
                }
                if let Some(id) = inst.line_material() {
                    referenced_line_materials.insert(id);
                }
                if let Some(id) = inst.point_material() {
                    referenced_point_materials.insert(id);
                }
            }
        }

        // Collect texture IDs referenced by retained face materials
        let mut referenced_textures = HashSet::new();
        for &mat_id in &referenced_face_materials {
            if let Some(mat) = self.face_materials.get(&mat_id) {
                if let Some(tex) = mat.base_color_texture() {
                    referenced_textures.insert(tex);
                }
                if let Some(tex) = mat.normal_texture() {
                    referenced_textures.insert(tex);
                }
                if let Some(tex) = mat.metallic_roughness_texture() {
                    referenced_textures.insert(tex);
                }
            }
        }

        // Remove unreferenced resources
        self.instances.retain(|id, _| referenced_instances.contains(id));
        self.meshes.retain(|id, _| referenced_meshes.contains(id));
        self.face_materials.retain(|id, _| referenced_face_materials.contains(id));
        self.line_materials.retain(|id, _| referenced_line_materials.contains(id));
        self.point_materials.retain(|id, _| referenced_point_materials.contains(id));
        self.textures.retain(|id, _| referenced_textures.contains(id));
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
        let children: Vec<NodeId> = node.children().to_vec();
        for child_id in children {
            self.invalidate_subtree_transforms(child_id);
        }
    }

    // ========== Node Transform Mutation API ==========

    /// Sets the payload of a node and invalidates ancestor bounds.
    pub fn set_node_payload(&mut self, node_id: NodeId, payload: NodePayload) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_payload(payload);
        self.node_generation += 1;
        self.invalidate_ancestor_bounds(node_id);
    }

    pub fn set_node_transform(&mut self, node_id: NodeId, transform: common::Transform) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_transform(transform);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the position of a node and invalidates ancestor bounds.
    pub fn set_node_position(&mut self, node_id: NodeId, position: Point3) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_position(position);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the rotation of a node and invalidates ancestor bounds.
    pub fn set_node_rotation(&mut self, node_id: NodeId, rotation: Quaternion) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_rotation(rotation);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets the scale of a node and invalidates ancestor bounds.
    pub fn set_node_scale(&mut self, node_id: NodeId, scale: Vector3) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_scale(scale);
        self.invalidate_subtree_transforms(node_id);
        self.invalidate_ancestor_bounds(node_id);
        self.node_generation += 1;
    }

    /// Sets a node's render-presentation behavior (layer, screen-space flags).
    ///
    /// The behavior inherits down the subtree and is resolved by the renderer,
    /// so no cache invalidation beyond bumping the node generation is needed.
    pub fn set_node_display(&mut self, node_id: NodeId, display: crate::DisplayBehavior) {
        let node = self.nodes.get_mut(&node_id).expect("Node not found");
        node.set_display(display);
        self.node_generation += 1;
    }

    // ========== Visibility API ==========

    /// Sets the visibility of a node and propagates invisibility to descendants.
    ///
    /// When setting to Invisible, all descendants are also marked invisible.
    /// When setting to Visible, only this node is changed.
    pub fn set_node_visibility(&mut self, node_id: NodeId, visibility: node::Visibility) {
        match visibility {
            node::Visibility::Visible => {
                let node = self.nodes.get_mut(&node_id).expect("Node not found");
                node.set_visibility(node::Visibility::Visible);
                self.invalidate_ancestor_effective_visibility(node_id);
            }
            node::Visibility::Invisible => {
                self.set_subtree_visibility_recursive(node_id, node::Visibility::Invisible);
                if let Some(node) = self.get_node(node_id)
                    && let Some(parent_id) = node.parent() {
                        self.invalidate_ancestor_effective_visibility(parent_id);
                    }
            }
        }
    }

    /// Recursively sets visibility for a node and all descendants.
    fn set_subtree_visibility_recursive(&mut self, node_id: NodeId, visibility: node::Visibility) {
        let Some(node) = self.nodes.get_mut(&node_id) else {
            return;
        };
        node.set_visibility(visibility);
        let children: Vec<NodeId> = node.children().to_vec();

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
    pub fn node_effective_visibility(
        &self,
        node_id: NodeId,
    ) -> node::EffectiveVisibility {
        let node = self.get_node(node_id).expect("Node not found");

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
    ) -> node::EffectiveVisibility {
        let node = self.get_node(node_id).expect("Node not found");

        if node.visibility() == node::Visibility::Invisible {
            return node::EffectiveVisibility::Invisible;
        }

        let mut all_visible = true;
        for &child_id in node.children() {
            let child_effective = self.node_effective_visibility(child_id);
            if child_effective != node::EffectiveVisibility::Visible {
                all_visible = false;
                break;
            }
        }

        if all_visible {
            node::EffectiveVisibility::Visible
        } else {
            node::EffectiveVisibility::Mixed
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

        for &root_id in &self.root_nodes {
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

        for &child_id in node.children() {
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
            NodePayload::Instance(instance_id) => {
                let Some(world_transform) = self.nodes_transform(node_id) else {
                    incomplete = true;
                    return BoundingResult { bounds: merged_bounds, incomplete };
                };
                let Some(instance) = self.get_instance(*instance_id) else {
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
            for &child_id in node.children() {
                if !self.nodes.contains_key(&child_id) {
                    return false;
                }
            }
            if let NodePayload::Instance(instance_id) = node.payload() {
                let Some(instance) = self.instances.get(instance_id) else {
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
}

#[cfg(test)]
mod scene_tests;
