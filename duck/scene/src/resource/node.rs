use crate::{CameraProjection, Light};
use super::handle::{InstanceHandle, NodeHandle};
use super::DisplayBehavior;
use super::RenderLayer;
use crate::common::{
    Aabb, Transform, apply_scale, compose_rotation, local_axes, local_axis_x, local_axis_y,
    local_axis_z, rotate_position_about_pivot, scale_position_about_pivot_local,
    scale_position_about_pivot_world,
};
use bitflags::bitflags;
use duck_engine_common::{Matrix4, Point3, Quaternion, Vector3};
use std::cell::Cell;

/// Unique identifier for a Node in the scene tree.
pub type NodeId = super::Id<Node>;

/// Trait for custom, user-defined node payloads.
///
/// External crates can implement this to attach arbitrary typed data to scene nodes.
/// Custom payloads serialize as `NodePayload::None` for now.
pub trait CustomNodePayload: Send + Sync {}

/// The typed content of a scene node.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NodePayload {
    /// Structural container with no content (default).
    None,
    /// References a mesh+material pair to be rendered. The handle owns the
    /// instance: it is removed when no longer referenced by any node or handle.
    Instance(InstanceHandle),
    /// A camera. Projection intrinsics are stored here; pose lives in the node's Transform.
    Camera(CameraProjection),
    /// A light source. Position and direction are derived from the node's world transform:
    /// translation column → position (Point, Spot); negative Z-axis → direction (Directional, Spot).
    Light(Light),
    /// A runtime-only custom payload defined by an external crate.
    /// Serializes as `None` for now.
    #[cfg_attr(feature = "serde", serde(skip))]
    Custom(Box<dyn CustomNodePayload>),
}

impl Default for NodePayload {
    fn default() -> Self {
        Self::None
    }
}

impl std::fmt::Debug for NodePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Instance(h) => f.debug_tuple("Instance").field(&h.id()).finish(),
            Self::Camera(c) => f.debug_tuple("Camera").field(c).finish(),
            Self::Light(l) => f.debug_tuple("Light").field(l).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").field(&"..").finish(),
        }
    }
}

impl Clone for NodePayload {
    fn clone(&self) -> Self {
        match self {
            Self::None => Self::None,
            Self::Instance(h) => Self::Instance(h.clone()),
            Self::Camera(c) => Self::Camera(c.clone()),
            Self::Light(l) => Self::Light(l.clone()),
            Self::Custom(_) => Self::None,
        }
    }
}

/// Explicit visibility state set by the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Visibility {
    /// Node is explicitly set to visible
    #[default]
    Visible,
    /// Node is explicitly set to invisible
    Invisible,
}

/// Effective visibility state computed during tree traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectiveVisibility {
    /// Node and all descendants are visible
    Visible,
    /// Node is explicitly invisible
    Invisible,
    /// Node is visible but has some invisible descendants
    Mixed,
}

bitflags! {
    /// Flags that dictate certain node behaviors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    pub struct NodeFlags: u32 {
        /// No special behavior.
        const NONE = 0;
        /// Marks a node as non-selectable. Geometry queries will not search this node or its children.
        const DO_NOT_SELECT = 1 << 0;
        /// Marks a node as not part of the scene bounding. This node will not be used for bounding calculations.
        const DOES_NOT_CONTRIBUTE_BOUNDING = 1 << 1;
    }
}

impl NodeFlags {
    /// A set of flags marking a node as (more or less) non-interactive. Geometry will be visible,
    /// but that is all.
    pub fn inert() -> Self {
        Self::DO_NOT_SELECT | Self::DOES_NOT_CONTRIBUTE_BOUNDING
    }
}

/// A node in the scene tree hierarchy.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Node {
    pub id: NodeId,
    pub name: Option<String>,

    transform: Transform,

    parent: Option<NodeId>,
    /// Owning references: an attached node is kept alive by its parent's entry
    /// here (or by the scene's root list).
    children: Vec<NodeHandle>,
    flags: NodeFlags,

    payload: NodePayload,

    // Render-presentation behavior (placement / screen-space). Inherits down
    // the subtree; resolved by the renderer, so no cache is needed here.
    display: DisplayBehavior,

    // Visibility
    visibility: Visibility,
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_effective_visibility: Cell<Option<EffectiveVisibility>>,

    // Cached computed values
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_world_transform: Cell<Option<Matrix4>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_bounds: Cell<Option<Aabb>>,
}

impl Node {
    /// Creates a new node with the given transform.
    pub fn new(name: Option<String>, transform: Transform, flags: NodeFlags) -> Self {
        Self {
            id: super::Id::new(),
            name,
            transform,
            parent: None,
            children: Vec::new(),
            flags,
            payload: NodePayload::None,
            display: DisplayBehavior::default(),
            visibility: Visibility::default(),
            cached_effective_visibility: Cell::new(None),
            cached_world_transform: Cell::new(None),
            cached_bounds: Cell::new(None),
        }
    }

    /// Creates a new node with default transform (identity).
    pub fn new_default() -> Self {
        Self::new(None, Transform::IDENTITY, NodeFlags::NONE)
    }

    // ============== Transform ==============

    /// Computes the local transform matrix from position, rotation, and scale.
    ///
    /// The order of operations is: Translation * Rotation * Scale (TRS)
    pub fn compute_local_transform(&self) -> Matrix4 {
        self.transform.to_matrix()
    }

    pub fn transform(&self) -> Transform {
        self.transform
    }

    pub fn set_transform(&mut self, transform: Transform) {
        self.transform = transform;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    // Methods to get and set individual parts of the transform. This is
    // for convince and to make sure that individual mutations of the transform
    // properly update the dirty state.
    pub fn position(&self) -> Point3 {
        self.transform.position
    }

    pub fn set_position(&mut self, position: Point3) {
        self.transform.position = position;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    pub fn rotation(&self) -> Quaternion {
        self.transform.rotation
    }

    pub fn set_rotation(&mut self, rotation: Quaternion) {
        self.transform.rotation = rotation;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    pub fn scale(&self) -> Vector3 {
        self.transform.scale
    }

    pub fn set_scale(&mut self, scale: Vector3) {
        self.transform.scale = scale;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    // ========== Transform Manipulation Methods ==========

    /// Rotates the node's position and orientation around a world-space pivot point.
    pub fn rotate_about_pivot(&mut self, pivot: Point3, rotation: Quaternion) {
        let new_position = rotate_position_about_pivot(self.transform.position, pivot, rotation);
        let new_rotation = compose_rotation(self.transform.rotation, rotation);
        self.set_position(new_position);
        self.set_rotation(new_rotation);
    }

    /// Scales the node's position and scale relative to a world-space pivot point.
    pub fn scale_about_pivot(&mut self, pivot: Point3, scale_factor: Vector3) {
        let new_position = scale_position_about_pivot_world(self.transform.position, pivot, scale_factor);
        let new_scale = apply_scale(self.transform.scale, scale_factor);
        self.set_position(new_position);
        self.set_scale(new_scale);
    }

    /// Scales the node's position and scale relative to a pivot point in local space.
    ///
    /// The local space is defined by the given orientation.
    pub fn scale_about_pivot_local(
        &mut self,
        pivot: Point3,
        scale_factor: Vector3,
        local_orientation: Quaternion,
    ) {
        let new_position =
            scale_position_about_pivot_local(self.transform.position, pivot, scale_factor, local_orientation);
        let new_scale = apply_scale(self.transform.scale, scale_factor);
        self.set_position(new_position);
        self.set_scale(new_scale);
    }

    /// Translates the node by the given world space offset.
    pub fn translate(&mut self, offset: Vector3) {
        self.set_position(self.transform.position + offset);
    }

    /// Returns the local X axis (right) in world space.
    pub fn local_x_axis(&self) -> Vector3 {
        local_axis_x(self.transform.rotation)
    }

    /// Returns the local Y axis (up) in world space.
    pub fn local_y_axis(&self) -> Vector3 {
        local_axis_y(self.transform.rotation)
    }

    /// Returns the local Z axis (forward) in world space.
    pub fn local_z_axis(&self) -> Vector3 {
        local_axis_z(self.transform.rotation)
    }

    /// Returns all local axes (right, up, forward) in world space.
    pub fn local_axes(&self) -> (Vector3, Vector3, Vector3) {
        local_axes(self.transform.rotation)
    }

    // ============== Hierarchy ==============

    /// Gets the parent node ID.
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Sets the parent node ID without maintaining tree consistency.
    ///
    /// The caller must ensure the parent-child relationship is consistent
    /// on both sides. Prefer `SceneData` methods for safe tree manipulation.
    pub fn set_parent_unchecked(&mut self, parent: Option<NodeId>) {
        self.parent = parent;
        self.mark_transform_dirty();
        self.mark_bounds_dirty();
    }

    /// Iterates the child node IDs.
    pub fn children(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.children.iter().map(NodeHandle::id)
    }

    /// Returns the number of children.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Returns true if `child` is a direct child of this node.
    pub fn has_child(&self, child: NodeId) -> bool {
        self.children.iter().any(|h| h.id() == child)
    }

    /// The owning child handles.
    pub(crate) fn child_handles(&self) -> &[NodeHandle] {
        &self.children
    }

    /// Adds an owning child handle without maintaining tree consistency.
    ///
    /// The caller must ensure the child's parent pointer is set correspondingly.
    /// Prefer `SceneData` methods for safe tree manipulation.
    pub fn add_child_unchecked(&mut self, child: NodeHandle) {
        if !self.children.contains(&child) {
            self.children.push(child);
            self.mark_bounds_dirty();
        }
    }

    /// Removes a child without maintaining tree consistency, dropping the
    /// owning handle.
    ///
    /// The caller must ensure the child's parent pointer is updated correspondingly.
    /// Prefer `SceneData` methods for safe tree manipulation.
    pub fn remove_child_unchecked(&mut self, child: NodeId) {
        self.children.retain(|h| h.id() != child);
        self.mark_bounds_dirty();
    }

    /// Removes and returns a child's owning handle without maintaining tree
    /// consistency.
    pub(crate) fn take_child_unchecked(&mut self, child: NodeId) -> Option<NodeHandle> {
        let pos = self.children.iter().position(|h| h.id() == child)?;
        self.mark_bounds_dirty();
        Some(self.children.remove(pos))
    }

    /// Replaces the children list without maintaining tree consistency.
    ///
    /// The caller must ensure all children reference valid nodes and that
    /// their parent pointers are consistent. Prefer `SceneData` methods for safe tree manipulation.
    pub fn set_children_unchecked(&mut self, children: Vec<NodeHandle>) {
        self.children = children;
        self.mark_bounds_dirty();
    }

    pub fn flags(&self) -> NodeFlags {
        self.flags
    }

    pub fn set_flags(&mut self, flags: NodeFlags) {
        self.flags = flags;
        self.mark_bounds_dirty();
    }

    /// Returns the node's payload.
    pub fn payload(&self) -> &NodePayload {
        &self.payload
    }

    /// Sets the node's payload and invalidates this node's bounds cache.
    ///
    /// Ancestor bounds propagation is the caller's (SceneData's) responsibility.
    pub fn set_payload(&mut self, payload: NodePayload) {
        self.payload = payload;
        self.cached_bounds.set(None);
    }

    // ========== Display Behavior ==========

    /// Returns this node's render-presentation behavior.
    pub fn display(&self) -> DisplayBehavior {
        self.display
    }

    /// Sets this node's render-presentation behavior.
    ///
    /// Does not touch the world-transform or bounds caches: the layer affects
    /// neither, and the screen-space effects are applied by the renderer
    /// downstream of the camera-independent cached transform (which is reused
    /// for picking and bounding).
    pub fn set_display(&mut self, display: DisplayBehavior) {
        self.display = display;
    }

    /// Sets which render layer / pass this node draws in.
    pub fn set_render_layer(&mut self, layer: RenderLayer) {
        self.display.layer = layer;
    }

    /// Sets the node's constant on-screen size, or `None` to leave it at its
    /// authored world size. See [`DisplayBehavior::screen_size`].
    pub fn set_screen_size(&mut self, screen_size: Option<f32>) {
        self.display.screen_size = screen_size;
    }

    /// Sets whether this node orients to face the camera (billboard).
    pub fn set_screen_facing(&mut self, screen_facing: bool) {
        self.display.screen_facing = screen_facing;
    }

    // ============== Dirty State ==============

    /// Marks this node's world transform as dirty (needs recomputation).
    /// Note: This only marks this node, not descendants. The SceneData is responsible
    /// for propagating dirty flags to children.
    pub(crate) fn mark_transform_dirty(&self) {
        self.cached_world_transform.set(None);
    }

    /// Marks this node's bounds as dirty (needs recomputation).
    /// Note: This only marks this node, not descendants.
    pub(crate) fn mark_bounds_dirty(&self) {
        self.cached_bounds.set(None);
    }

    pub fn transform_dirty(&self) -> bool {
        self.cached_world_transform.get().is_none()
    }

    pub fn bounds_dirty(&self) -> bool {
        self.cached_bounds.get().is_none()
    }

    /// Gets the cached world transform if valid
    /// You probably want [crate::SceneData::nodes_transform]
    pub fn cached_world_transform(&self) -> Option<Matrix4> {
        self.cached_world_transform.get()
    }

    /// Sets the cached world transform
    pub fn set_cached_world_transform(&self, transform: Matrix4) {
        self.cached_world_transform.set(Some(transform));
    }

    /// Gets the cached bounding box if valid
    /// You probably want [crate::SceneData::nodes_bounding]
    pub(crate) fn cached_bounds(&self) -> Option<Aabb> {
        self.cached_bounds.get()
    }

    /// Sets the cached bounding box
    pub(crate) fn set_cached_bounds(&self, bounds: Option<Aabb>) {
        self.cached_bounds.set(bounds);
    }

    // ========== Visibility Management ==========

    /// Gets the explicit visibility state of this node.
    pub fn visibility(&self) -> Visibility {
        self.visibility
    }

    /// Sets the explicit visibility state of this node.
    pub fn set_visibility(&mut self, visibility: Visibility) {
        self.visibility = visibility;
        self.cached_effective_visibility.set(None);
    }

    /// Gets the cached effective visibility if valid.
    pub(crate) fn cached_effective_visibility(&self) -> Option<EffectiveVisibility> {
        self.cached_effective_visibility.get()
    }

    /// Sets the cached effective visibility.
    pub(crate) fn set_cached_effective_visibility(&self, visibility: EffectiveVisibility) {
        self.cached_effective_visibility.set(Some(visibility));
    }

    /// Returns true if effective visibility needs recomputation.
    pub fn effective_visibility_dirty(&self) -> bool {
        self.cached_effective_visibility.get().is_none()
    }

    /// Marks visibility cache as dirty.
    pub fn mark_visibility_dirty(&self) {
        self.cached_effective_visibility.set(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{EPSILON, Transform};
    use duck_engine_common::{Deg, EuclideanSpace, Quaternion, Rotation3, Vector3};

    // ========================================================================
    // Node Creation Tests
    // ========================================================================

    #[test]
    fn test_node_new() {
        let position = Point3::new(1.0, 2.0, 3.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(2.0, 2.0, 2.0);

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);

        assert_eq!(node.position(), position);
        assert_eq!(node.rotation(), rotation);
        assert_eq!(node.scale(), scale);
    }

    #[test]
    fn test_node_default_values() {
        let node = Node::new_default();

        assert_eq!(node.name, None);
        assert_eq!(node.parent(), None);
        assert_eq!(node.child_count(), 0);
        assert!(matches!(node.payload(), NodePayload::None));
    }

    #[test]
    fn test_node_local_transform_identity() {
        let node = Node::new_default();
        let transform = node.compute_local_transform();

        // Identity transform should be close to Matrix4::from_scale(1.0)
        let identity = Matrix4::from_scale(1.0);

        // Check each element is close to identity
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (transform[i][j] - identity[i][j]).abs() < EPSILON,
                    "Transform element [{i}][{j}] = {}, expected {}",
                    transform[i][j],
                    identity[i][j]
                );
            }
        }
    }

    // ========================================================================
    // Node Transform Tests
    // ========================================================================

    #[test]
    fn test_compute_local_transform_translation_only() {
        let position = Point3::new(5.0, 10.0, 15.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity
        let scale = Vector3::new(1.0, 1.0, 1.0); // Unity scale

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // Check translation components (last column)
        assert!((transform[3][0] - 5.0).abs() < EPSILON);
        assert!((transform[3][1] - 10.0).abs() < EPSILON);
        assert!((transform[3][2] - 15.0).abs() < EPSILON);
        assert!((transform[3][3] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_local_transform_rotation_only() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::from_angle_z(Deg(90.0)); // 90 degrees around Z
        let scale = Vector3::new(1.0, 1.0, 1.0);

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // Apply transform to point (1, 0, 0) - should become roughly (0, 1, 0)
        let point = Vector3::new(1.0, 0.0, 0.0);
        let rotated_x = transform[0][0] * point.x + transform[1][0] * point.y + transform[2][0] * point.z;
        let rotated_y = transform[0][1] * point.x + transform[1][1] * point.y + transform[2][1] * point.z;
        let rotated_z = transform[0][2] * point.x + transform[1][2] * point.y + transform[2][2] * point.z;

        assert!(rotated_x.abs() < EPSILON, "Expected x ≈ 0, got {}", rotated_x);
        assert!((rotated_y - 1.0).abs() < EPSILON, "Expected y ≈ 1, got {}", rotated_y);
        assert!(rotated_z.abs() < EPSILON, "Expected z ≈ 0, got {}", rotated_z);
    }

    #[test]
    fn test_compute_local_transform_scale_only() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(2.0, 3.0, 4.0);

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // Check diagonal elements (scale factors)
        assert!((transform[0][0] - 2.0).abs() < EPSILON);
        assert!((transform[1][1] - 3.0).abs() < EPSILON);
        assert!((transform[2][2] - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_compute_local_transform_trs_composition() {
        // Translation × Rotation × Scale
        let position = Point3::new(10.0, 20.0, 30.0);
        let rotation = Quaternion::from_angle_y(Deg(45.0));
        let scale = Vector3::new(2.0, 2.0, 2.0);

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // Manually compute expected transform
        let translation_matrix = Matrix4::from_translation(position.to_vec());
        let rotation_matrix = Matrix4::from(rotation);
        let scale_matrix = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
        let expected = translation_matrix * rotation_matrix * scale_matrix;

        // Compare all elements
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (transform[i][j] - expected[i][j]).abs() < EPSILON,
                    "Transform element [{i}][{j}] = {}, expected {}",
                    transform[i][j],
                    expected[i][j]
                );
            }
        }
    }

    // ========================================================================
    // Node Hierarchy Tests
    // ========================================================================

    #[test]
    fn test_set_parent() {
        let mut node = Node::new_default();
        assert_eq!(node.parent(), None);

        let parent_id = crate::resource::Id::new();
        node.set_parent_unchecked(Some(parent_id));
        assert_eq!(node.parent(), Some(parent_id));

        node.set_parent_unchecked(None);
        assert_eq!(node.parent(), None);
    }

    #[test]
    fn test_add_child() {
        let mut parent = Node::new_default();
        let mut child1 = Node::new_default();
        let mut child2 = Node::new_default();

        let parent_id = parent.id;
        let child1_id = child1.id;
        let child2_id = child2.id;

        assert_eq!(parent.child_count(), 0);
        assert_eq!(child1.parent(), None);
        assert_eq!(child2.parent(), None);

        // Add first child (bidirectional setup)
        parent.add_child_unchecked(NodeHandle::unbound(child1_id));
        child1.set_parent_unchecked(Some(parent_id));

        assert_eq!(parent.child_count(), 1);
        assert_eq!(parent.children().next(), Some(child1_id));
        assert_eq!(child1.parent(), Some(parent_id));

        // Add second child (bidirectional setup)
        parent.add_child_unchecked(NodeHandle::unbound(child2_id));
        child2.set_parent_unchecked(Some(parent_id));

        assert_eq!(parent.child_count(), 2);
        assert!(parent.has_child(child1_id));
        assert!(parent.has_child(child2_id));
        assert_eq!(child1.parent(), Some(parent_id));
        assert_eq!(child2.parent(), Some(parent_id));
    }

    #[test]
    fn test_add_child_duplicate_ignored() {
        let mut node = Node::new_default();
        let child_id = crate::resource::Id::new();

        node.add_child_unchecked(NodeHandle::unbound(child_id));
        node.add_child_unchecked(NodeHandle::unbound(child_id)); // Duplicate
        node.add_child_unchecked(NodeHandle::unbound(child_id)); // Duplicate

        // Should only have one child
        assert_eq!(node.child_count(), 1);
        assert_eq!(node.children().next(), Some(child_id));
    }

    #[test]
    fn test_remove_child() {
        let mut node = Node::new_default();
        let id_a = crate::resource::Id::new();
        let id_b = crate::resource::Id::new();
        let id_c = crate::resource::Id::new();

        node.add_child_unchecked(NodeHandle::unbound(id_a));
        node.add_child_unchecked(NodeHandle::unbound(id_b));
        node.add_child_unchecked(NodeHandle::unbound(id_c));

        assert_eq!(node.child_count(), 3);

        node.remove_child_unchecked(id_b);
        assert_eq!(node.child_count(), 2);
        assert!(node.has_child(id_a));
        assert!(!node.has_child(id_b));
        assert!(node.has_child(id_c));
    }

    #[test]
    fn test_remove_child_nonexistent() {
        let mut node = Node::new_default();
        let child_id = crate::resource::Id::new();
        node.add_child_unchecked(NodeHandle::unbound(child_id));

        // Removing non-existent child should not panic
        node.remove_child_unchecked(crate::resource::Id::new());
        assert_eq!(node.child_count(), 1);
        assert_eq!(node.children().next(), Some(child_id));
    }

    // ========================================================================
    // Node Cache Tests
    // ========================================================================

    #[test]
    fn test_cached_world_transform_initially_none() {
        let node = Node::new_default();
        assert!(node.cached_world_transform().is_none());
    }

    #[test]
    fn test_set_cached_world_transform() {
        let node = Node::new_default();
        let transform = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));

        node.set_cached_world_transform(transform);

        let cached = node.cached_world_transform();
        assert!(cached.is_some());

        let cached_transform = cached.unwrap();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(cached_transform[i][j], transform[i][j]);
            }
        }
    }

    #[test]
    fn test_cached_world_transform_retrieval() {
        let node = Node::new_default();

        // Initially None
        assert!(node.cached_world_transform().is_none());

        // Set and retrieve
        let transform1 = Matrix4::from_scale(2.0);
        node.set_cached_world_transform(transform1);
        assert!(node.cached_world_transform().is_some());

        // Set different value
        let transform2 = Matrix4::from_scale(3.0);
        node.set_cached_world_transform(transform2);
        let cached = node.cached_world_transform().unwrap();
        assert!((cached[0][0] - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_cached_bounds_initially_none() {
        let node = Node::new_default();
        assert!(node.cached_bounds().is_none());
    }

    #[test]
    fn test_set_cached_bounds() {
        let node = Node::new_default();
        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };

        node.set_cached_bounds(Some(bounds));

        let cached = node.cached_bounds();
        assert!(cached.is_some());

        let cached_bounds = cached.unwrap();
        assert_eq!(cached_bounds.min, bounds.min);
        assert_eq!(cached_bounds.max, bounds.max);
    }

    #[test]
    fn test_cached_bounds_retrieval() {
        let node = Node::new_default();

        // Initially None
        assert!(node.cached_bounds().is_none());

        // Set and retrieve
        let bounds = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(10.0, 10.0, 10.0),
        };
        node.set_cached_bounds(Some(bounds));
        assert!(node.cached_bounds().is_some());

        // Can set to None
        node.set_cached_bounds(None);
        assert!(node.cached_bounds().is_none());
    }

    #[test]
    fn test_bounds_dirty_flag() {
        let node = Node::new_default();

        // Initially dirty (no cache)
        assert!(node.bounds_dirty());

        // Set cache - no longer dirty
        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };
        node.set_cached_bounds(Some(bounds));
        assert!(!node.bounds_dirty());

        // Mark bounds dirty
        node.mark_bounds_dirty();
        assert!(node.bounds_dirty());
    }

    #[test]
    fn test_transform_dirty_flag() {
        let node = Node::new_default();

        // Initially dirty (no cache)
        assert!(node.transform_dirty());

        // Set cache - no longer dirty
        let transform = Matrix4::from_scale(1.0);
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Mark transform dirty
        node.mark_transform_dirty();
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_dirty_flags_are_independent() {
        let node = Node::new_default();

        // Set both caches
        let transform = Matrix4::from_scale(1.0);
        node.set_cached_world_transform(transform);

        let bounds = Aabb {
            min: Point3::new(-1.0, -1.0, -1.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };
        node.set_cached_bounds(Some(bounds));

        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Mark only transform dirty
        node.mark_transform_dirty();
        assert!(node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Reset transform cache
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Mark only bounds dirty
        node.mark_bounds_dirty();
        assert!(!node.transform_dirty());
        assert!(node.bounds_dirty());
    }

    #[test]
    fn test_set_position_marks_dirty() {
        let mut node = Node::new_default();

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change position should mark dirty
        node.set_position(Point3::new(5.0, 5.0, 5.0));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_rotation_marks_dirty() {
        let mut node = Node::new_default();

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change rotation should mark dirty
        node.set_rotation(Quaternion::from_angle_z(Deg(45.0)));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_scale_marks_dirty() {
        let mut node = Node::new_default();

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change scale should mark dirty
        node.set_scale(Vector3::new(2.0, 2.0, 2.0));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_set_parent_marks_dirty() {
        let mut node = Node::new_default();

        // Set cache
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        assert!(!node.transform_dirty());

        // Change parent should mark dirty
        node.set_parent_unchecked(Some(crate::resource::Id::new()));
        assert!(node.transform_dirty());
    }

    #[test]
    fn test_add_child_marks_bounds_dirty() {
        let mut node = Node::new_default();

        // Set both caches
        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        node.set_cached_bounds(Some(Aabb::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.)
        )));
        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());

        // Add child should only mark bounds dirty, not transform
        node.add_child_unchecked(NodeHandle::unbound(crate::resource::Id::new()));
        assert!(!node.transform_dirty());
        assert!(node.bounds_dirty());
    }

    // ========================================================================
    // Node Payload Tests
    // ========================================================================

    #[test]
    fn test_payload_none_by_default() {
        let node = Node::new_default();
        assert!(matches!(node.payload(), NodePayload::None));
    }

    #[test]
    fn test_set_payload_instance() {
        let mut node = Node::new_default();
        let instance_id_a = crate::resource::Id::new();
        let instance_id_b = crate::resource::Id::new();

        node.set_payload(NodePayload::Instance(InstanceHandle::unbound(instance_id_a)));
        assert!(matches!(node.payload(), NodePayload::Instance(h) if h.id() == instance_id_a));

        node.set_payload(NodePayload::Instance(InstanceHandle::unbound(instance_id_b)));
        assert!(matches!(node.payload(), NodePayload::Instance(h) if h.id() == instance_id_b));
    }

    #[test]
    fn test_set_payload_none() {
        let mut node = Node::new_default();
        let instance_id = crate::resource::Id::new();

        node.set_payload(NodePayload::Instance(InstanceHandle::unbound(instance_id)));
        assert!(matches!(node.payload(), NodePayload::Instance(h) if h.id() == instance_id));

        node.set_payload(NodePayload::None);
        assert!(matches!(node.payload(), NodePayload::None));
    }

    #[test]
    fn test_set_payload_marks_bounds_dirty() {
        let mut node = Node::new_default();

        node.set_cached_world_transform(Matrix4::from_scale(1.0));
        node.set_cached_bounds(Some(Aabb::new(
            Point3::new(-1., -1., -1.),
            Point3::new(1., 1., 1.)
        )));
        assert!(!node.transform_dirty());

        node.set_payload(NodePayload::Instance(InstanceHandle::unbound(crate::resource::Id::new())));
        assert!(!node.transform_dirty()); // payload doesn't affect transform
        assert!(node.bounds_dirty()); // payload affects bounds
    }

    // ========================================================================
    // Visibility Tests
    // ========================================================================

    #[test]
    fn test_visibility_default() {
        let node = Node::new_default();
        assert_eq!(node.visibility(), Visibility::Visible);
    }

    #[test]
    fn test_set_visibility() {
        let mut node = Node::new_default();

        // Default is visible
        assert_eq!(node.visibility(), Visibility::Visible);

        // Set to invisible
        node.set_visibility(Visibility::Invisible);
        assert_eq!(node.visibility(), Visibility::Invisible);

        // Set back to visible
        node.set_visibility(Visibility::Visible);
        assert_eq!(node.visibility(), Visibility::Visible);
    }

    #[test]
    fn test_visibility_cache_invalidation() {
        let mut node = Node::new_default();

        // Set some cached effective visibility
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Setting visibility should invalidate cache
        node.set_visibility(Visibility::Invisible);
        assert!(node.effective_visibility_dirty());
    }

    #[test]
    fn test_effective_visibility_dirty() {
        let node = Node::new_default();

        // Initially dirty (no cache)
        assert!(node.effective_visibility_dirty());

        // Set cache - no longer dirty
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Mark dirty
        node.mark_visibility_dirty();
        assert!(node.effective_visibility_dirty());
    }

    #[test]
    fn test_cached_effective_visibility() {
        let node = Node::new_default();

        // Initially None
        assert_eq!(node.cached_effective_visibility(), None);

        // Set to Visible
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Visible));

        // Set to Invisible
        node.set_cached_effective_visibility(EffectiveVisibility::Invisible);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Invisible));

        // Set to Mixed
        node.set_cached_effective_visibility(EffectiveVisibility::Mixed);
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Mixed));
    }

    #[test]
    fn test_visibility_independent_from_transform_cache() {
        let mut node = Node::new_default();

        // Set transform cache
        let transform = Matrix4::from_scale(2.0);
        node.set_cached_world_transform(transform);
        assert!(!node.transform_dirty());

        // Setting visibility should NOT invalidate transform cache
        node.set_visibility(Visibility::Invisible);
        assert!(!node.transform_dirty());
        assert!(node.effective_visibility_dirty()); // but should invalidate visibility cache
    }

    #[test]
    fn test_mark_dirty_does_not_affect_visibility_cache() {
        let node = Node::new_default();

        // Set visibility cache
        node.set_cached_effective_visibility(EffectiveVisibility::Visible);
        assert!(!node.effective_visibility_dirty());

        // Mark transform/bounds dirty
        node.mark_bounds_dirty();

        // Visibility cache should still be valid
        assert!(!node.effective_visibility_dirty());
        assert_eq!(node.cached_effective_visibility(), Some(EffectiveVisibility::Visible));
    }

    // ========================================================================
    // Display Behavior Tests
    // ========================================================================

    #[test]
    fn test_display_default_is_ordinary() {
        let node = Node::new_default();
        let d = node.display();
        assert!(d.screen_size.is_none());
        assert!(!d.screen_facing);
        assert_eq!(d.layer, RenderLayer::Scene);
    }

    #[test]
    fn test_set_display_does_not_dirty_caches() {
        let mut node = Node::new_default();

        // Prime both caches so we can detect any unwanted invalidation.
        node.set_cached_world_transform(Matrix4::from_scale(2.0));
        node.set_cached_bounds(Some(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
        )));
        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());

        node.set_display(DisplayBehavior {
            screen_size: Some(16.0),
            screen_facing: true,
            layer: RenderLayer::Overlay,
        });

        // Display behavior is resolved by the renderer and applies camera-side,
        // so it must not invalidate the camera-independent caches.
        assert!(!node.transform_dirty());
        assert!(!node.bounds_dirty());
        assert_eq!(node.display().layer, RenderLayer::Overlay);
    }

    #[test]
    fn test_display_field_setters() {
        let mut node = Node::new_default();
        node.set_render_layer(RenderLayer::Overlay);
        node.set_screen_size(Some(20.0));
        assert_eq!(node.display().layer, RenderLayer::Overlay);
        assert_eq!(node.display().screen_size, Some(20.0));
        assert!(!node.display().screen_facing);
    }

    // ========================================================================
    // Additional Edge Case Tests
    // ========================================================================

    #[test]
    fn test_node_with_name() {
        let mut node = Node::new_default();
        assert_eq!(node.name, None);

        node.name = Some("TestNode".to_string());
        assert_eq!(node.name, Some("TestNode".to_string()));
    }

    #[test]
    fn test_zero_scale() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(0.0, 0.0, 0.0);

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // Should produce a zero-scale transform (degenerate)
        assert_eq!(transform[0][0], 0.0);
        assert_eq!(transform[1][1], 0.0);
        assert_eq!(transform[2][2], 0.0);
    }

    #[test]
    fn test_negative_scale() {
        let position = Point3::new(0.0, 0.0, 0.0);
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let scale = Vector3::new(-1.0, 1.0, 1.0); // Flip X

        let node = Node::new(None, Transform::new(position, rotation, scale), NodeFlags::NONE);
        let transform = node.compute_local_transform();

        // X axis should be flipped
        assert!((transform[0][0] - (-1.0)).abs() < EPSILON);
        assert!((transform[1][1] - 1.0).abs() < EPSILON);
        assert!((transform[2][2] - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_large_hierarchy() {
        let mut parent = Node::new_default();
        let child_ids: Vec<NodeId> = (0..1000).map(|_| crate::resource::Id::new()).collect();

        for &id in &child_ids {
            parent.add_child_unchecked(NodeHandle::unbound(id));
        }

        assert_eq!(parent.child_count(), 1000);

        for &id in &child_ids {
            assert!(parent.has_child(id));
        }
    }
}
