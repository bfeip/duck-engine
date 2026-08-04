use duck_engine_common::{Matrix4, SquareMatrix};

use crate::common::Aabb;
use crate::{
    DisplayBehavior, InstanceId, Mesh, NodeFlags, NodeId, NodePayload, PositionedCamera, Scene, SceneData,
};

/// A query that can pick objects by traversing the scene tree.
///
/// Implement this trait to create custom picking behaviors. The generic
/// traversal handles tree walking, AABB culling, and coordinate space
/// transformations - implementors only need to define the actual tests.
pub trait PickQuery {
    /// Result type returned for hits. A single mesh test may produce multiple results
    /// (e.g., ray hitting multiple triangles).
    type Result;

    /// Broad phase test: does this query potentially intersect geometry within the bounds?
    ///
    /// Return `true` if the subtree should be explored, `false` to skip it entirely.
    /// This should be a fast, conservative test - false positives are acceptable,
    /// but false negatives will cause missed results.
    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool;

    /// Transform this query to a different coordinate space.
    ///
    /// Used to transform the query from world space to local mesh space.
    fn transform(&self, matrix: &Matrix4) -> Self;

    /// Narrow phase test: test the query against a mesh and collect results.
    ///
    /// Called when a node passes the broad phase and has an instance attached.
    /// The query has already been transformed to local mesh space.
    ///
    /// # Arguments
    /// * `mesh` - The mesh to test against (in local space)
    /// * `node_id` - ID of the node being tested
    /// * `instance_id` - ID of the instance being tested
    /// * `world_transform` - The node's world transform (for result computation)
    /// * `results` - Vector to push results into
    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node_id: NodeId,
        instance_id: InstanceId,
        world_transform: &Matrix4,
        results: &mut Vec<Self::Result>,
    );
}

/// Camera context that makes picking agree with a screen-space render.
///
/// Screen-sized / screen-facing nodes ([`DisplayBehavior::is_screen_space`]) are
/// drawn at a camera-dependent effective transform, not their authored world
/// transform. Supplying a `PickView` lets the traversal test against that same
/// effective transform so picks land where the geometry visually appears.
/// Without one (`pick_all`), screen-space nodes are picked at their authored
/// world transform.
pub struct PickView<'a> {
    pub camera: &'a PositionedCamera,
    /// Viewport size in pixels (width, height).
    pub viewport: (u32, u32),
}

/// Picks all instances in the scene that match the given query.
///
/// Walks the scene tree from all root nodes, using cached bounding boxes
/// for efficient culling. The query is transformed to local space for each
/// instance test. Screen-space nodes are tested at their authored world
/// transform; use [`pick_all_with_view`] to test them at their on-screen
/// (camera-dependent) placement instead.
pub fn pick_all<Q: PickQuery>(query: &Q, scene: &Scene) -> Vec<Q::Result> {
    pick_all_with_view(query, scene, None)
}

/// Like [`pick_all`], but resolves screen-space nodes against `view` so picking
/// matches the rendered (camera-dependent) placement of screen-sized /
/// screen-facing geometry.
pub fn pick_all_with_view<Q: PickQuery>(
    query: &Q,
    scene: &Scene,
    view: Option<&PickView>,
) -> Vec<Q::Result> {
    let scene = scene.lock();
    let mut results = Vec::new();

    for &root_id in scene.root_nodes() {
        pick_node(query, root_id, &scene, DisplayBehavior::default(), view, &mut results);
    }

    results
}

/// Recursively tests a query against a node and its descendants.
fn pick_node<Q: PickQuery>(
    query: &Q,
    node_id: NodeId,
    scene: &SceneData,
    parent_display: DisplayBehavior,
    view: Option<&PickView>,
    results: &mut Vec<Q::Result>,
) {
    let Some(node) = scene.get_node(node_id) else { return };

    if node.flags().contains(NodeFlags::DO_NOT_SELECT) {
        return;
    }

    // Resolve this node's presentation against the inherited one so screen-space
    // flags set on a group root reach their descendants.
    let display = DisplayBehavior::inherit(parent_display, node.display());

    // A screen-space node is drawn at a camera-dependent effective transform. We
    // only honor that when a view is supplied; otherwise fall back to the
    // authored world transform.
    let screen_space = view.is_some() && display.is_screen_space();

    // Broad phase: skip entire subtree if bounds are unavailable or don't intersect.
    // nodes_bounding returns None for both "no geometry" and "geometry not yet
    // present", both of which mean there is nothing pickable here yet.
    //
    // Skipped for screen-space subtrees: their cached world-space bounds are
    // authored-size and would false-negative once the effective (on-screen) scale
    // exceeds the authored one.
    //
    // TODO: This means that the odd case of a screen-space node that is parented to a
    // non-screen-space node might be incorrectly skipped — though generally that does
    // not happen and so the case isn't considered here.
    if !screen_space {
        let Some(bounds) = scene.nodes_bounding(node_id).bounds else { return };
        if !query.might_intersect_bounds(&bounds) {
            return;
        }
    }

    // Narrow phase: test this node's instance (if any). A missing instance/mesh/
    // transform just means the resource hasn't arrived yet; skip it but still
    // recurse to children.
    if let NodePayload::Instance(instance_id) = node.payload()
        && let Some(transform) = pick_transform(scene, node_id, display, view)
    {
        collect_instance_hits(query, node_id, *instance_id, scene, &transform, results);
    }

    // Recurse to children. Missing child nodes are silently skipped.
    for &child_id in node.children() {
        pick_node(query, child_id, scene, display, view, results);
    }
}

/// The model matrix to pick `node_id` at.
///
/// Screen-space nodes ([`DisplayBehavior::is_screen_space`]) are drawn at the
/// renderer's camera-dependent effective transform, so — when a [`PickView`] is
/// available — they are picked there too; everything else picks at the cached
/// world transform. Returns `None` while the node's world transform is
/// unavailable.
fn pick_transform(
    scene: &SceneData,
    node_id: NodeId,
    display: DisplayBehavior,
    view: Option<&PickView>,
) -> Option<Matrix4> {
    let world = scene.nodes_transform(node_id)?;
    match view {
        Some(view) if display.is_screen_space() => {
            Some(display.effective_transform(world, view.camera, view.viewport))
        }
        _ => Some(world),
    }
}

/// Narrow phase: transforms the query into `transform`'s local space and collects
/// the node instance's mesh hits. A no-op if the instance or its mesh is missing,
/// or if `transform` is singular.
fn collect_instance_hits<Q: PickQuery>(
    query: &Q,
    node_id: NodeId,
    instance_id: InstanceId,
    scene: &SceneData,
    transform: &Matrix4,
    results: &mut Vec<Q::Result>,
) {
    let Some(instance) = scene.get_instance(instance_id) else { return };
    let Some(mesh) = scene.get_mesh(instance.mesh()) else { return };
    let Some(world_to_local) = transform.invert() else { return };
    let local_query = query.transform(&world_to_local);
    local_query.collect_mesh_hits(mesh, node_id, instance_id, transform, results);
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::Matrix4;
    use crate::{
        Mesh, MeshPrimitive, NodeFlags, PrimitiveType, Vertex,
        common::Transform,
    };

    struct AlwaysHitQuery;

    impl PickQuery for AlwaysHitQuery {
        type Result = NodeId;

        fn might_intersect_bounds(&self, _bounds: &Aabb) -> bool {
            true
        }

        fn transform(&self, _matrix: &Matrix4) -> Self {
            AlwaysHitQuery
        }

        fn collect_mesh_hits(
            &self,
            _mesh: &Mesh,
            node_id: NodeId,
            _instance_id: InstanceId,
            _world_transform: &Matrix4,
            results: &mut Vec<NodeId>,
        ) {
            results.push(node_id);
        }
    }

    fn make_geometry_node(scene: &mut Scene, parent: Option<NodeId>, flags: NodeFlags) -> NodeId {
        let mut scene = scene.lock();

        let mesh = Mesh::from_raw(
            vec![
                Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
            ],
            vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
        );
        let mesh_id = scene.add_mesh(mesh);
        let instance = crate::Instance::new(mesh_id).with_face_material(crate::FaceMaterialId::new());
        scene.add_instance_node(parent, instance, None, Transform::IDENTITY, flags).unwrap()
    }

    #[test]
    fn test_normal_node_is_pickable() {
        let mut scene = Scene::default();
        let node_id = make_geometry_node(&mut scene, None, NodeFlags::NONE);

        let hits = pick_all(&AlwaysHitQuery, &scene);
        assert_eq!(hits, vec![node_id]);
    }

    #[test]
    fn test_do_not_select_skips_node() {
        let mut scene = Scene::default();
        let _node_id = make_geometry_node(&mut scene, None, NodeFlags::DO_NOT_SELECT);

        let hits = pick_all(&AlwaysHitQuery, &scene);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_do_not_select_skips_children() {
        let mut scene = Scene::default();
        // DO_NOT_SELECT group parent with a selectable child
        let parent = scene.add_node(None, None, Transform::IDENTITY, NodeFlags::DO_NOT_SELECT).unwrap();
        let _child = make_geometry_node(&mut scene, Some(parent), NodeFlags::NONE);

        let hits = pick_all(&AlwaysHitQuery, &scene);
        assert!(hits.is_empty());
    }
}
