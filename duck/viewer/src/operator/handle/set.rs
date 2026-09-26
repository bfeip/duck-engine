//! The scene resources backing a displayed set of handles.

use duck_engine_common::{InnerSpace, Point3, Quaternion, Transform, Vector3};
use duck_engine_scene::resource::{
    DisplayBehavior, FaceMaterialHandle, Instance, NodeFlags, NodeHandle, NodeId, RenderLayer,
};
use duck_engine_scene::{PositionedCamera, Scene, SceneData};

use crate::common::{Axis, Ray, RgbaColor};
use crate::geom_query::{pick_all_from_ray_with_view, PickView, RayPickQuery};
use crate::operator::drag::DragGeometry;

use super::{shape, DragKind, Handle, HandleId, HandleShape};

/// On-screen size, in pixels, of a handle's unit extent. Shapes are built at
/// unit size and held here by [`DisplayBehavior::screen_size`], so handles
/// never scale with zoom.
const HANDLE_SCREEN_SIZE: f32 = 64.0;

/// Scene resources of one displayed handle. The owning handles keep the node —
/// and through it the mesh and material — alive.
struct Shown {
    node: NodeHandle,
    material: FaceMaterialHandle,
    id: HandleId,
    /// The shape the node's mesh was built from, so a sync can tell a move from
    /// a rebuild.
    shape: HandleShape,
    /// Kept so a grab can rebuild the drag locus without re-consulting the owner.
    drag: DragKind,
    anchor: Point3,
    direction: Vector3,
    color: RgbaColor,
}

/// Owns the scene side of a set of handles: their nodes, hover highlighting,
/// hit-testing, and the drag locus of whichever one is grabbed.
///
/// Handle nodes are parented under a single root drawn on the overlay layer at
/// a constant on-screen size, which they inherit.
pub struct HandleSet {
    /// Root for all handle geometry. Created when first needed.
    root: Option<NodeHandle>,
    shown: Vec<Shown>,
    highlighted: Option<HandleId>,
    screen_size: f32,
}

impl Default for HandleSet {
    fn default() -> Self {
        Self::new()
    }
}

impl HandleSet {
    pub fn new() -> Self {
        Self::with_screen_size(HANDLE_SCREEN_SIZE)
    }

    /// A set whose handles span `px` pixels instead of the default.
    pub fn with_screen_size(px: f32) -> Self {
        Self { root: None, shown: Vec::new(), highlighted: None, screen_size: px }
    }

    pub fn is_empty(&self) -> bool {
        self.shown.is_empty()
    }

    /// Nodes of the displayed handles, for callers that must exclude them from
    /// their own scene queries — snapping, most obviously, which would
    /// otherwise lock onto the grip being dragged.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.shown.iter().map(|h| h.node.id()).collect()
    }

    /// Brings the scene in line with `handles`.
    ///
    /// Called every frame, so the common case — the same handles at new
    /// placements — only rewrites transforms and colors. Geometry is rebuilt
    /// only when the ids or shapes themselves change.
    pub fn sync(&mut self, handles: &[Handle], scene: &Scene) {
        let mut scene = scene.lock();

        // A root held across a scene swap is stale, and so is everything under
        // it; drop the lot and rebuild against the new scene.
        if self.root.as_ref().is_some_and(|r| !scene.has_node(r.id())) {
            self.root = None;
            self.shown.clear();
            self.highlighted = None;
        }

        if !self.matches(handles) {
            self.rebuild(handles, &mut scene);
            return;
        }

        for (shown, handle) in self.shown.iter_mut().zip(handles) {
            if shown.anchor != handle.anchor || shown.direction != handle.direction {
                shown.anchor = handle.anchor;
                shown.direction = handle.direction;
                scene.set_node_transform(shown.node.id(), placement(handle));
            }
            shown.drag = handle.drag;
            // A highlighted handle wears the lit color; leave it alone or the
            // hover would flicker off on every move.
            if shown.color != handle.color {
                shown.color = handle.color;
                if self.highlighted != Some(shown.id)
                    && let Some(material) = scene.get_face_material_mut(shown.material.id())
                {
                    material.set_base_color_factor(handle.color);
                }
            }
        }
    }

    /// Whether the displayed handles can be moved into `handles`, or have to be
    /// rebuilt. Identity is the id and the shape: everything else is a write.
    fn matches(&self, handles: &[Handle]) -> bool {
        self.shown.len() == handles.len()
            && self
                .shown
                .iter()
                .zip(handles)
                .all(|(shown, handle)| shown.id == handle.id && shown.shape == handle.shape)
    }

    /// Discards the displayed handles and builds `handles` in their place.
    fn rebuild(&mut self, handles: &[Handle], scene: &mut SceneData) {
        for shown in self.shown.drain(..) {
            scene.remove_node(shown.node.id());
        }
        self.highlighted = None;

        if handles.is_empty() {
            return;
        }

        let root = self.ensure_root(scene);
        for handle in handles {
            // A custom shape is already a scene resource; the built-ins are
            // generated here and uploaded on first use.
            let mesh = match &handle.shape {
                HandleShape::Custom(mesh) => mesh.clone(),
                builtin => scene.add_mesh(shape::mesh(builtin)),
            };
            let material = scene.add_face_material(shape::material(handle.color));
            let node = scene
                .add_instance_node(
                    Some(root),
                    Instance::new(mesh).with_face_material(material.clone()),
                    None,
                    placement(handle),
                    // Pickable — that is how `pick` finds them — but kept out of
                    // the scene's bounds, which annotation must never affect.
                    NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING,
                )
                .expect("Failed to add handle node");

            self.shown.push(Shown {
                node,
                material,
                id: handle.id,
                shape: handle.shape.clone(),
                drag: handle.drag,
                anchor: handle.anchor,
                direction: handle.direction,
                color: handle.color,
            });
        }
    }

    /// The root all handles hang from, created on first use.
    fn ensure_root(&mut self, scene: &mut SceneData) -> NodeId {
        self.root
            .get_or_insert_with(|| {
                let root = scene
                    .add_node(None, Some("Handle root".to_owned()), Transform::IDENTITY,
                        NodeFlags::DOES_NOT_CONTRIBUTE_BOUNDING)
                    .expect("Failed to create handle root node");
                // Drawn over the scene at a constant on-screen size. Both
                // inherit down, so the handles themselves set neither.
                scene.set_node_display(
                    root.id(),
                    DisplayBehavior {
                        screen_size: Some(self.screen_size),
                        layer: RenderLayer::Overlay,
                        ..Default::default()
                    },
                );
                root
            })
            .id()
    }

    /// Removes every handle from the scene.
    pub fn clear(&mut self, scene: &Scene) {
        self.sync(&[], scene);
    }

    /// The handle `ray` hits, nearest first.
    ///
    /// Handles are drawn at a constant on-screen size, so the pick has to
    /// resolve them against that same camera-dependent transform through
    /// [`PickView`] — testing the authored unit geometry would miss entirely.
    pub fn pick(
        &self,
        ray: Ray,
        scene: &Scene,
        camera: &PositionedCamera,
        viewport: (u32, u32),
    ) -> Option<HandleId> {
        if self.is_empty() {
            return None;
        }
        let view = PickView { camera, viewport };
        let hits = pick_all_from_ray_with_view(&RayPickQuery::faces(ray), scene, Some(&view));

        hits.iter().find_map(|hit| {
            self.shown.iter().find(|s| s.node.id() == hit.node_id).map(|s| s.id)
        })
    }

    /// Lights one handle, or clears the highlight with `None`.
    pub fn set_highlight(&mut self, id: Option<HandleId>, scene: &Scene) {
        if self.highlighted == id {
            return;
        }
        let mut scene = scene.lock();

        // Restore whatever was lit, then light the new one.
        for (target, lit) in [(self.highlighted, false), (id, true)] {
            let Some(target) = target else { continue };
            let Some(shown) = self.shown.iter().find(|s| s.id == target) else { continue };
            let color = match lit {
                true => shown.color.lightened(Axis::HIGHLIGHT_LIGHTEN),
                false => shown.color,
            };
            if let Some(material) = scene.get_face_material_mut(shown.material.id()) {
                material.set_base_color_factor(color);
            }
        }

        self.highlighted = id;
    }

    /// The handle currently lit, if any.
    pub fn highlighted(&self) -> Option<HandleId> {
        self.highlighted
    }

    /// The locus a drag on `id` is confined to.
    pub fn drag_geometry(&self, id: HandleId, camera: &PositionedCamera) -> Option<DragGeometry> {
        let shown = self.shown.iter().find(|s| s.id == id)?;
        Some(match shown.drag {
            DragKind::Axis => DragGeometry::axis(shown.anchor, shown.direction),
            DragKind::Plane => DragGeometry::plane(shown.direction, shown.anchor),
            DragKind::View => DragGeometry::plane(camera.forward(), shown.anchor),
        })
    }

    /// The material backing a handle, for tests.
    #[cfg(test)]
    fn material_of(&self, id: HandleId) -> Option<duck_engine_scene::resource::FaceMaterialId> {
        self.shown.iter().find(|s| s.id == id).map(|s| s.material.id())
    }
}

/// Where a handle's node sits: at its anchor, with `+Y` — the axis every shape
/// is built along — turned onto its direction.
fn placement(handle: &Handle) -> Transform {
    let rotation = if handle.direction.magnitude2() < f32::EPSILON {
        Quaternion::from_sv(1.0, Vector3::new(0.0, 0.0, 0.0))
    } else {
        // `from_arc` picks an arbitrary perpendicular for an exact reversal,
        // which is all a rotationally symmetric shape needs.
        Quaternion::from_arc(Vector3::unit_y(), handle.direction.normalize(), None)
    };
    Transform {
        position: handle.anchor,
        rotation,
        scale: Vector3::new(1.0, 1.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::Rotation;
    use duck_engine_scene::Projection;

    const EPSILON: f32 = 1e-5;

    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
        }
    }

    fn handle(id: u32, shape: HandleShape, anchor: Point3) -> Handle {
        Handle::new(HandleId(id), shape, anchor)
    }

    #[test]
    fn sync_adds_a_handle_per_description() {
        let scene = Scene::default();
        let mut set = HandleSet::new();

        set.sync(
            &[
                handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0)),
                handle(1, HandleShape::Cube, Point3::new(0.0, 2.0, 0.0)),
            ],
            &scene,
        );

        assert_eq!(set.node_ids().len(), 2);
        assert!(!set.is_empty());
    }

    /// The per-frame poll must not churn the scene: the same handles at new
    /// placements keep their nodes, so nothing re-uploads to the GPU.
    #[test]
    fn sync_of_the_same_handles_reuses_their_nodes() {
        let scene = Scene::default();
        let mut set = HandleSet::new();

        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);
        let before = set.node_ids();

        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(5.0, 0.0, 0.0))], &scene);
        assert_eq!(set.node_ids(), before);

        // ...and the node actually moved.
        let moved = scene.lock().get_node(before[0]).unwrap().transform().position;
        assert!((moved.x - 5.0).abs() < EPSILON);
    }

    #[test]
    fn sync_with_a_changed_shape_rebuilds() {
        let scene = Scene::default();
        let mut set = HandleSet::new();

        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);
        let before = set.node_ids();

        set.sync(&[handle(0, HandleShape::Ball, Point3::new(1.0, 0.0, 0.0))], &scene);
        assert_ne!(set.node_ids(), before);
    }

    #[test]
    fn sync_with_a_changed_id_rebuilds() {
        let scene = Scene::default();
        let mut set = HandleSet::new();

        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);
        let before = set.node_ids();

        set.sync(&[handle(1, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);
        assert_ne!(set.node_ids(), before);
    }

    #[test]
    fn syncing_to_empty_clears_the_scene() {
        let scene = Scene::default();
        let mut set = HandleSet::new();

        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);
        let nodes = set.node_ids();

        set.clear(&scene);
        assert!(set.is_empty());
        assert!(set.node_ids().is_empty());
        for node in nodes {
            assert!(!scene.lock().has_node(node), "handle node outlived the set");
        }
    }

    /// A set held across a scene swap points at nodes that no longer exist;
    /// the next sync has to notice and rebuild rather than write into nothing.
    #[test]
    fn sync_after_a_scene_swap_rebuilds() {
        let scene = Scene::default();
        let mut set = HandleSet::new();
        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &scene);

        let fresh = Scene::default();
        set.sync(&[handle(0, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0))], &fresh);

        assert_eq!(set.node_ids().len(), 1);
        assert!(fresh.lock().has_node(set.node_ids()[0]));
    }

    #[test]
    fn highlighting_recolors_only_the_lit_handle() {
        let scene = Scene::default();
        let mut set = HandleSet::new();
        let base = RgbaColor { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
        set.sync(
            &[
                handle(0, HandleShape::Arrow, Point3::new(0.0, 0.0, 0.0)).with_color(base),
                handle(1, HandleShape::Arrow, Point3::new(1.0, 0.0, 0.0)).with_color(base),
            ],
            &scene,
        );

        set.set_highlight(Some(HandleId(0)), &scene);
        let lit = set.material_of(HandleId(0)).unwrap();
        let dim = set.material_of(HandleId(1)).unwrap();
        assert!(scene.lock().get_face_material(lit).unwrap().base_color_factor().r > base.r);
        assert_eq!(scene.lock().get_face_material(dim).unwrap().base_color_factor().r, base.r);

        // Clearing puts it back.
        set.set_highlight(None, &scene);
        assert_eq!(scene.lock().get_face_material(lit).unwrap().base_color_factor().r, base.r);
    }

    #[test]
    fn drag_geometry_follows_the_handles_drag_kind() {
        let scene = Scene::default();
        let mut set = HandleSet::new();
        set.sync(
            &[
                handle(0, HandleShape::Arrow, Point3::new(0.0, 0.0, 0.0))
                    .with_direction(Vector3::unit_x())
                    .with_drag(DragKind::Axis),
                handle(1, HandleShape::Quad, Point3::new(0.0, 0.0, 0.0))
                    .with_direction(Vector3::unit_y())
                    .with_drag(DragKind::Plane),
            ],
            &scene,
        );

        let camera = camera();
        assert!(matches!(
            set.drag_geometry(HandleId(0), &camera),
            Some(DragGeometry::Axis { .. })
        ));
        assert!(matches!(
            set.drag_geometry(HandleId(1), &camera),
            Some(DragGeometry::Plane(_))
        ));
        assert!(set.drag_geometry(HandleId(9), &camera).is_none());
    }

    /// Shapes are built along +Y, so the placement has to turn that onto the
    /// handle's direction or every arm would point the same way.
    #[test]
    fn placement_turns_y_onto_the_direction() {
        let h = handle(0, HandleShape::Arrow, Point3::new(0.0, 0.0, 0.0))
            .with_direction(Vector3::unit_x());
        let turned = placement(&h).rotation * Vector3::unit_y();
        assert!((turned - Vector3::unit_x()).magnitude() < EPSILON);
    }

    #[test]
    fn placement_of_a_degenerate_direction_does_not_produce_nans() {
        let h = handle(0, HandleShape::Ball, Point3::new(1.0, 2.0, 3.0))
            .with_direction(Vector3::new(0.0, 0.0, 0.0));
        let turned = placement(&h).rotation * Vector3::unit_y();
        assert!(turned.magnitude().is_finite());
    }
}
