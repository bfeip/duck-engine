//! Fixtures shared by the modeler's tests.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::common::{
    InnerSpace, Matrix4, Plane, Point3, Ray, Real, RgbaColor, Transform, Vector3,
};
use duck_engine_scene::resource::{NodeId, SubGeometryElement, SubGeometryKind, Visibility};
use duck_engine_scene::Scene;
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers, NamedKey, PhysicalKey};
use duck_engine_viewer::operator::{HandleDrag, HandleId};
use duck_engine_viewer::selection::SelectionItem;
use glam::DVec3;
use opencascade::primitives::{Face, Shape, Shell, Wire};

use crate::construction::ConstructionOptions;
use crate::document::{unify_same_domain, unwrap_single_solid, Document, PartId, SourceFate};
use crate::notifications::Notifications;
use crate::ops::loft::LoftProfile;
use crate::snap::{Snap, SnapKind};
use crate::tools::{Pointer, Workspace};

/// A document holding `shape` as a part named "part", and the part's node.
pub fn doc_with_shape(shape: Shape) -> (Document, NodeId) {
    let mut doc = Document::new(Scene::default());
    let part = doc
        .add_part("part", shape, Transform::IDENTITY, &CadTessellationOptions::default())
        .expect("shape tessellates");
    let node = doc.node_for_part(part).expect("part has a node");
    (doc, node)
}

/// A document holding a 2×2×2 box centred on the origin, and its node.
pub fn doc_with_box() -> (Document, NodeId) {
    doc_with_shape(Shape::box_centered(2.0, 2.0, 2.0))
}

/// A document holding a radius-1, height-4 cylinder standing on the origin
/// along +Y, framed by the world, and its part and node.
pub fn doc_with_cylinder() -> (Document, PartId, NodeId) {
    let (doc, node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 1.0, DVec3::Y, 4.0));
    let part = doc.part_for_node(node).expect("node is a part");
    (doc, part, node)
}

/// Asserts the rigid frame `actual` is the placement `expected`.
pub fn assert_frame(actual: Transform, expected: &Matrix4) {
    const EPSILON: Real = 1e-6;
    let actual = actual.to_matrix();
    let close = (0..4).all(|c| (0..4).all(|r| (actual[c][r] - expected[c][r]).abs() < EPSILON));
    assert!(close, "frame {actual:?} is not {expected:?}");
}

/// The extent of `shape`'s tessellation along each world axis.
pub fn mesh_extents(shape: &Shape) -> DVec3 {
    let vertices = shape.mesh().expect("shape meshes").vertices;
    let min = vertices.iter().copied().reduce(DVec3::min).expect("mesh has vertices");
    let max = vertices.iter().copied().reduce(DVec3::max).expect("mesh has vertices");
    max - min
}

/// A document holding `count` 2-unit cubes named "box 0", "box 1", ….
pub fn doc_with_boxes(count: usize) -> (Document, Vec<(PartId, NodeId)>) {
    let mut doc = Document::new(Scene::default());
    let parts = (0..count)
        .map(|i| {
            let part = doc
                .add_part(format!("box {i}"), Shape::cube(2.0), Transform::IDENTITY, &CadTessellationOptions::default())
                .expect("box tessellates");
            (part, doc.node_for_part(part).expect("part has a node"))
        })
        .collect();
    (doc, parts)
}

/// A document holding the six faces of a 2-unit cube on the origin, each a
/// part of its own named "Face-001"…, and their nodes.
pub fn doc_with_box_faces() -> (Document, Vec<NodeId>) {
    let mut doc = Document::new(Scene::default());
    let nodes = Shape::cube(2.0)
        .faces()
        .map(|face| {
            let face = Shape::from(face).deep_copy();
            let part = doc
                .add_numbered_part("Face", face, Transform::IDENTITY, &CadTessellationOptions::default())
                .expect("face tessellates");
            doc.node_for_part(part).expect("part has a node")
        })
        .collect();
    (doc, nodes)
}

/// A document holding a 2-unit cube on the origin and a unit sphere on its far
/// corner, and their nodes.
pub fn doc_with_box_and_sphere() -> (Document, NodeId, NodeId) {
    let mut doc = Document::new(Scene::default());
    let options = CadTessellationOptions::default();
    let cube = doc.add_part("box", Shape::cube(2.0), Transform::IDENTITY, &options).expect("box tessellates");
    let sphere = Shape::sphere(1.0).at(DVec3::splat(2.0)).build();
    let sphere = doc.add_part("sphere", sphere, Transform::IDENTITY, &options).expect("sphere tessellates");
    let cube = doc.node_for_part(cube).expect("part has a node");
    let sphere = doc.node_for_part(sphere).expect("part has a node");
    (doc, cube, sphere)
}

/// A document holding a radius-2 ball with a bite out of its side, as a boolean
/// leaves it, and its node. The ball's seam runs through the bite and splits its
/// rim, a circle, in two.
pub fn doc_with_bitten_ball() -> (Document, NodeId) {
    let ball = Shape::sphere(2.0).build();
    let bite = Shape::sphere(1.0).at(DVec3::new(2.0, 0.0, 0.3)).build();
    let bitten = ball.subtract(&bite).expect("the bite cuts").shape;
    doc_with_shape(unify_same_domain(unwrap_single_solid(bitten)))
}

/// The selection index of the bitten ball's rim, the ball's one edge.
pub fn bite_rim(doc: &Document, node: NodeId) -> u32 {
    let edges = doc.elements(node).expect("the ball is a part").edges().len();
    assert_eq!(edges, 1, "the ball has an edge besides its bite's rim");
    0
}

/// A document holding two unit square faces, one 2 above the other, and an
/// edge of each as a loft profile.
pub fn doc_with_two_squares() -> (Document, [LoftProfile; 2]) {
    let mut doc = Document::new(Scene::default());
    let mut square = |name: &str, y: f64| {
        let outline = Wire::from_ordered_points([
            DVec3::new(0.0, y, 0.0),
            DVec3::new(1.0, y, 0.0),
            DVec3::new(1.0, y, 1.0),
            DVec3::new(0.0, y, 1.0),
        ])
        .expect("square builds");
        let face = Shape::from(Face::from_wire(&outline).expect("face builds"));
        let options = CadTessellationOptions::default();
        let part = doc.add_part(name, face, Transform::IDENTITY, &options).expect("square tessellates");
        LoftProfile { node: doc.node_for_part(part).expect("part has a node"), edge_index: 0 }
    };
    let profiles = [square("lower", 0.0), square("upper", 2.0)];
    (doc, profiles)
}

/// A flat region through `corners`, in the XY plane.
pub fn xy_region(corners: &[(f64, f64)]) -> Shape {
    let outline = Wire::from_ordered_points(corners.iter().map(|&(x, y)| DVec3::new(x, y, 0.0)))
        .expect("outline builds");
    Face::from_wire(&outline).expect("region builds").into()
}

/// An open square tube lofted between unit squares two apart: four faces of
/// one sheet.
pub fn tube() -> Shape {
    let square = |y: f64| {
        Wire::from_ordered_points([
            DVec3::new(-0.5, y, -0.5),
            DVec3::new(0.5, y, -0.5),
            DVec3::new(0.5, y, 0.5),
            DVec3::new(-0.5, y, 0.5),
        ])
        .expect("square builds")
    };
    Shell::loft([square(0.0), square(2.0)]).into()
}

/// A plane through `origin` aligned with no world axis, so a mistaken basis
/// shows up.
pub fn skewed_plane(origin: Point3) -> Plane {
    Plane::from_point(Vector3::new(1.0, 2.0, 3.0).normalize(), origin)
}

/// The shape of the part at `node`.
pub fn part_shape(doc: &Document, node: NodeId) -> &Shape {
    doc.part_at(node).expect("node is a part").shape()
}

/// The faces of the part at `node`, numbered as its selections number them.
pub fn part_faces(doc: &Document, node: NodeId) -> &[Face] {
    doc.elements(node).expect("node is a part").faces()
}

/// The volume of the part at `node`.
pub fn part_volume(doc: &Document, node: NodeId) -> f64 {
    part_shape(doc, node).volume()
}

/// The volume of every part, in document order.
pub fn volumes(doc: &Document) -> Vec<f64> {
    doc.parts().map(|part| part.shape().volume()).collect()
}

/// The index of the face of the part at `node` whose outward normal at its
/// centre is `normal`.
pub fn face_along(doc: &Document, node: NodeId, normal: DVec3) -> u32 {
    part_faces(doc, node)
        .iter()
        .position(|face| face.normal_at_center().is_ok_and(|n| n.normalize().distance(normal) < 1e-6))
        .expect("a face matches") as u32
}

/// Volume of a frustum between squares of side `a` and `b`, `height` apart.
pub fn square_frustum(a: f64, b: f64, height: f64) -> f64 {
    height / 3.0 * (a * a + b * b + a * b)
}

/// Commits `result`, grown from the part at `node`, as `fate` says, under the
/// label "Test" and, for a new part, the name series "Result".
pub fn commit(doc: &mut Document, node: NodeId, result: Shape, fate: SourceFate) -> anyhow::Result<()> {
    let part = doc.part_for_node(node).expect("node is a part");
    doc.commit_result(part, result, fate, "Test", "Result", &CadTessellationOptions::default())
}

/// A workspace on `doc`, with default construction settings.
pub fn workspace(doc: Document) -> Workspace {
    Workspace {
        document: Arc::new(Mutex::new(doc)),
        construction: Rc::new(RefCell::new(ConstructionOptions::new())),
        notifications: Notifications::default(),
    }
}

/// A workspace on [`doc_with_shape`]'s document, and the part's node.
pub fn workspace_with(shape: Shape) -> (Workspace, NodeId) {
    let (doc, node) = doc_with_shape(shape);
    (workspace(doc), node)
}

/// A workspace on [`doc_with_box`]'s document, and the box's node.
pub fn workspace_with_box() -> (Workspace, NodeId) {
    workspace_with(Shape::box_centered(2.0, 2.0, 2.0))
}

/// A workspace on a document holding [`doc_with_box`]'s box and a unit cube
/// off to one side, and their nodes.
pub fn workspace_with_box_and_cube() -> (Workspace, NodeId, NodeId) {
    let (mut doc, main) = doc_with_box();
    let cube = Shape::box_from_corners(DVec3::splat(3.0), DVec3::splat(4.0));
    let part = doc.add_part("cube", cube, Transform::IDENTITY, &CadTessellationOptions::default()).expect("cube tessellates");
    let other = doc.node_for_part(part).expect("part has a node");
    (workspace(doc), main, other)
}

/// The visibility of `node` in the workspace's scene.
pub fn visibility(ws: &Workspace, node: NodeId) -> Visibility {
    let scene = ws.document.lock().unwrap().scene().clone();
    let visibility = scene.lock().get_node(node).expect("node exists").visibility();
    visibility
}

/// Asserts `actual` matches `expected` in every channel.
pub fn assert_color(actual: RgbaColor, expected: RgbaColor) {
    const EPSILON: f32 = 1e-6;
    let close = (actual.r - expected.r).abs() < EPSILON
        && (actual.g - expected.g).abs() < EPSILON
        && (actual.b - expected.b).abs() < EPSILON
        && (actual.a - expected.a).abs() < EPSILON;
    assert!(close, "expected color {expected:?}, got {actual:?}");
}

/// The volume of the part at `node` in the workspace's document.
pub fn volume(ws: &Workspace, node: NodeId) -> f64 {
    part_volume(&ws.document.lock().unwrap(), node)
}

/// Face `index` of the part at `node`, as a selection item.
pub fn face_item(node: NodeId, index: u32) -> SelectionItem {
    SelectionItem::SubGeometry { node_id: node, element: SubGeometryElement::new(SubGeometryKind::Face, index) }
}

/// Edge `index` of the part at `node`, as a selection item.
pub fn edge_item(node: NodeId, index: u32) -> SelectionItem {
    SelectionItem::SubGeometry { node_id: node, element: SubGeometryElement::new(SubGeometryKind::Edge, index) }
}

/// The face of the part at `node` whose outward normal is `normal`, as a
/// selection item.
pub fn face_item_along(ws: &Workspace, node: NodeId, normal: DVec3) -> SelectionItem {
    face_item(node, face_along(&ws.document.lock().unwrap(), node, normal))
}

/// A press of the unmodified character key `c`.
pub fn key(c: char) -> KeyEvent {
    KeyEvent {
        physical_key: PhysicalKey::Unidentified,
        logical_key: Key::Character(c),
        state: ElementState::Pressed,
        repeat: false,
    }
}

/// A press of the named key `named`, such as Delete.
pub fn named_key(named: NamedKey) -> KeyEvent {
    KeyEvent {
        physical_key: PhysicalKey::Unidentified,
        logical_key: Key::Named(named),
        state: ElementState::Pressed,
        repeat: false,
    }
}

/// A pointer snapped to `at` on the construction plane, its ray passing level
/// through it.
pub fn pointer_at(at: Point3) -> Pointer {
    Pointer {
        snap: Some(Snap { position: at, direction: None, kind: SnapKind::ConstructionPlane }),
        ray: Ray::new(at + Vector3::unit_x() * 10.0, -Vector3::unit_x()),
    }
}

/// A drag of grip `id` by `offset` from where it was grabbed, as the handle
/// machinery reports it.
pub fn drag(id: HandleId, grab: Point3, offset: Vector3) -> HandleDrag {
    HandleDrag { id, grab, point: grab + offset, modifiers: Modifiers::default() }
}
