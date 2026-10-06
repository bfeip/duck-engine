//! Fixtures shared by the modeler's tests.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::common::{Point3, Vector3};
use duck_engine_scene::resource::{NodeId, SubGeometryElement, SubGeometryKind, Visibility};
use duck_engine_scene::Scene;
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers, PhysicalKey};
use duck_engine_viewer::operator::{HandleDrag, HandleId};
use duck_engine_viewer::selection::SelectionItem;
use glam::DVec3;
use opencascade::primitives::{Shape, Shell, Wire};

use crate::construction::ConstructionOptions;
use crate::document::{Document, PartId, SourceFate};
use crate::notifications::Notifications;
use crate::tools::Workspace;

/// A document holding `shape` as a part named "part", and the part's node.
pub fn doc_with_shape(shape: Shape) -> (Document, NodeId) {
    let mut doc = Document::new(Scene::default());
    let part = doc
        .add_part("part", shape, &CadTessellationOptions::default())
        .expect("shape tessellates");
    let node = doc.node_for_part(part).expect("part has a node");
    (doc, node)
}

/// A document holding a 2×2×2 box centred on the origin, and its node.
pub fn doc_with_box() -> (Document, NodeId) {
    doc_with_shape(Shape::box_centered(2.0, 2.0, 2.0))
}

/// A document holding `count` 2-unit cubes named "box 0", "box 1", ….
pub fn doc_with_boxes(count: usize) -> (Document, Vec<(PartId, NodeId)>) {
    let mut doc = Document::new(Scene::default());
    let parts = (0..count)
        .map(|i| {
            let part = doc
                .add_part(format!("box {i}"), Shape::cube(2.0), &CadTessellationOptions::default())
                .expect("box tessellates");
            (part, doc.node_for_part(part).expect("part has a node"))
        })
        .collect();
    (doc, parts)
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

/// The shape of the part at `node`.
pub fn part_shape(doc: &Document, node: NodeId) -> &Shape {
    &doc.part_at(node).expect("node is a part").shape
}

/// The volume of the part at `node`.
pub fn part_volume(doc: &Document, node: NodeId) -> f64 {
    part_shape(doc, node).volume()
}

/// The volume of every part, in document order.
pub fn volumes(doc: &Document) -> Vec<f64> {
    doc.parts().map(|part| part.shape.volume()).collect()
}

/// The index of the face of the part at `node` whose outward normal at its
/// centre is `normal`.
pub fn face_along(doc: &Document, node: NodeId, normal: DVec3) -> u32 {
    part_shape(doc, node)
        .faces()
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
    let part = doc.add_part("cube", cube, &CadTessellationOptions::default()).expect("cube tessellates");
    let other = doc.node_for_part(part).expect("part has a node");
    (workspace(doc), main, other)
}

/// The visibility of `node` in the workspace's scene.
pub fn visibility(ws: &Workspace, node: NodeId) -> Visibility {
    let scene = ws.document.lock().unwrap().scene().clone();
    let visibility = scene.lock().get_node(node).expect("node exists").visibility();
    visibility
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

/// A drag of grip `id` by `offset` from where it was grabbed, as the handle
/// machinery reports it.
pub fn drag(id: HandleId, grab: Point3, offset: Vector3) -> HandleDrag {
    HandleDrag { id, grab, point: grab + offset, modifiers: Modifiers::default() }
}
