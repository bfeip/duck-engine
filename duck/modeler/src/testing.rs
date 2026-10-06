//! Fixtures shared by the modeler's tests.

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::resource::NodeId;
use duck_engine_scene::Scene;
use glam::DVec3;
use opencascade::primitives::{Shape, Shell, Wire};

use crate::document::{Document, PartId, SourceFate};

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
