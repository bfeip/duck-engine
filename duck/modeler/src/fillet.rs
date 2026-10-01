use anyhow::{bail, ensure, Context, Result};
use duck_engine_scene::common::{Point3, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::Shape;
use opencascade::FilletError;

use crate::document::{dvec3_to_point3, dvec3_to_vec3, unwrap_single_solid, Document};

/// A size at or below this is degenerate: there is nothing to round or bevel.
const MIN_SIZE: Real = 1e-6;

/// Two face normals summing to less than this fold back onto each other,
/// leaving no corner to bisect.
const MIN_NORMAL_SUM: f64 = 1e-6;

/// Whether edges are rounded or bevelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendKind {
    /// A round of constant radius.
    Fillet,
    /// A flat bevel, cut back the same distance along both faces.
    Chamfer,
}

impl BlendKind {
    /// Name of the operation, for the panel and the undo step.
    pub fn name(self) -> &'static str {
        match self {
            BlendKind::Fillet => "Fillet",
            BlendKind::Chamfer => "Chamfer",
        }
    }
}

/// The edges being blended, all on one part, identified the way the selection
/// system reports them: by tessellation order within the part's mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilletTarget {
    pub node: NodeId,
    /// Edge indices, the primary edge first.
    pub edges: Vec<u32>,
}

/// Where the grip rides: through the primary edge, along the line bisecting the
/// corner there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilletFrame {
    /// The primary edge's parametric midpoint.
    pub apex: Point3,
    /// Unit bisector of the two faces' outward normals at the apex, pointing out
    /// of the corner into the space beside it — for a concave edge as for a
    /// convex one.
    pub outward: Vector3,
}

impl FilletFrame {
    /// Resolves the frame at edge `edge_index` of the part at `node`.
    pub fn new(doc: &Document, node: NodeId, edge_index: u32) -> Result<Self> {
        let part = doc
            .part_for_node(node)
            .and_then(|part| doc.get_part(part))
            .context("Fillet target is not a known CAD part")?;
        let edge = part
            .shape
            .edge_at(edge_index as usize)
            .context("Selected edge is not part of a known CAD part")?;
        let faces = part.shape.adjacent_faces(&edge);
        let [first, second] = faces.as_slice() else {
            bail!("Only an edge between two faces can be filleted or chamfered");
        };

        let apex = edge.midpoint();
        // `normal_at` goes through `BRepGProp_Face::Normal`, which applies each
        // face's orientation: both normals point out of the material.
        let sum = first.normal_at(apex)? + second.normal_at(apex)?;
        ensure!(sum.length() > MIN_NORMAL_SUM, "The faces at this edge fold back onto each other");

        Ok(Self { apex: dvec3_to_point3(apex), outward: dvec3_to_vec3(sum.normalize()) })
    }
}

/// A fillet or chamfer as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilletParams {
    pub frame: FilletFrame,
    pub kind: BlendKind,
    /// A fillet's radius, or how far a chamfer cuts back along each face.
    /// Never negative.
    pub size: Real,
}

impl FilletParams {
    /// A zero-size fillet on `frame`.
    pub fn new(frame: FilletFrame) -> Self {
        Self { frame, kind: BlendKind::Fillet, size: 0.0 }
    }

    /// The size, signed by kind: positive for a fillet, negative for a chamfer.
    pub fn signed_size(&self) -> Real {
        match self.kind {
            BlendKind::Fillet => self.size,
            BlendKind::Chamfer => -self.size,
        }
    }

    /// Sets the size from a signed one: positive makes a fillet and negative a
    /// chamfer, while exactly zero keeps the kind.
    pub fn set_signed_size(&mut self, signed: Real) {
        if signed > 0.0 {
            self.kind = BlendKind::Fillet;
        } else if signed < 0.0 {
            self.kind = BlendKind::Chamfer;
        }
        self.size = signed.abs();
    }

    /// Where the grip sits: out of the corner by a fillet's radius, into the
    /// part by a chamfer's distance.
    pub fn grip(&self) -> Point3 {
        self.frame.apex + self.frame.outward * self.signed_size()
    }

    /// Whether the blend is too small to build.
    pub fn is_degenerate(&self) -> bool {
        self.size <= MIN_SIZE
    }
}

/// The target's part with its edges rounded or bevelled.
pub fn build_fillet(doc: &Document, target: &FilletTarget, params: &FilletParams) -> Result<Shape> {
    ensure!(!params.is_degenerate(), "Nothing to {} at zero size", params.kind.name().to_lowercase());
    let part = doc
        .part_for_node(target.node)
        .and_then(|part| doc.get_part(part))
        .context("Fillet target is not a known CAD part")?;

    // OCCT's blend raises the tolerance of vertices it touches on its input, so
    // it works on a copy: an abandoned preview must leave the part as it was.
    let body = part.shape.deep_copy();
    let edges = target
        .edges
        .iter()
        .map(|&index| {
            body.edge_at(index as usize)
                .with_context(|| format!("Selected edge {index} is not part of the part"))
        })
        .collect::<Result<Vec<_>>>()?;

    let size = f64::from(params.size);
    let blended = match params.kind {
        BlendKind::Fillet => body.fillet_edges(size, &edges),
        BlendKind::Chamfer => body.chamfer_edges(size, &edges),
    }
    .map_err(|error| {
        let hint = failure_hint(params.kind, &error);
        anyhow::Error::new(error).context(hint)
    })?;

    // The blend comes back wrapped in a compound; a lone solid stays a solid.
    Ok(unwrap_single_solid(blended))
}

/// A plain-language account of why a blend failed, to lead the kernel's reason.
fn failure_hint(kind: BlendKind, error: &opencascade::Error) -> String {
    let size = match kind {
        BlendKind::Fillet => "radius",
        BlendKind::Chamfer => "distance",
    };
    match error {
        opencascade::Error::FilletFailed(FilletError::NoSuitableEdges) => {
            "Only an edge between two faces can be filleted or chamfered".to_owned()
        }
        opencascade::Error::FilletFailed(FilletError::InvalidResult) => {
            format!("The {size} is too large: neighbouring blends overlap")
        }
        _ => format!("The {size} may be too large for the faces beside these edges"),
    }
}

/// Apply the blend: rebuild the target's part with it, in place, as one undo
/// step.
pub fn execute_fillet(doc: &mut Document, target: &FilletTarget, params: &FilletParams) -> Result<()> {
    let shape = build_fillet(doc, target, params)?;
    let part = doc.part_for_node(target.node).context("Fillet target is not a known CAD part")?;
    doc.reshape_part(part, shape, params.kind.name())
}



#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::consts;
    use duck_engine_scene::common::{EuclideanSpace, InnerSpace};
    use duck_engine_scene::Scene;
    use opencascade::primitives::{Face, Wire};

    use crate::document::PartKind;

    const EPSILON: Real = 1e-5;

    fn doc_with_shape(shape: Shape) -> (Document, NodeId) {
        let mut doc = Document::new(Scene::default());
        let part = doc
            .add_part("part", shape, &CadTessellationOptions::default())
            .expect("shape tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (doc, node)
    }

    /// A 2×2×2 box centred on the origin.
    fn doc_with_box() -> (Document, NodeId) {
        doc_with_shape(Shape::box_centered(2.0, 2.0, 2.0))
    }

    /// An L-shaped prism one unit tall, notched out of a 2×2 square at its
    /// `(1..2, 1..2)` corner, so the notch's inner edge is concave.
    fn l_prism() -> Shape {
        let corners = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
        let wire = Wire::from_ordered_points(corners.iter().map(|&(x, y)| glam::dvec3(x, y, 0.0)))
            .expect("wire builds");
        let face = Face::from_wire(&wire).expect("face builds");
        face.extrude(glam::dvec3(0.0, 0.0, 1.0)).into()
    }

    /// The index of the first edge whose midpoint is `point`.
    fn edge_at_point(doc: &Document, node: NodeId, point: glam::DVec3) -> u32 {
        let part = doc.get_part(doc.part_for_node(node).unwrap()).unwrap();
        let index = part
            .shape
            .edges()
            .position(|edge| edge.midpoint().distance(point) < 1e-9)
            .expect("an edge runs through the point");
        index as u32
    }

    fn target(node: NodeId, edges: &[u32]) -> FilletTarget {
        FilletTarget { node, edges: edges.to_vec() }
    }

    fn params(doc: &Document, node: NodeId, edge: u32, kind: BlendKind, size: Real) -> FilletParams {
        let frame = FilletFrame::new(doc, node, edge).expect("edge resolves");
        FilletParams { kind, size, ..FilletParams::new(frame) }
    }

    fn part_volume(doc: &Document, node: NodeId) -> f64 {
        doc.get_part(doc.part_for_node(node).unwrap()).unwrap().shape.volume()
    }

    /// Every edge of a box centred on the origin has its midpoint at two unit
    /// coordinates and a zero, and its corner bisects straight away from the
    /// centre through it.
    #[test]
    fn a_box_edge_frame_sits_on_the_edge_and_points_out_of_the_corner() {
        let (doc, node) = doc_with_box();
        let edge_count = doc.get_part(doc.part_for_node(node).unwrap()).unwrap().shape.edges().count();

        for edge in 0..edge_count as u32 {
            let frame = FilletFrame::new(&doc, node, edge).expect("box edge resolves");
            let from_centre = frame.apex.to_vec();
            assert!((from_centre.magnitude() - consts::SQRT_2).abs() < EPSILON, "edge {edge}: apex off the edge");
            assert!((frame.outward.magnitude() - 1.0).abs() < EPSILON, "edge {edge}: outward not unit");
            assert!(
                (frame.outward - from_centre.normalize()).magnitude() < EPSILON,
                "edge {edge}: outward {:?} does not bisect the corner at {:?}",
                frame.outward,
                frame.apex
            );
        }
    }

    /// Both faces at the notch's inner edge face into the notch, so their
    /// bisector points into the empty corner rather than into the material.
    #[test]
    fn a_concave_edge_frame_points_into_the_empty_corner() {
        let (doc, node) = doc_with_shape(l_prism());
        let edge = edge_at_point(&doc, node, glam::dvec3(1.0, 1.0, 0.5));

        let frame = FilletFrame::new(&doc, node, edge).expect("inner edge resolves");
        let into_notch = Vector3::new(1.0, 1.0, 0.0).normalize();
        assert!((frame.outward - into_notch).magnitude() < EPSILON, "got {:?}", frame.outward);
    }

    #[test]
    fn a_sheet_edge_is_refused() {
        let wire = Wire::rect(2.0, 2.0).expect("rectangle builds");
        let sheet: Shape = Face::from_wire(&wire).expect("face builds").into();
        let (doc, node) = doc_with_shape(sheet);

        let Err(error) = FilletFrame::new(&doc, node, 0) else {
            panic!("a sheet's edge borders only one face");
        };
        assert!(format!("{error:#}").contains("between two faces"), "got {error:#}");
    }

    #[test]
    fn the_signed_size_picks_the_kind() {
        let (doc, node) = doc_with_box();
        let mut params = params(&doc, node, 0, BlendKind::Fillet, 0.5);
        assert_eq!(params.signed_size(), 0.5);

        params.set_signed_size(-0.25);
        assert_eq!(params.kind, BlendKind::Chamfer);
        assert_eq!(params.size, 0.25);
        assert_eq!(params.signed_size(), -0.25);

        // At exactly zero there is no sign to go by, so the kind stays.
        params.set_signed_size(0.0);
        assert_eq!(params.kind, BlendKind::Chamfer);
        assert!(params.is_degenerate());
    }

    /// The grip rides out of the corner for a fillet and into the part for a
    /// chamfer, by the size either way.
    #[test]
    fn the_grip_follows_the_signed_size() {
        let (doc, node) = doc_with_box();
        let fillet = params(&doc, node, 0, BlendKind::Fillet, 0.5);
        let chamfer = FilletParams { kind: BlendKind::Chamfer, ..fillet };
        let (apex, outward) = (fillet.frame.apex, fillet.frame.outward);

        assert!((fillet.grip() - (apex + outward * 0.5)).magnitude() < EPSILON);
        assert!((chamfer.grip() - (apex - outward * 0.5)).magnitude() < EPSILON);
    }

    /// A rounded edge of the box loses a groove of r² less a quarter circle,
    /// all along its length of 2; a bevelled one a right-angled prism with legs
    /// d. Both stay solids, not the compound OCCT wraps them in.
    #[test]
    fn a_box_edge_rounds_and_bevels_to_the_expected_volume() {
        let (doc, node) = doc_with_box();
        let size = 0.5;
        let cases = [
            (BlendKind::Fillet, 8.0 - 2.0 * size * size * (1.0 - std::f64::consts::FRAC_PI_4)),
            (BlendKind::Chamfer, 8.0 - size * size),
        ];
        for (kind, expected) in cases {
            let params = params(&doc, node, 0, kind, size as Real);
            let shape = build_fillet(&doc, &target(node, &[0]), &params).expect("the blend builds");
            assert_eq!(shape.shape_type(), opencascade::primitives::ShapeType::Solid, "{kind:?}");
            assert!((shape.volume() - expected).abs() < 1e-5, "{kind:?}: expected {expected}, got {}", shape.volume());
        }
    }

    /// Both occurrences of a shared edge (one per face) may be selected; the
    /// edge is blended once.
    #[test]
    fn both_occurrences_of_an_edge_blend_it_once() {
        let (doc, node) = doc_with_box();
        let first = 0;
        let edge = doc.edge_subshape(node, first).unwrap();
        let part = doc.get_part(doc.part_for_node(node).unwrap()).unwrap();
        let second = part
            .shape
            .edges()
            .enumerate()
            .find(|(index, other)| *index != first as usize && other.is_same(&edge))
            .map(|(index, _)| index as u32)
            .expect("a box edge borders two faces");

        let params = params(&doc, node, first, BlendKind::Chamfer, 0.5);
        let shape = build_fillet(&doc, &target(node, &[first, second]), &params).expect("the blend builds");
        assert!((shape.volume() - (8.0 - 0.25)).abs() < 1e-5, "got {}", shape.volume());
    }

    #[test]
    fn an_oversized_fillet_is_an_error_and_leaves_the_part_alone() {
        let (doc, node) = doc_with_box();
        let params = params(&doc, node, 0, BlendKind::Fillet, 5.0);

        let Err(error) = build_fillet(&doc, &target(node, &[0]), &params) else {
            panic!("a radius wider than the faces cannot be built");
        };
        assert!(format!("{error:#}").contains("too large"), "got {error:#}");
        assert!((part_volume(&doc, node) - 8.0).abs() < 1e-9);
    }

    #[test]
    fn a_zero_size_blend_is_refused() {
        let (doc, node) = doc_with_box();
        let params = params(&doc, node, 0, BlendKind::Fillet, 0.0);
        assert!(build_fillet(&doc, &target(node, &[0]), &params).is_err());
    }

    /// Edges are resolved on a deep copy by position, which only works if the
    /// copy explores its edges in the original's order.
    #[test]
    fn a_deep_copy_keeps_the_edge_order() {
        let bored = Shape::box_centered(4.0, 4.0, 4.0)
            .subtract(&Shape::cylinder_centered(glam::DVec3::ZERO, 1.0, glam::DVec3::Z, 6.0))
            .expect("the bore cuts")
            .shape;
        for shape in [Shape::box_centered(2.0, 2.0, 2.0), l_prism(), bored] {
            let copy = shape.deep_copy();
            let original: Vec<_> = shape.edges().map(|edge| edge.midpoint()).collect();
            let copied: Vec<_> = copy.edges().map(|edge| edge.midpoint()).collect();
            assert_eq!(original.len(), copied.len());
            for (index, (a, b)) in original.iter().zip(&copied).enumerate() {
                assert!(a.distance(*b) < 1e-12, "edge {index} moved in the copy: {a} vs {b}");
            }
        }
    }

    /// Applying a fillet reshapes the part where it stands, as one undo step
    /// that undo and redo replay.
    #[test]
    fn execute_reshapes_in_place_as_one_undo_step() {
        let (mut doc, node) = doc_with_box();
        let part = doc.part_for_node(node).unwrap();
        let params = params(&doc, node, 0, BlendKind::Fillet, 0.5);
        let filleted = 8.0 - 0.5 * (1.0 - std::f64::consts::FRAC_PI_4);

        execute_fillet(&mut doc, &target(node, &[0]), &params).expect("the fillet applies");
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.node_for_part(part), Some(node), "the part keeps its node");
        assert_eq!(doc.get_part(part).unwrap().kind(), PartKind::Solid);
        assert!((part_volume(&doc, node) - filleted).abs() < 1e-5);
        assert_eq!(doc.undo_label(), Some("Fillet"));

        doc.undo().expect("undo the fillet");
        assert!((part_volume(&doc, node) - 8.0).abs() < 1e-9);

        doc.redo().expect("redo the fillet");
        assert!((part_volume(&doc, node) - filleted).abs() < 1e-5);
    }

    #[test]
    fn a_chamfer_is_its_own_undo_step() {
        let (mut doc, node) = doc_with_box();
        let params = params(&doc, node, 0, BlendKind::Chamfer, 0.5);

        execute_fillet(&mut doc, &target(node, &[0]), &params).expect("the chamfer applies");
        assert_eq!(doc.undo_label(), Some("Chamfer"));
    }
}
