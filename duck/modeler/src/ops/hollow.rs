use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::{InnerSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use glam::DVec3;
use opencascade::primitives::{Face, JoinType, Shape, ShapeType};
use opencascade::OffsetError;

use crate::document::{
    dvec3_to_point3, dvec3_to_vec3, has_solid, selected_face, unwrap_single_solid, Document,
};

/// A wall at or below this thickness is null to OCCT's offset.
const MIN_THICKNESS: Real = 1e-3;

/// The solid being hollowed and the faces opened in it, identified the way the
/// selection system reports them: by tessellation order within the part's mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HollowTarget {
    pub node: NodeId,
    /// The faces to remove, the primary first. None hollows the part closed.
    pub faces: Vec<u32>,
}

impl HollowTarget {
    /// Whether the walls close around a void, with no faces opened.
    pub fn is_closed(&self) -> bool {
        self.faces.is_empty()
    }
}

/// Where the wall's grip measures from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HollowFrame {
    /// A point on the part's surface: on the rim of the primary opening, or in
    /// the middle of the largest face of a closed hollow.
    pub anchor: Point3,
    /// Unit outward normal of the surface at the anchor.
    pub normal: Vector3,
}

impl HollowFrame {
    pub fn new(doc: &Document, target: &HollowTarget) -> Result<Self> {
        let shape = &doc.part_at(target.node).context("Hollow target is not a known CAD part")?.shape;
        ensure!(has_solid(shape), "Only a solid can be hollowed");
        let removed = target.faces.iter().map(|&index| selected_face(shape, index)).collect::<Result<Vec<_>>>()?;

        let (anchor, normal) = match removed.first().and_then(|primary| rim(shape, primary, &removed)) {
            Some(rim) => rim,
            None => largest_face_middle(shape)?,
        };
        Ok(Self { anchor: dvec3_to_point3(anchor), normal: dvec3_to_vec3(normal).normalize() })
    }
}

/// The middle of an edge `opening` shares with a face that stays, and that
/// face's outward normal there.
fn rim(shape: &Shape, opening: &Face, removed: &[Face]) -> Option<(DVec3, DVec3)> {
    opening.edges().find_map(|edge| {
        let kept = shape
            .adjacent_faces(&edge)
            .into_iter()
            .find(|face| !removed.iter().any(|gone| gone.is_same(face)))?;
        let middle = edge.midpoint();
        Some((middle, kept.normal_at(middle).ok()?))
    })
}

/// The middle of the shape's largest face, the first of equals, and its
/// outward normal there.
fn largest_face_middle(shape: &Shape) -> Result<(DVec3, DVec3)> {
    let largest = shape
        .faces()
        .fold(None, |largest: Option<(f64, Face)>, face| {
            let area = face.surface_area();
            match largest {
                Some((most, _)) if most >= area => largest,
                _ => Some((area, face)),
            }
        })
        .context("The part has no faces")?
        .1;
    let middle = largest.midpoint();
    let normal = largest.normal_at(middle).context("The part's largest face has no well-defined normal")?;
    Ok((middle, normal))
}

/// A hollow as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HollowParams {
    pub frame: HollowFrame,
    /// Wall thickness. Positive grows the walls inward from the part's
    /// surface, keeping its outside; negative grows them outward.
    pub thickness: Real,
}

impl HollowParams {
    /// No wall yet.
    pub fn new(frame: HollowFrame) -> Self {
        Self { frame, thickness: 0.0 }
    }

    /// Where the grip sits: on the far side of the wall from the anchor.
    pub fn grip(&self) -> Point3 {
        self.frame.anchor - self.frame.normal * self.thickness
    }

    /// Whether the wall is too thin to build.
    pub fn is_degenerate(&self) -> bool {
        self.thickness.abs() <= MIN_THICKNESS
    }
}

/// The target's part hollowed, to reshape the part with.
pub fn build_hollow(doc: &Document, target: &HollowTarget, params: &HollowParams) -> Result<Shape> {
    ensure!(!params.is_degenerate(), "Nothing to hollow without a wall thickness");
    let part = doc.part_at(target.node).context("Hollow target is not a known CAD part")?;
    ensure!(has_solid(&part.shape), "Only a solid can be hollowed");

    // OCCT's offset may retouch its input, so the hollow works on a copy: an
    // abandoned preview must leave the part as it was.
    let body = unwrap_single_solid(part.shape.deep_copy());
    ensure!(body.shape_type() == ShapeType::Solid, "Only a single solid can be hollowed");
    ensure!(body.sub_shapes().count() == 1, "A part with a void inside can't be hollowed");
    let faces = target.faces.iter().map(|&index| selected_face(&body, index)).collect::<Result<Vec<_>>>()?;

    body.hollow(-f64::from(params.thickness), &faces, JoinType::Intersection).map_err(explained)
}

/// `error`, led by a plain-language account of why the hollow failed.
fn explained(error: opencascade::Error) -> anyhow::Error {
    let hint = match &error {
        opencascade::Error::OffsetFailed(OffsetError::NullOffset) => "The wall is too thin to build",
        opencascade::Error::OffsetFailed(OffsetError::InvalidResult) => "The wall is too thick for this part",
        opencascade::Error::OffsetFailed(OffsetError::C0Geometry) => {
            "The part has creased surfaces the walls can't follow"
        }
        _ => "The wall may be too thick for this part",
    };
    anyhow::Error::new(error).context(hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    use opencascade::primitives::Wire;

    use crate::document::SourceFate;
    use crate::testing::{commit, doc_with_box, doc_with_shape, face_along, part_shape};

    const EPSILON: Real = 1e-5;

    fn target(node: NodeId, faces: &[u32]) -> HollowTarget {
        HollowTarget { node, faces: faces.to_vec() }
    }

    fn params(doc: &Document, target: &HollowTarget, thickness: Real) -> HollowParams {
        let frame = HollowFrame::new(doc, target).expect("target resolves");
        HollowParams { thickness, ..HollowParams::new(frame) }
    }

    fn hollowed_volume(doc: &Document, target: &HollowTarget, thickness: Real) -> f64 {
        let shape = build_hollow(doc, target, &params(doc, target, thickness)).expect("the part hollows");
        assert_eq!(shape.shape_type(), ShapeType::Solid);
        shape.volume()
    }

    /// Closed walls keep the box's outside around a smaller box of void, or
    /// grown outward, wrap the box's own volume as the void.
    #[test]
    fn a_closed_hollow_leaves_a_void() {
        let (doc, node) = doc_with_box();
        let closed = target(node, &[]);
        for (thickness, expected) in [(0.2, 8.0 - 1.6f64.powi(3)), (-0.2, 2.4f64.powi(3) - 8.0)] {
            let volume = hollowed_volume(&doc, &closed, thickness);
            assert!((volume - expected).abs() < 1e-6, "thickness {thickness}: expected {expected}, got {volume}");
        }
    }

    /// Opened at the top, the box is a cup: walls inside its outline, or
    /// outside and below it when grown outward.
    #[test]
    fn opening_the_top_makes_a_cup() {
        let (doc, node) = doc_with_box();
        let top = target(node, &[face_along(&doc, node, DVec3::Y)]);
        for (thickness, expected) in [(0.2, 8.0 - 1.6 * 1.8 * 1.6), (-0.2, 2.4 * 2.2 * 2.4 - 8.0)] {
            let volume = hollowed_volume(&doc, &top, thickness);
            assert!((volume - expected).abs() < 1e-6, "thickness {thickness}: expected {expected}, got {volume}");
        }
    }

    #[test]
    fn opening_two_faces_opens_a_corner() {
        let (doc, node) = doc_with_box();
        let corner = target(node, &[face_along(&doc, node, DVec3::Y), face_along(&doc, node, DVec3::X)]);
        let expected = 8.0 - 1.8 * 1.8 * 1.6;
        let volume = hollowed_volume(&doc, &corner, 0.2);
        assert!((volume - expected).abs() < 1e-6, "expected {expected}, got {volume}");
    }

    /// A cylinder opened at one cap is a tube with a floor.
    #[test]
    fn a_cylinder_opened_at_one_cap_is_a_cup() {
        let (doc, node) = doc_with_shape(Shape::cylinder_centered(DVec3::ZERO, 1.0, DVec3::Y, 2.0));
        let cap = target(node, &[face_along(&doc, node, DVec3::Y)]);
        let expected = std::f64::consts::PI * (2.0 - 0.9 * 0.9 * 1.9);
        let volume = hollowed_volume(&doc, &cap, 0.1);
        assert!((volume - expected).abs() < 1e-4, "expected {expected}, got {volume}");
    }

    /// The grip of an open hollow rides the rim of the opening, on the face
    /// beside it.
    #[test]
    fn an_open_frame_sits_on_the_rim_of_the_opening() {
        let (doc, node) = doc_with_box();
        let frame = HollowFrame::new(&doc, &target(node, &[face_along(&doc, node, DVec3::Y)])).expect("it resolves");

        assert!((frame.anchor.y - 1.0).abs() < EPSILON, "anchor {:?} is off the top", frame.anchor);
        assert!(frame.normal.y.abs() < EPSILON, "normal {:?} is not a wall's", frame.normal);
        assert!((frame.normal.magnitude() - 1.0).abs() < EPSILON);
        let across = frame.anchor.x * frame.normal.x + frame.anchor.z * frame.normal.z;
        assert!((across - 1.0).abs() < EPSILON, "anchor {:?} is not on the wall", frame.anchor);
    }

    #[test]
    fn a_closed_frame_sits_in_the_middle_of_the_largest_face() {
        let (doc, node) = doc_with_shape(Shape::box_centered(4.0, 1.0, 2.0));
        let frame = HollowFrame::new(&doc, &target(node, &[])).expect("it resolves");
        assert!(frame.normal.y.abs() > 1.0 - EPSILON, "normal {:?}", frame.normal);
        assert!((frame.anchor.x.abs() + frame.anchor.z.abs()) < EPSILON, "anchor {:?}", frame.anchor);
        assert!((frame.anchor.y.abs() - 0.5).abs() < EPSILON, "anchor {:?}", frame.anchor);
    }

    #[test]
    fn the_grip_sits_across_the_wall() {
        let (doc, node) = doc_with_box();
        let hollow = params(&doc, &target(node, &[face_along(&doc, node, DVec3::Y)]), 0.3);
        let expected = hollow.frame.anchor - hollow.frame.normal * 0.3;
        assert!((hollow.grip() - expected).magnitude() < EPSILON);
    }

    /// A wall thicker than half the box can't be built, and the part is left
    /// as it was.
    #[test]
    fn a_wall_too_thick_for_the_part_is_refused() {
        let (doc, node) = doc_with_box();
        for faces in [vec![], vec![face_along(&doc, node, DVec3::Y)]] {
            let hollow = target(node, &faces);
            for thickness in [1.0, 1.5] {
                let Err(error) = build_hollow(&doc, &hollow, &params(&doc, &hollow, thickness)) else {
                    panic!("{faces:?} at {thickness}: walls thicker than half the box were built");
                };
                assert!(format!("{error:#}").contains("too thick"), "{faces:?} at {thickness}: got {error:#}");
            }
        }
        assert!((part_shape(&doc, node).volume() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn a_sheet_is_refused() {
        let region: Shape = Face::from_wire(&Wire::rect(2.0, 2.0).unwrap()).unwrap().into();
        let (doc, node) = doc_with_shape(region);
        let Err(error) = HollowFrame::new(&doc, &target(node, &[])) else {
            panic!("a sheet has nothing to hollow");
        };
        assert!(format!("{error:#}").contains("Only a solid"), "got {error:#}");
    }

    #[test]
    fn a_wall_too_thin_to_build_is_refused() {
        let (doc, node) = doc_with_box();
        let closed = target(node, &[]);
        let thin = params(&doc, &closed, 1e-4);
        assert!(thin.is_degenerate());
        assert!(build_hollow(&doc, &closed, &thin).is_err());
    }

    /// A hollow reshapes its part where it stands, as one undo step that undo
    /// and redo replay.
    #[test]
    fn a_hollow_reshapes_its_part_in_place_as_one_undo_step() {
        let (mut doc, node) = doc_with_box();
        let part = doc.part_for_node(node).unwrap();
        let closed = target(node, &[]);
        let hollowed = 8.0 - 1.6f64.powi(3);

        let shape = build_hollow(&doc, &closed, &params(&doc, &closed, 0.2)).expect("the part hollows");
        commit(&mut doc, node, shape, SourceFate::Reshape).expect("the hollow applies");
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.node_for_part(part), Some(node), "the part keeps its node");
        assert!((part_shape(&doc, node).volume() - hollowed).abs() < 1e-6);

        doc.undo().expect("undo the hollow");
        assert!((part_shape(&doc, node).volume() - 8.0).abs() < 1e-9);
        doc.redo().expect("redo the hollow");
        assert!((part_shape(&doc, node).volume() - hollowed).abs() < 1e-6);
    }
}
