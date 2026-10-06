use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::{InnerSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::{JoinType, Shape, Shell};
use opencascade::OffsetError;

use crate::document::{dvec3_to_point3, dvec3_to_vec3, has_solid, selected_face, Document, SourceFate};

/// A side at or below this thickness is none: OCCT's offset takes anything
/// smaller as null.
const MIN_THICKNESS: Real = 1e-3;

/// The faces being thickened, all on one part, identified the way the
/// selection system reports them: by tessellation order within the part's mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThickenTarget {
    pub node: NodeId,
    /// The faces to thicken, the primary first. None thickens the whole part.
    pub faces: Vec<u32>,
}

/// Where a slab grows from, resolved once from its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThickenFrame {
    /// The middle of the primary face, where the grips measure from.
    pub origin: Point3,
    /// Unit normal of the primary face there; out of the material for a face
    /// of a solid.
    pub normal: Vector3,
    /// Whether the faces belong to a solid, which the slab can join.
    pub on_solid: bool,
    /// Whether the faces are all of a sheet, which the slab then replaces.
    pub whole_sheet: bool,
}

impl ThickenFrame {
    pub fn new(doc: &Document, target: &ThickenTarget) -> Result<Self> {
        let shape = &doc.part_at(target.node).context("Thicken target is not a known CAD part")?.shape;
        let on_solid = has_solid(shape);
        ensure!(!on_solid || !target.faces.is_empty(), "Select faces of the solid to thicken");
        let face_count = shape.faces().count() as u32;
        ensure!(face_count > 0, "Only faces can be thickened");
        let whole_sheet =
            !on_solid && (0..face_count).all(|index| target.faces.is_empty() || target.faces.contains(&index));

        let primary = selected_face(shape, target.faces.first().copied().unwrap_or(0))?;
        let origin = primary.midpoint();
        let normal = primary.normal_at(origin).context("The face to thicken has no well-defined normal")?;
        Ok(Self {
            origin: dvec3_to_point3(origin),
            normal: dvec3_to_vec3(normal).normalize(),
            on_solid,
            whole_sheet,
        })
    }
}

/// A thickening as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThickenParams {
    pub frame: ThickenFrame,
    /// Thickness along the faces' normals. Never negative.
    pub front: Real,
    /// Thickness against the faces' normals. Never negative, and unused when
    /// joining a solid.
    pub back: Real,
    /// Whether front and back are held equal.
    pub lock: bool,
    /// Whether faces of a solid thicken into a part of their own rather than
    /// joining it.
    pub new_body: bool,
}

impl ThickenParams {
    /// No thickness yet, joining any solid.
    pub fn new(frame: ThickenFrame) -> Self {
        Self { frame, front: 0.0, back: 0.0, lock: false, new_body: false }
    }

    /// What the slab does with the part its faces belong to.
    pub fn fate(&self) -> SourceFate {
        match (self.frame.on_solid, self.frame.whole_sheet) {
            (true, _) if self.new_body => SourceFate::Keep,
            (true, _) => SourceFate::Fuse,
            (false, true) => SourceFate::Replace,
            (false, false) => SourceFate::Keep,
        }
    }

    /// Whether the back thickness counts: not when joining a solid, whose
    /// material it would lie inside.
    pub fn back_applies(&self) -> bool {
        self.fate() != SourceFate::Fuse
    }

    /// The thicknesses to build each side with, zero where one doesn't count
    /// or is too thin to.
    fn sides(&self) -> (Real, Real) {
        let side = |thickness: Real| if thickness > MIN_THICKNESS { thickness } else { 0.0 };
        (side(self.front), if self.back_applies() { side(self.back) } else { 0.0 })
    }

    /// Whether there is no thickness to build.
    pub fn is_degenerate(&self) -> bool {
        let (front, back) = self.sides();
        front <= 0.0 && back <= 0.0
    }

    /// Where the front grip sits: out along the normal by the front thickness.
    pub fn front_grip(&self) -> Point3 {
        self.frame.origin + self.frame.normal * self.front
    }

    /// Where the back grip sits: back against the normal by the back thickness.
    pub fn back_grip(&self) -> Point3 {
        self.frame.origin - self.frame.normal * self.back
    }

    /// Sets the front thickness, and the back with it while locked.
    pub fn set_front(&mut self, front: Real) {
        self.front = front.max(0.0);
        if self.lock {
            self.back = self.front;
        }
    }

    /// Sets the back thickness, and the front with it while locked.
    pub fn set_back(&mut self, back: Real) {
        self.back = back.max(0.0);
        if self.lock {
            self.front = self.back;
        }
    }

    /// Holds both sides equal, at the front's thickness, or lets them go.
    pub fn set_lock(&mut self, lock: bool) {
        self.lock = lock;
        if lock {
            self.back = self.front;
        }
    }
}

/// The slab the target's faces thicken into, on its own, to commit with its
/// part as [`ThickenParams::fate`] says.
pub fn build_thicken(doc: &Document, target: &ThickenTarget, params: &ThickenParams) -> Result<Shape> {
    ensure!(!params.is_degenerate(), "Nothing to thicken without a thickness");
    let part = doc.part_at(target.node).context("Thicken target is not a known CAD part")?;

    // OCCT's offset reuses the faces it thickens and may retouch them, so the
    // slab grows from a copy: an abandoned preview must leave the part as it was.
    let body = part.shape.deep_copy();
    let sheet: Shape = if params.frame.whole_sheet {
        body
    } else {
        let faces = target.faces.iter().map(|&index| selected_face(&body, index)).collect::<Result<Vec<_>>>()?;
        match faces.as_slice() {
            [face] => face.into(),
            faces => Shell::from_faces(faces).into(),
        }
    };

    let (front, back) = params.sides();
    let (front, back) = (f64::from(front), f64::from(back));
    let join = JoinType::Intersection;
    let slab = match (front > 0.0, back > 0.0) {
        (true, false) => sheet.thicken(front, join),
        (false, true) => sheet.thicken(-back, join),
        _ => sheet.offset_surface(-back, join).and_then(|base| base.thicken(front + back, join)),
    };
    slab.map_err(explained)
}

/// `error`, led by a plain-language account of why the thickening failed.
fn explained(error: opencascade::Error) -> anyhow::Error {
    let hint = match &error {
        opencascade::Error::OffsetFailed(OffsetError::NotConnectedShell) => "The faces to thicken must touch",
        opencascade::Error::OffsetFailed(OffsetError::InvalidResult) => {
            "The thickness is too large for these faces"
        }
        _ => "The thickness may be too large for these faces",
    };
    anyhow::Error::new(error).context(hint)
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::common::EuclideanSpace;
    use glam::DVec3;
    use opencascade::bounding_box::aabb;
    use opencascade::primitives::{Face, FaceType, ShapeType, Wire};

    use crate::document::PartKind;
    use crate::testing::{commit, doc_with_box, doc_with_shape, face_along, part_shape, tube, volumes};

    const EPSILON: Real = 1e-5;

    /// Slack for bounding boxes, which OCCT pads by the shape's tolerance.
    const BOUNDS: f64 = 5e-3;

    /// A 2×3 sheet on the XY plane, centred on the origin.
    fn sheet() -> Shape {
        Face::from_wire(&Wire::rect(2.0, 3.0).expect("rectangle builds")).expect("face builds").into()
    }

    fn target(node: NodeId, faces: &[u32]) -> ThickenTarget {
        ThickenTarget { node, faces: faces.to_vec() }
    }

    fn params(doc: &Document, target: &ThickenTarget, front: Real, back: Real) -> ThickenParams {
        let frame = ThickenFrame::new(doc, target).expect("target resolves");
        ThickenParams { front, back, ..ThickenParams::new(frame) }
    }

    /// Builds the slab and commits it with its part as the parameters' fate says.
    fn thicken(doc: &mut Document, target: &ThickenTarget, params: &ThickenParams) {
        let slab = build_thicken(doc, target, params).expect("the faces thicken");
        commit(doc, target.node, slab, params.fate()).expect("the slab commits");
    }

    #[test]
    fn a_face_of_a_solid_thickens_out_of_it() {
        let (doc, node) = doc_with_box();
        for index in 0..6 {
            let frame = ThickenFrame::new(&doc, &target(node, &[index])).expect("box face resolves");
            assert!(frame.on_solid && !frame.whole_sheet);
            assert!((frame.normal.magnitude() - 1.0).abs() < EPSILON);
            assert!(frame.normal.dot(frame.origin.to_vec()) > 0.0, "face {index} thickens into the box");
        }
    }

    /// Joined, the top face's slab makes the box taller, in place; the back
    /// would lie inside the box and is left out.
    #[test]
    fn a_joined_slab_grows_its_solid_in_place() {
        let (mut doc, node) = doc_with_box();
        let part = doc.part_for_node(node).unwrap();
        let top = target(node, &[face_along(&doc, node, DVec3::Y)]);
        let joined = params(&doc, &top, 0.5, 0.3);
        assert_eq!(joined.fate(), SourceFate::Fuse);
        assert!(!joined.back_applies());

        thicken(&mut doc, &top, &joined);
        assert_eq!(volumes(&doc).len(), 1);
        assert_eq!(doc.node_for_part(part), Some(node), "the part keeps its node");
        assert!((part_shape(&doc, node).volume() - 10.0).abs() < 1e-6, "got {}", part_shape(&doc, node).volume());

        doc.undo().expect("undo the thickening");
        assert!((part_shape(&doc, node).volume() - 8.0).abs() < 1e-9);
    }

    /// Two faces meeting at an edge join with the corner between their slabs
    /// filled.
    #[test]
    fn faces_meeting_at_an_edge_join_with_their_corner_filled() {
        let (mut doc, node) = doc_with_box();
        let corner = target(node, &[face_along(&doc, node, DVec3::Y), face_along(&doc, node, DVec3::X)]);
        let joined = params(&doc, &corner, 0.5, 0.0);

        thicken(&mut doc, &corner, &joined);
        assert!((part_shape(&doc, node).volume() - 12.5).abs() < 1e-6, "got {}", part_shape(&doc, node).volume());
    }

    /// As a new body, the slab stands beside its untouched source with both
    /// sides counted.
    #[test]
    fn a_new_body_slab_leaves_its_solid_alone() {
        let (mut doc, node) = doc_with_box();
        let top = target(node, &[face_along(&doc, node, DVec3::Y)]);
        let slab = ThickenParams { new_body: true, ..params(&doc, &top, 0.5, 0.25) };
        assert_eq!(slab.fate(), SourceFate::Keep);

        thicken(&mut doc, &top, &slab);
        let names: Vec<_> = doc.parts().map(|part| part.name.as_str()).collect();
        assert_eq!(names, ["part", "Result-001"], "the solid stays, and the slab is a part of its own");
        let volumes = volumes(&doc);
        assert!((volumes[0] - 8.0).abs() < 1e-9);
        assert!((volumes[1] - 3.0).abs() < 1e-6, "got {}", volumes[1]);
    }

    /// With only a back thickness, the slab grows into the solid from its face.
    #[test]
    fn a_back_slab_grows_against_the_normal() {
        let (doc, node) = doc_with_box();
        let top = target(node, &[face_along(&doc, node, DVec3::Y)]);
        let slab = ThickenParams { new_body: true, ..params(&doc, &top, 0.0, 0.5) };

        let shape = build_thicken(&doc, &top, &slab).expect("the face thickens");
        assert!((shape.volume() - 2.0).abs() < 1e-6, "got {}", shape.volume());
        let bounds = aabb(&shape);
        assert!((bounds.min().y - 0.5).abs() < BOUNDS && (bounds.max().y - 1.0).abs() < BOUNDS, "y spans {} to {}", bounds.min().y, bounds.max().y);
    }

    #[test]
    fn a_cylinders_side_thickens_into_a_tube() {
        let (doc, node) = doc_with_shape(Shape::cylinder_centered(DVec3::ZERO, 1.0, DVec3::Y, 2.0));
        let side = part_shape(&doc, node).faces().position(|face| face.face_type() == FaceType::Cylinder).unwrap();
        let side = target(node, &[side as u32]);
        let pi = std::f64::consts::PI;

        let tube = ThickenParams { new_body: true, ..params(&doc, &side, 0.1, 0.0) };
        let volume = build_thicken(&doc, &side, &tube).expect("the side thickens").volume();
        assert!((volume - 0.42 * pi).abs() < 1e-4, "got {volume}");

        let (mut doc, node) = (doc, node);
        let joined = params(&doc, &side, 0.1, 0.0);
        thicken(&mut doc, &side, &joined);
        assert!((part_shape(&doc, node).volume() - 2.42 * pi).abs() < 1e-4);
    }

    /// A sheet thickened whole is replaced by its slab, under its own name, to
    /// either side or both.
    #[test]
    fn a_whole_sheet_is_replaced_by_its_slab() {
        for (front, back, low, high) in [(0.5, 0.0, 0.0, 0.5), (0.0, 0.5, -0.5, 0.0), (0.25, 0.25, -0.25, 0.25)] {
            let (mut doc, node) = doc_with_shape(sheet());
            let whole = target(node, &[]);
            let slab = params(&doc, &whole, front, back);
            assert!(slab.frame.whole_sheet);
            assert_eq!(slab.fate(), SourceFate::Replace);
            let side = f64::from(slab.frame.normal.z.signum());

            thicken(&mut doc, &whole, &slab);
            let parts: Vec<_> = doc.parts().collect();
            assert_eq!(parts.len(), 1, "the sheet was consumed");
            assert_eq!(parts[0].name, "part");
            assert_eq!(parts[0].kind(), PartKind::Solid);
            assert!((parts[0].shape.volume() - 3.0).abs() < 1e-6, "{front}/{back}: got {}", parts[0].shape.volume());
            let bounds = aabb(&parts[0].shape);
            let (min, max) = (bounds.min().z * side, bounds.max().z * side);
            let (min, max) = (min.min(max), min.max(max));
            assert!((min - low).abs() < BOUNDS && (max - high).abs() < BOUNDS, "{front}/{back}: spans {min} to {max}");
        }
    }

    /// Selecting every face of a sheet is selecting the sheet; selecting some
    /// makes a part of their own beside it.
    #[test]
    fn some_faces_of_a_sheet_make_a_part_beside_it() {
        let (mut doc, node) = doc_with_shape(tube());
        let all: Vec<u32> = (0..4).collect();
        assert_eq!(params(&doc, &target(node, &all), 0.1, 0.0).fate(), SourceFate::Replace);

        let one = target(node, &[0]);
        let slab = params(&doc, &one, 0.1, 0.0);
        assert_eq!(slab.fate(), SourceFate::Keep);
        thicken(&mut doc, &one, &slab);
        let volumes = volumes(&doc);
        assert_eq!(volumes.len(), 2, "the tube stays");
        assert!((volumes[1] - 0.2).abs() < 1e-6, "got {}", volumes[1]);
    }

    #[test]
    fn a_whole_tube_thickens_into_walls() {
        let (mut doc, node) = doc_with_shape(tube());
        let all: Vec<u32> = (0..4).collect();
        let walls = params(&doc, &target(node, &all), 0.1, 0.0);
        // Out of the tube or into it, by which way the lofted faces face.
        let outward = walls.frame.normal.dot(walls.frame.origin.to_vec() - Vector3::unit_y() * walls.frame.origin.y) > 0.0;
        let expected = if outward { (1.2 * 1.2 - 1.0) * 2.0 } else { (1.0 - 0.8 * 0.8) * 2.0 };

        thicken(&mut doc, &target(node, &all), &walls);
        let parts: Vec<_> = doc.parts().collect();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].shape.shape_type(), ShapeType::Solid);
        assert!((parts[0].shape.volume() - expected).abs() < 1e-6, "got {}", parts[0].shape.volume());
    }

    #[test]
    fn faces_apart_are_refused() {
        let (doc, node) = doc_with_box();
        let apart = target(node, &[face_along(&doc, node, DVec3::Y), face_along(&doc, node, DVec3::NEG_Y)]);
        let Err(error) = build_thicken(&doc, &apart, &params(&doc, &apart, 0.5, 0.0)) else {
            panic!("faces that don't touch can't make one slab");
        };
        assert!(format!("{error:#}").contains("must touch"), "got {error:#}");
        assert!((part_shape(&doc, node).volume() - 8.0).abs() < 1e-9, "the part is left as it was");
    }

    #[test]
    fn a_whole_solid_is_refused() {
        let (doc, node) = doc_with_box();
        let Err(error) = ThickenFrame::new(&doc, &target(node, &[])) else {
            panic!("a solid thickens by its faces");
        };
        assert!(format!("{error:#}").contains("faces of the solid"), "got {error:#}");
    }

    #[test]
    fn no_thickness_is_refused() {
        let (doc, node) = doc_with_box();
        let top = target(node, &[face_along(&doc, node, DVec3::Y)]);
        let none = params(&doc, &top, 0.0, 0.5);
        assert!(none.is_degenerate(), "the back doesn't count when joining");
        assert!(build_thicken(&doc, &top, &none).is_err());
    }

    #[test]
    fn locked_sides_move_together() {
        let (doc, node) = doc_with_shape(sheet());
        let mut slab = params(&doc, &target(node, &[]), 0.4, 0.1);
        let sides = |slab: &ThickenParams| (slab.front, slab.back);
        let near = |(front, back): (Real, Real), (f, b): (Real, Real)| (front - f).abs() < EPSILON && (back - b).abs() < EPSILON;

        slab.set_lock(true);
        assert!(near(sides(&slab), (0.4, 0.4)), "locking evens the back up: {:?}", sides(&slab));
        slab.set_back(0.7);
        assert!(near(sides(&slab), (0.7, 0.7)), "{:?}", sides(&slab));
        slab.set_lock(false);
        slab.set_front(-1.0);
        assert!(near(sides(&slab), (0.0, 0.7)), "a side never goes negative: {:?}", sides(&slab));
    }
}
