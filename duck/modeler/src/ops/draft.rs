use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::{InnerSpace, Matrix3, Point3, Rad, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::{FaceType, Shape};
use opencascade::DraftError;

use crate::document::{dvec3_to_point3, dvec3_to_vec3, point3_to_dvec3, vec3_to_dvec3, Document};

/// A change of angle at or below this many radians drafts nothing; OCCT
/// ignores anything smaller.
const MIN_DRAFT: Real = 1e-4;

/// A face whose normal leans less than this from the pull lies along the
/// neutral plane, with no line to hinge on.
const MIN_HINGE: Real = 1e-6;

/// A face centre this close to the neutral plane lies on it.
const ON_PLANE: Real = 1e-6;

/// The faces being drafted and the face they hinge on, all on one part,
/// identified the way the selection system reports them: by tessellation order
/// within the part's mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftTarget {
    pub node: NodeId,
    /// The flat face whose plane the drafted faces hinge on.
    pub neutral: u32,
    /// The faces to draft, the one carrying the grip first.
    pub faces: Vec<u32>,
}

/// The plane and pull every face drafts against, and the hinge the grip's face
/// turns about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftFrame {
    /// A point on the neutral plane.
    pub origin: Point3,
    /// Unit normal of the neutral plane, pointing from it toward the drafted
    /// faces.
    pub pull: Vector3,
    /// Where the grip's face crosses the neutral plane, below its middle.
    pub hinge: Point3,
    /// Unit direction of the hinge line. Turning about it drafts the grip's
    /// face further.
    pub axis: Vector3,
    /// From the hinge to the grip's face's middle, as the face stands.
    pub lever: Vector3,
    /// The grip's face's angle to the pull as it stands, in radians.
    pub initial: Real,
}

impl DraftFrame {
    /// Resolves the frame from the target's neutral face and its first face to
    /// draft.
    pub fn new(doc: &Document, target: &DraftTarget) -> Result<Self> {
        let part = doc
            .part_for_node(target.node)
            .and_then(|part| doc.get_part(part))
            .context("Draft target is not a known CAD part")?;
        let face = |index: u32| {
            part.shape
                .face_at(index as usize)
                .with_context(|| format!("Selected face {index} is not part of the part"))
        };
        let neutral = face(target.neutral)?;
        ensure!(neutral.face_type() == FaceType::Plane, "The neutral face must be flat");
        let lever_face = face(*target.faces.first().context("There are no faces to draft")?)?;

        // `normal_at_center` goes through `BRepGProp_Face::Normal`, which applies
        // each face's orientation: both normals point out of the material.
        let origin = dvec3_to_point3(neutral.center_of_mass());
        let normal = dvec3_to_vec3(neutral.normal_at_center()?).normalize();
        let centre = lever_face.midpoint();
        let outward = lever_face
            .normal_at(centre)
            .context("The face to draft has no well-defined normal")?;
        let (centre, outward) = (dvec3_to_point3(centre), dvec3_to_vec3(outward).normalize());

        // Away from the neutral plane toward the drafted face. One straddling
        // the plane pulls into the material behind the neutral face.
        let pull = if (centre - origin).dot(normal) > ON_PLANE { normal } else { -normal };

        let axis = outward.cross(pull);
        ensure!(
            axis.magnitude() > MIN_HINGE,
            "The face to draft lies parallel to the neutral face"
        );
        let axis = axis.normalize();
        // The pull, laid into the face: straight up it, away from the hinge.
        let up = axis.cross(outward).normalize();
        let hinge = centre - up * ((centre - origin).dot(pull) / up.dot(pull));
        let initial = outward.dot(pull).clamp(-1.0, 1.0).asin();

        Ok(Self { origin, pull, hinge, axis, lever: centre - hinge, initial })
    }
}

/// A draft as configured.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftParams {
    pub frame: DraftFrame,
    /// The drafted faces' angle to the pull, in radians. Positive leans them so
    /// material comes away on the pull's side of the neutral plane: a boss
    /// narrows away from it and a pocket widens.
    pub angle: Real,
}

impl DraftParams {
    /// No draft: the grip's face at the angle it already stands at.
    pub fn new(frame: DraftFrame) -> Self {
        Self { frame, angle: frame.initial }
    }

    /// Where the grip sits: the grip's face's middle, turned with the face.
    pub fn grip(&self) -> Point3 {
        let turn = Matrix3::from_axis_angle(self.frame.axis, Rad(self.angle - self.frame.initial));
        self.frame.hinge + turn * self.frame.lever
    }

    /// Whether the angle is where the grip's face already stands, leaving
    /// nothing to draft.
    pub fn is_degenerate(&self) -> bool {
        (self.angle - self.frame.initial).abs() <= MIN_DRAFT
    }
}

/// The target's part with its faces drafted.
pub fn build_draft(doc: &Document, target: &DraftTarget, params: &DraftParams) -> Result<Shape> {
    ensure!(!params.is_degenerate(), "Nothing to draft at the faces' own angle");
    let part = doc
        .part_for_node(target.node)
        .and_then(|part| doc.get_part(part))
        .context("Draft target is not a known CAD part")?;

    // OCCT's modifications may raise tolerances on their input, so the draft
    // works on a copy: an abandoned preview must leave the part as it was.
    let body = part.shape.deep_copy();
    let faces = target
        .faces
        .iter()
        .map(|&index| {
            body.face_at(index as usize)
                .with_context(|| format!("Selected face {index} is not part of the part"))
        })
        .collect::<Result<Vec<_>>>()?;

    let pull = vec3_to_dvec3(params.frame.pull);
    let origin = point3_to_dvec3(params.frame.origin);
    body.draft_faces(&faces, pull, f64::from(params.angle), origin, pull)
        .map_err(|error| {
            let hint = failure_hint(&error);
            anyhow::Error::new(error).context(hint)
        })
}

/// A plain-language account of why a draft failed, to lead the kernel's reason.
fn failure_hint(error: &opencascade::Error) -> &'static str {
    match error {
        opencascade::Error::DraftFailed(DraftError::FaceRecomputation) => {
            "Only flat faces, and cylinders or cones standing along the pull, can be drafted"
        }
        _ => "The drafted faces can't be rejoined to their neighbours; try a smaller angle",
    }
}

/// Apply the draft: rebuild the target's part with it, in place, as one undo
/// step.
pub fn execute_draft(doc: &mut Document, target: &DraftTarget, params: &DraftParams) -> Result<()> {
    let shape = build_draft(doc, target, params)?;
    let part = doc.part_for_node(target.node).context("Draft target is not a known CAD part")?;
    doc.reshape_part(part, shape, "Draft")
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::Vector3;
    use duck_engine_scene::Scene;
    use glam::DVec3;
    use opencascade::primitives::{Edge, Face, ShapeType, Wire};

    use crate::document::unwrap_single_solid;

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

    fn part_shape(doc: &Document, node: NodeId) -> &Shape {
        &doc.get_part(doc.part_for_node(node).unwrap()).unwrap().shape
    }

    /// The index of the face whose outward normal at its centre is `normal`
    /// and whose centre has `height` along it.
    fn face_at(doc: &Document, node: NodeId, normal: DVec3, height: f64) -> u32 {
        part_shape(doc, node)
            .faces()
            .position(|face| {
                let centre = face.center_of_mass();
                face.normal_at(centre).is_ok_and(|n| n.distance(normal) < 1e-6)
                    && (centre.dot(normal) - height).abs() < 1e-6
            })
            .expect("a face matches") as u32
    }

    /// A face of the box by its outward normal.
    fn box_face(doc: &Document, node: NodeId, normal: DVec3) -> u32 {
        face_at(doc, node, normal, 1.0)
    }

    fn target(node: NodeId, neutral: u32, faces: &[u32]) -> DraftTarget {
        DraftTarget { node, neutral, faces: faces.to_vec() }
    }

    fn params(doc: &Document, target: &DraftTarget, angle: Real) -> DraftParams {
        let frame = DraftFrame::new(doc, target).expect("target resolves");
        DraftParams { angle, ..DraftParams::new(frame) }
    }

    /// The bottom of the box and its four walls.
    fn walls(doc: &Document, node: NodeId) -> DraftTarget {
        let faces = [DVec3::X, DVec3::Z, DVec3::NEG_X, DVec3::NEG_Z].map(|n| box_face(doc, node, n));
        target(node, box_face(doc, node, DVec3::NEG_Y), &faces)
    }

    /// Volume of a frustum between squares of side `a` and `b`, `height` apart.
    fn square_frustum(a: f64, b: f64, height: f64) -> f64 {
        height / 3.0 * (a * a + b * b + a * b)
    }

    fn degrees(angle: f64) -> Real {
        angle.to_radians() as Real
    }

    #[test]
    fn a_box_wall_hinges_on_the_floor_it_stands_on() {
        let (doc, node) = doc_with_box();
        let wall = target(node, box_face(&doc, node, DVec3::NEG_Y), &[box_face(&doc, node, DVec3::X)]);
        let frame = DraftFrame::new(&doc, &wall).expect("wall resolves");

        assert!((frame.pull - Vector3::unit_y()).magnitude() < EPSILON, "pull {:?}", frame.pull);
        assert!((frame.hinge - Point3::new(1.0, -1.0, 0.0)).magnitude() < EPSILON, "hinge {:?}", frame.hinge);
        assert!((frame.axis - Vector3::unit_z()).magnitude() < EPSILON, "axis {:?}", frame.axis);
        assert!((frame.lever - Vector3::unit_y()).magnitude() < EPSILON, "lever {:?}", frame.lever);
        assert!(frame.initial.abs() < EPSILON);
    }

    /// Hinged on the top instead, the wall pulls down and away from it.
    #[test]
    fn the_pull_points_from_the_neutral_face_toward_the_drafted_one() {
        let (doc, node) = doc_with_box();
        let wall = target(node, box_face(&doc, node, DVec3::Y), &[box_face(&doc, node, DVec3::X)]);
        let frame = DraftFrame::new(&doc, &wall).expect("wall resolves");
        assert!((frame.pull + Vector3::unit_y()).magnitude() < EPSILON, "pull {:?}", frame.pull);
        assert!((frame.hinge - Point3::new(1.0, 1.0, 0.0)).magnitude() < EPSILON);
    }

    #[test]
    fn a_face_parallel_to_the_neutral_face_is_refused() {
        let (doc, node) = doc_with_box();
        let top = target(node, box_face(&doc, node, DVec3::NEG_Y), &[box_face(&doc, node, DVec3::Y)]);
        let Err(error) = DraftFrame::new(&doc, &top) else { panic!("the top has no hinge on the floor") };
        assert!(format!("{error:#}").contains("parallel"), "got {error:#}");
    }

    #[test]
    fn a_curved_neutral_face_is_refused() {
        let (doc, node) = doc_with_shape(Shape::cylinder_centered(DVec3::ZERO, 1.0, DVec3::Y, 2.0));
        let shape = part_shape(&doc, node);
        let side = shape.faces().position(|f| f.face_type() == FaceType::Cylinder).unwrap() as u32;
        let cap = shape.faces().position(|f| f.face_type() == FaceType::Plane).unwrap() as u32;

        let Err(error) = DraftFrame::new(&doc, &target(node, side, &[cap])) else {
            panic!("a cylinder has no neutral plane");
        };
        assert!(format!("{error:#}").contains("must be flat"), "got {error:#}");
    }

    /// One wall leaning in by θ over the box's height of 2 cuts away a wedge
    /// with legs 2 and 2·tanθ, all along its depth of 2.
    #[test]
    fn one_drafted_wall_cuts_a_wedge() {
        let (doc, node) = doc_with_box();
        let wall = target(node, box_face(&doc, node, DVec3::NEG_Y), &[box_face(&doc, node, DVec3::X)]);
        let angle = degrees(10.0);

        let shape = build_draft(&doc, &wall, &params(&doc, &wall, angle)).expect("the wall drafts");
        assert_eq!(shape.shape_type(), ShapeType::Solid);
        let expected = 8.0 - 4.0 * f64::from(angle).tan();
        assert!((shape.volume() - expected).abs() < 1e-6, "expected {expected}, got {}", shape.volume());
    }

    /// All four walls drafted together taper the box into a frustum; a negative
    /// angle flares it.
    #[test]
    fn four_drafted_walls_make_a_frustum() {
        let (doc, node) = doc_with_box();
        let walls = walls(&doc, node);
        for sign in [1.0, -1.0] {
            let angle = sign * degrees(5.0);
            let shape = build_draft(&doc, &walls, &params(&doc, &walls, angle)).expect("the walls draft");
            let expected = square_frustum(2.0, 2.0 - 4.0 * f64::from(angle).tan(), 2.0);
            assert!(
                (shape.volume() - expected).abs() < 1e-6,
                "sign {sign}: expected {expected}, got {}",
                shape.volume()
            );
        }
    }

    /// OCCT sets a face's angle to the pull rather than turning it further, so
    /// a drafted face starts from the angle it was given.
    #[test]
    fn a_drafted_face_starts_from_its_own_angle() {
        let (mut doc, node) = doc_with_box();
        let floor = box_face(&doc, node, DVec3::NEG_Y);
        let wall = target(node, floor, &[box_face(&doc, node, DVec3::X)]);
        let angle = degrees(10.0);
        let draft = params(&doc, &wall, angle);
        execute_draft(&mut doc, &wall, &draft).expect("the wall drafts");

        let leaning = part_shape(&doc, node)
            .faces()
            .position(|face| face.normal_at_center().is_ok_and(|n| n.x > 0.5 && n.y > 0.1))
            .expect("the drafted wall") as u32;
        let floor = box_face(&doc, node, DVec3::NEG_Y);
        let frame = DraftFrame::new(&doc, &target(node, floor, &[leaning])).expect("it resolves");
        assert!((frame.initial - angle).abs() < 1e-6, "got {}°", frame.initial.to_degrees());
        assert!(DraftParams { angle, ..DraftParams::new(frame) }.is_degenerate());
    }

    /// Hinged on a plate's top, the walls of a boss standing on it narrow as
    /// they rise away from it.
    #[test]
    fn a_boss_narrows_away_from_the_plate_it_stands_on() {
        let plate = Shape::box_from_corners(DVec3::new(-2.0, -1.0, -2.0), DVec3::new(2.0, 0.0, 2.0));
        let boss = Shape::box_from_corners(DVec3::new(-0.5, 0.0, -0.5), DVec3::new(0.5, 1.0, 0.5));
        let fused = unwrap_single_solid(plate.union(&boss).expect("the boss fuses").shape);
        let (doc, node) = doc_with_shape(fused);

        let walls: Vec<u32> = [DVec3::X, DVec3::Z, DVec3::NEG_X, DVec3::NEG_Z]
            .iter()
            .map(|&n| face_at(&doc, node, n, 0.5))
            .collect();
        let bosses = target(node, face_at(&doc, node, DVec3::Y, 0.0), &walls);
        let angle = degrees(5.0);
        let draft = params(&doc, &bosses, angle);
        assert!((draft.frame.pull - Vector3::unit_y()).magnitude() < EPSILON);

        let shape = build_draft(&doc, &bosses, &draft).expect("the boss drafts");
        let expected = 16.0 + square_frustum(1.0, 1.0 - 2.0 * f64::from(angle).tan(), 1.0);
        assert!((shape.volume() - expected).abs() < 1e-6, "expected {expected}, got {}", shape.volume());
    }

    /// Pulled along its axis from its base, a cylinder's side drafts into a
    /// cone.
    #[test]
    fn a_standing_cylinder_drafts_into_a_cone() {
        let (doc, node) = doc_with_shape(Shape::cylinder_centered(DVec3::ZERO, 1.0, DVec3::Y, 2.0));
        let side = part_shape(&doc, node).faces().position(|f| f.face_type() == FaceType::Cylinder).unwrap();
        let base = face_at(&doc, node, DVec3::NEG_Y, 1.0);
        let cylinder = target(node, base, &[side as u32]);
        let angle = degrees(10.0);
        let draft = params(&doc, &cylinder, angle);
        assert!(draft.frame.initial.abs() < EPSILON);
        assert!((draft.frame.pull - Vector3::unit_y()).magnitude() < EPSILON);

        let shape = build_draft(&doc, &cylinder, &draft).expect("the cylinder drafts");
        let top = 1.0 - 2.0 * f64::from(angle).tan();
        let expected = std::f64::consts::PI * 2.0 / 3.0 * (1.0 + top * top + top);
        assert!((shape.volume() - expected).abs() < 1e-4, "expected {expected}, got {}", shape.volume());
    }

    /// A spline's swept side can't take a draft: that is reported, and the part
    /// is left as it was.
    #[test]
    fn a_spline_wall_is_refused_and_the_part_left_alone() {
        let spline = Edge::spline_from_points(
            [
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(2.0, 0.0, 0.5),
                DVec3::new(1.5, 0.0, 2.0),
                DVec3::new(-0.5, 0.0, 1.0),
            ],
            None,
            true,
        )
        .expect("spline builds");
        let profile = Face::from_wire(&Wire::from_edges([&spline]).expect("wire builds")).expect("face builds");
        let (doc, node) = doc_with_shape(profile.extrude(DVec3::Y).into());
        let before = part_shape(&doc, node).volume();
        let shape = part_shape(&doc, node);
        let floor = face_at(&doc, node, DVec3::NEG_Y, 0.0);
        let side = shape.faces().position(|f| f.face_type() != FaceType::Plane).unwrap() as u32;
        let spline_wall = target(node, floor, &[side]);

        let Err(error) = build_draft(&doc, &spline_wall, &params(&doc, &spline_wall, degrees(5.0))) else {
            panic!("a spline wall can't be drafted");
        };
        assert!(format!("{error:#}").contains("Only flat faces"), "got {error:#}");
        assert!((part_shape(&doc, node).volume() - before).abs() < 1e-9);
    }

    /// Walls leaning so far that they would cross inside the box don't make a
    /// solid.
    #[test]
    fn walls_drafted_past_meeting_are_refused() {
        let (doc, node) = doc_with_box();
        let walls = walls(&doc, node);
        let result = build_draft(&doc, &walls, &params(&doc, &walls, degrees(60.0)));
        match result {
            Err(_) => {}
            Ok(shape) => panic!("got a shape: valid {}, volume {}", shape.is_valid(), shape.volume()),
        }
    }

    #[test]
    fn the_grip_turns_with_the_face_about_the_hinge() {
        let (doc, node) = doc_with_box();
        let wall = target(node, box_face(&doc, node, DVec3::NEG_Y), &[box_face(&doc, node, DVec3::X)]);
        let flat = params(&doc, &wall, 0.0);
        assert!((flat.grip() - Point3::new(1.0, 0.0, 0.0)).magnitude() < EPSILON);

        // Leaning in by 30°, the centre one unit up the wall swings in toward −X.
        let leaning = DraftParams { angle: degrees(30.0), ..flat };
        let expected = Point3::new(1.0 - 0.5, -1.0 + 0.75_f64.sqrt() as Real, 0.0);
        assert!((leaning.grip() - expected).magnitude() < EPSILON, "got {:?}", leaning.grip());
        assert!(((leaning.grip() - leaning.frame.hinge).magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn an_undrafted_angle_is_refused() {
        let (doc, node) = doc_with_box();
        let walls = walls(&doc, node);
        let none = params(&doc, &walls, 0.0);
        assert!(none.is_degenerate());
        assert!(build_draft(&doc, &walls, &none).is_err());
    }

    /// Faces are resolved on a deep copy by position, which only works if the
    /// copy explores its faces in the original's order.
    #[test]
    fn a_deep_copy_keeps_the_face_order() {
        let bored = Shape::box_centered(4.0, 4.0, 4.0)
            .subtract(&Shape::cylinder_centered(DVec3::ZERO, 1.0, DVec3::Z, 6.0))
            .expect("the bore cuts")
            .shape;
        for shape in [Shape::box_centered(2.0, 2.0, 2.0), bored] {
            let copy = shape.deep_copy();
            let original: Vec<_> = shape.faces().map(|face| face.center_of_mass()).collect();
            let copied: Vec<_> = copy.faces().map(|face| face.center_of_mass()).collect();
            assert_eq!(original, copied);
        }
    }

    /// Applying a draft reshapes the part where it stands, as one undo step
    /// that undo and redo replay.
    #[test]
    fn execute_reshapes_in_place_as_one_undo_step() {
        let (mut doc, node) = doc_with_box();
        let part = doc.part_for_node(node).unwrap();
        let walls = walls(&doc, node);
        let angle = degrees(5.0);
        let drafted = square_frustum(2.0, 2.0 - 4.0 * f64::from(angle).tan(), 2.0);

        let draft = params(&doc, &walls, angle);
        execute_draft(&mut doc, &walls, &draft).expect("the draft applies");
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.node_for_part(part), Some(node), "the part keeps its node");
        assert!((part_shape(&doc, node).volume() - drafted).abs() < 1e-6);
        assert_eq!(doc.undo_label(), Some("Draft"));

        doc.undo().expect("undo the draft");
        assert!((part_shape(&doc, node).volume() - 8.0).abs() < 1e-9);

        doc.redo().expect("redo the draft");
        assert!((part_shape(&doc, node).volume() - drafted).abs() < 1e-6);
    }
}
