//! The geometry of parts' edges and faces, in the world: the point of one
//! nearest another, where a perpendicular meets it, its middle, its circle.
//!
//! A [`PartEdge`] or [`PartFace`] is an element of a part, found in its B-rep.
//! Their queries take and return world points and directions.

use std::fmt;

use anyhow::{ensure, Result};
use duck_engine_scene::cad::EdgeElement;
use duck_engine_scene::common::{InnerSpace, MetricSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::{SubGeometryElement, SubGeometryKind};
use glam::DVec3;
use opencascade::primitives::{EdgePoint, EdgeType, Face, FaceType};

use crate::document::{dvec3_to_point3, dvec3_to_vec3, point3_to_dvec3, CadPart, Document, PartId};

/// How far out of one plane two circles may lie and still share it: the sine
/// of the angle between their normals, and the gap between their planes
/// relative to their size.
const COPLANAR: Real = 1e-6;

/// The factor from `part`'s B-rep to the world, which tessellation scales it
/// into.
fn scale(part: &CadPart) -> f64 {
    part.options().scale_factor
}

/// World `point`, in `part`'s B-rep.
fn to_shape(part: &CadPart, point: Point3) -> DVec3 {
    point3_to_dvec3(point) / scale(part)
}

/// `point` of `part`'s B-rep, in the world.
fn to_world(part: &CadPart, point: DVec3) -> Point3 {
    dvec3_to_point3(point * scale(part))
}

/// An element of a part's B-rep, numbered as its selections number it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartElement {
    pub part: PartId,
    pub element: SubGeometryElement,
}

impl PartElement {
    /// The edge this names, if it names an edge of a part of `document`.
    pub fn edge<'a>(&self, document: &'a Document) -> Option<PartEdge<'a>> {
        if self.element.kind != SubGeometryKind::Edge {
            return None;
        }
        PartEdge::new(document.get_part(self.part)?, self.element.index).ok()
    }
}

/// A point of a curve, and the curve's unit tangent there, pointing along it
/// (zero where the curve has no direction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurvePoint {
    pub point: Point3,
    pub tangent: Vector3,
}

/// A point, and the unit normal of a surface there, pointing out of the
/// material.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfacePoint {
    pub point: Point3,
    pub normal: Vector3,
}

/// A circle in space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    pub center: Point3,
    /// Unit normal of the circle's plane.
    pub normal: Vector3,
    pub radius: Real,
}

impl Circle {
    /// The segments tangent to this circle and `other`, each from where it
    /// touches this one to where it touches the other. Four for circles apart,
    /// two for circles that cross, none for one inside the other or for
    /// circles in different planes.
    pub fn common_tangents(&self, other: &Circle) -> Vec<(Point3, Point3)> {
        let between = other.center - self.center;
        let distance = between.magnitude();
        let tolerance = COPLANAR * (self.radius + other.radius + distance);
        let coplanar = self.normal.cross(other.normal).magnitude() <= COPLANAR
            && between.dot(self.normal).abs() <= tolerance;
        if !coplanar || distance <= Real::EPSILON {
            return Vec::new();
        }

        let x = between / distance;
        let y = self.normal.normalize().cross(x);
        let mut tangents = Vec::new();
        // Touching both circles on the same side of the line, then on opposite
        // sides (crossing between them).
        for side in [1.0, -1.0] {
            // The line's unit normal `n` makes this cosine with `x`.
            let cos = (self.radius - side * other.radius) / distance;
            if cos.abs() > 1.0 {
                continue;
            }
            let sin = (1.0 - cos * cos).sqrt();
            for turn in [1.0, -1.0] {
                let n = x * cos + y * (turn * sin);
                tangents.push((self.center + n * self.radius, other.center + n * (side * other.radius)));
            }
        }
        // Circles that touch share a tangent of no length there.
        tangents.retain(|(leaves, touches)| leaves.distance(*touches) > tolerance);
        tangents
    }
}

/// An edge of a part.
#[derive(Clone, Copy)]
pub struct PartEdge<'a> {
    part: &'a CadPart,
    index: u32,
    edge: &'a EdgeElement,
}

impl<'a> PartEdge<'a> {
    /// The edge `part`'s selections number `index`.
    pub fn new(part: &'a CadPart, index: u32) -> Result<Self> {
        Ok(Self { part, index, edge: part.edge(index)? })
    }

    pub fn element(&self) -> PartElement {
        PartElement { part: self.part.id, element: SubGeometryElement::new(SubGeometryKind::Edge, self.index) }
    }

    /// The point of the edge nearest `to`.
    pub fn nearest(&self, to: Point3) -> Result<CurvePoint> {
        let nearest = self.edge.curve().project(to_shape(self.part, to))?;
        Ok(self.in_world(nearest))
    }

    /// Every point of the edge where the line from `from` meets it at a right
    /// angle. The ends count only if it does there.
    pub fn feet(&self, from: Point3) -> Result<Vec<CurvePoint>> {
        let feet = self.edge.curve().projections(to_shape(self.part, from))?;
        Ok(feet.into_iter().map(|foot| self.in_world(foot)).collect())
    }

    /// The point halfway along the edge's length.
    pub fn midpoint(&self) -> Result<CurvePoint> {
        Ok(self.in_world(self.edge.curve().length_midpoint()?))
    }

    /// The edge's circle, if it is a circle or an arc of one.
    pub fn circle(&self) -> Option<Circle> {
        let (center, normal) = self.edge.circle_axis()?;
        let scale = scale(self.part) as Real;
        Some(Circle { center: center * scale, normal, radius: self.edge.circle_radius()? * scale })
    }

    /// The unit direction from the edge's start to its end, if it is straight.
    pub fn direction(&self) -> Option<Vector3> {
        let curve = self.edge.curve();
        (curve.edge_type() == EdgeType::Line)
            .then(|| dvec3_to_vec3(curve.end_point() - curve.start_point()).normalize())
    }

    /// Whether `point` lies on the edge, within `tolerance`.
    pub fn holds(&self, point: Point3, tolerance: Real) -> bool {
        self.nearest(point).is_ok_and(|nearest| nearest.point.distance(point) <= tolerance)
    }

    fn in_world(&self, at: EdgePoint) -> CurvePoint {
        CurvePoint { point: to_world(self.part, at.point), tangent: dvec3_to_vec3(at.tangent) }
    }
}

impl fmt::Display for PartEdge<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "edge {} of {}", self.index, self.part.name)
    }
}

/// A face of a part.
#[derive(Clone, Copy)]
pub struct PartFace<'a> {
    part: &'a CadPart,
    index: u32,
    face: &'a Face,
}

impl<'a> PartFace<'a> {
    /// The face `part`'s selections number `index`.
    pub fn new(part: &'a CadPart, index: u32) -> Result<Self> {
        Ok(Self { part, index, face: part.face(index)? })
    }

    pub fn element(&self) -> PartElement {
        PartElement { part: self.part.id, element: SubGeometryElement::new(SubGeometryKind::Face, self.index) }
    }

    pub fn is_flat(&self) -> bool {
        self.face.face_type() == FaceType::Plane
    }

    /// The point of the face nearest `to`: the foot of a perpendicular inside
    /// it, or a point of its boundary.
    pub fn nearest(&self, to: Point3) -> Result<Point3> {
        Ok(to_world(self.part, self.face.project(to_shape(self.part, to))?))
    }

    /// Every point inside the face where the line from `from` meets it at a
    /// right angle.
    pub fn feet(&self, from: Point3) -> Result<Vec<Point3>> {
        let feet = self.face.projections(to_shape(self.part, from))?;
        Ok(feet.into_iter().map(|foot| to_world(self.part, foot)).collect())
    }

    /// The face's unit normal at `point` of it, pointing out of the material.
    pub fn normal_at(&self, point: Point3) -> Result<Vector3> {
        self.normal_in_shape(to_shape(self.part, point))
    }

    /// The centre of the face's area, which a curved face may not pass
    /// through, and the face's normal at its point nearest there.
    pub fn centroid(&self) -> Result<SurfacePoint> {
        let center = self.face.center_of_mass();
        Ok(SurfacePoint { point: to_world(self.part, center), normal: self.normal_in_shape(center)? })
    }

    fn normal_in_shape(&self, point: DVec3) -> Result<Vector3> {
        // The kernel applies the face's orientation: its normal points out of
        // the material for reversed and forward faces alike.
        let normal = dvec3_to_vec3(self.face.normal_at(point)?);
        ensure!(normal.magnitude2() > 0.0, "The surface has no normal there");
        Ok(normal.normalize())
    }
}

impl fmt::Display for PartFace<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "face {} of {}", self.index, self.part.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::{Deg, Matrix4};
    use duck_engine_scene::resource::NodeId;
    use opencascade::primitives::Shape;

    use crate::testing::{bite_rim, doc_with_bitten_ball, doc_with_box, doc_with_shape, face_along, rim};

    const EPSILON: Real = 1e-6;

    fn close(a: Point3, b: Point3) -> bool {
        a.distance(b) < EPSILON
    }

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    /// A document holding a radius-2 cylinder standing on the XZ plane, 1
    /// high, and its node.
    fn doc_with_cylinder() -> (Document, NodeId) {
        doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0))
    }

    /// The face of the part at `node` facing `normal`.
    fn face(document: &Document, node: NodeId, normal: DVec3) -> PartFace<'_> {
        let part = document.part_at(node).expect("node is a part");
        PartFace::new(part, face_along(document, node, normal)).expect("the part has the face")
    }

    #[test]
    fn the_nearest_point_of_a_rim_lies_on_it_along_its_tangent() {
        let (document, _node) = doc_with_cylinder();
        let rim = rim(&document, origin());

        let nearest = rim.nearest(Point3::new(5.0, 0.0, 0.0)).unwrap();

        assert!(close(nearest.point, Point3::new(2.0, 0.0, 0.0)), "{:?}", nearest.point);
        let tangent = nearest.tangent;
        assert!((tangent.magnitude() - 1.0).abs() < EPSILON, "{tangent:?}");
        assert!(tangent.x.abs() < EPSILON && tangent.y.abs() < EPSILON, "{tangent:?}");
    }

    #[test]
    fn perpendiculars_meet_a_rim_at_its_near_and_far_points() {
        let (document, _node) = doc_with_cylinder();
        let rim = rim(&document, origin());

        let mut feet: Vec<Point3> =
            rim.feet(Point3::new(5.0, 0.0, 0.0)).unwrap().iter().map(|foot| foot.point).collect();
        feet.sort_by(|a, b| a.x.total_cmp(&b.x));

        assert_eq!(feet.len(), 2, "{feet:?}");
        assert!(close(feet[0], Point3::new(-2.0, 0.0, 0.0)) && close(feet[1], Point3::new(2.0, 0.0, 0.0)), "{feet:?}");
    }

    #[test]
    fn a_straight_edge_has_a_direction_and_a_middle_between_its_ends() {
        let (document, node) = doc_with_box();
        let part = document.part_at(node).unwrap();
        let curve = part.edge(0).unwrap().curve();
        let (start, end) = (dvec3_to_point3(curve.start_point()), dvec3_to_point3(curve.end_point()));
        let edge = PartEdge::new(part, 0).unwrap();

        let direction = edge.direction().expect("a box edge is straight");
        assert!((direction - (end - start).normalize()).magnitude() < EPSILON, "{direction:?}");

        let middle = edge.midpoint().unwrap();
        assert!(close(middle.point, start + (end - start) * 0.5), "{:?}", middle.point);
        assert!((middle.tangent - direction).magnitude() < EPSILON, "{:?}", middle.tangent);
        assert!(edge.circle().is_none());
    }

    #[test]
    fn a_rim_is_a_circle_with_no_direction() {
        let (document, _node) = doc_with_cylinder();
        let rim = rim(&document, origin());

        let circle = rim.circle().expect("a rim is a circle");

        assert!(close(circle.center, origin()), "{:?}", circle.center);
        assert!((circle.normal.y.abs() - 1.0).abs() < EPSILON, "{:?}", circle.normal);
        assert!((circle.radius - 2.0).abs() < EPSILON, "{}", circle.radius);
        assert!(rim.direction().is_none());
    }

    #[test]
    fn an_arc_holds_only_the_points_of_its_circle_it_runs_through() {
        // A half-cylinder whose base rim is the arc on z ≥ 0 (a box cut away
        // z < 0).
        let cylinder = Shape::cylinder(DVec3::ZERO, 1.0, DVec3::Y, 1.0);
        let cut = Shape::box_from_corners(DVec3::new(-2.0, -1.0, -2.0), DVec3::new(2.0, 2.0, 0.0));
        let (document, _node) = doc_with_shape(cylinder.subtract(&cut).expect("the cut succeeds").shape);
        let arc = rim(&document, origin());

        assert!(arc.holds(Point3::new(0.0, 0.0, 1.0), EPSILON));
        assert!(!arc.holds(Point3::new(0.0, 0.0, -1.0), EPSILON));
    }

    /// A ball's seam splits the rim of a bite out of it in two; the rim still
    /// answers as the one circle it is.
    #[test]
    fn a_seam_split_rim_answers_as_one_circle() {
        let (document, node) = doc_with_bitten_ball();
        let part = document.part_at(node).unwrap();
        let index = bite_rim(&document, node);
        let rim = PartEdge::new(part, index).unwrap();

        let circle = rim.circle().expect("the rim is a circle");
        let middle = rim.midpoint().unwrap().point;
        assert!((middle.distance(circle.center) - circle.radius).abs() < EPSILON, "{middle:?}");

        let pieces = part.edge(index).unwrap().pieces();
        assert_eq!(pieces.len(), 2);
        for piece in pieces {
            assert!(rim.holds(dvec3_to_point3(piece.midpoint()), EPSILON));
        }
    }

    #[test]
    fn the_nearest_point_of_a_face_beside_it_is_on_its_boundary() {
        // A 2-unit box centred on the origin; its top face is y = 1.
        let (document, node) = doc_with_box();
        let top = face(&document, node, DVec3::Y);
        let beside = Point3::new(3.0, 4.0, 0.5);

        assert!(close(top.nearest(beside).unwrap(), Point3::new(1.0, 1.0, 0.5)));
        assert!(top.feet(beside).unwrap().is_empty(), "no perpendicular lands inside the face");

        let above = Point3::new(0.3, 4.0, 0.2);
        let feet = top.feet(above).unwrap();
        let [foot] = feet.as_slice() else { panic!("expected one foot: {feet:?}") };
        assert!(close(*foot, Point3::new(0.3, 1.0, 0.2)), "{foot:?}");
        assert!(close(top.nearest(above).unwrap(), *foot));
    }

    #[test]
    fn a_flat_faces_centroid_is_its_middle_facing_out() {
        let (document, node) = doc_with_box();
        let top = face(&document, node, DVec3::Y);

        let SurfacePoint { point, normal } = top.centroid().unwrap();

        assert!(top.is_flat());
        assert!(close(point, Point3::new(0.0, 1.0, 0.0)), "{point:?}");
        assert!((normal - Vector3::unit_y()).magnitude() < EPSILON, "{normal:?}");
    }

    /// The kernel's normal on a cylinder is as long as its radius.
    #[test]
    fn the_normal_of_a_curved_face_is_unit_and_points_out() {
        let (document, node) = doc_with_cylinder();
        let part = document.part_at(node).unwrap();
        let side = part.elements().faces().iter().position(|face| face.face_type() == FaceType::Cylinder).unwrap();
        let side = PartFace::new(part, side as u32).unwrap();

        let normal = side.normal_at(Point3::new(2.0, 0.5, 0.0)).unwrap();

        assert!(!side.is_flat());
        assert!((normal - Vector3::unit_x()).magnitude() < EPSILON, "{normal:?}");
    }

    #[test]
    fn queries_follow_a_moved_part() {
        let (mut document, node) = doc_with_box();
        let part = document.part_for_node(node).unwrap();
        let moved = Matrix4::from_translation(Vector3::new(3.0, 0.0, 0.0)) * Matrix4::from_angle_y(Deg(90.0));
        document.bake_transform(part, moved).expect("the move bakes");

        let top = face(&document, node, DVec3::Y).centroid().unwrap();
        assert!(close(top.point, Point3::new(3.0, 1.0, 0.0)), "{:?}", top.point);

        // Each edge's middle is on two of the moved box's faces.
        let part = document.part_at(node).unwrap();
        for index in 0..part.elements().edges().len() as u32 {
            let middle = PartEdge::new(part, index).unwrap().midpoint().unwrap().point - Point3::new(3.0, 0.0, 0.0);
            let mut reach = [middle.x.abs(), middle.y.abs(), middle.z.abs()];
            reach.sort_by(Real::total_cmp);
            assert!(reach[0] < EPSILON && (reach[1] - 1.0).abs() < EPSILON && (reach[2] - 1.0).abs() < EPSILON, "{middle:?}");
        }
    }

    #[test]
    fn an_element_resolves_to_an_edge_only_if_it_names_one() {
        let (document, node) = doc_with_box();
        let part = document.part_for_node(node).unwrap();
        let element = |kind, index| PartElement { part, element: SubGeometryElement::new(kind, index) };

        let edge = element(SubGeometryKind::Edge, 3);
        assert_eq!(edge.edge(&document).map(|found| found.element()), Some(edge));
        assert!(element(SubGeometryKind::Face, 3).edge(&document).is_none());
        assert!(element(SubGeometryKind::Edge, 99).edge(&document).is_none());
        assert!(PartElement { part: PartId::new(), ..edge }.edge(&document).is_none());
    }

    fn circle(x: Real, radius: Real) -> Circle {
        Circle { center: Point3::new(x, 0.0, 0.0), normal: Vector3::unit_z(), radius }
    }

    #[test]
    fn common_tangents_of_circles_apart_crossing_and_nested() {
        const EXACT: Real = 1e-12;
        let tangents = circle(0.0, 1.0).common_tangents(&circle(4.0, 1.0));
        assert_eq!(tangents.len(), 4);
        let far = Point3::new(4.0, 0.0, 0.0);
        for (a, b) in &tangents {
            // Each touches its circle, square to the line between them.
            assert!((a.distance(origin()) - 1.0).abs() < EXACT && (b.distance(far) - 1.0).abs() < EXACT);
            assert!((a - origin()).dot(b - a).abs() < EXACT, "not tangent to the first: {a:?} {b:?}");
        }
        let (top, far_top) = (Point3::new(0.0, 1.0, 0.0), Point3::new(4.0, 1.0, 0.0));
        assert!(tangents.iter().any(|(a, b)| a.distance(top) < EXACT && b.distance(far_top) < EXACT));

        assert_eq!(circle(0.0, 2.0).common_tangents(&circle(3.0, 2.0)).len(), 2);
        assert!(circle(0.0, 3.0).common_tangents(&circle(1.0, 1.0)).is_empty());
    }

    #[test]
    fn circles_in_different_planes_share_no_tangent() {
        let raised = Circle { center: Point3::new(4.0, 0.0, 1.0), ..circle(4.0, 1.0) };
        assert!(circle(0.0, 1.0).common_tangents(&raised).is_empty());

        let turned = Circle { normal: Vector3::unit_y(), ..circle(4.0, 1.0) };
        assert!(circle(0.0, 1.0).common_tangents(&turned).is_empty());
    }
}
