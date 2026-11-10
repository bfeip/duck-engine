//! Snaps relating the next point of a [`Stroke`] to the points placed before
//! it: back onto its first point, along a construction axis or parallel to a
//! segment from its last ([`StrokeSnap`]), and the closest, perpendicular and
//! tangent points of the hovered geometry ([`edge_relations`],
//! [`face_relations`]).

use duck_engine_viewer::common::{InnerSpace, Point3, Ray, Vector3, EPSILON};
use duck_engine_viewer::scene::resource::SubGeometryKind;
use glam::DVec3;
use opencascade::primitives::{EdgePoint, EdgeType};

use super::providers::{PartEdge, PartFace};
use super::{Candidate, Snap, SnapFlags, SnapInput, SnapKind, SnapProvider, SnapSettings};
use crate::document::{dvec3_to_point3, dvec3_to_vec3, point3_to_dvec3, Document};

/// How far, relative to its radius, a point may lie off an arc and still be on
/// it.
const ON_ARC: f64 = 1e-6;

/// The points a tool has placed in the path it is drawing, for the snaps that
/// relate the next point to them.
#[derive(Debug, Clone)]
pub struct Stroke {
    /// The points placed so far, as they were snapped.
    placed: Vec<Snap>,
    /// Whether the path runs straight from point to point.
    straight: bool,
    /// The direction of the straight edge the cursor last snapped onto.
    acquired: Option<Vector3>,
}

impl Stroke {
    /// An empty stroke; `straight` if it runs straight from point to point,
    /// which its parallels and tangents need.
    pub fn new(straight: bool) -> Self {
        Self { placed: Vec::new(), straight, acquired: None }
    }

    /// The points placed so far.
    pub fn points(&self) -> Vec<Point3> {
        self.placed.iter().map(|snap| snap.position).collect()
    }

    /// The points, were `snap` placed next.
    pub fn points_with(&self, snap: &Snap) -> Vec<Point3> {
        let mut points = self.points();
        if let (Some(last), Some(moved)) = (points.last_mut(), snap.moves_last) {
            *last = moved;
        }
        points.push(snap.position);
        points
    }

    pub fn len(&self) -> usize {
        self.placed.len()
    }

    pub fn is_empty(&self) -> bool {
        self.placed.is_empty()
    }

    /// The point placed last.
    pub fn last(&self) -> Option<&Snap> {
        self.placed.last()
    }

    /// Places `snap` as the next point, first moving the last where it says.
    pub fn place(&mut self, snap: Snap) {
        if let (Some(last), Some(moved)) = (self.placed.last_mut(), snap.moves_last) {
            *last = Snap { position: moved, direction: snap.direction, ..*last };
        }
        self.placed.push(Snap { moves_last: None, ..snap });
    }

    /// Drops the points placed and the edge noted.
    pub fn clear(&mut self) {
        self.placed.clear();
        self.acquired = None;
    }

    /// Takes note of `snap`, where the cursor is: the straight edge it lies on
    /// becomes the one the next points may run parallel to.
    pub fn observe(&mut self, snap: &Snap, document: &Document) {
        let Some(on) = snap.on.filter(|on| on.element.kind == SubGeometryKind::Edge) else { return };
        let Some(edge) = document.get_part(on.part).and_then(|part| part.edge(on.element.index).ok()) else {
            return;
        };
        let curve = edge.curve();
        if curve.edge_type() == EdgeType::Line {
            self.acquired = Some(dvec3_to_vec3(curve.end_point() - curve.start_point()).normalize());
        }
    }

    /// The directions a parallel from the last point runs in: each segment's
    /// so far, and the noted edge's.
    fn parallels(&self) -> impl Iterator<Item = Vector3> + '_ {
        let segments = self.placed.windows(2).map(|pair| pair[1].position - pair[0].position);
        segments
            .filter(|segment| segment.magnitude() > EPSILON)
            .map(InnerSpace::normalize)
            .chain(self.acquired)
    }
}

/// The stroke's own snaps: its first point, to close it on
/// ([`SnapKind::WireStart`]), and lines from its last point along the
/// construction axes ([`SnapKind::AxisAligned`]) and, on a straight stroke,
/// parallel to its segments and the edge it noted ([`SnapKind::Parallel`]).
pub(super) struct StrokeSnap;

impl SnapProvider for StrokeSnap {
    fn produces(&self) -> SnapFlags {
        SnapFlags::WIRE_START | SnapFlags::AXIS_ALIGNED | SnapFlags::PARALLEL
    }

    fn collect(&self, input: &SnapInput, _document: &Document, _settings: &SnapSettings) -> Vec<Candidate> {
        let Some(stroke) = input.stroke else { return Vec::new() };
        let (Some(first), Some(last)) = (stroke.placed.first(), stroke.last()) else { return Vec::new() };
        let mut out = Vec::new();
        // It takes three points to enclose anything.
        if stroke.len() >= 3 && input.requested.contains(SnapFlags::WIRE_START) {
            out.push(Snap::new(first.position, SnapKind::WireStart).into());
        }
        if input.requested.contains(SnapFlags::AXIS_ALIGNED) {
            let (u, v) = input.plane.basis();
            for axis in [u, v, input.plane.normal.normalize()] {
                out.extend(along(&input.ray, last.position, axis, SnapKind::AxisAligned));
            }
        }
        if stroke.straight && input.requested.contains(SnapFlags::PARALLEL) {
            for direction in stroke.parallels() {
                out.extend(along(&input.ray, last.position, direction, SnapKind::Parallel));
            }
        }
        out
    }
}

/// The point of the line through `origin` along unit `direction` that `ray`
/// passes nearest, as a `kind` snap along the line.
fn along(ray: &Ray, origin: Point3, direction: Vector3, kind: SnapKind) -> Option<Candidate> {
    let t = ray.closest_param_on_axis(origin, direction)?;
    Some(Snap::new(origin + direction * t, kind).with_direction(direction).into())
}

/// Pushes into `out` the `kinds` of snap relating `edge`, hovered at `found`,
/// to `stroke`'s last point: the edge's point closest to it, the feet of its
/// perpendiculars from it, and on a straight stroke, the points where segments
/// from the circle it lies on touch the edge's, tangent to both.
pub(super) fn edge_relations(
    edge: &PartEdge,
    found: Point3,
    stroke: &Stroke,
    kinds: SnapFlags,
    document: &Document,
    out: &mut Vec<Candidate>,
) {
    let Some(last) = stroke.last() else { return };
    let PartEdge { part, index } = *edge;
    let curve = match part.edge(index) {
        Ok(element) => element.curve(),
        Err(e) => {
            log::debug!("No relations to edge {index} of {}: {e}", part.name);
            return;
        }
    };
    // The B-rep is in the world, scaled into the scene at tessellation.
    let scale = part.options().scale_factor;
    let from = point3_to_dvec3(last.position) / scale;
    let relation = |kind, at: EdgePoint| Candidate {
        snap: Snap::new(dvec3_to_point3(at.point * scale), kind)
            .with_direction(dvec3_to_vec3(at.tangent))
            .with_on(edge.element()),
        found_at: found,
    };

    if kinds.contains(SnapFlags::CLOSEST) {
        match curve.project(from) {
            Ok(nearest) if apart(nearest.point, from) => out.push(relation(SnapKind::Closest, nearest)),
            Ok(_) => {}
            Err(e) => log::debug!("No closest snap on edge {index} of {}: {e}", part.name),
        }
    }
    if kinds.contains(SnapFlags::PERPENDICULAR) {
        match curve.projections(from) {
            Ok(feet) => out.extend(
                feet.into_iter()
                    .filter(|foot| apart(foot.point, from))
                    .map(|foot| relation(SnapKind::Perpendicular, foot)),
            ),
            Err(e) => log::debug!("No perpendicular snap on edge {index} of {}: {e}", part.name),
        }
    }
    if stroke.straight && kinds.contains(SnapFlags::TANGENT_TANGENT) {
        for (leaves, touches) in tangents_from(last, edge, document) {
            let snap = Snap::new(dvec3_to_point3(touches), SnapKind::TangentTangent)
                .with_direction(dvec3_to_vec3((touches - leaves).normalize()))
                .with_on(edge.element());
            let moves_last = Some(dvec3_to_point3(leaves));
            out.push(Candidate { snap: Snap { moves_last, ..snap }, found_at: found });
        }
    }
}

/// Pushes into `out` the `kinds` of snap relating `face`, hovered at `found`,
/// to `stroke`'s last point: the face's point closest to it, and the feet of
/// its perpendiculars from it.
pub(super) fn face_relations(
    face: &PartFace,
    found: Point3,
    stroke: &Stroke,
    kinds: SnapFlags,
    out: &mut Vec<Candidate>,
) {
    let Some(last) = stroke.last() else { return };
    let PartFace { part, index } = *face;
    let surface = match part.face(index) {
        Ok(surface) => surface,
        Err(e) => {
            log::debug!("No relations to face {index} of {}: {e}", part.name);
            return;
        }
    };
    // The B-rep is in the world, scaled into the scene at tessellation.
    let scale = part.options().scale_factor;
    let from = point3_to_dvec3(last.position) / scale;
    let relation = |kind, point: DVec3| {
        let snap = Snap::new(dvec3_to_point3(point * scale), kind).with_on(face.element());
        let snap = match surface.normal_at(point) {
            Ok(normal) => snap.with_direction(dvec3_to_vec3(normal)),
            Err(_) => snap,
        };
        Candidate { snap, found_at: found }
    };

    if kinds.contains(SnapFlags::CLOSEST) {
        match surface.project(from) {
            Ok(point) if apart(point, from) => out.push(relation(SnapKind::Closest, point)),
            Ok(_) => {}
            Err(e) => log::debug!("No closest snap on face {index} of {}: {e}", part.name),
        }
    }
    if kinds.contains(SnapFlags::PERPENDICULAR) {
        match surface.projections(from) {
            Ok(feet) => out.extend(
                feet.into_iter()
                    .filter(|&point| apart(point, from))
                    .map(|point| relation(SnapKind::Perpendicular, point)),
            ),
            Err(e) => log::debug!("No perpendicular snap on face {index} of {}: {e}", part.name),
        }
    }
}

/// Whether `point` is far enough from `from` to make a segment.
fn apart(point: DVec3, from: DVec3) -> bool {
    point.distance(from) > f64::from(EPSILON)
}

/// The segments from the circle `last` lies on to the circle `edge` is, tangent
/// to both, where each touches the arcs themselves: where they leave and where
/// they touch `edge`, in the world.
fn tangents_from(last: &Snap, edge: &PartEdge, document: &Document) -> Vec<(DVec3, DVec3)> {
    let Some(on) = last.on.filter(|on| on.element.kind == SubGeometryKind::Edge) else { return Vec::new() };
    if on == edge.element() {
        return Vec::new();
    }
    let Some(from) = document.get_part(on.part).map(|part| PartEdge { part, index: on.element.index }) else {
        return Vec::new();
    };
    let (Some(leaving), Some(touched)) = (Circle::of(from), Circle::of(*edge)) else { return Vec::new() };
    // Coplanar: one normal, and both centres in the plane.
    let size = leaving.radius + touched.radius + leaving.center.distance(touched.center);
    let tolerance = ON_ARC * size;
    let coplanar = leaving.normal.cross(touched.normal).length() <= ON_ARC
        && (touched.center - leaving.center).dot(leaving.normal).abs() <= tolerance;
    if !coplanar {
        return Vec::new();
    }

    let tangents = common_tangents(leaving.center, leaving.radius, touched.center, touched.radius, leaving.normal);
    tangents
        .into_iter()
        .filter(|&(leaves, touches)| leaves.distance(touches) > tolerance)
        .filter(|&(leaves, touches)| leaving.holds(leaves) && touched.holds(touches))
        .collect()
}

/// A part's circular edge, in the world.
struct Circle<'a> {
    edge: PartEdge<'a>,
    center: DVec3,
    normal: DVec3,
    radius: f64,
}

impl<'a> Circle<'a> {
    /// `edge`, if it is a circle or an arc of one.
    fn of(edge: PartEdge<'a>) -> Option<Self> {
        let PartEdge { part, index } = edge;
        let curve = part.edge(index).ok()?.curve();
        let (center, normal) = curve.circle_axis()?;
        let radius = curve.circle_radius()?;
        let scale = part.options().scale_factor;
        Some(Self { edge, center: center * scale, normal, radius: radius * scale })
    }

    /// Whether `point`, on the edge's circle, lies on the edge itself.
    fn holds(&self, point: DVec3) -> bool {
        let PartEdge { part, index } = self.edge;
        let scale = part.options().scale_factor;
        let Ok(element) = part.edge(index) else { return false };
        element
            .curve()
            .project(point / scale)
            .is_ok_and(|nearest| (nearest.point * scale).distance(point) <= ON_ARC * self.radius)
    }
}

/// The lines tangent to the circles about `c1` and `c2` of radii `r1` and `r2`,
/// both in the plane through them facing `normal`: where each touches the
/// first circle and the second. Four for circles apart, two for circles that
/// cross, none for one inside the other.
fn common_tangents(c1: DVec3, r1: f64, c2: DVec3, r2: f64, normal: DVec3) -> Vec<(DVec3, DVec3)> {
    let between = c2 - c1;
    let distance = between.length();
    if distance <= f64::EPSILON {
        return Vec::new();
    }
    let x = between / distance;
    let y = normal.normalize().cross(x);
    let mut tangents = Vec::new();
    // Touching both circles on the same side of the line, then on opposite
    // sides (crossing between them).
    for side in [1.0, -1.0] {
        // The line's unit normal `n` makes this cosine with `x`.
        let cos = (r1 - side * r2) / distance;
        if cos.abs() > 1.0 {
            continue;
        }
        let sin = (1.0 - cos * cos).sqrt();
        for turn in [1.0, -1.0] {
            let n = x * cos + y * (turn * sin);
            tangents.push((c1 + n * r1, c2 + n * (side * r2)));
        }
    }
    tangents
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_viewer::common::{Plane, Real};
    use duck_engine_viewer::scene::{PositionedCamera, Projection, Scene};
    use opencascade::primitives::Shape;

    use duck_engine_viewer::common::Transform;
    use duck_engine_viewer::scene::cad::CadTessellationOptions;

    use crate::grid::GridConfig;
    use crate::testing::{doc_with_box, doc_with_shape};

    const CLOSE: Real = 1e-6;

    fn close(a: Point3, b: Point3) -> bool {
        (a - b).magnitude() < CLOSE
    }

    fn at(x: Real, y: Real, z: Real) -> Snap {
        Snap::new(Point3::new(x, y, z), SnapKind::ConstructionPlane)
    }

    fn stroke(straight: bool, points: &[Snap]) -> Stroke {
        let mut stroke = Stroke::new(straight);
        for &point in points {
            stroke.place(point);
        }
        stroke
    }

    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 10.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 0.0, -1.0),
            aspect: 4.0 / 3.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.01, zfar: 1000.0 },
        }
    }

    /// The stroke's own snaps for a ray straight down through `(x, z)`.
    fn stroke_snaps(stroke: &Stroke, x: Real, z: Real) -> Vec<Snap> {
        let cam = camera();
        let plane = Plane::xz();
        let grid = GridConfig::default();
        let input = SnapInput {
            ray: Ray::new(Point3::new(x, 10.0, z), Vector3::new(0.0, -1.0, 0.0)),
            cursor: (0.0, 0.0),
            viewport: (800, 600),
            camera: &cam,
            plane: &plane,
            grid: &grid,
            requested: SnapFlags::all(),
            exclude_nodes: &[],
            stroke: Some(stroke),
        };
        let document = Document::new(Scene::default());
        let candidates = StrokeSnap.collect(&input, &document, &SnapSettings::default());
        candidates.into_iter().map(|candidate| candidate.snap).collect()
    }

    fn of_kind(snaps: &[Snap], kind: SnapKind) -> Vec<Snap> {
        snaps.iter().copied().filter(|snap| snap.kind == kind).collect()
    }

    /// The edge of a part of `document` that is a circle, or an arc of one,
    /// about `center`.
    fn rim(document: &Document, center: Point3) -> PartEdge<'_> {
        document
            .parts()
            .find_map(|part| {
                let mut edges = part.elements().edges().iter();
                let index = edges.position(|edge| edge.circle_axis().is_some_and(|(c, _)| close(c, center)))?;
                Some(PartEdge { part, index: index as u32 })
            })
            .expect("a rim about the centre")
    }

    /// A document holding `shape`, and a radius-1 cylinder standing on
    /// (4, 0, 0) along +Y.
    fn beside_a_cylinder(shape: Shape) -> Document {
        let (mut document, _node) = doc_with_shape(shape);
        let cylinder = Shape::cylinder(DVec3::new(4.0, 0.0, 0.0), 1.0, DVec3::Y, 1.0);
        let options = CadTessellationOptions::default();
        document.add_part("cylinder", cylinder, Transform::IDENTITY, &options).expect("the cylinder tessellates");
        document
    }

    /// A last point on `edge`.
    fn on(edge: &PartEdge, position: Point3) -> Snap {
        Snap::new(position, SnapKind::Edge).with_on(edge.element())
    }

    fn relations(edge: &PartEdge, stroke: &Stroke, kinds: SnapFlags, document: &Document) -> Vec<Snap> {
        let mut out = Vec::new();
        edge_relations(edge, Point3::new(0.0, 0.0, 0.0), stroke, kinds, document, &mut out);
        out.into_iter().map(|candidate| candidate.snap).collect()
    }

    #[test]
    fn placing_a_snap_that_moves_the_last_point_moves_it() {
        let mut stroke = stroke(true, &[at(0.0, 0.0, 0.0), at(1.0, 0.0, 0.0)]);
        let moving = Snap { moves_last: Some(Point3::new(1.0, 0.0, 1.0)), ..at(3.0, 0.0, 1.0) };

        let preview = stroke.points_with(&moving);
        stroke.place(moving);

        let expected = [Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 1.0), Point3::new(3.0, 0.0, 1.0)];
        assert_eq!(preview, expected);
        assert_eq!(stroke.points(), expected);
        assert!(stroke.last().unwrap().moves_last.is_none());
    }

    #[test]
    fn the_start_is_offered_once_three_points_can_enclose() {
        let two = stroke(false, &[at(0.0, 0.0, 0.0), at(2.0, 0.0, 0.0)]);
        assert!(of_kind(&stroke_snaps(&two, 0.1, 0.1), SnapKind::WireStart).is_empty());

        let three = stroke(false, &[at(0.0, 0.0, 0.0), at(2.0, 0.0, 0.0), at(1.0, 0.0, 2.0)]);
        let starts = of_kind(&stroke_snaps(&three, 0.1, 0.1), SnapKind::WireStart);
        let [start] = starts.as_slice() else { panic!("expected the start: {starts:?}") };
        assert!(close(start.position, Point3::new(0.0, 0.0, 0.0)));
    }

    #[test]
    fn axis_lines_run_through_the_last_point() {
        let stroke = stroke(false, &[at(1.0, 0.0, 2.0)]);

        let axes = of_kind(&stroke_snaps(&stroke, 4.0, 2.3), SnapKind::AxisAligned);

        // Looking straight down N, only U (X) and V (Z) have a nearest point.
        assert_eq!(axes.len(), 2, "{axes:?}");
        let along_x = axes.iter().find(|snap| snap.direction.unwrap().x.abs() > 0.9).unwrap();
        assert!(close(along_x.position, Point3::new(4.0, 0.0, 2.0)), "{:?}", along_x.position);
        let along_z = axes.iter().find(|snap| snap.direction.unwrap().z.abs() > 0.9).unwrap();
        assert!(close(along_z.position, Point3::new(1.0, 0.0, 2.3)), "{:?}", along_z.position);
    }

    #[test]
    fn parallels_follow_each_segment_on_straight_strokes_only() {
        let points = [at(0.0, 0.0, 0.0), at(2.0, 0.0, 2.0), at(4.0, 0.0, 2.0)];

        // Parallel to the first segment through (4, 0, 2): the cursor at (5, 3)
        // lies on it.
        let parallels = of_kind(&stroke_snaps(&stroke(true, &points), 5.0, 3.0), SnapKind::Parallel);
        assert!(parallels.iter().any(|snap| close(snap.position, Point3::new(5.0, 0.0, 3.0))), "{parallels:?}");

        assert!(of_kind(&stroke_snaps(&stroke(false, &points), 5.0, 3.0), SnapKind::Parallel).is_empty());
    }

    #[test]
    fn noting_a_straight_edge_offers_its_parallel() {
        let (document, _node) = doc_with_box();
        let part = document.parts().next().unwrap();
        let straight = (0..part.elements().edges().len() as u32)
            .find(|&index| part.edge(index).unwrap().curve().edge_type() == EdgeType::Line)
            .unwrap();
        let curve = part.edge(straight).unwrap().curve();
        let direction = dvec3_to_vec3(curve.end_point() - curve.start_point()).normalize();
        let edge = PartEdge { part, index: straight };

        let mut stroke = stroke(true, &[at(5.0, 0.0, 5.0)]);
        stroke.observe(&on(&edge, Point3::new(0.0, 0.0, 0.0)), &document);

        let parallels: Vec<Vector3> = stroke.parallels().collect();
        let [noted] = parallels.as_slice() else { panic!("expected the noted edge: {parallels:?}") };
        assert!((noted - direction).magnitude() < CLOSE, "{noted:?}");
    }

    #[test]
    fn curved_edges_are_not_noted() {
        let cylinder = Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0);
        let (document, _node) = doc_with_shape(cylinder);
        let rim = rim(&document, Point3::new(0.0, 0.0, 0.0));

        let mut stroke = stroke(true, &[at(5.0, 0.0, 5.0)]);
        stroke.observe(&on(&rim, Point3::new(2.0, 0.0, 0.0)), &document);

        assert_eq!(stroke.parallels().count(), 0);
    }

    #[test]
    fn common_tangents_of_circles_apart_crossing_and_nested() {
        const EXACT: f64 = 1e-12;
        let tangents = common_tangents(DVec3::ZERO, 1.0, DVec3::new(4.0, 0.0, 0.0), 1.0, DVec3::Z);
        assert_eq!(tangents.len(), 4);
        for (a, b) in &tangents {
            // Each touches its circle, square to the line between them.
            assert!((a.length() - 1.0).abs() < EXACT && ((*b - DVec3::new(4.0, 0.0, 0.0)).length() - 1.0).abs() < EXACT);
            assert!(a.dot(*b - *a).abs() < EXACT, "not tangent to the first: {a} {b}");
        }
        assert!(tangents.iter().any(|(a, b)| (*a - DVec3::Y).length() < EXACT && (*b - DVec3::new(4.0, 1.0, 0.0)).length() < EXACT));

        assert_eq!(common_tangents(DVec3::ZERO, 2.0, DVec3::new(3.0, 0.0, 0.0), 2.0, DVec3::Z).len(), 2);
        assert!(common_tangents(DVec3::ZERO, 3.0, DVec3::new(1.0, 0.0, 0.0), 1.0, DVec3::Z).is_empty());
    }

    #[test]
    fn closest_and_perpendicular_on_a_rim() {
        // A radius-2 cylinder standing on the XZ plane; the last point is out
        // along +X at its base.
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let rim = rim(&document, Point3::new(0.0, 0.0, 0.0));
        let stroke = stroke(false, &[at(5.0, 0.0, 0.0)]);

        let closest = relations(&rim, &stroke, SnapFlags::CLOSEST, &document);
        let [closest] = closest.as_slice() else { panic!("expected one: {closest:?}") };
        assert_eq!(closest.kind, SnapKind::Closest);
        assert!(close(closest.position, Point3::new(2.0, 0.0, 0.0)), "{:?}", closest.position);
        assert_eq!(closest.on, Some(rim.element()));

        let mut feet: Vec<Point3> =
            relations(&rim, &stroke, SnapFlags::PERPENDICULAR, &document).iter().map(|snap| snap.position).collect();
        feet.sort_by(|a, b| a.x.total_cmp(&b.x));
        assert_eq!(feet.len(), 2, "{feet:?}");
        assert!(close(feet[0], Point3::new(-2.0, 0.0, 0.0)) && close(feet[1], Point3::new(2.0, 0.0, 0.0)), "{feet:?}");
    }

    #[test]
    fn closest_on_a_face_beside_it_is_on_its_boundary() {
        // A 2-unit box centred on the origin, the last point above and beside
        // its top face y = 1.
        let (document, _node) = doc_with_box();
        let part = document.parts().next().unwrap();
        let top = (0..part.elements().faces().len() as u32)
            .find(|&index| (part.face(index).unwrap().center_of_mass().y - 1.0).abs() < 1e-9)
            .unwrap();
        let face = PartFace { part, index: top };
        let stroke = stroke(false, &[at(3.0, 4.0, 0.5)]);

        let mut out = Vec::new();
        face_relations(&face, Point3::new(0.0, 1.0, 0.0), &stroke, SnapFlags::CLOSEST | SnapFlags::PERPENDICULAR, &mut out);

        let [closest] = out.as_slice() else { panic!("expected only the closest: {out:?}") };
        assert_eq!(closest.snap.kind, SnapKind::Closest);
        assert!(close(closest.snap.position, Point3::new(1.0, 1.0, 0.5)), "{:?}", closest.snap.position);
        assert_eq!(closest.snap.on.map(|on| on.element.kind), Some(SubGeometryKind::Face));
    }

    #[test]
    fn tangent_tangent_moves_the_last_point_round_its_circle() {
        let document = beside_a_cylinder(Shape::cylinder(DVec3::ZERO, 1.0, DVec3::Y, 1.0));
        let from = rim(&document, Point3::new(0.0, 0.0, 0.0));
        let to = rim(&document, Point3::new(4.0, 0.0, 0.0));
        let stroke = stroke(true, &[on(&from, Point3::new(-1.0, 0.0, 0.0))]);

        let tangents = relations(&to, &stroke, SnapFlags::TANGENT_TANGENT, &document);

        assert_eq!(tangents.len(), 4, "{tangents:?}");
        let outer = tangents
            .iter()
            .find(|snap| close(snap.position, Point3::new(4.0, 0.0, 1.0)))
            .unwrap_or_else(|| panic!("expected the outer tangent along z = 1: {tangents:?}"));
        assert!(close(outer.moves_last.unwrap(), Point3::new(0.0, 0.0, 1.0)), "{:?}", outer.moves_last);
        assert!(outer.direction.is_some_and(|d| (d - Vector3::new(1.0, 0.0, 0.0)).magnitude() < CLOSE));

        // Not on a curved stroke, and not from a point on no circle.
        assert!(relations(&to, &self::stroke(false, stroke.placed.as_slice()), SnapFlags::TANGENT_TANGENT, &document).is_empty());
        let loose = self::stroke(true, &[at(-1.0, 0.0, 0.0)]);
        assert!(relations(&to, &loose, SnapFlags::TANGENT_TANGENT, &document).is_empty());
    }

    #[test]
    fn tangents_touching_off_an_arc_are_dropped() {
        // A half-cylinder whose base rim is the arc on z ≥ 0 (a box cut away
        // z < 0), beside a full cylinder.
        let cylinder = Shape::cylinder(DVec3::ZERO, 1.0, DVec3::Y, 1.0);
        let cut = Shape::box_from_corners(DVec3::new(-2.0, -1.0, -2.0), DVec3::new(2.0, 2.0, 0.0));
        let document = beside_a_cylinder(cylinder.subtract(&cut).expect("the cut succeeds").shape);
        let arc = rim(&document, Point3::new(0.0, 0.0, 0.0));
        let to = rim(&document, Point3::new(4.0, 0.0, 0.0));
        let stroke = stroke(true, &[on(&arc, Point3::new(0.0, 0.0, 1.0))]);

        let tangents = relations(&to, &stroke, SnapFlags::TANGENT_TANGENT, &document);

        // Of the four, two leave the arc where z ≥ 0: the outer along z = 1,
        // and the crossing one leaving at z = √3/2.
        assert_eq!(tangents.len(), 2, "{tangents:?}");
        for tangent in &tangents {
            assert!(tangent.moves_last.unwrap().z > 0.0, "{tangent:?}");
        }
    }

    #[test]
    fn a_point_on_the_hovered_edge_relates_to_nothing_there() {
        let (document, _node) = doc_with_shape(Shape::cylinder(DVec3::ZERO, 2.0, DVec3::Y, 1.0));
        let rim = rim(&document, Point3::new(0.0, 0.0, 0.0));
        let stroke = stroke(true, &[on(&rim, Point3::new(2.0, 0.0, 0.0))]);

        let snaps = relations(&rim, &stroke, SnapFlags::all(), &document);

        // The far foot only: the closest point is the last point itself, and a
        // circle has no tangent to itself.
        let [foot] = snaps.as_slice() else { panic!("expected the far foot: {snaps:?}") };
        assert_eq!(foot.kind, SnapKind::Perpendicular);
        assert!(close(foot.position, Point3::new(-2.0, 0.0, 0.0)), "{:?}", foot.position);
    }
}
