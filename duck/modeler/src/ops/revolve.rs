use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::{consts, InnerSpace, MetricSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::{Compound, Shape, Wire};

use crate::document::{
    dvec3_to_point3, has_solid, interactive_fuzz, point3_to_dvec3, vec3_to_dvec3, Document,
    SourceFate,
};

/// A sweep at or below this many radians is degenerate: there is nothing to
/// revolve.
const MIN_ANGLE: Real = 1e-6;

/// Axis picks closer together than this define no direction.
const MIN_AXIS_LENGTH: Real = 1e-6;

/// What a revolve sweeps, on one part, identified the way the selection system
/// reports it: by tessellation order within the part's mesh, primary first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RevolveTarget {
    pub node: NodeId,
    pub profile: Profile,
}

/// The sub-geometry a revolve sweeps.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Profile {
    /// Faces, each sweeping into a solid.
    Faces(Vec<u32>),
    /// Edges, each sweeping into a sheet. An edge of a sketch brings the rest
    /// of its wire.
    Edges(Vec<u32>),
}

impl Profile {
    /// How many faces or edges the profile names.
    pub fn count(&self) -> usize {
        match self {
            Profile::Faces(indices) | Profile::Edges(indices) => indices.len(),
        }
    }
}

/// Where a revolve's profile stands, resolved once from its target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevolveFrame {
    /// The middle of the primary face or edge, which the angle grip swings
    /// from.
    pub lever: Point3,
    /// The diagonal of the profile's bounding box.
    pub size: Real,
    /// What the revolve does with the part it sweeps from.
    pub fate: SourceFate,
}

impl RevolveFrame {
    /// Resolves `target` against its part.
    pub fn new(doc: &Document, target: &RevolveTarget) -> Result<Self> {
        let part = doc.part_at(target.node).context("The revolve target is not a known CAD part")?;
        let elements = part.elements();
        let lever = match &target.profile {
            Profile::Faces(faces) => elements.face(primary(faces)?)?.midpoint(),
            Profile::Edges(edges) => elements.edge(primary(edges)?)?.curve().midpoint(),
        };
        let bounds = opencascade::bounding_box::aabb(&profile(doc, target)?);
        let size = (bounds.max() - bounds.min()).length() as Real;

        // A bare sketch becomes its revolution. A face of a solid sweeps out
        // more of the solid; an edge of one sweeps a sheet beside it.
        let fate = match (&target.profile, has_solid(part.shape())) {
            (_, false) => SourceFate::Replace,
            (Profile::Faces(_), true) => SourceFate::Fuse,
            (Profile::Edges(_), true) => SourceFate::Keep,
        };
        Ok(Self { lever: dvec3_to_point3(lever), size, fate })
    }
}

/// The first of a profile's indices.
fn primary(indices: &[u32]) -> Result<usize> {
    indices.first().map(|&index| index as usize).context("The profile is empty")
}

/// The axis a revolve turns about, as far as it has been picked.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Axis {
    /// Nothing picked yet.
    Unpicked,
    /// Its start picked.
    From(Point3),
    /// Through both picks, pointing from the first to the second.
    Through(Point3, Point3),
}

/// A revolve as configured: the profile it sweeps, the axis it turns about and
/// how far it turns.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RevolveParams {
    pub frame: RevolveFrame,
    pub axis: Axis,
    /// Signed sweep in radians, within ±2π: positive turns counter-clockwise
    /// looking down the axis.
    pub angle: Real,
}

impl RevolveParams {
    /// A full turn about an axis not yet picked.
    pub fn new(frame: RevolveFrame) -> Self {
        Self { frame, axis: Axis::Unpicked, angle: consts::TAU }
    }

    /// A point on the axis and its unit direction, once both ends are picked.
    pub fn line(&self) -> Option<(Point3, Vector3)> {
        let Axis::Through(from, to) = self.axis else { return None };
        let along = to - from;
        (along.magnitude() > MIN_AXIS_LENGTH).then(|| (from, along.normalize()))
    }

    /// Whether the revolve has no axis yet, or too small a sweep to build.
    pub fn is_degenerate(&self) -> bool {
        self.line().is_none() || self.angle.abs() <= MIN_ANGLE
    }
}

/// Whether `from` and `to` are far enough apart to pick an axis.
pub fn spans_axis(from: Point3, to: Point3) -> bool {
    from.distance(to) > MIN_AXIS_LENGTH
}

/// The shape the target sweeps: its faces, or its edges, those of a sketch
/// each with the rest of its wire, once.
fn profile(doc: &Document, target: &RevolveTarget) -> Result<Shape> {
    let part = doc.part_at(target.node).context("The revolve target is not a known CAD part")?;
    let elements = part.elements();
    let mut pieces: Vec<Shape> = Vec::new();
    match &target.profile {
        Profile::Faces(faces) => {
            for &index in faces {
                pieces.push(elements.face(index as usize)?.into());
            }
        }
        Profile::Edges(edges) if has_solid(part.shape()) => {
            for &index in edges {
                pieces.push(elements.edge(index as usize)?.curve().into());
            }
        }
        Profile::Edges(edges) => {
            let mut wires: Vec<Wire> = Vec::new();
            for &index in edges {
                let element = elements.edge(index as usize)?;
                let listed = |wire: &Wire| wire.edges().any(|edge| element.pieces().iter().any(|piece| edge.is_same(piece)));
                if wires.iter().any(listed) {
                    continue;
                }
                match doc.wire_for_edge(target.node, index) {
                    Some(wire) => wires.push(wire),
                    None => pieces.push(element.curve().into()),
                }
            }
            pieces.extend(wires.iter().map(Shape::from));
        }
    }
    Ok(match <[Shape; 1]>::try_from(pieces) {
        Ok([piece]) => piece,
        Err(pieces) => Compound::from_shapes(pieces).into(),
    })
}

/// `point`, moved onto the vertex of `shape` it was picked at, if any.
///
/// Picks snap to corners read from the `f32` vertex buffer, a few ulps off the
/// B-rep: enough to leave a profile edge meant to lie on the axis a sliver
/// away from it, which sweeps into a hairline tube.
fn onto_vertex(point: Point3, shape: &Shape) -> Point3 {
    let tolerance = interactive_fuzz(std::iter::once(shape)) as Real;
    shape
        .vertices()
        .map(|vertex| dvec3_to_point3(vertex.point()))
        .map(|vertex| (vertex.distance(point), vertex))
        .filter(|&(distance, _)| distance <= tolerance)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(point, |(_, vertex)| vertex)
}

/// The revolved profile, to preview and to commit.
pub fn build_revolve(doc: &Document, target: &RevolveTarget, params: &RevolveParams) -> Result<Shape> {
    let Axis::Through(from, to) = params.axis else { anyhow::bail!("Pick an axis to revolve about") };
    ensure!(!params.is_degenerate(), "The revolve has no sweep");
    let part = doc.part_at(target.node).context("The revolve target is not a known CAD part")?;
    let (from, to) = (onto_vertex(from, part.shape()), onto_vertex(to, part.shape()));
    ensure!(spans_axis(from, to), "The axis has no length");

    // OCCT turns only forward: a negative sweep turns forward about the
    // reversed axis.
    let direction = (to - from).normalize() * params.angle.signum();
    let angle = params.angle.abs().min(consts::TAU);
    let revolved = profile(doc, target)?
        .revolve(point3_to_dvec3(from), vec3_to_dvec3(direction), f64::from(angle))
        .context("The profile could not be revolved")?;
    ensure!(revolved.is_valid(), "The revolved profile is not a valid shape");
    Ok(revolved)
}

#[cfg(test)]
mod tests {
    use super::*;

    use opencascade::primitives::ShapeType;

    use crate::ops::primitives::polyline;
    use crate::testing::{doc_with_box, doc_with_shape, xy_region};

    const EPSILON: f64 = 1e-6;

    /// A 1 × 2 region standing on the origin, its left edge on the Y axis.
    fn upright() -> Shape {
        xy_region(&[(0.0, 0.0), (1.0, 0.0), (1.0, 2.0), (0.0, 2.0)])
    }

    fn faces(node: NodeId) -> RevolveTarget {
        RevolveTarget { node, profile: Profile::Faces(vec![0]) }
    }

    /// `target` turned `angle` about the axis from `from` to `to`.
    fn params(doc: &Document, target: &RevolveTarget, from: Point3, to: Point3, angle: Real) -> RevolveParams {
        let frame = RevolveFrame::new(doc, target).expect("target resolves");
        RevolveParams { axis: Axis::Through(from, to), angle, ..RevolveParams::new(frame) }
    }

    fn y_axis() -> (Point3, Point3) {
        (Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0))
    }

    fn revolved(shape: Shape, target: impl Fn(NodeId) -> RevolveTarget, angle: Real) -> Result<Shape> {
        let (doc, node) = doc_with_shape(shape);
        let target = target(node);
        let (from, to) = y_axis();
        build_revolve(&doc, &target, &params(&doc, &target, from, to, angle))
    }

    #[test]
    fn a_full_turn_about_the_regions_edge_is_a_cylinder() {
        let solid = revolved(upright(), faces, consts::TAU).expect("revolves");
        assert!(solid.contains_type(ShapeType::Solid));
        assert!((solid.volume() - 2.0 * std::f64::consts::PI).abs() < EPSILON, "{}", solid.volume());
    }

    #[test]
    fn a_quarter_turn_is_a_quarter_cylinder() {
        let solid = revolved(upright(), faces, consts::FRAC_PI_2).expect("revolves");
        assert!((solid.volume() - std::f64::consts::FRAC_PI_2).abs() < EPSILON, "{}", solid.volume());
    }

    /// Counter-clockwise looking down +Y carries +X toward −Z; a negative
    /// sweep goes the other way.
    #[test]
    fn a_negative_sweep_turns_the_other_way() {
        let forward = revolved(upright(), faces, consts::FRAC_PI_2).expect("revolves");
        let backward = revolved(upright(), faces, -consts::FRAC_PI_2).expect("revolves");
        assert!((forward.volume() - backward.volume()).abs() < EPSILON);

        let forward = opencascade::bounding_box::aabb(&forward);
        let backward = opencascade::bounding_box::aabb(&backward);
        assert!(forward.min().z < -0.5 && forward.max().z < EPSILON, "{:?}", forward.min());
        assert!(backward.max().z > 0.5 && backward.min().z > -EPSILON, "{:?}", backward.max());
    }

    /// Clicking one segment of a polyline sketch revolves the whole polyline.
    #[test]
    fn an_edge_of_a_sketch_brings_its_whole_wire() {
        let points = [(1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0)].map(|(x, y)| Point3::new(x, y, 0.0));
        let line = Shape::from(polyline(&points).expect("polyline builds"));
        let edges = |node| RevolveTarget { node, profile: Profile::Edges(vec![1]) };
        let sheet = revolved(line, edges, consts::TAU).expect("revolves");
        assert_eq!(sheet.faces().count(), 3);
        assert!(!sheet.contains_type(ShapeType::Solid));
    }

    #[test]
    fn edges_of_one_wire_revolve_it_once() {
        let edges = |node| RevolveTarget { node, profile: Profile::Edges(vec![0, 2]) };
        let sheet = revolved(upright(), edges, consts::TAU).expect("revolves");
        // The left edge lies on the axis and sweeps into nothing.
        assert_eq!(sheet.faces().count(), 3);
    }

    #[test]
    fn a_region_straddling_the_axis_is_refused() {
        let straddling = xy_region(&[(-0.5, 0.0), (0.5, 0.0), (0.5, 1.0), (-0.5, 1.0)]);
        let error = revolved(straddling, faces, consts::TAU).err().expect("the revolve self-intersects");
        assert!(format!("{error:#}").contains("crosses the axis"), "{error:#}");
    }

    #[test]
    fn a_revolve_needs_an_axis() {
        let (doc, node) = doc_with_shape(upright());
        let target = faces(node);
        let frame = RevolveFrame::new(&doc, &target).expect("target resolves");
        let mut params = RevolveParams::new(frame);
        assert!(params.is_degenerate());
        assert!(build_revolve(&doc, &target, &params).is_err());

        params.axis = Axis::From(Point3::new(0.0, 0.0, 0.0));
        assert!(params.is_degenerate());
        assert!(build_revolve(&doc, &target, &params).is_err());
    }

    /// Millimetre-scale regions whose edge on the axis sits at coordinates no
    /// `f32` holds, with the axis picked through the `f32` corners a snap
    /// reports. Rounded inward, the axis would cut the profile; rounded
    /// outward, it would leave a hairline tube down the middle. Either way it
    /// is moved back onto the B-rep, and the revolve closes on the axis.
    #[test]
    fn an_axis_picked_through_rounded_corners_lands_on_the_profile() {
        // 100.3 rounds up into the region, 100.1 down out of it.
        for x0 in [100.3, 100.1] {
            let (x1, height) = (x0 + 30.0, 50.3);
            let (doc, node) = doc_with_shape(xy_region(&[(x0, 0.0), (x1, 0.0), (x1, height), (x0, height)]));
            let target = faces(node);
            let rounded = |x: f64, y: f64| Point3::new(x as f32 as Real, y as f32 as Real, 0.0);

            let params = params(&doc, &target, rounded(x0, 0.0), rounded(x0, height), consts::TAU);
            let solid = build_revolve(&doc, &target, &params).expect("revolves");
            assert_eq!(solid.faces().count(), 3, "two caps and a wall, no tube");
            let expected = std::f64::consts::PI * (x1 - x0) * (x1 - x0) * height;
            assert!((solid.volume() - expected).abs() < 1e-6 * expected, "{} vs {expected}", solid.volume());
        }
    }

    #[test]
    fn a_sketch_is_replaced_by_its_revolution() {
        let (doc, node) = doc_with_shape(upright());
        assert_eq!(RevolveFrame::new(&doc, &faces(node)).unwrap().fate, SourceFate::Replace);
    }

    #[test]
    fn a_face_of_a_solid_fuses_and_an_edge_stands_beside_it() {
        let (doc, node) = doc_with_box();
        assert_eq!(RevolveFrame::new(&doc, &faces(node)).unwrap().fate, SourceFate::Fuse);
        let edge = RevolveTarget { node, profile: Profile::Edges(vec![0]) };
        assert_eq!(RevolveFrame::new(&doc, &edge).unwrap().fate, SourceFate::Keep);
    }

    #[test]
    fn the_lever_lies_on_the_primary_edge() {
        let (doc, node) = doc_with_shape(upright());
        let target = RevolveTarget { node, profile: Profile::Edges(vec![1, 0]) };
        let frame = RevolveFrame::new(&doc, &target).expect("target resolves");
        let elements = doc.elements(node).unwrap();
        let middle = dvec3_to_point3(elements.edge(1).unwrap().curve().midpoint());
        assert!(f64::from(frame.lever.distance(middle)) < EPSILON);
        // OCCT pads the bounding box a little.
        assert!((frame.size - (5.0 as Real).sqrt()).abs() < 0.1, "{}", frame.size);
    }
}
