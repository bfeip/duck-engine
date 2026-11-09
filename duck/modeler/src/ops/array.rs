use anyhow::Result;
use duck_engine_scene::common::{consts, EuclideanSpace, InnerSpace, Matrix4, Point3, Rad, Real, Vector3, Zero};

use crate::document::{Document, PartId};
use crate::ops::duplicate::duplicate_parts;
use crate::ops::OFFSET_TOLERANCE;

/// What an array's reach measures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArrayMode {
    /// From the first member to the last.
    #[default]
    Extent,
    /// From one member to the next.
    Spacing,
}

/// Evenly spaced members along one direction, the first at the source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArrayAxis {
    /// Members along the axis, the source's own included.
    pub count: u32,
    /// How far the axis reaches, as the array's [`ArrayMode`] measures it.
    pub reach: Vector3,
}

impl ArrayAxis {
    /// An axis of `count` members with no reach yet.
    pub fn new(count: u32) -> Self {
        Self { count, reach: Vector3::zero() }
    }

    /// From one member to the next, with the reach measured by `mode`.
    pub fn step(&self, mode: ArrayMode) -> Vector3 {
        match mode {
            ArrayMode::Extent if self.count > 1 => self.reach / (self.count - 1) as Real,
            ArrayMode::Extent => Vector3::zero(),
            ArrayMode::Spacing => self.reach,
        }
    }

    /// The members' offsets from the source, the source's own zero first. An
    /// axis whose step is too short to tell its members apart has the source
    /// alone.
    fn members(&self, mode: ArrayMode) -> Vec<Vector3> {
        let step = self.step(mode);
        if step.magnitude() <= OFFSET_TOLERANCE {
            return vec![Vector3::zero()];
        }
        (0..self.count.max(1)).map(|i| step * i as Real).collect()
    }

    /// This axis with its reach measured by `to` instead of `from`, laying out
    /// the same members.
    pub fn converted(self, from: ArrayMode, to: ArrayMode) -> Self {
        let gaps = self.count.saturating_sub(1) as Real;
        let reach = match (from, to) {
            _ if self.count < 2 => self.reach,
            (ArrayMode::Extent, ArrayMode::Spacing) => self.reach / gaps,
            (ArrayMode::Spacing, ArrayMode::Extent) => self.reach * gaps,
            _ => self.reach,
        };
        Self { reach, ..self }
    }
}

/// Copies of a selection on a grid: one member for every pairing of the two
/// axes' members.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RectangularArray {
    pub mode: ArrayMode,
    pub axes: [ArrayAxis; 2],
}

impl RectangularArray {
    /// Every member's offset from the source but the source's own, running
    /// along the first axis within each step of the second.
    pub fn offsets(&self) -> Vec<Vector3> {
        let [first, second] = self.axes.map(|axis| axis.members(self.mode));
        second
            .iter()
            .flat_map(|&across| first.iter().map(move |&along| along + across))
            .skip(1)
            .collect()
    }

    /// Every member's placement but the source's own, in [`offsets`](Self::offsets)'s order.
    pub fn placements(&self) -> Vec<Matrix4> {
        self.offsets().into_iter().map(Matrix4::from_translation).collect()
    }

    /// Measures the reaches by `mode`, keeping the members where they are.
    pub fn set_mode(&mut self, mode: ArrayMode) {
        self.axes = self.axes.map(|axis| axis.converted(self.mode, mode));
        self.mode = mode;
    }
}

/// Copies each of `sources` to every member of `array` but its own, as one
/// undo step. Returns the copies, source-major.
pub fn rectangular_array(doc: &mut Document, sources: &[PartId], array: &RectangularArray) -> Result<Vec<PartId>> {
    let mut doc = doc.undo_scope("Rectangular array");
    duplicate_parts(&mut doc, sources, &array.placements())
}

/// A turn, in radians, too small to tell two members apart.
const MIN_TURN: Real = 1e-6;

/// Evenly turned members about an axis, the first at the source.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngularAxis {
    /// Members around the axis, the source's own included.
    pub count: u32,
    /// How far the members turn, in radians, as the array's [`ArrayMode`]
    /// measures it.
    pub angle: Real,
}

impl AngularAxis {
    /// From one member to the next, with the angle measured by `mode`. An extent
    /// of a full turn or more spreads the members evenly round one turn, where
    /// a last member at the extent would sit on the first.
    pub fn step(&self, mode: ArrayMode) -> Real {
        match mode {
            ArrayMode::Extent if self.count < 2 => 0.0,
            ArrayMode::Extent if self.angle.abs() >= consts::TAU - MIN_TURN => {
                self.angle.signum() * consts::TAU / self.count as Real
            }
            ArrayMode::Extent => self.angle / (self.count - 1) as Real,
            ArrayMode::Spacing => self.angle,
        }
    }

    /// The members' turns from the source, the source's own zero first. An axis
    /// whose step is too small to tell its members apart has the source alone.
    fn members(&self, mode: ArrayMode) -> Vec<Real> {
        let step = self.step(mode);
        if step.abs() <= MIN_TURN {
            return vec![0.0];
        }
        (0..self.count.max(1)).map(|i| step * i as Real).collect()
    }

    /// This axis with its angle measured by `to` instead of `from`, laying out
    /// the same members, unless they wrap past a full turn, which no extent
    /// lays out.
    pub fn converted(self, from: ArrayMode, to: ArrayMode) -> Self {
        let count = self.count as Real;
        let angle = match (from, to) {
            _ if self.count < 2 => self.angle,
            (ArrayMode::Extent, ArrayMode::Spacing) => self.step(from),
            (ArrayMode::Spacing, ArrayMode::Extent) => {
                let around = self.angle * count;
                match (around.abs() - consts::TAU).abs() <= MIN_TURN {
                    true => around,
                    false => self.angle * (count - 1.0),
                }
            }
            _ => self.angle,
        };
        Self { angle, ..self }
    }
}

/// Copies of a selection turned about an axis, in rings stepping outward from
/// it: one member for every pairing of a turn and a ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadialArray {
    /// A point on the axis.
    pub origin: Point3,
    /// The axis's direction, unit length. A positive angle turns
    /// counter-clockwise seen with the axis pointing at the viewer.
    pub axis: Vector3,
    pub mode: ArrayMode,
    /// The members round each ring.
    pub around: AngularAxis,
    /// The rings, reaching outward square to the axis.
    pub rings: ArrayAxis,
}

impl RadialArray {
    /// Every member's placement but the source's own, running round each ring
    /// from the innermost out.
    pub fn placements(&self) -> Vec<Matrix4> {
        let onto_axis = Matrix4::from_translation(-self.origin.to_vec());
        let off_axis = Matrix4::from_translation(self.origin.to_vec());
        let turns = self.around.members(self.mode);
        let axis = self.axis;
        self.rings
            .members(self.mode)
            .into_iter()
            .flat_map(|out| {
                let ring = onto_axis * Matrix4::from_translation(out);
                turns.iter().map(move |&turn| off_axis * Matrix4::from_axis_angle(axis, Rad(turn)) * ring)
            })
            .skip(1)
            .collect()
    }

    /// Measures the angle and the rings' reach by `mode`, keeping the members
    /// where they are.
    pub fn set_mode(&mut self, mode: ArrayMode) {
        self.around = self.around.converted(self.mode, mode);
        self.rings = self.rings.converted(self.mode, mode);
        self.mode = mode;
    }
}

/// Copies each of `sources` to every member of `array` but its own, as one
/// undo step. Returns the copies, source-major.
pub fn radial_array(doc: &mut Document, sources: &[PartId], array: &RadialArray) -> Result<Vec<PartId>> {
    let mut doc = doc.undo_scope("Radial array");
    duplicate_parts(&mut doc, sources, &array.placements())
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::{transform_point, Deg, Transform};
    use duck_engine_scene::Scene;
    use opencascade::primitives::Shape;

    const EPSILON: Real = 1e-6;

    fn array(mode: ArrayMode, first: (u32, Vector3), second: (u32, Vector3)) -> RectangularArray {
        let axis = |(count, reach)| ArrayAxis { count, reach };
        RectangularArray { mode, axes: [axis(first), axis(second)] }
    }

    fn assert_offsets(actual: &[Vector3], expected: &[Vector3]) {
        let close = actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(a, e)| (a - e).magnitude() < EPSILON);
        assert!(close, "expected {expected:?}, got {actual:?}");
    }

    fn x(v: Real) -> Vector3 {
        Vector3::new(v, 0.0, 0.0)
    }

    #[test]
    fn extent_reaches_the_last_member() {
        let row = array(ArrayMode::Extent, (3, x(10.0)), (1, Vector3::zero()));
        assert_offsets(&row.offsets(), &[x(5.0), x(10.0)]);
    }

    #[test]
    fn spacing_reaches_the_next_member() {
        let row = array(ArrayMode::Spacing, (3, x(10.0)), (1, Vector3::zero()));
        assert_offsets(&row.offsets(), &[x(10.0), x(20.0)]);
    }

    #[test]
    fn a_grid_runs_along_the_first_axis_within_the_second() {
        let y = Vector3::new(0.0, 4.0, 0.0);
        let grid = array(ArrayMode::Spacing, (3, x(10.0)), (2, y));
        assert_offsets(&grid.offsets(), &[x(10.0), x(20.0), y, x(10.0) + y, x(20.0) + y]);
    }

    #[test]
    fn an_axis_without_reach_or_count_has_the_source_alone() {
        // No reach yet, whatever the count.
        assert!(array(ArrayMode::Extent, (3, Vector3::zero()), (3, Vector3::zero())).offsets().is_empty());
        // A single member: Extent has nowhere to spread it.
        assert!(array(ArrayMode::Extent, (1, x(10.0)), (1, x(10.0))).offsets().is_empty());
        assert!(array(ArrayMode::Spacing, (1, x(10.0)), (0, x(10.0))).offsets().is_empty());
    }

    #[test]
    fn switching_mode_keeps_the_members() {
        let y = Vector3::new(0.0, 4.0, 0.0);
        for mode in [ArrayMode::Extent, ArrayMode::Spacing] {
            let before = array(mode, (4, x(12.0)), (2, y));
            let mut switched = before;
            switched.set_mode(match mode {
                ArrayMode::Extent => ArrayMode::Spacing,
                ArrayMode::Spacing => ArrayMode::Extent,
            });
            assert_offsets(&switched.offsets(), &before.offsets());
        }
    }

    #[test]
    fn a_single_member_keeps_its_reach_across_modes() {
        let mut single = array(ArrayMode::Extent, (1, x(10.0)), (1, Vector3::zero()));
        single.set_mode(ArrayMode::Spacing);
        assert!((single.axes[0].reach - x(10.0)).magnitude() < EPSILON);
    }

    #[test]
    fn the_array_is_one_undo_step_of_placed_copies() {
        let mut doc = Document::new(Scene::default());
        let source = doc
            .add_part("Box-001", Shape::cube_centered(1.0), Transform::IDENTITY, &CadTessellationOptions::default())
            .expect("tessellates");
        let grid = array(ArrayMode::Spacing, (3, x(2.0)), (2, Vector3::new(0.0, 0.0, 3.0)));

        let copies = rectangular_array(&mut doc, &[source], &grid).expect("copies");

        assert_eq!(copies.len(), 5);
        for (&copy, offset) in copies.iter().zip(grid.offsets()) {
            let position = doc.get_part(copy).unwrap().frame().position;
            assert!((position - (Point3::new(0.0, 0.0, 0.0) + offset)).magnitude() < EPSILON, "{position:?}");
        }
        assert_eq!(doc.undo_label(), Some("Rectangular array"));
        doc.undo().expect("undo succeeds");
        assert_eq!(doc.parts().count(), 1, "one step removes every copy");
    }

    /// An array about the Y axis through the origin.
    fn radial(mode: ArrayMode, around: (u32, Real), rings: (u32, Vector3)) -> RadialArray {
        RadialArray {
            origin: Point3::origin(),
            axis: Vector3::unit_y(),
            mode,
            around: AngularAxis { count: around.0, angle: around.1.to_radians() },
            rings: ArrayAxis { count: rings.0, reach: rings.1 },
        }
    }

    /// Where each of `array`'s members takes `point`.
    fn members_of(array: &RadialArray, point: Point3) -> Vec<Point3> {
        array.placements().iter().map(|placement| transform_point(placement, point)).collect()
    }

    fn assert_points(actual: &[Point3], expected: &[Point3]) {
        let close = actual.len() == expected.len()
            && actual.iter().zip(expected).all(|(a, e)| (a - e).magnitude() < EPSILON);
        assert!(close, "expected {expected:?}, got {actual:?}");
    }

    /// `point` turned `degrees` about the Y axis.
    fn turned(point: Point3, degrees: Real) -> Point3 {
        transform_point(&Matrix4::from_angle_y(Deg(degrees)), point)
    }

    #[test]
    fn a_full_turn_spreads_the_members_evenly_round_it() {
        let ring = radial(ArrayMode::Extent, (4, 360.0), (1, Vector3::zero()));
        let p = Point3::new(2.0, 0.0, 0.0);
        assert_points(&members_of(&ring, p), &[turned(p, 90.0), turned(p, 180.0), turned(p, 270.0)]);
        // Counter-clockwise looking down +Y onto the XZ plane: +X turns to −Z.
        assert_points(&members_of(&ring, p)[..1], &[Point3::new(0.0, 0.0, -2.0)]);

        let backwards = radial(ArrayMode::Extent, (4, -360.0), (1, Vector3::zero()));
        assert_points(&members_of(&backwards, p), &[turned(p, -90.0), turned(p, -180.0), turned(p, -270.0)]);
    }

    #[test]
    fn a_partial_extent_reaches_the_last_member() {
        let fan = radial(ArrayMode::Extent, (4, 90.0), (1, Vector3::zero()));
        let p = Point3::new(1.0, 0.0, 0.0);
        assert_points(&members_of(&fan, p), &[turned(p, 30.0), turned(p, 60.0), turned(p, 90.0)]);
    }

    #[test]
    fn spacing_turns_to_the_next_member() {
        let fan = radial(ArrayMode::Spacing, (4, 45.0), (1, Vector3::zero()));
        let p = Point3::new(1.0, 0.0, 0.0);
        assert_points(&members_of(&fan, p), &[turned(p, 45.0), turned(p, 90.0), turned(p, 135.0)]);
    }

    #[test]
    fn rings_step_outward_each_turned_round() {
        let rings = radial(ArrayMode::Extent, (3, 360.0), (2, x(3.0)));
        let (inner, outer) = (Point3::new(2.0, 0.0, 0.0), Point3::new(5.0, 0.0, 0.0));
        assert_points(
            &members_of(&rings, inner),
            &[turned(inner, 120.0), turned(inner, 240.0), outer, turned(outer, 120.0), turned(outer, 240.0)],
        );
    }

    #[test]
    fn members_turn_about_the_axis_through_its_origin() {
        let mut half = radial(ArrayMode::Extent, (2, 360.0), (1, Vector3::zero()));
        half.origin = Point3::new(5.0, 0.0, 0.0);
        assert_points(&members_of(&half, Point3::new(2.0, 1.0, 0.0)), &[Point3::new(8.0, 1.0, 0.0)]);
    }

    #[test]
    fn nothing_to_turn_has_the_source_alone() {
        assert!(radial(ArrayMode::Extent, (1, 360.0), (1, Vector3::zero())).placements().is_empty());
        assert!(radial(ArrayMode::Extent, (4, 0.0), (1, x(3.0))).placements().is_empty());
        assert!(radial(ArrayMode::Spacing, (4, 0.0), (1, x(3.0))).placements().is_empty());
    }

    #[test]
    fn switching_radial_mode_keeps_the_members() {
        let p = Point3::new(2.0, 0.0, 1.0);
        for (mode, around) in [
            (ArrayMode::Extent, (6, 360.0)),
            (ArrayMode::Extent, (4, 90.0)),
            (ArrayMode::Spacing, (8, 45.0)),
            (ArrayMode::Spacing, (3, 30.0)),
            (ArrayMode::Extent, (5, -360.0)),
        ] {
            let before = radial(mode, around, (3, x(4.0)));
            let mut switched = before;
            switched.set_mode(match mode {
                ArrayMode::Extent => ArrayMode::Spacing,
                ArrayMode::Spacing => ArrayMode::Extent,
            });
            assert_points(&members_of(&switched, p), &members_of(&before, p));
        }
    }

    #[test]
    fn the_radial_array_is_one_undo_step_of_turned_copies() {
        let mut doc = Document::new(Scene::default());
        let source = doc
            .add_part("Box-001", Shape::cube_centered(1.0), Transform::IDENTITY, &CadTessellationOptions::default())
            .expect("tessellates");
        let mut ring = radial(ArrayMode::Extent, (4, 360.0), (1, Vector3::zero()));
        ring.origin = Point3::new(-2.0, 0.0, 0.0);

        let copies = radial_array(&mut doc, &[source], &ring).expect("copies");

        assert_eq!(copies.len(), 3);
        for (i, &copy) in copies.iter().enumerate() {
            let degrees = 90.0 * (i + 1) as Real;
            let frame = doc.get_part(copy).unwrap().frame();
            let expected = turned(Point3::new(2.0, 0.0, 0.0), degrees) + Vector3::new(-2.0, 0.0, 0.0);
            assert!((frame.position - expected).magnitude() < EPSILON, "{:?}", frame.position);
            let facing = frame.rotation * Vector3::unit_x();
            let expected = turned(Point3::new(1.0, 0.0, 0.0), degrees).to_vec();
            assert!((facing - expected).magnitude() < EPSILON, "{facing:?}");
        }
        assert_eq!(doc.undo_label(), Some("Radial array"));
        doc.undo().expect("undo succeeds");
        assert_eq!(doc.parts().count(), 1, "one step removes every copy");
    }
}
