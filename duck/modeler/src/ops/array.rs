use anyhow::Result;
use duck_engine_scene::common::{InnerSpace, Matrix4, Real, Vector3, Zero};

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

    /// Measures the reaches by `mode`, keeping the members where they are.
    pub fn set_mode(&mut self, mode: ArrayMode) {
        self.axes = self.axes.map(|axis| axis.converted(self.mode, mode));
        self.mode = mode;
    }
}

/// Copies each of `sources` to every member of `array` but its own, as one
/// undo step. Returns the copies, source-major.
pub fn rectangular_array(doc: &mut Document, sources: &[PartId], array: &RectangularArray) -> Result<Vec<PartId>> {
    let placements: Vec<Matrix4> = array.offsets().into_iter().map(Matrix4::from_translation).collect();
    let mut doc = doc.undo_scope("Rectangular array");
    duplicate_parts(&mut doc, sources, &placements)
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::{Point3, Transform};
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
}
