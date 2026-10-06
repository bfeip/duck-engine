use anyhow::Result;
use duck_engine_common::{Plane, Point3, Real, Transform, Vector3};
use duck_engine_viewer::operator::{Handle, HandleDrag, HandleId, HandleReach, HandleShape};
use opencascade::primitives::Shape;

use crate::ops::primitives::cylinder;
use crate::snap::Snap;
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{dimension_field, grip_dimension, Params, MIN_DIMENSION};
use super::primitive::{
    flat, height_along, radius_to, seat, unit_disk, Next, Pointer, Primitive, PrimitiveTool, Track,
};

/// The cylinder's dimension grips.
const RADIUS_HANDLE: HandleId = HandleId(0);
const HEIGHT_HANDLE: HandleId = HandleId(1);

/// The dimensions of a placed cylinder, adjustable before it is committed.
/// `base` is the first point picked and never moves: the radius grows about the
/// axis through it and the height grows away from it along `plane.normal`.
#[derive(Clone, Copy)]
pub struct CylinderParams {
    base: Point3,
    plane: Plane,
    radius: Real,
    height: Real,
}

impl CylinderParams {
    /// Parameters for a finished pick, normalized so the height is positive and
    /// grows away from `base`. A downward pick flips the plane normal rather
    /// than moving the base off the picked point, so later height edits move
    /// only the far cap.
    fn from_pick(base: Point3, radius: Real, height: Real, plane: Plane) -> Self {
        let (plane, height) = if height >= 0.0 {
            (plane, height)
        } else {
            (Plane::from_point(-plane.normal, base), -height)
        };
        Self { base, plane, radius, height }
    }
}

/// Placement of a cylinder between its base and its height.
#[derive(Clone, Copy, Debug)]
pub enum CylinderStage {
    /// The base picked; the pointer sizes the radius about it.
    Radius { base: Point3, plane: Plane },
    /// The radius sized; the pointer sets the height.
    Height { base: Point3, plane: Plane, radius: Real },
}

impl Params for CylinderParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = dimension_field(ui, "Radius", &mut self.radius);
        changed |= dimension_field(ui, "Height", &mut self.height);
        changed
    }

    /// A radius grip on the base rim and a height grip on the far cap, each tied
    /// back to the base by a leader line so both meet at the bottom centre.
    fn handles(&self) -> Vec<Handle> {
        let (u, _) = self.plane.basis();
        let grip = |id, anchor, direction: Vector3| {
            Handle::new(id, HandleShape::Cube, anchor)
                .with_direction(direction)
                .with_reach(HandleReach::Leader(self.base))
        };
        vec![
            grip(RADIUS_HANDLE, self.base + u * self.radius, u),
            grip(HEIGHT_HANDLE, self.base + self.plane.normal * self.height, self.plane.normal),
        ]
    }

    /// The grabbed rim or cap follows the cursor one for one: the radius is
    /// measured from the axis and the height from `base`, neither of which moves.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        match drag.id {
            RADIUS_HANDLE => {
                let (u, _) = grabbed.plane.basis();
                self.radius = grip_dimension(grabbed.radius, drag.distance_along(u));
            }
            HEIGHT_HANDLE => {
                self.height =
                    grip_dimension(grabbed.height, drag.distance_along(grabbed.plane.normal));
            }
            _ => {}
        }
    }
}

/// Places a cylinder: the centre of its base, a point on its rim, then its
/// height.
pub type CylinderTool = PrimitiveTool<CylinderParams>;

impl Primitive for CylinderParams {
    type Stage = CylinderStage;

    const TOOL: ToolInfo = ToolInfo { id: "cylinder", icon: icons::CYLINDER, shortcut: None };
    const NAME: &'static str = "Cylinder";

    fn start(first: &Snap, construction: &Plane) -> CylinderStage {
        CylinderStage::Radius { base: first.position, plane: seat(first, construction) }
    }

    fn track(stage: CylinderStage, pointer: &Pointer) -> Track<Self> {
        match stage {
            CylinderStage::Radius { base, plane } => {
                let rim = pointer.snap.map(|snap| snap.position);
                let radius = rim.and_then(|rim| radius_to(base, rim));
                Track {
                    cursor: rim,
                    preview: radius.map(|radius| flat(base, &plane, radius, radius)),
                    click: radius
                        .map(|radius| Next::Stage(CylinderStage::Height { base, plane, radius })),
                }
            }
            CylinderStage::Height { base, plane, radius } => {
                let height = height_along(base, plane.normal, &pointer.ray);
                let placed = (height.abs() > MIN_DIMENSION)
                    .then(|| CylinderParams::from_pick(base, radius, height, plane));
                Track {
                    cursor: Some(base + plane.normal * height),
                    preview: placed.map(|params| params.preview_transform()),
                    click: placed.map(Next::Placed),
                }
            }
        }
    }

    /// A unit disk while the radius is sized, then the unit cylinder: base at
    /// the origin, axis along local +Z.
    fn reference(stage: CylinderStage) -> Result<Shape> {
        match stage {
            CylinderStage::Radius { .. } => unit_disk(),
            CylinderStage::Height { .. } => Ok(Shape::cylinder_radius_height(1.0, 1.0)),
        }
    }

    /// Scales the unit cylinder to these dimensions. Every scale component
    /// stays non-negative — a negative one would make the baked transform a
    /// reflection, flipping the face normals inward.
    fn preview_transform(&self) -> Transform {
        Transform {
            position: self.base,
            rotation: self.plane.rotation(),
            scale: Vector3::new(self.radius, self.radius, self.height),
        }
    }

    fn build(&self) -> Result<Shape> {
        Ok(cylinder(self.base, self.plane.normal, self.radius, self.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_common::InnerSpace;

    use crate::testing::skewed_plane;
    use super::super::primitive::pointer_at;

    const EPSILON: Real = 1e-6;

    /// A drag of `offset` on the grip `id`, as the handle machinery reports it.
    fn drag(id: HandleId, offset: Vector3) -> HandleDrag {
        crate::testing::drag(id, Point3::new(0.0, 0.0, 0.0), offset)
    }

    #[test]
    fn the_rim_leads_to_the_height_stage() {
        let stage = CylinderStage::Radius { base: Point3::new(0.0, 0.0, 0.0), plane: Plane::xz() };
        let track = CylinderParams::track(stage, &pointer_at(Point3::new(3.0, 0.0, 4.0)));
        let Some(Next::Stage(CylinderStage::Height { radius, .. })) = track.click else {
            panic!("the rim sizes the radius");
        };
        assert!((radius - 5.0).abs() < EPSILON);
    }

    #[test]
    fn a_rim_on_the_base_cannot_be_clicked() {
        let base = Point3::new(1.0, 2.0, 3.0);
        let stage = CylinderStage::Radius { base, plane: Plane::xz() };
        let track = CylinderParams::track(stage, &pointer_at(base));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }

    #[test]
    fn a_flat_height_cannot_be_clicked() {
        let base = Point3::new(1.0, 2.0, 3.0);
        let stage = CylinderStage::Height { base, plane: Plane::xz(), radius: 2.0 };
        let track = CylinderParams::track(stage, &pointer_at(base));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }

    #[test]
    fn a_downward_height_flips_the_plane_about_the_base() {
        let plane = Plane::xz();
        let base = Point3::new(1.0, 2.0, 3.0);
        let stage = CylinderStage::Height { base, plane, radius: 2.0 };
        let track = CylinderParams::track(stage, &pointer_at(base - plane.normal * 3.0));
        let Some(Next::Placed(params)) = track.click else {
            panic!("the height places the cylinder");
        };

        // The picked point stays the base; the axis flips instead.
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.height - 3.0).abs() < EPSILON);
        assert!((params.plane.normal + plane.normal).magnitude() < EPSILON);

        let t = params.preview_transform();
        // Scale stays non-negative after the flip.
        assert!(t.scale.x >= 0.0 && t.scale.y >= 0.0 && t.scale.z >= 0.0);
        assert!((t.position - base).magnitude() < EPSILON);
    }

    #[test]
    fn height_edits_leave_the_base_cap_in_place() {
        let plane = skewed_plane(Point3::new(0.0, 0.0, 0.0));
        let base = Point3::new(-4.0, 5.0, 6.0);
        let mut params = CylinderParams::from_pick(base, 2.0, 3.0, plane);
        params.height = 10.0;
        // Only the far cap moves: base, axis and radius are untouched.
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.plane.normal - plane.normal).magnitude() < EPSILON);
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }

    #[test]
    fn grips_sit_on_the_rim_and_the_far_cap() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let params = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let handles = params.handles();
        assert_eq!(handles.len(), 2);

        let expected = [
            (RADIUS_HANDLE, base + u * 2.0, u),
            (HEIGHT_HANDLE, base + plane.normal * 5.0, plane.normal),
        ];
        for (id, anchor, direction) in expected {
            let handle = handles.iter().find(|h| h.id == id).expect("grip is present");
            assert!((handle.anchor - anchor).magnitude() < 1e-5, "{id:?} anchor");
            assert!((handle.direction - direction).magnitude() < 1e-5, "{id:?} direction");
        }
    }

    #[test]
    fn every_grips_leader_runs_back_to_the_base() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let params = CylinderParams::from_pick(base, 2.0, 5.0, skewed_plane(base));

        for handle in params.handles() {
            let HandleReach::Leader(origin) = handle.reach else {
                panic!("{:?} has no leader", handle.id);
            };
            assert!((origin - base).magnitude() < EPSILON, "{:?} leader origin", handle.id);
            assert!(
                (handle.anchor - origin).magnitude() > 1e-5,
                "{:?} leader is degenerate",
                handle.id
            );
        }
    }

    /// The radius is measured from the axis, not across the diameter, so unlike
    /// the box footprint it must not be doubled.
    #[test]
    fn both_grips_follow_the_drag_one_for_one() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.5), &grabbed);
        assert!((params.radius - 3.5).abs() < 1e-5);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 1.5), &grabbed);
        assert!((params.height - 6.5).abs() < 1e-5);
    }

    #[test]
    fn a_grip_moves_only_its_own_dimension() {
        let base = Point3::new(-4.0, 5.0, 6.0);
        let plane = skewed_plane(base);
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        assert!((params.height - grabbed.height).abs() < EPSILON);
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.plane.normal - plane.normal).magnitude() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 9.0), &grabbed);
        assert!((params.radius - grabbed.radius).abs() < EPSILON);
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }

    #[test]
    fn a_grip_ignores_motion_across_its_axis() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, v) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0 + v * 5.0 + plane.normal * 5.0), &grabbed);
        assert!((params.radius - 3.0).abs() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 1.0 + u * 5.0 + v * 5.0), &grabbed);
        assert!((params.height - 6.0).abs() < EPSILON);
    }

    /// Dragging a grip through the axis or past the base flattens the cylinder
    /// instead of inverting it: a negative scale would reflect the baked transform.
    #[test]
    fn a_grip_dragged_through_the_cylinder_clamps() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * -50.0), &grabbed);
        assert_eq!(params.radius, MIN_DIMENSION);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * -50.0), &grabbed);
        assert_eq!(params.height, MIN_DIMENSION);
        assert!(params.preview_transform().scale.z >= 0.0);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = CylinderParams::from_pick(base, 2.0, 5.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        params.apply_handle(&drag(RADIUS_HANDLE, u * 1.0), &grabbed);
        assert!((params.radius - 3.0).abs() < EPSILON);
    }
}
