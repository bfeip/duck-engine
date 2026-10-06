use anyhow::Result;
use duck_engine_common::{InnerSpace, Plane, Point3, Quaternion, Real, Transform, Vector3};
use duck_engine_viewer::operator::{Handle, HandleDrag, HandleId, HandleReach, HandleShape};
use opencascade::primitives::Shape;

use crate::ops::primitives::sphere;
use crate::snap::Snap;
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{dimension_field, grip_dimension, Params};
use super::primitive::{radius_to, Next, Pointer, Primitive, PrimitiveTool, Track};

/// The sphere's one grip, on the pole.
const RADIUS_HANDLE: HandleId = HandleId(0);

/// The dimensions of a placed sphere, adjustable before it is committed.
/// `center` is the first point picked and never moves: the radius grows about it.
#[derive(Clone, Copy)]
pub struct SphereParams {
    center: Point3,
    /// Polar axis, chosen at placement (see [`SphereParams::start`]).
    axis: Vector3,
    radius: Real,
}

/// Placement of a sphere: the centre picked, and the pointer sizing the radius
/// about it.
#[derive(Clone, Copy, Debug)]
pub struct SphereStage {
    center: Point3,
    axis: Vector3,
}

impl Params for SphereParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        dimension_field(ui, "Radius", &mut self.radius)
    }

    /// A radius grip on the pole, tied back to the center by a leader.
    fn handles(&self) -> Vec<Handle> {
        let axis = self.axis.normalize();
        vec![
            Handle::new(RADIUS_HANDLE, HandleShape::Cube, self.center + axis * self.radius)
                .with_direction(axis)
                .with_reach(HandleReach::Leader(self.center)),
        ]
    }

    /// The center never moves, so the pole follows the cursor one for one.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        if drag.id == RADIUS_HANDLE {
            self.radius = grip_dimension(grabbed.radius, drag.distance_along(grabbed.axis));
        }
    }
}

/// Places a sphere: its centre, then a point on its surface.
pub type SphereTool = PrimitiveTool<SphereParams>;

impl Primitive for SphereParams {
    type Stage = SphereStage;

    const TOOL: ToolInfo = ToolInfo { id: "sphere", icon: icons::SPHERE, shortcut: None };
    const NAME: &'static str = "Sphere";

    /// The polar axis is the snapped direction, such as a face normal, where
    /// there is one.
    fn start(first: &Snap, _construction: &Plane) -> SphereStage {
        // Otherwise a skewed axis keeps the seam and poles off every world axis,
        // so a later boolean's cutting plane isn't near-coincident with them.
        let axis = first.direction.unwrap_or_else(|| Vector3::new(1.0, 2.0, 3.0).normalize());
        SphereStage { center: first.position, axis }
    }

    fn track(stage: SphereStage, pointer: &Pointer) -> Track<Self> {
        let SphereStage { center, axis } = stage;
        let rim = pointer.snap.map(|snap| snap.position);
        let placed = rim
            .and_then(|rim| radius_to(center, rim))
            .map(|radius| SphereParams { center, axis, radius });
        Track {
            cursor: rim,
            preview: placed.map(|params| params.preview_transform()),
            click: placed.map(Next::Placed),
        }
    }

    /// The unit sphere about the origin.
    fn reference(_stage: SphereStage) -> Result<Shape> {
        Ok(Shape::sphere(1.0).build())
    }

    fn preview_transform(&self) -> Transform {
        Transform {
            position: self.center,
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            scale: Vector3::new(self.radius, self.radius, self.radius),
        }
    }

    fn build(&self) -> Result<Shape> {
        Ok(sphere(self.center, self.axis, self.radius))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::snap::SnapKind;
    use crate::tools::edit::MIN_DIMENSION;
    use super::super::primitive::pointer_at;

    const EPSILON: Real = 1e-6;

    /// A sphere whose polar axis is aligned with no world axis, so a mistaken
    /// direction shows up.
    fn skewed_sphere() -> SphereParams {
        SphereParams {
            center: Point3::new(-1.0, 0.5, 2.0),
            axis: Vector3::new(1.0, 2.0, 3.0).normalize(),
            radius: 4.0,
        }
    }

    /// A drag of `offset` on the radius grip, as the handle machinery reports it.
    fn drag(offset: Vector3) -> HandleDrag {
        crate::testing::drag(RADIUS_HANDLE, Point3::new(0.0, 0.0, 0.0), offset)
    }

    #[test]
    fn the_polar_axis_follows_a_snapped_direction() {
        let normal = Vector3::new(0.0, 0.0, 1.0);
        let at = Point3::new(1.0, 2.0, 3.0);
        let face = Snap { position: at, direction: Some(normal), kind: SnapKind::Face };
        assert!((SphereParams::start(&face, &Plane::xz()).axis - normal).magnitude() < EPSILON);
    }

    /// With no direction to follow, the axis lies along none of the world's.
    #[test]
    fn the_polar_axis_otherwise_runs_off_the_world_axes() {
        let at = Point3::new(1.0, 2.0, 3.0);
        let free = Snap { position: at, direction: None, kind: SnapKind::ConstructionPlane };
        let axis = SphereParams::start(&free, &Plane::xz()).axis;
        for world in [Vector3::unit_x(), Vector3::unit_y(), Vector3::unit_z()] {
            assert!(axis.dot(world).abs() < 1.0 - 1e-3, "{axis:?} runs along {world:?}");
        }
    }

    #[test]
    fn a_point_on_the_surface_places_the_sphere() {
        let center = Point3::new(1.0, 2.0, 3.0);
        let stage = SphereStage { center, axis: Vector3::unit_y() };
        let track = SphereParams::track(stage, &pointer_at(center + Vector3::new(3.0, 0.0, 4.0)));
        let Some(Next::Placed(params)) = track.click else { panic!("the point places the sphere") };
        assert!((params.radius - 5.0).abs() < EPSILON);
        assert!((params.center - center).magnitude() < EPSILON);
    }

    #[test]
    fn a_point_on_the_centre_cannot_be_clicked() {
        let center = Point3::new(1.0, 2.0, 3.0);
        let stage = SphereStage { center, axis: Vector3::unit_y() };
        let track = SphereParams::track(stage, &pointer_at(center));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }

    #[test]
    fn the_grip_sits_on_the_pole_with_a_leader_to_the_center() {
        let params = skewed_sphere();
        let handles = params.handles();
        assert_eq!(handles.len(), 1);

        let handle = &handles[0];
        assert_eq!(handle.id, RADIUS_HANDLE);
        assert!((handle.anchor - (params.center + params.axis * 4.0)).magnitude() < 1e-5);
        assert!((handle.direction - params.axis).magnitude() < 1e-5);
        let HandleReach::Leader(origin) = handle.reach else { panic!("grip has no leader") };
        assert!((origin - params.center).magnitude() < EPSILON);
    }

    /// The center never moves, so the radius follows the drag one for one.
    #[test]
    fn a_drag_along_the_axis_moves_the_radius_one_for_one() {
        let grabbed = skewed_sphere();
        let mut params = grabbed;
        params.apply_handle(&drag(grabbed.axis * 1.5), &grabbed);

        assert!((params.radius - 5.5).abs() < 1e-5);
        assert!((params.center - grabbed.center).magnitude() < EPSILON);
        assert!((params.axis - grabbed.axis).magnitude() < EPSILON);
    }

    #[test]
    fn the_grip_ignores_motion_across_its_axis() {
        let grabbed = SphereParams {
            center: Point3::new(0.0, 0.0, 0.0),
            axis: Vector3::unit_z(),
            radius: 2.0,
        };
        let mut params = grabbed;
        params.apply_handle(&drag(Vector3::new(5.0, -5.0, 1.0)), &grabbed);
        assert!((params.radius - 3.0).abs() < EPSILON);
    }

    #[test]
    fn a_grip_dragged_through_the_center_clamps() {
        let grabbed = skewed_sphere();
        let mut params = grabbed;
        params.apply_handle(&drag(grabbed.axis * -50.0), &grabbed);
        assert_eq!(params.radius, MIN_DIMENSION);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let grabbed = skewed_sphere();
        let mut params = grabbed;
        params.apply_handle(&drag(grabbed.axis * 1.0), &grabbed);
        params.apply_handle(&drag(grabbed.axis * 1.0), &grabbed);
        assert!((params.radius - 5.0).abs() < 1e-5);
    }
}
