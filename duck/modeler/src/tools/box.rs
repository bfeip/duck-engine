use anyhow::Result;
use duck_engine_common::{Plane, Point3, Real, Transform, Vector3};
use duck_engine_viewer::operator::{Handle, HandleDrag, HandleId, HandleReach, HandleShape};
use glam::dvec3;
use opencascade::primitives::Shape;

use crate::ops::primitives::{prism, rectangle_corners};
use crate::snap::Snap;
use crate::tools::ToolInfo;
use crate::ui::icons;
use super::edit::{dimension_field, grip_dimension, Params, MIN_DIMENSION};
use super::primitive::{
    flat, footprint, height_along, seat, unit_square, Next, Pointer, Primitive, PrimitiveTool,
    Track,
};

/// The box's dimension grips, one per axis of [`BoxParams`].
const WIDTH_HANDLE: HandleId = HandleId(0);
const DEPTH_HANDLE: HandleId = HandleId(1);
const HEIGHT_HANDLE: HandleId = HandleId(2);

/// The dimensions of a placed box. `base` is the first point picked and never
/// moves: the footprint grows about it and the height grows away from it along
/// `plane.normal`.
#[derive(Clone, Copy)]
pub struct BoxParams {
    base: Point3,
    plane: Plane,
    width: Real,
    depth: Real,
    height: Real,
}

impl BoxParams {
    /// Parameters for a finished pick, normalized so the height is positive and
    /// grows away from `base`. A downward pick flips the plane normal rather
    /// than moving the base off the picked point, so later height edits move
    /// only the far face.
    fn from_pick(base: Point3, width: Real, depth: Real, height: Real, plane: Plane) -> Self {
        let (plane, height) = if height >= 0.0 {
            (plane, height)
        } else {
            (Plane::from_point(-plane.normal, base), -height)
        };
        Self { base, plane, width, depth, height }
    }

    /// The footprint rectangle in the plane's basis: the vector from base to the
    /// footprint center, and the half extents along the basis vectors `(u, v)`.
    ///
    /// The only anchor-dependent part of the box. A corner-anchored box returns
    /// `u * width/2 + v * depth/2` as the offset here and needs no other change:
    /// both the preview transform and the committed shape read the rectangle
    /// from this one place.
    fn local_rect(&self) -> (Vector3, Real, Real) {
        // As we only support center boxes at the moment, this is all that is needed
        (Vector3::new(0.0, 0.0, 0.0), 0.5 * self.width, 0.5 * self.depth)
    }

    /// The footprint's four world-space corners, in wire order.
    fn footprint_corners(&self) -> [Point3; 4] {
        let (offset, half_width, half_depth) = self.local_rect();
        rectangle_corners(self.base + offset, &self.plane, 2.0 * half_width, 2.0 * half_depth)
    }
}

/// Placement of a box between its base and its height.
#[derive(Clone, Copy, Debug)]
pub enum BoxStage {
    /// The base picked; the pointer sizes the footprint about it.
    Footprint { base: Point3, plane: Plane },
    /// The footprint sized; the pointer sets the height.
    Height { base: Point3, plane: Plane, width: Real, depth: Real },
}

impl Params for BoxParams {
    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = dimension_field(ui, "Width", &mut self.width);
        changed |= dimension_field(ui, "Length", &mut self.depth);
        changed |= dimension_field(ui, "Height", &mut self.height);
        changed
    }

    /// One grip per dimension, on the face that dimension moves, each tied back
    /// to the base by a leader line so all three meet at the bottom centre.
    ///
    /// The footprint is read from [`local_rect`](BoxParams::local_rect) so the
    /// grips follow the anchor along with everything else.
    fn handles(&self) -> Vec<Handle> {
        let (u, v) = self.plane.basis();
        let (offset, half_width, half_depth) = self.local_rect();
        let centre = self.base + offset;
        let grip = |id, anchor, direction: Vector3| {
            Handle::new(id, HandleShape::Cube, anchor)
                .with_direction(direction)
                .with_reach(HandleReach::Leader(centre))
        };
        vec![
            grip(WIDTH_HANDLE, centre + u * half_width, u),
            grip(DEPTH_HANDLE, centre + v * half_depth, v),
            grip(HEIGHT_HANDLE, centre + self.plane.normal * self.height, self.plane.normal),
        ]
    }

    /// The grabbed face follows the cursor; the opposite one stays put.
    fn apply_handle(&mut self, drag: &HandleDrag, grabbed: &Self) {
        let (u, v) = grabbed.plane.basis();
        match drag.id {
            // The footprint grows about its centre, so a grip on one face
            // carries only half the width: doubling keeps it under the cursor.
            WIDTH_HANDLE => self.width = grip_dimension(grabbed.width, 2.0 * drag.distance_along(u)),
            DEPTH_HANDLE => self.depth = grip_dimension(grabbed.depth, 2.0 * drag.distance_along(v)),
            // Height grows away from `base`, which never moves, so the far face
            // follows the cursor one for one.
            HEIGHT_HANDLE => {
                self.height =
                    grip_dimension(grabbed.height, drag.distance_along(grabbed.plane.normal));
            }
            _ => {}
        }
    }
}

/// Places a box: the centre of its base, a corner of its footprint, then its
/// height.
pub type BoxTool = PrimitiveTool<BoxParams>;

impl Primitive for BoxParams {
    type Stage = BoxStage;

    const TOOL: ToolInfo = ToolInfo { id: "box", icon: icons::BOX, shortcut: None };
    const NAME: &'static str = "Box";

    fn start(first: &Snap, construction: &Plane) -> BoxStage {
        BoxStage::Footprint { base: first.position, plane: seat(first, construction) }
    }

    fn track(stage: BoxStage, pointer: &Pointer) -> Track<Self> {
        match stage {
            BoxStage::Footprint { base, plane } => {
                let corner = pointer.snap.map(|snap| snap.position);
                let size = corner.and_then(|corner| footprint(base, corner, &plane));
                Track {
                    cursor: corner,
                    preview: size.map(|(width, depth)| flat(base, &plane, width, depth)),
                    click: size.map(|(width, depth)| {
                        Next::Stage(BoxStage::Height { base, plane, width, depth })
                    }),
                }
            }
            BoxStage::Height { base, plane, width, depth } => {
                let height = height_along(base, plane.normal, &pointer.ray);
                let placed = (height.abs() > MIN_DIMENSION)
                    .then(|| BoxParams::from_pick(base, width, depth, height, plane));
                Track {
                    cursor: Some(base + plane.normal * height),
                    preview: placed.map(|params| params.preview_transform()),
                    click: placed.map(Next::Placed),
                }
            }
        }
    }

    /// A unit square while the footprint is sized, then the unit box:
    /// footprint centred in local XY, height along local +Z.
    fn reference(stage: BoxStage) -> Result<Shape> {
        match stage {
            BoxStage::Footprint { .. } => unit_square(),
            BoxStage::Height { .. } => {
                Ok(Shape::box_from_corners(dvec3(-0.5, -0.5, 0.0), dvec3(0.5, 0.5, 1.0)))
            }
        }
    }

    /// Scales the unit box to these dimensions. Every scale component stays
    /// non-negative — a negative one would make the baked GTransform a
    /// reflection, flipping the box's face normals inward.
    fn preview_transform(&self) -> Transform {
        let (offset, _, _) = self.local_rect();
        Transform {
            position: self.base + offset,
            rotation: self.plane.rotation(),
            scale: Vector3::new(self.width, self.depth, self.height),
        }
    }

    /// World-space box with analytic planar faces: the footprint rectangle on
    /// the plane, extruded along its normal.
    fn build(&self) -> Result<Shape> {
        prism(&self.footprint_corners(), self.plane.normal * self.height)
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
    fn the_footprint_corner_leads_to_the_height_stage() {
        let stage = BoxStage::Footprint { base: Point3::new(0.0, 0.0, 0.0), plane: Plane::xz() };
        let track = BoxParams::track(stage, &pointer_at(Point3::new(1.5, 0.0, 2.0)));
        let Some(Next::Stage(BoxStage::Height { width, depth, .. })) = track.click else {
            panic!("the corner sizes the footprint");
        };
        assert!((width - 3.0).abs() < EPSILON);
        assert!((depth - 4.0).abs() < EPSILON);
    }

    #[test]
    fn a_footprint_needs_a_snapped_corner_off_its_axes() {
        let stage = BoxStage::Footprint { base: Point3::new(0.0, 0.0, 0.0), plane: Plane::xz() };
        let on_an_axis = BoxParams::track(stage, &pointer_at(Point3::new(1.0, 0.0, 0.0)));
        assert!(on_an_axis.click.is_none());
        assert!(on_an_axis.preview.is_none());

        let unsnapped = Pointer { snap: None, ..pointer_at(Point3::new(1.0, 0.0, 1.0)) };
        assert!(BoxParams::track(stage, &unsnapped).click.is_none());
    }

    #[test]
    fn a_flat_height_cannot_be_clicked() {
        let base = Point3::new(1.0, 2.0, 3.0);
        let stage = BoxStage::Height { base, plane: Plane::xz(), width: 4.0, depth: 5.0 };
        let track = BoxParams::track(stage, &pointer_at(base));
        assert!(track.click.is_none());
        assert!(track.preview.is_none());
    }

    #[test]
    fn a_downward_height_flips_the_plane_about_the_base() {
        let base = Point3::new(1.0, 2.0, 3.0);
        let plane = Plane::xz();
        let stage = BoxStage::Height { base, plane, width: 4.0, depth: 5.0 };
        let track = BoxParams::track(stage, &pointer_at(base - plane.normal * 6.0));
        let Some(Next::Placed(params)) = track.click else { panic!("the height places the box") };

        // The picked point stays the base; the normal flips instead.
        assert!((params.base - base).magnitude() < EPSILON);
        assert!((params.height - 6.0).abs() < EPSILON);
        assert!((params.plane.normal + plane.normal).magnitude() < EPSILON);

        let t = params.preview_transform();
        // Scale stays non-negative: a negative one would reflect the baked
        // transform and invert the face normals.
        assert!(t.scale.x >= 0.0 && t.scale.y >= 0.0 && t.scale.z >= 0.0);
        assert!((t.position - base).magnitude() < EPSILON);
    }

    #[test]
    fn footprint_is_centred_on_the_base() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let params = BoxParams::from_pick(base, 4.0, 6.0, 2.0, skewed_plane(base));
        let corners = params.footprint_corners();
        let offset = corners.iter().fold(Vector3::new(0.0, 0.0, 0.0), |acc, c| acc + (c - base))
            / corners.len() as Real;
        assert!(offset.magnitude() < 1e-5);
    }

    #[test]
    fn height_edits_leave_the_footprint_in_place() {
        let base = Point3::new(-4.0, 5.0, 6.0);
        let plane = skewed_plane(base);
        let mut params = BoxParams::from_pick(base, 3.0, 7.0, 2.0, plane);
        let before = params.footprint_corners();

        params.height = 11.0;

        // Only the top face moves: base, plane and footprint are untouched.
        let after = params.footprint_corners();
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).magnitude() < EPSILON);
        }
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }

    #[test]
    fn handles_sit_on_the_face_each_one_moves() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let params = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, v) = plane.basis();

        let handles = params.handles();
        assert_eq!(handles.len(), 3);

        // Each grip sits half an extent out along its own axis — except height,
        // which grows a full extent away from the base.
        let expected = [
            (WIDTH_HANDLE, base + u * 2.0, u),
            (DEPTH_HANDLE, base + v * 3.0, v),
            (HEIGHT_HANDLE, base + plane.normal * 2.0, plane.normal),
        ];
        for (id, anchor, direction) in expected {
            let handle = handles.iter().find(|h| h.id == id).expect("grip is present");
            assert!((handle.anchor - anchor).magnitude() < 1e-5, "{id:?} anchor");
            assert!((handle.direction - direction).magnitude() < 1e-5, "{id:?} direction");
        }
    }

    /// Every grip's leader runs back to the bottom centre, so the three meet
    /// there however the box is sized.
    #[test]
    fn every_grips_leader_runs_back_to_the_base() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let params = BoxParams::from_pick(base, 4.0, 6.0, 2.0, skewed_plane(base));
        let (offset, _, _) = params.local_rect();
        let centre = base + offset;

        for handle in params.handles() {
            let HandleReach::Leader(origin) = handle.reach else {
                panic!("{:?} has no leader", handle.id);
            };
            assert!((origin - centre).magnitude() < 1e-5, "{:?} leader origin", handle.id);
            // A leader only reads as a leader if it actually spans something.
            assert!(
                (handle.anchor - origin).magnitude() > 1e-5,
                "{:?} leader is degenerate",
                handle.id
            );
        }
    }

    /// The footprint grows about its centre, so a face only travels half as far
    /// as the dimension grows. Doubling is what keeps the grabbed face under
    /// the cursor; height, anchored at the base, must not be doubled.
    #[test]
    fn footprint_grips_double_the_drag_and_height_does_not() {
        let base = Point3::new(-1.0, 0.5, 2.0);
        let plane = skewed_plane(base);
        let grabbed = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, v) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(WIDTH_HANDLE, u * 1.5), &grabbed);
        assert!((params.width - 7.0).abs() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(DEPTH_HANDLE, v * 1.5), &grabbed);
        assert!((params.depth - 9.0).abs() < EPSILON);

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 1.5), &grabbed);
        assert!((params.height - 3.5).abs() < EPSILON);
    }

    #[test]
    fn a_grip_moves_only_its_own_dimension() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = skewed_plane(base);
        let grabbed = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(WIDTH_HANDLE, u * 1.0), &grabbed);
        assert!((params.depth - grabbed.depth).abs() < EPSILON);
        assert!((params.height - grabbed.height).abs() < EPSILON);
        assert!((params.base - grabbed.base).magnitude() < EPSILON);
    }

    /// Off-axis motion is the common case — the cursor rarely tracks the grip's
    /// axis exactly — and must not bleed into the dimension.
    #[test]
    fn a_grip_ignores_motion_across_its_axis() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, v) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(WIDTH_HANDLE, u * 1.0 + v * 5.0 + plane.normal * 5.0), &grabbed);
        assert!((params.width - 6.0).abs() < EPSILON);
    }

    /// Dragging a grip past the opposite face flattens the box instead of
    /// inverting it: a negative extent would reflect the baked transform.
    #[test]
    fn a_grip_dragged_through_the_box_clamps() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(WIDTH_HANDLE, u * -50.0), &grabbed);
        assert_eq!(params.width, MIN_DIMENSION);
        assert!(params.preview_transform().scale.x >= 0.0);
    }

    /// A drag carries its total offset from the grab, so applying successive
    /// reports to the same snapshot must not compound them.
    #[test]
    fn successive_drags_from_one_grab_do_not_compound() {
        let base = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::xz();
        let grabbed = BoxParams::from_pick(base, 4.0, 6.0, 2.0, plane);
        let (u, _) = plane.basis();

        let mut params = grabbed;
        params.apply_handle(&drag(WIDTH_HANDLE, u * 1.0), &grabbed);
        params.apply_handle(&drag(WIDTH_HANDLE, u * 1.0), &grabbed);
        assert!((params.width - 6.0).abs() < EPSILON);
    }

    /// The grips are the 3D twin of the panel fields, so a grip drag has to
    /// leave the box where the equivalent typed value would.
    #[test]
    fn a_height_grip_leaves_the_footprint_in_place() {
        let base = Point3::new(-4.0, 5.0, 6.0);
        let plane = skewed_plane(base);
        let grabbed = BoxParams::from_pick(base, 3.0, 7.0, 2.0, plane);
        let before = grabbed.footprint_corners();

        let mut params = grabbed;
        params.apply_handle(&drag(HEIGHT_HANDLE, plane.normal * 9.0), &grabbed);

        assert!((params.height - 11.0).abs() < EPSILON);
        for (a, b) in before.iter().zip(params.footprint_corners().iter()) {
            assert!((a - b).magnitude() < EPSILON);
        }
        assert!((params.preview_transform().position - base).magnitude() < EPSILON);
    }
}
