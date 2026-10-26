//! The grips a transform shows: arrows, rings or cubes along the axes, quads
//! for the planes between them, and a ball for free motion.

use duck_engine_viewer::common::{Axis, Real, RgbaColor};
use duck_engine_viewer::operator::{DragKind, Handle, HandleId, HandleReach, HandleShape};

use super::manipulation::{Frame, Space, TransformMode};

/// How much larger than a plain grip the gizmo is drawn.
const GIZMO_SCALE: Real = 1.4;
/// Rings run round at the radius the arrows reach, twice a plain ring's.
const RING_SCALE: Real = 2.0 * GIZMO_SCALE;
/// Plane quads are smaller, tucked in between the arrows.
const QUAD_SCALE: Real = 0.9;
/// How far out along each of its two axes a plane quad sits, before
/// [`QUAD_SCALE`].
const QUAD_OFFSET: Real = 0.47;
/// Opacity of a plane quad.
const PLANE_ALPHA: f32 = 0.5;
/// The ball claims no axis, so it is neutral.
const BALL_COLOR: RgbaColor = RgbaColor { r: 0.85, g: 0.85, b: 0.85, a: 1.0 };

/// What dragging a gizmo grip constrains to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grip {
    /// Along an axis, or about it for a rotation.
    Axis(Axis),
    /// In the plane normal to an axis.
    Plane(Axis),
    /// Free: in the view plane, or uniformly for a scale.
    Ball,
}

impl Grip {
    pub fn id(self) -> HandleId {
        HandleId(match self {
            Grip::Axis(axis) => index(axis),
            Grip::Plane(normal) => 3 + index(normal),
            Grip::Ball => 6,
        })
    }

    /// The grip `id` names, if it is one of the gizmo's.
    pub fn of(id: HandleId) -> Option<Self> {
        match id.0 {
            0..=2 => Some(Grip::Axis(Axis::ALL[id.0 as usize])),
            3..=5 => Some(Grip::Plane(Axis::ALL[id.0 as usize - 3])),
            6 => Some(Grip::Ball),
            _ => None,
        }
    }
}

fn index(axis: Axis) -> u32 {
    match axis {
        Axis::X => 0,
        Axis::Y => 1,
        Axis::Z => 2,
    }
}

/// The two axes in the plane normal to `normal`.
pub fn plane_axes(normal: Axis) -> (Axis, Axis) {
    match normal {
        Axis::X => (Axis::Y, Axis::Z),
        Axis::Y => (Axis::Z, Axis::X),
        Axis::Z => (Axis::X, Axis::Y),
    }
}

/// The grips for `mode` about `frame`, along `space`'s axes.
pub fn handles(mode: TransformMode, frame: &Frame, space: Space) -> Vec<Handle> {
    let direction = |axis| frame.axis(axis, space);
    let grip = |grip: Grip, shape| Handle::new(grip.id(), shape, frame.pivot);

    let arms = |shape: HandleShape| {
        Axis::ALL.map(|axis| {
            grip(Grip::Axis(axis), shape.clone())
                .with_direction(direction(axis))
                .with_reach(HandleReach::Arm)
                .with_color(axis.color())
                .with_scale(GIZMO_SCALE)
        })
    };
    let planes = Axis::ALL.map(|normal| {
        let (u, v) = plane_axes(normal);
        grip(Grip::Plane(normal), HandleShape::Quad)
            .with_drag(DragKind::Plane)
            .with_direction(direction(normal))
            .with_reach(HandleReach::Offset((direction(u) + direction(v)) * QUAD_OFFSET))
            .with_color(u.color().mix(v.color(), 0.5).with_alpha(PLANE_ALPHA))
            .with_scale(QUAD_SCALE)
    });
    let ball = grip(Grip::Ball, HandleShape::Ball).with_drag(DragKind::View).with_color(BALL_COLOR);

    match mode {
        TransformMode::Translate => arms(HandleShape::Cone).into_iter().chain(planes).chain([ball]).collect(),
        TransformMode::Rotate => Axis::ALL
            .map(|axis| {
                grip(Grip::Axis(axis), HandleShape::Ring)
                    .with_drag(DragKind::Plane)
                    .with_direction(direction(axis))
                    .with_color(axis.color())
                    .with_scale(RING_SCALE)
            })
            .to_vec(),
        TransformMode::Scale if frame.uniform_scale => vec![ball],
        TransformMode::Scale => arms(HandleShape::Cube).into_iter().chain(planes).chain([ball]).collect(),
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::{consts, quaternion_from_axis_angle_safe, InnerSpace, Point3, Vector3};

    use super::*;

    const EPSILON: Real = 1e-6;

    fn frame() -> Frame {
        Frame {
            pivot: Point3::new(1.0, 2.0, 3.0),
            rotation: quaternion_from_axis_angle_safe(Vector3::unit_z(), consts::PI / 2.0),
            uniform_scale: false,
        }
    }

    fn grips(handles: &[Handle]) -> Vec<Grip> {
        handles.iter().map(|h| Grip::of(h.id).expect("a gizmo grip")).collect()
    }

    #[test]
    fn grip_ids_round_trip() {
        for grip in Axis::ALL.map(Grip::Axis).into_iter().chain(Axis::ALL.map(Grip::Plane)).chain([Grip::Ball]) {
            assert_eq!(Grip::of(grip.id()), Some(grip));
        }
        assert_eq!(Grip::of(HandleId(7)), None);
    }

    #[test]
    fn each_mode_shows_its_grips_at_the_pivot() {
        let translate = handles(TransformMode::Translate, &frame(), Space::World);
        assert_eq!(translate.len(), 7);
        assert!(translate.iter().all(|h| (h.anchor - frame().pivot).magnitude() < EPSILON));
        assert!(translate[..3].iter().all(|h| h.shape == HandleShape::Cone && h.drag == DragKind::Axis));

        let rotate = handles(TransformMode::Rotate, &frame(), Space::World);
        assert_eq!(grips(&rotate), Axis::ALL.map(Grip::Axis));
        assert!(rotate.iter().all(|h| h.shape == HandleShape::Ring && h.drag == DragKind::Plane));

        let scale = handles(TransformMode::Scale, &frame(), Space::World);
        assert!(scale[..3].iter().all(|h| h.shape == HandleShape::Cube));
        assert_eq!(grips(&scale).last(), Some(&Grip::Ball));
    }

    #[test]
    fn local_grips_turn_with_the_frame() {
        let local = handles(TransformMode::Translate, &frame(), Space::Local);
        // A quarter turn about Z carries X onto Y.
        assert!((local[0].direction - Vector3::unit_y()).magnitude() < EPSILON, "{:?}", local[0].direction);
        let world = handles(TransformMode::Translate, &frame(), Space::World);
        assert!((world[0].direction - Vector3::unit_x()).magnitude() < EPSILON);
    }

    #[test]
    fn a_uniform_scale_shows_only_the_ball() {
        let uniform = Frame { uniform_scale: true, ..frame() };
        assert_eq!(grips(&handles(TransformMode::Scale, &uniform, Space::Local)), [Grip::Ball]);
    }
}
