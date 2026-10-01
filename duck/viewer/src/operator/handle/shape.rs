//! Geometry for the built-in handle forms and reaches.
//!
//! A form is built at unit size about the origin along `+Y`, matching the axis
//! the mesh primitives already build along, so nothing needs reorienting here —
//! [`HandleSet`](super::HandleSet) rotates `+Y` onto the handle's direction when
//! it places the node. A unit-size form spans roughly `0..1`, which the renderer
//! holds at a constant pixel size.
//!
//! [`HandleReach`] is applied on top: it moves the form off the origin, and for
//! an arm adds the shaft that carries it. Everything a reach does is baked into
//! the mesh, so that it scales with the handle instead of staying a fixed world
//! size — the one exception is [`HandleReach::Leader`], whose whole point is to
//! span a real distance, and which gets its own node.

use duck_engine_common::{Deg, Matrix4, Quaternion, Real, RgbaColor, Rotation, Vector3};
use duck_engine_scene::resource::{
    FaceMaterial, LineMaterial, MaterialFlags, Mesh, MeshPrimitive, PrimitiveType, Vertex,
};

use super::{HandleReach, HandleShape};

/// Radial resolution of the round forms.
const SEGMENTS: u32 = 16;

/// Handles are annotation, not scene geometry: unlit so they read as a flat
/// color, and double-sided so a thin form stays visible from behind.
const HANDLE_FLAGS: MaterialFlags =
    MaterialFlags::DO_NOT_LIGHT.union(MaterialFlags::DOUBLE_SIDED);

/// Edge of a cube grab.
const CUBE_SIZE: Real = 0.22;
/// Radius of a ball grab.
const BALL_RADIUS: Real = 0.12;
/// Base radius and length of a cone grab.
const CONE_RADIUS: Real = 0.11;
const CONE_LENGTH: Real = 0.28;
/// Edge of a quad grab.
const QUAD_SIZE: Real = 0.35;
/// Radius of a ring grab, and of its tube.
const RING_RADIUS: Real = 0.5;
const RING_TUBE: Real = 0.03;

/// Radius and length of an arm's shaft.
const SHAFT_RADIUS: Real = 0.025;
const SHAFT_LENGTH: Real = 0.8;
/// Where an arm's grab is centred along `+Y`, leaving it inside the unit extent.
const ARM_TIP: Real = 0.9;
/// A grab carried on an arm is scaled down: at full size it would dwarf the
/// shaft under it.
const ARM_TIP_SCALE: Real = 0.72;

/// The material a handle of `color` draws with.
pub(super) fn material(color: RgbaColor) -> FaceMaterial {
    FaceMaterial::new().with_base_color_factor(color).with_flags(HANDLE_FLAGS)
}

/// The geometry for a built-in `shape` under `reach`, in the node's local frame.
///
/// `rotation` is the node's own `+Y`→direction turn. It is needed only for
/// [`HandleReach::Offset`], whose displacement is given in world axes and so has
/// to be pre-rotated here for the node's rotation to cancel back out.
/// [`HandleShape::Custom`] carries its own mesh and never reaches here.
pub(super) fn mesh(shape: &HandleShape, reach: &HandleReach, rotation: Quaternion) -> Mesh {
    let grab = form(shape);
    match reach {
        HandleReach::Arm => arm(grab),
        HandleReach::Offset(world) => {
            grab.transformed(&Matrix4::from_translation(rotation.invert() * *world))
        }
        HandleReach::None | HandleReach::Leader(_) => grab,
    }
}

/// A grab's form, at unit size about the origin along `+Y`.
fn form(shape: &HandleShape) -> Mesh {
    match shape {
        HandleShape::Cone => {
            Mesh::cone(CONE_RADIUS, CONE_LENGTH, SEGMENTS, true, PrimitiveType::TriangleList)
        }
        HandleShape::Cube => Mesh::cube(CUBE_SIZE, PrimitiveType::TriangleList),
        HandleShape::Ball => {
            Mesh::sphere(BALL_RADIUS, SEGMENTS, SEGMENTS / 2, PrimitiveType::TriangleList)
        }
        // Already in the XZ plane, so its axis is +Y like everything else.
        HandleShape::Ring => Mesh::torus(
            RING_RADIUS,
            RING_TUBE,
            SEGMENTS * 2,
            SEGMENTS / 2,
            PrimitiveType::TriangleList,
        ),
        HandleShape::Quad => Mesh::quad(QUAD_SIZE, QUAD_SIZE, PrimitiveType::TriangleList)
            .transformed(&Matrix4::from_angle_x(Deg(90.0))),
        HandleShape::Custom(_) => unreachable!("a custom shape supplies its own mesh"),
    }
}

/// A shaft along `+Y` carrying `grab` at its far end.
fn arm(grab: Mesh) -> Mesh {
    let shaft =
        Mesh::cylinder(SHAFT_RADIUS, SHAFT_LENGTH, SEGMENTS, false, PrimitiveType::TriangleList)
            .transformed(&Matrix4::from_translation(Vector3::new(0.0, SHAFT_LENGTH / 2.0, 0.0)));
    let tip = grab.transformed(
        &(Matrix4::from_translation(Vector3::new(0.0, ARM_TIP, 0.0))
            * Matrix4::from_scale(ARM_TIP_SCALE)),
    );
    shaft.merged(&tip)
}

/// A leader line's geometry: one segment from the origin to `+Y` at unit length.
///
/// Its node is scaled to the real distance it spans, so — unlike the forms —
/// this one is authored to be stretched.
pub(super) fn leader_mesh() -> Mesh {
    let vertices = vec![
        Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] },
        Vertex { position: [0.0, 1.0, 0.0], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] },
    ];
    Mesh::from_raw(
        vertices,
        vec![MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: vec![0, 1] }],
    )
}

/// The material a leader line of `color` draws with.
pub(super) fn leader_material(color: RgbaColor) -> LineMaterial {
    LineMaterial::new(color)
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::{array_to_vec3, InnerSpace, One};

    const BUILTINS: [HandleShape; 5] = [
        HandleShape::Cone,
        HandleShape::Cube,
        HandleShape::Ball,
        HandleShape::Ring,
        HandleShape::Quad,
    ];

    fn no_turn() -> Quaternion {
        Quaternion::one()
    }

    fn extent_along_y(mesh: &Mesh) -> (Real, Real) {
        let ys = mesh.vertices().iter().map(|v| v.position[1] as Real);
        ys.fold((Real::INFINITY, Real::NEG_INFINITY), |(lo, hi), y| (lo.min(y), hi.max(y)))
    }

    /// Every form has to produce pickable triangles — a handle with no faces
    /// could be drawn but never grabbed.
    #[test]
    fn every_builtin_form_has_triangles() {
        for shape in BUILTINS {
            let mesh = form(&shape);
            assert!(!mesh.vertices().is_empty(), "{shape:?} has no vertices");
            assert!(
                mesh.has_primitive_type(PrimitiveType::TriangleList),
                "{shape:?} has no triangles"
            );
        }
    }

    /// Every form is authored about the origin, so `HandleReach::None` puts the
    /// grab on the anchor whatever form it wears.
    #[test]
    fn a_form_sits_on_the_origin() {
        for shape in BUILTINS {
            let (lo, hi) = extent_along_y(&form(&shape));
            assert!(lo <= 0.0 && hi >= 0.0, "{shape:?} does not straddle the origin: {lo}..{hi}");
        }
    }

    /// An arm reaches out to the unit extent and does not dip behind its anchor,
    /// so the set can turn it onto any direction.
    #[test]
    fn an_arm_spans_the_unit_extent_along_positive_y() {
        for shape in BUILTINS {
            let mesh = mesh(&shape, &HandleReach::Arm, no_turn());
            let (lo, hi) = extent_along_y(&mesh);
            assert!(lo >= -RING_RADIUS, "{shape:?} arm dips below the anchor: {lo}");
            assert!((hi - 1.0).abs() < 0.15, "{shape:?} arm misses the unit extent: {hi}");
        }
    }

    /// The same form serves as a lone grab and as an arm's tip — that is the
    /// point of splitting reach out of shape — so an arm must be the form plus a
    /// shaft, not a different form.
    #[test]
    fn an_arm_adds_a_shaft_to_the_same_form() {
        let lone = mesh(&HandleShape::Cube, &HandleReach::None, no_turn());
        let armed = mesh(&HandleShape::Cube, &HandleReach::Arm, no_turn());
        assert!(armed.vertices().len() > lone.vertices().len(), "arm added no shaft");
        // The lone grab straddles the anchor; the armed one is pushed out to the tip.
        assert!(extent_along_y(&lone).0 < 0.0);
        assert!(extent_along_y(&armed).1 > SHAFT_LENGTH);
    }

    /// An offset is given in world axes, so the node's own rotation must cancel
    /// out of it — otherwise a handle turned onto a direction would fling its
    /// grab somewhere unrelated.
    #[test]
    fn an_offset_is_pre_rotated_into_the_local_frame() {
        // A node whose +Y has been turned onto +X.
        let rotation = Quaternion::from_arc(Vector3::unit_y(), Vector3::unit_x(), None);
        let offset = Vector3::new(0.0, 0.5, 0.0);
        // A cube, whose vertices are symmetric about its centre, so the centroid
        // below is exactly where the grab sits.
        let mesh = mesh(&HandleShape::Cube, &HandleReach::Offset(offset), rotation);

        // Turning the mesh by the node rotation must put the grab back at the
        // world offset that was asked for.
        let centre = mesh
            .vertices()
            .iter()
            .fold(Vector3::new(0.0, 0.0, 0.0), |acc, v| acc + array_to_vec3(v.position))
            / mesh.vertices().len() as Real;
        let world = rotation * centre;
        assert!((world - offset).magnitude() < 1e-5, "offset landed at {world:?}");
    }

    #[test]
    fn a_leader_leaves_the_form_on_the_anchor() {
        // The leader is a separate node, so the form itself must not move.
        let plain = mesh(&HandleShape::Cube, &HandleReach::None, no_turn());
        let led = mesh(
            &HandleShape::Cube,
            &HandleReach::Leader(duck_engine_common::Point3::new(9.0, 9.0, 9.0)),
            no_turn(),
        );
        assert_eq!(plain.vertices().len(), led.vertices().len());
        assert_eq!(extent_along_y(&plain), extent_along_y(&led));
    }
}
