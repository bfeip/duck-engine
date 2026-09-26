//! Geometry for the built-in [`HandleShape`]s.
//!
//! Every shape is built at unit size about the origin along `+Y`, matching the
//! axis the mesh primitives already build along, so no shape needs reorienting
//! here — [`HandleSet`](super::HandleSet) rotates `+Y` onto the handle's
//! direction when it places the node. A unit-size shape spans roughly `0..1`,
//! which the renderer holds at a constant pixel size.

use duck_engine_common::{Deg, Matrix4, RgbaColor, Vector3};
use duck_engine_scene::resource::{FaceMaterial, MaterialFlags, Mesh, PrimitiveType};

use super::HandleShape;

/// Radial resolution of the round shapes.
const SEGMENTS: u32 = 16;

/// Handles are annotation, not scene geometry: unlit so they read as a flat
/// color, and double-sided so a thin shape stays visible from behind.
const HANDLE_FLAGS: MaterialFlags =
    MaterialFlags::DO_NOT_LIGHT.union(MaterialFlags::DOUBLE_SIDED);

/// Radius of an arm's shaft.
const SHAFT_RADIUS: f32 = 0.025;
/// Length of an arm's shaft, leaving the rest of the unit extent for its tip.
const SHAFT_LENGTH: f32 = 0.8;
/// Extent of an arm's tip, along the arm.
const TIP_LENGTH: f32 = 0.2;
/// Radius of a cone tip.
const CONE_RADIUS: f32 = 0.08;
/// Edge of a cube tip.
const CUBE_SIZE: f32 = 0.16;
/// Radius of a ball handle. Larger than a tip, since it is the whole grab.
const BALL_RADIUS: f32 = 0.12;
/// Edge of a quad handle.
const QUAD_SIZE: f32 = 0.35;
/// Tube radius of a ring handle.
const RING_TUBE: f32 = 0.03;

/// The material a handle of `color` draws with.
pub(super) fn material(color: RgbaColor) -> FaceMaterial {
    FaceMaterial::new().with_base_color_factor(color).with_flags(HANDLE_FLAGS)
}

/// The geometry for a built-in `shape`, at unit size about the origin along
/// `+Y`. [`HandleShape::Custom`] carries its own mesh and never reaches here.
pub(super) fn mesh(shape: &HandleShape) -> Mesh {
    match shape {
        HandleShape::Arrow => arm(cone_tip()),
        HandleShape::Cube => arm(cube_tip()),
        HandleShape::Ball => {
            Mesh::sphere(BALL_RADIUS, SEGMENTS, SEGMENTS / 2, PrimitiveType::TriangleList)
        }
        // Already in the XZ plane, so its axis is +Y like everything else.
        HandleShape::Ring => Mesh::torus(
            0.5,
            RING_TUBE,
            SEGMENTS * 2,
            SEGMENTS / 2,
            PrimitiveType::TriangleList,
        ),
        HandleShape::Quad => quad(),
        HandleShape::Custom(_) => unreachable!("a custom shape supplies its own mesh"),
    }
}

/// A shaft along `+Y` carrying `tip`, which is already placed at its end.
fn arm(tip: Mesh) -> Mesh {
    Mesh::cylinder(SHAFT_RADIUS, SHAFT_LENGTH, SEGMENTS, false, PrimitiveType::TriangleList)
        .transformed(&Matrix4::from_translation(Vector3::new(0.0, SHAFT_LENGTH / 2.0, 0.0)))
        .merged(&tip)
}

/// A cone at the end of an arm, apex pointing along `+Y`.
fn cone_tip() -> Mesh {
    Mesh::cone(CONE_RADIUS, TIP_LENGTH, SEGMENTS, true, PrimitiveType::TriangleList).transformed(
        &Matrix4::from_translation(Vector3::new(0.0, SHAFT_LENGTH + TIP_LENGTH / 2.0, 0.0)),
    )
}

/// A cube at the end of an arm.
fn cube_tip() -> Mesh {
    Mesh::cube(CUBE_SIZE, PrimitiveType::TriangleList).transformed(&Matrix4::from_translation(
        Vector3::new(0.0, SHAFT_LENGTH + TIP_LENGTH / 2.0, 0.0),
    ))
}

/// A square normal to `+Y`, turned up from the authored XY plane.
fn quad() -> Mesh {
    Mesh::quad(QUAD_SIZE, QUAD_SIZE, PrimitiveType::TriangleList)
        .transformed(&Matrix4::from_angle_x(Deg(90.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every built-in shape has to produce pickable triangles — a handle with
    /// no faces could be drawn but never grabbed.
    #[test]
    fn every_builtin_shape_has_triangles() {
        let shapes = [
            HandleShape::Arrow,
            HandleShape::Cube,
            HandleShape::Ball,
            HandleShape::Ring,
            HandleShape::Quad,
        ];
        for shape in shapes {
            let mesh = mesh(&shape);
            assert!(!mesh.vertices().is_empty(), "{shape:?} has no vertices");
            assert!(
                mesh.has_primitive_type(PrimitiveType::TriangleList),
                "{shape:?} has no triangles"
            );
        }
    }

    /// Arms grow along +Y from the origin, so the set can rotate +Y onto any
    /// direction without the geometry straddling the anchor.
    #[test]
    fn an_arm_spans_the_unit_extent_along_positive_y() {
        let mesh = mesh(&HandleShape::Arrow);
        let ys: Vec<f32> = mesh.vertices().iter().map(|v| v.position[1]).collect();
        let lo = ys.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(lo >= -CONE_RADIUS, "arm dips below the anchor: {lo}");
        assert!((hi - 1.0).abs() < 0.05, "arm does not reach the unit extent: {hi}");
    }
}
