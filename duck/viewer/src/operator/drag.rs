//! The world-space locus an interactive drag is solved against.
//!
//! A drag is measured by resolving two cursor rays — one through the pixel the
//! drag was anchored at, one through that pixel offset by the accumulated
//! mouse motion — against the same [`DragGeometry`]. The difference between
//! the two solutions is the drag, which keeps the grabbed point under the
//! cursor.

use duck_engine_common::{InnerSpace, Point3, Vector3, EPSILON};
use duck_engine_scene::common::{Plane, Ray};
use duck_engine_scene::PositionedCamera;

/// The locus a drag point is confined to.
pub enum DragGeometry {
    /// Ray plane intersection. The grabbed point stays exactly under the cursor.
    Plane(Plane),

    /// The point on an infinite line closest to the ray.
    Axis { origin: Point3, direction: Vector3 },
}

impl DragGeometry {
    /// The line through `origin` along `direction`, which need not be
    /// normalized.
    pub fn axis(origin: Point3, direction: Vector3) -> Self {
        let direction = direction.normalize();
        DragGeometry::Axis { origin, direction }
    }

    /// The plane through `point` with the given normal.
    pub fn plane(normal: Vector3, point: Point3) -> Self {
        DragGeometry::Plane(Plane::from_point(normal, point))
    }

    /// Resolves `ray` to a point on this geometry.
    ///
    /// `None` when the solve is unbounded or flipped: the ray is
    /// (near-)parallel to the geometry, or the solution lies behind the ray's
    /// origin. Past a vanishing line the solution inverts, and a drag must not
    /// jump to the mirrored side.
    pub fn solve(&self, ray: &Ray) -> Option<Point3> {
        match self {
            // `intersect_plane` already rejects both near-parallel rays and
            // solutions behind the origin.
            DragGeometry::Plane(plane) => ray.intersect_plane(plane).map(|(_, point)| point),
            DragGeometry::Axis { origin, direction } => {
                if direction.magnitude2() < EPSILON {
                    return None;
                }
                let t = ray.closest_param_on_axis(*origin, *direction)?;
                let point = origin + direction * t;
                ((point - ray.origin).dot(ray.direction) > 0.0).then_some(point)
            }
        }
    }
}

/// Solves `geometry` against the rays through two screen pixels: the one a drag
/// was anchored at, and the one it has reached.
///
/// Returns `(anchor_point, cursor_point)`; the drag is their difference. `None`
/// when either end is degenerate — a caller that must keep a drag alive across
/// a vanishing line falls back to a view-plane solve.
pub fn solve_drag(
    geometry: &DragGeometry,
    anchor: (f32, f32),
    cursor: (f32, f32),
    camera: &PositionedCamera,
    size: (u32, u32),
) -> Option<(Point3, Point3)> {
    let (width, height) = size;
    let anchor_ray = camera.ray_from_screen_point(anchor.0, anchor.1, width, height);
    let cursor_ray = camera.ray_from_screen_point(cursor.0, cursor.1, width, height);
    Some((geometry.solve(&anchor_ray)?, geometry.solve(&cursor_ray)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::Projection;

    const EPSILON: crate::common::Real = 1e-6;

    /// A camera looking down -Z from five units out, with a square viewport.
    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
        }
    }

    #[test]
    fn solve_drag_returns_both_ends_on_the_geometry() {
        let geometry = DragGeometry::plane(Vector3::unit_z(), Point3::new(0.0, 0.0, 0.0));
        let (anchor, cursor) =
            solve_drag(&geometry, (100.0, 100.0), (140.0, 100.0), &camera(), (200, 200))
                .expect("both rays cross the plane");

        let DragGeometry::Plane(plane) = &geometry else { unreachable!() };
        assert!(plane.signed_distance(anchor).abs() < EPSILON);
        assert!(plane.signed_distance(cursor).abs() < EPSILON);
        // Dragging right moves the solved point in +x and nothing else.
        assert!(cursor.x > anchor.x);
        assert!((cursor.y - anchor.y).abs() < EPSILON);
    }

    #[test]
    fn solve_drag_of_an_unmoved_cursor_is_zero() {
        let geometry = DragGeometry::axis(Point3::new(0.0, 0.0, 0.0), Vector3::unit_x());
        let (anchor, cursor) =
            solve_drag(&geometry, (120.0, 90.0), (120.0, 90.0), &camera(), (200, 200))
                .expect("the ray is skew to the axis");

        assert!((cursor - anchor).magnitude() < EPSILON);
    }

    #[test]
    fn solve_drag_rejects_a_degenerate_end() {
        // A plane seen exactly edge-on from this camera: neither ray solves.
        let geometry = DragGeometry::plane(Vector3::unit_z(), Point3::new(0.0, 0.0, 20.0));
        assert!(
            solve_drag(&geometry, (100.0, 100.0), (140.0, 100.0), &camera(), (200, 200)).is_none()
        );
    }

    #[test]
    fn plane_solve_lies_on_plane() {
        let geometry = DragGeometry::plane(Vector3::unit_y(), Point3::new(0.0, 0.0, 0.0));
        let ray = Ray::new(Point3::new(2.0, 3.0, 1.0), Vector3::new(1.0, -1.0, 0.0));

        let point = geometry.solve(&ray).expect("ray crosses the plane");
        let DragGeometry::Plane(plane) = &geometry else { unreachable!() };
        assert!(plane.signed_distance(point).abs() < EPSILON);
    }

    #[test]
    fn plane_solve_rejects_ray_pointing_away() {
        // The plane is behind the ray origin: this is the horizon case, where a
        // drag must degrade rather than jump to the mirrored intersection.
        let geometry = DragGeometry::plane(Vector3::unit_y(), Point3::new(0.0, 0.0, 0.0));
        let ray = Ray::new(Point3::new(0.0, 3.0, 0.0), Vector3::new(0.0, 1.0, 0.0));

        assert!(geometry.solve(&ray).is_none());
    }

    #[test]
    fn plane_solve_rejects_edge_on_ray() {
        let geometry = DragGeometry::plane(Vector3::unit_y(), Point3::new(0.0, 0.0, 0.0));
        let ray = Ray::new(Point3::new(0.0, 3.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(geometry.solve(&ray).is_none());
    }

    #[test]
    fn axis_solve_lies_on_axis() {
        let origin = Point3::new(1.0, 2.0, 3.0);
        let direction = Vector3::unit_x();
        let geometry = DragGeometry::axis(origin, direction);
        let ray = Ray::new(Point3::new(4.0, 8.0, 5.0), Vector3::new(0.2, -1.0, -0.3));

        let point = geometry.solve(&ray).expect("ray is skew to the axis");
        assert!((point - origin).cross(direction).magnitude() < EPSILON);
    }

    #[test]
    fn axis_solve_ignores_direction_magnitude() {
        // `closest_param_on_axis` returns its parameter in normalized units, so
        // a non-unit direction must not scale the solution.
        let origin = Point3::new(0.0, 0.0, 0.0);
        let ray = Ray::new(Point3::new(3.0, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0));

        let unit = DragGeometry::axis(origin, Vector3::unit_x()).solve(&ray).unwrap();
        let scaled =
            DragGeometry::axis(origin, Vector3::unit_x() * 10.0).solve(&ray).unwrap();

        assert!((unit - scaled).magnitude() < EPSILON);
        assert!((unit.x - 3.0).abs() < EPSILON);
    }

    #[test]
    fn axis_solve_rejects_parallel_ray() {
        let geometry = DragGeometry::axis(Point3::new(0.0, 0.0, 0.0), Vector3::unit_x());
        let ray = Ray::new(Point3::new(0.0, 1.0, 0.0), Vector3::unit_x());

        assert!(geometry.solve(&ray).is_none());
    }

    #[test]
    fn axis_solve_rejects_solution_behind_ray_origin() {
        // Closest point on the axis sits behind the ray origin — the flipped
        // side of the axis's vanishing line.
        let geometry = DragGeometry::axis(Point3::new(0.0, 0.0, 0.0), Vector3::unit_x());
        let ray = Ray::new(Point3::new(3.0, 1.0, 0.0), Vector3::new(0.0, 1.0, 0.0));

        assert!(geometry.solve(&ray).is_none());
    }

    #[test]
    fn degenerate_axis_direction_solves_to_nothing() {
        let geometry = DragGeometry::axis(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 0.0));
        let ray = Ray::new(Point3::new(3.0, 5.0, 0.0), Vector3::new(0.0, -1.0, 0.0));

        assert!(geometry.solve(&ray).is_none());
    }
}
