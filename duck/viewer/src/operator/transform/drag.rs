//! The world-space locus an interactive drag is solved against.
//!
//! A drag is measured by resolving two cursor rays — one through the pixel the
//! drag was anchored at, one through that pixel offset by the accumulated
//! mouse motion — against the same [`DragGeometry`]. The difference between
//! the two solutions is the drag, which keeps the grabbed point under the
//! cursor.

use duck_engine_common::{InnerSpace, Point3, Vector3, EPSILON};
use duck_engine_scene::common::{Plane, Ray};

/// The locus a drag point is confined to.
pub(super) enum DragGeometry {
    /// Ray plane intersection. The grabbed point stays exactly under the cursor.
    Plane(Plane),

    /// The point on an infinite line closest to the ray.
    Axis { origin: Point3, direction: Vector3 },
}

impl DragGeometry {
    /// The line through `origin` along `direction`, which need not be
    /// normalized.
    pub(super) fn axis(origin: Point3, direction: Vector3) -> Self {
        let direction = direction.normalize();
        DragGeometry::Axis { origin, direction }
    }

    /// The plane through `point` with the given normal.
    pub(super) fn plane(normal: Vector3, point: Point3) -> Self {
        DragGeometry::Plane(Plane::from_point(normal, point))
    }

    /// Resolves `ray` to a point on this geometry.
    ///
    /// `None` when the solve is unbounded or flipped: the ray is
    /// (near-)parallel to the geometry, or the solution lies behind the ray's
    /// origin. Past a vanishing line the solution inverts, and a drag must not
    /// jump to the mirrored side.
    pub(super) fn solve(&self, ray: &Ray) -> Option<Point3> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

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
