use crate::{
    EuclideanSpace, InnerSpace, Matrix, Matrix3, Matrix4, Point3, Quaternion, SquareMatrix,
    Vector3, Vector4,
};

use crate::{Real, EPSILON};

/// A plane in 3D space defined by a normal and distance from origin.
///
/// The plane equation is: normal · point + distance = 0
/// Points with negative signed distance are "inside" (on the normal's opposite side).
/// This convention is useful for convex polyhedra where "inside" means inside the volume.
#[derive(Debug, Copy, Clone)]
pub struct Plane {
    /// Unit normal pointing "outside" the half-space
    pub normal: Vector3,
    /// Signed distance from origin along the normal
    pub d: Real,
}

impl Plane {
    /// Creates a plane from a normal and a distance.
    /// The normal will be normalized automatically.
    pub fn new(normal: Vector3, d: Real) -> Self {
        let normal = normal.normalize();
        Self { normal, d }
    }

    /// Creates a plane from a normal vector and a point on the plane.
    /// The normal will be normalized automatically.
    pub fn from_point(normal: Vector3, point: Point3) -> Self {
        let normal = normal.normalize();
        let distance = -normal.dot(point.to_vec());
        Self { normal, d: distance }
    }

    /// Creates a plane from the coefficients of the plane equation ax + by + cz + d = 0.
    /// The coefficients will be normalized so that (a, b, c) becomes a unit vector.
    pub fn from_coefficients(a: Real, b: Real, c: Real, d: Real) -> Self {
        let length = (a * a + b * b + c * c).sqrt();
        if length < EPSILON {
            // Degenerate plane, return a default
            return Self {
                normal: Vector3::new(0.0, 1.0, 0.0),
                d: 0.0,
            };
        }
        Self {
            normal: Vector3::new(a / length, b / length, c / length),
            d: d / length,
        }
    }

    pub fn xz() -> Self {
        Self { normal: Vector3::unit_y(), d: 0.0 }
    }

    pub fn xy() -> Self {
        Self { normal: Vector3::unit_z(), d: 0.0 }
    }

    pub fn yz() -> Self {
        Self { normal: Vector3::unit_x(), d: 0.0 }
    }

    /// Computes the signed distance from a point to the plane.
    ///
    /// - Positive: point is on the "outside" (same side as normal)
    /// - Zero: point is on the plane
    /// - Negative: point is on the "inside" (opposite side from normal)
    pub fn signed_distance(&self, point: Point3) -> Real {
        self.normal.dot(point.to_vec()) + self.d
    }

    /// Returns true if the point is on the inside (negative side) of the plane.
    pub fn contains_point(&self, point: Point3) -> bool {
        self.signed_distance(point) <= EPSILON
    }

    /// Transforms the plane by a 4x4 transformation matrix.
    ///
    /// For correct plane transformation, we need to use the inverse-transpose
    /// of the matrix to transform the plane normal correctly under non-uniform scaling.
    pub fn transform(&self, matrix: &Matrix4) -> Self {
        // Planes transform by the inverse-transpose of the matrix.
        // A plane can be represented as a 4D vector (a, b, c, d) where ax + by + cz + d = 0.
        // If M transforms points, then M^(-T) transforms planes.
        let inv_transpose = matrix
            .invert()
            .unwrap_or(Matrix4::identity())
            .transpose();

        let plane_vec = Vector4::new(self.normal.x, self.normal.y, self.normal.z, self.d);
        let transformed = inv_transpose * plane_vec;

        // Re-normalize the result
        Self::from_coefficients(transformed.x, transformed.y, transformed.z, transformed.w)
    }

    /// Finds the intersection point of a line segment with the plane.
    /// Returns Some((t, point)) where t is the parameter (0 to 1) and point is the intersection.
    /// Returns None if the segment is parallel to the plane or doesn't intersect.
    pub fn intersect_segment(
        &self,
        start: Point3,
        end: Point3,
    ) -> Option<(Real, Point3)> {
        let direction = end - start;
        let denom = self.normal.dot(direction);

        // Check if segment is parallel to plane
        if denom.abs() < EPSILON {
            return None;
        }

        let t = -(self.normal.dot(start.to_vec()) + self.d) / denom;

        // Check if intersection is within segment bounds
        if !(0.0..=1.0).contains(&t) {
            return None;
        }

        let point = start + direction * t;
        Some((t, point))
    }

    /// Projects `point` onto the plane, returning the closest point on it.
    ///
    /// Assumes a unit normal (the invariant maintained by every constructor).
    pub fn project_point(&self, point: Point3) -> Point3 {
        point - self.normal * self.signed_distance(point)
    }

    /// Returns two orthonormal vectors `(u, v)` that span the plane.
    ///
    /// The choice is deterministic but otherwise arbitrary: `u` is the world axis
    /// least aligned with the normal, projected into the plane (so it stays
    /// well-defined even when the plane is axis-aligned). `(u, normal, v)` form a
    /// right-handed frame, i.e. `v × u == normal` and `Matrix3::from_cols(u,
    /// normal, v)` is a proper rotation. Handy for laying out 2D content on an
    /// arbitrary plane.
    pub fn basis(&self) -> (Vector3, Vector3) {
        let n = self.normal.normalize();
        let candidate = [Vector3::unit_x(), Vector3::unit_y(), Vector3::unit_z()]
            .into_iter()
            .min_by(|a, b| a.dot(n).abs().partial_cmp(&b.dot(n).abs()).unwrap())
            .unwrap();
        let u = (candidate - n * candidate.dot(n)).normalize();
        let v = u.cross(n);
        (u, v)
    }

    /// Orthonormal right-handed frame for this plane as a [`Matrix3`].
    /// 
    /// Provides a rotation matrix whose columns are the world directions of
    /// the plane's local axes: local +X → the in-plane tangent `u`,
    /// local +Y → `normal × u`, local +Z → `normal`. Use it to place local-XY
    /// content with +Z as the up/extrude axis onto an arbitrary plane.
    ///
    /// Note this differs from [`basis`](Self::basis), whose right-handed frame is
    /// `(u, normal, v)`; here the normal is on +Z, the usual surface-frame
    /// convention.
    pub fn frame(&self) -> Matrix3 {
        let (u, _v) = self.basis();
        Matrix3::from_cols(u, self.normal.cross(u), self.normal)
    }

    /// This plane's [`frame`](Self::frame) as a rotation quaternion (local +Z maps
    /// to the plane normal, local XY lies in the plane).
    pub fn rotation(&self) -> Quaternion {
        Quaternion::from(self.frame())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Matrix4, Point3, Vector3, Rad, EPSILON};

    #[test]
    fn frame_is_proper_rotation_mapping_z_to_normal() {
        let planes = [
            Plane::xz(),
            Plane::xy(),
            Plane::yz(),
            Plane::from_point(Vector3::new(1.0, 2.0, 3.0), Point3::new(0.0, 0.0, 0.0)),
        ];
        for plane in planes {
            let f = plane.frame();
            // Proper rotation: no reflection.
            assert!((f.determinant() - 1.0).abs() < 1e-5);
            // Columns are orthonormal.
            assert!(f.x.dot(f.y).abs() < 1e-5);
            assert!(f.x.dot(f.z).abs() < 1e-5);
            assert!(f.y.dot(f.z).abs() < 1e-5);
            // Local +Z maps onto the plane normal.
            assert!((plane.rotation() * Vector3::unit_z() - plane.normal).magnitude() < 1e-5);
        }
    }

    #[test]
    fn test_plane_from_normal_and_point() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::new(0.0, 5.0, 0.0));

        // Normal should be normalized (already unit)
        assert!((plane.normal.magnitude() - 1.0).abs() < EPSILON);

        // Point on plane should have zero distance
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(10.0, 5.0, -3.0)).abs() < EPSILON);

        // Points above should be positive
        assert!(plane.signed_distance(Point3::new(0.0, 6.0, 0.0)) > 0.0);

        // Points below should be negative
        assert!(plane.signed_distance(Point3::new(0.0, 4.0, 0.0)) < 0.0);
    }

    #[test]
    fn test_plane_from_coefficients() {
        // Plane y = 5, or 0x + 1y + 0z - 5 = 0
        let plane = Plane::from_coefficients(0.0, 1.0, 0.0, -5.0);

        assert!((plane.normal.y - 1.0).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_from_coefficients_normalizes() {
        // Non-normalized coefficients: 0, 2, 0, -10 (equivalent to y = 5)
        let plane = Plane::from_coefficients(0.0, 2.0, 0.0, -10.0);

        assert!((plane.normal.magnitude() - 1.0).abs() < EPSILON);
        assert!((plane.normal.y - 1.0).abs() < EPSILON);
        assert!(plane.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_signed_distance() {
        // XZ plane at origin (normal pointing +Y)
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        assert!((plane.signed_distance(Point3::new(0.0, 0.0, 0.0))).abs() < EPSILON);
        assert!((plane.signed_distance(Point3::new(0.0, 3.0, 0.0)) - 3.0).abs() < EPSILON);
        assert!((plane.signed_distance(Point3::new(0.0, -2.0, 0.0)) - -2.0).abs() < EPSILON);
    }

    #[test]
    fn test_plane_contains_point() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Points on or below the plane are "inside"
        assert!(plane.contains_point(Point3::new(0.0, 0.0, 0.0)));
        assert!(plane.contains_point(Point3::new(0.0, -1.0, 0.0)));
        assert!(plane.contains_point(Point3::new(5.0, -0.5, 3.0)));

        // Points above are "outside"
        assert!(!plane.contains_point(Point3::new(0.0, 1.0, 0.0)));
    }

    #[test]
    fn test_plane_transform_translation() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());
        let translation = Matrix4::from_translation(Vector3::new(0.0, 5.0, 0.0));

        let transformed = plane.transform(&translation);

        // Normal should remain the same
        assert!((transformed.normal.y - 1.0).abs() < EPSILON);

        // Plane should now pass through y=5
        assert!(transformed.signed_distance(Point3::new(0.0, 5.0, 0.0)).abs() < EPSILON);
    }

    #[test]
    fn test_plane_transform_rotation() {
        // XZ plane (normal +Y)
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Rotate 90 degrees around X axis: Y -> Z
        let rotation = Matrix4::from_angle_x(Rad(crate::consts::FRAC_PI_2));
        let transformed = plane.transform(&rotation);

        // Normal should now point in +Z direction
        assert!(transformed.normal.x.abs() < 0.01);
        assert!(transformed.normal.y.abs() < 0.01);
        assert!((transformed.normal.z - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_plane_intersect_segment_hit() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        let start = Point3::new(0.0, -1.0, 0.0);
        let end = Point3::new(0.0, 1.0, 0.0);

        let result = plane.intersect_segment(start, end);
        assert!(result.is_some());

        let (t, point) = result.unwrap();
        assert!((t - 0.5).abs() < EPSILON);
        assert!(point.y.abs() < EPSILON);
    }

    #[test]
    fn test_plane_intersect_segment_miss_parallel() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Segment parallel to plane
        let start = Point3::new(0.0, 1.0, 0.0);
        let end = Point3::new(1.0, 1.0, 0.0);

        assert!(plane.intersect_segment(start, end).is_none());
    }

    #[test]
    fn test_plane_intersect_segment_miss_outside() {
        let plane = Plane::from_point(Vector3::new(0.0, 1.0, 0.0), Point3::origin());

        // Segment entirely above plane
        let start = Point3::new(0.0, 1.0, 0.0);
        let end = Point3::new(0.0, 2.0, 0.0);

        assert!(plane.intersect_segment(start, end).is_none());
    }

    #[test]
    fn test_plane_project_point() {
        // y = 5
        let plane = Plane::from_coefficients(0.0, 1.0, 0.0, -5.0);

        // An off-plane point keeps its in-plane coords, snaps onto the plane.
        let p = plane.project_point(Point3::new(3.0, 10.0, 4.0));
        assert!((p.x - 3.0).abs() < EPSILON);
        assert!((p.y - 5.0).abs() < EPSILON);
        assert!((p.z - 4.0).abs() < EPSILON);

        // A point already on the plane is unchanged.
        let on = plane.project_point(Point3::new(-2.0, 5.0, 7.0));
        assert!((on - Point3::new(-2.0, 5.0, 7.0)).magnitude() < EPSILON);

        // The world origin projects to the plane's nearest point.
        let o = plane.project_point(Point3::origin());
        assert!((o - Point3::new(0.0, 5.0, 0.0)).magnitude() < EPSILON);
    }

    #[test]
    fn test_plane_basis_is_orthonormal_right_handed() {
        let planes = [
            Plane::xz(),
            Plane::xy(),
            Plane::yz(),
            Plane::from_point(Vector3::new(1.0, 2.0, 3.0), Point3::new(1.0, 0.0, 0.0)),
        ];
        for plane in planes {
            let n = plane.normal.normalize();
            let (u, v) = plane.basis();

            // Unit length.
            assert!((u.magnitude() - 1.0).abs() < EPSILON);
            assert!((v.magnitude() - 1.0).abs() < EPSILON);
            // Mutually orthogonal and both in-plane.
            assert!(u.dot(v).abs() < EPSILON);
            assert!(u.dot(n).abs() < EPSILON);
            assert!(v.dot(n).abs() < EPSILON);
            // Right-handed: v × u == normal.
            assert!((v.cross(u) - n).magnitude() < EPSILON);
        }
    }
}
