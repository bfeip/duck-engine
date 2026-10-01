use crate::{Matrix4, Point3};

use crate::{EPSILON, Real, ray::Ray};

/// An axis-aligned bounding box (AABB) in 3D space.
#[derive(Debug, Copy, Clone)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    /// Creates a new AABB from min and max points.
    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    /// Creates an AABB that encompasses all the given points.
    /// Returns None if the points slice is empty.
    pub fn from_points(points: &[Point3]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }

        let mut min = points[0];
        let mut max = points[0];

        for point in points.iter().skip(1) {
            min.x = min.x.min(point.x);
            min.y = min.y.min(point.y);
            min.z = min.z.min(point.z);

            max.x = max.x.max(point.x);
            max.y = max.y.max(point.y);
            max.z = max.z.max(point.z);
        }

        Some(Self { min, max })
    }

    /// Returns the 8 corner points of the AABB.
    pub fn corners(&self) -> [Point3; 8] {
        [
            Point3::new(self.min.x, self.min.y, self.min.z),
            Point3::new(self.max.x, self.min.y, self.min.z),
            Point3::new(self.min.x, self.max.y, self.min.z),
            Point3::new(self.max.x, self.max.y, self.min.z),
            Point3::new(self.min.x, self.min.y, self.max.z),
            Point3::new(self.max.x, self.min.y, self.max.z),
            Point3::new(self.min.x, self.max.y, self.max.z),
            Point3::new(self.max.x, self.max.y, self.max.z),
        ]
    }

    /// Transforms the AABB by the given 4x4 transformation matrix.
    /// This handles rotation/scaling/shearing by transforming all 8 corners
    /// and computing a new axis-aligned bounding box.
    pub fn transform(&self, matrix: &Matrix4) -> Self {
        // Transform all 8 corners of the box
        let corners = self.corners();

        let transformed_corners: Vec<Point3> = corners
            .iter()
            .map(|corner| {
                let homogeneous = matrix * corner.to_homogeneous();
                Point3::from_homogeneous(homogeneous)
            })
            .collect();

        // Unwrap is safe because we know we have 8 corners
        Self::from_points(&transformed_corners).unwrap()
    }

    /// Tests if a ray intersects this AABB using the slab method.
    /// Returns the t parameter of the intersection point if it hits, None otherwise.
    /// If the ray originates inside the box, returns Some(0.0).
    pub fn intersects_ray(&self, ray: &Ray) -> Option<Real> {
        #[derive(Copy, Clone)]
        enum Axis { X, Y, Z }

        let mut tmin = Real::NEG_INFINITY;
        let mut tmax = Real::INFINITY;

        // Test intersection with each pair of parallel planes
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let (origin_component, dir_component, min_component, max_component) = match axis {
                Axis::X => (ray.origin.x, ray.direction.x, self.min.x, self.max.x),
                Axis::Y => (ray.origin.y, ray.direction.y, self.min.y, self.max.y),
                Axis::Z => (ray.origin.z, ray.direction.z, self.min.z, self.max.z),
            };

            if dir_component.abs() < EPSILON {
                // Ray is parallel to the slab
                if origin_component < min_component || origin_component > max_component {
                    return None;
                }
            } else {
                // Compute intersection t values
                let inv_dir = 1.0 / dir_component;
                let mut t1 = (min_component - origin_component) * inv_dir;
                let mut t2 = (max_component - origin_component) * inv_dir;

                if t1 > t2 {
                    std::mem::swap(&mut t1, &mut t2);
                }

                tmin = tmin.max(t1);
                tmax = tmax.min(t2);

                if tmin > tmax {
                    return None;
                }
            }
        }

        // Return the near intersection point (or 0 if inside)
        if tmin >= 0.0 {
            Some(tmin)
        } else if tmax >= 0.0 {
            Some(0.0) // Ray origin is inside the box
        } else {
            None // Box is behind the ray
        }
    }

    /// Expands the AABB to include the given point.
    pub fn expand(&self, point: Point3) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(point.x),
                self.min.y.min(point.y),
                self.min.z.min(point.z),
            ),
            max: Point3::new(
                self.max.x.max(point.x),
                self.max.y.max(point.y),
                self.max.z.max(point.z),
            ),
        }
    }

    /// Merges this AABB with another, returning the bounding box that encompasses both.
    pub fn merge(&self, other: &Aabb) -> Self {
        Self {
            min: Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Returns the center point of the AABB.
    pub fn center(&self) -> Point3 {
        Point3::new(
            (self.min.x + self.max.x) / 2.0,
            (self.min.y + self.max.y) / 2.0,
            (self.min.z + self.max.z) / 2.0,
        )
    }

    /// Returns the size (extents) of the AABB along each axis.
    pub fn size(&self) -> (Real, Real, Real) {
        (
            self.max.x - self.min.x,
            self.max.y - self.min.y,
            self.max.z - self.min.z,
        )
    }

    /// Returns the diagonal length of the AABB (distance from min to max corner).
    pub fn diagonal(&self) -> Real {
        let (sx, sy, sz) = self.size();
        (sx * sx + sy * sy + sz * sz).sqrt()
    }

    /// Returns the bounding sphere radius (half the diagonal).
    /// This represents the smallest sphere centered at the AABB center that contains the box.
    pub fn bounding_sphere_radius(&self) -> Real {
        self.diagonal() / 2.0
    }

    /// Tests if a point is inside the AABB (inclusive of boundaries).
    pub fn contains_point(&self, point: Point3) -> bool {
        point.x >= self.min.x && point.x <= self.max.x &&
        point.y >= self.min.y && point.y <= self.max.y &&
        point.z >= self.min.z && point.z <= self.max.z
    }

    /// Tests if this AABB intersects another AABB.
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x &&
        self.min.y <= other.max.y && self.max.y >= other.min.y &&
        self.min.z <= other.max.z && self.max.z >= other.min.z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{InnerSpace, Matrix4, Vector3, Rad, EPSILON};

    #[test]
    fn test_aabb_new() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        assert_eq!(aabb.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(aabb.max, Point3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_from_points() {
        let points = vec![
            Point3::new(1.0, 2.0, 3.0),
            Point3::new(-1.0, 4.0, 2.0),
            Point3::new(2.0, 1.0, 5.0),
        ];
        let aabb = Aabb::from_points(&points).unwrap();

        assert_eq!(aabb.min, Point3::new(-1.0, 1.0, 2.0));
        assert_eq!(aabb.max, Point3::new(2.0, 4.0, 5.0));
    }

    #[test]
    fn test_aabb_from_single_point() {
        let points = vec![Point3::new(5.0, 10.0, 15.0)];
        let aabb = Aabb::from_points(&points).unwrap();

        assert_eq!(aabb.min, Point3::new(5.0, 10.0, 15.0));
        assert_eq!(aabb.max, Point3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn test_aabb_from_empty_points() {
        let points: Vec<Point3> = vec![];
        assert!(Aabb::from_points(&points).is_none());
    }

    #[test]
    fn test_aabb_corners() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 2.0, 3.0));
        let corners = aabb.corners();

        assert_eq!(corners.len(), 8);

        // Verify min corner
        assert_eq!(corners[0], Point3::new(0.0, 0.0, 0.0));
        // Verify max corner
        assert_eq!(corners[7], Point3::new(1.0, 2.0, 3.0));

        // Verify all corners are within bounds
        for corner in corners.iter() {
            assert!(corner.x >= aabb.min.x && corner.x <= aabb.max.x);
            assert!(corner.y >= aabb.min.y && corner.y <= aabb.max.y);
            assert!(corner.z >= aabb.min.z && corner.z <= aabb.max.z);
        }
    }

    #[test]
    fn test_aabb_center() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 20.0, 30.0));
        let center = aabb.center();

        assert_eq!(center, Point3::new(5.0, 10.0, 15.0));
    }

    #[test]
    fn test_aabb_size() {
        let aabb = Aabb::new(Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 7.0, 10.0));
        let (width, height, depth) = aabb.size();

        assert!((width - 3.0).abs() < EPSILON);
        assert!((height - 5.0).abs() < EPSILON);
        assert!((depth - 7.0).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_expand() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let expanded = aabb.expand(Point3::new(2.0, -1.0, 0.5));

        assert_eq!(expanded.min, Point3::new(0.0, -1.0, 0.0));
        assert_eq!(expanded.max, Point3::new(2.0, 1.0, 1.0));
    }

    #[test]
    fn test_aabb_expand_internal_point() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        // Point already inside
        let internal_point = Point3::new(5.0, 5.0, 5.0);
        let expanded = aabb.expand(internal_point);

        // Should be unchanged
        assert_eq!(expanded.min, aabb.min);
        assert_eq!(expanded.max, aabb.max);
    }

    #[test]
    fn test_aabb_merge() {
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let aabb2 = Aabb::new(Point3::new(0.5, 0.5, 0.5), Point3::new(2.0, 2.0, 2.0));
        let merged = aabb1.merge(&aabb2);

        assert_eq!(merged.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(merged.max, Point3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_aabb_merge_disjoint() {
        // Two boxes that don't overlap
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let aabb2 = Aabb::new(Point3::new(5.0, 5.0, 5.0), Point3::new(6.0, 6.0, 6.0));

        let merged = aabb1.merge(&aabb2);

        // Should encompass both
        assert_eq!(merged.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(merged.max, Point3::new(6.0, 6.0, 6.0));
    }

    #[test]
    fn test_aabb_merge_contained() {
        // Small box inside large box
        let large = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let small = Aabb::new(Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0));

        let merged = large.merge(&small);

        // Should be same as large box
        assert_eq!(merged.min, large.min);
        assert_eq!(merged.max, large.max);
    }

    #[test]
    fn test_aabb_contains_point_inside() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        assert!(aabb.contains_point(Point3::new(5.0, 5.0, 5.0)));
        assert!(aabb.contains_point(Point3::new(0.0, 0.0, 0.0))); // Corner
        assert!(aabb.contains_point(Point3::new(10.0, 10.0, 10.0))); // Corner
    }

    #[test]
    fn test_aabb_contains_point_outside() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        assert!(!aabb.contains_point(Point3::new(-1.0, 5.0, 5.0)));
        assert!(!aabb.contains_point(Point3::new(11.0, 5.0, 5.0)));
        assert!(!aabb.contains_point(Point3::new(5.0, -1.0, 5.0)));
        assert!(!aabb.contains_point(Point3::new(5.0, 11.0, 5.0)));
    }

    #[test]
    fn test_aabb_intersects_overlapping() {
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let aabb2 = Aabb::new(Point3::new(5.0, 5.0, 5.0), Point3::new(15.0, 15.0, 15.0));

        assert!(aabb1.intersects(&aabb2));
        assert!(aabb2.intersects(&aabb1)); // Symmetric
    }

    #[test]
    fn test_aabb_intersects_disjoint() {
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let aabb2 = Aabb::new(Point3::new(20.0, 20.0, 20.0), Point3::new(30.0, 30.0, 30.0));

        assert!(!aabb1.intersects(&aabb2));
        assert!(!aabb2.intersects(&aabb1));
    }

    #[test]
    fn test_aabb_intersects_touching() {
        let aabb1 = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let aabb2 = Aabb::new(Point3::new(10.0, 0.0, 0.0), Point3::new(20.0, 10.0, 10.0));

        // Touching counts as intersecting
        assert!(aabb1.intersects(&aabb2));
    }

    #[test]
    fn test_aabb_intersects_contained() {
        let large = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let small = Aabb::new(Point3::new(2.0, 2.0, 2.0), Point3::new(3.0, 3.0, 3.0));

        assert!(large.intersects(&small));
        assert!(small.intersects(&large));
    }

    // ===== Ray Intersection Tests =====

    #[test]
    fn test_ray_aabb_intersection_hit() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        let t = aabb.intersects_ray(&ray);
        assert!(t.is_some());
        assert!((t.unwrap() - 4.0).abs() < EPSILON); // Should hit at t=4
    }

    #[test]
    fn test_ray_aabb_intersection_miss() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none());
    }

    #[test]
    fn test_ray_aabb_intersection_inside() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        let t = aabb.intersects_ray(&ray);
        assert!(t.is_some());
        assert!((t.unwrap() - 0.0).abs() < EPSILON); // Origin inside returns 0
    }

    #[test]
    fn test_ray_aabb_intersection_behind_ray() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        let ray = Ray::new(Point3::new(5.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none()); // Box is behind the ray
    }

    #[test]
    fn test_ray_aabb_parallel_to_plane() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
        // Ray parallel to X axis, but outside the box in Y
        let ray = Ray::new(Point3::new(-5.0, 5.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

        assert!(aabb.intersects_ray(&ray).is_none());
    }

    #[test]
    fn test_aabb_intersects_ray_grazing() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Ray just touching the top edge
        let ray = Ray::new(Point3::new(-1.0, 1.0, 0.5), Vector3::new(1.0, 0.0, 0.0));

        let result = aabb.intersects_ray(&ray);
        assert!(result.is_some());
    }

    #[test]
    fn test_aabb_intersects_ray_from_corner() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Ray starting at corner, shooting out
        let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(-1.0, -1.0, -1.0));

        let result = aabb.intersects_ray(&ray);
        // Should return 0 since origin is at corner (inside/on surface)
        assert!(result.is_some());
        assert!((result.unwrap() - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_intersects_ray_diagonal() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));

        // Ray shooting diagonally through the box
        let ray = Ray::new(Point3::new(-1.0, -1.0, -1.0), Vector3::new(1.0, 1.0, 1.0).normalize());

        let result = aabb.intersects_ray(&ray);
        assert!(result.is_some());
        assert!(result.unwrap() > 0.0);
    }

    // ===== Transform Tests =====

    #[test]
    fn test_aabb_transform_identity() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let identity = Matrix4::from_scale(1.0);

        let transformed = aabb.transform(&identity);

        assert_eq!(transformed.min, aabb.min);
        assert_eq!(transformed.max, aabb.max);
    }

    #[test]
    fn test_aabb_transform_translation() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let translation = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));

        let transformed = aabb.transform(&translation);

        assert_eq!(transformed.min, Point3::new(5.0, 10.0, 15.0));
        assert_eq!(transformed.max, Point3::new(6.0, 11.0, 16.0));
    }

    #[test]
    fn test_aabb_transform_scale() {
        let aabb = Aabb::new(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 1.0, 1.0));
        let scale = Matrix4::from_scale(2.0);

        let transformed = aabb.transform(&scale);

        assert_eq!(transformed.min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(transformed.max, Point3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn test_aabb_transform_rotation() {
        // Unit cube at origin
        let aabb = Aabb::new(Point3::new(-0.5, -0.5, -0.5), Point3::new(0.5, 0.5, 0.5));

        // Rotate 45 degrees around Z
        let rotation = Matrix4::from_angle_z(Rad(crate::consts::PI / 4.0));
        let transformed = aabb.transform(&rotation);

        // Rotated AABB should be larger in X and Y (but same in Z)
        let original_size_x = aabb.max.x - aabb.min.x;
        let transformed_size_x = transformed.max.x - transformed.min.x;

        assert!(transformed_size_x > original_size_x);

        // Z size should be unchanged
        let original_size_z = aabb.max.z - aabb.min.z;
        let transformed_size_z = transformed.max.z - transformed.min.z;
        assert!((transformed_size_z - original_size_z).abs() < EPSILON);
    }

    #[test]
    fn test_aabb_transform_combined() {
        let aabb = Aabb::new(Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));

        // Combine translation, rotation, and scale
        let transform = Matrix4::from_translation(Vector3::new(10.0, 0.0, 0.0))
            * Matrix4::from_angle_z(Rad(crate::consts::PI / 4.0))
            * Matrix4::from_scale(2.0);

        let transformed = aabb.transform(&transform);

        // Should have moved and grown
        assert!(transformed.max.x > aabb.max.x);
        assert!(transformed.min.x > aabb.min.x);
    }
}
