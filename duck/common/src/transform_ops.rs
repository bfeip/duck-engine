//! Transform operations for manipulating positions, rotations, and scales.
//!
//! This module provides pure mathematical functions for common transform operations
//! like rotating around pivots, scaling about pivots, and computing local axes.
//! These functions are designed to be reusable across operators, animation systems,
//! gizmos, and other subsystems.

use crate::{
    EuclideanSpace, InnerSpace, Matrix, Matrix4, Point3, Quaternion, Rad, Rotation, Rotation3,
    SquareMatrix, Vector3,
};

use crate::EPSILON;

// =============================================================================
// Matrix Operations
// =============================================================================

/// Transforms a point by a 4x4 matrix (applies translation, rotation, and scale).
pub fn transform_point(matrix: &Matrix4, point: Point3) -> Point3 {
    Point3::from_homogeneous(matrix * point.to_homogeneous())
}

/// Transforms a surface normal by a 4x4 matrix, returning a unit vector.
///
/// Uses the inverse-transpose so normals stay perpendicular to their surface
/// under non-uniform scale; falls back to a plain direction transform when the
/// matrix is singular.
pub fn transform_normal(matrix: &Matrix4, normal: Vector3) -> Vector3 {
    let m = match matrix.invert() {
        Some(inverse) => inverse.transpose(),
        None => *matrix,
    };
    (m * normal.extend(0.0)).truncate().normalize()
}

// =============================================================================
// Pivot-Based Transforms
// =============================================================================

/// Rotates a position around a pivot point.
///
/// # Arguments
/// * `position` - The position to rotate
/// * `pivot` - The world-space pivot point
/// * `rotation` - The rotation to apply
///
/// # Returns
/// The new position after rotation around the pivot
pub fn rotate_position_about_pivot(
    position: Point3,
    pivot: Point3,
    rotation: Quaternion,
) -> Point3 {
    let offset = position - pivot;
    let rotated_offset = rotation.rotate_vector(offset);
    pivot + rotated_offset
}

/// Composes a rotation by applying a new rotation to an existing orientation.
///
/// Applies a world-space rotation to an existing orientation (rotation * current).
///
/// # Arguments
/// * `current_rotation` - The current orientation
/// * `rotation` - The rotation to apply (in world space)
///
/// # Returns
/// The new orientation
pub fn compose_rotation(
    current_rotation: Quaternion,
    rotation: Quaternion,
) -> Quaternion {
    rotation * current_rotation
}

/// Scales a position offset from a pivot point in world space.
///
/// # Arguments
/// * `position` - The position to scale
/// * `pivot` - The world-space pivot point
/// * `scale` - The scale factors (x, y, z)
///
/// # Returns
/// The new position after scaling about the pivot
pub fn scale_position_about_pivot_world(
    position: Point3,
    pivot: Point3,
    scale: Vector3,
) -> Point3 {
    let offset = position - pivot;
    let scaled_offset = Vector3::new(offset.x * scale.x, offset.y * scale.y, offset.z * scale.z);
    pivot + scaled_offset
}

/// Scales a position offset from a pivot point in local space.
///
/// The local space is defined by a reference orientation (typically the
/// primary selected object's rotation in multi-object operations).
///
/// # Arguments
/// * `position` - The position to scale
/// * `pivot` - The world-space pivot point
/// * `scale` - The scale factors in local space (x, y, z)
/// * `local_orientation` - The orientation defining local space
///
/// # Returns
/// The new position after scaling about the pivot in local space
pub fn scale_position_about_pivot_local(
    position: Point3,
    pivot: Point3,
    scale: Vector3,
    local_orientation: Quaternion,
) -> Point3 {
    let offset = position - pivot;
    // Transform offset to local space
    let local_offset = local_orientation.conjugate().rotate_vector(offset);
    // Apply scale in local space
    let scaled_local = Vector3::new(
        local_offset.x * scale.x,
        local_offset.y * scale.y,
        local_offset.z * scale.z,
    );
    // Transform back to world space
    let world_offset = local_orientation.rotate_vector(scaled_local);
    pivot + world_offset
}

/// Applies component-wise scale to an existing scale vector.
///
/// # Arguments
/// * `current_scale` - The current scale
/// * `scale_factor` - The scale factors to apply (multiplied component-wise)
///
/// # Returns
/// The new scale
pub fn apply_scale(current_scale: Vector3, scale_factor: Vector3) -> Vector3 {
    Vector3::new(
        current_scale.x * scale_factor.x,
        current_scale.y * scale_factor.y,
        current_scale.z * scale_factor.z,
    )
}

// =============================================================================
// Axis Computation
// =============================================================================

/// Computes the local X axis (right) for a given orientation.
///
/// # Arguments
/// * `rotation` - The orientation quaternion
///
/// # Returns
/// A unit vector representing the local X axis in world space
pub fn local_axis_x(rotation: Quaternion) -> Vector3 {
    rotation.rotate_vector(Vector3::unit_x())
}

/// Computes the local Y axis (up) for a given orientation.
///
/// # Arguments
/// * `rotation` - The orientation quaternion
///
/// # Returns
/// A unit vector representing the local Y axis in world space
pub fn local_axis_y(rotation: Quaternion) -> Vector3 {
    rotation.rotate_vector(Vector3::unit_y())
}

/// Computes the local Z axis (forward) for a given orientation.
///
/// # Arguments
/// * `rotation` - The orientation quaternion
///
/// # Returns
/// A unit vector representing the local Z axis in world space
pub fn local_axis_z(rotation: Quaternion) -> Vector3 {
    rotation.rotate_vector(Vector3::unit_z())
}

/// Computes all three local axes for a given orientation.
///
/// # Arguments
/// * `rotation` - The orientation quaternion
///
/// # Returns
/// A tuple of (right, up, forward) unit vectors in world space
pub fn local_axes(rotation: Quaternion) -> (Vector3, Vector3, Vector3) {
    (
        local_axis_x(rotation),
        local_axis_y(rotation),
        local_axis_z(rotation),
    )
}

// =============================================================================
// Quaternion Utilities
// =============================================================================

/// Creates a rotation quaternion from an axis and angle, with safety for zero-length axes.
///
/// If the axis has near-zero magnitude, returns an identity quaternion.
///
/// # Arguments
/// * `axis` - The axis to rotate around (does not need to be normalized)
/// * `angle` - The rotation angle in radians
///
/// # Returns
/// A rotation quaternion, or identity if the axis is degenerate
pub fn quaternion_from_axis_angle_safe(axis: Vector3, angle: f32) -> Quaternion {
    if axis.magnitude2() > EPSILON {
        Quaternion::from_axis_angle(axis.normalize(), Rad(angle))
    } else {
        Quaternion::new(1.0, 0.0, 0.0, 0.0)
    }
}

// =============================================================================
// Point Set Operations
// =============================================================================

/// Computes the centroid (average position) of a set of points.
///
/// # Arguments
/// * `points` - An iterator of points
///
/// # Returns
/// The centroid, or None if the iterator is empty
pub fn centroid<'a>(points: impl Iterator<Item = &'a Point3>) -> Option<Point3> {
    let mut sum = Vector3::new(0.0, 0.0, 0.0);
    let mut count = 0;

    for point in points {
        sum += point.to_vec();
        count += 1;
    }

    if count == 0 {
        None
    } else {
        Some(Point3::from_vec(sum / count as f32))
    }
}

/// Computes the centroid from a slice of points.
///
/// # Arguments
/// * `points` - A slice of points
///
/// # Returns
/// The centroid, or None if the slice is empty
pub fn centroid_of_slice(points: &[Point3]) -> Option<Point3> {
    centroid(points.iter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Deg, Rotation3};

    const TEST_EPSILON: f32 = 1e-5;

    // ===== transform_normal Tests =====

    #[test]
    fn test_transform_normal_rotation() {
        let matrix = Matrix4::from(Quaternion::from_angle_z(Deg(90.0)));
        let result = transform_normal(&matrix, Vector3::unit_x());

        assert!((result.x - 0.0).abs() < TEST_EPSILON);
        assert!((result.y - 1.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_transform_normal_non_uniform_scale() {
        // A surface with tangent (1,-1,0) and normal (1,1,0)/√2, scaled by
        // (2,1,1): the tangent becomes (2,-1,0), so the correct normal is
        // proportional to (1,2,0), not the naively-transformed (2,1,0).
        let matrix = Matrix4::from_nonuniform_scale(2.0, 1.0, 1.0);
        let n = Vector3::new(1.0, 1.0, 0.0).normalize();

        let result = transform_normal(&matrix, n);
        let expected = Vector3::new(1.0, 2.0, 0.0).normalize();

        assert!((result.x - expected.x).abs() < TEST_EPSILON);
        assert!((result.y - expected.y).abs() < TEST_EPSILON);
        assert!((result.z - expected.z).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_transform_normal_result_is_unit_length() {
        let matrix = Matrix4::from_scale(3.0);
        let result = transform_normal(&matrix, Vector3::new(0.0, 1.0, 0.0));

        assert!((result.magnitude() - 1.0).abs() < TEST_EPSILON);
    }

    // ===== rotate_position_about_pivot Tests =====

    #[test]
    fn test_rotate_position_about_pivot_identity() {
        let position = Point3::new(1.0, 0.0, 0.0);
        let pivot = Point3::origin();
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0); // identity

        let result = rotate_position_about_pivot(position, pivot, rotation);

        assert!((result.x - 1.0).abs() < TEST_EPSILON);
        assert!((result.y - 0.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_rotate_position_about_pivot_90_degrees_z() {
        let position = Point3::new(1.0, 0.0, 0.0);
        let pivot = Point3::origin();
        let rotation = Quaternion::from_angle_z(Deg(90.0));

        let result = rotate_position_about_pivot(position, pivot, rotation);

        // (1,0,0) rotated 90 degrees around Z should become (0,1,0)
        assert!((result.x - 0.0).abs() < TEST_EPSILON);
        assert!((result.y - 1.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_rotate_position_about_offset_pivot() {
        let position = Point3::new(2.0, 0.0, 0.0);
        let pivot = Point3::new(1.0, 0.0, 0.0);
        let rotation = Quaternion::from_angle_z(Deg(90.0));

        let result = rotate_position_about_pivot(position, pivot, rotation);

        // Offset (1,0,0) rotated 90 degrees around Z becomes (0,1,0), plus pivot
        assert!((result.x - 1.0).abs() < TEST_EPSILON);
        assert!((result.y - 1.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    // ===== compose_rotation Tests =====

    #[test]
    fn test_compose_rotation_identity() {
        let current = Quaternion::from_angle_y(Deg(45.0));
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0); // identity

        let result = compose_rotation(current, rotation);

        assert!((result.s - current.s).abs() < TEST_EPSILON);
        assert!((result.v.x - current.v.x).abs() < TEST_EPSILON);
        assert!((result.v.y - current.v.y).abs() < TEST_EPSILON);
        assert!((result.v.z - current.v.z).abs() < TEST_EPSILON);
    }

    // ===== scale_position_about_pivot_world Tests =====

    #[test]
    fn test_scale_position_about_pivot_world_uniform() {
        let position = Point3::new(2.0, 0.0, 0.0);
        let pivot = Point3::origin();
        let scale = Vector3::new(2.0, 2.0, 2.0);

        let result = scale_position_about_pivot_world(position, pivot, scale);

        assert!((result.x - 4.0).abs() < TEST_EPSILON);
        assert!((result.y - 0.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_scale_position_about_offset_pivot() {
        let position = Point3::new(3.0, 0.0, 0.0);
        let pivot = Point3::new(1.0, 0.0, 0.0);
        let scale = Vector3::new(2.0, 1.0, 1.0);

        let result = scale_position_about_pivot_world(position, pivot, scale);

        // Offset is (2,0,0), scaled X gives (4,0,0), plus pivot (1,0,0) = (5,0,0)
        assert!((result.x - 5.0).abs() < TEST_EPSILON);
    }

    // ===== scale_position_about_pivot_local Tests =====

    #[test]
    fn test_scale_position_local_identity_orientation() {
        // With identity orientation, local scale should equal world scale
        let position = Point3::new(2.0, 0.0, 0.0);
        let pivot = Point3::origin();
        let scale = Vector3::new(2.0, 1.0, 1.0);
        let orientation = Quaternion::new(1.0, 0.0, 0.0, 0.0);

        let result = scale_position_about_pivot_local(position, pivot, scale, orientation);

        assert!((result.x - 4.0).abs() < TEST_EPSILON);
        assert!((result.y - 0.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_scale_position_local_with_rotation() {
        // With 90deg Z rotation, local X becomes world Y
        let position = Point3::new(0.0, 2.0, 0.0);
        let pivot = Point3::origin();
        let scale = Vector3::new(2.0, 1.0, 1.0); // Scale local X by 2
        let orientation = Quaternion::from_angle_z(Deg(90.0));

        let result = scale_position_about_pivot_local(position, pivot, scale, orientation);

        // Position (0,2,0) in world space.
        // Local coords: conjugate of 90deg Z rotation transforms (0,2,0) to (2,0,0).
        // Scale local X by 2: (4,0,0)
        // Transform back to world: (0,4,0)
        assert!((result.x - 0.0).abs() < TEST_EPSILON);
        assert!((result.y - 4.0).abs() < TEST_EPSILON);
        assert!((result.z - 0.0).abs() < TEST_EPSILON);
    }

    // ===== apply_scale Tests =====

    #[test]
    fn test_apply_scale() {
        let current = Vector3::new(2.0, 3.0, 4.0);
        let factor = Vector3::new(2.0, 0.5, 1.0);

        let result = apply_scale(current, factor);

        assert!((result.x - 4.0).abs() < TEST_EPSILON);
        assert!((result.y - 1.5).abs() < TEST_EPSILON);
        assert!((result.z - 4.0).abs() < TEST_EPSILON);
    }

    // ===== local_axes Tests =====

    #[test]
    fn test_local_axes_identity() {
        let rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        let (right, up, forward) = local_axes(rotation);

        assert!((right.x - 1.0).abs() < TEST_EPSILON);
        assert!((up.y - 1.0).abs() < TEST_EPSILON);
        assert!((forward.z - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_local_axes_90_deg_y_rotation() {
        let rotation = Quaternion::from_angle_y(Deg(90.0));
        let (right, up, forward) = local_axes(rotation);

        // After 90deg Y rotation: X -> -Z, Y -> Y, Z -> X
        assert!((right.z - (-1.0)).abs() < TEST_EPSILON);
        assert!((up.y - 1.0).abs() < TEST_EPSILON);
        assert!((forward.x - 1.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_local_axis_x() {
        let rotation = Quaternion::from_angle_z(Deg(90.0));
        let axis = local_axis_x(rotation);

        // After 90deg Z rotation: X -> Y
        assert!((axis.x - 0.0).abs() < TEST_EPSILON);
        assert!((axis.y - 1.0).abs() < TEST_EPSILON);
        assert!((axis.z - 0.0).abs() < TEST_EPSILON);
    }

    // ===== quaternion_from_axis_angle_safe Tests =====

    #[test]
    fn test_quaternion_safe_normal_axis() {
        let axis = Vector3::new(0.0, 1.0, 0.0);
        let angle = std::f32::consts::FRAC_PI_2;

        let result = quaternion_from_axis_angle_safe(axis, angle);
        let expected = Quaternion::from_angle_y(Deg(90.0));

        assert!((result.s - expected.s).abs() < TEST_EPSILON);
        assert!((result.v.x - expected.v.x).abs() < TEST_EPSILON);
        assert!((result.v.y - expected.v.y).abs() < TEST_EPSILON);
        assert!((result.v.z - expected.v.z).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_quaternion_safe_zero_axis() {
        let axis = Vector3::new(0.0, 0.0, 0.0);
        let angle = 1.0;

        let result = quaternion_from_axis_angle_safe(axis, angle);

        // Should return identity
        assert!((result.s - 1.0).abs() < TEST_EPSILON);
        assert!((result.v.x - 0.0).abs() < TEST_EPSILON);
        assert!((result.v.y - 0.0).abs() < TEST_EPSILON);
        assert!((result.v.z - 0.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_quaternion_safe_unnormalized_axis() {
        let axis = Vector3::new(0.0, 2.0, 0.0); // Not normalized
        let angle = std::f32::consts::FRAC_PI_2;

        let result = quaternion_from_axis_angle_safe(axis, angle);
        let expected = Quaternion::from_angle_y(Deg(90.0));

        // Should normalize and produce same result
        assert!((result.s - expected.s).abs() < TEST_EPSILON);
        assert!((result.v.y - expected.v.y).abs() < TEST_EPSILON);
    }

    // ===== centroid Tests =====

    #[test]
    fn test_centroid_single_point() {
        let points = vec![Point3::new(1.0, 2.0, 3.0)];
        let result = centroid_of_slice(&points).unwrap();

        assert!((result.x - 1.0).abs() < TEST_EPSILON);
        assert!((result.y - 2.0).abs() < TEST_EPSILON);
        assert!((result.z - 3.0).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_centroid_multiple_points() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.0, 0.0, 2.0),
        ];
        let result = centroid_of_slice(&points).unwrap();

        assert!((result.x - 0.5).abs() < TEST_EPSILON);
        assert!((result.y - 0.5).abs() < TEST_EPSILON);
        assert!((result.z - 0.5).abs() < TEST_EPSILON);
    }

    #[test]
    fn test_centroid_empty() {
        let points: Vec<Point3> = vec![];
        let result = centroid_of_slice(&points);

        assert!(result.is_none());
    }

    #[test]
    fn test_centroid_iterator() {
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(4.0, 4.0, 4.0),
        ];
        let result = centroid(points.iter()).unwrap();

        assert!((result.x - 2.0).abs() < TEST_EPSILON);
        assert!((result.y - 2.0).abs() < TEST_EPSILON);
        assert!((result.z - 2.0).abs() < TEST_EPSILON);
    }
}
