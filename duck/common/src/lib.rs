

/// Epsilon value for floating-point comparisons
pub const EPSILON: f32 = 1e-6;

mod ray;
mod aabb;
mod plane;
mod convex_polyhedron;
mod transform;
mod transform_ops;

// Re-export common types
pub use ray::{Ray, SegmentApproach};
pub use aabb::Aabb;
pub use plane::Plane;
pub use convex_polyhedron::ConvexPolyhedron;
pub use transform::Transform;
pub use transform_ops::{
    apply_scale, centroid, centroid_of_slice, compose_rotation, local_axes, local_axis_x,
    local_axis_y, local_axis_z, quaternion_from_axis_angle_safe, rotate_position_about_pivot,
    scale_position_about_pivot_local, scale_position_about_pivot_world, transform_normal,
    transform_point,
};

// Math type aliases — all cgmath generics fixed to f32
pub type Point3 = cgmath::Point3<f32>;
pub type Vector2 = cgmath::Vector2<f32>;
pub type Vector3 = cgmath::Vector3<f32>;
pub type Vector4 = cgmath::Vector4<f32>;
pub type Matrix3 = cgmath::Matrix3<f32>;
pub type Matrix4 = cgmath::Matrix4<f32>;
pub type Quaternion = cgmath::Quaternion<f32>;
pub type PerspectiveFov = cgmath::PerspectiveFov<f32>;
pub type Euler = cgmath::Euler<cgmath::Rad<f32>>;

// Math trait re-exports (must be in scope to call methods like .normalize(), .dot(), .invert())
pub use cgmath::{InnerSpace, EuclideanSpace, MetricSpace, SquareMatrix, Matrix, Rotation, Rotation3, Angle, Zero, One};

// Angle types: re-exported generically so constructors like Deg(90.0) and Rad(PI) work
pub use cgmath::{Rad, Deg};

// Math free function re-exports
pub use cgmath::{point3, vec2, vec3, vec4, ortho, perspective};

/// An RGBA color, with values between 0.0 and 1.0
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RgbaColor {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl RgbaColor {
    pub const WHITE: Self = Self { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const BLACK: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const RED: Self = Self { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    pub const GREEN: Self = Self { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const BLUE: Self = Self { r: 0.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const MAGENTA: Self = Self { r: 1.0, g: 0.0, b: 1.0, a: 1.0 };
    pub const CYAN: Self = Self { r: 0.0, g: 1.0, b: 1.0, a: 1.0 };
    pub const YELLOW: Self = Self { r: 1.0, g: 1.0, b: 0.0, a: 1.0 };
    pub const ORANGE: Self = Self { r: 1.0, g: 0.5, b: 0.0, a: 1.0 };
    pub const GRAY: Self = Self { r: 0.5, g: 0.5, b: 0.5, a: 1.0 };
    pub const TRANSPARENT: Self = Self { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };
}

/// Computes the normal matrix from a world transform matrix.
///
/// The normal matrix is the inverse-transpose of the upper-left 3x3 portion
/// of the transform matrix. This is necessary for correct normal transformation
/// when non-uniform scaling is present.
///
/// If the matrix is not invertible, returns an identity matrix.
pub fn compute_normal_matrix(world_transform: &Matrix4) -> Matrix3 {
    // Extract the upper-left 3x3 matrix
    let mat3 = Matrix3::from([
        [world_transform[0][0], world_transform[0][1], world_transform[0][2]],
        [world_transform[1][0], world_transform[1][1], world_transform[1][2]],
        [world_transform[2][0], world_transform[2][1], world_transform[2][2]],
    ]);

    // Compute inverse-transpose
    match mat3.invert() {
        Some(inv) => inv.transpose(),
        None => {
            // If not invertible, use identity (shouldn't happen in practice)
            Matrix3::identity()
        }
    }
}

/// Decomposes a 4x4 transformation matrix into translation, rotation, and scale components.
///
/// # Arguments
/// * `matrix` - A 4x4 transformation matrix (column-major)
///
/// # Returns
/// A [`Transform`] containing the decomposed translation, rotation, and scale.
pub fn decompose_matrix(matrix: &Matrix4) -> Transform {
    // Extract translation from the last column
    let position = Point3::new(matrix[3][0], matrix[3][1], matrix[3][2]);

    // Extract basis vectors (first three columns)
    let basis_x = Vector3::new(matrix[0][0], matrix[0][1], matrix[0][2]);
    let basis_y = Vector3::new(matrix[1][0], matrix[1][1], matrix[1][2]);
    let basis_z = Vector3::new(matrix[2][0], matrix[2][1], matrix[2][2]);

    // Extract scale from the length of each basis vector
    let scale = Vector3::new(
        basis_x.magnitude(),
        basis_y.magnitude(),
        basis_z.magnitude(),
    );

    // Normalize basis vectors to remove scale, giving us the rotation matrix
    let rot_x = basis_x / scale.x;
    let rot_y = basis_y / scale.y;
    let rot_z = basis_z / scale.z;

    // Build rotation matrix from normalized basis vectors
    let rotation_matrix = Matrix3::from([rot_x.into(), rot_y.into(), rot_z.into()]);

    // Convert rotation matrix to quaternion
    let rotation: Quaternion = rotation_matrix.into();

    Transform::new(position, rotation, scale)
}

/// Converts a [`Matrix4`] (column-major `f32`, this crate's convention) into a
/// row-major `[[f64; 4]; 4]`, the layout expected by row-major `f64` matrix APIs.
/// 
/// This is transparently a convince function opencascade-rs functionality. It
/// Probably shouldn't exist here.
pub fn matrix4_to_row_major_f64(matrix: &Matrix4) -> [[f64; 4]; 4] {
    let mut out = [[0.0f64; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = matrix[col][row] as f64;
        }
    }
    out
}

pub fn rgba_to_array(c: RgbaColor) -> [f32; 4] {
    [c.r, c.g, c.b, c.a]
}

pub fn array_to_rgba(a: [f32; 4]) -> RgbaColor {
    RgbaColor {
        r: a[0],
        g: a[1],
        b: a[2],
        a: a[3],
    }
}

pub fn point3_to_array(p: Point3) -> [f32; 3] {
    [p.x, p.y, p.z]
}

pub fn array_to_point3(a: [f32; 3]) -> Point3 {
    Point3::new(a[0], a[1], a[2])
}

pub fn vec3_to_array(v: Vector3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

pub fn array_to_vec3(a: [f32; 3]) -> Vector3 {
    Vector3::new(a[0], a[1], a[2])
}

/// Computes an orthonormal basis from a direction vector.
///
/// Returns `(right, up)` vectors perpendicular to `direction` and to each other.
/// The input direction does not need to be normalized (it will be internally).
///
/// # Panics
/// Panics if `direction` is a zero vector.
pub fn orthonormal_basis(direction: Vector3) -> (Vector3, Vector3) {
    let dir = direction.normalize();
    let up_hint = if dir.y.abs() > 0.99 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let right = dir.cross(up_hint).normalize();
    let up = right.cross(dir).normalize();
    (right, up)
}

/// One of the three principal coordinate axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Axis {
    /// The axis named by a key character (`'x'`/`'y'`/`'z'`), else `None`.
    pub fn from_char(c: char) -> Option<Axis> {
        match c {
            'x' => Some(Axis::X),
            'y' => Some(Axis::Y),
            'z' => Some(Axis::Z),
            _ => None,
        }
    }
    
    /// Unit direction vector for this axis.
    pub fn direction(&self) -> Vector3 {
        match self {
            Axis::X => Vector3::unit_x(),
            Axis::Y => Vector3::unit_y(),
            Axis::Z => Vector3::unit_z(),
        }
    }

    /// RGB color convention: X=red, Y=green, Z=blue.
    pub fn color(&self) -> RgbaColor {
        match self {
            Axis::X => RgbaColor::RED,
            Axis::Y => RgbaColor::GREEN,
            Axis::Z => RgbaColor::BLUE,
        }
    }

    /// Rotation matrix that transforms Y-up geometry to align with this axis.
    ///
    /// Useful for orienting primitives (cylinders, cones) that are built along +Y
    /// to instead point along the desired axis.
    pub fn rotation_from_y(&self) -> Matrix4 {
        match self {
            Axis::Y => Matrix4::from_scale(1.0),
            Axis::X => Matrix4::from_angle_z(Deg(-90.0)),
            Axis::Z => Matrix4::from_angle_x(Deg(90.0)),
        }
    }

    /// Quaternion that rotates Y-up geometry to align with this axis.
    pub fn quaternion_from_y(&self) -> Quaternion {
        match self {
            Axis::Y => Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Axis::X => Quaternion::from_angle_z(Deg(-90.0)),
            Axis::Z => Quaternion::from_angle_x(Deg(90.0)),
        }
    }

    /// All three axes in order.
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== compute_normal_matrix Tests =====

    #[test]
    fn test_normal_matrix_identity() {
        let identity = Matrix4::identity();
        let normal_mat = compute_normal_matrix(&identity);
        let expected = Matrix3::identity();

        // Compare each element
        for i in 0..3 {
            for j in 0..3 {
                let diff: f32 = normal_mat[i][j] - expected[i][j];
                assert!(diff.abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_normal_matrix_uniform_scale() {
        // Uniform scale by 2
        let scale = Matrix4::from_scale(2.0);
        let normal_mat = compute_normal_matrix(&scale);

        // For uniform scale s, normal matrix should be scaled by 1/s
        let expected_scale = 0.5;
        assert!((normal_mat[0][0] - expected_scale).abs() < EPSILON);
        assert!((normal_mat[1][1] - expected_scale).abs() < EPSILON);
        assert!((normal_mat[2][2] - expected_scale).abs() < EPSILON);
    }

    #[test]
    fn test_normal_matrix_non_uniform_scale() {
        // Non-uniform scale
        let scale = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let normal_mat = compute_normal_matrix(&scale);

        // Normal matrix should correct for non-uniform scaling
        // inv-transpose of diagonal should give reciprocals
        assert!((normal_mat[0][0] - 0.5).abs() < EPSILON);
        assert!((normal_mat[1][1] - (1.0/3.0)).abs() < EPSILON);
        assert!((normal_mat[2][2] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_normal_matrix_rotation() {
        // Rotation should preserve the rotation in normal matrix
        let rotation = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0));
        let normal_mat = compute_normal_matrix(&rotation);

        // For pure rotation, normal matrix should equal rotation matrix (upper 3x3)
        for i in 0..3 {
            for j in 0..3 {
                assert!((normal_mat[i][j] - rotation[i][j]).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn test_normal_matrix_combined_transforms() {
        // Translation doesn't affect normal matrix, but rotation and scale do
        // Transform: T(10,20,30) * R_y(90°) * S(2)
        let transform = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0))
            * Matrix4::from_angle_y(Rad(std::f32::consts::PI / 2.0))
            * Matrix4::from_scale(2.0);

        let normal_mat = compute_normal_matrix(&transform);

        // Normal matrix = inverse-transpose of upper 3x3
        // For this transform:
        // - Translation is ignored (doesn't affect 3x3 portion)
        // - 90° rotation around Y: X→-Z, Y→Y, Z→X
        // - Uniform scale by 2 → normal matrix scaled by 1/2 = 0.5
        //
        // Expected normal matrix (in row-major for clarity):
        //   Row 0: [ ~0    0   0.5]
        //   Row 1: [  0   0.5   0 ]
        //   Row 2: [-0.5  0   ~0 ]
        //
        // In cgmath (column-major), matrix[col][row]:

        // Check key elements that define the transformation
        assert!((normal_mat[0][0]).abs() < EPSILON);           // col 0, row 0 ≈ 0
        assert!((normal_mat[0][2] - -0.5).abs() < 0.001);      // col 0, row 2 ≈ -0.5
        assert!((normal_mat[1][1] - 0.5).abs() < EPSILON);     // col 1, row 1 ≈ 0.5
        assert!((normal_mat[2][0] - 0.5).abs() < 0.001);       // col 2, row 0 ≈ 0.5
        assert!((normal_mat[2][2]).abs() < EPSILON);           // col 2, row 2 ≈ 0

        // Verify the matrix represents correct normal transformation behavior
        // A normal pointing in +X should transform correctly under rotation + scale
        let normal_x = Vector3::new(1.0, 0.0, 0.0);
        let transformed_x = normal_mat * normal_x;
        // After 90° Y rotation and scale: X → -Z direction, scaled by 0.5
        assert!((transformed_x.x).abs() < 0.001);
        assert!((transformed_x.y).abs() < 0.001);
        assert!((transformed_x.z - -0.5).abs() < 0.001);
    }

    #[test]
    fn test_normal_matrix_non_invertible() {
        // Zero scale matrix is non-invertible
        let zero_scale = Matrix4::from_scale(0.0);
        let normal_mat = compute_normal_matrix(&zero_scale);

        // Should return identity for non-invertible matrix
        let identity = Matrix3::identity();
        for i in 0..3 {
            for j in 0..3 {
                let diff: f32 = normal_mat[i][j] - identity[i][j];
                assert!(diff.abs() < EPSILON);
            }
        }
    }

    // ===== decompose_matrix Tests =====

    #[test]
    fn test_decompose_identity() {
        let identity = Matrix4::identity();
        let t = decompose_matrix(&identity);

        assert!((t.position.x - 0.0).abs() < EPSILON);
        assert!((t.position.y - 0.0).abs() < EPSILON);
        assert!((t.position.z - 0.0).abs() < EPSILON);

        // Identity quaternion is (w=1, x=0, y=0, z=0) or (s=1, v=0)
        assert!((t.rotation.s - 1.0).abs() < EPSILON || (t.rotation.s + 1.0).abs() < EPSILON);

        assert!((t.scale.x - 1.0).abs() < EPSILON);
        assert!((t.scale.y - 1.0).abs() < EPSILON);
        assert!((t.scale.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_translation_only() {
        let translation_vec = Vector3::new(5.0, 10.0, 15.0);
        let matrix = Matrix4::from_translation(translation_vec);
        let t = decompose_matrix(&matrix);

        assert!((t.position.x - 5.0).abs() < EPSILON);
        assert!((t.position.y - 10.0).abs() < EPSILON);
        assert!((t.position.z - 15.0).abs() < EPSILON);

        assert!((t.scale.x - 1.0).abs() < EPSILON);
        assert!((t.scale.y - 1.0).abs() < EPSILON);
        assert!((t.scale.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_rotation_only() {
        let matrix = Matrix4::from_angle_y(Rad(std::f32::consts::PI / 2.0));
        let t = decompose_matrix(&matrix);

        assert!((t.position.x - 0.0).abs() < EPSILON);
        assert!((t.position.y - 0.0).abs() < EPSILON);
        assert!((t.position.z - 0.0).abs() < EPSILON);

        // Scale should be 1
        assert!((t.scale.x - 1.0).abs() < EPSILON);
        assert!((t.scale.y - 1.0).abs() < EPSILON);
        assert!((t.scale.z - 1.0).abs() < EPSILON);

        // Rotation should be 90° around Y axis
        // Quaternion for rotation around Y by θ: (cos(θ/2), 0, sin(θ/2), 0)
        // For θ = π/2: (cos(π/4), 0, sin(π/4), 0) = (√2/2, 0, √2/2, 0)
        let sqrt2_over_2 = std::f32::consts::SQRT_2 / 2.0;
        assert!((t.rotation.s - sqrt2_over_2).abs() < 0.001);  // scalar part
        assert!((t.rotation.v.x - 0.0).abs() < EPSILON);        // x component
        assert!((t.rotation.v.y - sqrt2_over_2).abs() < 0.001); // y component
        assert!((t.rotation.v.z - 0.0).abs() < EPSILON);        // z component
    }

    #[test]
    fn test_decompose_scale_only() {
        let matrix = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let t = decompose_matrix(&matrix);

        assert!((t.position.x - 0.0).abs() < EPSILON);
        assert!((t.position.y - 0.0).abs() < EPSILON);
        assert!((t.position.z - 0.0).abs() < EPSILON);

        assert!((t.scale.x - 2.0).abs() < EPSILON);
        assert!((t.scale.y - 3.0).abs() < EPSILON);
        assert!((t.scale.z - 4.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_trs_composition() {
        // Create a TRS matrix
        let t = Matrix4::from_translation(Vector3::new(1.0, 2.0, 3.0));
        let r = Matrix4::from_angle_z(Rad(std::f32::consts::PI / 4.0));
        let s = Matrix4::from_scale(2.0);
        let trs = t * r * s;

        let t = decompose_matrix(&trs);

        assert!((t.position.x - 1.0).abs() < EPSILON);
        assert!((t.position.y - 2.0).abs() < EPSILON);
        assert!((t.position.z - 3.0).abs() < EPSILON);

        assert!((t.scale.x - 2.0).abs() < EPSILON);
        assert!((t.scale.y - 2.0).abs() < EPSILON);
        assert!((t.scale.z - 2.0).abs() < EPSILON);

        // Rotation should be 45° around Z axis
        // Quaternion for rotation around Z by θ: (cos(θ/2), 0, 0, sin(θ/2))
        // For θ = π/4: (cos(π/8), 0, 0, sin(π/8))
        let half_angle = std::f32::consts::PI / 8.0;
        let expected_s = half_angle.cos();  // cos(π/8) ≈ 0.9239
        let expected_z = half_angle.sin();  // sin(π/8) ≈ 0.3827

        assert!((t.rotation.s - expected_s).abs() < 0.001);     // scalar part
        assert!((t.rotation.v.x - 0.0).abs() < EPSILON);        // x component
        assert!((t.rotation.v.y - 0.0).abs() < EPSILON);        // y component
        assert!((t.rotation.v.z - expected_z).abs() < 0.001);   // z component
    }

    #[test]
    fn test_decompose_negative_scale() {
        let matrix = Matrix4::from_nonuniform_scale(-1.0, 2.0, 3.0);
        let t = decompose_matrix(&matrix);

        // Negative scale should be preserved in magnitude
        assert!((t.scale.x.abs() - 1.0).abs() < EPSILON);
        assert!((t.scale.y - 2.0).abs() < EPSILON);
        assert!((t.scale.z - 3.0).abs() < EPSILON);
    }

    // ===== orthonormal_basis Tests =====

    #[test]
    fn test_orthonormal_basis_z_axis() {
        let dir = Vector3::new(0.0, 0.0, 1.0);
        let (right, up) = orthonormal_basis(dir);

        // right and up should be perpendicular to dir
        assert!(right.dot(dir).abs() < EPSILON);
        assert!(up.dot(dir).abs() < EPSILON);
        // right and up should be perpendicular to each other
        assert!(right.dot(up).abs() < EPSILON);
        // all should be unit vectors
        assert!((right.magnitude() - 1.0).abs() < EPSILON);
        assert!((up.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_orthonormal_basis_y_axis() {
        // Tests the near-Y edge case where we switch the hint vector
        let dir = Vector3::new(0.0, 1.0, 0.0);
        let (right, up) = orthonormal_basis(dir);

        assert!(right.dot(dir).abs() < EPSILON);
        assert!(up.dot(dir).abs() < EPSILON);
        assert!(right.dot(up).abs() < EPSILON);
        assert!((right.magnitude() - 1.0).abs() < EPSILON);
        assert!((up.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_orthonormal_basis_negative_y() {
        let dir = Vector3::new(0.0, -1.0, 0.0);
        let (right, up) = orthonormal_basis(dir);

        assert!(right.dot(dir).abs() < EPSILON);
        assert!(up.dot(dir).abs() < EPSILON);
        assert!(right.dot(up).abs() < EPSILON);
    }

    #[test]
    fn test_orthonormal_basis_arbitrary_direction() {
        let dir = Vector3::new(1.0, 2.0, 3.0);
        let (right, up) = orthonormal_basis(dir);
        let dir_n = dir.normalize();

        assert!(right.dot(dir_n).abs() < EPSILON);
        assert!(up.dot(dir_n).abs() < EPSILON);
        assert!(right.dot(up).abs() < EPSILON);
        assert!((right.magnitude() - 1.0).abs() < EPSILON);
        assert!((up.magnitude() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_decompose_recompose_identity() {
        // Decompose and recompose should give us back the original (approximately)
        let original = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0))
            * Matrix4::from_scale(2.0);

        let t = decompose_matrix(&original);

        // Recompose: T * R * S
        let recomposed = t.to_matrix();

        // Compare matrices (they should be very close)
        for i in 0..4 {
            for j in 0..4 {
                let diff: f32 = original[i][j] - recomposed[i][j];
                assert!(diff.abs() < 0.001);
            }
        }
    }

}
