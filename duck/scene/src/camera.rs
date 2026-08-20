use duck_engine_common::{
    Deg, InnerSpace, Matrix3, Matrix4, MetricSpace, Point3, Quaternion, SquareMatrix, Vector3,
    ortho, perspective,
};

/// Matrix to convert from OpenGL clip-space depth [-1, 1] to WGPU depth [0, 1].
///
/// WGPU uses a different depth convention than OpenGL:
/// - OpenGL NDC depth: [-1, 1] (near to far)
/// - WGPU NDC depth: [0, 1] (near to far)
///
/// This matrix remaps Z: `z' = 0.5 * z + 0.5`
#[rustfmt::skip]
pub(crate) const OPENGL_TO_WGPU_MATRIX: Matrix4 = Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

/// A fully-positioned camera combining projection intrinsics with world-space pose.
///
/// # Example
///
/// ```
/// use duck_engine_scene::PositionedCamera;
/// use duck_engine_scene::common::{Point3, Vector3};
///
/// let camera = PositionedCamera {
///     eye: Point3::new(0.0, 0.0, 5.0),
///     target: Point3::new(0.0, 0.0, 0.0),
///     up: Vector3::new(0.0, 1.0, 0.0),
///     aspect: 16.0 / 9.0,
///     fovy: 45.0,
///     znear: 0.1,
///     zfar: 100.0,
///     ortho: false,
/// };
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PositionedCamera {
    /// The position of the camera in world space.
    pub eye: Point3,
    /// The point the camera is looking at in world space.
    pub target: Point3,
    /// The up direction vector.
    pub up: Vector3,
    /// The aspect ratio of the viewport (width / height).
    pub aspect: f32,
    /// Vertical field of view in degrees (used for perspective projection).
    pub fovy: f32,
    /// Distance to the near clipping plane.
    pub znear: f32,
    /// Distance to the far clipping plane.
    pub zfar: f32,
    /// When true, use orthographic projection instead of perspective.
    ///
    /// The orthographic view size is derived from the camera distance and fovy,
    /// so zoom (changing distance) works naturally for both projection modes.
    pub ortho: bool,
}

impl PositionedCamera {
    /// Builds the combined view-projection matrix for this camera.
    ///
    /// The resulting matrix transforms world-space coordinates to clip-space,
    /// ready for the GPU rasterizer. It combines:
    /// - View matrix: transforms world space to camera/view space
    /// - Projection matrix: transforms view space to clip space (perspective or orthographic)
    /// - Depth remapping: converts OpenGL depth convention to WGPU convention
    pub fn build_view_projection_matrix(&self) -> Matrix4 {
        let view = Matrix4::look_at_rh(self.eye, self.target, self.up);
        let proj = if self.ortho {
            // Derive orthographic bounds from camera distance and fovy.
            // This allows zoom (changing distance) to work naturally.
            let half_height = self.length() * (self.fovy.to_radians() / 2.0).tan();
            let half_width = half_height * self.aspect;
            ortho(-half_width, half_width, -half_height, half_height, self.znear, self.zfar)
        } else {
            perspective(Deg(self.fovy), self.aspect, self.znear, self.zfar)
        };

        OPENGL_TO_WGPU_MATRIX * proj * view
    }

    /// Returns the camera's forward vector
    pub fn forward(&self) -> Vector3 {
        (self.target - self.eye).normalize()
    }

    /// Returns the right vector of the camera
    pub fn right(&self) -> Vector3 {
        self.forward().cross(self.up).normalize()
    }

    /// Returns length of the camera's look vector
    /// (the distance from the camera eye to the target)
    pub fn length(&self) -> f32 {
        self.eye.distance(self.target)
    }

    /// World-space size of one pixel at the given depth from the camera.
    ///
    /// For perspective, pixel size grows with depth. For orthographic, depth is
    /// ignored and the camera distance is used instead (pixel size is constant).
    pub fn world_size_per_pixel(&self, depth: f32, viewport_height: u32) -> f32 {
        let effective_depth = if self.ortho { self.length() } else { depth };
        let half_fovy = (self.fovy.to_radians() / 2.0).tan();
        2.0 * effective_depth * half_fovy / viewport_height as f32
    }

    /// Adjusts the camera to fit a bounding box in view.
    ///
    /// Positions the camera so the entire bounding box is visible while maintaining
    /// the current view direction (from eye towards target). The camera is moved
    /// along this direction to ensure the bounds fit within the field of view.
    pub fn fit_to_bounds(&mut self, bounds: &crate::common::Aabb) {
        let center = bounds.center();
        let (size_x, size_y, size_z) = bounds.size();

        // Compute the bounding sphere radius (half the diagonal of the AABB)
        let bounding_radius = (size_x * size_x + size_y * size_y + size_z * size_z).sqrt() / 2.0;

        // Calculate the distance needed to fit the bounding sphere in view
        // Using the vertical field of view and accounting for aspect ratio
        let half_fov_rad = (self.fovy / 2.0).to_radians();

        // Calculate distance for vertical fit
        let vertical_distance = bounding_radius / half_fov_rad.sin();

        // Calculate distance for horizontal fit (accounting for aspect ratio)
        let half_hfov_rad = (half_fov_rad.tan() * self.aspect).atan();
        let horizontal_distance = bounding_radius / half_hfov_rad.sin();

        // Use the larger distance to ensure the object fits in both dimensions
        let distance = vertical_distance.max(horizontal_distance);

        // Get current view direction (or default to -Z if eye == target)
        let view_dir = if self.eye.distance(self.target) < 1e-6 {
            Vector3::new(0.0, 0.0, -1.0)
        } else {
            (self.target - self.eye).normalize()
        };

        // Position camera at the calculated distance from the center
        self.target = center;
        self.eye = center - view_dir * distance;

        // Adjust near/far planes to encompass the scene
        // Near plane: at least 1/1000th of the distance, but not less than 0.001
        // Far plane: generous enough for significant zoom-out
        self.znear = (distance * 0.001).max(0.001);
        self.zfar = (distance + bounding_radius) * 10.0;
    }

    /// Projects a world-space point to normalized device coordinates: X and Y
    /// in [-1, 1] (Y-up), Z in [0, 1] (WGPU depth convention).
    pub fn project_point_ndc(&self, world_point: Point3) -> Point3 {
        let vp = self.build_view_projection_matrix();
        let homogeneous = vp * world_point.to_homogeneous();

        // Perform perspective division
        Point3::from_homogeneous(homogeneous)
    }

    /// Unprojects an NDC point (X/Y in [-1, 1] Y-up, Z in [0, 1]) back to
    /// world space, or `None` if the view-projection matrix is not invertible.
    pub fn unproject_point_ndc(&self, ndc_point: Point3) -> Option<Point3> {
        let viewproj = self.build_view_projection_matrix();

        let inv_vp = viewproj.invert()?;

        // Convert NDC point to homogeneous coordinates
        let homogeneous = inv_vp * ndc_point.to_homogeneous();

        // Perform perspective division
        Some(Point3::from_homogeneous(homogeneous))
    }

    /// Projects a world-space point to screen-space pixels: X in
    /// [0, screen_width] left-to-right, Y in [0, screen_height] top-to-bottom,
    /// Z the [0, 1] depth.
    pub fn project_point_screen(
        &self,
        world_point: Point3,
        screen_width: u32,
        screen_height: u32,
    ) -> Point3 {
        let ndc = self.project_point_ndc(world_point);

        // Convert NDC to screen coordinates
        // NDC: [-1, 1] × [-1, 1], Y-up
        // Screen: [0, width] × [0, height], Y-down
        let screen_x = (ndc.x + 1.0) * 0.5 * screen_width as f32;
        let screen_y = (1.0 - ndc.y) * 0.5 * screen_height as f32; // Flip Y
        let screen_z = ndc.z; // Keep depth as-is

        Point3::new(screen_x, screen_y, screen_z)
    }


    /// Unprojects a screen-space pixel (Y-down) at `depth` (0 = near plane,
    /// 1 = far plane) back to world space, or `None` if the view-projection
    /// matrix is not invertible.
    pub fn unproject_point_screen(
        &self,
        screen_x: f32,
        screen_y: f32,
        depth: f32,
        screen_width: u32,
        screen_height: u32,
    ) -> Option<Point3> {
        // Convert screen coordinates to NDC
        // Screen: [0, width] × [0, height], Y-down
        // NDC: [-1, 1] × [-1, 1], Y-up
        let ndc_x = (screen_x / screen_width as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (screen_y / screen_height as f32) * 2.0; // Flip Y
        let ndc_z = depth;

        let ndc_point = Point3::new(ndc_x, ndc_y, ndc_z);
        self.unproject_point_ndc(ndc_point)
    }

    /// The world-space ray originating at the near plane and pointing through
    /// the given NDC position (X/Y in [-1, 1], Y-up).
    pub fn ray_from_ndc_point(&self, ndc_x: f32, ndc_y: f32) -> crate::common::Ray {
        let world_near = self
            .unproject_point_ndc(Point3::new(ndc_x, ndc_y, 0.0))
            .expect("Camera view-projection matrix should be invertible");

        let world_far = self
            .unproject_point_ndc(Point3::new(ndc_x, ndc_y, 1.0))
            .expect("Camera view-projection matrix should be invertible");

        let direction = (world_far - world_near).normalize();

        crate::common::Ray::new(world_near, direction)
    }

    /// The world-space ray originating at the near plane and pointing through
    /// the given screen pixel (Y-down). The usual starting point for mouse
    /// picking; see [`geom_query`](crate::geom_query).
    pub fn ray_from_screen_point(
        &self,
        screen_x: f32,
        screen_y: f32,
        screen_width: u32,
        screen_height: u32,
    ) -> crate::common::Ray {
        // Unproject points at near and far planes
        let world_near = self.unproject_point_screen(
            screen_x,
            screen_y,
            0.0, // Near plane
            screen_width,
            screen_height,
        ).expect("Camera view-projection matrix should be invertible");

        let world_far = self.unproject_point_screen(
            screen_x,
            screen_y,
            1.0, // Far plane
            screen_width,
            screen_height,
        ).expect("Camera view-projection matrix should be invertible");

        // Direction is from near point to far point
        let direction = (world_far - world_near).normalize();

        crate::common::Ray::new(world_near, direction)
    }

    /// Pose interpolated toward `other` at `t` in [0, 1]: the orientation
    /// follows the shortest arc while the target, distance, and projection
    /// parameters blend linearly, keeping the eye on an orbit arc rather than
    /// a straight chord. `aspect` and `ortho` are not interpolable and switch
    /// from `self`'s values to `other`'s at `t >= 1`.
    pub fn interpolated(&self, other: &PositionedCamera, t: f32) -> PositionedCamera {
        fn orientation(camera: &PositionedCamera) -> Quaternion {
            let forward = (camera.target - camera.eye).normalize();
            let right = forward.cross(camera.up).normalize();
            let up = right.cross(forward);
            Quaternion::from(Matrix3::from_cols(right, up, -forward))
        }

        let from = orientation(self);
        let mut to = orientation(other);
        // Shortest arc: q and -q encode the same orientation.
        if from.dot(to) < 0.0 {
            to = -to;
        }
        let rotation = from.slerp(to, t).normalize();

        let target = self.target + (other.target - self.target) * t;
        let distance = self.length() + (other.length() - self.length()) * t;
        let forward = rotation * -Vector3::unit_z();

        let done = t >= 1.0;
        PositionedCamera {
            eye: target - forward * distance,
            target,
            up: rotation * Vector3::unit_y(),
            aspect: if done { other.aspect } else { self.aspect },
            fovy: self.fovy + (other.fovy - self.fovy) * t,
            znear: self.znear + (other.znear - self.znear) * t,
            zfar: self.zfar + (other.zfar - self.zfar) * t,
            ortho: if done { other.ortho } else { self.ortho },
        }
    }

    /// Builds a `Transform` that encodes this camera's pose (eye + orientation).
    ///
    /// Column convention: right = +X, corrected-up = +Y, forward = -Z
    /// (camera looks down -Z in a right-handed coordinate system).
    pub fn pose_transform(&self) -> crate::common::Transform {
        let forward = (self.target - self.eye).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward);
        let mat = Matrix3::from_cols(right, up, -forward);
        crate::common::Transform::new(
            self.eye,
            Quaternion::from(mat),
            Vector3::new(1.0, 1.0, 1.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::{Point3, Vector3, Vector4, InnerSpace, SquareMatrix};

    const EPSILON: f32 = 1e-6;

    fn create_test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    // ===== PositionedCamera Struct Tests =====

    #[test]
    fn test_camera_forward() {
        let camera = create_test_camera();
        let forward = camera.forward();

        assert!((forward.x - 0.0).abs() < EPSILON);
        assert!((forward.y - 0.0).abs() < EPSILON);
        assert!((forward.z - -1.0).abs() < EPSILON);

        let magnitude = forward.magnitude();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_right() {
        let camera = create_test_camera();
        let forward = camera.forward();
        let right = camera.right();

        let dot_forward = forward.dot(right);
        assert!(dot_forward.abs() < EPSILON);

        let dot_up = camera.up.dot(right);
        assert!(dot_up.abs() < EPSILON);

        assert!((right.x - 1.0).abs() < EPSILON);
        assert!((right.y - 0.0).abs() < EPSILON);
        assert!((right.z - 0.0).abs() < EPSILON);

        let magnitude = right.magnitude();
        assert!((magnitude - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_length() {
        let camera = PositionedCamera {
            eye: Point3::new(3.0, 4.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };

        let length = camera.length();
        assert!((length - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_camera_length_zero() {
        let camera = PositionedCamera {
            eye: Point3::new(1.0, 2.0, 3.0),
            target: Point3::new(1.0, 2.0, 3.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };

        let length = camera.length();
        assert!(length.abs() < EPSILON);
    }

    #[test]
    fn test_build_view_projection_identity() {
        let camera = PositionedCamera {
            eye: Point3::new(0.0, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        };

        let vp = camera.build_view_projection_matrix();

        for i in 0..4 {
            for j in 0..4 {
                assert!(vp[i][j].is_finite());
            }
        }

        let det = vp.determinant();
        assert!(det.abs() > EPSILON);
    }

    #[test]
    fn test_build_view_projection_aspect_ratio() {
        let mut camera1 = create_test_camera();
        camera1.aspect = 16.0 / 9.0;

        let mut camera2 = create_test_camera();
        camera2.aspect = 4.0 / 3.0;

        let vp1 = camera1.build_view_projection_matrix();
        let vp2 = camera2.build_view_projection_matrix();

        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (vp1[i][j] - vp2[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "Aspect ratio should affect the view-projection matrix");
    }

    #[test]
    fn test_build_view_projection_fov() {
        let mut camera1 = create_test_camera();
        camera1.fovy = 45.0;

        let mut camera2 = create_test_camera();
        camera2.fovy = 90.0;

        let vp1 = camera1.build_view_projection_matrix();
        let vp2 = camera2.build_view_projection_matrix();

        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (vp1[i][j] - vp2[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "FOV should affect the view-projection matrix");
    }

    // ===== OpenGL-to-WGPU Matrix Tests =====

    #[test]
    fn test_depth_remapping() {

        let m = OPENGL_TO_WGPU_MATRIX;

        let near_clip = Vector4::new(0.0, 0.0, -1.0, 1.0);
        let near_result = m * near_clip;

        assert!((near_result.z - 0.0).abs() < EPSILON);
        assert!((near_result.w - 1.0).abs() < EPSILON);

        let z_ndc = near_result.z / near_result.w;
        assert!((z_ndc - 0.0).abs() < EPSILON);

        let far_clip = Vector4::new(0.0, 0.0, 1.0, 1.0);
        let far_result = m * far_clip;
        assert!((far_result.z - 1.0).abs() < EPSILON);
        assert!((far_result.w - 1.0).abs() < EPSILON);

        let z_ndc_far = far_result.z / far_result.w;
        assert!((z_ndc_far - 1.0).abs() < EPSILON);

        let mid_clip = Vector4::new(0.0, 0.0, 0.0, 1.0);
        let mid_result = m * mid_clip;
        assert!((mid_result.z - 0.5).abs() < EPSILON);
        assert!((mid_result.w - 1.0).abs() < EPSILON);

        let z_ndc_mid = mid_result.z / mid_result.w;
        assert!((z_ndc_mid - 0.5).abs() < EPSILON);

        let test_point = Vector4::new(3.5, -2.7, 0.0, 1.0);
        let transformed = m * test_point;
        assert!((transformed.x - 3.5).abs() < EPSILON);
        assert!((transformed.y - -2.7).abs() < EPSILON);
    }

    // ===== Projection/Unprojection Tests =====

    #[test]
    fn test_project_unproject_ndc_roundtrip() {
        let camera = create_test_camera();

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(2.5, 1.5, -3.0),
        ];

        for original_point in test_points {
            let ndc = camera.project_point_ndc(original_point);
            let unprojected = camera.unproject_point_ndc(ndc)
                .expect("Failed to unproject point");

            assert!(
                (unprojected.x - original_point.x).abs() < 1e-4,
                "X mismatch: original={}, unprojected={}", original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-4,
                "Y mismatch: original={}, unprojected={}", original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-4,
                "Z mismatch: original={}, unprojected={}", original_point.z, unprojected.z
            );
        }
    }

    #[test]
    fn test_project_camera_target_to_ndc_center() {
        let camera = create_test_camera();
        let ndc = camera.project_point_ndc(camera.target);

        assert!(ndc.x.abs() < 1e-4, "Target should project to NDC center X, got {}", ndc.x);
        assert!(ndc.y.abs() < 1e-4, "Target should project to NDC center Y, got {}", ndc.y);
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "NDC Z should be in [0, 1], got {}", ndc.z);
    }

    #[test]
    fn test_project_ndc_bounds() {
        let camera = create_test_camera();
        let point_in_front = Point3::new(0.0, 0.0, 2.0);
        let ndc = camera.project_point_ndc(point_in_front);
        assert!(ndc.z >= 0.0 && ndc.z <= 1.0, "Point in frustum should have NDC Z in [0, 1]");
    }

    #[test]
    fn test_project_ndc_depth_ordering() {
        let camera = create_test_camera();
        let point_near = Point3::new(0.0, 0.0, 1.0);
        let point_far = Point3::new(0.0, 0.0, -2.0);

        let ndc_near = camera.project_point_ndc(point_near);
        let ndc_far = camera.project_point_ndc(point_far);

        assert!(
            ndc_near.z < ndc_far.z,
            "Closer point should have smaller NDC Z: near={}, far={}",
            ndc_near.z, ndc_far.z
        );
    }

    #[test]
    fn test_project_unproject_screen_roundtrip() {
        let camera = create_test_camera();
        let screen_width = 1920;
        let screen_height = 1080;

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(-0.5, 2.0, -1.0),
        ];

        for original_point in test_points {
            let screen = camera.project_point_screen(original_point, screen_width, screen_height);
            let unprojected = camera.unproject_point_screen(
                screen.x, screen.y, screen.z, screen_width, screen_height
            ).expect("Failed to unproject screen point");

            assert!(
                (unprojected.x - original_point.x).abs() < 1e-3,
                "Screen roundtrip X mismatch: original={}, unprojected={}",
                original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-3,
                "Screen roundtrip Y mismatch: original={}, unprojected={}",
                original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-3,
                "Screen roundtrip Z mismatch: original={}, unprojected={}",
                original_point.z, unprojected.z
            );
        }
    }

    // ===== Orthographic Tests =====

    fn create_ortho_test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            aspect: 16.0 / 9.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: true,
        }
    }

    #[test]
    fn test_ortho_project_unproject_ndc_roundtrip() {
        let camera = create_ortho_test_camera();

        let test_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, -1.0),
            Point3::new(2.5, 1.5, -3.0),
        ];

        for original_point in test_points {
            let ndc = camera.project_point_ndc(original_point);
            let unprojected = camera.unproject_point_ndc(ndc)
                .expect("Failed to unproject ortho point");

            assert!(
                (unprojected.x - original_point.x).abs() < 1e-4,
                "Ortho X mismatch: original={}, unprojected={}", original_point.x, unprojected.x
            );
            assert!(
                (unprojected.y - original_point.y).abs() < 1e-4,
                "Ortho Y mismatch: original={}, unprojected={}", original_point.y, unprojected.y
            );
            assert!(
                (unprojected.z - original_point.z).abs() < 1e-4,
                "Ortho Z mismatch: original={}, unprojected={}", original_point.z, unprojected.z
            );
        }
    }

    #[test]
    fn test_ortho_no_perspective_distortion() {
        let camera = create_ortho_test_camera();

        let point_near = Point3::new(1.0, 1.0, 2.0);
        let point_far = Point3::new(1.0, 1.0, -5.0);

        let ndc_near = camera.project_point_ndc(point_near);
        let ndc_far = camera.project_point_ndc(point_far);

        assert!(
            (ndc_near.x - ndc_far.x).abs() < 1e-5,
            "Ortho X should be same at different depths: near={}, far={}",
            ndc_near.x, ndc_far.x
        );
        assert!(
            (ndc_near.y - ndc_far.y).abs() < 1e-5,
            "Ortho Y should be same at different depths: near={}, far={}",
            ndc_near.y, ndc_far.y
        );
    }

    #[test]
    fn test_ortho_vs_perspective_different_matrices() {
        let mut camera = create_test_camera();
        let persp_vp = camera.build_view_projection_matrix();
        camera.ortho = true;
        let ortho_vp = camera.build_view_projection_matrix();

        let mut found_difference = false;
        for i in 0..4 {
            for j in 0..4 {
                if (persp_vp[i][j] - ortho_vp[i][j]).abs() > EPSILON {
                    found_difference = true;
                    break;
                }
            }
        }
        assert!(found_difference, "Perspective and orthographic matrices should differ");
    }

    // ===== Interpolation tests =====

    fn axis_view_camera(eye: Point3, up: Vector3) -> PositionedCamera {
        PositionedCamera {
            eye,
            target: Point3::new(0.0, 0.0, 0.0),
            up,
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    fn assert_cameras_close(a: &PositionedCamera, b: &PositionedCamera) {
        assert!(a.eye.distance(b.eye) < 1e-4, "eye {:?} vs {:?}", a.eye, b.eye);
        assert!(a.target.distance(b.target) < 1e-4, "target {:?} vs {:?}", a.target, b.target);
        assert!((a.up - b.up).magnitude() < 1e-4, "up {:?} vs {:?}", a.up, b.up);
    }

    #[test]
    fn interpolated_reproduces_endpoints() {
        let a = axis_view_camera(Point3::new(5.0, 0.0, 0.0), Vector3::unit_y());
        let b = axis_view_camera(Point3::new(0.0, 0.0, 8.0), Vector3::unit_y());

        assert_cameras_close(&a.interpolated(&b, 0.0), &a);
        assert_cameras_close(&a.interpolated(&b, 1.0), &b);
        assert_eq!(a.interpolated(&b, 0.5).ortho, a.ortho);
    }

    #[test]
    fn interpolated_midpoint_bisects_arc() {
        let a = axis_view_camera(Point3::new(5.0, 0.0, 0.0), Vector3::unit_y());
        let b = axis_view_camera(Point3::new(0.0, 0.0, 5.0), Vector3::unit_y());

        let mid = a.interpolated(&b, 0.5);
        assert!((mid.length() - 5.0).abs() < 1e-4, "distance preserved on the arc");
        let expected_dir = Vector3::new(1.0, 0.0, 1.0).normalize();
        let dir = (mid.eye - mid.target).normalize();
        assert!((dir - expected_dir).magnitude() < 1e-4, "eye direction bisects: {:?}", dir);
        assert!((mid.up - Vector3::unit_y()).magnitude() < 1e-4);
    }

    #[test]
    fn interpolated_blends_distance() {
        let a = axis_view_camera(Point3::new(4.0, 0.0, 0.0), Vector3::unit_y());
        let b = axis_view_camera(Point3::new(0.0, 0.0, 10.0), Vector3::unit_y());

        let mid = a.interpolated(&b, 0.5);
        assert!((mid.length() - 7.0).abs() < 1e-4);
    }

    #[test]
    fn interpolated_antipodal_views_are_finite() {
        // +X view to -X view: a 180-degree turn about the shared up axis.
        let a = axis_view_camera(Point3::new(5.0, 0.0, 0.0), Vector3::unit_y());
        let b = axis_view_camera(Point3::new(-5.0, 0.0, 0.0), Vector3::unit_y());

        for i in 0..=10 {
            let t = i as f32 / 10.0;
            let cam = a.interpolated(&b, t);
            assert!(cam.eye.x.is_finite() && cam.eye.y.is_finite() && cam.eye.z.is_finite());
            assert!((cam.length() - 5.0).abs() < 1e-3, "distance held at t={t}");
            assert!((cam.up - Vector3::unit_y()).magnitude() < 1e-3, "up held at t={t}");
        }
    }

    #[test]
    fn interpolated_up_stays_orthonormal() {
        let a = axis_view_camera(Point3::new(0.0, 6.0, 0.0), Vector3::unit_z());
        let b = axis_view_camera(Point3::new(4.0, 3.0, 4.0), Vector3::unit_y());

        for i in 1..10 {
            let t = i as f32 / 10.0;
            let cam = a.interpolated(&b, t);
            assert!((cam.up.magnitude() - 1.0).abs() < 1e-4, "unit up at t={t}");
            assert!(cam.up.dot(cam.forward()).abs() < 1e-4, "up orthogonal to forward at t={t}");
        }
    }

    // ===== Pose transform test =====

    #[test]
    fn pose_transform_encodes_eye_and_orientation() {
        let camera = create_test_camera();
        let mat = camera.pose_transform().to_matrix();

        // Column 3 is the eye position.
        assert!((mat[3][0] - camera.eye.x).abs() < 1e-4);
        assert!((mat[3][1] - camera.eye.y).abs() < 1e-4);
        assert!((mat[3][2] - camera.eye.z).abs() < 1e-4);

        // Column 2 is -forward (camera looks down -Z).
        let forward = (camera.target - camera.eye).normalize();
        assert!((mat[2][0] + forward.x).abs() < 1e-4);
        assert!((mat[2][1] + forward.y).abs() < 1e-4);
        assert!((mat[2][2] + forward.z).abs() < 1e-4);
    }
}
