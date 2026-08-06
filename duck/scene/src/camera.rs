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

/// Projection-only camera parameters, stored in a scene node's payload.
///
/// Describes how the camera sees (its optics) without encoding where it is.
/// Pose (position, orientation) lives in the owning node's `Transform`.
///
/// `focus_distance` is the eye-to-target distance used when constructing a
/// [`PositionedCamera`]. It is not a rendering parameter — it preserves the
/// orbit pivot distance across node-transform round-trips so that interactive
/// navigation (zoom, pan) remains stable after scene reload.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CameraProjection {
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
    /// Distance from the camera eye to its look-at target.
    ///
    /// Used when reconstructing a [`PositionedCamera`] from this projection and
    /// a world transform — target = eye + forward * focus_distance.
    pub focus_distance: f32,
}

impl CameraProjection {
    /// Builds a [`PositionedCamera`] by combining this projection with a world
    /// transform matrix and a viewport aspect ratio.
    ///
    /// The world transform is expected to follow the convention used by
    /// [`PositionedCamera::to_node_transform`]: column 0 = right, column 1 = up,
    /// column 2 = -forward (camera looks down -Z), column 3 = eye position.
    pub fn into_positioned(self, world_transform: Matrix4, aspect: f32) -> PositionedCamera {
        let eye = Point3::new(
            world_transform[3][0],
            world_transform[3][1],
            world_transform[3][2],
        );
        let up = Vector3::new(
            world_transform[1][0],
            world_transform[1][1],
            world_transform[1][2],
        ).normalize();
        // Column 2 stores -forward (camera looks down -Z in right-handed coords).
        let forward = -Vector3::new(
            world_transform[2][0],
            world_transform[2][1],
            world_transform[2][2],
        ).normalize();
        let distance = self.focus_distance.max(0.001);
        let target = eye + forward * distance;
        PositionedCamera {
            eye,
            target,
            up,
            aspect,
            fovy: self.fovy,
            znear: self.znear,
            zfar: self.zfar,
            ortho: self.ortho,
        }
    }
}

/// A fully-positioned camera combining projection intrinsics with world-space pose.
///
/// Used by the viewer and navigation operators wherever the camera's position and
/// orientation must be manipulated directly.  The canonical pose for a camera node
/// in the scene graph lives in that node's `Transform`; this type is the convenient
/// working representation during a frame or an interaction gesture.
///
/// # Example
///
/// ```
/// use duck_engine_scene::camera::PositionedCamera;
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
    ///
    /// # Arguments
    /// * `bounds` - The axis-aligned bounding box to fit in view
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

    /// Projects a 3D world-space point to normalized device coordinates (NDC).
    ///
    /// # Arguments
    /// * `world_point` - Point in world space
    ///
    /// # Returns
    /// A 3D point in NDC space where:
    /// - X and Y are in range [-1, 1] (left/right, bottom/top)
    /// - Z is in range [0, 1] (near/far in WGPU depth convention)
    pub fn project_point_ndc(&self, world_point: Point3) -> Point3 {
        let vp = self.build_view_projection_matrix();
        let homogeneous = vp * world_point.to_homogeneous();

        // Perform perspective division
        Point3::from_homogeneous(homogeneous)
    }

    /// Unprojects a point from normalized device coordinates (NDC) to world space.
    ///
    /// # Arguments
    /// * `ndc_point` - Point in NDC space where:
    ///   - X and Y are in range [-1, 1] (left/right, bottom/top)
    ///   - Z is in range [0, 1] (near/far in WGPU depth convention)
    ///
    /// # Returns
    /// A 3D point in world space, or None if the view-projection matrix is not invertible.
    pub fn unproject_point_ndc(&self, ndc_point: Point3) -> Option<Point3> {
        let viewproj = self.build_view_projection_matrix();

        let inv_vp = viewproj.invert()?;

        // Convert NDC point to homogeneous coordinates
        let homogeneous = inv_vp * ndc_point.to_homogeneous();

        // Perform perspective division
        Some(Point3::from_homogeneous(homogeneous))
    }

    /// Projects a 3D world-space point to screen-space pixel coordinates.
    ///
    /// # Arguments
    /// * `world_point` - Point in world space
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    ///
    /// # Returns
    /// A 3D point in screen space where:
    /// - X is in range [0, screen_width] (left to right)
    /// - Y is in range [0, screen_height] (top to bottom)
    /// - Z is the depth value in range [0, 1]
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


    /// Unprojects a screen-space pixel coordinate to a point in world space.
    ///
    /// # Arguments
    /// * `screen_x` - X coordinate in screen space (0 = left edge)
    /// * `screen_y` - Y coordinate in screen space (0 = top edge)
    /// * `depth` - Depth value in range [0, 1] (0 = near plane, 1 = far plane)
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    ///
    /// # Returns
    /// A 3D point in world space, or None if the view-projection matrix is not invertible.
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

    /// Creates a ray from a normalized device coordinate point.
    ///
    /// The ray originates at the near plane and points through the specified
    /// NDC position into the 3D world.
    ///
    /// # Arguments
    /// * `ndc_x` - X coordinate in NDC space [-1, 1] (left to right)
    /// * `ndc_y` - Y coordinate in NDC space [-1, 1] (bottom to top)
    ///
    /// # Returns
    /// A ray originating at the near plane, pointing through the NDC point
    /// into the 3D world.
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

    /// Creates a ray from a screen-space point, unprojecting it to world space.
    ///
    /// The ray originates at the near plane and points through the specified
    /// screen pixel into the 3D world. This is useful for mouse picking and selection.
    ///
    /// # Arguments
    /// * `screen_x` - X coordinate in screen space (0 = left edge)
    /// * `screen_y` - Y coordinate in screen space (0 = top edge)
    /// * `screen_width` - Width of the screen/viewport in pixels
    /// * `screen_height` - Height of the screen/viewport in pixels
    ///
    /// # Returns
    /// A ray originating at the near plane, pointing through the screen point
    /// into the 3D world.
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

    /// Extracts the projection intrinsics from this positioned camera.
    pub fn projection(&self) -> CameraProjection {
        CameraProjection {
            fovy: self.fovy,
            znear: self.znear,
            zfar: self.zfar,
            ortho: self.ortho,
            focus_distance: self.length(),
        }
    }

    /// Builds a node `Transform` that encodes this camera's pose (eye + orientation).
    ///
    /// Column convention: right = +X, corrected-up = +Y, forward = -Z
    /// (camera looks down -Z in a right-handed coordinate system).
    pub fn to_node_transform(&self) -> crate::common::Transform {
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

    // ===== CameraProjection round-trip test =====

    #[test]
    fn test_projection_roundtrip() {
        let original = create_test_camera();
        let proj = original.projection();

        // focus_distance should equal the eye-to-target distance
        assert!((proj.focus_distance - 5.0).abs() < EPSILON);

        // Reconstruct via to_node_transform + into_positioned
        let transform_mat = original.to_node_transform().to_matrix();
        let reconstructed = proj.into_positioned(transform_mat, original.aspect);

        assert!((reconstructed.eye.x - original.eye.x).abs() < 1e-4);
        assert!((reconstructed.eye.y - original.eye.y).abs() < 1e-4);
        assert!((reconstructed.eye.z - original.eye.z).abs() < 1e-4);
        assert!((reconstructed.target.x - original.target.x).abs() < 1e-4);
        assert!((reconstructed.target.y - original.target.y).abs() < 1e-4);
        assert!((reconstructed.target.z - original.target.z).abs() < 1e-4);
        assert!((reconstructed.up.x - original.up.x).abs() < 1e-4);
        assert!((reconstructed.up.y - original.up.y).abs() < 1e-4);
        assert!((reconstructed.up.z - original.up.z).abs() < 1e-4);
    }

    #[test]
    fn positioned_camera_for_node_uses_given_aspect() {
        use crate::resource::{NodeFlags, NodePayload};
        use crate::SceneData;
        use crate::common;

        let mut scene = SceneData::new();
        let camera = scene
            .add_node(None, Some("cam".into()), common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        let proj = CameraProjection {
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
            focus_distance: 5.0,
        };
        scene.set_node_payload(camera, NodePayload::Camera(proj));

        let positioned = scene.positioned_camera_for_node(camera, 2.0).expect("camera resolves");
        assert!((positioned.aspect - 2.0).abs() < EPSILON);
    }
}
