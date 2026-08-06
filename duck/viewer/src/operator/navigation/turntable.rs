use duck_engine_common::{InnerSpace, Point3, Rotation, point3, vec3};

use crate::scene::PositionedCamera;
use crate::scene::common::quaternion_from_axis_angle_safe;

use super::ORBIT_SENSITIVITY;

/// Internal state for orbit-style navigation (orbit / pan / zoom).
pub(super) struct TurntableState {
    /// Azimuth angle in radians (horizontal rotation around target).
    pub azimuth: f32,
    /// Elevation angle in radians (vertical rotation).
    pub elevation: f32,
    /// Base distance from camera to target.
    pub radius: f32,
    /// Custom orbit pivot point. When set, orbit rotates the camera around
    /// this point instead of `camera.target`.
    pub pivot: Option<Point3>,
}

impl TurntableState {
    pub fn new() -> Self {
        Self {
            azimuth: 0.0,
            elevation: 0.0,
            radius: 5.0,
            pivot: None,
        }
    }

    /// Initialize parameters from current camera state.
    pub fn init(&mut self, camera: &PositionedCamera) {
        self.pivot = None;
        self.radius = camera.length();

        // Calculate direction vector from target to eye
        let direction = camera.eye - camera.target;

        // Calculate azimuth (horizontal angle around Y-axis)
        self.azimuth = f32::atan2(direction.x, direction.z);

        // Calculate elevation (vertical angle from horizontal plane)
        let horizontal_distance =
            f32::sqrt(direction.x * direction.x + direction.z * direction.z);
        self.elevation = f32::atan2(direction.y, horizontal_distance);
    }

    /// Initialize orbit parameters for orbiting around an explicit pivot point.
    pub fn init_with_pivot(&mut self, camera: &PositionedCamera, pivot: Point3) {
        self.pivot = Some(pivot);

        // Compute elevation from eye-to-pivot direction (for elevation clamping)
        let direction = camera.eye - pivot;
        let horizontal_distance =
            f32::sqrt(direction.x * direction.x + direction.z * direction.z);
        self.elevation = f32::atan2(direction.y, horizontal_distance);
    }

    /// Update camera position based on current orbit parameters (non-pivot orbit only).
    pub fn update_camera_position(&self, camera: &mut PositionedCamera) {
        // Convert spherical coordinates to Cartesian
        let x = camera.target.x + self.radius * self.elevation.cos() * self.azimuth.sin();
        let y = camera.target.y + self.radius * self.elevation.sin();
        let z = camera.target.z + self.radius * self.elevation.cos() * self.azimuth.cos();

        camera.eye = point3(x, y, z);

        // Update up vector to maintain proper orientation
        let forward = (camera.target - camera.eye).normalize();
        let world_up = vec3(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();
        camera.up = forward.cross(right).normalize();
    }

    /// Handle orbit around the pivot point using incremental rotations.
    ///
    /// Rotates both eye and target around the pivot by the same rotation,
    /// keeping the pivot visually stationary on screen.
    fn handle_pivot_orbit(&mut self, dx: f64, dy: f64, camera: &mut PositionedCamera) {
        let pivot = self.pivot.unwrap();

        let d_azimuth = -(dx as f32) * ORBIT_SENSITIVITY;
        let d_elevation = dy as f32 * ORBIT_SENSITIVITY;

        // Clamp cumulative elevation to prevent going over the poles
        let new_elevation = self.elevation + d_elevation;
        const MAX_ELEVATION: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        let clamped = new_elevation.clamp(-MAX_ELEVATION, MAX_ELEVATION);
        let actual_d_elevation = clamped - self.elevation;
        self.elevation = clamped;

        // Compute the camera's right axis for elevation rotation
        let forward = (camera.target - camera.eye).normalize();
        let world_up = vec3(0.0, 1.0, 0.0);
        let right = world_up.cross(forward).normalize();

        // Build combined rotation quaternion: azimuth (around Y) then elevation (around right)
        let azimuth_rot = quaternion_from_axis_angle_safe(world_up, d_azimuth);
        let elevation_rot = quaternion_from_axis_angle_safe(right, actual_d_elevation);
        let rotation = elevation_rot * azimuth_rot;

        // Rotate both eye and target offsets around the pivot
        let eye_offset = rotation.rotate_vector(camera.eye - pivot);
        let target_offset = rotation.rotate_vector(camera.target - pivot);

        camera.eye = pivot + eye_offset;
        camera.target = pivot + target_offset;

        // Update up vector
        let forward = (camera.target - camera.eye).normalize();
        let right = world_up.cross(forward).normalize();
        camera.up = forward.cross(right).normalize();
    }

    /// Handle zoom via mouse wheel by adjusting camera distance.
    pub fn handle_zoom(&mut self, delta: f32, camera: &mut PositionedCamera, model_radius: f32) {
        self.radius = super::zoom_radius(self.radius, delta, model_radius);
        self.update_camera_position(camera);
    }

    /// Handle orbit rotation based on mouse movement.
    pub fn handle_orbit(&mut self, dx: f64, dy: f64, camera: &mut PositionedCamera) {
        if self.pivot.is_some() {
            return self.handle_pivot_orbit(dx, dy, camera);
        }

        let dx = dx as f32 * ORBIT_SENSITIVITY;
        let dy = dy as f32 * ORBIT_SENSITIVITY;

        self.azimuth -= dx;

        self.elevation += dy;
        const MAX_ELEVATION: f32 = std::f32::consts::FRAC_PI_2 - 0.01;
        self.elevation = self.elevation.clamp(-MAX_ELEVATION, MAX_ELEVATION);

        self.update_camera_position(camera);
    }

    /// Handle panning based on mouse movement.
    pub fn handle_pan(&self, dx: f32, dy: f32, camera: &mut PositionedCamera, viewport: (u32, u32)) {
        super::pan(dx, dy, camera, viewport);
    }
}
