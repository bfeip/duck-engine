use duck_engine_common::{InnerSpace, Vector3};

use crate::bindings::InputMap;
use crate::input::{ElementState, Key, Modifiers};
use crate::scene::PositionedCamera;
use crate::scene_scale;

use super::NavigationAction;

/// Mouse look sensitivity in radians per pixel of mouse movement.
const LOOK_SENSITIVITY: f32 = 0.003;

/// Maximum pitch angle (looking up/down) in radians. Just under 90 degrees.
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Internal state for walk-style navigation (movement keys + mouse look).
pub(super) struct WalkState {
    /// Current yaw angle (horizontal rotation) in radians.
    pub yaw: f32,
    /// Current pitch angle (vertical rotation) in radians.
    pub pitch: f32,
    forward_pressed: bool,
    backward_pressed: bool,
    left_pressed: bool,
    right_pressed: bool,
}

impl WalkState {
    pub fn new() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.0,
            forward_pressed: false,
            backward_pressed: false,
            left_pressed: false,
            right_pressed: false,
        }
    }

    /// Initialize yaw and pitch from current camera orientation.
    pub fn init_from_camera(&mut self, camera: &PositionedCamera) {
        let forward = camera.forward();
        self.yaw = f32::atan2(forward.x, forward.z);
        let horizontal_length = (forward.x * forward.x + forward.z * forward.z).sqrt();
        self.pitch = f32::atan2(forward.y, horizontal_length);
    }

    /// Handle mouse look based on mouse drag delta.
    pub fn handle_look(&mut self, dx: f32, dy: f32, camera: &mut PositionedCamera) {
        self.yaw -= dx * LOOK_SENSITIVITY;
        self.pitch -= dy * LOOK_SENSITIVITY;
        self.pitch = self.pitch.clamp(-MAX_PITCH, MAX_PITCH);
        self.update_camera_target(camera);
    }

    /// Update camera target based on current yaw and pitch.
    pub fn update_camera_target(&self, camera: &mut PositionedCamera) {
        let distance = (camera.target - camera.eye).magnitude();
        let forward = Vector3::new(
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.cos(),
        );
        camera.target = camera.eye + forward * distance;
        let world_up = Vector3::new(0.0, 1.0, 0.0);
        let right = forward.cross(world_up).normalize();
        camera.up = right.cross(forward).normalize();
    }

    /// Apply movement based on currently pressed keys.
    /// Returns true if any movement was applied.
    pub fn apply_movement(
        &self,
        camera: &mut PositionedCamera,
        delta_time: f32,
        model_radius: f32,
    ) -> bool {
        let mut movement = Vector3::new(0.0, 0.0, 0.0);
        let forward_flat = Vector3::new(self.yaw.sin(), 0.0, self.yaw.cos()).normalize();
        let right = Vector3::new(-self.yaw.cos(), 0.0, self.yaw.sin());

        if self.forward_pressed { movement += forward_flat; }
        if self.backward_pressed { movement -= forward_flat; }
        if self.right_pressed { movement += right; }
        if self.left_pressed { movement -= right; }

        if movement.magnitude2() > 0.0 {
            let walk_speed = scene_scale::walk_speed(model_radius);
            movement = movement.normalize() * walk_speed * delta_time;
            camera.eye += movement;
            camera.target += movement;
            return true;
        }
        false
    }

    /// Handle a key press/release event using the provided bindings.
    ///
    /// Only responds to [`NavigationAction`] movement variants; all other
    /// bound actions are ignored. Returns true if the key was handled.
    pub fn handle_key(
        &mut self,
        key: &Key,
        state: ElementState,
        bindings: &InputMap<NavigationAction>,
    ) -> bool {
        let pressed = state == ElementState::Pressed;
        let actions = bindings.actions_for_key(key, Modifiers::default());
        if actions.is_empty() {
            return false;
        }
        let mut handled = false;
        for action in actions {
            match action {
                NavigationAction::MoveForward => { self.forward_pressed = pressed; handled = true; }
                NavigationAction::MoveBackward => { self.backward_pressed = pressed; handled = true; }
                NavigationAction::MoveLeft => { self.left_pressed = pressed; handled = true; }
                NavigationAction::MoveRight => { self.right_pressed = pressed; handled = true; }
                _ => {}
            }
        }
        handled
    }

    /// Check if any movement key is currently pressed.
    pub fn is_moving(&self) -> bool {
        self.forward_pressed || self.backward_pressed || self.left_pressed || self.right_pressed
    }

    /// Reset all pressed key state.
    pub fn reset_keys(&mut self) {
        self.forward_pressed = false;
        self.backward_pressed = false;
        self.left_pressed = false;
        self.right_pressed = false;
    }
}
