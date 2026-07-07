//! Selection-agnostic core of an interactive transform: input bindings, the
//! axis-constraint state machine, and the mouse-delta → transform math.
//!
//! [`TransformInteraction`] knows nothing about what is being transformed —
//! consumers start it about a pivot, feed it mouse deltas, and read back the
//! resulting translation/rotation/scale (or the composed [`Matrix4`] delta)
//! to apply however they see fit.

use duck_engine_common::{EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Vector3};
use duck_engine_scene::common::Plane;
use duck_engine_scene::PositionedCamera;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::common::{
    local_axis_x, local_axis_y, local_axis_z, quaternion_from_axis_angle_safe, Axis, RgbaColor,
};
use crate::gizmo::GizmoType;
use crate::input::{Key, Modifiers, MouseButton, NamedKey};

/// Semantic actions for an interactive transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransformAction {
    /// Begin a freeform transform in this interaction's mode.
    StartTransform,
    /// Cycle the axis constraint to X (world then local on repeated press).
    ConstrainX,
    /// Cycle the axis constraint to Y (world then local on repeated press).
    ConstrainY,
    /// Cycle the axis constraint to Z (world then local on repeated press).
    ConstrainZ,
    /// Confirm the active transform via keyboard.
    KeyConfirm,
    /// Cancel the active transform via keyboard.
    KeyCancel,
    /// Confirm the active transform via mouse click.
    MouseConfirm,
    /// Cancel the active transform via mouse click.
    MouseCancel,
    /// Drag interaction with a gizmo handle (drag start, drag, and drag end).
    GizmoDrag,
}

/// The type of transform being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

impl TransformMode {
    /// The gizmo handle set for this mode.
    pub fn gizmo_type(self) -> GizmoType {
        match self {
            TransformMode::Translate => GizmoType::Translate,
            TransformMode::Rotate => GizmoType::Rotate,
            TransformMode::Scale => GizmoType::Scale,
        }
    }

    /// The keyboard key that starts a freeform transform in this mode.
    fn start_key(self) -> char {
        match self {
            TransformMode::Translate => 'g',
            TransformMode::Rotate => 'r',
            TransformMode::Scale => 's',
        }
    }
}

/// Axis constraint for the transform operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisConstraint {
    /// No constraint - free transform
    None,
    /// Constrain to world X axis
    WorldX,
    /// Constrain to world Y axis
    WorldY,
    /// Constrain to world Z axis
    WorldZ,
    /// Constrain to local X axis (relative to the interaction frame)
    LocalX,
    /// Constrain to local Y axis (relative to the interaction frame)
    LocalY,
    /// Constrain to local Z axis (relative to the interaction frame)
    LocalZ,
}

impl AxisConstraint {
    /// Returns the color for visual feedback (RGB = XYZ convention).
    pub fn color(&self) -> Option<RgbaColor> {
        match self {
            AxisConstraint::None => None,
            AxisConstraint::WorldX | AxisConstraint::LocalX => Some(RgbaColor::RED),
            AxisConstraint::WorldY | AxisConstraint::LocalY => Some(RgbaColor::GREEN),
            AxisConstraint::WorldZ | AxisConstraint::LocalZ => Some(RgbaColor::BLUE),
        }
    }

    /// Returns whether this is a local axis constraint.
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            AxisConstraint::LocalX | AxisConstraint::LocalY | AxisConstraint::LocalZ
        )
    }
}

/// Maps an axis constraint to the corresponding Axis (e.g. for gizmo highlighting).
pub fn axis_from_constraint(constraint: &AxisConstraint) -> Option<Axis> {
    match constraint {
        AxisConstraint::WorldX | AxisConstraint::LocalX => Some(Axis::X),
        AxisConstraint::WorldY | AxisConstraint::LocalY => Some(Axis::Y),
        AxisConstraint::WorldZ | AxisConstraint::LocalZ => Some(Axis::Z),
        AxisConstraint::None => None,
    }
}

/// State machine for one in-progress transform (translate, rotate, or scale):
/// activation about a pivot, axis-constraint cycling, mouse-delta
/// accumulation, and the resulting transform values.
pub struct TransformInteraction {
    /// The fixed operation this interaction performs.
    mode: TransformMode,

    /// Whether a transform is currently being applied (driven by mouse motion).
    active: bool,

    /// Current axis constraint.
    axis_constraint: AxisConstraint,

    /// Accumulated mouse movement since the transform started.
    accumulated_delta: (f32, f32),

    /// Pivot point in world space (center of rotation/scale).
    pivot_world: Point3,

    /// Orientation of the local constraint axes.
    frame_rotation: Quaternion,

    /// Model radius for scaling sensitivity.
    model_radius: f32,

    pub bindings: InputMap<TransformAction>,
}

impl TransformInteraction {
    /// Creates a new interaction locked to the given mode.
    pub fn new(mode: TransformMode) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::Key { key: Key::Character(mode.start_key()), modifiers: Modifiers::default() },
                TransformAction::StartTransform,
            )
            .bind(
                InputBinding::Key { key: Key::Character('x'), modifiers: Modifiers::default() },
                TransformAction::ConstrainX,
            )
            .bind(
                InputBinding::Key { key: Key::Character('y'), modifiers: Modifiers::default() },
                TransformAction::ConstrainY,
            )
            .bind(
                InputBinding::Key { key: Key::Character('z'), modifiers: Modifiers::default() },
                TransformAction::ConstrainZ,
            )
            .bind(
                InputBinding::Key { key: Key::Named(NamedKey::Enter), modifiers: Modifiers::default() },
                TransformAction::KeyConfirm,
            )
            .bind(
                InputBinding::Key { key: Key::Named(NamedKey::Escape), modifiers: Modifiers::default() },
                TransformAction::KeyCancel,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::MouseConfirm,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                TransformAction::MouseCancel,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            )
            .bind(
                InputBinding::MouseDragEnd { button: MouseButton::Left, modifiers: Modifiers::default() },
                TransformAction::GizmoDrag,
            );
        Self {
            mode,
            active: false,
            axis_constraint: AxisConstraint::None,
            accumulated_delta: (0.0, 0.0),
            pivot_world: Point3::origin(),
            frame_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            model_radius: 1.0,
            bindings,
        }
    }

    pub fn mode(&self) -> TransformMode {
        self.mode
    }

    /// Returns true if a transform operation is currently active.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Begin a transform about `pivot`, unconstrained and with no accumulated
    /// movement. `frame_rotation` orients the local constraint axes (`None`
    /// keeps the previous frame); `model_radius` scales sensitivity.
    pub fn start(&mut self, pivot: Point3, frame_rotation: Option<Quaternion>, model_radius: f32) {
        self.pivot_world = pivot;
        if let Some(rotation) = frame_rotation {
            self.frame_rotation = rotation;
        }
        self.model_radius = model_radius;
        self.active = true;
        self.axis_constraint = AxisConstraint::None;
        self.accumulated_delta = (0.0, 0.0);
    }

    /// Deactivate and clear the constraint and accumulated movement (after a
    /// confirm or cancel).
    pub fn finish(&mut self) {
        self.active = false;
        self.axis_constraint = AxisConstraint::None;
        self.accumulated_delta = (0.0, 0.0);
    }

    /// Add mouse movement to the accumulated drag.
    pub fn accumulate(&mut self, dx: f32, dy: f32) {
        self.accumulated_delta.0 += dx;
        self.accumulated_delta.1 += dy;
    }

    pub fn pivot(&self) -> Point3 {
        self.pivot_world
    }

    pub fn frame_rotation(&self) -> Quaternion {
        self.frame_rotation
    }

    pub fn model_radius(&self) -> f32 {
        self.model_radius
    }

    pub fn axis_constraint(&self) -> AxisConstraint {
        self.axis_constraint
    }

    pub fn set_axis_constraint(&mut self, constraint: AxisConstraint) {
        self.axis_constraint = constraint;
    }

    /// Cycles the axis constraint for a given axis key.
    /// None → World → Local → None
    pub fn cycle_axis_constraint(&mut self, axis: char) {
        self.axis_constraint = match (axis, &self.axis_constraint) {
            // X axis cycling
            ('x', AxisConstraint::None) => AxisConstraint::WorldX,
            ('x', AxisConstraint::WorldX) => AxisConstraint::LocalX,
            ('x', AxisConstraint::LocalX) => AxisConstraint::None,
            ('x', _) => AxisConstraint::WorldX, // Switch from other axis

            // Y axis cycling
            ('y', AxisConstraint::None) => AxisConstraint::WorldY,
            ('y', AxisConstraint::WorldY) => AxisConstraint::LocalY,
            ('y', AxisConstraint::LocalY) => AxisConstraint::None,
            ('y', _) => AxisConstraint::WorldY, // Switch from other axis

            // Z axis cycling
            ('z', AxisConstraint::None) => AxisConstraint::WorldZ,
            ('z', AxisConstraint::WorldZ) => AxisConstraint::LocalZ,
            ('z', AxisConstraint::LocalZ) => AxisConstraint::None,
            ('z', _) => AxisConstraint::WorldZ, // Switch from other axis

            _ => self.axis_constraint,
        };
    }

    /// Get the constraint axis direction in world space.
    pub fn constraint_axis(&self) -> Option<Vector3> {
        match self.axis_constraint {
            AxisConstraint::None => None,
            AxisConstraint::WorldX => Some(Vector3::unit_x()),
            AxisConstraint::WorldY => Some(Vector3::unit_y()),
            AxisConstraint::WorldZ => Some(Vector3::unit_z()),
            AxisConstraint::LocalX => Some(local_axis_x(self.frame_rotation)),
            AxisConstraint::LocalY => Some(local_axis_y(self.frame_rotation)),
            AxisConstraint::LocalZ => Some(local_axis_z(self.frame_rotation)),
        }
    }

    /// Compute the translation delta based on mouse movement and constraints.
    /// `size` is the viewport size in pixels.
    pub fn translation(&self, camera: &PositionedCamera, size: (u32, u32)) -> Vector3 {
        let pivot = &self.pivot_world;
        let (width, height) = size;
        let (dx, dy) = self.accumulated_delta;

        let movement_plane = Plane::from_point(camera.forward(), *pivot);
        let Point3 { x: screen_x, y: screen_y, .. } = camera.project_point_screen(*pivot, width, height);
        let diff_ray = camera.ray_from_screen_point(screen_x + dx, screen_y + dy, width, height);
        let new_pivot = diff_ray.intersect_plane(&movement_plane)
            .map_or(*pivot, |intersection| intersection.1);
        let move_vector = new_pivot - pivot;

        match self.constraint_axis() {
            None => {
                move_vector
            }
            Some(axis) => {
                axis * axis.dot(move_vector)
            }
        }
    }

    /// Compute the rotation based on mouse movement and constraints.
    pub fn rotation(&self, camera: &PositionedCamera) -> Quaternion {
        // 0.5 degrees per pixel
        let sensitivity = 0.5_f32.to_radians();
        let angle = self.accumulated_delta.0 * sensitivity;

        let axis = match self.constraint_axis() {
            None => {
                // Free rotation: rotate around view axis
                camera.forward()
            }
            Some(axis) => axis,
        };

        quaternion_from_axis_angle_safe(axis, angle)
    }

    /// Compute the scale factor based on mouse movement and constraints.
    pub fn scale(&self) -> Vector3 {
        // 0.5% change per pixel
        let sensitivity = 0.005;
        let factor = 1.0 + self.accumulated_delta.0 * sensitivity;
        // Clamp to prevent negative or zero scale
        let factor = factor.max(0.01);

        match self.axis_constraint {
            AxisConstraint::None => Vector3::new(factor, factor, factor),
            AxisConstraint::WorldX | AxisConstraint::LocalX => Vector3::new(factor, 1.0, 1.0),
            AxisConstraint::WorldY | AxisConstraint::LocalY => Vector3::new(1.0, factor, 1.0),
            AxisConstraint::WorldZ | AxisConstraint::LocalZ => Vector3::new(1.0, 1.0, factor),
        }
    }

    /// The current transform for this mode as a single world-space matrix:
    /// rotation and scale act about the pivot (scale in the local frame under
    /// a local constraint), so `new_world = delta * old_world` for points.
    /// `size` is the viewport size in pixels.
    pub fn delta_matrix(&self, camera: &PositionedCamera, size: (u32, u32)) -> Matrix4 {
        let to_pivot = Matrix4::from_translation(self.pivot_world.to_vec());
        let from_pivot = Matrix4::from_translation(-self.pivot_world.to_vec());
        match self.mode {
            TransformMode::Translate => Matrix4::from_translation(self.translation(camera, size)),
            TransformMode::Rotate => to_pivot * Matrix4::from(self.rotation(camera)) * from_pivot,
            TransformMode::Scale => {
                let s = self.scale();
                let scale = Matrix4::from_nonuniform_scale(s.x, s.y, s.z);
                let scale = if self.axis_constraint.is_local() {
                    let frame = Matrix4::from(self.frame_rotation);
                    let frame_inv = Matrix4::from(self.frame_rotation.conjugate());
                    frame * scale * frame_inv
                } else {
                    scale
                };
                to_pivot * scale * from_pivot
            }
        }
    }
}
