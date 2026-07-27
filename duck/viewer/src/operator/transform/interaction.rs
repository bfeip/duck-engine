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
use crate::input::{Key, Modifiers, MouseButton, NamedKey};

use super::gizmo::{GizmoHandleId, GizmoType};

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
    /// Cycle the plane constraint excluding X, i.e. the YZ plane (world then local).
    ConstrainPlaneX,
    /// Cycle the plane constraint excluding Y, i.e. the XZ plane (world then local).
    ConstrainPlaneY,
    /// Cycle the plane constraint excluding Z, i.e. the XY plane (world then local).
    ConstrainPlaneZ,
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

/// The reference frame a constraint is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintSpace {
    /// World coordinate axes.
    World,
    /// Axes relative to the interaction frame.
    Local,
}

/// Axis constraint for the transform operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisConstraint {
    /// No constraint - free transform.
    None,
    /// Constrain to a single axis in the given space.
    Axis(Axis, ConstraintSpace),
    /// Constrain to the plane whose normal is the given (excluded) axis, in the
    /// given space. The operation runs on the other two axes (e.g. `Plane(Z)`
    /// is the XY plane).
    Plane(Axis, ConstraintSpace),
}

impl AxisConstraint {
    /// Returns the color for visual feedback (RGB = XYZ convention). Only
    /// single-axis constraints have a single feedback color; planes are drawn
    /// with their two in-plane axis colors by the annotations.
    pub fn color(&self) -> Option<RgbaColor> {
        match self {
            AxisConstraint::Axis(axis, _) => Some(axis.color()),
            AxisConstraint::None | AxisConstraint::Plane(..) => None,
        }
    }

    /// Returns whether this constraint is expressed in local space.
    pub fn is_local(&self) -> bool {
        matches!(
            self,
            AxisConstraint::Axis(_, ConstraintSpace::Local)
                | AxisConstraint::Plane(_, ConstraintSpace::Local)
        )
    }
}

/// Maps a single-axis constraint to its `Axis` (e.g. for gizmo highlighting).
/// Returns `None` for plane and unconstrained states.
pub fn axis_from_constraint(constraint: &AxisConstraint) -> Option<Axis> {
    match constraint {
        AxisConstraint::Axis(axis, _) => Some(*axis),
        AxisConstraint::None | AxisConstraint::Plane(..) => None,
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

    /// Whether the active drag came from grabbing a gizmo handle.
    directional: bool,

    /// Screen position where a directional (handle) drag started. Used by
    /// plane/uniform scale to measure radial distance from the pivot.
    drag_start_screen: Option<(f32, f32)>,

    /// Input bindings consumed by the driver's dispatch loop.
    pub(in crate::operator::transform) bindings: InputMap<TransformAction>,
}

/// The default input bindings for a transform interaction: the mode's start
/// key, X/Y/Z axis and Shift+X/Y/Z plane constraints, Enter/LMB confirm,
/// Escape/RMB cancel, and LMB drags on gizmo handles.
fn default_bindings(mode: TransformMode) -> InputMap<TransformAction> {
    InputMap::new()
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
                InputBinding::Key { key: Key::Character('x'), modifiers: Modifiers { shift: true, ..Modifiers::default() } },
                TransformAction::ConstrainPlaneX,
            )
            .bind(
                InputBinding::Key { key: Key::Character('y'), modifiers: Modifiers { shift: true, ..Modifiers::default() } },
                TransformAction::ConstrainPlaneY,
            )
            .bind(
                InputBinding::Key { key: Key::Character('z'), modifiers: Modifiers { shift: true, ..Modifiers::default() } },
                TransformAction::ConstrainPlaneZ,
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
        )
}

impl TransformInteraction {
    /// Creates a new interaction locked to the given mode.
    pub fn new(mode: TransformMode) -> Self {
        Self {
            mode,
            active: false,
            axis_constraint: AxisConstraint::None,
            accumulated_delta: (0.0, 0.0),
            pivot_world: Point3::origin(),
            frame_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            model_radius: 1.0,
            directional: false,
            drag_start_screen: None,
            bindings: default_bindings(mode),
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
        self.directional = false;
        self.drag_start_screen = None;
    }

    /// Deactivate and clear the constraint and accumulated movement (after a
    /// confirm or cancel).
    pub fn finish(&mut self) {
        self.active = false;
        self.axis_constraint = AxisConstraint::None;
        self.accumulated_delta = (0.0, 0.0);
        self.directional = false;
        self.drag_start_screen = None;
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

    /// Constrain to a gizmo handle and mark the drag directional. `screen_pos`
    /// is the cursor position where the drag began (used by plane/uniform scale).
    pub fn constrain_to_handle(&mut self, id: GizmoHandleId, screen_pos: (f32, f32)) {
        self.axis_constraint = match id {
            GizmoHandleId::Axis(axis) => AxisConstraint::Axis(axis, ConstraintSpace::World),
            GizmoHandleId::Plane(normal) => AxisConstraint::Plane(normal, ConstraintSpace::World),
            GizmoHandleId::Ball => AxisConstraint::None,
        };
        self.directional = true;
        self.drag_start_screen = Some(screen_pos);
    }

    /// Cycles the single-axis constraint for a given axis key.
    /// None → World → Local → None (switching from another constraint jumps to World).
    pub fn cycle_axis_constraint(&mut self, axis: char) {
        let Some(a) = Axis::from_char(axis) else { return };
        use ConstraintSpace::{Local, World};
        self.axis_constraint = match self.axis_constraint {
            AxisConstraint::Axis(cur, World) if cur == a => AxisConstraint::Axis(a, Local),
            AxisConstraint::Axis(cur, Local) if cur == a => AxisConstraint::None,
            _ => AxisConstraint::Axis(a, World),
        };
    }

    /// Cycles the plane constraint excluding the given axis key (Blender Shift+axis).
    /// None → World → Local → None (switching from another constraint jumps to World).
    pub fn cycle_plane_constraint(&mut self, axis: char) {
        let Some(a) = Axis::from_char(axis) else { return };
        use ConstraintSpace::{Local, World};
        self.axis_constraint = match self.axis_constraint {
            AxisConstraint::Plane(cur, World) if cur == a => AxisConstraint::Plane(a, Local),
            AxisConstraint::Plane(cur, Local) if cur == a => AxisConstraint::None,
            _ => AxisConstraint::Plane(a, World),
        };
    }

    /// The world-space direction of an axis in the given space.
    fn axis_direction(&self, axis: Axis, space: ConstraintSpace) -> Vector3 {
        match space {
            ConstraintSpace::World => axis.direction(),
            ConstraintSpace::Local => match axis {
                Axis::X => local_axis_x(self.frame_rotation),
                Axis::Y => local_axis_y(self.frame_rotation),
                Axis::Z => local_axis_z(self.frame_rotation),
            },
        }
    }

    /// Get the single-axis constraint direction in world space (`None` unless a
    /// single-axis constraint is active).
    pub fn constraint_axis(&self) -> Option<Vector3> {
        match self.axis_constraint {
            AxisConstraint::Axis(axis, space) => Some(self.axis_direction(axis, space)),
            _ => None,
        }
    }

    /// The gizmo handle that should be highlighted for the current constraint.
    pub fn highlight_handle(&self) -> Option<GizmoHandleId> {
        match self.axis_constraint {
            AxisConstraint::None => None,
            AxisConstraint::Axis(axis, _) => Some(GizmoHandleId::Axis(axis)),
            AxisConstraint::Plane(normal, _) => Some(GizmoHandleId::Plane(normal)),
        }
    }

    /// Get the constraint plane's world-space normal (`None` unless a plane
    /// constraint is active).
    pub fn constraint_plane_normal(&self) -> Option<Vector3> {
        match self.axis_constraint {
            AxisConstraint::Plane(axis, space) => Some(self.axis_direction(axis, space)),
            _ => None,
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

        match self.axis_constraint {
            AxisConstraint::None => move_vector,
            AxisConstraint::Axis(..) => {
                let axis = self.constraint_axis().unwrap();
                axis * axis.dot(move_vector)
            }
            AxisConstraint::Plane(..) => {
                let normal = self.constraint_plane_normal().unwrap();
                move_vector - normal * normal.dot(move_vector)
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

    /// Signed change (in pixels) of the cursor's distance from the pivot's
    /// screen projection since a directional drag began. Used for plane and
    /// uniform scale, where there is no single handle axis to project onto:
    /// dragging outward grows, inward shrinks. Falls back to horizontal drag.
    fn radial_magnitude(&self, camera: &PositionedCamera, size: (u32, u32)) -> f32 {
        let (w, h) = size;
        let (dx, dy) = self.accumulated_delta;
        let Some((sx, sy)) = self.drag_start_screen else { return dx };
        let p = camera.project_point_screen(self.pivot_world, w, h);
        let start_dist = ((sx - p.x).powi(2) + (sy - p.y).powi(2)).sqrt();
        let cur_dist = (((sx + dx) - p.x).powi(2) + ((sy + dy) - p.y).powi(2)).sqrt();
        cur_dist - start_dist
    }

    /// Compute the scale factor based on mouse movement and constraints.
    /// `size` is the viewport size in pixels.
    pub fn scale(&self, camera: &PositionedCamera, size: (u32, u32)) -> Vector3 {
        // 0.5% change per pixel
        let sensitivity = 0.005;

        let (dx, dy) = self.accumulated_delta;
        let magnitude = match (self.directional, self.axis_constraint) {
            // Single-axis handle drag: project the handle's world axis to screen
            // space and measure the drag along it, so dragging toward the handle
            // grows scale and away shrinks it.
            (true, AxisConstraint::Axis(..)) => {
                let axis = self.constraint_axis().unwrap();
                let (w, h) = size;
                let p0 = camera.project_point_screen(self.pivot_world, w, h);
                let p1 = camera.project_point_screen(self.pivot_world + axis, w, h);
                let (ex, ey) = (p1.x - p0.x, p1.y - p0.y);
                let len = (ex * ex + ey * ey).sqrt();
                if len > 1e-3 {
                    // Signed drag along the handle, in pixels.
                    (dx * ex + dy * ey) / len
                } else {
                    // Axis ~parallel to the view direction: fall back to horizontal.
                    dx
                }
            }
            // Plane or uniform handle drag: radial distance from the pivot.
            (true, AxisConstraint::Plane(..)) | (true, AxisConstraint::None) => {
                self.radial_magnitude(camera, size)
            }
            // Keyboard / non-directional: horizontal drag.
            _ => dx,
        };

        // Clamp to prevent negative or zero scale
        let factor = (1.0 + magnitude * sensitivity).max(0.01);

        match self.axis_constraint {
            AxisConstraint::None => Vector3::new(factor, factor, factor),
            AxisConstraint::Axis(Axis::X, _) => Vector3::new(factor, 1.0, 1.0),
            AxisConstraint::Axis(Axis::Y, _) => Vector3::new(1.0, factor, 1.0),
            AxisConstraint::Axis(Axis::Z, _) => Vector3::new(1.0, 1.0, factor),
            AxisConstraint::Plane(Axis::X, _) => Vector3::new(1.0, factor, factor),
            AxisConstraint::Plane(Axis::Y, _) => Vector3::new(factor, 1.0, factor),
            AxisConstraint::Plane(Axis::Z, _) => Vector3::new(factor, factor, 1.0),
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
                let s = self.scale(camera, size);
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
