//! Selection-agnostic core of an interactive transform: input bindings, the
//! axis-constraint state machine, and the mouse-delta → transform math.
//!
//! [`TransformInteraction`] knows nothing about what is being transformed —
//! consumers start it about a pivot, feed it mouse deltas, and read back the
//! resulting translation/rotation/scale (or the composed [`Matrix4`] delta)
//! to apply however they see fit.

use duck_engine_common::{
    EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Vector3, Zero, EPSILON,
};
use duck_engine_scene::common::Ray;
use duck_engine_scene::camera::PositionedCamera;
use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::common::{
    local_axis_x, local_axis_y, local_axis_z, quaternion_from_axis_angle_safe, Axis, RgbaColor,
};
use crate::input::{Key, Modifiers, MouseButton, NamedKey};

use super::drag::DragGeometry;
use super::gizmo::{GizmoHandleId, GizmoType};

/// Rotation rate used only when the rotation plane is too edge-on to solve an
/// angle against.
const ROTATE_DEGREES_PER_PIXEL: f32 = 0.5;

/// Scale rate used only when a drag cannot be resolved into a distance ratio
/// (the drag was anchored on the pivot, or the geometry is degenerate).
const SCALE_FRACTION_PER_PIXEL: f32 = 0.005;

/// Minimum anchor-to-pivot screen distance for scaling by distance ratio.
/// Closer in, the ratio is hypersensitive — a few pixels would double the scale
/// — so the drag uses [`SCALE_FRACTION_PER_PIXEL`] instead.
const MIN_SCALE_ANCHOR_PIXELS: f32 = 24.0;

/// Wraps an angle into `(-π, π]`, so an unwrapped difference between successive
/// solved angles takes the short way round.
fn wrap_to_pi(angle: f32) -> f32 {
    use std::f32::consts::{PI, TAU};
    let wrapped = (angle + PI).rem_euclid(TAU);
    wrapped - PI
}

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

    /// Pivot point in world space (center of rotation/scale).
    pivot_world: Point3,

    /// Orientation of the local constraint axes.
    frame_rotation: Quaternion,

    /// Screen pixel the drag is anchored at: the grabbed handle's pixel, or the
    /// cursor when the transform was started from the keyboard.
    anchor_screen: (f32, f32),

    /// Where the drag has reached, in the same screen pixels as the anchor.
    cursor_screen: (f32, f32),

    /// Whether an absolute cursor update landed since the last relative one, in
    /// which case that relative delta is redundant and must be dropped.
    absolute_since_motion: bool,

    /// Total angle swept since the drag began, integrated across events so it
    /// can exceed a full turn. Rotate mode only.
    swept_angle: f32,

    /// The last solved angle the sweep was unwrapped against. Rotate mode only.
    last_solved_angle: f32,

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
            pivot_world: Point3::origin(),
            frame_rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            anchor_screen: (0.0, 0.0),
            cursor_screen: (0.0, 0.0),
            absolute_since_motion: false,
            swept_angle: 0.0,
            last_solved_angle: 0.0,
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
    /// keeps the previous frame); `anchor_screen` is the pixel the drag is
    /// measured from.
    pub fn start(
        &mut self,
        pivot: Point3,
        frame_rotation: Option<Quaternion>,
        anchor_screen: (f32, f32),
    ) {
        self.pivot_world = pivot;
        if let Some(rotation) = frame_rotation {
            self.frame_rotation = rotation;
        }
        self.anchor_screen = anchor_screen;
        self.cursor_screen = anchor_screen;
        self.absolute_since_motion = false;
        self.active = true;
        self.axis_constraint = AxisConstraint::None;
        self.reset_rotation();
    }

    /// Deactivate and clear the constraint and accumulated movement (after a
    /// confirm or cancel).
    pub fn finish(&mut self) {
        self.active = false;
        self.axis_constraint = AxisConstraint::None;
        self.cursor_screen = self.anchor_screen;
        self.absolute_since_motion = false;
        self.reset_rotation();
    }

    /// Move the drag to an absolute cursor position.
    ///
    /// Authoritative: this is the pixel the user sees the pointer at, so it is
    /// what the grabbed geometry has to line up with.
    pub fn set_cursor(
        &mut self,
        position: (f32, f32),
        camera: &PositionedCamera,
        size: (u32, u32),
    ) {
        let dx = position.0 - self.cursor_screen.0;
        self.cursor_screen = position;
        self.absolute_since_motion = true;
        if self.mode == TransformMode::Rotate {
            self.integrate_rotation(dx, camera, size);
        }
    }

    /// Advance the drag by relative mouse movement.
    ///
    /// Only takes effect when no absolute position accompanied this motion,
    /// which is how a drag keeps working once the pointer leaves the window and
    /// `CursorMoved` stops arriving. See [`Self::set_cursor`].
    pub fn accumulate(&mut self, dx: f32, dy: f32, camera: &PositionedCamera, size: (u32, u32)) {
        if self.absolute_since_motion {
            self.absolute_since_motion = false;
            return;
        }
        self.cursor_screen.0 += dx;
        self.cursor_screen.1 += dy;
        if self.mode == TransformMode::Rotate {
            self.integrate_rotation(dx, camera, size);
        }
    }

    /// Screen-space displacement of the drag so far.
    fn drag_delta(&self) -> (f32, f32) {
        (self.cursor_screen.0 - self.anchor_screen.0, self.cursor_screen.1 - self.anchor_screen.1)
    }

    pub fn pivot(&self) -> Point3 {
        self.pivot_world
    }

    pub fn frame_rotation(&self) -> Quaternion {
        self.frame_rotation
    }

    pub fn axis_constraint(&self) -> AxisConstraint {
        self.axis_constraint
    }

    pub fn set_axis_constraint(&mut self, constraint: AxisConstraint) {
        self.axis_constraint = constraint;
    }

    /// Constrain to the axis or plane a gizmo handle represents.
    pub fn constrain_to_handle(&mut self, id: GizmoHandleId) {
        self.axis_constraint = match id {
            GizmoHandleId::Axis(axis) => AxisConstraint::Axis(axis, ConstraintSpace::World),
            GizmoHandleId::Plane(normal) => AxisConstraint::Plane(normal, ConstraintSpace::World),
            GizmoHandleId::Ball => AxisConstraint::None,
        };
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

    /// The pair of rays a drag is measured between: through the anchor pixel,
    /// and through the anchor pixel offset by the accumulated motion.
    ///
    /// Relative motion rather than the live cursor, so a drag that leaves the
    /// window keeps going.
    fn drag_rays(&self, camera: &PositionedCamera, size: (u32, u32)) -> (Ray, Ray) {
        let (width, height) = size;
        let (ax, ay) = self.anchor_screen;
        let (cx, cy) = self.cursor_screen;
        (
            camera.ray_from_screen_point(ax, ay, width, height),
            camera.ray_from_screen_point(cx, cy, width, height),
        )
    }

    /// The plane parallel to the image plane through the pivot.
    fn view_plane(&self, camera: &PositionedCamera) -> DragGeometry {
        DragGeometry::plane(camera.forward(), self.pivot_world)
    }

    /// The locus the drag point is confined to by the current constraint.
    fn drag_geometry(&self, camera: &PositionedCamera) -> DragGeometry {
        match self.axis_constraint {
            AxisConstraint::None => self.view_plane(camera),
            AxisConstraint::Axis(axis, space) => {
                DragGeometry::axis(self.pivot_world, self.axis_direction(axis, space))
            }
            AxisConstraint::Plane(axis, space) => {
                DragGeometry::plane(self.axis_direction(axis, space), self.pivot_world)
            }
        }
    }

    /// How far the drag point moved across `geometry` between the two rays.
    /// `None` when either end of the solve is degenerate.
    fn solve_delta(geometry: &DragGeometry, (anchor, cursor): &(Ray, Ray)) -> Option<Vector3> {
        Some(geometry.solve(cursor)? - geometry.solve(anchor)?)
    }

    /// Slides the unconstrained drag onto the constraint by dropping whatever
    /// part of it the constraint disallows.
    ///
    /// Always undershoots, and badly for an axis angled away from the camera:
    /// such an axis looks shortened on screen, so keeping up with the cursor
    /// along it takes *more* world movement than the cursor covered, while
    /// dropping a component can only ever give less. Used only when the exact
    /// solve is unavailable — moving a little beats freezing.
    fn projected_translation(&self, view_delta: Vector3) -> Vector3 {
        match self.axis_constraint {
            AxisConstraint::None => view_delta,
            AxisConstraint::Axis(..) => {
                let axis = self.constraint_axis().unwrap();
                axis * axis.dot(view_delta)
            }
            AxisConstraint::Plane(..) => {
                let normal = self.constraint_plane_normal().unwrap();
                view_delta - normal * normal.dot(view_delta)
            }
        }
    }

    /// Compute the translation delta based on mouse movement and constraints.
    /// `size` is the viewport size in pixels.
    pub fn translation(&self, camera: &PositionedCamera, size: (u32, u32)) -> Vector3 {
        let rays = self.drag_rays(camera, size);

        if let Some(exact) = Self::solve_delta(&self.drag_geometry(camera), &rays)
            && exact.magnitude().is_finite()
        {
            return exact;
        }

        // Degradation path: project the always-solvable view-plane drag onto the
        // constraint. Loses cos²θ of the drag, but keeps a drag alive that would
        // otherwise freeze.
        Self::solve_delta(&self.view_plane(camera), &rays)
            .map(|reference| self.projected_translation(reference))
            .unwrap_or_else(Vector3::zero)
    }

    /// The world-space axis the rotation turns about: the constraint axis, or
    /// the view direction when unconstrained.
    fn rotation_axis(&self, camera: &PositionedCamera) -> Vector3 {
        self.constraint_axis().unwrap_or_else(|| camera.forward())
    }

    /// The angle the cursor has swept about the pivot since the drag anchor.
    /// 
    /// Measured in the rotation plane. `None` when that plane is too close to
    /// edge-on, or the cursor is effectively on the pivot, for the angle to
    /// mean anything.
    ///
    /// Measured from the anchor, so it is zero at drag start — which is what
    /// lets [`Self::start`] seed the winding without a camera.
    fn solved_rotation_angle(&self, camera: &PositionedCamera, size: (u32, u32)) -> Option<f32> {
        let axis = self.rotation_axis(camera);
        if axis.magnitude2() < EPSILON {
            return None;
        }
        let axis = axis.normalize();

        let geometry = DragGeometry::plane(axis, self.pivot_world);
        let (anchor_ray, cursor_ray) = self.drag_rays(camera, size);
        let anchor = geometry.solve(&anchor_ray)? - self.pivot_world;
        let cursor = geometry.solve(&cursor_ray)? - self.pivot_world;
        if anchor.magnitude2() < EPSILON || cursor.magnitude2() < EPSILON {
            return None;
        }

        Some(axis.dot(anchor.cross(cursor)).atan2(anchor.dot(cursor)))
    }

    /// Fold this event's motion into the swept rotation angle.
    ///
    /// The swept angle has to be integrated rather than recomputed: a solved
    /// angle is only defined modulo a full turn, and a rotate handle gets spun
    /// well past half a turn. Each event's solved angle is unwrapped against
    /// the previous one, which is unambiguous at per-event motion sizes.
    fn integrate_rotation(&mut self, dx: f32, camera: &PositionedCamera, size: (u32, u32)) {
        match self.solved_rotation_angle(camera, size) {
            Some(solved) => {
                self.swept_angle += wrap_to_pi(solved - self.last_solved_angle);
                self.last_solved_angle = solved;
            }
            // Rotation plane edge-on: no angle to solve, so fall back to a
            // pixel rate for this event only. Integrating makes that continuous
            // — the drag slows down rather than freezing or popping.
            None => self.swept_angle += dx * ROTATE_DEGREES_PER_PIXEL.to_radians(),
        }
    }

    /// Discard the swept rotation.
    /// 
    /// Called when the constraint changes mid-drag:
    /// the axis itself moved, so the angle swept about the old one is
    /// meaningless.
    pub fn reset_rotation(&mut self) {
        self.swept_angle = 0.0;
        self.last_solved_angle = 0.0;
    }

    /// Compute the rotation based on mouse movement and constraints.
    pub fn rotation(&self, camera: &PositionedCamera) -> Quaternion {
        quaternion_from_axis_angle_safe(self.rotation_axis(camera), self.swept_angle)
    }

    /// The scale factor as the ratio of the cursor's distance from the pivot to
    /// the anchor's, measured on the constraint geometry — so the grabbed point
    /// tracks the cursor.
    ///
    /// `None` when the ratio is meaningless or hypersensitive: the geometry is
    /// degenerate, or the drag was anchored too close to the pivot, where a few
    /// pixels would double the scale. The anchor distance is fixed for the whole
    /// drag, so this decision never flips partway through.
    fn solved_scale_factor(&self, camera: &PositionedCamera, size: (u32, u32)) -> Option<f32> {
        let (width, height) = size;
        let projected_pivot = camera.project_point_screen(self.pivot_world, width, height);
        let (ax, ay) = self.anchor_screen;
        let anchor_pixels = ((ax - projected_pivot.x).hypot(ay - projected_pivot.y)).abs();
        if anchor_pixels < MIN_SCALE_ANCHOR_PIXELS {
            return None;
        }

        let geometry = self.drag_geometry(camera);
        let (anchor_ray, cursor_ray) = self.drag_rays(camera, size);
        let anchor = geometry.solve(&anchor_ray)? - self.pivot_world;
        let cursor = geometry.solve(&cursor_ray)? - self.pivot_world;

        match self.axis_constraint {
            // Signed along the axis, so dragging back through the pivot shrinks
            // toward zero and then trips the clamp below rather than mirroring.
            AxisConstraint::Axis(..) => {
                let axis = self.constraint_axis()?;
                if axis.magnitude2() < EPSILON {
                    return None;
                }
                let axis = axis.normalize();
                let anchor_distance = anchor.dot(axis);
                if anchor_distance.abs() < EPSILON {
                    return None;
                }
                Some(cursor.dot(axis) / anchor_distance)
            }
            // Plane and uniform have no single axis to measure along, so use
            // radial distance: dragging outward grows, inward shrinks.
            _ => {
                let anchor_distance = anchor.magnitude();
                if anchor_distance < EPSILON {
                    return None;
                }
                Some(cursor.magnitude() / anchor_distance)
            }
        }
    }

    /// Pixel drag magnitude for when no distance ratio can be solved.
    ///
    /// Grabbing the gizmo's center ball anchors within a few pixels of the
    /// pivot, so uniform scale reaches this path routinely rather than only in
    /// degenerate cases — it has to stay usable in every drag direction.
    fn fallback_scale_magnitude(&self, camera: &PositionedCamera, size: (u32, u32)) -> f32 {
        let (dx, dy) = self.drag_delta();
        match self.axis_constraint {
            // A single axis already has a direction to measure along.
            AxisConstraint::Axis(..) => dx,
            // Nothing to measure along, so use the signed change in distance
            // from the pivot: dragging outward grows, inward shrinks.
            _ => {
                let projected = camera.project_point_screen(self.pivot_world, size.0, size.1);
                let (ax, ay) = self.anchor_screen;
                let anchor_distance = (ax - projected.x).hypot(ay - projected.y);
                let cursor_distance =
                    (ax + dx - projected.x).hypot(ay + dy - projected.y);
                cursor_distance - anchor_distance
            }
        }
    }

    /// Compute the scale factor based on mouse movement and constraints.
    /// `size` is the viewport size in pixels.
    pub fn scale(&self, camera: &PositionedCamera, size: (u32, u32)) -> Vector3 {
        let factor = self.solved_scale_factor(camera, size).unwrap_or_else(|| {
            1.0 + self.fallback_scale_magnitude(camera, size) * SCALE_FRACTION_PER_PIXEL
        });
        // Clamp to prevent negative or zero scale
        let factor = factor.max(0.01);

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

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::Plane;
    use std::f32::consts::PI;

    const EPSILON: f32 = 1e-6;
    /// Pixel round-trips project and unproject through separate f32 matrix
    /// inversions, so they hold to a small fraction of a pixel rather than to
    /// `EPSILON`. Still three orders of magnitude tighter than the errors these
    /// tests exist to catch, which run to tens of pixels.
    const PIXEL_EPSILON: f32 = 0.01;
    /// Tolerance for comparing world vectors computed by two different but
    /// mathematically equivalent routes; the magnitudes here are order 1.
    const WORLD_EPSILON: f32 = 1e-4;
    const SIZE: (u32, u32) = (800, 800);
    const CENTER: (f32, f32) = (400.0, 400.0);

    fn camera_at(eye: (f32, f32, f32), ortho: bool) -> PositionedCamera {
        PositionedCamera {
            eye: eye.into(),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho,
        }
    }

    /// An interaction mid-drag about the origin, anchored at `anchor`.
    fn started(
        mode: TransformMode,
        anchor: (f32, f32),
        constraint: AxisConstraint,
    ) -> TransformInteraction {
        let mut interaction = TransformInteraction::new(mode);
        interaction.start(Point3::new(0.0, 0.0, 0.0), None, anchor);
        interaction.set_axis_constraint(constraint);
        interaction
    }

    fn projected_pivot(camera: &PositionedCamera) -> (f32, f32) {
        let p = camera.project_point_screen(Point3::new(0.0, 0.0, 0.0), SIZE.0, SIZE.1);
        (p.x, p.y)
    }

    /// Where the pivot ends up on screen after applying `translation`. Anchoring
    /// at the pivot's own projection makes the pivot the grabbed point, so this
    /// should land exactly on the virtual cursor.
    fn pivot_pixel_after(
        interaction: &TransformInteraction,
        camera: &PositionedCamera,
    ) -> (f32, f32) {
        let moved = Point3::new(0.0, 0.0, 0.0) + interaction.translation(camera, SIZE);
        let p = camera.project_point_screen(moved, SIZE.0, SIZE.1);
        (p.x, p.y)
    }

    fn assert_pixels_close(actual: (f32, f32), expected: (f32, f32), what: &str) {
        assert!(
            (actual.0 - expected.0).abs() < PIXEL_EPSILON
                && (actual.1 - expected.1).abs() < PIXEL_EPSILON,
            "{what}: expected {expected:?}, got {actual:?}"
        );
    }

    /// The unit screen-space direction of a world axis through the pivot.
    fn projected_axis_direction(camera: &PositionedCamera, axis: Vector3) -> (f32, f32) {
        let origin = camera.project_point_screen(Point3::new(0.0, 0.0, 0.0), SIZE.0, SIZE.1);
        let tip = camera.project_point_screen(Point3::new(0.0, 0.0, 0.0) + axis, SIZE.0, SIZE.1);
        let (ex, ey) = (tip.x - origin.x, tip.y - origin.y);
        let len = (ex * ex + ey * ey).sqrt();
        (ex / len, ey / len)
    }

    // ── Translation ──────────────────────────────────────────────────────────

    #[test]
    fn foreshortened_axis_drag_tracks_cursor_in_ortho() {
        // The headline property, which projecting the drag onto the axis fails
        // by cos²θ. Under an orthographic camera the 3D closest approach reduces
        // exactly to the screen-space perpendicular foot of the cursor on the
        // projected axis, so this is an equality, not an approximation.
        let camera = camera_at((8.0, 2.0, 2.0), true);
        let anchor = projected_pivot(&camera);
        let mut interaction =
            started(TransformMode::Translate, anchor, AxisConstraint::Axis(Axis::X, ConstraintSpace::World));

        let (ex, ey) = projected_axis_direction(&camera, Vector3::unit_x());
        let drag = 120.0;
        interaction.accumulate(ex * drag, ey * drag, &camera, SIZE);

        assert_pixels_close(
            pivot_pixel_after(&interaction, &camera),
            (anchor.0 + ex * drag, anchor.1 + ey * drag),
            "grabbed point should stay under the cursor",
        );

        let translation = interaction.translation(&camera, SIZE);
        assert!(translation.y.abs() < EPSILON && translation.z.abs() < EPSILON);
        assert!(translation.x > 0.0);
    }

    #[test]
    fn foreshortened_axis_drag_beats_projected_math() {
        // Documents the defect the solve replaces: the retained projection
        // fallback loses cos²θ of the drag on a steeply foreshortened axis.
        let camera = camera_at((8.0, 2.0, 2.0), true);
        let anchor = projected_pivot(&camera);
        let mut interaction =
            started(TransformMode::Translate, anchor, AxisConstraint::Axis(Axis::X, ConstraintSpace::World));

        let (ex, ey) = projected_axis_direction(&camera, Vector3::unit_x());
        interaction.accumulate(ex * 120.0, ey * 120.0, &camera, SIZE);

        let rays = interaction.drag_rays(&camera, SIZE);
        let reference =
            TransformInteraction::solve_delta(&interaction.view_plane(&camera), &rays).unwrap();
        let projected = interaction.projected_translation(reference).magnitude();
        let solved = interaction.translation(&camera, SIZE).magnitude();

        assert!(solved > projected * 1.5, "solved {solved}, projected {projected}");
    }

    #[test]
    fn plane_constrained_drag_keeps_grabbed_point_under_cursor() {
        for ortho in [false, true] {
            let camera = camera_at((6.0, 5.0, 7.0), ortho);
            let anchor = projected_pivot(&camera);
            let mut interaction = started(
                TransformMode::Translate,
                anchor,
                AxisConstraint::Plane(Axis::Y, ConstraintSpace::World),
            );

            let (dx, dy) = (90.0, -70.0);
            interaction.accumulate(dx, dy, &camera, SIZE);

            assert_pixels_close(
                pivot_pixel_after(&interaction, &camera),
                (anchor.0 + dx, anchor.1 + dy),
                if ortho { "ortho plane drag" } else { "perspective plane drag" },
            );
            // Plane(Y) excludes Y, so the drag stays in the XZ plane.
            assert!(interaction.translation(&camera, SIZE).y.abs() < EPSILON);
        }
    }

    #[test]
    fn unconstrained_translation_matches_pivot_projection_math() {
        // Freeform translate must not regress: the view plane is parallel to the
        // image plane, so the solve reproduces the previous pivot-projection
        // formula exactly.
        let camera = camera_at((3.0, 4.0, 9.0), false);
        let anchor = projected_pivot(&camera);
        let mut interaction = started(TransformMode::Translate, anchor, AxisConstraint::None);
        let (dx, dy) = (57.0, -31.0);
        interaction.accumulate(dx, dy, &camera, SIZE);

        let pivot = Point3::new(0.0, 0.0, 0.0);
        let plane = Plane::from_point(camera.forward(), pivot);
        let ray = camera.ray_from_screen_point(anchor.0 + dx, anchor.1 + dy, SIZE.0, SIZE.1);
        let expected = ray.intersect_plane(&plane).unwrap().1 - pivot;

        let actual = interaction.translation(&camera, SIZE);
        assert!((actual - expected).magnitude() < WORLD_EPSILON, "{actual:?} vs {expected:?}");
    }

    #[test]
    fn unconstrained_translation_is_anchor_independent() {
        // World units per pixel are constant across a plane parallel to the
        // image plane, so where the freeform drag was anchored cannot matter.
        let camera = camera_at((3.0, 4.0, 9.0), false);
        let (dx, dy) = (44.0, 61.0);

        let mut near_pivot = started(TransformMode::Translate, projected_pivot(&camera), AxisConstraint::None);
        near_pivot.accumulate(dx, dy, &camera, SIZE);
        let mut far_corner = started(TransformMode::Translate, (60.0, 730.0), AxisConstraint::None);
        far_corner.accumulate(dx, dy, &camera, SIZE);

        let a = near_pivot.translation(&camera, SIZE);
        let b = far_corner.translation(&camera, SIZE);
        assert!((a - b).magnitude() < WORLD_EPSILON, "{a:?} vs {b:?}");
    }

    #[test]
    fn zero_delta_yields_zero_translation() {
        // No motion must mean no motion, however far the anchor is from the
        // pivot — the difference-of-solves form guarantees it, so grabbing an
        // arrow tip never makes the target jump.
        let camera = camera_at((5.0, 4.0, 6.0), false);
        for constraint in [
            AxisConstraint::None,
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
            AxisConstraint::Plane(Axis::Z, ConstraintSpace::World),
        ] {
            let interaction = started(TransformMode::Translate, (612.0, 208.0), constraint);
            let translation = interaction.translation(&camera, SIZE);
            assert!(translation.magnitude() < EPSILON, "{constraint:?} gave {translation:?}");
        }
    }

    #[test]
    fn changing_constraint_mid_drag_re_solves_from_anchor() {
        // Nothing caches a delta: the constraint selects the geometry and the
        // drag is always re-solved from the anchor.
        let camera = camera_at((5.0, 4.0, 6.0), false);
        let mut interaction = started(
            TransformMode::Translate,
            projected_pivot(&camera),
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
        );
        interaction.accumulate(83.0, -44.0, &camera, SIZE);

        let first = interaction.translation(&camera, SIZE);
        interaction.set_axis_constraint(AxisConstraint::Plane(Axis::Y, ConstraintSpace::World));
        let plane_constrained = interaction.translation(&camera, SIZE);
        interaction.set_axis_constraint(AxisConstraint::Axis(Axis::X, ConstraintSpace::World));
        let again = interaction.translation(&camera, SIZE);

        assert!((first - again).magnitude() < EPSILON);
        assert!((first - plane_constrained).magnitude() > EPSILON, "constraints should differ");
    }

    #[test]
    fn ground_plane_drag_toward_the_horizon_is_not_capped() {
        // Regression: a ground-plane drag toward the vanishing line needs large
        // amplification to keep up with the cursor. Bounding the solve against
        // the view-plane drag left the target short of the cursor, further short
        // the further it was dragged.
        let camera = camera_at((0.0, 4.0, 9.0), false);
        let anchor = projected_pivot(&camera);
        let mut interaction = started(
            TransformMode::Translate,
            anchor,
            AxisConstraint::Plane(Axis::Y, ConstraintSpace::World),
        );
        // Drag most of the way up the viewport, well past where a gain cap bites
        // but still below the (off-screen) horizon.
        let (dx, dy) = (-140.0, -330.0);
        interaction.accumulate(dx, dy, &camera, SIZE);

        assert_pixels_close(
            pivot_pixel_after(&interaction, &camera),
            (anchor.0 + dx, anchor.1 + dy),
            "grabbed point should stay under the cursor near the horizon",
        );

        // The amplification this needs is far beyond any plausible cap, which is
        // why the solve must stay unbounded.
        let rays = interaction.drag_rays(&camera, SIZE);
        let reference =
            TransformInteraction::solve_delta(&interaction.view_plane(&camera), &rays).unwrap();
        let gain = interaction.translation(&camera, SIZE).magnitude() / reference.magnitude();
        assert!(gain > 8.0, "expected a large gain, got {gain}");
    }

    #[test]
    fn degenerate_axis_translation_stays_finite_and_on_axis() {
        // Axis pointing almost straight at the camera. The solve is inherently
        // sensitive there, but it must stay finite and confined to the axis.
        let camera = camera_at((10.0, 0.02, 0.03), false);
        let mut interaction = started(
            TransformMode::Translate,
            projected_pivot(&camera),
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
        );
        interaction.accumulate(120.0, 65.0, &camera, SIZE);

        let translation = interaction.translation(&camera, SIZE);
        assert!(translation.magnitude().is_finite(), "{translation:?}");
        assert!(translation.y.abs() < EPSILON && translation.z.abs() < EPSILON);
    }

    #[test]
    fn plane_constraint_past_horizon_falls_back_instead_of_freezing() {
        // Dragging above a near-level constraint plane's horizon leaves nothing
        // to intersect. The previous code silently froze; the fallback keeps the
        // drag alive and in-plane.
        let camera = camera_at((0.0, 0.5, 10.0), false);
        let mut interaction = started(
            TransformMode::Translate,
            projected_pivot(&camera),
            AxisConstraint::Plane(Axis::Y, ConstraintSpace::World),
        );
        interaction.accumulate(150.0, -600.0, &camera, SIZE);

        let rays = interaction.drag_rays(&camera, SIZE);
        assert!(
            TransformInteraction::solve_delta(&interaction.drag_geometry(&camera), &rays).is_none(),
            "expected the constraint solve to fail past the horizon"
        );

        let translation = interaction.translation(&camera, SIZE);
        assert!(translation.magnitude() > EPSILON, "drag froze: {translation:?}");
        assert!(translation.magnitude().is_finite());
        assert!(translation.y.abs() < EPSILON, "should stay in the XZ plane");
    }

    #[test]
    fn local_axis_constraint_follows_frame_rotation() {
        let camera = camera_at((4.0, 5.0, 8.0), false);
        let rotation = quaternion_from_axis_angle_safe(Vector3::unit_y(), PI / 4.0);
        let mut interaction = TransformInteraction::new(TransformMode::Translate);
        interaction.start(Point3::new(0.0, 0.0, 0.0), Some(rotation), projected_pivot(&camera));
        interaction.set_axis_constraint(AxisConstraint::Axis(Axis::X, ConstraintSpace::Local));
        interaction.accumulate(70.0, 25.0, &camera, SIZE);

        let translation = interaction.translation(&camera, SIZE);
        let local_x = local_axis_x(rotation);
        assert!(translation.magnitude() > EPSILON);
        assert!(
            translation.normalize().cross(local_x).magnitude() < 1e-4,
            "{translation:?} should lie along the rotated X axis {local_x:?}"
        );
    }

    // ── Cursor tracking ──────────────────────────────────────────────────────

    #[test]
    fn absolute_cursor_position_overrides_the_motion_delta() {
        // The core of the tracking fix: `MouseMotion` reports raw device
        // movement, which pointer acceleration makes a different quantity from
        // on-screen pixels. When both describe the same movement the absolute
        // position wins, in either arrival order.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let anchor = (300.0, 300.0);
        let truth = (500.0, 380.0);
        let lying_delta = (150.0, 60.0); // 0.75x the real movement

        // Absolute first, then the redundant delta.
        let mut absolute_first = started(TransformMode::Translate, anchor, AxisConstraint::None);
        absolute_first.set_cursor(truth, &camera, SIZE);
        absolute_first.accumulate(lying_delta.0, lying_delta.1, &camera, SIZE);
        assert_eq!(absolute_first.cursor_screen, truth);

        // Delta first, then the authoritative absolute.
        let mut motion_first = started(TransformMode::Translate, anchor, AxisConstraint::None);
        motion_first.accumulate(lying_delta.0, lying_delta.1, &camera, SIZE);
        motion_first.set_cursor(truth, &camera, SIZE);
        assert_eq!(motion_first.cursor_screen, truth);
    }

    #[test]
    fn a_stream_of_paired_events_does_not_drift() {
        // Repeated absolute/relative pairs must not accumulate the delta's error,
        // which is what left the target progressively further behind the cursor.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let anchor = (200.0, 200.0);
        let mut interaction = started(TransformMode::Translate, anchor, AxisConstraint::None);

        let mut truth = anchor;
        for _ in 0..40 {
            truth = (truth.0 + 10.0, truth.1 + 4.0);
            interaction.set_cursor(truth, &camera, SIZE);
            // The device reports only three quarters of the real movement.
            interaction.accumulate(7.5, 3.0, &camera, SIZE);
        }

        assert_eq!(interaction.cursor_screen, truth);
        assert_eq!(interaction.drag_delta(), (truth.0 - anchor.0, truth.1 - anchor.1));
    }

    #[test]
    fn relative_motion_carries_the_drag_with_no_absolute_updates() {
        // Off-window, `CursorMoved` stops arriving and deltas are all there is;
        // the drag has to keep going.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let anchor = (400.0, 400.0);
        let mut interaction = started(TransformMode::Translate, anchor, AxisConstraint::None);

        interaction.accumulate(-30.0, 12.0, &camera, SIZE);
        interaction.accumulate(-30.0, 12.0, &camera, SIZE);

        assert_eq!(interaction.cursor_screen, (anchor.0 - 60.0, anchor.1 + 24.0));
    }

    // ── Rotation ─────────────────────────────────────────────────────────────

    #[test]
    fn rotation_winds_past_half_turn() {
        // A solved angle is only defined modulo a turn, so the sweep is
        // integrated. Walking the cursor 270° around the pivot must accumulate
        // 270°, not wrap back at 180°.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let radius = 100.0;
        let mut previous = (CENTER.0 + radius, CENTER.1);
        let mut interaction = started(TransformMode::Rotate, previous, AxisConstraint::None);

        let mut last = 0.0_f32;
        let steps = 27;
        for step in 1..=steps {
            let theta = (step as f32) * 10.0_f32.to_radians();
            let position = (CENTER.0 + radius * theta.cos(), CENTER.1 + radius * theta.sin());
            interaction.accumulate(position.0 - previous.0, position.1 - previous.1, &camera, SIZE);
            previous = position;

            let swept = interaction.swept_angle.abs();
            assert!(swept > last, "sweep should grow monotonically at step {step}");
            last = swept;
        }

        let expected = (steps as f32) * 10.0_f32.to_radians();
        assert!(last > PI, "sweep wrapped at half a turn: {last}");
        assert!((last - expected).abs() < 0.02, "expected {expected}, got {last}");
    }

    #[test]
    fn rotation_resets_on_constraint_change() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction =
            started(TransformMode::Rotate, (CENTER.0 + 120.0, CENTER.1), AxisConstraint::None);
        interaction.accumulate(0.0, 60.0, &camera, SIZE);
        assert!(interaction.swept_angle.abs() > EPSILON);

        interaction.reset_rotation();
        assert!(interaction.swept_angle.abs() < EPSILON);
        // Identity rotation: w == 1.
        assert!((interaction.rotation(&camera).s - 1.0).abs() < EPSILON);
    }

    #[test]
    fn rotation_falls_back_to_pixel_rate_when_anchored_on_pivot() {
        // No radius means no angle to solve, so the drag degrades to the pixel
        // rate rather than producing a wild or zero rotation.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction = started(TransformMode::Rotate, CENTER, AxisConstraint::None);
        interaction.accumulate(100.0, 0.0, &camera, SIZE);

        let expected = 100.0 * ROTATE_DEGREES_PER_PIXEL.to_radians();
        assert!((interaction.swept_angle - expected).abs() < EPSILON);
    }

    // ── Scale ────────────────────────────────────────────────────────────────

    #[test]
    fn scale_factor_is_ratio_of_cursor_distances() {
        // Anchored 100 px along the projected X axis and dragged to 200 px, so
        // the grabbed point is twice as far from the pivot: exactly 2×.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction = started(
            TransformMode::Scale,
            (CENTER.0 + 100.0, CENTER.1),
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
        );
        interaction.accumulate(100.0, 0.0, &camera, SIZE);

        let scale = interaction.scale(&camera, SIZE);
        assert!((scale.x - 2.0).abs() < 1e-4, "{scale:?}");
        assert!((scale.y - 1.0).abs() < EPSILON && (scale.z - 1.0).abs() < EPSILON);
    }

    #[test]
    fn uniform_scale_factor_is_ratio_of_radial_distances() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction =
            started(TransformMode::Scale, (CENTER.0 + 120.0, CENTER.1), AxisConstraint::None);
        interaction.accumulate(60.0, 0.0, &camera, SIZE);

        let scale = interaction.scale(&camera, SIZE);
        assert!((scale.x - 1.5).abs() < 1e-4, "{scale:?}");
        assert!((scale.x - scale.y).abs() < EPSILON && (scale.y - scale.z).abs() < EPSILON);
    }

    #[test]
    fn scale_anchored_on_pivot_falls_back_to_pixel_rate() {
        // A ratio taken a pixel from the pivot would be hypersensitive, so a
        // drag anchored there uses the pixel rate for its whole duration.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction = started(
            TransformMode::Scale,
            projected_pivot(&camera),
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
        );
        interaction.accumulate(100.0, 0.0, &camera, SIZE);

        let expected = 1.0 + 100.0 * SCALE_FRACTION_PER_PIXEL;
        assert!((interaction.scale(&camera, SIZE).x - expected).abs() < EPSILON);
    }

    #[test]
    fn uniform_scale_from_the_center_ball_responds_in_any_direction() {
        // The gizmo's center ball sits on the pivot, so uniform scale takes the
        // fallback path routinely. It must stay radial there: dragging away from
        // the pivot grows whichever way the cursor goes, and back in shrinks.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let outward: Vec<f32> = [(90.0, 0.0), (0.0, 90.0), (-90.0, 0.0), (0.0, -90.0)]
            .into_iter()
            .map(|(dx, dy)| {
                let mut interaction =
                    started(TransformMode::Scale, projected_pivot(&camera), AxisConstraint::None);
                interaction.accumulate(dx, dy, &camera, SIZE);
                interaction.scale(&camera, SIZE).x
            })
            .collect();

        let expected = 1.0 + 90.0 * SCALE_FRACTION_PER_PIXEL;
        for factor in outward {
            assert!((factor - expected).abs() < 1e-4, "expected {expected}, got {factor}");
        }
    }

    #[test]
    fn scale_never_goes_negative() {
        // Dragging back through the pivot must collapse toward zero, not mirror.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut interaction = started(
            TransformMode::Scale,
            (CENTER.0 + 100.0, CENTER.1),
            AxisConstraint::Axis(Axis::X, ConstraintSpace::World),
        );
        interaction.accumulate(-400.0, 0.0, &camera, SIZE);

        let scale = interaction.scale(&camera, SIZE);
        assert!(scale.x > 0.0, "{scale:?}");
    }
}
