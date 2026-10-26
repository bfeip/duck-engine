//! One move, rotation or scale: its constraint, what drives it, and the delta
//! that comes out.
//!
//! A [`Manipulation`] is held by a tool and fed its events. It answers with an
//! [`Outcome`] the tool acts on, and with the gizmo [`Handle`]s to show; it
//! never touches what is being transformed.
//!
//! Either driver reduces to a pair of points on the constraint's locus: where
//! the drag began, and where it has reached. A gizmo grip gets the pair from the
//! handle machinery; the cursor, after a key starts a transform, solves its own
//! rays for it.

use duck_engine_viewer::common::{
    consts, quaternion_from_axis_angle_safe, Axis, EuclideanSpace, InnerSpace, Matrix4, Point3,
    Quaternion, Real, Rotation, SquareMatrix, Vector3, Zero, EPSILON,
};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers, MouseButton, NamedKey};
use duck_engine_viewer::operator::{solve_drag, DragGeometry, Handle, HandleEvent, HandleId};
use duck_engine_viewer::scene::PositionedCamera;

use super::gizmo::{self, Grip};

/// Rotation rate for an event whose rotation plane is too edge-on to solve an
/// angle against.
const ROTATE_DEGREES_PER_PIXEL: Real = 0.5;

/// Scale rate for a drag that started too close to the pivot to scale by a
/// distance ratio.
const SCALE_FRACTION_PER_PIXEL: Real = 0.005;

/// Closest a drag may start to the pivot, on screen, and still scale by
/// distance ratio. Any closer and the ratio is hypersensitive: a few pixels
/// would double the scale.
const MIN_SCALE_ANCHOR_PIXELS: f32 = 24.0;

/// Smallest scale factor, so a scale dragged back through the pivot collapses
/// rather than mirrors.
const MIN_SCALE: Real = 0.01;

/// What a transform does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransformMode {
    Translate,
    Rotate,
    Scale,
}

impl TransformMode {
    /// The key that starts a free transform in this mode.
    pub fn key(self) -> char {
        match self {
            TransformMode::Translate => 'g',
            TransformMode::Rotate => 'r',
            TransformMode::Scale => 's',
        }
    }
}

/// The axes a constraint runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// The transformed thing's own axes.
    Local,
    World,
}

impl Space {
    pub fn other(self) -> Self {
        match self {
            Space::Local => Space::World,
            Space::World => Space::Local,
        }
    }
}

/// What a transform is confined to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constraint {
    Free,
    /// Along an axis, or about it for a rotation.
    Axis(Axis, Space),
    /// In the plane normal to an axis: `Plane(Z, _)` is the XY plane.
    Plane(Axis, Space),
}

/// What a transform acts about, as the thing being transformed offers it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    /// Where rotation and scale act about, and the gizmo sits.
    pub pivot: Point3,
    /// The thing's own axes, which [`Space::Local`] constraints run along.
    pub rotation: Quaternion,
    /// Whether it can only scale uniformly, as a face re-solved into its body
    /// can.
    pub uniform_scale: bool,
}

impl Frame {
    /// The world direction of `axis` in `space`.
    pub fn axis(&self, axis: Axis, space: Space) -> Vector3 {
        match space {
            Space::World => axis.direction(),
            Space::Local => self.rotation.rotate_vector(axis.direction()),
        }
    }
}

/// What a transform reads of the event context.
#[derive(Clone, Copy)]
pub struct Input<'a> {
    pub camera: &'a PositionedCamera,
    pub size: (u32, u32),
    /// The cursor in view pixels, or `None` when it is off the view.
    pub cursor: Option<(f32, f32)>,
    pub modifiers: Modifiers,
}

impl<'a> Input<'a> {
    pub fn of(ctx: &'a EventContext) -> Self {
        Self {
            camera: &*ctx.camera,
            size: ctx.size,
            cursor: *ctx.cursor_position,
            modifiers: ctx.modifiers,
        }
    }

    /// Where `point` lands on screen.
    fn pixel(&self, point: Point3) -> (f32, f32) {
        let p = self.camera.project_point_screen(point, self.size.0, self.size.1);
        (p.x as f32, p.y as f32)
    }
}

/// What a tool should do after feeding its transform an event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    /// Not the transform's: pass the event on.
    Ignored,
    /// Taken, with nothing to do.
    Consumed,
    /// A transform began: lock what it moves, or [`Manipulation::abort`] if
    /// that fails.
    Started,
    /// The delta changed: preview it.
    Changed,
    /// Finished: apply this delta.
    Confirmed(Matrix4),
    /// Abandoned: put back what was previewed.
    Cancelled,
}

/// What a key asks of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Start,
    Axis(Axis),
    Plane(Axis),
    Confirm,
    Cancel,
}

impl Command {
    /// The command `key` gives a transform in `mode` while `modifiers` are
    /// held.
    fn read(key: &KeyEvent, modifiers: Modifiers, mode: TransformMode) -> Option<Self> {
        let plain = modifiers == Modifiers::default();
        let shift = modifiers == Modifiers { shift: true, ..Modifiers::default() };
        match key.logical_key {
            Key::Named(NamedKey::Enter) if plain => Some(Command::Confirm),
            Key::Named(NamedKey::Escape) if plain => Some(Command::Cancel),
            Key::Character(c) => {
                let c = c.to_ascii_lowercase();
                if plain && c == mode.key() {
                    return Some(Command::Start);
                }
                let axis = Axis::from_char(c)?;
                match (plain, shift) {
                    (true, _) => Some(Command::Axis(axis)),
                    // A rotation runs about one axis, so a plane names the axis
                    // normal to it.
                    (_, true) if mode == TransformMode::Rotate => Some(Command::Axis(axis)),
                    (_, true) => Some(Command::Plane(axis)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// What drives an active transform.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Drive {
    /// A gizmo grip, solved by the handle machinery.
    Grip(HandleId),
    /// The cursor, measured from the pixel the transform started at.
    Cursor { anchor: (f32, f32), last: (f32, f32) },
}

/// A transform in progress.
struct Active {
    frame: Frame,
    constraint: Constraint,
    drive: Drive,
    /// How far a move has gone.
    translation: Vector3,
    /// The angle a rotation has swept, integrated across updates so it can pass
    /// half a turn.
    swept: Real,
    /// The last solved angle `swept` was unwrapped against.
    last_angle: Real,
    /// The axis a rotation turns about, as of the last update.
    rotation_axis: Vector3,
    /// How far a scale has gone.
    factor: Real,
}

impl Active {
    fn new(frame: Frame, constraint: Constraint, drive: Drive, camera: &PositionedCamera) -> Self {
        let mut active = Self {
            frame,
            constraint,
            drive,
            translation: Vector3::zero(),
            swept: 0.0,
            last_angle: 0.0,
            rotation_axis: Vector3::zero(),
            factor: 1.0,
        };
        active.rotation_axis = active.rotation_axis_for(camera);
        active
    }

    /// The axis a rotation turns about: the constraint's, or the view direction
    /// when free.
    fn rotation_axis_for(&self, camera: &PositionedCamera) -> Vector3 {
        match self.constraint {
            Constraint::Axis(axis, space) | Constraint::Plane(axis, space) => {
                self.frame.axis(axis, space)
            }
            Constraint::Free => camera.forward(),
        }
    }

    /// The locus a cursor-driven drag in `mode` is solved against.
    fn locus(&self, mode: TransformMode, camera: &PositionedCamera) -> DragGeometry {
        let pivot = self.frame.pivot;
        if mode == TransformMode::Rotate {
            return DragGeometry::plane(self.rotation_axis_for(camera), pivot);
        }
        match self.constraint {
            Constraint::Free => DragGeometry::plane(camera.forward(), pivot),
            Constraint::Axis(axis, space) => DragGeometry::axis(pivot, self.frame.axis(axis, space)),
            Constraint::Plane(axis, space) => DragGeometry::plane(self.frame.axis(axis, space), pivot),
        }
    }

    /// Takes the drag from `grab` to `point`, on the locus, as how far the
    /// transform has gone. Returns false when it could not be read as one, which
    /// leaves the amount where it was.
    fn update(&mut self, mode: TransformMode, grab: Point3, point: Point3, input: &Input) -> bool {
        match mode {
            TransformMode::Translate => {
                self.translation = point - grab;
                true
            }
            TransformMode::Rotate => {
                self.rotation_axis = self.rotation_axis_for(input.camera);
                let Some(solved) = angle_about(self.frame.pivot, self.rotation_axis, grab, point)
                else {
                    return false;
                };
                // A solved angle is only defined modulo a turn; unwrapped
                // against the last one it is unambiguous at per-event motion.
                self.swept += wrap_to_pi(solved - self.last_angle);
                self.last_angle = solved;
                true
            }
            TransformMode::Scale => match self.scale_factor(grab, point, input) {
                Some(factor) => {
                    self.factor = factor.max(MIN_SCALE);
                    true
                }
                None => false,
            },
        }
    }

    /// The scale factor of a drag from `grab` to `point`: the ratio of their
    /// distances from the pivot, signed along an axis and radial otherwise, so
    /// the grabbed point tracks the cursor.
    ///
    /// A drag that started too close to the pivot on screen scales by pixels
    /// instead. `None` when the ratio is degenerate.
    fn scale_factor(&self, grab: Point3, point: Point3, input: &Input) -> Option<Real> {
        let pivot = self.frame.pivot;
        let (from, to, centre) = (input.pixel(grab), input.pixel(point), input.pixel(pivot));
        let grab_pixels = distance(from, centre);
        if grab_pixels < MIN_SCALE_ANCHOR_PIXELS {
            let pixels = match self.constraint {
                Constraint::Axis(axis, space) => {
                    let axis = screen_direction(input, pivot, self.frame.axis(axis, space));
                    (to.0 - from.0) * axis.0 + (to.1 - from.1) * axis.1
                }
                // Nothing to measure along: dragging away from the pivot grows,
                // and back toward it shrinks.
                _ => distance(to, centre) - grab_pixels,
            };
            return Some(1.0 + pixels as Real * SCALE_FRACTION_PER_PIXEL);
        }

        match self.constraint {
            // Signed, so dragging back through the pivot shrinks toward the
            // clamp rather than mirroring.
            Constraint::Axis(axis, space) => {
                let axis = self.frame.axis(axis, space);
                let from = (grab - pivot).dot(axis);
                (from.abs() > EPSILON).then(|| (point - pivot).dot(axis) / from)
            }
            _ => {
                let from = (grab - pivot).magnitude();
                (from > EPSILON).then(|| (point - pivot).magnitude() / from)
            }
        }
    }

    /// The scale as a matrix about the origin: stretching the constraint's
    /// axes, in its own space.
    fn scale_matrix(&self) -> Matrix4 {
        let f = self.factor;
        let (scale, space) = match self.constraint {
            Constraint::Free => return Matrix4::from_scale(f),
            Constraint::Axis(axis, space) => (axis_scale(axis, f, 1.0), space),
            Constraint::Plane(normal, space) => (axis_scale(normal, 1.0, f), space),
        };
        let scale = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z);
        match space {
            Space::World => scale,
            Space::Local => {
                let rotation = Matrix4::from(self.frame.rotation);
                rotation * scale * Matrix4::from(self.frame.rotation.conjugate())
            }
        }
    }
}

/// A transform of the selection: idle, or driven by a gizmo grip or the cursor.
pub struct Manipulation {
    mode: TransformMode,
    /// The space grips constrain in, and the first an axis key picks.
    orientation: Space,
    active: Option<Active>,
}

impl Manipulation {
    pub fn new(mode: TransformMode) -> Self {
        Self { mode, orientation: Space::Local, active: None }
    }

    pub fn mode(&self) -> TransformMode {
        self.mode
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    /// The space grips constrain in and the gizmo is drawn along, and the first
    /// an axis key picks.
    pub fn orientation(&self) -> Space {
        self.orientation
    }

    pub fn set_orientation(&mut self, orientation: Space) {
        self.orientation = orientation;
    }

    /// Drops the transform in progress, if any, without an outcome: for a
    /// start the tool could not lock, and for teardown.
    pub fn abort(&mut self) {
        self.active = None;
    }

    /// The transform so far, as a world-space matrix: `new = delta * old` for
    /// points. Rotation and scale act about the pivot. Identity when idle.
    pub fn delta(&self) -> Matrix4 {
        let Some(active) = &self.active else { return Matrix4::identity() };
        let pivot = active.frame.pivot.to_vec();
        let about_pivot = |m: Matrix4| {
            Matrix4::from_translation(pivot) * m * Matrix4::from_translation(-pivot)
        };
        match self.mode {
            TransformMode::Translate => Matrix4::from_translation(active.translation),
            TransformMode::Rotate => about_pivot(Matrix4::from(quaternion_from_axis_angle_safe(
                active.rotation_axis,
                active.swept,
            ))),
            TransformMode::Scale => about_pivot(active.scale_matrix()),
        }
    }

    /// The gizmo grips to show: about `frame` while idle, riding along with a
    /// grip drag, and none while the cursor drives — a press then must not
    /// grab one.
    pub fn handles(&self, frame: Option<&Frame>) -> Vec<Handle> {
        match &self.active {
            None => frame.map_or_else(Vec::new, |frame| gizmo::handles(self.mode, frame, self.orientation)),
            Some(active) => match active.drive {
                Drive::Cursor { .. } => Vec::new(),
                Drive::Grip(_) => {
                    // The gizmo moves with what it moves, so the grabbed grip
                    // stays under the cursor.
                    let mut frame = active.frame;
                    if self.mode == TransformMode::Translate {
                        frame.pivot += active.translation;
                    }
                    gizmo::handles(self.mode, &frame, self.orientation)
                }
            },
        }
    }

    /// Feeds one viewport event. `frame` is what a transform would start about
    /// now; `None` refuses a start.
    pub fn dispatch(&mut self, event: &Event, frame: Option<Frame>, input: &Input) -> Outcome {
        let Event::Device(event) = event else { return Outcome::Ignored };
        let cursor_driven = matches!(self.drive(), Some(Drive::Cursor { .. }));
        match event {
            DeviceEvent::KeyboardInput { event: key, .. } => self.on_key(key, frame, input),
            DeviceEvent::CursorMoved { position } if cursor_driven => {
                self.move_cursor((position.0 as f32, position.1 as f32), input);
                Outcome::Changed
            }
            // The cursor already moves the transform; a button held across it
            // must not orbit the camera as well.
            DeviceEvent::MouseMotion { .. }
            | DeviceEvent::MouseDragStart { .. }
            | DeviceEvent::MouseDrag { .. }
            | DeviceEvent::MouseDragEnd { .. }
                if cursor_driven =>
            {
                Outcome::Consumed
            }
            DeviceEvent::MouseClick { button: MouseButton::Left, .. } if cursor_driven => {
                self.confirm()
            }
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } if cursor_driven => {
                self.cancel()
            }
            _ => Outcome::Ignored,
        }
    }

    /// Feeds one event on a gizmo grip. `frame` is what a grab would start
    /// about.
    pub fn on_handle(&mut self, event: &HandleEvent, frame: Option<Frame>, input: &Input) -> Outcome {
        match event {
            HandleEvent::Begin(id) if !self.is_active() => {
                let (Some(frame), Some(grip)) = (frame, Grip::of(*id)) else {
                    return Outcome::Ignored;
                };
                let constraint = self.grip_constraint(grip, &frame);
                self.active = Some(Active::new(frame, constraint, Drive::Grip(*id), input.camera));
                Outcome::Started
            }
            HandleEvent::Drag(drag) if self.drive() == Some(Drive::Grip(drag.id)) => {
                let mode = self.mode;
                let active = self.active.as_mut().expect("a grip drives the transform");
                active.update(mode, drag.grab, drag.point, input);
                Outcome::Changed
            }
            HandleEvent::End(id) if self.drive() == Some(Drive::Grip(*id)) => self.confirm(),
            HandleEvent::Cancel(id) if self.drive() == Some(Drive::Grip(*id)) => self.cancel(),
            _ => Outcome::Ignored,
        }
    }

    fn drive(&self) -> Option<Drive> {
        self.active.as_ref().map(|active| active.drive)
    }

    /// What dragging `grip` constrains to: its axis or plane in the
    /// orientation, unless `frame` only scales uniformly.
    fn grip_constraint(&self, grip: Grip, frame: &Frame) -> Constraint {
        if self.mode == TransformMode::Scale && frame.uniform_scale {
            return Constraint::Free;
        }
        match grip {
            Grip::Axis(axis) => Constraint::Axis(axis, self.orientation),
            Grip::Plane(normal) => Constraint::Plane(normal, self.orientation),
            Grip::Ball => Constraint::Free,
        }
    }

    fn on_key(&mut self, key: &KeyEvent, frame: Option<Frame>, input: &Input) -> Outcome {
        if key.state != ElementState::Pressed || key.repeat {
            return Outcome::Ignored;
        }
        let Some(command) = Command::read(key, input.modifiers, self.mode) else {
            return Outcome::Ignored;
        };
        // A grip drag is confined to the locus the grip was grabbed on; its keys
        // are taken so they reach nothing else meanwhile.
        if matches!(self.drive(), Some(Drive::Grip(_))) {
            return Outcome::Consumed;
        }
        match command {
            Command::Start if self.is_active() => Outcome::Consumed,
            Command::Start => match self.start_at_cursor(frame, input) {
                true => Outcome::Started,
                false => Outcome::Ignored,
            },
            Command::Axis(axis) | Command::Plane(axis) => {
                let starting = !self.is_active();
                if starting && !self.start_at_cursor(frame, input) {
                    return Outcome::Ignored;
                }
                self.cycle(command, axis, input);
                match starting {
                    true => Outcome::Started,
                    false => Outcome::Changed,
                }
            }
            Command::Confirm if self.is_active() => self.confirm(),
            Command::Cancel if self.is_active() => self.cancel(),
            Command::Confirm | Command::Cancel => Outcome::Ignored,
        }
    }

    /// Starts a free transform about `frame`, driven by the cursor from where
    /// it is now — or from the pivot, with the cursor off the view.
    fn start_at_cursor(&mut self, frame: Option<Frame>, input: &Input) -> bool {
        let Some(frame) = frame else { return false };
        let anchor = input.cursor.unwrap_or_else(|| input.pixel(frame.pivot));
        let drive = Drive::Cursor { anchor, last: anchor };
        self.active = Some(Active::new(frame, Constraint::Free, drive, input.camera));
        true
    }

    /// Cycles the constraint for an axis or plane key: the orientation's space,
    /// then the other, then free. A different axis or kind jumps straight to the
    /// orientation's space.
    fn cycle(&mut self, command: Command, axis: Axis, input: &Input) {
        let (mode, first) = (self.mode, self.orientation);
        let active = self.active.as_mut().expect("a transform is active");
        // Accepted but not acted on: what is being scaled only scales evenly.
        if mode == TransformMode::Scale && active.frame.uniform_scale {
            return;
        }
        let make = |space| match command {
            Command::Plane(_) => Constraint::Plane(axis, space),
            _ => Constraint::Axis(axis, space),
        };
        active.constraint = if active.constraint == make(first) {
            make(first.other())
        } else if active.constraint == make(first.other()) {
            Constraint::Free
        } else {
            make(first)
        };
        // A swept angle belongs to the axis it was swept about; everything else
        // re-solves from the anchor.
        active.swept = 0.0;
        active.last_angle = 0.0;
        if let Drive::Cursor { last, .. } = active.drive {
            self.move_cursor(last, input);
        }
    }

    /// Moves a cursor-driven transform to `cursor`.
    fn move_cursor(&mut self, cursor: (f32, f32), input: &Input) {
        let mode = self.mode;
        let Some(active) = &mut self.active else { return };
        let Drive::Cursor { anchor, last } = active.drive else { return };
        active.drive = Drive::Cursor { anchor, last: cursor };

        let locus = active.locus(mode, input.camera);
        let solved = solve_drag(&locus, anchor, cursor, input.camera, input.size)
            .is_some_and(|(grab, point)| active.update(mode, grab, point, input));
        // An edge-on rotation plane has no angle to solve; a pixel rate for
        // this event alone keeps the rotation moving, which integrating makes
        // continuous. Anything else holds where it last solved.
        if !solved && mode == TransformMode::Rotate {
            active.swept += (cursor.0 - last.0) as Real * ROTATE_DEGREES_PER_PIXEL.to_radians();
        }
    }

    fn confirm(&mut self) -> Outcome {
        let delta = self.delta();
        self.active = None;
        Outcome::Confirmed(delta)
    }

    fn cancel(&mut self) -> Outcome {
        self.active = None;
        Outcome::Cancelled
    }
}

/// The angle from `from` to `to` about `axis` through `pivot`, positive
/// counter-clockwise looking down `axis`. `None` when either sits on the axis.
fn angle_about(pivot: Point3, axis: Vector3, from: Point3, to: Point3) -> Option<Real> {
    if axis.magnitude2() < EPSILON {
        return None;
    }
    let axis = axis.normalize();
    // Only the swing about the axis is angular.
    let flatten = |p: Point3| {
        let v = p - pivot;
        v - axis * axis.dot(v)
    };
    let (from, to) = (flatten(from), flatten(to));
    if from.magnitude2() < EPSILON || to.magnitude2() < EPSILON {
        return None;
    }
    Some(from.cross(to).dot(axis).atan2(from.dot(to)))
}

/// Wraps an angle into `(-π, π]`, so a difference of solved angles takes the
/// short way round.
fn wrap_to_pi(angle: Real) -> Real {
    use consts::{PI, TAU};
    (angle + PI).rem_euclid(TAU) - PI
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// The unit screen direction `direction` runs in from `point`; right when it
/// points straight at the camera.
fn screen_direction(input: &Input, point: Point3, direction: Vector3) -> (f32, f32) {
    let (from, to) = (input.pixel(point), input.pixel(point + direction));
    let (dx, dy) = (to.0 - from.0, to.1 - from.1);
    let length = dx.hypot(dy);
    if length < f32::EPSILON { (1.0, 0.0) } else { (dx / length, dy / length) }
}

/// A per-axis scale of `on` along `axis` and `off` along the other two.
fn axis_scale(axis: Axis, on: Real, off: Real) -> Vector3 {
    match axis {
        Axis::X => Vector3::new(on, off, off),
        Axis::Y => Vector3::new(off, on, off),
        Axis::Z => Vector3::new(off, off, on),
    }
}

#[cfg(test)]
mod tests {
    use duck_engine_viewer::common::Transform;
    use duck_engine_viewer::operator::HandleDrag;
    use duck_engine_viewer::scene::Projection;

    use super::*;
    use crate::testing::{key, named_key};

    const EPSILON: Real = 1e-6;
    /// Pixel round trips project and unproject through separate matrix
    /// inversions, so they hold to a small fraction of a pixel rather than to
    /// `EPSILON` — still far tighter than the tens of pixels these tests catch.
    const PIXEL_EPSILON: f32 = 0.01;
    /// For world vectors computed by two different but equivalent routes.
    const WORLD_EPSILON: Real = 1e-4;
    const SIZE: (u32, u32) = (800, 800);
    const CENTER: (f32, f32) = (400.0, 400.0);

    fn camera_at(eye: (Real, Real, Real), ortho: bool) -> PositionedCamera {
        let mut camera = PositionedCamera {
            eye: eye.into(),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: 1.0,
            projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
        };
        if ortho {
            camera.make_orthographic();
        }
        camera
    }

    fn input(camera: &PositionedCamera) -> Input<'_> {
        Input { camera, size: SIZE, cursor: Some(CENTER), modifiers: Modifiers::default() }
    }

    /// A frame on the origin along the world axes.
    fn frame() -> Frame {
        Frame { pivot: Point3::new(0.0, 0.0, 0.0), rotation: Transform::IDENTITY.rotation, uniform_scale: false }
    }

    /// A transform about the origin driven by the cursor from `anchor`, under
    /// `constraint`.
    fn started(
        mode: TransformMode,
        anchor: (f32, f32),
        constraint: Constraint,
        camera: &PositionedCamera,
    ) -> Manipulation {
        let mut m = Manipulation::new(mode);
        let input = Input { cursor: Some(anchor), ..input(camera) };
        assert!(m.start_at_cursor(Some(frame()), &input));
        m.active.as_mut().unwrap().constraint = constraint;
        m
    }

    fn constraint(m: &Manipulation) -> Constraint {
        m.active.as_ref().map_or(Constraint::Free, |active| active.constraint)
    }

    fn translation(m: &Manipulation) -> Vector3 {
        m.active.as_ref().unwrap().translation
    }

    fn swept(m: &Manipulation) -> Real {
        m.active.as_ref().unwrap().swept
    }

    fn factor(m: &Manipulation) -> Real {
        m.active.as_ref().unwrap().factor
    }

    fn projected_pivot(camera: &PositionedCamera) -> (f32, f32) {
        input(camera).pixel(Point3::new(0.0, 0.0, 0.0))
    }

    /// Where the pivot lands after the move. Anchored on the pivot's own
    /// projection, the pivot is the grabbed point, so this should be the cursor.
    fn pivot_pixel_after(m: &Manipulation, camera: &PositionedCamera) -> (f32, f32) {
        input(camera).pixel(Point3::new(0.0, 0.0, 0.0) + translation(m))
    }

    fn assert_pixels_close(actual: (f32, f32), expected: (f32, f32), what: &str) {
        assert!(
            (actual.0 - expected.0).abs() < PIXEL_EPSILON && (actual.1 - expected.1).abs() < PIXEL_EPSILON,
            "{what}: expected {expected:?}, got {actual:?}"
        );
    }

    fn press(event: KeyEvent) -> Event {
        Event::Device(DeviceEvent::KeyboardInput { event, is_synthetic: false })
    }

    fn cursor_moved(position: (f32, f32)) -> Event {
        Event::Device(DeviceEvent::CursorMoved { position: (position.0 as f64, position.1 as f64) })
    }

    fn click(button: MouseButton) -> Event {
        Event::Device(DeviceEvent::MouseClick { button, position: CENTER, duration_ms: 50 })
    }

    fn grip_drag(grip: Grip, grab: Point3, point: Point3) -> HandleEvent {
        HandleEvent::Drag(HandleDrag { id: grip.id(), grab, point, modifiers: Modifiers::default() })
    }

    // ── Translation ──────────────────────────────────────────────────────────

    #[test]
    fn foreshortened_axis_drag_tracks_cursor_in_ortho() {
        // Projecting the drag onto the axis would lose cos²θ of it. Under an
        // orthographic camera the closest approach is exactly the screen-space
        // foot of the cursor on the projected axis, so this is an equality.
        let camera = camera_at((8.0, 2.0, 2.0), true);
        let anchor = projected_pivot(&camera);
        let mut m = started(TransformMode::Translate, anchor, Constraint::Axis(Axis::X, Space::World), &camera);

        let (ex, ey) = screen_direction(&input(&camera), Point3::new(0.0, 0.0, 0.0), Vector3::unit_x());
        let cursor = (anchor.0 + ex * 120.0, anchor.1 + ey * 120.0);
        m.move_cursor(cursor, &input(&camera));

        assert_pixels_close(pivot_pixel_after(&m, &camera), cursor, "grabbed point should stay under the cursor");
        let t = translation(&m);
        assert!(t.y.abs() < EPSILON && t.z.abs() < EPSILON && t.x > 0.0, "{t:?}");
    }

    #[test]
    fn plane_constrained_drag_keeps_grabbed_point_under_cursor() {
        for ortho in [false, true] {
            let camera = camera_at((6.0, 5.0, 7.0), ortho);
            let anchor = projected_pivot(&camera);
            let mut m =
                started(TransformMode::Translate, anchor, Constraint::Plane(Axis::Y, Space::World), &camera);

            let cursor = (anchor.0 + 90.0, anchor.1 - 70.0);
            m.move_cursor(cursor, &input(&camera));

            assert_pixels_close(pivot_pixel_after(&m, &camera), cursor, if ortho { "ortho" } else { "perspective" });
            // Plane(Y) excludes Y, so the drag stays in the XZ plane.
            assert!(translation(&m).y.abs() < EPSILON);
        }
    }

    #[test]
    fn free_translation_is_anchor_independent() {
        // World units per pixel are constant across a plane parallel to the
        // image plane, so where the drag was anchored cannot matter.
        let camera = camera_at((3.0, 4.0, 9.0), false);
        let mut near = started(TransformMode::Translate, projected_pivot(&camera), Constraint::Free, &camera);
        let mut far = started(TransformMode::Translate, (60.0, 730.0), Constraint::Free, &camera);
        let pivot = projected_pivot(&camera);
        near.move_cursor((pivot.0 + 44.0, pivot.1 + 61.0), &input(&camera));
        far.move_cursor((60.0 + 44.0, 730.0 + 61.0), &input(&camera));

        assert!((translation(&near) - translation(&far)).magnitude() < WORLD_EPSILON);
    }

    #[test]
    fn an_unmoved_cursor_moves_nothing() {
        // However far the anchor is from the pivot — grabbing an arrow tip must
        // never make the target jump.
        let camera = camera_at((5.0, 4.0, 6.0), false);
        for constraint in [
            Constraint::Free,
            Constraint::Axis(Axis::X, Space::World),
            Constraint::Plane(Axis::Z, Space::World),
        ] {
            let mut m = started(TransformMode::Translate, (612.0, 208.0), constraint, &camera);
            m.move_cursor((612.0, 208.0), &input(&camera));
            assert!(translation(&m).magnitude() < EPSILON, "{constraint:?}");
        }
    }

    #[test]
    fn a_new_constraint_re_solves_from_the_anchor() {
        // Nothing caches a delta: the constraint picks the locus and the drag is
        // solved again from the anchor.
        let camera = camera_at((5.0, 4.0, 6.0), false);
        let anchor = projected_pivot(&camera);
        let mut m = started(TransformMode::Translate, anchor, Constraint::Free, &camera);
        let input = input(&camera);
        m.move_cursor((anchor.0 + 83.0, anchor.1 - 44.0), &input);

        m.cycle(Command::Axis(Axis::X), Axis::X, &input);
        assert_eq!(constraint(&m), Constraint::Axis(Axis::X, Space::Local));
        let along_x = translation(&m);
        m.cycle(Command::Plane(Axis::Y), Axis::Y, &input);
        let in_plane = translation(&m);

        assert!(along_x.y.abs() < EPSILON && along_x.z.abs() < EPSILON, "{along_x:?}");
        assert!(in_plane.y.abs() < EPSILON && (in_plane - along_x).magnitude() > EPSILON, "{in_plane:?}");
    }

    #[test]
    fn ground_plane_drag_toward_the_horizon_is_not_capped() {
        // A ground-plane drag toward the vanishing line needs a large gain to
        // keep up with the cursor; nothing may bound it.
        let camera = camera_at((0.0, 4.0, 9.0), false);
        let anchor = projected_pivot(&camera);
        let mut m = started(TransformMode::Translate, anchor, Constraint::Plane(Axis::Y, Space::World), &camera);
        let cursor = (anchor.0 - 140.0, anchor.1 - 330.0);
        m.move_cursor(cursor, &input(&camera));

        assert_pixels_close(pivot_pixel_after(&m, &camera), cursor, "near the horizon");
    }

    #[test]
    fn a_drag_past_the_horizon_holds_where_it_last_solved() {
        // Above a near-level plane's horizon there is nothing to intersect; the
        // drag must neither freeze at zero nor jump to the mirrored side.
        let camera = camera_at((0.0, 0.5, 10.0), false);
        let anchor = projected_pivot(&camera);
        let mut m = started(TransformMode::Translate, anchor, Constraint::Plane(Axis::Y, Space::World), &camera);
        m.move_cursor((anchor.0 + 30.0, anchor.1 + 20.0), &input(&camera));
        let last = translation(&m);
        assert!(last.magnitude() > EPSILON);

        m.move_cursor((anchor.0 + 150.0, anchor.1 - 600.0), &input(&camera));
        assert!((translation(&m) - last).magnitude() < EPSILON, "{:?}", translation(&m));
    }

    #[test]
    fn a_local_axis_follows_the_frame() {
        let camera = camera_at((4.0, 5.0, 8.0), false);
        let rotation = quaternion_from_axis_angle_safe(Vector3::unit_y(), consts::PI / 4.0);
        let mut m = Manipulation::new(TransformMode::Translate);
        let anchor = projected_pivot(&camera);
        let frame = Frame { rotation, ..frame() };
        m.start_at_cursor(Some(frame), &Input { cursor: Some(anchor), ..input(&camera) });
        m.cycle(Command::Axis(Axis::X), Axis::X, &input(&camera));
        m.move_cursor((anchor.0 + 70.0, anchor.1 + 25.0), &input(&camera));

        let t = translation(&m);
        let local_x = rotation.rotate_vector(Vector3::unit_x());
        assert!(t.magnitude() > EPSILON);
        assert!(t.normalize().cross(local_x).magnitude() < WORLD_EPSILON, "{t:?} is not along {local_x:?}");
    }

    // ── Keys ─────────────────────────────────────────────────────────────────

    #[test]
    fn axis_keys_cycle_through_the_orientation_first() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let input = input(&camera);
        let mut m = Manipulation::new(TransformMode::Translate);
        let shift = Modifiers { shift: true, ..Modifiers::default() };

        assert_eq!(m.dispatch(&press(key('y')), Some(frame()), &input), Outcome::Started);
        assert_eq!(constraint(&m), Constraint::Axis(Axis::Y, Space::Local));
        assert_eq!(m.dispatch(&press(key('y')), Some(frame()), &input), Outcome::Changed);
        assert_eq!(constraint(&m), Constraint::Axis(Axis::Y, Space::World));
        m.dispatch(&press(key('y')), Some(frame()), &input);
        assert_eq!(constraint(&m), Constraint::Free);

        // Shift+Z arrives as an uppercase character.
        let shifted = Input { modifiers: shift, ..input };
        m.dispatch(&press(key('Z')), Some(frame()), &shifted);
        assert_eq!(constraint(&m), Constraint::Plane(Axis::Z, Space::Local));
        m.dispatch(&press(key('Z')), Some(frame()), &shifted);
        assert_eq!(constraint(&m), Constraint::Plane(Axis::Z, Space::World));
    }

    /// A rotation runs about one axis, so Shift names the axis like a plain key.
    #[test]
    fn a_rotation_reads_plane_keys_as_their_axis() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let shifted = Input { modifiers: Modifiers { shift: true, ..Modifiers::default() }, ..input(&camera) };
        let mut m = Manipulation::new(TransformMode::Rotate);
        m.dispatch(&press(key('X')), Some(frame()), &shifted);
        assert_eq!(constraint(&m), Constraint::Axis(Axis::X, Space::Local));
    }

    #[test]
    fn the_mode_key_starts_at_the_cursor_and_a_click_confirms() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let input = input(&camera);
        let mut m = Manipulation::new(TransformMode::Translate);

        assert_eq!(m.dispatch(&press(key('g')), Some(frame()), &input), Outcome::Started);
        assert_eq!(m.dispatch(&cursor_moved((CENTER.0 + 50.0, CENTER.1)), None, &input), Outcome::Changed);
        let moved = translation(&m);
        assert!(moved.x > 0.0 && moved.y.abs() < EPSILON, "{moved:?}");

        let Outcome::Confirmed(delta) = m.dispatch(&click(MouseButton::Left), None, &input) else {
            panic!("a click confirms");
        };
        assert!((delta.w.truncate() - moved).magnitude() < EPSILON);
        assert!(!m.is_active());
    }

    #[test]
    fn escape_and_right_click_cancel() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let input = input(&camera);
        for cancel in [press(named_key(NamedKey::Escape)), click(MouseButton::Right)] {
            let mut m = Manipulation::new(TransformMode::Rotate);
            m.dispatch(&press(key('r')), Some(frame()), &input);
            assert_eq!(m.dispatch(&cancel, None, &input), Outcome::Cancelled);
            assert!(!m.is_active());
        }
    }

    /// With nothing to transform the keys are someone else's — X deletes.
    #[test]
    fn keys_are_left_alone_with_nothing_to_transform() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Translate);
        for c in ['g', 'x'] {
            assert_eq!(m.dispatch(&press(key(c)), None, &input(&camera)), Outcome::Ignored);
        }
        assert_eq!(m.dispatch(&press(named_key(NamedKey::Enter)), Some(frame()), &input(&camera)), Outcome::Ignored);
    }

    /// A button held while the cursor drives must not orbit the camera.
    #[test]
    fn a_cursor_driven_transform_takes_mouse_drags() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Translate);
        m.dispatch(&press(key('g')), Some(frame()), &input(&camera));
        let drag = Event::Device(DeviceEvent::MouseDrag {
            button: MouseButton::Left,
            start_pos: CENTER,
            current_pos: CENTER,
            delta: (5.0, 0.0),
        });
        assert_eq!(m.dispatch(&drag, None, &input(&camera)), Outcome::Consumed);
    }

    #[test]
    fn a_uniform_scale_takes_axis_keys_without_constraining() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Scale);
        let uniform = Frame { uniform_scale: true, ..frame() };
        assert_eq!(m.dispatch(&press(key('x')), Some(uniform), &input(&camera)), Outcome::Started);
        assert_eq!(constraint(&m), Constraint::Free);
    }

    // ── Grips ────────────────────────────────────────────────────────────────

    #[test]
    fn a_grip_constrains_in_the_orientation() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Translate);
        m.set_orientation(Space::World);
        let begin = HandleEvent::Begin(Grip::Plane(Axis::X).id());
        assert_eq!(m.on_handle(&begin, Some(frame()), &input(&camera)), Outcome::Started);
        assert_eq!(constraint(&m), Constraint::Plane(Axis::X, Space::World));
    }

    #[test]
    fn a_grip_drag_moves_by_its_points_and_ends_confirmed() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let input = input(&camera);
        let mut m = Manipulation::new(TransformMode::Translate);
        let grip = Grip::Axis(Axis::X);
        m.on_handle(&HandleEvent::Begin(grip.id()), Some(frame()), &input);

        let drag = grip_drag(grip, Point3::new(1.0, 0.0, 0.0), Point3::new(3.5, 0.0, 0.0));
        assert_eq!(m.on_handle(&drag, None, &input), Outcome::Changed);
        let Outcome::Confirmed(delta) = m.on_handle(&HandleEvent::End(grip.id()), None, &input) else {
            panic!("releasing the grip confirms");
        };
        assert!((delta.w.truncate() - Vector3::new(2.5, 0.0, 0.0)).magnitude() < EPSILON);
    }

    /// The grabbed grip's locus is fixed, so keys may not change it mid-drag —
    /// but they are taken, or X would delete the part being dragged.
    #[test]
    fn keys_during_a_grip_drag_are_taken_and_change_nothing() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Translate);
        m.on_handle(&HandleEvent::Begin(Grip::Axis(Axis::X).id()), Some(frame()), &input(&camera));
        assert_eq!(m.dispatch(&press(key('y')), None, &input(&camera)), Outcome::Consumed);
        assert_eq!(constraint(&m), Constraint::Axis(Axis::X, Space::Local));
    }

    #[test]
    fn a_uniform_scale_grip_is_free() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Scale);
        let uniform = Frame { uniform_scale: true, ..frame() };
        m.on_handle(&HandleEvent::Begin(Grip::Axis(Axis::X).id()), Some(uniform), &input(&camera));
        assert_eq!(constraint(&m), Constraint::Free);
    }

    #[test]
    fn the_gizmo_rides_a_translate_drag_and_hides_while_the_cursor_drives() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Translate);
        let grip = Grip::Axis(Axis::X);
        m.on_handle(&HandleEvent::Begin(grip.id()), Some(frame()), &input(&camera));
        m.on_handle(&grip_drag(grip, Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)), None, &input(&camera));
        let anchors: Vec<Point3> = m.handles(None).iter().map(|h| h.anchor).collect();
        assert!(!anchors.is_empty());
        assert!(anchors.iter().all(|a| (a - Point3::new(2.0, 0.0, 0.0)).magnitude() < EPSILON));

        let mut keyed = Manipulation::new(TransformMode::Translate);
        keyed.dispatch(&press(key('g')), Some(frame()), &input(&camera));
        assert!(keyed.handles(Some(&frame())).is_empty());
    }

    // ── Rotation ─────────────────────────────────────────────────────────────

    #[test]
    fn rotation_winds_past_half_turn() {
        // Walking the cursor 270° around the pivot must sweep 270°, not wrap
        // back at 180°.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let radius = 100.0;
        let mut m = started(TransformMode::Rotate, (CENTER.0 + radius, CENTER.1), Constraint::Free, &camera);

        let mut last = 0.0;
        let steps = 27;
        for step in 1..=steps {
            let theta = (step as f32 * 10.0).to_radians();
            m.move_cursor((CENTER.0 + radius * theta.cos(), CENTER.1 + radius * theta.sin()), &input(&camera));
            let sweep = swept(&m).abs();
            assert!(sweep > last, "the sweep should grow at step {step}");
            last = sweep;
        }

        let expected = (steps as Real * 10.0).to_radians();
        assert!(last > consts::PI, "the sweep wrapped at half a turn: {last}");
        assert!((last - expected).abs() < 0.02, "expected {expected}, got {last}");
    }

    #[test]
    fn a_ring_drag_winds_past_half_turn() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = Manipulation::new(TransformMode::Rotate);
        let grip = Grip::Axis(Axis::Z);
        m.on_handle(&HandleEvent::Begin(grip.id()), Some(frame()), &input(&camera));
        let grab = Point3::new(1.0, 0.0, 0.0);
        for step in 1..=27 {
            let theta = (step as Real * 10.0).to_radians();
            let point = Point3::new(theta.cos(), theta.sin(), 0.0);
            m.on_handle(&grip_drag(grip, grab, point), None, &input(&camera));
        }
        assert!((swept(&m) - (270.0 as Real).to_radians()).abs() < 1e-4, "{}", swept(&m));
    }

    #[test]
    fn rotation_resets_on_a_new_axis() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Rotate, (CENTER.0 + 120.0, CENTER.1), Constraint::Free, &camera);
        m.move_cursor((CENTER.0 + 120.0, CENTER.1 + 60.0), &input(&camera));
        assert!(swept(&m).abs() > EPSILON);

        // About X, seen down -Z, the rotation plane is edge-on, so nothing
        // re-solves the reset sweep.
        m.cycle(Command::Axis(Axis::X), Axis::X, &input(&camera));
        assert!(swept(&m).abs() < EPSILON, "{}", swept(&m));
    }

    #[test]
    fn rotation_falls_back_to_pixel_rate_when_anchored_on_the_pivot() {
        // No radius, no angle to solve: the pixel rate, not a wild or zero turn.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Rotate, CENTER, Constraint::Free, &camera);
        m.move_cursor((CENTER.0 + 100.0, CENTER.1), &input(&camera));

        let expected = 100.0 * ROTATE_DEGREES_PER_PIXEL.to_radians();
        assert!((swept(&m) - expected).abs() < EPSILON);
    }

    #[test]
    fn a_rotation_turns_about_the_pivot() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let pivot = Point3::new(2.0, 1.0, 0.0);
        let mut m = Manipulation::new(TransformMode::Rotate);
        let grip = Grip::Axis(Axis::Z);
        m.on_handle(&HandleEvent::Begin(grip.id()), Some(Frame { pivot, ..frame() }), &input(&camera));
        m.on_handle(&grip_drag(grip, pivot + Vector3::unit_x(), pivot + Vector3::unit_y()), None, &input(&camera));

        let delta = m.delta();
        let moved = Point3::from_homogeneous(delta * (pivot + Vector3::unit_x()).to_homogeneous());
        assert!((Point3::from_homogeneous(delta * pivot.to_homogeneous()) - pivot).magnitude() < EPSILON);
        assert!((moved - (pivot + Vector3::unit_y())).magnitude() < EPSILON, "{moved:?}");
    }

    // ── Scale ────────────────────────────────────────────────────────────────

    #[test]
    fn scale_factor_is_the_ratio_of_cursor_distances() {
        // Anchored 100 px out along X and dragged to 200 px: exactly 2×.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Scale, (CENTER.0 + 100.0, CENTER.1), Constraint::Axis(Axis::X, Space::World), &camera);
        m.move_cursor((CENTER.0 + 200.0, CENTER.1), &input(&camera));
        assert!((factor(&m) - 2.0).abs() < 1e-4, "{}", factor(&m));
    }

    #[test]
    fn uniform_scale_factor_is_the_ratio_of_radial_distances() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Scale, (CENTER.0 + 120.0, CENTER.1), Constraint::Free, &camera);
        m.move_cursor((CENTER.0 + 180.0, CENTER.1), &input(&camera));
        assert!((factor(&m) - 1.5).abs() < 1e-4, "{}", factor(&m));
    }

    #[test]
    fn scale_anchored_on_the_pivot_falls_back_to_pixel_rate() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Scale, projected_pivot(&camera), Constraint::Axis(Axis::X, Space::World), &camera);
        m.move_cursor((CENTER.0 + 100.0, CENTER.1), &input(&camera));
        let expected = 1.0 + 100.0 * SCALE_FRACTION_PER_PIXEL;
        assert!((factor(&m) - expected).abs() < 1e-4, "{}", factor(&m));
    }

    #[test]
    fn uniform_scale_from_the_pivot_grows_in_any_direction() {
        // The gizmo's ball sits on the pivot, so this path is routine: it has
        // to grow whichever way the drag leaves the pivot.
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let expected = 1.0 + 90.0 * SCALE_FRACTION_PER_PIXEL;
        for (dx, dy) in [(90.0, 0.0), (0.0, 90.0), (-90.0, 0.0), (0.0, -90.0)] {
            let mut m = started(TransformMode::Scale, projected_pivot(&camera), Constraint::Free, &camera);
            m.move_cursor((CENTER.0 + dx, CENTER.1 + dy), &input(&camera));
            assert!((factor(&m) - expected).abs() < 1e-4, "({dx}, {dy}) gave {}", factor(&m));
        }
    }

    #[test]
    fn scale_never_goes_negative() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let mut m = started(TransformMode::Scale, (CENTER.0 + 100.0, CENTER.1), Constraint::Axis(Axis::X, Space::World), &camera);
        m.move_cursor((CENTER.0 - 300.0, CENTER.1), &input(&camera));
        assert!((factor(&m) - MIN_SCALE).abs() < EPSILON, "{}", factor(&m));
    }

    #[test]
    fn a_local_scale_stretches_along_the_frame() {
        let camera = camera_at((0.0, 0.0, 10.0), false);
        let rotation = quaternion_from_axis_angle_safe(Vector3::unit_z(), consts::PI / 2.0);
        let mut m = Manipulation::new(TransformMode::Scale);
        let grip = Grip::Axis(Axis::X);
        m.on_handle(&HandleEvent::Begin(grip.id()), Some(Frame { rotation, ..frame() }), &input(&camera));
        // Local X runs along world Y here.
        m.on_handle(&grip_drag(grip, Point3::new(0.0, 1.0, 0.0), Point3::new(0.0, 3.0, 0.0)), None, &input(&camera));

        let delta = m.delta();
        let y = (delta * Vector3::unit_y().extend(0.0)).truncate();
        let x = (delta * Vector3::unit_x().extend(0.0)).truncate();
        assert!((y - Vector3::new(0.0, 3.0, 0.0)).magnitude() < EPSILON, "{y:?}");
        assert!((x - Vector3::unit_x()).magnitude() < EPSILON, "{x:?}");
    }
}
