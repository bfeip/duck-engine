//! Moves, turns and scales the selected nodes, Blender style.
//!
//! G, R or S starts a move, rotation or scale that follows the cursor. X, Y or
//! Z then confines it to that world axis; pressed again, to the primary node's
//! own, and a third time it frees it. With Shift, the same goes for the plane
//! normal to the axis. Another mode's key turns the transform into one of that
//! mode. A click or Enter keeps the transform, and a right-click or Escape puts
//! the nodes back.
//!
//! Between transforms the selection wears grips for the last mode used, which
//! drag directly.

use duck_engine_common::{
    consts, EuclideanSpace, InnerSpace, Matrix4, Point3, Quaternion, Real, Rotation,
    SquareMatrix, Vector3, Zero, EPSILON,
};
use duck_engine_scene::{PositionedCamera, Scene, SceneData};

use crate::common::{
    centroid_of_slice, decompose_matrix, quaternion_from_axis_angle_safe, Axis, RgbaColor,
    Transform,
};
use crate::event::{DeviceEvent, Event, EventContext};
use crate::input::{ElementState, Key, KeyEvent, Modifiers, MouseButton, NamedKey};
use crate::operator::{
    solve_drag, DragGeometry, DragKind, Handle, HandleEvent, HandleId, HandleInput, HandleOutcome,
    HandleReach, HandleSet, HandleShape, Operator,
};
use crate::scene::resource::NodeId;
use crate::selection::SelectionManager;

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

/// How much larger than a plain grip the gizmo is drawn.
const GIZMO_SCALE: Real = 1.4;
/// Rings run round at the radius the arrows reach, twice a plain ring's.
const RING_SCALE: Real = 2.0 * GIZMO_SCALE;
/// Plane quads are smaller, tucked in between the arrows.
const QUAD_SCALE: Real = 0.9;
/// How far out along each of its two axes a plane quad sits, before
/// [`QUAD_SCALE`].
const QUAD_OFFSET: Real = 0.47;
/// Opacity of a plane quad.
const PLANE_ALPHA: f32 = 0.5;
/// The ball claims no axis, so it is neutral.
const BALL_COLOR: RgbaColor = RgbaColor { r: 0.85, g: 0.85, b: 0.85, a: 1.0 };

/// What a transform does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Translate,
    Rotate,
    Scale,
}

impl Mode {
    /// The mode key `c` starts.
    fn of(c: char) -> Option<Self> {
        match c {
            'g' => Some(Mode::Translate),
            'r' => Some(Mode::Rotate),
            's' => Some(Mode::Scale),
            _ => None,
        }
    }
}

/// The axes a constraint runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    World,
    /// The primary node's own.
    Local,
}

/// What a transform is confined to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Constraint {
    Free,
    /// Along an axis, or about it for a rotation.
    Axis(Axis, Space),
    /// In the plane normal to an axis: `Plane(Z, _)` is the XY plane.
    Plane(Axis, Space),
}

impl Constraint {
    fn space(self) -> Option<Space> {
        match self {
            Constraint::Free => None,
            Constraint::Axis(_, space) | Constraint::Plane(_, space) => Some(space),
        }
    }

    /// The grip that drags under this constraint, in either space.
    fn grip(self) -> HandleId {
        HandleId(match self {
            Constraint::Axis(axis, _) => axis as u32,
            Constraint::Plane(normal, _) => 3 + normal as u32,
            Constraint::Free => 6,
        })
    }

    /// What dragging grip `id` confines to. Grips run along the world axes.
    fn of_grip(id: HandleId) -> Option<Self> {
        let axis = |index: u32| Axis::ALL[index as usize];
        match id.0 {
            0..=2 => Some(Constraint::Axis(axis(id.0), Space::World)),
            3..=5 => Some(Constraint::Plane(axis(id.0 - 3), Space::World)),
            6 => Some(Constraint::Free),
            _ => None,
        }
    }
}

/// What a key asks of the operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Command {
    Start(Mode),
    Axis(Axis),
    Plane(Axis),
    Confirm,
    Cancel,
}

impl Command {
    /// The command `key` gives while `modifiers` are held, for a transform in
    /// `mode`.
    fn read(key: &Key, modifiers: Modifiers, mode: Mode) -> Option<Self> {
        let plain = modifiers == Modifiers::default();
        let shift = modifiers == Modifiers { shift: true, ..Modifiers::default() };
        match key {
            Key::Named(NamedKey::Enter) if plain => Some(Command::Confirm),
            Key::Named(NamedKey::Escape) if plain => Some(Command::Cancel),
            Key::Character(c) => {
                let c = c.to_ascii_lowercase();
                if let Some(mode) = Mode::of(c).filter(|_| plain) {
                    return Some(Command::Start(mode));
                }
                let axis = Axis::from_char(c)?;
                match (plain, shift) {
                    (true, _) => Some(Command::Axis(axis)),
                    // A rotation turns about one axis, so a plane names the axis
                    // normal to it.
                    (_, true) if mode == Mode::Rotate => Some(Command::Axis(axis)),
                    (_, true) => Some(Command::Plane(axis)),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

/// What the selection transforms about.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// The centre of the selected nodes' bounds.
    pivot: Point3,
    /// The primary node's world rotation, which [`Space::Local`] runs along.
    rotation: Quaternion,
}

impl Frame {
    /// The frame of the selected nodes; `None` with nothing selected.
    fn of(selection: &SelectionManager, scene: &SceneData) -> Option<Self> {
        let centres: Vec<Point3> = selection
            .selected_nodes()
            .iter()
            .filter_map(|&node| {
                scene.nodes_bounding(node).bounds.map(|bounds| bounds.center()).or_else(|| {
                    scene.nodes_transform(node).map(|world| decompose_matrix(&world).position)
                })
            })
            .collect();
        let pivot = centroid_of_slice(&centres)?;
        let rotation = selection
            .primary()
            .and_then(|primary| scene.nodes_transform(primary.node_id()))
            .map_or(Transform::IDENTITY.rotation, |world| decompose_matrix(&world).rotation);
        Some(Self { pivot, rotation })
    }

    /// The world direction of `axis` in `space`.
    fn axis(&self, axis: Axis, space: Space) -> Vector3 {
        match space {
            Space::World => axis.direction(),
            Space::Local => self.rotation.rotate_vector(axis.direction()),
        }
    }
}

/// A node as it stood when the transform began.
struct Rest {
    node: NodeId,
    local: Transform,
    world: Matrix4,
    /// From world space into the parent's.
    to_parent: Matrix4,
}

impl Rest {
    fn of(node: NodeId, scene: &SceneData) -> Option<Self> {
        let local = scene.get_node(node)?.transform();
        let world = scene.nodes_transform(node)?;
        let parent = scene.get_node(node)?.parent().and_then(|parent| scene.nodes_transform(parent));
        let to_parent = parent.and_then(|parent| parent.invert()).unwrap_or(Matrix4::identity());
        Some(Self { node, local, world, to_parent })
    }
}

/// What drives a transform.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Drive {
    /// A grip, solved by the handle machinery.
    Grip,
    /// The cursor, measured from the pixel the transform started at.
    Cursor { anchor: (f32, f32), last: (f32, f32) },
}

/// A transform in progress.
struct Active {
    frame: Frame,
    constraint: Constraint,
    drive: Drive,
    nodes: Vec<Rest>,
    /// How far a move has gone.
    translation: Vector3,
    /// The angle a rotation has swept, integrated across updates so it can pass
    /// half a turn.
    swept: Real,
    /// The last solved angle `swept` was unwrapped against.
    last_angle: Real,
    /// The axis a rotation turns about.
    rotation_axis: Vector3,
    /// How far a scale has gone.
    factor: Real,
}

impl Active {
    /// Takes the transform back to nothing done under its constraint, seen by
    /// `camera`.
    fn reset(&mut self, camera: &PositionedCamera) {
        self.translation = Vector3::zero();
        self.swept = 0.0;
        self.last_angle = 0.0;
        self.factor = 1.0;
        self.rotation_axis = match self.constraint {
            Constraint::Axis(axis, space) | Constraint::Plane(axis, space) => {
                self.frame.axis(axis, space)
            }
            Constraint::Free => camera.forward(),
        };
    }

    /// The locus a cursor drag in `mode` is solved against.
    fn locus(&self, mode: Mode, camera: &PositionedCamera) -> DragGeometry {
        let pivot = self.frame.pivot;
        match (mode, self.constraint) {
            (Mode::Rotate, _) => DragGeometry::plane(self.rotation_axis, pivot),
            (_, Constraint::Free) => DragGeometry::plane(camera.forward(), pivot),
            (_, Constraint::Axis(axis, space)) => {
                DragGeometry::axis(pivot, self.frame.axis(axis, space))
            }
            (_, Constraint::Plane(axis, space)) => {
                DragGeometry::plane(self.frame.axis(axis, space), pivot)
            }
        }
    }

    /// Takes a drag from `grab` to `point`, on the locus, as how far the
    /// transform has gone. Returns false when it could not be read as one, which
    /// leaves the amount where it was.
    fn update(&mut self, mode: Mode, grab: Point3, point: Point3, ctx: &EventContext) -> bool {
        match mode {
            Mode::Translate => {
                self.translation = point - grab;
                true
            }
            Mode::Rotate => {
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
            Mode::Scale => match self.scale_factor(grab, point, ctx) {
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
    fn scale_factor(&self, grab: Point3, point: Point3, ctx: &EventContext) -> Option<Real> {
        let pivot = self.frame.pivot;
        let (from, to, centre) = (pixel(ctx, grab), pixel(ctx, point), pixel(ctx, pivot));
        let grab_pixels = distance(from, centre);
        if grab_pixels < MIN_SCALE_ANCHOR_PIXELS {
            let pixels = match self.constraint {
                Constraint::Axis(axis, space) => {
                    let axis = screen_direction(ctx, pivot, self.frame.axis(axis, space));
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

    /// The transform in `mode` so far, as a world-space matrix: `new = delta *
    /// old` for points. Rotation and scale act about the pivot.
    fn delta(&self, mode: Mode) -> Matrix4 {
        let pivot = self.frame.pivot.to_vec();
        let about_pivot =
            |m: Matrix4| Matrix4::from_translation(pivot) * m * Matrix4::from_translation(-pivot);
        match mode {
            Mode::Translate => Matrix4::from_translation(self.translation),
            Mode::Rotate => about_pivot(Matrix4::from(quaternion_from_axis_angle_safe(
                self.rotation_axis,
                self.swept,
            ))),
            Mode::Scale => about_pivot(self.scale_matrix()),
        }
    }

    /// The scale so far as a matrix about the origin: stretching the
    /// constraint's axes, in its own space.
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

/// Moves, turns and scales the selected nodes, by grip or from the keyboard.
pub struct TransformOperator {
    /// The last mode started, which the grips show.
    mode: Mode,
    active: Option<Active>,
    /// Scene side of the grips.
    handles: HandleSet,
    /// Pointer state of a drag on one of them.
    input: HandleInput,
}

impl Default for TransformOperator {
    fn default() -> Self {
        Self::new()
    }
}

impl TransformOperator {
    pub fn new() -> Self {
        Self {
            mode: Mode::Translate,
            active: None,
            handles: HandleSet::new(),
            input: HandleInput::new(),
        }
    }

    fn drive(&self) -> Option<Drive> {
        self.active.as_ref().map(|active| active.drive)
    }

    fn cursor_driven(&self) -> bool {
        matches!(self.drive(), Some(Drive::Cursor { .. }))
    }

    /// The grips to show: about the selection between transforms, and with one
    /// in progress, along its constraint and moving with what it moves.
    fn shown(&self, ctx: &EventContext) -> Vec<Handle> {
        let Some(active) = &self.active else {
            let frame = Frame::of(&ctx.selection, &ctx.scene.lock());
            return frame.map_or_else(Vec::new, |frame| gizmo(self.mode, &frame, Space::World));
        };
        let mut frame = active.frame;
        if self.mode == Mode::Translate {
            frame.pivot += active.translation;
        }
        gizmo(self.mode, &frame, active.constraint.space().unwrap_or(Space::World))
    }

    /// Starts a `mode` transform about `frame` of the selected nodes. Returns
    /// false when none can be moved.
    fn begin(
        &mut self,
        mode: Mode,
        frame: Frame,
        constraint: Constraint,
        drive: Drive,
        ctx: &EventContext,
    ) -> bool {
        let nodes: Vec<Rest> = {
            let scene = ctx.scene.lock();
            let selected = ctx.selection.selected_nodes();
            selected.into_iter().filter_map(|node| Rest::of(node, &scene)).collect()
        };
        if nodes.is_empty() {
            return false;
        }
        let mut active = Active {
            frame,
            constraint,
            drive,
            nodes,
            translation: Vector3::zero(),
            swept: 0.0,
            last_angle: 0.0,
            rotation_axis: Vector3::zero(),
            factor: 1.0,
        };
        active.reset(ctx.camera);
        self.mode = mode;
        self.active = Some(active);
        true
    }

    /// Starts a free `mode` transform driven by the cursor from where it is now,
    /// or from the pivot with the cursor off the view.
    fn start_at_cursor(&mut self, mode: Mode, ctx: &EventContext) -> bool {
        let Some(frame) = Frame::of(&ctx.selection, &ctx.scene.lock()) else { return false };
        let anchor = ctx.cursor_position.unwrap_or_else(|| pixel(ctx, frame.pivot));
        let drive = Drive::Cursor { anchor, last: anchor };
        self.begin(mode, frame, Constraint::Free, drive, ctx)
    }

    fn on_grip(&mut self, event: &HandleEvent, ctx: &EventContext) {
        match event {
            HandleEvent::Begin(id) => {
                let frame = Frame::of(&ctx.selection, &ctx.scene.lock());
                if let (Some(frame), Some(constraint)) = (frame, Constraint::of_grip(*id)) {
                    self.begin(self.mode, frame, constraint, Drive::Grip, ctx);
                }
            }
            HandleEvent::Drag(drag) => {
                let mode = self.mode;
                if let Some(active) = &mut self.active
                    && active.update(mode, drag.grab, drag.point, ctx)
                {
                    self.preview(ctx);
                }
            }
            HandleEvent::End(_) => self.confirm(&ctx.scene),
            HandleEvent::Cancel(_) => self.cancel(&ctx.scene),
        }
    }

    fn on_key(&mut self, key: &KeyEvent, ctx: &EventContext) -> bool {
        if key.state != ElementState::Pressed || key.repeat {
            return false;
        }
        let Some(command) = Command::read(&key.logical_key, ctx.modifiers, self.mode) else {
            return false;
        };
        let drive = self.drive();
        match command {
            // A grip drag stays on the locus it was grabbed on; its keys reach
            // nothing else meanwhile.
            _ if drive == Some(Drive::Grip) => true,
            Command::Start(mode) if drive.is_none() => self.start_at_cursor(mode, ctx),
            Command::Start(mode) => {
                if mode != self.mode {
                    self.mode = mode;
                    self.reconstrain(Constraint::Free, ctx);
                }
                true
            }
            Command::Axis(axis) => self.cycle(|space| Constraint::Axis(axis, space), ctx),
            Command::Plane(axis) => self.cycle(|space| Constraint::Plane(axis, space), ctx),
            Command::Confirm if drive.is_some() => {
                self.confirm(&ctx.scene);
                true
            }
            Command::Cancel if drive.is_some() => {
                self.cancel(&ctx.scene);
                true
            }
            Command::Confirm | Command::Cancel => false,
        }
    }

    /// Cycles the constraint an axis or plane key gives: `make`'s in world
    /// space, then in local, then free. Starts a transform at the cursor first
    /// when none is in progress. Returns whether one is.
    fn cycle(&mut self, make: impl Fn(Space) -> Constraint, ctx: &EventContext) -> bool {
        if self.active.is_none() && !self.start_at_cursor(self.mode, ctx) {
            return false;
        }
        let current = self.active.as_ref().map(|active| active.constraint);
        let next = if current == Some(make(Space::World)) {
            make(Space::Local)
        } else if current == Some(make(Space::Local)) {
            Constraint::Free
        } else {
            make(Space::World)
        };
        self.reconstrain(next, ctx);
        true
    }

    /// Puts the cursor transform under `constraint`, from where it started.
    fn reconstrain(&mut self, constraint: Constraint, ctx: &EventContext) {
        let Some(active) = &mut self.active else { return };
        active.constraint = constraint;
        active.reset(ctx.camera);
        if let Drive::Cursor { last, .. } = active.drive {
            self.move_cursor(last, ctx);
        }
    }

    /// Moves a cursor-driven transform to `cursor`.
    fn move_cursor(&mut self, cursor: (f32, f32), ctx: &EventContext) {
        let mode = self.mode;
        let Some(active) = &mut self.active else { return };
        let Drive::Cursor { anchor, last } = active.drive else { return };
        active.drive = Drive::Cursor { anchor, last: cursor };

        let locus = active.locus(mode, ctx.camera);
        let solved = solve_drag(&locus, anchor, cursor, ctx.camera, ctx.size)
            .is_some_and(|(grab, point)| active.update(mode, grab, point, ctx));
        // An edge-on rotation plane has no angle to solve; a pixel rate for
        // this event alone keeps the rotation moving, which integrating makes
        // continuous. Anything else holds where it last solved.
        if !solved && mode == Mode::Rotate {
            active.swept += (cursor.0 - last.0) as Real * ROTATE_DEGREES_PER_PIXEL.to_radians();
        }
        self.preview(ctx);
    }

    /// Shows the transform so far on the nodes.
    fn preview(&self, ctx: &EventContext) {
        let Some(active) = &self.active else { return };
        let delta = active.delta(self.mode);
        let mut scene = ctx.scene.lock();
        for rest in &active.nodes {
            if !scene.has_node(rest.node) {
                continue;
            }
            let moved = decompose_matrix(&(rest.to_parent * delta * rest.world));
            let transform = match self.mode {
                Mode::Translate => Transform { position: moved.position, ..rest.local },
                Mode::Rotate => Transform { scale: rest.local.scale, ..moved },
                // A node holds no shear: a scale off its own axes stretches each
                // of them by as much as it lengthens it, and leaves it turned as
                // it was.
                Mode::Scale => Transform { rotation: rest.local.rotation, ..moved },
            };
            scene.set_node_transform(rest.node, transform);
        }
    }

    /// Keeps the transform as previewed.
    fn confirm(&mut self, scene: &Scene) {
        self.active = None;
        self.handles.set_highlight(None, scene);
    }

    /// Puts the nodes back where they started.
    fn cancel(&mut self, scene: &Scene) {
        if let Some(active) = self.active.take() {
            let mut data = scene.lock();
            for rest in &active.nodes {
                if data.has_node(rest.node) {
                    data.set_node_transform(rest.node, rest.local);
                }
            }
        }
        self.handles.set_highlight(None, scene);
    }
}

impl Operator for TransformOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // Frame tick: bring the grips in line with the selection, or with the
        // transform in progress, whose constraint a keyboard one lights.
        if matches!(event, Event::Device(DeviceEvent::Update { .. })) {
            let shown = self.shown(ctx);
            self.handles.sync(&shown, &ctx.scene);
            if self.cursor_driven() {
                let constraint = self.active.as_ref().map(|active| active.constraint.grip());
                self.handles.set_highlight(constraint, &ctx.scene);
            }
            return false;
        }

        // While the cursor drives, a press confirms rather than grabbing a grip.
        if !self.cursor_driven() {
            match self.input.dispatch(event, &mut self.handles, ctx) {
                HandleOutcome::Ignored => {}
                HandleOutcome::Consumed => return true,
                HandleOutcome::Event(grip) => {
                    self.on_grip(&grip, ctx);
                    return true;
                }
            }
        }

        let Event::Device(event) = event else { return false };
        let cursor_driven = self.cursor_driven();
        match event {
            DeviceEvent::KeyboardInput { event: key, .. } => self.on_key(key, ctx),
            DeviceEvent::CursorMoved { position } if cursor_driven => {
                self.move_cursor((position.0 as f32, position.1 as f32), ctx);
                true
            }
            // The cursor already moves the nodes; a button held across it must
            // not orbit the camera as well.
            DeviceEvent::MouseMotion { .. }
            | DeviceEvent::MouseDragStart { .. }
            | DeviceEvent::MouseDrag { .. }
            | DeviceEvent::MouseDragEnd { .. }
                if cursor_driven =>
            {
                true
            }
            DeviceEvent::MouseClick { button: MouseButton::Left, .. } if cursor_driven => {
                self.confirm(&ctx.scene);
                true
            }
            DeviceEvent::MouseClick { button: MouseButton::Right, .. } if cursor_driven => {
                self.cancel(&ctx.scene);
                true
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Transform"
    }
}

/// The grips for `mode` about `frame`, along `space`'s axes: arrows, rings or
/// cubes along the axes, quads for the planes between them, and a ball for a
/// free move or a uniform scale.
fn gizmo(mode: Mode, frame: &Frame, space: Space) -> Vec<Handle> {
    let direction = |axis| frame.axis(axis, space);
    let grip = |constraint: Constraint, shape| Handle::new(constraint.grip(), shape, frame.pivot);

    let arms = |shape: HandleShape| {
        Axis::ALL.map(|axis| {
            grip(Constraint::Axis(axis, space), shape.clone())
                .with_direction(direction(axis))
                .with_reach(HandleReach::Arm)
                .with_color(axis.color())
                .with_scale(GIZMO_SCALE)
        })
    };
    let planes = Axis::ALL.map(|normal| {
        let (u, v) = plane_axes(normal);
        grip(Constraint::Plane(normal, space), HandleShape::Quad)
            .with_drag(DragKind::Plane)
            .with_direction(direction(normal))
            .with_reach(HandleReach::Offset((direction(u) + direction(v)) * QUAD_OFFSET))
            .with_color(u.color().mix(v.color(), 0.5).with_alpha(PLANE_ALPHA))
            .with_scale(QUAD_SCALE)
    });
    let ball = grip(Constraint::Free, HandleShape::Ball)
        .with_drag(DragKind::View)
        .with_color(BALL_COLOR);

    match mode {
        Mode::Translate => arms(HandleShape::Cone).into_iter().chain(planes).chain([ball]).collect(),
        Mode::Rotate => Axis::ALL
            .map(|axis| {
                grip(Constraint::Axis(axis, space), HandleShape::Ring)
                    .with_drag(DragKind::Plane)
                    .with_direction(direction(axis))
                    .with_color(axis.color())
                    .with_scale(RING_SCALE)
            })
            .to_vec(),
        Mode::Scale => arms(HandleShape::Cube).into_iter().chain(planes).chain([ball]).collect(),
    }
}

/// The two axes in the plane normal to `normal`.
fn plane_axes(normal: Axis) -> (Axis, Axis) {
    match normal {
        Axis::X => (Axis::Y, Axis::Z),
        Axis::Y => (Axis::Z, Axis::X),
        Axis::Z => (Axis::X, Axis::Y),
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

/// Where `point` lands on screen.
fn pixel(ctx: &EventContext, point: Point3) -> (f32, f32) {
    let p = ctx.camera.project_point_screen(point, ctx.size.0, ctx.size.1);
    (p.x as f32, p.y as f32)
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// The unit screen direction `direction` runs in from `point`; right when it
/// points straight at the camera.
fn screen_direction(ctx: &EventContext, point: Point3, direction: Vector3) -> (f32, f32) {
    let (from, to) = (pixel(ctx, point), pixel(ctx, point + direction));
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
    use duck_engine_common::consts::FRAC_PI_2;

    use super::*;
    use crate::event::AppEvent;
    use crate::input::PhysicalKey;
    use crate::scene::resource::{NodeFlags, NodeHandle};
    use crate::scene::Projection;
    use crate::selection::SelectionItem;

    const EPSILON: Real = 1e-6;
    /// Pixel round trips project and unproject through separate matrix
    /// inversions, so they hold to a small fraction of a pixel.
    const PIXEL_EPSILON: f32 = 0.05;
    const SIZE: (u32, u32) = (800, 600);

    /// An operator over one selected node, seen from `eye`.
    struct Bench {
        operator: TransformOperator,
        scene: Scene,
        camera: PositionedCamera,
        selection: SelectionManager,
        cursor: Option<(f32, f32)>,
        modifiers: Modifiers,
        node: NodeHandle,
    }

    impl Bench {
        /// A selected node at the origin, turned by `rotation`, seen from
        /// `eye`.
        fn new(rotation: Quaternion, eye: (Real, Real, Real)) -> Self {
            let scene = Scene::default();
            let transform = Transform { rotation, ..Transform::IDENTITY };
            let node = scene
                .lock()
                .add_node(None, Some("node".to_owned()), transform, NodeFlags::NONE)
                .unwrap();
            let mut selection = SelectionManager::new();
            selection.extend([SelectionItem::Node(node.id())]);
            let camera = PositionedCamera {
                eye: eye.into(),
                target: Point3::new(0.0, 0.0, 0.0),
                up: Vector3::unit_y(),
                aspect: SIZE.0 as Real / SIZE.1 as Real,
                projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
            };
            let mut bench = Self {
                operator: TransformOperator::new(),
                scene,
                camera,
                selection,
                cursor: None,
                modifiers: Modifiers::default(),
                node,
            };
            bench.cursor = Some(bench.pixel(Point3::new(0.0, 0.0, 0.0)));
            bench
        }

        /// A node on the origin, seen head-on down `-Z`.
        fn head_on() -> Self {
            Self::new(Transform::IDENTITY.rotation, (0.0, 0.0, 10.0))
        }

        fn send(&mut self, event: DeviceEvent) -> bool {
            let mut ctx = EventContext::new(
                SIZE,
                &mut self.cursor,
                self.scene.clone(),
                &mut self.camera,
                &mut self.selection,
                self.modifiers,
            );
            self.operator.dispatch(&Event::Device(event), &mut ctx)
        }

        fn update(&mut self) {
            self.send(DeviceEvent::Update { delta_time: 0.0 });
        }

        fn key(&mut self, key: Key) -> bool {
            let event = KeyEvent {
                physical_key: PhysicalKey::Unidentified,
                logical_key: key,
                state: ElementState::Pressed,
                repeat: false,
            };
            self.send(DeviceEvent::KeyboardInput { event, is_synthetic: false })
        }

        /// Presses the character keys of `keys` in turn.
        fn type_keys(&mut self, keys: &str) {
            for c in keys.chars() {
                assert!(self.key(Key::Character(c)), "{c:?} should be taken");
            }
        }

        fn move_to(&mut self, at: (f32, f32)) -> bool {
            self.cursor = Some(at);
            self.send(DeviceEvent::CursorMoved { position: (at.0 as f64, at.1 as f64) })
        }

        fn click(&mut self, button: MouseButton) -> bool {
            let position = self.cursor.unwrap();
            self.send(DeviceEvent::MouseClick { button, position, duration_ms: 50 })
        }

        fn cancel_by_escape(&mut self) -> bool {
            self.key(Key::Named(NamedKey::Escape))
        }

        fn cancel_by_right_click(&mut self) -> bool {
            self.click(MouseButton::Right)
        }

        fn transform(&self) -> Transform {
            self.scene.lock().get_node(self.node.id()).unwrap().transform()
        }

        fn pixel(&self, point: Point3) -> (f32, f32) {
            let p = self.camera.project_point_screen(point, SIZE.0, SIZE.1);
            (p.x as f32, p.y as f32)
        }

        /// The pixel `right` of and `up` above the origin's.
        fn off_centre(&self, right: f32, up: f32) -> (f32, f32) {
            let centre = self.pixel(Point3::new(0.0, 0.0, 0.0));
            (centre.0 + right, centre.1 - up)
        }

        fn constraint(&self) -> Option<Constraint> {
            self.operator.active.as_ref().map(|active| active.constraint)
        }

        /// The ids of the grips shown after a frame tick.
        fn grips(&mut self) -> Vec<HandleId> {
            self.update();
            let ctx = EventContext::new(
                SIZE,
                &mut self.cursor,
                self.scene.clone(),
                &mut self.camera,
                &mut self.selection,
                self.modifiers,
            );
            self.operator.shown(&ctx).iter().map(|handle| handle.id).collect()
        }
    }

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        distance(a, b) < PIXEL_EPSILON
    }

    fn quarter_turn_about_z() -> Quaternion {
        quaternion_from_axis_angle_safe(Vector3::unit_z(), FRAC_PI_2)
    }

    #[test]
    fn a_plane_move_keeps_the_node_under_the_cursor() {
        // Looking down onto the ground plane, where a drag along it is
        // foreshortened on screen.
        let mut bench = Bench::new(Transform::IDENTITY.rotation, (0.0, 4.0, 9.0));
        bench.type_keys("g");
        bench.modifiers = Modifiers { shift: true, ..Modifiers::default() };
        assert!(bench.key(Key::Character('Y')));
        bench.modifiers = Modifiers::default();
        assert_eq!(bench.constraint(), Some(Constraint::Plane(Axis::Y, Space::World)));

        let mut cursor = bench.cursor.unwrap();
        for _ in 0..10 {
            cursor = (cursor.0 - 14.0, cursor.1 - 22.0);
            assert!(bench.move_to(cursor));
            // Raw device motion is not the cursor, and moves nothing.
            assert!(bench.send(DeviceEvent::MouseMotion { delta: (-10.5, -16.5) }));
        }

        let position = bench.transform().position;
        assert!(close(bench.pixel(position), cursor), "{position:?} should be under {cursor:?}");
        assert!(position.y.abs() < EPSILON, "{position:?} left the ground plane");
    }

    #[test]
    fn axis_keys_cycle_world_then_local_then_free() {
        let mut bench = Bench::new(quarter_turn_about_z(), (0.0, 0.0, 10.0));
        bench.type_keys("gx");
        assert_eq!(bench.constraint(), Some(Constraint::Axis(Axis::X, Space::World)));
        bench.type_keys("x");
        assert_eq!(bench.constraint(), Some(Constraint::Axis(Axis::X, Space::Local)));

        // Turned a quarter about Z, the node's own X runs up the world's Y.
        let up = bench.off_centre(0.0, 100.0);
        bench.move_to(up);
        let position = bench.transform().position;
        assert!(position.x.abs() < EPSILON && position.y > 0.0, "{position:?}");

        bench.type_keys("x");
        assert_eq!(bench.constraint(), Some(Constraint::Free));
        bench.type_keys("z");
        assert_eq!(bench.constraint(), Some(Constraint::Axis(Axis::Z, Space::World)));
    }

    #[test]
    fn a_click_keeps_the_move_and_escape_or_a_right_click_puts_it_back() {
        let mut bench = Bench::head_on();
        bench.type_keys("gx");
        let right = bench.off_centre(80.0, 0.0);
        bench.move_to(right);
        assert!(bench.click(MouseButton::Left));
        assert!(bench.operator.active.is_none());
        let kept = bench.transform().position;
        assert!(close(bench.pixel(kept), right), "{kept:?}");

        for cancel in [Bench::cancel_by_escape, Bench::cancel_by_right_click] {
            bench.type_keys("gy");
            let up = bench.off_centre(80.0, 60.0);
            bench.move_to(up);
            assert!(cancel(&mut bench));
            assert!(bench.operator.active.is_none());
            let position = bench.transform().position;
            assert!((position - kept).magnitude() < EPSILON, "{position:?}");
        }
    }

    #[test]
    fn a_keyboard_scale_anchors_at_the_cursor() {
        // Started 120 px right of the pivot and carried 120 px further, the
        // grabbed point is twice as far out: a 2x scale.
        let mut bench = Bench::head_on();
        bench.cursor = Some(bench.off_centre(120.0, 0.0));
        bench.type_keys("s");
        bench.move_to(bench.off_centre(240.0, 0.0));
        let scale = bench.transform().scale;
        assert!((scale - Vector3::new(2.0, 2.0, 2.0)).magnitude() < 1e-3, "{scale:?}");
    }

    #[test]
    fn a_world_axis_scale_stretches_the_nodes_own_axis_and_keeps_it_turned() {
        // Turned a quarter about Z, the node's own Y runs along world -X, so a
        // world-X stretch is a stretch of its Y.
        let mut bench = Bench::new(quarter_turn_about_z(), (0.0, 0.0, 10.0));
        bench.cursor = Some(bench.off_centre(100.0, 0.0));
        bench.type_keys("sx");
        bench.move_to(bench.off_centre(300.0, 0.0));
        let transform = bench.transform();
        let scale = transform.scale;
        assert!((scale - Vector3::new(1.0, 3.0, 1.0)).magnitude() < 1e-3, "{scale:?}");
        let turned = transform.rotation.rotate_vector(Vector3::unit_x());
        assert!((turned - Vector3::unit_y()).magnitude() < EPSILON, "{turned:?}");
    }

    #[test]
    fn a_rotation_follows_the_cursor_round_the_pivot() {
        // From right of the pivot round to above it is a quarter turn about +Z.
        let mut bench = Bench::head_on();
        bench.cursor = Some(bench.off_centre(100.0, 0.0));
        bench.type_keys("rz");
        bench.move_to(bench.off_centre(70.0, 70.0));
        bench.move_to(bench.off_centre(0.0, 100.0));
        let turned = bench.transform().rotation.rotate_vector(Vector3::unit_x());
        assert!((turned - Vector3::unit_y()).magnitude() < 1e-4, "{turned:?}");
    }

    #[test]
    fn another_modes_key_turns_the_transform_into_that_mode() {
        let mut bench = Bench::head_on();
        bench.cursor = Some(bench.off_centre(100.0, 0.0));
        bench.type_keys("g");
        bench.move_to(bench.off_centre(100.0, 100.0));
        assert!(bench.transform().position.y > 0.0);

        // The move is dropped, and the cursor now turns the node about the view.
        bench.type_keys("r");
        assert_eq!(bench.operator.mode, Mode::Rotate);
        let transform = bench.transform();
        assert!(transform.position.to_vec().magnitude() < EPSILON, "{transform:?}");
        let turned = transform.rotation.rotate_vector(Vector3::unit_x());
        assert!((turned - Vector3::unit_x()).magnitude() > 0.1, "{turned:?}");
    }

    #[test]
    fn the_grips_show_the_last_mode_started() {
        let mut bench = Bench::head_on();
        assert_eq!(bench.grips().len(), 7, "arrows, quads and a ball");
        bench.type_keys("r");
        assert!(bench.key(Key::Named(NamedKey::Enter)));
        assert_eq!(bench.grips(), [HandleId(0), HandleId(1), HandleId(2)], "three rings");
    }

    #[test]
    fn nothing_selected_shows_no_grips_and_starts_nothing() {
        let mut bench = Bench::head_on();
        bench.selection.clear();
        assert!(bench.grips().is_empty());
        assert!(!bench.key(Key::Character('g')));
        assert!(!bench.key(Key::Character('x')));
        assert!(bench.operator.active.is_none());
    }

    #[test]
    fn a_grip_drag_moves_along_its_axis() {
        let mut bench = Bench::head_on();
        bench.update();
        // The X arrow's tip, out to the right of the pivot.
        let tip = bench.off_centre(80.0, 0.0);
        let start = DeviceEvent::MouseDragStart {
            button: MouseButton::Left,
            start_pos: tip,
            current_pos: tip,
        };
        assert!(bench.send(start), "the X arrow should be grabbed");
        assert_eq!(bench.constraint(), Some(Constraint::Axis(Axis::X, Space::World)));

        // Off the axis, the arrow carries the node only along it, by about as
        // far as the cursor went along it: under perspective the point nearest
        // a ray off the axis is not quite the cursor's own.
        let along = bench.off_centre(160.0, 40.0);
        assert!(bench.move_to(along));
        let end = DeviceEvent::MouseDragEnd {
            button: MouseButton::Left,
            start_pos: tip,
            end_pos: along,
        };
        assert!(bench.send(end));
        assert!(bench.operator.active.is_none());
        let position = bench.transform().position;
        let moved = bench.pixel(position).0 - bench.off_centre(0.0, 0.0).0;
        assert!((moved - 80.0).abs() < 1.0, "{position:?} moved {moved} px");
        assert!(position.y.abs() < EPSILON && position.z.abs() < EPSILON, "{position:?}");
    }

    #[test]
    fn drags_off_the_grips_pass_an_idle_operator() {
        // Or navigation could never orbit.
        let mut bench = Bench::head_on();
        bench.update();
        let corner = (20.0, 20.0);
        let drag = DeviceEvent::MouseDrag {
            button: MouseButton::Left,
            start_pos: corner,
            current_pos: (30.0, 25.0),
            delta: (10.0, 5.0),
        };
        let start = DeviceEvent::MouseDragStart {
            button: MouseButton::Left,
            start_pos: corner,
            current_pos: corner,
        };
        assert!(!bench.send(start));
        assert!(!bench.send(drag));
        assert!(!bench.send(DeviceEvent::MouseMotion { delta: (10.0, 5.0) }));
    }

    #[test]
    fn a_keyboard_transform_takes_drags_without_moving_on_them() {
        // The dispatcher synthesizes drags from the motion the cursor reports,
        // and navigation behind would orbit on them.
        let mut bench = Bench::head_on();
        bench.type_keys("g");
        let start = DeviceEvent::MouseDragStart {
            button: MouseButton::Left,
            start_pos: (20.0, 20.0),
            current_pos: (30.0, 25.0),
        };
        let drag = DeviceEvent::MouseDrag {
            button: MouseButton::Left,
            start_pos: (20.0, 20.0),
            current_pos: (30.0, 25.0),
            delta: (10.0, 5.0),
        };
        assert!(bench.send(start));
        assert!(bench.send(drag));
        assert!(bench.send(DeviceEvent::MouseMotion { delta: (10.0, 5.0) }));
        assert!(bench.transform().position.to_vec().magnitude() < EPSILON);
    }

    #[test]
    fn app_events_are_not_consumed() {
        let mut bench = Bench::head_on();
        let mut ctx = EventContext::new(
            SIZE,
            &mut bench.cursor,
            bench.scene.clone(),
            &mut bench.camera,
            &mut bench.selection,
            Modifiers::default(),
        );
        let event = Event::App(AppEvent::CameraInteractionStart);
        assert!(!bench.operator.dispatch(&event, &mut ctx));
    }
}
