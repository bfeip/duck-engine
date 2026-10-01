//! Draggable 3D manipulator handles, owned by whatever is being edited.
//!
//! A [`Handle`] is a description, not a resource: an id, a shape, a world
//! anchor and the locus a drag on it is confined to. Owners rebuild the whole
//! list from their own parameters whenever those change and hand it to
//! [`HandleSet::sync`], which reconciles it against the scene.
//!
//! [`HandleInput`] turns pointer events into [`HandleEvent`]s against a set.
//! A drag reports the world point its locus solves to, not a scalar, so the
//! owner decides what the motion means: [`HandleDrag::distance_along`] for a
//! dimension or radius, [`HandleDrag::angle_about`] for an angle.
//!
//! A handle's appearance is three independent choices, so that the combinations
//! nobody has needed yet cost nothing: [`HandleShape`] is the form of the grab,
//! [`HandleReach`] is where that grab sits relative to the anchor, and
//! [`DragKind`] is the locus a drag on it solves against. A ring that reads as
//! an angle control is a [`HandleShape::Ring`] over an ordinary
//! [`DragKind::Plane`]; a dimension grip out on a face with a line back to its
//! origin is a [`HandleShape::Cube`] with a [`HandleReach::Leader`].

mod set;
mod shape;

pub use set::HandleSet;

use duck_engine_common::{InnerSpace, Point3, Real, Vector3};
use duck_engine_scene::resource::MeshHandle;

use crate::common::RgbaColor;
use crate::event::{DeviceEvent, Event, EventContext};
use crate::input::{ElementState, Key, Modifiers, NamedKey};
use crate::operator::drag::{solve_drag, DragGeometry};

/// Default color of a handle: neutral, so it reads as a grip rather than
/// claiming an axis meaning the owner may not intend.
const HANDLE_COLOR: RgbaColor = RgbaColor { r: 0.85, g: 0.85, b: 0.85, a: 1.0 };

/// Identifies one handle within a set.
///
/// Owners mint their own values — typically named constants, one per
/// parameter — and the machinery only ever compares them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HandleId(pub u32);

/// The locus a handle's drag point is confined to, through its anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DragKind {
    /// Slide along the handle's direction.
    Axis,
    /// Slide in the plane whose normal is the handle's direction.
    Plane,
    /// Slide in the plane parallel to the image plane. The direction is unused.
    View,
}

/// The form of a handle's grab.
///
/// Built at unit size about the origin along `+Y` — the axis the mesh
/// primitives already build along — and turned onto the handle's direction when
/// the set places it. The renderer holds it at a constant on-screen size.
///
/// Where the grab sits relative to the anchor, and what connects the two, is the
/// separate concern of [`HandleReach`].
#[derive(Debug, Clone, PartialEq)]
pub enum HandleShape {
    /// Cone pointing along the direction. Offsets and distances.
    Cone,
    /// Cube. Extents.
    Cube,
    /// Sphere. Free grabs.
    Ball,
    /// Torus in the plane normal to the direction. Angles.
    Ring,
    /// Square in the plane normal to the direction. Plane-constrained grabs.
    Quad,
    /// Caller-supplied geometry, authored to the same convention.
    ///
    /// A scene resource rather than a bare [`Mesh`](duck_engine_scene::resource::Mesh)
    /// so that a set polled every frame can tell one custom shape from another
    /// by id, and so the geometry is uploaded once however often the handles
    /// around it are rebuilt.
    Custom(MeshHandle),
}

/// How far a handle's grab sits from its anchor, and what bridges the gap.
///
/// The anchor is always the point the drag locus passes through. `Arm` puts the
/// grab out at the end of a stem, so there the anchor is the tail; every other
/// reach leaves the grab on the anchor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandleReach {
    /// The grab sits on the anchor.
    None,
    /// A stem from the anchor out along the direction, with the grab at its far
    /// end. Constant on-screen length, like the shape.
    Arm,
    /// A displacement from the anchor, in world axes but at a constant
    /// on-screen magnitude, with nothing drawn between. For a grab that wants
    /// to sit clear of its anchor without a visible connection.
    Offset(Vector3),
    /// A leader line running back to a world point, tying the grab to whatever
    /// it measures from.
    ///
    /// Unlike `Arm`, the leader spans the real world distance rather than a
    /// constant pixel length, so it still meets that point at any zoom. It is
    /// annotation only and never pickable.
    Leader(Point3),
}

impl HandleReach {
    /// Whether a handle displaying `self` can be updated into one displaying
    /// `other` by writing transforms, rather than rebuilding its geometry.
    ///
    /// Only a leader's origin is a transform; `Arm` and `Offset` are baked into
    /// the mesh, and `Leader` alone owns a second node.
    fn reusable_as(&self, other: &HandleReach) -> bool {
        match (self, other) {
            (HandleReach::Leader(_), HandleReach::Leader(_)) => true,
            (a, b) => a == b,
        }
    }
}

/// One draggable manipulator.
///
/// Rebuilt from the owner's parameters whenever they change, so it holds no
/// state of its own — the scene resources backing it live in the [`HandleSet`].
#[derive(Debug, Clone)]
pub struct Handle {
    pub id: HandleId,
    pub shape: HandleShape,
    pub drag: DragKind,
    /// World point the handle sits at, and that its drag locus passes through.
    pub anchor: Point3,
    /// Orients both the shape and the drag locus.
    pub direction: Vector3,
    pub color: RgbaColor,
    /// Where the grab sits relative to the anchor. See [`HandleReach`].
    pub reach: HandleReach,
}

impl Handle {
    /// A handle at `anchor`, pointing along `+Y` and dragging along it.
    pub fn new(id: HandleId, shape: HandleShape, anchor: Point3) -> Self {
        Self {
            id,
            shape,
            drag: DragKind::Axis,
            anchor,
            direction: Vector3::unit_y(),
            color: HANDLE_COLOR,
            reach: HandleReach::None,
        }
    }

    /// Orients the shape and the drag locus. Need not be normalized.
    pub fn with_direction(mut self, direction: Vector3) -> Self {
        self.direction = direction;
        self
    }

    /// Sets where the grab sits relative to the anchor. See [`HandleReach`].
    pub fn with_reach(mut self, reach: HandleReach) -> Self {
        self.reach = reach;
        self
    }

    pub fn with_drag(mut self, drag: DragKind) -> Self {
        self.drag = drag;
        self
    }

    pub fn with_color(mut self, color: RgbaColor) -> Self {
        self.color = color;
        self
    }
}

/// A drag in progress on one handle.
///
/// Both points are solved against the handle's locus, so the offset between
/// them is the drag in world units. It is measured from the grab rather than
/// accumulated per event, which is why an owner snapshots its parameters on
/// [`HandleEvent::Begin`] and edits from that snapshot.
#[derive(Debug, Clone, Copy)]
pub struct HandleDrag {
    pub id: HandleId,
    /// Where the locus solved when the handle was grabbed.
    pub grab: Point3,
    /// Where it solves now.
    pub point: Point3,
    pub modifiers: Modifiers,
}

impl HandleDrag {
    /// The world-space offset from the grab.
    pub fn delta(&self) -> Vector3 {
        self.point - self.grab
    }

    /// The drag projected onto `axis`, in world units. Signed: negative when
    /// the drag ran against `axis`.
    pub fn distance_along(&self, axis: Vector3) -> Real {
        if axis.magnitude2() < Real::EPSILON {
            return 0.0;
        }
        self.delta().dot(axis.normalize())
    }

    /// The angle in radians the drag swept about `axis` through `pivot`,
    /// positive counter-clockwise looking down `axis`.
    ///
    /// Zero when either end sits on the axis, where no angle is defined. The
    /// result is in `-π..=π`: a drag that sweeps further than half a turn in one
    /// event wraps, so an owner that must track full turns integrates its own.
    pub fn angle_about(&self, pivot: Point3, axis: Vector3) -> Real {
        if axis.magnitude2() < Real::EPSILON {
            return 0.0;
        }
        let axis = axis.normalize();
        // Drop the component along the axis: only the swing about it is angular.
        let flatten = |p: Point3| {
            let v = p - pivot;
            v - axis * axis.dot(v)
        };
        let (from, to) = (flatten(self.grab), flatten(self.point));
        if from.magnitude2() < Real::EPSILON || to.magnitude2() < Real::EPSILON {
            return 0.0;
        }
        // atan2 of the cross (signed about the axis) against the dot keeps the
        // sign and stays accurate near both 0 and π, unlike acos of the dot.
        from.cross(to).dot(axis).atan2(from.dot(to))
    }
}

/// What happened to a handle.
#[derive(Debug, Clone, Copy)]
pub enum HandleEvent {
    /// Grabbed. The cue to snapshot whatever [`HandleEvent::Drag`] will edit.
    Begin(HandleId),
    /// Moved. Carries the total offset from the grab, not an increment.
    Drag(HandleDrag),
    /// Released, keeping the drag.
    End(HandleId),
    /// Abandoned. The owner should restore its snapshot.
    Cancel(HandleId),
}

/// What a dispatched event did to a handle set.
pub enum HandleOutcome {
    /// Not a handle interaction; the event should continue down the stack.
    Ignored,
    /// Consumed, with nothing for the owner to do.
    Consumed,
    /// Consumed, and the owner should act on this.
    Event(HandleEvent),
}

/// The handle a drag is holding, and what that drag is solved against.
struct Grab {
    id: HandleId,
    geometry: DragGeometry,
    /// Pixel the drag is measured from. The grab point is re-solved from it
    /// each time rather than cached, so both ends of a drag are always read in
    /// the same frame.
    anchor_screen: (f32, f32),
}

/// Turns pointer events into drags on a [`HandleSet`].
///
/// Holds no scene state: the caller owns the set and applies the resulting
/// events. Feed every event through [`dispatch`](Self::dispatch) ahead of
/// whatever else would consume pointer input.
#[derive(Default)]
pub struct HandleInput {
    grabbed: Option<Grab>,
}

impl HandleInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_dragging(&self) -> bool {
        self.grabbed.is_some()
    }

    /// The handle currently held, if any.
    pub fn grabbed(&self) -> Option<HandleId> {
        self.grabbed.as_ref().map(|g| g.id)
    }

    /// Drops any drag without an event context, for teardown that happens
    /// outside dispatch. Idempotent.
    pub fn abort(&mut self) {
        self.grabbed = None;
    }

    /// Feeds one event to the handle set.
    pub fn dispatch(
        &mut self,
        event: &Event,
        set: &mut HandleSet,
        ctx: &mut EventContext,
    ) -> HandleOutcome {
        let Event::Device(event) = event else {
            return HandleOutcome::Ignored;
        };

        match event {
            DeviceEvent::MouseDragStart { start_pos, .. } if !self.is_dragging() => {
                self.begin(*start_pos, set, ctx)
            }

            // Consumed but deliberately not applied: the dispatcher synthesizes
            // this from the `MouseMotion` that `CursorMoved` also reports, so
            // acting on both would double-count the drag. It must still be
            // consumed, or the navigation operator orbits underneath.
            DeviceEvent::MouseDrag { .. } | DeviceEvent::MouseMotion { .. }
                if self.is_dragging() =>
            {
                HandleOutcome::Consumed
            }

            // The absolute cursor, not the raw motion delta: pointer
            // acceleration makes the two different quantities, and only this
            // one keeps the grabbed point under the cursor.
            DeviceEvent::CursorMoved { position } => {
                let position = (position.0 as f32, position.1 as f32);
                match self.is_dragging() {
                    true => self.solve(position, ctx),
                    false => self.hover(position, set, ctx),
                }
            }

            DeviceEvent::MouseDragEnd { .. } => match self.grabbed.take() {
                Some(grab) => HandleOutcome::Event(HandleEvent::End(grab.id)),
                None => HandleOutcome::Ignored,
            },

            DeviceEvent::KeyboardInput { event, .. }
                if self.is_dragging()
                    && event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape) =>
            {
                let grab = self.grabbed.take().expect("a drag is in progress");
                HandleOutcome::Event(HandleEvent::Cancel(grab.id))
            }

            _ => HandleOutcome::Ignored,
        }
    }

    /// Hit-tests a press and takes the handle under it.
    ///
    /// The ray comes from `start_pos` rather than the dispatcher's current
    /// cursor: `MouseDragStart` is synthesized once the cursor has already
    /// passed the drag threshold, so the current cursor is several pixels off
    /// the pixel the user actually grabbed at.
    fn begin(
        &mut self,
        start_pos: (f32, f32),
        set: &mut HandleSet,
        ctx: &mut EventContext,
    ) -> HandleOutcome {
        let (width, height) = ctx.size;
        let ray = ctx.camera.ray_from_screen_point(start_pos.0, start_pos.1, width, height);

        let Some(id) = set.pick(ray, &ctx.scene, ctx.camera, ctx.size) else {
            return HandleOutcome::Ignored;
        };
        let Some(geometry) = set.drag_geometry(id, ctx.camera) else {
            return HandleOutcome::Ignored;
        };
        // Refuse a grab whose locus does not solve under the grabbed pixel —
        // edge-on to the camera, there is no drag to measure.
        if geometry.solve(&ray).is_none() {
            return HandleOutcome::Ignored;
        }

        set.set_highlight(Some(id), &ctx.scene);
        self.grabbed = Some(Grab { id, geometry, anchor_screen: start_pos });
        HandleOutcome::Event(HandleEvent::Begin(id))
    }

    /// Resolves the drag at `cursor` against the grabbed handle's locus.
    fn solve(&mut self, cursor: (f32, f32), ctx: &mut EventContext) -> HandleOutcome {
        let grab = self.grabbed.as_ref().expect("a drag is in progress");

        // A degenerate solve means the locus has swung edge-on to the camera.
        // Consume without reporting, so the owner holds its last offset rather
        // than jumping to a mirrored one.
        let Some((grab_point, point)) =
            solve_drag(&grab.geometry, grab.anchor_screen, cursor, ctx.camera, ctx.size)
        else {
            return HandleOutcome::Consumed;
        };

        HandleOutcome::Event(HandleEvent::Drag(HandleDrag {
            id: grab.id,
            grab: grab_point,
            point,
            modifiers: ctx.modifiers,
        }))
    }

    /// Lights the handle under an idle cursor.
    ///
    /// Never consumes: a cursor move is information everything downstream wants,
    /// and swallowing it here would freeze the rest of the stack.
    fn hover(
        &mut self,
        position: (f32, f32),
        set: &mut HandleSet,
        ctx: &mut EventContext,
    ) -> HandleOutcome {
        if set.is_empty() {
            return HandleOutcome::Ignored;
        }
        let (width, height) = ctx.size;
        let ray = ctx.camera.ray_from_screen_point(position.0, position.1, width, height);
        let hit = set.pick(ray, &ctx.scene, ctx.camera, ctx.size);
        set.set_highlight(hit, &ctx.scene);
        HandleOutcome::Ignored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::Zero;

    const EPSILON: crate::common::Real = 1e-4;

    fn drag(grab: Point3, point: Point3) -> HandleDrag {
        HandleDrag { id: HandleId(0), grab, point, modifiers: Modifiers::default() }
    }

    #[test]
    fn distance_along_ignores_the_off_axis_component() {
        let d = drag(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 9.0, -4.0));
        assert!((d.distance_along(Vector3::unit_x()) - 3.0).abs() < EPSILON);
        assert!((d.distance_along(Vector3::unit_z()) + 4.0).abs() < EPSILON);
    }

    #[test]
    fn distance_along_ignores_axis_magnitude() {
        let d = drag(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
        let unit = d.distance_along(Vector3::unit_x());
        let scaled = d.distance_along(Vector3::unit_x() * 10.0);
        assert!((unit - scaled).abs() < EPSILON);
    }

    #[test]
    fn distance_along_a_degenerate_axis_is_zero() {
        let d = drag(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0));
        assert_eq!(d.distance_along(Vector3::zero()), 0.0);
    }

    #[test]
    fn angle_about_is_signed_by_the_axis() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        // A quarter turn from +x to +y is positive about +z, negative about -z.
        let d = drag(Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        assert!((d.angle_about(pivot, Vector3::unit_z()) - duck_engine_common::consts::FRAC_PI_2).abs() < EPSILON);
        assert!((d.angle_about(pivot, -Vector3::unit_z()) + duck_engine_common::consts::FRAC_PI_2).abs() < EPSILON);
    }

    #[test]
    fn angle_about_ignores_the_component_along_the_axis_and_the_radius() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        // Same quarter turn, but at a different radius and lifted along +z.
        let d = drag(Point3::new(5.0, 0.0, 2.0), Point3::new(0.0, 0.3, -7.0));
        assert!((d.angle_about(pivot, Vector3::unit_z()) - duck_engine_common::consts::FRAC_PI_2).abs() < EPSILON);
    }

    #[test]
    fn angle_about_is_measured_from_the_pivot() {
        // The same two world points sweep a different angle about a moved pivot.
        let d = drag(Point3::new(1.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0));
        let origin = d.angle_about(Point3::new(0.0, 0.0, 0.0), Vector3::unit_z());
        let offset = d.angle_about(Point3::new(-1.0, -1.0, 0.0), Vector3::unit_z());
        assert!((origin - duck_engine_common::consts::FRAC_PI_2).abs() < EPSILON);
        assert!((offset - origin).abs() > EPSILON);
    }

    #[test]
    fn angle_about_an_end_on_the_axis_is_zero() {
        let pivot = Point3::new(0.0, 0.0, 0.0);
        // The grab sits on the axis, so no starting direction is defined.
        let d = drag(Point3::new(0.0, 0.0, 3.0), Point3::new(0.0, 1.0, 0.0));
        assert_eq!(d.angle_about(pivot, Vector3::unit_z()), 0.0);
    }

    #[test]
    fn a_handle_defaults_to_an_axis_drag_along_z() {
        let h = Handle::new(HandleId(7), HandleShape::Ball, Point3::new(1.0, 2.0, 3.0));
        assert_eq!(h.drag, DragKind::Axis);
        assert_eq!(h.direction, Vector3::unit_y());
        assert_eq!(h.color, HANDLE_COLOR);
    }

    // --- the input state machine ---

    use crate::selection::SelectionManager;
    use duck_engine_scene::{PositionedCamera, Projection, Scene, SceneData};

    const VIEWPORT: (u32, u32) = (800, 600);
    /// The pixel a camera on +Z sees the world origin through.
    const CENTRE: (f32, f32) = (400.0, 300.0);

    /// The pieces an [`EventContext`] borrows, owned by the test.
    struct Parts {
        cursor: Option<(f32, f32)>,
        scene: Scene,
        selection: SelectionManager,
        camera: PositionedCamera,
    }

    fn parts() -> Parts {
        Parts {
            cursor: Some(CENTRE),
            scene: Scene::new(SceneData::new()),
            selection: SelectionManager::new(),
            camera: PositionedCamera {
                eye: Point3::new(0.0, 0.0, 5.0),
                target: Point3::new(0.0, 0.0, 0.0),
                up: Vector3::unit_y(),
                aspect: VIEWPORT.0 as Real / VIEWPORT.1 as Real,
                projection: Projection::Perspective { fovy: 45.0, znear: 0.1, zfar: 100.0 },
            },
        }
    }

    fn context(parts: &mut Parts) -> EventContext<'_> {
        EventContext {
            size: VIEWPORT,
            cursor_position: &mut parts.cursor,
            scene: parts.scene.clone(),
            camera: &mut parts.camera,
            selection: &mut parts.selection,
            modifiers: Modifiers::default(),
            emit_queue: Vec::new(),
        }
    }

    /// A set holding one ball at the origin, draggable across the view plane so
    /// any screen motion resolves.
    fn ball_at_origin(scene: &Scene) -> (HandleSet, HandleId) {
        let id = HandleId(1);
        let mut set = HandleSet::new();
        set.sync(
            &[Handle::new(id, HandleShape::Ball, Point3::new(0.0, 0.0, 0.0))
                .with_drag(DragKind::View)],
            scene,
        );
        (set, id)
    }

    fn drag_start(pos: (f32, f32)) -> Event {
        Event::Device(DeviceEvent::MouseDragStart {
            button: crate::input::MouseButton::Left,
            start_pos: pos,
            current_pos: pos,
        })
    }

    fn cursor_moved(pos: (f32, f32)) -> Event {
        Event::Device(DeviceEvent::CursorMoved { position: (pos.0 as f64, pos.1 as f64) })
    }

    fn escape() -> Event {
        Event::Device(DeviceEvent::KeyboardInput {
            event: crate::input::KeyEvent {
                physical_key: crate::input::PhysicalKey::Unidentified,
                logical_key: Key::Named(NamedKey::Escape),
                state: ElementState::Pressed,
                repeat: false,
            },
            is_synthetic: false,
        })
    }

    #[test]
    fn a_press_on_a_handle_grabs_it() {
        let mut parts = parts();
        let (mut set, id) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();

        let outcome = input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));
        assert!(matches!(outcome, HandleOutcome::Event(HandleEvent::Begin(got)) if got == id));
        assert_eq!(input.grabbed(), Some(id));
    }

    /// A press that hits nothing has to fall through, or the tool underneath
    /// would never see a click again.
    #[test]
    fn a_press_that_misses_every_handle_is_ignored() {
        let mut parts = parts();
        let (mut set, _) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();

        let outcome = input.dispatch(&drag_start((5.0, 5.0)), &mut set, &mut context(&mut parts));
        assert!(matches!(outcome, HandleOutcome::Ignored));
        assert!(!input.is_dragging());
    }

    /// The dispatcher synthesizes `MouseDrag` from the same motion that
    /// `CursorMoved` reports. Acting on both would double-count the drag, but
    /// leaving it unconsumed would let the camera orbit under the grip.
    #[test]
    fn mouse_drag_is_consumed_without_reporting_a_drag() {
        let mut parts = parts();
        let (mut set, _) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        let event = Event::Device(DeviceEvent::MouseDrag {
            button: crate::input::MouseButton::Left,
            start_pos: CENTRE,
            current_pos: (500.0, 300.0),
            delta: (100.0, 0.0),
        });
        assert!(matches!(
            input.dispatch(&event, &mut set, &mut context(&mut parts)),
            HandleOutcome::Consumed
        ));

        let motion = Event::Device(DeviceEvent::MouseMotion { delta: (100.0, 0.0) });
        assert!(matches!(
            input.dispatch(&motion, &mut set, &mut context(&mut parts)),
            HandleOutcome::Consumed
        ));
    }

    #[test]
    fn a_cursor_move_while_grabbed_reports_the_offset_from_the_grab() {
        let mut parts = parts();
        let (mut set, id) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        let outcome =
            input.dispatch(&cursor_moved((500.0, 300.0)), &mut set, &mut context(&mut parts));
        let HandleOutcome::Event(HandleEvent::Drag(drag)) = outcome else {
            panic!("a grabbed handle reports its drag");
        };
        assert_eq!(drag.id, id);
        // Dragging right moves the point in +x, and nowhere else.
        assert!(drag.delta().x > 0.0);
        assert!(drag.delta().y.abs() < EPSILON);
    }

    /// The offset is measured from the grab every time, not accumulated, so
    /// returning to the grab pixel returns to a zero offset.
    #[test]
    fn a_drag_back_to_the_grab_pixel_reports_no_offset() {
        let mut parts = parts();
        let (mut set, _) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        input.dispatch(&cursor_moved((500.0, 300.0)), &mut set, &mut context(&mut parts));
        let outcome = input.dispatch(&cursor_moved(CENTRE), &mut set, &mut context(&mut parts));

        let HandleOutcome::Event(HandleEvent::Drag(drag)) = outcome else {
            panic!("a grabbed handle reports its drag");
        };
        assert!(drag.delta().magnitude() < EPSILON);
    }

    #[test]
    fn releasing_ends_the_drag() {
        let mut parts = parts();
        let (mut set, id) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        let event = Event::Device(DeviceEvent::MouseDragEnd {
            button: crate::input::MouseButton::Left,
            start_pos: CENTRE,
            end_pos: (500.0, 300.0),
        });
        let outcome = input.dispatch(&event, &mut set, &mut context(&mut parts));
        assert!(matches!(outcome, HandleOutcome::Event(HandleEvent::End(got)) if got == id));
        assert!(!input.is_dragging());
    }

    #[test]
    fn escape_cancels_the_drag() {
        let mut parts = parts();
        let (mut set, id) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        let outcome = input.dispatch(&escape(), &mut set, &mut context(&mut parts));
        assert!(matches!(outcome, HandleOutcome::Event(HandleEvent::Cancel(got)) if got == id));
        assert!(!input.is_dragging());
    }

    /// Escape with nothing grabbed belongs to whatever is behind: for a tool,
    /// that is "discard the operation".
    #[test]
    fn escape_without_a_grab_falls_through() {
        let mut parts = parts();
        let (mut set, _) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();

        assert!(matches!(
            input.dispatch(&escape(), &mut set, &mut context(&mut parts)),
            HandleOutcome::Ignored
        ));
    }

    /// Hovering lights the handle but must not consume: everything downstream
    /// needs cursor moves too.
    #[test]
    fn hovering_highlights_without_consuming() {
        let mut parts = parts();
        let (mut set, id) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();

        let outcome = input.dispatch(&cursor_moved(CENTRE), &mut set, &mut context(&mut parts));
        assert!(matches!(outcome, HandleOutcome::Ignored));
        assert_eq!(set.highlighted(), Some(id));

        // Moving off clears it again.
        input.dispatch(&cursor_moved((5.0, 5.0)), &mut set, &mut context(&mut parts));
        assert_eq!(set.highlighted(), None);
    }

    #[test]
    fn abort_drops_a_grab_without_an_event() {
        let mut parts = parts();
        let (mut set, _) = ball_at_origin(&parts.scene);
        let mut input = HandleInput::new();
        input.dispatch(&drag_start(CENTRE), &mut set, &mut context(&mut parts));

        input.abort();
        assert!(!input.is_dragging());
        input.abort();
        assert!(!input.is_dragging());
    }
}
