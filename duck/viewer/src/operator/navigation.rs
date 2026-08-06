use serde::{Deserialize, Serialize};

use crate::bindings::{InputBinding, InputMap};
use crate::common;
use crate::scene::camera::PositionedCamera;
use crate::scene::geom_query::{pick_all_from_ray, RayPickQuery};
use crate::event::{AppEvent, DeviceEvent, Event, EventContext};
use crate::input::{Key, Modifiers, MouseButton, MouseScrollDelta};
use crate::operator::Operator;
use crate::scene_scale;

mod turntable;
mod trackball;
mod walk;

use turntable::TurntableState;
use trackball::TrackballState;
use walk::WalkState;

pub(super) const ORBIT_SENSITIVITY: f32 = 0.005;

pub(super) fn pan(dx: f32, dy: f32, camera: &mut PositionedCamera, viewport: (u32, u32)) {
    let (width, height) = viewport;
    let pivot = camera.target;
    let movement_plane = common::Plane::from_point(camera.forward(), pivot);
    let screen = camera.project_point_screen(pivot, width, height);
    let diff_ray = camera.ray_from_screen_point(screen.x - dx, screen.y - dy, width, height);
    if let Some((_, new_pivot)) = diff_ray.intersect_plane(&movement_plane) {
        let offset = new_pivot - pivot;
        camera.eye += offset;
        camera.target += offset;
    }
}

pub(super) fn zoom_radius(radius: f32, delta: f32, model_radius: f32) -> f32 {
    let zoom_factor = scene_scale::zoom_factor();
    let factor = if delta > 0.0 { 1.0 - zoom_factor } else { 1.0 + zoom_factor };
    (radius * factor.powf(delta.abs())).clamp(
        scene_scale::min_camera_radius(model_radius),
        scene_scale::max_camera_radius(model_radius),
    )
}

/// The navigation interaction mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NavigationMode {
    /// Orbit/pan/zoom around a target point with the camera kept upright (default).
    #[default]
    Turntable,
    /// Orbit/pan/zoom allowing unrestricted camera roll. Good for CAD workflows
    /// where parts may be in arbitrary orientations.
    Trackball,
    /// First-person walk with movement keys and mouse look.
    Walk,
}

/// Semantic actions for the navigation operator across all modes.
///
/// Mouse drag actions apply to turntable/trackball and walk modes:
/// - [`Orbit`] rotates around the target (or free-looks in walk mode)
/// - [`Pan`] moves the camera parallel to the view plane
/// - [`PivotOrbit`] picks a new pivot under the cursor before orbiting
/// - [`Zoom`] adjusts camera distance (scroll or pinch)
///
/// Key actions apply exclusively to walk mode:
/// - [`MoveForward`], [`MoveBackward`], [`MoveLeft`], [`MoveRight`]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NavigationAction {
    /// Orbit (turntable/trackball) or free-look (walk).
    Orbit,
    /// Pan the camera parallel to the view plane.
    Pan,
    /// Pick a new pivot point under the cursor, then orbit around it.
    PivotOrbit,
    /// Zoom in/out via scroll wheel or pinch gesture.
    Zoom,
    /// Walk mode: move forward.
    MoveForward,
    /// Walk mode: move backward.
    MoveBackward,
    /// Walk mode: strafe left.
    MoveLeft,
    /// Walk mode: strafe right.
    MoveRight,
}

/// Operator for camera navigation. See [`NavigationMode`] for available modes.
///
/// **Turntable / Trackball mode** (default: Turntable):
/// - Orbit drag: Orbit camera around target
/// - Pan drag: Pan camera perpendicular to view direction
/// - Pivot-orbit drag: Orbit around point under cursor
/// - Mouse wheel: Zoom in/out
///
/// **Walk mode**:
/// - Movement key bindings: Move forward/backward/strafe
/// - Orbit drag: Look around
///
/// Use [`NavigationOperator::mode`] / [`NavigationOperator::set_mode`] to change the active
/// mode. Use [`NavigationOperator::bindings`] to remap any action.
pub struct NavigationOperator {
    mode: NavigationMode,
    turntable: TurntableState,
    trackball: TrackballState,
    walk: WalkState,
    /// Whether a drag-based navigation interaction is currently active.
    interacting: bool,
    /// All navigation bindings: orbit/pan/zoom drags and walk movement keys.
    pub bindings: InputMap<NavigationAction>,
}

impl NavigationOperator {
    /// Creates a new navigation operator with default bindings.
    pub fn new() -> Self {
        let bindings = InputMap::new()
            // Orbit/pan/pivot drag
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Left, modifiers: Modifiers::default() },
                NavigationAction::Orbit,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Left, modifiers: Modifiers::default() },
                NavigationAction::Orbit,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Right, modifiers: Modifiers::default() },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Right, modifiers: Modifiers::default() },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDragStart {
                    button: MouseButton::Left,
                    modifiers: Modifiers { shift: true, ..Default::default() },
                },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDrag {
                    button: MouseButton::Left,
                    modifiers: Modifiers { shift: true, ..Default::default() },
                },
                NavigationAction::Pan,
            )
            .bind(
                InputBinding::MouseDragStart { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            .bind(
                InputBinding::MouseDrag { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            .bind(
                InputBinding::MouseDragEnd { button: MouseButton::Middle, modifiers: Modifiers::default() },
                NavigationAction::PivotOrbit,
            )
            // Zoom
            .bind(InputBinding::MouseScroll, NavigationAction::Zoom)
            // Walk movement keys
            .bind(
                InputBinding::Key { key: Key::Character('w'), modifiers: Modifiers::default() },
                NavigationAction::MoveForward,
            )
            .bind(
                InputBinding::Key { key: Key::Character('s'), modifiers: Modifiers::default() },
                NavigationAction::MoveBackward,
            )
            .bind(
                InputBinding::Key { key: Key::Character('a'), modifiers: Modifiers::default() },
                NavigationAction::MoveLeft,
            )
            .bind(
                InputBinding::Key { key: Key::Character('d'), modifiers: Modifiers::default() },
                NavigationAction::MoveRight,
            );

        Self {
            mode: NavigationMode::default(),
            turntable: TurntableState::new(),
            trackball: TrackballState::new(),
            walk: WalkState::new(),
            interacting: false,
            bindings,
        }
    }

    /// Returns the current navigation mode.
    pub fn mode(&self) -> NavigationMode {
        self.mode
    }

    /// Sets the navigation mode.
    pub fn set_mode(&mut self, mode: NavigationMode) {
        self.mode = mode;
    }

    fn handle_drag_start(
        &mut self,
        action: NavigationAction,
        start_pos: (f32, f32),
        ctx: &mut EventContext,
    ) -> bool {
        let camera = ctx.camera();
        let scene_center = {
            ctx.scene.bounding().bounds.map(|b| b.center()).unwrap_or(camera.target)
        };

        match (self.mode, action) {
            (NavigationMode::Turntable, NavigationAction::Orbit | NavigationAction::Pan) => {
                self.turntable.init_with_pivot(&camera, scene_center);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Orbit | NavigationAction::Pan) => {
                self.trackball.init(&camera, scene_center);
                true
            }
            (_, NavigationAction::PivotOrbit) => {
                let ray = camera.ray_from_screen_point(
                    start_pos.0, start_pos.1, ctx.size.0, ctx.size.1,
                );
                let hits = {
                    pick_all_from_ray(&RayPickQuery::faces(ray), &ctx.scene)
                };
                let pivot = hits.first().map(|h| h.hit_point).unwrap_or(scene_center);
                match self.mode {
                    NavigationMode::Turntable => self.turntable.init_with_pivot(&camera, pivot),
                    NavigationMode::Trackball => self.trackball.init(&camera, pivot),
                    NavigationMode::Walk => {}
                }
                true
            }
            (NavigationMode::Walk, NavigationAction::Orbit) => {
                self.walk.init_from_camera(&camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag(
        &mut self,
        action: NavigationAction,
        delta: &(f32, f32),
        camera: &mut PositionedCamera,
        viewport: (u32, u32),
    ) -> bool {
        match (self.mode, action) {
            (NavigationMode::Turntable, NavigationAction::Orbit) => {
                self.turntable.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Turntable, NavigationAction::Pan) => {
                self.turntable.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Orbit) => {
                self.trackball.handle_orbit(delta.0 as f64, delta.1 as f64, camera);
                true
            }
            (NavigationMode::Trackball, NavigationAction::Pan) => {
                self.trackball.handle_pan(delta.0, delta.1, camera, viewport);
                true
            }
            (NavigationMode::Turntable | NavigationMode::Trackball, NavigationAction::PivotOrbit) => {
                match self.mode {
                    NavigationMode::Turntable => {
                        self.turntable.handle_orbit(delta.0 as f64, delta.1 as f64, camera)
                    }
                    NavigationMode::Trackball => {
                        self.trackball.handle_orbit(delta.0 as f64, delta.1 as f64, camera)
                    }
                    NavigationMode::Walk => {}
                }
                true
            }
            (NavigationMode::Walk, NavigationAction::Orbit) => {
                self.walk.handle_look(delta.0, delta.1, camera);
                true
            }
            _ => false,
        }
    }

    fn handle_drag_end(&mut self, action: NavigationAction) {
        if action == NavigationAction::PivotOrbit {
            self.turntable.pivot = None;
            self.trackball.pivot = None;
        }
    }

    fn handle_wheel(
        &mut self,
        scroll_amount: f32,
        camera: &mut PositionedCamera,
        model_radius: f32,
    ) -> bool {
        match self.mode {
            NavigationMode::Turntable => {
                self.turntable.init(camera);
                self.turntable.handle_zoom(scroll_amount, camera, model_radius);
                true
            }
            NavigationMode::Trackball => {
                self.trackball.handle_zoom(scroll_amount, camera, model_radius);
                true
            }
            NavigationMode::Walk => false,
        }
    }

    fn handle_update(
        &mut self,
        delta_time: f32,
        camera: &mut PositionedCamera,
        model_radius: f32,
    ) -> bool {
        if self.mode != NavigationMode::Walk {
            self.walk.reset_keys();
            return false;
        }
        if delta_time > 1.0 {
            return false;
        }
        if self.walk.is_moving() {
            self.walk.apply_movement(camera, delta_time, model_radius);
            return true;
        }
        false
    }
}

impl Operator for NavigationOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseDragStart { button, start_pos, .. } => {
                let actions = self.bindings.actions_for_drag_start(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= self.handle_drag_start(action, *start_pos, ctx);
                }
                if handled && !self.interacting {
                    self.interacting = true;
                    ctx.emit(AppEvent::CameraInteractionStart);
                }
                handled
            }
            DeviceEvent::MouseDrag { button, delta, .. } => {
                let actions = self.bindings.actions_for_drag(*button, ctx.modifiers).to_vec();
                let size = ctx.size;
                let mut handled = false;
                ctx.with_camera_mut(|cam| {
                    for action in &actions {
                        handled |= self.handle_drag(*action, delta, cam, size);
                    }
                });
                handled
            }
            DeviceEvent::MouseDragEnd { button, .. } => {
                // Modifiers may not be held on release; match with no modifiers.
                let actions =
                    self.bindings.actions_for_drag_end(*button, Modifiers::default()).to_vec();
                for action in actions {
                    self.handle_drag_end(action);
                }
                if self.interacting {
                    self.interacting = false;
                    ctx.emit(AppEvent::CameraInteractionEnd);
                }
                false
            }
            DeviceEvent::MouseWheel { delta } => {
                if !self.bindings.actions_for_scroll().contains(&NavigationAction::Zoom) {
                    return false;
                }
                let scroll_amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(_x, y) => *y / 100.0,
                };
                let model_radius = {
                    scene_scale::model_radius_from_bounds(ctx.scene.bounding().bounds.as_ref())
                };
                ctx.with_camera_mut(|cam| {
                    self.handle_wheel(scroll_amount, cam, model_radius);
                });
                true
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if self.mode != NavigationMode::Walk {
                    return false;
                }
                let mut handled = false;
                ctx.with_camera_mut(|cam| {
                    if key_event.state == crate::input::ElementState::Pressed && !key_event.repeat {
                        self.walk.init_from_camera(cam);
                    }
                    handled = self.walk.handle_key(&key_event.logical_key, key_event.state, &self.bindings);
                });
                handled
            }
            DeviceEvent::Update { delta_time } => {
                let model_radius = {
                    scene_scale::model_radius_from_bounds(ctx.scene.bounding().bounds.as_ref())
                };
                ctx.with_camera_mut(|cam| {
                    self.handle_update(*delta_time, cam, model_radius);
                });
                false
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Navigation"
    }
}
