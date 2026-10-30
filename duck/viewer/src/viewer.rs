use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use duck_engine_common::{Real, Vector3};
use duck_engine_scene::Scene;
use web_time::Instant;

use crate::{
    axis_triad::{self, AxisTriad, AxisTriadConfig},
    compositor::{Compositor, VIEW_FORMAT},
    overlay::{OverlayPainter, ScreenOverlay},
    event::{AppEvent, DeviceEvent, Event, EventContext, EventDispatcher},
    input::{ElementState, TouchPhase},
    scene::{PositionedCamera, PositionedLight, Projection, common::RgbaColor},
    selection::SelectionManager,
    renderer::{
        Gpu, HighlightQuery, PassBuilder, RenderContext, Renderer, SceneResources,
    },
    view::{
        HeadlightMode, PixelRect, View, ViewId, ViewLayout, ViewTarget, default_headlight_rig,
    },
};

/// Per-scene state shared by every view of that scene: the GPU resource caches
/// and the selection. Selecting in one view of a scene selects in all of them.
struct SceneSlot {
    scene: Scene,
    resources: SceneResources,
    selection: SelectionManager,
    /// Lights shared by every view of this scene. Per-scene rather than
    /// viewer-global so that a view over its own scene
    /// is unaffected by the lighting of any other scene.
    lights: Vec<PositionedLight>,
}

/// Pointer capture: while any button or touch that started in a view is held,
/// all pointer events route to that view regardless of where the cursor is.
struct PointerCapture {
    view: ViewId,
    /// Held mouse buttons + active touches that started this capture.
    holds: usize,
}

/// Surface-free core viewer. Hosts any number of [`View`]s, routes input to
/// them, renders each into its own texture, and composites the result into an
/// sRGB [`wgpu::TextureView`] via [`render_to_target`](Self::render_to_target).
/// 
/// The concrete output target (a window surface or an offscreen texture) is
/// owned by a wrapper such as [`SurfacedViewer`](crate::SurfacedViewer) or
/// [`OffscreenViewer`](crate::OffscreenViewer).
///
/// A new viewer has no views; add them with [`add_view`](Self::add_view). Views
/// are composited in stack order (last on top) and hit-tested topmost-first.
/// Event coordinates given to [`handle_event`](Self::handle_event) are local to
/// the viewer target; the viewer rebases them to the receiving view.
pub struct Viewer {
    /// Device, target configuration, and the pipelines shared by every view.
    ctx: RenderContext,
    /// Full target size in physical pixels.
    size: (u32, u32),
    /// One slot per distinct scene (by handle identity).
    scenes: Vec<SceneSlot>,
    /// Draw order == stack order; the last view renders on top.
    views: Vec<View>,
    /// Keyboard target: the view last clicked (or the first view added).
    focused_view: Option<ViewId>,
    capture: Option<PointerCapture>,
    /// Cursor position in viewer-target coordinates.
    cursor_position: Option<(f32, f32)>,
    compositor: Compositor,
    /// Draws operators' screen overlays over the composited views.
    overlay_painter: OverlayPainter,
    /// Color of target areas no view covers.
    clear_color: wgpu::Color,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
    next_view_id: u64,
    /// Axis-triad overlays, one record per triad view.
    axis_triads: Vec<AxisTriad>,
}

impl Viewer {
    /// Build a viewer around an existing GPU context.
    ///
    /// `composite_format` is the output format: the sRGB format of the target
    /// views passed to [`render_to_target`](Self::render_to_target), as the
    /// surface or texture behind them dictates (e.g. `Bgra8UnormSrgb`). Views
    /// themselves render at `Rgba8UnormSrgb` and `sample_count`, whatever the
    /// output; the compositor converts between the two. The viewer starts with
    /// no views.
    pub fn from_gpu(
        gpu: Gpu,
        composite_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
    ) -> Self {
        debug_assert!(
            composite_format.is_srgb(),
            "composite target {composite_format:?} is not sRGB"
        );
        let compositor = Compositor::blend(&gpu.device, composite_format);
        let overlay_painter = OverlayPainter::new(&gpu.device, composite_format);
        Self {
            ctx: RenderContext::new(gpu, VIEW_FORMAT, sample_count),
            size: (width.max(1), height.max(1)),
            scenes: Vec::new(),
            views: Vec::new(),
            focused_view: None,
            capture: None,
            cursor_position: None,
            compositor,
            overlay_painter,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            last_update_time: None,
            next_view_id: 0,
            axis_triads: Vec::new(),
        }
    }

    // ========== Views ==========

    /// Add a view of `scene` at `layout`, on top of existing views.
    ///
    /// The first view of a scene registers the scene with the viewer (GPU
    /// resources + selection, shared by later views of the same handle). The
    /// view starts with a default camera and [`HeadlightMode::Auto`]
    /// headlights; replace the camera with [`View::set_camera`].
    pub fn add_view(&mut self, name: impl Into<String>, scene: Scene, layout: ViewLayout) -> ViewId {
        self.ensure_scene(&scene);

        let rect = layout.resolve(self.size);
        let renderer = Renderer::new(&mut self.ctx, rect.width, rect.height);
        let target = create_view_target(
            self.ctx.device(),
            &self.compositor,
            self.ctx.format(),
            rect,
        );

        let mut camera = default_camera();
        camera.aspect = rect.width as Real / rect.height.max(1) as Real;

        let id = ViewId(self.next_view_id);
        self.next_view_id += 1;
        self.views.push(View {
            id,
            name: name.into(),
            scene,
            camera,
            lights: Vec::new(),
            headlight_mode: HeadlightMode::default(),
            headlight_rig: default_headlight_rig(),
            layout,
            rect,
            visible: true,
            renderer,
            dispatcher: EventDispatcher::new(),
            cursor_position: None,
            target,
            transition: None,
        });
        if self.focused_view.is_none() {
            self.focused_view = Some(id);
        }
        id
    }

    /// Remove a view. When this was the scene's last view, the scene's shared
    /// GPU resources and selection are dropped too. Axis triads mirroring the
    /// view are removed with it.
    pub fn remove_view(&mut self, id: ViewId) {
        self.axis_triads.retain(|t| t.view != id);
        let dependents: Vec<ViewId> =
            self.axis_triads.iter().filter(|t| t.target == id).map(|t| t.view).collect();
        self.axis_triads.retain(|t| t.target != id);

        let Some(index) = self.views.iter().position(|v| v.id == id) else { return };
        let view = self.views.remove(index);
        if !self.views.iter().any(|v| v.scene.ptr_eq(&view.scene)) {
            self.scenes.retain(|s| !s.scene.ptr_eq(&view.scene));
        }
        if self.capture.as_ref().is_some_and(|c| c.view == id) {
            self.capture = None;
        }
        if self.focused_view == Some(id) {
            self.focused_view = self.views.first().map(|v| v.id);
        }

        for dependent in dependents {
            self.remove_view(dependent);
        }
    }

    /// Rebind a view to a different scene, keeping everything view-local — its
    /// layout, stack position, operator stack, camera, and renderer (workflow,
    /// targets, background color). The old scene's shared state is dropped if
    /// this was its last view.
    pub fn set_view_scene(&mut self, id: ViewId, scene: Scene) {
        let Some(index) = self.views.iter().position(|v| v.id == id) else { return };
        if self.views[index].scene.ptr_eq(&scene) {
            return;
        }

        // Detach from the old scene.
        let old_scene = self.views[index].scene.clone();
        if !self
            .views
            .iter()
            .enumerate()
            .any(|(i, v)| i != index && v.scene.ptr_eq(&old_scene))
        {
            self.scenes.retain(|s| !s.scene.ptr_eq(&old_scene));
        }

        // Attach to the new one. The renderer holds no scene state — targets,
        // workflow, and uniforms are all view-local — so it carries over.
        self.ensure_scene(&scene);
        self.views[index].scene = scene;
    }

    /// Get a view by ID.
    pub fn view(&self, id: ViewId) -> Option<&View> {
        self.views.iter().find(|v| v.id == id)
    }

    /// Get a view by ID, for mutation.
    pub fn view_mut(&mut self, id: ViewId) -> Option<&mut View> {
        self.views.iter_mut().find(|v| v.id == id)
    }

    /// Views in stack order (last is topmost).
    pub fn views(&self) -> impl Iterator<Item = &View> {
        self.views.iter()
    }

    /// Views in stack order (last is topmost), for mutation.
    pub fn views_mut(&mut self) -> impl Iterator<Item = &mut View> {
        self.views.iter_mut()
    }

    /// The selection shared by every view of view `id`'s scene.
    pub fn selection(&self, id: ViewId) -> Option<&SelectionManager> {
        self.slot(id).map(|s| &s.selection)
    }

    /// The selection shared by every view of view `id`'s scene, for mutation.
    pub fn selection_mut(&mut self, id: ViewId) -> Option<&mut SelectionManager> {
        self.slot_mut(id).map(|s| &mut s.selection)
    }

    /// Lights shared by every view of view `id`'s scene.
    pub fn scene_lights(&self, id: ViewId) -> Option<&[PositionedLight]> {
        self.slot(id).map(|s| s.lights.as_slice())
    }

    /// Lights shared by every view of view `id`'s scene, for mutation.
    pub fn scene_lights_mut(&mut self, id: ViewId) -> Option<&mut Vec<PositionedLight>> {
        self.slot_mut(id).map(|s| &mut s.lights)
    }

    /// Re-place a view; its targets resize if the resolved pixel size changed.
    pub fn set_view_layout(&mut self, id: ViewId, layout: ViewLayout) {
        let Some(view) = self.views.iter_mut().find(|v| v.id == id) else { return };
        view.layout = layout;
        let rect = layout.resolve(self.size);
        if rect == view.rect {
            return;
        }
        let size_changed = (rect.width, rect.height) != (view.rect.width, view.rect.height);
        view.rect = rect;
        if size_changed {
            view.camera.aspect = rect.width as Real / rect.height.max(1) as Real;
            view.renderer.resize((rect.width, rect.height));
            view.target = create_view_target(
                self.ctx.device(),
                &self.compositor,
                self.ctx.format(),
                rect,
            );
            self.dispatch_to(id, &Event::Device(DeviceEvent::Resized((rect.width, rect.height))));
        }
    }

    /// Hidden views are neither rendered nor hit-tested.
    pub fn set_view_visible(&mut self, id: ViewId, visible: bool) {
        if let Some(view) = self.views.iter_mut().find(|v| v.id == id) {
            view.visible = visible;
        }
    }

    /// Move a view to the top of the stack (rendered last, hit-tested first).
    pub fn move_view_to_top(&mut self, id: ViewId) {
        if let Some(index) = self.views.iter().position(|v| v.id == id) {
            let view = self.views.remove(index);
            self.views.push(view);
        }
    }

    /// Add an axis-triad overlay mirroring `target`'s camera orientation.
    /// 
    /// Clicking one of its six axis handles animates `target`'s camera to look
    /// down that axis; clicking the axis already faced flips to the opposite
    /// side. Each snap is reported to `target`'s operators as
    /// [`AppEvent::ViewSnapped`]. The triad renders its own small scene in an
    /// anchored corner view kept on top of the stack.
    ///
    /// The returned id is a normal view id: layout, visibility, and removal go
    /// through the usual view APIs. Removing `target` removes the triad too.
    ///
    /// Pointer events over the triad's corner stay with it: clicks beside the
    /// handles and scroll wheel input are not forwarded to views beneath.
    pub fn add_axis_triad(&mut self, target: ViewId, config: AxisTriadConfig) -> ViewId {
        let scene = Scene::default();
        let pending = Arc::new(Mutex::new(None));
        let operator = axis_triad::build_triad(&scene, Arc::clone(&pending));

        let id = self.add_view(
            "Axis Triad",
            scene,
            ViewLayout::Anchored {
                corner: config.corner,
                size: (config.size, config.size),
                margin: config.margin,
            },
        );
        {
            let view = self.view_mut(id).expect("view was just added");
            view.set_background_color(RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 0.0 });
            view.dispatcher_mut().push_back(operator);
            let mut camera = PositionedCamera {
                eye: (0.0, 0.0, config.camera_distance).into(),
                target: (0.0, 0.0, 0.0).into(),
                up: Vector3::unit_y(),
                aspect: 1.0,
                projection: Projection::Perspective {
                    fovy: 45.0,
                    znear: 0.1,
                    zfar: config.camera_distance * 4.0,
                },
            };
            if config.ortho {
                camera.make_orthographic();
            }
            view.set_camera(camera);
        }

        self.axis_triads.push(AxisTriad { view: id, target, config, pending_snap: pending });
        self.sync_triad_cameras();
        id
    }

    /// Remove an axis triad added with [`add_axis_triad`](Self::add_axis_triad).
    pub fn remove_axis_triad(&mut self, id: ViewId) {
        self.remove_view(id);
    }

    /// The keyboard-focused view (the view last clicked).
    pub fn focused_view(&self) -> Option<ViewId> {
        self.focused_view
    }

    pub fn set_focused_view(&mut self, id: ViewId) {
        if self.views.iter().any(|v| v.id == id) {
            self.focused_view = Some(id);
        }
    }

    // ========== Events ==========

    /// Handle a single event, routing it to the appropriate view(s).
    ///
    /// Coordinates are viewer-target-local: the host rebases window/panel
    /// coordinates to the viewer target, the viewer rebases to the view.
    /// Pointer events go to the view under the cursor (topmost first); a press
    /// captures its view until all buttons/touches release; keys go to the
    /// last-clicked view; `Update` goes to every view.
    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::Device(DeviceEvent::Resized(physical_size)) => {
                let (w, h) = *physical_size;
                self.resize(w, h);
            }
            Event::Device(DeviceEvent::Update { .. }) => {
                for i in 0..self.views.len() {
                    let id = self.views[i].id;
                    self.dispatch_to(id, event);
                }
            }
            Event::Device(DeviceEvent::CursorMoved { position }) => {
                let target_pos = (position.0 as f32, position.1 as f32);
                self.cursor_position = Some(target_pos);
                if let Some(id) = self.pointer_target() {
                    let rect = self.view_rect(id);
                    let local = (
                        position.0 - rect.x as f64,
                        position.1 - rect.y as f64,
                    );
                    self.dispatch_to(id, &Event::Device(DeviceEvent::CursorMoved { position: local }));
                }
            }
            Event::Device(DeviceEvent::MouseInput { state, .. }) => {
                let target = self.pointer_target();
                match state {
                    ElementState::Pressed => {
                        if let Some(id) = target {
                            let capture =
                                self.capture.get_or_insert(PointerCapture { view: id, holds: 0 });
                            capture.holds += 1;
                            self.focused_view = Some(id);
                            self.dispatch_to(id, event);
                        }
                    }
                    ElementState::Released => {
                        self.release_hold();
                        if let Some(id) = target {
                            self.dispatch_to(id, event);
                        }
                    }
                }
            }
            Event::Device(DeviceEvent::Touch { id: touch_id, phase, position }) => {
                let target = match phase {
                    TouchPhase::Started => {
                        let target = self
                            .capture
                            .as_ref()
                            .map(|c| c.view)
                            .or_else(|| self.hit_test(position.0 as f32, position.1 as f32));
                        if let Some(id) = target {
                            let capture =
                                self.capture.get_or_insert(PointerCapture { view: id, holds: 0 });
                            capture.holds += 1;
                            self.focused_view = Some(id);
                        }
                        target
                    }
                    TouchPhase::Moved => self.pointer_target_at(*position),
                    TouchPhase::Ended | TouchPhase::Cancelled => {
                        let target = self.pointer_target_at(*position);
                        self.release_hold();
                        target
                    }
                };
                if let Some(id) = target {
                    let rect = self.view_rect(id);
                    let local = (
                        position.0 - rect.x as f64,
                        position.1 - rect.y as f64,
                    );
                    self.dispatch_to(
                        id,
                        &Event::Device(DeviceEvent::Touch {
                            id: *touch_id,
                            phase: *phase,
                            position: local,
                        }),
                    );
                }
            }
            Event::Device(DeviceEvent::KeyboardInput { .. }) => {
                if let Some(id) = self.focused_view {
                    self.dispatch_to(id, event);
                }
            }
            // Remaining device events are pointer-shaped (wheel, raw motion,
            // and any pre-synthesized drag/click a host injects).
            Event::Device(_) => {
                if let Some(id) = self.pointer_target().or(self.focused_view) {
                    self.dispatch_to(id, event);
                }
            }
            // Host-injected app events go to the focused view; use
            // `handle_view_event` to address a specific one.
            Event::App(_) => {
                if let Some(id) = self.focused_view {
                    self.dispatch_to(id, event);
                }
            }
        }
    }

    /// Dispatch an event directly to one view, bypassing routing. Coordinates
    /// (if any) are taken as already view-local.
    pub fn handle_view_event(&mut self, id: ViewId, event: &Event) {
        self.dispatch_to(id, event);
    }

    /// Dispatch an Update event with delta_time since last update to every view.
    ///
    /// Call this once per frame before rendering to enable smooth continuous
    /// operations like WASD movement in walk mode, or camera transitions from axis triads.
    pub fn update(&mut self) {
        let now = Instant::now();
        let delta_time = match self.last_update_time {
            Some(last) => now.duration_since(last).as_secs_f32(),
            None => 1.0 / 60.0, // Assume 60 FPS on first frame
        };
        self.last_update_time = Some(now);

        let event = Event::Device(DeviceEvent::Update { delta_time });
        self.handle_event(&event);

        // Triad clicks start transitions before this frame's advance; triad
        // cameras mirror the frame's final poses afterwards.
        self.apply_triad_snaps();
        self.advance_camera_transitions(delta_time);
        self.sync_triad_cameras();
        self.float_triads_to_top();
    }

    /// Start a camera transition on each triad's target view with a pending
    /// handle click, and tell the view's operators with [`AppEvent::ViewSnapped`].
    fn apply_triad_snaps(&mut self) {
        let snaps: Vec<_> = self
            .axis_triads
            .iter()
            .filter_map(|triad| {
                let request = triad.pending_snap.lock().unwrap().take()?;
                Some((triad.target, triad.config.snap_duration, triad.config.fit_on_snap, request))
            })
            .collect();

        for (target, duration, fit, request) in snaps {
            let Some(view) = self.view_mut(target) else { continue };
            let bounds = if fit { view.scene.lock().bounding().bounds } else { None };
            let direction = axis_triad::snap_direction(&view.camera, &request);
            let camera = axis_triad::snap_camera(&view.camera, &request, bounds.as_ref());
            view.transition_camera_to(camera, duration);
            self.dispatch_to(target, &Event::App(AppEvent::ViewSnapped { direction }));
        }
    }

    /// Orient each triad's camera to match its target view's.
    fn sync_triad_cameras(&mut self) {
        for i in 0..self.axis_triads.len() {
            let (triad_view, target_view, distance) = {
                let triad = &self.axis_triads[i];
                (triad.view, triad.target, triad.config.camera_distance)
            };
            let Some(reference) = self.view(target_view).map(|v| v.camera.clone()) else {
                continue;
            };
            if let Some(view) = self.views.iter_mut().find(|v| v.id == triad_view) {
                axis_triad::aim_triad_camera(&mut view.camera, &reference, distance);
            }
        }
    }

    /// Keep triad views above later-added views (stable partition; no-op scan
    /// in the common case).
    fn float_triads_to_top(&mut self) {
        let is_triad =
            |id: ViewId, triads: &[AxisTriad]| triads.iter().any(|t| t.view == id);

        let mut seen_triad = false;
        let buried = self.views.iter().any(|v| {
            if is_triad(v.id, &self.axis_triads) {
                seen_triad = true;
                false
            } else {
                seen_triad
            }
        });
        if !buried {
            return;
        }

        let triad_ids: Vec<ViewId> = self.axis_triads.iter().map(|t| t.view).collect();
        let (rest, triads): (Vec<View>, Vec<View>) =
            self.views.drain(..).partition(|v| !triad_ids.contains(&v.id));
        self.views = rest;
        self.views.extend(triads);
    }

    /// Advance in-flight camera transitions, dropping any whose camera was
    /// written by someone else since the last update.
    fn advance_camera_transitions(&mut self, delta_time: f32) {
        for view in &mut self.views {
            let Some(transition) = view.transition.as_mut() else { continue };
            if transition.externally_modified(&view.camera) {
                view.transition = None;
                continue;
            }
            let aspect = view.camera.aspect;
            view.camera = transition.advance(delta_time);
            view.camera.aspect = aspect;
            if transition.finished() {
                view.transition = None;
            }
        }
    }

    /// The view pointer events currently route to: the capturing view, else
    /// the topmost visible view under the cursor.
    fn pointer_target(&self) -> Option<ViewId> {
        self.capture
            .as_ref()
            .map(|c| c.view)
            .or_else(|| self.cursor_position.and_then(|(x, y)| self.hit_test(x, y)))
    }

    /// Like [`Self::pointer_target`], but hit-tests `position` instead of the
    /// last cursor position, for touches that carry their own.
    fn pointer_target_at(&self, position: (f64, f64)) -> Option<ViewId> {
        self.capture
            .as_ref()
            .map(|c| c.view)
            .or_else(|| self.hit_test(position.0 as f32, position.1 as f32))
    }

    /// Drop one held button or touch from the capture, ending it when none remain.
    fn release_hold(&mut self) {
        if let Some(capture) = self.capture.as_mut() {
            capture.holds = capture.holds.saturating_sub(1);
            if capture.holds == 0 {
                self.capture = None;
            }
        }
    }

    /// Topmost visible view containing the point, in target coordinates.
    fn hit_test(&self, x: f32, y: f32) -> Option<ViewId> {
        self.views
            .iter()
            .rev()
            .find(|v| v.visible && v.rect.contains(x, y))
            .map(|v| v.id)
    }

    /// The view's rect in target coordinates, or a 1×1 rect at the origin if
    /// `id` is unknown.
    fn view_rect(&self, id: ViewId) -> PixelRect {
        self.views
            .iter()
            .find(|v| v.id == id)
            .map(|v| v.rect)
            .unwrap_or(PixelRect { x: 0, y: 0, width: 1, height: 1 })
    }

    /// Dispatch `event` through one view's operator stack. Cursor state in the
    /// event is expected to be view-local.
    fn dispatch_to(&mut self, id: ViewId, event: &Event) {
        let Some((view, slot)) = self.view_and_slot(id) else { return };
        let view = &mut self.views[view];
        let slot = &mut self.scenes[slot];

        if let Event::Device(DeviceEvent::CursorMoved { position }) = event {
            view.cursor_position = Some((position.0 as f32, position.1 as f32));
        }

        let mut ctx = EventContext {
            size: (view.rect.width, view.rect.height),
            cursor_position: &mut view.cursor_position,
            scene: view.scene.clone(), // pointer clone
            camera: &mut view.camera,
            selection: &mut slot.selection,
            modifiers: Default::default(), // dispatcher overwrites this in dispatch()
            emit_queue: Vec::new(),
        };
        view.dispatcher.dispatch(event, &mut ctx);
    }

    // ========== Rendering ==========

    /// Resize the viewer target. Re-places every view; views whose pixel size
    /// changed get resized render targets and a view-local `Resized` event.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = (width, height);

        let mut resized = Vec::new();
        for view in &mut self.views {
            let rect = view.layout.resolve(self.size);
            if rect == view.rect {
                continue;
            }
            let size_changed = (rect.width, rect.height) != (view.rect.width, view.rect.height);
            view.rect = rect;
            if size_changed {
                view.camera.aspect = rect.width as Real / rect.height.max(1) as Real;
                view.renderer.resize((rect.width, rect.height));
                view.target = create_view_target(
                    self.ctx.device(),
                    &self.compositor,
                    self.ctx.format(),
                    rect,
                );
                resized.push((view.id, rect.width, rect.height));
            }
        }
        for (id, w, h) in resized {
            self.dispatch_to(id, &Event::Device(DeviceEvent::Resized((w, h))));
        }
    }

    /// Render every visible view into its texture and composite the stack onto
    /// `target`, a view at the format given to [`from_gpu`](Self::from_gpu),
    /// with each view's operator [overlay](crate::operator::Operator::overlay)
    /// on top.
    /// The encoder is not submitted — the caller is responsible for that.
    ///
    /// Each scene's GPU resources are prepared exactly once per call, however
    /// many views share the scene.
    pub fn render_to_target(
        &mut self,
        target: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), anyhow::Error> {
        for slot in &mut self.scenes {
            let mut guard = slot.scene.lock();
            slot.resources.prepare(&mut self.ctx, &mut guard)?;
        }

        for view in self.views.iter_mut().filter(|v| v.visible) {
            let Some(slot) = self.scenes.iter_mut().find(|s| s.scene.ptr_eq(&view.scene)) else {
                continue;
            };
            let guard = view.scene.lock();
            // Defensive: the aspect is stamped on every rect change, but raw
            // camera mutation may have overwritten it.
            view.camera.aspect = view.rect.width as Real / view.rect.height.max(1) as Real;
            let lights = view.effective_lights(&slot.lights, &guard);
            let highlight: Option<&dyn HighlightQuery> = if slot.selection.config().outline_enabled
            {
                Some(&slot.selection)
            } else {
                None
            };
            view.renderer.render_scene_to_view(
                &mut self.ctx,
                &mut slot.resources,
                &guard,
                &view.camera,
                &lights,
                &view.target.render_view,
                encoder,
                highlight,
            )?;
        }

        self.compositor.composite(
            encoder,
            target,
            self.clear_color,
            self.views
                .iter()
                .filter(|v| v.visible)
                .map(|v| (&v.target.bind_group, v.rect)),
        );

        // Above every view, triads included: the overlays are the topmost UI.
        let overlays: Vec<(PixelRect, ScreenOverlay)> = self
            .views
            .iter()
            .filter(|v| v.visible)
            .map(|v| (v.rect, v.dispatcher.overlay()))
            .filter(|(_, overlay)| !overlay.is_empty())
            .collect();
        self.overlay_painter.draw(
            self.ctx.device(),
            self.ctx.queue(),
            encoder,
            target,
            self.size,
            &overlays,
        );
        Ok(())
    }

    /// Render view `id`'s scene from `camera` and read the result back into an
    /// RGBA image (blocking). For headless still-image / thumbnail rendering.
    /// The output size is the view's current pixel size. The view's lights
    /// apply, resolved against `camera`.
    pub fn render_view_to_image(
        &mut self,
        id: ViewId,
        camera: &PositionedCamera,
    ) -> Result<image::RgbaImage, anyhow::Error> {
        let (view, slot) = self.view_and_slot(id).with_context(|| format!("no view {id:?}"))?;
        let view = &mut self.views[view];
        let slot = &mut self.scenes[slot];
        let lights = {
            let guard = view.scene.lock();
            view.effective_lights(&slot.lights, &guard)
        };
        let highlight: Option<&dyn HighlightQuery> = if slot.selection.config().outline_enabled {
            Some(&slot.selection)
        } else {
            None
        };
        view.renderer.render_scene_to_image(
            &mut self.ctx,
            &mut slot.resources,
            &mut view.scene,
            camera,
            &lights,
            highlight,
        )
    }

    /// A [`PassBuilder`] configured for view `id`'s target size, format and
    /// MSAA settings, for constructing a pass to put in its workflow.
    pub fn pass_builder(&mut self, id: ViewId) -> Option<PassBuilder<'_>> {
        let view = self.views.iter().find(|v| v.id == id)?;
        Some(view.renderer.pass_builder(&mut self.ctx))
    }

    /// Color of target areas no view covers.
    pub fn set_clear_color(&mut self, color: RgbaColor) {
        self.clear_color = wgpu::Color {
            r: color.r as f64,
            g: color.g as f64,
            b: color.b as f64,
            a: color.a as f64,
        };
    }

    // ========== GPU context ==========

    /// Get the current target size as (width, height)
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Target aspect ratio.
    pub fn aspect(&self) -> Real {
        let (w, h) = self.size;
        if h > 0 { w as Real / h as Real } else { 16.0 / 9.0 }
    }

    /// The render context every view draws through: the device and queue, the
    /// target format and sample count, and the builders for custom pipelines
    /// and user shaders.
    pub fn context(&self) -> &RenderContext {
        &self.ctx
    }

    /// Register `scene` if no view of it exists yet, creating its shared GPU
    /// resources and selection.
    fn ensure_scene(&mut self, scene: &Scene) {
        if self.scenes.iter().any(|s| s.scene.ptr_eq(scene)) {
            return;
        }
        self.scenes.push(SceneSlot {
            scene: scene.clone(),
            resources: SceneResources::new(&self.ctx),
            selection: SelectionManager::new(),
            lights: Vec::new(),
        });
    }

    /// Indices of view `id` and of its scene's slot.
    fn view_and_slot(&self, id: ViewId) -> Option<(usize, usize)> {
        let view = self.views.iter().position(|v| v.id == id)?;
        let slot = self.scenes.iter().position(|s| s.scene.ptr_eq(&self.views[view].scene))?;
        Some((view, slot))
    }

    fn slot(&self, id: ViewId) -> Option<&SceneSlot> {
        self.view_and_slot(id).map(|(_, slot)| &self.scenes[slot])
    }

    fn slot_mut(&mut self, id: ViewId) -> Option<&mut SceneSlot> {
        self.view_and_slot(id).map(|(_, slot)| &mut self.scenes[slot])
    }
}

fn default_camera() -> PositionedCamera {
    PositionedCamera {
        eye: (0.0, 0.1, 0.2).into(),
        target: (0.0, 0.0, 0.0).into(),
        up: Vector3::unit_y(),
        aspect: 16.0 / 9.0,
        projection: Projection::Perspective { fovy: 45.0, znear: 0.001, zfar: 100.0 },
    }
}

/// Allocate a view's color texture at `rect`'s size (at least 1×1), bound for
/// sampling by the compositor.
fn create_view_target(
    device: &wgpu::Device,
    compositor: &Compositor,
    format: wgpu::TextureFormat,
    rect: PixelRect,
) -> ViewTarget {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("View Color Target"),
        size: wgpu::Extent3d {
            width: rect.width.max(1),
            height: rect.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = compositor.bind(device, &render_view);
    ViewTarget { texture, render_view, bind_group }
}
