use std::ops::{Deref, DerefMut};

use duck_engine_common::Vector3;
use duck_engine_scene::Scene;
use web_time::Instant;

use crate::{
    camera_transition::CameraTransition,
    compositor::Compositor,
    event::{DeviceEvent, Event, EventContext, EventDispatcher},
    input::{ElementState, TouchPhase},
    scene::{PositionedCamera, common::RgbaColor},
    selection::SelectionManager,
    renderer::{
        Gpu, HiddenLineConfig, HiddenLineWorkflow, HighlightQuery, RenderContext, Renderer,
        SceneResources, SceneWorkflow, ShadedWorkflow,
    },
    view::{
        CameraLight, HeadlightMode, PixelRect, View, ViewId, ViewLayout, ViewTarget,
        default_camera_lights,
    },
};

/// Per-scene state shared by every view of that scene: the GPU resource caches
/// and the selection. Selecting in one view of a scene selects in all of them.
struct SceneSlot {
    scene: Scene,
    resources: SceneResources,
    selection: SelectionManager,
}

/// Pointer capture: while any button or touch that started in a view is held,
/// all pointer events route to that view regardless of where the cursor is.
struct Capture {
    view: ViewId,
    /// Held mouse buttons + active touches that started this capture.
    holds: usize,
}

/// Surface-free core viewer: hosts any number of [`View`]s, routes input to
/// them, renders each into its own texture, and composites the result into an
/// arbitrary [`wgpu::TextureView`] via [`render_to_target`](Self::render_to_target).
/// The concrete output target (a window surface or an offscreen texture) is
/// owned by a wrapper such as [`SurfacedViewer`] or [`OffscreenViewer`].
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
    capture: Option<Capture>,
    /// Cursor position in viewer-target coordinates.
    cursor_position: Option<(f32, f32)>,
    compositor: Compositor,
    /// Color of target areas no view covers.
    clear_color: wgpu::Color,
    /// Last time update() was called, for delta_time calculation
    last_update_time: Option<Instant>,
    next_view_id: u64,
}

impl Viewer {
    /// Build a viewer around an existing GPU context.
    ///
    /// `color_format` is the format of the target the viewer will render into
    /// (a surface format or an offscreen texture format); all views render at
    /// this format and `sample_count`. The viewer starts with no views.
    pub fn from_gpu(
        gpu: Gpu,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let compositor = Compositor::new(&gpu.device, color_format);
        Self {
            ctx: RenderContext::new(gpu, color_format, sample_count, has_compute),
            size: (width.max(1), height.max(1)),
            scenes: Vec::new(),
            views: Vec::new(),
            focused_view: None,
            capture: None,
            cursor_position: None,
            compositor,
            clear_color: wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 },
            last_update_time: None,
            next_view_id: 0,
        }
    }

    // ========== Views ==========

    /// Add a view of `scene` at `layout`, on top of existing views.
    ///
    /// The first view of a scene registers the scene with the viewer (GPU
    /// resources + selection, shared by later views of the same handle). The
    /// view starts with a default camera and [`HeadlightMode::Auto`]
    /// headlights; replace the camera with
    /// [`ViewMut::set_camera`](ViewMut::set_camera).
    pub fn add_view(&mut self, name: impl Into<String>, scene: Scene, layout: ViewLayout) -> ViewId {
        self.attach_scene(&scene);

        let rect = layout.resolve(self.size);
        let renderer = Renderer::new(&mut self.ctx, rect.width, rect.height);
        let target = create_view_target(
            self.ctx.device(),
            &self.compositor,
            self.ctx.format(),
            rect,
        );

        let mut camera = default_camera();
        camera.aspect = rect.width as f32 / rect.height.max(1) as f32;

        let id = ViewId(self.next_view_id);
        self.next_view_id += 1;
        self.views.push(View {
            id,
            name: name.into(),
            scene,
            camera,
            headlight: HeadlightMode::default(),
            camera_lights: default_camera_lights(),
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
    /// GPU resources and selection are dropped too.
    pub fn remove_view(&mut self, id: ViewId) {
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
        self.attach_scene(&scene);
        self.views[index].scene = scene;
    }

    pub fn view(&self, id: ViewId) -> Option<&View> {
        self.views.iter().find(|v| v.id == id)
    }

    /// Mutable access to a view together with its scene's shared state.
    pub fn view_mut(&mut self, id: ViewId) -> Option<ViewMut<'_>> {
        let index = self.views.iter().position(|v| v.id == id)?;
        let slot = self
            .scenes
            .iter()
            .position(|s| s.scene.ptr_eq(&self.views[index].scene))?;
        Some(ViewMut {
            view: &mut self.views[index],
            slot: &mut self.scenes[slot],
            ctx: &mut self.ctx,
        })
    }

    /// Views in stack order (last is topmost).
    pub fn views(&self) -> impl Iterator<Item = &View> {
        self.views.iter()
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
            view.camera.aspect = rect.width as f32 / rect.height.max(1) as f32;
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
                                self.capture.get_or_insert(Capture { view: id, holds: 0 });
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
                                self.capture.get_or_insert(Capture { view: id, holds: 0 });
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
    /// operations like WASD movement in walk mode.
    pub fn update(&mut self) {
        let now = Instant::now();
        let delta_time = match self.last_update_time {
            Some(last) => now.duration_since(last).as_secs_f32(),
            None => 1.0 / 60.0, // Assume 60 FPS on first frame
        };
        self.last_update_time = Some(now);

        let event = Event::Device(DeviceEvent::Update { delta_time });
        self.handle_event(&event);

        self.advance_camera_transitions(delta_time);
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

    fn pointer_target_at(&self, position: (f64, f64)) -> Option<ViewId> {
        self.capture
            .as_ref()
            .map(|c| c.view)
            .or_else(|| self.hit_test(position.0 as f32, position.1 as f32))
    }

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
        let Some(index) = self.views.iter().position(|v| v.id == id) else { return };
        let Some(slot) = self
            .scenes
            .iter()
            .position(|s| s.scene.ptr_eq(&self.views[index].scene))
        else {
            return;
        };
        let view = &mut self.views[index];
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
                view.camera.aspect = rect.width as f32 / rect.height.max(1) as f32;
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
    /// `target`. The encoder is not submitted — the caller is responsible for
    /// that.
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
            view.camera.aspect = view.rect.width as f32 / view.rect.height.max(1) as f32;
            let headlights = view.active_camera_lights(&view.camera, &guard);
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
                &headlights,
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
        Ok(())
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
    pub fn aspect(&self) -> f32 {
        let (w, h) = self.size;
        if h > 0 { w as f32 / h as f32 } else { 16.0 / 9.0 }
    }

    /// The render context every view draws through. Custom pipelines and user
    /// shaders are built from here.
    pub fn context(&self) -> &RenderContext {
        &self.ctx
    }

    /// Get the render target texture format.
    /// Useful for creating render pipelines that need to match the target format.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.ctx.format()
    }

    /// The MSAA sample count every view renders with.
    pub fn sample_count(&self) -> u32 {
        self.ctx.sample_count()
    }

    /// Get a reference to the wgpu device
    pub fn device(&self) -> &wgpu::Device {
        self.ctx.device()
    }

    /// Get a reference to the wgpu queue
    pub fn queue(&self) -> &wgpu::Queue {
        self.ctx.queue()
    }

    /// Returns a clone of the GPU handle pair, for sharing the device/queue with
    /// other owners (e.g. an offscreen viewer or an egui renderer).
    pub fn gpu(&self) -> Gpu {
        self.ctx.gpu().clone()
    }

    /// Register `scene` if no view of it exists yet, creating its shared GPU
    /// resources and selection.
    fn attach_scene(&mut self, scene: &Scene) {
        if self.scenes.iter().any(|s| s.scene.ptr_eq(scene)) {
            return;
        }
        self.scenes.push(SceneSlot {
            scene: scene.clone(),
            resources: SceneResources::new(&self.ctx),
            selection: SelectionManager::new(),
        });
    }
}

/// Mutable access to a [`View`] bundled with its scene's shared state
/// (selection, GPU resources) and the viewer's render context, obtained from
/// [`Viewer::view_mut`]. Derefs to [`View`] for reads.
pub struct ViewMut<'a> {
    view: &'a mut View,
    slot: &'a mut SceneSlot,
    ctx: &'a mut RenderContext,
}

impl Deref for ViewMut<'_> {
    type Target = View;
    fn deref(&self) -> &View {
        self.view
    }
}

impl ViewMut<'_> {
    /// The view's operator stack.
    pub fn dispatcher_mut(&mut self) -> &mut EventDispatcher {
        &mut self.view.dispatcher
    }

    /// The selection shared by all views of this view's scene.
    pub fn selection(&self) -> &SelectionManager {
        &self.slot.selection
    }

    /// Mutable access to the shared selection.
    pub fn selection_mut(&mut self) -> &mut SelectionManager {
        &mut self.slot.selection
    }

    /// Mutable access to the view's camera. The aspect is re-stamped from the
    /// view's pixel size before rendering, so callers need not maintain it.
    pub fn camera_mut(&mut self) -> &mut PositionedCamera {
        &mut self.view.camera
    }

    /// Replace the view's camera, stamping its aspect from the view's pixel size.
    pub fn set_camera(&mut self, camera: PositionedCamera) {
        let aspect = self.view.aspect();
        self.view.camera = camera;
        self.view.camera.aspect = aspect;
    }

    /// Animate the view's camera to `camera` over `duration` seconds
    /// (smoothstep-eased orbit arc). Replaces any in-flight transition,
    /// continuing from the current pose; a non-positive duration applies the
    /// camera immediately. Any other write to the camera — by an operator,
    /// [`camera_mut`](Self::camera_mut), or [`set_camera`](Self::set_camera) —
    /// cancels the transition at the viewer's next update. The aspect is
    /// stamped from the view's pixel size; resizing does not cancel.
    pub fn transition_camera_to(&mut self, camera: PositionedCamera, duration: f32) {
        if duration <= 0.0 {
            self.view.transition = None;
            self.set_camera(camera);
            return;
        }
        let mut camera = camera;
        camera.aspect = self.view.aspect();
        self.view.transition =
            Some(CameraTransition::new(self.view.camera.clone(), camera, duration));
    }

    /// Cancel any in-flight camera transition, leaving the camera where it is.
    pub fn cancel_camera_transition(&mut self) {
        self.view.transition = None;
    }

    /// Set when the view's camera-space lights contribute to rendering.
    pub fn set_headlight(&mut self, mode: HeadlightMode) {
        self.view.headlight = mode;
    }

    /// Replace the view's camera-space lights (the default is a key + fill
    /// directional pair).
    pub fn set_camera_lights(&mut self, lights: Vec<CameraLight>) {
        self.view.camera_lights = lights;
    }

    /// This view's background color. Alpha below 1 shows through to views
    /// beneath (zero alpha makes the view a pure overlay).
    pub fn set_background_color(&mut self, color: RgbaColor) {
        self.view.renderer.set_background_color(color);
    }

    /// Replace the view's rendering workflow.
    pub fn set_workflow(&mut self, workflow: Box<SceneWorkflow>) {
        self.view.renderer.set_workflow(workflow);
    }

    /// Create a [`ShadedWorkflow`] configured for this view.
    pub fn shaded_workflow(&mut self) -> ShadedWorkflow {
        self.view.renderer.shaded_workflow(self.ctx)
    }

    /// Create a [`HiddenLineWorkflow`] configured for this view.
    pub fn hidden_line_workflow(&mut self, config: HiddenLineConfig) -> HiddenLineWorkflow {
        self.view.renderer.hidden_line_workflow(self.ctx, config)
    }

    /// Clear the view's scene, removing all geometry, materials, textures, and
    /// associated GPU resources, then restore a default camera and selection.
    ///
    /// The scene clear affects every view of the scene; only this view's
    /// camera is reset.
    pub fn clear_scene(&mut self) {
        self.view.scene.clear();
        self.slot.selection.clear();
        self.slot.resources.clear();
        self.set_camera(default_camera());
    }

    /// Render this view's scene from the given camera and read the result back
    /// into an RGBA image (blocking). For headless still-image / thumbnail
    /// rendering. The output size is the view's current pixel size. The view's
    /// headlights apply, resolved against the given camera.
    pub fn render_to_image(
        &mut self,
        camera: &PositionedCamera,
    ) -> Result<image::RgbaImage, anyhow::Error> {
        let headlights = {
            let guard = self.view.scene.lock();
            self.view.active_camera_lights(camera, &guard)
        };
        let highlight: Option<&dyn HighlightQuery> = if self.slot.selection.config().outline_enabled
        {
            Some(&self.slot.selection)
        } else {
            None
        };
        self.view.renderer.render_scene_to_image(
            self.ctx,
            &mut self.slot.resources,
            &mut self.view.scene,
            camera,
            &headlights,
            highlight,
        )
    }
}

fn default_camera() -> PositionedCamera {
    PositionedCamera {
        eye: (0.0, 0.1, 0.2).into(),
        target: (0.0, 0.0, 0.0).into(),
        up: Vector3::unit_y(),
        aspect: 16.0 / 9.0,
        fovy: 45.0,
        znear: 0.001,
        zfar: 100.0,
        ortho: false,
    }
}

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

/// A configured window/canvas surface paired with the GPU context created for
/// it. [`SurfacedViewer`] composes this with a viewer
/// core on top of it; applications that present through their own compositor
/// (e.g. egui) can hold a `WindowSurface` directly and drive an offscreen
/// viewer with `gpu()`.
pub struct WindowSurface<'a> {
    surface: wgpu::Surface<'a>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    sample_count: u32,
    has_compute: bool,
}

impl<'a> WindowSurface<'a> {
    /// Create and configure a surface for the given target, bootstrapping the
    /// wgpu instance, adapter, and device/queue.
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        // Create wgpu instance
        #[cfg(not(target_arch = "wasm32"))]
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        #[cfg(target_arch = "wasm32")]
        let instance = wgpu::util::new_instance_with_webgpu_detection(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..Default::default()
        }).await;

        let surface = instance.create_surface(surface_target).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let is_gl_backend = adapter.get_info().backend == wgpu::Backend::Gl;
        let downlevel_flags = adapter.get_downlevel_capabilities().flags;
        let has_compute = downlevel_flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);

        if cfg!(target_arch = "wasm32") {
            if is_gl_backend {
                log::info!("WebGPU not available, falling back to WebGL.");
            } else {
                log::info!("Using WebGPU backend.");
            }
        }

        // Set DUCK_FORCE_WEBGL2_LIMITS to cap any backend to the WebGL2
        // floor, for testing behavior on minimum supported hardware.
        let required_limits = if std::env::var("DUCK_FORCE_WEBGL2_LIMITS").is_ok() {
            log::info!("DUCK_FORCE_WEBGL2_LIMITS set: capping to downlevel WebGL2 limits");
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            adapter.limits()
        };
        log::info!(
            "Requested max_texture_dimension_2d: {}",
            required_limits.max_texture_dimension_2d,
        );

        // Depth formats only guarantee up to 4x MSAA in the WebGPU baseline;
        // higher sample counts (e.g. 8x) require this adapter-specific feature
        // to be enabled at device creation. Request it when available so the
        // MSAA probe below can pick counts above 4.
        let required_features = if adapter
            .features()
            .contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        {
            wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features,
                required_limits,
                label: None,
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
                experimental_features: Default::default(),
            })
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let present_mode = surface_caps
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::Fifo)
            .unwrap_or(surface_caps.present_modes[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        // The renderer picks the count: it owns the attachments that have to
        // support it.
        let sample_count = Renderer::preferred_sample_count(&adapter, surface_format);

        if sample_count > 1 {
            log::info!("Using {sample_count}x MSAA");
        }
        else {
            log::warn!("MSAA unavailable");
        }

        Self {
            surface,
            config,
            gpu: Gpu::new(device, queue),
            sample_count,
            has_compute,
        }
    }

    /// A clone of the shared GPU handle, for building renderers on the same
    /// device/queue (e.g. an [`OffscreenViewer`] or an egui renderer).
    pub fn gpu(&self) -> Gpu {
        self.gpu.clone()
    }

    /// The surface color format (an sRGB format when available).
    pub fn format(&self) -> wgpu::TextureFormat {
        self.config.format
    }

    /// The MSAA sample count supported for this surface format.
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Whether the adapter supports compute shaders.
    pub fn has_compute(&self) -> bool {
        self.has_compute
    }

    /// Current surface size as (width, height).
    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    /// Reconfigure the surface to a new size. Ignores zero dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
    }

    /// Acquire the next surface texture to render into and present.
    pub fn acquire(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

/// A [`Viewer`] that owns a window/canvas surface and presents to it.
pub struct SurfacedViewer<'a> {
    surface: WindowSurface<'a>,
    core: Viewer,
}

impl<'a> Deref for SurfacedViewer<'a> {
    type Target = Viewer;
    fn deref(&self) -> &Viewer {
        &self.core
    }
}

impl<'a> DerefMut for SurfacedViewer<'a> {
    fn deref_mut(&mut self) -> &mut Viewer {
        &mut self.core
    }
}

impl<'a> SurfacedViewer<'a> {
    /// Create a new viewer with the given surface target.
    pub async fn new<T>(surface_target: T, width: u32, height: u32) -> Self
    where
        T: Into<wgpu::SurfaceTarget<'a>>,
    {
        let surface = WindowSurface::new(surface_target, width, height).await;
        Self::from_surface(surface, width, height)
    }

    /// Build a surfaced viewer around an already-configured [`WindowSurface`].
    pub fn from_surface(surface: WindowSurface<'a>, width: u32, height: u32) -> Self {
        let core = Viewer::from_gpu(
            surface.gpu(),
            surface.format(),
            width,
            height,
            surface.sample_count(),
            surface.has_compute(),
        );
        Self { surface, core }
    }

    /// Create a new viewer from a winit Window (native platforms).
    /// The viewer size is automatically determined from the window's inner size.
    #[cfg(feature = "winit-support")]
    pub async fn from_window(window: std::sync::Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        Self::new(window, size.width, size.height).await
    }

    /// Create a new viewer from an HTML canvas element (WebAssembly).
    /// The viewer size is automatically determined from the canvas dimensions.
    #[cfg(target_arch = "wasm32")]
    pub async fn from_canvas(canvas: web_sys::HtmlCanvasElement) -> Self {
        let width = canvas.width();
        let height = canvas.height();
        Self::new(wgpu::SurfaceTarget::Canvas(canvas), width, height).await
    }

    /// Handle a single event, reconfiguring the surface on resize before
    /// delegating to the core viewer.
    pub fn handle_event(&mut self, event: &Event) {
        if let Event::Device(DeviceEvent::Resized(physical_size)) = event {
            let (w, h) = *physical_size;
            self.surface.resize(w, h);
        }
        self.core.handle_event(event);
    }

    /// Render all views using the default rendering path.
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let (output, _view, encoder) = self.render_scene()?;
        self.present(encoder, output);
        Ok(())
    }

    /// Render all views into the surface, returning the surface output, view,
    /// and command encoder for further rendering (overlays, post-processing,
    /// etc.).
    ///
    /// Call [`present()`](Self::present) when done to submit commands and display the frame.
    pub fn render_scene(&mut self) -> Result<(wgpu::SurfaceTexture, wgpu::TextureView, wgpu::CommandEncoder), anyhow::Error> {
        let output = self.surface.acquire()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.core.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") },
        );

        self.core.render_to_target(&view, &mut encoder)?;

        Ok((output, view, encoder))
    }

    /// Submit the command encoder and present the surface texture.
    pub fn present(&self, encoder: wgpu::CommandEncoder, output: wgpu::SurfaceTexture) {
        self.core.queue().submit(std::iter::once(encoder.finish()));
        output.present();
    }
}

/// A [`Viewer`] that renders into an owned offscreen texture instead of a
/// surface. The texture is sampleable (`TEXTURE_BINDING`), so consumers can
/// display it inside a UI panel (e.g. `egui::Image` via
/// `egui_wgpu::Renderer::register_native_texture`) or read it back to an image.
pub struct OffscreenViewer {
    color_format: wgpu::TextureFormat,
    color_texture: wgpu::Texture,
    render_view: wgpu::TextureView,
    sample_view: wgpu::TextureView,
    core: Viewer,
}

impl Deref for OffscreenViewer {
    type Target = Viewer;
    fn deref(&self) -> &Viewer {
        &self.core
    }
}

impl DerefMut for OffscreenViewer {
    fn deref_mut(&mut self) -> &mut Viewer {
        &mut self.core
    }
}

impl OffscreenViewer {
    /// Build an offscreen viewer around an existing GPU context.
    ///
    /// Use this to share the device/queue with the rest of an application (e.g.
    /// the device that owns the window surface and the egui renderer) so the
    /// rendered texture can be sampled by that same device. Views render at
    /// `sample_count` internally; the owned color texture is single-sampled.
    pub fn from_gpu(
        gpu: Gpu,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        sample_count: u32,
        has_compute: bool,
    ) -> Self {
        let (color_texture, render_view, sample_view) =
            Self::create_color(&gpu.device, color_format, width, height);
        let core = Viewer::from_gpu(gpu, color_format, width, height, sample_count, has_compute);
        Self {
            color_format,
            color_texture,
            render_view,
            sample_view,
            core,
        }
    }

    /// Create an offscreen viewer with its own headless GPU context.
    ///
    /// Convenience for thumbnails / server-side rendering where no surface or
    /// external device is involved. Uses `Rgba8UnormSrgb` and no MSAA.
    pub async fn headless(width: u32, height: u32) -> anyhow::Result<Self> {
        let (gpu, caps) = Gpu::headless().await?;
        Ok(Self::from_gpu(
            gpu,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            width,
            height,
            1,
            caps.has_compute,
        ))
    }

    fn create_color(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
        // The sample view uses the non-sRGB variant so samplers read the raw
        // (gamma-encoded) bytes; for a non-sRGB `format` this is the same format.
        let sample_format = format.remove_srgb_suffix();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Color Target"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[sample_format],
        });
        let render_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sample_view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(sample_format),
            ..Default::default()
        });
        (texture, render_view, sample_view)
    }

    /// Resize the offscreen target, recreating the color texture.
    pub fn resize(&mut self, width: u32, height: u32) {
        let (texture, render_view, sample_view) =
            Self::create_color(self.core.device(), self.color_format, width, height);
        self.color_texture = texture;
        self.render_view = render_view;
        self.sample_view = sample_view;
        self.core.resize(width, height);
    }

    /// Render all views into the owned offscreen texture and submit. Does not
    /// present (there is no surface); read the result via [`texture_view`] or
    /// [`ViewMut::render_to_image`].
    ///
    /// [`texture_view`]: Self::texture_view
    pub fn render(&mut self) -> Result<(), anyhow::Error> {
        let mut encoder = self.core.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Offscreen Render Encoder") },
        );
        // Disjoint field borrows: `&self.render_view` alongside `&mut self.core`.
        self.core.render_to_target(&self.render_view, &mut encoder)?;
        self.core.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// The offscreen color texture being rendered into.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.color_texture
    }

    /// A non-sRGB view of the offscreen color texture, for sampling (e.g. egui).
    /// See the type docs for why this is not the sRGB render view.
    pub fn texture_view(&self) -> &wgpu::TextureView {
        &self.sample_view
    }
}
