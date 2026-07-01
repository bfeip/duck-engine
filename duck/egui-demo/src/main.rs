mod io;
mod platform;
mod ui;

use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
#[cfg(target_arch = "wasm32")]
use std::{cell::RefCell, rc::Rc};

use egui_wgpu::RendererOptions;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};
#[cfg(target_arch = "wasm32")]
use winit::event_loop::EventLoopProxy;

use duck_engine_common::Point3;
use duck_engine_viewer::event::Event;
use duck_engine_viewer::{common::RgbaColor, scene::NodeFlags};
use duck_engine_viewer::input::{ElementState, Key};
use duck_engine_viewer::operator::{NavigationMode, NavigationOperator, SelectionOperator, TransformMode, TransformOperator};
use duck_engine_viewer::common::Transform;
use duck_engine_viewer::scene::{Light, LightType, NodePayload, Scene};
use duck_engine_viewer::winit_support;
use duck_engine_viewer::{OffscreenViewer, WindowSurface};

/// Debug actions triggered by key presses
enum DebugAction {
    CycleOperator,
    ToggleOrtho,
    CycleWorkflow,
}

/// Custom event delivered through the winit event loop.
///
/// On web, wgpu init is async and cannot block, so the fully-built
/// [`ViewerState`] is created off the event loop and handed back via the proxy.
/// On native this enum is uninhabited (init happens synchronously in `resumed`).
enum UserEvent {
    #[cfg(target_arch = "wasm32")]
    Initialized(ViewerState<'static>),
    /// The browser window was resized. winit's web backend doesn't seem to emit
    /// `Resized` for browser-window resizes, so we drive it ourselves from a 
    /// resize` listener.
    #[cfg(target_arch = "wasm32")]
    Resized(PhysicalSize<u32>),
}

/// Owns all rendering state: egui context + GPU renderer, the window surface
/// egui presents to, and the [`OffscreenViewer`] that renders the 3D scene into
/// a texture displayed inside the central panel.
///
/// Field order matters: Rust drops fields in declaration order, so egui
/// resources are released before the viewer and surface. This prevents
/// segfaults from background threads on Wayland during shutdown.
struct ViewerState<'a> {
    egui_renderer: egui_wgpu::Renderer,
    egui_winit: egui_winit::State,
    egui_ctx: egui::Context,
    /// Stable egui texture id the offscreen color texture is registered under.
    scene_texture_id: egui::TextureId,
    /// The central-panel image rect in physical pixels, stashed each frame for
    /// input routing. `None` until the first frame is built.
    viewport_rect: Option<egui::Rect>,
    /// True while a pointer drag that began inside the viewport is in progress;
    /// keeps routing to the viewer even if the cursor crosses a panel.
    viewport_drag_active: bool,
    /// Latest cursor position in physical pixels (window space).
    last_cursor: Option<(f32, f32)>,
    viewer: OffscreenViewer,
    surface: WindowSurface<'a>,
    window: Arc<Window>,
    nav_op: Arc<Mutex<NavigationOperator>>,
}

impl ViewerState<'static> {
    /// Build the viewer + egui state for an existing window. `size` overrides
    /// the window's reported inner size (used on web, where it can lag).
    async fn from_window(window: Arc<Window>, size: Option<PhysicalSize<u32>>) -> Self {
        let size = size.unwrap_or(window.inner_size());
        let surface = WindowSurface::new(Arc::clone(&window), size.width, size.height).await;

        let mut viewer = OffscreenViewer::from_gpu(
            surface.gpu(),
            surface.format(),
            size.width,
            size.height,
            surface.sample_count(),
            surface.has_compute(),
        );

        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Translate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Rotate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Scale))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(SelectionOperator::new())));
        let nav_op = Arc::new(Mutex::new(NavigationOperator::new()));
        viewer.dispatcher_mut().push_back(nav_op.clone());

        let egui_ctx = egui::Context::default();
        let egui_winit = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(
            viewer.device(),
            surface.format(),
            RendererOptions::default(),
        );

        // Register the offscreen scene texture once; the id is stable and
        // re-pointed on resize via `update_egui_texture_from_wgpu_texture`.
        let scene_texture_id = egui_renderer.register_native_texture(
            viewer.device(),
            viewer.texture_view(),
            wgpu::FilterMode::Linear,
        );

        Self {
            egui_renderer,
            egui_winit,
            egui_ctx,
            scene_texture_id,
            viewport_rect: None,
            viewport_drag_active: false,
            last_cursor: None,
            viewer,
            surface,
            window,
            nav_op,
        }
    }
}

impl<'a> ViewerState<'a> {
    /// Handle a window event. egui always sees it first (for its own hover /
    /// focus state); events that belong to the 3D viewport are additionally
    /// routed to the viewer with pointer-capture semantics and viewport-local
    /// coordinates.
    fn handle_window_event(&mut self, event: &WindowEvent) {
        // The surface tracks the whole window; the offscreen viewer is sized
        // from the central panel each frame, not from the window.
        if let WindowEvent::Resized(size) = event {
            self.surface.resize(size.width, size.height);
        }
        if let WindowEvent::CursorMoved { position, .. } = event {
            self.last_cursor = Some((position.x as f32, position.y as f32));
        }

        let _ = self.egui_winit.on_window_event(&self.window, event);

        let Some(app_event) = winit_support::convert_window_event(event.clone()) else {
            return;
        };
        if self.should_route_to_viewport(&app_event) {
            let app_event = self.to_viewport_local(app_event);
            self.viewer.handle_event(&app_event);
        }
    }

    /// Raw mouse motion drives active gizmo/transform drags; only forward it
    /// while a viewport drag is in progress.
    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if !self.viewport_drag_active {
            return;
        }
        if let Some(app_event) = winit_support::convert_device_event(event.clone()) {
            self.viewer.handle_event(&app_event);
        }
    }

    /// Whether the latest cursor position sits inside the 3D viewport rect.
    fn cursor_in_viewport(&self) -> bool {
        match (self.viewport_rect, self.last_cursor) {
            (Some(rect), Some((x, y))) => rect.contains(egui::pos2(x, y)),
            _ => false,
        }
    }

    /// Decide whether a converted event should be routed to the 3D viewer,
    /// updating the pointer-capture flag on press/release.
    fn should_route_to_viewport(&mut self, event: &Event) -> bool {
        use duck_engine_viewer::event::DeviceEvent as DE;
        match event {
            Event::Device(DE::MouseInput { state, .. }) => match state {
                ElementState::Pressed => {
                    if self.cursor_in_viewport() {
                        self.viewport_drag_active = true;
                        true
                    } else {
                        false
                    }
                }
                ElementState::Released => {
                    if self.viewport_drag_active {
                        self.viewport_drag_active = false;
                        true
                    } else {
                        false
                    }
                }
            },
            Event::Device(DE::CursorMoved { .. }) => {
                self.viewport_drag_active
                    || (self.cursor_in_viewport() && !self.egui_ctx.is_using_pointer())
            }
            Event::Device(DE::MouseWheel { .. }) => self.cursor_in_viewport(),
            Event::Device(DE::KeyboardInput { .. }) => !self.egui_ctx.wants_keyboard_input(),
            _ => false,
        }
    }

    /// Translate absolute cursor coordinates from window space into the 3D
    /// viewport's local pixel space (its top-left is the offscreen origin).
    /// Only `CursorMoved` carries an absolute position; drags/clicks are
    /// synthesized from it downstream, so translating it is sufficient.
    fn to_viewport_local(&self, event: Event) -> Event {
        use duck_engine_viewer::event::DeviceEvent as DE;
        match (event, self.viewport_rect) {
            (Event::Device(DE::CursorMoved { position }), Some(rect)) => {
                Event::Device(DE::CursorMoved {
                    position: (position.0 - rect.min.x as f64, position.1 - rect.min.y as f64),
                })
            }
            (event, _) => event,
        }
    }
}

/// Application state for the winit event loop.
struct App<'a> {
    state: Option<ViewerState<'a>>,
    ui: ui::UiState,
    /// Index of the currently active workflow (cycled by the W debug key).
    workflow_index: usize,
    /// Pending HDR environment path to load
    #[cfg(not(target_arch = "wasm32"))]
    pending_hdr_path: Option<PathBuf>,
    /// Pending scene file path to load
    #[cfg(not(target_arch = "wasm32"))]
    pending_scene_load_path: Option<PathBuf>,
    /// Pending scene file path to save
    #[cfg(not(target_arch = "wasm32"))]
    pending_scene_save_path: Option<PathBuf>,
    /// Proxy used to hand the asynchronously-built viewer state back to the loop.
    #[cfg(target_arch = "wasm32")]
    proxy: EventLoopProxy<UserEvent>,
    /// Scene bytes picked from the browser file dialog, awaiting load.
    #[cfg(target_arch = "wasm32")]
    pending_scene_bytes: Rc<RefCell<Option<Vec<u8>>>>,
    /// HDR bytes picked from the browser file dialog, awaiting load.
    #[cfg(target_arch = "wasm32")]
    pending_hdr_bytes: Rc<RefCell<Option<Vec<u8>>>>,
}

impl<'a> App<'a> {
    fn handle_redraw_requested(&mut self) {
        if self.state.is_none() {
            return;
        }
        self.process_pending_io();

        let mut ui_actions = ui::UiActions::default();
        if let Some(state) = self.state.as_mut() {
            state.viewer.update();

            // Build the egui frame: side/top panels, then the central panel
            // holding the (stable) 3D scene texture. The central image rect is
            // captured to size the offscreen viewer and route viewport input.
            let raw_input = state.egui_winit.take_egui_input(&state.window);
            let ui = &mut self.ui;
            let scene_texture_id = state.scene_texture_id;
            let mut viewport_rect = None;
            let full_output = state.egui_ctx.run(raw_input, |ctx| {
                ui_actions = ui.build(ctx, &mut state.viewer, &state.nav_op);
                egui::CentralPanel::default()
                    .frame(egui::Frame::NONE)
                    .show(ctx, |ui| {
                        let size = ui.available_size();
                        let image = egui::Image::new(egui::load::SizedTexture::new(
                            scene_texture_id,
                            size,
                        ));
                        viewport_rect = Some(ui.add(image).rect);
                    });
            });

            state.egui_winit.handle_platform_output(
                &state.window,
                full_output.platform_output.clone(),
            );

            // Reconcile the offscreen texture size with the central panel, then
            // re-point the (stable) egui texture id at the new view. The image
            // widget already references the id, so egui scales last frame's
            // texture for the one frame the sizes disagree.
            let ppp = full_output.pixels_per_point;
            state.viewport_rect = viewport_rect.map(|r| {
                egui::Rect::from_min_size(
                    egui::pos2(r.min.x * ppp, r.min.y * ppp),
                    egui::vec2(r.width() * ppp, r.height() * ppp),
                )
            });
            if let Some(rect) = state.viewport_rect {
                let w = (rect.width().round() as u32).max(1);
                let h = (rect.height().round() as u32).max(1);
                if (w, h) != state.viewer.size() {
                    state.viewer.resize(w, h);
                    state.egui_renderer.update_egui_texture_from_wgpu_texture(
                        state.viewer.device(),
                        state.viewer.texture_view(),
                        wgpu::FilterMode::Linear,
                        state.scene_texture_id,
                    );
                }
            }

            // Render the 3D scene into the offscreen texture (own encoder+submit).
            if let Err(e) = state.viewer.render() {
                log::error!("Offscreen render error: {}", e);
            }

            // Present: egui paints the whole window (including the scene image)
            // into the surface.
            match state.surface.acquire() {
                Ok(output) => {
                    let view = output
                        .texture
                        .create_view(&wgpu::TextureViewDescriptor::default());
                    let mut encoder = state.viewer.device().create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("egui Encoder") },
                    );
                    render_egui_overlay(
                        &mut state.egui_renderer,
                        &state.egui_ctx,
                        &full_output,
                        state.surface.size(),
                        ppp,
                        state.viewer.device(),
                        state.viewer.queue(),
                        &mut encoder,
                        &view,
                    );
                    state.viewer.queue().submit(std::iter::once(encoder.finish()));
                    output.present();
                }
                Err(e) => log::error!("Surface acquire error: {}", e),
            }

            state.window.request_redraw();
        }

        // Handle UI actions (after releasing the state borrow)
        if ui_actions.load_scene {
            self.open_scene_file_dialog();
        }
        if ui_actions.save_scene {
            self.save_scene_file_dialog();
        }
        if ui_actions.clear_scene {
            self.clear_scene();
        }
        if let Some(light_type) = ui_actions.add_light {
            self.add_light(light_type);
        }
        if ui_actions.load_environment {
            self.open_hdr_file_dialog();
        }
        if ui_actions.clear_environment {
            self.clear_environment();
        }

        {
            let scene_arc = self.state.as_mut().unwrap().viewer.scene();
            let mut scene = scene_arc.lock().unwrap();
            for change in ui_actions.visibility_changes {
                scene.set_node_visibility(change.node_id, change.new_visibility);
            }
        }

        if let Some(camera) = ui_actions.set_camera {
            self.state.as_mut().unwrap().viewer.set_camera(camera);
        }
        #[cfg(feature = "streaming")]
        if let Some(url) = ui_actions.connect_stream {
            let viewer = &mut self.state.as_mut().unwrap().viewer;
            match viewer.connect_stream(&url) {
                Ok(()) => {
                    self.ui.left.network.status =
                        ui::network_tab::NetworkStatus::Connected;
                }
                Err(e) => {
                    self.ui.left.network.status =
                        ui::network_tab::NetworkStatus::Error(e.to_string());
                }
            }
        }
        #[cfg(feature = "streaming")]
        if ui_actions.disconnect_stream {
            self.state.as_mut().unwrap().viewer.disconnect_stream();
        }
    }

    fn clear_scene(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.viewer.set_scene(Arc::new(Mutex::new(Scene::new())));
            log::info!("Scene cleared");
        }
    }

    fn add_light(&mut self, light_type: LightType) {
        let Some(state) = self.state.as_mut() else { return };
        let viewer = &mut state.viewer;
        let white = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };

        let (light, transform) = match light_type {
            LightType::Point => (
                Light::point(white, 1.0),
                Transform::from_position(Point3::new(0.0, 3.0, 0.0)),
            ),
            LightType::Directional => (Light::directional(white, 1.0), Transform::IDENTITY),
            LightType::Spot => (
                Light::spot(white, 1.0, 30.0_f32.to_radians(), 45.0_f32.to_radians()),
                Transform::from_position(Point3::new(0.0, 3.0, 0.0)),
            ),
        };

        let scene_arc = viewer.scene();
        let mut scene = scene_arc.lock().unwrap();
        let node_id = scene.add_node(None, None, transform, NodeFlags::NONE).expect("add light node");
        scene.set_node_payload(node_id, NodePayload::Light(light));
        log::info!("Added {:?} light", light_type);
    }

    fn clear_environment(&mut self) {
        if let Some(state) = self.state.as_mut() {
            let scene_arc = state.viewer.scene();
            let mut scene = scene_arc.lock().unwrap();
            scene.set_active_environment_map(None);
            log::info!("Environment cleared");
        }
    }

    fn handle_debug_key_action(&mut self, action: DebugAction, _event_loop: &ActiveEventLoop) {
        match action {
            DebugAction::CycleOperator => self.cycle_operator_mode(),
            DebugAction::ToggleOrtho => self.toggle_ortho(),
            DebugAction::CycleWorkflow => self.cycle_workflow(),
        }
    }

    fn get_debug_key_action(event: &duck_engine_viewer::event::Event) -> Option<DebugAction> {
        let duck_engine_viewer::event::Event::Device(
            duck_engine_viewer::event::DeviceEvent::KeyboardInput { event: key_event, .. },
        ) = event
        else {
            return None;
        };
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return None;
        }
        match &key_event.logical_key {
            Key::Character('c') => Some(DebugAction::CycleOperator),
            Key::Character('o') => Some(DebugAction::ToggleOrtho),
            Key::Character('w') => Some(DebugAction::CycleWorkflow),
            _ => None,
        }
    }

    fn cycle_operator_mode(&mut self) {
        let Some(state) = self.state.as_mut() else { return };
        let new_mode = match state.nav_op.lock().unwrap().mode() {
            NavigationMode::Turntable => NavigationMode::Walk,
            NavigationMode::Walk => NavigationMode::Trackball,
            NavigationMode::Trackball => NavigationMode::Turntable,
        };
        state.nav_op.lock().unwrap().set_mode(new_mode);
    }

    fn toggle_ortho(&mut self) {
        if let Some(state) = self.state.as_mut() {
            state.viewer.with_camera_mut(|c| c.ortho = !c.ortho);
        }
    }

    fn cycle_workflow(&mut self) {
        use duck_engine_viewer::renderer::SceneWorkflow;
        let Some(state) = self.state.as_mut() else { return };
        self.workflow_index = (self.workflow_index + 1) % 2;
        let workflow: Box<SceneWorkflow> = match self.workflow_index {
            0 => Box::new(state.viewer.shaded_workflow()),
            _ => Box::new(state.viewer.hidden_line_workflow(Default::default())),
        };
        log::info!("Switched to '{}' workflow", workflow.name());
        state.viewer.set_workflow(workflow);
    }
}

impl<'a> ApplicationHandler<UserEvent> for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        platform::resume(self, event_loop);
    }

    #[cfg(target_arch = "wasm32")]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Initialized(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
            UserEvent::Resized(size) => {
                let Some(state) = self.state.as_mut() else { return };
                // Update winit's canvas + reported inner size, then reconfigure
                // the surface. The offscreen viewer tracks the central panel and
                // is resized during the next frame build.
                let _ = state.window.request_inner_size(size);
                state.surface.resize(size.width, size.height);
                state.window.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.handle_redraw_requested();
            }
            _ => {
                let mut wants_keyboard = false;
                if let Some(state) = self.state.as_mut() {
                    state.handle_window_event(&event);
                    wants_keyboard = state.egui_ctx.wants_keyboard_input();
                }
                // Debug keys only fire when egui isn't capturing keyboard input
                // (e.g. not typing in a text field).
                if !wants_keyboard
                    && let Some(app_event) = winit_support::convert_window_event(event)
                    && let Some(action) = Self::get_debug_key_action(&app_event)
                {
                    self.handle_debug_key_action(action, event_loop);
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(state) = self.state.as_mut() {
            state.handle_device_event(&event);
        }
    }
}

/// Render the full egui frame (panels + the 3D scene image) into `view`.
fn render_egui_overlay(
    egui_renderer: &mut egui_wgpu::Renderer,
    egui_ctx: &egui::Context,
    full_output: &egui::FullOutput,
    viewer_size: (u32, u32),
    scale_factor: f32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    for (id, image_delta) in &full_output.textures_delta.set {
        egui_renderer.update_texture(device, queue, *id, image_delta);
    }

    let clipped_primitives =
        egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point);

    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [viewer_size.0, viewer_size.1],
        pixels_per_point: scale_factor,
    };

    egui_renderer.update_buffers(device, queue, encoder, &clipped_primitives, &screen_descriptor);

    {
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });
        egui_renderer.render(
            &mut render_pass.forget_lifetime(),
            &clipped_primitives,
            &screen_descriptor,
        );
    }

    for id in &full_output.textures_delta.free {
        egui_renderer.free_texture(id);
    }
}

fn main() {
    platform::run();
}
