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
use duck_engine_viewer::{common::RgbaColor, scene::NodeFlags};
use duck_engine_viewer::input::{ElementState, Key};
use duck_engine_viewer::operator::{NavigationMode, NavigationOperator, SelectionOperator, TransformMode, TransformOperator};
use duck_engine_viewer::common::Transform;
use duck_engine_viewer::scene::{Light, LightType, NodePayload, Scene};
use duck_engine_viewer::winit_support;
use duck_engine_viewer::SurfacedViewer;

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

/// Owns all rendering state: the 3D viewer plus egui context and GPU renderer.
///
/// Field order matters: Rust drops fields in declaration order, so egui
/// resources are released before the viewer and window. This prevents
/// segfaults from background threads on Wayland during shutdown.
struct ViewerState<'a> {
    egui_renderer: egui_wgpu::Renderer,
    egui_winit: egui_winit::State,
    egui_ctx: egui::Context,
    viewer: SurfacedViewer<'a>,
    window: Arc<Window>,
    nav_op: Arc<Mutex<NavigationOperator>>,
}

impl ViewerState<'static> {
    /// Build the viewer + egui state for an existing window. `size` overrides
    /// the window's reported inner size (used on web, where it can lag).
    async fn from_window(window: Arc<Window>, size: Option<PhysicalSize<u32>>) -> Self {
        let size = size.unwrap_or(window.inner_size());
        let mut viewer = SurfacedViewer::new(Arc::clone(&window), size.width, size.height).await;

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
        let egui_renderer = egui_wgpu::Renderer::new(
            viewer.device(),
            viewer.surface_format(),
            RendererOptions::default(),
        );

        Self { egui_renderer, egui_winit, egui_ctx, viewer, window, nav_op }
    }
}

impl<'a> ViewerState<'a> {
    /// Handle a window event: egui gets first priority, viewer gets the rest.
    fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_winit.on_window_event(&self.window, event);
        if !response.consumed {
            if let Some(app_event) = winit_support::convert_window_event(event.clone()) {
                self.viewer.handle_event(&app_event);
            }
        }
        response.consumed
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let Some(app_event) = winit_support::convert_device_event(event.clone()) {
            self.viewer.handle_event(&app_event);
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

            let raw_input = state.egui_winit.take_egui_input(&state.window);
            let ui = &mut self.ui;
            let full_output = state.egui_ctx.run(raw_input, |ctx| {
                ui_actions = ui.build(ctx, &mut state.viewer, &state.nav_op);
            });

            state.egui_winit.handle_platform_output(
                &state.window,
                full_output.platform_output.clone(),
            );

            match state.viewer.render_scene() {
                Ok((output, view, mut encoder)) => {
                    render_egui_overlay(
                        &mut state.egui_renderer,
                        &state.egui_ctx,
                        &full_output,
                        state.viewer.size(),
                        state.window.scale_factor() as f32,
                        state.viewer.device(),
                        state.viewer.queue(),
                        &mut encoder,
                        &view,
                    );
                    state.viewer.present(encoder, output);
                }
                Err(e) => log::error!("Render error: {}", e),
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
        use duck_engine_viewer::event::{DeviceEvent, Event};
        match event {
            UserEvent::Initialized(state) => {
                state.window.request_redraw();
                self.state = Some(state);
            }
            UserEvent::Resized(size) => {
                let Some(state) = self.state.as_mut() else { return };
                // Update winit's canvas + reported inner size...
                let _ = state.window.request_inner_size(size);
                // ...and reconfigure the viewer.
                state
                    .viewer
                    .handle_event(&Event::Device(DeviceEvent::Resized((size.width, size.height))));
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
                if let Some(state) = self.state.as_mut() {
                    state.handle_window_event(&event);
                }
                if let Some(app_event) = winit_support::convert_window_event(event)
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

/// Render egui on top of the 3D scene already in `encoder`/`view`.
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
                    load: wgpu::LoadOp::Load,
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
