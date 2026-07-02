mod boolean;
mod cursor;
mod document;
mod extrude;
mod grid;
mod io;
mod loft;
mod operators;
mod preview;
mod snap;
mod tool;
mod tool_manager;
mod ui;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use egui_wgpu::RendererOptions;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::winit_support;
use duck_engine_viewer::{OffscreenViewer, WindowSurface};
use duck_engine_viewer::event::Event;
use duck_engine_viewer::input::ElementState;
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, TransformMode};
use duck_engine_viewer::common::{
    Vector3, InnerSpace
};
use duck_engine_viewer::scene::{
    Scene, NodeFlags, NodePayload, PositionedCamera,
};

use crate::operators::{
    BooleanOperator, BoxOperator, CircleOperator, ConstructionOptions, CurveOperator,
    CylinderOperator, ExtrudeOperator, LineOperator, LoftOperator, RectangleOperator,
    SphereOperator, TransformTool,
};
use crate::tool_manager::ToolManager;
use crate::ui::{ModelerUi, UiAction};

use document::Document;

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
    ui: ModelerUi,
    /// Stable egui texture id the offscreen color texture is registered under.
    /// Re-pointed (not re-created) when the offscreen texture is resized.
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

    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    tools: ToolManager,
}

impl ViewerState<'static> {
    async fn new(event_loop: &ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes()
                    .with_title("Modeler")
                    .with_inner_size(winit::dpi::LogicalSize::new(1200, 1000))
                ).expect("Failed to create window"),
        );

        let size = window.inner_size();
        let surface = WindowSurface::new(Arc::clone(&window), size.width, size.height).await;
        let mut viewer = OffscreenViewer::from_gpu(
            surface.gpu(),
            surface.format(),
            size.width,
            size.height,
            surface.sample_count(),
            surface.has_compute(),
        );

        let egui_ctx = egui::Context::default();
        egui_extras::install_image_loaders(&egui_ctx);

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

        let construction_options = Rc::new(RefCell::new(ConstructionOptions::new()));
        let document = Arc::new(Mutex::new(Document::new(viewer.scene())));

        let sel_op = Arc::new(Mutex::new(SelectionOperator::new()));
        viewer.dispatcher_mut().push_back(sel_op.clone());
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(NavigationOperator::new())));

        let mut tools = ToolManager::new(sel_op);
        tools.install(viewer.dispatcher_mut());
        tools.register(TransformTool::new(TransformMode::Translate, Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(TransformTool::new(TransformMode::Rotate, Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(TransformTool::new(TransformMode::Scale, Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(SphereOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(BoxOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(RectangleOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(LineOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CurveOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CircleOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(CylinderOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(BooleanOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(ExtrudeOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));
        tools.register(LoftOperator::new(Rc::clone(&construction_options), Arc::clone(&document)));

        Self {
            egui_renderer,
            egui_winit,
            egui_ctx,
            ui: ModelerUi::default(),
            scene_texture_id,
            viewport_rect: None,
            viewport_drag_active: false,
            last_cursor: None,
            viewer,
            surface,
            window,
            construction_options,
            document,
            tools,
        }
    }

    fn set_default_scene(&mut self) {
        let mut scene = Scene::new();

        // Setup default camera and lighting
        let eye = [75.0, 50.0, 75.0].into();
        let target = [0.0, 0.0, 0.0].into();
        let forward: Vector3 = target - eye;
        let right = forward.cross([0.0, 1.0, 0.0].into()).normalize();
        let up = right.cross(forward);

        let size = self.viewer.size();

        let camera = PositionedCamera {
            eye,
            target,
            up,
            aspect: size.0 as f32 / size.1 as f32,
            fovy: 35.0,
            znear: 1.0,
            zfar: 5_000f32,
            ortho: false
        };
        let camera_transform = camera.to_node_transform();
        let camera_projection = camera.projection();
        let camera_node_id = scene.add_node(
            None,
            Some("Main camera".to_owned()),
            camera_transform,
            NodeFlags::NONE
        ).expect("Failed to add camera on default scene");
        scene.set_node_payload(camera_node_id, NodePayload::Camera(camera_projection));
        scene.set_active_camera(Some(camera_node_id));
        scene.set_default_light_nodes(camera_node_id);

        let coptions = self.construction_options.borrow();
        let _grid = grid::Grid::add_to_scene(&mut scene, &coptions.grid, &coptions.construction_plane);
        drop(coptions);

        let scene_arc = Arc::new(Mutex::new(scene));
        self.viewer.set_scene(Arc::clone(&scene_arc));
        self.document.lock().unwrap().set_scene(scene_arc);
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

    /// Returns true when the user asked to quit via the menu.
    fn handle_redraw(&mut self) -> bool {
        self.viewer.update();

        // Build the egui frame: docked panels, then the central panel holding
        // the (stable) 3D scene texture. The central image rect is captured to
        // size the offscreen viewer and route viewport input.
        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let egui_ctx = self.egui_ctx.clone();
        let scene_texture_id = self.scene_texture_id;
        let mut viewport_rect = None;
        let mut ui_actions = Vec::new();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            ui_actions =
                self.ui.show(ctx, &self.document, self.viewer.selection_mut(), &mut self.tools);
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let size = ui.available_size();
                    let image =
                        egui::Image::new(egui::load::SizedTexture::new(scene_texture_id, size));
                    viewport_rect = Some(ui.add(image).rect);
                });
        });
        self.egui_winit.handle_platform_output(&self.window, full_output.platform_output.clone());

        // After egui so a panel-driven finish (e.g. boolean Apply) cedes back
        // to selection in the same frame.
        self.tools.update(&self.viewer.scene());

        // UI actions run outside the frame closure: the file dialogs block.
        for action in ui_actions {
            match action {
                UiAction::ImportCad => {
                    let options = self.construction_options.borrow().geometry_options.clone();
                    if let Err(e) = io::import_cad_dialog(&self.document, &options) {
                        log::error!("CAD import failed: {e:#}");
                    }
                }
                UiAction::ExportCad => {
                    if let Err(e) = io::export_cad_dialog(&self.document) {
                        log::error!("CAD export failed: {e:#}");
                    }
                }
                UiAction::Quit => return true,
            }
        }

        // Reconcile the offscreen texture size with the central panel, then
        // re-point the (stable) egui texture id at the new view.
        let ppp = full_output.pixels_per_point;
        self.viewport_rect = viewport_rect.map(|r| {
            egui::Rect::from_min_size(
                egui::pos2(r.min.x * ppp, r.min.y * ppp),
                egui::vec2(r.width() * ppp, r.height() * ppp),
            )
        });
        if let Some(rect) = self.viewport_rect {
            let w = (rect.width().round() as u32).max(1);
            let h = (rect.height().round() as u32).max(1);
            if (w, h) != self.viewer.size() {
                self.viewer.resize(w, h);
                self.egui_renderer.update_egui_texture_from_wgpu_texture(
                    self.viewer.device(),
                    self.viewer.texture_view(),
                    wgpu::FilterMode::Linear,
                    self.scene_texture_id,
                );
            }
        }

        // Render the 3D scene into the offscreen texture (own encoder+submit).
        if let Err(e) = self.viewer.render() {
            log::error!("Offscreen render error: {}", e);
        }

        // Present: egui paints the whole window (including the scene image)
        // into the surface.
        match self.surface.acquire() {
            Ok(output) => {
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let mut encoder = self.viewer.device().create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("egui Encoder") },
                );
                self.render_egui_overlay(&full_output, ppp, &mut encoder, &view);
                self.viewer.queue().submit(std::iter::once(encoder.finish()));
                output.present();
            }
            Err(e) => log::error!("Surface acquire error: {}", e),
        }

        self.window.request_redraw();
        false
    }

    /// Render the full egui frame (panels + the 3D scene image) into `view`.
    fn render_egui_overlay(
        &mut self,
        full_output: &egui::FullOutput,
        pixels_per_point: f32,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
    ) {
        let device = self.viewer.device();
        let queue = self.viewer.queue();

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(device, queue, *id, image_delta);
        }

        let clipped_primitives =
            self.egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point);

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: {
                let (w, h) = self.surface.size();
                [w, h]
            },
            pixels_per_point,
        };

        self.egui_renderer.update_buffers(
            device,
            queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

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
            self.egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &clipped_primitives,
                &screen_descriptor,
            );
        }

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

struct App<'a> {
    state: Option<ViewerState<'a>>,
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_none() {
            let mut state = pollster::block_on(ViewerState::new(event_loop));
            state.set_default_scene();
            state.window.request_redraw();
            self.state = Some(state);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if let Some(state) = self.state.as_mut() {
                    if state.handle_redraw() {
                        event_loop.exit();
                    }
                }
            }
            _ => {
                if let Some(state) = self.state.as_mut() {
                    state.handle_window_event(&event);
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

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.run_app(&mut App { state: None }).unwrap();
}
