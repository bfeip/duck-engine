mod boolean;
mod cursor;
mod document;
mod extrude;
mod grid;
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
use duck_engine_viewer::SurfacedViewer;
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
use crate::ui::ModelerUi;

use document::Document;

/// Owns all rendering state: the 3D viewer plus egui context and GPU renderer.
///
/// Field order matters: Rust drops fields in declaration order, so egui
/// resources are released before the viewer and window. This prevents
/// segfaults from background threads on Wayland during shutdown.
struct ViewerState<'a> {
    egui_renderer: egui_wgpu::Renderer,
    egui_winit: egui_winit::State,
    egui_ctx: egui::Context,
    ui: ModelerUi,
    viewer: SurfacedViewer<'a>,
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

        let mut viewer = SurfacedViewer::from_window(Arc::clone(&window)).await;

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
        let egui_renderer = egui_wgpu::Renderer::new(
            viewer.device(),
            viewer.surface_format(),
            RendererOptions::default(),
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
            viewer,
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
    fn handle_window_event(&mut self, event: &WindowEvent) {
        let response = self.egui_winit.on_window_event(&self.window, event);
        if !response.consumed {
            if let Some(app_event) = winit_support::convert_window_event(event.clone()) {
                self.viewer.handle_event(&app_event);
            }
        }
    }

    fn handle_device_event(&mut self, event: &DeviceEvent) {
        if let Some(app_event) = winit_support::convert_device_event(event.clone()) {
            self.viewer.handle_event(&app_event);
        }
    }

    fn handle_redraw(&mut self) {
        self.viewer.update();

        let raw_input = self.egui_winit.take_egui_input(&self.window);
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| {
            self.ui.show(ctx, &self.document, self.viewer.selection_mut(), &mut self.tools);
        });
        self.egui_winit.handle_platform_output(&self.window, full_output.platform_output.clone());

        // After egui so a panel-driven finish (e.g. boolean Apply) cedes back
        // to selection in the same frame.
        self.tools.update(&self.viewer.scene());

        match self.viewer.render_scene() {
            Ok((output, view, mut encoder)) => {
                self.render_egui_overlay(&full_output, &mut encoder, &view);
                self.viewer.present(encoder, output);
            }
            Err(e) => log::error!("Render error: {}", e),
        }

        self.window.request_redraw();
    }

    fn render_egui_overlay(
        &mut self,
        full_output: &egui::FullOutput,
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
                let (w, h) = self.viewer.size();
                [w, h]
            },
            pixels_per_point: self.window.scale_factor() as f32,
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
                        load: wgpu::LoadOp::Load,
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
                    state.handle_redraw();
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
