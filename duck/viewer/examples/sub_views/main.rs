//! Multi-view demonstration: a four-pane top/front/right/iso layout of one
//! scene plus a built-in axis-triad overlay.
//!
//! Exercises every sub-view mechanism:
//! - Four `Fractional` quadrant views share one scene: each pane has its own
//!   camera and operator stack (orbit them independently), while selection is
//!   shared — click a shape in any pane and the outline appears in all four.
//! - `Viewer::add_axis_triad` overlays the iso pane's corner with an axis
//!   triad mirroring the iso camera. Clicking one of its six handles animates
//!   the iso camera to look down that axis; clicking the axis already faced
//!   flips to the opposite side.
//!
//! Run with `cargo run --example sub_views -p duck-engine-viewer`.

use std::sync::Arc;

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::common::{InnerSpace, Point3, RgbaColor, Transform, Vector3};
use duck_engine_viewer::event::{DeviceEvent as DE, Event};
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator};
use duck_engine_viewer::scene::Scene;
use duck_engine_viewer::scene::resource::{
    FaceMaterial, Instance, Mesh, NodeFlags, PrimitiveType,
};
use duck_engine_viewer::{
    AxisTriadConfig, SurfacedViewer, ViewLayout, winit_support,
};

/// Build the shared model: a few solids around the origin.
fn build_model_scene(scene: &Scene) {
    let mut scene = scene.lock();

    let mut add = |mesh: Mesh, material: FaceMaterial, name: &str, position: Point3| {
        let mesh = scene.add_mesh(mesh);
        let material = scene.add_face_material(material);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh).with_face_material(material),
                Some(name.to_string()),
                Transform::from_position(position),
                NodeFlags::NONE,
            )
            .unwrap();
    };

    add(
        Mesh::sphere(0.6, 32, 16, PrimitiveType::TriangleList),
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor { r: 0.8, g: 0.3, b: 0.2, a: 1.0 })
            .with_roughness_factor(0.35),
        "Sphere",
        Point3::new(-1.2, 0.6, 0.0),
    );
    add(
        Mesh::cube(1.0, PrimitiveType::TriangleList),
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor { r: 0.25, g: 0.5, b: 0.85, a: 1.0 })
            .with_roughness_factor(0.5),
        "Cube",
        Point3::new(1.0, 0.5, 0.6),
    );
    add(
        Mesh::torus(0.7, 0.22, 40, 20, PrimitiveType::TriangleList),
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor { r: 0.9, g: 0.75, b: 0.2, a: 1.0 })
            .with_metallic_factor(0.8)
            .with_roughness_factor(0.3),
        "Torus",
        Point3::new(0.2, 0.25, -1.3),
    );
    add(
        Mesh::plane(8.0, 8.0, 1, 1, PrimitiveType::TriangleList),
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor { r: 0.35, g: 0.35, b: 0.38, a: 1.0 })
            .with_roughness_factor(0.9),
        "Ground",
        Point3::new(0.0, 0.0, 0.0),
    );
}

struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<SurfacedViewer<'a>>,
}

impl<'a> App<'a> {
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs = Window::default_attributes()
            .with_title("Duck Engine - Sub-Views (click axes / shapes, orbit any pane)")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 960));
        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());

        let size = window.inner_size();
        let mut viewer = pollster::block_on(SurfacedViewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        let model = Scene::default();
        build_model_scene(&model);

        // Four quadrants of one scene. Selection is shared automatically; each
        // pane owns its camera and operators.
        let quadrant = |x, y| ViewLayout::Fractional { x, y, width: 0.5, height: 0.5 };
        let panes = [
            ("Top", quadrant(0.0, 0.0), Vector3::unit_y(), Vector3::new(0.0, 0.0, -1.0), true),
            ("Iso", quadrant(0.5, 0.0), Vector3::new(1.0, 0.8, 1.0).normalize(), Vector3::unit_y(), false),
            ("Front", quadrant(0.0, 0.5), Vector3::unit_z(), Vector3::unit_y(), true),
            ("Right", quadrant(0.5, 0.5), Vector3::unit_x(), Vector3::unit_y(), true),
        ];

        let bounds = model.lock().bounding().bounds;
        let mut iso_view = None;
        for (name, layout, direction, up, ortho) in panes {
            let id = viewer.add_view(name, model.clone(), layout);
            let mut view = viewer.view_mut(id).unwrap();
            view.dispatcher_mut()
                .push_back(Arc::new(std::sync::Mutex::new(SelectionOperator::new())));
            view.dispatcher_mut()
                .push_back(Arc::new(std::sync::Mutex::new(NavigationOperator::new())));
            let camera = view.camera_mut();
            camera.target = Point3::new(0.0, 0.0, 0.0);
            camera.eye = Point3::new(0.0, 0.0, 0.0) + direction * 10.0;
            camera.up = up;
            camera.ortho = ortho;
            if let Some(bounds) = &bounds {
                camera.fit_to_bounds(bounds);
            }
            if name == "Iso" {
                iso_view = Some(id);
            }
        }

        // The built-in axis triad, overlaying the iso pane's corner.
        viewer.add_axis_triad(iso_view.unwrap(), AxisTriadConfig::default());

        window.request_redraw();
        self.window = Some(window);
        self.viewer = Some(viewer);
    }
}

impl<'a> ApplicationHandler for App<'a> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.initialize(event_loop);
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
                let viewer = self.viewer.as_mut().unwrap();
                viewer.update();
                if let Err(e) = viewer.render() {
                    log::error!("Render error: {}", e);
                }
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }

        if let Some(app_event) = winit_support::convert_window_event(event) {
            let viewer = self.viewer.as_mut().unwrap();
            viewer.handle_event(&app_event);

            if let Event::Device(DE::KeyboardInput { event: key_event, .. }) = &app_event {
                if matches!(
                    key_event.logical_key,
                    duck_engine_viewer::input::Key::Named(duck_engine_viewer::input::NamedKey::Escape)
                ) {
                    event_loop.exit();
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(app_event) = winit_support::convert_device_event(event) {
            self.viewer.as_mut().unwrap().handle_event(&app_event);
        }
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App { window: None, viewer: None };

    event_loop.run_app(&mut app).unwrap();
}
