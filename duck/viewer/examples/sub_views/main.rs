//! Multi-view demonstration: a four-pane top/front/right/iso layout of one
//! scene plus an axis-triad overlay.
//!
//! Exercises every sub-view mechanism:
//! - Four `Fractional` quadrant views share one scene: each pane has its own
//!   camera and operator stack (orbit them independently), while selection is
//!   shared — click a shape in any pane and the outline appears in all four.
//! - An `Anchored` triad view overlays the iso pane's corner, rendering its own
//!   little axis scene over a transparent background. Each frame its camera is
//!   oriented from the iso pane's; clicking an axis arrow snaps the iso camera
//!   to look down that axis (a custom operator reaching across views through a
//!   scene handle + camera node).
//!
//! Run with `cargo run --example sub_views -p duck-engine-viewer`.

use std::sync::{Arc, Mutex};

use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::common::{
    Deg, InnerSpace, Matrix4, Point3, RgbaColor, Transform, Vector3,
};
use duck_engine_viewer::event::{DeviceEvent as DE, Event, EventContext};
use duck_engine_viewer::geom_query::{RayPickQuery, pick_all_from_ray};
use duck_engine_viewer::input::MouseButton;
use duck_engine_viewer::operator::{NavigationOperator, Operator, SelectionOperator};
use duck_engine_viewer::scene::resource::{
    FaceMaterial, FaceMaterialHandle, Instance, MaterialFlags, Mesh, NodeFlags, NodeId,
    PrimitiveType,
};
use duck_engine_viewer::scene::{Scene, SceneData};
use duck_engine_viewer::{Corner, SurfacedViewer, ViewId, ViewLayout, Viewer, winit_support};

const TRIAD_CAMERA_DISTANCE: f32 = 3.2;

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

/// One clickable arrow of the triad; maps its scene nodes to a camera pose.
struct TriadAxis {
    nodes: Vec<NodeId>,
    /// Where the eye goes, as a unit offset from the target.
    direction: Vector3,
    up: Vector3,
}

/// Build the triad's own scene: three unlit axis arrows around a hub, and
/// return the clickable axes.
fn build_triad_scene(scene: &Scene) -> Vec<TriadAxis> {
    let mut guard = scene.lock();

    fn add_part(
        scene: &mut SceneData,
        mesh: Mesh,
        material: &FaceMaterialHandle,
        name: String,
    ) -> NodeId {
        let mesh = scene.add_mesh(mesh);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh).with_face_material(material.clone()),
                Some(name),
                Transform::IDENTITY,
                NodeFlags::NONE,
            )
            .unwrap()
            .id()
    }

    let axes = [
        ("X", Vector3::unit_x(), RgbaColor { r: 0.85, g: 0.2, b: 0.2, a: 1.0 }, Matrix4::from_angle_z(Deg(-90.0))),
        ("Y", Vector3::unit_y(), RgbaColor { r: 0.2, g: 0.75, b: 0.2, a: 1.0 }, Matrix4::from_angle_x(Deg(0.0))),
        ("Z", Vector3::unit_z(), RgbaColor { r: 0.25, g: 0.4, b: 0.9, a: 1.0 }, Matrix4::from_angle_x(Deg(90.0))),
    ];

    let mut result = Vec::new();
    for (name, dir, color, rotation) in axes {
        let material = guard.add_face_material(
            // Unlit so the triad reads the same from every angle.
            FaceMaterial::new()
                .with_base_color_factor(color)
                .with_flags(MaterialFlags::DO_NOT_LIGHT),
        );
        // Shaft along +Y, rotated onto the axis and shifted outward.
        let shaft = Mesh::cylinder(0.05, 0.7, 16, true, PrimitiveType::TriangleList)
            .transformed(&(rotation * Matrix4::from_translation(Vector3::unit_y() * 0.35)));
        let head = Mesh::cone_directed(
            Point3::new(0.0, 0.0, 0.0) + dir * 1.0,
            -dir,
            0.12,
            0.3,
            16,
            true,
            PrimitiveType::TriangleList,
        );
        let nodes = vec![
            add_part(&mut guard, shaft, &material, format!("{name} shaft")),
            add_part(&mut guard, head, &material, format!("{name} head")),
        ];
        // Looking down the axis; Y-axis views need a different up vector.
        let up = if dir.y.abs() > 0.5 { Vector3::unit_z() } else { Vector3::unit_y() };
        result.push(TriadAxis { nodes, direction: dir, up });
    }

    let hub_material = guard.add_face_material(
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor { r: 0.7, g: 0.7, b: 0.7, a: 1.0 })
            .with_flags(MaterialFlags::DO_NOT_LIGHT),
    );
    add_part(
        &mut guard,
        Mesh::sphere(0.12, 16, 8, PrimitiveType::TriangleList),
        &hub_material,
        "Hub".to_string(),
    );

    result
}

/// Operator on the triad view: clicking an axis arrow orients the iso view's
/// camera to look down that axis, refit to the model. Cross-view control needs
/// only the target view's scene handle and camera node.
struct TriadOperator {
    axes: Vec<TriadAxis>,
    iso_scene: Scene,
    iso_camera: NodeId,
}

impl Operator for TriadOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DE::MouseClick { button: MouseButton::Left, position, .. }) = event
        else {
            return false;
        };

        let camera = ctx.camera();
        let ray = camera.ray_from_screen_point(position.0, position.1, ctx.size.0, ctx.size.1);
        let query = RayPickQuery::for_kinds(ray, 0.0, true, false, false);
        let results = pick_all_from_ray(&query, &ctx.scene);
        let Some(hit) = results.first() else { return false };
        let Some(axis) = self.axes.iter().find(|a| a.nodes.contains(&hit.node_id)) else {
            return false;
        };

        let bounds = self.iso_scene.lock().bounding().bounds;
        let direction = axis.direction;
        let up = axis.up;
        self.iso_scene.with_camera_for_node_mut(self.iso_camera, 1.0, |camera| {
            let distance = (camera.eye - camera.target).magnitude();
            camera.eye = camera.target + direction * distance;
            camera.up = up;
            if let Some(bounds) = &bounds {
                camera.fit_to_bounds(bounds);
            }
        });
        true
    }

    fn name(&self) -> &str {
        "Triad"
    }
}

struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<SurfacedViewer<'a>>,
    iso_view: Option<ViewId>,
    triad_view: Option<ViewId>,
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
                .push_back(Arc::new(Mutex::new(SelectionOperator::new())));
            view.dispatcher_mut()
                .push_back(Arc::new(Mutex::new(NavigationOperator::new())));
            view.with_camera_mut(|camera| {
                camera.target = Point3::new(0.0, 0.0, 0.0);
                camera.eye = Point3::new(0.0, 0.0, 0.0) + direction * 10.0;
                camera.up = up;
                camera.ortho = ortho;
                if let Some(bounds) = &bounds {
                    camera.fit_to_bounds(bounds);
                }
            });
            if name == "Iso" {
                iso_view = Some(id);
            }
        }
        let iso_view = iso_view.unwrap();

        // The triad: its own tiny scene in an anchored view overlapping the iso
        // pane, composited over it thanks to the zero-alpha background.
        let triad_scene = Scene::default();
        let axes = build_triad_scene(&triad_scene);
        let triad_view = viewer.add_view(
            "Triad",
            triad_scene,
            ViewLayout::Anchored { corner: Corner::TopRight, size: (140, 140), margin: (16, 16) },
        );
        {
            let iso_camera = viewer.view(iso_view).unwrap().camera_node();
            let mut view = viewer.view_mut(triad_view).unwrap();
            view.set_background_color(RgbaColor { r: 0.0, g: 0.0, b: 0.0, a: 0.0 });
            view.dispatcher_mut().push_back(Arc::new(Mutex::new(TriadOperator {
                axes,
                iso_scene: model,
                iso_camera,
            })));
        }

        window.request_redraw();
        self.window = Some(window);
        self.viewer = Some(viewer);
        self.iso_view = Some(iso_view);
        self.triad_view = Some(triad_view);
    }

    /// Point the triad camera the way the iso pane looks, so the arrows always
    /// mirror the iso orientation.
    fn sync_triad_camera(viewer: &mut Viewer, iso_view: ViewId, triad_view: ViewId) {
        let Some(iso_camera) = viewer.view(iso_view).and_then(|v| v.camera()) else { return };
        let direction = (iso_camera.eye - iso_camera.target).normalize();
        let up = iso_camera.up;
        if let Some(mut triad) = viewer.view_mut(triad_view) {
            triad.with_camera_mut(|camera| {
                camera.target = Point3::new(0.0, 0.0, 0.0);
                camera.eye = Point3::new(0.0, 0.0, 0.0) + direction * TRIAD_CAMERA_DISTANCE;
                camera.up = up;
            });
        }
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
                Self::sync_triad_camera(
                    viewer,
                    self.iso_view.unwrap(),
                    self.triad_view.unwrap(),
                );
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

    let mut app = App {
        window: None,
        viewer: None,
        iso_view: None,
        triad_view: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
