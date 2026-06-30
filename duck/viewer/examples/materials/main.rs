use std::sync::{Arc, Mutex};

use duck_engine_scene::NodeFlags;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use duck_engine_viewer::common::{RgbaColor, Transform, Point3};
use duck_engine_viewer::input::{ElementState, Key};
use duck_engine_viewer::scene::{
    DisplayBehavior, EnvironmentMapId, FaceMaterial, Instance, LineMaterial, MaterialFlags, Mesh,
    NodePayload, PointMaterial, PositionedCamera, PrimitiveType, ViewportRect,
};
use duck_engine_viewer::{SurfacedViewer, Viewer, winit_support};
use duck_engine_viewer::operator::{NavigationOperator, SelectionOperator, TransformMode, TransformOperator};

const SPHERE_RADIUS: f32 = 0.4;
const SPHERE_SEGMENTS: u32 = 32;
const SPHERE_RINGS: u32 = 16;
const COLS: usize = 5;
const ROWS: usize = 7;
const SPACING: f32 = 1.2;

/// Compute the grid position for a (row, col) cell, centered at the origin.
fn grid_position(row: usize, col: usize) -> Point3 {
    let x = (col as f32 - (COLS as f32 - 1.0) / 2.0) * SPACING;
    let z = (row as f32 - (ROWS as f32 - 1.0) / 2.0) * SPACING;
    Point3::new(x, 0.0, z)
}

/// Build the material showcase scene: a 5x5 grid of spheres demonstrating PBR properties.
fn build_material_scene(viewer: &mut Viewer) {
    let scene_arc = viewer.scene();
    let mut scene = scene_arc.lock().unwrap();

    // Attach default camera-space key + fill lights to a placeholder camera node.
    let camera_node = scene.add_node(
        None, Some("Camera".to_string()), Default::default(), NodeFlags::NONE
    ).unwrap();
    scene.set_default_light_nodes(camera_node);

    // Shared sphere mesh for all instances
    let mesh_id = scene.add_mesh(Mesh::sphere(
        SPHERE_RADIUS,
        SPHERE_SEGMENTS,
        SPHERE_RINGS,
        PrimitiveType::TriangleList,
    ));

    // grid_position returns a Point3, wrap it in a Transform for add_instance_node
    let grid_transform = |row, col| Transform::from_position(grid_position(row, col));

    // Row 0: Roughness gradient (dielectric blue, roughness 0.0 -> 1.0)
    for col in 0..COLS {
        let roughness = col as f32 / (COLS - 1) as f32;
        let mat = FaceMaterial::new()
            .with_base_color_factor(RgbaColor::BLUE)
            .with_metallic_factor(0.0)
            .with_roughness_factor(roughness);
        let mat_id = scene.add_face_material(mat);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some(format!("Roughness {roughness:.2}")),
                grid_transform(0, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 1: Metallic gradient (blue, roughness 0.3, metallic 0.0 -> 1.0)
    for col in 0..COLS {
        let metallic = col as f32 / (COLS - 1) as f32;
        let mat = FaceMaterial::new()
            .with_base_color_factor(RgbaColor::BLUE)
            .with_roughness_factor(0.3)
            .with_metallic_factor(metallic);
        let mat_id = scene.add_face_material(mat);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some(format!("Metallic {metallic:.2}")),
                grid_transform(1, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 2: Base colors (dielectric, roughness 0.4)
    let colors = [
        ("Red", RgbaColor::RED),
        ("Green", RgbaColor::GREEN),
        ("Blue", RgbaColor::BLUE),
        ("White", RgbaColor::WHITE),
        ("Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in colors.iter().enumerate() {
        let mat = FaceMaterial::new()
            .with_base_color_factor(*color)
            .with_metallic_factor(0.0)
            .with_roughness_factor(0.4);
        let mat_id = scene.add_face_material(mat);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some(name.to_string()),
                grid_transform(2, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 3: Metallic colors (metallic 1.0, roughness 0.25)
    let metals = [
        ("Red", RgbaColor::RED),
        ("Green", RgbaColor::GREEN),
        ("Blue", RgbaColor::BLUE),
        ("White", RgbaColor::WHITE),
        ("Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in metals.iter().enumerate() {
        let mat = FaceMaterial::new()
            .with_base_color_factor(*color)
            .with_metallic_factor(1.0)
            .with_roughness_factor(0.25);
        let mat_id = scene.add_face_material(mat);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some(name.to_string()),
                grid_transform(3, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 4: Unlit (DO_NOT_LIGHT) vs lit
    let unlit_colors = [
        ("Unlit Red", RgbaColor::RED),
        ("Unlit Green", RgbaColor::GREEN),
        ("Unlit Blue", RgbaColor::BLUE),
        ("Unlit White", RgbaColor::WHITE),
        ("Unlit Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in unlit_colors.iter().enumerate() {
        let mat = FaceMaterial::new()
            .with_base_color_factor(*color)
            .with_flags(MaterialFlags::DO_NOT_LIGHT);
        let mat_id = scene.add_face_material(mat);
        scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(mat_id),
                Some(name.to_string()),
                grid_transform(4, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 5: Wireframe spheres (lines) with various colors
    let line_mesh_id = scene.add_mesh(Mesh::sphere(
        SPHERE_RADIUS,
        SPHERE_SEGMENTS,
        SPHERE_RINGS,
        PrimitiveType::LineList,
    ));
    let line_colors = [
        ("Lines Red", RgbaColor::RED),
        ("Lines Green", RgbaColor::GREEN),
        ("Lines Blue", RgbaColor::BLUE),
        ("Lines White", RgbaColor::WHITE),
        ("Lines Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in line_colors.iter().enumerate() {
        let mat_id = scene.add_line_material(LineMaterial::new(*color));
        scene
            .add_instance_node(
                None,
                Instance::new(line_mesh_id).with_line_material(mat_id),
                Some(name.to_string()),
                grid_transform(5, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Row 6: Point spheres with various colors
    let point_mesh_id = scene.add_mesh(Mesh::sphere(
        SPHERE_RADIUS,
        SPHERE_SEGMENTS,
        SPHERE_RINGS,
        PrimitiveType::PointList,
    ));
    let point_colors = [
        ("Points Red", RgbaColor::RED),
        ("Points Green", RgbaColor::GREEN),
        ("Points Blue", RgbaColor::BLUE),
        ("Points White", RgbaColor::WHITE),
        ("Points Yellow", RgbaColor::YELLOW),
    ];
    for (col, (name, color)) in point_colors.iter().enumerate() {
        let mat_id = scene.add_point_material(PointMaterial::new(*color));
        scene
            .add_instance_node(
                None,
                Instance::new(point_mesh_id).with_point_material(mat_id),
                Some(name.to_string()),
                grid_transform(6, col),
                NodeFlags::NONE
            )
            .unwrap();
    }

    // Screen-space demo: a flat quad above the grid that always faces the camera
    // (screen_facing) and holds a constant on-screen size (screen_size).
    // Orbiting/zooming should keep it the same pixel size and edge-on to the
    // viewer while the spheres rotate and shrink with distance.
    let quad_mesh = scene.add_mesh(Mesh::quad(1.0, 1.6, PrimitiveType::TriangleList));
    let quad_mat = scene.add_face_material(
        FaceMaterial::new()
            .with_base_color_factor(RgbaColor::MAGENTA)
            .with_flags(MaterialFlags::DO_NOT_LIGHT | MaterialFlags::DOUBLE_SIDED),
    );
    let billboard = scene
        .add_instance_node(
            None,
            Instance::new(quad_mesh).with_face_material(quad_mat),
            Some("Billboard".to_string()),
            Transform::from_position(Point3::new(0.0, 2.5, 0.0)),
            NodeFlags::NONE,
        )
        .unwrap();
    scene.set_node_display(
        billboard,
        DisplayBehavior {
            screen_facing: true,
            screen_size: Some(120.0),
            ..Default::default()
        },
    );

    // ---- Sub-view demo --------------------------------------------------
    // A separate subtree (a single red sphere at the origin) rendered into an
    // inset region in the top-right corner, viewed through its own camera. The
    // subtree is drawn *only* in the sub-view, never in the main grid.
    let sub_root = scene
        .add_node(None, Some("SubViewRoot".to_string()), Transform::IDENTITY, NodeFlags::NONE)
        .unwrap();
    let sub_mat = scene.add_face_material(
        FaceMaterial::new().with_base_color_factor(RgbaColor::RED).with_roughness_factor(0.4),
    );
    scene
        .add_instance_node(
            Some(sub_root),
            Instance::new(mesh_id).with_face_material(sub_mat),
            Some("SubViewSphere".to_string()),
            Transform::IDENTITY,
            NodeFlags::NONE,
        )
        .unwrap();

    // A dedicated camera node looking head-on at the sub-view sphere. The aspect
    // is overridden per frame from the sub-view's pixel rectangle.
    let sub_cam = PositionedCamera {
        eye: Point3::new(0.0, 0.5, 2.0),
        target: Point3::new(0.0, 0.0, 0.0),
        up: duck_engine_viewer::common::Vector3::new(0.0, 1.0, 0.0),
        aspect: 1.0,
        fovy: 45.0,
        znear: 0.1,
        zfar: 100.0,
        ortho: false,
    };
    let sub_cam_node = scene
        .add_node(
            None,
            Some("SubViewCamera".to_string()),
            sub_cam.to_node_transform(),
            NodeFlags::NONE,
        )
        .unwrap();
    scene.set_node_payload(sub_cam_node, NodePayload::Camera(sub_cam.projection()));

    // Top-right inset: 24% of the surface, inset slightly from the corner.
    scene.add_sub_view(ViewportRect::new(0.72, 0.04, 0.24, 0.24), sub_root, sub_cam_node);

    // Release the scene lock before re-borrowing the viewer for the camera.
    drop(scene);

    // Camera: elevated view looking down at the grid
    viewer.with_camera_mut(|camera| {
        camera.eye = Point3::new(0.0, 6.0, 8.0);
        camera.target = Point3::new(0.0, 0.0, 0.0);
    });
    let bounds = scene_arc.lock().unwrap().bounding().bounds;
    if let Some(bounds) = bounds {
        viewer.with_camera_mut(|camera| camera.fit_to_bounds(&bounds));
    }
}

/// Application state for the winit event loop.
struct App<'a> {
    window: Option<Arc<Window>>,
    viewer: Option<SurfacedViewer<'a>>,
    env_map_id: Option<EnvironmentMapId>,
}

impl<'a> App<'a> {
    fn initialize(&mut self, event_loop: &ActiveEventLoop) {
        let window_attrs =
            Window::default_attributes().with_title("Duck Engine - PBR Materials Showcase");

        let window = Arc::new(event_loop.create_window(window_attrs).unwrap());
        let size = window.inner_size();
        let mut viewer = pollster::block_on(SurfacedViewer::new(
            Arc::clone(&window),
            size.width,
            size.height,
        ));

        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Translate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Rotate))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(TransformOperator::new(TransformMode::Scale))));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(SelectionOperator::new())));
        viewer.dispatcher_mut().push_back(Arc::new(Mutex::new(NavigationOperator::new())));

        build_material_scene(&mut viewer);

        // Load environment map for IBL (toggled with 'e' key)
        let env_map_path: std::path::PathBuf =
            [env!("CARGO_MANIFEST_DIR"), "..", "..", "assets", "studio_small_09_4k.hdr"].iter().collect();
        let env_map_id = viewer
            .scene()
            .lock()
            .unwrap()
            .add_environment_map_from_hdr_path(env_map_path);
        self.env_map_id = Some(env_map_id);

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

        let Some(app_event) = winit_support::convert_window_event(event) else {
            return;
        };
        let viewer = self.viewer.as_mut().unwrap();
        viewer.handle_event(&app_event);

        let duck_engine_viewer::event::Event::Device(
            duck_engine_viewer::event::DeviceEvent::KeyboardInput { event: key_event, .. },
        ) = &app_event
        else {
            return;
        };
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return;
        }
        match &key_event.logical_key {
            Key::Named(duck_engine_viewer::input::NamedKey::Escape) => event_loop.exit(),
            Key::Character('e') => {
                if let Some(env_id) = self.env_map_id {
                    let scene = viewer.scene();
                    let mut scene = scene.lock().unwrap();
                    let ibl_active = scene.active_environment_map().is_some();
                    if ibl_active {
                        scene.set_active_environment_map(None);
                    } else {
                        scene.set_active_environment_map(Some(env_id));
                    }
                }
            }
            _ => {}
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
        env_map_id: None,
    };

    event_loop.run_app(&mut app).unwrap();
}
