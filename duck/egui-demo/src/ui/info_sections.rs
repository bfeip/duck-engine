use duck_engine_viewer::event::EventDispatcher;
use duck_engine_viewer::scene::resource::SubGeometryKind;
use duck_engine_viewer::scene::{PositionedCamera, PositionedLight, Projection, Scene};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};
use duck_engine_viewer::{ViewId, Viewer};

pub fn show(ui: &mut egui::Ui, viewer: &Viewer, view_id: ViewId) {
    let view = viewer.view(view_id).expect("main view");
    let scene = view.scene();
    build_camera_section(ui, view.camera());
    ui.separator();
    build_operators_section(ui, view.dispatcher());
    ui.separator();
    build_selection_section(ui, viewer.selection(view_id).expect("main view"), &scene);
    ui.separator();
    build_scene_info_section(ui, &scene, viewer.scene_lights(view_id).expect("main view"));
}

fn build_camera_section(ui: &mut egui::Ui, camera: &PositionedCamera) {
    ui.heading("Camera");

    ui.label(match camera.projection {
        Projection::Perspective { fovy, .. } => format!("Projection: Perspective ({fovy:.1}°)"),
        Projection::Orthographic { half_height, .. } => {
            format!("Projection: Orthographic ({:.2} high)", half_height * 2.0)
        }
    });
    ui.label(format!(
        "Position: ({:.2}, {:.2}, {:.2})",
        camera.eye.x, camera.eye.y, camera.eye.z
    ));
    ui.label(format!(
        "Target: ({:.2}, {:.2}, {:.2})",
        camera.target.x, camera.target.y, camera.target.z
    ));
    let (near, far) = camera.projection.depth_range();
    ui.label(format!("Near: {near:.4}"));
    ui.label(format!("Far: {far:.4}"));
}

fn build_operators_section(ui: &mut egui::Ui, dispatcher: &EventDispatcher) {
    ui.heading("Operators");

    for name in dispatcher.iter_names() {
        ui.label(format!("  {}", name));
    }
}

fn selection_item_label(item: SelectionItem, scene: &Scene) -> String {
    let node_id = item.node_id();
    let node_label = scene
        .get_node(node_id)
        .and_then(|n| n.name.clone())
        .unwrap_or_else(|| format!("Node #{}", node_id));

    match item {
        SelectionItem::Node(_) => node_label,
        SelectionItem::SubGeometry { element, .. } => {
            let kind = match element.kind {
                SubGeometryKind::Face => "Face",
                SubGeometryKind::Edge => "Edge",
                SubGeometryKind::Pointset => "Point",
            };
            format!("{} #{} ({})", kind, element.index, node_label)
        }
    }
}

fn build_selection_section(ui: &mut egui::Ui, selection: &SelectionManager, scene: &Scene) {
    ui.heading("Selection");

    if selection.is_empty() {
        ui.label("(none)");
    } else {
        ui.label(format!("Count: {}", selection.len()));

        if let Some(primary) = selection.primary() {
            ui.label(format!("Primary: {}", selection_item_label(primary, scene)));
        }

        ui.label("Selected:");
        for item in selection.iter() {
            ui.label(format!("  • {}", selection_item_label(*item, scene)));
        }
    }
}

fn build_scene_info_section(ui: &mut egui::Ui, scene: &Scene, scene_lights: &[PositionedLight]) {
    ui.heading("Scene Info");
    ui.label(format!("Meshes: {}", scene.mesh_count()));
    ui.label(format!("Instances: {}", scene.instance_count()));
    ui.label(format!("Nodes: {}", scene.node_count()));
    ui.label(format!("Lights: {}", scene_lights.len()));
}
