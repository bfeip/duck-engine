use duck_engine_viewer::scene::resource::SubGeometryKind;
use duck_engine_viewer::selection::SelectionItem;
use duck_engine_viewer::Viewer;

pub fn show(ui: &mut egui::Ui, viewer: &Viewer) {
    build_camera_section(ui, viewer);
    ui.separator();
    build_operators_section(ui, viewer);
    ui.separator();
    build_selection_section(ui, viewer);
    ui.separator();
    build_scene_info_section(ui, viewer);
}

fn build_camera_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Camera");

    let camera = viewer.camera();
    ui.label(format!(
        "Projection: {}",
        if camera.ortho { "Orthographic" } else { "Perspective" }
    ));
    ui.label(format!(
        "Position: ({:.2}, {:.2}, {:.2})",
        camera.eye.x, camera.eye.y, camera.eye.z
    ));
    ui.label(format!(
        "Target: ({:.2}, {:.2}, {:.2})",
        camera.target.x, camera.target.y, camera.target.z
    ));
    ui.label(format!("Near: {:.4}", camera.znear));
    ui.label(format!("Far: {:.4}", camera.zfar));
}

fn build_operators_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Operators");

    for name in viewer.dispatcher().iter_names() {
        ui.label(format!("  {}", name));
    }
}

fn selection_item_label(item: SelectionItem, viewer: &Viewer) -> String {
    let node_id = item.node_id();
    let node_label = viewer
        .scene()
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

fn build_selection_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Selection");

    let selection = viewer.selection();

    if selection.is_empty() {
        ui.label("(none)");
    } else {
        ui.label(format!("Count: {}", selection.len()));

        if let Some(primary) = selection.primary() {
            ui.label(format!("Primary: {}", selection_item_label(primary, viewer)));
        }

        ui.label("Selected:");
        for item in selection.iter() {
            ui.label(format!("  • {}", selection_item_label(*item, viewer)));
        }
    }
}

fn build_scene_info_section(ui: &mut egui::Ui, viewer: &Viewer) {
    ui.heading("Scene Info");
    let scene = viewer.scene();
    ui.label(format!("Meshes: {}", scene.mesh_count()));
    ui.label(format!("Instances: {}", scene.instance_count()));
    ui.label(format!("Nodes: {}", scene.node_count()));
    ui.label(format!("Lights: {}", scene.light_count()));
}
