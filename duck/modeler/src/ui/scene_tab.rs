//! The scene tab: settings that belong to the scene rather than any one part —
//! camera projection, construction plane, grid, and snapping.

use duck_engine_viewer::common::{EuclideanSpace, InnerSpace, Plane, Point3, Vector3};
use duck_engine_viewer::scene::resource::NodePayload;

use crate::document::Document;
use crate::operators::ConstructionOptions;
use crate::snap::SnapFlags;
use crate::ui::UiAction;

/// The scene tab, owning the state local to it.
#[derive(Default)]
pub struct SceneTab;

impl SceneTab {
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        document: &mut Document,
        construction: &mut ConstructionOptions,
        actions: &mut Vec<UiAction>,
    ) {
        egui::CollapsingHeader::new("Camera")
            .default_open(true)
            .show(ui, |ui| camera_ui(ui, document));

        egui::CollapsingHeader::new("Construction plane")
            .default_open(true)
            .show(ui, |ui| {
                if plane_ui(ui, construction) {
                    actions.push(UiAction::ConstructionChanged);
                }
            });

        egui::CollapsingHeader::new("Grid")
            .default_open(true)
            .show(ui, |ui| {
                if grid_ui(ui, construction) {
                    actions.push(UiAction::ConstructionChanged);
                }
            });

        egui::CollapsingHeader::new("Snap")
            .default_open(true)
            .show(ui, |ui| snap_ui(ui, construction));
    }
}

/// Projection controls for the active camera node. Pose is left to the
/// navigation operator.
fn camera_ui(ui: &mut egui::Ui, document: &mut Document) {
    let scene = document.scene().clone();

    let projection = scene.active_camera().and_then(|node_id| {
        let node = scene.get_node(node_id)?;
        match node.payload() {
            NodePayload::Camera(projection) => Some((node_id, projection.clone())),
            _ => None,
        }
    });
    let Some((node_id, mut projection)) = projection else {
        ui.weak("No active camera");
        return;
    };

    let mut changed = false;
    egui::Grid::new("camera_settings").num_columns(2).show(ui, |ui| {
        ui.label("Field of view");
        changed |= ui
            .add(egui::Slider::new(&mut projection.fovy, 10.0..=120.0).suffix("°"))
            .changed();
        ui.end_row();

        let (znear, zfar) = (projection.znear, projection.zfar);
        ui.label("Near clip");
        changed |= ui
            .add(egui::DragValue::new(&mut projection.znear).speed(0.1).range(0.001..=zfar - 0.001))
            .changed();
        ui.end_row();

        ui.label("Far clip");
        changed |= ui
            .add(egui::DragValue::new(&mut projection.zfar).speed(10.0).range(znear + 0.001..=f32::MAX))
            .changed();
        ui.end_row();

        ui.label("Projection");
        changed |= ui.checkbox(&mut projection.ortho, "Orthographic").changed();
        ui.end_row();
    });

    if changed {
        scene.set_node_payload(node_id, NodePayload::Camera(projection));
    }
}

/// Axis-aligned plane presets plus an offset along the plane normal.
/// Returns true when the plane changed (grid must be rebuilt).
fn plane_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) -> bool {
    let mut changed = false;
    // The plane convention is normal·p + d = 0, so the offset of the plane
    // along its own normal is -d.
    let mut offset = -construction.construction_plane.d;

    ui.horizontal(|ui| {
        let presets: [(&str, Vector3); 3] = [
            ("XY", Vector3::unit_z()),
            ("XZ", Vector3::unit_y()),
            ("YZ", Vector3::unit_x()),
        ];
        for (label, normal) in presets {
            let selected = construction.construction_plane.normal.dot(normal) > 0.999;
            if ui.selectable_label(selected, label).clicked() && !selected {
                construction.construction_plane =
                    Plane::from_point(normal, Point3::origin() + normal * offset);
                changed = true;
            }
        }
    });

    egui::Grid::new("plane_settings").num_columns(2).show(ui, |ui| {
        ui.label("Offset");
        if ui.add(egui::DragValue::new(&mut offset).speed(0.5)).changed() {
            let normal = construction.construction_plane.normal;
            construction.construction_plane =
                Plane::from_point(normal, Point3::origin() + normal * offset);
            changed = true;
        }
        ui.end_row();
    });

    changed
}

/// Grid visibility and dimensions. Returns true when the grid must be rebuilt.
fn grid_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) -> bool {
    let grid = &mut construction.grid;
    let mut changed = false;

    changed |= ui.checkbox(&mut grid.visible, "Show grid").changed();

    egui::Grid::new("grid_settings").num_columns(2).show(ui, |ui| {
        ui.label("Size");
        changed |= ui
            .add(egui::DragValue::new(&mut grid.size).speed(10.0).range(1.0..=f32::MAX))
            .changed();
        ui.end_row();

        ui.label("Minor spacing");
        changed |= ui
            .add(egui::DragValue::new(&mut grid.minor_spacing).speed(0.5).range(0.01..=f32::MAX))
            .changed();
        ui.end_row();

        ui.label("Major every");
        changed |= ui
            .add(egui::DragValue::new(&mut grid.major_every).range(1..=1000))
            .changed();
        ui.end_row();
    });

    changed
}

/// Snap master switch, per-kind toggles, and pixel tolerance. Read live by
/// operators, so no rebuild action is needed.
fn snap_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) {
    let settings = &mut construction.snap.settings;

    ui.checkbox(&mut settings.enabled, "Enable snapping");

    ui.add_enabled_ui(settings.enabled, |ui| {
        let kinds: [(&str, SnapFlags); 7] = [
            ("Origin", SnapFlags::ORIGIN),
            ("Grid guides", SnapFlags::GRID_GUIDE),
            ("Grid axes", SnapFlags::GRID_AXIS),
            ("Faces", SnapFlags::FACE),
            ("Corners", SnapFlags::CORNER),
            ("Edges", SnapFlags::EDGE),
            ("Wire start", SnapFlags::WIRE_START),
        ];
        for (label, flag) in kinds {
            let mut on = settings.enabled_kinds.contains(flag);
            if ui.checkbox(&mut on, label).changed() {
                settings.enabled_kinds.set(flag, on);
            }
        }

        egui::Grid::new("snap_settings").num_columns(2).show(ui, |ui| {
            ui.label("Tolerance");
            ui.add(
                egui::DragValue::new(&mut settings.pixel_tolerance)
                    .speed(1.0)
                    .range(1.0..=100.0)
                    .suffix(" px"),
            );
            ui.end_row();
        });
    });
}
