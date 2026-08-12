use duck_engine_viewer::common::RgbaColor;
use duck_engine_viewer::scene::{Light, LightType};
use duck_engine_viewer::scene::resource::{NodeId, NodePayload};
use duck_engine_viewer::ViewMut;

use super::UiActions;

pub fn show(ui: &mut egui::Ui, view: &mut ViewMut<'_>, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        ui.label("Add:");
        if ui.button("Point").clicked() {
            actions.add_light = Some(LightType::Point);
        }
        if ui.button("Dir").clicked() {
            actions.add_light = Some(LightType::Directional);
        }
        if ui.button("Spot").clicked() {
            actions.add_light = Some(LightType::Spot);
        }
    });

    let scene_arc = view.scene();
    let light_nodes: Vec<(NodeId, Light)> = scene_arc.light_nodes();

    ui.label(format!("({} lights)", light_nodes.len()));
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if light_nodes.is_empty() {
                ui.label("No lights in scene");
            } else {
                let mut to_delete: Option<NodeId> = None;
                let mut to_update: Option<(NodeId, Light)> = None;

                for (i, (node_id, mut light)) in light_nodes.into_iter().enumerate() {
                    let (deleted, updated) = build_light_editor(ui, i, node_id, &mut light);
                    if deleted {
                        to_delete = Some(node_id);
                    }
                    if updated {
                        to_update = Some((node_id, light));
                    }
                    ui.separator();
                }

                if let Some(id) = to_delete {
                    scene_arc.remove_node(id);
                }
                if let Some((id, light)) = to_update {
                    scene_arc.set_node_payload(id, NodePayload::Light(light));
                }
            }
        });
}

/// Build editor UI for a single light. Returns (delete_requested, was_modified).
fn build_light_editor(
    ui: &mut egui::Ui,
    index: usize,
    node_id: NodeId,
    light: &mut Light,
) -> (bool, bool) {
    let mut delete_requested = false;
    let mut modified = false;

    let light_type_name = match light {
        Light::Point { .. } => "Point",
        Light::Directional { .. } => "Directional",
        Light::Spot { .. } => "Spot",
    };

    let header_id = ui.make_persistent_id(format!("light_{}_{}", index, node_id));

    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), header_id, true)
        .show_header(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("{} #{}", light_type_name, index));
                if ui.small_button("X").clicked() {
                    delete_requested = true;
                }
            });
        })
        .body(|ui| {
            match light {
                Light::Point { color, intensity, range, .. } => {
                    modified |= build_color_edit(ui, color);
                    modified |= build_intensity_edit(ui, intensity);
                    modified |= build_range_edit(ui, range);
                }
                Light::Directional { color, intensity, .. } => {
                    modified |= build_color_edit(ui, color);
                    modified |= build_intensity_edit(ui, intensity);
                }
                Light::Spot {
                    color,
                    intensity,
                    range,
                    inner_cone_angle,
                    outer_cone_angle,
                    ..
                } => {
                    modified |= build_color_edit(ui, color);
                    modified |= build_intensity_edit(ui, intensity);
                    modified |= build_range_edit(ui, range);
                    modified |= build_cone_angles_edit(ui, inner_cone_angle, outer_cone_angle);
                }
            }
        });

    (delete_requested, modified)
}

fn build_color_edit(ui: &mut egui::Ui, color: &mut RgbaColor) -> bool {
    ui.horizontal(|ui| {
        ui.label("Color:");
        let mut rgb = [color.r, color.g, color.b];
        if ui.color_edit_button_rgb(&mut rgb).changed() {
            color.r = rgb[0];
            color.g = rgb[1];
            color.b = rgb[2];
            true
        } else {
            false
        }
    })
    .inner
}

fn build_intensity_edit(ui: &mut egui::Ui, intensity: &mut f32) -> bool {
    ui.horizontal(|ui| {
        ui.label("Intensity:");
        ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=100.0))
            .changed()
    })
    .inner
}

fn build_range_edit(ui: &mut egui::Ui, range: &mut f32) -> bool {
    ui.horizontal(|ui| {
        ui.label("Range:");
        let changed = ui
            .add(egui::DragValue::new(range).speed(0.1).range(0.0..=1000.0))
            .changed();
        if *range == 0.0 {
            ui.label("(infinite)");
        }
        changed
    })
    .inner
}

fn build_cone_angles_edit(ui: &mut egui::Ui, inner: &mut f32, outer: &mut f32) -> bool {
    let mut inner_deg = inner.to_degrees();
    let mut outer_deg = outer.to_degrees();
    let mut changed = false;

    ui.horizontal(|ui| {
        ui.label("Inner cone:");
        if ui
            .add(egui::DragValue::new(&mut inner_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
            .changed()
        {
            *inner = inner_deg.to_radians();
            if *inner > *outer {
                *outer = *inner;
            }
            changed = true;
        }
    });

    ui.horizontal(|ui| {
        ui.label("Outer cone:");
        if ui
            .add(egui::DragValue::new(&mut outer_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
            .changed()
        {
            *outer = outer_deg.to_radians();
            if *outer < *inner {
                *inner = *outer;
            }
            changed = true;
        }
    });

    changed
}
