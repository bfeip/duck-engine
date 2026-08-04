use duck_engine_viewer::scene::{EffectiveVisibility, NodeId, NodePayload, Visibility};
use duck_engine_viewer::Viewer;

use super::{UiActions, VisibilityChange};

pub fn show(ui: &mut egui::Ui, viewer: &Viewer, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        if ui.button("Open...").clicked() {
            actions.load_scene = true;
        }
        if ui.button("Clear").clicked() {
            actions.clear_scene = true;
        }
    });

    ui.separator();

    ui.heading("Scene Tree");

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // The scene must not be locked here: this closure builds widgets,
            // and `render_node_tree` recurses through them.
            let scene = viewer.scene();
            let roots = scene.root_nodes();
            if roots.is_empty() {
                ui.label("(empty)");
            } else {
                for root_id in roots {
                    render_node_tree(ui, &scene, root_id, 0, actions);
                }
            }
        });
}

/// Recursively render a node and its children in the scene tree.
pub fn render_node_tree(
    ui: &mut egui::Ui,
    scene: &duck_engine_viewer::scene::Scene,
    node_id: NodeId,
    depth: usize,
    actions: &mut UiActions,
) {
    let Some(node) = scene.get_node(node_id) else {
        return;
    };

    let has_children = !node.children().is_empty();
    let has_instance = matches!(node.payload(), NodePayload::Instance(_));

    let visibility = node.visibility();
    let effective_visibility = scene.node_effective_visibility(node_id);
    let mut is_visible = visibility == Visibility::Visible;
    let is_indeterminate = effective_visibility == EffectiveVisibility::Mixed;

    let label = if let Some(ref name) = node.name {
        name.clone()
    } else if has_instance {
        format!("Instance #{}", node_id)
    } else {
        format!("Node #{}", node_id)
    };

    let icon = if has_instance || has_children { "+" } else { "-" };
    let display_label = format!("{} {}", icon, label);

    let text_alpha = if effective_visibility == EffectiveVisibility::Invisible {
        0.5
    } else {
        1.0
    };

    let children: Vec<NodeId> = node.children().to_vec();

    if has_children {
        let id = ui.make_persistent_id(format!("node_{}", node_id));
        let mut state =
            egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, depth < 2);

        ui.horizontal(|ui| {
            let checkbox =
                egui::Checkbox::without_text(&mut is_visible).indeterminate(is_indeterminate);
            if ui.add(checkbox).changed() {
                actions.visibility_changes.push(VisibilityChange {
                    node_id,
                    new_visibility: if is_visible {
                        Visibility::Visible
                    } else {
                        Visibility::Invisible
                    },
                });
            }

            state.show_toggle_button(ui, egui::collapsing_header::paint_default_icon);

            let text_color = ui.visuals().text_color().gamma_multiply(text_alpha);
            ui.colored_label(text_color, &display_label);
        });

        state.show_body_unindented(ui, |ui| {
            ui.indent(id, |ui| {
                for &child_id in &children {
                    render_node_tree(ui, scene, child_id, depth + 1, actions);
                }
            });
        });
    } else {
        ui.horizontal(|ui| {
            let checkbox =
                egui::Checkbox::without_text(&mut is_visible).indeterminate(is_indeterminate);
            if ui.add(checkbox).changed() {
                actions.visibility_changes.push(VisibilityChange {
                    node_id,
                    new_visibility: if is_visible {
                        Visibility::Visible
                    } else {
                        Visibility::Invisible
                    },
                });
            }

            let text_color = ui.visuals().text_color().gamma_multiply(text_alpha);
            ui.colored_label(text_color, &display_label);
        });
    }
}
