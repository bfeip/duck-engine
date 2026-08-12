use duck_engine_viewer::ViewMut;

use super::UiActions;

pub fn show(ui: &mut egui::Ui, view: &mut ViewMut<'_>, actions: &mut UiActions) {
    ui.horizontal(|ui| {
        if ui.button("Load HDR...").clicked() {
            actions.load_environment = true;
        }
        if ui.button("Clear").clicked() {
            actions.clear_environment = true;
        }
    });

    ui.separator();

    let scene_arc = view.scene();
    let env_id = scene_arc.active_environment_map();
    if let Some(env_id) = env_id {
        let env = scene_arc.get_environment_map(env_id);
        let mut intensity = env.as_ref().map_or(1.0, |e| e.intensity());
        let rotation_deg = env.as_ref().map_or(0.0, |e| e.rotation().to_degrees());

        ui.label(format!("Active: Environment #{}", env_id));
        if ui
            .add(egui::Slider::new(&mut intensity, 0.0..=5.0).text("Intensity"))
            .changed()
        {
            scene_arc.set_environment_map_intensity(env_id, intensity);
        }
        ui.label(format!("Rotation: {:.1}°", rotation_deg));
    } else {
        ui.label("No environment map active");
        ui.label("");
        ui.label("Load an HDR file to enable");
        ui.label("image-based lighting (IBL)");
    }
}
