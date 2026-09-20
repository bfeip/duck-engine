use duck_engine_viewer::common::{
    Deg, Euler, Point3, Quaternion, Rad, RgbaColor, Rotation3, Transform,
};
use duck_engine_viewer::scene::{Light, LightType, PositionedLight};
use duck_engine_viewer::ViewMut;

use super::UiActions;

pub fn show(ui: &mut egui::Ui, view: &mut ViewMut<'_>, _actions: &mut UiActions) {
    let mut to_add: Option<LightType> = None;
    ui.horizontal(|ui| {
        ui.label("Add:");
        if ui.button("Point").clicked() {
            to_add = Some(LightType::Point);
        }
        if ui.button("Dir").clicked() {
            to_add = Some(LightType::Directional);
        }
        if ui.button("Spot").clicked() {
            to_add = Some(LightType::Spot);
        }
        if ui.button("Hemi").clicked() {
            to_add = Some(LightType::Hemisphere);
        }
    });

    let lights = view.scene_lights_mut();
    if let Some(light_type) = to_add {
        lights.push(new_light(light_type));
    }

    ui.label(format!("({} lights)", lights.len()));
    ui.separator();

    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            if lights.is_empty() {
                ui.label("No lights in scene");
                return;
            }

            let mut to_delete: Option<usize> = None;
            for (i, light) in lights.iter_mut().enumerate() {
                if build_light_editor(ui, i, light) {
                    to_delete = Some(i);
                }
                ui.separator();
            }
            if let Some(i) = to_delete {
                lights.remove(i);
            }
        });
}

/// A new light of `light_type`, posed where it will be visible in a default scene.
fn new_light(light_type: LightType) -> PositionedLight {
    let white = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    let overhead = Transform::from_position(Point3::new(0.0, 3.0, 0.0));
    match light_type {
        LightType::Point => PositionedLight::world(Light::point(white, 1.0), overhead),
        LightType::Directional => {
            PositionedLight::world(Light::directional(white, 1.0), Transform::IDENTITY)
        }
        LightType::Spot => PositionedLight::world(
            Light::spot(white, 1.0, 30.0_f32.to_radians(), 45.0_f32.to_radians()),
            overhead,
        ),
        // Identity points -Z, so rotate the sky axis onto +Y.
        LightType::Hemisphere => PositionedLight::world(
            Light::hemisphere(
                RgbaColor { r: 0.16, g: 0.19, b: 0.24, a: 1.0 },
                RgbaColor { r: 0.045, g: 0.042, b: 0.040, a: 1.0 },
                1.0,
            ),
            Transform::from_rotation(Quaternion::from_angle_x(Deg(-90.0))),
        ),
    }
}

/// Build editor UI for a single light. Returns whether deletion was requested.
fn build_light_editor(ui: &mut egui::Ui, index: usize, posed: &mut PositionedLight) -> bool {
    let mut delete_requested = false;

    let light = &mut posed.light;
    let light_type_name = match light.light_type() {
        LightType::Point => "Point",
        LightType::Directional => "Directional",
        LightType::Spot => "Spot",
        LightType::Hemisphere => "Hemisphere",
    };

    let header_id = ui.make_persistent_id(format!("light_{index}"));

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
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_range_edit(ui, range);
                }
                Light::Directional { color, intensity, .. } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                }
                Light::Spot {
                    color,
                    intensity,
                    range,
                    inner_cone_angle,
                    outer_cone_angle,
                    ..
                } => {
                    build_color_edit(ui, color);
                    build_intensity_edit(ui, intensity);
                    build_range_edit(ui, range);
                    build_cone_angles_edit(ui, inner_cone_angle, outer_cone_angle);
                }
                Light::Hemisphere { sky_color, ground_color, intensity } => {
                    ui.label("Sky");
                    build_color_edit(ui, sky_color);
                    ui.label("Ground");
                    build_color_edit(ui, ground_color);
                    build_intensity_edit(ui, intensity);
                }
            }
            ui.separator();
            build_pose_edit(ui, &mut posed.transform);
        });

    delete_requested
}

/// Position and orientation. The transform's -Z axis is the light direction,
/// so the rotation matters for every type but Point.
fn build_pose_edit(ui: &mut egui::Ui, transform: &mut Transform) {
    ui.horizontal(|ui| {
        ui.label("Position:");
        ui.add(egui::DragValue::new(&mut transform.position.x).speed(0.1).prefix("x "));
        ui.add(egui::DragValue::new(&mut transform.position.y).speed(0.1).prefix("y "));
        ui.add(egui::DragValue::new(&mut transform.position.z).speed(0.1).prefix("z "));
    });

    let euler = Euler::from(transform.rotation);
    let mut degrees = [
        Deg::from(Rad(euler.x.0)).0,
        Deg::from(Rad(euler.y.0)).0,
        Deg::from(Rad(euler.z.0)).0,
    ];
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Rotation:");
        for value in &mut degrees {
            changed |= ui
                .add(egui::DragValue::new(value).speed(1.0).suffix("°"))
                .changed();
        }
    });
    if changed {
        transform.rotation = Quaternion::from(Euler::new(
            Rad::from(Deg(degrees[0])),
            Rad::from(Deg(degrees[1])),
            Rad::from(Deg(degrees[2])),
        ));
    }
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
