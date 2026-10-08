//! The lights tab: the view's camera-following headlights and the lights
//! placed in the scene.

use duck_engine_viewer::common::{
    Deg, EuclideanSpace, Point3, Quaternion, Rad, Real, RgbaColor, Rotation, Rotation3, Transform,
    Vector3,
};
use duck_engine_viewer::scene::{
    Light, LightSpace, LightType, MAX_LIGHTS, PositionedCamera, PositionedLight,
};
use duck_engine_viewer::{HeadlightMode, ViewMut, default_headlight_rig};

use crate::AppAction;

/// A copy of a view's lighting for the UI to edit, written back on
/// [`AppAction::LightingChanged`].
#[derive(Clone)]
pub struct Lighting {
    pub headlight: HeadlightMode,
    pub headlight_rig: Vec<PositionedLight>,
    /// The lights of the view's scene.
    pub lights: Vec<PositionedLight>,
}

impl Lighting {
    /// The lighting `view` renders with.
    pub fn of(view: &ViewMut<'_>) -> Self {
        Self {
            headlight: view.headlight(),
            headlight_rig: view.headlight_rig().to_vec(),
            lights: view.scene_lights().to_vec(),
        }
    }

    /// Write this lighting back to `view`.
    pub fn apply(&self, view: &mut ViewMut<'_>) {
        view.set_headlight(self.headlight);
        view.set_headlight_rig(self.headlight_rig.clone());
        *view.scene_lights_mut() = self.lights.clone();
    }

    /// Whether the headlight rig renders. The modeler gives its view no lights
    /// of its own and no environment map, so under Auto only scene lights turn
    /// the rig off.
    fn rig_active(&self) -> bool {
        match self.headlight {
            HeadlightMode::On => true,
            HeadlightMode::Off => false,
            HeadlightMode::Auto => self.lights.is_empty(),
        }
    }
}

pub fn show(
    ui: &mut egui::Ui,
    lighting: &mut Lighting,
    camera: &PositionedCamera,
    actions: &mut Vec<AppAction>,
) {
    let mut changed = false;
    egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        egui::CollapsingHeader::new("Headlights")
            .default_open(true)
            .show(ui, |ui| changed |= headlights_ui(ui, lighting, camera));

        egui::CollapsingHeader::new(format!("Scene lights ({}/{MAX_LIGHTS})", lighting.lights.len()))
            .id_salt("scene_lights")
            .default_open(true)
            .show(ui, |ui| changed |= scene_lights_ui(ui, lighting, camera));
    });
    if changed {
        actions.push(AppAction::LightingChanged);
    }
}

/// Headlight mode and the rig itself. Returns true when anything changed.
fn headlights_ui(ui: &mut egui::Ui, lighting: &mut Lighting, camera: &PositionedCamera) -> bool {
    let mut changed = false;
    egui::Grid::new("headlight_settings").num_columns(2).show(ui, |ui| {
        ui.label("Mode");
        egui::ComboBox::from_id_salt("headlight_mode")
            .selected_text(mode_label(lighting.headlight))
            .show_ui(ui, |ui| {
                for mode in [HeadlightMode::Auto, HeadlightMode::On, HeadlightMode::Off] {
                    changed |= ui
                        .selectable_value(&mut lighting.headlight, mode, mode_label(mode))
                        .changed();
                }
            });
        ui.end_row();
    });
    if lighting.headlight == HeadlightMode::Auto {
        ui.weak(if lighting.rig_active() {
            "On while the scene has no lights"
        } else {
            "Off while the scene has lights"
        });
    }

    changed |= add_light_ui(ui, &mut lighting.headlight_rig, LightSpace::Camera, camera);
    if lighting.headlight_rig.is_empty() {
        ui.weak("No headlights");
    }
    changed |= light_list_ui(ui, "headlight_rig", &mut lighting.headlight_rig, camera, false);
    if ui.button("Reset").on_hover_text("Restore the default headlights").clicked() {
        lighting.headlight_rig = default_headlight_rig();
        changed = true;
    }
    changed
}

fn mode_label(mode: HeadlightMode) -> &'static str {
    match mode {
        HeadlightMode::Auto => "Auto",
        HeadlightMode::On => "On",
        HeadlightMode::Off => "Off",
    }
}

/// Add buttons and the scene's lights. Returns true when anything changed.
fn scene_lights_ui(ui: &mut egui::Ui, lighting: &mut Lighting, camera: &PositionedCamera) -> bool {
    let mut changed = add_light_ui(ui, &mut lighting.lights, LightSpace::World, camera);

    // The rig follows the scene's lights, so it is what the cap drops.
    let rig = if lighting.rig_active() { lighting.headlight_rig.len() } else { 0 };
    let dropped = (lighting.lights.len() + rig).saturating_sub(MAX_LIGHTS);
    if dropped > 0 {
        ui.colored_label(
            ui.visuals().warn_fg_color,
            format!("Only {MAX_LIGHTS} lights render: {dropped} headlights are dropped"),
        );
    }

    if lighting.lights.is_empty() {
        ui.weak("No lights in the scene");
    }
    changed |= light_list_ui(ui, "scene_lights", &mut lighting.lights, camera, true);
    changed
}

/// A button per light type adding a new light to `lights` in `space`, disabled
/// once `lights` alone fills [`MAX_LIGHTS`]. Returns true when one was added.
fn add_light_ui(
    ui: &mut egui::Ui,
    lights: &mut Vec<PositionedLight>,
    space: LightSpace,
    camera: &PositionedCamera,
) -> bool {
    let mut added = false;
    let full = lights.len() >= MAX_LIGHTS;
    ui.horizontal_wrapped(|ui| {
        ui.label("Add");
        for light_type in
            [LightType::Directional, LightType::Point, LightType::Spot, LightType::Hemisphere]
        {
            if ui.add_enabled(!full, egui::Button::new(type_label(light_type))).clicked() {
                lights.push(new_light(light_type, space, camera));
                added = true;
            }
        }
    });
    added
}

fn type_label(light_type: LightType) -> &'static str {
    match light_type {
        LightType::Point => "Point",
        LightType::Directional => "Directional",
        LightType::Spot => "Spot",
        LightType::Hemisphere => "Hemisphere",
    }
}

/// A new light of `light_type` in `space`, posed to light what the camera
/// sees.
fn new_light(
    light_type: LightType,
    space: LightSpace,
    camera: &PositionedCamera,
) -> PositionedLight {
    let white = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
    let (light, transform) = match light_type {
        LightType::Point => (Light::point(white, 5.0), view_pose(space, camera)),
        LightType::Spot => (
            Light::spot(white, 5.0, 20.0_f32.to_radians(), 30.0_f32.to_radians()),
            view_pose(space, camera),
        ),
        LightType::Directional => (
            Light::directional(white, 5.0),
            Transform::from_rotation(rotation_from(45.0, 45.0)),
        ),
        LightType::Hemisphere => (
            Light::hemisphere(
                RgbaColor { r: 0.16, g: 0.19, b: 0.24, a: 1.0 },
                RgbaColor { r: 0.045, g: 0.042, b: 0.040, a: 1.0 },
                0.9,
            ),
            Transform::from_rotation(rotation_from(0.0, 90.0)),
        ),
    };
    PositionedLight { light, transform, space }
}

/// One collapsible editor per light, with From view and remove buttons in its
/// header. `show_space` offers the World/Camera toggle. Returns true when
/// anything changed.
fn light_list_ui(
    ui: &mut egui::Ui,
    id_salt: &str,
    lights: &mut Vec<PositionedLight>,
    camera: &PositionedCamera,
    show_space: bool,
) -> bool {
    let mut changed = false;
    let mut to_remove = None;
    for (index, posed) in lights.iter_mut().enumerate() {
        ui.push_id((id_salt, index), |ui| {
            let id = ui.make_persistent_id("light");
            let state =
                egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
            let mut title_clicked = false;
            let mut header = state.show_header(ui, |ui| {
                // The title toggles too, as a collapsing header's does.
                let title = format!("{} {}", type_label(posed.light.light_type()), index + 1);
                title_clicked = ui
                    .add(egui::Label::new(title).selectable(false).sense(egui::Sense::click()))
                    .clicked();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.small_button("🗑").on_hover_text("Remove").clicked() {
                        to_remove = Some(index);
                    }
                    // A hemisphere has no position, and aiming its sky
                    // along the view is never wanted.
                    if posed.light.light_type() != LightType::Hemisphere
                        && ui
                            .small_button("From view")
                            .on_hover_text("Move to the camera and aim along the view")
                            .clicked()
                    {
                        posed.transform = view_pose(posed.space, camera);
                        changed = true;
                    }
                });
            });
            if title_clicked {
                header.toggle();
            }
            header.body(|ui| changed |= light_ui(ui, posed, camera, show_space));
        });
    }
    if let Some(index) = to_remove {
        lights.remove(index);
        changed = true;
    }
    changed
}

/// The parameters of one light. Returns true when anything changed.
fn light_ui(
    ui: &mut egui::Ui,
    posed: &mut PositionedLight,
    camera: &PositionedCamera,
    show_space: bool,
) -> bool {
    let mut changed = false;
    egui::Grid::new("light_settings").num_columns(2).show(ui, |ui| {
        match &mut posed.light {
            Light::Point { color, intensity, range } => {
                changed |= color_row(ui, "Color", color);
                changed |= intensity_row(ui, intensity);
                changed |= range_row(ui, range);
            }
            Light::Directional { color, intensity } => {
                changed |= color_row(ui, "Color", color);
                changed |= intensity_row(ui, intensity);
            }
            Light::Spot { color, intensity, range, inner_cone_angle, outer_cone_angle } => {
                changed |= color_row(ui, "Color", color);
                changed |= intensity_row(ui, intensity);
                changed |= range_row(ui, range);
                changed |= cone_rows(ui, inner_cone_angle, outer_cone_angle);
            }
            Light::Hemisphere { sky_color, ground_color, intensity } => {
                changed |= color_row(ui, "Sky", sky_color);
                changed |= color_row(ui, "Ground", ground_color);
                changed |= intensity_row(ui, intensity);
            }
        }

        if show_space {
            changed |= space_row(ui, posed, camera);
        }
        let light_type = posed.light.light_type();
        if matches!(light_type, LightType::Point | LightType::Spot) {
            changed |= position_row(ui, &mut posed.transform.position);
        }
        match light_type {
            LightType::Point => {}
            LightType::Hemisphere => {
                changed |= direction_rows(ui, &mut posed.transform.rotation, "Where the sky is")
            }
            _ => {
                changed |= direction_rows(
                    ui,
                    &mut posed.transform.rotation,
                    "Where the light comes from",
                )
            }
        }
    });
    changed
}

fn color_row(ui: &mut egui::Ui, label: &str, color: &mut RgbaColor) -> bool {
    ui.label(label);
    let mut rgb = [color.r, color.g, color.b];
    let changed = ui.color_edit_button_rgb(&mut rgb).changed();
    if changed {
        [color.r, color.g, color.b] = rgb;
    }
    ui.end_row();
    changed
}

fn intensity_row(ui: &mut egui::Ui, intensity: &mut f32) -> bool {
    ui.label("Intensity");
    let changed =
        ui.add(egui::DragValue::new(intensity).speed(0.1).range(0.0..=100.0)).changed();
    ui.end_row();
    changed
}

fn range_row(ui: &mut egui::Ui, range: &mut f32) -> bool {
    ui.label("Range").on_hover_text("Distance at which the light fades out; 0 for no falloff");
    let changed = ui.add(egui::DragValue::new(range).speed(1.0).range(0.0..=f32::MAX)).changed();
    ui.end_row();
    changed
}

/// Inner and outer cone angles in degrees, each pushing the other so the
/// inner cone never exceeds the outer.
fn cone_rows(ui: &mut egui::Ui, inner: &mut f32, outer: &mut f32) -> bool {
    let mut changed = false;
    let mut inner_deg = inner.to_degrees();
    ui.label("Inner cone");
    if ui
        .add(egui::DragValue::new(&mut inner_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
        .changed()
    {
        *inner = inner_deg.to_radians();
        *outer = outer.max(*inner);
        changed = true;
    }
    ui.end_row();

    let mut outer_deg = outer.to_degrees();
    ui.label("Outer cone");
    if ui
        .add(egui::DragValue::new(&mut outer_deg).speed(1.0).range(0.0..=90.0).suffix("°"))
        .changed()
    {
        *outer = outer_deg.to_radians();
        *inner = inner.min(*outer);
        changed = true;
    }
    ui.end_row();
    changed
}

/// World or camera space. Switching keeps the light where it is as seen from
/// `camera` right now.
fn space_row(ui: &mut egui::Ui, posed: &mut PositionedLight, camera: &PositionedCamera) -> bool {
    let mut changed = false;
    ui.label("Space");
    ui.horizontal(|ui| {
        let spaces = [
            ("World", LightSpace::World, "Stays put in the scene"),
            ("Camera", LightSpace::Camera, "Follows the camera, like the headlights"),
        ];
        for (label, space, hover) in spaces {
            let selected = posed.space == space;
            if ui.selectable_label(selected, label).on_hover_text(hover).clicked() && !selected {
                posed.transform = respace(&posed.transform, posed.space, space, camera);
                posed.space = space;
                changed = true;
            }
        }
    });
    ui.end_row();
    changed
}

fn position_row(ui: &mut egui::Ui, position: &mut Point3) -> bool {
    let mut changed = false;
    ui.label("Position");
    ui.horizontal(|ui| {
        for (axis, value) in [("x ", &mut position.x), ("y ", &mut position.y), ("z ", &mut position.z)] {
            changed |= ui.add(egui::DragValue::new(value).speed(1.0).prefix(axis)).changed();
        }
    });
    ui.end_row();
    changed
}

/// Azimuth and elevation of the direction the light faces away from.
fn direction_rows(ui: &mut egui::Ui, rotation: &mut Quaternion, hover: &str) -> bool {
    let (mut azimuth, mut elevation) = from_angles(*rotation);
    let mut changed = false;
    ui.label("Azimuth").on_hover_text(hover);
    changed |= ui
        .add(egui::DragValue::new(&mut azimuth).speed(1.0).max_decimals(1).suffix("°"))
        .changed();
    ui.end_row();

    ui.label("Elevation").on_hover_text(hover);
    changed |= ui
        .add(
            egui::DragValue::new(&mut elevation)
                .speed(1.0)
                .range(-90.0..=90.0)
                .max_decimals(1)
                .suffix("°"),
        )
        .changed();
    ui.end_row();

    if changed {
        *rotation = rotation_from(azimuth, elevation);
    }
    changed
}

/// Azimuth and elevation, in degrees, of the direction a light posed by
/// `rotation` comes from: azimuth about +Y from +Z toward +X, elevation above
/// the XZ plane. Roll is lost, as no light depends on it.
fn from_angles(rotation: Quaternion) -> (Real, Real) {
    // The light shines along -Z, so it comes from +Z.
    let from = rotation * Vector3::unit_z();
    let elevation = from.y.clamp(-1.0, 1.0).asin();
    let azimuth = from.x.atan2(from.z);
    (Deg::from(Rad(azimuth)).0, Deg::from(Rad(elevation)).0)
}

/// The rotation of a light coming from `azimuth` and `elevation` (degrees);
/// the inverse of [`from_angles`].
fn rotation_from(azimuth: Real, elevation: Real) -> Quaternion {
    Quaternion::from_angle_y(Deg(azimuth)) * Quaternion::from_angle_x(Deg(-elevation))
}

/// A pose at the camera's eye aimed along its view, in `space`.
fn view_pose(space: LightSpace, camera: &PositionedCamera) -> Transform {
    if space == LightSpace::Camera {
        Transform::IDENTITY
    } else {
        camera.pose_transform()
    }
}

/// `transform` re-expressed from space `from` to space `to` without moving the
/// light as `camera` sees it. Only World and Camera convert.
fn respace(
    transform: &Transform,
    from: LightSpace,
    to: LightSpace,
    camera: &PositionedCamera,
) -> Transform {
    let pose = camera.pose_transform();
    match (from, to) {
        (LightSpace::Camera, LightSpace::World) => compose(&pose, transform),
        (LightSpace::World, LightSpace::Camera) => relative_to(&pose, transform),
        _ => *transform,
    }
}

/// `local`, posed relative to `frame`, in the space `frame` is given in.
/// Ignores `frame`'s scale.
fn compose(frame: &Transform, local: &Transform) -> Transform {
    Transform::new(
        frame.position + frame.rotation * local.position.to_vec(),
        frame.rotation * local.rotation,
        local.scale,
    )
}

/// `transform` re-expressed relative to `frame`; the inverse of [`compose`].
fn relative_to(frame: &Transform, transform: &Transform) -> Transform {
    let inverse = frame.rotation.invert();
    Transform::new(
        Point3::from_vec(inverse * (transform.position - frame.position)),
        inverse * transform.rotation,
        transform.scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_viewer::common::InnerSpace;
    use duck_engine_viewer::scene::Projection;

    const EPSILON: Real = 1e-6;

    fn camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(75.0, 50.0, 75.0),
            target: Point3::new(10.0, 0.0, -5.0),
            up: Vector3::unit_y(),
            aspect: 1.5,
            projection: Projection::Orthographic { half_height: 35.0, half_depth: 1000.0 },
        }
    }

    /// The direction a light posed by `transform` shines in.
    fn shine(transform: &Transform) -> Vector3 {
        transform.rotation * -Vector3::unit_z()
    }

    fn assert_close(a: Vector3, b: Vector3) {
        assert!((a - b).magnitude() < EPSILON, "{a:?} != {b:?}");
    }

    #[test]
    fn angles_round_trip_through_a_rotation() {
        for (azimuth, elevation) in
            [(0.0, 0.0), (45.0, 45.0), (-120.0, 30.0), (170.0, -60.0), (90.0, 0.0)]
        {
            let (a, e) = from_angles(rotation_from(azimuth, elevation));
            assert!((a - azimuth).abs() < EPSILON, "azimuth {a} != {azimuth}");
            assert!((e - elevation).abs() < EPSILON, "elevation {e} != {elevation}");
        }
    }

    #[test]
    fn straight_up_and_down_keep_their_elevation() {
        for elevation in [90.0, -90.0] {
            let rotation = rotation_from(30.0, elevation);
            let (_, e) = from_angles(rotation);
            assert!((e - elevation).abs() < EPSILON, "elevation {e} != {elevation}");
            let sign = elevation.signum();
            assert_close(shine(&Transform::from_rotation(rotation)), -sign * Vector3::unit_y());
        }
    }

    #[test]
    fn angles_read_any_rotation_by_its_direction() {
        let rotation = Quaternion::from_angle_x(Deg(-40.0)) * Quaternion::from_angle_y(Deg(-30.0));
        let (a, e) = from_angles(rotation);
        let rebuilt = Transform::from_rotation(rotation_from(a, e));
        assert_close(shine(&rebuilt), shine(&Transform::from_rotation(rotation)));
    }

    #[test]
    fn respacing_keeps_the_light_in_place() {
        let camera = camera();
        let local = Transform::new(
            Point3::new(3.0, -2.0, 5.0),
            rotation_from(-35.0, 20.0),
            Vector3::new(1.0, 1.0, 1.0),
        );

        let world = respace(&local, LightSpace::Camera, LightSpace::World, &camera);
        // The renderer resolves a camera-space light as pose · local.
        let resolved = camera.pose_transform().to_matrix() * local.to_matrix();
        assert_close(world.position.to_vec(), resolved.w.truncate());
        assert_close(shine(&world), -resolved.z.truncate().normalize());

        let back = respace(&world, LightSpace::World, LightSpace::Camera, &camera);
        assert_close(back.position.to_vec(), local.position.to_vec());
        assert_close(shine(&back), shine(&local));
    }

    #[test]
    fn from_view_aims_along_the_camera() {
        let camera = camera();
        let world = view_pose(LightSpace::World, &camera);
        assert_close(world.position.to_vec(), camera.eye.to_vec());
        assert_close(shine(&world), camera.forward());

        // The identity camera-space pose resolves to the same world pose.
        let local = view_pose(LightSpace::Camera, &camera);
        let resolved = respace(&local, LightSpace::Camera, LightSpace::World, &camera);
        assert_close(resolved.position.to_vec(), camera.eye.to_vec());
        assert_close(shine(&resolved), camera.forward());
    }

    #[test]
    fn new_lights_shine_down_on_the_scene() {
        let camera = camera();
        let directional = new_light(LightType::Directional, LightSpace::World, &camera);
        assert!(shine(&directional.transform).y < 0.0, "a new directional light comes from above");

        // A hemisphere's direction runs from sky to ground.
        let hemisphere = new_light(LightType::Hemisphere, LightSpace::World, &camera);
        assert_close(shine(&hemisphere.transform), -Vector3::unit_y());
    }

    /// Draw one frame of a light list and return the text it painted, with
    /// where.
    fn draw_list(
        ctx: &egui::Context,
        event: Option<egui::Event>,
        lights: &mut Vec<PositionedLight>,
    ) -> Vec<(String, egui::Rect)> {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 600.0))),
            events: event.into_iter().collect(),
            ..Default::default()
        };
        let output = ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                light_list_ui(ui, "lights", lights, &camera(), false);
            });
        });
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::epaint::Shape::Text(text) => Some((
                    text.galley.text().to_owned(),
                    text.galley.rect.translate(text.pos.to_vec2()),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn clicking_a_lights_title_opens_its_editor() {
        let ctx = egui::Context::default();
        ctx.style_mut(|style| style.animation_time = 0.0);
        let mut lights = vec![new_light(LightType::Spot, LightSpace::Camera, &camera())];

        let texts = draw_list(&ctx, None, &mut lights);
        assert!(!texts.iter().any(|(text, _)| text == "Position"), "a light starts closed");
        let (_, title) = texts.iter().find(|(text, _)| text == "Spot 1").expect("the title");
        let at = title.center();
        let press = |pressed| egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        for event in [egui::Event::PointerMoved(at), press(true), press(false)] {
            draw_list(&ctx, Some(event), &mut lights);
        }

        let texts = draw_list(&ctx, None, &mut lights);
        assert!(texts.iter().any(|(text, _)| text == "Position"), "the editor opened");
    }

    #[test]
    fn new_headlights_start_at_the_eye() {
        let camera = camera();
        let spot = new_light(LightType::Spot, LightSpace::Camera, &camera);
        assert_eq!(spot.space, LightSpace::Camera);
        let resolved = respace(&spot.transform, spot.space, LightSpace::World, &camera);
        assert_close(resolved.position.to_vec(), camera.eye.to_vec());
        assert_close(shine(&resolved), camera.forward());
    }
}
