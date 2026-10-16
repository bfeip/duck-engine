//! The scene tab: settings that belong to the scene rather than any one part —
//! display units, camera projection, construction plane, grid, and snapping.

use duck_engine_viewer::common::{
    EuclideanSpace, InnerSpace, LengthDisplay, LengthUnit, Plane, Point3, Real, Vector3,
};
use duck_engine_viewer::scene::{PositionedCamera, Projection};

use crate::construction::ConstructionOptions;
use crate::document::DOCUMENT_UNITS;
use crate::snap::SnapFlags;
use crate::ui::length_value;
use crate::AppAction;

/// Closest a clip plane may sit to the eye, and the least gap kept between
/// near and far, in document units: 1 µm.
const MIN_CLIP: Real = 1e-3;

/// Floors on the grid's extent and ruling, in document units: 1 mm and 10 µm.
/// The grid's line count is extent over spacing, so they bound its mesh.
const MIN_GRID_SIZE: Real = 1.0;
const MIN_GRID_SPACING: Real = 0.01;

/// Display steps a pixel of drag moves the lengths that run to metres — far
/// clip, clip depth, grid size — where the rest run to millimetres.
const LONG_STEPS: f64 = 100.0;

/// The scene tab, owning the state local to it.
pub struct SceneTab {
    /// The perspective projection to restore when switching back from
    /// orthographic, so the toggle is lossless.
    last_perspective: Projection,
}

impl Default for SceneTab {
    fn default() -> Self {
        Self {
            last_perspective: Projection::Perspective { fovy: 35.0, znear: 1.0, zfar: 5_000.0 },
        }
    }
}

impl SceneTab {
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        camera: &mut PositionedCamera,
        construction: &mut ConstructionOptions,
        actions: &mut Vec<AppAction>,
    ) {
        egui::CollapsingHeader::new("Units")
            .default_open(true)
            .show(ui, |ui| units_ui(ui, &mut construction.units));

        let units = construction.units;
        egui::CollapsingHeader::new("Camera")
            .default_open(true)
            .show(ui, |ui| camera_ui(ui, camera, &mut self.last_perspective, units, actions));

        egui::CollapsingHeader::new("Construction plane")
            .default_open(true)
            .show(ui, |ui| {
                if plane_ui(ui, construction) {
                    actions.push(AppAction::ConstructionChanged);
                }
            });

        egui::CollapsingHeader::new("Grid")
            .default_open(true)
            .show(ui, |ui| {
                if grid_ui(ui, construction) {
                    actions.push(AppAction::ConstructionChanged);
                }
            });

        egui::CollapsingHeader::new("Snap")
            .default_open(true)
            .show(ui, |ui| snap_ui(ui, construction));

        egui::CollapsingHeader::new("Debug")
            .default_open(false)
            .show(ui, |ui| {
                if debug_ui(ui, construction) {
                    actions.push(AppAction::TessellationChanged);
                }
            });
    }
}

/// The unit lengths are shown and typed in, and to how many decimals.
/// Presentation only: the document models in [`DOCUMENT_UNITS`] whatever is
/// picked here, so nothing is rebuilt.
fn units_ui(ui: &mut egui::Ui, units: &mut LengthDisplay) {
    egui::Grid::new("unit_settings").num_columns(2).show(ui, |ui| {
        ui.label("Display unit");
        egui::ComboBox::from_id_salt("display_unit")
            .selected_text(units.unit.name())
            .show_ui(ui, |ui| {
                for unit in LengthUnit::ALL {
                    ui.selectable_value(&mut units.unit, unit, unit.name());
                }
            });
        ui.end_row();

        ui.label("Decimals");
        ui.add(egui::DragValue::new(&mut units.decimals).range(0..=6));
        ui.end_row();
    });

    if let Some(unit) = DOCUMENT_UNITS.nearest_unit() {
        ui.weak(format!("The model is stored in {}s.", unit.name().to_lowercase()));
    }
}

/// Projection controls for the view's camera, lengths shown in `units`. Pose
/// is left to the navigation operator.
///
/// The two projections carry different parameters, so the grid swaps rows with
/// the mode: orthographic has no near plane to set, and its extent is a world
/// size rather than an angle.
fn camera_ui(
    ui: &mut egui::Ui,
    camera: &mut PositionedCamera,
    last_perspective: &mut Projection,
    units: LengthDisplay,
    actions: &mut Vec<AppAction>,
) {
    let long_speed = units.step(DOCUMENT_UNITS) * LONG_STEPS;
    let mut changed = false;
    egui::Grid::new("camera_settings").num_columns(2).show(ui, |ui| {
        match &mut camera.projection {
            Projection::Perspective { fovy, znear, zfar } => {
                ui.label("Field of view");
                changed |=
                    ui.add(egui::Slider::new(fovy, 10.0..=120.0).suffix("°")).changed();
                ui.end_row();

                let far = *zfar;
                ui.label("Near clip");
                changed |=
                    ui.add(length_value(znear, units).range(MIN_CLIP..=far - MIN_CLIP)).changed();
                ui.end_row();

                let near = *znear;
                ui.label("Far clip");
                let far = length_value(zfar, units).speed(long_speed);
                changed |= ui.add(far.range(near + MIN_CLIP..=Real::MAX)).changed();
                ui.end_row();
            }
            Projection::Orthographic { half_height, half_depth } => {
                let mut height = *half_height * 2.0;
                ui.label("View height");
                if ui.add(length_value(&mut height, units).range(MIN_CLIP..=Real::MAX)).changed() {
                    *half_height = height / 2.0;
                    changed = true;
                }
                ui.end_row();

                ui.label("Clip depth");
                let depth = length_value(half_depth, units).speed(long_speed);
                changed |= ui.add(depth.range(MIN_CLIP..=Real::MAX)).changed();
                ui.end_row();
            }
        }

        ui.label("Projection");
        let mut ortho = camera.projection.is_ortho();
        if ui.checkbox(&mut ortho, "Orthographic").changed() {
            if ortho {
                *last_perspective = camera.projection;
                camera.make_orthographic();
            } else if let Projection::Perspective { fovy, znear, zfar } = *last_perspective {
                camera.make_perspective(fovy, znear, zfar);
            }
            changed = true;
        }
        ui.end_row();
    });

    if changed {
        actions.push(AppAction::CameraChanged);
    }
}

/// Axis-aligned plane presets plus an offset along the plane normal, and
/// whether view snaps turn the plane. Returns true when the plane changed
/// (grid must be rebuilt).
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
            // Either facing: a view snap turns the normal toward the eye.
            let selected = construction.construction_plane.normal.dot(normal).abs() > 0.999;
            if ui.selectable_label(selected, label).clicked() && !selected {
                construction.construction_plane =
                    Plane::from_point(normal, Point3::origin() + normal * offset);
                changed = true;
            }
        }
    });

    egui::Grid::new("plane_settings").num_columns(2).show(ui, |ui| {
        ui.label("Offset");
        if ui.add(length_value(&mut offset, construction.units)).changed() {
            let normal = construction.construction_plane.normal;
            construction.construction_plane =
                Plane::from_point(normal, Point3::origin() + normal * offset);
            changed = true;
        }
        ui.end_row();
    });

    ui.checkbox(&mut construction.follow_view, "Follow view snaps")
        .on_hover_text("Snapping the view to an axis turns the plane to face it until you orbit away");

    changed
}

/// Grid visibility and dimensions. Returns true when the grid must be rebuilt.
fn grid_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) -> bool {
    let units = construction.units;
    let grid = &mut construction.grid;
    let mut changed = false;

    changed |= ui.checkbox(&mut grid.visible, "Show grid").changed();

    egui::Grid::new("grid_settings").num_columns(2).show(ui, |ui| {
        ui.label("Size");
        let long_speed = units.step(DOCUMENT_UNITS) * LONG_STEPS;
        let size = length_value(&mut grid.size, units).speed(long_speed);
        changed |= ui.add(size.range(MIN_GRID_SIZE..=Real::MAX)).changed();
        ui.end_row();

        ui.label("Minor spacing");
        let spacing = length_value(&mut grid.minor_spacing, units);
        changed |= ui.add(spacing.range(MIN_GRID_SPACING..=Real::MAX)).changed();
        ui.end_row();

        ui.label("Major every");
        changed |= ui
            .add(egui::DragValue::new(&mut grid.major_every).range(1..=1000))
            .changed();
        ui.end_row();
    });

    changed
}

/// Diagnostic toggles. Returns true when existing parts must be re-tessellated.
fn debug_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) -> bool {
    let options = &mut construction.geometry_options;
    ui.checkbox(&mut options.show_seam_edges, "Show seam edges").changed()
}

/// Snap master switch, per-kind toggles, and pixel tolerance. Read live by
/// operators, so no rebuild action is needed.
fn snap_ui(ui: &mut egui::Ui, construction: &mut ConstructionOptions) {
    let settings = &mut construction.snap.settings;

    ui.checkbox(&mut settings.enabled, "Enable snapping");

    ui.add_enabled_ui(settings.enabled, |ui| {
        let kinds: [(&str, SnapFlags); 9] = [
            ("Origin", SnapFlags::ORIGIN),
            ("Grid guides", SnapFlags::GRID_GUIDE),
            ("Grid axes", SnapFlags::GRID_AXIS),
            ("Faces", SnapFlags::FACE),
            ("Face centers", SnapFlags::FACE_CENTER),
            ("Corners", SnapFlags::CORNER),
            ("Edges", SnapFlags::EDGE),
            ("Midpoints", SnapFlags::MIDPOINT),
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
