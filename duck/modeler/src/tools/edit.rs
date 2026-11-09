//! Settings a tool edits live, by its panel fields and its grips in the
//! viewport, until it applies them; and the panel parts every tool shares.

use std::ops::RangeInclusive;

use duck_engine_viewer::common::{consts, InnerSpace, LengthDisplay, Plane, Point3, Real, Vector3};
use duck_engine_viewer::operator::{DragKind, Handle, HandleDrag, HandleEvent, HandleId, HandleReach, HandleShape};

use crate::ui::length_value;

/// Smallest value a dimension field accepts. Anything at or below it is
/// degenerate and can't be built.
pub(super) const MIN_DIMENSION: Real = 1e-6;

/// Settings a tool holds live, as preview geometry, until they are applied.
///
/// Two editors drive the same values: the panel's fields via
/// [`ui`](Params::ui), and the grips via [`handles`](Params::handles) and
/// [`apply_handle`](Params::apply_handle).
///
/// `Copy` because a grip drag edits from a snapshot taken when it was grabbed.
pub trait Params: Copy {
    /// The fields, one row each of a two-column grid, lengths shown in
    /// `units`. Returns true when a value changed.
    fn ui(&mut self, ui: &mut egui::Ui, units: LengthDisplay) -> bool;

    /// The grips for these settings, or empty for none.
    fn handles(&self) -> Vec<Handle> {
        Vec::new()
    }

    /// Applies a grip drag, editing from `grabbed` — the settings as they were
    /// when the grip was taken.
    ///
    /// A [`HandleDrag`] carries its total offset from the grab rather than an
    /// increment, so editing from the live value would compound it.
    fn apply_handle(&mut self, _drag: &HandleDrag, _grabbed: &Self) {}

    /// Whether these settings leave nothing to build.
    fn is_degenerate(&self) -> bool {
        false
    }
}

/// Settings under edit: the live values, and the snapshot a held grip drags
/// from.
#[derive(Clone, Copy)]
pub(super) struct Edit<P> {
    params: P,
    grabbed: Option<P>,
}

impl<P: Params> Edit<P> {
    pub(super) fn new(params: P) -> Self {
        Self { params, grabbed: None }
    }

    pub(super) fn params(&self) -> &P {
        &self.params
    }

    /// Replaces the settings, as a key that edits them does.
    pub(super) fn set(&mut self, params: P) {
        self.params = params;
    }

    pub(super) fn handles(&self) -> Vec<Handle> {
        self.params.handles()
    }

    /// Runs one grip event against the settings. Returns whether they changed.
    pub(super) fn on_handle(&mut self, event: &HandleEvent) -> bool {
        match event {
            HandleEvent::Begin(_) => {
                self.grabbed = Some(self.params);
                false
            }
            HandleEvent::Drag(drag) => {
                let Some(grabbed) = self.grabbed else { return false };
                self.params.apply_handle(drag, &grabbed);
                true
            }
            HandleEvent::End(_) => {
                self.grabbed = None;
                false
            }
            // Put the settings back as they were when the grip was taken.
            HandleEvent::Cancel(_) => match self.grabbed.take() {
                Some(grabbed) => {
                    self.params = grabbed;
                    true
                }
                None => false,
            },
        }
    }

    /// The panel's body: the fields, lengths shown in `units`, then Cancel /
    /// Apply.
    pub(super) fn panel(&mut self, ui: &mut egui::Ui, units: LengthDisplay) -> PanelAction {
        let changed = egui::Grid::new("tool_params")
            .num_columns(2)
            .show(ui, |ui| self.params.ui(ui, units))
            .inner;
        ui.separator();
        match apply_row(ui) {
            PanelAction::None if changed => PanelAction::Changed,
            action => action,
        }
    }
}

/// What the user asked of a tool's panel this frame.
pub(super) enum PanelAction {
    None,
    /// A value changed; the preview needs refreshing.
    Changed,
    Apply,
    Cancel,
}

/// The Cancel / Apply row that closes a tool's panel.
pub(super) fn apply_row(ui: &mut egui::Ui) -> PanelAction {
    let mut action = PanelAction::None;
    ui.horizontal(|ui| {
        if ui.button("Cancel").clicked() {
            action = PanelAction::Cancel;
        }
        if ui.button("Apply  ↵").clicked() {
            action = PanelAction::Apply;
        }
    });
    action
}

/// `error`, in the error color.
pub(super) fn error_line(ui: &mut egui::Ui, error: &str) {
    ui.colored_label(ui.visuals().error_fg_color, error);
}

/// One labelled dimension row of a panel's grid: a length, shown in `units`,
/// that must stay above [`MIN_DIMENSION`]. Returns true when the value changed.
pub(super) fn dimension_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Real,
    units: LengthDisplay,
) -> bool {
    length_field(ui, label, value, MIN_DIMENSION..=Real::MAX, units)
}

/// One labelled length row of a panel's grid, held within `range` and shown in
/// `units`. Returns true when the value changed.
pub(super) fn length_field(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Real,
    range: RangeInclusive<Real>,
    units: LengthDisplay,
) -> bool {
    ui.label(label);
    let changed = ui.add(length_value(value, units).range(range)).changed();
    ui.end_row();
    changed
}

/// One labelled angle row of a panel's grid, shown in degrees and held within
/// `±limit` radians. Returns true when the value changed.
pub(super) fn angle_field(ui: &mut egui::Ui, label: &str, radians: &mut Real, limit: Real) -> bool {
    ui.label(label);
    let limit = limit.to_degrees();
    let mut degrees = radians.to_degrees();
    let changed = ui
        .add(egui::DragValue::new(&mut degrees).speed(0.5).range(-limit..=limit).suffix("°"))
        .changed();
    // Written back only on an edit, so an untouched value never drifts through
    // the degree round trip.
    if changed {
        *radians = degrees.to_radians();
    }
    ui.end_row();
    changed
}

/// A dimension moved by `delta`, held at or above [`MIN_DIMENSION`] so a grip
/// dragged past the opposite face flattens the shape rather than inverting it.
pub(super) fn grip_dimension(from: Real, delta: Real) -> Real {
    (from + delta).max(MIN_DIMENSION)
}

/// The arm an angle grip swings on: from its pivot on an axis, square to the
/// axis, out to the lever.
pub(super) struct Lever {
    pivot: Point3,
    axis: Vector3,
    arm: Vector3,
}

impl Lever {
    /// The lever out to `end` from the axis through `origin` along the unit
    /// `axis`. An end on the axis itself swings at `size` instead, along some
    /// square direction.
    pub(super) fn new(origin: Point3, axis: Vector3, end: Point3, size: Real) -> Self {
        let pivot = origin + axis * (end - origin).dot(axis);
        let arm = end - pivot;
        let arm = if arm.magnitude() > MIN_DIMENSION {
            arm
        } else {
            Plane::from_point(axis, pivot).basis().0 * size.max(MIN_DIMENSION)
        };
        Self { pivot, axis, arm }
    }

    /// The end of the arm, swung `angle` counter-clockwise looking down the
    /// axis.
    pub(super) fn swung(&self, angle: Real) -> Point3 {
        self.pivot + self.arm * angle.cos() + self.axis.cross(self.arm) * angle.sin()
    }

    /// The grip `id`: a ball at the end of the arm swung `angle`, tied by a
    /// leader to the axis, dragged round it.
    pub(super) fn handle(&self, id: HandleId, angle: Real) -> Handle {
        Handle::new(id, HandleShape::Ball, self.swung(angle))
            .with_direction(self.axis)
            .with_drag(DragKind::Plane)
            .with_reach(HandleReach::Leader(self.pivot))
    }

    /// How far `drag` has swung the grip about the axis since it was grabbed,
    /// given the swing `so_far`. The drag's own angle wraps at half a turn
    /// either way, so this takes the turn nearest the swing so far, and a drag
    /// carries on round.
    pub(super) fn swept(&self, drag: &HandleDrag, so_far: Real) -> Real {
        let swept = drag.angle_about(self.pivot, self.axis);
        swept + consts::TAU * ((so_far - swept) / consts::TAU).round()
    }
}
