use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_common::Point3;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    operator::Operator,
    scene::PositionedCamera,
    selection::SelectionManager,
};
use glam::{dvec3, DVec3};
use log::warn;
use opencascade::primitives::{Edge, Shape, Wire};

use crate::document::Document;
use crate::preview::PreviewSession;
use crate::snap::{Snap, SnapKind, SnapProvider, WireStartSnap};
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CurveAction {
    /// Place a point (or close the curve when hovering the start point).
    AddPoint,
    /// Finalize the current open curve.
    Finish,
}

enum Phase {
    Idle,
    /// Building a curve: `points` are the placed interpolation points and `closing`
    /// records whether the cursor is currently snapped onto the start point.
    Building {
        points: Vec<Point3>,
        closing: bool,
    },
}

pub struct CurveOperator {
    phase: Phase,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    preview: PreviewSession,
    bindings: InputMap<CurveAction>,
    /// Where the modeler's 3D cursor should sit (the latest snap point), or `None`
    /// to hide it. Read by the modeler via [`ModelingTool::cursor_target`].
    cursor_target: Option<Point3>,
    /// Set when a curve has been committed so the modeler cedes back to selection.
    finished: bool,
}

fn to_dvec3(p: &Point3) -> DVec3 {
    dvec3(p.x as f64, p.y as f64, p.z as f64)
}

/// Builds an open curve wire shape: a B-spline interpolated through `points`.
/// Returns `None` if there are fewer than two points (two points are accepted so
/// the rubber-band preview works, but a committed curve needs three).
fn open_curve_shape(points: &[Point3]) -> Option<Shape> {
    if points.len() < 2 {
        return None;
    }
    let edge = Edge::spline_from_points(points.iter().map(to_dvec3), None, false)
        .map_err(|e| warn!("Failed to build curve edge: {e}"))
        .ok()?;
    let wire = Wire::from_edges(&[edge])
        .map_err(|e| warn!("Failed to build curve wire: {e}"))
        .ok()?;
    Some(Shape::from(&wire))
}

/// Builds a closed wire from a smooth periodic B-spline through `points` (the
/// start point is not repeated; the interpolation closes the loop). Returns
/// `None` with fewer than three points.
fn closed_curve(points: &[Point3]) -> Option<Wire> {
    if points.len() < 3 {
        return None;
    }
    let edge = Edge::spline_from_points(points.iter().map(to_dvec3), None, true)
        .map_err(|e| warn!("Failed to build closed curve edge: {e}"))
        .ok()?;
    Wire::from_edges(&[edge])
        .map_err(|e| warn!("Failed to build closed curve wire: {e}"))
        .ok()
}

/// Builds a closed curve shape (the loop, with no fill) from `points`. Used as a
/// fallback when the points are not co-planar and a face cannot be built.
fn closed_curve_shape(points: &[Point3]) -> Option<Shape> {
    Some(Shape::from(&closed_curve(points)?))
}

/// Builds a closed planar face shape bounded by the periodic curve. Returns `None`
/// with fewer than three points, or when the points are not co-planar (no face can
/// be built).
fn closed_face_shape(points: &[Point3]) -> Option<Shape> {
    let wire = closed_curve(points)?;
    let face = wire.to_face().map_err(|e| warn!("Failed to build face from curve: {e}")).ok()?;
    Some(Shape::from(&face))
}

impl CurveOperator {
    pub fn new(
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
    ) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                CurveAction::AddPoint,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                CurveAction::Finish,
            );
        let preview = PreviewSession::new(Arc::clone(&document));
        Self {
            phase: Phase::Idle,
            construction_options,
            document,
            preview,
            bindings,
            cursor_target: None,
            finished: false,
        }
    }

    /// Resolves a snapped point. While building (≥3 points), offers the curve's
    /// start point as a [`SnapKind::WireStart`] candidate so the snap engine can
    /// close the loop; callers read `snap.kind == WireStart` to learn whether the
    /// cursor is closing the curve.
    fn snapped_point(
        &self,
        cursor: (f32, f32),
        exclude: &[NodeId],
        camera: &PositionedCamera,
        ctx: &EventContext,
    ) -> Option<Snap> {
        // A periodic curve needs at least three interpolation points.
        let wire_start = match &self.phase {
            Phase::Building { points, .. } if points.len() >= 3 => {
                Some(WireStartSnap { start: points[0] })
            }
            _ => None,
        };
        let extra: Vec<&dyn SnapProvider> =
            wire_start.iter().map(|p| p as &dyn SnapProvider).collect();

        self.construction_options
            .borrow()
            .resolve_snap(cursor, exclude, camera, ctx, &extra)
    }

    /// Rebuilds the preview geometry. `cursor_point` is the live (snapped) cursor
    /// position to draw a rubber-band curve to, or `None` to show only the placed
    /// points. When `closing` is set, shows a filled face preview instead.
    fn rebuild_preview(&mut self, cursor_point: Option<Point3>, closing: bool) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };

        let shape = if closing {
            // Non-coplanar points can't form a face; fall back to the closed curve.
            closed_face_shape(&points).or_else(|| closed_curve_shape(&points))
        } else {
            let mut all = points;
            if let Some(c) = cursor_point {
                all.push(c);
            }
            open_curve_shape(&all)
        };

        // `rebuild` keeps the last valid preview if construction fails (e.g. a snap
        // produced a degenerate point), so the curve doesn't momentarily disappear.
        let Some(shape) = shape else { return };
        let preview_options = self.construction_options.borrow().preview_options();
        if self.preview.try_replace_preview(&shape, &preview_options, "curve").is_some()
            && let Phase::Building { closing: c, .. } = &mut self.phase
        {
            *c = closing;
        }
    }

    /// Adds a point to the curve or starts building a curve if there were no previous points.
    /// Returns true if a point was successfully added.
    fn on_add_point(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let camera = ctx.camera.clone();
        let Some(snap) = self.snapped_point(position, self.preview.preview_nodes(), &camera, ctx) else {
            return false;
        };
        let point = snap.position;
        let closing = snap.kind == SnapKind::WireStart;

        match &mut self.phase {
            Phase::Idle => {
                self.phase = Phase::Building { points: vec![point], closing: false };
            }
            Phase::Building { .. } => {
                if closing {
                    self.commit_closed();
                } else {
                    if let Phase::Building { points, .. } = &mut self.phase {
                        points.push(point);
                    }
                    // Show the placed curve; the next cursor move adds the rubber band.
                    self.rebuild_preview(None, false);
                }
            }
        }

        true
    }

    /// Commits the placed points as a closed planar region bounded by a smooth
    /// periodic curve, and ends the tool.
    fn commit_closed(&mut self) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };
        let _ = self.preview.commit();

        if let Some(shape) = closed_face_shape(&points).or_else(|| closed_curve_shape(&points)) {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            if doc
                .add_numbered_part(
                    "Region",
                    shape,
                    &coptions.geometry_options,
                )
                .is_ok()
            {
                self.finished = true;
            }
        }
        else {
            warn!("Failed to build closed face shape for curve.")
        }
        self.phase = Phase::Idle;
    }

    /// Commits the placed points as an open curve and ends the tool. Does nothing
    /// with fewer than three placed points (the curve tool's minimum).
    fn finish(&mut self) -> bool {
        let points = match &self.phase {
            Phase::Building { points, .. } if points.len() >= 3 => points.clone(),
            _ => return false,
        };
        let _ = self.preview.commit();

        let mut committed = false;
        if let Some(shape) = open_curve_shape(&points) {
            let coptions = self.construction_options.borrow();
            let mut doc = self.document.lock().unwrap();
            committed = doc
                .add_numbered_part(
                    "Curve",
                    shape,
                    &coptions.geometry_options,
                )
                .is_ok();
        }
        self.phase = Phase::Idle;
        if committed {
            self.finished = true;
        }
        committed
    }

    /// Discards the in-progress curve, removing its preview.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);
        let camera = ctx.camera.clone();
        let snapped = self.snapped_point(cursor, self.preview.preview_nodes(), &camera, ctx);

        // Record where the 3D cursor should sit
        self.cursor_target = snapped.map(|s| s.position);

        if matches!(self.phase, Phase::Building { .. }) {
            if let Some(snap) = snapped {
                let closing = snap.kind == SnapKind::WireStart;
                self.rebuild_preview(Some(snap.position), closing);
            }
        }
    }
}

impl ModelingTool for CurveOperator {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "curve", icon: icons::CURVE, shortcut: None }
    }

    fn deactivate(&mut self) {
        self.cancel();
        self.cursor_target = None;
        self.finished = false;
    }

    /// Leaving the tool ends the curve where it stands, as right-click and
    /// Enter do; too few points is no result, and `deactivate` then drops it.
    fn finalize(&mut self, _selection: &mut SelectionManager) -> anyhow::Result<()> {
        self.finish();
        Ok(())
    }

    fn is_finished(&self) -> bool {
        self.finished
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor_target
    }
}

impl Operator for CurveOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        CurveAction::AddPoint => self.on_add_point(*position, ctx),
                        CurveAction::Finish => self.finish(),
                    };
                }
                handled
            }
            DeviceEvent::CursorMoved { position } => {
                self.on_cursor_moved(*position, ctx);
                false
            }
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                match key_event.logical_key {
                    Key::Named(NamedKey::Enter) => self.finish(),
                    Key::Named(NamedKey::Escape) => {
                        let was_building = matches!(self.phase, Phase::Building { .. });
                        if was_building {
                            self.cancel();
                        }
                        was_building
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn name(&self) -> &str {
        "Curve"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32, z: f32) -> Point3 {
        Point3::new(x, y, z)
    }

    #[test]
    fn open_curve_needs_two_points() {
        assert!(open_curve_shape(&[p(0.0, 0.0, 0.0)]).is_none());
        assert!(open_curve_shape(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 1.0)]).is_some());
    }

    #[test]
    fn open_curve_through_three_points() {
        let pts = [p(0.0, 0.0, 0.0), p(1.0, 0.0, 2.0), p(3.0, 0.0, 1.0)];
        assert!(open_curve_shape(&pts).is_some());
    }

    #[test]
    fn closed_curve_needs_three_points() {
        assert!(closed_curve(&[p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0)]).is_none());
        let pts = [p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(1.0, 0.0, 2.0)];
        assert!(closed_curve(&pts).is_some());
    }

    #[test]
    fn closed_face_from_planar_points() {
        let pts = [p(0.0, 0.0, 0.0), p(2.0, 0.0, 0.0), p(2.0, 0.0, 2.0), p(0.0, 0.0, 2.0)];
        assert!(closed_face_shape(&pts).is_some());
    }
}
