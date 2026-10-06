use duck_engine_common::Point3;
use duck_engine_scene::resource::NodeId;
use duck_engine_viewer::{
    bindings::{InputBinding, InputMap},
    event::{DeviceEvent, Event, EventContext},
    input::{ElementState, Key, Modifiers, MouseButton, NamedKey},
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::ops::primitives::{closed_polyline, polyline, region};
use crate::preview::PreviewSession;
use crate::snap::{Snap, SnapKind, SnapProvider, WireStartSnap};
use crate::tools::{ModelingTool, ToolInfo, Workspace};
use crate::ui::icons;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LineAction {
    /// Place a point (or close the wire when hovering the start point).
    AddPoint,
    /// Finalize the current open polyline.
    Finish,
}

enum Phase {
    Idle,
    /// Building a polyline: `points` are the placed vertices and `closing` records
    /// whether the cursor is currently snapped onto the start point.
    Building {
        points: Vec<Point3>,
        closing: bool,
    },
}

pub struct LineTool {
    phase: Phase,
    workspace: Workspace,
    preview: PreviewSession,
    bindings: InputMap<LineAction>,
    /// Where the modeler's 3D cursor should sit (the latest snap point), or `None`
    /// to hide it. Read by the modeler via [`ModelingTool::cursor_target`].
    cursor_target: Option<Point3>,
    /// Set when a line has been committed so the modeler cedes back to selection.
    finished: bool,
}


impl LineTool {
    pub fn new(workspace: &Workspace) -> Self {
        let bindings = InputMap::new()
            .bind(
                InputBinding::MouseClick { button: MouseButton::Left, modifiers: Modifiers::default() },
                LineAction::AddPoint,
            )
            .bind(
                InputBinding::MouseClick { button: MouseButton::Right, modifiers: Modifiers::default() },
                LineAction::Finish,
            );
        let preview = workspace.preview_session();
        Self {
            phase: Phase::Idle,
            workspace: workspace.clone(),
            preview,
            bindings,
            cursor_target: None,
            finished: false,
        }
    }

    /// Resolves a snapped point. While building a face (≥3 points), offers the
    /// wire's start point as a [`SnapKind::WireStart`] candidate so the snap
    /// engine can close the path; callers read `snap.kind == WireStart` to learn
    /// whether the cursor is closing the wire.
    fn snapped_point(
        &self,
        cursor: (f32, f32),
        exclude: &[NodeId],
        ctx: &EventContext,
    ) -> Option<Snap> {
        // A face needs at least three vertices to enclose an area.
        let wire_start = match &self.phase {
            Phase::Building { points, .. } if points.len() >= 3 => {
                Some(WireStartSnap { start: points[0] })
            }
            _ => None,
        };
        let extra: Vec<&dyn SnapProvider> =
            wire_start.iter().map(|p| p as &dyn SnapProvider).collect();

        self.workspace.snap_with(cursor, exclude, ctx, &extra)
    }

    /// Rebuilds the preview geometry. `cursor_point` is the live (snapped) cursor
    /// position to draw a rubber-band segment to, or `None` to show only the placed
    /// points. When `closing` is set, shows a filled face preview instead.
    fn rebuild_preview(&mut self, cursor_point: Option<Point3>, closing: bool) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };

        let shape = if closing {
            // Non-coplanar points bound no face; the region is then the loop alone.
            closed_polyline(&points).map(region)
        } else {
            let mut all = points;
            all.extend(cursor_point);
            polyline(&all).map(|wire| Shape::from(&wire))
        };

        // `rebuild` keeps the last valid preview if construction fails (e.g. a snap
        // produced a degenerate point), so the line doesn't momentarily disappear.
        let Ok(shape) = shape else { return };
        let preview_options = self.workspace.preview_options();
        if self.preview.try_replace_preview(&shape, &preview_options, "line").is_some()
            && let Phase::Building { closing: c, .. } = &mut self.phase
        {
            *c = closing;
        }
    }

    /// Adds a point to the wire or starts building a wire if there were no previous points.
    /// Returns true if a point was successfully added.
    fn on_add_point(&mut self, position: (f32, f32), ctx: &mut EventContext) -> bool {
        let Some(snap) = self.snapped_point(position, self.preview.preview_nodes(), ctx) else {
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
                    // Show the placed polyline; the next cursor move adds the rubber band.
                    self.rebuild_preview(None, false);
                }
            }
        }

        true
    }

    /// Commits the placed points as a closed planar region and ends the tool.
    fn commit_closed(&mut self) {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return,
        };
        let _ = self.preview.commit();

        let committed = closed_polyline(&points)
            .map(region)
            .and_then(|shape| self.workspace.add_numbered_part("Region", shape));
        match committed {
            Ok(_) => self.finished = true,
            Err(e) => self.workspace.notifications.failure("Line", &e),
        }
        self.phase = Phase::Idle;
    }

    /// Commits the placed points as an open polyline wire and ends the tool. Does
    /// nothing if fewer than one segment has been defined.
    fn finish(&mut self) -> bool {
        let points = match &self.phase {
            Phase::Building { points, .. } => points.clone(),
            Phase::Idle => return false,
        };
        let _ = self.preview.commit();
        self.phase = Phase::Idle;
        // A lone point is no line: there is nothing to commit.
        if points.len() < 2 {
            return false;
        }

        let committed = polyline(&points).and_then(|wire| self.workspace.add_numbered_part("Line", Shape::from(&wire)));
        match committed {
            Ok(_) => {
                self.finished = true;
                true
            }
            Err(e) => {
                self.workspace.notifications.failure("Line", &e);
                false
            }
        }
    }

    /// Discards the in-progress line, removing its preview.
    pub fn cancel(&mut self) {
        self.preview.cancel();
        self.phase = Phase::Idle;
    }

    fn on_cursor_moved(&mut self, position: (f64, f64), ctx: &mut EventContext) {
        let cursor = (position.0 as f32, position.1 as f32);
        let snapped = self.snapped_point(cursor, self.preview.preview_nodes(), ctx);

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

impl ModelingTool for LineTool {
    fn info(&self) -> ToolInfo {
        ToolInfo { id: "line", icon: icons::LINE, shortcut: None }
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::MouseClick { button, position, .. } => {
                let actions = self.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                let mut handled = false;
                for action in actions {
                    handled |= match action {
                        LineAction::AddPoint => self.on_add_point(*position, ctx),
                        LineAction::Finish => self.finish(),
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

    fn deactivate(&mut self) {
        self.cancel();
        self.cursor_target = None;
        self.finished = false;
    }

    /// Leaving the tool ends the polyline where it stands, as right-click and
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
