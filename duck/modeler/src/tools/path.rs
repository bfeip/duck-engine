use anyhow::Result;
use duck_engine_common::Point3;
use duck_engine_viewer::{
    event::{Event, EventContext},
    selection::SelectionManager,
};
use opencascade::primitives::Shape;

use crate::ops::primitives::{closed_polyline, closed_spline, polyline, region, spline};
use crate::preview::PreviewSession;
use crate::snap::{Snap, SnapKind, SnapProvider, WireStartSnap};
use crate::ui::icons;
use super::{Gesture, ModelingTool, ToolInfo, Workspace};

/// How a path runs through its points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    /// Straight from point to point.
    Polyline,
    /// One smooth curve through every point.
    Spline,
}

impl PathKind {
    /// Palette identity.
    fn tool(self) -> ToolInfo {
        match self {
            PathKind::Polyline => ToolInfo { id: "line", icon: icons::LINE, shortcut: None },
            PathKind::Spline => ToolInfo { id: "curve", icon: icons::CURVE, shortcut: None },
        }
    }

    /// The name series an open path's parts are numbered in.
    fn name(self) -> &'static str {
        match self {
            PathKind::Polyline => "Line",
            PathKind::Spline => "Curve",
        }
    }

    /// The fewest points an open path runs through.
    fn min_points(self) -> usize {
        match self {
            PathKind::Polyline => 2,
            PathKind::Spline => 3,
        }
    }

    /// The open path through `points`.
    fn open(self, points: &[Point3]) -> Result<Shape> {
        let wire = match self {
            PathKind::Polyline => polyline(points)?,
            PathKind::Spline => spline(points)?,
        };
        Ok(Shape::from(&wire))
    }

    /// What the path through `points` and back to the first encloses.
    fn closed(self, points: &[Point3]) -> Result<Shape> {
        let wire = match self {
            PathKind::Polyline => closed_polyline(points)?,
            PathKind::Spline => closed_spline(points)?,
        };
        Ok(region(wire))
    }
}

/// How a path ends.
#[derive(Clone, Copy)]
enum Ending {
    /// At its last point.
    Open,
    /// Back at its first point, enclosing a region.
    Closed,
}

/// Draws a line or curve through clicked points.
///
/// Right-click or Enter ends the path at its last point; clicking its first
/// point again closes it into a region. Either adds it as a part and leaves the
/// tool. Escape discards it.
pub struct PathTool {
    kind: PathKind,
    workspace: Workspace,
    preview: PreviewSession,
    /// The points placed so far, none before the first click.
    points: Vec<Point3>,
    /// Where the 3D cursor marks.
    cursor: Option<Point3>,
    /// Set once the path is added or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

impl PathTool {
    pub fn new(kind: PathKind, workspace: &Workspace) -> Self {
        Self {
            kind,
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            points: Vec::new(),
            cursor: None,
            finished: false,
        }
    }

    /// The first point, offered as a snap once the path can close on it: it
    /// takes three points to enclose anything.
    fn closing_snap(&self) -> Option<WireStartSnap> {
        (self.points.len() >= 3).then(|| WireStartSnap { start: self.points[0] })
    }

    /// The snapped point under the cursor at `at`, looking through the preview.
    fn snap(&self, at: (f32, f32), ctx: &EventContext) -> Option<Snap> {
        let start = self.closing_snap();
        let extra: Vec<&dyn SnapProvider> =
            start.iter().map(|start| start as &dyn SnapProvider).collect();
        self.workspace.snap_with(at, self.preview.preview_nodes(), ctx, &extra)
    }

    /// Follows `snap` with the cursor and the path's loose end, or with the
    /// region the path would close into.
    fn hover(&mut self, snap: Option<Snap>) {
        self.cursor = snap.map(|snap| snap.position);
        let Some(snap) = snap.filter(|_| !self.points.is_empty()) else { return };
        let shape = if snap.kind == SnapKind::WireStart {
            self.kind.closed(&self.points)
        } else {
            let points: Vec<Point3> = self.points.iter().copied().chain([snap.position]).collect();
            self.kind.open(&points)
        };
        self.show(shape);
    }

    /// Takes a click at `snap`: the next point, or the first point again, which
    /// closes the path.
    fn click(&mut self, snap: Option<Snap>) {
        let Some(snap) = snap else { return };
        if snap.kind == SnapKind::WireStart {
            self.apply_and_report(Ending::Closed);
        } else {
            self.points.push(snap.position);
            // The next hover adds the loose end.
            self.show(self.kind.open(&self.points));
        }
    }

    /// Previews `shape`, keeping the last preview when there is none to show.
    fn show(&mut self, shape: Result<Shape>) {
        let Ok(shape) = shape else { return };
        let options = self.workspace.preview_options();
        let name = format!("{} preview", self.kind.name());
        self.preview.try_replace_preview(&shape, &options, &name);
    }

    /// Whether the path can end open where it is.
    fn is_complete(&self) -> bool {
        self.points.len() >= self.kind.min_points()
    }

    /// Ends the path open, or drops one too short to stand and stays. Returns
    /// whether there was a path.
    fn finish(&mut self) -> bool {
        if self.points.is_empty() {
            return false;
        }
        if self.is_complete() {
            self.apply_and_report(Ending::Open);
        } else {
            self.reset();
        }
        true
    }

    /// Adds the path as a part, ended as `ending` says, and finishes the tool.
    /// A failure keeps the points, so the path can go on.
    fn apply(&mut self, ending: Ending) -> Result<()> {
        let (name, shape) = match ending {
            Ending::Open => (self.kind.name(), self.kind.open(&self.points)?),
            Ending::Closed => ("Region", self.kind.closed(&self.points)?),
        };
        self.workspace.add_numbered_part(name, shape)?;
        self.reset();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: a closing click, Enter, right-click.
    fn apply_and_report(&mut self, ending: Ending) {
        if let Err(e) = self.apply(ending) {
            self.workspace.notifications.failure(self.kind.name(), &e);
        }
    }

    /// Discards the path and finishes the tool.
    fn cancel(&mut self) {
        self.reset();
        self.finished = true;
    }

    /// Drops the points and their preview, ready for a first click.
    fn reset(&mut self) {
        self.preview.cancel();
        self.points.clear();
    }
}

impl ModelingTool for PathTool {
    fn info(&self) -> ToolInfo {
        self.kind.tool()
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        match gesture {
            Gesture::Hover(at) => {
                self.hover(self.snap(at, ctx));
                false
            }
            Gesture::Click { at, .. } => {
                self.click(self.snap(at, ctx));
                true
            }
            Gesture::Finish => self.finish(),
            Gesture::Cancel => {
                self.cancel();
                true
            }
            Gesture::Frame | Gesture::Key(_) => false,
        }
    }

    fn cursor_target(&self) -> Option<Point3> {
        self.cursor
    }

    /// Leaving the tool ends a path that can end open, as right-click and Enter
    /// do.
    fn finalize(&mut self, _selection: &mut SelectionManager) -> Result<()> {
        if self.is_complete() { self.apply(Ending::Open) } else { Ok(()) }
    }

    fn deactivate(&mut self) {
        self.reset();
        self.cursor = None;
        self.finished = false;
    }

    fn is_finished(&self) -> bool {
        self.finished
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_common::Real;
    use duck_engine_scene::Scene;
    use opencascade::primitives::ShapeType;

    use crate::document::Document;
    use crate::testing::workspace;

    fn tool(kind: PathKind) -> (PathTool, Workspace) {
        let ws = workspace(Document::new(Scene::default()));
        (PathTool::new(kind, &ws), ws)
    }

    /// A snap to `(x, 0, z)` on the construction plane.
    fn at(x: Real, z: Real) -> Option<Snap> {
        let position = Point3::new(x, 0.0, z);
        Some(Snap { position, direction: None, kind: SnapKind::ConstructionPlane })
    }

    /// The path's first point, offered back to close it.
    fn start_of(tool: &PathTool) -> Option<Snap> {
        let start = tool.closing_snap().expect("the path can close").start;
        Some(Snap { position: start, direction: None, kind: SnapKind::WireStart })
    }

    /// The names and shape types of the document's parts.
    fn parts(ws: &Workspace) -> Vec<(String, ShapeType)> {
        let doc = ws.document.lock().unwrap();
        doc.parts().map(|part| (part.name.clone(), part.shape.shape_type())).collect()
    }

    #[test]
    fn finishing_adds_the_open_path_and_leaves() {
        let (mut tool, ws) = tool(PathKind::Polyline);
        tool.click(at(0.0, 0.0));
        tool.click(at(1.0, 0.0));

        assert!(tool.finish());
        assert!(tool.is_finished());
        assert!(tool.preview.is_empty());
        assert_eq!(parts(&ws), [("Line-001".to_owned(), ShapeType::Wire)]);
    }

    /// A curve takes three points; a line, two.
    #[test]
    fn finishing_short_of_a_path_drops_the_points_and_stays() {
        let (mut tool, ws) = tool(PathKind::Spline);
        tool.click(at(0.0, 0.0));
        tool.click(at(1.0, 0.0));

        assert!(tool.finish());
        assert!(!tool.is_finished());
        assert!(tool.points.is_empty());
        assert!(tool.preview.is_empty());
        assert!(parts(&ws).is_empty());
        // Nothing placed is nothing to finish.
        assert!(!tool.finish());
    }

    #[test]
    fn clicking_the_first_point_again_closes_a_region() {
        let (mut tool, ws) = tool(PathKind::Spline);
        assert!(tool.closing_snap().is_none());
        tool.click(at(0.0, 0.0));
        tool.click(at(2.0, 0.0));
        tool.click(at(1.0, 2.0));

        tool.click(start_of(&tool));
        assert!(tool.is_finished());
        assert_eq!(parts(&ws), [("Region-001".to_owned(), ShapeType::Face)]);
    }

    #[test]
    fn cancelling_discards_the_path_and_leaves() {
        let (mut tool, ws) = tool(PathKind::Polyline);
        tool.click(at(0.0, 0.0));
        tool.hover(at(1.0, 0.0));
        assert!(!tool.preview.is_empty());

        tool.cancel();
        assert!(tool.is_finished());
        assert!(tool.preview.is_empty());
        assert!(parts(&ws).is_empty());
    }

    #[test]
    fn leaving_the_tool_ends_only_a_complete_path() {
        let (mut tool, ws) = tool(PathKind::Polyline);
        tool.click(at(0.0, 0.0));
        tool.finalize(&mut SelectionManager::new()).expect("nothing to add");
        assert!(parts(&ws).is_empty());

        tool.click(at(1.0, 0.0));
        tool.finalize(&mut SelectionManager::new()).expect("the line is added");
        assert_eq!(parts(&ws).len(), 1);
    }
}
