//! The driver for tools that place a new shape by picking points: box,
//! cylinder, sphere, rectangle and circle.
//!
//! A [`Primitive`] says how its picks go: the [`Stage`](Primitive::Stage) the
//! first one begins, what the pointer means at each stage, and the unit shape
//! the preview scales meanwhile. [`PrimitiveTool`] runs the rest the same way
//! for every primitive:
//!
//! - Each click takes the next pick. One that would leave the shape degenerate
//!   takes nothing, and no click reaches the selection.
//! - The last pick places the shape. From then its panel and grips edit the
//!   settings, moving the preview without re-tessellating it.
//! - Enter, right-click, Apply, or switching tools adds the placed shape as a
//!   part and leaves the tool. Before it is placed, Enter and right-click drop
//!   the picks so far and stay.
//! - Escape or Cancel discards everything and leaves the tool.

use anyhow::{Context, Result};
use duck_engine_common::{InnerSpace, MetricSpace, Plane, Point3, Ray, Real, Transform, Vector3};
use duck_engine_scene::resource::Visibility;
use duck_engine_viewer::{
    event::{Event, EventContext},
    operator::{Handle, HandleEvent},
    selection::SelectionManager,
};
use opencascade::primitives::{Shape, Wire};

use crate::ops::primitives::{circle, region};
use crate::preview::PreviewSession;
use crate::snap::Snap;
use super::edit::{Edit, PanelAction, Params, MIN_DIMENSION};
use super::{Gesture, ModelingTool, PanelContext, ToolInfo, Workspace};

/// A shape placed by picking points, run by [`PrimitiveTool`]. Implemented by
/// the settings the panel and grips edit once it is placed.
pub trait Primitive: Params + 'static {
    /// Where placement stands between the first pick and the last.
    type Stage: Copy;

    /// Palette identity.
    const TOOL: ToolInfo;
    /// The panel title, and the name series its parts are numbered in.
    const NAME: &'static str;

    /// The stage a first pick at `first` begins, with `construction` the
    /// construction plane.
    fn start(first: &Snap, construction: &Plane) -> Self::Stage;

    /// What `pointer` means at `stage`.
    fn track(stage: Self::Stage, pointer: &Pointer) -> Track<Self>;

    /// The unit shape the preview scales at `stage`, and once placed from it.
    fn reference(stage: Self::Stage) -> Result<Shape>;

    /// Places the reference shape for these settings.
    fn preview_transform(&self) -> Transform;

    /// The shape to add as a part.
    fn build(&self) -> Result<Shape>;
}

/// Where the cursor points: the snapped point under it, and the ray through
/// it.
#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    pub snap: Option<Snap>,
    pub ray: Ray,
}

/// What the pointer means at a stage.
pub struct Track<P: Primitive> {
    /// Where the 3D cursor marks, or `None` to hide it.
    pub cursor: Option<Point3>,
    /// Where the stage's reference shape goes, or `None` to hide it.
    pub preview: Option<Transform>,
    /// What a click would do, or `None` if it would leave the shape
    /// degenerate.
    pub click: Option<Next<P>>,
}

/// Where a click takes placement.
pub enum Next<P: Primitive> {
    /// On to another stage.
    Stage(P::Stage),
    /// Placed, with these settings.
    Placed(P),
}

/// A tool that places a new `P`, previewing it live until applied.
pub struct PrimitiveTool<P: Primitive> {
    workspace: Workspace,
    preview: PreviewSession,
    placement: Placement<P>,
    /// Where the 3D cursor marks while picking.
    cursor: Option<Point3>,
    /// Set once the shape is applied or discarded, so the tool cedes back to
    /// selection. Cleared on [`ModelingTool::deactivate`].
    finished: bool,
}

enum Placement<P: Primitive> {
    /// Waiting for the first pick.
    Idle,
    /// Picked up to this stage.
    Picking(P::Stage),
    /// Every point picked; the panel and grips edit the settings.
    Placed(Edit<P>),
}

impl<P: Primitive> PrimitiveTool<P> {
    pub fn new(workspace: &Workspace) -> Self {
        Self {
            workspace: workspace.clone(),
            preview: workspace.preview_session(),
            placement: Placement::Idle,
            cursor: None,
            finished: false,
        }
    }

    fn is_placed(&self) -> bool {
        matches!(self.placement, Placement::Placed(_))
    }

    /// Follows `pointer` with the cursor and the preview.
    fn hover(&mut self, pointer: &Pointer) {
        match self.placement {
            Placement::Idle => self.cursor = pointer.snap.map(|snap| snap.position),
            Placement::Picking(stage) => {
                let track = P::track(stage, pointer);
                self.cursor = track.cursor;
                self.show(track.preview);
            }
            Placement::Placed(_) => {}
        }
    }

    /// Takes the pick at `pointer`.
    fn click(&mut self, pointer: &Pointer) {
        let next = match self.placement {
            Placement::Idle => {
                let Some(first) = pointer.snap else { return };
                let construction = self.workspace.construction.borrow().construction_plane;
                Next::Stage(P::start(&first, &construction))
            }
            Placement::Picking(stage) => match P::track(stage, pointer).click {
                Some(next) => next,
                None => return,
            },
            Placement::Placed(_) => return,
        };
        match next {
            Next::Stage(stage) => {
                if let Err(e) = self.begin(stage) {
                    self.workspace.notifications.failure(P::NAME, &e);
                }
            }
            Next::Placed(params) => {
                self.show(Some(params.preview_transform()));
                self.placement = Placement::Placed(Edit::new(params));
            }
        }
    }

    /// Enters `stage`, previewing its reference shape, hidden until the pointer
    /// gives it a size.
    fn begin(&mut self, stage: P::Stage) -> Result<()> {
        let reference = P::reference(stage)?;
        let options = self.workspace.geometry_options();
        self.preview
            .try_replace_preview(&reference, &options, &format!("{} preview", P::NAME))
            .context("The preview could not be tessellated")?;
        self.preview.set_preview_visibility(Visibility::Invisible);
        self.placement = Placement::Picking(stage);
        Ok(())
    }

    /// Applies the placed shape, or drops the picks of one not yet placed and
    /// stays. Returns whether there was anything to finish.
    fn finish(&mut self) -> bool {
        match self.placement {
            Placement::Idle => return false,
            Placement::Picking(_) => self.reset(),
            Placement::Placed(_) => self.apply_and_report(),
        }
        true
    }

    /// Adds the placed shape as a part and finishes the tool. A failed build
    /// keeps the panel so the settings can be corrected.
    fn apply(&mut self) -> Result<()> {
        let Placement::Placed(edit) = &self.placement else { return Ok(()) };
        let shape = edit.params().build()?;
        self.workspace.add_numbered_part(P::NAME, shape)?;
        self.reset();
        self.finished = true;
        Ok(())
    }

    /// Apply, reporting a failure. For the gestures that keep the tool active
    /// and so must report for themselves: the panel's Apply, Enter,
    /// right-click.
    fn apply_and_report(&mut self) {
        if let Err(e) = self.apply() {
            self.workspace.notifications.failure(P::NAME, &e);
        }
    }

    /// Discards everything and finishes the tool.
    fn cancel(&mut self) {
        self.reset();
        self.finished = true;
    }

    /// Drops whatever is placed or being placed, ready for a first pick.
    fn reset(&mut self) {
        self.preview.cancel();
        self.placement = Placement::Idle;
    }

    /// Shows the preview at `transform`, or hides it.
    fn show(&self, transform: Option<Transform>) {
        match transform {
            Some(transform) => {
                self.preview.set_preview_transform(transform);
                self.preview.set_preview_visibility(Visibility::Visible);
            }
            None => self.preview.set_preview_visibility(Visibility::Invisible),
        }
    }

    /// What the cursor at `at` points at, looking through the preview.
    fn pointer(&self, at: (f32, f32), ctx: &EventContext) -> Pointer {
        Pointer {
            snap: self.workspace.snap(at, self.preview.preview_nodes(), ctx),
            ray: ctx.camera.ray_from_screen_point(at.0, at.1, ctx.size.0, ctx.size.1),
        }
    }
}

impl<P: Primitive> ModelingTool for PrimitiveTool<P> {
    fn info(&self) -> ToolInfo {
        P::TOOL
    }

    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Some(gesture) = Gesture::read(event, ctx.modifiers) else { return false };
        match gesture {
            // Once placed, the pointer has nothing left to pick.
            Gesture::Hover(at) => {
                if !self.is_placed() {
                    self.hover(&self.pointer(at, ctx));
                }
                false
            }
            Gesture::Click { at, .. } => {
                if !self.is_placed() {
                    self.click(&self.pointer(at, ctx));
                }
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

    /// Grips only once placed: until then the pointer is still sizing the
    /// shape.
    fn handles(&self) -> Vec<Handle> {
        match &self.placement {
            Placement::Placed(edit) => edit.handles(),
            _ => Vec::new(),
        }
    }

    fn on_handle(&mut self, event: &HandleEvent) {
        let Placement::Placed(edit) = &mut self.placement else { return };
        if edit.on_handle(event) {
            self.preview.set_preview_transform(edit.params().preview_transform());
        }
    }

    fn cursor_target(&self) -> Option<Point3> {
        if self.is_placed() { None } else { self.cursor }
    }

    fn panel_title(&self) -> Option<&str> {
        self.is_placed().then_some(P::NAME)
    }

    fn panel_ui(&mut self, ui: &mut egui::Ui, _panel: &mut PanelContext) {
        let Placement::Placed(edit) = &mut self.placement else { return };
        match edit.panel(ui) {
            PanelAction::Changed => {
                self.preview.set_preview_transform(edit.params().preview_transform());
            }
            PanelAction::Apply => self.apply_and_report(),
            PanelAction::Cancel => self.cancel(),
            PanelAction::None => {}
        }
    }

    /// A placed shape is fully defined, so leaving the tool adds it.
    fn finalize(&mut self, _selection: &mut SelectionManager) -> Result<()> {
        self.apply()
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

/// The plane a primitive's base sits on, through `first`: facing the snapped
/// geometry's direction where it has one, else the `construction` plane's way.
pub(super) fn seat(first: &Snap, construction: &Plane) -> Plane {
    Plane::from_point(first.direction.unwrap_or(construction.normal), first.position)
}

/// The width and depth, along `plane`'s basis, of the rectangle centred on
/// `center` with a corner at `corner`; `None` if it is degenerate.
pub(super) fn footprint(center: Point3, corner: Point3, plane: &Plane) -> Option<(Real, Real)> {
    let (u, v) = plane.basis();
    let offset = corner - center;
    let (width, depth) = (2.0 * offset.dot(u).abs(), 2.0 * offset.dot(v).abs());
    (width > MIN_DIMENSION && depth > MIN_DIMENSION).then_some((width, depth))
}

/// The radius from `center` out to `rim`; `None` if it is degenerate.
pub(super) fn radius_to(center: Point3, rim: Point3) -> Option<Real> {
    Some(center.distance(rim)).filter(|&radius| radius > MIN_DIMENSION)
}

/// How far along `axis` from `base` the `ray` passes closest, signed. Zero
/// where the ray runs along the axis.
pub(super) fn height_along(base: Point3, axis: Vector3, ray: &Ray) -> Real {
    ray.closest_param_on_axis(base, axis).unwrap_or(0.0)
}

/// Lays a flat unit reference shape (local XY, facing +Z) on `plane` at
/// `center`, scaled by `x` and `y` across it.
pub(super) fn flat(center: Point3, plane: &Plane, x: Real, y: Real) -> Transform {
    Transform { position: center, rotation: plane.rotation(), scale: Vector3::new(x, y, 1.0) }
}

/// A unit square face, centred on the origin in local XY.
pub(super) fn unit_square() -> Result<Shape> {
    Ok(region(Wire::rect(1.0, 1.0).context("Failed to build the unit square")?))
}

/// A unit-radius disk face, centred on the origin in local XY.
pub(super) fn unit_disk() -> Result<Shape> {
    Ok(region(circle(Point3::new(0.0, 0.0, 0.0), Vector3::unit_z(), 1.0)?))
}

/// A pointer snapped to `at`, its ray passing level through it, for the
/// primitives' tests.
#[cfg(test)]
pub(super) fn pointer_at(at: Point3) -> Pointer {
    use crate::snap::SnapKind;

    Pointer {
        snap: Some(Snap { position: at, direction: None, kind: SnapKind::ConstructionPlane }),
        ray: Ray::new(at + Vector3::unit_x() * 10.0, -Vector3::unit_x()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::Scene;

    use crate::document::Document;
    use crate::testing::{visibility, volumes, workspace};
    use crate::tools::r#box::{BoxStage, BoxTool};

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn tool() -> (BoxTool, Workspace) {
        let ws = workspace(Document::new(Scene::default()));
        (BoxTool::new(&ws), ws)
    }

    /// Picks a box based at the origin, 2 × 3 across and 4 high.
    fn place(tool: &mut BoxTool) {
        for at in [origin(), Point3::new(1.0, 0.0, 1.5), Point3::new(1.0, 4.0, 1.5)] {
            tool.hover(&pointer_at(at));
            tool.click(&pointer_at(at));
        }
    }

    fn part_count(ws: &Workspace) -> usize {
        ws.document.lock().unwrap().parts().count()
    }

    /// The volume of the one part the tool added.
    fn added_volume(ws: &Workspace) -> f64 {
        let volumes = volumes(&ws.document.lock().unwrap());
        assert_eq!(volumes.len(), 1, "one part was added");
        volumes[0]
    }

    #[test]
    fn each_click_takes_a_pick_until_the_panel_holds_the_shape() {
        let (mut tool, _ws) = tool();
        tool.click(&pointer_at(origin()));
        assert!(matches!(tool.placement, Placement::Picking(BoxStage::Footprint { .. })));
        tool.click(&pointer_at(Point3::new(1.0, 0.0, 1.5)));
        assert!(matches!(tool.placement, Placement::Picking(BoxStage::Height { .. })));
        tool.click(&pointer_at(Point3::new(1.0, 4.0, 1.5)));

        assert_eq!(tool.panel_title(), Some("Box"));
        assert_eq!(tool.handles().len(), 3);
        assert_eq!(tool.cursor_target(), None);
    }

    #[test]
    fn a_degenerate_click_takes_no_pick() {
        let (mut tool, _ws) = tool();
        tool.click(&pointer_at(origin()));
        tool.click(&pointer_at(origin()));
        assert!(matches!(tool.placement, Placement::Picking(BoxStage::Footprint { .. })));
    }

    #[test]
    fn the_preview_shows_only_while_the_pointer_sizes_it() {
        let (mut tool, ws) = tool();
        tool.click(&pointer_at(origin()));
        let preview = tool.preview.preview_node().expect("the stage is previewed");
        assert_eq!(visibility(&ws, preview), Visibility::Invisible);

        tool.hover(&pointer_at(Point3::new(1.0, 0.0, 1.5)));
        assert_eq!(visibility(&ws, preview), Visibility::Visible);
        tool.hover(&pointer_at(origin()));
        assert_eq!(visibility(&ws, preview), Visibility::Invisible);
    }

    #[test]
    fn finishing_adds_the_placed_shape_and_leaves() {
        let (mut tool, ws) = tool();
        place(&mut tool);

        assert!(tool.finish());
        assert!(tool.is_finished());
        assert!(tool.preview.is_empty());
        assert!((added_volume(&ws) - 24.0).abs() < 1e-6);
    }

    #[test]
    fn finishing_mid_pick_drops_the_picks_and_stays() {
        let (mut tool, ws) = tool();
        tool.click(&pointer_at(origin()));

        assert!(tool.finish());
        assert!(matches!(tool.placement, Placement::Idle));
        assert!(tool.preview.is_empty());
        assert!(!tool.is_finished());
        assert_eq!(part_count(&ws), 0);
        // Nothing picked is nothing to finish.
        assert!(!tool.finish());
    }

    #[test]
    fn cancelling_discards_the_shape_and_leaves() {
        let (mut tool, ws) = tool();
        place(&mut tool);

        tool.cancel();
        assert!(tool.is_finished());
        assert!(tool.preview.is_empty());
        assert_eq!(part_count(&ws), 0);
    }

    #[test]
    fn leaving_the_tool_adds_only_a_placed_shape() {
        let (mut tool, ws) = tool();
        tool.click(&pointer_at(origin()));
        tool.finalize(&mut SelectionManager::new()).expect("nothing to add");
        tool.deactivate();
        assert_eq!(part_count(&ws), 0);

        place(&mut tool);
        tool.finalize(&mut SelectionManager::new()).expect("the box is added");
        assert_eq!(part_count(&ws), 1);
    }

    #[test]
    fn once_placed_the_pointer_changes_nothing() {
        let (mut tool, ws) = tool();
        place(&mut tool);
        tool.hover(&pointer_at(Point3::new(5.0, 7.0, 5.0)));
        tool.click(&pointer_at(Point3::new(5.0, 7.0, 5.0)));

        assert!(tool.finish());
        assert!((added_volume(&ws) - 24.0).abs() < 1e-6);
    }

    #[test]
    fn a_footprint_flat_along_either_axis_is_degenerate() {
        assert!(footprint(origin(), origin(), &Plane::xz()).is_none());
        assert!(footprint(origin(), Point3::new(1.0, 0.0, 0.0), &Plane::xz()).is_none());
        assert!(footprint(origin(), Point3::new(0.0, 0.0, 1.0), &Plane::xz()).is_none());
    }

    #[test]
    fn the_unit_references_build() {
        assert!(unit_square().is_ok());
        assert!(unit_disk().is_ok());
    }
}
