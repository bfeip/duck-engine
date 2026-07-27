use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::SubGeometryKind;
use duck_engine_viewer::common::{decompose_matrix, InnerSpace, Point3, Quaternion, Vector3};
use duck_engine_viewer::event::{Event, EventContext};
use duck_engine_viewer::operator::{
    NodeTransformTarget, Operator, SelectionKinds, SelectionMode, TransformCaps, TransformDriver,
    TransformFrame, TransformInteraction, TransformMode, TransformTarget,
};
use duck_engine_viewer::scene::{NodeId, Scene};
use duck_engine_viewer::selection::SelectionItem;
use opencascade::primitives::FaceOrientation;

use crate::document::{Document, PartId};
use crate::notifications::Notifications;
use crate::preview::PreviewSession;
use crate::tool::{ModelingTool, ToolInfo};
use crate::ui::icons;
use super::ConstructionOptions;

/// CAD-aware transform tool for one operation (move, rotate, *or* scale). Each
/// [`TransformMode`] is registered as its own palette tool.
///
/// The tool is a viewer [`TransformDriver`] over a [`ModelerTarget`], which
/// routes between two transform grains based on the selection:
///
/// - **Whole parts** (node selection) transform through the viewer's
///   [`NodeTransformTarget`] — the drag previews by mutating the *scene node*
///   transform, which moves the tessellated mesh but leaves the CAD B-Rep
///   untouched. On commit the node's transform is baked into the part's
///   geometry via [`Document::bake_transform`].
/// - **Faces** (a face drilled into as the primary selection) transform
///   through [`FaceTweakTarget`]: the drag moves a ghost of the picked face,
///   and committing re-solves the B-Rep around the transformed face via
///   [`Document::tweak_faces`].
///
/// Selecting the tool shows its handle set immediately; drag a handle to
/// transform. The mode's key (G/R/S) starts a freeform transform and X/Y/Z
/// start an axis-constrained one; left-click or Enter confirms, right-click or
/// Escape cancels.
pub struct TransformTool {
    mode: TransformMode,
    driver: TransformDriver<ModelerTarget>,
    /// Held only to reach the scene on [`ModelingTool::deactivate`], which has
    /// no event context.
    document: Arc<Mutex<Document>>,
}

impl TransformTool {
    pub fn new(
        mode: TransformMode,
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        let target = ModelerTarget {
            nodes: NodeTransformTarget::new(),
            face: FaceTweakTarget::new(Arc::clone(&document)),
            active: None,
            construction_options,
            document: Arc::clone(&document),
            notifications,
        };
        Self {
            mode,
            driver: TransformDriver::with_target(mode, target),
            document,
        }
    }
}

impl Operator for TransformTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        self.driver.dispatch(event, ctx)
    }

    fn name(&self) -> &str {
        "TransformTool"
    }
}

impl ModelingTool for TransformTool {
    fn info(&self) -> ToolInfo {
        match self.mode {
            TransformMode::Translate => ToolInfo { id: "move", icon: icons::MOVE, shortcut: Some('g') },
            TransformMode::Rotate => ToolInfo { id: "rotate", icon: icons::ROTATE, shortcut: Some('r') },
            TransformMode::Scale => ToolInfo { id: "scale", icon: icons::SCALE, shortcut: Some('s') },
        }
    }

    fn activate(&mut self) {
        // Show this mode's handle set as soon as the tool is selected; the next
        // frame syncs it onto the current selection.
        self.driver.set_gizmo_enabled(true);
    }

    fn deactivate(&mut self) {
        // Abort any in-progress transform and remove gizmo/ghost nodes.
        let scene_arc = self.document.lock().unwrap().scene().clone();
        self.driver.teardown(&scene_arc);
    }

    /// Whole parts first; a second click drills into a face for tweaking.
    fn selection_mode(&self) -> SelectionMode {
        SelectionMode::Progressive(SelectionKinds::FACE)
    }
}

/// The picked face, identified the way the selection system reports it: by
/// tessellation order within a part's mesh.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FaceTarget {
    node: NodeId,
    part: PartId,
    face_index: u32,
}

/// Which transform grain an in-progress drag locked onto at
/// [`TransformTarget::begin`].
enum ActiveRoute {
    Nodes,
    Face(FaceTarget),
}

/// [`TransformTarget`] routing between whole-part node transforms and single
/// face tweaks based on the current selection.
///
/// The route is resolved from the selection in [`frame`](TransformTarget::frame)
/// (so the idle gizmo follows the selection grain) and locked in
/// [`begin`](TransformTarget::begin); preview/commit/cancel always act on the
/// locked route, so a drag can never change grain midway.
struct ModelerTarget {
    nodes: NodeTransformTarget,
    face: FaceTweakTarget,
    active: Option<ActiveRoute>,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    notifications: Notifications,
}

impl ModelerTarget {
    /// The face currently drilled into as the primary selection, if it belongs
    /// to a known CAD part.
    fn selected_face(&self, ctx: &EventContext) -> Option<FaceTarget> {
        let SelectionItem::SubGeometry { node_id, element } = ctx.selection.primary()? else {
            return None;
        };
        if element.kind != SubGeometryKind::Face {
            return None;
        }
        let part = self.document.lock().unwrap().part_for_node(node_id)?;
        Some(FaceTarget { node: node_id, part, face_index: element.index })
    }

    /// Snapshot of the tessellation options for one begin/commit.
    fn geometry_options(&self) -> CadTessellationOptions {
        self.construction_options.borrow().geometry_options.clone()
    }

    /// Bake every committed node's transform into its CAD part.
    fn bake_nodes(&mut self, nodes: &[NodeId], ctx: &mut EventContext) {
        let options = self.geometry_options();
        let mut doc = self.document.lock().unwrap();
        // One undo step covers a multi-select bake.
        let mut doc = doc.undo_scope("Transform");
        for &node in nodes {
            let Some(part) = doc.part_for_node(node) else { continue };
            // Stay in common's matrix type; the OCCT array conversion happens
            // inside bake_transform.
            let delta = ctx
                .scene
                .lock()
                .unwrap()
                .get_node(node)
                .map(|n| n.transform().to_matrix());
            if let Some(delta) = delta
                && let Err(e) = doc.bake_transform(part, delta, &options) {
                    log::error!("transform bake failed for node {node:?}: {e}");
                    self.notifications.error(format!("Transform failed: {e}"));
                }
        }
    }
}

impl TransformTarget for ModelerTarget {
    fn frame(&mut self, ctx: &mut EventContext) -> Option<TransformFrame> {
        match self.selected_face(ctx) {
            Some(target) => self.face.frame_for(target),
            None => self.nodes.frame(ctx),
        }
    }

    fn caps(&self) -> TransformCaps {
        match self.active {
            // A non-uniform similarity can't be baked into the B-Rep.
            Some(ActiveRoute::Face(_)) => TransformCaps { uniform_scale_only: true },
            _ => TransformCaps::default(),
        }
    }

    fn begin(&mut self, ctx: &mut EventContext) -> bool {
        self.active = match self.selected_face(ctx) {
            Some(target) => {
                let options = self.geometry_options();
                self.face.begin(target, &options).then_some(ActiveRoute::Face(target))
            }
            None => self.nodes.begin(ctx).then_some(ActiveRoute::Nodes),
        };
        self.active.is_some()
    }

    fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        match self.active {
            Some(ActiveRoute::Nodes) => self.nodes.preview(interaction, ctx),
            Some(ActiveRoute::Face(_)) => self.face.preview(interaction, ctx),
            None => {}
        }
    }

    fn commit(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        match self.active.take() {
            Some(ActiveRoute::Nodes) => {
                let nodes = self.nodes.nodes();
                self.nodes.commit(interaction, ctx);
                self.bake_nodes(&nodes, ctx);
            }
            Some(ActiveRoute::Face(target)) => {
                let options = self.geometry_options();
                self.face.commit(target, interaction, ctx, &options, &self.notifications);
            }
            None => {}
        }
    }

    fn cancel(&mut self, ctx: &mut EventContext) {
        match self.active.take() {
            Some(ActiveRoute::Nodes) => self.nodes.cancel(ctx),
            Some(ActiveRoute::Face(_)) => self.face.cancel(),
            None => {}
        }
    }

    fn abort(&mut self, scene: &Arc<Mutex<Scene>>) {
        self.nodes.abort(scene);
        self.face.abort();
        self.active = None;
    }
}

/// Face half of [`ModelerTarget`]: interactive transform of one face of a
/// solid.
///
/// The drag is previewed cheaply by rigidly transforming a ghost tessellation
/// of just the picked face — no OCCT work per frame. Committing runs the
/// actual tweak ([`Document::tweak_faces`]), which re-solves the body around
/// the moved face and re-tessellates in place; a tweak the kernel cannot
/// re-solve leaves the part untouched and reports the error.
struct FaceTweakTarget {
    /// Ghost of the picked face shown while dragging.
    preview: PreviewSession,
    /// Pivot/frame resolved from the last-targeted face (`None` when the face
    /// has no usable pivot), cached because gizmo syncing runs every update.
    resolved: Option<(FaceTarget, Option<(Point3, Quaternion)>)>,
    document: Arc<Mutex<Document>>,
}

impl FaceTweakTarget {
    fn new(document: Arc<Mutex<Document>>) -> Self {
        Self {
            preview: PreviewSession::new(Arc::clone(&document)),
            resolved: None,
            document,
        }
    }

    /// Pivot (face centroid) and frame (local Z = outward face normal) for
    /// `target`, resolved from the B-Rep and cached.
    fn resolve(&mut self, target: FaceTarget) -> Option<(Point3, Quaternion)> {
        if let Some((cached, resolved)) = self.resolved
            && cached == target {
                return resolved;
            }
        let resolved = (|| {
            let doc = self.document.lock().unwrap();
            let face = doc.face_subshape(target.node, target.face_index)?;
            // `normal_at_center` returns the surface's parametric normal, which
            // points inward for Reversed faces — flip it so local Z points out.
            let sign = if face.orientation() == FaceOrientation::Forward { 1.0 } else { -1.0 };
            let n = match face.normal_at_center() {
                Ok(n) => n,
                Err(e) => {
                    log::warn!("Tweak target has no usable pivot: {e}");
                    return None;
                }
            };
            let normal = (Vector3::new(n.x as f32, n.y as f32, n.z as f32) * sign).normalize();
            let c = face.center_of_mass();
            let pivot = Point3::new(c.x as f32, c.y as f32, c.z as f32);
            let frame = Quaternion::from_arc(Vector3::unit_z(), normal, None);
            Some((pivot, frame))
        })();
        self.resolved = Some((target, resolved));
        resolved
    }

    /// The gizmo anchor for `target`, if the face can be resolved.
    fn frame_for(&mut self, target: FaceTarget) -> Option<TransformFrame> {
        let (pivot, frame) = self.resolve(target)?;
        Some(TransformFrame { pivot, frame_rotation: Some(frame) })
    }

    /// Lock onto `target`: ghost the face for the upcoming drag. Fails
    /// (refusing the start) if the face can't be resolved or ghosted.
    fn begin(&mut self, target: FaceTarget, options: &CadTessellationOptions) -> bool {
        if self.resolve(target).is_none() {
            log::warn!("Tweak target could not be resolved to a CAD face");
            return false;
        }

        let ghost_shape = {
            let doc = self.document.lock().unwrap();
            doc.face_subshape(target.node, target.face_index)
                .map(opencascade::primitives::Shape::from)
        };
        let Some(ghost_shape) = ghost_shape else { return false };
        if self.preview.add_preview_from_shape(&ghost_shape, options, "Tweak preview").is_none() {
            log::warn!("Tweak preview could not be tessellated");
            return false;
        }
        true
    }

    /// Move the ghost to the interaction's current delta.
    fn preview(&mut self, interaction: &TransformInteraction, ctx: &mut EventContext) {
        let Some(ghost) = self.preview.preview_node() else { return };
        let camera = ctx.camera();
        let delta = interaction.delta_matrix(&camera, ctx.size);
        let transform = decompose_matrix(&delta);
        ctx.scene.lock().unwrap().set_node_transform(ghost, transform);
    }

    /// Run the tweak on the B-Rep. On success the selection is cleared — the
    /// part re-tessellates, so face indices no longer identify the same
    /// sub-geometry.
    fn commit(
        &mut self,
        target: FaceTarget,
        interaction: &TransformInteraction,
        ctx: &mut EventContext,
        options: &CadTessellationOptions,
        notifications: &Notifications,
    ) {
        let camera = ctx.camera();
        let delta = interaction.delta_matrix(&camera, ctx.size);

        self.preview.cancel();
        self.resolved = None;

        let result = self.document.lock().unwrap().tweak_faces(
            target.part,
            &[target.face_index],
            delta,
            options,
        );
        match result {
            Ok(()) => ctx.selection.clear(),
            Err(e) => {
                log::error!("face tweak failed: {e}");
                notifications.error(format!("Face tweak failed: {e}"));
            }
        }
    }

    /// Discard the drag without touching the B-Rep.
    fn cancel(&mut self) {
        self.preview.cancel();
    }

    /// Drop everything held for the current face (route change or tool
    /// deactivation). Idempotent.
    fn abort(&mut self) {
        self.preview.cancel();
        self.resolved = None;
    }
}
