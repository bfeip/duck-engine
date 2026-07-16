use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::CadTessellationOptions;
use duck_engine_scene::SubGeometryKind;
use duck_engine_viewer::common::{decompose_matrix, Axis, InnerSpace, Point3, Quaternion, Vector3};
use duck_engine_viewer::event::{AppEvent, DeviceEvent, Event, EventContext};
use duck_engine_viewer::gizmo::GizmoState;
use duck_engine_viewer::input::{ElementState, Modifiers};
use duck_engine_viewer::operator::{
    axis_from_constraint, AxisConstraint, Operator, SelectionKinds, SelectionMode, TransformAction,
    TransformAnnotations, TransformInteraction, TransformMode, TransformOperator,
};
use duck_engine_viewer::scene::NodeId;
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
/// Selection is progressive: the first click selects a whole part, a second
/// click drills into a face. The two grains transform differently:
///
/// - **Whole parts** go through the viewer's [`TransformOperator`], which does
///   all the interactive work (gizmos, axis constraints, mouse math, preview,
///   confirm/cancel) by mutating the *scene node* transform. That moves the
///   tessellated mesh but leaves the underlying CAD B-Rep untouched, so on
///   every confirmed transform we bake the node's final transform into the
///   part's geometry via [`Document::bake_transform`].
/// - **Faces** go through [`FaceTweak`], built from the same interaction
///   units: the drag moves a ghost of the picked face, and confirming re-solves
///   the B-Rep around the transformed face via [`Document::tweak_faces`].
///
/// Selecting the tool shows its handle set immediately; drag a handle to
/// transform. The mode's key (G/R/S) starts a freeform transform and X/Y/Z
/// start an axis-constrained one; left-click or Enter confirms, right-click or
/// Escape cancels.
pub struct TransformTool {
    mode: TransformMode,
    transform_op: TransformOperator,
    face: FaceTweak,
    /// Whether events are currently routed to the face path (primary selection
    /// is a face) rather than the node path.
    routing_face: bool,
    construction_options: Rc<RefCell<ConstructionOptions>>,
    document: Arc<Mutex<Document>>,
    notifications: Notifications,
}

impl TransformTool {
    pub fn new(
        mode: TransformMode,
        construction_options: Rc<RefCell<ConstructionOptions>>,
        document: Arc<Mutex<Document>>,
        notifications: Notifications,
    ) -> Self {
        Self {
            mode,
            transform_op: TransformOperator::new(mode),
            face: FaceTweak::new(mode, Arc::clone(&document), notifications.clone()),
            routing_face: false,
            construction_options,
            document,
            notifications,
        }
    }

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

    /// Tear down whichever path is being left when the selection grain changes.
    fn set_route(&mut self, face: bool, ctx: &mut EventContext) {
        if self.routing_face == face {
            return;
        }
        self.routing_face = face;
        if face {
            // Abort any node transform and hide its gizmo while a face is targeted.
            self.transform_op.teardown(&mut ctx.scene.lock().unwrap());
        } else {
            self.face.teardown();
            self.transform_op.set_gizmo_enabled(true);
        }
    }

    /// Bake every just-confirmed node's transform into its CAD part.
    fn bake_committed(&mut self, nodes: &[NodeId], ctx: &mut EventContext) {
        let options = self.construction_options.borrow().geometry_options.clone();
        let mut doc = self.document.lock().unwrap();
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
            if let Some(delta) = delta {
                if let Err(e) = doc.bake_transform(part, delta, &options) {
                    log::error!("transform bake failed for node {node:?}: {e}");
                    self.notifications.error(format!("Transform failed: {e}"));
                }
            }
        }
    }
}

impl Operator for TransformTool {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        // The inner operator emits `AppEvent::TransformCommitted` on confirm; the
        // dispatcher re-dispatches it back to us. Bake the committed nodes when we see
        // it.
        if let Event::App(AppEvent::TransformCommitted { nodes }) = event {
            self.bake_committed(nodes, ctx);
            return false;
        }

        let target = self.selected_face(ctx);
        self.set_route(target.is_some(), ctx);
        match target {
            Some(target) => {
                let options = self.construction_options.borrow().geometry_options.clone();
                self.face.dispatch(target, event, ctx, &options)
            }
            None => self.transform_op.dispatch(event, ctx),
        }
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
        self.transform_op.set_gizmo_enabled(true);
    }

    fn deactivate(&mut self) {
        self.face.teardown();
        self.routing_face = false;
        // Abort any in-progress transform and remove gizmo/annotation nodes.
        let scene_arc = self.document.lock().unwrap().scene().clone();
        self.transform_op.teardown(&mut scene_arc.lock().unwrap());
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

/// Interactive transform of one face of a solid, sharing the node operator's
/// interaction units (gizmo, constraint cycling, mouse math, annotations).
///
/// The drag is previewed cheaply by rigidly transforming a ghost tessellation
/// of just the picked face — no OCCT work per frame. Confirming runs the
/// actual tweak ([`Document::tweak_faces`]), which re-solves the body around
/// the moved face and re-tessellates in place; a tweak the kernel cannot
/// re-solve leaves the part untouched and reports the error.
struct FaceTweak {
    interaction: TransformInteraction,
    gizmo: GizmoState,
    annotations: TransformAnnotations,
    /// Ghost of the picked face shown while dragging.
    preview: PreviewSession,
    /// The face locked at drag start.
    active_target: Option<FaceTarget>,
    /// Pivot/frame resolved from the last-targeted face (`None` when the face
    /// has no usable pivot), cached because gizmo syncing runs every update.
    resolved: Option<(FaceTarget, Option<(Point3, Quaternion)>)>,
    document: Arc<Mutex<Document>>,
    notifications: Notifications,
}

impl FaceTweak {
    fn new(mode: TransformMode, document: Arc<Mutex<Document>>, notifications: Notifications) -> Self {
        Self {
            interaction: TransformInteraction::new(mode),
            gizmo: GizmoState::new(),
            annotations: TransformAnnotations::new(),
            preview: PreviewSession::new(Arc::clone(&document)),
            active_target: None,
            resolved: None,
            document,
            notifications,
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

    /// Lock onto `target` and begin a transform: ghost the face and activate
    /// the interaction about the face pivot. Fails silently (staying idle) if
    /// the face can't be resolved or ghosted.
    fn start(&mut self, target: FaceTarget, ctx: &mut EventContext, options: &CadTessellationOptions) {
        let Some((pivot, frame)) = self.resolve(target) else {
            log::warn!("Tweak target could not be resolved to a CAD face");
            return;
        };

        let ghost_shape = {
            let doc = self.document.lock().unwrap();
            doc.face_subshape(target.node, target.face_index).map(opencascade::primitives::Shape::from)
        };
        let Some(ghost_shape) = ghost_shape else { return };
        if self.preview.add_preview_from_shape(&ghost_shape, options, "Tweak preview").is_none() {
            log::warn!("Tweak preview could not be tessellated");
            return;
        }

        let model_radius = {
            let scene = ctx.scene.lock().unwrap();
            scene
                .bounding()
                .bounds
                .as_ref()
                .map(|aabb| aabb.bounding_sphere_radius())
                .filter(|r| *r > 0.0)
                .unwrap_or(1.0)
        };
        self.interaction.start(pivot, Some(frame), model_radius);
        self.active_target = Some(target);
    }

    /// Move the ghost to the interaction's current delta.
    fn update_ghost(&mut self, ctx: &mut EventContext) {
        let Some(ghost) = self.preview.preview_node() else { return };
        let camera = ctx.camera();
        let delta = self.interaction.delta_matrix(&camera, ctx.size);
        let transform = decompose_matrix(&delta);
        ctx.scene.lock().unwrap().set_node_transform(ghost, transform);
    }

    /// Run the tweak on the B-Rep and clear the drag state. On success the
    /// selection is cleared — the part re-tessellates, so face indices no
    /// longer identify the same sub-geometry.
    fn confirm(&mut self, ctx: &mut EventContext, options: &CadTessellationOptions) {
        let Some(target) = self.active_target else { return };
        let camera = ctx.camera();
        let delta = self.interaction.delta_matrix(&camera, ctx.size);

        self.finish_drag(ctx);
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
                self.notifications.error(format!("Face tweak failed: {e}"));
            }
        }
    }

    /// Discard the drag without touching the B-Rep.
    fn cancel(&mut self, ctx: &mut EventContext) {
        self.finish_drag(ctx);
    }

    /// Remove the ghost and per-drag visuals and deactivate the interaction.
    fn finish_drag(&mut self, ctx: &mut EventContext) {
        self.preview.cancel();
        {
            let mut scene = ctx.scene.lock().unwrap();
            self.annotations.clear(&mut scene);
            self.gizmo.set_highlight(None, &mut scene);
        }
        self.interaction.finish();
        self.active_target = None;
    }

    /// Remove everything this path owns in the scene (route change or tool
    /// deactivation). Idempotent.
    fn teardown(&mut self) {
        self.preview.cancel();
        let scene_arc = self.document.lock().unwrap().scene().clone();
        let mut scene = scene_arc.lock().unwrap();
        self.annotations.clear(&mut scene);
        self.gizmo.hide(&mut scene);
        self.interaction.finish();
        self.active_target = None;
        self.resolved = None;
    }

    /// Show or reposition the gizmo at the targeted face's pivot.
    fn sync_gizmo(&mut self, target: FaceTarget, ctx: &mut EventContext) {
        let gizmo_type = self.interaction.mode().gizmo_type();
        let Some((pivot, _)) = self.resolve(target) else {
            let mut scene = ctx.scene.lock().unwrap();
            self.gizmo.hide(&mut scene);
            return;
        };
        let mut scene = ctx.scene.lock().unwrap();
        if self.gizmo.current_type() == Some(gizmo_type) {
            self.gizmo.update_position(pivot, &mut scene);
        } else {
            let model_radius = scene
                .bounding()
                .bounds
                .as_ref()
                .map(|aabb| aabb.bounding_sphere_radius())
                .filter(|r| *r > 0.0)
                .unwrap_or(1.0);
            self.gizmo.show(gizmo_type, pivot, model_radius * 0.15, &mut scene);
        }
    }

    /// Handle an axis key: start constrained if idle, otherwise cycle. Scale
    /// tweaks are uniform-only (a non-uniform similarity can't be baked into
    /// the B-Rep), so axis keys are accepted but don't constrain.
    fn constrain_axis(
        &mut self,
        axis: char,
        target: FaceTarget,
        ctx: &mut EventContext,
        options: &CadTessellationOptions,
    ) -> bool {
        if !self.interaction.is_active() {
            self.start(target, ctx, options);
            if !self.interaction.is_active() {
                return false;
            }
        }
        if self.interaction.mode() == TransformMode::Scale {
            return true;
        }
        self.interaction.cycle_axis_constraint(axis);
        self.update_ghost(ctx);
        let highlight = axis_from_constraint(&self.interaction.axis_constraint());
        let mut scene = ctx.scene.lock().unwrap();
        self.annotations.update(&self.interaction, &mut scene);
        self.gizmo.set_highlight(highlight, &mut scene);
        true
    }

    fn dispatch(
        &mut self,
        target: FaceTarget,
        event: &Event,
        ctx: &mut EventContext,
        options: &CadTessellationOptions,
    ) -> bool {
        let Event::Device(event) = event else { return false };
        match event {
            DeviceEvent::KeyboardInput { event: key_event, .. } => {
                if key_event.state != ElementState::Pressed || key_event.repeat {
                    return false;
                }
                let actions = self.interaction.bindings
                    .actions_for_key(&key_event.logical_key, ctx.modifiers)
                    .to_vec();
                for action in actions {
                    match action {
                        TransformAction::StartTransform if !self.interaction.is_active() => {
                            self.start(target, ctx, options);
                            return self.interaction.is_active();
                        }
                        TransformAction::ConstrainX => {
                            if self.constrain_axis('x', target, ctx, options) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainY => {
                            if self.constrain_axis('y', target, ctx, options) {
                                return true;
                            }
                        }
                        TransformAction::ConstrainZ => {
                            if self.constrain_axis('z', target, ctx, options) {
                                return true;
                            }
                        }
                        TransformAction::KeyConfirm if self.interaction.is_active() => {
                            self.confirm(ctx, options);
                            return true;
                        }
                        TransformAction::KeyCancel if self.interaction.is_active() => {
                            self.cancel(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            DeviceEvent::MouseMotion { delta } => {
                if self.interaction.is_active() {
                    self.interaction.accumulate(delta.0 as f32, delta.1 as f32);
                    self.update_ghost(ctx);
                    true
                } else {
                    false
                }
            }

            DeviceEvent::MouseClick { button, .. } => {
                if !self.interaction.is_active() {
                    return false;
                }
                let actions =
                    self.interaction.bindings.actions_for_click(*button, ctx.modifiers).to_vec();
                for action in actions {
                    match action {
                        TransformAction::MouseConfirm => {
                            self.confirm(ctx, options);
                            return true;
                        }
                        TransformAction::MouseCancel => {
                            self.cancel(ctx);
                            return true;
                        }
                        _ => {}
                    }
                }
                false
            }

            DeviceEvent::MouseDragStart { button, start_pos, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag_start(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.interaction.is_active() || !self.gizmo.has_gizmo() {
                    return false;
                }
                let picked = {
                    let ray = ctx.camera().ray_from_screen_point(
                        start_pos.0, start_pos.1, ctx.size.0, ctx.size.1,
                    );
                    let scene = ctx.scene.lock().unwrap();
                    self.gizmo.pick_handle(ray, &scene)
                };
                if let Some(axis) = picked {
                    self.start(target, ctx, options);
                    if !self.interaction.is_active() {
                        return false;
                    }
                    // Scale stays uniform; the handle only starts the drag.
                    if self.interaction.mode() != TransformMode::Scale {
                        self.interaction.set_axis_constraint(match axis {
                            Axis::X => AxisConstraint::WorldX,
                            Axis::Y => AxisConstraint::WorldY,
                            Axis::Z => AxisConstraint::WorldZ,
                        });
                    }
                    let mut scene = ctx.scene.lock().unwrap();
                    self.gizmo.set_highlight(Some(axis), &mut scene);
                    self.annotations.update(&self.interaction, &mut scene);
                    return true;
                }
                false
            }

            DeviceEvent::MouseDrag { button, delta, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag(*button, ctx.modifiers)
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.interaction.is_active() {
                    self.interaction.accumulate(delta.0, delta.1);
                    self.update_ghost(ctx);
                    return true;
                }
                false
            }

            DeviceEvent::MouseDragEnd { button, .. } => {
                if !self.interaction.bindings
                    .actions_for_drag_end(*button, Modifiers::default())
                    .contains(&TransformAction::GizmoDrag)
                {
                    return false;
                }
                if self.interaction.is_active() {
                    self.confirm(ctx, options);
                    return true;
                }
                false
            }

            DeviceEvent::CursorMoved { position } => {
                // Hover highlight on gizmo handles while idle.
                if self.gizmo.has_gizmo() && !self.interaction.is_active() {
                    let ray = ctx.camera().ray_from_screen_point(
                        position.0 as f32, position.1 as f32, ctx.size.0, ctx.size.1,
                    );
                    let mut scene = ctx.scene.lock().unwrap();
                    let axis = self.gizmo.pick_handle(ray, &scene);
                    self.gizmo.set_highlight(axis, &mut scene);
                }
                false
            }

            DeviceEvent::Update { .. } => {
                if !self.interaction.is_active() {
                    self.sync_gizmo(target, ctx);
                }
                false
            }

            _ => false,
        }
    }
}


