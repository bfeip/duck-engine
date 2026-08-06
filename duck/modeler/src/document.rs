use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use anyhow::{Context, Result};
use duck_engine_scene::resource::{Id, NodeId, Visibility};
use duck_engine_scene::Scene;
use duck_engine_scene::cad::{CadTessellationOptions, retessellate_node, tessellate_into};
use duck_engine_scene::common::{matrix4_to_row_major_f64, Matrix4, Transform};
use opencascade::primitives::{Edge, Face, Shape, ShapeType, Wire};

use crate::history::{Delta, History, PartSnapshot};

pub type PartId = Id;

/// Topological classification of a part, derived from its B-Rep shape type.
/// A presentation-level version of OCCT [`ShapeType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartKind {
    Solid,
    Shell,
    Face,
    Wire,
    Point,
    Compound,
    Other,
}

impl PartKind {
    /// Short uppercase badge label, e.g. "SOLID".
    pub fn label(self) -> &'static str {
        match self {
            PartKind::Solid => "SOLID",
            PartKind::Shell => "SHELL",
            PartKind::Face => "FACE",
            PartKind::Wire => "WIRE",
            PartKind::Point => "POINT",
            PartKind::Compound => "COMPOUND",
            PartKind::Other => "SHAPE",
        }
    }
}

pub struct CadPart {
    pub id: PartId,
    pub name: String,
    pub shape: Shape,
    /// Tessellation options the part was last (re)tessellated with, kept for
    /// history capture — `remove_part` receives none from its caller.
    options: CadTessellationOptions,
}

impl CadPart {
    /// Classify the part by its top-level B-Rep shape type.
    pub fn kind(&self) -> PartKind {
        match self.shape.shape_type() {
            ShapeType::Solid | ShapeType::CompoundSolid => PartKind::Solid,
            ShapeType::Shell => PartKind::Shell,
            ShapeType::Face => PartKind::Face,
            ShapeType::Wire | ShapeType::Edge => PartKind::Wire,
            ShapeType::Vertex => PartKind::Point,
            ShapeType::Compound => PartKind::Compound,
            ShapeType::Shape => PartKind::Other,
        }
    }
}

pub struct Document {
    parts: Vec<CadPart>,
    part_to_node: HashMap<PartId, NodeId>,
    node_to_part: HashMap<NodeId, PartId>,
    scene: Scene,
    history: History,
}

impl Document {
    pub fn new(scene: Scene) -> Self {
        Self {
            parts: Vec::new(),
            part_to_node: HashMap::new(),
            node_to_part: HashMap::new(),
            scene,
            history: History::default(),
        }
    }

    pub fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.history.clear();
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Tessellate `shape`, add the resulting node to the scene, store the CAD part,
    /// and record the mapping — all atomically. If tessellation fails, nothing is modified.
    pub fn add_part(
        &mut self,
        name: impl Into<String>,
        shape: Shape,
        options: &CadTessellationOptions,
    ) -> Result<PartId> {
        let name = name.into();
        let node = tessellate_into(&shape, &self.scene, options, None, Some(&name))
            .context("Failed to tessellate part")?
            .id();
        let id = PartId::new();
        self.parts.push(CadPart { id, name, shape, options: options.clone() });
        self.part_to_node.insert(id, node);
        self.node_to_part.insert(node, id);
        if let Some(snapshot) = self.snapshot_part(id) {
            let label = format!("Add {}", self.get_part(id).unwrap().name);
            self.history.record(&label, Delta::Added(snapshot));
        }
        Ok(id)
    }

    /// Remove a part from the CAD store, the mapping, and the scene tree.
    ///
    /// The recorded undo snapshot keeps the detached node chain alive until
    /// the step leaves history; then its mesh and materials are freed too.
    pub fn remove_part(&mut self, id: PartId) {
        if let Some(snapshot) = self.snapshot_part(id) {
            let label = format!("Delete {}", snapshot.name);
            self.history.record(&label, Delta::Removed(snapshot));
        }
        self.remove_part_inner(id);
    }

    /// [`remove_part`](Self::remove_part) without history recording, shared
    /// with undo/redo replay.
    fn remove_part_inner(&mut self, id: PartId) {
        self.parts.retain(|p| p.id != id);
        if let Some(node) = self.part_to_node.remove(&id) {
            self.node_to_part.remove(&node);
            self.scene.remove_node(node);
        }
    }

    pub fn get_part(&self, id: PartId) -> Option<&CadPart> {
        self.parts.iter().find(|p| p.id == id)
    }

    pub fn get_part_mut(&mut self, id: PartId) -> Option<&mut CadPart> {
        self.parts.iter_mut().find(|p| p.id == id)
    }

    /// Current visibility of the part's scene node, or `None` if the part is unknown.
    pub fn part_visibility(&self, id: PartId) -> Option<Visibility> {
        let node = self.node_for_part(id)?;
        let scene = self.scene.lock();
        scene.get_node(node).map(|n| n.visibility())
    }

    /// Set the visibility of the part's scene node. No-op for an unknown part.
    pub fn set_part_visibility(&mut self, id: PartId, visibility: Visibility) {
        if let Some(node) = self.node_for_part(id) {
            self.scene.set_node_visibility(node, visibility);
        }
    }

    /// Bake a transform into the part's CAD geometry then
    /// re-tessellate the part in place (preserving its `NodeId`) and reset the node
    /// transform to identity.
    pub fn bake_transform(
        &mut self,
        part: PartId,
        transform: Matrix4,
        options: &CadTessellationOptions,
    ) -> Result<()> {
        let node = self
            .node_for_part(part)
            .context("bake_transform: no node for part")?;

        // Transpose common's column-major f32 matrix into OCCT's row-major f64 array.
        let mat = matrix4_to_row_major_f64(&transform);

        let before = {
            let cad_part = self
                .get_part_mut(part)
                .context("bake_transform: part not found")?;
            let before = cad_part.shape.clone();
            // A similarity keeps surfaces analytic (planes stay planes); only a
            // non-uniform scale needs the B-spline-converting general transform.
            cad_part.shape = match cad_part.shape.transformed(mat) {
                Ok(shape) => shape,
                Err(_) => cad_part.shape.gtransform(mat),
            };
            cad_part.options = options.clone();
            before
        };

        let after = {
            let cad_part = self
                .get_part(part)
                .context("bake_transform: part not found")?;
            retessellate_node(&cad_part.shape, &self.scene, options, node)?;
            self.scene.set_node_transform(node, Transform::IDENTITY);
            cad_part.shape.clone()
        };

        self.history.record(
            "Transform",
            Delta::Reshaped { part, before, after, options: options.clone() },
        );

        Ok(())
    }

    /// Transform the given faces (by tessellation index) of a part's B-Rep and
    /// re-solve the body around them, then re-tessellate the part in place
    /// (preserving its `NodeId`). The part is untouched on error.
    ///
    /// `transform` must be a similarity (rotation + translation + uniform
    /// scale); a tweak the body cannot re-solve reports an error.
    pub fn tweak_faces(
        &mut self,
        part: PartId,
        face_indices: &[u32],
        transform: Matrix4,
        options: &CadTessellationOptions,
    ) -> Result<()> {
        let node = self
            .node_for_part(part)
            .context("tweak_faces: no node for part")?;

        let tweaked = {
            let cad_part = self.get_part(part).context("tweak_faces: part not found")?;
            let faces: Vec<_> = face_indices
                .iter()
                .map(|&index| {
                    cad_part
                        .shape
                        .faces()
                        .nth(index as usize)
                        .with_context(|| format!("tweak_faces: no face at index {index}"))
                })
                .collect::<Result<_>>()?;
            let mat = matrix4_to_row_major_f64(&transform);
            cad_part.shape.tweak_faces(faces, mat)?
        };


        retessellate_node(&tweaked, &self.scene, options, node)?;

        let cad_part = self.get_part_mut(part).context("tweak_faces: part not found")?;
        let before = std::mem::replace(&mut cad_part.shape, tweaked.clone());
        cad_part.options = options.clone();

        self.history.record(
            "Tweak face",
            Delta::Reshaped { part, before, after: tweaked, options: options.clone() },
        );

        Ok(())
    }

    /// Groups every mutation made through the returned guard into one undo step.
    /// Nested scopes merge into the outermost one, whose label wins.
    pub fn undo_scope(&mut self, label: impl Into<String>) -> UndoScope<'_> {
        self.history.begin_group(label);
        UndoScope { doc: self }
    }

    /// Undo the most recent step. Returns its label, or `None` on an empty stack.
    pub fn undo(&mut self) -> Result<Option<String>> {
        let Some(step) = self.history.pop_undo() else {
            return Ok(None);
        };
        let mut result = Ok(());
        for delta in step.deltas.iter().rev() {
            let applied = match delta {
                Delta::Added(snapshot) => {
                    self.remove_part_inner(snapshot.part);
                    Ok(())
                }
                Delta::Removed(snapshot) => self.resurrect(snapshot),
                Delta::Reshaped { part, before, options, .. } => {
                    self.reshape(*part, before, options)
                }
            };
            if result.is_ok() {
                result = applied;
            }
        }
        let label = step.label.clone();
        self.history.push_redo(step);
        result.map(|()| Some(label))
    }

    /// Redo the most recently undone step. Returns its label, or `None` on an
    /// empty stack. Replays captured results; no CAD operation is re-run.
    pub fn redo(&mut self) -> Result<Option<String>> {
        let Some(step) = self.history.pop_redo() else {
            return Ok(None);
        };
        let mut result = Ok(());
        for delta in &step.deltas {
            let applied = match delta {
                Delta::Added(snapshot) => self.resurrect(snapshot),
                Delta::Removed(snapshot) => {
                    self.remove_part_inner(snapshot.part);
                    Ok(())
                }
                Delta::Reshaped { part, after, options, .. } => {
                    self.reshape(*part, after, options)
                }
            };
            if result.is_ok() {
                result = applied;
            }
        }
        let label = step.label.clone();
        self.history.restore_undo(step);
        result.map(|()| Some(label))
    }

    /// Label of the step [`undo`](Self::undo) would revert, if any.
    pub fn undo_label(&self) -> Option<&str> {
        self.history.undo_label()
    }

    /// Label of the step [`redo`](Self::redo) would replay, if any.
    pub fn redo_label(&self) -> Option<&str> {
        self.history.redo_label()
    }

    /// Captures the state needed to delete and later resurrect a part.
    /// `None` for an unknown part or one whose scene node is gone.
    ///
    /// The snapshot's node handle keeps the part's scene chain (node →
    /// instance → mesh/materials) alive off-tree after removal, so resurrection
    /// is a reattach — no retessellation, all ids preserved.
    fn snapshot_part(&self, id: PartId) -> Option<PartSnapshot> {
        let part = self.get_part(id)?;
        let node = self.scene.node_handle(self.node_for_part(id)?)?;
        Some(PartSnapshot {
            part: id,
            node,
            name: part.name.clone(),
            shape: part.shape.clone(),
            options: part.options.clone(),
        })
    }

    /// Reattaches a removed part's still-alive subtree under its original part,
    /// node, and material ids.
    fn resurrect(&mut self, snapshot: &PartSnapshot) -> Result<()> {
        let node = snapshot.node.id();
        self.scene
            .reparent_node(node, None)
            .context("resurrect: part node no longer exists")?;
        // Parts hidden at removal time (e.g. sources consumed by a commit)
        // come back visible.
        self.scene.set_node_visibility(node, Visibility::Visible);
        self.parts.push(CadPart {
            id: snapshot.part,
            name: snapshot.name.clone(),
            shape: snapshot.shape.clone(),
            options: snapshot.options.clone(),
        });
        self.part_to_node.insert(snapshot.part, node);
        self.node_to_part.insert(node, snapshot.part);
        Ok(())
    }

    /// Swaps a part's shape to a captured snapshot and re-tessellates in place,
    /// preserving its node.
    fn reshape(
        &mut self,
        part: PartId,
        shape: &Shape,
        options: &CadTessellationOptions,
    ) -> Result<()> {
        let node = self.node_for_part(part).context("reshape: no node for part")?;
        retessellate_node(shape, &self.scene, options, node)?;
        self.scene.set_node_transform(node, Transform::IDENTITY);

        let cad_part = self.get_part_mut(part).context("reshape: part not found")?;
        cad_part.shape = shape.clone();
        cad_part.options = options.clone();
        Ok(())
    }

    pub fn parts(&self) -> impl Iterator<Item = &CadPart> {
        self.parts.iter()
    }

    pub fn part_for_node(&self, node: NodeId) -> Option<PartId> {
        self.node_to_part.get(&node).copied()
    }

    pub fn node_for_part(&self, part: PartId) -> Option<NodeId> {
        self.part_to_node.get(&part).copied()
    }

    /// Resolve a picked face — identified by its tessellation order (`face_index`,
    /// as carried by a `SubGeometryKind::Face` selection) — back to its OCCT [`Face`] sub-shape.
    pub fn face_subshape(&self, node: NodeId, face_index: u32) -> Option<Face> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        part.shape.faces().nth(face_index as usize)
    }

    /// Resolve a picked edge — identified by its tessellation order (`edge_index`,
    /// as carried by a `SubGeometryKind::Edge` selection) — back to its OCCT [`Edge`] sub-shape.
    pub fn edge_subshape(&self, node: NodeId, edge_index: u32) -> Option<Edge> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        part.shape.edges().nth(edge_index as usize)
    }

    /// Resolve a picked edge to the [`Wire`] that contains it in the part's B-Rep.
    ///
    /// A loft profile is a wire, but the selection system reports the individual
    /// edge the user clicked; this finds the wire that edge belongs to (the first
    /// wire containing a topologically-identical edge), so one click selects a
    /// whole multi-edge profile.
    pub fn wire_for_edge(&self, node: NodeId, edge_index: u32) -> Option<Wire> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        let target = part.shape.edges().nth(edge_index as usize)?;
        part.shape.wires().find(|wire| wire.edges().any(|e| e.is_same(&target)))
    }
}

/// Guard for [`Document::undo_scope`]: mutations made through it accumulate
/// into one undo step, committed when the guard drops.
/// 
/// Derefs into `Document` so mutations can be made directly though it.
pub struct UndoScope<'a> {
    doc: &'a mut Document,
}

impl Deref for UndoScope<'_> {
    type Target = Document;

    fn deref(&self) -> &Document {
        self.doc
    }
}

impl DerefMut for UndoScope<'_> {
    fn deref_mut(&mut self) -> &mut Document {
        self.doc
    }
}

impl Drop for UndoScope<'_> {
    fn drop(&mut self) {
        self.doc.history.end_group();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::Vector3;
    use duck_engine_scene::resource::NodePayload;

    fn doc_with_box() -> (Document, PartId, NodeId) {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let part = doc
            .add_part("box", opencascade::primitives::Shape::cube(2.0), &CadTessellationOptions::default())
            .expect("box tessellates");
        let node = doc.node_for_part(part).expect("part has a node");
        (doc, part, node)
    }

    /// Tessellation index of the box face whose center has the largest y (the top).
    fn top_face_index(doc: &Document, part: PartId) -> u32 {
        let shape = &doc.get_part(part).unwrap().shape;
        shape
            .faces()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.center_of_mass().y.total_cmp(&b.center_of_mass().y))
            .map(|(index, _)| index as u32)
            .expect("box has faces")
    }

    fn max_y(doc: &Document, part: PartId) -> f64 {
        let shape = &doc.get_part(part).unwrap().shape;
        shape
            .mesh()
            .unwrap()
            .vertices
            .iter()
            .map(|v| v.y)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    #[test]
    fn tweak_faces_resolves_brep_and_keeps_node() {
        let (mut doc, part, node) = doc_with_box();
        let face = top_face_index(&doc, part);
        let transform = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));

        doc.tweak_faces(part, &[face], transform, &CadTessellationOptions::default())
            .expect("translating the top face up re-solves");

        assert_eq!(doc.parts().count(), 1, "tweak edits the part in place");
        assert_eq!(doc.node_for_part(part), Some(node), "node id must be preserved");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6, "top must land at y=3");
    }

    #[test]
    fn tweak_faces_bad_index_leaves_part_untouched() {
        let (mut doc, part, _) = doc_with_box();
        let transform = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));

        let result = doc.tweak_faces(part, &[99], transform, &CadTessellationOptions::default());

        assert!(result.is_err());
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "failed tweak must not modify the shape");
    }

    #[test]
    fn tweak_after_boolean_direct_path() {
        use crate::boolean::{execute_boolean, BooleanKind};

        let (mut doc, box_part, _) = doc_with_box();
        let sphere = doc
            .add_part(
                "sphere",
                opencascade::primitives::Shape::sphere(1.0).at(glam::dvec3(2.0, 2.0, 2.0)).build(),
                &CadTessellationOptions::default(),
            )
            .expect("sphere tessellates");
        let box_node = doc.node_for_part(box_part).unwrap();
        let sphere_node = doc.node_for_part(sphere).unwrap();
        execute_boolean(
            BooleanKind::Subtract,
            box_node,
            &[sphere_node],
            &mut doc,
            &CadTessellationOptions::default(),
        )
        .expect("subtract succeeds");

        let part = doc.parts().next().expect("boolean leaves one part").id;
        let node = doc.node_for_part(part).unwrap();
        let faces_before = doc.get_part(part).unwrap().shape.faces().count();
        let face = top_face_index(&doc, part);
        let transform = Matrix4::from_translation(Vector3::new(0.0, 0.5, 0.0));

        doc.tweak_faces(part, &[face], transform, &CadTessellationOptions::default())
            .expect("tweaking the boolean result re-solves");

        assert_eq!(doc.node_for_part(part), Some(node), "node id must be preserved");
        assert!((max_y(&doc, part) - 2.5).abs() < 1e-6, "top must land at y=2.5");
        let faces_after = doc.get_part(part).unwrap().shape.faces().count();
        assert_eq!(faces_after, faces_before, "the spherical cavity must survive the re-solve");
    }

    #[test]
    fn undo_add_detaches_part_and_keeps_it_for_redo() {
        let (mut doc, part, node) = doc_with_box();

        let label = doc.undo().expect("undo succeeds");
        assert_eq!(label.as_deref(), Some("Add box"));
        assert_eq!(doc.parts().count(), 0);
        assert!(doc.node_for_part(part).is_none());
        {
            let scene = doc.scene().lock();
            // The redo snapshot owns the detached subtree: nothing renders,
            // but the resources survive for an id-stable redo.
            assert!(!scene.is_node_attached(node));
            assert_eq!(scene.mesh_count(), 1);
        }

        // Dropping the history releases the snapshot; everything is freed.
        doc.history.clear();
        let scene = doc.scene().lock();
        assert!(scene.get_node(node).is_none());
        assert_eq!(scene.mesh_count(), 0);
        assert_eq!(scene.face_material_count(), 0);
        assert_eq!(scene.line_material_count(), 0);
    }

    #[test]
    fn undo_remove_resurrects_with_stable_ids() {
        let (mut doc, part, node) = doc_with_box();
        let face_material = {
            let scene = doc.scene().lock();
            let NodePayload::Instance(instance) = scene.get_node(node).unwrap().payload()
            else {
                panic!("expected instance payload");
            };
            scene.get_instance(instance.id()).unwrap().face_material().unwrap()
        };

        doc.remove_part(part);
        assert_eq!(doc.parts().count(), 0);

        let label = doc.undo().expect("undo succeeds");
        assert_eq!(label.as_deref(), Some("Delete box"));
        assert_eq!(doc.get_part(part).unwrap().name, "box");
        assert_eq!(doc.node_for_part(part), Some(node));
        assert_eq!(doc.part_for_node(node), Some(part));
        assert_eq!(doc.part_visibility(part), Some(Visibility::Visible));
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "geometry restored");
        let scene = doc.scene().lock();
        assert!(
            scene.get_face_material(face_material).is_some(),
            "face material resurrected under its original id"
        );
        let restored = scene.get_node(node).unwrap();
        assert_eq!(restored.transform(), Transform::IDENTITY);
    }

    #[test]
    fn redo_replays_add_and_remove() {
        let (mut doc, part, node) = doc_with_box();

        doc.undo().expect("undo add");
        let label = doc.redo().expect("redo add");
        assert_eq!(label.as_deref(), Some("Add box"));
        assert_eq!(doc.node_for_part(part), Some(node), "ids stable across redo");

        doc.remove_part(part);
        doc.undo().expect("undo remove");
        let label = doc.redo().expect("redo remove");
        assert_eq!(label.as_deref(), Some("Delete box"));
        assert_eq!(doc.parts().count(), 0);
    }

    #[test]
    fn undo_redo_restore_baked_transform() {
        let (mut doc, part, node) = doc_with_box();
        let up = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));
        doc.bake_transform(part, up, &CadTessellationOptions::default())
            .expect("bake succeeds");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);

        let label = doc.undo().expect("undo bake");
        assert_eq!(label.as_deref(), Some("Transform"));
        assert_eq!(doc.node_for_part(part), Some(node), "node preserved");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "shape restored");

        doc.redo().expect("redo bake");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6, "captured result replayed");
    }

    #[test]
    fn undo_redo_restore_tweaked_faces() {
        let (mut doc, part, node) = doc_with_box();
        let face = top_face_index(&doc, part);
        let up = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));
        doc.tweak_faces(part, &[face], up, &CadTessellationOptions::default())
            .expect("tweak succeeds");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);

        let label = doc.undo().expect("undo tweak");
        assert_eq!(label.as_deref(), Some("Tweak face"));
        assert_eq!(doc.node_for_part(part), Some(node), "node preserved");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "shape restored");

        doc.redo().expect("redo tweak");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn undo_scope_groups_mutations_into_one_step() {
        let (mut doc, part, _) = doc_with_box();
        {
            let mut doc = doc.undo_scope("Combine");
            doc.add_part(
                "sphere",
                opencascade::primitives::Shape::sphere(1.0).build(),
                &CadTessellationOptions::default(),
            )
            .expect("sphere tessellates");
            doc.remove_part(part);
        }
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.undo_label(), Some("Combine"));

        doc.undo().expect("undo group");
        assert_eq!(doc.parts().count(), 1, "sphere removed, box resurrected");
        assert!(doc.get_part(part).is_some());

        doc.redo().expect("redo group");
        let names: Vec<_> = doc.parts().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["sphere"]);
    }

    #[test]
    fn new_commit_clears_redo() {
        let (mut doc, _, _) = doc_with_box();
        doc.undo().expect("undo add");
        assert_eq!(doc.redo_label(), Some("Add box"));
        doc.add_part(
            "sphere",
            opencascade::primitives::Shape::sphere(1.0).build(),
            &CadTessellationOptions::default(),
        )
        .expect("sphere tessellates");
        assert!(doc.redo_label().is_none());
    }

    #[test]
    fn empty_stacks_are_a_no_op() {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        assert!(doc.undo().expect("empty undo ok").is_none());
        assert!(doc.redo().expect("empty redo ok").is_none());
        assert!(doc.undo_label().is_none());
        assert!(doc.redo_label().is_none());
    }

    #[test]
    fn tweak_faces_rejects_non_similarity_transform() {
        let (mut doc, part, _) = doc_with_box();
        let face = top_face_index(&doc, part);
        let squash = Matrix4::from_nonuniform_scale(1.0, 0.5, 1.0);

        let result = doc.tweak_faces(part, &[face], squash, &CadTessellationOptions::default());

        assert!(result.is_err(), "non-uniform scale must be rejected");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "failed tweak must not modify the shape");
    }
}
