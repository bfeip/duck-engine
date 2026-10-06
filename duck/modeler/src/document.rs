use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

use anyhow::{Context, Result};
use duck_engine_scene::resource::{Id, NodeId, Visibility};
use duck_engine_scene::Scene;
use duck_engine_scene::cad::{CadTessellationOptions, retessellate_node, tessellate_into};
use duck_engine_scene::common::{matrix4_to_row_major_f64, Matrix4, Point3, Real, Transform, Vector3};
use glam::DVec3;
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

/// What a result grown from a part does with that part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceFate {
    /// The result fuses into its source, which is reshaped in place.
    Fuse,
    /// The result replaces its source outright, under its name.
    Replace,
    /// The result is a new part beside its untouched source.
    Keep,
}

/// Whether `shape` is or contains a solid body, as opposed to bare sheets,
/// wires and points.
pub fn has_solid(shape: &Shape) -> bool {
    shape.contains_type(ShapeType::Solid)
}

/// Merges faces and edges left split across a shared surface or curve — chiefly
/// the two halves a boolean leaves when it divides a periodic face at its seam,
/// which would otherwise be separate geometry.
///
/// A cleanup failure is not an operation failure: the original shape is kept.
pub fn unify_same_domain(shape: Shape) -> Shape {
    let cleaned = match shape.clean() {
        Ok(cleaned) => cleaned,
        Err(e) => {
            log::warn!("Failed to unify same-domain geometry, keeping the original: {e}");
            return shape;
        }
    };
    // OCCT healing can silently discard geometry; losing every face means the
    // cleanup ate the body.
    if shape.faces().next().is_some() && cleaned.faces().next().is_none() {
        log::warn!("Unifying same-domain geometry lost all faces; keeping the original");
        return shape;
    }
    cleaned
}

/// Unwraps compound shapes that contain a single solid, such as from the output
/// of a 'fuse' operation,
pub fn unwrap_single_solid(shape: Shape) -> Shape {
    if shape.shape_type() != ShapeType::Compound {
        return shape;
    }
    let mut children = shape.sub_shapes();
    match (children.next(), children.next()) {
        (Some(only), None) if only.shape_type() == ShapeType::Solid => only,
        _ => shape,
    }
}

/// `addition` fused into `body`: a lone solid, with any seams the fuse split
/// merged back.
fn fuse(body: &Shape, addition: &Shape) -> Result<Shape> {
    // An OCCT boolean may mutate its inputs, which must survive a failed fuse
    // unchanged.
    let (body, addition) = (body.deep_copy(), addition.deep_copy());
    let fuzz = interactive_fuzz([&body, &addition].into_iter());
    let fused = body.union_with_fuzz(&addition, fuzz)?;
    if let Some(warnings) = &fused.warnings {
        log::warn!("Fuse completed with warnings:\n{warnings}");
    }
    // The fuse wraps its result in a compound and can split a periodic face at
    // its seam; unwrap a lone solid, then merge the halves back.
    Ok(unify_same_domain(unwrap_single_solid(fused.shape)))
}

/// `shape` moved by `placement`. A similarity keeps surfaces analytic (planes
/// stay planes); only a non-uniform scale needs the B-spline-converting general
/// transform.
pub fn place(shape: &Shape, placement: &Matrix4) -> Shape {
    let mat = matrix4_to_row_major_f64(placement);
    match shape.transformed(mat) {
        Ok(placed) => placed,
        Err(_) => shape.gtransform(mat),
    }
}

/// An engine point as an OCCT point.
pub fn point3_to_dvec3(p: Point3) -> DVec3 {
    DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))
}

/// An engine vector as an OCCT vector.
pub fn vec3_to_dvec3(v: Vector3) -> DVec3 {
    DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
}

/// An OCCT point as an engine point.
pub fn dvec3_to_point3(v: DVec3) -> Point3 {
    Point3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// An OCCT vector as an engine vector.
pub fn dvec3_to_vec3(v: DVec3) -> Vector3 {
    Vector3::new(v.x as Real, v.y as Real, v.z as Real)
}

/// Additional boolean intersection tolerance for interactively placed parts.
///
/// Snaps onto existing parts read positions from the tessellated `f32` vertex
/// buffer, so inputs meant to coincide can sit a few f32 ulps of the coordinate
/// magnitude apart — far beyond OCCT's 1e-7 default, in the near-coincidence
/// band where the BOP misclassifies splits and a subtract silently removes
/// nothing. Four ulps of the inputs' extent covers that placement error with
/// margin.
pub fn interactive_fuzz<'a>(shapes: impl Iterator<Item = &'a Shape>) -> f64 {
    let extent = shapes
        .map(|shape| {
            let aabb = opencascade::bounding_box::aabb(shape);
            aabb.min().abs().max_element().max(aabb.max().abs().max_element())
        })
        .fold(0.0f64, f64::max);
    4.0 * f64::from(f32::EPSILON) * extent
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
    /// Tessellation options the part was last built with. Imported parts carry
    /// their own colors here, so a copy made with these keeps its appearance.
    pub fn options(&self) -> &CadTessellationOptions {
        &self.options
    }

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

/// The series a part name belongs to, i.e. the name with any
/// [`numbered_name`](Document::numbered_name) suffix removed: `Box-003` → `Box`,
/// `Bracket` → `Bracket`, `Part-A` → `Part-A`.
fn numbering_base(name: &str) -> &str {
    match name.rsplit_once('-') {
        Some((base, suffix)) if !base.is_empty() && suffix.parse::<u32>().is_ok() => base,
        _ => name,
    }
}

pub struct Document {
    parts: Vec<CadPart>,
    part_to_node: HashMap<PartId, NodeId>,
    node_to_part: HashMap<NodeId, PartId>,
    scene: Scene,
    history: History,
    generation: u64,
}

impl Document {
    pub fn new(scene: Scene) -> Self {
        Self {
            parts: Vec::new(),
            part_to_node: HashMap::new(),
            node_to_part: HashMap::new(),
            scene,
            history: History::default(),
            generation: 0,
        }
    }

    pub fn set_scene(&mut self, scene: Scene) {
        self.scene = scene;
        self.history.clear();
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Bumped whenever a part is added, removed, reshaped, or renamed.
    pub fn generation(&self) -> u64 {
        self.generation
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
        self.generation += 1;
        if let Some(snapshot) = self.snapshot_part(id) {
            let label = format!("Add {}", self.get_part(id).unwrap().name);
            self.history.record(&label, Delta::Added(snapshot));
        }
        Ok(id)
    }

    /// [`add_part`](Self::add_part) under an auto-numbered name derived from `base`.
    pub fn add_numbered_part(
        &mut self,
        base: &str,
        shape: Shape,
        options: &CadTessellationOptions,
    ) -> Result<PartId> {
        let name = self.numbered_name(base);
        self.add_part(name, shape, options)
    }

    /// The next free numbered name for `base` — `Box-001`, `Box-002`, … — one
    /// past the highest suffix currently in use.
    ///
    /// Derived from the live part names rather than a stored counter, so undo,
    /// redo, import, and [`set_scene`](Self::set_scene) cannot desync it.
    pub fn numbered_name(&self, base: &str) -> String {
        let highest = self
            .parts
            .iter()
            .filter_map(|part| {
                part.name.strip_prefix(base)?.strip_prefix('-')?.parse::<u32>().ok()
            })
            .max();
        format!("{base}-{:03}", highest.map_or(1, |n| n + 1))
    }

    /// The next free numbered name in `source`'s series — `Box-003` yields
    /// `Box-004`, an unnumbered `Bracket` yields `Bracket-001`.
    ///
    /// Falls back to `Copy` for an unknown part.
    pub fn duplicate_name(&self, source: PartId) -> String {
        let base = match self.get_part(source) {
            Some(part) => numbering_base(&part.name),
            None => "Copy",
        };
        self.numbered_name(base)
    }

    /// Rename a part and its scene node, recorded as its own undo step.
    ///
    /// A blank, unchanged, or unknown-part rename is a no-op and records nothing.
    pub fn rename_part(&mut self, id: PartId, name: impl Into<String>) {
        let name = name.into();
        let name = name.trim();
        if name.is_empty() {
            return;
        }
        let Some(part) = self.get_part(id) else { return };
        if part.name == name {
            return;
        }
        let before = part.name.clone();
        let after = name.to_owned();
        self.set_part_name(id, &after);
        self.history
            .record(&format!("Rename {before}"), Delta::Renamed { part: id, before, after });
    }

    /// [`rename_part`](Self::rename_part) without validation or history
    /// recording, shared with undo/redo replay.
    fn set_part_name(&mut self, id: PartId, name: &str) {
        if let Some(part) = self.get_part_mut(id) {
            part.name = name.to_owned();
        }
        if let Some(node) = self.node_for_part(id) {
            self.scene.set_node_name(node, Some(name.to_owned()));
        }
        self.generation += 1;
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
        self.generation += 1;
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

    /// Bake a transform into the part's CAD geometry, then re-tessellate the
    /// part in place (preserving its `NodeId`) and reset the node transform to
    /// identity. The part is untouched on error.
    pub fn bake_transform(&mut self, part: PartId, transform: Matrix4) -> Result<()> {
        let baked = {
            let cad_part = self.get_part(part).context("bake_transform: part not found")?;
            place(&cad_part.shape, &transform)
        };
        self.reshape_part(part, baked, "Transform")
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
    ) -> Result<()> {
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
        self.reshape_part(part, tweaked, "Tweak face")
    }

    /// Replace a part's shape in place, preserving its `NodeId` and appearance,
    /// as one undo step labelled `label`. The part is untouched on error.
    ///
    /// Re-tessellates with the part's own options, which keep an imported
    /// part's colors.
    pub fn reshape_part(&mut self, part: PartId, shape: Shape, label: &str) -> Result<()> {
        let (before, options) = {
            let cad_part = self.get_part(part).context("reshape_part: part not found")?;
            (cad_part.shape.clone(), cad_part.options.clone())
        };
        self.reshape(part, &shape, &options)?;
        self.history.record(label, Delta::Reshaped { part, before, after: shape, options });
        Ok(())
    }

    /// Commits `result`, grown from part `source`, as `fate` says, as one undo
    /// step labelled `label`. A new part is named in `base`'s series and
    /// tessellated with `options`; a fused source keeps its own.
    pub fn commit_result(
        &mut self,
        source: PartId,
        result: Shape,
        fate: SourceFate,
        label: &str,
        base: &str,
        options: &CadTessellationOptions,
    ) -> Result<()> {
        let source_part = self.get_part(source).context("Source part not found")?;
        match fate {
            SourceFate::Fuse => {
                let fused = fuse(&source_part.shape, &result)
                    .context("Failed to fuse into the source part")?;
                self.reshape_part(source, fused, label)
            }
            // A new part rather than a reshape: a sheet's node carries sketch
            // materials, which retessellating in place would keep.
            SourceFate::Replace => {
                let name = source_part.name.clone();
                let mut doc = self.undo_scope(label);
                // Tessellates atomically — if this fails, nothing is changed.
                doc.add_part(name, result, options).context("Failed to tessellate the result")?;
                doc.remove_part(source);
                Ok(())
            }
            SourceFate::Keep => {
                let mut doc = self.undo_scope(label);
                doc.add_numbered_part(base, result, options).context("Failed to tessellate the result")?;
                Ok(())
            }
        }
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
                Delta::Renamed { part, before, .. } => {
                    self.set_part_name(*part, before);
                    Ok(())
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
                Delta::Renamed { part, after, .. } => {
                    self.set_part_name(*part, after);
                    Ok(())
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
        self.generation += 1;
        // The node may carry a name from before a rename that this resurrection
        // predates; the snapshot is the authority.
        self.scene.set_node_name(node, Some(snapshot.name.clone()));
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
        self.generation += 1;
        Ok(())
    }

    /// Re-tessellates every part with seam edges shown or hidden.
    ///
    /// Purely a display change — the shapes are untouched, sub-geometry indices
    /// stay aligned, and nothing is recorded on the undo stack.
    pub fn set_seam_edges_visible(&mut self, visible: bool) {
        for index in 0..self.parts.len() {
            let part = &mut self.parts[index];
            if part.options.show_seam_edges == visible {
                continue;
            }
            part.options.show_seam_edges = visible;
            let id = part.id;
            let Some(node) = self.node_for_part(id) else { continue };
            let part = &self.parts[index];
            if let Err(e) = retessellate_node(&part.shape, &self.scene, &part.options, node) {
                log::warn!("Failed to re-tessellate {}: {e:#}", part.name);
            }
        }
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
        part.shape.face_at(face_index as usize)
    }

    /// Resolve a picked edge — identified by its tessellation order (`edge_index`,
    /// as carried by a `SubGeometryKind::Edge` selection) — back to its OCCT [`Edge`] sub-shape.
    pub fn edge_subshape(&self, node: NodeId, edge_index: u32) -> Option<Edge> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        part.shape.edge_at(edge_index as usize)
    }

    /// Resolve a picked edge to the [`Wire`] that contains it in the part's B-Rep.
    ///
    /// A loft profile is a wire, but the selection system reports the individual
    /// edge the user clicked; this finds the wire that edge belongs to (the first
    /// wire containing a topologically-identical edge), so one click selects a
    /// whole multi-edge profile.
    pub fn wire_for_edge(&self, node: NodeId, edge_index: u32) -> Option<Wire> {
        let part = self.part_for_node(node).and_then(|id| self.get_part(id))?;
        let target = part.shape.edge_at(edge_index as usize)?;
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
    use duck_engine_scene::common::{
        Deg, EuclideanSpace, InnerSpace, Quaternion, RgbaColor, Rotation3, Vector3,
    };
    use duck_engine_scene::resource::FaceMaterial;

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

        doc.tweak_faces(part, &[face], transform)
            .expect("translating the top face up re-solves");

        assert_eq!(doc.parts().count(), 1, "tweak edits the part in place");
        assert_eq!(doc.node_for_part(part), Some(node), "node id must be preserved");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6, "top must land at y=3");
    }

    /// `steps` small turns about arbitrary axes, composed the way an
    /// interactive rotate drag accumulates them.
    fn dragged_rotation(seed: &mut u32, steps: usize) -> Quaternion {
        let mut next = || {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 17;
            *seed ^= *seed << 5;
            (*seed >> 8) as Real / (1u32 << 24) as Real
        };
        let mut rotation = Quaternion::new(1.0, 0.0, 0.0, 0.0);
        for _ in 0..steps {
            let axis = Vector3::new(next() - 0.5, next() - 0.5, next() - 0.5).normalize();
            rotation = Quaternion::from_axis_angle(axis, Deg(next() * 20.0)) * rotation;
        }
        rotation
    }

    /// Engine-side rotations must reach OCCT as similarities, or `place` falls
    /// back to the B-spline-converting general transform. (In `f32` most of
    /// these fail OCCT's 1e-7 similarity check.)
    #[test]
    fn dragged_rotations_bake_as_similarities() {
        let cube = Shape::cube(2.0);
        let mut seed = 0x1234_5678;
        for i in 0..200 {
            let placement = Transform {
                position: Point3::new(123.4, 56.7, 89.0),
                ..Transform::from_rotation(dragged_rotation(&mut seed, 20))
            };
            let mat = matrix4_to_row_major_f64(&placement.to_matrix());
            assert!(cube.transformed(mat).is_ok(), "rotation {i} is not a similarity");
        }
    }

    #[test]
    fn tweak_faces_accepts_a_dragged_tilt() {
        let (mut doc, part, _) = doc_with_box();
        let face = top_face_index(&doc, part);
        let pivot = {
            let shape = &doc.get_part(part).unwrap().shape;
            dvec3_to_point3(shape.faces().nth(face as usize).unwrap().center_of_mass())
        };
        // Tilt the top face about a skewed in-plane axis, one drag step at a time.
        let axis = Vector3::new(1.0, 0.0, 2.0).normalize();
        let tilt = (0..20).fold(Quaternion::new(1.0, 0.0, 0.0, 0.0), |q, _| {
            Quaternion::from_axis_angle(axis, Deg(1.3)) * q
        });
        let transform = Matrix4::from_translation(pivot.to_vec())
            * Matrix4::from(tilt)
            * Matrix4::from_translation(-pivot.to_vec());

        doc.tweak_faces(part, &[face], transform).expect("a tilted top face re-solves");
        assert!(max_y(&doc, part) > 2.0 + 1e-3, "one side of the top must rise");
    }

    #[test]
    fn tweak_faces_bad_index_leaves_part_untouched() {
        let (mut doc, part, _) = doc_with_box();
        let transform = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));

        let result = doc.tweak_faces(part, &[99], transform);

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

        doc.tweak_faces(part, &[face], transform)
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
            let instance_id = scene.get_node(node).unwrap().instance().expect("expected an instance");
            scene.get_instance(instance_id).unwrap().face_material().unwrap()
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
        doc.bake_transform(part, up)
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
        doc.tweak_faces(part, &[face], up)
            .expect("tweak succeeds");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);

        let label = doc.undo().expect("undo tweak");
        assert_eq!(label.as_deref(), Some("Tweak face"));
        assert_eq!(doc.node_for_part(part), Some(node), "node preserved");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "shape restored");

        doc.redo().expect("redo tweak");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);
    }

    fn assert_part_color(doc: &Document, part: PartId, expected: RgbaColor) {
        let color = doc.get_part(part).unwrap().options().face_material.base_color_factor();
        let channels = [(color.r, expected.r), (color.g, expected.g), (color.b, expected.b), (color.a, expected.a)];
        let close = channels.iter().all(|(got, want)| (got - want).abs() < 1e-6);
        assert!(close, "expected {expected:?}, got {color:?}");
    }

    /// Moving or tweaking a part keeps its own tessellation options, which carry
    /// an imported part's color for later copies to inherit. Undo brings back
    /// the same options.
    #[test]
    fn reshaping_keeps_the_part_options() {
        let red = RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
        let options = CadTessellationOptions {
            face_material: FaceMaterial::new().with_base_color_factor(red),
            ..Default::default()
        };
        let mut doc = Document::new(Scene::default());
        let part = doc.add_part("imported", Shape::cube(2.0), &options).expect("cube tessellates");
        let up = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));

        doc.bake_transform(part, up).expect("bake succeeds");
        let face = top_face_index(&doc, part);
        doc.tweak_faces(part, &[face], up).expect("tweak succeeds");
        assert_part_color(&doc, part, red);

        doc.undo().expect("undo tweak");
        doc.undo().expect("undo bake");
        assert_part_color(&doc, part, red);
    }

    #[test]
    fn reshape_part_keeps_the_node_and_undoes_as_one_step() {
        let (mut doc, part, node) = doc_with_box();

        let taller = opencascade::primitives::Shape::box_with_dimensions(2.0, 3.0, 2.0);
        doc.reshape_part(part, taller, "Stretch").expect("reshape succeeds");
        assert_eq!(doc.node_for_part(part), Some(node), "node id must be preserved");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6);
        assert_eq!(doc.undo_label(), Some("Stretch"));

        doc.undo().expect("undo reshape");
        assert_eq!(doc.node_for_part(part), Some(node));
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "shape restored");

        doc.redo().expect("redo reshape");
        assert!((max_y(&doc, part) - 3.0).abs() < 1e-6, "captured result replayed");
    }

    /// A 2×1×2 pad standing on the 2-cube's top face.
    fn pad() -> Shape {
        Shape::box_from_corners(DVec3::new(0.0, 2.0, 0.0), DVec3::new(2.0, 3.0, 2.0))
    }

    /// A unit-square sketch region.
    fn region() -> Shape {
        Face::from_wire(&Wire::rect(1.0, 1.0).expect("rectangle builds")).expect("face builds").into()
    }

    fn part_volume(doc: &Document, part: PartId) -> f64 {
        doc.get_part(part).unwrap().shape.volume()
    }

    /// A fused result reshapes its source in place, keeping the part, its node
    /// and its appearance, as one undo step.
    #[test]
    fn a_fused_result_reshapes_its_source_in_place() {
        let red = RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
        let options = CadTessellationOptions {
            face_material: FaceMaterial::new().with_base_color_factor(red),
            ..Default::default()
        };
        let mut doc = Document::new(Scene::default());
        let part = doc.add_part("imported", Shape::cube(2.0), &options).expect("cube tessellates");
        let node = doc.node_for_part(part);

        doc.commit_result(part, pad(), SourceFate::Fuse, "Pad", "Pad", &CadTessellationOptions::default())
            .expect("the pad fuses");
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.node_for_part(part), node);
        assert_eq!(doc.get_part(part).unwrap().kind(), PartKind::Solid);
        assert!((part_volume(&doc, part) - 12.0).abs() < 1e-6, "got {}", part_volume(&doc, part));
        assert_part_color(&doc, part, red);
        assert_eq!(doc.undo_label(), Some("Pad"));

        doc.undo().expect("undo the pad");
        assert!((part_volume(&doc, part) - 8.0).abs() < 1e-9);
        doc.redo().expect("redo the pad");
        assert!((part_volume(&doc, part) - 12.0).abs() < 1e-6);
    }

    /// A result that replaces its source takes over its name; undo brings the
    /// source back.
    #[test]
    fn a_replacing_result_takes_over_its_sources_name() {
        let mut doc = Document::new(Scene::default());
        let sketch = doc.add_part("sketch", region(), &CadTessellationOptions::default()).expect("sketch tessellates");

        doc.commit_result(sketch, Shape::cube(1.0), SourceFate::Replace, "Thicken", "Thickened", &CadTessellationOptions::default())
            .expect("the result replaces the sketch");
        let parts: Vec<_> = doc.parts().collect();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].name, "sketch");
        assert_eq!(parts[0].kind(), PartKind::Solid);
        assert!(doc.get_part(sketch).is_none(), "the sketch was consumed");
        assert_eq!(doc.undo_label(), Some("Thicken"));

        doc.undo().expect("undo the replacement");
        assert_eq!(doc.parts().count(), 1);
        assert_eq!(doc.get_part(sketch).map(CadPart::kind), Some(PartKind::Face));
    }

    /// A kept source stands untouched beside a new, numbered part.
    #[test]
    fn a_kept_source_stands_beside_a_new_part() {
        let (mut doc, part, _) = doc_with_box();

        doc.commit_result(part, pad(), SourceFate::Keep, "Thicken", "Thickened", &CadTessellationOptions::default())
            .expect("the result is added");
        let names: Vec<_> = doc.parts().map(|part| part.name.as_str()).collect();
        assert_eq!(names, ["box", "Thickened-001"]);
        assert!((part_volume(&doc, part) - 8.0).abs() < 1e-9);
        assert_eq!(doc.undo_label(), Some("Thicken"));

        doc.undo().expect("undo the new part");
        assert_eq!(doc.parts().count(), 1);
    }

    #[test]
    fn only_solid_geometry_has_a_solid() {
        use opencascade::primitives::Compound;

        assert!(has_solid(&Shape::cube(2.0)));
        assert!(has_solid(&Compound::from_shapes([Shape::cube(2.0)]).into()));
        assert!(!has_solid(&region()));
        assert!(!has_solid(&Compound::from_shapes([region()]).into()));
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

    /// The node's display name, which must track the part's.
    fn node_name(doc: &Document, part: PartId) -> Option<String> {
        let node = doc.node_for_part(part)?;
        let scene = doc.scene().lock();
        scene.get_node(node)?.name.clone()
    }

    #[test]
    fn numbered_names_count_up_and_fill_past_the_highest() {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let opts = CadTessellationOptions::default();

        assert_eq!(doc.numbered_name("Box"), "Box-001");
        doc.add_numbered_part("Box", opencascade::primitives::Shape::cube(2.0), &opts)
            .expect("box tessellates");
        assert_eq!(doc.numbered_name("Box"), "Box-002");

        // A different base numbers independently.
        assert_eq!(doc.numbered_name("Sphere"), "Sphere-001");

        // Gaps are not filled: the next name is one past the highest in use.
        doc.add_part("Box-007", opencascade::primitives::Shape::cube(2.0), &opts)
            .expect("box tessellates");
        assert_eq!(doc.numbered_name("Box"), "Box-008");
    }

    #[test]
    fn duplicate_names_continue_the_source_series() {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let opts = CadTessellationOptions::default();
        let cube = || opencascade::primitives::Shape::cube(2.0);

        let numbered = doc.add_part("Box-001", cube(), &opts).expect("tessellates");
        assert_eq!(doc.duplicate_name(numbered), "Box-002");

        // An unnumbered name starts its own series rather than becoming Bracket-Bracket.
        let plain = doc.add_part("Bracket", cube(), &opts).expect("tessellates");
        assert_eq!(doc.duplicate_name(plain), "Bracket-001");

        // A non-numeric suffix is part of the name, not a counter.
        let lettered = doc.add_part("Part-A", cube(), &opts).expect("tessellates");
        assert_eq!(doc.duplicate_name(lettered), "Part-A-001");
    }

    #[test]
    fn rename_updates_the_part_and_its_node() {
        let (mut doc, part, _) = doc_with_box();

        doc.rename_part(part, "Bracket");

        assert_eq!(doc.get_part(part).unwrap().name, "Bracket");
        assert_eq!(node_name(&doc, part).as_deref(), Some("Bracket"));
        assert_eq!(doc.undo_label(), Some("Rename box"));
    }

    #[test]
    fn rename_undo_redo_round_trips_both_copies() {
        let (mut doc, part, _) = doc_with_box();
        doc.rename_part(part, "Bracket");

        doc.undo().expect("undo rename");
        assert_eq!(doc.get_part(part).unwrap().name, "box");
        assert_eq!(node_name(&doc, part).as_deref(), Some("box"));

        doc.redo().expect("redo rename");
        assert_eq!(doc.get_part(part).unwrap().name, "Bracket");
        assert_eq!(node_name(&doc, part).as_deref(), Some("Bracket"));
    }

    #[test]
    fn rename_trims_and_ignores_blank_or_unchanged_names() {
        let (mut doc, part, _) = doc_with_box();
        let steps_before = doc.undo_label().map(str::to_owned);

        doc.rename_part(part, "   ");
        assert_eq!(doc.get_part(part).unwrap().name, "box");
        assert_eq!(doc.undo_label().map(str::to_owned), steps_before, "blank records nothing");

        doc.rename_part(part, "box");
        assert_eq!(doc.undo_label().map(str::to_owned), steps_before, "unchanged records nothing");

        doc.rename_part(part, "  Bracket  ");
        assert_eq!(doc.get_part(part).unwrap().name, "Bracket", "surrounding space is trimmed");
    }

    #[test]
    fn rename_survives_delete_and_undo() {
        let (mut doc, part, _) = doc_with_box();
        doc.rename_part(part, "Bracket");
        doc.remove_part(part);

        doc.undo().expect("undo delete");
        assert_eq!(doc.get_part(part).unwrap().name, "Bracket", "resurrects under the new name");
        assert_eq!(node_name(&doc, part).as_deref(), Some("Bracket"));

        doc.undo().expect("undo rename");
        assert_eq!(doc.get_part(part).unwrap().name, "box");
        assert_eq!(node_name(&doc, part).as_deref(), Some("box"));
    }

    #[test]
    fn generation_rises_with_every_part_change() {
        let (mut doc, part, _) = doc_with_box();
        let up = Matrix4::from_translation(Vector3::new(0.0, 1.0, 0.0));
        let mut seen = vec![doc.generation()];

        doc.rename_part(part, "Bracket");
        seen.push(doc.generation());
        doc.bake_transform(part, up).expect("bake succeeds");
        seen.push(doc.generation());
        doc.undo().expect("undo bake");
        seen.push(doc.generation());
        doc.remove_part(part);
        seen.push(doc.generation());
        doc.undo().expect("undo remove");
        seen.push(doc.generation());

        assert!(seen.windows(2).all(|w| w[1] > w[0]), "generation must rise at every step: {seen:?}");
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

        let result = doc.tweak_faces(part, &[face], squash);

        assert!(result.is_err(), "non-uniform scale must be rejected");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "failed tweak must not modify the shape");
    }
}
