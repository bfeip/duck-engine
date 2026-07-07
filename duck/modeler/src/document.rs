use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use duck_engine_scene::{Id, NodeId, Scene, Visibility};
use duck_engine_scene::cad::{CadTessellationOptions, retessellate_node, tessellate_into};
use duck_engine_scene::common::{matrix4_to_row_major_f64, Matrix4, Transform};
use opencascade::primitives::{Edge, Face, Shape, ShapeType, Wire};

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
    scene: Arc<Mutex<Scene>>,
}

impl Document {
    pub fn new(scene: Arc<Mutex<Scene>>) -> Self {
        Self {
            parts: Vec::new(),
            part_to_node: HashMap::new(),
            node_to_part: HashMap::new(),
            scene,
        }
    }

    pub fn set_scene(&mut self, scene: Arc<Mutex<Scene>>) {
        self.scene = scene;
    }

    pub fn scene(&self) -> &Arc<Mutex<Scene>> {
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
        let node = {
            let mut scene = self.scene.lock().unwrap();
            tessellate_into(&shape, &mut *scene, options, None, Some(&name))
                .context("Failed to tessellate part")?
        };
        let id = PartId::new();
        self.parts.push(CadPart { id, name, shape });
        self.part_to_node.insert(id, node);
        self.node_to_part.insert(node, id);
        Ok(id)
    }

    /// Remove a part from the CAD store, the mapping, and the scene.
    pub fn remove_part(&mut self, id: PartId) {
        self.parts.retain(|p| p.id != id);
        if let Some(node) = self.part_to_node.remove(&id) {
            self.node_to_part.remove(&node);
            self.scene.lock().unwrap().remove_node(node);
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
        let scene = self.scene.lock().unwrap();
        scene.get_node(node).map(|n| n.visibility())
    }

    /// Set the visibility of the part's scene node. No-op for an unknown part.
    pub fn set_part_visibility(&mut self, id: PartId, visibility: Visibility) {
        if let Some(node) = self.node_for_part(id) {
            self.scene.lock().unwrap().set_node_visibility(node, visibility);
        }
    }

    /// Bake a transform into the part's CAD geometry via an affine GTransform then
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

        {
            let cad_part = self
                .get_part_mut(part)
                .context("bake_transform: part not found")?;
            cad_part.shape = cad_part.shape.gtransform(mat);
        }

        let cad_part = self
            .get_part(part)
            .context("bake_transform: part not found")?;
        let mut scene = self.scene.lock().unwrap();
        retessellate_node(&cad_part.shape, &mut scene, options, node)?;
        scene.set_node_transform(node, Transform::IDENTITY);

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

        {
            let mut scene = self.scene.lock().unwrap();
            retessellate_node(&tweaked, &mut scene, options, node)?;
        }
        self.get_part_mut(part)
            .context("tweak_faces: part not found")?
            .shape = tweaked;

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

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::Vector3;

    fn doc_with_box() -> (Document, PartId, NodeId) {
        let scene = Arc::new(Mutex::new(Scene::new()));
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
    fn tweak_faces_rejects_non_similarity_transform() {
        let (mut doc, part, _) = doc_with_box();
        let face = top_face_index(&doc, part);
        let squash = Matrix4::from_nonuniform_scale(1.0, 0.5, 1.0);

        let result = doc.tweak_faces(part, &[face], squash, &CadTessellationOptions::default());

        assert!(result.is_err(), "non-uniform scale must be rejected");
        assert!((max_y(&doc, part) - 2.0).abs() < 1e-6, "failed tweak must not modify the shape");
    }
}
