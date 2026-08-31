use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::{
    classify_shape, tessellate_into_with_materials, CadTessellationOptions, GeometryClass,
};
use duck_engine_scene::common::Transform;
use duck_engine_scene::resource::{FaceMaterialHandle, LineMaterialHandle, NodeId, Visibility};
use duck_engine_scene::{Scene, SceneData};
use opencascade::primitives::Shape;

use crate::document::Document;

/// Tracks the transient scene geometry of an in-progress operation: the preview
/// node(s) it creates and the source node(s) it hides.
///
/// [`cancel`](Self::cancel) removes the previews and restores hidden sources;
/// [`commit`](Self::commit) removes the previews and hands the still-hidden
/// sources back to the caller to delete. After either — and on drop — the
/// session is inert. Removing a preview node releases everything it owned, so
/// its mesh (and, at session end, the shared materials) are freed too.
pub struct PreviewSession {
    document: Arc<Mutex<Document>>,
    previews: Vec<NodeId>,
    hidden: Vec<NodeId>,
    /// Materials shared by every preview tessellation of the same geometry
    /// class. Created on first use, dropped with the session.
    materials: Option<PreviewMaterials>,
}

/// The material pair a session hands to its preview tessellations, along with
/// the [`GeometryClass`] it was instantiated for. Reused across rebuilds — e.g.
/// per cursor move during a drag — so only a class change (a sketch becoming a
/// solid) mints a fresh pair.
struct PreviewMaterials {
    class: GeometryClass,
    face: FaceMaterialHandle,
    line: LineMaterialHandle,
}

impl PreviewSession {
    /// A session with no previews and no hidden sources, bound to `document`.
    pub fn new(document: Arc<Mutex<Document>>) -> Self {
        Self { document, previews: Vec::new(), hidden: Vec::new(), materials: None }
    }

    /// The document's current scene.
    fn scene(&self) -> Scene {
        self.document.lock().unwrap().scene().clone()
    }

    /// The preview materials for `shape`, instantiated from the option templates
    /// for its geometry class and reused while that class holds.
    fn materials(
        &mut self,
        scene: &Scene,
        shape: &Shape,
        options: &CadTessellationOptions,
    ) -> (FaceMaterialHandle, LineMaterialHandle) {
        let class = classify_shape(shape);
        let reusable = self.materials.as_ref().is_some_and(|materials| {
            // A pair from a previous scene (after a scene swap) is stale, as is
            // one whose class no longer matches the shape being previewed.
            materials.class == class && scene.get_face_material(materials.face.id()).is_some()
        });
        if !reusable {
            let (face, line) = options.materials_for(shape);
            let mut scene = scene.lock();
            // Dropped under the guard; the outgoing preview node releases its own
            // reference when it is removed, and the materials are reaped then.
            self.materials = Some(PreviewMaterials {
                class,
                face: scene.add_face_material(face.clone().with_fresh_id()),
                line: scene.add_line_material(line.clone().with_fresh_id()),
            });
        }
        let materials = self.materials.as_ref().expect("just instantiated");
        (materials.face.clone(), materials.line.clone())
    }

    /// True while no previews are tracked and no sources are hidden.
    pub fn is_empty(&self) -> bool {
        self.previews.is_empty() && self.hidden.is_empty()
    }

    /// The tracked preview nodes, e.g. to exclude from snap resolution.
    pub fn preview_nodes(&self) -> &[NodeId] {
        &self.previews
    }

    /// The sole tracked preview node, or `None` if there are zero or more than one.
    pub fn preview_node(&self) -> Option<NodeId> {
        match self.previews.as_slice() {
            [node] => Some(*node),
            _ => None,
        }
    }

    /// Tessellate `shape` into the scene and track the resulting node as a
    /// preview. Returns `None` without modifying anything if tessellation fails.
    pub fn add_preview_from_shape(
        &mut self,
        shape: &Shape,
        options: &CadTessellationOptions,
        name: &str,
    ) -> Option<NodeId> {
        let scene = self.scene();
        let (face, line) = self.materials(&scene, shape, options);
        let node =
            tessellate_into_with_materials(shape, &scene, options, &face, &line, None, Some(name))
                .ok()?
                .id();
        self.previews.push(node);
        Some(node)
    }

    /// Track an externally-built node as a preview.
    pub fn add_preview_node(&mut self, node: NodeId) {
        self.previews.push(node);
    }

    /// Replace all tracked previews with a freshly-tessellated node, but only on
    /// success: if the build fails the existing previews are left untouched, so
    /// the preview never flickers. Returns the new node on success.
    pub fn try_replace_preview(
        &mut self,
        shape: &Shape,
        options: &CadTessellationOptions,
        name: &str,
    ) -> Option<NodeId> {
        let scene = self.scene();
        let (face, line) = self.materials(&scene, shape, options);
        let node =
            tessellate_into_with_materials(shape, &scene, options, &face, &line, None, Some(name))
                .ok()?
                .id();

        let mut scene = scene.lock();
        for old in self.previews.drain(..) {
            scene.remove_node(old);
        }
        self.previews.push(node);
        Some(node)
    }

    /// Remove all previews and restore hidden sources, leaving the session ready
    /// to be rebuilt from scratch.
    pub fn clear_previews(&mut self) {
        let scene = self.scene();
        let mut scene = scene.lock();
        for node in self.previews.drain(..) {
            scene.remove_node(node);
        }
        Self::restore_hidden(&mut scene, &mut self.hidden);
    }

    /// Set the visibility of every tracked preview.
    pub fn set_preview_visibility(&self, visibility: Visibility) {
        let scene = self.scene();
        let mut scene = scene.lock();
        for &node in &self.previews {
            scene.set_node_visibility(node, visibility);
        }
    }

    /// Set the transform of every tracked preview.
    pub fn set_preview_transform(&self, transform: Transform) {
        let scene = self.scene();
        let mut scene = scene.lock();
        for &node in &self.previews {
            scene.set_node_transform(node, transform);
        }
    }

    /// Hide `node` for the preview's duration and track it for restoration on
    /// cancel or drop. Does nothing if it is already hidden by this session.
    pub fn hide_source_node(&mut self, node: NodeId) {
        if self.hidden.contains(&node) {
            return;
        }
        self.scene().set_node_visibility(node, Visibility::Invisible);
        self.hidden.push(node);
    }

    /// Remove all previews and restore every hidden source. Idempotent; the
    /// session is inert afterwards.
    pub fn cancel(&mut self) {
        let scene = self.scene();
        let mut scene = scene.lock();
        for node in self.previews.drain(..) {
            scene.remove_node(node);
        }
        Self::restore_hidden(&mut scene, &mut self.hidden);
        // Dropped under the guard; reaped when it releases.
        self.materials = None;
    }

    /// Remove all previews and return the still-hidden source nodes, transferring
    /// ownership to the caller — the committing operation is expected to delete
    /// them (they stay hidden, not restored). Idempotent; the session is inert
    /// afterwards.
    #[must_use = "the returned hidden sources must be deleted by the committing operation"]
    pub fn commit(&mut self) -> Vec<NodeId> {
        let scene = self.scene();
        let mut scene = scene.lock();
        for node in self.previews.drain(..) {
            scene.remove_node(node);
        }
        self.materials = None;
        std::mem::take(&mut self.hidden)
    }

    fn restore_hidden(scene: &mut SceneData, hidden: &mut Vec<NodeId>) {
        for node in hidden.drain(..) {
            scene.set_node_visibility(node, Visibility::Visible);
        }
    }
}

impl Drop for PreviewSession {
    /// Safety net: an un-cancelled, un-committed session tears itself down like
    /// [`cancel`](Self::cancel). After `cancel`/`commit` both lists are empty, so
    /// this is a no-op.
    fn drop(&mut self) {
        if self.is_empty() {
            return;
        }
        // A panic in drop while the document lock is poisoned would abort the
        // process; skip teardown on poison instead. The scene handle absorbs
        // poison itself, so it needs no such guard.
        let scene = match self.document.lock() {
            Ok(doc) => doc.scene().clone(),
            Err(_) => return,
        };
        let mut scene = scene.lock();
        for &node in &self.previews {
            scene.remove_node(node);
        }
        for &node in &self.hidden {
            scene.set_node_visibility(node, Visibility::Visible);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_scene::common::RgbaColor;
    use duck_engine_scene::resource::{FaceMaterial, NodePayload};
    use opencascade::primitives::{Face, Wire};

    fn document() -> Arc<Mutex<Document>> {
        let scene = Scene::default();
        Arc::new(Mutex::new(Document::new(scene)))
    }

    fn unit_shape() -> Shape {
        Shape::sphere(1.0).build()
    }

    /// A lone face — free geometry, the class a sketch preview has.
    fn unit_face() -> Shape {
        Face::from_wire(&Wire::rect(1.0, 1.0).unwrap()).unwrap().into()
    }

    const SOLID_COLOR: RgbaColor = RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 };
    const FREE_COLOR: RgbaColor = RgbaColor { r: 0.0, g: 0.0, b: 1.0, a: 0.3 };

    /// Options whose two classes are told apart by base color.
    fn classed_options() -> CadTessellationOptions {
        CadTessellationOptions {
            face_material: FaceMaterial::new().with_base_color_factor(SOLID_COLOR),
            free_face_material: Some(FaceMaterial::new().with_base_color_factor(FREE_COLOR)),
            ..Default::default()
        }
    }

    const EPSILON: f32 = 1e-6;

    fn assert_color(actual: RgbaColor, expected: RgbaColor) {
        let close = (actual.r - expected.r).abs() < EPSILON
            && (actual.g - expected.g).abs() < EPSILON
            && (actual.b - expected.b).abs() < EPSILON
            && (actual.a - expected.a).abs() < EPSILON;
        assert!(close, "expected color {expected:?}, got {actual:?}");
    }

    /// The base color of the face material a preview node's instance is drawn with.
    fn preview_color(document: &Arc<Mutex<Document>>, node: NodeId) -> RgbaColor {
        with_scene(document, |scene| {
            let NodePayload::Instance(instance) = scene.get_node(node).unwrap().payload() else {
                panic!("preview node carries no instance");
            };
            let material = scene.get_instance(instance.id()).unwrap().face_material().unwrap();
            scene.get_face_material(material).unwrap().base_color_factor()
        })
    }

    /// Tessellate a standalone node directly into the document's scene (a stand-in
    /// source part).
    fn add_source(document: &Arc<Mutex<Document>>) -> NodeId {
        let scene = document.lock().unwrap().scene().clone();
        duck_engine_scene::cad::tessellate_into(
            &unit_shape(),
            &scene,
            &CadTessellationOptions::default(),
            None,
            Some("src"),
        )
        .unwrap()
        .id()
    }

    fn with_scene<R>(document: &Arc<Mutex<Document>>, f: impl FnOnce(&SceneData) -> R) -> R {
        let scene = document.lock().unwrap().scene().clone();
        let scene = scene.lock();
        f(&scene)
    }

    fn visibility(document: &Arc<Mutex<Document>>, node: NodeId) -> Visibility {
        with_scene(document, |s| s.get_node(node).unwrap().visibility())
    }

    #[test]
    fn add_preview_then_cancel_frees_resources() {
        let document = document();
        let mut session = PreviewSession::new(document.clone());
        assert!(session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p").is_some());
        with_scene(&document, |s| {
            assert_eq!(s.node_count(), 1);
            assert!(s.mesh_count() >= 1);
        });

        session.cancel();
        with_scene(&document, |s| {
            assert_eq!(s.node_count(), 0);
            assert_eq!(s.mesh_count(), 0);
            assert_eq!(s.instance_count(), 0);
        });
    }

    #[test]
    fn preview_replacements_share_one_material_pair() {
        let document = document();
        let mut session = PreviewSession::new(document.clone());
        // Simulates a drag: the preview is rebuilt repeatedly. Every rebuild
        // must reuse the session's material pair instead of minting a new one.
        for _ in 0..3 {
            assert!(session
                .try_replace_preview(&unit_shape(), &CadTessellationOptions::default(), "p")
                .is_some());
        }
        with_scene(&document, |s| {
            assert_eq!(s.face_material_count(), 1);
            assert_eq!(s.line_material_count(), 1);
            assert_eq!(s.mesh_count(), 1, "replaced preview meshes must be freed");
        });

        session.cancel();
        with_scene(&document, |s| {
            assert_eq!(s.face_material_count(), 0);
            assert_eq!(s.line_material_count(), 0);
            assert_eq!(s.mesh_count(), 0);
        });
    }

    #[test]
    fn replacement_of_another_class_restyles_the_preview() {
        let document = document();
        let options = classed_options();
        let mut session = PreviewSession::new(document.clone());

        // A sketch stage (lone face) followed by a solid stage, as the box and
        // cylinder tools do: the solid must not inherit the sketch materials.
        let sketch = session.add_preview_from_shape(&unit_face(), &options, "p").unwrap();
        assert_color(preview_color(&document, sketch), FREE_COLOR);

        let solid = session.try_replace_preview(&unit_shape(), &options, "p").unwrap();
        assert_color(preview_color(&document, solid), SOLID_COLOR);
        // The superseded pair goes with the preview node that held it.
        with_scene(&document, |s| {
            assert_eq!(s.face_material_count(), 1);
            assert_eq!(s.line_material_count(), 1);
        });

        session.cancel();
        with_scene(&document, |s| {
            assert_eq!(s.face_material_count(), 0);
            assert_eq!(s.line_material_count(), 0);
        });
    }

    #[test]
    fn hide_source_restored_on_cancel() {
        let document = document();
        let source = add_source(&document);
        let mut session = PreviewSession::new(document.clone());

        session.hide_source_node(source);
        assert_eq!(visibility(&document, source), Visibility::Invisible);

        session.cancel();
        assert_eq!(visibility(&document, source), Visibility::Visible);
    }

    #[test]
    fn commit_hands_back_hidden_sources_kept_hidden() {
        let document = document();
        let source = add_source(&document);
        let mut session = PreviewSession::new(document.clone());

        session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p");
        session.hide_source_node(source);

        let hidden = session.commit();
        assert_eq!(hidden, vec![source]);
        // Preview gone, source still hidden for the committing op to delete.
        assert_eq!(visibility(&document, source), Visibility::Invisible);
        with_scene(&document, |s| assert_eq!(s.node_count(), 1));
    }

    #[test]
    fn commit_then_drop_is_noop() {
        let document = document();
        let source = add_source(&document);
        {
            let mut session = PreviewSession::new(document.clone());
            session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p");
            session.hide_source_node(source);
            let _ = session.commit();
            // Drop here must not restore the source or touch the scene.
        }
        assert_eq!(visibility(&document, source), Visibility::Invisible);
        with_scene(&document, |s| assert_eq!(s.node_count(), 1));
    }

    #[test]
    fn drop_reverts_like_cancel() {
        let document = document();
        let source = add_source(&document);
        let baseline = with_scene(&document, |s| s.mesh_count());
        {
            let mut session = PreviewSession::new(document.clone());
            session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p");
            session.hide_source_node(source);
            // No cancel/commit: Drop must behave like cancel.
        }
        assert_eq!(visibility(&document, source), Visibility::Visible);
        with_scene(&document, |s| assert_eq!(s.mesh_count(), baseline));
    }

    #[test]
    fn rebuild_swaps_the_single_preview() {
        let document = document();
        let mut session = PreviewSession::new(document.clone());
        let first = session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p").unwrap();
        let second = session.try_replace_preview(&unit_shape(), &CadTessellationOptions::default(), "p").unwrap();

        assert_ne!(first, second);
        assert_eq!(session.preview_node(), Some(second));
        // Old preview removed: only one node remains.
        with_scene(&document, |s| assert_eq!(s.node_count(), 1));
    }

    #[test]
    fn resolves_current_scene_after_swap() {
        let document = document();
        let mut session = PreviewSession::new(document.clone());

        // Swap in a fresh scene, as new-document / file-load does.
        document.lock().unwrap().set_scene(Scene::default());

        // The preview must land in the new scene, not the one present at construction.
        session.add_preview_from_shape(&unit_shape(), &CadTessellationOptions::default(), "p");
        with_scene(&document, |s| assert_eq!(s.node_count(), 1));
    }
}
