use std::sync::{Arc, Mutex};

use duck_engine_scene::cad::{tessellate_into, CadTessellationOptions};
use duck_engine_scene::{NodeId, Scene, SceneData, Visibility};
use opencascade::primitives::Shape;

use crate::document::Document;

/// Tracks the transient scene geometry of an in-progress operation: the preview
/// node(s) it creates and the source node(s) it hides.
///
/// [`cancel`](Self::cancel) removes the previews and restores hidden sources;
/// [`commit`](Self::commit) removes the previews and hands the still-hidden
/// sources back to the caller to delete. After either — and on drop — the
/// session is inert. Previews are removed with [`SceneData::cleanup_node`] so their
/// meshes and materials are freed too.
pub struct PreviewSession {
    document: Arc<Mutex<Document>>,
    previews: Vec<NodeId>,
    hidden: Vec<NodeId>,
}

impl PreviewSession {
    /// A session with no previews and no hidden sources, bound to `document`.
    pub fn new(document: Arc<Mutex<Document>>) -> Self {
        Self { document, previews: Vec::new(), hidden: Vec::new() }
    }

    /// The document's current scene.
    fn scene(&self) -> Scene {
        self.document.lock().unwrap().scene().clone()
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
        let node = tessellate_into(shape, &scene, options, None, Some(name)).ok()?;
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
        let node = tessellate_into(shape, &scene, options, None, Some(name)).ok()?;

        let mut scene = scene.lock();
        for old in self.previews.drain(..) {
            scene.cleanup_node(old);
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
            scene.cleanup_node(node);
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
            scene.cleanup_node(node);
        }
        Self::restore_hidden(&mut scene, &mut self.hidden);
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
            scene.cleanup_node(node);
        }
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
            scene.cleanup_node(node);
        }
        for &node in &self.hidden {
            scene.set_node_visibility(node, Visibility::Visible);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> Arc<Mutex<Document>> {
        let scene = Scene::default();
        Arc::new(Mutex::new(Document::new(scene)))
    }

    fn unit_shape() -> Shape {
        Shape::sphere(1.0).build()
    }

    /// Tessellate a standalone node directly into the document's scene (a stand-in
    /// source part).
    fn add_source(document: &Arc<Mutex<Document>>) -> NodeId {
        let scene = document.lock().unwrap().scene().clone();
        tessellate_into(&unit_shape(), &scene, &CadTessellationOptions::default(), None, Some("src"))
            .unwrap()
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
