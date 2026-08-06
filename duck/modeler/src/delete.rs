//! Keyboard-driven deletion of selected parts.

use std::sync::Mutex;

use duck_engine_viewer::bindings::{InputBinding, InputMap};
use duck_engine_viewer::event::{DeviceEvent, Event, EventContext};
use duck_engine_viewer::input::{ElementState, Key, KeyEvent, Modifiers, NamedKey};
use duck_engine_viewer::operator::Operator;
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::Document;

/// Last-priority operator that turns an unclaimed 'X'/Delete press into a
/// deferred delete request, applied by the app's per-frame update.
///
/// Registered behind the tool host so an active tool keeps any key it
/// consumes (e.g. 'x' as the transform axis constraint).
pub struct DeleteOperator {
    bindings: InputMap<()>,
    pending: bool,
}

impl DeleteOperator {
    pub fn new() -> Self {
        Self {
            bindings: InputMap::new()
                .bind(
                    InputBinding::Key { key: Key::Character('x'), modifiers: Modifiers::default() },
                    (),
                )
                .bind(
                    InputBinding::Key {
                        key: Key::Named(NamedKey::Delete),
                        modifiers: Modifiers::default(),
                    },
                    (),
                ),
            pending: false,
        }
    }

    /// Records a delete request if the key matches a binding.
    /// Returns `true` when the key was claimed.
    fn handle_key(&mut self, key_event: &KeyEvent, modifiers: Modifiers) -> bool {
        if key_event.state != ElementState::Pressed || key_event.repeat {
            return false;
        }
        if self.bindings.actions_for_key(&key_event.logical_key, modifiers).is_empty() {
            return false;
        }
        self.pending = true;
        true
    }

    /// Takes the pending delete request, if any.
    pub fn take_pending(&mut self) -> bool {
        std::mem::take(&mut self.pending)
    }
}

impl Operator for DeleteOperator {
    fn dispatch(&mut self, event: &Event, ctx: &mut EventContext) -> bool {
        let Event::Device(DeviceEvent::KeyboardInput { event: key_event, .. }) = event else {
            return false;
        };
        self.handle_key(key_event, ctx.modifiers)
    }

    fn name(&self) -> &str {
        "Delete"
    }
}

/// Deletes every whole-part selection ([`SelectionItem::Node`]) from the
/// document and purges the deleted nodes from the selection. Sub-geometry
/// selections and nodes without a part are left untouched.
///
/// Returns the number of parts deleted.
pub fn delete_selected_parts(document: &Mutex<Document>, selection: &mut SelectionManager) -> usize {
    let nodes: Vec<_> = selection
        .iter()
        .filter_map(|item| match item {
            SelectionItem::Node(node) => Some(*node),
            SelectionItem::SubGeometry { .. } => None,
        })
        .collect();

    let mut document = document.lock().unwrap();
    // One undo step covers the whole multi-selection.
    let mut document = document.undo_scope("Delete");
    let mut deleted = 0;
    for node in nodes {
        let Some(part) = document.part_for_node(node) else { continue };
        document.remove_part(part);
        selection.remove_node(node);
        deleted += 1;
    }
    deleted
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::Transform;
    use duck_engine_scene::resource::{NodeFlags, NodeId, SubGeometryElement, SubGeometryKind};
    use duck_engine_scene::Scene;
    use duck_engine_viewer::input::PhysicalKey;

    use super::*;
    use crate::document::PartId;

    fn key_press(key: Key) -> KeyEvent {
        KeyEvent {
            physical_key: PhysicalKey::Unidentified,
            logical_key: key,
            state: ElementState::Pressed,
            repeat: false,
        }
    }

    #[test]
    fn matches_bound_keys() {
        let mut op = DeleteOperator::new();
        assert!(op.handle_key(&key_press(Key::Character('x')), Modifiers::default()));
        assert!(op.take_pending());

        // Case-insensitive via InputMap normalization.
        assert!(op.handle_key(&key_press(Key::Character('X')), Modifiers::default()));
        assert!(op.take_pending());

        assert!(op.handle_key(&key_press(Key::Named(NamedKey::Delete)), Modifiers::default()));
        assert!(op.take_pending());
        assert!(!op.take_pending(), "take_pending must reset the request");

        assert!(!op.handle_key(&key_press(Key::Character('q')), Modifiers::default()));
        assert!(!op.take_pending());
    }

    #[test]
    fn ignores_repeat_release_and_modifiers() {
        let mut op = DeleteOperator::new();

        let mut repeat = key_press(Key::Character('x'));
        repeat.repeat = true;
        assert!(!op.handle_key(&repeat, Modifiers::default()));

        let mut released = key_press(Key::Character('x'));
        released.state = ElementState::Released;
        assert!(!op.handle_key(&released, Modifiers::default()));

        let ctrl = Modifiers { control: true, ..Modifiers::default() };
        assert!(!op.handle_key(&key_press(Key::Character('x')), ctrl));

        assert!(!op.take_pending());
    }

    fn doc_with_boxes(count: usize) -> (Arc<Mutex<Document>>, Vec<(PartId, NodeId)>) {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let parts = (0..count)
            .map(|i| {
                let part = doc
                    .add_part(
                        format!("box {i}"),
                        opencascade::primitives::Shape::cube(2.0),
                        &CadTessellationOptions::default(),
                    )
                    .expect("box tessellates");
                (part, doc.node_for_part(part).expect("part has a node"))
            })
            .collect();
        (Arc::new(Mutex::new(doc)), parts)
    }

    #[test]
    fn deletes_node_selections_only() {
        let (doc, parts) = doc_with_boxes(2);
        let (deleted_part, deleted_node) = parts[0];
        let (kept_part, kept_node) = parts[1];

        let mut selection = SelectionManager::new();
        selection.add(SelectionItem::Node(deleted_node));
        selection.add(SelectionItem::SubGeometry {
            node_id: kept_node,
            element: SubGeometryElement::new(SubGeometryKind::Face, 0),
        });

        assert_eq!(delete_selected_parts(&doc, &mut selection), 1);

        let doc = doc.lock().unwrap();
        assert!(doc.get_part(deleted_part).is_none());
        assert_eq!(doc.node_for_part(deleted_part), None);
        // The undo snapshot keeps the node alive, but detached from the tree.
        assert!(!doc.scene().lock().is_node_attached(deleted_node));
        assert!(!selection.is_node_selected(deleted_node), "selection entries must be purged");

        assert!(doc.get_part(kept_part).is_some(), "sub-geometry selection must not delete");
        assert!(doc.scene().get_node(kept_node).is_some());
        assert_eq!(selection.len(), 1, "kept part's selection survives");
    }

    #[test]
    fn skips_nodes_without_a_part() {
        let (doc, _) = doc_with_boxes(0);
        let unmapped = doc
            .lock()
            .unwrap()
            .scene()
            .lock()
            .add_node(None, Some("free node".into()), Transform::IDENTITY, NodeFlags::NONE)
            .expect("node adds")
            .id();

        let mut selection = SelectionManager::new();
        selection.add(SelectionItem::Node(unmapped));

        assert_eq!(delete_selected_parts(&doc, &mut selection), 0);
        assert!(selection.is_node_selected(unmapped), "unmapped selection is left untouched");
    }

    #[test]
    fn multi_delete_is_one_undo_step() {
        let (doc, parts) = doc_with_boxes(3);

        let mut selection = SelectionManager::new();
        for &(_, node) in &parts {
            selection.add(SelectionItem::Node(node));
        }
        assert_eq!(delete_selected_parts(&doc, &mut selection), 3);

        let mut doc = doc.lock().unwrap();
        assert_eq!(doc.undo_label(), Some("Delete"));
        doc.undo().expect("undo succeeds");
        assert_eq!(doc.parts().count(), 3, "one undo restores the whole selection");
        for &(part, node) in &parts {
            assert_eq!(doc.node_for_part(part), Some(node));
        }
    }
}
