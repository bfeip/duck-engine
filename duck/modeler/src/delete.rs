//! Deleting the selected parts.

use std::sync::Mutex;

use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::Document;
use crate::ops::delete::delete_parts;

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

    let deleted = delete_parts(&mut document.lock().unwrap(), &nodes);
    for &node in &deleted {
        selection.remove_node(node);
    }
    deleted.len()
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::common::Transform;
    use duck_engine_scene::resource::{NodeFlags, NodeId, SubGeometryElement, SubGeometryKind};

    use super::*;
    use crate::document::PartId;
    use crate::testing;

    /// A document holding `count` boxes, shared as the app holds it.
    fn doc_with_boxes(count: usize) -> (Mutex<Document>, Vec<(PartId, NodeId)>) {
        let (doc, parts) = testing::doc_with_boxes(count);
        (Mutex::new(doc), parts)
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
}
