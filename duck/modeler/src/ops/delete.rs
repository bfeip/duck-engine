use duck_engine_scene::resource::NodeId;

use crate::document::Document;

/// Deletes the parts at `nodes` as one undo step, skipping nodes that aren't
/// parts. Returns the nodes whose parts were deleted.
pub fn delete_parts(doc: &mut Document, nodes: &[NodeId]) -> Vec<NodeId> {
    let mut doc = doc.undo_scope("Delete");
    let mut deleted = Vec::new();
    for &node in nodes {
        let Some(part) = doc.part_for_node(node) else { continue };
        doc.remove_part(part);
        deleted.push(node);
    }
    deleted
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::doc_with_boxes;

    #[test]
    fn deleting_several_parts_is_one_undo_step() {
        let (mut doc, parts) = doc_with_boxes(3);
        let nodes: Vec<_> = parts.iter().map(|&(_, node)| node).collect();

        assert_eq!(delete_parts(&mut doc, &nodes), nodes);
        assert_eq!(doc.parts().count(), 0);
        assert_eq!(doc.undo_label(), Some("Delete"));

        doc.undo().expect("undo succeeds");
        assert_eq!(doc.parts().count(), 3, "one undo restores every part");
        for &(part, node) in &parts {
            assert_eq!(doc.node_for_part(part), Some(node));
        }
    }
}
