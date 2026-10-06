//! What the selection designates for a tool to act on.

use duck_engine_scene::resource::{NodeId, SubGeometryKind};
use duck_engine_viewer::selection::{SelectionItem, SelectionManager};

use crate::document::Document;
use crate::ops::boolean::BooleanTarget;
use crate::ops::loft::LoftProfile;

/// A target the selection designates, the part it lies on, and how many
/// selected items it leaves out.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Selected<T> {
    pub node: NodeId,
    pub target: T,
    pub ignored: usize,
}

impl<T> Selected<T> {
    /// The same designation, with the target remade by `f` from its part and
    /// itself.
    pub fn map<U>(self, f: impl FnOnce(NodeId, T) -> U) -> Selected<U> {
        Selected { node: self.node, target: f(self.node, self.target), ignored: self.ignored }
    }
}

/// The selected sub-shapes of `kind` on one part, primary first.
///
/// The part is `locked` once editing has begun. Until then it is the primary's:
/// the selection's primary if that is of `kind`, else the first of `kind`
/// selected. Sub-shapes on other parts are left out.
pub fn selected_on_part(
    selection: &SelectionManager,
    kind: SubGeometryKind,
    locked: Option<NodeId>,
) -> Option<Selected<Vec<u32>>> {
    let of_kind = |item: &SelectionItem| match *item {
        SelectionItem::SubGeometry { node_id, element } if element.kind == kind => {
            Some((node_id, element.index))
        }
        _ => None,
    };
    let picked: Vec<_> = selection.iter().filter_map(of_kind).collect();
    let primary = selection.primary().as_ref().and_then(of_kind).or_else(|| picked.first().copied());
    let node = locked.or(primary.map(|(node, _)| node))?;

    let mut on_part: Vec<u32> =
        picked.iter().filter(|(on, _)| *on == node).map(|&(_, index)| index).collect();
    let ignored = picked.len() - on_part.len();
    if let Some((_, index)) = primary.filter(|(on, _)| *on == node) {
        on_part.retain(|&other| other != index);
        on_part.insert(0, index);
    }
    (!on_part.is_empty()).then_some(Selected { node, target: on_part, ignored })
}

/// The selected faces on one part, as [`selected_on_part`] picks them, or with
/// none, that part whole — no faces — if its node is selected. Selected items
/// on other parts are left out.
pub fn selected_faces_or_part(
    selection: &SelectionManager,
    locked: Option<NodeId>,
) -> Option<Selected<Vec<u32>>> {
    let (node, faces) = match selected_on_part(selection, SubGeometryKind::Face, locked) {
        Some(faces) => (faces.node, faces.target),
        None => {
            let node = locked
                .or_else(|| selection.primary().map(|item| item.node_id()))
                .filter(|&node| selection.contains(&SelectionItem::Node(node)))?;
            (node, Vec::new())
        }
    };
    let ignored = selection.iter().filter(|item| item.node_id() != node).count();
    Some(Selected { node, target: faces, ignored })
}

/// The boolean the selection designates: the primary part as its target, and
/// the other selected parts as its tools.
pub fn selected_boolean(selection: &SelectionManager) -> Option<BooleanTarget> {
    let Some(SelectionItem::Node(target)) = selection.primary() else { return None };
    let tools = selection
        .iter()
        .filter_map(|item| match *item {
            SelectionItem::Node(node) if node != target => Some(node),
            _ => None,
        })
        .collect();
    Some(BooleanTarget { target, tools })
}

/// The selected edges, in the order they were picked, as loft profiles.
pub fn selected_profiles(selection: &SelectionManager) -> Vec<LoftProfile> {
    selection
        .iter()
        .filter_map(|item| match *item {
            SelectionItem::SubGeometry { node_id, element }
                if element.kind == SubGeometryKind::Edge =>
            {
                Some(LoftProfile { node: node_id, edge_index: element.index })
            }
            _ => None,
        })
        .collect()
}

/// The selected parts, in the order they were picked.
pub fn selected_parts(selection: &SelectionManager, doc: &Document) -> Vec<NodeId> {
    selection
        .iter()
        .filter_map(|item| match *item {
            SelectionItem::Node(node) if doc.part_at(node).is_some() => Some(node),
            _ => None,
        })
        .collect()
}

/// "3 edges", noting any on other parts that are left out.
pub fn count_summary(noun: &str, count: usize, ignored: usize) -> String {
    let counted = match count {
        1 => format!("1 {noun}"),
        count => format!("{count} {noun}s"),
    };
    match ignored {
        0 => counted,
        ignored => format!("{counted} ({ignored} on other parts ignored)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::testing::{doc_with_box, edge_item, face_item};

    fn select(items: &[SelectionItem]) -> SelectionManager {
        let mut selection = SelectionManager::new();
        selection.extend(items.iter().copied());
        selection
    }

    fn faces(node: NodeId, faces: &[u32], ignored: usize) -> Option<Selected<Vec<u32>>> {
        Some(Selected { node, target: faces.to_vec(), ignored })
    }

    #[test]
    fn faces_on_one_part_are_its_target() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[face_item(part, 2), face_item(part, 0), face_item(other, 1)]);
        assert_eq!(selected_faces_or_part(&selection, None), faces(part, &[2, 0], 1));
    }

    #[test]
    fn a_part_selected_whole_is_its_target_with_no_faces() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(part), SelectionItem::Node(other)]);
        assert_eq!(selected_faces_or_part(&selection, None), faces(part, &[], 1));
    }

    /// Faces picked on a part selected whole say which of its faces to use.
    #[test]
    fn faces_win_over_their_own_part() {
        let part = NodeId::new();
        let selection = select(&[SelectionItem::Node(part), face_item(part, 3)]);
        assert_eq!(selected_faces_or_part(&selection, None), faces(part, &[3], 0));
    }

    /// Faces on one part outrank another part selected whole first, which is
    /// left out.
    #[test]
    fn faces_outrank_another_part_selected_whole() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(other), face_item(part, 1)]);
        assert_eq!(selected_faces_or_part(&selection, None), faces(part, &[1], 1));
    }

    #[test]
    fn a_locked_part_must_itself_be_selected() {
        let (part, other) = (NodeId::new(), NodeId::new());
        let selection = select(&[SelectionItem::Node(other)]);
        assert_eq!(selected_faces_or_part(&selection, Some(part)), None);

        let selection = select(&[SelectionItem::Node(other), SelectionItem::Node(part)]);
        assert_eq!(selected_faces_or_part(&selection, Some(part)), faces(part, &[], 1));
    }

    #[test]
    fn a_boolean_targets_the_primary_part_with_the_others_as_tools() {
        let (target, a, b) = (NodeId::new(), NodeId::new(), NodeId::new());
        let selection = select(&[
            SelectionItem::Node(target),
            face_item(a, 0),
            SelectionItem::Node(a),
            SelectionItem::Node(b),
        ]);
        assert_eq!(selected_boolean(&selection), Some(BooleanTarget { target, tools: vec![a, b] }));
    }

    #[test]
    fn a_boolean_needs_a_part_as_its_primary() {
        let part = NodeId::new();
        let selection = select(&[face_item(part, 0), SelectionItem::Node(part)]);
        assert_eq!(selected_boolean(&selection), None);
    }

    #[test]
    fn profiles_are_the_selected_edges_in_picking_order() {
        let (a, b) = (NodeId::new(), NodeId::new());
        let selection =
            select(&[edge_item(b, 4), face_item(a, 0), SelectionItem::Node(a), edge_item(a, 1)]);
        assert_eq!(
            selected_profiles(&selection),
            [LoftProfile { node: b, edge_index: 4 }, LoftProfile { node: a, edge_index: 1 }]
        );
    }

    #[test]
    fn selected_parts_are_the_part_nodes_in_picking_order() {
        let (doc, part) = doc_with_box();
        let stranger = NodeId::new();
        let selection =
            select(&[SelectionItem::Node(stranger), face_item(part, 0), SelectionItem::Node(part)]);
        assert_eq!(selected_parts(&selection, &doc), [part]);
    }
}
