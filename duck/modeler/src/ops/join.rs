use anyhow::{ensure, Context, Result};
use duck_engine_scene::common::Real;
use duck_engine_scene::resource::NodeId;
use opencascade::primitives::{Edge, Sewing, Shape, ShapeType, Shell};

use crate::document::{has_solid, interactive_fuzz, CadPart, Document, PartId};

/// OCCT's own sewing tolerance, the least a join sews with.
const MIN_TOLERANCE: f64 = 1e-6;

/// Sheets sewn together along the edges they share: each group that touches,
/// as a solid where it closes and a shell where it doesn't.
pub struct Join {
    /// The groups the join changes, to commit in place of their sources.
    pub groups: Vec<JoinGroup>,
    /// The edges of the sewn sheets that bound a single face.
    pub open_edges: Vec<Edge>,
}

/// One joined piece, and the parts it consumes in picking order.
pub struct JoinGroup {
    pub shape: Shape,
    pub sources: Vec<NodeId>,
}

impl Join {
    /// The parts the join consumes, each once.
    pub fn consumed(&self) -> Vec<NodeId> {
        let mut consumed = Vec::new();
        for &node in self.groups.iter().flat_map(|group| &group.sources) {
            if !consumed.contains(&node) {
                consumed.push(node);
            }
        }
        consumed
    }
}

/// Whether `part` is a sheet a join can sew: faces, but no solid.
pub fn is_sheet(part: &CadPart) -> bool {
    !has_solid(part.shape()) && !part.elements().faces().is_empty()
}

/// The tolerance a join sews `nodes` with unless told otherwise: enough to
/// close the gaps that snapping to tessellated geometry leaves.
pub fn default_tolerance(doc: &Document, nodes: &[NodeId]) -> Real {
    let shapes = nodes.iter().filter_map(|&node| doc.part_at(node)).map(CadPart::shape);
    interactive_fuzz(shapes).max(MIN_TOLERANCE) as Real
}

/// Sews the sheets among `nodes` together wherever their edges lie within
/// `tolerance`, without touching the document. Parts that aren't sheets are
/// left out.
///
/// A part is consumed when its faces join another's, or when joining makes
/// it something else, such as a closed shell a solid. Every piece of a
/// consumed part is committed; a sheet that touches nothing is left alone.
pub fn build_join(doc: &Document, nodes: &[NodeId], tolerance: Real) -> Result<Join> {
    // Deep copies: sewing may retouch its inputs, and the parts must survive
    // an abandoned preview as they were.
    let sources: Vec<(NodeId, ShapeType, Shape)> = nodes
        .iter()
        .filter_map(|&node| {
            let part = doc.part_at(node).filter(|part| is_sheet(part))?;
            Some((node, part.shape().shape_type(), part.shape().deep_copy()))
        })
        .collect();
    ensure!(!sources.is_empty(), "Nothing to join: select sheets");

    let sewing = Sewing::new(sources.iter().map(|(_, _, shape)| shape), f64::from(tolerance))?;
    let sewn = sewing.shape();
    let pieces: Vec<Shape> = match sewn.shape_type() {
        ShapeType::Compound => sewn.sub_shapes().collect(),
        _ => vec![sewn],
    };

    // The sources whose faces each piece holds, in picking order.
    let mut holders: Vec<Vec<usize>> = vec![Vec::new(); pieces.len()];
    for (index, (_, _, shape)) in sources.iter().enumerate() {
        for face in shape.faces() {
            let sewn = sewing.sewn(&Shape::from(face));
            let Some(piece) =
                pieces.iter().position(|piece| piece.faces().any(|face| Shape::from(face).is_same(&sewn)))
            else {
                continue;
            };
            if !holders[piece].contains(&index) {
                holders[piece].push(index);
            }
        }
    }

    let pieces: Vec<Shape> = pieces.into_iter().map(close).collect();
    let consumed: Vec<bool> = sources
        .iter()
        .enumerate()
        .map(|(index, (_, shape_type, _))| {
            let held: Vec<_> = holders.iter().zip(&pieces).filter(|(holders, _)| holders.contains(&index)).collect();
            let joins_another = held.iter().any(|(holders, _)| holders.len() > 1);
            let becomes_another = matches!(held[..], [(_, piece)] if piece.shape_type() != *shape_type);
            joins_another || becomes_another
        })
        .collect();

    let groups: Vec<JoinGroup> = pieces
        .into_iter()
        .zip(holders)
        .filter(|(_, holders)| holders.iter().any(|&index| consumed[index]))
        .map(|(shape, holders)| JoinGroup {
            shape,
            sources: holders.into_iter().map(|index| sources[index].0).collect(),
        })
        .collect();
    ensure!(!groups.is_empty(), "Nothing to join: no two sheets meet within the tolerance");
    Ok(Join { groups, open_edges: sewing.free_edges() })
}

/// `piece` as the solid it bounds if it closes, else as it is.
fn close(piece: Shape) -> Shape {
    let shell: Shape = match piece.shape_type() {
        ShapeType::Shell => piece.clone(),
        // A face closed on itself, such as a whole sphere, is a shell of one.
        ShapeType::Face => Shell::from_faces(piece.faces()).into(),
        _ => return piece,
    };
    if !shell.is_closed() {
        return piece;
    }
    match shell.to_solid() {
        Ok(solid) => solid,
        Err(e) => {
            log::warn!("A closed join could not be made solid, keeping it a shell: {e}");
            piece
        }
    }
}

/// Commits each joined group as a part in place of the parts it consumes, as
/// one undo step. Returns the new parts.
///
/// A group takes the name and appearance of its first source; a later group
/// from the same source is numbered in its series.
pub fn commit_join(doc: &mut Document, join: &Join) -> Result<Vec<PartId>> {
    let consumed = join
        .consumed()
        .into_iter()
        .map(|node| doc.part_for_node(node).context("Join source is not a known CAD part"))
        .collect::<Result<Vec<_>>>()?;

    let mut doc = doc.undo_scope("Join");
    let mut named = Vec::new();
    let mut joined = Vec::with_capacity(join.groups.len());
    for group in &join.groups {
        let first = group.sources.first().and_then(|&node| doc.part_for_node(node));
        let source = first.context("A joined group has no source part")?;
        let (name, frame, options) = {
            let part = doc.get_part(source).context("Join source part not found")?;
            let name = if named.contains(&source) { doc.duplicate_name(source) } else { part.name.clone() };
            (name, part.frame(), part.options().clone())
        };
        named.push(source);
        // A new part rather than a reshape: a sheet's node carries sketch
        // materials, which retessellating in place would keep.
        let shape = group.shape.clone();
        joined.push(doc.add_part(name, shape, frame, &options).context("Failed to tessellate the joined part")?);
    }
    for part in consumed {
        doc.remove_part(part);
    }
    Ok(joined)
}

/// Whether `part` has faces to unjoin: any, unless it is a lone face already.
pub fn can_unjoin(part: &CadPart) -> bool {
    part.shape().shape_type() != ShapeType::Face && !part.elements().faces().is_empty()
}

/// Breaks each part among `nodes` that [`can_unjoin`] into its faces, a part
/// for each that shares no edge with the others, as one undo step. Returns
/// the new parts.
pub fn unjoin_parts(doc: &mut Document, nodes: &[NodeId]) -> Result<Vec<PartId>> {
    let sources: Vec<PartId> = nodes
        .iter()
        .filter_map(|&node| doc.part_for_node(node))
        .filter(|&part| doc.get_part(part).is_some_and(can_unjoin))
        .collect();
    ensure!(!sources.is_empty(), "Nothing to unjoin: select a solid or shell");

    let mut doc = doc.undo_scope("Unjoin");
    let mut faces = Vec::new();
    for source in sources {
        let (pieces, frame, options) = {
            let part = doc.get_part(source).context("Unjoin source part not found")?;
            // Each face copied on its own, so no two share an edge.
            let pieces: Vec<Shape> =
                part.elements().faces().iter().map(|face| Shape::from(face).deep_copy()).collect();
            (pieces, part.frame(), part.options().clone())
        };
        for piece in pieces {
            let face = doc.add_numbered_part("Face", piece, frame, &options).context("Failed to tessellate a face")?;
            faces.push(face);
        }
        doc.remove_part(source);
    }
    Ok(faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    use duck_engine_scene::cad::CadTessellationOptions;
    use duck_engine_scene::common::Transform;
    use glam::DVec3;

    use crate::document::PartKind;
    use crate::testing::{doc_with_box, doc_with_box_faces, doc_with_shape, part_shape, xy_region};

    fn kinds(doc: &Document) -> Vec<PartKind> {
        doc.parts().map(CadPart::kind).collect()
    }

    fn names(doc: &Document) -> Vec<&str> {
        doc.parts().map(|part| part.name.as_str()).collect()
    }

    /// Builds the join of `nodes` at the default tolerance and commits it.
    fn join(doc: &mut Document, nodes: &[NodeId]) -> Result<Vec<PartId>> {
        let join = build_join(doc, nodes, default_tolerance(doc, nodes))?;
        commit_join(doc, &join)
    }

    /// The face part among `nodes` centred at `center`.
    fn face_at(doc: &Document, nodes: &[NodeId], center: DVec3) -> NodeId {
        *nodes
            .iter()
            .find(|&&node| part_shape(doc, node).faces().next().unwrap().center_of_mass().distance(center) < 1e-6)
            .expect("a face is centred there")
    }

    /// Adds `shape` as a part named `name`, returning its node.
    fn add(doc: &mut Document, name: &str, shape: Shape) -> NodeId {
        let part = doc.add_part(name, shape, Transform::IDENTITY, &CadTessellationOptions::default()).expect("shape tessellates");
        doc.node_for_part(part).expect("part has a node")
    }

    #[test]
    fn unjoining_a_box_makes_a_part_of_each_face() {
        let (mut doc, node) = doc_with_box();
        let box_part = doc.part_for_node(node).unwrap();

        let faces = unjoin_parts(&mut doc, &[node]).expect("the box unjoins");
        assert_eq!(faces.len(), 6);
        assert_eq!(kinds(&doc), [PartKind::Face; 6]);
        assert_eq!(names(&doc), ["Face-001", "Face-002", "Face-003", "Face-004", "Face-005", "Face-006"]);
        // The box is centred on the origin, so each face still faces outward
        // if its normal points the way its centre lies.
        for part in doc.parts() {
            let face = part.shape().faces().next().unwrap();
            assert!(face.normal_at_center().unwrap().dot(face.center_of_mass()) > 0.0, "{} faces inward", part.name);
        }
        assert_eq!(doc.undo_label(), Some("Unjoin"));

        doc.undo().expect("undo succeeds");
        assert_eq!(kinds(&doc), [PartKind::Solid]);
        assert_eq!(doc.node_for_part(box_part), Some(node));
    }

    #[test]
    fn unjoined_faces_share_no_edges() {
        let (mut doc, node) = doc_with_box();
        unjoin_parts(&mut doc, &[node]).expect("the box unjoins");

        let shapes: Vec<_> = doc.parts().map(CadPart::shape).collect();
        for (i, a) in shapes.iter().enumerate() {
            for b in &shapes[i + 1..] {
                let shared = a.edges().any(|ea| b.edges().any(|eb| Shape::from(&ea).is_same(&Shape::from(&eb))));
                assert!(!shared, "two unjoined faces share an edge");
            }
        }
    }

    #[test]
    fn a_lone_face_has_nothing_to_unjoin() {
        let (mut doc, node) = doc_with_shape(xy_region(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]));

        let error = unjoin_parts(&mut doc, &[node]).expect_err("a face is already unjoined");
        assert!(error.to_string().contains("Nothing to unjoin"), "got {error}");
        assert_eq!(kinds(&doc), [PartKind::Face]);
    }

    /// A sphere is one face closed on itself: unjoined it is a sheet, and
    /// joined again a solid.
    #[test]
    fn a_sphere_unjoins_into_a_sheet_and_joins_back_solid() {
        let (mut doc, node) = doc_with_shape(Shape::sphere(1.0).build());

        let faces = unjoin_parts(&mut doc, &[node]).expect("the sphere unjoins");
        assert_eq!(kinds(&doc), [PartKind::Face]);

        let sheet = doc.node_for_part(faces[0]).unwrap();
        join(&mut doc, &[sheet]).expect("the sheet closes");
        assert_eq!(kinds(&doc), [PartKind::Solid]);
        let volume = doc.parts().next().unwrap().shape().volume();
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((volume - expected).abs() < 1e-6, "got {volume}");
    }

    #[test]
    fn the_faces_of_a_box_join_into_a_solid_as_one_undo_step() {
        let (mut doc, nodes) = doc_with_box_faces();

        join(&mut doc, &nodes).expect("the faces join");
        assert_eq!(kinds(&doc), [PartKind::Solid]);
        assert_eq!(names(&doc), ["Face-001"], "the solid is named after the first face picked");
        assert!((doc.parts().next().unwrap().shape().volume() - 8.0).abs() < 1e-9);
        assert_eq!(doc.undo_label(), Some("Join"));

        doc.undo().expect("undo succeeds");
        assert_eq!(kinds(&doc), [PartKind::Face; 6]);
        for &node in &nodes {
            assert!(doc.part_for_node(node).is_some(), "every face comes back on its own node");
        }
    }

    #[test]
    fn five_faces_join_into_a_shell_with_an_open_rim() {
        let (doc, nodes) = doc_with_box_faces();
        let top = face_at(&doc, &nodes, DVec3::new(1.0, 1.0, 2.0));
        let rest: Vec<_> = nodes.iter().copied().filter(|&node| node != top).collect();

        let join = build_join(&doc, &rest, default_tolerance(&doc, &rest)).expect("the faces join");
        assert_eq!(join.groups.len(), 1);
        assert_eq!(join.groups[0].shape.shape_type(), ShapeType::Shell);
        assert_eq!(join.groups[0].sources, rest);
        assert_eq!(join.open_edges.len(), 4);
    }

    /// Two sides meeting at an edge join; a square off on its own touches
    /// nothing and stays the part it was.
    #[test]
    fn a_sheet_apart_is_left_alone() {
        let (mut doc, nodes) = doc_with_box_faces();
        let side = face_at(&doc, &nodes, DVec3::new(0.0, 1.0, 1.0));
        let front = face_at(&doc, &nodes, DVec3::new(1.0, 0.0, 1.0));
        let far = add(&mut doc, "far", xy_region(&[(10.0, 0.0), (11.0, 0.0), (11.0, 1.0), (10.0, 1.0)]));
        let far_part = doc.part_for_node(far).unwrap();

        let join = build_join(&doc, &[side, far, front], 1e-6).expect("the sides join");
        assert_eq!(join.groups.len(), 1);
        assert_eq!(join.groups[0].sources, [side, front]);
        commit_join(&mut doc, &join).expect("the join commits");

        assert_eq!(doc.part_for_node(far), Some(far_part), "the far square is untouched");
        assert!(doc.part_for_node(side).is_none() && doc.part_for_node(front).is_none());
        assert_eq!(doc.parts().count(), 4 + 1 + 1);
    }

    #[test]
    fn a_closed_shell_alone_becomes_a_solid() {
        let cube = Shape::cube(2.0);
        let (mut doc, node) = doc_with_shape(Shell::from_faces(cube.faces()).into());
        assert_eq!(kinds(&doc), [PartKind::Shell]);

        join(&mut doc, &[node]).expect("the shell closes");
        assert_eq!(kinds(&doc), [PartKind::Solid]);
        assert_eq!(names(&doc), ["part"]);
    }

    #[test]
    fn solids_are_left_out_and_nothing_joining_is_refused() {
        let (mut doc, solid) = doc_with_box();
        let square = add(&mut doc, "square", xy_region(&[(5.0, 0.0), (6.0, 0.0), (6.0, 1.0), (5.0, 1.0)]));

        let Err(error) = build_join(&doc, &[solid, square], 1e-6) else {
            panic!("a lone square joins nothing");
        };
        assert!(error.to_string().contains("Nothing to join"), "got {error}");
    }

    /// Squares 1e-4 apart meet only once the tolerance spans the gap.
    #[test]
    fn a_gap_within_the_tolerance_is_sewn_shut() {
        let mut doc = Document::new(duck_engine_scene::Scene::default());
        let gap = 1e-4;
        let left = add(&mut doc, "left", xy_region(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]));
        let right =
            add(&mut doc, "right", xy_region(&[(1.0 + gap, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0 + gap, 1.0)]));

        assert!(build_join(&doc, &[left, right], 1e-6).is_err(), "the gap is wider than the tolerance");
        let join = build_join(&doc, &[left, right], 1e-3).expect("the gap is within the tolerance");
        assert_eq!(join.groups.len(), 1);
        assert_eq!(join.groups[0].shape.shape_type(), ShapeType::Shell);
    }

    #[test]
    fn building_a_join_leaves_the_sources_untouched() {
        let (doc, nodes) = doc_with_box_faces();

        let join = build_join(&doc, &nodes, default_tolerance(&doc, &nodes)).expect("the faces join");
        let joined = &join.groups[0].shape;
        for &node in &nodes {
            for face in part_shape(&doc, node).faces() {
                assert!(!joined.faces().any(|f| f.is_same(&face)), "the join shares a face with a source part");
            }
        }
    }
}
