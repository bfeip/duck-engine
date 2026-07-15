use anyhow::{Context, Result};
use duck_engine_scene::NodeId;
use duck_engine_scene::cad::{CadTessellationOptions, tessellate_into};
use opencascade::primitives::{Shape, ShapeType};

use crate::document::{Document, PartId};

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum BooleanKind {
    #[default]
    Subtract,
    Union,
    Intersect,
}

struct BooleanResult {
    shape: Shape,
    target_part_id: PartId,
    tool_part_ids: Vec<PartId>,
}

/// Resolve nodes to shapes and compute the boolean result, without touching the scene.
fn compute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &Document,
) -> Result<BooleanResult> {
    let target_part_id = doc.part_for_node(target)
        .context("Target node is not a known CAD part")?;
    let tool_part_ids: Vec<_> = tools.iter()
        .map(|&node| doc.part_for_node(node).context("Tool node is not a known CAD part"))
        .collect::<Result<_>>()?;

    let target_part = doc.get_part(target_part_id)
        .context("Target part not found")?;
    // Deep copies: OCCT booleans run destructively by default (tolerance bumps,
    // added PCurves on the *inputs*), and Shape::clone shares B-Rep data — so
    // operating on clones would corrupt the document's parts, which outlive a
    // cancelled or failed operation.
    let tool_shapes: Vec<_> = tool_part_ids.iter()
        .map(|&id| doc.get_part(id).map(|p| p.shape.deep_copy()).context("Tool part not found"))
        .collect::<Result<_>>()?;

    let mut shape = target_part.shape.deep_copy();
    for tool in &tool_shapes {
        shape = match kind {
            BooleanKind::Subtract  => shape.subtract(tool)?.shape,
            BooleanKind::Union     => shape.union(tool)?.shape,
            BooleanKind::Intersect => shape.intersect(tool)?.shape,
        };
    }

    Ok(BooleanResult { shape: normalize_boolean_result(shape), target_part_id, tool_part_ids })
}

/// A boolean result is wrapped in a TopoDS_COMPOUND even when it holds a
/// single solid; unwrap that case so the part is a plain solid. Multi-solid
/// or mixed compounds are legitimately multi-body and stay as-is.
fn normalize_boolean_result(shape: Shape) -> Shape {
    if shape.shape_type() != ShapeType::Compound {
        return shape;
    }
    let mut children = shape.sub_shapes();
    match (children.next(), children.next()) {
        (Some(only), None) if only.shape_type() == ShapeType::Solid => only,
        _ => shape,
    }
}

pub fn execute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &mut Document,
    options: &CadTessellationOptions,
) -> Result<()> {
    let computed = compute_boolean(kind, target, tools, doc)?;

    // Tessellates atomically — if this fails, nothing is changed.
    doc.add_part("Boolean result".to_owned(), computed.shape, options)
        .context("Failed to tessellate boolean result")?;

    // Tessellation succeeded — remove inputs.
    for &part_id in computed.tool_part_ids.iter() {
        doc.remove_part(part_id);
    }
    doc.remove_part(computed.target_part_id);

    Ok(())
}

/// Non-destructive preview: compute the boolean and add a temporary scene node
/// without modifying the source parts or document. The caller owns the returned
/// NodeId and must remove it when done.
pub fn preview_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &Document,
    options: &CadTessellationOptions,
) -> Result<NodeId> {
    let computed = compute_boolean(kind, target, tools, doc)?;
    let mut scene = doc.scene().lock().unwrap();
    tessellate_into(&computed.shape, &mut *scene, options, None, Some("Boolean preview"))
        .context("Failed to tessellate boolean preview")
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use duck_engine_scene::Scene;
    use glam::dvec3;

    use super::*;
    use crate::document::PartKind;

    fn doc_with_box_and_sphere() -> (Document, NodeId, NodeId) {
        let scene = Arc::new(Mutex::new(Scene::new()));
        let mut doc = Document::new(scene);
        let options = CadTessellationOptions::default();
        let box_part = doc
            .add_part("box", Shape::cube(2.0), &options)
            .expect("box tessellates");
        let sphere_part = doc
            .add_part("sphere", Shape::sphere(1.0).at(dvec3(2.0, 2.0, 2.0)).build(), &options)
            .expect("sphere tessellates");
        let box_node = doc.node_for_part(box_part).unwrap();
        let sphere_node = doc.node_for_part(sphere_part).unwrap();
        (doc, box_node, sphere_node)
    }

    /// The boolean must run on deep copies: OCCT BOPs are destructive toward
    /// their inputs, and the result reuses unsplit input faces — with shallow
    /// clones a preview would corrupt the document's parts, surviving cancel.
    #[test]
    fn boolean_shares_no_faces_with_document_parts() {
        let (doc, box_node, sphere_node) = doc_with_box_and_sphere();

        let result = compute_boolean(BooleanKind::Subtract, box_node, &[sphere_node], &doc)
            .expect("subtract succeeds");

        for node in [box_node, sphere_node] {
            let part_id = doc.part_for_node(node).unwrap();
            let source = &doc.get_part(part_id).unwrap().shape;
            for face in source.faces() {
                assert!(
                    !result.shape.faces().any(|f| f.is_same(&face)),
                    "boolean result shares a face with a source part"
                );
            }
        }
    }

    #[test]
    fn boolean_result_is_solid_part() {
        let (mut doc, box_node, sphere_node) = doc_with_box_and_sphere();

        execute_boolean(
            BooleanKind::Subtract,
            box_node,
            &[sphere_node],
            &mut doc,
            &CadTessellationOptions::default(),
        )
        .expect("subtract succeeds");

        let part = doc.parts().next().expect("boolean leaves one part");
        assert_eq!(doc.parts().count(), 1, "inputs are consumed");
        assert_eq!(part.kind(), PartKind::Solid, "single-solid compound must be unwrapped");
    }
}
