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

    let target_volume = target_part.shape.volume();
    let mut shape = target_part.shape.deep_copy();
    let fuzz = interactive_fuzz(std::iter::once(&shape).chain(&tool_shapes));
    for tool in &tool_shapes {
        let result = match kind {
            BooleanKind::Subtract  => shape.subtract_with_fuzz(tool, fuzz)?,
            BooleanKind::Union     => shape.union_with_fuzz(tool, fuzz)?,
            BooleanKind::Intersect => shape.intersect_with_fuzz(tool, fuzz)?,
        };
        if let Some(warnings) = &result.warnings {
            log::warn!("boolean completed with warnings:\n{warnings}");
        }
        shape = result.shape;
    }

    // A subtract that removes nothing is either a non-intersecting tool or an
    // OCCT classification failure on near-coincident geometry (the cut reports
    // success but only imprints the section edge). Either way, fail instead of
    // consuming the inputs for a no-op result.
    if kind == BooleanKind::Subtract && target_volume - shape.volume() <= 1e-9 * target_volume {
        anyhow::bail!(
            "Subtract removed no material — the tools may not intersect the target, \
             or the inputs sit in a degenerate near-coincident position (try nudging a tool)"
        );
    }

    Ok(BooleanResult { shape: normalize_boolean_result(shape), target_part_id, tool_part_ids })
}

/// Additional boolean intersection tolerance for interactively placed parts.
///
/// Placement flows through f32 (snaps, tessellated pick positions), so inputs
/// meant to coincide can sit a few f32 ulps of the coordinate magnitude apart
/// — far beyond OCCT's 1e-7 default, in the near-coincidence band where the
/// BOP misclassifies splits and a subtract silently removes nothing. Four
/// ulps of the inputs' extent covers that placement error with margin.
fn interactive_fuzz<'a>(shapes: impl Iterator<Item = &'a Shape>) -> f64 {
    let extent = shapes
        .map(|shape| {
            let aabb = opencascade::bounding_box::aabb(shape);
            aabb.min().abs().max_element().max(aabb.max().abs().max_element())
        })
        .fold(0.0f64, f64::max);
    4.0 * f32::EPSILON as f64 * extent
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

    // One undo step covers the added result and the removed inputs.
    let mut doc = doc.undo_scope("Boolean");

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
    tessellate_into(&computed.shape, doc.scene(), options, None, Some("Boolean preview"))
        .map(|node| node.id())
        .context("Failed to tessellate boolean preview")
}

#[cfg(test)]
mod tests {
    use duck_engine_scene::Scene;
    use glam::dvec3;

    use super::*;
    use crate::document::PartKind;

    fn doc_with_box_and_sphere() -> (Document, NodeId, NodeId) {
        let scene = Scene::default();
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

    /// The interactive-placement failure mode: a default-parametrization
    /// sphere whose seam sits a few microns off the box's top face plane (f32
    /// snap error at mm scale) makes a plain OCCT cut "succeed" while removing
    /// nothing. The extent-scaled fuzzy value must rescue the cut through the
    /// full modeler path.
    #[test]
    fn subtract_rescues_near_coincident_sphere() {
        use opencascade::primitives::{Face, Wire};

        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let options = CadTessellationOptions::default();

        let wire = Wire::from_ordered_points([
            dvec3(0.0, 0.0, 0.0),
            dvec3(100.0, 0.0, 0.0),
            dvec3(100.0, 0.0, 100.0),
            dvec3(0.0, 0.0, 100.0),
        ])
        .expect("rectangle wire");
        let world_box: Shape =
            Face::from_wire(&wire).expect("rectangle face").extrude(dvec3(0.0, 50.0, 0.0)).into();
        let box_volume = world_box.volume();
        let box_part = doc.add_part("box", world_box, &options).expect("box tessellates");
        let sphere_part = doc
            .add_part(
                "sphere",
                Shape::sphere(20.0).at(dvec3(40.0, 50.0 + 4e-6, 60.0)).build(),
                &options,
            )
            .expect("sphere tessellates");

        let result = compute_boolean(
            BooleanKind::Subtract,
            doc.node_for_part(box_part).unwrap(),
            &[doc.node_for_part(sphere_part).unwrap()],
            &doc,
        )
        .expect("near-coincident subtract succeeds");

        let removed = box_volume - result.shape.volume();
        let expected = 2.0 / 3.0 * std::f64::consts::PI * 20.0f64.powi(3);
        assert!(
            (removed - expected).abs() < 5e-3 * expected,
            "expected a half-ball cavity ({expected:.1}), removed {removed:.1}"
        );
    }

    /// A subtract whose tools don't intersect the target must fail loudly
    /// instead of consuming the inputs for a no-op result.
    #[test]
    fn subtract_removing_nothing_errors() {
        let (mut doc, box_node, _) = doc_with_box_and_sphere();
        let options = CadTessellationOptions::default();
        let far_part = doc
            .add_part("far sphere", Shape::sphere(1.0).at(dvec3(10.0, 10.0, 10.0)).build(), &options)
            .expect("sphere tessellates");
        let far_node = doc.node_for_part(far_part).unwrap();

        let err = execute_boolean(
            BooleanKind::Subtract,
            box_node,
            &[far_node],
            &mut doc,
            &options,
        )
        .expect_err("no-op subtract must fail");
        assert!(err.to_string().contains("removed no material"), "unexpected error: {err}");
        assert_eq!(doc.parts().count(), 3, "a failed boolean must not consume inputs");
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

    #[test]
    fn boolean_is_one_undo_step() {
        let (mut doc, box_node, sphere_node) = doc_with_box_and_sphere();

        execute_boolean(
            BooleanKind::Subtract,
            box_node,
            &[sphere_node],
            &mut doc,
            &CadTessellationOptions::default(),
        )
        .expect("subtract succeeds");
        assert_eq!(doc.undo_label(), Some("Boolean"));

        doc.undo().expect("undo succeeds");
        let names: Vec<_> = doc.parts().map(|p| p.name.as_str()).collect();
        assert_eq!(names.len(), 2, "one undo restores both inputs");
        assert!(names.contains(&"box") && names.contains(&"sphere"));
        assert_eq!(doc.node_for_part(doc.part_for_node(box_node).unwrap()), Some(box_node));

        doc.redo().expect("redo succeeds");
        assert_eq!(doc.parts().count(), 1, "one redo replays the boolean");
        assert_eq!(doc.parts().next().unwrap().name, "Boolean result");
    }
}
