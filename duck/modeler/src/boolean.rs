use anyhow::{Context, Result};
use duck_engine_scene::resource::NodeId;
use duck_engine_scene::cad::{CadTessellationOptions, tessellate_into};
use opencascade::primitives::{BooleanPair, Shape};

use crate::document::{unify_same_domain, unwrap_single_solid, Document, PartId};

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
    /// See [`BooleanPreview::removed`]; gathered only on request.
    removed: Vec<Shape>,
}

/// Resolve nodes to shapes and compute the boolean result, without touching the scene.
/// With `keep_removed`, also gather the input material the result drops.
fn compute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &Document,
    keep_removed: bool,
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
    let mut removed = Vec::new();
    for tool in &tool_shapes {
        let result = match kind {
            BooleanKind::Subtract => {
                // A subtract keeps none of its tool.
                if keep_removed {
                    removed.push(tool.clone());
                }
                shape.subtract_with_fuzz(tool, fuzz)?
            }
            BooleanKind::Union => shape.union_with_fuzz(tool, fuzz)?,
            BooleanKind::Intersect => {
                // One intersection serves the result and what this step drops
                // from either side.
                let pair = BooleanPair::new(&shape, tool, fuzz)?;
                if keep_removed {
                    for piece in [pair.subtract(), pair.subtract_reversed()] {
                        match piece {
                            // Unified like the result (see below).
                            Ok(piece) if piece.faces().next().is_some() => {
                                removed.push(unify_same_domain(unwrap_single_solid(piece.shape)));
                            }
                            // That side lost nothing.
                            Ok(_) => {}
                            Err(e) => log::warn!("Could not compute removed material: {e}"),
                        }
                    }
                }
                pair.intersect()?
            }
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

    // Unify last: the BOP splits periodic faces at their seam, leaving same-domain
    // halves that would otherwise be drawn and picked as separate geometry.
    let shape = unify_same_domain(unwrap_single_solid(shape));
    Ok(BooleanResult { shape, target_part_id, tool_part_ids, removed })
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

pub fn execute_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &mut Document,
    options: &CadTessellationOptions,
) -> Result<()> {
    let computed = compute_boolean(kind, target, tools, doc, false)?;

    // The result supersedes the target, so it inherits the target's name.
    let name = doc
        .get_part(computed.target_part_id)
        .map_or_else(|| "Boolean result".to_owned(), |part| part.name.clone());

    // One undo step covers the added result and the removed inputs.
    let mut doc = doc.undo_scope("Boolean");

    // Tessellates atomically — if this fails, nothing is changed.
    doc.add_part(name, computed.shape, options)
        .context("Failed to tessellate boolean result")?;

    // Tessellation succeeded — remove inputs.
    for &part_id in computed.tool_part_ids.iter() {
        doc.remove_part(part_id);
    }
    doc.remove_part(computed.target_part_id);

    Ok(())
}

/// A non-destructive boolean preview.
pub struct BooleanPreview {
    /// Temporary scene node showing the result. The caller owns it and must
    /// remove it when done.
    pub node: NodeId,
    /// The input material the result drops, as separate shapes: a subtract's
    /// tools, the part of each intersect input outside the result, and nothing
    /// for a union.
    pub removed: Vec<Shape>,
}

/// Non-destructive preview: compute the boolean and add a temporary scene node
/// without modifying the source parts or document.
pub fn preview_boolean(
    kind: BooleanKind,
    target: NodeId,
    tools: &[NodeId],
    doc: &Document,
    options: &CadTessellationOptions,
) -> Result<BooleanPreview> {
    let computed = compute_boolean(kind, target, tools, doc, true)?;
    let node = tessellate_into(&computed.shape, doc.scene(), options, None, Some("Boolean preview"))
        .context("Failed to tessellate boolean preview")?
        .id();
    Ok(BooleanPreview { node, removed: computed.removed })
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

        let result = compute_boolean(BooleanKind::Subtract, box_node, &[sphere_node], &doc, false)
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

    /// A BOP splits a periodic face at its seam, leaving two faces on the same
    /// sphere. Left alone they are drawn and picked as separate geometry, so the
    /// result must come back unified.
    #[test]
    fn boolean_result_has_no_same_domain_faces() {
        let scene = Scene::default();
        let mut doc = Document::new(scene);
        let options = CadTessellationOptions::default();
        let box_part = doc.add_part("box", Shape::cube(2.0), &options).expect("box tessellates");
        let sphere = Shape::sphere(2.0).at(dvec3(0.5, 1.0, 0.5)).build();
        let sphere_part = doc.add_part("sphere", sphere, &options).expect("sphere tessellates");
        let (box_node, sphere_node) =
            (doc.node_for_part(box_part).unwrap(), doc.node_for_part(sphere_part).unwrap());

        let result = compute_boolean(BooleanKind::Subtract, box_node, &[sphere_node], &doc, false)
            .expect("subtract succeeds");

        // Cleaning again must find nothing left to merge.
        let faces = result.shape.faces().count();
        assert_eq!(faces, result.shape.clean().expect("clean").faces().count());
        assert_eq!(faces, 5);
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
            false,
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
        assert_eq!(doc.parts().next().unwrap().name, "box");
    }

    #[test]
    fn result_inherits_the_target_name() {
        let (mut doc, box_node, sphere_node) = doc_with_box_and_sphere();

        execute_boolean(
            BooleanKind::Subtract,
            box_node,
            &[sphere_node],
            &mut doc,
            &CadTessellationOptions::default(),
        )
        .expect("subtract succeeds");

        let names: Vec<_> = doc.parts().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["box"], "the result supersedes the target and keeps its name");
    }

    fn assert_volume(shape: &Shape, expected: f64) {
        let volume = shape.volume();
        assert!(
            (volume - expected).abs() < 1e-4 * expected,
            "expected volume {expected}, got {volume}"
        );
    }

    /// The box and the sphere share an eighth of the sphere; an intersect keeps
    /// that and reports the rest of each input, target first.
    #[test]
    fn intersect_preview_removes_each_input_outside_the_result() {
        use std::f64::consts::PI;

        let (doc, box_node, sphere_node) = doc_with_box_and_sphere();
        let preview = preview_boolean(
            BooleanKind::Intersect,
            box_node,
            &[sphere_node],
            &doc,
            &CadTessellationOptions::default(),
        )
        .expect("intersect succeeds");

        let shared = PI / 6.0;
        assert_eq!(preview.removed.len(), 2);
        assert_volume(&preview.removed[0], 8.0 - shared);
        assert_volume(&preview.removed[1], 4.0 / 3.0 * PI - shared);
    }

    /// A subtract keeps none of its tools, so each is removed whole.
    #[test]
    fn subtract_preview_removes_the_tools_whole() {
        use std::f64::consts::PI;

        let (doc, box_node, sphere_node) = doc_with_box_and_sphere();
        let preview = preview_boolean(
            BooleanKind::Subtract,
            box_node,
            &[sphere_node],
            &doc,
            &CadTessellationOptions::default(),
        )
        .expect("subtract succeeds");

        assert_eq!(preview.removed.len(), 1);
        assert_volume(&preview.removed[0], 4.0 / 3.0 * PI);
    }

    #[test]
    fn union_preview_removes_nothing() {
        let (doc, box_node, sphere_node) = doc_with_box_and_sphere();
        let preview = preview_boolean(
            BooleanKind::Union,
            box_node,
            &[sphere_node],
            &doc,
            &CadTessellationOptions::default(),
        )
        .expect("union succeeds");

        assert!(preview.removed.is_empty());
    }

    #[test]
    fn removed_material_is_only_gathered_on_request() {
        let (doc, box_node, sphere_node) = doc_with_box_and_sphere();
        for kind in [BooleanKind::Subtract, BooleanKind::Union, BooleanKind::Intersect] {
            let result = compute_boolean(kind, box_node, &[sphere_node], &doc, false)
                .expect("boolean succeeds");
            assert!(result.removed.is_empty());
        }
    }
}
