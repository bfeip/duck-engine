use std::collections::{HashMap, HashSet};

use duck_engine_common::{Matrix3, Matrix4, Point3, SquareMatrix};

use crate::scene::camera::PositionedCamera;
use crate::scene::light::Light;
use crate::scene::resource::{
    AlphaMode, DisplayBehavior, FaceMaterialId, Instance, InstanceId, LineMaterialId,
    MaterialProperties, MeshId, NodeId, NodePayload, PointMaterialId, PrimitiveType, RenderLayer,
    SubGeometryElement, SubGeometryKind, Visibility,
};
use crate::scene::SceneData;
use crate::scene::common;
use crate::highlight_query::HighlightQuery;

/// A draw call targeting a sub-range of a mesh's index buffer for a single instance.
///
/// Used to render individual faces or edges (e.g. for highlight outlines), but
/// applicable wherever only part of a mesh needs to be drawn.
pub struct SubGeomBatch {
    pub mesh_id: MeshId,
    pub instance_transform: InstanceTransform,
    pub primitive_type: PrimitiveType,
    /// First raw index within the primitive's index buffer.
    /// For `TriangleList`: `range.start * 3`. For `LineList`: `range.start * 2`.
    pub first_index: u32,
    /// Number of raw indices to draw.
    /// For `TriangleList`: `range.count * 3`. For `LineList`: `range.count * 2`.
    pub index_count: u32,
}

/// A typed reference to the material a batch draws with.
///
/// The variant *is* the material kind, so it carries the matching typed id
/// (`FaceMaterialId`/`LineMaterialId`/`PointMaterialId`) — no erasure, and the
/// draw path looks it up in the correctly-typed cache. The kind always agrees
/// with the batch's `primitive_type` (faces↔triangles, etc.).
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub enum BatchMaterial {
    Face(FaceMaterialId),
    Line(LineMaterialId),
    Point(PointMaterialId),
}

/// Key used to group instances into draw batches: instances sharing a mesh and
/// material render together.
///
/// `primitive_type` is intentionally absent: [`BatchMaterial`] already encodes
/// the kind, and a given material id only ever pairs with one primitive type, so
/// the type is fully determined by `material` and would be redundant here.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchKey {
    pub mesh_id: MeshId,
    pub material: BatchMaterial,
}

/// Represents an instance with its computed world transform.
// TODO: Has grown considerably. Needs renaming.
#[derive(Clone)]
pub struct InstanceTransform {
    pub node_id: NodeId,
    pub instance_id: InstanceId,
    /// Camera-independent world transform from the scene graph. Reused for
    /// centroid sorting and highlight sub-geometry; never camera-adjusted.
    pub world_transform: Matrix4,
    pub normal_matrix: Matrix3,
    /// Effective (inherited) render-presentation behavior for this instance.
    pub display: DisplayBehavior,
    /// Transform actually uploaded to the GPU. Equal to `world_transform` for
    /// ordinary geometry; the camera-dependent screen-space adjustments
    /// (screen-sizing, billboarding) replace it at render time. Kept separate
    /// so the camera-independent `world_transform` stays usable for sorting.
    pub effective_transform: Matrix4,
    /// Normal matrix matching `effective_transform`.
    pub effective_normal_matrix: Matrix3,
}

impl InstanceTransform {
    /// Creates a new InstanceTransform with the given world transform.
    /// The normal matrix is computed from the world transform. The effective
    /// transform starts equal to the world transform (no screen-space
    /// adjustment) and display defaults to ordinary scene geometry.
    pub fn new(node_id: NodeId, instance_id: InstanceId, world_transform: Matrix4) -> Self {
        let normal_matrix = common::compute_normal_matrix(&world_transform);
        Self {
            node_id,
            instance_id,
            world_transform,
            normal_matrix,
            display: DisplayBehavior::default(),
            effective_transform: world_transform,
            effective_normal_matrix: normal_matrix,
        }
    }

    /// Sets the effective render-presentation behavior for this instance.
    pub fn with_display(mut self, display: DisplayBehavior) -> Self {
        self.display = display;
        self
    }
}

/// Represents a batch of instances that share the same mesh, material, and primitive type.
///
/// Batching allows us to minimize draw calls and state changes by grouping
/// instances that can be rendered together.
pub struct DrawBatch {
    pub mesh_id: MeshId,
    /// Typed material this batch draws with. Used to look up the GPU bind group
    /// in the correctly-typed material cache.
    pub material: BatchMaterial,
    /// Shader/pipeline properties resolved from the typed material at batch
    /// collection time, so the draw loop needs no further scene material lookup.
    pub material_props: MaterialProperties,
    /// Geometry primitive selecting the mesh's index buffer and pipeline topology.
    /// Always agrees with `material`'s kind.
    pub primitive_type: PrimitiveType,
    /// Number of indices this batch draws, resolved from the mesh at collection
    /// time so the draw loops need no scene lookup.
    pub index_count: u32,
    pub instances: Vec<InstanceTransform>,
}

impl DrawBatch {
    pub fn new(
        mesh_id: MeshId,
        material: BatchMaterial,
        primitive_type: PrimitiveType,
        index_count: u32,
    ) -> Self {
        Self {
            mesh_id,
            material,
            material_props: MaterialProperties::UNLIT_OPAQUE,
            primitive_type,
            index_count,
            instances: Vec::new(),
        }
    }

    /// Sets the resolved material properties (chainable).
    pub fn with_material_props(mut self, props: MaterialProperties) -> Self {
        self.material_props = props;
        self
    }

    pub fn key(&self) -> BatchKey {
        BatchKey {
            mesh_id: self.mesh_id,
            material: self.material,
        }
    }

    pub fn add_instance(&mut self, instance_transform: InstanceTransform) {
        self.instances.push(instance_transform);
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }
}

/// A light resolved to world space, combining photometric data from `NodePayload::Light`
/// with position and direction derived from the node's world transform.
pub(crate) struct ResolvedLight {
    pub light: Light,
    /// World-space position (relevant for Point and Spot lights).
    pub position: [f32; 3],
    /// World-space direction pointing away from lit surfaces (relevant for Directional and Spot lights).
    pub direction: [f32; 3],
}

/// All scene data collected in a single tree traversal.
pub(crate) struct SceneFrameData {
    pub instance_transforms: Vec<InstanceTransform>,
    pub lights: Vec<ResolvedLight>,
}

fn collect_scene_data_recursive(
    scene: &SceneData,
    node_id: NodeId,
    parent_transform: Matrix4,
    parent_display: DisplayBehavior,
    parent_changed: bool,
    data: &mut SceneFrameData,
) {
    let Some(node) = scene.get_node(node_id) else { return };

    let needs_recompute = parent_changed || node.transform_dirty();
    let world_transform = if needs_recompute {
        let wt = parent_transform * node.compute_local_transform();
        node.set_cached_world_transform(wt);
        wt
    } else {
        node.cached_world_transform().unwrap()
    };

    if node.visibility() == Visibility::Invisible {
        return;
    }

    let display = DisplayBehavior::inherit(parent_display, node.display());

    match node.payload() {
        NodePayload::Instance(instance) => {
            data.instance_transforms.push(
                InstanceTransform::new(node.id, instance.id(), world_transform).with_display(display),
            );
        }
        NodePayload::Light(light) => {
            let (position, direction) = Light::world_position_and_direction(&world_transform);
            data.lights.push(ResolvedLight { light: light.clone(), position: position.into(), direction: direction.into() });
        }
        _ => {}
    }

    for child_id in node.children() {
        collect_scene_data_recursive(scene, child_id, world_transform, display, needs_recompute, data);
    }
}

/// Walks the entire scene tree and collects all instances and lights in one pass.
pub(crate) fn collect_scene_frame_data(scene: &SceneData) -> SceneFrameData {
    let mut data = SceneFrameData {
        instance_transforms: Vec::new(),
        lights: Vec::new(),
    };
    for root_id in scene.root_nodes() {
        collect_scene_data_recursive(
            scene,
            root_id,
            Matrix4::identity(),
            DisplayBehavior::default(),
            false,
            &mut data,
        );
    }
    data
}

/// Collects all instances grouped into batches by mesh, material, and primitive type.
///
/// This walks the scene tree, computes world transforms, and groups
/// instances that share the same mesh, material, and primitive type into batches.
/// Each mesh can have multiple primitive types (triangles, lines, points), so
/// a single instance may generate multiple batches.
/// Batches are sorted to minimize state changes during rendering:
/// 1. By material ID (to minimize bind group changes)
/// 2. By primitive type (to minimize pipeline changes)
/// 3. By mesh ID (for GPU cache locality)
pub(crate) fn collect_draw_batches(scene: &SceneData) -> Vec<DrawBatch> {
    let instance_transforms = collect_scene_frame_data(scene).instance_transforms;
    let mut batch_map: HashMap<BatchKey, DrawBatch> = HashMap::new();

    for inst_transform in instance_transforms {
        let Some(instance) = scene.get_instance(inst_transform.instance_id) else {
            continue;
        };
        let Some(mesh) = scene.get_mesh(instance.mesh()) else {
            continue;
        };

        // Create a separate batch for each primitive type the mesh supports and
        // for which the instance has a material assigned.
        for primitive_type in [
            PrimitiveType::TriangleList,
            PrimitiveType::LineList,
            PrimitiveType::PointList,
        ] {
            if !mesh.has_primitive_type(primitive_type) {
                continue;
            }

            let Some((material, material_props)) =
                resolve_batch_material(scene, instance, primitive_type)
            else {
                continue;
            };

            let key = BatchKey { mesh_id: instance.mesh(), material };
            batch_map
                .entry(key)
                .or_insert_with(|| {
                    let index_count = mesh.index_count(primitive_type);
                    DrawBatch::new(instance.mesh(), material, primitive_type, index_count)
                        .with_material_props(material_props)
                })
                .add_instance(inst_transform.clone());
        }
    }

    // Convert to Vec and sort for optimal rendering
    let mut batches: Vec<DrawBatch> = batch_map.into_values().collect();
    // Sort by primitive type first so all triangles are drawn before lines/points.
    // This ensures the depth buffer is fully populated with triangle depths before
    // coplanar lines test against it.
    batches.sort_by_key(|b| (b.primitive_type as u8, b.material, b.mesh_id));
    batches
}

/// Reorders batches so opaque/mask batches come first (preserving existing sort),
/// followed by transparent (Blend) batches sorted back-to-front by distance from camera.
pub(crate) fn sort_batches_for_transparency(
    batches: &mut Vec<DrawBatch>,
    camera_position: Point3,
) {
    // Stable partition: opaque/mask first, then blend
    batches.sort_by(|a, b| {
        let a_transparent = is_transparent_batch(a);
        let b_transparent = is_transparent_batch(b);
        match (a_transparent, b_transparent) {
            (false, true) => std::cmp::Ordering::Less,
            (true, false) => std::cmp::Ordering::Greater,
            (true, true) => {
                // Both transparent: sort back-to-front (farthest first)
                let dist_a = batch_centroid_distance_sq(a, camera_position);
                let dist_b = batch_centroid_distance_sq(b, camera_position);
                dist_b
                    .partial_cmp(&dist_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            (false, false) => std::cmp::Ordering::Equal,
        }
    });
}

/// Resolves the typed material and shader properties for one primitive kind of
/// an instance, or `None` when that kind has no assigned material (so it is not
/// drawn). Face materials carry full PBR properties; line/point materials are
/// unlit but may bind a base-color texture.
fn resolve_batch_material(
    scene: &SceneData,
    instance: &Instance,
    primitive_type: PrimitiveType,
) -> Option<(BatchMaterial, MaterialProperties)> {
    match primitive_type {
        PrimitiveType::TriangleList => {
            let id = instance.face_material()?;
            let material = scene.get_face_material(id)?;
            Some((BatchMaterial::Face(id), material.properties()))
        }
        PrimitiveType::LineList => {
            let id = instance.line_material()?;
            let material = scene.get_line_material(id)?;
            Some((BatchMaterial::Line(id), material.properties()))
        }
        PrimitiveType::PointList => {
            let id = instance.point_material()?;
            let material = scene.get_point_material(id)?;
            Some((BatchMaterial::Point(id), material.properties()))
        }
    }
}

fn is_transparent_batch(batch: &DrawBatch) -> bool {
    batch.material_props.alpha_mode == AlphaMode::Blend
}

fn batch_centroid_distance_sq(batch: &DrawBatch, camera_position: Point3) -> f32 {
    if batch.instances.is_empty() {
        return 0.0;
    }
    let count = batch.instances.len() as f32;
    let (sx, sy, sz) = batch.instances.iter().fold((0.0f32, 0.0f32, 0.0f32), |(x, y, z), i| {
        (
            x + i.world_transform[3][0],
            y + i.world_transform[3][1],
            z + i.world_transform[3][2],
        )
    });
    let cx = sx / count - camera_position.x;
    let cy = sy / count - camera_position.y;
    let cz = sz / count - camera_position.z;
    cx * cx + cy * cy + cz * cz
}

/// Partitions batches by a predicate on instances.
///
/// Takes a list of batches and splits each batch's instances based on the predicate.
/// Returns two sets of batches: those where the predicate returned `true` and those
/// where it returned `false`.
///
/// Batches are preserved with their mesh/material/primitive type, but may be split
/// if instances within a batch have different predicate results.
///
/// Empty batches (after partitioning) are not included in the output.
pub(crate) fn partition_batches<F>(batches: &[DrawBatch], predicate: F) -> (Vec<DrawBatch>, Vec<DrawBatch>)
where
    F: Fn(&InstanceTransform) -> bool,
{
    // Use Vec + index map instead of HashMap to preserve the input ordering.
    // This is important because batches arrive sorted (e.g. opaque-first,
    // transparent back-to-front) and both partitions must maintain that order.
    let mut matched: Vec<DrawBatch> = Vec::new();
    let mut unmatched: Vec<DrawBatch> = Vec::new();
    let mut matched_index: HashMap<BatchKey, usize> = HashMap::new();
    let mut unmatched_index: HashMap<BatchKey, usize> = HashMap::new();

    for batch in batches {
        let key = batch.key();

        for instance in &batch.instances {
            if predicate(instance) {
                let idx = *matched_index.entry(key).or_insert_with(|| {
                    matched.push(
                        DrawBatch::new(
                            batch.mesh_id,
                            batch.material,
                            batch.primitive_type,
                            batch.index_count,
                        )
                        .with_material_props(batch.material_props.clone()),
                    );
                    matched.len() - 1
                });
                matched[idx].add_instance(instance.clone());
            } else {
                let idx = *unmatched_index.entry(key).or_insert_with(|| {
                    unmatched.push(
                        DrawBatch::new(
                            batch.mesh_id,
                            batch.material,
                            batch.primitive_type,
                            batch.index_count,
                        )
                        .with_material_props(batch.material_props.clone()),
                    );
                    unmatched.len() - 1
                });
                unmatched[idx].add_instance(instance.clone());
            }
        }
    }

    (matched, unmatched)
}

/// Overwrites the effective transform of every instance that requests a
/// camera-dependent presentation (`screen_size` and/or `screen_facing`).
/// Ordinary instances are left untouched — their effective transform already
/// equals the camera-independent world transform.
fn apply_screen_space_transforms(
    batches: &mut [DrawBatch],
    camera: &PositionedCamera,
    viewport: (u32, u32),
) {
    for batch in batches.iter_mut() {
        for inst in batch.instances.iter_mut() {
            if !inst.display.is_screen_space() {
                continue;
            }
            let m = inst.display.effective_transform(inst.world_transform, camera, viewport);
            inst.effective_transform = m;
            inst.effective_normal_matrix = common::compute_normal_matrix(&m);
        }
    }
}

/// Sub-geometry highlight draw calls for the current frame, split into the
/// primary selection and the secondary (everything else) tier so each can be
/// drawn in its own color.
#[derive(Default)]
struct SubGeomHighlights {
    primary: Vec<SubGeomBatch>,
    secondary: Vec<SubGeomBatch>,
}

/// Builds sub-geometry draw calls for all highlighted faces, edges, and pointsets
/// in the current frame.
///
/// For each node with highlighted sub-geometry, looks up the instance's mesh
/// topology and converts each highlighted face/edge/pointset index into a
/// `SubGeomBatch` targeting that element's index range. The single primary
/// element (whatever its kind) is sorted into the primary tier; everything else
/// is secondary.
fn collect_highlight_sub_geom_batches(
    batches: &[DrawBatch],
    scene: &SceneData,
    highlight: &dyn HighlightQuery,
) -> SubGeomHighlights {
    // Build a node_id → InstanceTransform index so we can resolve elements by node.
    let instance_by_node: HashMap<NodeId, &InstanceTransform> = batches
        .iter()
        .flat_map(|b| b.instances.iter())
        .map(|it| (it.node_id, it))
        .collect();

    let primary = highlight.primary_sub_geometry();

    let mut out = SubGeomHighlights::default();

    // Each kind maps to a primitive type whose ranges are counted in primitives
    // (triangles/segments/points), so scale by the primitive's index stride.
    for kind in SubGeometryKind::ALL {
        let primitive_type = kind.primitive_type();
        let stride = primitive_type.indices_per_primitive();

        for node_id in highlight.nodes_with_highlighted(kind) {
            let Some(it) = instance_by_node.get(&node_id) else { continue };
            let Some(instance) = scene.get_instance(it.instance_id) else { continue };
            let Some(mesh) = scene.get_mesh(instance.mesh()) else { continue };
            let Some(topology) = mesh.topology() else { continue };

            for index in highlight.highlighted_for_node(node_id, kind) {
                let Some(range) = topology.ranges(kind).get(index as usize) else { continue };
                let batch = SubGeomBatch {
                    mesh_id: instance.mesh(),
                    instance_transform: (*it).clone(),
                    primitive_type,
                    first_index: range.start * stride,
                    index_count: range.count * stride,
                };
                if primary == Some((node_id, SubGeometryElement { kind, index })) {
                    out.primary.push(batch);
                } else {
                    out.secondary.push(batch);
                }
            }
        }
    }

    out
}

/// Frame-scoped collection of draw batches, sorted and partitioned for rendering.
///
/// Constructed once per frame from the scene, camera, and optional selection.
/// Internally uses `collect_draw_batches`, `sort_batches_for_transparency`,
/// and `partition_batches`.
pub struct DrawData {
    /// All batches, sorted: opaque first (by material/primitive/mesh),
    /// then transparent back-to-front.
    batches: Vec<DrawBatch>,
    /// Subset of batches containing only the primary highlighted instance.
    /// Empty if no highlight is active.
    highlighted_batches: Vec<DrawBatch>,
    /// Subset of batches containing secondary (non-primary) highlighted instances.
    /// Empty if there is only one selection or no highlight is active.
    secondary_highlighted_batches: Vec<DrawBatch>,
    /// Batches on `RenderLayer::Overlay`, rendered in a separate overlay pass.
    overlay_batches: Vec<DrawBatch>,
    /// Sub-geometry draw calls for highlighted faces/edges belonging to the primary selection.
    /// Each entry targets a specific index range within a mesh's index buffer.
    highlight_sub_geom_batches: Vec<SubGeomBatch>,
    /// Sub-geometry draw calls for highlighted faces/edges belonging to secondary selections.
    secondary_highlight_sub_geom_batches: Vec<SubGeomBatch>,
    /// Resolved outline configuration for the primary selection. `Some` when a
    /// non-empty highlight is active; `None` otherwise.
    highlight_config: Option<crate::highlight_query::HighlightConfig>,
    /// Resolved outline configuration for secondary selections. `Some` when secondary
    /// highlights exist; `None` otherwise.
    secondary_highlight_config: Option<crate::highlight_query::HighlightConfig>,
}

impl DrawData {
    /// Build draw data for the current frame.
    ///
    /// - Walks the scene tree to collect instance transforms
    /// - Groups into batches by mesh/material/primitive
    /// - Sorts for transparency (opaque first, transparent back-to-front)
    /// - Partitions highlighted instances if a non-empty highlight is provided
    ///
    /// `camera` and `viewport` resolve transparency ordering and the
    /// camera-dependent screen-space adjustments (screen-sizing / billboarding),
    /// which are computed by [`DisplayBehavior::effective_transform`].
    pub(crate) fn new(
        scene: &SceneData,
        camera: &PositionedCamera,
        viewport: (u32, u32),
        highlight: Option<&dyn HighlightQuery>,
    ) -> Self {
        let mut batches = collect_draw_batches(scene);
        sort_batches_for_transparency(&mut batches, camera.eye);

        // Replace the effective transform of screen-space instances before any
        // partitioning. `partition_batches` clones instances, so doing this on
        // `batches` first propagates the adjusted transforms into every derived
        // partition (overlay, highlight) automatically.
        apply_screen_space_transforms(&mut batches, camera, viewport);

        // Partition out overlay batches (RenderLayer::Overlay) so they render in
        // a separate pass. The effective layer is resolved per instance during
        // the scene traversal, so two instances sharing a mesh+material but
        // differing in layer are separated correctly.
        let (overlay_batches, normal_batches) =
            partition_batches(&batches, |inst| inst.display.layer == RenderLayer::Overlay);
        batches = normal_batches;

        let active_highlight = highlight.filter(|h| !h.is_empty());

        // Whole-node outlines: only nodes selected *as a node* are outlined.
        // Sub-geometry selections do not contribute here — they are handled by
        // the sub-geom batches below.
        let primary_outlined_node = active_highlight.and_then(|h| h.primary_outlined_node());

        let highlighted_batches = active_highlight
            .map(|_| partition_batches(&batches, |inst| Some(inst.node_id) == primary_outlined_node).0)
            .unwrap_or_default();

        let secondary_highlighted_batches = active_highlight
            .map(|h| {
                let outlined: HashSet<NodeId> = h.outlined_nodes().into_iter().collect();
                partition_batches(&batches, |inst| {
                    outlined.contains(&inst.node_id) && Some(inst.node_id) != primary_outlined_node
                })
                .0
            })
            .unwrap_or_default();

        let sub_geom = active_highlight
            .map(|h| collect_highlight_sub_geom_batches(&batches, scene, h))
            .unwrap_or_default();
        let highlight_sub_geom_batches = sub_geom.primary;
        let secondary_highlight_sub_geom_batches = sub_geom.secondary;

        let highlight_config = active_highlight.map(|h| h.highlight_config());
        let secondary_highlight_config = active_highlight.and_then(|h| h.secondary_highlight_config());

        Self {
            batches,
            highlighted_batches,
            secondary_highlighted_batches,
            overlay_batches,
            highlight_sub_geom_batches,
            secondary_highlight_sub_geom_batches,
            highlight_config,
            secondary_highlight_config,
        }
    }

    /// All batches (opaque, transparent, selected, etc.), sorted for rendering.
    pub fn all_batches(&self) -> &[DrawBatch] {
        &self.batches
    }

    /// Batches containing only the primary highlighted instance, for outline/mask rendering.
    /// Empty if no highlight is active.
    pub fn highlighted_batches(&self) -> &[DrawBatch] {
        &self.highlighted_batches
    }

    /// Batches containing secondary (non-primary) highlighted instances.
    /// Empty if there is only one selection or no highlight is active.
    pub fn secondary_highlighted_batches(&self) -> &[DrawBatch] {
        &self.secondary_highlighted_batches
    }

    /// Whether any instances or sub-geometry are highlighted (primary or secondary).
    pub fn has_highlights(&self) -> bool {
        self.has_node_highlights() || self.has_sub_geom_highlights()
    }

    /// Whether any whole-node outlines need to be drawn (primary or secondary).
    pub fn has_node_highlights(&self) -> bool {
        !self.highlighted_batches.is_empty() || !self.secondary_highlighted_batches.is_empty()
    }

    /// Whether any sub-geometry highlights need to be drawn (primary or secondary).
    pub fn has_sub_geom_highlights(&self) -> bool {
        !self.highlight_sub_geom_batches.is_empty()
            || !self.secondary_highlight_sub_geom_batches.is_empty()
    }

    /// Sub-geometry draw calls for primary highlighted faces/edges.
    pub fn highlight_sub_geom_batches(&self) -> &[SubGeomBatch] {
        &self.highlight_sub_geom_batches
    }

    /// Sub-geometry draw calls for secondary highlighted faces/edges.
    pub fn secondary_highlight_sub_geom_batches(&self) -> &[SubGeomBatch] {
        &self.secondary_highlight_sub_geom_batches
    }

    /// Batches on `RenderLayer::Overlay`, rendered in a separate overlay pass.
    pub fn overlay_batches(&self) -> &[DrawBatch] {
        &self.overlay_batches
    }

    /// Whether any overlay (always-on-top) batches exist.
    pub fn has_overlay(&self) -> bool {
        !self.overlay_batches.is_empty()
    }

    /// Outline configuration for the primary selection, or `None` if nothing is highlighted.
    pub fn highlight_config(&self) -> Option<&crate::highlight_query::HighlightConfig> {
        self.highlight_config.as_ref()
    }

    /// Outline configuration for secondary selections, or `None` if there are none.
    pub fn secondary_highlight_config(&self) -> Option<&crate::highlight_query::HighlightConfig> {
        self.secondary_highlight_config.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duck_engine_common::{
        Deg, EuclideanSpace, InnerSpace, Matrix4, Quaternion, Rotation3, SquareMatrix, Vector3,
    };
    use duck_engine_scene::resource::NodeFlags;
    use crate::scene::common::EPSILON;

    fn nid() -> NodeId { NodeId::new() }
    fn iid() -> InstanceId { InstanceId::new() }
    fn mid() -> MeshId { MeshId::new() }
    fn matid() -> BatchMaterial { BatchMaterial::Face(crate::scene::resource::FaceMaterialId::new()) }

    // ========================================================================
    // InstanceTransform Tests
    // ========================================================================

    #[test]
    fn test_instance_transform_creation() {
        let node_id = nid();
        let instance_id = iid();
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(node_id, instance_id, transform);

        assert_eq!(instance_transform.node_id, node_id);
        assert_eq!(instance_transform.instance_id, instance_id);
        assert_eq!(instance_transform.world_transform, transform);
    }

    #[test]
    fn test_instance_transform_identity() {
        let identity = Matrix4::identity();
        let instance_transform = InstanceTransform::new(nid(), iid(), identity);

        // Verify identity transform
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    assert_eq!(instance_transform.world_transform[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.world_transform[i][j], 0.0);
                }
            }
        }

        // Verify identity normal matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(instance_transform.normal_matrix[i][j], 1.0);
                } else {
                    assert_eq!(instance_transform.normal_matrix[i][j], 0.0);
                }
            }
        }
    }

    #[test]
    fn test_instance_transform_normal_matrix_computed() {
        let transform = Matrix4::from_scale(2.0);
        let instance_transform = InstanceTransform::new(nid(), iid(), transform);

        // Normal matrix should be inverse-transpose
        // For uniform scale of 2.0, normal matrix should be 0.5
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.5).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_translation() {
        let transform = Matrix4::from_translation(Vector3::new(5.0, 10.0, 15.0));
        let instance_transform = InstanceTransform::new(nid(), iid(), transform);

        // Translation should be in the transform
        assert_eq!(instance_transform.world_transform[3][0], 5.0);
        assert_eq!(instance_transform.world_transform[3][1], 10.0);
        assert_eq!(instance_transform.world_transform[3][2], 15.0);

        // Normal matrix should remain identity (translation doesn't affect normals)
        assert_eq!(instance_transform.normal_matrix[0][0], 1.0);
        assert_eq!(instance_transform.normal_matrix[1][1], 1.0);
        assert_eq!(instance_transform.normal_matrix[2][2], 1.0);
    }

    #[test]
    fn test_instance_transform_rotation() {
        let rotation = Quaternion::from_angle_z(Deg(90.0));
        let transform = Matrix4::from(rotation);
        let instance_transform = InstanceTransform::new(nid(), iid(), transform);

        // Normal matrix should match rotation (orthogonal matrices)
        // For 90 degree Z rotation: (1,0,0) -> (0,1,0)
        let normal = instance_transform.normal_matrix;

        // Check that applying normal matrix to (1,0,0) gives approximately (0,1,0)
        let x = normal[0][0] * 1.0 + normal[1][0] * 0.0 + normal[2][0] * 0.0;
        let y = normal[0][1] * 1.0 + normal[1][1] * 0.0 + normal[2][1] * 0.0;

        assert!(x.abs() < EPSILON);
        assert!((y - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_non_uniform_scale() {
        let transform = Matrix4::from_nonuniform_scale(2.0, 3.0, 4.0);
        let instance_transform = InstanceTransform::new(nid(), iid(), transform);

        // Normal matrix should handle non-uniform scale correctly
        // Inverse of diagonal matrix (2,3,4) is (0.5, 0.333..., 0.25)
        assert!((instance_transform.normal_matrix[0][0] - 0.5).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[1][1] - 1.0/3.0).abs() < EPSILON);
        assert!((instance_transform.normal_matrix[2][2] - 0.25).abs() < EPSILON);
    }

    #[test]
    fn test_instance_transform_different_instances_different_ids() {
        let transform = Matrix4::identity();
        let instance1 = InstanceTransform::new(nid(), iid(), transform);
        let instance2 = InstanceTransform::new(nid(), iid(), transform);
        let instance3 = InstanceTransform::new(nid(), iid(), transform);

        assert_ne!(instance1.instance_id, instance2.instance_id);
        assert_ne!(instance1.instance_id, instance3.instance_id);
        assert_ne!(instance2.instance_id, instance3.instance_id);
    }

    #[test]
    fn test_instance_transform_with_complex_transform() {
        // Combine translation, rotation, and scale
        let translation = Matrix4::from_translation(Vector3::new(10.0, 20.0, 30.0));
        let rotation = Matrix4::from(Quaternion::from_angle_y(Deg(45.0)));
        let scale = Matrix4::from_scale(2.0);
        let transform = translation * rotation * scale;

        let node_id = nid();
        let instance_id = iid();
        let instance_transform = InstanceTransform::new(node_id, instance_id, transform);

        assert_eq!(instance_transform.node_id, node_id);
        assert_eq!(instance_transform.instance_id, instance_id);
        assert_eq!(instance_transform.world_transform, transform);

        // Verify normal matrix was computed (not identity)
        let expected_normal = common::compute_normal_matrix(&transform);

        for i in 0..3 {
            for j in 0..3 {
                assert!((instance_transform.normal_matrix[i][j] - expected_normal[i][j]).abs() < EPSILON);
            }
        }
    }

    // ========================================================================
    // DrawBatch Tests
    // ========================================================================

    #[test]
    fn test_draw_batch_new() {
        let mesh_id = mid();
        let material_id = matid();
        let batch = DrawBatch::new(mesh_id, material_id, PrimitiveType::TriangleList, 3);

        assert_eq!(batch.mesh_id, mesh_id);
        assert_eq!(batch.material, material_id);
        assert_eq!(batch.primitive_type, PrimitiveType::TriangleList);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_draw_batch_add_instance() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);
        let instance_id = iid();

        let instance_transform = InstanceTransform::new(nid(), instance_id, Matrix4::identity());
        batch.add_instance(instance_transform);

        assert!(!batch.is_empty());
        assert_eq!(batch.len(), 1);
        assert_eq!(batch.instances[0].instance_id, instance_id);
    }

    #[test]
    fn test_draw_batch_add_multiple_instances() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);
        let instance_ids: Vec<InstanceId> = (0..5).map(|_| iid()).collect();

        for &id in &instance_ids {
            batch.add_instance(InstanceTransform::new(nid(), id, Matrix4::identity()));
        }

        assert_eq!(batch.len(), 5);
        assert!(!batch.is_empty());

        for (i, &expected_id) in instance_ids.iter().enumerate() {
            assert_eq!(batch.instances[i].instance_id, expected_id);
        }
    }

    #[test]
    fn test_draw_batch_mesh_material_ids() {
        let mesh1 = mid();
        let mat1 = matid();
        let mesh2 = mid();
        let mat2 = matid();
        let batch1 = DrawBatch::new(mesh1, mat1, PrimitiveType::TriangleList, 3);
        let batch2 = DrawBatch::new(mesh2, mat2, PrimitiveType::LineList, 2);

        assert_eq!(batch1.mesh_id, mesh1);
        assert_eq!(batch1.material, mat1);
        assert_eq!(batch1.primitive_type, PrimitiveType::TriangleList);

        assert_eq!(batch2.mesh_id, mesh2);
        assert_eq!(batch2.material, mat2);
        assert_eq!(batch2.primitive_type, PrimitiveType::LineList);
    }

    #[test]
    fn test_draw_batch_instance_count() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);

        assert_eq!(batch.len(), 0);

        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        assert_eq!(batch.len(), 1);

        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        assert_eq!(batch.len(), 2);

        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_draw_batch_instances_with_different_transforms() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);

        let transform1 = Matrix4::from_scale(1.0);
        let transform2 = Matrix4::from_scale(2.0);
        let transform3 = Matrix4::from_translation(Vector3::new(5.0, 0.0, 0.0));

        batch.add_instance(InstanceTransform::new(nid(), iid(), transform1));
        batch.add_instance(InstanceTransform::new(nid(), iid(), transform2));
        batch.add_instance(InstanceTransform::new(nid(), iid(), transform3));

        assert_eq!(batch.len(), 3);

        // Verify transforms are preserved
        assert_eq!(batch.instances[0].world_transform, transform1);
        assert_eq!(batch.instances[1].world_transform, transform2);
        assert_eq!(batch.instances[2].world_transform, transform3);
    }

    #[test]
    fn test_draw_batch_large_number_of_instances() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);

        for _ in 0..1000 {
            batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        }

        assert_eq!(batch.len(), 1000);
        assert!(!batch.is_empty());
    }

    // ========================================================================
    // partition_batches Tests
    // ========================================================================

    #[test]
    fn test_partition_batches_empty() {
        let batches: Vec<DrawBatch> = vec![];
        let (matched, unmatched) = partition_batches(&batches, |_| true);
        assert!(matched.is_empty());
        assert!(unmatched.is_empty());
    }

    #[test]
    fn test_partition_batches_all_match() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);
        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |_| true);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].instances.len(), 2);
        assert!(unmatched.is_empty());
    }

    #[test]
    fn test_partition_batches_none_match() {
        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);
        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(nid(), iid(), Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |_| false);
        assert!(matched.is_empty());
        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].instances.len(), 2);
    }

    #[test]
    fn test_partition_batches_split() {
        let node_a = nid();
        let node_b = nid();
        let node_c = nid();
        // We'll match only node_b
        let match_id = node_b;

        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 3);
        batch.add_instance(InstanceTransform::new(node_a, iid(), Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(node_b, iid(), Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(node_c, iid(), Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |inst| inst.node_id == match_id);

        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].instances.len(), 1);
        assert_eq!(matched[0].instances[0].node_id, match_id);

        assert_eq!(unmatched.len(), 1);
        assert_eq!(unmatched[0].instances.len(), 2);
    }

    #[test]
    fn test_partition_batches_preserves_index_count() {
        let node_a = nid();
        let node_b = nid();

        let mut batch = DrawBatch::new(mid(), matid(), PrimitiveType::TriangleList, 42);
        batch.add_instance(InstanceTransform::new(node_a, iid(), Matrix4::identity()));
        batch.add_instance(InstanceTransform::new(node_b, iid(), Matrix4::identity()));

        let (matched, unmatched) = partition_batches(&[batch], |inst| inst.node_id == node_a);

        // The draw loops read index_count off the batch, so a partition that
        // dropped it would silently draw the wrong number of indices.
        assert_eq!(matched[0].index_count, 42);
        assert_eq!(unmatched[0].index_count, 42);
    }

    // ========================================================================
    // DrawData Tests
    // ========================================================================

    use crate::highlight_query::HighlightConfig;

    struct MockHighlight {
        highlighted_nodes: Vec<NodeId>,
    }

    impl HighlightQuery for MockHighlight {
        fn is_empty(&self) -> bool {
            self.highlighted_nodes.is_empty()
        }

        fn outlined_nodes(&self) -> Vec<NodeId> {
            self.highlighted_nodes.clone()
        }

        fn primary_outlined_node(&self) -> Option<NodeId> {
            self.highlighted_nodes.first().copied()
        }

        fn highlighted_for_node(&self, _node_id: NodeId, _kind: SubGeometryKind) -> Vec<u32> {
            Vec::new()
        }

        fn nodes_with_highlighted(&self, _kind: SubGeometryKind) -> Vec<NodeId> {
            Vec::new()
        }

        fn primary_sub_geometry(&self) -> Option<(NodeId, SubGeometryElement)> {
            None
        }

        fn highlight_config(&self) -> HighlightConfig {
            HighlightConfig::default()
        }

        fn secondary_highlight_config(&self) -> Option<HighlightConfig> {
            if self.highlighted_nodes.len() > 1 {
                Some(HighlightConfig::default())
            } else {
                None
            }
        }
    }

    /// A simple camera for tests that exercise `DrawData::new`.
    fn test_camera() -> PositionedCamera {
        PositionedCamera {
            eye: Point3::new(0.0, 0.0, 5.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: duck_engine_common::Vector3::new(0.0, 1.0, 0.0),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
            ortho: false,
        }
    }

    /// Creates a scene with one instance node (empty mesh + default material).
    fn build_simple_scene() -> (SceneData, NodeId) {
        let mut scene = SceneData::new();
        let mesh_id = scene.add_mesh(crate::scene::resource::Mesh::new());
        let material_id = scene.add_face_material(crate::scene::resource::FaceMaterial::new());
        let node_id = scene.add_instance_node(
            None, Instance::new(mesh_id).with_face_material(material_id), None, common::Transform::IDENTITY, NodeFlags::NONE
        ).unwrap().id();
        (scene, node_id)
    }

    #[test]
    fn test_draw_data_no_highlight() {
        let scene = SceneData::new();
        let draw_data = DrawData::new(&scene, &test_camera(), (256, 256), None);

        assert!(draw_data.all_batches().is_empty());
        assert!(draw_data.highlighted_batches().is_empty());
        assert!(!draw_data.has_highlights());
    }

    #[test]
    fn test_draw_data_empty_highlight() {
        let (scene, _node_id) = build_simple_scene();
        let highlight = MockHighlight {
            highlighted_nodes: vec![],
        };
        let draw_data = DrawData::new(
            &scene,
            &test_camera(),
            (256, 256),
            Some(&highlight),
        );

        assert!(!draw_data.has_highlights());
        assert!(draw_data.highlighted_batches().is_empty());
    }

    #[test]
    fn test_draw_data_with_highlight() {
        let (scene, node_id) = build_simple_scene();
        let highlight = MockHighlight {
            highlighted_nodes: vec![node_id],
        };
        let draw_data = DrawData::new(
            &scene,
            &test_camera(),
            (256, 256),
            Some(&highlight),
        );

        // Scene has no mesh primitives (empty mesh), so no batches are generated
        // This test verifies the highlight path runs without errors
        assert!(!draw_data.has_highlights() || !draw_data.highlighted_batches().is_empty());
    }

    /// Two instance nodes share the same mesh + material but differ in render
    /// layer; the overlay partition must split them per instance.
    #[test]
    fn test_overlay_partition_is_per_instance() {
        use crate::scene::resource::{
            DisplayBehavior, Mesh, MeshPrimitive, PrimitiveType, RenderLayer, Vertex,
        };

        let vertex = Vertex {
            position: [0.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let mesh = Mesh::from_raw(
            vec![vertex, vertex, vertex],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: vec![0, 1, 2],
            }],
        );

        let mut scene = SceneData::new();
        let mesh_id = scene.add_mesh(mesh);
        let material_id = scene.add_face_material(crate::scene::resource::FaceMaterial::new());

        let scene_node = scene
            .add_instance_node(None, Instance::new(mesh_id.clone()).with_face_material(material_id.clone()), None, common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        let overlay_node = scene
            .add_instance_node(None, Instance::new(mesh_id).with_face_material(material_id), None, common::Transform::IDENTITY, NodeFlags::NONE)
            .unwrap()
            .id();
        scene.set_node_display(
            overlay_node,
            DisplayBehavior { layer: RenderLayer::Overlay, ..Default::default() },
        );

        let draw_data = DrawData::new(&scene, &test_camera(), (256, 256), None);

        // Exactly one overlay batch and one normal batch, each with one instance.
        let overlay_instances: usize = draw_data.overlay_batches().iter().map(|b| b.len()).sum();
        let normal_instances: usize = draw_data.all_batches().iter().map(|b| b.len()).sum();
        assert_eq!(overlay_instances, 1, "overlay node should produce one overlay instance");
        assert_eq!(normal_instances, 1, "scene node should produce one normal instance");
        assert!(draw_data.has_overlay());

        // Sanity: the overlay instance is the one we flagged.
        let overlay_node_ids: Vec<_> = draw_data
            .overlay_batches()
            .iter()
            .flat_map(|b| b.instances.iter().map(|i| i.node_id))
            .collect();
        assert_eq!(overlay_node_ids, vec![overlay_node]);
        let _ = scene_node;
    }

    /// A `screen_facing` instance gets an effective transform whose rotation
    /// basis faces the camera (forward column points at the eye, columns are
    /// orthonormal and right-handed) while its world position is preserved.
    #[test]
    fn test_billboard_effective_transform() {
        use crate::scene::resource::{Mesh, MeshPrimitive, PrimitiveType, Vertex};

        let vertex = Vertex {
            position: [0.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        };
        let mesh = Mesh::from_raw(
            vec![vertex, vertex, vertex],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: vec![0, 1, 2],
            }],
        );

        let mut scene = SceneData::new();
        let mesh_id = scene.add_mesh(mesh);
        let material_id = scene.add_face_material(crate::scene::resource::FaceMaterial::new());

        // Place the node off the view axis so the billboard basis is non-trivial.
        let p = Point3::new(5.0, 0.0, 0.0);
        let node = scene
            .add_instance_node(
                None,
                Instance::new(mesh_id).with_face_material(material_id),
                None,
                common::Transform::from_position(p),
                NodeFlags::NONE,
            )
            .unwrap()
            .id();
        scene.set_node_display(node, DisplayBehavior { screen_facing: true, ..Default::default() });

        let camera = test_camera();
        let draw_data = DrawData::new(&scene, &camera, (256, 256), None);

        let inst = &draw_data.all_batches()[0].instances[0];
        let m = inst.effective_transform;
        let right = m.x.truncate();
        let up = m.y.truncate();
        let fwd = m.z.truncate();

        // World position is preserved in the translation column.
        assert!((m.w.truncate() - p.to_vec()).magnitude() < EPSILON);

        // Forward column faces the eye.
        let expected_fwd = (camera.eye - p).normalize();
        assert!((fwd - expected_fwd).magnitude() < EPSILON);

        // Orthonormal columns.
        for c in [right, up, fwd] {
            assert!((c.magnitude() - 1.0).abs() < EPSILON);
        }
        assert!(right.dot(up).abs() < EPSILON);
        assert!(right.dot(fwd).abs() < EPSILON);
        assert!(up.dot(fwd).abs() < EPSILON);

        // Right-handed basis (right × up == fwd) and right ⟂ camera up.
        assert!((right.cross(up) - fwd).magnitude() < EPSILON);
        assert!(right.dot(camera.up).abs() < EPSILON);

        // The camera-independent world transform is untouched.
        assert!((inst.world_transform.w.truncate() - p.to_vec()).magnitude() < EPSILON);
    }
}
