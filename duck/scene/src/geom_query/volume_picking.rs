use duck_engine_common::Matrix4;

use crate::common::{Aabb, ConvexPolyhedron};
use crate::Scene;
use crate::resource::{InstanceId, Mesh, Node, NodeId};

use super::mesh_intersection::{self, MeshVolumeHit};
use super::pick_query::{pick_all, PickQuery};

/// Result of a volume-instance intersection test.
#[derive(Debug, Clone)]
pub struct VolumePickResult {
    /// The node that was hit
    pub node_id: NodeId,
    /// The instance that was hit
    pub instance_id: InstanceId,
    /// The model matrix the instance was tested at, taking its mesh's local
    /// space to world space.
    pub transform: Matrix4,
    /// Where each of the mesh's primitives lies relative to the volume.
    pub hit: MeshVolumeHit,
}

/// Volume picking query; a [`PickQuery`] over a convex volume.
pub struct VolumePickQuery {
    /// The volume in current coordinate space (may be transformed to local space)
    volume: ConvexPolyhedron,
    /// Whether to use thorough (but slower) edge-triangle intersection tests
    thorough: bool,
}

impl VolumePickQuery {
    /// Creates a query for a world-space convex volume. `thorough` enables
    /// more accurate but slower edge-triangle tests.
    pub fn new(volume: ConvexPolyhedron, thorough: bool) -> Self {
        Self { volume, thorough }
    }
}

impl PickQuery for VolumePickQuery {
    type Result = VolumePickResult;

    fn might_intersect_bounds(&self, bounds: &Aabb) -> bool {
        self.volume.intersects_aabb(bounds)
    }

    fn transform(&self, matrix: &Matrix4) -> Self {
        Self {
            volume: self.volume.transform(matrix),
            thorough: self.thorough,
        }
    }

    fn collect_mesh_hits(
        &self,
        mesh: &Mesh,
        node: &Node,
        instance_id: InstanceId,
        world_transform: &Matrix4,
        results: &mut Vec<Self::Result>,
    ) {
        // Test against mesh (volume is already in local space)
        if let Some(hit) = mesh_intersection::intersect_volume(mesh, &self.volume, self.thorough) {
            results.push(VolumePickResult {
                node_id: node.id,
                instance_id,
                transform: *world_transform,
                hit,
            });
        }
    }
}

/// Picks all instances intersected by a world-space convex volume.
///
/// Each [`VolumePickResult`] records where each of the instance's primitives
/// lies relative to the volume. `thorough` enables more accurate but slower
/// edge-triangle intersection tests, catching volumes that pass through a
/// triangle without any of its vertices being inside and without its edges
/// crossing the volume boundary.
pub fn pick_all_from_volume(
    volume: &ConvexPolyhedron,
    scene: &Scene,
    thorough: bool,
) -> Vec<VolumePickResult> {
    let query = VolumePickQuery::new(volume.clone(), thorough);
    pick_all(&query, scene)
}
