//! Per-mesh spatial acceleration: flat BVHs over a mesh's triangles and line
//! segments. Obtain one via [`Mesh::spatial_index`](crate::Mesh::spatial_index),
//! which caches it keyed by the mesh generation.

use duck_engine_common::Point3;

use crate::common::{Aabb, Ray, SegmentApproach};
use crate::Mesh;

use super::mesh_intersection::TriangleMeshHit;

/// Primitives per leaf. Small enough that a leaf test is a handful of
/// intersections, large enough to keep the tree shallow.
const LEAF_SIZE: usize = 8;

/// Traversal stack capacity. The median split yields a balanced tree, so depth
/// is ~log2(n / LEAF_SIZE) + 1; 64 covers any `u32`-indexed primitive count.
const MAX_DEPTH: usize = 64;

/// Spatial index over one mesh's primitives: one flat BVH for triangles and one
/// for line segments. Primitive indices match the enumeration order of
/// [`Mesh::triangles`] / [`Mesh::segments`].
#[derive(Clone)]
pub struct MeshSpatialIndex {
    triangles: FlatBvh,
    segments: FlatBvh,
}

impl MeshSpatialIndex {
    /// Builds the index for a mesh. O(n log n) in the primitive count.
    pub fn build(mesh: &Mesh) -> Self {
        let triangle_bounds: Vec<Aabb> = mesh
            .triangles()
            .map(|[v0, v1, v2]| {
                Aabb::from_points(&[
                    Point3::from(v0.position),
                    Point3::from(v1.position),
                    Point3::from(v2.position),
                ])
                .expect("three points")
            })
            .collect();
        let segment_bounds: Vec<Aabb> = mesh
            .segments()
            .map(|[v0, v1]| {
                Aabb::from_points(&[Point3::from(v0.position), Point3::from(v1.position)])
                    .expect("two points")
            })
            .collect();

        Self {
            triangles: FlatBvh::build(&triangle_bounds),
            segments: FlatBvh::build(&segment_bounds),
        }
    }

    /// The nearest ray–triangle intersection, like
    /// [`intersect_ray_nearest`](super::intersect_ray_nearest). The ray must be
    /// in local mesh space.
    pub fn nearest_triangle(&self, mesh: &Mesh, ray: &Ray) -> Option<TriangleMeshHit> {
        let mut best: Option<TriangleMeshHit> = None;
        // Shared with the descend closure (which prunes subtrees entirely
        // beyond the best hit) without aliasing the `best` borrow.
        let best_distance = std::cell::Cell::new(f32::INFINITY);

        self.triangles.traverse(
            ray,
            0.0,
            |entry| entry < best_distance.get(),
            |triangle_index| {
                let Some([v0, v1, v2]) = mesh.triangle(triangle_index) else {
                    return;
                };
                let p0 = Point3::from(v0.position);
                let p1 = Point3::from(v1.position);
                let p2 = Point3::from(v2.position);
                if let Some((t, u, v)) = ray.intersect_triangle(p0, p1, p2) {
                    if t < best_distance.get() {
                        best_distance.set(t);
                        best = Some(TriangleMeshHit {
                            distance: t,
                            hit_point: ray.point_at(t),
                            triangle_index,
                            barycentric: (u, v, 1.0 - u - v),
                        });
                    }
                }
            },
        );

        best
    }

    /// Appends every ray–triangle intersection to `out`, like
    /// [`intersect_ray`](super::intersect_ray) but skipping subtrees the ray
    /// misses. Hit order is unspecified. The ray must be in local mesh space.
    pub fn all_triangle_hits(&self, mesh: &Mesh, ray: &Ray, out: &mut Vec<TriangleMeshHit>) {
        self.triangles.traverse(ray, 0.0, |_| true, |triangle_index| {
            let Some([v0, v1, v2]) = mesh.triangle(triangle_index) else {
                return;
            };
            let p0 = Point3::from(v0.position);
            let p1 = Point3::from(v1.position);
            let p2 = Point3::from(v2.position);
            if let Some((t, u, v)) = ray.intersect_triangle(p0, p1, p2) {
                out.push(TriangleMeshHit {
                    distance: t,
                    hit_point: ray.point_at(t),
                    triangle_index,
                    barycentric: (u, v, 1.0 - u - v),
                });
            }
        });
    }

    /// Calls `f(segment_index, approach)` for every segment whose closest
    /// approach to the ray is within `tolerance`, in unspecified order. The ray
    /// and tolerance must be in local mesh space.
    pub fn for_each_segment_within(
        &self,
        mesh: &Mesh,
        ray: &Ray,
        tolerance: f32,
        mut f: impl FnMut(usize, &SegmentApproach),
    ) {
        self.segments.traverse(ray, tolerance, |_| true, |segment_index| {
            let Some([v0, v1]) = mesh.segment(segment_index) else {
                return;
            };
            let p0 = Point3::from(v0.position);
            let p1 = Point3::from(v1.position);
            if let Some(approach) = ray.closest_approach_to_segment(p0, p1) {
                if approach.distance <= tolerance {
                    f(segment_index, &approach);
                }
            }
        });
    }
}

/// One node of a [`FlatBvh`], in depth-first preorder.
#[derive(Clone)]
struct BvhNode {
    aabb: Aabb,
    /// Leaf (`count > 0`): start of this leaf's range in `prims`.
    /// Internal (`count == 0`): index of the right child (left child is the
    /// next node).
    index: u32,
    count: u32,
}

/// A flat BVH over primitive bounding boxes. Leaves reference ranges of
/// `prims`, which holds the primitive indices reordered by the build.
#[derive(Clone)]
struct FlatBvh {
    nodes: Vec<BvhNode>,
    prims: Vec<u32>,
}

impl FlatBvh {
    fn build(bounds: &[Aabb]) -> Self {
        let mut prims: Vec<u32> = (0..bounds.len() as u32).collect();
        let mut nodes = Vec::new();
        if !bounds.is_empty() {
            nodes.reserve(2 * bounds.len().div_ceil(LEAF_SIZE));
            build_node(bounds, &mut prims, 0, &mut nodes);
        }
        Self { nodes, prims }
    }

    /// Visits every leaf primitive in subtrees whose (tolerance-inflated) AABB
    /// the ray hits. `descend(entry_t)` can prune a node after its AABB test —
    /// nearest-hit queries use it to skip nodes beyond the current best.
    fn traverse(
        &self,
        ray: &Ray,
        tolerance: f32,
        mut descend: impl FnMut(f32) -> bool,
        mut visit: impl FnMut(usize),
    ) {
        if self.nodes.is_empty() {
            return;
        }

        let mut stack = [0u32; MAX_DEPTH];
        let mut top = 1; // stack[0] = 0, the root

        while top > 0 {
            top -= 1;
            let node_index = stack[top];
            let node = &self.nodes[node_index as usize];
            let aabb = inflate(&node.aabb, tolerance);
            let Some(entry) = aabb.intersects_ray(ray) else {
                continue;
            };
            if !descend(entry) {
                continue;
            }

            if node.count > 0 {
                let start = node.index as usize;
                for &prim in &self.prims[start..start + node.count as usize] {
                    visit(prim as usize);
                }
            } else {
                debug_assert!(top + 2 <= MAX_DEPTH, "BVH deeper than traversal stack");
                // Right child first so the left child pops first.
                stack[top] = node.index;
                stack[top + 1] = node_index + 1;
                top += 2;
            }
        }
    }
}

/// Recursively builds the subtree for `prims` (a slice of the global primitive
/// array starting at `offset`), appending nodes in preorder.
fn build_node(bounds: &[Aabb], prims: &mut [u32], offset: u32, nodes: &mut Vec<BvhNode>) {
    let mut aabb = bounds[prims[0] as usize];
    for &p in &prims[1..] {
        aabb = aabb.merge(&bounds[p as usize]);
    }

    let node_index = nodes.len();
    nodes.push(BvhNode {
        aabb,
        index: 0, // placeholder
        count: 0
    });

    if prims.len() <= LEAF_SIZE {
        // This is a leaf node. Assign final values and return.
        nodes[node_index].index = offset;
        nodes[node_index].count = prims.len() as u32;
        return;
    }

    // Median split on the longest axis of the centroid bounds: balanced by
    // construction, so traversal depth is bounded regardless of geometry.
    let mut centroid_bounds = Aabb::new(bounds[prims[0] as usize].center(), bounds[prims[0] as usize].center());
    for &p in &prims[1..] {
        centroid_bounds = centroid_bounds.expand(bounds[p as usize].center());
    }
    let (sx, sy, sz) = centroid_bounds.size();
    let axis = if sx >= sy && sx >= sz {
        0
    } else if sy >= sz {
        1
    } else {
        2
    };
    let centroid = |p: u32| -> f32 {
        let c = bounds[p as usize].center();
        [c.x, c.y, c.z][axis]
    };

    let mid = prims.len() / 2;
    prims.select_nth_unstable_by(mid, |&a, &b| centroid(a).total_cmp(&centroid(b)));

    let (left, right) = prims.split_at_mut(mid);
    build_node(bounds, left, offset, nodes);
    nodes[node_index].index = nodes.len() as u32;
    build_node(bounds, right, offset + mid as u32, nodes);
}

fn inflate(aabb: &Aabb, amount: f32) -> Aabb {
    if amount == 0.0 {
        return *aabb;
    }
    Aabb::new(
        Point3::new(aabb.min.x - amount, aabb.min.y - amount, aabb.min.z - amount),
        Point3::new(aabb.max.x + amount, aabb.max.y + amount, aabb.max.z + amount),
    )
}

#[cfg(test)]
mod tests {
    use duck_engine_common::{InnerSpace, Point3, Vector3};

    use crate::common::Ray;
    use crate::geom_query::mesh_intersection;
    use crate::{Mesh, MeshPrimitive, PrimitiveType, Vertex};

    use super::MeshSpatialIndex;

    const EPSILON: f32 = 1e-6;

    /// Deterministic xorshift PRNG so failures reproduce.
    struct Rng(u32);

    impl Rng {
        /// Uniform in [-1, 1).
        fn coord(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            (self.0 >> 8) as f32 / (1 << 23) as f32 - 1.0
        }

        fn vertex(&mut self, center: [f32; 3], spread: f32) -> Vertex {
            Vertex {
                position: [
                    center[0] + self.coord() * spread,
                    center[1] + self.coord() * spread,
                    center[2] + self.coord() * spread,
                ],
                tex_coords: [0.0; 3],
                normal: [0.0; 3],
            }
        }

        fn ray(&mut self) -> Ray {
            let origin = Point3::new(
                self.coord() * 3.0,
                self.coord() * 3.0,
                self.coord() * 3.0,
            );
            let target = Point3::new(self.coord(), self.coord(), self.coord());
            let direction = (target - origin).normalize();
            Ray::new(origin, direction)
        }
    }

    /// Random small triangles and segments scattered in [-1, 1]³, split across
    /// two primitives per type (index parity across primitives matters), with a
    /// zero-area triangle and a zero-length segment mixed in.
    fn random_mesh(seed: u32, triangle_count: usize, segment_count: usize) -> Mesh {
        let mut rng = Rng(seed);
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        let mut segments = Vec::new();

        for i in 0..triangle_count {
            let center = [rng.coord(), rng.coord(), rng.coord()];
            let base = vertices.len() as u32;
            let spread = if i == 0 { 0.0 } else { 0.3 }; // i == 0: zero-area
            let v = rng.vertex(center, spread);
            vertices.push(v);
            vertices.push(if i == 0 { v } else { rng.vertex(center, spread) });
            vertices.push(if i == 0 { v } else { rng.vertex(center, spread) });
            triangles.extend([base, base + 1, base + 2]);
        }
        for i in 0..segment_count {
            let center = [rng.coord(), rng.coord(), rng.coord()];
            let base = vertices.len() as u32;
            let v = rng.vertex(center, 0.3);
            vertices.push(v);
            vertices.push(if i == 0 { v } else { rng.vertex(center, 0.3) }); // i == 0: zero-length
            segments.extend([base, base + 1]);
        }

        let (tri_a, tri_b) = triangles.split_at(triangle_count / 2 * 3);
        let (seg_a, seg_b) = segments.split_at(segment_count / 2 * 2);
        Mesh::from_raw(
            vertices,
            vec![
                MeshPrimitive {
                    primitive_type: PrimitiveType::TriangleList,
                    indices: tri_a.to_vec(),
                },
                MeshPrimitive {
                    primitive_type: PrimitiveType::LineList,
                    indices: seg_a.to_vec(),
                },
                MeshPrimitive {
                    primitive_type: PrimitiveType::TriangleList,
                    indices: tri_b.to_vec(),
                },
                MeshPrimitive {
                    primitive_type: PrimitiveType::LineList,
                    indices: seg_b.to_vec(),
                },
            ],
        )
    }

    #[test]
    fn nearest_triangle_matches_brute_force() {
        for seed in 1..6 {
            let mesh = random_mesh(seed, 64, 0);
            let index = MeshSpatialIndex::build(&mesh);
            let mut rng = Rng(seed * 1000 + 7);
            for _ in 0..50 {
                let ray = rng.ray();
                let bvh = index.nearest_triangle(&mesh, &ray);
                let brute = mesh_intersection::intersect_ray(&mesh, &ray)
                    .into_iter()
                    .min_by(|a, b| a.distance.total_cmp(&b.distance));
                match (&bvh, &brute) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert!(
                            (a.distance - b.distance).abs() <= EPSILON,
                            "seed {seed}: bvh {a:?} vs brute {b:?}"
                        );
                    }
                    _ => panic!("seed {seed}: bvh {bvh:?} vs brute {brute:?}"),
                }
            }
        }
    }

    #[test]
    fn all_triangle_hits_match_brute_force() {
        for seed in 1..6 {
            let mesh = random_mesh(seed, 64, 0);
            let index = MeshSpatialIndex::build(&mesh);
            let mut rng = Rng(seed * 1000 + 13);
            for _ in 0..50 {
                let ray = rng.ray();
                let mut bvh = Vec::new();
                index.all_triangle_hits(&mesh, &ray, &mut bvh);
                let mut bvh: Vec<usize> = bvh.into_iter().map(|h| h.triangle_index).collect();
                bvh.sort_unstable();
                let mut brute: Vec<usize> = mesh_intersection::intersect_ray(&mesh, &ray)
                    .into_iter()
                    .map(|h| h.triangle_index)
                    .collect();
                brute.sort_unstable();
                assert_eq!(bvh, brute, "seed {seed}");
            }
        }
    }

    #[test]
    fn segments_within_match_brute_force() {
        for seed in 1..6 {
            let mesh = random_mesh(seed, 0, 64);
            let index = MeshSpatialIndex::build(&mesh);
            let mut rng = Rng(seed * 1000 + 29);
            for tolerance in [0.05, 0.3] {
                for _ in 0..25 {
                    let ray = rng.ray();
                    let mut bvh = Vec::new();
                    index.for_each_segment_within(&mesh, &ray, tolerance, |i, approach| {
                        assert!(approach.distance <= tolerance);
                        bvh.push(i);
                    });
                    bvh.sort_unstable();
                    let mut brute: Vec<usize> =
                        mesh_intersection::intersect_ray_with_lines(&mesh, &ray, tolerance)
                            .into_iter()
                            .map(|h| h.segment_index)
                            .collect();
                    brute.sort_unstable();
                    assert_eq!(bvh, brute, "seed {seed} tolerance {tolerance}");
                }
            }
        }
    }

    #[test]
    fn empty_and_single_kind_meshes() {
        let ray = Ray::new(Point3::new(0.0, 0.0, -5.0), Vector3::new(0.0, 0.0, 1.0));

        let empty = Mesh::new();
        let index = MeshSpatialIndex::build(&empty);
        assert!(index.nearest_triangle(&empty, &ray).is_none());
        index.for_each_segment_within(&empty, &ray, 1.0, |_, _| panic!("no segments"));

        let tri_only = random_mesh(3, 16, 0);
        let index = MeshSpatialIndex::build(&tri_only);
        index.for_each_segment_within(&tri_only, &ray, 1.0, |_, _| panic!("no segments"));

        let seg_only = random_mesh(4, 0, 16);
        let index = MeshSpatialIndex::build(&seg_only);
        assert!(index.nearest_triangle(&seg_only, &ray).is_none());
    }

    #[test]
    fn mesh_cache_invalidates_on_mutation() {
        let vertex = |x: f32| Vertex {
            position: [x, 0.0, 5.0],
            tex_coords: [0.0; 3],
            normal: [0.0; 3],
        };
        let mut mesh = Mesh::from_raw(
            vec![vertex(-1.0), vertex(1.0), {
                let mut v = vertex(0.0);
                v.position[1] = 1.0;
                v
            }],
            vec![MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: vec![0, 1, 2],
            }],
        );
        let ray = Ray::new(Point3::new(0.0, 0.3, 0.0), Vector3::new(0.0, 0.0, 1.0));

        assert!(mesh.spatial_index().nearest_triangle(&mesh, &ray).is_some());

        // Repeated access reuses the cached index.
        let a = mesh.spatial_index();
        let b = mesh.spatial_index();
        assert!(std::ptr::eq(&*a, &*b));
        drop((a, b));

        // Move the triangle out of the ray's path; the stale index would still hit.
        mesh.translate(Vector3::new(100.0, 0.0, 0.0));
        assert!(mesh.spatial_index().nearest_triangle(&mesh, &ray).is_none());
    }
}
