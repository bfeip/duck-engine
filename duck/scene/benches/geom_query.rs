//! Benchmarks for the geometry query hot paths: per-mesh ray intersection and
//! the scene-walking pick queries. These are the loops interactive snapping and
//! picking run per cursor event, so they must stay fast on dense meshes.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use duck_engine_scene::common::{Point3, Ray, Transform, Vector3};
use duck_engine_scene::geom_query::{
    intersect_ray, intersect_ray_nearest, intersect_ray_with_lines, pick_all_from_ray,
    MeshSpatialIndex, RayPickQuery,
};
use duck_engine_scene::{
    Instance, Mesh, MeshPrimitive, NodeFlags, PrimitiveType, Scene, SceneData, Vertex,
};

/// World-space tolerance for line picks, roughly a few pixels at working distance.
const LINE_TOLERANCE: f32 = 0.05;

/// An `n`×`n`-vertex displaced grid over `[0, size]²` in XZ: triangulated
/// surface plus every grid row/column as line segments — the shape of a
/// tessellated CAD face with its iso/boundary wires.
fn grid_mesh(n: usize, size: f32) -> Mesh {
    let step = size / (n - 1) as f32;
    let mut vertices = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let x = i as f32 * step;
            let z = j as f32 * step;
            let y = ((x * 1.3).sin() + (z * 0.7).cos()) * 0.1;
            vertices.push(Vertex {
                position: [x, y, z],
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            });
        }
    }

    let mut triangles = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = a + 1;
            let c = a + n as u32;
            let d = c + 1;
            triangles.extend([a, b, c, b, d, c]);
        }
    }

    let mut lines = Vec::with_capacity(n * (n - 1) * 4);
    for j in 0..n {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            lines.extend([a, a + 1]);
        }
    }
    for j in 0..n - 1 {
        for i in 0..n {
            let a = (j * n + i) as u32;
            lines.extend([a, a + n as u32]);
        }
    }

    Mesh::from_raw(
        vertices,
        vec![
            MeshPrimitive {
                primitive_type: PrimitiveType::TriangleList,
                indices: triangles,
            },
            MeshPrimitive {
                primitive_type: PrimitiveType::LineList,
                indices: lines,
            },
        ],
    )
}

/// A scene with `count`×`count` instances of one shared dense mesh, laid out on
/// a grid with gaps so a ray hits at most one instance.
fn instanced_scene(mesh: Mesh, count: usize, spacing: f32) -> Scene {
    let mut scene = SceneData::new();
    let mesh = scene.add_mesh(mesh);
    for j in 0..count {
        for i in 0..count {
            let position = Point3::new(i as f32 * spacing, 0.0, j as f32 * spacing);
            scene
                .add_instance_node(
                    None,
                    Instance::new(mesh.clone()),
                    Some(format!("part_{i}_{j}")),
                    Transform::from_position(position),
                    NodeFlags::NONE,
                )
                .expect("instance node");
        }
    }
    Scene::new(scene)
}

/// Straight-down ray through `(x, z)`.
fn ray_down(x: f32, z: f32) -> Ray {
    Ray::new(Point3::new(x, 10.0, z), Vector3::new(0.0, -1.0, 0.0))
}

fn bench_mesh_queries(c: &mut Criterion) {
    // ~79k triangles, ~80k segments.
    let mesh = grid_mesh(200, 10.0);
    let ray = ray_down(5.02, 5.03);

    let mut group = c.benchmark_group("mesh");
    group.bench_function("intersect_ray_80k_tris", |b| {
        b.iter(|| black_box(intersect_ray(black_box(&mesh), black_box(&ray))))
    });
    group.bench_function("intersect_ray_nearest_80k_tris", |b| {
        b.iter(|| black_box(intersect_ray_nearest(black_box(&mesh), black_box(&ray))))
    });
    group.bench_function("intersect_ray_with_lines_80k_segs", |b| {
        b.iter(|| {
            black_box(intersect_ray_with_lines(
                black_box(&mesh),
                black_box(&ray),
                LINE_TOLERANCE,
            ))
        })
    });
    group.finish();
}

fn bench_spatial_index(c: &mut Criterion) {
    let mesh = grid_mesh(200, 10.0);
    let ray = ray_down(5.02, 5.03);
    let index = MeshSpatialIndex::build(&mesh);

    let mut group = c.benchmark_group("spatial");
    group.bench_function("build_80k", |b| {
        b.iter(|| black_box(MeshSpatialIndex::build(black_box(&mesh))))
    });
    group.bench_function("nearest_triangle_80k", |b| {
        b.iter(|| black_box(index.nearest_triangle(black_box(&mesh), black_box(&ray))))
    });
    group.bench_function("segments_within_80k", |b| {
        b.iter(|| {
            let mut count = 0u32;
            index.for_each_segment_within(
                black_box(&mesh),
                black_box(&ray),
                LINE_TOLERANCE,
                |_, _| count += 1,
            );
            black_box(count)
        })
    });
    group.finish();
}

fn bench_scene_queries(c: &mut Criterion) {
    // One dense part alone, and a 10×10 grid of instances of the same part.
    let single = instanced_scene(grid_mesh(200, 10.0), 1, 0.0);
    let many = instanced_scene(grid_mesh(200, 10.0), 10, 15.0);

    // Hits the interior of the part at grid cell (3, 3).
    let hit_ray = ray_down(3.0 * 15.0 + 5.02, 3.0 * 15.0 + 5.03);
    // Passes through the gap between parts: broad-phase culls everything.
    let miss_ray = ray_down(12.5, 12.5);

    let mut group = c.benchmark_group("scene");
    group.sample_size(20);
    group.bench_function("pick_faces_single_80k", |b| {
        let ray = ray_down(5.02, 5.03);
        b.iter(|| black_box(pick_all_from_ray(&RayPickQuery::faces(ray), &single)))
    });
    group.bench_function("pick_all_10x10_hit", |b| {
        b.iter(|| {
            black_box(pick_all_from_ray(
                &RayPickQuery::all(hit_ray, LINE_TOLERANCE),
                &many,
            ))
        })
    });
    group.bench_function("pick_all_10x10_miss", |b| {
        b.iter(|| {
            black_box(pick_all_from_ray(
                &RayPickQuery::all(miss_ray, LINE_TOLERANCE),
                &many,
            ))
        })
    });
    group.finish();
}

criterion_group!(benches, bench_mesh_queries, bench_spatial_index, bench_scene_queries);
criterion_main!(benches);
