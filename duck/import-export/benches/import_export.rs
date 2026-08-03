use std::path::PathBuf;

use criterion::{Criterion, criterion_group, criterion_main};
use duck_engine_import_export::{LoadOptions, SceneSource, load_sync};

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets")
}

fn glb_path() -> PathBuf {
    assets_dir().join("1987_mazda_rx-7_fc.glb")
}

fn gltf_path() -> PathBuf {
    assets_dir().join("Camera_01_4k.gltf/Camera_01_4k.gltf")
}

fn bench_gltf_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("gltf_loading");

    group.bench_function("load_glb", |b| {
        b.iter(|| {
            load_sync(
                SceneSource::Path(glb_path()),
                LoadOptions::default(),
            )
            .unwrap()
        });
    });

    group.bench_function("load_gltf_with_textures", |b| {
        b.iter(|| {
            load_sync(
                SceneSource::Path(gltf_path()),
                LoadOptions::default(),
            )
            .unwrap()
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Harness
// ============================================================================

criterion_group!(benches, bench_gltf_loading);
criterion_main!(benches);
