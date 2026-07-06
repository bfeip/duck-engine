//! Construction grid: minor/major lines, axis cross, oriented to a construction plane.

use duck_engine_viewer::common::{
    EuclideanSpace, InnerSpace, Matrix3, Plane, Point3, Quaternion, RgbaColor, Transform, Vector3,
};
use duck_engine_viewer::scene::{
    Instance, LineMaterial, Mesh, MeshIndex, MeshPrimitive, NodeFlags, NodeId, PrimitiveType, Scene,
    Vertex,
};

const GRID_NORMAL: [f32; 3] = [0.0, 1.0, 0.0];

/// Visual and dimensional parameters for the construction grid.
pub struct GridConfig {
    /// Whether the grid is drawn at all. Snapping is unaffected.
    pub visible: bool,
    /// Full side length of the grid in world units.
    pub size: f32,
    /// Distance between adjacent minor lines in world units.
    pub minor_spacing: f32,
    /// A major line is drawn every Nth minor line.
    pub major_every: u32,
    /// Color of the minor (fine) gridlines.
    pub minor_color: RgbaColor,
    /// Color of the major (coarse) gridlines.
    pub major_color: RgbaColor,
    /// Color of the first in-plane axis line (analogous to world X).
    pub axis_u_color: RgbaColor,
    /// Color of the second in-plane axis line (analogous to world Z).
    pub axis_v_color: RgbaColor,
    /// Color of the construction-plane normal indicator line.
    pub axis_normal_color: RgbaColor,
    /// Length of the short normal-axis indicator line, in world units.
    pub normal_axis_length: f32,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            visible: true,
            size: 1000.0,
            minor_spacing: 5.0,
            major_every: 5,
            minor_color: RgbaColor { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
            major_color: RgbaColor { r: 0.10, g: 0.10, b: 0.10, a: 1.0 },
            axis_u_color: RgbaColor { r: 0.65, g: 0.20, b: 0.20, a: 1.0 },
            axis_v_color: RgbaColor { r: 0.20, g: 0.30, b: 0.65, a: 1.0 },
            axis_normal_color: RgbaColor { r: 0.20, g: 0.55, b: 0.25, a: 1.0 },
            normal_axis_length: 25.0,
        }
    }
}

/// A grid installed in a scene: handles to the nodes that comprise it, so it
/// can be removed (e.g., when the construction plane changes).
pub struct Grid {
    nodes: Vec<NodeId>,
}

impl Grid {
    /// Creates the grid meshes, materials, and inert instance nodes in `scene`,
    /// oriented to lie on `plane`. When the config is not visible, creates
    /// nothing (an empty grid).
    pub fn add_to_scene(scene: &mut Scene, config: &GridConfig, plane: &Plane) -> Self {
        if !config.visible {
            return Self { nodes: Vec::new() };
        }
        let transform = plane_to_transform(plane);

        let minor = build_grid_mesh(config, GridLayer::Minor);
        let major = build_grid_mesh(config, GridLayer::Major);
        let axis_u = build_axis_mesh([1.0, 0.0, 0.0], config.size);
        let axis_v = build_axis_mesh([0.0, 0.0, 1.0], config.size);
        let axis_n = build_axis_mesh([0.0, 1.0, 0.0], config.normal_axis_length);

        let nodes = [
            ("Grid (minor)", minor, config.minor_color),
            ("Grid (major)", major, config.major_color),
            ("Grid axis U", axis_u, config.axis_u_color),
            ("Grid axis V", axis_v, config.axis_v_color),
            ("Grid axis N", axis_n, config.axis_normal_color),
        ]
        .into_iter()
        .map(|(name, mesh, color)| {
            let mesh_id = scene.add_mesh(mesh);
            let material_id = scene.add_line_material(LineMaterial::new(color));
            scene
                .add_instance_node(
                    None,
                    Instance::new(mesh_id).with_line_material(material_id),
                    Some(name.to_owned()),
                    transform,
                    NodeFlags::inert(),
                )
                .expect("Grid instance node creation failed")
        })
        .collect();

        Self { nodes }
    }

    /// Removes the grid's nodes and their meshes/materials from `scene`.
    pub fn remove_from_scene(self, scene: &mut Scene) {
        for node in self.nodes {
            scene.cleanup_node(node);
        }
    }
}

/// Which subset of gridlines a mesh contains. Layers are mutually exclusive
/// and exclude the lines that fall on an axis, so no two line segments in the
/// grid are ever coincident (eliminates z-fighting).
#[derive(Copy, Clone)]
enum GridLayer {
    Minor,
    Major,
}

fn build_grid_mesh(config: &GridConfig, layer: GridLayer) -> Mesh {
    let half = config.size / 2.0;
    let spacing = config.minor_spacing.max(f32::EPSILON);
    let major_every = config.major_every.max(1) as i32;
    let half_steps = (half / spacing).floor() as i32;

    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<MeshIndex> = Vec::new();

    let push_segment = |vertices: &mut Vec<Vertex>,
                        indices: &mut Vec<MeshIndex>,
                        a: [f32; 3],
                        b: [f32; 3]| {
        let i = vertices.len() as MeshIndex;
        vertices.push(Vertex { position: a, tex_coords: [0.0; 3], normal: GRID_NORMAL });
        vertices.push(Vertex { position: b, tex_coords: [0.0; 3], normal: GRID_NORMAL });
        indices.push(i);
        indices.push(i + 1);
    };

    for step in -half_steps..=half_steps {
        let is_axis = step == 0;
        let is_major = step.rem_euclid(major_every) == 0 && !is_axis;
        let include = match layer {
            GridLayer::Minor => !is_axis && !is_major,
            GridLayer::Major => is_major,
        };
        if !include {
            continue;
        }
        let coord = step as f32 * spacing;
        push_segment(&mut vertices, &mut indices, [-half, 0.0, coord], [half, 0.0, coord]);
        push_segment(&mut vertices, &mut indices, [coord, 0.0, -half], [coord, 0.0, half]);
    }

    Mesh::from_raw(
        vertices,
        vec![MeshPrimitive { primitive_type: PrimitiveType::LineList, indices }],
    )
}

fn build_axis_mesh(direction: [f32; 3], length: f32) -> Mesh {
    let half = length / 2.0;
    let a = [-direction[0] * half, -direction[1] * half, -direction[2] * half];
    let b = [direction[0] * half, direction[1] * half, direction[2] * half];
    let vertices = vec![
        Vertex { position: a, tex_coords: [0.0; 3], normal: GRID_NORMAL },
        Vertex { position: b, tex_coords: [0.0; 3], normal: GRID_NORMAL },
    ];
    Mesh::from_raw(
        vertices,
        vec![MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: vec![0, 1] }],
    )
}

/// Rigid transform that places the canonical grid geometry onto `plane`.
///
/// The grid meshes (`build_grid_mesh`, `build_axis_mesh`) are authored once in a
/// fixed local space — the XZ plane with +Y as the normal (`GRID_NORMAL`). This
/// returns the transform that rotates and translates that local space onto the
/// construction plane, so the very same meshes can be reused for any plane
/// orientation instead of rebuilding them per plane.
///
/// The rotation's columns are the plane's frame `(u, normal, v)`: `normal` is the
/// middle column because the grid's local up axis is +Y, and `u`/`v` are the
/// in-plane axes that the grid's local X and Z map onto (see [`Plane::basis`]).
/// The translation is the plane point nearest the world origin, keeping the grid
/// centred near the origin.
fn plane_to_transform(plane: &Plane) -> Transform {
    let (u, v) = plane.basis();
    let n = plane.normal.normalize();
    let rotation: Quaternion = Matrix3::from_cols(u, n, v).into();
    let position = plane.project_point(Point3::origin());
    Transform { position, rotation, scale: Vector3::new(1.0, 1.0, 1.0) }
}
