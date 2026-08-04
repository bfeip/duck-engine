use std::{
    cell::{Cell, Ref, RefCell},
    collections::HashSet,
    fs::File,
    io::BufReader,
    path::Path,
};

use anyhow::{Context, Result};
use duck_engine_common::{transform_point, InnerSpace, Matrix4, Point3, Vector3};

mod primitives;

use crate::common::Aabb;
use crate::geom_query::MeshSpatialIndex;

/// Unique identifier for a mesh in the scene.
pub type MeshId = crate::Id<Mesh>;

/// Index type used for mesh index buffers.
pub type MeshIndex = u32;

/// Primitive types for mesh rendering
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrimitiveType {
    TriangleList,
    LineList,
    PointList,
}

impl PrimitiveType {
    /// Number of vertex indices consumed by one primitive of this type:
    /// 3 for triangles, 2 for line segments, 1 for points. Used to convert
    /// element-counted [`SubMeshRange`]s into raw index offsets.
    pub fn indices_per_primitive(self) -> u32 {
        match self {
            PrimitiveType::TriangleList => 3,
            PrimitiveType::LineList => 2,
            PrimitiveType::PointList => 1,
        }
    }
}

/// A kind of mesh sub-geometry that can be selected or highlighted independently.
///
/// Each kind maps 1:1 to a [`PrimitiveType`] and to one of [`Topology`]'s range
/// lists, letting selection/highlight code be written once and parametrized by kind
/// rather than triplicated across faces, edges, and points.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum SubGeometryKind {
    Face,
    Edge,
    Pointset,
}

impl SubGeometryKind {
    /// All sub-geometry kinds, for iterating over every kind.
    pub const ALL: [SubGeometryKind; 3] =
        [SubGeometryKind::Face, SubGeometryKind::Edge, SubGeometryKind::Pointset];

    /// The mesh primitive type this kind is built from.
    pub fn primitive_type(self) -> PrimitiveType {
        match self {
            SubGeometryKind::Face => PrimitiveType::TriangleList,
            SubGeometryKind::Edge => PrimitiveType::LineList,
            SubGeometryKind::Pointset => PrimitiveType::PointList,
        }
    }
}

/// A single sub-geometry element within a mesh: a [`SubGeometryKind`] plus the
/// element's index into that kind's [`Topology`] range list.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SubGeometryElement {
    pub kind: SubGeometryKind,
    pub index: u32,
}

impl SubGeometryElement {
    pub fn new(kind: SubGeometryKind, index: u32) -> Self {
        Self { kind, index }
    }
}

/// A collection of indices representing a single primitive type in a mesh
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshPrimitive {
    pub primitive_type: PrimitiveType,
    pub indices: Vec<MeshIndex>,
}

impl MeshPrimitive {
    /// Converts a TriangleList to a LineList with deduplicated edges.
    ///
    /// Each triangle `[a, b, c]` produces edges `[a,b], [b,c], [c,a]`. Shared
    /// edges between adjacent triangles appear only once in the output.
    ///
    /// Returns `None` if this primitive is not a TriangleList.
    pub fn to_line_list(&self) -> Option<MeshPrimitive> {
        if self.primitive_type != PrimitiveType::TriangleList {
            return None;
        }

        let mut edge_set: HashSet<(MeshIndex, MeshIndex)> = HashSet::new();
        let mut line_indices: Vec<MeshIndex> = Vec::new();

        for tri in self.indices.chunks_exact(3) {
            let (a, b, c) = (tri[0], tri[1], tri[2]);
            for (v0, v1) in [(a, b), (b, c), (c, a)] {
                let edge = if v0 <= v1 { (v0, v1) } else { (v1, v0) };
                if edge_set.insert(edge) {
                    line_indices.push(v0);
                    line_indices.push(v1);
                }
            }
        }

        Some(MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: line_indices,
        })
    }

    /// Converts a TriangleList or LineList to a PointList with unique vertex indices.
    ///
    /// Returns `None` if this primitive is already a PointList.
    pub fn to_point_list(&self) -> Option<MeshPrimitive> {
        if self.primitive_type == PrimitiveType::PointList {
            return None;
        }

        let mut seen: HashSet<MeshIndex> = HashSet::new();
        let mut point_indices: Vec<MeshIndex> = Vec::new();

        for &idx in &self.indices {
            if seen.insert(idx) {
                point_indices.push(idx);
            }
        }

        Some(MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices: point_indices,
        })
    }
}

/// Index range within a mesh primitive's index list.
///
/// Units depend on the primitive type:
/// - `TriangleList`: `start` and `count` are in **triangles** (multiply by 3 for raw index offset)
/// - `LineList`: `start` and `count` are in **segments** (multiply by 2 for raw index offset)
/// - `PointList`: `start` and `count` are in **points** (1:1 with raw indices)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubMeshRange {
    /// First element (in primitive-type units) within the primitive's index list.
    pub start: u32,
    /// Number of elements.
    pub count: u32,
}

/// Semantic grouping of a mesh's index ranges into named sub-geometry elements.
///
/// A mesh with topology knows which triangles form "face 3", which line segments
/// form "edge 7", etc. This enables sub-geometry selection and highlighting.
/// All fields are optional — set only the slices that are meaningful for a given mesh.
///
/// Ranges are parallel with the primitives list: `face_ranges` index into the first
/// `TriangleList` primitive, `edge_ranges` into the first `LineList` primitive, and
/// `point_ranges` into the first `PointList` primitive.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Topology {
    /// One entry per face element, in tessellation order.
    /// Indexes into the mesh's first `TriangleList` primitive's indices (in triangles).
    pub face_ranges: Vec<SubMeshRange>,
    /// One entry per edge/wire element, in iteration order.
    /// Indexes into the mesh's first `LineList` primitive's indices (in segments).
    pub edge_ranges: Vec<SubMeshRange>,
    /// One entry per point element.
    /// Indexes into the mesh's first `PointList` primitive's indices.
    pub pointset_ranges: Vec<SubMeshRange>,
}

impl Topology {
    /// Returns `true` if there are no face, edge, or point ranges.
    pub fn is_empty(&self) -> bool {
        self.face_ranges.is_empty() && self.edge_ranges.is_empty() && self.pointset_ranges.is_empty()
    }

    /// Returns the range list for the given sub-geometry kind.
    pub fn ranges(&self, kind: SubGeometryKind) -> &[SubMeshRange] {
        match kind {
            SubGeometryKind::Face => &self.face_ranges,
            SubGeometryKind::Edge => &self.edge_ranges,
            SubGeometryKind::Pointset => &self.pointset_ranges,
        }
    }
}

/// GPU-compatible vertex structure containing position, texture coordinates, and normal.
///
/// This struct is laid out in memory to match the vertex shader's expectations.
/// Each vertex is 36 bytes: 12 bytes position + 12 bytes tex_coords + 12 bytes normal.
///
/// # Memory Layout
/// - Uses `#[repr(C)]` for predictable layout
/// - Implements `Pod` and `Zeroable` for zero-copy GPU buffer uploads
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Vertex {
    /// Vertex position in local mesh space [x, y, z]
    pub position: [f32; 3],
    /// Texture coordinates [u, v, w] (w unused, reserved for 3D textures)
    pub tex_coords: [f32; 3],
    /// Vertex normal vector [x, y, z]
    pub normal: [f32; 3],
}

/// Source data for loading a Wavefront OBJ mesh.
///
/// OBJ meshes can be loaded from either in-memory bytes or a file path.
pub enum ObjMesh<'a> {
    /// OBJ data from an in-memory byte slice
    Bytes(&'a [u8]),
    /// OBJ data from a file path
    Path(&'a Path),
}

/// Descriptor for creating a mesh from various sources.
///
/// Meshes can be created as empty containers, loaded from OBJ files,
/// or constructed from raw vertex and primitive data.
pub enum MeshDescriptor<'a> {
    /// Creates an empty mesh with no vertices or primitives
    Empty,
    /// Loads mesh data from a Wavefront OBJ file
    Obj(ObjMesh<'a>),
    /// Creates mesh from raw vertex and primitive data
    Raw {
        /// Vertex data for the mesh
        vertices: Vec<Vertex>,
        /// Primitives (triangle lists, line lists, etc.) referencing the vertices
        primitives: Vec<MeshPrimitive>,
    },
}

/// A mesh composed of vertices and primitives
///
/// Meshes store vertex data (positions, normals, texture coordinates) and primitives
/// (triangle lists, line lists, point lists).
///
/// # Examples
///
/// ```
/// use duck_engine_scene::{Mesh, MeshPrimitive, Vertex, SceneData, PrimitiveType};
///
/// // Create from raw data (no device needed)
/// let vertices = vec![
///     Vertex { position: [0.0, 0.0, 0.0], tex_coords: [0.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [1.0, 0.0, 0.0], tex_coords: [1.0, 0.0, 0.0], normal: [0.0, 1.0, 0.0] },
///     Vertex { position: [0.5, 1.0, 0.0], tex_coords: [0.5, 1.0, 0.0], normal: [0.0, 1.0, 0.0] },
/// ];
/// let primitives = vec![MeshPrimitive {
///     primitive_type: PrimitiveType::TriangleList,
///     indices: vec![0, 1, 2],
/// }];
/// let mesh = Mesh::from_raw(vertices, primitives);
///
/// // Add to scene
/// let mut scene = SceneData::new();
/// let mesh_id = scene.add_mesh(mesh);
///
/// // GPU resources are created automatically during rendering
/// ```
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh {
    /// Unique identifier for this mesh
    pub id: MeshId,
    /// CPU-side vertex data
    vertices: Vec<Vertex>,
    /// CPU-side primitive data (index lists grouped by type)
    primitives: Vec<MeshPrimitive>,
    /// Generation counter - increments on any mutation (for change tracking)
    #[cfg_attr(feature = "serde", serde(skip, default = "crate::initial_generation"))]
    generation: u64,
    /// Cached local-space axis-aligned bounding box
    #[cfg_attr(feature = "serde", serde(skip))]
    cached_bounding: Cell<Option<Aabb>>,
    /// Lazily built spatial index over triangles/segments, cleared on mutation.
    #[cfg_attr(feature = "serde", serde(skip))]
    spatial_index: RefCell<Option<MeshSpatialIndex>>,
    /// Optional sub-geometry topology: maps faces/edges/points to index ranges.
    topology: Option<Topology>,
}

impl Mesh {
    /// Creates a new empty mesh with no vertices or primitives.
    pub fn new() -> Self {
        Self {
            id: crate::Id::new(),
            vertices: Vec::new(),
            primitives: Vec::new(),
            generation: crate::initial_generation(),
            cached_bounding: Cell::new(None),
            spatial_index: RefCell::new(None),
            topology: None,
        }
    }

    /// Creates a mesh from raw vertex and primitive data.
    ///
    /// # Arguments
    /// * `vertices` - Vertex data (positions, normals, texture coordinates)
    /// * `primitives` - Primitive data (index lists grouped by type)
    pub fn from_raw(vertices: Vec<Vertex>, primitives: Vec<MeshPrimitive>) -> Self {
        Self {
            id: crate::Id::new(),
            vertices,
            primitives,
            generation: crate::initial_generation(),
            cached_bounding: Cell::new(None),
            spatial_index: RefCell::new(None),
            topology: None,
        }
    }

    /// Creates a mesh from a descriptor.
    ///
    /// # Arguments
    /// * `descriptor` - Source data for the mesh (empty, OBJ file, or raw data)
    ///
    /// # Errors
    /// Returns an error if loading from OBJ fails.
    pub fn from_descriptor(descriptor: MeshDescriptor) -> Result<Self> {
        match descriptor {
            MeshDescriptor::Empty => Ok(Self::new()),
            MeshDescriptor::Obj(obj_desc) => match obj_desc {
                ObjMesh::Bytes(bytes) => Self::from_obj_bytes(bytes),
                ObjMesh::Path(path) => Self::from_obj_path(path),
            },
            MeshDescriptor::Raw { vertices, primitives } => Ok(Self::from_raw(vertices, primitives)),
        }
    }

    /// Loads a mesh from OBJ data in a byte slice.
    ///
    /// # Arguments
    /// * `obj_bytes` - OBJ file data as bytes
    ///
    /// # Errors
    /// Returns an error if the OBJ data is malformed or cannot be parsed
    pub fn from_obj_bytes(obj_bytes: &[u8]) -> Result<Self> {
        let obj: obj::Obj<obj::TexturedVertex> =
            obj::load_obj(obj_bytes).context("Failed to parse OBJ data")?;
        Self::from_obj(obj)
    }

    /// Loads a mesh from an OBJ file on disk.
    ///
    /// # Arguments
    /// * `obj_path` - File path to the OBJ file
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or the OBJ data is malformed
    pub fn from_obj_path<P: AsRef<Path>>(obj_path: P) -> Result<Self> {
        let path = obj_path.as_ref();
        let obj_file =
            File::open(path).with_context(|| format!("Failed to open OBJ file: {:?}", path))?;
        let obj_reader = BufReader::new(obj_file);
        let obj: obj::Obj<obj::TexturedVertex> =
            obj::load_obj(obj_reader).with_context(|| format!("Failed to parse OBJ file: {:?}", path))?;
        Self::from_obj(obj)
    }

    /// Creates a mesh from a parsed OBJ object.
    fn from_obj(obj: obj::Obj<obj::TexturedVertex>) -> Result<Self> {
        let vertices: Vec<Vertex> = obj
            .vertices
            .iter()
            .map(|v: &obj::TexturedVertex| Vertex {
                position: v.position,
                tex_coords: v.texture,
                normal: v.normal,
            })
            .collect();

        // OBJ files contain triangles, so create a triangle primitive
        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: obj.indices.iter().map(|&i| i as MeshIndex).collect(),
        }];

        Ok(Self::from_raw(vertices, primitives))
    }

    /// Set the mesh's vertex data.
    pub fn set_vertices(&mut self, vertices: Vec<Vertex>) {
        self.vertices = vertices;
        self.generation += 1;
        self.cached_bounding.set(None);
        *self.spatial_index.get_mut() = None;
    }

    /// Set the mesh's primitive data.
    pub fn set_primitives(&mut self, primitives: Vec<MeshPrimitive>) {
        self.primitives = primitives;
        self.generation += 1;
        *self.spatial_index.get_mut() = None;
    }

    /// Add vertices to the mesh.
    pub fn add_vertices(&mut self, vertices: &[Vertex]) {
        self.vertices.extend_from_slice(vertices);
        self.generation += 1;
        self.cached_bounding.set(None);
        *self.spatial_index.get_mut() = None;
    }

    /// Add a primitive to the mesh.
    pub fn add_primitive(&mut self, primitive: MeshPrimitive) {
        self.primitives.push(primitive);
        self.generation += 1;
        *self.spatial_index.get_mut() = None;
    }

    /// Translates all vertex positions by the given offset.
    pub fn translate(&mut self, offset: Vector3) {
        for v in &mut self.vertices {
            v.position[0] += offset.x;
            v.position[1] += offset.y;
            v.position[2] += offset.z;
        }
        self.generation += 1;
        self.cached_bounding.set(None);
        *self.spatial_index.get_mut() = None;
    }

    /// Translates all vertex positions by the given offset (consuming variant).
    pub fn translated(mut self, offset: Vector3) -> Self {
        self.translate(offset);
        self
    }

    /// Transforms all vertex positions and normals by a 4x4 matrix.
    ///
    /// Positions are transformed as points (affected by translation).
    /// Normals are transformed by the inverse-transpose of the upper 3x3
    /// to handle non-uniform scaling correctly.
    pub fn transform(&mut self, matrix: &Matrix4) {
        let normal_matrix = crate::common::compute_normal_matrix(matrix);

        for v in &mut self.vertices {
            // Transform position as a point
            let pos = Point3::new(v.position[0], v.position[1], v.position[2]);
            let transformed = transform_point(matrix, pos);
            v.position = [transformed.x, transformed.y, transformed.z];

            // Transform normal by inverse-transpose
            let normal = Vector3::new(v.normal[0], v.normal[1], v.normal[2]);
            let transformed_normal = (normal_matrix * normal).normalize();
            v.normal = [transformed_normal.x, transformed_normal.y, transformed_normal.z];
        }
        self.generation += 1;
        self.cached_bounding.set(None);
        *self.spatial_index.get_mut() = None;
    }

    /// Transforms all vertex positions and normals by a 4x4 matrix (consuming variant).
    pub fn transformed(mut self, matrix: &Matrix4) -> Self {
        self.transform(matrix);
        self
    }

    /// Merges another mesh into this one, combining vertices and primitives.
    ///
    /// Indices from `other` are offset by the current vertex count. Primitives
    /// of the same type are combined; new primitive types are appended.
    pub fn merge(&mut self, other: &Mesh) {
        let base_index = self.vertices.len() as MeshIndex;
        self.vertices.extend_from_slice(other.vertices());

        for other_prim in other.primitives() {
            let offset_indices: Vec<MeshIndex> =
                other_prim.indices.iter().map(|&i| i + base_index).collect();

            if let Some(existing) = self
                .primitives
                .iter_mut()
                .find(|p| p.primitive_type == other_prim.primitive_type)
            {
                existing.indices.extend(offset_indices);
            } else {
                self.primitives.push(MeshPrimitive {
                    primitive_type: other_prim.primitive_type,
                    indices: offset_indices,
                });
            }
        }

        self.generation += 1;
        self.cached_bounding.set(None);
        *self.spatial_index.get_mut() = None;
    }

    /// Merges another mesh into this one (consuming variant).
    pub fn merged(mut self, other: &Mesh) -> Self {
        self.merge(other);
        self
    }

    /// Returns the current generation counter.
    ///
    /// This value increments on any mutation to the mesh data.
    /// Used by renderers to track when GPU resources need updating.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns a reference to the mesh's vertex data.
    pub fn vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    /// Returns a reference to the mesh's primitives.
    pub fn primitives(&self) -> &[MeshPrimitive] {
        &self.primitives
    }

    /// Returns true if this mesh has any primitives of the specified type.
    pub fn has_primitive_type(&self, primitive_type: PrimitiveType) -> bool {
        self.primitives
            .iter()
            .any(|p| p.primitive_type == primitive_type)
    }

    /// Extracts all triangle indices from the mesh.
    ///
    /// Collects indices from all triangle list primitives in the mesh into a single vector.
    /// Each group of 3 indices defines one triangle.
    ///
    /// # Returns
    /// A vector of indices for all triangles. Empty if the mesh contains no triangle primitives.
    pub fn triangle_indices(&self) -> Vec<MeshIndex> {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
            .flat_map(|p| p.indices.iter().copied())
            .collect()
    }

    /// Extracts all line segment indices from the mesh.
    ///
    /// Collects indices from all line list primitives in the mesh into a single vector.
    /// Each group of 2 indices defines one segment.
    ///
    /// # Returns
    /// A vector of indices for all segments. Empty if the mesh contains no line primitives.
    pub fn line_indices(&self) -> Vec<MeshIndex> {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::LineList)
            .flat_map(|p| p.indices.iter().copied())
            .collect()
    }

    /// Extracts all point indices from the mesh.
    ///
    /// Collects indices from all point list primitives in the mesh into a single vector.
    ///
    /// # Returns
    /// A vector of indices for all points. Empty if the mesh contains no point primitives.
    pub fn point_indices(&self) -> Vec<MeshIndex> {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::PointList)
            .flat_map(|p| p.indices.iter().copied())
            .collect()
    }

    /// Returns triangle `index` (in [`Self::triangles`] order) as three resolved
    /// vertices, or `None` if the mesh has fewer triangles.
    pub fn triangle(&self, index: usize) -> Option<[Vertex; 3]> {
        let mut remaining = index;
        for primitive in self
            .primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
        {
            let count = primitive.indices.len() / 3;
            if remaining < count {
                let tri = &primitive.indices[remaining * 3..remaining * 3 + 3];
                return Some([
                    self.vertices[tri[0] as usize],
                    self.vertices[tri[1] as usize],
                    self.vertices[tri[2] as usize],
                ]);
            }
            remaining -= count;
        }
        None
    }

    /// Returns segment `index` (in [`Self::segments`] order) as two resolved
    /// vertices, or `None` if the mesh has fewer segments.
    pub fn segment(&self, index: usize) -> Option<[Vertex; 2]> {
        let mut remaining = index;
        for primitive in self
            .primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::LineList)
        {
            let count = primitive.indices.len() / 2;
            if remaining < count {
                let seg = &primitive.indices[remaining * 2..remaining * 2 + 2];
                return Some([
                    self.vertices[seg[0] as usize],
                    self.vertices[seg[1] as usize],
                ]);
            }
            remaining -= count;
        }
        None
    }

    /// Returns point `index` (in [`Self::points`] order) as a resolved vertex,
    /// or `None` if the mesh has fewer points.
    pub fn point(&self, index: usize) -> Option<Vertex> {
        let mut remaining = index;
        for primitive in self
            .primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::PointList)
        {
            let count = primitive.indices.len();
            if remaining < count {
                return Some(self.vertices[primitive.indices[remaining] as usize]);
            }
            remaining -= count;
        }
        None
    }

    /// Iterates over each triangle as three resolved vertices.
    ///
    /// Yields `[v0, v1, v2]` for every triangle in the mesh's `TriangleList` primitives,
    /// in order. Callers do not need to know the indexing convention.
    pub fn triangles(&self) -> impl Iterator<Item = [Vertex; 3]> + '_ {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::TriangleList)
            .flat_map(|p| p.indices.chunks_exact(3))
            .map(|tri| [
                self.vertices[tri[0] as usize],
                self.vertices[tri[1] as usize],
                self.vertices[tri[2] as usize],
            ])
    }

    /// Iterates over each line segment as two resolved vertices.
    ///
    /// Yields `[v0, v1]` for every segment in the mesh's `LineList` primitives,
    /// in order. Callers do not need to know the indexing convention.
    pub fn segments(&self) -> impl Iterator<Item = [Vertex; 2]> + '_ {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::LineList)
            .flat_map(|p| p.indices.chunks_exact(2))
            .map(|seg| [
                self.vertices[seg[0] as usize],
                self.vertices[seg[1] as usize],
            ])
    }

    /// Iterates over each point as a resolved vertex.
    ///
    /// Yields one `Vertex` for every point in the mesh's `PointList` primitives, in order.
    pub fn points(&self) -> impl Iterator<Item = Vertex> + '_ {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == PrimitiveType::PointList)
            .flat_map(|p| p.indices.iter())
            .map(|&i| self.vertices[i as usize])
    }

    /// Get the count of indices for a primitive type.
    pub fn index_count(&self, primitive_type: PrimitiveType) -> u32 {
        self.primitives
            .iter()
            .filter(|p| p.primitive_type == primitive_type)
            .map(|p| p.indices.len())
            .sum::<usize>() as u32
    }

    /// Computes the local-space axis-aligned bounding box for a mesh.
    /// Returns None if the mesh has no vertices.
    pub fn bounding(&self) -> Option<Aabb> {
        let cached_bounding = self.cached_bounding.get();
        if cached_bounding.is_some() {
            return cached_bounding;
        }

        if self.vertices.is_empty() {
            return None;
        }

        // Extract positions from vertices
        let positions: Vec<Point3> = self
            .vertices
            .iter()
            .map(|v| Point3::new(v.position[0], v.position[1], v.position[2]))
            .collect();

        let bounding = Aabb::from_points(&positions);
        self.cached_bounding.set(bounding);
        bounding
    }

    /// Spatial index over this mesh's triangles and segments, built lazily and
    /// rebuilt after any mutation.
    ///
    /// The returned guard borrows the cache; drop it before mutating the mesh.
    pub fn spatial_index(&self) -> Ref<'_, MeshSpatialIndex> {
        if self.spatial_index.borrow().is_none() {
            *self.spatial_index.borrow_mut() = Some(MeshSpatialIndex::build(self));
        }
        Ref::map(self.spatial_index.borrow(), |slot| {
            slot.as_ref().expect("just built")
        })
    }

    /// Returns the mesh's sub-geometry topology, if any.
    pub fn topology(&self) -> Option<&Topology> {
        self.topology.as_ref()
    }

    /// Sets the sub-geometry topology for this mesh.
    ///
    /// Does not increment the generation counter because topology does not affect
    /// GPU buffer contents — it is CPU-side metadata used for picking and selection.
    ///
    /// Note: if mesh editing is ever supported, topology changes may need to trigger
    /// a generation increment (and GPU re-upload) if the vertex/index layout changes.
    pub fn set_topology(&mut self, topology: Topology) {
        self.topology = Some(topology);
    }

    /// Returns the sub-geometry element of `kind` that contains `primitive_index`
    /// (counting individual primitives — triangles, segments, or points — in the
    /// matching primitive list).
    ///
    /// Returns `None` if this mesh has no topology or `primitive_index` is out of range.
    pub fn sub_geometry_for_primitive(
        &self,
        kind: SubGeometryKind,
        primitive_index: u32,
    ) -> Option<u32> {
        let topo = self.topology.as_ref()?;
        topo.ranges(kind)
            .iter()
            .position(|r| primitive_index >= r.start && primitive_index < r.start + r.count)
            .map(|i| i as u32)
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vertex(x: f32, y: f32, z: f32) -> Vertex {
        Vertex { position: [x, y, z], tex_coords: [0.0; 3], normal: [0.0, 1.0, 0.0] }
    }

    fn make_mixed_mesh() -> Mesh {
        let vertices = vec![
            make_vertex(0.0, 0.0, 0.0),
            make_vertex(1.0, 0.0, 0.0),
            make_vertex(0.0, 1.0, 0.0),
            make_vertex(0.0, 0.0, 1.0),
        ];
        let primitives = vec![
            MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] },
            MeshPrimitive { primitive_type: PrimitiveType::LineList, indices: vec![0, 3, 1, 3] },
            MeshPrimitive { primitive_type: PrimitiveType::PointList, indices: vec![0, 1, 2, 3] },
        ];
        Mesh::from_raw(vertices, primitives)
    }

    // ========== to_line_list ==========

    #[test]
    fn single_triangle_to_line_list() {
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![0, 1, 2],
        };
        let lines = prim.to_line_list().unwrap();
        assert_eq!(lines.primitive_type, PrimitiveType::LineList);
        // 3 edges: (0,1), (1,2), (2,0)
        assert_eq!(lines.indices.len(), 6);
        assert_eq!(lines.indices, vec![0, 1, 1, 2, 2, 0]);
    }

    #[test]
    fn shared_edge_deduplicated() {
        // Two triangles sharing edge (0,2): [0,1,2] and [0,2,3]
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        let lines = prim.to_line_list().unwrap();
        // 5 unique edges, not 6: (0,1), (1,2), (0,2) shared, (2,3), (0,3)
        assert_eq!(lines.indices.len(), 10);

        // Collect edges as pairs for verification
        let edges: Vec<(u32, u32)> = lines
            .indices
            .chunks_exact(2)
            .map(|e| {
                let (a, b) = (e[0], e[1]);
                if a <= b { (a, b) } else { (b, a) }
            })
            .collect();
        assert_eq!(edges.len(), 5);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
        assert!(edges.contains(&(0, 2)));
        assert!(edges.contains(&(2, 3)));
        assert!(edges.contains(&(0, 3)));
    }

    #[test]
    fn to_line_list_empty_indices() {
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![],
        };
        let lines = prim.to_line_list().unwrap();
        assert_eq!(lines.indices.len(), 0);
    }

    #[test]
    fn to_line_list_non_triangle_returns_none() {
        let line_prim = MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1],
        };
        assert!(line_prim.to_line_list().is_none());

        let point_prim = MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices: vec![0],
        };
        assert!(point_prim.to_line_list().is_none());
    }

    // ========== to_point_list ==========

    #[test]
    fn triangle_to_point_list() {
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::TriangleList,
            indices: vec![0, 1, 2, 0, 2, 3],
        };
        let points = prim.to_point_list().unwrap();
        assert_eq!(points.primitive_type, PrimitiveType::PointList);
        assert_eq!(points.indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn line_to_point_list() {
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1, 1, 2, 2, 0],
        };
        let points = prim.to_point_list().unwrap();
        assert_eq!(points.primitive_type, PrimitiveType::PointList);
        assert_eq!(points.indices, vec![0, 1, 2]);
    }

    #[test]
    fn point_list_to_point_list_returns_none() {
        let prim = MeshPrimitive {
            primitive_type: PrimitiveType::PointList,
            indices: vec![0, 1, 2],
        };
        assert!(prim.to_point_list().is_none());
    }

    // ========== Mesh index accessors and primitive iterators ==========

    #[test]
    fn line_indices_returns_correct_indices() {
        let mesh = make_mixed_mesh();
        assert_eq!(mesh.line_indices(), vec![0, 3, 1, 3]);
    }

    #[test]
    fn line_indices_empty_for_triangle_only_mesh() {
        let mesh = Mesh::from_raw(
            vec![make_vertex(0.0, 0.0, 0.0), make_vertex(1.0, 0.0, 0.0), make_vertex(0.0, 1.0, 0.0)],
            vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
        );
        assert!(mesh.line_indices().is_empty());
    }

    #[test]
    fn triangles_iter_yields_correct_vertices() {
        let mesh = make_mixed_mesh();
        let tris: Vec<_> = mesh.triangles().collect();
        assert_eq!(tris.len(), 1);
        assert_eq!(tris[0][0].position, [0.0, 0.0, 0.0]);
        assert_eq!(tris[0][1].position, [1.0, 0.0, 0.0]);
        assert_eq!(tris[0][2].position, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn segments_iter_yields_correct_vertices() {
        let mesh = make_mixed_mesh();
        let segs: Vec<_> = mesh.segments().collect();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0][0].position, [0.0, 0.0, 0.0]);
        assert_eq!(segs[0][1].position, [0.0, 0.0, 1.0]);
        assert_eq!(segs[1][0].position, [1.0, 0.0, 0.0]);
        assert_eq!(segs[1][1].position, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn points_iter_yields_correct_vertices() {
        let mesh = make_mixed_mesh();
        let pts: Vec<_> = mesh.points().collect();
        assert_eq!(pts.len(), 4);
        assert_eq!(pts[0].position, [0.0, 0.0, 0.0]);
        assert_eq!(pts[3].position, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn segments_iter_empty_for_triangle_only_mesh() {
        let mesh = Mesh::from_raw(
            vec![make_vertex(0.0, 0.0, 0.0), make_vertex(1.0, 0.0, 0.0), make_vertex(0.0, 1.0, 0.0)],
            vec![MeshPrimitive { primitive_type: PrimitiveType::TriangleList, indices: vec![0, 1, 2] }],
        );
        assert_eq!(mesh.segments().count(), 0);
    }

    // ========== point_for_vertex ==========

    #[test]
    fn point_for_vertex_resolves_ranges() {
        let mut mesh = make_mixed_mesh();
        // Two logical points: point 0 spans the first 2 vertices, point 1 the last 2.
        mesh.set_topology(Topology {
            face_ranges: vec![],
            edge_ranges: vec![],
            pointset_ranges: vec![
                SubMeshRange { start: 0, count: 2 },
                SubMeshRange { start: 2, count: 2 },
            ],
        });
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 0), Some(0));
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 1), Some(0));
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 2), Some(1));
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 3), Some(1));
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 4), None); // out of range
    }

    #[test]
    fn point_for_vertex_none_without_topology() {
        let mesh = make_mixed_mesh();
        assert_eq!(mesh.sub_geometry_for_primitive(SubGeometryKind::Pointset, 0), None);
    }
}
