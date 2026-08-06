use std::f32::consts::PI;

use duck_engine_common::{InnerSpace, Matrix4, Point3, Vector3};

use super::{Mesh, MeshIndex, MeshPrimitive, PrimitiveType, Vertex};

impl Mesh {
    /// Creates a box (cuboid) mesh centered at the origin.
    pub fn upright_box(width: f32, height: f32, depth: f32, primitive_type: PrimitiveType) -> Self {
        struct Face {
            normal: [f32; 3],
            corners: [[f32; 3]; 4],
        }

        let hw = width / 2.0;
        let hh = height / 2.0;
        let hd = depth / 2.0;

        let faces = [
            Face {
                normal: [0.0, 0.0, 1.0],
                corners: [[-hw, -hh, hd], [hw, -hh, hd], [hw, hh, hd], [-hw, hh, hd]],
            },
            Face {
                normal: [0.0, 0.0, -1.0],
                corners: [[hw, -hh, -hd], [-hw, -hh, -hd], [-hw, hh, -hd], [hw, hh, -hd]],
            },
            Face {
                normal: [0.0, 1.0, 0.0],
                corners: [[-hw, hh, hd], [hw, hh, hd], [hw, hh, -hd], [-hw, hh, -hd]],
            },
            Face {
                normal: [0.0, -1.0, 0.0],
                corners: [[-hw, -hh, -hd], [hw, -hh, -hd], [hw, -hh, hd], [-hw, -hh, hd]],
            },
            Face {
                normal: [1.0, 0.0, 0.0],
                corners: [[hw, -hh, hd], [hw, -hh, -hd], [hw, hh, -hd], [hw, hh, hd]],
            },
            Face {
                normal: [-1.0, 0.0, 0.0],
                corners: [[-hw, -hh, -hd], [-hw, -hh, hd], [-hw, hh, hd], [-hw, hh, -hd]],
            },
        ];

        // UV coordinates for each face corner
        let uvs: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        let mut vertices = Vec::with_capacity(3 * 8);

        for face in &faces {
            for (i, pos) in face.corners.iter().enumerate() {
                vertices.push(Vertex {
                    position: *pos,
                    tex_coords: [uvs[i][0], uvs[i][1], 0.0],
                    normal: face.normal,
                });
            }
        }

        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::with_capacity(36);
                for face_idx in 0..6 {
                    let base = (face_idx * 4) as MeshIndex;
                    indices.extend_from_slice(&[
                        base,
                        base + 1,
                        base + 2,
                        base,
                        base + 2,
                        base + 3,
                    ]);
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::with_capacity(48);
                for face_idx in 0..6 {
                    let base = (face_idx * 4) as MeshIndex;
                    // 4 edges per face
                    indices.extend_from_slice(&[
                        base, base + 1,
                        base + 1, base + 2,
                        base + 2, base + 3,
                        base + 3, base,
                    ]);
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a cube mesh centered at the origin.
    ///
    /// Convenience method equivalent to `Mesh::box_mesh(size, size, size)`.
    pub fn cube(size: f32, primitive_type: PrimitiveType) -> Self {
        Self::upright_box(size, size, size, primitive_type)
    }

    /// Creates a UV sphere mesh centered at the origin.
    pub fn sphere(radius: f32, segments: u32, rings: u32, primitive_type: PrimitiveType) -> Self {
        let segments = segments.max(3);
        let rings = rings.max(2);

        let mut vertices = Vec::new();

        // Generate vertices
        for ring in 0..=rings {
            let phi = PI * ring as f32 / rings as f32; // 0 to PI (top to bottom)
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();
            let v = ring as f32 / rings as f32;

            for seg in 0..=segments {
                let theta = 2.0 * PI * seg as f32 / segments as f32; // 0 to 2PI
                let sin_theta = theta.sin();
                let cos_theta = theta.cos();
                let u = seg as f32 / segments as f32;

                let x = sin_phi * cos_theta;
                let y = cos_phi;
                let z = sin_phi * sin_theta;

                vertices.push(Vertex {
                    position: [x * radius, y * radius, z * radius],
                    tex_coords: [u, v, 0.0],
                    normal: [x, y, z],
                });
            }
        }

        // Generate indices
        let verts_per_ring = segments + 1;
        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::new();
                for ring in 0..rings {
                    for seg in 0..segments {
                        let current = ring * verts_per_ring + seg;
                        let next = current + verts_per_ring;

                        // Skip degenerate triangles at poles
                        if ring != 0 {
                            indices.push(current as MeshIndex);
                            indices.push((current + 1) as MeshIndex);
                            indices.push(next as MeshIndex);
                        }
                        if ring != rings - 1 {
                            indices.push((current + 1) as MeshIndex);
                            indices.push((next + 1) as MeshIndex);
                            indices.push(next as MeshIndex);
                        }
                    }
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::new();
                for ring in 0..=rings {
                    for seg in 0..segments {
                        let current = ring * verts_per_ring + seg;
                        // Horizontal line along ring
                        indices.push(current as MeshIndex);
                        indices.push((current + 1) as MeshIndex);
                    }
                }
                for ring in 0..rings {
                    for seg in 0..=segments {
                        let current = ring * verts_per_ring + seg;
                        let next = current + verts_per_ring;
                        // Vertical line between rings
                        indices.push(current as MeshIndex);
                        indices.push(next as MeshIndex);
                    }
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a cylinder mesh centered at the origin, extending along the Y axis.
    pub fn cylinder(
        radius: f32,
        height: f32,
        segments: u32,
        capped: bool,
        primitive_type: PrimitiveType,
    ) -> Self {
        let segments = segments.max(3);
        let half_height = height / 2.0;

        let mut vertices = Vec::new();

        // Side vertices (two rings): bottom at even indices, top at odd
        for i in 0..=segments {
            let theta = 2.0 * PI * i as f32 / segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let u = i as f32 / segments as f32;

            let x = radius * cos_theta;
            let z = radius * sin_theta;
            let normal = [cos_theta, 0.0, sin_theta];

            // Bottom vertex
            vertices.push(Vertex {
                position: [x, -half_height, z],
                tex_coords: [u, 1.0, 0.0],
                normal,
            });
            // Top vertex
            vertices.push(Vertex {
                position: [x, half_height, z],
                tex_coords: [u, 0.0, 0.0],
                normal,
            });
        }

        let side_vertex_count = vertices.len();

        // Caps
        if capped {
            // Top cap center
            vertices.push(Vertex {
                position: [0.0, half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, 1.0, 0.0],
            });

            // Top cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (sin_theta + 1.0) / 2.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                });
            }

            // Bottom cap center
            vertices.push(Vertex {
                position: [0.0, -half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, -1.0, 0.0],
            });

            // Bottom cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, -half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (1.0 - sin_theta) / 2.0, 0.0],
                    normal: [0.0, -1.0, 0.0],
                });
            }
        }

        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::new();
                // Side indices
                for i in 0..segments {
                    let base = i * 2;
                    indices.extend_from_slice(&[
                        base as MeshIndex,
                        (base + 1) as MeshIndex,
                        (base + 3) as MeshIndex,
                        base as MeshIndex,
                        (base + 3) as MeshIndex,
                        (base + 2) as MeshIndex,
                    ]);
                }
                if capped {
                    let top_center_idx = side_vertex_count as MeshIndex;
                    let top_ring_start = top_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.extend_from_slice(&[
                            top_center_idx,
                            top_ring_start + i,
                            top_ring_start + i + 1,
                        ]);
                    }
                    let bottom_center_idx = top_ring_start + segments as MeshIndex + 1;
                    let bottom_ring_start = bottom_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.extend_from_slice(&[
                            bottom_center_idx,
                            bottom_ring_start + i + 1,
                            bottom_ring_start + i,
                        ]);
                    }
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::new();
                // Bottom ring
                for i in 0..segments {
                    let base = i * 2;
                    indices.push(base as MeshIndex);
                    indices.push((base + 2) as MeshIndex);
                }
                // Top ring
                for i in 0..segments {
                    let base = i * 2 + 1;
                    indices.push(base as MeshIndex);
                    indices.push((base + 2) as MeshIndex);
                }
                // Vertical lines
                for i in 0..segments {
                    let base = i * 2;
                    indices.push(base as MeshIndex);
                    indices.push((base + 1) as MeshIndex);
                }
                if capped {
                    let top_center_idx = side_vertex_count as MeshIndex;
                    let top_ring_start = top_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.push(top_center_idx);
                        indices.push(top_ring_start + i);
                    }
                    let bottom_center_idx = top_ring_start + segments as MeshIndex + 1;
                    let bottom_ring_start = bottom_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.push(bottom_center_idx);
                        indices.push(bottom_ring_start + i);
                    }
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a cone mesh centered at the origin, with the apex pointing up (+Y).
    pub fn cone(
        radius: f32,
        height: f32,
        segments: u32,
        capped: bool,
        primitive_type: PrimitiveType,
    ) -> Self {
        let segments = segments.max(3);
        let half_height = height / 2.0;

        let mut vertices = Vec::new();

        // Calculate the normal slope for the cone sides
        let slope = radius / height;
        let normal_y = slope / (1.0 + slope * slope).sqrt();
        let normal_xz = 1.0 / (1.0 + slope * slope).sqrt();

        // Apex vertex (duplicated for each segment for proper normals)
        let apex_y = half_height;

        // Side faces: base at even indices, apex at odd
        for i in 0..=segments {
            let theta = 2.0 * PI * i as f32 / segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();

            let nx = normal_xz * cos_theta;
            let nz = normal_xz * sin_theta;

            // Base vertex
            vertices.push(Vertex {
                position: [radius * cos_theta, -half_height, radius * sin_theta],
                tex_coords: [i as f32 / segments as f32, 1.0, 0.0],
                normal: [nx, normal_y, nz],
            });

            // Apex vertex (with matching normal for this segment)
            vertices.push(Vertex {
                position: [0.0, apex_y, 0.0],
                tex_coords: [i as f32 / segments as f32, 0.0, 0.0],
                normal: [nx, normal_y, nz],
            });
        }

        let side_vertex_count = vertices.len();

        // Bottom cap
        if capped {
            vertices.push(Vertex {
                position: [0.0, -half_height, 0.0],
                tex_coords: [0.5, 0.5, 0.0],
                normal: [0.0, -1.0, 0.0],
            });

            // Cap ring
            for i in 0..=segments {
                let theta = 2.0 * PI * i as f32 / segments as f32;
                let cos_theta = theta.cos();
                let sin_theta = theta.sin();

                vertices.push(Vertex {
                    position: [radius * cos_theta, -half_height, radius * sin_theta],
                    tex_coords: [(cos_theta + 1.0) / 2.0, (1.0 - sin_theta) / 2.0, 0.0],
                    normal: [0.0, -1.0, 0.0],
                });
            }
        }

        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::new();
                for i in 0..segments {
                    let base = i * 2;
                    indices.extend_from_slice(&[
                        base as MeshIndex,
                        (base + 1) as MeshIndex,
                        (base + 2) as MeshIndex,
                    ]);
                }
                if capped {
                    let cap_center_idx = side_vertex_count as MeshIndex;
                    let cap_ring_start = cap_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.extend_from_slice(&[
                            cap_center_idx,
                            cap_ring_start + i + 1,
                            cap_ring_start + i,
                        ]);
                    }
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::new();
                // Base ring
                for i in 0..segments {
                    let base = i * 2;
                    indices.push(base as MeshIndex);
                    indices.push((base + 2) as MeshIndex);
                }
                // Lines from base to apex
                for i in 0..segments {
                    let base = i * 2;
                    indices.push(base as MeshIndex);
                    indices.push((base + 1) as MeshIndex);
                }
                if capped {
                    let cap_center_idx = side_vertex_count as MeshIndex;
                    let cap_ring_start = cap_center_idx + 1;
                    for i in 0..segments as MeshIndex {
                        indices.push(cap_center_idx);
                        indices.push(cap_ring_start + i);
                    }
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a cone mesh with the apex at the given point, extending along a direction.
    ///
    /// This is a convenience wrapper around [`Mesh::cone`] that orients the cone
    /// so the apex is at `apex` and the base extends `height` units along `direction`.
    pub fn cone_directed(
        apex: Point3,
        direction: Vector3,
        radius: f32,
        height: f32,
        segments: u32,
        capped: bool,
        primitive_type: PrimitiveType,
    ) -> Self {
        let dir = direction.normalize();

        // Build rotation: the default cone has apex at +Y and base at -Y.
        // We want apex at `apex` with base extending along `direction`.
        // So -Y (base direction) must map to +dir, meaning +Y maps to -dir.
        let neg_dir = -dir;
        let (right, up) = crate::common::orthonormal_basis(neg_dir);

        // Rotation matrix: maps (X, Y, Z) -> (right, -dir, up)
        #[rustfmt::skip]
        let rotation = Matrix4::new(
            right.x,   neg_dir.x, up.x, 0.0,
            right.y,   neg_dir.y, up.y, 0.0,
            right.z,   neg_dir.z, up.z, 0.0,
            0.0,       0.0,       0.0,  1.0,
        );

        // The default cone has apex at y = +half_height.
        // After rotation, +Y maps to -dir, so apex is at -dir * half_height.
        // Translate so the apex lands at the desired point.
        let half_height = height / 2.0;
        let apex_offset = neg_dir * half_height;
        let translation = Vector3::new(apex.x, apex.y, apex.z) - apex_offset;
        let transform = Matrix4::from_translation(translation) * rotation;

        Self::cone(radius, height, segments, capped, primitive_type).transformed(&transform)
    }

    /// Creates a torus mesh centered at the origin, lying in the XZ plane.
    pub fn torus(
        major_radius: f32,
        minor_radius: f32,
        major_segments: u32,
        minor_segments: u32,
        primitive_type: PrimitiveType,
    ) -> Self {
        let major_segments = major_segments.max(3);
        let minor_segments = minor_segments.max(3);

        let mut vertices = Vec::new();

        for i in 0..=major_segments {
            let theta = 2.0 * PI * i as f32 / major_segments as f32;
            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let u = i as f32 / major_segments as f32;

            for j in 0..=minor_segments {
                let phi = 2.0 * PI * j as f32 / minor_segments as f32;
                let cos_phi = phi.cos();
                let sin_phi = phi.sin();
                let v = j as f32 / minor_segments as f32;

                // Position on the tube surface
                let x = (major_radius + minor_radius * cos_phi) * cos_theta;
                let y = minor_radius * sin_phi;
                let z = (major_radius + minor_radius * cos_phi) * sin_theta;

                // Normal vector (points from tube center to surface)
                let nx = cos_phi * cos_theta;
                let ny = sin_phi;
                let nz = cos_phi * sin_theta;

                vertices.push(Vertex {
                    position: [x, y, z],
                    tex_coords: [u, v, 0.0],
                    normal: [nx, ny, nz],
                });
            }
        }

        // Generate indices
        let verts_per_ring = minor_segments + 1;
        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::new();
                for i in 0..major_segments {
                    for j in 0..minor_segments {
                        let current = i * verts_per_ring + j;
                        let next = (i + 1) * verts_per_ring + j;

                        indices.extend_from_slice(&[
                            current as MeshIndex,
                            (current + 1) as MeshIndex,
                            next as MeshIndex,
                            (current + 1) as MeshIndex,
                            (next + 1) as MeshIndex,
                            next as MeshIndex,
                        ]);
                    }
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::new();
                // Lines along minor circles (tube cross-sections)
                for i in 0..=major_segments {
                    for j in 0..minor_segments {
                        let current = i * verts_per_ring + j;
                        indices.push(current as MeshIndex);
                        indices.push((current + 1) as MeshIndex);
                    }
                }
                // Lines along major circles (around the main ring)
                for i in 0..major_segments {
                    for j in 0..=minor_segments {
                        let current = i * verts_per_ring + j;
                        let next = (i + 1) * verts_per_ring + j;
                        indices.push(current as MeshIndex);
                        indices.push(next as MeshIndex);
                    }
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a flat plane mesh in the XZ plane, centered at the origin, and
    /// divided into segments with lines separating each segment
    pub fn plane(
        width: f32,
        depth: f32,
        width_segments: u32,
        depth_segments: u32,
        primitive_type: PrimitiveType,
    ) -> Self {
        let width_segments = width_segments.max(1);
        let depth_segments = depth_segments.max(1);

        let hw = width / 2.0;
        let hd = depth / 2.0;

        let mut vertices = Vec::new();

        for zi in 0..=depth_segments {
            let v = zi as f32 / depth_segments as f32;
            let z = -hd + v * depth;

            for xi in 0..=width_segments {
                let u = xi as f32 / width_segments as f32;
                let x = -hw + u * width;

                vertices.push(Vertex {
                    position: [x, 0.0, z],
                    tex_coords: [u, v, 0.0],
                    normal: [0.0, 1.0, 0.0],
                });
            }
        }

        let verts_per_row = width_segments + 1;
        let indices = match primitive_type {
            PrimitiveType::TriangleList => {
                let mut indices = Vec::new();
                for zi in 0..depth_segments {
                    for xi in 0..width_segments {
                        let current = zi * verts_per_row + xi;
                        let next = current + verts_per_row;

                        indices.extend_from_slice(&[
                            current as MeshIndex,
                            next as MeshIndex,
                            (current + 1) as MeshIndex,
                            (current + 1) as MeshIndex,
                            next as MeshIndex,
                            (next + 1) as MeshIndex,
                        ]);
                    }
                }
                indices
            }
            PrimitiveType::LineList => {
                let mut indices = Vec::new();
                // Horizontal lines (along width)
                for zi in 0..=depth_segments {
                    for xi in 0..width_segments {
                        let current = zi * verts_per_row + xi;
                        indices.push(current as MeshIndex);
                        indices.push((current + 1) as MeshIndex);
                    }
                }
                // Vertical lines (along depth)
                for zi in 0..depth_segments {
                    for xi in 0..=width_segments {
                        let current = zi * verts_per_row + xi;
                        let next = current + verts_per_row;
                        indices.push(current as MeshIndex);
                        indices.push(next as MeshIndex);
                    }
                }
                indices
            }
            PrimitiveType::PointList => (0..vertices.len() as MeshIndex).collect(),
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a simple quad (two triangles) in the XY plane, facing +Z.
    pub fn quad(width: f32, height: f32, primitive_type: PrimitiveType) -> Self {
        let hw = width / 2.0;
        let hh = height / 2.0;

        let vertices = vec![
            Vertex {
                position: [-hw, -hh, 0.0],
                tex_coords: [0.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [hw, -hh, 0.0],
                tex_coords: [1.0, 1.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [hw, hh, 0.0],
                tex_coords: [1.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-hw, hh, 0.0],
                tex_coords: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
            },
        ];

        let indices = match primitive_type {
            PrimitiveType::TriangleList => vec![0, 1, 2, 0, 2, 3],
            PrimitiveType::LineList => vec![0, 1, 1, 2, 2, 3, 3, 0],
            PrimitiveType::PointList => vec![0, 1, 2, 3],
        };

        Self::from_raw(
            vertices,
            vec![MeshPrimitive {
                primitive_type,
                indices,
            }],
        )
    }

    /// Creates a line segment going from start to end.
    pub fn line(start: Point3, end: Point3) -> Self {
        let vertices = vec![
            Vertex {
                position: start.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: end.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            },
        ];

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices: vec![0, 1],
        }];

        Self::from_raw(vertices, primitives)
    }

    /// Creates a polyline linking each point to the next. If closed, links the final
    /// Point back to the first.
    pub fn polyline(points: Vec<Point3>, closed: bool) -> Self {
        let vertices: Vec<Vertex> = points
            .iter()
            .map(|&p| Vertex {
                position: p.into(),
                tex_coords: [0.0; 3],
                normal: [0.0, 1.0, 0.0],
            })
            .collect();

        let mut indices = Vec::new();
        for i in 0..points.len().saturating_sub(1) {
            indices.push(i as u32);
            indices.push((i + 1) as u32);
        }
        if closed && points.len() > 2 {
            indices.push((points.len() - 1) as u32);
            indices.push(0);
        }

        let primitives = vec![MeshPrimitive {
            primitive_type: PrimitiveType::LineList,
            indices,
        }];

        Mesh::from_raw(vertices, primitives)
    }
}
