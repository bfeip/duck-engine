use bytemuck::bytes_of;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

use crate::scene::{Light, LightType, PositionedCamera, SceneData, MAX_LIGHTS};

use super::batching::ResolvedLight;
use super::bind_group_layouts::BindGroupLayouts;

/// GPU uniform buffer layout for camera data.
///
/// Contains the view-projection matrix and eye position, uploaded to
/// bind group 0 for use by shaders.
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CameraUniform {
    /// Combined view-projection matrix (64 bytes, 4x4 f32).
    view_proj: [[f32; 4]; 4],
    /// Camera eye position in world space (for view direction calculation in PBR).
    eye_position: [f32; 3],
    /// Padding for 16-byte alignment.
    _padding: u32,
}

impl CameraUniform {
    /// Creates a `CameraUniform` from a [`PositionedCamera`].
    pub fn from_positioned_camera(camera: &PositionedCamera) -> Self {
        Self {
            view_proj: camera.build_view_projection_matrix().into(),
            eye_position: [camera.eye.x, camera.eye.y, camera.eye.z],
            _padding: 0,
        }
    }

    /// Creates a new camera uniform initialized to identity matrix and origin.
    pub fn new() -> Self {
        use duck_engine_common::{Matrix4, SquareMatrix};
        Self {
            view_proj: Matrix4::identity().into(),
            eye_position: [0.0, 0.0, 0.0],
            _padding: 0,
        }
    }
}

/// GPU-compatible representation of a single light for shader uniforms.
///
/// This struct is laid out to match WGSL uniform buffer alignment requirements.
/// vec3<f32> types require 16-byte alignment in WGSL, so scalar fields are
/// grouped at the start to pack efficiently.
///
/// # Memory Layout (64 bytes total)
///
/// | Offset | Size | Field          | Notes                              |
/// |--------|------|----------------|------------------------------------|
/// | 0      | 4    | light_type     | 0=Point, 1=Directional, 2=Spot     |
/// | 4      | 4    | range          | 0 = infinite range                 |
/// | 8      | 4    | inner_cone_cos | Spot: cos(inner angle)             |
/// | 12     | 4    | outer_cone_cos | Spot: cos(outer angle)             |
/// | 16     | 12   | position       | Point/Spot: world position         |
/// | 28     | 4    | intensity      | Light intensity multiplier         |
/// | 32     | 12   | direction      | Directional/Spot: light direction  |
/// | 44     | 4    | _padding1      | Alignment padding                  |
/// | 48     | 12   | color          | RGB color                          |
/// | 60     | 4    | _padding2      | Alignment padding                  |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightUniform {
    light_type: u32,
    range: f32,
    inner_cone_cos: f32,
    outer_cone_cos: f32,
    position: [f32; 3],
    intensity: f32,
    direction: [f32; 3],
    _padding1: f32,
    color: [f32; 3],
    _padding2: f32,
}

impl LightUniform {
    pub fn from_resolved_light(resolved: &ResolvedLight) -> Self {
        match &resolved.light {
            Light::Point { color, intensity, range } => LightUniform {
                light_type: LightType::Point as u32,
                range: *range,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: resolved.position,
                intensity: *intensity,
                direction: [0.0, 0.0, 0.0],
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Directional { color, intensity } => LightUniform {
                light_type: LightType::Directional as u32,
                range: 0.0,
                inner_cone_cos: 0.0,
                outer_cone_cos: 0.0,
                position: [0.0, 0.0, 0.0],
                intensity: *intensity,
                direction: resolved.direction,
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
            Light::Spot { color, intensity, range, inner_cone_angle, outer_cone_angle } => LightUniform {
                light_type: LightType::Spot as u32,
                range: *range,
                inner_cone_cos: inner_cone_angle.cos(),
                outer_cone_cos: outer_cone_angle.cos(),
                position: resolved.position,
                intensity: *intensity,
                direction: resolved.direction,
                _padding1: 0.0,
                color: [color.r, color.g, color.b],
                _padding2: 0.0,
            },
        }
    }
}

/// GPU-compatible array of lights with count.
///
/// # Memory Layout (528 bytes total)
///
/// | Offset | Size     | Field       |
/// |--------|----------|-------------|
/// | 0      | 4        | light_count |
/// | 4      | 12       | _padding    |
/// | 16     | 64 * 8   | lights      |
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LightsArrayUniform {
    pub light_count: u32,
    _padding: [u32; 3],
    pub lights: [LightUniform; MAX_LIGHTS],
}

impl LightsArrayUniform {
    pub fn from_resolved_lights(lights: &[ResolvedLight]) -> Self {
        let mut uniform = Self {
            light_count: lights.len().min(MAX_LIGHTS) as u32,
            _padding: [0; 3],
            lights: [bytemuck::Zeroable::zeroed(); MAX_LIGHTS],
        };
        for (i, light) in lights.iter().take(MAX_LIGHTS).enumerate() {
            uniform.lights[i] = LightUniform::from_resolved_light(light);
        }
        uniform
    }
}

/// GPU instance data for one camera: a uniform buffer plus its bind group.
pub(crate) struct CameraBinding {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl CameraBinding {
    /// Create a camera buffer and bind group against the shared camera layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> CameraBinding {
        let buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[CameraUniform::new()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        CameraBinding { buffer, bind_group }
    }
}

/// GPU instance data for lighting uniforms: a buffer plus its bind group.
pub(crate) struct LightsBinding {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub synced_generation: u64,
}

impl LightsBinding {
    /// Create the light buffer and bind group against the shared light layout.
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> LightsBinding {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light buffer"),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            size: std::mem::size_of::<LightsArrayUniform>() as wgpu::BufferAddress,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("Light bind group"),
        });

        LightsBinding {
            buffer,
            bind_group,
            synced_generation: 0,
        }
    }
}

/// The persistent scene-level bind groups owned by the renderer: the camera
/// and the lights array.
///
/// These are uniform buffers the renderer fills every frame, so it owns them
/// outright. IBL (group 3) is deliberately *not* here: that bind group lives in
/// [`IblResources`](crate::ibl), keyed per environment map and rebuilt by GPU
/// preprocessing, so it can only be borrowed per frame — it joins camera/lights
/// in [`SceneBindingRefs`], the per-frame view assembled in
/// [`refs`](Self::refs).
pub(crate) struct SceneBindings {
    camera: CameraBinding,
    lights: LightsBinding,
}

impl SceneBindings {
    pub fn new(device: &wgpu::Device, layouts: &BindGroupLayouts) -> Self {
        Self {
            camera: CameraBinding::new(device, &layouts.camera),
            lights: LightsBinding::new(device, &layouts.light),
        }
    }

    /// Write `camera` into the shared camera uniform buffer.
    pub fn write_camera(&self, queue: &wgpu::Queue, camera: &PositionedCamera) {
        let uniform = [CameraUniform::from_positioned_camera(camera)];
        queue.write_buffer(&self.camera.buffer, 0, bytemuck::cast_slice(&uniform));
    }

    /// Re-upload the lights uniform if the scene's node generation has changed
    /// since the last sync. `resolve` supplies the resolved lights for the
    /// current frame (the renderer gathers them from the scene graph), and is
    /// only called when an upload is actually needed.
    pub fn sync_lights(
        &mut self,
        queue: &wgpu::Queue,
        scene: &SceneData,
        resolve: impl FnOnce() -> LightsArrayUniform,
    ) {
        let node_gen = scene.node_generation();
        if self.lights.synced_generation != node_gen {
            queue.write_buffer(&self.lights.buffer, 0, bytes_of(&resolve()));
            self.lights.synced_generation = node_gen;
        }
    }

    /// Force the next [`sync_lights`](Self::sync_lights) to re-upload, e.g. after
    /// the scene's GPU resources are cleared.
    pub fn invalidate_lights(&mut self) {
        self.lights.synced_generation = 0;
    }

    /// Assemble the per-frame view of all scene-level bind groups: the owned
    /// camera/lights plus the borrowed, optional IBL group for this frame.
    pub fn refs<'a>(&'a self, ibl: Option<&'a wgpu::BindGroup>) -> SceneBindingRefs<'a> {
        SceneBindingRefs {
            camera: &self.camera.bind_group,
            lights: &self.lights.bind_group,
            ibl,
        }
    }
}

/// Per-frame references to the scene-level bind groups (camera, lights, IBL).
///
/// Bundles the three standard scene groups so passes can bind them together.
/// Group indices follow the standard shader ABI ([`crate::abi`]); a pass chooses
/// whether and where to bind each.
#[derive(Clone, Copy)]
pub struct SceneBindingRefs<'a> {
    pub camera: &'a wgpu::BindGroup,
    pub lights: &'a wgpu::BindGroup,
    /// `Some` if there is an active, fully-processed environment map for IBL.
    pub ibl: Option<&'a wgpu::BindGroup>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::common::{RgbaColor, EPSILON};

    fn point_resolved(position: [f32; 3], color: RgbaColor, intensity: f32) -> ResolvedLight {
        ResolvedLight { light: Light::point(color, intensity), position, direction: [0.0, 0.0, -1.0] }
    }

    fn directional_resolved(direction: [f32; 3], color: RgbaColor, intensity: f32) -> ResolvedLight {
        ResolvedLight { light: Light::directional(color, intensity), position: [0.0; 3], direction }
    }

    fn spot_resolved(position: [f32; 3], direction: [f32; 3], color: RgbaColor, intensity: f32, inner: f32, outer: f32) -> ResolvedLight {
        ResolvedLight { light: Light::spot(color, intensity, inner, outer), position, direction }
    }

    #[test]
    fn test_light_uniform_from_point_light() {
        let color = RgbaColor { r: 0.5, g: 0.6, b: 0.7, a: 1.0 };
        let resolved = point_resolved([1.0, 2.0, 3.0], color, 2.0);
        let uniform = LightUniform::from_resolved_light(&resolved);

        assert_eq!(uniform.light_type, LightType::Point as u32);
        assert!((uniform.position[0] - 1.0).abs() < EPSILON);
        assert!((uniform.position[1] - 2.0).abs() < EPSILON);
        assert!((uniform.position[2] - 3.0).abs() < EPSILON);
        assert!((uniform.color[0] - 0.5).abs() < EPSILON);
        assert!((uniform.color[1] - 0.6).abs() < EPSILON);
        assert!((uniform.color[2] - 0.7).abs() < EPSILON);
        assert!((uniform.intensity - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_directional_light() {
        let color = RgbaColor { r: 1.0, g: 1.0, b: 0.9, a: 1.0 };
        let resolved = directional_resolved([0.0, -1.0, 0.0], color, 1.5);
        let uniform = LightUniform::from_resolved_light(&resolved);

        assert_eq!(uniform.light_type, LightType::Directional as u32);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.intensity - 1.5).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_from_spotlight() {
        let color = RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 };
        let inner_angle = std::f32::consts::PI / 6.0;
        let outer_angle = std::f32::consts::PI / 4.0;
        let resolved = spot_resolved([0.0, 5.0, 0.0], [0.0, -1.0, 0.0], color, 3.0, inner_angle, outer_angle);
        let uniform = LightUniform::from_resolved_light(&resolved);

        assert_eq!(uniform.light_type, LightType::Spot as u32);
        assert!((uniform.position[1] - 5.0).abs() < EPSILON);
        assert!((uniform.direction[1] - (-1.0)).abs() < EPSILON);
        assert!((uniform.inner_cone_cos - inner_angle.cos()).abs() < EPSILON);
        assert!((uniform.outer_cone_cos - outer_angle.cos()).abs() < EPSILON);
    }

    #[test]
    fn test_light_uniform_layout() {
        assert_eq!(std::mem::size_of::<LightUniform>(), 64);
        assert_eq!(std::mem::size_of::<LightsArrayUniform>(), 528);
    }

    #[test]
    fn test_lights_array_uniform_from_resolved_lights() {
        let lights = vec![
            point_resolved([1.0, 0.0, 0.0], RgbaColor { r: 1.0, g: 0.0, b: 0.0, a: 1.0 }, 1.0),
            directional_resolved([0.0, -1.0, 0.0], RgbaColor { r: 1.0, g: 1.0, b: 1.0, a: 1.0 }, 1.0),
        ];
        let uniform = LightsArrayUniform::from_resolved_lights(&lights);

        assert_eq!(uniform.light_count, 2);
        assert_eq!(uniform.lights[0].light_type, LightType::Point as u32);
        assert_eq!(uniform.lights[1].light_type, LightType::Directional as u32);
    }
}
