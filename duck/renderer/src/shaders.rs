use crate::render_core::ShaderLibrary;

// Embed shader sources at compile time for WASM compatibility
const ENGINE_MODULES: [(&str, &str); 16] = [
    // Unified configurable surface shader (entry point) + its modules
    ("package::surface", include_str!("shaders/surface.wesl")),
    ("package::material", include_str!("shaders/material.wesl")),
    ("package::lit_surface", include_str!("shaders/lit_surface.wesl")),
    ("package::common", include_str!("shaders/common.wesl")),
    ("package::camera", include_str!("shaders/camera.wesl")),
    ("package::constants", include_str!("shaders/constants.wesl")),
    ("package::lighting", include_str!("shaders/lighting.wesl")),
    ("package::vertex", include_str!("shaders/vertex.wesl")),
    ("package::pbr", include_str!("shaders/pbr.wesl")),
    ("package::normal_mapping", include_str!("shaders/normal_mapping.wesl")),
    // IBL module
    ("package::ibl", include_str!("shaders/ibl.wesl")),
    // Standalone flat-color overlay shader (camera + color uniform)
    ("package::material_color", include_str!("shaders/material_color.wesl")),
    ("package::flat_color", include_str!("shaders/flat_color.wesl")),
    // Screen-space outline modules
    ("package::outline_mask", include_str!("shaders/outline_mask.wesl")),
    ("package::outline_screenspace", include_str!("shaders/outline_screenspace.wesl")),
    // Silhouette edge detection shader
    ("package::silhouette_edges", include_str!("shaders/silhouette_edges.wesl")),
];

/// Build a [`ShaderLibrary`] pre-loaded with all engine shader modules.
fn engine_library() -> ShaderLibrary {
    ShaderLibrary::new(ENGINE_MODULES)
}

/// Compile a user-supplied WESL shader with access to all engine shader modules.
///
/// The user's shader source is compiled with engine modules available for
/// import (`package::common`, `package::camera`, `package::lighting`, etc.).
///
/// The user module is registered as `package::user` so that `package::` imports
/// in the user's WESL resolve correctly. In WESL, `package::` means "the current
/// package", so user code must live in the same namespace as the engine modules.
/// Use specific item imports, e.g.:
/// ```wesl
/// import package::common::{VertexInput, InstanceInput, LIGHT_TYPE_DIRECTIONAL};
/// import package::camera::camera;    // the camera uniform global
/// import package::lighting::lights;  // the lights uniform global
/// ```
pub(crate) fn compile_user_wesl(device: &wgpu::Device, source: &str) -> anyhow::Result<wgpu::ShaderModule> {
    engine_library().compile_adhoc(device, source)
}

/// Typed entry points for the engine's shader variants.
///
/// Thin wrapper over a [`ShaderLibrary`] holding the engine WESL modules: each
/// method maps typed renderer state (material/scene properties, MSAA flags) to
/// the module path + feature flags the library compiles. Variant caching lives
/// in the library.
pub(crate) struct ShaderGenerator {
    library: ShaderLibrary,
}

impl ShaderGenerator {
    /// Create a new ShaderGenerator
    ///
    /// All engine shader sources are embedded at compile time, enabling
    /// compatibility with WASM and other environments without filesystem access.
    pub fn new() -> Self {
        Self { library: engine_library() }
    }

    /// Compile a variant of the unified surface shader from its feature flags.
    ///
    /// Features come from `SurfaceConfig::features`; the caller owns the config
    /// so this stays a thin module/feature mapping.
    pub fn generate_surface_shader(
        &mut self,
        device: &wgpu::Device,
        features: &[(&str, bool)],
        label: &str,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        self.library.compile(device, "package::surface", features, label)
    }

    /// Generate the outline mask shader module.
    /// Renders selected objects to a mask texture.
    pub fn generate_outline_mask_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        self.library
            .compile(device, "package::outline_mask", &[], "Outline Mask Shader")
    }

    /// Generate the screen-space outline shader module.
    pub fn generate_outline_screenspace_shader(
        &mut self,
        device: &wgpu::Device,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        self.library.compile(
            device,
            "package::outline_screenspace",
            &[],
            "Outline Screenspace Shader",
        )
    }

    /// Generate the flat-color shader module.
    /// Renders geometry using a color from the material uniform (group 2) with no lighting.
    pub fn generate_flat_color_shader(&mut self, device: &wgpu::Device) -> anyhow::Result<wgpu::ShaderModule> {
        self.library
            .compile(device, "package::flat_color", &[], "Flat Color Shader")
    }

    /// Generate the silhouette edge detection shader module.
    /// `depth_multisampled = true` produces a variant that reads from a
    /// `texture_depth_multisampled_2d`; `false` reads a `texture_depth_2d`.
    pub fn generate_silhouette_shader(
        &mut self,
        device: &wgpu::Device,
        depth_multisampled: bool,
    ) -> anyhow::Result<wgpu::ShaderModule> {
        let label = if depth_multisampled {
            "Silhouette Edges Shader (MSAA)"
        } else {
            "Silhouette Edges Shader"
        };
        self.library.compile(
            device,
            "package::silhouette_edges",
            &[("depth_multisampled", depth_multisampled)],
            label,
        )
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::engine_library;
    use wgpu::naga;

    /// Compile the standalone shader modules to WGSL and run them through naga's
    /// parser and validator. Both steps are device-free, so this is the only
    /// automated coverage available for shader sources; it catches WESL import
    /// breakage and WGSL type errors that would otherwise surface as a panic on
    /// the first frame that builds the pipeline.
    ///
    /// The surface shader is excluded: its variants are driven by
    /// `SurfaceConfig::features` and are covered by the pipeline cache instead.
    #[test]
    fn standalone_shader_modules_are_valid_wgsl() {
        let mut library = engine_library();
        for (module, features) in [
            ("package::outline_mask", &[][..]),
            ("package::outline_screenspace", &[][..]),
            ("package::flat_color", &[][..]),
            ("package::silhouette_edges", &[("depth_multisampled", false)][..]),
            ("package::silhouette_edges", &[("depth_multisampled", true)][..]),
        ] {
            let wgsl = library
                .compile_to_wgsl(module, features)
                .unwrap_or_else(|e| panic!("{module}: WESL compilation failed: {e:?}"));
            let parsed = naga::front::wgsl::parse_str(&wgsl)
                .unwrap_or_else(|e| panic!("{module}: WGSL parse failed: {}", e.emit_to_string(&wgsl)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&parsed)
            .unwrap_or_else(|e| panic!("{module}: WGSL validation failed: {e:?}"));
        }
    }
}
