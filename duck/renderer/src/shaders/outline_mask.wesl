// outline_mask.wesl
// Renders highlighted geometry into the coverage mask.
//
// The fragment writes 1.0 to both channels; the pipeline's color write mask
// selects which one actually lands, so one shader serves both highlight tiers
// (red = primary, green = secondary).

import package::common::{VertexInput, InstanceInput};
import package::vertex::vertex_position_only;

// Vertex shader - standard transform, no normal expansion
@vertex
fn vs_mask(model: VertexInput, instance: InstanceInput) -> @builtin(position) vec4<f32> {
    return vertex_position_only(model, instance);
}

@fragment
fn fs_mask() -> @location(0) vec2<f32> {
    return vec2<f32>(1.0, 1.0);
}
