__kernel void silu_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    const float x = in[i];
    out[i] = x / (1.f + exp(-x));
  }
}

__kernel void scale_kernel(
    __global const float* in,
    __global float* out,
    const float scale,
    const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = in[i] * scale;
  }
}

__kernel void add_scalar_kernel(
    __global const float* in,
    __global float* out,
    const float scalar,
    const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = in[i] + scalar;
  }
}
