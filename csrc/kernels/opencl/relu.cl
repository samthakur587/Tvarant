__kernel void relu_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = in[i] > 0.f ? in[i] : 0.f;
  }
}
