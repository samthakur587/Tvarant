__kernel void fill_kernel(__global float* out, const float value, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = value;
  }
}
