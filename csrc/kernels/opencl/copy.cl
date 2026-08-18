__kernel void copy_kernel(__global float* dst, __global const float* src, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    dst[i] = src[i];
  }
}
