__kernel void add_kernel(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const float alpha,
    const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = a[i] + alpha * b[i];
  }
}

__kernel void mul_kernel(
    __global const float* a,
    __global const float* b,
    __global float* out,
    const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = a[i] * b[i];
  }
}
