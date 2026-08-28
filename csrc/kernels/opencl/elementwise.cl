__kernel void eye_kernel(__global float* out, const int rows, const int cols, const int diag) {
  const int i = get_global_id(0);
  if (i < diag) {
    out[i * cols + i] = 1.f;
  }
}

__kernel void silu_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    const float x = in[i];
    out[i] = x / (1.f + exp(-x));
  }
}

__kernel void neg_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = -in[i];
  }
}

__kernel void abs_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = fabs(in[i]);
  }
}

__kernel void exp_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = exp(in[i]);
  }
}

__kernel void exp2_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = exp2(in[i]);
  }
}

__kernel void sqrt_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = sqrt(in[i]);
  }
}

__kernel void rsqrt_kernel(__global const float* in, __global float* out, const int n) {
  const int i = get_global_id(0);
  if (i < n) {
    out[i] = rsqrt(in[i]);
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
