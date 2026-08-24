// One work-item per (outer, inner) pair; reduces along dim.
__kernel void softmax_kernel(
    __global const float* in,
    __global float* out,
    const int outer,
    const int dim,
    const int inner) {
  const int gid = get_global_id(0);
  const int total = outer * inner;
  if (gid >= total) {
    return;
  }
  const int o = gid / inner;
  const int i = gid - o * inner;
  float m = in[(o * dim) * inner + i];
  for (int d = 1; d < dim; ++d) {
    const float v = in[(o * dim + d) * inner + i];
    m = v > m ? v : m;
  }
  float sum = 0.f;
  for (int d = 0; d < dim; ++d) {
    const float e = exp(in[(o * dim + d) * inner + i] - m);
    out[(o * dim + d) * inner + i] = e;
    sum += e;
  }
  const float inv = 1.f / sum;
  for (int d = 0; d < dim; ++d) {
    out[(o * dim + d) * inner + i] *= inv;
  }
}

__kernel void layernorm_kernel(
    __global const float* in,
    __global const float* weight,
    __global const float* bias,
    __global float* out,
    __global float* mean,
    __global float* rstd,
    const int outer,
    const int n,
    const float eps,
    const int has_weight,
    const int has_bias) {
  const int row = get_global_id(0);
  if (row >= outer) {
    return;
  }
  __global const float* src = in + row * n;
  float mu = 0.f;
  for (int j = 0; j < n; ++j) {
    mu += src[j];
  }
  mu /= (float)n;
  float var = 0.f;
  for (int j = 0; j < n; ++j) {
    const float d = src[j] - mu;
    var += d * d;
  }
  var /= (float)n;
  const float rs = 1.f / sqrt(var + eps);
  mean[row] = mu;
  rstd[row] = rs;
  __global float* dst = out + row * n;
  for (int j = 0; j < n; ++j) {
    float y = (src[j] - mu) * rs;
    if (has_weight) {
      y *= weight[j];
    }
    if (has_bias) {
      y += bias[j];
    }
    dst[j] = y;
  }
}

__kernel void embedding_kernel(
    __global const float* weight,
    __global const int* indices,
    __global float* out,
    const int n_idx,
    const int dim,
    const int vocab) {
  const int gid = get_global_id(0);
  const int row = gid / dim;
  const int col = gid - row * dim;
  if (row >= n_idx) {
    return;
  }
  const int id = indices[row];
  if (id < 0 || id >= vocab) {
    out[row * dim + col] = 0.f;
    return;
  }
  out[row * dim + col] = weight[id * dim + col];
}
