// Row-major SGEMM: C = act(alpha * A[M,K] @ B + beta * bias_or_C)
// B is [K,N] if trans_b==0, else [N,K].
// Launch with global size rounded up to a multiple of 32 (Tvarant lws=32).
__kernel void gemm_kernel(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta) {
  const int gid = get_global_id(0);
  const int i = gid / N;
  const int j = gid - i * N;
  if (i >= M || j >= N) {
    return;
  }
  float acc = 0.f;
  for (int p = 0; p < K; ++p) {
    acc += a[i * K + p] * b[p * N + j];
  }
  const float prev = (beta == 0.f) ? 0.f : c[i * N + j];
  c[i * N + j] = alpha * acc + beta * prev;
}

__kernel void gemm_bias_act_kernel(
    __global const float* a,
    __global const float* b,
    __global const float* bias,
    __global float* c,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta,
    const int act,
    const int trans_b,
    const int has_bias) {
  const int gid = get_global_id(0);
  const int i = gid / N;
  const int j = gid - i * N;
  if (i >= M || j >= N) {
    return;
  }
  float acc = 0.f;
  for (int p = 0; p < K; ++p) {
    const float bv = trans_b ? b[j * K + p] : b[p * N + j];
    acc += a[i * K + p] * bv;
  }
  float prev = 0.f;
  if (has_bias) {
    prev = bias[j];
  } else if (beta != 0.f) {
    prev = c[i * N + j];
  }
  acc = alpha * acc + beta * prev;
  if (act == 1) {
    acc = acc > 0.f ? acc : 0.f;
  } else if (act == 2) {
    acc = acc / (1.f + exp(-acc));
  }
  c[i * N + j] = acc;
}

__kernel void bmm_kernel(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int B,
    const int M,
    const int N,
    const int K,
    const int trans_b) {
  const int gid = get_global_id(0);
  const int total = B * M * N;
  if (gid >= total) {
    return;
  }
  int tmp = gid;
  const int j = tmp % N;
  tmp /= N;
  const int i = tmp % M;
  tmp /= M;
  const int bi = tmp;
  const int a_off = bi * M * K;
  const int b_off = trans_b ? bi * N * K : bi * K * N;
  const int c_off = bi * M * N;
  float acc = 0.f;
  for (int p = 0; p < K; ++p) {
    const float bv = trans_b ? b[b_off + j * K + p] : b[b_off + p * N + j];
    acc += a[a_off + i * K + p] * bv;
  }
  c[c_off + i * N + j] = acc;
}
