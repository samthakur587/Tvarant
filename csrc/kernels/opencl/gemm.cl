// Row-major SGEMM: C = alpha * A[M,K] @ B[K,N] + beta * C[M,N]
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
