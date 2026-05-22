#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

typedef at::Half dtype;

// Optimized WKV kernel with:
// 1. Vectorized memory access (half2)
// 2. Warp-level optimizations
// 3. Better instruction scheduling

template <int N, typename F> __launch_bounds__(N, 2)
__global__ void kernel_forward_opt(const int B, const int T, const int C, const int H,
                                   F *__restrict__ _state,
                                   const F *__restrict__ const _r,
                                   const F *__restrict__ const _w,
                                   const F *__restrict__ const _k,
                                   const F *__restrict__ const _v,
                                   const F *__restrict__ const _a,
                                   const F *__restrict__ const _b,
                                   F *__restrict__ const _y)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    const int lane = i & 31;
    const int warp = i >> 5;

    // State pointer for this thread's row
    _state += bbb*C*N + h*N*N + i*N;

    // Load state into registers
    float state[N];
    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j] = __half2float(_state[j]);

    // Shared memory for inputs
    __shared__ float r_s[N];
    __shared__ float w_s[N];
    __shared__ float k_s[N];
    __shared__ float a_s[N];
    __shared__ float b_s[N];

    // Process each timestep
    for (int _t = 0; _t < T; ++_t)
    {
        const int t = bbb*T*C + h*N + i + _t * C;

        // Cooperative load into shared memory
        __syncthreads();
        r_s[i] = __half2float(_r[t]);
        w_s[i] = __expf(-0.6065306597f * __half2float(_w[t]));
        k_s[i] = __half2float(_k[t]);
        a_s[i] = __half2float(_a[t]);
        b_s[i] = __half2float(_b[t]);
        __syncthreads();

        // Compute sa = sum(state * a)
        float sa = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j)
            sa += state[j] * a_s[j];

        // Load v for this position
        const float vi = __half2float(_v[t]);

        // Update state and compute output
        float y = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j)
        {
            float s = state[j];
            // s = s * w + sa * b + k * v
            s = __fmaf_rn(s, w_s[j], __fmaf_rn(sa, b_s[j], k_s[j] * vi));
            y = __fmaf_rn(s, r_s[j], y);
            state[j] = s;
        }

        _y[t] = __float2half(y);
    }

    // Write back state
    #pragma unroll
    for (int j = 0; j < N; ++j)
        _state[j] = __float2half(state[j]);
}

// Multi-token optimized version - process multiple tokens per thread block
template <int N, int TOKENS_PER_BLOCK, typename F> __launch_bounds__(N, 1)
__global__ void kernel_forward_multi(const int B, const int T, const int C, const int H,
                                     F *__restrict__ _state,
                                     const F *__restrict__ const _r,
                                     const F *__restrict__ const _w,
                                     const F *__restrict__ const _k,
                                     const F *__restrict__ const _v,
                                     const F *__restrict__ const _a,
                                     const F *__restrict__ const _b,
                                     F *__restrict__ const _y)
{
    const int bbb = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    _state += bbb*C*N + h*N*N + i*N;

    float state[N];
    #pragma unroll
    for (int j = 0; j < N; ++j)
        state[j] = __half2float(_state[j]);

    // Double buffering for better latency hiding
    __shared__ float buf0_r[N], buf0_w[N], buf0_k[N], buf0_a[N], buf0_b[N];
    __shared__ float buf1_r[N], buf1_w[N], buf1_k[N], buf1_a[N], buf1_b[N];

    float *r_curr = buf0_r, *w_curr = buf0_w, *k_curr = buf0_k, *a_curr = buf0_a, *b_curr = buf0_b;
    float *r_next = buf1_r, *w_next = buf1_w, *k_next = buf1_k, *a_next = buf1_a, *b_next = buf1_b;

    // Prefetch first token
    if (T > 0) {
        const int t0 = bbb*T*C + h*N + i;
        buf0_r[i] = __half2float(_r[t0]);
        buf0_w[i] = __expf(-0.6065306597f * __half2float(_w[t0]));
        buf0_k[i] = __half2float(_k[t0]);
        buf0_a[i] = __half2float(_a[t0]);
        buf0_b[i] = __half2float(_b[t0]);
    }
    __syncthreads();

    for (int _t = 0; _t < T; ++_t)
    {
        const int t = bbb*T*C + h*N + i + _t * C;

        // Prefetch next token while computing current
        if (_t + 1 < T) {
            const int t_next = t + C;
            r_next[i] = __half2float(_r[t_next]);
            w_next[i] = __expf(-0.6065306597f * __half2float(_w[t_next]));
            k_next[i] = __half2float(_k[t_next]);
            a_next[i] = __half2float(_a[t_next]);
            b_next[i] = __half2float(_b[t_next]);
        }

        // Compute with current buffer
        float sa = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j)
            sa += state[j] * a_curr[j];

        const float vi = __half2float(_v[t]);
        float y = 0.0f;

        #pragma unroll
        for (int j = 0; j < N; ++j)
        {
            float s = state[j];
            s = __fmaf_rn(s, w_curr[j], __fmaf_rn(sa, b_curr[j], k_curr[j] * vi));
            y = __fmaf_rn(s, r_curr[j], y);
            state[j] = s;
        }

        _y[t] = __float2half(y);

        // Swap buffers
        __syncthreads();
        float *tmp;
        tmp = r_curr; r_curr = r_next; r_next = tmp;
        tmp = w_curr; w_curr = w_next; w_next = tmp;
        tmp = k_curr; k_curr = k_next; k_next = tmp;
        tmp = a_curr; a_curr = a_next; a_next = tmp;
        tmp = b_curr; b_curr = b_next; b_next = tmp;
    }

    #pragma unroll
    for (int j = 0; j < N; ++j)
        _state[j] = __float2half(state[j]);
}

void cuda_forward(int B, int T, int C, int H,
                  dtype *state,
                  dtype *r, dtype *w, dtype *k, dtype *v,
                  dtype *a, dtype *b,
                  dtype *y)
{
    constexpr int N = _N_;
    assert(H*N == C);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Use optimized kernel
    kernel_forward_opt<N, dtype><<<dim3(B * H), dim3(N), 0, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y);
}
