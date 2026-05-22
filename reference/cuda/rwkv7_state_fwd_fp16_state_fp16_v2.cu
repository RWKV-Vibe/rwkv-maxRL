#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

typedef at::Half dtype;

// Optimized FP16 state kernel with:
// 1. Vectorized half2 loads/stores for state
// 2. __ldg for cached input reads
// 3. Better register allocation hints

template <int N, typename F>
__global__ void kernel_forward_v2(const int B, const int T, const int C, const int H,
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

    // State pointer for this thread's row
    _state += bbb*C*N + h*N*N + i*N;

    // Load state into registers using vectorized loads
    float state[N];

    // Vectorized load: 2 halfs at a time
    const half2* state_h2 = (const half2*)_state;
    #pragma unroll
    for (int j = 0; j < N/2; ++j) {
        half2 val = state_h2[j];
        state[j*2] = __half2float(val.x);
        state[j*2+1] = __half2float(val.y);
    }

    __shared__ float r_s[N];
    __shared__ float w_s[N];
    __shared__ float k_s[N];
    __shared__ float a_s[N];
    __shared__ float b_s[N];

    for (int _t = 0; _t < T; ++_t)
    {
        const int t = bbb*T*C + h*N + i + _t * C;

        __syncthreads();
        // Use __ldg for cached reads
        r_s[i] = __half2float(__ldg((const __half*)&_r[t]));
        w_s[i] = __expf(-0.6065306597f * __half2float(__ldg((const __half*)&_w[t])));
        k_s[i] = __half2float(__ldg((const __half*)&_k[t]));
        a_s[i] = __half2float(__ldg((const __half*)&_a[t]));
        b_s[i] = __half2float(__ldg((const __half*)&_b[t]));
        __syncthreads();

        float sa = 0.0f;
        #pragma unroll
        for (int j = 0; j < N; ++j)
            sa += state[j] * a_s[j];

        const float vi = __half2float(__ldg((const __half*)&_v[t]));
        float y = 0.0f;

        #pragma unroll
        for (int j = 0; j < N; ++j)
        {
            float s = state[j];
            s = __fmaf_rn(s, w_s[j], __fmaf_rn(sa, b_s[j], k_s[j] * vi));
            y = __fmaf_rn(s, r_s[j], y);
            state[j] = s;
        }

        _y[t] = F(y);
    }

    // Vectorized store: 2 halfs at a time
    half2* state_out_h2 = (half2*)_state;
    #pragma unroll
    for (int j = 0; j < N/2; ++j) {
        half2 val;
        val.x = __float2half(state[j*2]);
        val.y = __float2half(state[j*2+1]);
        state_out_h2[j] = val;
    }
}

// Alternative: keep original simple version but without launch_bounds
template <int N, typename F>
__global__ void kernel_forward_simple(const int B, const int T, const int C, const int H,
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
        state[j] = (float)_state[j];

    __shared__ float r[N], w[N], k[N], a[N], b[N];

    for (int _t = 0; _t < T; ++_t)
    {
        const int t = bbb*T*C + h*N + i + _t * C;
        __syncthreads();
        r[i] = (float)_r[t];
        w[i] = __expf(-0.6065306597f * (float)_w[t]);
        k[i] = (float)_k[t];
        a[i] = (float)_a[t];
        b[i] = (float)_b[t];
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < N; j++)
            sa += a[j] * state[j];

        float vv = (float)_v[t];
        float y = 0;
        #pragma unroll
        for (int j = 0; j < N; j++)
        {
            float& s = state[j];
            s = s * w[j] + k[j] * vv + sa * b[j];
            y += s * r[j];
        }
        _y[t] = F(y);
    }
    #pragma unroll
    for (int j = 0; j < N; j++)
        _state[j] = F(state[j]);
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

    // Use vectorized version
    kernel_forward_v2<N, dtype><<<dim3(B * H), dim3(N), 0, stream>>>(B, T, C, H, state, r, w, k, v, a, b, y);
}
