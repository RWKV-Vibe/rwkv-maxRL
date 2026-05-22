"""
Triton Fused Kernels for RWKV-7
"""

import torch
import triton
import triton.language as tl

########################################################################################################
# Fused Squared ReLU: out = (relu(x @ W1))^2 @ W2
# This fuses activation with matmul for better memory efficiency
########################################################################################################

@triton.jit
def _squared_relu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    """Fused squared relu: out = relu(x)^2"""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    x = tl.maximum(x, 0.0)
    out = x * x
    tl.store(out_ptr + offs, out, mask=mask)

def fused_squared_relu(x: torch.Tensor) -> torch.Tensor:
    """Apply squared relu: (relu(x))^2"""
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _squared_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

########################################################################################################
# Fused Token Shift: out = x + (x_prev - x) * mix
########################################################################################################

@triton.jit
def _token_shift_kernel(
    x_ptr, x_prev_ptr, mix_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask)
    x_prev = tl.load(x_prev_ptr + offs, mask=mask)
    mix = tl.load(mix_ptr + offs % (N // tl.load(x_ptr + 0).dtype.element_size), mask=mask)  # broadcast

    out = x + (x_prev - x) * mix
    tl.store(out_ptr + offs, out, mask=mask)

########################################################################################################
# Fused LayerNorm
########################################################################################################

@triton.jit
def _layer_norm_fwd_kernel(
    X, Y, W, B,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride
    Y += row * stride

    # Compute mean
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += x
    mean = tl.sum(_sum) / N

    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        _var += x_centered * x_centered
    var = tl.sum(_var) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and scale
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + cols, y.to(tl.float16), mask=mask)

def fused_layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Fused layer normalization"""
    assert x.is_contiguous()
    shape = x.shape
    x = x.view(-1, shape[-1])
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    if BLOCK_SIZE > 8192:
        BLOCK_SIZE = 8192
    grid = (M,)
    _layer_norm_fwd_kernel[grid](x, y, weight, bias, N, N, eps, BLOCK_SIZE=BLOCK_SIZE)
    return y.view(shape)

########################################################################################################
# Fused SiLU (Swish): out = x * sigmoid(x)
########################################################################################################

@triton.jit
def _silu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    out = x * sigmoid_x
    tl.store(out_ptr + offs, out.to(tl.float16), mask=mask)

def fused_silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out

########################################################################################################
# Benchmark utilities
########################################################################################################

def benchmark_kernels():
    """Benchmark Triton kernels vs PyTorch"""
    import time

    print("=" * 60)
    print("Triton Kernel Benchmarks")
    print("=" * 60)

    # Test squared relu
    x = torch.randn(1024, 2560, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(10):
        _ = fused_squared_relu(x)
        _ = torch.relu(x) ** 2
    torch.cuda.synchronize()

    # Benchmark
    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        _ = fused_squared_relu(x)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        _ = torch.relu(x) ** 2
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / N * 1000

    print(f"\nSquared ReLU (1024x2560):")
    print(f"  Triton:  {triton_time:.3f} ms")
    print(f"  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Speedup: {pytorch_time/triton_time:.2f}x")

    # Test layer norm
    x = torch.randn(1024, 2560, dtype=torch.float16, device='cuda')
    w = torch.ones(2560, dtype=torch.float16, device='cuda')
    b = torch.zeros(2560, dtype=torch.float16, device='cuda')

    # Warmup
    for _ in range(10):
        _ = fused_layer_norm(x, w, b)
        _ = torch.nn.functional.layer_norm(x, (2560,), w, b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(N):
        _ = fused_layer_norm(x, w, b)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - t0) / N * 1000

    t0 = time.perf_counter()
    for _ in range(N):
        _ = torch.nn.functional.layer_norm(x, (2560,), w, b)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - t0) / N * 1000

    print(f"\nLayerNorm (1024x2560):")
    print(f"  Triton:  {triton_time:.3f} ms")
    print(f"  PyTorch: {pytorch_time:.3f} ms")
    print(f"  Speedup: {pytorch_time/triton_time:.2f}x")

if __name__ == "__main__":
    benchmark_kernels()
