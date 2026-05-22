########################################################################################################
#
# RWKV-7 with CUDA Graph Optimization
# - FP16 State
# - CUDA Graph for reduced kernel launch overhead
#
########################################################################################################

from typing import List
import os
current_path = os.path.dirname(os.path.abspath(__file__))

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch._C._jit_set_autocast_mode(False)

from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method
MyStatic = torch.jit.script

DTYPE = torch.half
HEAD_SIZE = 64

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

load(name="rwkv7_state_fwd_fp16_state_fp16",
     sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16_state_fp16.cpp",
              f"{current_path}/cuda/rwkv7_state_fwd_fp16_state_fp16.cu"],
     is_python_module=False, verbose=True,
     extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization",
                        f"-D_N_={HEAD_SIZE}"] + (["-Xptxas -O3"] if os.name != "nt" else []))

class WKV_7_batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=r.device, dtype=DTYPE)
            torch.ops.rwkv7_state_fwd_fp16_state_fp16.forward(B, T, C, H, state, r, w, k, v, a, b, y)
            return y

def RWKV7_BATCH(state, r, w, k, v, a, b):
    return WKV_7_batch.apply(state, r, w, k, v, a, b)

########################################################################################################
# Model with CUDA Graph
########################################################################################################

class RWKV_x070_Graph(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        args.head_size = HEAD_SIZE
        self.eval()

        self.z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)
        z = self.z
        self.n_head, self.head_size = z['blocks.0.att.r_k'].shape
        args.n_embd = self.n_head * self.head_size

        keys = list(z.keys())
        max_layer = -1
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE, device="cuda")
            if k.endswith('att.r_k'):
                z[k] = z[k].flatten()
            z[k] = z[k].contiguous()
            kk = k.split('.')
            if kk[0] == 'blocks':
                max_layer = max(max_layer, int(kk[1]))

        args.n_layer = max_layer + 1
        print(f"Model: {args.n_layer} layers, {args.n_embd} dim, fp16 state")
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

        self._graph_ready = False

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None]
        state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
        state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size),
                               dtype=DTYPE, requires_grad=False, device="cuda")
        return state

    def setup_cuda_graph(self, batch_size: int):
        """Setup CUDA graph for fixed batch size single-token inference"""
        self._graph_batch_size = batch_size
        self._graph_state = self.generate_zero_state(batch_size)

        # Allocate static buffers
        self._graph_emb_input = torch.zeros((batch_size, 1, self.n_embd), dtype=DTYPE, device="cuda")
        self._graph_output = torch.zeros((batch_size, self.args.vocab_size), dtype=DTYPE, device="cuda")

        # Warmup runs
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                out = self._forward_batch_internal(self._graph_emb_input, self._graph_state)
                self._graph_output.copy_(out)
        torch.cuda.current_stream().wait_stream(s)

        # Capture graph
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            out = self._forward_batch_internal(self._graph_emb_input, self._graph_state)
            self._graph_output.copy_(out)

        self._graph_ready = True
        print(f"CUDA Graph captured for batch size {batch_size}")
        return self

    def forward_graph(self, token_ids: torch.Tensor):
        """Fast inference using CUDA graph. token_ids shape: [batch_size]"""
        assert self._graph_ready, "Call setup_cuda_graph first"
        # Update input embedding
        self._graph_emb_input.copy_(self.z['emb.weight'][token_ids].unsqueeze(1))
        self._cuda_graph.replay()
        return self._graph_output

    def forward_batch(self, tokens, state):
        """Standard batch forward (for prefill or variable batch)"""
        z = self.z
        if isinstance(tokens[0], list):
            x = z['emb.weight'][torch.tensor(tokens, device="cuda")]
        else:
            x = z['emb.weight'][torch.tensor(tokens, device="cuda")].unsqueeze(1)
        return self._forward_batch_internal(x, state)

    # @MyFunction  # Disable JIT for CUDA Graph compatibility
    def _forward_batch_internal(self, x: torch.Tensor, state: List[torch.Tensor]):
        with torch.no_grad():
            z = self.z
            B, T, C = x.shape
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = TMix_batch(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = CMix_batch(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            x = x[:, -1, :]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

########################################################################################################
# TMix / CMix
########################################################################################################

@MyStatic
def TMix_batch(layer_id: int, H: int, N: int, x, x_prev, v_first, state,
               x_r, x_w, x_k, x_v, x_a, x_g,
               w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2,
               k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    B, T, C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:, :-1, :]), dim=1) - x
    x_prev[0] = x[:, -1, :]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H*N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_BATCH(state, r, w, k, v, -kk, kk * a)

    xx = F.group_norm(xx.view(B*T, H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B, T, H*N)
    xx = xx + ((r * k * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def CMix_batch(x, x_prev, x_k, K_, V_):
    B, T, C = x.shape
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:, :-1, :]), dim=1) - x
    x_prev[1] = x[:, -1, :]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_
