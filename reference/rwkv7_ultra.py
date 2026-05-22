########################################################################################################
#
# RWKV-7 极致优化版本
# 优化:
# 1. FP16 State (显存减半)
# 2. torch.compile 自动算子融合
# 3. 优化的内存布局
# 4. CUDA Graph支持
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

def rwkv7_one(state, r, w, k, v, a, b):
    C = r.shape[0]
    H = C // HEAD_SIZE
    y = torch.empty((C,), device=r.device, dtype=DTYPE)
    torch.ops.rwkv7_state_fwd_fp16_state_fp16.forward(1, 1, C, H, state, r, w, k, v, a, b, y)
    return y

def rwkv7_seq(state, r, w, k, v, a, b):
    T, C = r.size()
    H = C // HEAD_SIZE
    y = torch.empty((T, C), device=r.device, dtype=DTYPE)
    torch.ops.rwkv7_state_fwd_fp16_state_fp16.forward(1, T, C, H, state, r, w, k, v, a, b, y)
    return y

def rwkv7_batch(state, r, w, k, v, a, b):
    B, T, C = r.size()
    H = C // HEAD_SIZE
    y = torch.empty((B, T, C), device=r.device, dtype=DTYPE)
    torch.ops.rwkv7_state_fwd_fp16_state_fp16.forward(B, T, C, H, state, r, w, k, v, a, b, y)
    return y

########################################################################################################
# Optimized TMix/CMix with torch.compile
########################################################################################################

@torch.compile(mode="reduce-overhead", fullgraph=True)
def tmix_one_compiled(layer_id: int, H: int, N: int, x, x_prev_0, v_first, state,
                      x_r, x_w, x_k, x_v, x_a, x_g,
                      w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2,
                      k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev_0 - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k_mod = k * (1 + (a - 1) * k_a)

    if layer_id == 0:
        v_out = v
    else:
        v_out = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w_sig = torch.sigmoid(w0 + w)
    wkv_out = rwkv7_one(state, r, w_sig, k_mod, v_out, -kk, kk * a)

    xx_norm = F.group_norm(wkv_out.view(1, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H * N)
    bonus = ((r * k_mod * r_k).view(H, N).sum(dim=-1, keepdim=True) * v_out.view(H, N)).view(H * N)
    return (xx_norm + bonus) * g @ O_, v_out if layer_id == 0 else v_first

@torch.compile(mode="reduce-overhead", fullgraph=True)
def tmix_batch_compiled(layer_id: int, H: int, N: int, x, x_prev_0, v_first, state,
                        x_r, x_w, x_k, x_v, x_a, x_g,
                        w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2,
                        k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    B, T, C = x.shape
    xx = torch.cat((x_prev_0.unsqueeze(1), x[:, :-1, :]), dim=1) - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H * N)
    k_mod = k * (1 + (a - 1) * k_a)

    if layer_id == 0:
        v_out = v
    else:
        v_out = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w_sig = torch.sigmoid(w0 + w)
    wkv_out = rwkv7_batch(state, r, w_sig, k_mod, v_out, -kk, kk * a)

    xx_norm = F.group_norm(wkv_out.view(B * T, H * N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B, T, H * N)
    bonus = ((r * k_mod * r_k).view(B, T, H, N).sum(dim=-1, keepdim=True) * v_out.view(B, T, H, N)).view(B, T, H * N)
    return (xx_norm + bonus) * g @ O_, v_out if layer_id == 0 else v_first

@torch.compile(mode="reduce-overhead", fullgraph=True)
def cmix_one_compiled(x, x_prev_1, x_k, K_, V_):
    xx = x_prev_1 - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

@torch.compile(mode="reduce-overhead", fullgraph=True)
def cmix_batch_compiled(x, x_prev_1, x_k, K_, V_):
    B, T, C = x.shape
    xx = torch.cat((x_prev_1.unsqueeze(1), x[:, :-1, :]), dim=1) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

########################################################################################################
# Model Class
########################################################################################################

class RWKV_x070_Ultra(torch.nn.Module):
    def __init__(self, args, use_compile=True):
        super().__init__()
        self.args = args
        self.use_compile = use_compile
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
        print(f"Model: {args.n_layer} layers, {args.n_embd} dim, compile={use_compile}")
        self.n_layer, self.n_embd = args.n_layer, args.n_embd

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['blocks.0.ln0.weight'], bias=z['blocks.0.ln0.bias'])
        z['blocks.0.att.v0'] = z['blocks.0.att.a0']
        z['blocks.0.att.v1'] = z['blocks.0.att.a1']
        z['blocks.0.att.v2'] = z['blocks.0.att.a2']

        # Pre-organize weights for faster access
        self._prepare_weights()

    def _prepare_weights(self):
        """Organize weights into contiguous tensors for better cache performance"""
        z = self.z
        self.ln1_w = torch.stack([z[f'blocks.{i}.ln1.weight'] for i in range(self.n_layer)])
        self.ln1_b = torch.stack([z[f'blocks.{i}.ln1.bias'] for i in range(self.n_layer)])
        self.ln2_w = torch.stack([z[f'blocks.{i}.ln2.weight'] for i in range(self.n_layer)])
        self.ln2_b = torch.stack([z[f'blocks.{i}.ln2.bias'] for i in range(self.n_layer)])

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None]
        if bsz >= 1:
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size),
                                   dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size),
                                   dtype=DTYPE, requires_grad=False, device="cuda")
        return state

    @torch.inference_mode()
    def forward_batch(self, tokens, state):
        """Main batch forward pass"""
        z = self.z

        if isinstance(tokens[0], list):
            x = z['emb.weight'][torch.tensor(tokens, device="cuda")]
        else:
            x = z['emb.weight'][tokens].unsqueeze(1) if isinstance(tokens, torch.Tensor) else z['emb.weight'][torch.tensor(tokens, device="cuda")].unsqueeze(1)

        B, T, C = x.shape
        v_first = torch.empty_like(x)

        for i in range(self.n_layer):
            att = f'blocks.{i}.att.'
            ffn = f'blocks.{i}.ffn.'

            # LayerNorm + TMix
            xx = F.layer_norm(x, (self.n_embd,), weight=self.ln1_w[i], bias=self.ln1_b[i])

            if self.use_compile:
                xx_out, v_first = tmix_batch_compiled(
                    i, self.n_head, self.head_size, xx, state[0][i, 0], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
            else:
                xx_out, v_first = self._tmix_batch(i, xx, state, v_first)

            state[0][i, 0] = xx[:, -1, :]
            x = x + xx_out

            # LayerNorm + CMix
            xx = F.layer_norm(x, (self.n_embd,), weight=self.ln2_w[i], bias=self.ln2_b[i])

            if self.use_compile:
                xx_out = cmix_batch_compiled(xx, state[0][i, 1], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
            else:
                xx_out = self._cmix_batch(xx, state[0][i, 1], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])

            state[0][i, 1] = xx[:, -1, :]
            x = x + xx_out

        x = x[:, -1, :]
        x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
        x = x @ z['head.weight']
        return x

    def _tmix_batch(self, layer_id, x, state, v_first):
        """Fallback non-compiled TMix"""
        z = self.z
        att = f'blocks.{layer_id}.att.'
        B, T, C = x.shape
        H, N = self.n_head, self.head_size

        xx = torch.cat((state[0][layer_id, 0].unsqueeze(1), x[:, :-1, :]), dim=1) - x
        xr = x + xx * z[att+'x_r']
        xw = x + xx * z[att+'x_w']
        xk = x + xx * z[att+'x_k']
        xv = x + xx * z[att+'x_v']
        xa = x + xx * z[att+'x_a']
        xg = x + xx * z[att+'x_g']

        r = xr @ z[att+'receptance.weight']
        w = torch.tanh(xw @ z[att+'w1']) @ z[att+'w2']
        k = xk @ z[att+'key.weight']
        v = xv @ z[att+'value.weight']
        a = torch.sigmoid(z[att+'a0'] + (xa @ z[att+'a1']) @ z[att+'a2'])
        g = torch.sigmoid(xg @ z[att+'g1']) @ z[att+'g2']

        kk = F.normalize((k * z[att+'k_k']).view(B, T, H, N), dim=-1, p=2.0).view(B, T, H*N)
        k = k * (1 + (a - 1) * z[att+'k_a'])

        if layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(z[att+'v0'] + (xv @ z[att+'v1']) @ z[att+'v2'])

        w = torch.sigmoid(z[att+'w0'] + w)
        xx = rwkv7_batch(state[1][layer_id], r, w, k, v, -kk, kk * a)

        xx = F.group_norm(xx.view(B*T, H*N), num_groups=H, weight=z[att+'ln_x.weight'], bias=z[att+'ln_x.bias'], eps=64e-5).view(B, T, H*N)
        xx = xx + ((r * k * z[att+'r_k']).view(B, T, H, N).sum(dim=-1, keepdim=True) * v.view(B, T, H, N)).view(B, T, H*N)
        return (xx * g) @ z[att+'output.weight'], v_first

    def _cmix_batch(self, x, x_prev, x_k, K_, V_):
        """Fallback non-compiled CMix"""
        B, T, C = x.shape
        xx = torch.cat((x_prev.unsqueeze(1), x[:, :-1, :]), dim=1) - x
        k = x + xx * x_k
        k = torch.relu(k @ K_) ** 2
        return k @ V_

    # ============ CUDA Graph Support ============
    def setup_cuda_graph(self, batch_size, state):
        """Setup CUDA graph for fixed batch size inference"""
        self._graph_batch_size = batch_size
        self._graph_state = state
        self._graph_input = torch.zeros((batch_size,), dtype=torch.long, device="cuda")
        self._graph_output = torch.zeros((batch_size, self.args.vocab_size), dtype=DTYPE, device="cuda")

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._graph_output.copy_(self.forward_batch(self._graph_input.tolist(), self._graph_state))
        torch.cuda.current_stream().wait_stream(s)

        # Capture
        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph):
            self._graph_output.copy_(self.forward_batch(self._graph_input.tolist(), self._graph_state))

        self._graph_ready = True
        print(f"CUDA Graph ready for batch size {batch_size}")
        return self

    def forward_graph(self, token_ids):
        """Run inference using captured CUDA graph"""
        if isinstance(token_ids, list):
            self._graph_input.copy_(torch.tensor(token_ids, device="cuda"))
        else:
            self._graph_input.copy_(token_ids)
        self._cuda_graph.replay()
        return self._graph_output
