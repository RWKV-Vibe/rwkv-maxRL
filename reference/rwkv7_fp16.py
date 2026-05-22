########################################################################################################
#
# RWKV-7 "Goose" 优化推理版本 (FP16 State Only)
# 优化点:
# 1. State使用fp16存储，显存减半，最大batch提升3倍
# 2. 精简代码，移除冗余
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
# CUDA Kernel Loading
########################################################################################################

from torch.utils.cpp_extension import load

# FP16 State Kernel v2 (optimized with vectorized half2 loads/stores)
load(name="rwkv7_fp16_state_v2",
     sources=[f"{current_path}/cuda/rwkv7_state_fwd_fp16_state_fp16_v2.cpp",
              f"{current_path}/cuda/rwkv7_state_fwd_fp16_state_fp16_v2.cu"],
     is_python_module=False, verbose=True,
     extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "--extra-device-vectorization",
                        f"-D_N_={HEAD_SIZE}"] + (["-Xptxas -O3"] if os.name != "nt" else []))

class WKV_7_one(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            C = r.shape[0]
            H = C // HEAD_SIZE
            y = torch.empty((C,), device=k.device, dtype=DTYPE, requires_grad=False)
            torch.ops.rwkv7_fp16_state_v2.forward(1, 1, C, H, state, r, w, k, v, a, b, y)
            return y

class WKV_7_seq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False)
            torch.ops.rwkv7_fp16_state_v2.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y

class WKV_7_batch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            B, T, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, requires_grad=False)
            torch.ops.rwkv7_fp16_state_v2.forward(B, T, C, H, state, r, w, k, v, a, b, y)
            return y

class WKV_7_batch_one(torch.autograd.Function):
    """Optimized for single token batch: [B, C] input instead of [B, 1, C]"""
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            B, C = r.size()
            H = C // HEAD_SIZE
            y = torch.empty((B, C), device=k.device, dtype=DTYPE, requires_grad=False)
            # Reshape to [B, 1, C] for kernel, then squeeze back
            torch.ops.rwkv7_fp16_state_v2.forward(B, 1, C, H, state,
                r.unsqueeze(1), w.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1),
                a.unsqueeze(1), b.unsqueeze(1), y.unsqueeze(1))
            return y

def RWKV7_ONE(state, r, w, k, v, a, b):
    return WKV_7_one.apply(state, r, w, k, v, a, b)
def RWKV7_SEQ(state, r, w, k, v, a, b):
    return WKV_7_seq.apply(state, r, w, k, v, a, b)
def RWKV7_BATCH(state, r, w, k, v, a, b):
    return WKV_7_batch.apply(state, r, w, k, v, a, b)
def RWKV7_BATCH_ONE(state, r, w, k, v, a, b):
    return WKV_7_batch_one.apply(state, r, w, k, v, a, b)

########################################################################################################
# Model Class
########################################################################################################

class RWKV_x070(MyModule):
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
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
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

    def generate_zero_state(self, bsz):
        args = self.args
        state = [None, None]
        if bsz >= 1:
            # state[0]: x_prev for TMix and CMix, shape [n_layer, 2, bsz, n_embd]
            state[0] = torch.zeros((args.n_layer, 2, bsz, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            # state[1]: WKV state, shape [n_layer, bsz, n_head, head_size, head_size], fp16
            state[1] = torch.zeros((args.n_layer, bsz, args.n_embd // args.head_size, args.head_size, args.head_size),
                                   dtype=DTYPE, requires_grad=False, device="cuda")
        else:
            state[0] = torch.zeros((args.n_layer, 2, args.n_embd), dtype=DTYPE, requires_grad=False, device="cuda")
            state[1] = torch.zeros((args.n_layer, args.n_embd // args.head_size, args.head_size, args.head_size),
                                   dtype=DTYPE, requires_grad=False, device="cuda")
        return state

    def forward(self, idx, state, full_output=False):
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    def forward_batch(self, tokens, state, full_output=False):
        """Batch forward. tokens: list of list of token ids or list of single tokens"""
        assert type(tokens) is list
        lengths = [len(x) if type(x) is list else 1 for x in tokens]

        # All same length -> fast path
        if len(set(lengths)) == 1 and not full_output:
            if type(tokens[0]) is list:
                return self.forward_seq_batch(tokens, state, full_output)
            else:
                # Single token batch
                return self.forward_one_batch(tokens, state)

        # Variable length handling
        bsz = len(tokens)
        pos = [0] * bsz
        out = torch.empty((bsz, self.args.vocab_size), dtype=DTYPE, requires_grad=False, device="cuda")

        while True:
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break
            step = min(lengths[i] - pos[i] for i in active)
            batch_tokens = [tokens[i][pos[i]:pos[i]+step] if type(tokens[i]) is list else [tokens[i]] for i in active]
            batch_state = [state[0][:,:,active], state[1][:,active]]
            new_out = self.forward_seq_batch(batch_tokens, batch_state, False)
            for k, i in enumerate(active):
                out[i] = new_out[k]
                state[0][:,:,i] = batch_state[0][:,:,k]
                state[1][:,i] = batch_state[1][:,k]
                pos[i] += step
        return out

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad():
            z = self.z
            vocab = z['emb.weight'].shape[0]
            if idx < 0 or idx >= vocab:
                idx = 0
            x = z['emb.weight'][idx]
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = TMix_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = CMix_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    @MyFunction
    def forward_one_batch(self, idxs:List[int], state:List[torch.Tensor]):
        """Batch of single tokens - optimized [B, C] path"""
        with torch.no_grad():
            z = self.z
            idx_t = torch.tensor(idxs, device=z['emb.weight'].device, dtype=torch.long)
            idx_t = torch.clamp(idx_t, 0, z['emb.weight'].shape[0] - 1)
            x = z['emb.weight'][idx_t]  # [B, C] - no unsqueeze
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = TMix_batch_one(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = CMix_batch_one(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            # x is already [B, C], no need for x[:, -1, :]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad():
            z = self.z
            idx_t = torch.tensor(idx, device=z['emb.weight'].device, dtype=torch.long)
            idx_t = torch.clamp(idx_t, 0, z['emb.weight'].shape[0] - 1)
            x = z['emb.weight'][idx_t]
            v_first = torch.empty_like(x)

            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])
                xx, v_first = TMix_seq(i, self.n_head, self.head_size, xx, state[0][i], v_first, state[1][i],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'], z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])
                xx = CMix_seq(xx, state[0][i], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

    @MyFunction
    def forward_seq_batch(self, idxs:List[List[int]], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad():
            z = self.z
            idx_t = torch.tensor(idxs, device=z['emb.weight'].device, dtype=torch.long)
            idx_t = torch.clamp(idx_t, 0, z['emb.weight'].shape[0] - 1)
            x = z['emb.weight'][idx_t]
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

            if not full_output: x = x[:,-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x

########################################################################################################
# TMix Functions
########################################################################################################

@MyStatic
def TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g,
             w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2
    kk = F.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_ONE(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(H*N)
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g,
             w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev[0].unsqueeze(0), x[:-1,:])) - x
    x_prev[0] = x[-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_SEQ(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def TMix_batch(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g,
               w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    B,T,C = x.shape
    xx = torch.cat((x_prev[0].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[0] = x[:,-1,:]
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B,T,H,N), dim=-1, p=2.0).view(B,T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_BATCH(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(B*T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B,T,H*N)
    xx = xx + ((r * k * r_k).view(B,T,H,N).sum(dim=-1, keepdim=True) * v.view(B,T,H,N)).view(B,T,H*N)
    return (xx * g) @ O_, v_first

########################################################################################################
# CMix Functions
########################################################################################################

@MyStatic
def CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

@MyStatic
def CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(0), x[:-1,:])) - x
    x_prev[1] = x[-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

@MyStatic
def CMix_batch(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev[1].unsqueeze(1), x[:,:-1,:]), dim=1) - x
    x_prev[1] = x[:,-1,:]
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_

########################################################################################################
# Optimized single-token batch functions (input shape [B, C] instead of [B, 1, C])
########################################################################################################

@MyStatic
def TMix_batch_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g,
                   w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    """TMix for batch of single tokens. x shape: [B, C]"""
    B = x.shape[0]
    # x_prev[0] shape: [B, C], no need for cat with T dim
    xx = x_prev[0] - x
    x_prev[0] = x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = F.normalize((k * k_k).view(B,H,N), dim=-1, p=2.0).view(B,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    w = torch.sigmoid(w0 + w)
    xx = RWKV7_BATCH_ONE(state, r, w, k, v, -kk, kk*a)

    xx = F.group_norm(xx.view(B,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps=64e-5).view(B,H*N)
    xx = xx + ((r * k * r_k).view(B,H,N).sum(dim=-1, keepdim=True) * v.view(B,H,N)).view(B,H*N)
    return (xx * g) @ O_, v_first

@MyStatic
def CMix_batch_one(x, x_prev, x_k, K_, V_):
    """CMix for batch of single tokens. x shape: [B, C]"""
    xx = x_prev[1] - x
    x_prev[1] = x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_
