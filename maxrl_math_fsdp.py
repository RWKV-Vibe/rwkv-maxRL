#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-turn MaxRL training for RWKV.

Code provenance in this file:
- Directly reused modules/helpers:
  - FSDP/backend utilities from `sft_tool_use_fsdp.py`
  - Rollout engine and judge path from `eval_math192_grpo.py`
- Hand-copied logic:
  - `compute_maxrl_outcome_advantage` from
    `maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py`
  - `_ppo_clipped_objective` and `_sync_infer_weights` logic from
    `grpo_math_local.py`
- Original glue in this file:
  - CLI/config wiring
  - single-turn data preprocessing
  - rollout/judge-to-trajectory assembly
  - prompt-group training loop that keeps the SFT FSDP backend structure
"""

import argparse
import contextlib
import functools
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import eval_math192_grpo as eval_base
import sft_tool_use_fsdp as sft_backend
from grpo_math_local import (
    FSDP,
    MemoryEfficientAdamW,
    MixedPrecision,
    ShardingStrategy,
    _dist_is_initialized,
    _ensure_cuda_toolkit_env,
    _get_tensor_by_dotted_name,
    _init_distributed_from_env,
    _parse_cuda_index,
    _resolve_torch_cuda_arch_list,
    enable_full_finetune,
    load_data,
    load_train_model_rwkv7_cuda,
    normalize_model_arg,
    now_str,
    transformer_auto_wrap_policy,
)


@dataclass
class MaxRLConfig:
    train_jsonl: str
    out_dir: str
    model: str
    tokenizer: str
    max_steps: int
    ctx_len: int
    grad_cp: int
    seed: int
    lr: float
    beta1: float
    beta2: float
    optimizer_eps: float
    grad_clip: float
    save_interval: int
    log_interval: int
    micro_batch_size: int
    global_batch_size: int
    memory_efficient_adamw: bool
    max_new_tokens: int
    rollout_batch_size: int
    group_size: int
    temperature: float
    top_p: float
    top_k: int
    use_rapid_sampling: bool
    clip_range: float
    dynamic_prompt_batch: int
    max_sampling_rounds: int
    sync_diag_interval: int
    sync_diag_sample_values: int
    sync_infer_offload_cpu: bool
    eval_jsonl: str
    eval_interval: int
    eval_batch_size: int


# Hand-copied from:
# maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py
def compute_maxrl_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    del norm_adv_by_std_in_grpo
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack([x.to(scores.device) for x in id2score[idx]])
                id2mean[idx] = torch.mean(score_tensor)
                id2std[idx] = torch.std(score_tensor.unsqueeze(0))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2mean[index[i]] + epsilon)

        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


# Hand-copied from:
# grpo_math_local.py::_ppo_clipped_objective
def _ppo_clipped_objective(ratio: torch.Tensor, adv: torch.Tensor, clip_range: float) -> torch.Tensor:
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv
    return torch.where(adv >= 0, torch.minimum(unclipped, clipped), torch.maximum(unclipped, clipped))


def _dist_barrier(train_gpu: int):
    if _dist_is_initialized():
        dist.barrier(device_ids=[int(train_gpu)])


_SYNC_TRANSPOSE_KEYS = ("key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight")


def _project_train_tensor_for_infer(name: str, tensor: torch.Tensor) -> torch.Tensor:
    x = tensor.detach()
    if any(k in name for k in _SYNC_TRANSPOSE_KEYS):
        x = x.t()
    x = x.squeeze()
    if name.endswith("att.r_k"):
        x = x.flatten()
    return x


def _sample_tensor_digest(tensor: Optional[torch.Tensor], sample_values: int) -> Optional[Dict[str, Any]]:
    if not torch.is_tensor(tensor):
        return None
    flat = tensor.reshape(-1)
    if flat.numel() <= 0:
        return {
            "shape": tuple(int(x) for x in tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(flat.numel()),
            "sample_n": 0,
            "sum": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
        }
    n = min(int(sample_values), int(flat.numel()))
    head = flat[:n].float()
    return {
        "shape": tuple(int(x) for x in tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": int(flat.numel()),
        "sample_n": int(n),
        "sum": float(head.sum().item()),
        "abs_mean": float(head.abs().mean().item()),
        "abs_max": float(head.abs().max().item()),
    }


def _sample_tensor_head_cpu(tensor: Optional[torch.Tensor], sample_values: int) -> Optional[torch.Tensor]:
    if not torch.is_tensor(tensor):
        return None
    flat = tensor.reshape(-1)
    if flat.numel() <= 0:
        return torch.empty((0,), dtype=torch.float32)
    n = min(int(sample_values), int(flat.numel()))
    return flat[:n].detach().to(device="cpu", dtype=torch.float32)


def _select_sync_diag_names(base_model: torch.nn.Module, infer_engine, n_layers: int) -> List[str]:
    if infer_engine is None or not hasattr(infer_engine, "infer_model") or not hasattr(infer_engine.infer_model, "z"):
        return []
    infer_keys = set(getattr(infer_engine.infer_model, "z", {}).keys())
    last = max(0, int(n_layers) - 1)
    mid = max(0, int(n_layers) // 2)
    preferred = [
        "head.weight",
        "blocks.0.att.key.weight",
        "blocks.0.att.value.weight",
        "blocks.0.ffn.key.weight",
        f"blocks.{mid}.att.key.weight",
        f"blocks.{mid}.ffn.value.weight",
        f"blocks.{last}.att.output.weight",
        f"blocks.{last}.ffn.receptance.weight",
    ]
    picked: List[str] = []
    for name in preferred:
        if name in infer_keys and _get_tensor_by_dotted_name(base_model, name) is not None and name not in picked:
            picked.append(name)
    if picked:
        return picked[:6]
    fallback = []
    for name in sorted(infer_keys):
        if name.startswith("emb.") or name.endswith("ln0.weight") or name.endswith("ln0.bias"):
            continue
        if _get_tensor_by_dotted_name(base_model, name) is not None:
            fallback.append(name)
        if len(fallback) >= 6:
            break
    return fallback


def _run_sync_diagnostics(
    *,
    step: int,
    base_model: torch.nn.Module,
    infer_engine,
    diag_names: List[str],
    sample_values: int,
    world_size: int,
    logger: sft_backend.Logger,
    is_main: bool,
):
    if not diag_names or infer_engine is None:
        return

    local_payload: Dict[str, Any] = {}
    infer_z = getattr(getattr(infer_engine, "infer_model", None), "z", {})
    for name in diag_names:
        src = _get_tensor_by_dotted_name(base_model, name)
        infer_t = infer_z.get(name) if isinstance(infer_z, dict) else None
        if not torch.is_tensor(src):
            local_payload[name] = {"missing_train": True}
            continue
        proj = _project_train_tensor_for_infer(name, src)
        proj_fp16 = proj.to(dtype=torch.half)
        proj_digest = _sample_tensor_digest(proj_fp16, sample_values=sample_values)
        infer_digest = _sample_tensor_digest(infer_t, sample_values=sample_values)
        item: Dict[str, Any] = {
            "train_proj": proj_digest,
            "infer": infer_digest,
        }
        if proj_digest is not None and infer_digest is not None and proj_digest["sample_n"] > 0 and infer_digest["sample_n"] > 0:
            n = min(int(proj_digest["sample_n"]), int(infer_digest["sample_n"]))
            proj_head = _sample_tensor_head_cpu(proj_fp16, sample_values=n)
            infer_head = _sample_tensor_head_cpu(infer_t, sample_values=n)
            if proj_head is None or infer_head is None:
                local_payload[name] = item
                continue
            diff = (infer_head - proj_head).abs()
            item["infer_diff_max"] = float(diff.max().item())
            item["infer_diff_mean"] = float(diff.mean().item())
        local_payload[name] = item

    gathered = [None for _ in range(world_size)] if world_size > 1 and _dist_is_initialized() else [local_payload]
    if world_size > 1 and _dist_is_initialized():
        dist.all_gather_object(gathered, local_payload)

    if not is_main:
        return

    lines = [f"[sync-diag step {step}] sampled_names={len(diag_names)} sample_values={int(sample_values)}"]
    for name in diag_names:
        rank_entries = []
        for rank_idx, payload in enumerate(gathered):
            if not isinstance(payload, dict):
                continue
            item = payload.get(name)
            if isinstance(item, dict):
                rank_entries.append((rank_idx, item))
        if not rank_entries:
            lines.append(f"  {name}: missing on all ranks")
            continue

        base_item = rank_entries[0][1]
        base_train = base_item.get("train_proj") if isinstance(base_item, dict) else None
        rank_mismatch = False
        infer_bad = False
        parts = []
        for rank_idx, item in rank_entries:
            train_proj = item.get("train_proj")
            infer_diff_max = float(item.get("infer_diff_max", float("nan")))
            infer_diff_mean = float(item.get("infer_diff_mean", float("nan")))
            train_sum = float("nan")
            train_absmax = float("nan")
            if isinstance(base_train, dict) and isinstance(train_proj, dict):
                if (
                    tuple(train_proj.get("shape", ())) != tuple(base_train.get("shape", ()))
                    or abs(float(train_proj.get("sum", 0.0)) - float(base_train.get("sum", 0.0))) > 1e-3
                    or abs(float(train_proj.get("abs_max", 0.0)) - float(base_train.get("abs_max", 0.0))) > 1e-3
                ):
                    rank_mismatch = True
            if isinstance(train_proj, dict):
                train_sum = float(train_proj.get("sum", float("nan")))
                train_absmax = float(train_proj.get("abs_max", float("nan")))
            if math.isfinite(infer_diff_max) and infer_diff_max > 1e-6:
                infer_bad = True
            parts.append(
                f"r{rank_idx}:train_sum={train_sum:.4e} "
                f"train_absmax={train_absmax:.4e} "
                f"infer_diff_max={infer_diff_max:.4e} infer_diff_mean={infer_diff_mean:.4e}"
            )
        flag = "WARN" if (rank_mismatch or infer_bad) else "OK"
        lines.append(f"  [{flag}] {name} | " + " | ".join(parts))

    for line in lines:
        logger.log(line)


# Hand-copied/adapted from:
# grpo_math_local.py::_sync_infer_weights
# WARNING:
# The user explicitly noted that the FSDP train->rollout sync path in
# grpo_math_local.py may contain a severe bug. This function is still copied
# first for fidelity, but should be treated as suspicious until validated.
# In particular, `sync_infer_offload_cpu` is added here to match the original
# grpo_math_local.py default, but this offload path is still UNVERIFIED in this
# script and may contain serious hazards.
@torch.no_grad()
def _sync_infer_weights(
    *,
    step: int,
    train_model: torch.nn.Module,
    base_model: torch.nn.Module,
    infer_engine,
    fsdp_enabled: bool,
    world_size: int,
    train_gpu: int,
    logger: sft_backend.Logger,
    is_main: bool,
    sync_diag_interval: int,
    sync_diag_sample_values: int,
    sync_infer_offload_cpu: bool,
):
    if infer_engine is None:
        return

    if fsdp_enabled and world_size > 1:
        do_sync = True
        if hasattr(infer_engine, "should_sync"):
            do_sync = bool(infer_engine.should_sync(step=step, force=True))
        if _dist_is_initialized():
            flag = [1 if do_sync else 0]
            dist.broadcast_object_list(flag, src=0)
            do_sync = bool(flag[0])
        if not do_sync:
            _dist_barrier(train_gpu)
            return

        if FSDP is None:
            raise RuntimeError("FSDP state-dict helpers are unavailable in current torch build.")

        n_layers = int(getattr(getattr(base_model, "args", None), "n_layer", 0))
        t_sync = time.time()
        diag_names = []
        if int(sync_diag_interval) > 0 and (int(step) == 1 or int(step) % int(sync_diag_interval) == 0):
            diag_names = _select_sync_diag_names(base_model, infer_engine, n_layers=n_layers)

        try:
            ctx = FSDP.summon_full_params(
                train_model,
                recurse=True,
                writeback=False,
                rank0_only=False,
                offload_to_cpu=bool(sync_infer_offload_cpu),
            )
        except TypeError:
            ctx = FSDP.summon_full_params(train_model, recurse=True, writeback=False)

        def _getter(name: str):
            return _get_tensor_by_dotted_name(base_model, name)

        with ctx:
            infer_engine.sync_infer_weights(
                step=step,
                force=True,
                train_tensor_getter=_getter,
                n_layers=n_layers,
            )
            if diag_names:
                _run_sync_diagnostics(
                    step=step,
                    base_model=base_model,
                    infer_engine=infer_engine,
                    diag_names=diag_names,
                    sample_values=int(sync_diag_sample_values),
                    world_size=world_size,
                    logger=logger,
                    is_main=is_main,
                )
        if is_main:
            logger.log(
                f"[sync] train->rollout done: step={step} "
                f"offload_cpu={int(bool(sync_infer_offload_cpu))} dt={time.time() - t_sync:.2f}s"
            )
        _dist_barrier(train_gpu)
        return

    infer_engine.sync_infer_weights(step=step, force=True)


def _normalize_train_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        problem = str(rec.get("problem", "")).strip()
        answer = str(rec.get("answer", rec.get("solution", ""))).strip()
        if not problem or not answer:
            continue
        normalized.append(
            {
                "problem": problem,
                "answer": answer,
            }
        )
    return normalized


def _split_global_count(total: int, world_size: int, rank: int) -> Tuple[int, int, int]:
    base = int(total) // max(1, int(world_size))
    rem = int(total) % max(1, int(world_size))
    start = int(rank) * base + min(int(rank), rem)
    local = base + (1 if int(rank) < rem else 0)
    end = start + local
    return local, start, end


def _group_is_all_zero_or_all_one(group: Dict[str, Any]) -> bool:
    rewards = [float(sample.get("reward", 0.0)) for sample in group.get("samples", [])]
    if not rewards:
        return True
    all_zero = all(r == 0.0 for r in rewards)
    all_one = all(r == 1.0 for r in rewards)
    return all_zero or all_one


def _build_prompt_ids(record: Dict[str, Any], tokenizer, ctx_len: int, max_new_tokens: int) -> List[int]:
    prompt = sft_backend.build_prompt(record["problem"])
    ids = tokenizer.encode(prompt)
    max_prompt_len = int(ctx_len) - int(max_new_tokens) - 4
    max_prompt_len = max(64, max_prompt_len)
    if len(ids) > max_prompt_len:
        ids = ids[-max_prompt_len:]
    return [int(x) for x in ids]


def _append_jsonl(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def _evaluate_math(
    *,
    step: int,
    eval_data: List[Dict[str, Any]],
    infer_engine,
    tokenizer,
    cfg: MaxRLConfig,
    device: str,
    rank: int,
    world_size: int,
    train_gpu: int,
    logger: sft_backend.Logger,
    is_main: bool,
):
    if infer_engine is None:
        raise RuntimeError("Evaluation requires infer_engine, but it is unavailable.")
    if not eval_data:
        raise RuntimeError("Evaluation dataset is empty.")

    idxs = list(range(len(eval_data)))
    if world_size > 1:
        local_pos = list(range(rank, len(idxs), world_size))
        local_idxs = [idxs[i] for i in local_pos]
    else:
        local_idxs = list(idxs)

    ex_list = [eval_data[i] for i in local_idxs]
    prompt_tokens_list = [_build_prompt_ids(ex, tokenizer, cfg.ctx_len, cfg.max_new_tokens) for ex in ex_list]
    gts = [str(ex.get("answer", ex.get("solution", ""))).strip() for ex in ex_list]

    comp_tokens: List[List[int]] = []
    comp_texts: List[str] = []
    truncated: List[bool] = []
    for start in range(0, len(prompt_tokens_list), int(cfg.eval_batch_size)):
        batch_prompts = prompt_tokens_list[start : start + int(cfg.eval_batch_size)]
        batch_seed = int(cfg.seed + step * 1009 + start * 17 + rank * 1000003 + 7)
        batch_tokens, _, batch_texts, batch_trunc = infer_engine.generate_group_parallel(
            prompt_tokens_list=batch_prompts,
            group_size=1,
            max_new_tokens=int(cfg.max_new_tokens),
            temperature=float(cfg.temperature),
            top_p=float(cfg.top_p),
            top_k=int(cfg.top_k),
            stop_on_think_close=False,
            stop_on_user=True,
            stop_on_boxed=True,
            stop_check_every=8,
            stop_check_window=96,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            penalty_decay=0.0,
            use_rollout_cache=False,
            rng_seed=batch_seed,
        )
        comp_tokens.extend(batch_tokens)
        comp_texts.extend(batch_texts)
        truncated.extend(batch_trunc)

    local_correct = 0
    local_trunc = 0
    local_len_sum = 0.0
    details: List[Dict[str, Any]] = []
    for i, ex in enumerate(ex_list):
        judge = eval_base._judge_one(comp_texts[i], gts[i], bool(truncated[i]))
        ok = bool(judge.get("ok", False))
        local_correct += int(ok)
        local_trunc += int(bool(truncated[i]))
        local_len_sum += float(len(comp_tokens[i]))
        details.append(
            {
                "idx": int(local_idxs[i]),
                "problem": ex.get("problem", ""),
                "gt": gts[i],
                "completion": comp_texts[i],
                "truncated": bool(truncated[i]),
                "judge": judge,
                "reward": 1.0 if ok else 0.0,
                "gen_len": len(comp_tokens[i]),
            }
        )

    stats = torch.tensor(
        [
            float(local_correct),
            float(local_trunc),
            float(local_len_sum),
            float(len(ex_list)),
        ],
        dtype=torch.float64,
        device=device,
    )
    gathered_details = [None for _ in range(world_size)] if world_size > 1 and _dist_is_initialized() else [details]
    if world_size > 1 and _dist_is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_gather_object(gathered_details, details)

    if not is_main:
        if world_size > 1 and _dist_is_initialized():
            _dist_barrier(train_gpu)
        return

    total_correct = int(stats[0].item())
    total_trunc = int(stats[1].item())
    total_len_sum = float(stats[2].item())
    total_n = int(stats[3].item())
    merged_details: List[Dict[str, Any]] = []
    for rank_details in gathered_details:
        if isinstance(rank_details, list):
            merged_details.extend(rank_details)
    merged_details.sort(key=lambda x: int(x.get("idx", -1)))

    eval_summary = {
        "time": now_str(),
        "step": int(step),
        "eval_n": int(total_n),
        "judge_acc": float(total_correct / max(1, total_n)),
        "trunc_rate": float(total_trunc / max(1, total_n)),
        "avg_len": float(total_len_sum / max(1, total_n)),
        "eval_temperature": float(cfg.temperature),
        "eval_top_p": float(cfg.top_p),
        "eval_top_k": int(cfg.top_k),
        "eval_max_new_tokens": int(cfg.max_new_tokens),
        "group_size": 1,
        "world_size": int(world_size),
        "eval_jsonl": str(cfg.eval_jsonl),
    }
    eval_outputs = dict(eval_summary)
    eval_outputs["details"] = merged_details
    _append_jsonl(os.path.join(cfg.out_dir, "eval.jsonl"), eval_summary)
    _append_jsonl(os.path.join(cfg.out_dir, "eval_gen_judgements.jsonl"), eval_outputs)
    logger.log(
        f"[EVAL step {step}] judge_acc={eval_summary['judge_acc']:.3f} "
        f"n={total_n} trunc={eval_summary['trunc_rate']:.3f} avg_len={eval_summary['avg_len']:.1f}"
    )
    if world_size > 1 and _dist_is_initialized():
        _dist_barrier(train_gpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="./out_sft_areal_part1234_from_base_bs16_150step_20260322_220805/ckpt_step150_full_model.pth")
    ap.add_argument("--train_jsonl", type=str, default="areal_train_excluding_pass8_part1234_merged.jsonl")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--grad_cp", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--optimizer_eps", type=float, default=1e-18)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--save_interval", type=int, default=50)
    ap.add_argument("--log_interval", type=int, default=1)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--global_batch_size", type=int, default=16)
    ap.add_argument("--memory_efficient_adamw", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--rollout_batch_size", type=int, default=192)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--use_rapid_sampling", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--sync_diag_interval", type=int, default=50)
    ap.add_argument("--sync_diag_sample_values", type=int, default=1024)
    ap.add_argument("--sync_infer_offload_cpu", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_jsonl", type=str, default="areal_test192.jsonl")
    ap.add_argument("--eval_interval", type=int, default=10)
    ap.add_argument("--eval_batch_size", type=int, default=192)

    ap.add_argument(
        "--no-fsdp_no_sync",
        dest="fsdp_no_sync",
        action="store_false",
        default=False,
        help="FSDP no_sync is forcibly disabled; this flag is kept only for CLI compatibility.",
    )
    if any(
        x == "--fsdp_no_sync"
        or x.startswith("--fsdp_no_sync=")
        or x == "--fsdp-no-sync"
        or x.startswith("--fsdp-no-sync=")
        for x in sys.argv[1:]
    ):
        raise SystemExit("Parameter forbidden: fsdp_no_sync is hard-disabled and must remain False.")
    args = ap.parse_args()

    if int(args.micro_batch_size) != 1:
        raise RuntimeError("This MaxRL entry currently expects --micro_batch_size=1 to match the SFT backend.")

    fixed_rollout = {
        "ctx_len": 8192,
        "max_new_tokens": 2048,
        "rollout_batch_size": 192,
        "group_size": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "tokenizer": "reference/rwkv_vocab_v20230424.txt",
        "seed": 42,
        "use_rapid_sampling": False,
    }
    mismatches = []
    for key, expected in fixed_rollout.items():
        actual = getattr(args, key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        raise SystemExit("Rollout parameter mismatch: " + ", ".join(mismatches))

    rank, world_size, local_rank = _init_distributed_from_env()
    is_main = rank == 0

    def rank0_print(msg: str):
        if is_main:
            print(msg, flush=True)

    auto_out_dir = False
    if args.out_dir is None or str(args.out_dir).strip() == "":
        auto_out_dir = True
        if is_main:
            args.out_dir = f"out_maxrl_math_{now_str()}"
    if world_size > 1 and _dist_is_initialized() and auto_out_dir:
        out_obj = [args.out_dir if is_main else None]
        dist.broadcast_object_list(out_obj, src=0)
        args.out_dir = str(out_obj[0])

    os.makedirs(args.out_dir, exist_ok=True)
    logger = sft_backend.Logger(args.out_dir, rank=rank, is_main=is_main)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    n_cuda = torch.cuda.device_count()
    if n_cuda < 1:
        raise RuntimeError("CUDA is required for this script, but no CUDA device is available.")

    available_gpu_indices = list(range(n_cuda))
    if world_size > len(available_gpu_indices):
        raise RuntimeError(
            f"WORLD_SIZE={world_size} exceeds available training GPUs={len(available_gpu_indices)} "
            f"(train_gpus={available_gpu_indices})."
        )

    active_train_gpus = available_gpu_indices[: max(1, world_size)]
    if world_size > 1:
        if local_rank >= world_size:
            raise RuntimeError(f"LOCAL_RANK={local_rank} out of range for WORLD_SIZE={world_size}.")
        train_gpu = int(local_rank)
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        rank0_print(
            f"[GPU] distributed mode: world_size={world_size} "
            f"train_gpus={active_train_gpus} (rank{rank} -> cuda:{train_gpu})"
        )
    else:
        train_gpu = active_train_gpus[0]
        device = f"cuda:{train_gpu}"
        torch.cuda.set_device(train_gpu)
        rank0_print(f"[GPU] single-process mode: total={n_cuda}, train_gpu={train_gpu}")

    os.environ["RWKV_HEAD_SIZE_A"] = "64"
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(int(args.ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    _ensure_cuda_toolkit_env()
    safe_arch = _resolve_torch_cuda_arch_list(device)
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if safe_arch:
        if arch_env and "10.0" in arch_env:
            os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch
            rank0_print(f"[CUDA-ARCH] override TORCH_CUDA_ARCH_LIST={arch_env} -> {safe_arch}")
        elif not arch_env:
            os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch
            rank0_print(f"[CUDA-ARCH] set TORCH_CUDA_ARCH_LIST={safe_arch}")

    cfg = MaxRLConfig(
        train_jsonl=str(args.train_jsonl),
        out_dir=str(args.out_dir),
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        max_steps=int(args.max_steps),
        ctx_len=int(args.ctx_len),
        grad_cp=int(args.grad_cp),
        seed=int(args.seed),
        lr=float(args.lr),
        beta1=float(args.beta1),
        beta2=float(args.beta2),
        optimizer_eps=float(args.optimizer_eps),
        grad_clip=float(args.grad_clip),
        save_interval=int(args.save_interval),
        log_interval=int(args.log_interval),
        micro_batch_size=int(args.micro_batch_size),
        global_batch_size=int(args.global_batch_size),
        memory_efficient_adamw=bool(args.memory_efficient_adamw),
        max_new_tokens=int(args.max_new_tokens),
        rollout_batch_size=int(args.rollout_batch_size),
        group_size=int(args.group_size),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        use_rapid_sampling=bool(args.use_rapid_sampling),
        clip_range=float(args.clip_range),
        dynamic_prompt_batch=24,
        max_sampling_rounds=10,
        sync_diag_interval=int(args.sync_diag_interval),
        sync_diag_sample_values=int(args.sync_diag_sample_values),
        sync_infer_offload_cpu=bool(args.sync_infer_offload_cpu),
        eval_jsonl=str(args.eval_jsonl),
        eval_interval=int(args.eval_interval),
        eval_batch_size=int(args.eval_batch_size),
    )
    if cfg.global_batch_size <= 0:
        raise RuntimeError("--global_batch_size must be > 0")
    if cfg.global_batch_size % max(1, world_size) != 0:
        raise RuntimeError(
            f"--global_batch_size={cfg.global_batch_size} must be divisible by world_size={world_size}."
        )
    local_update_group_target = cfg.global_batch_size // max(1, world_size)
    if local_update_group_target <= 0:
        raise RuntimeError("local update group target resolved to <= 0")

    if is_main:
        with open(os.path.join(args.out_dir, "maxrl_config.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    data_raw = load_data(args.train_jsonl)
    data = _normalize_train_records(data_raw)
    if not data:
        raise RuntimeError("No valid train records after preprocessing.")
    eval_raw = load_data(args.eval_jsonl)
    eval_data = _normalize_train_records(eval_raw)
    if not eval_data:
        raise RuntimeError("No valid eval records after preprocessing.")
    data_stats = {
        "loaded_records": len(data_raw),
        "valid_records": len(data),
        "eval_loaded_records": len(eval_raw),
        "eval_valid_records": len(eval_data),
        "first_problem_preview": data[0]["problem"][:200],
        "first_answer_preview": data[0]["answer"][:200],
    }
    if is_main:
        with open(os.path.join(args.out_dir, "data_stats.json"), "w", encoding="utf-8") as f:
            json.dump(data_stats, f, ensure_ascii=False, indent=2)

    logger.log(
        f"Data loaded: valid={data_stats['valid_records']}/{data_stats['loaded_records']} "
        f"eval={data_stats['eval_valid_records']}/{data_stats['eval_loaded_records']} "
        f"target_valid_groups={cfg.global_batch_size} local_update_group_target={local_update_group_target} "
        f"group_size={cfg.group_size}"
    )

    base_name, pth_path = normalize_model_arg(args.model)
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Cannot find model pth: {pth_path}")

    train_idx = _parse_cuda_index(device)
    if train_idx is not None:
        torch.cuda.set_device(train_idx)
    train_model_raw, _ = load_train_model_rwkv7_cuda(
        pth_path,
        device=device,
        ctx_len=int(args.ctx_len),
        grad_cp=int(args.grad_cp),
    )
    if train_idx is not None:
        torch.cuda.set_device(train_idx)
    trainable = enable_full_finetune(train_model_raw)
    if trainable <= 0:
        raise RuntimeError("No trainable parameters found.")
    rank0_print(f"Trainable parameters (full): {trainable}")

    train_model = train_model_raw
    if world_size > 1:
        if FSDP is None or ShardingStrategy is None or MixedPrecision is None or transformer_auto_wrap_policy is None:
            raise RuntimeError("Current torch build does not provide FSDP.")
        fsdp_force_fp32 = os.environ.get("RWKV_FSDP_FORCE_FP32", "0") == "1"
        fsdp_enable_mp = os.environ.get("RWKV_FSDP_ENABLE_MP", "0") == "1"
        if fsdp_force_fp32:
            fsdp_mp = None
            rank0_print("[GPU] FSDP mixed_precision disabled by RWKV_FSDP_FORCE_FP32=1")
        elif fsdp_enable_mp:
            fsdp_mp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            rank0_print("[GPU] FSDP mixed_precision enabled by RWKV_FSDP_ENABLE_MP=1")
        else:
            fsdp_mp = None
            rank0_print("[GPU] FSDP mixed_precision disabled by default (match sft_tool_use_fsdp.py)")
        auto_wrap_policy = None
        if hasattr(train_model_raw, "blocks") and len(getattr(train_model_raw, "blocks", [])) > 0:
            block_cls = type(train_model_raw.blocks[0])
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={block_cls},
            )
        train_model = FSDP(
            train_model_raw,
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
            sync_module_states=False,
            mixed_precision=fsdp_mp,
            auto_wrap_policy=auto_wrap_policy,
        )
        rank0_print(f"[GPU] FSDP enabled across {world_size} training ranks.")

    trainable_params = [p for p in train_model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable params found after wrapping.")

    if args.memory_efficient_adamw:
        opt = MemoryEfficientAdamW(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
            enabled=True,
        )
        logger.log("Optimizer: MemoryEfficientAdamW (CPU-offloaded states)")
    else:
        opt = torch.optim.Adam(
            trainable_params,
            lr=float(args.lr),
            betas=(float(args.beta1), float(args.beta2)),
            eps=float(args.optimizer_eps),
            weight_decay=0.0,
        )
        logger.log("Optimizer: Adam (no weight_decay)")

    fsdp_enabled = (FSDP is not None) and isinstance(train_model, FSDP)
    iterator = sft_backend.ShardedIterator(data, rank=rank, world_size=world_size, seed=int(args.seed))

    tok, encode, _decode, infer_engine, _ = eval_base._build_engine(
        model_path=args.model,
        device=device,
        ctx_len=int(args.ctx_len),
        tokenizer_path=args.tokenizer,
        use_rapid_sampling=bool(args.use_rapid_sampling),
    )

    logger.log(
        f"MaxRL train begin: steps={args.max_steps} global_batch={args.global_batch_size} "
        f"local_update_group_target={local_update_group_target} group_size={cfg.group_size} "
        f"rollout_batch_size={cfg.rollout_batch_size} dynamic_prompt_batch={cfg.dynamic_prompt_batch} "
        f"max_sampling_rounds={cfg.max_sampling_rounds} sync_diag_interval={cfg.sync_diag_interval} "
        f"sync_diag_sample_values={cfg.sync_diag_sample_values} eval_interval={cfg.eval_interval} "
        f"eval_batch_size={cfg.eval_batch_size} lr={args.lr:.2e}"
    )
    logger.log(
        "Rollout path: eval_math192_grpo.py::_build_engine + FP16BatchInference.generate_group_parallel "
        "(single-turn, no tool use, no multi-turn)."
    )
    logger.log(
        "MaxRL path: hand-copied compute_maxrl_outcome_advantage from "
        "maxrl-main/maxrl-main/verl/trainer/ppo/core_algos.py."
    )
    logger.log(
        "WARNING: FSDP train->rollout sync is copied from grpo_math_local.py as requested, "
        "but this path may contain a severe bug and is not trusted yet."
    )
    logger.log(
        "WARNING: sync_infer_offload_cpu is enabled to match grpo_math_local.py default, "
        "but this CPU-offload sync path is UNVERIFIED in this script and may contain serious hazards."
    )
    logger.log(
        "Precision path: train forward uses torch.autocast(cuda, bfloat16); "
        "cross-entropy is computed in float32; optimizer updates remain float32-compatible. "
        "FSDP mixed_precision stays disabled by default to match sft_tool_use_fsdp.py "
        "unless RWKV_FSDP_ENABLE_MP=1 is set."
    )
    logger.log(f"Allocator path: PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}")
    p0 = next(train_model.parameters())
    logger.log(f"model dtype={p0.dtype}, device={p0.device}")
    train_model.train()

    try:
        opt.zero_grad(set_to_none=True)
        for step in range(1, int(args.max_steps) + 1):
            step_start = time.time()
            _sync_infer_weights(
                step=step,
                train_model=train_model,
                base_model=train_model_raw,
                infer_engine=infer_engine,
                fsdp_enabled=fsdp_enabled,
                world_size=world_size,
                train_gpu=train_gpu,
                logger=logger,
                is_main=is_main,
                sync_diag_interval=int(cfg.sync_diag_interval),
                sync_diag_sample_values=int(cfg.sync_diag_sample_values),
                sync_infer_offload_cpu=bool(cfg.sync_infer_offload_cpu),
            )

            selected_groups_global = None
            prompts_per_batch = max(1, int(cfg.rollout_batch_size) // max(1, int(cfg.group_size)))
            local_round_prompt_n, _, _ = _split_global_count(int(cfg.dynamic_prompt_batch), world_size, rank)
            accumulated_valid_groups = [] if is_main else None

            for rollout_round in range(1, int(cfg.max_sampling_rounds) + 1):
                prompt_records = [iterator.next() for _ in range(local_round_prompt_n)]
                round_prompt_groups: List[Dict[str, Any]] = []
                for rec in prompt_records:
                    round_prompt_groups.append(
                        {
                            "problem": rec["problem"],
                            "answer": rec["answer"],
                            "prompt_ids": _build_prompt_ids(rec, tok, cfg.ctx_len, cfg.max_new_tokens),
                            "samples": [],
                        }
                    )

                for start in range(0, len(round_prompt_groups), prompts_per_batch):
                    chunk = round_prompt_groups[start : start + prompts_per_batch]
                    batch_prompts = [g["prompt_ids"] for g in chunk]
                    batch_seed = int(
                        cfg.seed
                        + step * 1009
                        + rollout_round * 10007
                        + start * 1009
                        + rank * 1000003
                    )

                    comp_tokens, old_logps, comp_texts, truncated = infer_engine.generate_group_parallel(
                        prompt_tokens_list=batch_prompts,
                        group_size=int(cfg.group_size),
                        max_new_tokens=int(cfg.max_new_tokens),
                        temperature=float(cfg.temperature),
                        top_p=float(cfg.top_p),
                        top_k=int(cfg.top_k),
                        stop_on_think_close=False,
                        stop_on_user=True,
                        stop_on_boxed=True,
                        stop_check_every=8,
                        stop_check_window=96,
                        presence_penalty=0.0,
                        frequency_penalty=0.0,
                        penalty_decay=0.0,
                        use_rollout_cache=False,
                        rng_seed=batch_seed,
                    )

                    for i, group in enumerate(chunk):
                        gt = group["answer"]
                        for gi in range(int(cfg.group_size)):
                            idx = i * int(cfg.group_size) + gi
                            judge = eval_base._judge_one(comp_texts[idx], gt, bool(truncated[idx]))
                            reward = 1.0 if bool(judge.get("ok", False)) else 0.0
                            group["samples"].append(
                                {
                                    "comp_tokens": [int(x) for x in comp_tokens[idx]],
                                    "old_logps": [float(x) for x in old_logps[idx]],
                                    "completion": comp_texts[idx],
                                    "truncated": bool(truncated[idx]),
                                    "judge": judge,
                                    "reward": float(reward),
                                    "adv": 0.0,
                                }
                            )

                local_valid_groups = [group for group in round_prompt_groups if not _group_is_all_zero_or_all_one(group)]
                local_all0_or_all1 = int(len(round_prompt_groups) - len(local_valid_groups))

                if world_size > 1 and _dist_is_initialized():
                    gathered_valid_groups: List[Optional[List[Dict[str, Any]]]] = [None for _ in range(world_size)]
                    dist.all_gather_object(gathered_valid_groups, local_valid_groups)
                else:
                    gathered_valid_groups = [local_valid_groups]

                status_obj = None
                if is_main:
                    for gathered_part in gathered_valid_groups:
                        if isinstance(gathered_part, list) and gathered_part:
                            accumulated_valid_groups.extend(gathered_part)
                    valid_total = len(accumulated_valid_groups)
                    sampled_total = int(rollout_round * cfg.dynamic_prompt_batch)
                    logger.log(
                        f"[sampling step {step}] round={rollout_round}/{cfg.max_sampling_rounds} "
                        f"sampled_global={sampled_total} valid_global={valid_total}/{cfg.global_batch_size} "
                        f"round_local_valid(rank0)={len(local_valid_groups)} round_local_all0or1(rank0)={local_all0_or_all1}"
                    )
                    if valid_total >= int(cfg.global_batch_size):
                        selected_groups_global = accumulated_valid_groups[: int(cfg.global_batch_size)]
                        status_obj = {
                            "done": True,
                            "ok": True,
                            "message": "",
                            "selected_groups": selected_groups_global,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }
                    elif rollout_round >= int(cfg.max_sampling_rounds):
                        status_obj = {
                            "done": True,
                            "ok": False,
                            "message": (
                                f"Dynamic sampling exhausted at step={step}: "
                                f"valid_groups={valid_total} < target={cfg.global_batch_size} "
                                f"after max_rounds={cfg.max_sampling_rounds}. "
                                "Treat as training collapse and terminate."
                            ),
                            "selected_groups": None,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }
                    else:
                        status_obj = {
                            "done": False,
                            "ok": True,
                            "message": "",
                            "selected_groups": None,
                            "valid_total": valid_total,
                            "rollout_round": rollout_round,
                        }

                if world_size > 1 and _dist_is_initialized():
                    status_box = [status_obj if is_main else None]
                    dist.broadcast_object_list(status_box, src=0)
                    status_obj = status_box[0]

                if not bool(status_obj["ok"]):
                    raise RuntimeError(str(status_obj["message"]))
                if bool(status_obj["done"]):
                    selected_groups_global = status_obj["selected_groups"]
                    break

            if selected_groups_global is None:
                raise RuntimeError(f"Dynamic sampling failed to produce selected_groups_global at step={step}.")

            _, shard_start, shard_end = _split_global_count(int(cfg.global_batch_size), world_size, rank)
            prompt_groups = selected_groups_global[shard_start:shard_end]
            if len(prompt_groups) != int(local_update_group_target):
                raise RuntimeError(
                    f"Rank {rank} received {len(prompt_groups)} selected groups, "
                    f"expected {local_update_group_target}."
                )

            flat_rewards = []
            flat_uid = []
            flat_refs = []
            for pi, group in enumerate(prompt_groups):
                for sample in group["samples"]:
                    flat_rewards.append([float(sample["reward"])])
                    flat_uid.append(pi)
                    flat_refs.append(sample)

            if not flat_refs:
                raise RuntimeError(f"No rollout samples collected at step={step}.")

            reward_tensor = torch.tensor(flat_rewards, dtype=torch.float32, device=device)
            reward_mask = torch.ones_like(reward_tensor, dtype=torch.float32, device=device)
            adv_tensor, _ = compute_maxrl_outcome_advantage(
                token_level_rewards=reward_tensor,
                response_mask=reward_mask,
                index=np.asarray(flat_uid),
            )
            for i, sample in enumerate(flat_refs):
                sample["adv"] = float(adv_tensor[i, 0].detach().item())

            local_target_tokens = 0
            sample_cnt_local = 0
            sample_correct_local = 0
            prompt_pass_local = 0
            reward_sum_local = 0.0
            adv_abs_sum_local = 0.0
            traj_cnt_local = 0
            for group in prompt_groups:
                group_any_ok = False
                for sample in group["samples"]:
                    keep = min(len(sample["comp_tokens"]), len(sample["old_logps"]))
                    if keep <= 0:
                        continue
                    local_target_tokens += int(keep)
                    sample_cnt_local += 1
                    ok = bool(sample["judge"].get("ok", False))
                    sample_correct_local += int(ok)
                    group_any_ok = group_any_ok or ok
                    reward_sum_local += float(sample["reward"])
                    adv_abs_sum_local += abs(float(sample["adv"]))
                    traj_cnt_local += 1
                prompt_pass_local += int(group_any_ok)

            global_target_tokens = sft_backend._all_reduce_scalar(
                local_target_tokens,
                device=device,
                op=dist.ReduceOp.SUM,
            )
            if global_target_tokens <= 0:
                raise RuntimeError(f"Resolved global_target_tokens={global_target_tokens} at step={step}.")

            opt.zero_grad(set_to_none=True)
            step_policy_local = 0.0

            for group in prompt_groups:
                valid_samples = []
                for sample in group["samples"]:
                    keep = min(len(sample["comp_tokens"]), len(sample["old_logps"]))
                    if keep <= 0:
                        continue
                    valid_samples.append(
                        {
                            "full_tokens": group["prompt_ids"] + sample["comp_tokens"][:keep],
                            "prompt_len": len(group["prompt_ids"]),
                            "old_logps": sample["old_logps"][:keep],
                            "adv": float(sample["adv"]),
                        }
                    )
                if not valid_samples:
                    continue

                for sample in valid_samples:
                    padded, lens = _pad_batch_local([sample["full_tokens"]], device=device, pad_id=0)
                    inp = padded[:, :-1].contiguous()
                    tgt = padded[:, 1:].contiguous()

                    with contextlib.nullcontext():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = train_model(inp)
                        if torch.is_tensor(logits) and logits.dim() == 2:
                            logits = logits.unsqueeze(0)
                        picked = -F.cross_entropy(
                            logits.float().reshape(-1, logits.size(-1)),
                            tgt.reshape(-1),
                            reduction="none",
                        ).reshape_as(tgt)

                        prompt_len = int(sample["prompt_len"])
                        full_len = int(lens[0])
                        comp_len = int(len(sample["old_logps"]))
                        start = max(0, prompt_len - 1)
                        end = max(start, min(full_len - 1, start + comp_len))
                        if end <= start:
                            continue

                        new_lp = picked[0, start:end].float()
                        old_lp = torch.tensor(sample["old_logps"], dtype=torch.float32, device=device)
                        if old_lp.numel() != new_lp.numel():
                            keep = min(old_lp.numel(), new_lp.numel())
                            if keep <= 0:
                                continue
                            new_lp = new_lp[:keep]
                            old_lp = old_lp[:keep]

                        adv = torch.full_like(new_lp, float(sample["adv"]))
                        ratio = torch.exp((new_lp - old_lp).clamp(min=-20.0, max=20.0))
                        obj = _ppo_clipped_objective(ratio, adv, clip_range=float(cfg.clip_range))
                        policy_sum = -obj.sum()
                        loss = policy_sum / float(global_target_tokens)
                        loss.backward()
                        step_policy_local += float(policy_sum.detach().item())

            raw_grad_sq_local = 0.0
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        raw_grad_sq_local += float((g.norm(2) ** 2).item())
            raw_grad_sq = sft_backend._all_reduce_scalar(raw_grad_sq_local, device=device, op=dist.ReduceOp.SUM)
            raw_grad = math.sqrt(max(0.0, raw_grad_sq))

            if float(args.grad_clip) > 0:
                if fsdp_enabled and hasattr(train_model, "clip_grad_norm_"):
                    train_model.clip_grad_norm_(float(args.grad_clip))
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))

            grad_sq_local = 0.0
            with torch.no_grad():
                for p in trainable_params:
                    if p.grad is not None:
                        g = p.grad.detach().float()
                        grad_sq_local += float((g.norm(2) ** 2).item())
            grad_sq = sft_backend._all_reduce_scalar(grad_sq_local, device=device, op=dist.ReduceOp.SUM)
            grad_norm = math.sqrt(max(0.0, grad_sq))

            opt.step()
            opt.zero_grad(set_to_none=True)

            step_policy = sft_backend._all_reduce_scalar(step_policy_local, device=device, op=dist.ReduceOp.SUM)
            sample_cnt = sft_backend._all_reduce_scalar(sample_cnt_local, device=device, op=dist.ReduceOp.SUM)
            sample_correct = sft_backend._all_reduce_scalar(sample_correct_local, device=device, op=dist.ReduceOp.SUM)
            prompt_pass = sft_backend._all_reduce_scalar(prompt_pass_local, device=device, op=dist.ReduceOp.SUM)
            reward_sum = sft_backend._all_reduce_scalar(reward_sum_local, device=device, op=dist.ReduceOp.SUM)
            adv_abs_sum = sft_backend._all_reduce_scalar(adv_abs_sum_local, device=device, op=dist.ReduceOp.SUM)
            traj_cnt = sft_backend._all_reduce_scalar(traj_cnt_local, device=device, op=dist.ReduceOp.SUM)
            avg_loss = step_policy / max(1.0, float(global_target_tokens))
            dt = time.time() - step_start

            if is_main and (step == 1 or step == int(args.max_steps) or step % max(1, int(args.log_interval)) == 0):
                logger.log(
                    f"[train step {step}/{int(args.max_steps)}] "
                    f"loss={avg_loss:.6f} resp_tok={int(global_target_tokens)} "
                    f"sample_acc={float(sample_correct) / max(1.0, float(sample_cnt)):.4f} "
                    f"prompt_pass={float(prompt_pass) / max(1.0, float(cfg.global_batch_size)):.4f} "
                    f"reward_mean={float(reward_sum) / max(1.0, float(sample_cnt)):.4f} "
                    f"adv_abs_mean={float(adv_abs_sum) / max(1.0, float(traj_cnt)):.4f} "
                    f"raw_grad={raw_grad:.6f} grad={grad_norm:.6f} "
                    f"lr={float(opt.param_groups[0]['lr']):.2e} step_time={dt:.2f}s"
                )

            if int(cfg.eval_interval) > 0 and step % int(cfg.eval_interval) == 0:
                _sync_infer_weights(
                    step=step,
                    train_model=train_model,
                    base_model=train_model_raw,
                    infer_engine=infer_engine,
                    fsdp_enabled=fsdp_enabled,
                    world_size=world_size,
                    train_gpu=train_gpu,
                    logger=logger,
                    is_main=is_main,
                    sync_diag_interval=int(cfg.sync_diag_interval),
                    sync_diag_sample_values=int(cfg.sync_diag_sample_values),
                    sync_infer_offload_cpu=bool(cfg.sync_infer_offload_cpu),
                )
                _evaluate_math(
                    step=step,
                    eval_data=eval_data,
                    infer_engine=infer_engine,
                    tokenizer=tok,
                    cfg=cfg,
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    train_gpu=train_gpu,
                    logger=logger,
                    is_main=is_main,
                )

            if int(args.save_interval) > 0 and (step % int(args.save_interval) == 0 or step == int(args.max_steps)):
                model_state = sft_backend._collect_model_state(train_model, train_model_raw, fsdp_enabled=fsdp_enabled)
                if is_main and model_state is not None:
                    ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pth")
                    torch.save(
                        {
                            "time": now_str(),
                            "step": step,
                            "cfg": asdict(cfg),
                            "model_state": model_state,
                        },
                        ckpt_path,
                    )
                    latest_full_path = os.path.join(args.out_dir, "latest_full_model.pth")
                    torch.save(model_state, latest_full_path)
                    logger.log(f"saved: {ckpt_path}")
                    logger.log(f"saved: {latest_full_path}")
                if world_size > 1 and _dist_is_initialized():
                    torch.cuda.set_device(train_gpu)
                    dist.barrier(device_ids=[int(train_gpu)])

        logger.log("train end.")
    finally:
        if world_size > 1 and _dist_is_initialized():
            dist.destroy_process_group()


def _pad_batch_local(seqs: List[List[int]], device: str, pad_id: int = 0):
    lens = [len(s) for s in seqs]
    tmax = max(lens)
    bsz = len(seqs)
    x = torch.full((bsz, tmax), pad_id, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        if seq:
            x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    return x, lens


if __name__ == "__main__":
    main()
