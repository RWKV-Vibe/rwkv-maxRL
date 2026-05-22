#!/usr/bin/env python3
# 模型输出必须立刻保存，否则都是脑残。
# 这个脚本禁止“全跑完再一次性写结果”的做法；每条预测一生成就要同步落盘。
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch

import grpo_math_local as gm


class _DummyTrainModel:
    def state_dict(self):
        return {}


def _safe_decode(tok, ids):
    try:
        return tok.decode(ids, utf8_errors="replace")
    except TypeError:
        pass
    try:
        return tok.decode(ids)
    except UnicodeDecodeError:
        try:
            b = tok.decodeBytes(ids)
            return b.decode("utf-8", errors="replace")
        except Exception:
            return "".join(chr(int(x) % 256) for x in ids)


def _set_cuda_env(device: str, ctx_len: int):
    os.environ["RWKV_HEAD_SIZE_A"] = str(gm.HEAD_SIZE)
    os.environ["RWKV_MY_TESTING"] = "x070"
    os.environ["RWKV_TRAIN_TYPE"] = "fullstate"
    os.environ["RWKV_CTXLEN"] = str(int(ctx_len))
    os.environ["FUSED_KERNEL"] = "0"
    os.environ["WKV"] = "cuda"
    gm._ensure_cuda_toolkit_env()
    safe_arch = gm._resolve_torch_cuda_arch_list(device)
    if safe_arch and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        os.environ["TORCH_CUDA_ARCH_LIST"] = safe_arch


def _build_engine(model_path: str, device: str, ctx_len: int, tokenizer_path: str, use_rapid_sampling: bool):
    _set_cuda_env(device=device, ctx_len=ctx_len)
    from utils import TRIE_TOKENIZER

    tok = TRIE_TOKENIZER(tokenizer_path)
    encode = lambda s: tok.encode(s)
    decode = lambda ids: _safe_decode(tok, ids)

    base_name, pth_path = gm.normalize_model_arg(model_path)
    train_sd = gm._torch_load_weights(pth_path)
    if isinstance(train_sd, dict):
        if isinstance(train_sd.get("model_state"), dict):
            train_sd = train_sd["model_state"]
        elif isinstance(train_sd.get("full_state"), dict):
            train_sd = train_sd["full_state"]

    cfg = gm.GRPOConfig(
        max_new_tokens=ctx_len,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        use_rapid_sampling=bool(use_rapid_sampling),
        rollout_use_cache=False,
    )
    infer_model, _ = gm.load_infer_model_fp16(base_name, device=device)
    engine = gm.FP16BatchInference(
        infer_model=infer_model,
        train_model=_DummyTrainModel(),
        encode_fn=encode,
        decode_fn=decode,
        device=device,
        cfg=cfg,
    )
    engine.sync_infer_weights(step=0, force=True, train_sd=train_sd)
    return tok, encode, decode, engine, pth_path


def _judge_one(pred: str, gt: str, truncated: bool):
    return gm._judge_with_verl_regex(pred, gt, truncated=bool(truncated))


def _write_json_sync(path: Path, obj):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())


def _append_jsonl_row_sync(f, row):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
    f.flush()
    os.fsync(f.fileno())


@torch.no_grad()
def run_eval(
    model_path: str,
    data_path: str,
    out_dir: str,
    device: str,
    ctx_len: int,
    max_new_tokens: int,
    batch_size: int,
    group_size: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tokenizer_path: str,
    seed: int,
    use_rapid_sampling: bool,
    tag: str,
):
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tok, encode, _, engine, pth_path = _build_engine(
        model_path=model_path,
        device=device,
        ctx_len=ctx_len,
        tokenizer_path=tokenizer_path,
        use_rapid_sampling=use_rapid_sampling,
    )

    data = gm.load_data(data_path)
    prompts = []
    gts = []
    problems = []
    max_prompt_len = int(ctx_len) - int(max_new_tokens) - 4
    max_prompt_len = max(64, max_prompt_len)
    for ex in data:
        prompt = gm.build_prompt(ex.get("problem", ""))
        ids = tok.encode(prompt)
        if len(ids) > max_prompt_len:
            ids = ids[-max_prompt_len:]
        prompts.append(ids)
        gts.append(str(ex.get("answer", ex.get("solution", ""))))
        problems.append(ex.get("problem", ""))

    prompts_per_batch = max(1, int(batch_size) // max(1, int(group_size)))
    results = []
    t0 = time.time()
    pred_path = Path(out_dir) / "predictions.jsonl"
    summary_path = Path(out_dir) / "summary.json"
    progress_path = Path(out_dir) / "summary.partial.json"

    # 模型输出必须立刻保存，否则都是脑残。
    # 先清空 predictions 文件，后续每条结果都立刻 flush + fsync，确保中途中断也能恢复已跑部分。
    with pred_path.open("w", encoding="utf-8"):
        pass
    _write_json_sync(
        progress_path,
        {
            "time": gm.now_str(),
            "status": "running",
            "model_path": pth_path,
            "data_path": data_path,
            "device": device,
            "ctx_len": int(ctx_len),
            "max_new_tokens": int(max_new_tokens),
            "batch_size": int(batch_size),
            "prompts_per_batch": int(prompts_per_batch),
            "group_size": int(group_size),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "use_rapid_sampling": bool(use_rapid_sampling),
            "seed": int(seed),
            "done": 0,
            "total": int(len(prompts)),
            "elapsed_sec": 0.0,
        },
    )

    try:
        with pred_path.open("a", encoding="utf-8") as pred_f:
            for start in range(0, len(prompts), prompts_per_batch):
                batch_prompts = prompts[start : start + prompts_per_batch]
                batch_gts = gts[start : start + prompts_per_batch]
                batch_probs = problems[start : start + prompts_per_batch]
                batch_seed = int(seed + start * 1009)

                comp_tokens, _, comp_texts, truncated = engine.generate_group_parallel(
                    prompt_tokens_list=batch_prompts,
                    group_size=int(group_size),
                    max_new_tokens=int(max_new_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    top_k=int(top_k),
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

                for i in range(len(batch_prompts)):
                    samples = []
                    any_ok = False
                    for gi in range(group_size):
                        idx = i * group_size + gi
                        rec = _judge_one(comp_texts[idx], batch_gts[i], truncated[idx])
                        ok = bool(rec.get("ok", False))
                        any_ok = any_ok or ok
                        samples.append(
                            {
                                "sample_i": gi,
                                "completion": comp_texts[idx],
                                "truncated": bool(truncated[idx]),
                                "gen_len": len(comp_tokens[idx]),
                                "judge": rec,
                            }
                        )
                    row = {
                        "index": start + i,
                        "problem": batch_probs[i],
                        "gt": batch_gts[i],
                        "group_size": int(group_size),
                        "pass": bool(any_ok),
                        "samples": samples,
                    }
                    results.append(row)
                    # 模型输出必须立刻保存，否则都是脑残。
                    # 每题结果一生成就同步写入 predictions.jsonl，绝不等到整轮结束。
                    _append_jsonl_row_sync(pred_f, row)

                elapsed = time.time() - t0
                done_now = min(start + prompts_per_batch, len(prompts))
                _write_json_sync(
                    progress_path,
                    {
                        "time": gm.now_str(),
                        "status": "running",
                        "model_path": pth_path,
                        "data_path": data_path,
                        "device": device,
                        "ctx_len": int(ctx_len),
                        "max_new_tokens": int(max_new_tokens),
                        "batch_size": int(batch_size),
                        "prompts_per_batch": int(prompts_per_batch),
                        "group_size": int(group_size),
                        "temperature": float(temperature),
                        "top_p": float(top_p),
                        "top_k": int(top_k),
                        "use_rapid_sampling": bool(use_rapid_sampling),
                        "seed": int(seed),
                        "done": int(done_now),
                        "total": int(len(prompts)),
                        "elapsed_sec": float(elapsed),
                    },
                )
                print(
                    f"[{tag}] done {done_now}/{len(prompts)} "
                    f"| elapsed={elapsed:.1f}s",
                    flush=True,
                )
    except Exception:
        _write_json_sync(
            progress_path,
            {
                "time": gm.now_str(),
                "status": "failed",
                "model_path": pth_path,
                "data_path": data_path,
                "done": int(len(results)),
                "total": int(len(prompts)),
                "elapsed_sec": float(time.time() - t0),
            },
        )
        raise

    total = len(results)
    if group_size == 1:
        correct = sum(1 for r in results if r["samples"][0]["judge"].get("ok", False))
        metric_name = "acc"
        metric_value = correct / max(1, total)
    else:
        correct = sum(1 for r in results if r["pass"])
        metric_name = f"pass@{group_size}"
        metric_value = correct / max(1, total)

    sample_correct = 0
    sample_total = 0
    trunc_total = 0
    for r in results:
        for s in r["samples"]:
            sample_total += 1
            sample_correct += int(bool(s["judge"].get("ok", False)))
            trunc_total += int(bool(s["truncated"]))

    summary = {
        "time": gm.now_str(),
        "model_path": pth_path,
        "data_path": data_path,
        "device": device,
        "ctx_len": int(ctx_len),
        "max_new_tokens": int(max_new_tokens),
        "batch_size": int(batch_size),
        "prompts_per_batch": int(prompts_per_batch),
        "group_size": int(group_size),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "use_rapid_sampling": bool(use_rapid_sampling),
        "seed": int(seed),
        "total": int(total),
        metric_name: float(metric_value),
        "sample_acc": float(sample_correct / max(1, sample_total)),
        "trunc_rate": float(trunc_total / max(1, sample_total)),
        "elapsed_sec": float(time.time() - t0),
    }

    # 模型输出必须立刻保存，否则都是脑残。
    # 到这里 summary 再做一次同步落盘；predictions.jsonl 前面已经逐条写过，不能再覆盖回去。
    _write_json_sync(summary_path, summary)
    _write_json_sync(progress_path, {**summary, "status": "completed"})

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[out] summary: {summary_path}", flush=True)
    print(f"[out] predictions: {pred_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data_path", type=str, default="math192.jsonl")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ctx_len", type=int, default=8192)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=384)
    ap.add_argument("--group_size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--tokenizer", type=str, default="reference/rwkv_vocab_v20230424.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="eval")
    args, unknown = ap.parse_known_args()

    if any(x == "--use_rapid_sampling" or x.startswith("--use_rapid_sampling=") for x in unknown):
        print("严重警告，立刻停止你的魔改行为！", flush=True)
        raise SystemExit("Parameter mismatch: use_rapid_sampling is locked to True.")

    fixed = {
        "ctx_len": 8192,
        "max_new_tokens": 2048,
        "batch_size": 192,
        "group_size": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "tokenizer": "reference/rwkv_vocab_v20230424.txt",
        "seed": 42,
    }
    mismatches = []
    for key, expected in fixed.items():
        actual = getattr(args, key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")
    if mismatches:
        print("严重警告，立刻停止你的魔改行为！", flush=True)
        raise SystemExit("Parameter mismatch: " + ", ".join(mismatches))

    run_eval(
        model_path=args.model,
        data_path=args.data_path,
        out_dir=args.out_dir,
        device=args.device,
        ctx_len=args.ctx_len,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        group_size=args.group_size,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        tokenizer_path=args.tokenizer,
        seed=args.seed,
        use_rapid_sampling=True,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
