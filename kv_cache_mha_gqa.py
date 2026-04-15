#!/usr/bin/env python3
"""
KV cache footprint: Multi-Head Attention (MHA) vs Grouped Query Attention (GQA).

Extracted from the project notebook for non-interactive runs (local GPU, CI, or batch jobs).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DTYPE_BYTES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
}


def sizeof_past_key_values(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "to_legacy_cache"):
        past_key_values = past_key_values.to_legacy_cache()

    total = 0
    for layer_kv in past_key_values:
        if layer_kv is None:
            continue
        if isinstance(layer_kv, (tuple, list)):
            for t in layer_kv:
                if t is None:
                    continue
                total += t.numel() * t.element_size()
        else:
            t = layer_kv
            if t is not None and hasattr(t, "numel"):
                total += t.numel() * t.element_size()
    return total


def analytical_kv_bytes(
    num_layers: int,
    batch_size: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> int:
    bytes_per = DTYPE_BYTES.get(dtype, torch.tensor([], dtype=dtype).element_size())
    return 2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim * bytes_per


def bytes_to_gb(nbytes: int) -> float:
    return nbytes / (1024**3)


def gpu_mem_gb() -> Optional[tuple[float, float]]:
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserv = torch.cuda.memory_reserved() / (1024**3)
    return alloc, reserv


def default_compute_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


# ---------------------------------------------------------------------------
# Synthetic Llama-config experiment
# ---------------------------------------------------------------------------


@dataclass
class SimConfig:
    hidden_size: int = 1024
    num_layers: int = 16
    num_heads: int = 16
    num_kv_heads: int = 16
    vocab_size: int = 32000
    max_position_embeddings: int = 16384
    batch_size: int = 1
    total_tokens: int = 4096
    prompt_tokens: int = 128
    measure_every: int = 256
    dtype: Optional[torch.dtype] = None
    device: Optional[str] = None

    def resolved_dtype(self) -> torch.dtype:
        d = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        return self.dtype if self.dtype is not None else default_compute_dtype(d)

    def resolved_device(self) -> str:
        return self.device if self.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")


def build_llama_model(cfg: SimConfig) -> LlamaForCausalLM:
    assert cfg.hidden_size % cfg.num_heads == 0
    dev = cfg.resolved_device()
    dt = cfg.resolved_dtype()

    llama_cfg = LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=4 * cfg.hidden_size,
        num_hidden_layers=cfg.num_layers,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=False,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(llama_cfg).to(dev).eval().to(dtype=dt)
    return model


@torch.no_grad()
def run_kv_growth_synthetic(cfg: SimConfig) -> Dict[str, List]:
    model = build_llama_model(cfg)
    head_dim = cfg.hidden_size // cfg.num_heads
    dev = cfg.resolved_device()

    prompt_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.prompt_tokens), device=dev)

    t0 = time.time()
    out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    prefill_ms = (time.time() - t0) * 1000

    results: Dict[str, List] = {
        "seq_len": [],
        "kv_gb_empirical": [],
        "kv_gb_analytical": [],
        "step_time_ms": [],
        "gpu_alloc_gb": [],
        "gpu_reserved_gb": [],
    }

    dt = cfg.resolved_dtype()

    def record(seq_len: int, step_ms: float) -> None:
        emp = sizeof_past_key_values(past)
        ana = analytical_kv_bytes(
            num_layers=cfg.num_layers,
            batch_size=cfg.batch_size,
            seq_len=seq_len,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=head_dim,
            dtype=dt,
        )
        results["seq_len"].append(seq_len)
        results["kv_gb_empirical"].append(bytes_to_gb(emp))
        results["kv_gb_analytical"].append(bytes_to_gb(ana))
        results["step_time_ms"].append(step_ms)
        gm = gpu_mem_gb()
        if gm is None:
            results["gpu_alloc_gb"].append(None)
            results["gpu_reserved_gb"].append(None)
        else:
            results["gpu_alloc_gb"].append(gm[0])
            results["gpu_reserved_gb"].append(gm[1])

    record(cfg.prompt_tokens, prefill_ms)

    current_len = cfg.prompt_tokens
    next_token = torch.randint(0, cfg.vocab_size, (cfg.batch_size, 1), device=dev)

    for _ in range(cfg.prompt_tokens, cfg.total_tokens):
        t1 = time.time()
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        dt_ms = (time.time() - t1) * 1000
        next_token = torch.randint(0, cfg.vocab_size, (cfg.batch_size, 1), device=dev)
        current_len += 1
        if (current_len % cfg.measure_every) == 0 or current_len == cfg.total_tokens:
            record(current_len, dt_ms)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def print_tail(name: str, cfg: SimConfig, res: Dict[str, List], k: int = 6) -> None:
    print(f"\n=== {name} ===")
    print(
        f"layers={cfg.num_layers} heads={cfg.num_heads} kv_heads={cfg.num_kv_heads} "
        f"hidden={cfg.hidden_size} dtype={cfg.resolved_dtype()}"
    )
    print("Last rows:")
    for i in range(max(0, len(res["seq_len"]) - k), len(res["seq_len"])):
        s = res["seq_len"][i]
        e = res["kv_gb_empirical"][i]
        a = res["kv_gb_analytical"][i]
        ms = res["step_time_ms"][i]
        ga = res["gpu_alloc_gb"][i]
        gr = res["gpu_reserved_gb"][i]
        print(
            f"  seq={s:6d}  KV_emp={e:.6f} GB  KV_ana={a:.6f} GB  "
            f"step={ms:7.2f} ms  gpu_alloc={ga}  gpu_res={gr}"
        )


def plot_synthetic(
    res_mha: Dict[str, List],
    res_gqa: Dict[str, List],
    title: str,
    output_path: Optional[str],
    show: bool,
) -> None:
    plt.figure()
    plt.plot(res_mha["seq_len"], res_mha["kv_gb_empirical"], label="MHA empirical")
    plt.plot(res_gqa["seq_len"], res_gqa["kv_gb_empirical"], label="GQA empirical")
    plt.plot(res_mha["seq_len"], res_mha["kv_gb_analytical"], linestyle="--", label="MHA analytical")
    plt.plot(res_gqa["seq_len"], res_gqa["kv_gb_analytical"], linestyle="--", label="GQA analytical")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("KV cache size (GB)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def cmd_synthetic(args: argparse.Namespace) -> int:
    base = SimConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        prompt_tokens=args.prompt_tokens,
        total_tokens=args.total_tokens,
        measure_every=args.measure_every,
        device=args.device,
    )
    cfg_mha = replace(base, num_kv_heads=base.num_heads)
    cfg_gqa = replace(base, num_kv_heads=args.gqa_kv_heads)

    print("Running MHA (synthetic Llama config)...")
    res_mha = run_kv_growth_synthetic(cfg_mha)
    print("Running GQA (synthetic Llama config)...")
    res_gqa = run_kv_growth_synthetic(cfg_gqa)

    print_tail("MHA", cfg_mha, res_mha)
    print_tail("GQA", cfg_gqa, res_gqa)

    final_mha = res_mha["kv_gb_empirical"][-1]
    final_gqa = res_gqa["kv_gb_empirical"][-1]
    ratio = cfg_mha.num_heads / max(cfg_gqa.num_kv_heads, 1)
    print("\nFinal KV (empirical):")
    print(f"  MHA: {final_mha:.6f} GB")
    print(f"  GQA: {final_gqa:.6f} GB")
    print(f"  Reduction factor: {final_mha / max(final_gqa, 1e-12):.2f}x (ideal ~{ratio:.2f}x)")

    out = os.path.join(args.output_dir, "kv_synthetic_mha_vs_gqa.png") if args.output_dir else None
    plot_synthetic(
        res_mha,
        res_gqa,
        title="KV cache growth: MHA vs GQA (synthetic Llama)",
        output_path=out,
        show=args.show,
    )
    return 0


# ---------------------------------------------------------------------------
# Pretrained Hugging Face models
# ---------------------------------------------------------------------------


@dataclass
class RunCfg:
    batch_size: int = 1
    total_tokens: int = 2048
    measure_every: int = 128
    prompt_text: str = "Summarize the following meeting:\n" + ("hello " * 200)


def load_model_and_tokenizer(model_id: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",
    ).eval()
    return tok, model


@torch.no_grad()
def run_kv_growth_pretrained(model, tokenizer, run_cfg: RunCfg) -> Dict[str, Any]:
    cfg = model.config
    num_layers = getattr(cfg, "num_hidden_layers")
    num_heads = getattr(cfg, "num_attention_heads")
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
    hidden_size = getattr(cfg, "hidden_size")
    head_dim = hidden_size // num_heads
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    prompt = tokenizer(run_cfg.prompt_text, return_tensors="pt")
    input_ids = prompt["input_ids"].to(device)
    if input_ids.shape[1] > run_cfg.total_tokens:
        input_ids = input_ids[:, : run_cfg.total_tokens]

    t0 = time.time()
    out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    prefill_ms = (time.time() - t0) * 1000
    seq_len = input_ids.shape[1]

    results: Dict[str, Any] = {
        "seq_len": [],
        "kv_gb_empirical": [],
        "kv_gb_analytical_actual": [],
        "kv_gb_analytical_mha": [],
        "step_time_ms": [],
        "gpu_alloc_gb": [],
        "gpu_reserved_gb": [],
        "meta": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "dtype": str(dtype),
            "max_pos": getattr(cfg, "max_position_embeddings", None),
            "model_type": getattr(cfg, "model_type", None),
        },
    }

    def record(cur_len: int, step_ms: float) -> None:
        emp = sizeof_past_key_values(past)
        ana_actual = analytical_kv_bytes(
            num_layers=num_layers,
            batch_size=run_cfg.batch_size,
            seq_len=cur_len,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        ana_mha = analytical_kv_bytes(
            num_layers=num_layers,
            batch_size=run_cfg.batch_size,
            seq_len=cur_len,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
        )
        results["seq_len"].append(cur_len)
        results["kv_gb_empirical"].append(bytes_to_gb(emp))
        results["kv_gb_analytical_actual"].append(bytes_to_gb(ana_actual))
        results["kv_gb_analytical_mha"].append(bytes_to_gb(ana_mha))
        results["step_time_ms"].append(step_ms)
        gm = gpu_mem_gb()
        if gm is None:
            results["gpu_alloc_gb"].append(None)
            results["gpu_reserved_gb"].append(None)
        else:
            results["gpu_alloc_gb"].append(gm[0])
            results["gpu_reserved_gb"].append(gm[1])

    record(seq_len, prefill_ms)

    vocab_size = getattr(cfg, "vocab_size")
    next_token = torch.randint(0, vocab_size, (run_cfg.batch_size, 1), device=device)

    while seq_len < run_cfg.total_tokens:
        t1 = time.time()
        out = model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True)
        past = out.past_key_values
        dt_ms = (time.time() - t1) * 1000
        next_token = torch.randint(0, vocab_size, (run_cfg.batch_size, 1), device=device)
        seq_len += 1
        if (seq_len % run_cfg.measure_every) == 0 or seq_len == run_cfg.total_tokens:
            record(seq_len, dt_ms)

    return results


def describe_pretrained(res: Dict[str, Any], name: str) -> None:
    meta = res["meta"]
    print(f"\n=== {name} ===")
    print(
        "layers:",
        meta["num_layers"],
        "heads:",
        meta["num_heads"],
        "kv_heads:",
        meta["num_kv_heads"],
        "hidden:",
        meta["hidden_size"],
        "head_dim:",
        meta["head_dim"],
        "dtype:",
        meta["dtype"],
        "max_pos:",
        meta["max_pos"],
    )
    print("Final seq_len:", res["seq_len"][-1])
    print("Final KV empirical (GB):", f"{res['kv_gb_empirical'][-1]:.6f}")
    print("Final KV analytical actual (GB):", f"{res['kv_gb_analytical_actual'][-1]:.6f}")
    print("Final KV analytical MHA baseline (GB):", f"{res['kv_gb_analytical_mha'][-1]:.6f}")


def plot_pretrained(res_mha: Dict[str, Any], res_gqa: Dict[str, Any], output_path: Optional[str], show: bool) -> None:
    plt.figure()
    plt.plot(
        res_mha["seq_len"],
        res_mha["kv_gb_empirical"],
        label=f"MHA empirical (kv_heads={res_mha['meta']['num_kv_heads']})",
    )
    plt.plot(
        res_gqa["seq_len"],
        res_gqa["kv_gb_empirical"],
        label=f"GQA empirical (kv_heads={res_gqa['meta']['num_kv_heads']})",
    )
    plt.plot(res_mha["seq_len"], res_mha["kv_gb_analytical_actual"], linestyle="--", label="MHA analytical")
    plt.plot(res_gqa["seq_len"], res_gqa["kv_gb_analytical_actual"], linestyle="--", label="GQA analytical")
    plt.xlabel("Sequence length (tokens)")
    plt.ylabel("KV cache size (GB)")
    plt.title("KV cache growth: pretrained MHA vs GQA")
    plt.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved figure: {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def cmd_pretrained(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        print(
            "Warning: pretrained mode expects a GPU for reasonable performance and memory layout.",
            file=sys.stderr,
        )
    dtype = torch.float16 if args.fp16 else torch.float32

    run_cfg = RunCfg(total_tokens=args.total_tokens, measure_every=args.measure_every)

    print("Loading MHA model:", args.mha_model)
    tok_mha, model_mha = load_model_and_tokenizer(args.mha_model, dtype=dtype)
    print("Running KV growth (MHA)...")
    res_mha = run_kv_growth_pretrained(model_mha, tok_mha, run_cfg)
    del model_mha
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nLoading GQA model:", args.gqa_model)
    tok_gqa, model_gqa = load_model_and_tokenizer(args.gqa_model, dtype=dtype)
    print("Running KV growth (GQA)...")
    res_gqa = run_kv_growth_pretrained(model_gqa, tok_gqa, run_cfg)
    del model_gqa
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    describe_pretrained(res_mha, "MHA model")
    describe_pretrained(res_gqa, "GQA model")

    final_mha = res_mha["kv_gb_empirical"][-1]
    final_gqa = res_gqa["kv_gb_empirical"][-1]
    print("\nFinal KV empirical:")
    print("  MHA:", f"{final_mha:.6f}", "GB")
    print("  GQA:", f"{final_gqa:.6f}", "GB")
    print("  Reduction factor (MHA/GQA):", f"{final_mha / max(final_gqa, 1e-12):.2f}x")

    out = os.path.join(args.output_dir, "kv_pretrained_mha_vs_gqa.png") if args.output_dir else None
    plot_pretrained(res_mha, res_gqa, out, args.show)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure KV cache growth for MHA vs GQA (synthetic or pretrained models)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("synthetic", help="Randomly initialized Llama-shaped model (no downloads).")
    ps.add_argument("--hidden-size", type=int, default=1024)
    ps.add_argument("--num-layers", type=int, default=16)
    ps.add_argument("--num-heads", type=int, default=16)
    ps.add_argument("--gqa-kv-heads", type=int, default=4, help="KV heads for the GQA run (MHA uses num-heads).")
    ps.add_argument("--batch-size", type=int, default=1)
    ps.add_argument("--prompt-tokens", type=int, default=128)
    ps.add_argument("--total-tokens", type=int, default=4096)
    ps.add_argument("--measure-every", type=int, default=256)
    ps.add_argument("--device", choices=("cuda", "cpu"), default=None, help="Override auto device selection.")
    ps.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, save plot as PNG under this directory.",
    )
    ps.add_argument("--show", action="store_true", help="Display matplotlib windows (if supported).")
    ps.set_defaults(func=cmd_synthetic)

    pp = sub.add_parser("pretrained", help="Compare two Hugging Face causal LMs (downloads weights).")
    pp.add_argument(
        "--mha-model",
        default="openlm-research/open_llama_3b",
        help="Model with full MHA (or any baseline).",
    )
    pp.add_argument(
        "--gqa-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="GQA-style model to compare.",
    )
    pp.add_argument("--total-tokens", type=int, default=2048)
    pp.add_argument("--measure-every", type=int, default=128)
    pp.add_argument("--fp16", action="store_true", default=True, help="Load in float16 (default: on).")
    pp.add_argument("--no-fp16", action="store_false", dest="fp16", help="Load in float32.")
    pp.add_argument("--output-dir", type=str, default=None)
    pp.add_argument("--show", action="store_true")
    pp.set_defaults(func=cmd_pretrained)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "output_dir", None):
        os.makedirs(args.output_dir, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
