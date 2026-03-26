"""
Experiment 7B: Norm Ablation + Adaptive Halting for KV Recurrence.

Exp7a showed KV recurrence works (KV-A +2pp on GSM8K) but degrades at higher
step counts due to hidden state drift from repeated norm application. Also
tested consolidation-based KV-aware generation, but frozen lower layers (0-11)
can't handle hidden-state inputs → consolidation hurts.

This experiment focuses on two remaining fixes:
  1. Norm ablation: norm_mode="final_only" — don't norm h between recurrence
     steps, only for diagnostics. Hypothesis: fixes drift, lets more steps help.
  2. Adaptive halting: monitor logit lens confidence over a sliding window,
     halt when confidence decays. Hypothesis: finds optimal step count per item.

Recurrence: partial-layer (12-35) with KV cache (same as exp7a).
Generation: prefix mode (prepend h to query embeddings, same as exp7a).

Configurations:
| Config  | Steps    | Norm   | Description                         |
|---------|----------|--------|-------------------------------------|
| KV7B-0  | 0        | —      | Baseline                            |
| KV7B-A  | 4        | every  | exp7a repro (control)               |
| KV7B-B  | 4        | final  | Norm ablation at 4 steps            |
| KV7B-C  | 8        | every  | 8 steps, norm every (exp7a-like)    |
| KV7B-D  | 8        | final  | Norm ablation at 8 steps            |
| KV7B-E  | 16       | final  | Scale test                          |
| KV7B-F  | 32       | final  | Scale test                          |
| KV7B-G  | adaptive | final  | Adaptive halt, max=64               |

Key comparisons:
  - KV7B-A vs KV7B-B → norm ablation effect at 4 steps
  - KV7B-C vs KV7B-D → norm ablation effect at 8 steps
  - KV7B-D/E/F → with norm fix, do more steps keep helping?
  - KV7B-G → does adaptive halting find the sweet spot?

Usage:
    python experiments/exp7b_kv_generation.py --eval-only
    python experiments/exp7b_kv_generation.py --benchmark gsm8k --subset 50
"""

import sys
import json
import argparse
import time
import gc
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import (
    RESULTS_DIR, DEVICE, MID_LAYER_INDEX,
    BENCHMARK_GSM8K_MAX_TOKENS, BENCHMARK_ARC_MAX_TOKENS,
)
from model_utils import load_model
from kv_recurrence import (
    kv_recurrence, kv_recurrence_full, kv_recurrence_gated,
    kv_recurrence_first_token_override, kv_recurrence_kl_gated,
    kv_recurrence_answer_mass_gated,
)
from benchmarks import load_benchmark, extract_answer, check_answer
from eval_prompts import PROMPTS as EVAL_PROMPTS, check_answer as eval_check_answer


CONFIGS = {
    "KV7B-0": {
        "description": "Baseline (no recurrence)",
        "type": "baseline",
    },
    # --- Norm ablation: does final_only fix drift at higher steps? ---
    "KV7B-A": {
        "description": "4 steps, norm=every (exp7a repro)",
        "type": "kv_enhanced",
        "n_steps": 4,
        "norm_mode": "every_step",
    },
    "KV7B-B": {
        "description": "4 steps, norm=final",
        "type": "kv_enhanced",
        "n_steps": 4,
        "norm_mode": "final_only",
    },
    "KV7B-C": {
        "description": "8 steps, norm=every",
        "type": "kv_enhanced",
        "n_steps": 8,
        "norm_mode": "every_step",
    },
    "KV7B-D": {
        "description": "8 steps, norm=final",
        "type": "kv_enhanced",
        "n_steps": 8,
        "norm_mode": "final_only",
    },
    "KV7B-E": {
        "description": "16 steps, norm=final",
        "type": "kv_enhanced",
        "n_steps": 16,
        "norm_mode": "final_only",
    },
    "KV7B-F": {
        "description": "32 steps, norm=final",
        "type": "kv_enhanced",
        "n_steps": 32,
        "norm_mode": "final_only",
    },
    # --- Adaptive halting ---
    "KV7B-G": {
        "description": "Adaptive halt, norm=final, max=64",
        "type": "kv_enhanced",
        "n_steps": 64,
        "norm_mode": "final_only",
        "adaptive_halt": True,
        "decay_window": 3,
        "decay_threshold": 0.0,
    },
    "KV7B-H": {
        "description": "Adaptive halt, norm=every, max=64",
        "type": "kv_enhanced",
        "n_steps": 64,
        "norm_mode": "every_step",
        "adaptive_halt": True,
        "decay_window": 3,
        "decay_threshold": 0.0,
    },
    # --- Split-layer generation: layers 0-11 clean, 12-35 with thought KV ---
    "KV7B-S4": {
        "description": "Split gen, 4 steps, norm=every",
        "type": "kv_split",
        "n_steps": 4,
        "norm_mode": "every_step",
    },
    "KV7B-S8": {
        "description": "Split gen, 8 steps, norm=every",
        "type": "kv_split",
        "n_steps": 8,
        "norm_mode": "every_step",
    },
    "KV7B-S16": {
        "description": "Split gen, 16 steps, norm=every",
        "type": "kv_split",
        "n_steps": 16,
        "norm_mode": "every_step",
    },
    "KV7B-S32": {
        "description": "Split gen, 32 steps, norm=every",
        "type": "kv_split",
        "n_steps": 32,
        "norm_mode": "every_step",
    },
    # --- Confidence-gated: route easy items to baseline, hard to recurrence ---
    "KV7B-G90": {
        "description": "Gated split, thresh=0.9, 4 steps",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.9,
        "norm_mode": "every_step",
        "generation_mode": "split",
    },
    "KV7B-G80": {
        "description": "Gated split, thresh=0.8, 4 steps",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.8,
        "norm_mode": "every_step",
        "generation_mode": "split",
    },
    "KV7B-G95": {
        "description": "Gated split, thresh=0.95, 4 steps",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.95,
        "norm_mode": "every_step",
        "generation_mode": "split",
    },
    # --- Prompt-weighted: blend baseline logits into first token ---
    "KV7B-P5": {
        "description": "Gated+prompt_w=0.5, thresh=0.9",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.9,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.5,
    },
    "KV7B-P7": {
        "description": "Gated+prompt_w=0.7, thresh=0.9",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.9,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.7,
    },
    "KV7B-P3": {
        "description": "Gated+prompt_w=0.3, thresh=0.9",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.9,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.3,
    },
    # --- Lower thresholds: route more items to baseline ---
    "KV7B-T50": {
        "description": "Gated, thresh=0.5, pw=0.7",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.5,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.7,
    },
    "KV7B-T60": {
        "description": "Gated, thresh=0.6, pw=0.7",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.6,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.7,
    },
    "KV7B-T70": {
        "description": "Gated, thresh=0.7, pw=0.7",
        "type": "kv_gated",
        "n_steps": 4,
        "confidence_threshold": 0.7,
        "norm_mode": "every_step",
        "generation_mode": "split",
        "prompt_weight": 0.7,
    },
    # --- Approach 1: First-token override (pw=1.0 on token 1 only) ---
    "KV7B-FTO": {
        "description": "First-token override, 4 steps",
        "type": "first_token_override",
        "n_steps": 4,
        "norm_mode": "every_step",
    },
    # --- Approach 2: KL-divergence gating ---
    "KV7B-KL1": {
        "description": "KL gate, thresh=1.0, pw=0.7",
        "type": "kl_gated",
        "n_steps": 4,
        "kl_threshold": 1.0,
        "norm_mode": "every_step",
        "prompt_weight": 0.7,
    },
    "KV7B-KL3": {
        "description": "KL gate, thresh=3.0, pw=0.7",
        "type": "kl_gated",
        "n_steps": 4,
        "kl_threshold": 3.0,
        "norm_mode": "every_step",
        "prompt_weight": 0.7,
    },
    # --- Approach 3: Answer-token mass gating ---
    "KV7B-AM3": {
        "description": "Answer mass gate, thresh=0.3, pw=0.7",
        "type": "answer_mass_gated",
        "n_steps": 4,
        "mass_threshold": 0.3,
        "norm_mode": "every_step",
        "prompt_weight": 0.7,
    },
    "KV7B-AM5": {
        "description": "Answer mass gate, thresh=0.5, pw=0.7",
        "type": "answer_mass_gated",
        "n_steps": 4,
        "mass_threshold": 0.5,
        "norm_mode": "every_step",
        "prompt_weight": 0.7,
    },
}

ALL_CONFIGS = list(CONFIGS.keys())


def run_single_item(model, tokenizer, config_name, cfg, prompt_text, max_tokens):
    """Run a single config on a single prompt."""
    ctype = cfg["type"]

    if ctype == "baseline":
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        answer = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return {"answer": answer, "n_steps_taken": 0}

    elif ctype == "kv_enhanced":
        result = kv_recurrence_full(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            adaptive_halt=cfg.get("adaptive_halt", False),
            decay_window=cfg.get("decay_window", 3),
            decay_threshold=cfg.get("decay_threshold", 0.0),
            norm_mode=cfg.get("norm_mode", "final_only"),
            generation_mode="prefix",
        )
        return result

    elif ctype == "kv_split":
        result = kv_recurrence_full(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            norm_mode=cfg.get("norm_mode", "every_step"),
            generation_mode="split",
        )
        return result

    elif ctype == "kv_gated":
        result = kv_recurrence_gated(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            confidence_threshold=cfg["confidence_threshold"],
            norm_mode=cfg.get("norm_mode", "every_step"),
            generation_mode=cfg.get("generation_mode", "split"),
            prompt_weight=cfg.get("prompt_weight", 0.0),
        )
        return result

    elif ctype == "first_token_override":
        result = kv_recurrence_first_token_override(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            norm_mode=cfg.get("norm_mode", "every_step"),
        )
        return result

    elif ctype == "kl_gated":
        result = kv_recurrence_kl_gated(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            kl_threshold=cfg["kl_threshold"],
            norm_mode=cfg.get("norm_mode", "every_step"),
            prompt_weight=cfg.get("prompt_weight", 0.7),
        )
        return result

    elif ctype == "answer_mass_gated":
        result = kv_recurrence_answer_mass_gated(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
            mass_threshold=cfg["mass_threshold"],
            norm_mode=cfg.get("norm_mode", "every_step"),
            prompt_weight=cfg.get("prompt_weight", 0.7),
        )
        return result


def run_eval_prompts(model, tokenizer, configs_to_run):
    """Run on 20 eval prompts."""
    print("\n" + "=" * 60)
    print("Phase 1: Eval Prompts (20 prompts)")
    print("=" * 60)

    results = {}

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            continue

        torch.cuda.empty_cache()
        gc.collect()

        cfg = CONFIGS[config_name]
        print(f"\n--- {config_name}: {cfg['description']} ---")

        correct = 0
        total_steps = 0

        for i, prompt_data in enumerate(EVAL_PROMPTS):
            prompt = prompt_data["prompt"]
            expected = prompt_data["answer"]

            result = run_single_item(
                model, tokenizer, config_name, cfg, prompt, max_tokens=64
            )

            answer = result["answer"].strip()
            is_correct = eval_check_answer(answer, expected)
            correct += int(is_correct)
            steps = result.get("n_steps_taken", 0)
            total_steps += steps

            status = "OK" if is_correct else "WRONG"
            print(f"  [{i+1:2d}/20] [{status}] steps={steps} "
                  f"ans={answer[:60].encode('ascii', 'replace').decode()}")

        acc = correct / len(EVAL_PROMPTS)
        avg_steps = total_steps / len(EVAL_PROMPTS)
        results[config_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(EVAL_PROMPTS),
            "avg_steps": avg_steps,
        }

        print(f"  {config_name}: {correct}/{len(EVAL_PROMPTS)} = {acc:.1%} "
              f"(avg steps={avg_steps:.1f})")

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Config':>12} | {'Description':<42} | {'Acc':>6} | {'Steps':>6}")
    print("-" * 70)
    for config_name in configs_to_run:
        if config_name in results:
            r = results[config_name]
            desc = CONFIGS[config_name]["description"]
            print(f"{config_name:>12} | {desc:<42} | {r['accuracy']:>5.1%} | "
                  f"{r['avg_steps']:>5.1f}")
    print("=" * 70)

    return results


def run_benchmark(model, tokenizer, benchmark_name, configs_to_run, items, output_dir):
    """Run on benchmark dataset."""
    max_tokens = (
        BENCHMARK_GSM8K_MAX_TOKENS if benchmark_name == "gsm8k"
        else BENCHMARK_ARC_MAX_TOKENS
    )

    all_results = {}
    results_path = output_dir / f"exp7b_{benchmark_name}_results.json"

    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            continue

        torch.cuda.empty_cache()
        gc.collect()

        cfg = CONFIGS[config_name]
        config_key = f"config_{config_name}"

        print(f"\n{'='*60}")
        print(f"{config_name}: {cfg['description']}")
        print(f"{'='*60}")

        completed_ids = set()
        if config_key in existing and "details" in existing[config_key]:
            completed_ids = {d["id"] for d in existing[config_key]["details"]}
            details = existing[config_key]["details"]
            print(f"  Resuming: {len(completed_ids)} already done")
        else:
            details = []

        correct = sum(1 for d in details if d["correct"])
        total_steps = sum(d.get("steps", 0) for d in details)
        t_start = time.time()

        for idx, item in enumerate(items):
            if item.id in completed_ids:
                continue

            try:
                result = run_single_item(
                    model, tokenizer, config_name, cfg, item.prompt, max_tokens
                )

                raw_answer = result["answer"]
                predicted = extract_answer(raw_answer, benchmark_name)
                is_correct = check_answer(predicted, item.expected, benchmark_name)
                steps = result.get("n_steps_taken", 0)

                correct += int(is_correct)
                total_steps += steps

                routed = result.get("routed", "")
                baseline_conf = result.get("baseline_confidence", None)

                detail = {
                    "id": item.id,
                    "correct": is_correct,
                    "predicted": predicted,
                    "expected": item.expected,
                    "raw_answer": raw_answer[:200],
                    "steps": steps,
                }
                if routed:
                    detail["routed"] = routed
                if baseline_conf is not None:
                    detail["baseline_confidence"] = round(baseline_conf, 4)
                details.append(detail)

                done = len(details)
                acc_so_far = correct / done if done > 0 else 0
                status = "OK" if is_correct else "WRONG"
                route_str = f" route={routed}" if routed else ""
                conf_str = f" conf={baseline_conf:.3f}" if baseline_conf is not None else ""
                print(f"  [{done}/{len(items)}] [{status}] {item.id}: "
                      f"pred={predicted} exp={item.expected} steps={steps}"
                      f"{route_str}{conf_str} "
                      f"(acc={acc_so_far:.1%})")

            except Exception as e:
                print(f"  [ERROR] {item.id}: {e}")
                import traceback
                traceback.print_exc()
                details.append({
                    "id": item.id,
                    "correct": False,
                    "predicted": "",
                    "expected": item.expected,
                    "raw_answer": f"ERROR: {str(e)[:100]}",
                    "steps": 0,
                    "error": True,
                })

            if len(details) % 10 == 0:
                _save_results(results_path, existing, config_key, config_name,
                              cfg, details, correct, total_steps, benchmark_name)

            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        elapsed = time.time() - t_start
        n_done = len(details)
        accuracy = correct / n_done if n_done > 0 else 0
        avg_steps = total_steps / n_done if n_done > 0 else 0

        all_results[config_name] = {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "correct": correct,
            "total": n_done,
            "elapsed_s": elapsed,
        }

        _save_results(results_path, existing, config_key, config_name,
                      cfg, details, correct, total_steps, benchmark_name)

        print(f"\n  {config_name} done: {accuracy:.1%} accuracy, "
              f"{avg_steps:.1f} avg steps, {elapsed:.0f}s")

    return all_results


def _save_results(results_path, existing, config_key, config_name,
                  cfg, details, correct, total_steps, benchmark_name):
    n_done = len(details)
    existing[config_key] = {
        "config": config_name,
        "description": cfg["description"],
        "accuracy": correct / n_done if n_done > 0 else 0,
        "avg_steps": total_steps / n_done if n_done > 0 else 0,
        "correct": correct,
        "total": n_done,
        "details": details,
    }
    existing["metadata"] = {
        "experiment": "exp7b_kv_generation",
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Exp 7B: Full-Layer KV Generation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run on 20 eval prompts")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "arc"])
    parser.add_argument("--configs", type=str, default=",".join(ALL_CONFIGS))
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]

    print("=" * 60)
    print("Experiment 7B: Full-Layer KV Recurrence + KV-Aware Generation")
    print("=" * 60)
    print(f"  Configs: {configs}")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    eval_results = run_eval_prompts(model, tokenizer, configs)

    eval_path = RESULTS_DIR / "exp7b_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved eval results: {eval_path}")

    if args.eval_only:
        return

    print(f"\nLoading {args.benchmark} benchmark...")
    items = load_benchmark(args.benchmark, subset_n=args.subset, seed=args.seed)
    print(f"  Loaded {len(items)} items")

    bench_results = run_benchmark(
        model, tokenizer, args.benchmark, configs, items, RESULTS_DIR,
    )

    print(f"\n{'='*70}")
    print(f"SUMMARY - {args.benchmark.upper()}")
    print(f"{'='*70}")
    print(f"{'Config':>12} | {'Description':<42} | {'Acc':>6} | {'Steps':>6}")
    print("-" * 70)
    for config_name in configs:
        if config_name in bench_results:
            r = bench_results[config_name]
            desc = CONFIGS[config_name]["description"]
            print(f"{config_name:>12} | {desc:<42} | {r['accuracy']:>5.1%} | "
                  f"{r['avg_steps']:>5.1f}")
    print("=" * 70)
    print("\nDone!")


if __name__ == "__main__":
    main()
