"""
Experiment 7A: KV-Cache Recurrence — Latent reasoning with prompt attention.

Tests whether maintaining the prompt's KV cache during recurrence (so thought
tokens can attend to the original question) improves over:
  1. No recurrence (baseline)
  2. Single-vector recurrence without KV cache (Phase 1 approach)
  3. Beam search without KV cache (exp6a)

Configurations:
| Config    | Type       | Steps/Depth | Description                     |
|-----------|-----------|-------------|----------------------------------|
| KV-0      | Baseline  | 0           | No recurrence                    |
| KV-A      | KV recur  | 4           | 4 thought steps with KV cache    |
| KV-B      | KV recur  | 8           | 8 thought steps                  |
| KV-C      | KV recur  | 16          | 16 thought steps                 |
| KV-D      | KV recur  | 32          | 32 thought steps                 |
| KV-E      | Old recur | 32          | Old single-vector (no KV cache)  |
| KV-BEAM-A | KV beam   | 3x5 d=4    | Beam search WITH KV cache        |

Usage:
    python experiments/exp7a_kv_recurrence.py --eval-only
    python experiments/exp7a_kv_recurrence.py --benchmark gsm8k --subset 50
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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RESULTS_DIR, DEVICE, MID_LAYER_INDEX,
    BENCHMARK_GSM8K_MAX_TOKENS, BENCHMARK_ARC_MAX_TOKENS,
)
from model_utils import load_model
from kv_recurrence import kv_recurrence, kv_beam_search
from continuous_recurrence import continuous_recurrence
from benchmarks import load_benchmark, extract_answer, check_answer
from eval_prompts import PROMPTS as EVAL_PROMPTS, check_answer as eval_check_answer


CONFIGS = {
    "KV-0": {
        "description": "Baseline (no recurrence)",
        "type": "baseline",
    },
    "KV-A": {
        "description": "KV recurrence, 4 steps",
        "type": "kv_recur", "n_steps": 4,
    },
    "KV-B": {
        "description": "KV recurrence, 8 steps",
        "type": "kv_recur", "n_steps": 8,
    },
    "KV-C": {
        "description": "KV recurrence, 16 steps",
        "type": "kv_recur", "n_steps": 16,
    },
    "KV-D": {
        "description": "KV recurrence, 32 steps",
        "type": "kv_recur", "n_steps": 32,
    },
    "KV-E": {
        "description": "Old single-vector recurrence, 32 steps (no KV)",
        "type": "old_recur", "n_steps": 32,
    },
    "KV-BEAM-A": {
        "description": "KV beam search, 3x5 depth=4 alpha=0.1",
        "type": "kv_beam", "beam_width": 3, "branch_factor": 5,
        "max_depth": 4, "injection_alpha": 0.1,
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

    elif ctype == "kv_recur":
        result = kv_recurrence(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
        )
        return result

    elif ctype == "old_recur":
        result = continuous_recurrence(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            n_steps=cfg["n_steps"],
            max_new_tokens=max_tokens,
        )
        return result

    elif ctype == "kv_beam":
        result = kv_beam_search(
            model, tokenizer,
            context_text=prompt_text,
            query_text=prompt_text,
            beam_width=cfg["beam_width"],
            branch_factor=cfg["branch_factor"],
            max_depth=cfg["max_depth"],
            injection_alpha=cfg["injection_alpha"],
            max_new_tokens=max_tokens,
        )
        return {
            "answer": result["answer"],
            "n_steps_taken": result["total_forward_calls"],
        }


def run_eval_prompts(model, tokenizer, configs_to_run):
    """Run on 20 eval prompts."""
    print("\n" + "=" * 60)
    print("Phase 1: Eval Prompts (20 prompts)")
    print("=" * 60)

    results = {}

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            continue

        # Force cleanup between configs
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
    print(f"\n{'='*65}")
    print(f"{'Config':>12} | {'Description':<40} | {'Acc':>6} | {'Steps':>6}")
    print("-" * 65)
    for config_name in configs_to_run:
        if config_name in results:
            r = results[config_name]
            desc = CONFIGS[config_name]["description"]
            print(f"{config_name:>12} | {desc:<40} | {r['accuracy']:>5.1%} | "
                  f"{r['avg_steps']:>5.1f}")
    print("=" * 65)

    return results


def run_benchmark(model, tokenizer, benchmark_name, configs_to_run, items, output_dir):
    """Run on benchmark dataset."""
    max_tokens = (
        BENCHMARK_GSM8K_MAX_TOKENS if benchmark_name == "gsm8k"
        else BENCHMARK_ARC_MAX_TOKENS
    )

    all_results = {}
    results_path = output_dir / f"exp7a_{benchmark_name}_results.json"

    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            continue

        # Force cleanup between configs to prevent OOM
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

                detail = {
                    "id": item.id,
                    "correct": is_correct,
                    "predicted": predicted,
                    "expected": item.expected,
                    "raw_answer": raw_answer[:200],
                    "steps": steps,
                }
                details.append(detail)

                done = len(details)
                acc_so_far = correct / done if done > 0 else 0
                status = "OK" if is_correct else "WRONG"
                print(f"  [{done}/{len(items)}] [{status}] {item.id}: "
                      f"pred={predicted} exp={item.expected} steps={steps} "
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
        "experiment": "exp7a_kv_recurrence",
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Exp 7A: KV-Cache Recurrence")
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
    print("Experiment 7A: KV-Cache Recurrence")
    print("=" * 60)
    print(f"  Configs: {configs}")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    eval_results = run_eval_prompts(model, tokenizer, configs)

    eval_path = RESULTS_DIR / "exp7a_eval_results.json"
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

    print(f"\n{'='*65}")
    print(f"SUMMARY — {args.benchmark.upper()}")
    print(f"{'='*65}")
    print(f"{'Config':>12} | {'Description':<40} | {'Acc':>6} | {'Steps':>6}")
    print("-" * 65)
    for config_name in configs:
        if config_name in bench_results:
            r = bench_results[config_name]
            desc = CONFIGS[config_name]["description"]
            print(f"{config_name:>12} | {desc:<40} | {r['accuracy']:>5.1%} | "
                  f"{r['avg_steps']:>5.1f}")
    print("=" * 65)
    print("\nDone!")


if __name__ == "__main__":
    main()
