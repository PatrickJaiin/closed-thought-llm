"""
Experiment 6A: Latent Beam Search — Tree-of-Thoughts in hidden space.

Tests latent beam search across multiple configurations on eval prompts
and benchmarks.

Configurations:
| Config | beam_width | branch_factor | max_depth | conf_threshold | alpha |
|--------|-----------|---------------|-----------|----------------|-------|
| BS-A   | 1         | 1             | 8         | 0.9            | 1.0   |
| BS-B   | 3         | 5             | 4         | 0.9            | 1.0   |
| BS-C   | 3         | 5             | 8         | 0.9            | 1.0   |
| BS-D   | 5         | 10            | 4         | 0.9            | 1.0   |
| BS-E   | 5         | 10            | 8         | 0.85           | 1.0   |
| BS-F   | 3         | 5             | 8         | 0.8            | 1.0   |
| BS-G   | 3         | 5             | 8         | 0.9            | 0.5   |
| BS-H   | 3         | 5             | 8         | 0.9            | 0.1   |

Usage:
    python experiments/exp6a_latent_beam_search.py --eval-only
    python experiments/exp6a_latent_beam_search.py --benchmark gsm8k --configs BS-B,BS-C --subset 50
    python experiments/exp6a_latent_beam_search.py --benchmark arc --subset 50
"""

import sys
import json
import argparse
import time
import gc
import math
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
from latent_beam_search import latent_beam_search, run_beam_search_on_item
from continuous_recurrence import _generate_with_prefix_state
from benchmarks import load_benchmark, extract_answer, check_answer
from eval_prompts import PROMPTS as EVAL_PROMPTS, check_answer as eval_check_answer


# ── Config registry ──────────────────────────────────────────────────

CONFIGS = {
    "BS-A": {
        "description": "Greedy single-path, no early halt",
        "beam_width": 1, "branch_factor": 1, "max_depth": 8,
        "confidence_threshold": 1.0, "injection_alpha": 0.1,
    },
    "BS-B": {
        "description": "3x5 depth=4 alpha=0.1 (gentle nudge)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 4,
        "confidence_threshold": 1.0, "injection_alpha": 0.1,
    },
    "BS-C": {
        "description": "3x5 depth=8 alpha=0.1 (gentle, deep)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 8,
        "confidence_threshold": 1.0, "injection_alpha": 0.1,
    },
    "BS-D": {
        "description": "3x5 depth=4 alpha=0.3 (moderate nudge)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 4,
        "confidence_threshold": 1.0, "injection_alpha": 0.3,
    },
    "BS-E": {
        "description": "3x5 depth=8 alpha=0.3 (moderate, deep)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 8,
        "confidence_threshold": 1.0, "injection_alpha": 0.3,
    },
    "BS-F": {
        "description": "3x5 depth=4 alpha=0.5 (strong nudge)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 4,
        "confidence_threshold": 1.0, "injection_alpha": 0.5,
    },
    "BS-G": {
        "description": "5x10 depth=4 alpha=0.1 (wide, gentle)",
        "beam_width": 5, "branch_factor": 10, "max_depth": 4,
        "confidence_threshold": 1.0, "injection_alpha": 0.1,
    },
    "BS-H": {
        "description": "3x5 depth=4 alpha=0.05 (very subtle)",
        "beam_width": 3, "branch_factor": 5, "max_depth": 4,
        "confidence_threshold": 1.0, "injection_alpha": 0.05,
    },
}

ALL_CONFIGS = list(CONFIGS.keys())


# ── Eval prompt runner ────────────────────────────────────────────────


def run_eval_prompts(model, tokenizer, configs_to_run):
    """Run beam search on the 20 eval prompts for quick validation."""
    print("\n" + "=" * 60)
    print("Phase 1: Eval Prompts (20 prompts, quick sanity check)")
    print("=" * 60)

    # Baseline: direct generation (no recurrence)
    print("\n--- Baseline (no recurrence) ---")
    baseline_correct = 0
    for prompt_data in EVAL_PROMPTS:
        prompt = prompt_data["prompt"]
        expected = prompt_data["answer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        answer = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        is_correct = eval_check_answer(answer, expected)
        baseline_correct += int(is_correct)

    baseline_acc = baseline_correct / len(EVAL_PROMPTS)
    print(f"  Baseline: {baseline_correct}/{len(EVAL_PROMPTS)} = {baseline_acc:.1%}")

    # Run each beam search config
    results = {"baseline": {"accuracy": baseline_acc, "correct": baseline_correct, "total": len(EVAL_PROMPTS)}}

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"  Skipping unknown config: {config_name}")
            continue

        cfg = CONFIGS[config_name]
        print(f"\n--- {config_name}: {cfg['description']} ---")

        correct = 0
        total_fwd = 0
        total_depth = 0

        for i, prompt_data in enumerate(EVAL_PROMPTS):
            prompt = prompt_data["prompt"]
            expected = prompt_data["answer"]

            result = latent_beam_search(
                model, tokenizer,
                context_text=prompt,
                query_text=prompt,
                beam_width=cfg["beam_width"],
                branch_factor=cfg["branch_factor"],
                max_depth=cfg["max_depth"],
                confidence_threshold=cfg["confidence_threshold"],
                injection_alpha=cfg["injection_alpha"],
                max_new_tokens=64,
            )

            answer = result["answer"].strip()
            is_correct = eval_check_answer(answer, expected)
            correct += int(is_correct)
            total_fwd += result["total_forward_calls"]
            total_depth += result["depth_reached"]

            status = "OK" if is_correct else "WRONG"
            print(f"  [{i+1:2d}/20] [{status}] depth={result['depth_reached']} "
                  f"fwd={result['total_forward_calls']} "
                  f"conf={result['best_beam']['confidence']:.3f} "
                  f"halted={result['halted']}")

        acc = correct / len(EVAL_PROMPTS)
        avg_fwd = total_fwd / len(EVAL_PROMPTS)
        avg_depth = total_depth / len(EVAL_PROMPTS)

        results[config_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": len(EVAL_PROMPTS),
            "avg_forward_calls": avg_fwd,
            "avg_depth": avg_depth,
        }

        print(f"  {config_name}: {correct}/{len(EVAL_PROMPTS)} = {acc:.1%} "
              f"(avg fwd={avg_fwd:.1f}, avg depth={avg_depth:.1f})")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Config':>8} | {'Accuracy':>8} | {'Avg FWD':>8} | {'Avg Depth':>9}")
    print("-" * 60)
    print(f"{'baseline':>8} | {baseline_acc:>7.1%} | {'0':>8} | {'0':>9}")
    for config_name in configs_to_run:
        if config_name in results:
            r = results[config_name]
            print(f"{config_name:>8} | {r['accuracy']:>7.1%} | "
                  f"{r.get('avg_forward_calls', 0):>7.1f} | "
                  f"{r.get('avg_depth', 0):>8.1f}")
    print("=" * 60)

    return results


# ── Benchmark runner ─────────────────────────────────────────────────


def run_benchmark(model, tokenizer, benchmark_name, configs_to_run, items, output_dir):
    """Run beam search configs on a benchmark dataset."""
    max_tokens = (
        BENCHMARK_GSM8K_MAX_TOKENS if benchmark_name == "gsm8k"
        else BENCHMARK_ARC_MAX_TOKENS
    )

    all_results = {}
    results_path = output_dir / f"exp6a_{benchmark_name}_results.json"

    # Load existing results for resume
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        print(f"  Loaded existing results from {results_path}")

    for config_name in configs_to_run:
        if config_name not in CONFIGS:
            print(f"  Skipping unknown config: {config_name}")
            continue

        cfg = CONFIGS[config_name]
        config_key = f"config_{config_name}"

        print(f"\n{'='*60}")
        print(f"{config_name}: {cfg['description']}")
        print(f"{'='*60}")

        # Resume support
        completed_ids = set()
        if config_key in existing and "details" in existing[config_key]:
            completed_ids = {d["id"] for d in existing[config_key]["details"]}
            details = existing[config_key]["details"]
            print(f"  Resuming: {len(completed_ids)} already done")
        else:
            details = []

        correct = sum(1 for d in details if d["correct"])
        total_fwd = sum(d.get("forward_calls", 0) for d in details)

        t_start = time.time()

        for idx, item in enumerate(items):
            if item.id in completed_ids:
                continue

            try:
                result = run_beam_search_on_item(
                    model, tokenizer, item, max_tokens,
                    beam_width=cfg["beam_width"],
                    branch_factor=cfg["branch_factor"],
                    max_depth=cfg["max_depth"],
                    confidence_threshold=cfg["confidence_threshold"],
                    injection_alpha=cfg["injection_alpha"],
                )

                raw_answer = result["answer"]
                predicted = extract_answer(raw_answer, benchmark_name)
                is_correct = check_answer(predicted, item.expected, benchmark_name)
                fwd_calls = result.get("n_steps_taken", 0)

                correct += int(is_correct)
                total_fwd += fwd_calls

                detail = {
                    "id": item.id,
                    "correct": is_correct,
                    "predicted": predicted,
                    "expected": item.expected,
                    "raw_answer": raw_answer[:200],
                    "forward_calls": fwd_calls,
                    "depth_reached": result.get("depth_reached", 0),
                    "best_confidence": result.get("best_confidence", 0.0),
                    "halted": result.get("halted", False),
                }
                details.append(detail)

                done = len(details)
                acc_so_far = correct / done if done > 0 else 0
                status = "OK" if is_correct else "WRONG"
                print(f"  [{done}/{len(items)}] [{status}] {item.id}: "
                      f"pred={predicted} exp={item.expected} "
                      f"fwd={fwd_calls} depth={result.get('depth_reached', 0)} "
                      f"conf={result.get('best_confidence', 0):.3f} "
                      f"(acc={acc_so_far:.1%})")

            except Exception as e:
                print(f"  [ERROR] {item.id}: {e}")
                details.append({
                    "id": item.id,
                    "correct": False,
                    "predicted": "",
                    "expected": item.expected,
                    "raw_answer": f"ERROR: {str(e)[:100]}",
                    "forward_calls": 0,
                    "error": True,
                })

            # Periodic save every 10 items
            if len(details) % 10 == 0:
                _save_results(
                    results_path, existing, config_key, config_name, cfg,
                    details, correct, total_fwd, len(items), benchmark_name,
                )

            # Memory cleanup every 50 items
            if (idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        elapsed = time.time() - t_start
        n_done = len(details)
        accuracy = correct / n_done if n_done > 0 else 0
        avg_fwd = total_fwd / n_done if n_done > 0 else 0

        all_results[config_name] = {
            "accuracy": accuracy,
            "avg_forward_calls": avg_fwd,
            "correct": correct,
            "total": n_done,
            "elapsed_s": elapsed,
        }

        # Final save
        _save_results(
            results_path, existing, config_key, config_name, cfg,
            details, correct, total_fwd, n_done, benchmark_name,
        )

        print(f"\n  {config_name} done: {accuracy:.1%} accuracy, "
              f"{avg_fwd:.1f} avg forward calls, {elapsed:.0f}s")

    return all_results


def _save_results(
    results_path, existing, config_key, config_name, cfg,
    details, correct, total_fwd, total, benchmark_name,
):
    """Save intermediate results to JSON (resumable)."""
    n_done = len(details)
    existing[config_key] = {
        "config": config_name,
        "description": cfg["description"],
        "params": {k: v for k, v in cfg.items() if k != "description"},
        "accuracy": correct / n_done if n_done > 0 else 0,
        "avg_forward_calls": total_fwd / n_done if n_done > 0 else 0,
        "correct": correct,
        "total": n_done,
        "details": details,
    }
    existing["metadata"] = {
        "experiment": "exp6a_latent_beam_search",
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


# ── Plotting ─────────────────────────────────────────────────────────


def plot_bar_chart(all_results, benchmark_name, save_path, baseline_acc=None):
    """Bar chart of accuracy by config."""
    configs = list(all_results.keys())
    accs = [all_results[c]["accuracy"] * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    bars = ax.bar(configs, accs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=8)

    if baseline_acc is not None:
        ax.axhline(y=baseline_acc * 100, color="red", linestyle="--",
                   linewidth=1.5, label=f"Baseline: {baseline_acc*100:.1f}%")
        ax.legend()

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Latent Beam Search: {benchmark_name.upper()} Accuracy")
    ax.set_ylim(0, max(accs + [baseline_acc * 100 if baseline_acc else 0]) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pareto(all_results, benchmark_name, save_path, baseline_acc=None):
    """Pareto plot: accuracy vs forward calls (compute)."""
    configs = list(all_results.keys())

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, config in enumerate(configs):
        r = all_results[config]
        ax.scatter(
            r["avg_forward_calls"], r["accuracy"] * 100,
            s=150, zorder=3, edgecolors="black", linewidth=0.5,
        )
        ax.annotate(
            f" {config}", (r["avg_forward_calls"], r["accuracy"] * 100),
            fontsize=9, fontweight="bold",
        )

    if baseline_acc is not None:
        ax.axhline(y=baseline_acc * 100, color="red", linestyle="--",
                   linewidth=1.5, label=f"Baseline: {baseline_acc*100:.1f}%")
        ax.legend()

    ax.set_xlabel("Average Forward Calls (Compute)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Pareto: Accuracy vs Compute — {benchmark_name.upper()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Experiment 6A: Latent Beam Search")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run on 20 eval prompts (quick test)")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "arc"],
                        help="Benchmark to run (default: gsm8k)")
    parser.add_argument("--configs", type=str, default=",".join(ALL_CONFIGS),
                        help="Comma-separated list of configs (default: all)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of benchmark items (default: full)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",")]

    print("=" * 60)
    print("Experiment 6A: Latent Beam Search")
    print("=" * 60)
    print(f"  Configs: {configs}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    # Phase 1: Eval prompts
    eval_results = run_eval_prompts(model, tokenizer, configs)

    # Save eval results
    eval_path = RESULTS_DIR / "exp6a_eval_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nSaved eval results: {eval_path}")

    if args.eval_only:
        print("\n--eval-only: skipping benchmark runs.")
        return

    # Phase 2: Benchmark
    print(f"\nLoading {args.benchmark} benchmark...")
    items = load_benchmark(args.benchmark, subset_n=args.subset, seed=args.seed)
    print(f"  Loaded {len(items)} items")

    bench_results = run_benchmark(
        model, tokenizer, args.benchmark, configs, items, RESULTS_DIR,
    )

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY — {args.benchmark.upper()}")
    print(f"{'='*70}")
    print(f"{'Config':>8} | {'Description':<45} | {'Acc':>7} | {'FWD':>7}")
    print("-" * 70)
    for config_name in configs:
        if config_name in bench_results:
            r = bench_results[config_name]
            desc = CONFIGS[config_name]["description"]
            print(f"{config_name:>8} | {desc:<45} | {r['accuracy']:>6.1%} | "
                  f"{r['avg_forward_calls']:>6.1f}")
    print("=" * 70)

    # Plots
    tag = args.benchmark
    if args.subset:
        tag += f"_n{args.subset}"

    # Get baseline accuracy from Phase 5 results for comparison
    baseline_acc = None
    phase5_path = RESULTS_DIR / f"exp5a_{args.benchmark}_results.json"
    if phase5_path.exists():
        with open(phase5_path) as f:
            phase5 = json.load(f)
        if "config_A" in phase5:
            baseline_acc = phase5["config_A"]["accuracy"]
            print(f"\n  Phase 5 baseline (Config A): {baseline_acc:.1%}")

    plot_bar_chart(
        bench_results, args.benchmark,
        save_path=RESULTS_DIR / f"exp6a_{tag}_bar.png",
        baseline_acc=baseline_acc,
    )
    plot_pareto(
        bench_results, args.benchmark,
        save_path=RESULTS_DIR / f"exp6a_{tag}_pareto.png",
        baseline_acc=baseline_acc,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
