"""
Experiment 5A: Full benchmark ablation across 9 configurations.

Runs GSM8K (1319 test) and ARC Challenge (1172 test) across:

| Config | Loop         | Halt Gate        | Memory | Description           |
|--------|-------------|------------------|--------|-----------------------|
| A      | None        | None             | None   | Normal prompting      |
| B      | Fixed N=32  | None             | None   | Phase 1 approach      |
| C      | Continuous  | Heuristic @0.6   | None   | Phase 2 best          |
| D      | Continuous  | Learned (RL)     | None   | Phase 3               |
| E      | Continuous  | Learned          | KV buf | Phase 4 Tier 1        |
| F      | Continuous  | Learned          | Neural | Phase 4 Tier 3        |
| G      | Continuous  | Learned+MemGate  | KV     | Full system           |
| H      | Text CoT    | None             | None   | Text baseline (128t)  |
| I      | Fixed R=3   | None             | None   | Lys et al. approx     |

Usage:
    python experiments/exp5a_ablation.py --benchmark gsm8k --configs A,B,C --subset 50
    python experiments/exp5a_ablation.py --benchmark arc --configs A,B,C,D,E,F,G,H,I
    python experiments/exp5a_ablation.py --benchmark gsm8k  # full run, all configs
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
    MAX_CONTINUOUS_STEPS, HIDDEN_DIM,
    BENCHMARK_GSM8K_MAX_TOKENS, BENCHMARK_ARC_MAX_TOKENS,
    BENCHMARK_DEFAULT_SUBSET, LYS_MID_LAYER_INDEX,
)
from model_utils import load_model, get_embeddings
from continuous_recurrence import continuous_recurrence
from recurrence import text_baseline
from gates import HaltGate, MemoryGate
from gates_heuristic import ConfidenceHalt
from memory import create_memory
from benchmarks import (
    load_benchmark, extract_answer, check_answer,
    ThresholdedHaltGate,
)


# ── Config registry ──────────────────────────────────────────────────


ALL_CONFIGS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

CONFIG_DESCRIPTIONS = {
    "A": "Normal prompting (no recurrence)",
    "B": "Fixed N=32 mid-layer loop",
    "C": "Continuous + heuristic confidence@0.6",
    "D": "Continuous + learned RL gate",
    "E": "Continuous + learned gate + KV memory",
    "F": "Continuous + learned gate + neural memory",
    "G": "Continuous + learned gate + MemoryGate + KV",
    "H": "Text CoT baseline (128 tokens)",
    "I": "Lys et al. approx (R=3, layer 18)",
}


def _load_halt_gate():
    """Load the RL-trained halt gate."""
    gate_path = RESULTS_DIR / "halt_gate_rl.pt"
    gate = HaltGate(hidden_dim=HIDDEN_DIM).to(DEVICE)
    gate.load_state_dict(torch.load(gate_path, map_location=DEVICE, weights_only=True))
    gate.eval()
    return gate


def _load_memory_gate():
    """Load the RL-trained memory gate."""
    gate_path = RESULTS_DIR / "memory_gate_rl.pt"
    gate = MemoryGate(hidden_dim=HIDDEN_DIM).to(DEVICE)
    gate.load_state_dict(torch.load(gate_path, map_location=DEVICE, weights_only=True))
    gate.eval()
    return gate


# ── Runner functions ─────────────────────────────────────────────────


def run_config_A(model, tokenizer, item, max_tokens):
    """Config A: direct model.generate(), no recurrence."""
    inputs = tokenizer(item.prompt, return_tensors="pt").to(DEVICE)
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


def run_config_B(model, tokenizer, item, max_tokens):
    """Config B: fixed N=32 mid-layer loop."""
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        n_steps=32,
        max_new_tokens=max_tokens,
    )
    return result


def run_config_C(model, tokenizer, item, max_tokens):
    """Config C: continuous + heuristic confidence@0.6."""
    halt = ConfidenceHalt(model, threshold=0.6, min_steps=1)
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        halt_fn=halt,
        max_steps=MAX_CONTINUOUS_STEPS,
        max_new_tokens=max_tokens,
    )
    return result


def run_config_D(model, tokenizer, item, max_tokens, halt_gate=None, halt_threshold=0.5):
    """Config D: continuous + learned RL halt gate."""
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        halt_fn=halt_gate,
        max_steps=MAX_CONTINUOUS_STEPS,
        max_new_tokens=max_tokens,
        halt_threshold=halt_threshold,
    )
    return result


def run_config_E(model, tokenizer, item, max_tokens, halt_gate=None, memory=None, halt_threshold=0.5):
    """Config E: continuous + learned gate + KV memory (no memory gate)."""
    if memory is not None:
        memory.reset()
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        halt_fn=halt_gate,
        max_steps=MAX_CONTINUOUS_STEPS,
        max_new_tokens=max_tokens,
        memory=memory,
        halt_threshold=halt_threshold,
    )
    return result


def run_config_F(model, tokenizer, item, max_tokens, halt_gate=None, memory=None, halt_threshold=0.5):
    """Config F: continuous + learned gate + neural memory (no memory gate)."""
    if memory is not None:
        memory.reset()
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        halt_fn=halt_gate,
        max_steps=MAX_CONTINUOUS_STEPS,
        max_new_tokens=max_tokens,
        memory=memory,
        halt_threshold=halt_threshold,
    )
    return result


def run_config_G(model, tokenizer, item, max_tokens, halt_gate=None, memory=None, memory_gate=None, halt_threshold=0.5):
    """Config G: continuous + learned gate + MemoryGate + KV memory."""
    if memory is not None:
        memory.reset()
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        halt_fn=halt_gate,
        max_steps=MAX_CONTINUOUS_STEPS,
        max_new_tokens=max_tokens,
        memory=memory,
        memory_gate=memory_gate,
        halt_threshold=halt_threshold,
    )
    return result


def run_config_H(model, tokenizer, item, max_tokens):
    """Config H: text CoT baseline (128 thinking tokens)."""
    result = text_baseline(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        n_thinking_tokens=128,
        max_new_tokens=max_tokens,
    )
    return {"answer": result["answer"], "n_steps_taken": 128}


def run_config_I(model, tokenizer, item, max_tokens):
    """Config I: Lys et al. approx — R=3 loops at 50% depth (layer 18)."""
    result = continuous_recurrence(
        model, tokenizer,
        context_text=item.prompt,
        query_text=item.prompt,
        n_steps=3,
        mid_layer=LYS_MID_LAYER_INDEX,
        max_new_tokens=max_tokens,
    )
    return result


# ── Main experiment runner ───────────────────────────────────────────


def run_ablation(
    model,
    tokenizer,
    benchmark_name: str,
    configs: list,
    items: list,
    output_dir: Path,
    halt_threshold: float = 0.5,
):
    """Run the full ablation across specified configs and items."""
    max_tokens = (
        BENCHMARK_GSM8K_MAX_TOKENS if benchmark_name == "gsm8k"
        else BENCHMARK_ARC_MAX_TOKENS
    )

    # Pre-load shared resources
    halt_gate = None
    memory_gate = None
    kv_memory = None
    neural_memory = None

    needs_halt = set(configs) & {"D", "E", "F", "G"}
    needs_mem_gate = "G" in configs
    needs_kv = set(configs) & {"E", "G"}
    needs_neural = "F" in configs

    if needs_halt:
        print("  Loading RL halt gate...")
        halt_gate = _load_halt_gate()

    if needs_mem_gate:
        print("  Loading RL memory gate...")
        memory_gate = _load_memory_gate()

    if needs_kv:
        print("  Creating KV memory...")
        kv_memory = create_memory("kv", hidden_dim=HIDDEN_DIM, device=DEVICE)

    if needs_neural:
        print("  Creating neural memory...")
        neural_memory = create_memory("neural", hidden_dim=HIDDEN_DIM, device=DEVICE)

    # Results storage
    all_results = {}
    results_path = output_dir / f"exp5a_{benchmark_name}_results.json"

    # Load existing results for resume
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)
        print(f"  Loaded existing results from {results_path}")

    for config in configs:
        config_key = f"config_{config}"
        print(f"\n{'='*60}")
        print(f"Config {config}: {CONFIG_DESCRIPTIONS[config]}")
        print(f"{'='*60}")

        # Resume: skip completed items
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
                if config == "A":
                    result = run_config_A(model, tokenizer, item, max_tokens)
                elif config == "B":
                    result = run_config_B(model, tokenizer, item, max_tokens)
                elif config == "C":
                    result = run_config_C(model, tokenizer, item, max_tokens)
                elif config == "D":
                    result = run_config_D(model, tokenizer, item, max_tokens, halt_gate, halt_threshold)
                elif config == "E":
                    result = run_config_E(model, tokenizer, item, max_tokens, halt_gate, kv_memory, halt_threshold)
                elif config == "F":
                    result = run_config_F(model, tokenizer, item, max_tokens, halt_gate, neural_memory, halt_threshold)
                elif config == "G":
                    result = run_config_G(model, tokenizer, item, max_tokens, halt_gate, kv_memory, memory_gate, halt_threshold)
                elif config == "H":
                    result = run_config_H(model, tokenizer, item, max_tokens)
                elif config == "I":
                    result = run_config_I(model, tokenizer, item, max_tokens)
                else:
                    raise ValueError(f"Unknown config: {config}")

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
                details.append({
                    "id": item.id,
                    "correct": False,
                    "predicted": "",
                    "expected": item.expected,
                    "raw_answer": f"ERROR: {str(e)[:100]}",
                    "steps": 0,
                    "error": True,
                })

            # Periodic save every 10 items
            if len(details) % 10 == 0:
                _save_config_results(
                    results_path, existing, config_key, config,
                    details, correct, total_steps, len(items), benchmark_name,
                )

            # Memory cleanup every 100 items
            if (idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        elapsed = time.time() - t_start
        n_done = len(details)
        accuracy = correct / n_done if n_done > 0 else 0
        avg_steps = total_steps / n_done if n_done > 0 else 0

        all_results[config] = {
            "accuracy": accuracy,
            "avg_steps": avg_steps,
            "correct": correct,
            "total": n_done,
            "elapsed_s": elapsed,
            "details": details,
        }

        # Final save
        _save_config_results(
            results_path, existing, config_key, config,
            details, correct, total_steps, n_done, benchmark_name,
        )

        print(f"\n  Config {config} done: {accuracy:.1%} accuracy, "
              f"{avg_steps:.1f} avg steps, {elapsed:.0f}s")

    return all_results


def _save_config_results(
    results_path, existing, config_key, config,
    details, correct, total_steps, total, benchmark_name,
):
    """Save intermediate results to JSON (resumable)."""
    n_done = len(details)
    existing[config_key] = {
        "config": config,
        "description": CONFIG_DESCRIPTIONS[config],
        "accuracy": correct / n_done if n_done > 0 else 0,
        "avg_steps": total_steps / n_done if n_done > 0 else 0,
        "correct": correct,
        "total": n_done,
        "details": details,
    }
    existing["metadata"] = {
        "experiment": "exp5a_ablation",
        "benchmark": benchmark_name,
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


# ── Output generation ────────────────────────────────────────────────


def print_table(all_results, benchmark_name):
    """Print ASCII ablation table."""
    print(f"\n{'='*75}")
    print(f"ABLATION TABLE — {benchmark_name.upper()}")
    print(f"{'='*75}")
    print(f"{'Config':>8} | {'Description':<40} | {'Acc':>7} | {'Steps':>7} | {'N':>5}")
    print("-" * 75)

    for config in ALL_CONFIGS:
        if config not in all_results:
            continue
        r = all_results[config]
        print(f"    {config:>4} | {CONFIG_DESCRIPTIONS[config]:<40} | "
              f"{r['accuracy']:>6.1%} | {r['avg_steps']:>6.1f} | {r['total']:>5}")

    print("=" * 75)


def print_latex_table(all_results, benchmark_name):
    """Print LaTeX ablation table."""
    print(f"\n% LaTeX table for {benchmark_name}")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\begin{tabular}{clccc}")
    print(r"\toprule")
    print(r"Config & Description & Accuracy & Avg Steps & N \\")
    print(r"\midrule")

    for config in ALL_CONFIGS:
        if config not in all_results:
            continue
        r = all_results[config]
        desc = CONFIG_DESCRIPTIONS[config].replace("&", r"\&")
        print(f"{config} & {desc} & {r['accuracy']:.1%} & {r['avg_steps']:.1f} & {r['total']} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(f"\\caption{{Ablation results on {benchmark_name.upper()}}}")
    print(f"\\label{{tab:ablation_{benchmark_name}}}")
    print(r"\end{table}")


def plot_bar_chart(all_results, benchmark_name, save_path):
    """Bar chart of accuracy by config."""
    configs = [c for c in ALL_CONFIGS if c in all_results]
    accs = [all_results[c]["accuracy"] * 100 for c in configs]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(configs)))
    bars = ax.bar(configs, accs, color=colors, edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Ablation: {benchmark_name.upper()} Accuracy by Configuration")
    ax.set_ylim(0, max(accs) * 1.15 if accs else 100)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


def plot_pareto(all_results, benchmark_name, save_path):
    """Pareto plot: accuracy vs average steps (compute)."""
    configs = [c for c in ALL_CONFIGS if c in all_results]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors_map = {
        "A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728",
        "E": "#9467bd", "F": "#8c564b", "G": "#e377c2", "H": "#7f7f7f",
        "I": "#bcbd22",
    }
    markers_map = {
        "A": "o", "B": "s", "C": "^", "D": "D",
        "E": "v", "F": "P", "G": "*", "H": "X",
        "I": "p",
    }

    for config in configs:
        r = all_results[config]
        ax.scatter(
            r["avg_steps"], r["accuracy"] * 100,
            s=150, marker=markers_map.get(config, "o"),
            color=colors_map.get(config, "gray"),
            edgecolors="black", linewidth=0.5, zorder=3,
        )
        ax.annotate(
            f" {config}", (r["avg_steps"], r["accuracy"] * 100),
            fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Average Recurrence Steps (Compute)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Pareto: Accuracy vs Compute — {benchmark_name.upper()}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Phase 5 ablation experiment")
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=["gsm8k", "arc"],
                        help="Benchmark to run (default: gsm8k)")
    parser.add_argument("--configs", type=str, default=",".join(ALL_CONFIGS),
                        help="Comma-separated list of configs to run (default: all)")
    parser.add_argument("--subset", type=int, default=None,
                        help="Number of items to use (default: full dataset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for subset selection")
    parser.add_argument("--halt-threshold", type=float, default=0.5,
                        help="Halt threshold for RL gate configs D/E/F/G (default: 0.5)")
    args = parser.parse_args()

    configs = [c.strip().upper() for c in args.configs.split(",")]
    for c in configs:
        if c not in ALL_CONFIGS:
            print(f"Error: unknown config '{c}'. Choose from {ALL_CONFIGS}")
            sys.exit(1)

    print("=" * 60)
    print("Experiment 5A: Full Benchmark Ablation")
    print("=" * 60)
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Configs: {configs}")
    print(f"  Subset: {args.subset or 'full'}")
    print(f"  Seed: {args.seed}")
    print(f"  Halt threshold: {args.halt_threshold}")
    print()

    # Load benchmark
    print("Loading benchmark data...")
    items = load_benchmark(args.benchmark, subset_n=args.subset, seed=args.seed)
    print(f"  Loaded {len(items)} items")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    # Output directory
    output_dir = RESULTS_DIR
    output_dir.mkdir(exist_ok=True)

    # Run ablation
    all_results = run_ablation(
        model, tokenizer, args.benchmark, configs, items, output_dir,
        halt_threshold=args.halt_threshold,
    )

    # Print tables
    print_table(all_results, args.benchmark)
    print_latex_table(all_results, args.benchmark)

    # Generate plots
    tag = f"{args.benchmark}"
    if args.subset:
        tag += f"_n{args.subset}"

    plot_bar_chart(
        all_results, args.benchmark,
        save_path=output_dir / f"exp5a_{tag}_bar.png",
    )
    plot_pareto(
        all_results, args.benchmark,
        save_path=output_dir / f"exp5a_{tag}_pareto.png",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
