"""
Experiment 2A: Long-horizon stability test.

Tests the continuous recurrence loop at N=64, 128, 256, 512 steps on 20 prompts.
Checks for:
- NaN/Inf detection
- Bounded norms (no explosion)
- Cosine similarity convergence behavior
- Plots per-prompt trajectories

Success criteria: loop stable for 256+ steps (no NaN, bounded norms).
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from config import RESULTS_DIR, MID_LAYER_INDEX
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from continuous_recurrence import continuous_recurrence_trajectory
from eval_prompts import get_prompts
from plotting import plot_cosine_similarity, plot_norms, plot_pca_trajectory

if WANDB_ENABLED:
    import wandb

STEP_COUNTS = [64, 128, 256, 512]


def run_experiment():
    print("=" * 60)
    print("Experiment 2A: Long-Horizon Stability Test")
    print(f"Steps: {STEP_COUNTS}")
    print(f"Mid-layer: {MID_LAYER_INDEX}")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp2a_long_horizon")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    prompts = get_prompts()
    results = {}

    for max_steps in STEP_COUNTS:
        print(f"\n--- Testing {max_steps} steps ---")
        step_results = []
        nan_count = 0

        for prompt_data in prompts:
            print(f"  {prompt_data['id']}...", end=" ")

            traj = continuous_recurrence_trajectory(
                model, tokenizer,
                context_text=prompt_data["prompt"],
                max_steps=max_steps,
                mid_layer=MID_LAYER_INDEX,
            )

            step_results.append({
                "id": prompt_data["id"],
                "steps_completed": traj["steps_completed"],
                "nan_detected": traj["nan_detected"],
                "final_norm": traj["norms"][-1] if traj["norms"] else None,
                "norm_range": [min(traj["norms"]), max(traj["norms"])] if traj["norms"] else None,
                "final_cos_sim": traj["cosine_sims"][-1] if traj["cosine_sims"] else None,
                "cos_sim_mean": float(np.mean(traj["cosine_sims"])) if traj["cosine_sims"] else None,
                "norms": traj["norms"],
                "cosine_sims": traj["cosine_sims"],
            })

            if traj["nan_detected"]:
                nan_count += 1
                print(f"NaN at step {traj['steps_completed']}!")
            else:
                print(f"OK (norm: {traj['norms'][-1]:.1f}, cos: {traj['cosine_sims'][-1]:.4f})")

        # Summary
        all_final_norms = [r["final_norm"] for r in step_results if r["final_norm"] is not None]
        all_cos_sims = [r["final_cos_sim"] for r in step_results if r["final_cos_sim"] is not None]

        results[max_steps] = {
            "step_results": step_results,
            "nan_count": nan_count,
            "total_prompts": len(prompts),
            "stable_count": len(prompts) - nan_count,
            "avg_final_norm": float(np.mean(all_final_norms)) if all_final_norms else None,
            "std_final_norm": float(np.std(all_final_norms)) if all_final_norms else None,
            "avg_final_cos_sim": float(np.mean(all_cos_sims)) if all_cos_sims else None,
        }

        print(f"\n  Summary ({max_steps} steps):")
        print(f"    Stable: {len(prompts) - nan_count}/{len(prompts)}")
        if all_final_norms:
            print(f"    Final norm: {np.mean(all_final_norms):.1f} +/- {np.std(all_final_norms):.1f}")
        if all_cos_sims:
            print(f"    Final cos_sim: {np.mean(all_cos_sims):.4f}")

        if WANDB_ENABLED:
            wandb.log({
                "max_steps": max_steps,
                "stable_count": len(prompts) - nan_count,
                "avg_final_norm": np.mean(all_final_norms) if all_final_norms else 0,
            })

    # Generate aggregate plots
    print("\nGenerating plots...")

    for max_steps in STEP_COUNTS:
        step_data = results[max_steps]["step_results"]
        stable = [r for r in step_data if not r["nan_detected"]]

        if not stable:
            continue

        # Average cosine similarities across stable prompts
        min_len = min(len(r["cosine_sims"]) for r in stable)
        if min_len > 0:
            avg_cos = np.mean(
                [r["cosine_sims"][:min_len] for r in stable],
                axis=0
            ).tolist()
            plot_cosine_similarity(
                avg_cos,
                title=f"Avg Cosine Similarity (N={max_steps}, {len(stable)} prompts)",
                save_path=RESULTS_DIR / f"exp2a_cos_sim_n{max_steps}.png",
            )

        # Average norms
        min_len_norms = min(len(r["norms"]) for r in stable)
        if min_len_norms > 0:
            avg_norms = np.mean(
                [r["norms"][:min_len_norms] for r in stable],
                axis=0
            ).tolist()
            plot_norms(
                avg_norms,
                title=f"Avg Hidden State Norms (N={max_steps}, {len(stable)} prompts)",
                save_path=RESULTS_DIR / f"exp2a_norms_n{max_steps}.png",
            )

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY: Long-Horizon Stability")
    print("=" * 60)
    print(f"{'Steps':>8} | {'Stable':>8} | {'Avg Norm':>10} | {'Avg CosSim':>10}")
    print("-" * 45)
    for max_steps in STEP_COUNTS:
        r = results[max_steps]
        norm_str = f"{r['avg_final_norm']:.1f}" if r["avg_final_norm"] else "N/A"
        cos_str = f"{r['avg_final_cos_sim']:.4f}" if r["avg_final_cos_sim"] else "N/A"
        print(f"{max_steps:>8} | {r['stable_count']:>5}/{r['total_prompts']:<2} | {norm_str:>10} | {cos_str:>10}")

    # Save results (without full trajectory data for JSON size)
    output_path = RESULTS_DIR / "exp2a_long_horizon.json"
    serializable = {}
    for max_steps in STEP_COUNTS:
        r = results[max_steps]
        serializable[str(max_steps)] = {
            "nan_count": r["nan_count"],
            "total_prompts": r["total_prompts"],
            "stable_count": r["stable_count"],
            "avg_final_norm": r["avg_final_norm"],
            "std_final_norm": r["std_final_norm"],
            "avg_final_cos_sim": r["avg_final_cos_sim"],
            "per_prompt": [
                {
                    "id": sr["id"],
                    "steps_completed": sr["steps_completed"],
                    "nan_detected": sr["nan_detected"],
                    "final_norm": sr["final_norm"],
                    "final_cos_sim": sr["final_cos_sim"],
                    "cos_sim_mean": sr["cos_sim_mean"],
                }
                for sr in r["step_results"]
            ],
        }

    serializable["metadata"] = {
        "experiment": "exp2a_long_horizon",
        "timestamp": datetime.now().isoformat(),
        "step_counts": STEP_COUNTS,
        "mid_layer_index": MID_LAYER_INDEX,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return results


if __name__ == "__main__":
    run_experiment()
