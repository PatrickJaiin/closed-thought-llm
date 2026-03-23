"""
Experiment 1C: Degeneration analysis.
Track per-step metrics during recurrence:
- Cosine similarity between consecutive hidden states
- L2 norm of hidden states
- PCA/t-SNE projection of hidden state trajectory
- Noise injection variant (small Gaussian noise each step)
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from config import RESULTS_DIR, NOISE_STD, MID_LAYER_INDEX, DEVICE
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model, get_embeddings, full_forward, partial_forward
from eval_prompts import get_prompts
from plotting import (
    plot_cosine_similarity,
    plot_norms,
    plot_pca_trajectory,
    plot_tsne_trajectory,
)

if WANDB_ENABLED:
    import wandb

MAX_RECURRENCE_STEPS = 64  # More steps to observe degeneration


def collect_trajectory(model, tokenizer, prompt_text, n_steps, mode="full", noise_std=0.0):
    """
    Run recurrence and collect hidden states at each step.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer.
        prompt_text: Input prompt.
        n_steps: Number of recurrence steps.
        mode: "full" for full-loop, "mid" for mid-layer loop.
        noise_std: Standard deviation of Gaussian noise to inject (0 = no noise).

    Returns:
        List of hidden state vectors (each shape: (hidden_dim,)).
    """
    trajectory = []

    with torch.no_grad():
        # Initial forward pass
        inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
        embeds = get_embeddings(model, inputs.input_ids)
        hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
        h = hidden[:, -1:, :]  # (1, 1, hidden_dim)
        trajectory.append(h.squeeze().cpu().float().numpy())

        for step in range(n_steps):
            if noise_std > 0:
                noise = torch.randn_like(h) * noise_std
                h = h + noise

            ones_mask = torch.ones(1, 1, device=DEVICE, dtype=torch.long)
            position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)

            if mode == "full":
                hidden = full_forward(model, h, attention_mask=ones_mask, position_ids=position_ids)
            elif mode == "mid":
                hidden = partial_forward(
                    model, h, start_layer=MID_LAYER_INDEX,
                    attention_mask=ones_mask, position_ids=position_ids,
                )
                # partial_forward already applies final norm
                trajectory.append(hidden.squeeze().cpu().float().numpy())
                h = hidden[:, -1:, :]
                continue
            else:
                raise ValueError(f"Unknown mode: {mode}")

            h = hidden[:, -1:, :]
            trajectory.append(h.squeeze().cpu().float().numpy())

    return trajectory


def compute_metrics(trajectory):
    """
    Compute per-step metrics from a trajectory of hidden state vectors.

    Returns:
        dict with:
            - cosine_sims: list of cosine similarities between consecutive steps
            - norms: list of L2 norms at each step
    """
    norms = [np.linalg.norm(h) for h in trajectory]

    cosine_sims = []
    for i in range(1, len(trajectory)):
        h_prev = trajectory[i - 1]
        h_curr = trajectory[i]
        cos = np.dot(h_prev, h_curr) / (np.linalg.norm(h_prev) * np.linalg.norm(h_curr) + 1e-8)
        cosine_sims.append(float(cos))

    return {
        "cosine_sims": cosine_sims,
        "norms": norms,
    }


def run_experiment():
    print("=" * 60)
    print("Experiment 1C: Degeneration Analysis")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp1c_degeneration")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    # Use a subset of prompts for detailed analysis
    prompts = get_prompts()[:5]
    all_results = {}

    for mode in ["full", "mid"]:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}-loop")
        print(f"{'='*40}")

        mode_results = {}

        for noise_label, noise_val in [("no_noise", 0.0), ("with_noise", NOISE_STD)]:
            print(f"\n  Noise: {noise_label} (std={noise_val})")
            noise_results = []

            for prompt_data in prompts:
                print(f"    Processing: {prompt_data['id']}...")
                trajectory = collect_trajectory(
                    model, tokenizer,
                    prompt_data["prompt"],
                    n_steps=MAX_RECURRENCE_STEPS,
                    mode=mode,
                    noise_std=noise_val,
                )

                metrics = compute_metrics(trajectory)
                noise_results.append({
                    "id": prompt_data["id"],
                    "metrics": metrics,
                    "trajectory": trajectory,
                })

                print(f"      Final norm: {metrics['norms'][-1]:.2f}, "
                      f"Final cos_sim: {metrics['cosine_sims'][-1]:.4f}")

            mode_results[noise_label] = noise_results

        all_results[mode] = mode_results

    # Generate plots
    print("\nGenerating plots...")

    for mode in ["full", "mid"]:
        for noise_label in ["no_noise", "with_noise"]:
            data = all_results[mode][noise_label]

            # Average metrics across prompts
            all_cosines = [d["metrics"]["cosine_sims"] for d in data]
            all_norms = [d["metrics"]["norms"] for d in data]

            avg_cosines = np.mean(all_cosines, axis=0).tolist()
            avg_norms = np.mean(all_norms, axis=0).tolist()

            prefix = f"{mode}_{noise_label}"

            plot_cosine_similarity(
                avg_cosines,
                title=f"Cosine Similarity ({mode}-loop, {noise_label})",
                save_path=RESULTS_DIR / f"{prefix}_cosine_sim.png",
            )

            plot_norms(
                avg_norms,
                title=f"Hidden State Norms ({mode}-loop, {noise_label})",
                save_path=RESULTS_DIR / f"{prefix}_norms.png",
            )

            # PCA/t-SNE on first prompt's trajectory
            first_traj = data[0]["trajectory"]
            traj_array = np.array(first_traj)

            plot_pca_trajectory(
                traj_array,
                title=f"PCA Trajectory ({mode}-loop, {noise_label})",
                save_path=RESULTS_DIR / f"{prefix}_pca.png",
            )

            if len(traj_array) >= 5:  # t-SNE needs enough points
                plot_tsne_trajectory(
                    traj_array,
                    title=f"t-SNE Trajectory ({mode}-loop, {noise_label})",
                    save_path=RESULTS_DIR / f"{prefix}_tsne.png",
                )

    # Save numeric results (without raw trajectories — too large)
    output_path = RESULTS_DIR / "exp1c_degeneration.json"
    serializable = {}
    for mode in ["full", "mid"]:
        serializable[mode] = {}
        for noise_label in ["no_noise", "with_noise"]:
            data = all_results[mode][noise_label]
            serializable[mode][noise_label] = [
                {"id": d["id"], "metrics": d["metrics"]}
                for d in data
            ]

    serializable["metadata"] = {
        "experiment": "exp1c_degeneration",
        "timestamp": datetime.now().isoformat(),
        "max_recurrence_steps": MAX_RECURRENCE_STEPS,
        "noise_std": NOISE_STD,
        "mid_layer_index": MID_LAYER_INDEX,
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {output_path}")
    print(f"Plots saved to {RESULTS_DIR}/")

    if WANDB_ENABLED:
        wandb.finish()

    return all_results


if __name__ == "__main__":
    run_experiment()
