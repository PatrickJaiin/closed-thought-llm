"""
Visualization utilities for closed-thought LLM experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path


def plot_cosine_similarity(cosine_sims, title="Cosine Similarity", save_path=None):
    """Plot cosine similarity between consecutive hidden states over recurrence steps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = list(range(1, len(cosine_sims) + 1))
    ax.plot(steps, cosine_sims, "b-o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Recurrence Step")
    ax.set_ylabel("Cosine Similarity (h_t, h_{t+1})")
    ax.set_title(title)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Fixed point")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_norms(norms, title="Hidden State Norms", save_path=None):
    """Plot L2 norms of hidden states over recurrence steps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = list(range(len(norms)))
    ax.plot(steps, norms, "g-o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Recurrence Step")
    ax.set_ylabel("L2 Norm")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_pca_trajectory(trajectory, title="PCA Trajectory", save_path=None):
    """
    Plot 2D PCA projection of hidden state trajectory.
    trajectory: numpy array of shape (n_steps, hidden_dim).
    """
    if len(trajectory) < 2:
        print("  Skipping PCA: need at least 2 points")
        return

    if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
        print("  Skipping PCA: trajectory contains NaN/Inf (degenerated)")
        return

    pca = PCA(n_components=2)
    projected = pca.fit_transform(trajectory)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by step number
    colors = np.arange(len(projected))
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1],
        c=colors, cmap="viridis", s=30, zorder=3,
    )

    # Draw lines connecting consecutive points
    ax.plot(projected[:, 0], projected[:, 1], "k-", alpha=0.3, linewidth=0.8)

    # Mark start and end
    ax.scatter(projected[0, 0], projected[0, 1], c="red", s=100, marker="^", zorder=4, label="Start")
    ax.scatter(projected[-1, 0], projected[-1, 1], c="blue", s=100, marker="s", zorder=4, label="End")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(title)
    ax.legend()
    plt.colorbar(scatter, label="Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_tsne_trajectory(trajectory, title="t-SNE Trajectory", save_path=None):
    """
    Plot 2D t-SNE projection of hidden state trajectory.
    trajectory: numpy array of shape (n_steps, hidden_dim).
    """
    n_points = len(trajectory)
    if n_points < 5:
        print("  Skipping t-SNE: need at least 5 points")
        return

    if np.any(np.isnan(trajectory)) or np.any(np.isinf(trajectory)):
        print("  Skipping t-SNE: trajectory contains NaN/Inf (degenerated)")
        return

    perplexity = min(30, n_points - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    projected = tsne.fit_transform(trajectory)

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = np.arange(len(projected))
    scatter = ax.scatter(
        projected[:, 0], projected[:, 1],
        c=colors, cmap="viridis", s=30, zorder=3,
    )

    ax.plot(projected[:, 0], projected[:, 1], "k-", alpha=0.3, linewidth=0.8)

    ax.scatter(projected[0, 0], projected[0, 1], c="red", s=100, marker="^", zorder=4, label="Start")
    ax.scatter(projected[-1, 0], projected[-1, 1], c="blue", s=100, marker="s", zorder=4, label="End")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend()
    plt.colorbar(scatter, label="Step")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_accuracy_comparison(results_dict, title="Accuracy Comparison", save_path=None):
    """
    Plot accuracy vs recurrence steps for multiple experiments.

    Args:
        results_dict: dict mapping experiment_name → {n_steps: {"accuracy": float}}.
        title: Plot title.
        save_path: Where to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D", "v"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, (name, data) in enumerate(results_dict.items()):
        steps = sorted(data.keys(), key=lambda x: int(x))
        accuracies = [data[s]["accuracy"] for s in steps]
        steps_int = [int(s) for s in steps]

        ax.plot(
            steps_int, accuracies,
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            linewidth=2, markersize=8,
            label=name,
        )

    ax.set_xlabel("Recurrence Steps / Thinking Tokens")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)
