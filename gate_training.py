"""
Training pipeline for learned gates (Phase 3).

Two training strategies:
1. Supervised bootstrap — train HaltGate from heuristic gate labels (BCE loss)
2. REINFORCE refinement — optimize halt timing for task accuracy

The LLM stays frozen; only the small gate MLPs are trained.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
from pathlib import Path

from config import (
    DEVICE, HIDDEN_DIM, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS,
    MAX_NEW_TOKENS, GATE_LR, GATE_RL_LR,
    GATE_RL_STEP_PENALTY, GATE_RL_CORRECT_REWARD, GATE_RL_INCORRECT_REWARD,
    RESULTS_DIR,
)
from model_utils import partial_forward, full_forward, get_embeddings
from gates import HaltGate


# ── Dataset for supervised bootstrap ──────────────────────────────────


class HaltLabelDataset(Dataset):
    """
    Dataset of (hidden_state, halt_label) pairs collected from
    running a heuristic gate on training problems.

    Each sample is a recurrence step where:
    - hidden_state: the h vector at that step (hidden_dim,)
    - halt_label: 1.0 if the heuristic said "halt here", 0.0 otherwise
    """

    def __init__(self, hidden_states, halt_labels):
        """
        Args:
            hidden_states: Tensor of shape (N, hidden_dim)
            halt_labels: Tensor of shape (N,) with values 0.0 or 1.0
        """
        self.hidden_states = hidden_states
        self.halt_labels = halt_labels

    def __len__(self):
        return len(self.hidden_states)

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.halt_labels[idx]


# ── Data collection ───────────────────────────────────────────────────


def collect_halt_labels(
    model,
    tokenizer,
    prompts: list,
    heuristic_halt_fn,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    mid_layer: int = MID_LAYER_INDEX,
) -> HaltLabelDataset:
    """
    Run heuristic halt function on a set of prompts and collect
    (hidden_state, halt_label) pairs for supervised training.

    For each prompt:
    - Run recurrence up to max_steps
    - At each step, record h and whether heuristic says halt
    - The step where halt first triggers gets label=1, all others get label=0

    Args:
        model: Frozen LLM.
        tokenizer: Tokenizer.
        prompts: List of prompt dicts with "prompt" key.
        heuristic_halt_fn: A heuristic gate (from gates_heuristic.py).
        max_steps: Max recurrence steps.
        mid_layer: Injection layer.

    Returns:
        HaltLabelDataset with all collected samples.
    """
    all_hidden = []
    all_labels = []

    print(f"Collecting halt labels from {len(prompts)} prompts...")

    with torch.no_grad():
        for i, prompt_data in enumerate(prompts):
            prompt_text = prompt_data["prompt"]

            # Initial forward
            inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
            embeds = get_embeddings(model, inputs.input_ids)
            hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
            h = hidden[:, -1:, :]

            h_prev = None
            halt_step = None

            for step in range(max_steps):
                # Build diagnostics for heuristic
                diag = {"step": step, "h_norm": h.norm().item()}
                if h_prev is not None:
                    diag["cos_sim"] = torch.nn.functional.cosine_similarity(
                        h.view(1, -1), h_prev.view(1, -1)
                    ).item()
                    diag["delta_norm"] = (h - h_prev).norm().item()

                # Record hidden state
                h_flat = h.squeeze().cpu().float()
                all_hidden.append(h_flat)

                # Check if heuristic says halt
                should_halt = heuristic_halt_fn(h, step, diag)

                if should_halt and halt_step is None:
                    halt_step = step
                    all_labels.append(1.0)
                    break  # Stop collecting for this prompt once halted
                else:
                    all_labels.append(0.0)

                h_prev = h.clone()

                # Recurrence step
                position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
                h = partial_forward(
                    model, h, start_layer=mid_layer, position_ids=position_ids
                )

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(prompts)} prompts")

    hidden_tensor = torch.stack(all_hidden)
    labels_tensor = torch.tensor(all_labels)

    print(f"  Collected {len(hidden_tensor)} samples")
    print(f"  Positive labels (halt): {(labels_tensor == 1.0).sum().item()}")
    print(f"  Negative labels (continue): {(labels_tensor == 0.0).sum().item()}")

    return HaltLabelDataset(hidden_tensor, labels_tensor)


# ── Supervised training ───────────────────────────────────────────────


def train_supervised(
    gate: HaltGate,
    dataset: HaltLabelDataset,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = GATE_LR,
    pos_weight: Optional[float] = None,
) -> dict:
    """
    Train HaltGate with BCE loss on heuristic-labeled data.

    Args:
        gate: The HaltGate module (moved to appropriate device).
        dataset: HaltLabelDataset with (hidden_state, halt_label) pairs.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        pos_weight: Weight for positive class (to handle imbalance).
                    If None, auto-computed from label distribution.

    Returns:
        dict with training metrics (losses, accuracies per epoch).
    """
    gate.train()

    # Auto-compute positive weight for class imbalance
    if pos_weight is None:
        n_pos = (dataset.halt_labels == 1.0).sum().item()
        n_neg = (dataset.halt_labels == 0.0).sum().item()
        if n_pos > 0:
            pos_weight = n_neg / n_pos
        else:
            pos_weight = 1.0
        print(f"  Auto pos_weight: {pos_weight:.1f} (neg/pos ratio)")

    criterion = nn.BCELoss(
        weight=None  # Per-sample weighting handled below
    )
    optimizer = optim.Adam(gate.parameters(), lr=lr)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"loss": [], "accuracy": [], "precision": [], "recall": []}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        tp = fp = fn = 0

        for h_batch, labels_batch in loader:
            h_batch = h_batch.to(DEVICE)
            labels_batch = labels_batch.to(DEVICE)

            # Forward: gate expects (batch, 1, hidden_dim) but we have (batch, hidden_dim)
            h_input = h_batch.unsqueeze(1)  # (batch, 1, hidden_dim)
            p_halt = gate(h_input).squeeze()  # (batch,)

            # Apply pos_weight manually
            weight = torch.where(
                labels_batch == 1.0,
                torch.full_like(labels_batch, pos_weight),
                torch.ones_like(labels_batch),
            )
            loss = nn.functional.binary_cross_entropy(p_halt, labels_batch, weight=weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(h_batch)

            # Metrics
            predicted = (p_halt > 0.5).float()
            correct += (predicted == labels_batch).sum().item()
            total += len(labels_batch)
            tp += ((predicted == 1) & (labels_batch == 1)).sum().item()
            fp += ((predicted == 1) & (labels_batch == 0)).sum().item()
            fn += ((predicted == 0) & (labels_batch == 1)).sum().item()

        avg_loss = total_loss / total
        accuracy = correct / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)
        history["precision"].append(precision)
        history["recall"].append(recall)

        print(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, "
              f"acc={accuracy:.3f}, prec={precision:.3f}, rec={recall:.3f}")

    gate.eval()
    return history


# ── REINFORCE training ────────────────────────────────────────────────


def train_reinforce(
    gate: HaltGate,
    model,
    tokenizer,
    prompts: list,
    epochs: int = 5,
    lr: float = GATE_RL_LR,
    step_penalty: float = GATE_RL_STEP_PENALTY,
    correct_reward: float = GATE_RL_CORRECT_REWARD,
    incorrect_reward: float = GATE_RL_INCORRECT_REWARD,
    max_steps: int = MAX_CONTINUOUS_STEPS,
    mid_layer: int = MID_LAYER_INDEX,
    check_answer_fn=None,
) -> dict:
    """
    REINFORCE refinement: optimize halt timing for actual task accuracy.

    Reward structure:
    - +1 for correct answer at halt point
    - -1 for incorrect answer at halt point
    - -0.01 per step (encourages efficiency)

    The gate samples halt decisions stochastically during training,
    and the REINFORCE gradient updates the gate to maximize expected reward.

    Args:
        gate: Pre-trained HaltGate (from supervised bootstrap).
        model: Frozen LLM.
        tokenizer: Tokenizer.
        prompts: List of prompt dicts with "prompt" and "answer" keys.
        epochs: Number of RL epochs over all prompts.
        lr: Learning rate.
        step_penalty: Per-step penalty.
        correct_reward: Reward for correct answer.
        incorrect_reward: Reward for incorrect answer.
        max_steps: Safety cap.
        mid_layer: Injection layer.
        check_answer_fn: Function(predicted, expected) -> bool.

    Returns:
        dict with training metrics.
    """
    from eval_prompts import check_answer
    if check_answer_fn is None:
        check_answer_fn = check_answer

    optimizer = optim.Adam(gate.parameters(), lr=lr)
    history = {"reward": [], "accuracy": [], "avg_steps": []}

    for epoch in range(epochs):
        epoch_rewards = []
        epoch_correct = 0
        epoch_steps = []

        # Shuffle prompts each epoch
        indices = np.random.permutation(len(prompts))

        for idx in indices:
            prompt_data = prompts[idx]
            gate.train()

            log_probs = []
            rewards = []

            with torch.no_grad():
                # Initial forward
                inputs = tokenizer(prompt_data["prompt"], return_tensors="pt").to(DEVICE)
                embeds = get_embeddings(model, inputs.input_ids)
                hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
                h = hidden[:, -1:, :]

            h_prev = None
            halted_step = max_steps

            for step in range(max_steps):
                # Gate forward (with gradients)
                h_input = h.detach()  # detach from LLM graph
                p_halt = gate(h_input)  # (1, 1) or scalar

                # Sample action from Bernoulli
                dist = torch.distributions.Bernoulli(probs=p_halt.clamp(0.01, 0.99))
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)

                # Accumulate step penalty
                rewards.append(step_penalty)

                if action.item() > 0.5:
                    halted_step = step
                    break

                h_prev = h.clone()

                # Recurrence step (no grad — LLM is frozen)
                with torch.no_grad():
                    position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
                    h = partial_forward(
                        model, h, start_layer=mid_layer, position_ids=position_ids
                    )

            # Generate answer at halt point and compute final reward
            with torch.no_grad():
                from continuous_recurrence import _generate_with_prefix_state
                answer = _generate_with_prefix_state(
                    model, tokenizer, h, prompt_data["prompt"], MAX_NEW_TOKENS
                )

            is_correct = check_answer_fn(answer, prompt_data["answer"])
            final_reward = correct_reward if is_correct else incorrect_reward

            # Add final reward to last step
            rewards[-1] += final_reward

            # Compute discounted returns (simple — no discount since episodes are short)
            returns = []
            G = 0
            for r in reversed(rewards):
                G += r
                returns.insert(0, G)
            returns = torch.tensor(returns, device=DEVICE, dtype=torch.float32)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # REINFORCE loss: -sum(log_prob * return)
            policy_loss = 0
            for lp, ret in zip(log_probs, returns):
                policy_loss -= lp.squeeze() * ret

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_rewards.append(sum(rewards))
            epoch_correct += int(is_correct)
            epoch_steps.append(halted_step)

        avg_reward = np.mean(epoch_rewards)
        accuracy = epoch_correct / len(prompts)
        avg_steps = np.mean(epoch_steps)

        history["reward"].append(avg_reward)
        history["accuracy"].append(accuracy)
        history["avg_steps"].append(avg_steps)

        print(f"  Epoch {epoch + 1}/{epochs}: reward={avg_reward:.3f}, "
              f"acc={accuracy:.1%}, avg_steps={avg_steps:.1f}")

    gate.eval()
    return history


def save_gate(gate, path):
    """Save gate checkpoint."""
    torch.save(gate.state_dict(), path)
    print(f"  Gate saved to {path}")


def load_gate(gate_class, path, **kwargs):
    """Load gate from checkpoint."""
    gate = gate_class(**kwargs)
    gate.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    gate.to(DEVICE)
    gate.eval()
    print(f"  Gate loaded from {path}")
    return gate
