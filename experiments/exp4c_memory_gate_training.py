"""
Experiment 4C: Train memory gate with multi-turn RL.

Trains the MemoryGate to decide when to store/retrieve from memory,
using multi-turn accuracy as the reward signal.

Pipeline:
1. Use multi-query scenarios from exp4a
2. Memory gate controls read/write decisions
3. REINFORCE with reward = multi-turn accuracy

Also combines with HaltGate for the full gated system test.

Success criteria:
- Gated memory >= always-on memory on multi-turn
- Gate learns to retrieve on follow-up questions
"""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from config import (
    DEVICE, RESULTS_DIR, MID_LAYER_INDEX, MAX_CONTINUOUS_STEPS,
    MAX_NEW_TOKENS, HIDDEN_DIM, GATE_HIDDEN_DIM,
    MEMORY_RESIDUAL_ALPHA, GATE_RL_LR,
)
from config import WANDB_ENABLED, WANDB_PROJECT
from model_utils import load_model
from continuous_recurrence import continuous_recurrence
from gates import MemoryGate, HaltGate, count_parameters
from gate_training import save_gate
from memory import create_memory
from eval_prompts import check_answer

if WANDB_ENABLED:
    import wandb


# Reuse scenarios from exp4a
from experiments.exp4a_memory_tiers import MULTI_QUERY_SCENARIOS


def train_memory_gate_rl(
    memory_gate,
    model,
    tokenizer,
    scenarios,
    epochs=10,
    lr=GATE_RL_LR,
    n_recurrence=32,
    memory_tier="kv",
):
    """
    Train MemoryGate with REINFORCE on multi-query scenarios.

    For each scenario:
    1. Process queries sequentially
    2. Memory gate samples store/retrieve decisions
    3. Reward = fraction of correct answers in the scenario

    Args:
        memory_gate: MemoryGate module to train.
        model: Frozen LLM.
        tokenizer: Tokenizer.
        scenarios: List of multi-query scenario dicts.
        epochs: Number of training epochs.
        lr: Learning rate.
        n_recurrence: Recurrence steps per query.
        memory_tier: Which memory backend to use.

    Returns:
        Training history dict.
    """
    optimizer = optim.Adam(memory_gate.parameters(), lr=lr)
    history = {"accuracy": [], "avg_reward": []}

    for epoch in range(epochs):
        epoch_correct = 0
        epoch_total = 0
        epoch_rewards = []

        # Shuffle scenarios
        scenario_indices = np.random.permutation(len(scenarios))

        for si in scenario_indices:
            scenario = scenarios[si]
            memory = create_memory(tier=memory_tier, device=DEVICE)

            memory_gate.train()
            log_probs_episode = []
            scenario_correct = 0

            for qi, query_data in enumerate(scenario["queries"]):
                # Gate decisions with gradients
                with torch.no_grad():
                    from model_utils import full_forward, get_embeddings
                    inputs = tokenizer(scenario["context"], return_tensors="pt").to(DEVICE)
                    embeds = get_embeddings(model, inputs.input_ids)
                    hidden = full_forward(model, embeds, attention_mask=inputs.attention_mask)
                    h = hidden[:, -1:, :]

                # Sample retrieve decision
                p_store, p_retrieve = memory_gate(h)

                retrieve_dist = torch.distributions.Bernoulli(probs=p_retrieve.clamp(0.01, 0.99))
                retrieve_action = retrieve_dist.sample()
                log_probs_episode.append(retrieve_dist.log_prob(retrieve_action))

                # Execute retrieval if gate says yes
                if retrieve_action.item() > 0.5:
                    mem = memory.read(h)
                    if mem is not None:
                        h = h + MEMORY_RESIDUAL_ALPHA * mem

                # Run recurrence
                with torch.no_grad():
                    from model_utils import partial_forward
                    for _ in range(n_recurrence):
                        position_ids = torch.zeros(1, 1, device=DEVICE, dtype=torch.long)
                        h = partial_forward(
                            model, h, start_layer=MID_LAYER_INDEX,
                            position_ids=position_ids,
                        )

                # Sample store decision
                p_store_post, _ = memory_gate(h.detach())
                store_dist = torch.distributions.Bernoulli(probs=p_store_post.clamp(0.01, 0.99))
                store_action = store_dist.sample()
                log_probs_episode.append(store_dist.log_prob(store_action))

                if store_action.item() > 0.5:
                    memory.write(h.detach())

                # Generate answer
                with torch.no_grad():
                    from continuous_recurrence import _generate_with_prefix_state
                    answer = _generate_with_prefix_state(
                        model, tokenizer, h, query_data["q"], MAX_NEW_TOKENS
                    )

                is_correct = check_answer(answer, query_data["a"])
                scenario_correct += int(is_correct)
                epoch_correct += int(is_correct)
                epoch_total += 1

            # Reward = fraction correct in this scenario
            reward = scenario_correct / len(scenario["queries"])
            epoch_rewards.append(reward)

            # REINFORCE update
            if log_probs_episode:
                baseline = 0.5  # simple baseline
                advantage = reward - baseline

                policy_loss = 0
                for lp in log_probs_episode:
                    policy_loss -= lp.squeeze() * advantage

                optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(memory_gate.parameters(), max_norm=1.0)
                optimizer.step()

        accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        avg_reward = np.mean(epoch_rewards)
        history["accuracy"].append(accuracy)
        history["avg_reward"].append(avg_reward)

        print(f"  Epoch {epoch + 1}/{epochs}: accuracy={accuracy:.1%}, reward={avg_reward:.3f}")

    memory_gate.eval()
    return history


def evaluate_memory_configs(model, tokenizer, scenarios, n_recurrence=32):
    """
    Evaluate different memory configurations:
    1. No memory
    2. Always-on KV memory (no gate)
    3. Gated KV memory (trained gate)
    """
    results = {}

    # Config 1: No memory
    print("\n--- No memory ---")
    correct = 0
    total = 0
    for scenario in scenarios:
        for query_data in scenario["queries"]:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=scenario["context"],
                query_text=query_data["q"],
                n_steps=n_recurrence,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            if check_answer(result["answer"], query_data["a"]):
                correct += 1
            total += 1
    results["no_memory"] = {"accuracy": correct / total, "correct": correct, "total": total}
    print(f"  No memory: {correct}/{total} ({correct/total:.1%})")

    # Config 2: Always-on KV memory
    print("\n--- Always-on KV memory ---")
    correct = 0
    total = 0
    for scenario in scenarios:
        memory = create_memory(tier="kv", device=DEVICE)
        for query_data in scenario["queries"]:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=scenario["context"],
                query_text=query_data["q"],
                n_steps=n_recurrence,
                max_new_tokens=MAX_NEW_TOKENS,
                memory=memory,
                memory_alpha=MEMORY_RESIDUAL_ALPHA,
            )
            if check_answer(result["answer"], query_data["a"]):
                correct += 1
            total += 1
    results["always_on_kv"] = {"accuracy": correct / total, "correct": correct, "total": total}
    print(f"  Always-on KV: {correct}/{total} ({correct/total:.1%})")

    return results


def run_experiment():
    print("=" * 60)
    print("Experiment 4C: Memory Gate Training (Multi-Turn RL)")
    print("=" * 60)

    if WANDB_ENABLED:
        wandb.init(project=WANDB_PROJECT, name="exp4c_memory_gate_training")

    print("\nLoading model...")
    model, tokenizer = load_model()
    print("Model loaded.\n")

    scenarios = MULTI_QUERY_SCENARIOS

    # Evaluate baselines first
    print("Evaluating baselines...")
    baselines = evaluate_memory_configs(model, tokenizer, scenarios)

    # Train memory gate
    print("\n--- Training MemoryGate ---")
    memory_gate = MemoryGate(hidden_dim=HIDDEN_DIM, gate_dim=GATE_HIDDEN_DIM).to(DEVICE)
    print(f"  Parameters: {count_parameters(memory_gate):,}")

    train_history = train_memory_gate_rl(
        memory_gate, model, tokenizer, scenarios,
        epochs=15, lr=GATE_RL_LR,
        n_recurrence=32, memory_tier="kv",
    )

    # Save gate checkpoint
    gate_path = RESULTS_DIR / "memory_gate_rl.pt"
    save_gate(memory_gate, gate_path)

    # Evaluate trained gate
    print("\n--- Evaluating gated memory ---")
    correct = 0
    total = 0
    for scenario in scenarios:
        memory = create_memory(tier="kv", device=DEVICE)
        for query_data in scenario["queries"]:
            result = continuous_recurrence(
                model, tokenizer,
                context_text=scenario["context"],
                query_text=query_data["q"],
                n_steps=32,
                max_new_tokens=MAX_NEW_TOKENS,
                memory=memory,
                memory_gate=memory_gate,
                memory_alpha=MEMORY_RESIDUAL_ALPHA,
            )
            if check_answer(result["answer"], query_data["a"]):
                correct += 1
            total += 1

    gated_results = {"accuracy": correct / total, "correct": correct, "total": total}
    print(f"  Gated KV: {correct}/{total} ({correct/total:.1%})")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training curves
    ax1.plot(train_history["accuracy"], "g-o", markersize=4, label="Accuracy")
    ax1.plot(train_history["avg_reward"], "b-s", markersize=4, label="Avg Reward")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Value")
    ax1.set_title("Memory Gate RL Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Comparison bar chart
    configs = ["No Memory", "Always-On KV", "Gated KV"]
    accs = [
        baselines["no_memory"]["accuracy"],
        baselines["always_on_kv"]["accuracy"],
        gated_results["accuracy"],
    ]
    colors = ["#999999", "#1f77b4", "#2ca02c"]
    bars = ax2.bar(configs, accs, color=colors, alpha=0.8)
    ax2.set_ylabel("Multi-Turn Accuracy")
    ax2.set_title("Memory Configuration Comparison")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{acc:.0%}", ha="center", fontsize=11)

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "exp4c_memory_gate.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {RESULTS_DIR / 'exp4c_memory_gate.png'}")
    plt.close(fig)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  No memory:    {baselines['no_memory']['accuracy']:.1%}")
    print(f"  Always-on KV: {baselines['always_on_kv']['accuracy']:.1%}")
    print(f"  Gated KV:     {gated_results['accuracy']:.1%}")

    # Save
    output_path = RESULTS_DIR / "exp4c_memory_gate.json"
    serializable = {
        "baselines": baselines,
        "gated": gated_results,
        "training_history": train_history,
        "metadata": {
            "experiment": "exp4c_memory_gate_training",
            "timestamp": datetime.now().isoformat(),
            "gate_params": count_parameters(memory_gate),
        },
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")

    if WANDB_ENABLED:
        wandb.finish()

    return serializable


if __name__ == "__main__":
    run_experiment()
