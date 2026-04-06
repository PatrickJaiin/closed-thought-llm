"""
Model configuration, paths, and hyperparameters for closed-thought LLM experiments.
"""

import torch
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Model ──────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"
DTYPE = torch.float16

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# bitsandbytes 4-bit only works on CUDA; on MPS/CPU we load in fp16
LOAD_IN_4BIT = DEVICE == "cuda"

# Qwen3-8B architecture constants
NUM_LAYERS = 36
HIDDEN_DIM = 4096
EMBED_DIM = 4096
NUM_HEADS = 32
VOCAB_SIZE = 151936

# ── Recurrence hyperparams ─────────────────────────────────────────────
RECURRENCE_STEPS = [0, 1, 2, 4, 8, 16, 32]
MID_LAYER_INDEX = NUM_LAYERS // 3  # ~layer 12 for 36-layer model

# ── Continuous loop settings (Phase 2+) ────────────────────────────────
MAX_CONTINUOUS_STEPS = 256  # safety cap for continuous recurrence loop
HALT_CONFIDENCE_THRESHOLD = 0.8  # halt when softmax(logits).max() > this
HALT_CONVERGENCE_THRESHOLD = 0.98  # halt when cos_sim(h_t, h_{t-1}) > this
HALT_ENTROPY_THRESHOLD = 1.0  # halt when logit entropy < this (nats)
HALT_DELTA_NORM_THRESHOLD = 0.5  # halt when ||h_t - h_{t-1}||_2 < this

# ── Learned gate settings (Phase 3) ───────────────────────────────────
GATE_HIDDEN_DIM = 256  # intermediate dim for gate MLPs
GATE_LR = 1e-3  # learning rate for gate training
GATE_RL_LR = 1e-4  # learning rate for REINFORCE refinement
GATE_RL_STEP_PENALTY = -0.01  # per-step penalty in RL reward
GATE_RL_CORRECT_REWARD = 1.0  # reward for correct halt
GATE_RL_INCORRECT_REWARD = -1.0  # reward for incorrect halt

# ── Memory settings (Phase 4) ─────────────────────────────────────────
MEMORY_SLOTS = 128  # number of slots in ring buffer / neural memory
MEMORY_DIM = 256  # projected memory dimension for neural memory
MEMORY_TEMPORAL_DECAY = 0.99  # per-step decay factor for ring buffer
MEMORY_SURPRISE_THRESHOLD = 0.05  # store when 1 - cos_sim > this
MEMORY_RESIDUAL_ALPHA = 0.1  # blending weight for memory retrieval

# ── Experiment settings ────────────────────────────────────────────────
MAX_NEW_TOKENS = 64  # eval answers are short; keeps generation fast without KV cache
TEMPERATURE = 0.0  # greedy decoding for reproducibility
NOISE_STD = 0.01   # for degeneration noise injection experiment

# ── Benchmark settings (Phase 5) ──────────────────────────────────────
BENCHMARK_GSM8K_MAX_TOKENS = 256  # GSM8K needs room for chain-of-thought
BENCHMARK_ARC_MAX_TOKENS = 16     # ARC just needs a letter
BENCHMARK_DEFAULT_SUBSET = 50     # default subset for quick sanity checks
LYS_MID_LAYER_INDEX = NUM_LAYERS // 2  # 18 — Lys et al. approx at 50% depth

# ── Beam search settings (Phase 6) ────────────────────────────────────
BEAM_WIDTH = 3                    # number of beams to keep after pruning
BEAM_BRANCH_FACTOR = 5            # top-k candidates from logit lens per step
BEAM_MAX_DEPTH = 8                # maximum branching depth
BEAM_CONFIDENCE_THRESHOLD = 0.9   # halt when logit lens confidence exceeds this
BEAM_INJECTION_ALPHA = 1.0        # scaling for token embedding injection

# ── Wandb ──────────────────────────────────────────────────────────────
WANDB_PROJECT = "closed-thought-llm"
WANDB_ENABLED = False  # set True when ready to log


if __name__ == "__main__":
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Num layers: {NUM_LAYERS}, Hidden dim: {HIDDEN_DIM}")
    print(f"Mid-layer index: {MID_LAYER_INDEX}")
    print(f"Recurrence steps: {RECURRENCE_STEPS}")
    print(f"Results dir: {RESULTS_DIR}")

    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model and print architecture summary
    from model_utils import load_model
    model, tokenizer = load_model()
    print(f"\nModel loaded successfully.")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Layers: {len(model.model.layers)}")
    print(f"Embed dim: {model.config.hidden_size}")
    print(f"Vocab size: {model.config.vocab_size}")
