# Always Thinking: Gated Mid-Layer Recurrence with Memory for Frozen LMs

**Project:** Continuous Loop + Gated I/O + Memory Architecture
**Author:** Shiv
**Hardware:** Alienware w/ RTX 4090 (24GB VRAM) + CMU cluster for scaling
**Target:** NeurIPS 2026 (deadline ~May 2026), post arXiv ASAP

---

## Core Vision

This is NOT "run N fixed steps then answer." Instead:
1. **Continuous thought loop** — mid-layer recurrence runs ALWAYS, not for fixed N
2. **Learned I/O gate** — decides WHEN to inject a query and WHEN to extract an answer
3. **Learned memory gate** — decides WHEN to store/retrieve thoughts from a memory unit
4. All gates operate INSIDE the recurrence loop

This is the key differentiator from all concurrent work (Lys et al., COCONUT, Huginn, etc.) — nobody has built a unified system combining latent recurrence + adaptive compute + memory + gating.

---

## Phase 0: Foundation (Week 1) — COMPLETE

- [x] Set up environment, download Qwen3-8B, verify inference
- [x] Read COCONUT, Huginn, Sleep-Time Compute, Titans papers

## Phase 1: Can Hidden States Recur Without Training? (Week 2–3) — COMPLETE

**Full findings:** See `results/PHASE1_FINDINGS.md`

- [x] **Exp 1A:** Full-loop recurrence — **FAILS** (NaN after 1 step)
- [x] **Exp 1A-mid:** Mid-layer loop at layer 12 (~1/3 depth) — **WORKS** (45% → 90% at N=32, no training/regularization)
- [x] **Exp 1B:** Text CoT baseline — 85% at N=128 (mid-layer beats it with fewer FLOPs)
- [x] **Exp 1C:** Degeneration analysis — stable cosine sim ~0.85-0.95, bounded norms, rich PCA trajectory

**Concurrent Work:** Lys et al. (arxiv 2602.14759) — they inject at 40-60% with R=3 + regularization; we inject at 33% with N=32, no regularization needed.

---

## Phase 2: Continuous Loop Foundation (Week 2-3) — NEXT

**Goal:** Convert fixed-N loop into always-running continuous loop. Add heuristic gates as proof-of-concept.

### Files
- `continuous_recurrence.py` — continuous loop engine with pluggable `halt_fn`
- `gates_heuristic.py` — confidence, convergence, entropy, delta-norm halt functions
- `experiments/exp2a_long_horizon.py` — stability test at N=64,128,256,512
- `experiments/exp2b_continuous_halting.py` — heuristic gate threshold sweep

### Key Implementation

**`continuous_recurrence.py`** — refactored from `recurrence.mid_layer_loop_recurrence()`:
- Replace `for step in range(n_steps)` with `while not halt_fn(h, step, diag)`
- Accept pluggable `halt_fn: Callable[[Tensor, int, dict], bool]`
- `max_steps=256` safety cap
- Reuses `model_utils.partial_forward()` and `_generate_with_prefix_state()` unchanged
- Integrated memory read/write hooks for Phase 4

**`gates_heuristic.py`** — four heuristic halting strategies:
1. **Confidence gate:** `softmax(logits).max() > threshold` via logit lens
2. **Convergence gate:** `cos_sim(h_t, h_{t-1}) > 0.98`
3. **Entropy gate:** logit entropy < threshold
4. **Delta-norm gate:** `||h_t - h_{t-1}||_2 < threshold` (ACT-style)

**`model_utils.py`** extended with `logit_lens(model, hidden_states, top_k=5)` — project hidden states through LM head, return top-k probabilities and entropy.

**`config.py`** extended with: `MAX_CONTINUOUS_STEPS=256`, gate thresholds, memory hyperparameters.

### Experiments

| Exp | Description | Key Metric |
|-----|-------------|-----------|
| 2A | 512-step stability on 20 prompts | Cosine sim, norms, NaN check |
| 2B | Sweep thresholds for each heuristic gate | Accuracy vs avg steps |
| 2C | Compare adaptive halting to fixed N=32 | Accuracy-per-FLOP |
| 2D | Easy vs hard: do easy problems halt earlier? | Steps-to-halt by difficulty |

### Success Criteria
- Loop stable for 256+ steps (no NaN, bounded norms)
- At least one heuristic gate matches 85%+ accuracy with fewer avg steps than N=32
- Adaptive behavior: easy questions halt earlier than hard ones

---

## Phase 3: Learned Gates (Week 4-6)

**Goal:** Replace heuristic gates with small trained MLPs. Train I/O gate (halt/inject) and memory gate (store/retrieve). Keep LLM frozen.

### Files
- `gates.py` — `HaltGate`, `InjectGate`, `MemoryGate` (nn.Module classes)
- `gate_training.py` — supervised bootstrap + RL refinement
- `experiments/exp3a_supervised_gate.py` — train halt gate from heuristic labels
- `experiments/exp3b_rl_gate.py` — REINFORCE refinement

### Gate Architectures (total ~4.2M params, ~8MB VRAM)

| Gate | Params | Architecture |
|------|--------|-------------|
| HaltGate | ~1.05M | `h → Linear(4096,256) → GELU → Linear(256,1) → Sigmoid` |
| InjectGate | ~2.1M | `[h_loop, h_query] → project → concat → Linear(512,256) → GELU → Linear(256,1) → Sigmoid` |
| MemoryGate | ~1.1M | `h → Linear(4096,256) → GELU → [store_head, retrieve_head]` (two sigmoids) |

### Training Strategy
1. **Phase 3A: Supervised bootstrap** — run heuristic confidence gate on prompts, collect (h_t, halt_label) pairs, train HaltGate with BCE loss
2. **Phase 3B: RL refinement** — REINFORCE with reward = +1 (correct), -1 (incorrect), -0.01/step
3. **Phase 3C: Memory gate** — deferred until Phase 4; trained with multi-turn accuracy reward

### Success Criteria
- Learned gate matches/exceeds heuristic gate accuracy
- Avg steps-to-halt decreases 20%+ vs fixed N=32 for comparable accuracy
- Gate shows per-problem adaptation (not just learning a fixed N)

---

## Phase 4: Memory System (Week 6-8)

**Goal:** Build the memory unit that the memory gate interacts with inside the loop. Store and retrieve latent thoughts.

### Files
- `memory.py` — `KVMemory`, `SurpriseMemory`, `NeuralMemory`
- `experiments/exp4a_memory_tiers.py` — compare memory types on multi-query
- `experiments/exp4b_forgetting.py` — forgetting curves
- `experiments/exp4c_memory_gate_training.py` — train memory gate with multi-turn RL

### Memory Tiers (ablation — all three implemented)

| Tier | Type | Params | VRAM | Description |
|------|------|--------|------|-------------|
| 1 | KV Ring Buffer | 0 | ~2MB | Cosine similarity retrieval, temporal decay |
| 2 | Surprise-Based | 0 | ~2MB | Store only when `1 - cos_sim > threshold` (Titans-inspired) |
| 3 | Neural Memory | ~6.4M | ~13MB | Learned read/write with attention over memory slots |

### Integration into the Loop
```python
# Inside continuous_recurrence loop:
for step in count():
    if memory_gate.should_retrieve(h):
        mem = memory.read(h)
        h = h + alpha * mem          # residual addition
    h = partial_forward(model, h, start_layer=mid_layer, ...)
    if memory_gate.should_store(h):
        memory.write(h)
    if halt_gate(h):
        break
```

Memory read happens BEFORE forward pass (retrieved context influences computation). Memory write happens AFTER (stored state reflects processed thought).

### Forgetting Mechanisms
- Temporal decay on ring buffer: `value *= decay^(age)`
- Access-based refresh: reading resets decay counter (rehearsal)
- Weight decay on NeuralMemory parameters

### Success Criteria
- Memory improves multi-turn accuracy by 10%+ over no-memory
- Forgetting curves show biologically-plausible decay
- Rehearsal works: re-accessed memories survive longer

---

## Phase 5: Full Evaluation + Paper (Week 8-12)

**Goal:** Rigorous benchmarking, ablation, cross-model validation, paper writing.

### Benchmarks
- GSM8K (1319 test, math), ARC Challenge (1172, science), AIME 2024/2025
- Download from HuggingFace: `gsm8k`, `allenai/ai2_arc`, `letta-ai/stateful-gsm-symbolic`, `letta-ai/stateful-aime-2024`

### Ablation Table (9 configurations)

| Config | Loop | Halt Gate | Memory | Description |
|--------|------|-----------|--------|-------------|
| A | None | None | None | Normal prompting |
| B | Fixed N=32 | None | None | Phase 1 approach |
| C | Continuous | Heuristic | None | Phase 2 |
| D | Continuous | Learned | None | Phase 3 |
| E | Continuous | Learned | KV buffer | Phase 4 Tier 1 |
| F | Continuous | Learned | Neural | Phase 4 Tier 3 |
| G | Continuous | Learned+inject | Neural | Full system |
| H | Text CoT | None | None | Text baseline |
| I | Lys et al. | R=3, regularized | None | Concurrent work |

### Cross-Model (CMU cluster)
- Llama 3-8B, Gemma 2-9B — direct comparison with Lys et al.'s models
- Test if ~1/3 injection depth is universal

### Paper: "Always Thinking: Gated Mid-Layer Recurrence with Memory for Frozen LMs"
- Lead with empirical result (45% → 90% without training)
- Key contribution: continuous loop + learned gates + memory = unified cognitive architecture
- Differentiate from Lys et al. (fixed R=3, no gates, no memory, needs regularization)
- Target: NeurIPS 2026 (deadline ~May 2026), post arXiv ASAP

---

## VRAM Budget

| Component | VRAM |
|-----------|------|
| Qwen3-8B (4-bit NF4) | ~5 GB |
| Activations | ~1 GB |
| All gates (4.2M params) | ~8 MB |
| Neural memory (6.4M params) | ~13 MB |
| Gate training overhead | ~2 GB |
| **Total (training)** | **~8 GB** |
| **Available on 4090** | **~17.2 GB** |
| **Headroom** | **~9 GB** |

---

## Build Order

1. **Phase 2:** `continuous_recurrence.py` → `gates_heuristic.py` → extend `model_utils.py` + `config.py` → `exp2a` → `exp2b`
2. **Phase 3:** `gates.py` → `gate_training.py` → `exp3a` → `exp3b`
3. **Phase 4:** `memory.py` → integrate into `continuous_recurrence.py` → `exp4a` → `exp4b` → `exp4c`
4. **Phase 5:** benchmarks → ablation → cross-model → paper

## Verification Checklist
1. Run `exp2a_long_horizon.py` — loop stable at 256+ steps
2. Run `exp2b_continuous_halting.py` — adaptive gate beats fixed N
3. Run `exp3a_supervised_gate.py` — learned gate ≥ heuristic accuracy
4. Run `exp4a_memory_tiers.py` — memory helps on multi-query
5. Run full ablation — full system beats all baselines on ≥2 benchmarks

---

## Key Risk Mitigations

| Risk | Mitigation | Status |
|------|-----------|--------|
| Latent recurrence degenerates | Mid-layer injection at ~1/3 depth is stable | **RESOLVED** |
| No improvement over Sleep-Time | Phase 1: 90% vs 85% — promising | Pending Phase 2 |
| Scooped by Lys et al. | Our approach differs (earlier injection, no reg, gates, memory) | **Manageable** |
| Memory doesn't help | Three tiers + forgetting + rehearsal give multiple angles | Pending Phase 4 |
| Small eval set inflates results | Phase 5 uses real benchmarks (GSM8K 1319, ARC, AIME) | Pending Phase 5 |

---

## Resources

**Papers to Cite:**
- COCONUT (Hao et al., 2024) — arxiv 2412.06769
- Sleep-Time Compute (Lin et al., 2025) — arxiv 2504.13171
- Huginn (Geiping et al., 2025) — arxiv 2502.05171
- Titans (Behrouz et al., 2024) — arxiv 2501.00663
- **Inner Loop Inference (Lys et al., 2026) — arxiv 2602.14759** ← concurrent work
- StreamingThinker (2025) — arxiv 2510.17238

**Datasets:**
- GSM8K: `huggingface.co/datasets/gsm8k`
- ARC: `huggingface.co/datasets/allenai/ai2_arc`
- Stateful GSM-Symbolic: `huggingface.co/datasets/letta-ai/stateful-gsm-symbolic`
- Stateful AIME: `huggingface.co/datasets/letta-ai/stateful-aime-2024`
