# Extension Plan: Mid-Layer Looping — Competitive Positioning

## Context
Our independent discovery of mid-layer latent recurrence on frozen LLMs overlaps with:
- **"Inner Loop Inference for Pretrained Transformers"** (Lys et al., Feb 2026, arxiv 2602.14759)
- They loop layers at 40-60% depth, R=3 iterations, with regularization, getting +1-2% on benchmarks
- We loop at ~33% depth (layer 12/36), up to N=32 iterations, no regularization, getting 45%→90% on small eval

## Our Advantages to Exploit
1. We push iteration count far beyond R=3 (up to N=32) and show continued improvement
2. We inject earlier (~33% vs 40-60%) and remain stable without regularization
3. Our accuracy gains appear much larger (pending real benchmarks)

---

## Phase A: Reproduce & Validate on Real Benchmarks
**Goal:** Get credible numbers comparable to their paper

- [ ] Run mid-layer loop (layer 12, N=1..32) on GSM8K (full 1319 test problems)
- [ ] Run on ARC Challenge (1172 problems, 25-shot)
- [ ] Run on MMLU (5-shot, standard splits)
- [ ] Run on HellaSwag (10-shot)
- [ ] Run on WinoGrande (5-shot)
- [ ] Compare N=0 (no loop) vs N=3 (their setting) vs N=8,16,32 (our extension)
- [ ] Report results in same format as their Table I

## Phase B: Layer Sweep Ablation
**Goal:** Map the full landscape of injection depth vs performance

- [ ] Sweep injection layer: 0, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32 (of 36 total)
- [ ] For each layer, run N=1,4,8,16,32 on GSM8K subset (200 problems)
- [ ] Generate heatmap: injection_layer x n_steps → accuracy
- [ ] Compare to their Figure 1 heatmap
- [ ] Identify if ~33% is genuinely better than 40-60% on Qwen3-8B
- [ ] Test if optimal depth is model-dependent

## Phase C: Regularization Comparison
**Goal:** Show regularization isn't needed at the right injection point

- [ ] Implement their three strategies: uniform averaging, moving average, auto-alignment
- [ ] Run each on GSM8K at layer 12 and layer 18 (their ~50% point)
- [ ] Compare: no-reg at layer 12 vs reg at layer 18 vs no-reg at layer 18
- [ ] Hypothesis: earlier injection is inherently more stable, eliminating need for regularization

## Phase D: Scaling Iteration Count
**Goal:** Show the scaling curve they missed (R=3 is way too few)

- [ ] Run N=1,2,4,8,16,32,64,128 on GSM8K with layer 12
- [ ] Plot accuracy vs N — does it plateau? When?
- [ ] Plot cosine similarity convergence vs N
- [ ] Identify "optimal N" for cost-accuracy tradeoff
- [ ] Compare FLOPs: N latent steps vs equivalent text thinking tokens

## Phase E: Cross-Model Generalization
**Goal:** Show it works beyond Qwen3-8B

- [ ] Test on Llama 3-8B (their model — direct comparison)
- [ ] Test on Gemma 2-2B and 9B (their models)
- [ ] Test on Mistral 7B (new model they didn't test)
- [ ] For each model, find optimal injection layer
- [ ] Report: is ~1/3 depth universal, or model-specific?

## Phase F: Paper Framing
**Positioning:** "We extend [Lys et al. 2026] by showing..."

1. Mid-layer looping scales with iteration count far beyond R=3
2. Earlier injection (~1/3 depth) is more stable and may eliminate regularization
3. Gains are larger than +1-2% when iteration count is properly tuned
4. The technique generalizes across model families
5. We provide mechanistic analysis (probing, attention patterns) of what happens during recurrence

## Key Differentiation Claims
- **Iteration scaling:** They stop at 3, we show 32+ helps
- **No regularization needed:** Simpler method, works at the right depth
- **Stronger results:** Larger accuracy improvements (pending real benchmarks)
- **Mechanistic understanding:** PCA trajectories, cosine similarity analysis, probing studies
