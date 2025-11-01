# MoirAI

**Dynamic hierarchical tokens (H-Net) + hierarchical recurrent reasoning (HRM-L/M/G) + hierarchical MoE experts**

Knowledge bank via *transplanted* FFN experts (Qwen by default; optional Mamba mixers), plus *trained-from-scratch* HRM experts (heterogeneous sizes). Efficient, compile-friendly, and staged for stability.

---

## 1) Architectural overview (what the subsystems are and why)

Modern large language models are powerful but face fundamental limitations: they rely on static, pre-defined tokenizers that struggle with raw, byte-level data, and their fixed-depth architectures limit multi-step reasoning. **MoirAI** is built to fix this by integrating three synergistic subsystems that learn to **reason efficiently and dynamically from bytes**.

### 1.1 H-Net (Dynamic Hierarchical Chunking) [^1][^2]

**What.** H-Net is a tokenizer-free input layer that learns to segment raw byte streams into **three token bands**: **v0 (low)** for fine details, **v1 (mid)** for clauses / semantic chunks, and **v2 (high)** for global context. The model, not a fixed vocab, decides boundaries.

**Why.** Static tokenizers decouple representation from reasoning. Learned chunking yields (a) semantically coherent units that adapt to data/task, (b) **fewer, larger high-level tokens** for efficient global passes, and (c) a **reverse lattice** guaranteeing perfect byte-level generation.

**Targets.** We softly target **v0≈32 bits**, **v1≈512 bits**, **v2≈8 kbits**, with a brief curriculum to reach them; training starts looser and anneals down.

### 1.2 HRM (Hierarchical Recurrent Reasoning)

**What.** HRM is a multi-timescale recurrent reasoning stack:

* **HRM-L** iterates rapidly over v0,
* **HRM-M** iterates more slowly over v1, and
* **HRM-G** performs a single, high-level update per outer step using pooled states.

A learned **halting** mechanism can stop early when extra compute won’t help.

**Why.** Hard problems need **iteration**, not just depth. HRM lets the model “think” as needed while summarizing easy parts—deep latent reasoning **without external chain-of-thought text**. A **global control signal** from HRM-G is **broadcast via FiLM** back to L before byte logits.

### 1.3 Hierarchical MoE (Conditional Knowledge & Reasoning)

**What.** Two conditional expert systems:

1. **FFN Knowledge Experts** transplanted from strong donor dense models (e.g. Qwen), serving as a ready-to-use **knowledge bank**.
2. **HRM Reasoning Experts** (heterogeneous sizes), trained from scratch, supplying varied **reasoning horsepower**.

Routing picks only a **small set of specialists** per step.

**Why.** Inputs differ in knowledge needs and difficulty. Routing to a targeted specialist keeps active compute low; transplantation **slashes pretraining cost**; heterogeneous HRM experts match compute budget to difficulty.

### 1.4 Control, stability, and policy

MoirAI also includes:

* **A Task Header Block (THB)** and **Task Header Policy Layer (THPL)** that encode per-sample runtime policy (task type, halting style, abstention mode, loop caps, etc.) in a fixed 64-byte header. This header is authoritative for routing bias, halting, abstention, and allowed iteration budgets.
* **Selective confidence + verify path**: calibrated uncertainty, a controlled abstain path, and an optional “verify” retry pass with extra compute. This is supervised and coverage-controlled.
* **Attention backends and stabilization**: multiple interchangeable attention variants (standard dot-product, adaptive/uncertainty-aware, head-gated, sigmoid, selective temperature, etc.) plus safeguards against attention sink and activation spikes.
* **Long-context strategy**: External Segment Encoder (ESE) to compress long documents into cached high-level latents; periodic or Power/DSA-style attention for scalable global mixing; hybrid positional schemes; static-shape routing with top-k=1 to preserve compile-friendliness.

### 1.5 Learned Residual Scales (S-REG)

**What.** A compile-safe **Scale Registry (S-REG)** adds non-negative learned scalars that modulate residual contributions **per module type** (attention, HRM experts, fixed paths, donor FFN retrieval, FiLM broadcast, UMoE-lite taps) and **per layer/band** (L/M/G, trunk attention, attention taps). For any residual path `Δ(·)` we apply:

```text
y = x + s_type · s_layer · DropPath(Δ(x))         # s_type ≥ 0, s_layer ≥ 0 (learned; softplus-param)
```

**Why.**
(1) Stabilizes early training and donor-FFN splice-in; (2) reduces attention sink by controlling post-mix magnitude; (3) prevents HRM expert over-dominance; (4) provides clean attribution telemetry (“who did the work?”) for verify/abstain and fixed-path repurposing; (5) costs ~0 params relative to model size and preserves compile invariants.

**Parameterization & Safety.**
Scales are parameterized as `s = softplus(θ)` with optional soft barrier `s ≤ s_max` via a tiny quadratic penalty (see §11.6). Initial values are conservative (see §16 (REPLACED)). GRUs remain unscaled; only **residual contributions** into GRU inputs or output heads are scaled. No dynamic control flow, no shape changes, and no changes to top-k dispatch.

**Placement (high-level).**
S-REG multiplies residuals from: HRM-L/M attention, HRM-L/M top-1 experts, HRM fixed paths, donor FFN retrieval into G, FiLM broadcast magnitude `(γ,β)`, trunk attention residuals (all backends), and optional UMoE-lite taps.

---

## 2) Model variants (trunks trained from scratch; Qwen‑aligned dims) [^14][^15]

**Canonical variant names include the final total parameter count and the estimated average *active* parameters per token.**

| Variant (canonical)   | Layers | d_model | HRM clusters | FFN clusters (per family) | Approx **total params** | **Avg active params / token** |
| --------------------- | :----: | :-----: | :----------: | :-----------------------: | :---------------------: | :---------------------------: |
| **MoirAI‑0.9B‑A0.3B** |   24   |   896   |       4      |             8             |          ≈0.93B         |          ≈0.30–0.38B          |
| **MoirAI‑2.9B‑A0.9B** |   28   |  1 536  |       6      |             8             |          ≈2.92B         |          ≈0.82–0.90B          |

**Definition of “active”.** For a token, active parameters include: the selected HRM cluster’s L/M experts (top‑1 each), any non‑zero shared fixed HRM path, the retrieved FFN expert (family→cluster→expert, top‑1), trunk attention projections in the executed layers/heads, FiLM broadcast, and the ByteHead. DropPath and halting are accounted for.

### 2.1 Trunk depth, indexing, and donor capture

* **Trunk depth** by variant: 24 (MoirAI‑0.9B‑A0.3B) and 28 (MoirAI‑2.9B‑A0.9B).
* **Layer indexing:** 1‑based over the full trunk; taps and S‑REG Layer‑ID scales use this indexing.
* **Donor capture:** donor layers forming the FFN family bank are chosen by the data‑driven selector (see §6.2.1) and do not renumber trunk layers.

### 2.2 HRM band widths & heterogeneous tiers (per variant)

This subsection fixes **band widths**, **tier ratios with rounding**, **per‑cluster multiplicities**, and **intended pick‑rate biases**. Rounding follows §6.1.0. Variant names include both total and average active parameter counts.

#### 2.2.1 **MoirAI‑0.9B‑A0.3B**

* **Band widths:** (d_L = 1920), (d_M = 2880); (d_G = 2\cdot d_L = 3840).
* **Ratios → rounded widths (L / M):**

  * **0.2667× (4/15)** → **512 / 768**  (512 is (2^9); 768 is 64‑aligned)
  * **0.50×** → **960 / 1440**
  * **0.75×** → **1440 / 2160**
  * **1.00×** → **1920 / 2880**
  * **1.25× (escape)** → **2432 / 3584**  (64‑aligned within ≤2% of 1.25×)
* **Per‑cluster multiplicities:**

  * **HRM‑L (8 total):** { **0.2667×: 4**, 0.50×: 1, 0.75×: 1, 1.00×: 1, 1.25×: 1 }
  * **HRM‑M (5 total):** { **0.2667×: 1**, 0.50×: 1, 0.75×: 1, 1.00×: 1, 1.25×: 1 }
* **Pick‑rate intent (steady state):**

  * **L:** ~88–92% to **0.2667×**, ~5–8% to 0.50×, remainder thin across larger tiers.
  * **M:** ~85–90% to **0.2667×**, remainder across larger tiers.

#### 2.2.2 **MoirAI‑2.9B‑A0.9B**

* **Band widths:** (d_L = 3072), (d_M = 4096) (reduced vs 1.5× to contain quadratic cost); (d_G = 2\cdot d_L = 6144).
* **Ratios → rounded widths (L / M):**

  * **0.3333× (1/3)** → **1024 / 1344**  (L snaps to 1024; M aligns to 1344)
  * **0.50×** → **1536 / 2048**
  * **0.75×** → **2304 / 3072**
  * **1.00×** → **3072 / 4096**
  * **1.20× (escape)** → **3712 / 4928**  (64‑aligned; >1.0 headroom)
* **Per‑cluster multiplicities:**

  * **HRM‑L (6 total):** { **0.3333×: 2**, 0.50×: 1, 0.75×: 1, 1.00×: 1, 1.20×: 1 }
  * **HRM‑M (5 total):** { 0.3333×: 1, 0.50×: 1, 0.75×: 1, 1.00×: 1, 1.20×: 1 }
* **Pick‑rate intent (steady state):**

  * **L:** ≥90% to **0.3333×**; escape 1.20× is rare.
  * **M:** ≥88% to **0.3333×**; remainder across larger tiers.

**Compute priors and curricula** are specified in §6.1.3. Shapes remain static; dispatch is top‑1.

### 2.3 Active‑parameter accounting and attention backend impact

* **Fixed backends:** choosing a specific per‑layer backend (e.g., SDPA, DSA, Power) fixes the attention projection parameter cost for that layer, so active counts stabilize quickly.
* **Learned→fixed backends:** a head‑scoped dual‑exec mixer may be used temporarily; an entropy/L1 penalty drives a **near‑binary** mixing coefficient, after which the layer behaves as a fixed‑backend layer and active counts converge.
* **Per‑token backend routing is not used.** Attention backends are either fixed or learned→fixed; dynamic per‑token routing of backends is out of scope to preserve compile invariants and predictable active counts.

---

## 3) Data & representation

* **Bytes in/out.** Everything is trained from bytes. Reverse lattice (from H-Net) ensures we can map the L state back to bytes exactly for generation and loss.
* **ARC-AGI packing.** 4 bits/cell; reserve two codes for **row** and **sample** separators; a 30×30 grid stays < 4 KB using this plan.
* **Task headers.** Every sample starts with a fixed 64-byte **Task Header Block (THB)** produced by THPL (see §9). H-Net is told *not to chunk across* that header. The header fields act as side-channel control features for halting, routing, abstention/verify policy, etc.

---

## 4) H-Net dynamic chunker

### 4.1 Targets (bits per chunk) [^17]

*   **v0:** **32 bits** (~4 bytes; ≈ BPE-token scale).
*   **v1:** **512 bits** (~64 bytes; ≈ 10–15 words / clause).
*   **v2:** **8 192 bits** (~1 KB; ≈ 150–200 words / ~⅘ of a page).

We start looser (64 / 1 024 / 16 384 bits) and anneal to these.

### 4.2 Level blocks (each level ℓ ∈ {0,1,2})

Pipeline per level:

1.  Input stream (bytes for v0; vℓ for ℓ>0) → **embed**.
2.  **1D Transformer** (or equivalent) processes the sequence.
3.  **Boundary head** predicts boundary likelihood σ(b).
4.  **Soft pooling** between boundaries produces the next-level tokens `vℓ+1`.

We also maintain a **reverse lattice** with (start, end, weights) per v0 segment, so we can reconstruct bytes exactly from v0/v1/v2 states during decoding.

### 4.3 Losses & regularizers

*   **Target-bits Huber loss** with a zero-penalty band ±15% (±20% during target anneal; see §4.5).
*   **Ratio term** to discourage pathological over/under compression globally.
*   **Boundary entropy term** to prevent degenerate “one point every N bytes” or “one gigantic span per doc” behavior.

### 4.4 Chunk-aware attention windows (speed)

We bias attention cost down while letting information flow:

*   **v0 attention:** within-chunk + **±16 neighbor chunks** (tiny slack window across boundary).
*   **v1 attention:** within-chunk + **±2 neighbor chunks**.
*   **v2 attention:** full.

This is block-sparse / windowed, making H-Net efficient.

### 4.5 Target‑bits curriculum

A two‑phase schedule anneals token sizes:

* **Early (M1 → early M2):** v0 = 64 bits, v1 = 1 024 bits, v2 = 16 384 bits with **±20%** Huber zero‑band.
* **Late (late M2 → M3):** cosine‑anneal to **32/512/8 192** bits and tighten the Huber zero‑band to **±10%**.
* Freeze Level‑2 for one epoch during the switchover, keep ratio/entropy losses active, and use the wider band only during anneal.

### 4.6 Early-phase “hockey-stick” guardrails (M1)

Empirically, healthy H-Net runs show a fast **hockey-stick** in early training (~first 500 steps): selection rate (L1/L0) drops, spikes, then settles near target. When we *don’t* see this, the run later degrades.

Policy:

*   Initialize selection rate around `L1/L0 ≈ 0.40`.
*   In the first **2k optimizer steps**, assert:
    *   selection rate **decreases**, then clearly **overshoots** above init, then **monotone declines toward the target**;
    *   spike timing between step 100 and step 800 (tunable).
*   If the pattern is missing:
    *   temporarily **raise ratio-loss weight** ×1.5 for 200 steps,
    *   temporarily **halve LR** on the main net,
    *   retry once with a new seed if it still fails.

Config:

```yaml
hnet.hockeystick:
  enable: true
  init_select_rate: 0.40
  window_steps: 2000
  spike_step_range: [100, 800]
  remediate: {ratio_loss_boost: 1.5, main_lr_scale: 0.5, retries: 1}
```

We unit-test this detector on synthetic traces and assert it triggers expected interventions.

### 4.7 Spacing / first‑byte pathology mitigations (text presets default)

**First‑Byte Assist (FBA) is ON by default for *text* presets via THPL; the boundary anchor prior remains OFF unless validated for the domain.**

1. **First‑Byte Assist (FBA)** – A tiny MLP emits a scalar bias only at “first‑byte‑after‑boundary” positions; added to the byte logits path. High weight decay; strictly additive; shape‑stable.

2. **Boundary Anchor Prior (soft)** – A small “is_space_or_delim” feature can be fed into the v0 boundary head MLP (coef ≤0.05). **Default OFF** for code/CJK/ARC unless validated.

Config (policy‑gated):

```yaml
hnet.spacing_helpers:
  fba: {enable: true, weight: 0.3}      # THPL enables for text presets
  anchor_prior: {enable: false, max_coef: 0.05}
```

Tests: first‑byte error rates at boundaries (English vs CamelCase vs CJK vs code); overall perplexity guard ≤0.1% regression.

### 4.8 Compression spikes are **signals**, not always failures

Large **mid-run compression ratio spikes** sometimes correlate with *better* cross-entropy. Over-penalizing them hurt runs. Policy:

*   Keep target-bits, ratio, and entropy terms as designed.
*   Do **not** auto-penalize transient ratio spikes if cross-entropy is still improving.
*   Only page an alert if spikes **persist > N steps** *and* CE stalls.

Config:

```yaml
hnet.ratio_spike_policy:
  benign_window_steps: 4000
  ce_improve_floor: 0.1   # nats/token per 1k steps; tune per scale
  alert_if_persistent: true
```

### 4.9 Clarify chunk semantics (E-span / M-point / D-span)

We explicitly distinguish three chunk definitions so instrumentation doesn’t conflate them:

*   **E-span:** the encoder’s contiguous byte span leading to the next selected byte.
*   **M-point:** the selected byte that feeds the main network at v0/v1/v2.
*   **D-span:** the bytes predicted (decoded) from one chunk vector.

We log metrics per view; “first-byte errors” live in D-span, not M-point.

### 4.10 Re‑chunk on demand (optional; gated & budgeted)

One sliding window may be re‑encoded when HRM‑G flags misalignment:

```yaml
hnet.rechunk_on_demand:
  enable: false
  trigger: {innov_pctl: 97, proto_sim_max: 0.15}
  window_tokens: 1024
  budget:
    max_triggers_per_1k_tokens: 1
    p95_latency_guard_pct: 5
```

Buffers are pre‑allocated; shapes remain static. Auto‑disable on budget violations.

### 4.11 Tests & monitors for H-Net

We extend tests/metrics (see §12–§13):

*   **Hockey-stick presence (M1).** Detect the three-phase pattern (drop → spike → settle) in first 2k steps. Auto-remediate, retry once.
*   **Spacing suite (M3).** Measure first-byte boundary error rates with and without First-Byte Assist / anchor prior across English, CamelCase/CJK, code. Require reduced first-byte boundary errors on English without hurting no-space/CJK/code.
*   **Spike benignity (M2–M3).** Label compression spikes “benign” if CE is improving within the window; ensure we don’t kill runs prematurely.
*   **Semantic view separation (M1).** Confirm logging separates E-span, M-point, D-span.
*   **Reverse lattice.** Round-trip bytes ↔ v0/v1/v2 must be exact in fuzz tests (≥1k random samples).

---

## 5) HRM hierarchical recurrent reasoning [^3][^4][^24]

We partition reasoning into **C_hrm clusters**. Each outer step:

1. We route to **exactly one HRM cluster** (`top-1` cluster routing).
2. That cluster runs its own **HRM-L**, **HRM-M**, and **HRM-G** stack.
3. We update global state `hG`, integrate transplanted FFN outputs, broadcast control back to L via FiLM, and either halt or continue for another outer step.

Each cluster has its own expert banks (see §6). Clusters are *not* shared across steps, but a per-step router chooses which cluster to activate. Shared fixed paths provide early stabilization.

### 5.1 HRM cluster routing [^16]

**Inputs.** The previous outer step’s global state `hG`.

**Mechanism.**

* Project `hG` to a query `q_hrm = W_hrm_q · hG`.
* Compare against learned cluster prototypes `{μ_c}` with logits `ℓ_c = q_hrm · μ_c`.
* **Select top-1** cluster `c* = argmax_c ℓ_c`.

**Load-balance and stability.**

* Apply auxiliary Switch-style load-balance loss (α≈1e-2).
* Apply z-loss (1e-3) on router logits.
* Maintain entropy targets (looser early, tighter later).
* HRM **always** picks **one** cluster per outer step; there is no “1→2 schedule” for HRM clusters at runtime.

When innovation-based router bias is enabled (see §7.2.1), we add a small bias term favoring clusters historically good at handling “surprising” spans.

#### 5.1.1 Optional top‑2 HRM cluster routing (policy‑gated; later)

A policy‑gated variant permits **top‑2** cluster routing for high‑complexity tasks under static shapes. Capacity factor is **1.25** with **pre‑allocated** buffers for the second cluster. A complexity predicate (e.g., high innovation and large prototype distance) enables this mode, typically **restricted to the verify path**. Disable automatically if step‑time or memory guards are exceeded.

#### 5.1.2 Soft co‑activation across non‑selected clusters (optional; later)

Two **low‑weight** S‑REG‑scaled summary readouts from non‑selected clusters can be added to the active cluster’s context. These paths are **always present** at compile time but start with very small priors and may be automatically frozen or pruned via S‑REG attribution if impact remains negligible.

### 5.2 Widths & iteration caps

We parameterize HRM widths by **band** and **variant**.

* Let **d_L** and **d_M** be the HRM‑L and HRM‑M band widths for the chosen model variant (see **§2.2**).
* **HRM‑G** width is **2·d_L**.
* Iteration caps are unchanged; only widths are variant‑specific.

**Iteration caps:**

* **HRM‑L:** iteration cap `k ≤ 4`.
* **HRM‑M:** iteration cap `U ≤ 2`.
* **HRM‑G:** one update per outer step.

**Routing:** still **top‑1** everywhere (clusters, HRM‑L/M experts, FFN family→cluster→expert).

**Notes.**

* For **Q0.5B**, we keep `d_M = 1.5·d_L`.
* For **Q1.5B**, we **override** `d_M` (not 1.5×) to reduce quadratic cost; see **§2.2**.

#### 5.2.1 Iteration heuristics & auto‑tuning triggers

**Targets.** Typical steady‑state usage aims at **L≈3 iterations** and **M≈1–2 iterations**, subject to per‑task limits.

**Signals.** Adjust within caps using:

* innovation magnitude (AFA),
* halter margin,
* (|\Delta h_G|) norm trends,
* local overflow/dead‑expert rates.

**Policy.** Within the configured caps, raise or lower the *suggested* inner‑loop counts per sample using a small bias on the halter’s continue logit and on per‑band “continue” thresholds. This works jointly with §9.4 auto‑tuning of **caps** (outer_max, l_max, m_max). All adjustments are logged and bounded by caps.

#### 5.2.2 Learned window indexing for HRM‑M (optional; later)

A pointer mechanism can **index one additional v1 chunk** into HRM‑M attention per inner step:

* **Inputs:** current (hM_{u-1}), pooled (hL_k), and optional task features.
* **Output:** a single extra key/value window chosen from nearby v1 chunks.
* **Regularization:** a small penalty discourages overuse; budgeted to ≤1 extra window per u.
* **Implementation:** pre‑allocate KV buffers for the extra window; keep shapes static.
* **Status:** optional later experiment; disabled by default.

### 5.3 Exact loop inside a chosen cluster

Below, `c*` is the chosen cluster this outer step. We drop `c*` from notation for brevity.

**HRM-L inner loop (t = 1…k)**

```text
ctxL_t = Attn_L(hL_{t-1}, v0)                      # [T0, d]
eL_t   = ExpertL(hL_{t-1})                         # top-1 expert from L bank (hetero sizes)
hL̃_t  = hL_{t-1} + DropPath(eL_t)
hL_t   = GRU_L(hL̃_t, ctxL_t)
hL_t  += w_fixL · FixedL(hL_{t-1})                 # shared fixed HRM-L path with per-cluster scalar gate
```

**HRM-M inner loop (u = 1…U≤2)**

```text
poolL = Pool(hL_k)                                  # mean/max over low-band tokens
ctxM  = Attn_M(hM_{u-1}, v1, extra_kv=poolL)
eM_u  = ExpertM(hM_{u-1})                           # top-1 expert from M bank
hM̃_u  = hM_{u-1} + DropPath(eM_u)
hM_u  = GRU_M(hM̃_u, ctxM)
hM_u += w_fixM · FixedM(hM_{u-1})                   # shared fixed HRM-M path with per-cluster scalar gate
```

```text
poolM = Pool(hM_U)
poolL = Pool(hL_k)
xG    = concat(poolL, poolM, CLS(v2))

hG'   = GRU_G(hG, xG)

# Query FFN knowledge bank
q     = Wq · hG'                                        # [d]
y_FFN = RetrieveFFN(q, v0, high_token_prior=v2)         # (family→cluster→expert; top-1)

# S-REG: scale routed FFN residual before integration
y_FFN = (s_type.ffn_retrieve · s_layer.G) · y_FFN

# Integrate and produce FiLM
hG''  = G_update(hG', y_FFN)                            # small MLP / GRU integration

(γ,β) = FiLM(hG'')                                      # produce FiLM parameters
(γ,β) = (s_type.film · s_layer.L) · (γ,β)               # S-REG: scale broadcast magnitude

# Apply FiLM and decode bytes
hL_mod = (1 + γ) ⊙ hL_k + β
logits = ByteHead(hL_mod)
```

**Notes.**

* The donor family’s **fixed FFN expert** (if added as residual) is also scaled: `y_fixed ← (s_type.ffn_fixed · s_layer.G) · y_fixed` before adding to `y_FFN`.
* S-REG multiplies existing fixed-path gates but does not replace them.

#### 5.3.1 Learned residual aggregator (replaces hard adds)

Residual merges that previously used a hard add are implemented as a **learned convex aggregator**:

[
\tilde{h} ;=; \alpha\cdot x ;+; (1-\alpha)\cdot r,\quad
\alpha ;=; \sigma!\big(w^\top f([x,r])\big),
]

where (f) is a small feature projection (e.g., LayerNorm(x), LayerNorm(r), and their dot). Optionally apply **LayerScale** on the output to keep magnitudes in range. **S‑REG scales remain outside** this aggregator so attribution is preserved. Shapes are unchanged.

### 5.4 Halting (outer steps) [^30]

The model iterates outer steps until the halter stops or policy caps are reached.

* **Halter:** 2‑layer MLP with ACT over (hG''); step penalty (\lambda_o=0.01).
* **Styles:** task‑specific (e.g., cosine_mlp, bce_no_extra_pass).
* **Cosine veto:** optional stop when (|\Delta h_G|) is small under high caps.
* **Convergence guard:** small regularizer when outer caps rise (see §11.2).
* **Train vs inference caps:** training **outer_max ≤16**; inference may allow higher, per policy.
* **Gradients:** default **truncated BPTT across outer steps** (full gradients across inner L/M loops of the final step).

**Inference halting bias** [^28] — THPL can set:

```yaml
policy.halt_offset: +5.0   # default 0.0; clipped to +7.5
```

**Mixed‑BPTT sprinkles** [^31] — with small probabilities, enable gradients through the first or first+second outer steps, run a backward pass, clear gradients, then run the final step and backward. Activation checkpointing and OOM‑based auto‑throttling are required.

#### 5.4.1 Inference halting bias (logit offset) [^28]

THPL exposes **policy.halt_offset** to bias continue logits **at inference only**:

```yaml
policy.halt_offset: +5.0   # default 0.0; clipped to +7.5
```

This reduces false positives/negatives on compute‑sensitive tasks without retraining. Effects are profiled per task.

#### 5.4.2 Mixed‑precision hygiene for the halter

Halter MLP runs in the model’s global precision (see §11). When instability is detected (rare), only the *final* halter linear may be promoted to fp32 compute with fp16 weights to stabilize thresholds; this retains compile invariants.

#### 5.4.3 Mixed‑BPTT sprinkles (prefix passes) [^31]

To provide upstream credit while bounding memory:

* With small probabilities per batch, **enable gradients through the first or first+second outer steps**, perform a backward pass, **clear grads**, then run the final step and backward again.
* Activation checkpointing on outer‑step boundaries is required; on OOM sentinels, automatically reduce sprinkle rates.
* Counters for truncated vs sprinkle modes are logged.

### 5.5 FiLM broadcast and byte head

After integrating FFN knowledge (hG''), we generate FiLM parameters and steer output token logits.

- Predict (γ, β) either per channel or in 8–16 channel groups.
- Apply to the final HRM-L state hL_k:
  hL_mod = (1 + γ) ⊙ hL_k + β
- Final byte logits: ByteHead(hL_mod).

FiLM magnitude can optionally be gated by innovation (see §7.2.1): scale (γ,β) by σ(w · innov_mean) to avoid over-steering when the model is already confident.

#### 5.5.1 Multi‑token prediction head (optional; later)

An optional **MTP head** predicts the next (m) bytes jointly:

* **Placement:** parallel to the ByteHead over (hL_{\text{mod}}).
* **Training:** teacher forcing with a joint cross‑entropy on (m) future bytes; small loss weight.
* **Constraints:** fixed small (m) (e.g., 2–4) with pre‑allocated projection matrices; no autoregressive graph changes.
* **Ablation:** enable only for a small, budgeted run; carry forward only if quality/latency trade‑offs are favorable.

### 5.6 HRM‑L micro‑experts (variant‑gated)

**MoirAI‑2.9B‑A0.9B:** **ON** by default (k = 3–4 tiny GRU cells; top‑1 router; capacity factor 1.25) with a **Δ step‑time ≤5%** guard; auto‑disable if exceeded.
**MoirAI‑0.9B‑A0.3B:** **OFF** by default; enable only on targeted workloads that show net quality or step‑time benefit.
Monitor local pick‑rates, outer‑step reductions, and net step‑time.

### 5.7 Dump head (introspection-only)

We allow a **no-op dump head**: a parallel byte-distribution head that emits logits/probs purely for telemetry and downstream tools, **without affecting gradients**.

* This head can be placed at `pre_film` (hL_k) or `post_film` (hL_mod).
* It is fully detached from loss, halting, routing, FiLM, etc.
* Config:

```yaml
dump_head:
  enable: true
  location: "pre_film"     # or "post_film"
  temp: 1.0
```

* Use: compare dump logits vs final logits, log disagreement per token, visualize hallucination-like divergences.
* Safety: latency overhead should be <2%. Tests assert zero effect on main loss/grad.

### 5.8 Scale hooks (HRM-L/M/G)

We apply S-REG to **non-GRU** residuals inside HRM loops. GRU updates themselves remain unscaled.

**HRM-L inner loop (t = 1…k)**

```text
ctxL_t = Attn_L(hL_{t-1}, v0)                           # [T0, d]
eL_t   = ExpertL(hL_{t-1})                              # top-1 expert from L bank

# Residual aggregations with S-REG
hL̃_t  = hL_{t-1} + (s_type.attn        · s_layer.L) · DropPath(ctxL_t)
hL̃_t  = hL̃_t    + (s_type.hrmL_expert · s_layer.L) · DropPath(eL_t)

# GRU update
hL_t   = GRU_L(hL̃_t, hL_{t-1})

# Shared fixed path (preserves per-cluster gate w_fixL)
hL_t  += (s_type.hrmL_fixed · s_layer.L) · (w_fixL · FixedL(hL_{t-1}))
```

**HRM-M inner loop (u = 1…U≤2)**

```text
poolL = Pool(hL_k)
ctxM  = Attn_M(hM_{u-1}, v1, extra_kv=poolL)
eM_u  = ExpertM(hM_{u-1})

hM̃_u = hM_{u-1} + (s_type.attn        · s_layer.M) · DropPath(ctxM)
hM̃_u = hM̃_u    + (s_type.hrmM_expert · s_layer.M) · DropPath(eM_u)

hM_u  = GRU_M(hM̃_u, hM_{u-1})
hM_u += (s_type.hrmM_fixed · s_layer.M) · (w_fixM · FixedM(hM_{u-1}))
```

**Global & FiLM** are handled as in §5.3.

#### 5.8.1 Innovation‑modulated S‑REG clamp

A runtime clamp modulates S‑REG application using AFA innovation while **leaving S‑REG parameters stop‑grad**:

Let `innov_mean` be the per‑sequence EMA‑normalized innovation scalar from AFA (0 at low surprise; ↑ with surprise). Define

[
s_\text{eff} ;=; s \cdot \mathrm{clamp}!\Big(1 - \kappa_\text{mod}\cdot \mathrm{norm_innov},; [\alpha, 1]\Big),
]

with defaults **(\alpha = 0.5)** and **(\kappa_\text{mod}\in [0.0, 0.3])**. `norm_innov` is a z‑score of `innov_mean` computed with a **stop‑grad EMA** baseline. The clamp is **multiplicative** and applied only to the **applied scale**; the S‑REG parameters ( \theta ) do **not** receive gradients from this path.

**Scope.** Apply to residual types that can destabilize early (e.g., `ffn_retrieve`, `film`, HRM fixed paths), and log the realized attenuation **(1 - s_\text{eff}/s)** for attribution audits. The clamp is disabled for GRU cores (they are never S‑REG‑scaled).

**Safety.** The floor (\alpha) prevents over‑suppression under sustained surprise. When innovation is low, (s_\text{eff}\to s); when high, (s_\text{eff}) reduces smoothly toward (\alpha s).

---

## 6) Hierarchical experts and transplanted knowledge

We have two MoE systems:

1.  **HRM reasoning experts** for HRM-L and HRM-M. These are *trained from scratch*, are **heterogeneous in size**, and are organized per HRM cluster.
2.  **FFN knowledge experts** transplanted from donor dense models (Qwen by default; optional Mamba mixers). These act as a knowledge bank, exposed to HRM-G via RetrieveFFN.

Both use **top-1 routing** (capacity factor 1.25). Both use Switch-style load balancing (Switch-LBL) and z-loss to avoid collapse. Both rely on static shapes and static `top-k=1` dispatch for compile-friendliness.

### 6.1 HRM reasoning experts (heterogeneous sizes + shared fixed paths)

Each HRM cluster has its own **HRM-L** and **HRM-M** expert banks. Banks are **not shared across clusters**.

#### 6.1.0 HRM width rounding & recording policy

**Purpose.** Hardware‑friendly expert widths, static shapes, and reproducible configs.

**Apply to** all HRM‑L/M experts: `raw_width = ratio × band_width`.

**Rounding rules (in order):**

1. If `raw_width` is within **±5%** of a power‑of‑two, **snap** to that power‑of‑two.
2. Otherwise, **round to nearest even**; prefer a multiple of **64** if the change is **≤2%**.
3. Enforce **minimum expert width**: **0.5B ≥ 512**, **1.5B ≥ 1024**.
4. **Record** in configs: `ratio_recorded = rounded_width / band_width` (so the recorded ratio can differ slightly per band).

**Reference implementation:**

```python
def round_width(raw, min_width, pow2_tol=0.05, align=64, align_tol=0.02):
    raw = max(raw, min_width)
    p = 2**round(log2(raw))
    if abs(raw - p) / raw <= pow2_tol:
        return int(p)
    a = round(raw / align) * align
    if abs(raw - a) / raw <= align_tol:
        return int(a)
    return int(2 * round(raw / 2))
```

#### 6.1.1 HRM‑L (operates on v0; state width = d_L)

**Expert bank per cluster (top‑1): five size tiers + escape; multiplicities differ by variant.** Widths come from §2.2 via the rounding policy in §6.1.0.

* **MoirAI‑0.9B‑A0.3B (d_L = 1920)** — **8 experts/cluster:**

  * **0.2667×:** **×4**
  * **0.50×:** ×1
  * **0.75×:** ×1
  * **1.00×:** ×1
  * **1.25× (escape):** ×1

* **MoirAI‑2.9B‑A0.9B (d_L = 3072)** — **6 experts/cluster:**

  * **0.3333×:** **×2**
  * **0.50×:** ×1
  * **0.75×:** ×1
  * **1.00×:** ×1
  * **1.20× (escape):** ×1

**Per‑expert wrapper.** `In: d_L → d_e` → 2‑matrix MLP @ `d_e` → `Out: d_e → d_L` as a residual around GRU‑L.
**Shared fixed path.** `FixedL_shared: d_L → d_L` (tied across clusters), applied with per‑cluster scalar gate `w_fixL_c`; anneal schedule is S‑REG/fragility‑aware (§6.1.7).

#### 6.1.2 HRM‑M (operates on v1; state width = d_M)

**Expert bank per cluster (top‑1): five size tiers + escape; one of each per cluster (both variants).** Widths from §2.2 via §6.1.0.

* **MoirAI‑0.9B‑A0.3B (d_M = 2880)** — 5 experts/cluster: { 0.2667×, 0.50×, 0.75×, 1.00×, 1.25× }
* **MoirAI‑2.9B‑A0.9B (d_M = 4096)** — 5 experts/cluster: { 0.3333×, 0.50×, 0.75×, 1.00×, 1.20× }

**Per‑expert wrapper.** `In: d_M → d_e` → 2‑matrix MLP @ `d_e` → `Out: d_e → d_M` as a residual around GRU‑M.
**Shared fixed path.** `FixedM_shared: d_M → d_M`, per‑cluster gate `w_fixM_c`; anneal per §6.1.7.

#### 6.1.3 Routing, stability, and curricula (HRM‑L/M)

* **Routing:** **top‑1** expert with capacity factor **1.25**.

* **Stabilizers:** Switch‑LBL α: **0.02 → 0.01 → 0.005** (warm / main / late); **z‑loss = 1e‑3**; router τ: **1.2 → 1.0**; router logit noise: **1.0 → 0.2**.

* **Compute priors (raised to bias small tiers):**

  * **κ_L = 0.34** (range 0.33–0.35 acceptable),
  * **κ_M = 0.25**.
  * Subtract `κ·(d_e/d_band − 1)` from router logits.

* **Size curricula (small → large unlock):**

  * Early enable only the **smallest tier** (0.2667× on Q0.5B; 0.3333× on Q1.5B).
  * Then enable 0.50× and 0.75×.
  * Then 1.00×; keep **escape (>1.0)** last and **rare**.
  * Maintain **tier‑level Expert‑Choice fallback** when starvation/overflow persists (>1k steps) without changing global top‑1 policy.

* **Pick‑rate intent (steady state):** see **§2.2**; overwhelming mass to the smallest tier in both bands.

* **Halting / early stabilization:** keep halter behavior; allow **+1 outer step** temporarily if small‑tier dominance causes transient overflow during warm‑start; revert once routers settle.

#### 6.1.4 Remove vs repurpose shared fixed paths

Shared fixed paths (`FixedL_shared`, `FixedM_shared`) start as stabilizers. We then either remove or repurpose.

**Default (remove):**

*   Track “fixed-share” = average scalar gate usage.
*   If global fixed-share <20% for ≥N steps *and* forcibly zeroing all fixed gates (`a_c, a'_c → 0`) causes ≤0.5% absolute drop on our main metric, we leave them off and consider them removed.

**Repurpose (keep):**

*   Freeze routed experts.
*   Train `FixedL_shared` / `FixedM_shared` to imitate routed experts using reverse-KL (routed → fixed) plus orthogonality on pooled expert outputs.
*   Only unfreeze per-cluster adapters / LN gains *inside* the fixed modules; do not widen.
*   Goal: make the fixed path a cheap “summary expert” for each cluster.

Cluster scaling per model variant:

*   **Q0.5B → 4 clusters**
*   **Q1.5B → 6 clusters**

We size clusters and fixed paths so HRM total params sit at ≳~¼ of FFN bank parameters.

#### 6.1.5 HRM parameter budget (formulas & accounting)

Let `d_L, d_M` be the band widths, and `d_e` the rounded tier width.

**Per‑expert (two‑matrix MLP) approximate params:**
[
P_{\text{expert}} ;\approx; 2,d_{\text{band}},d_e ;+; 2,d_e^2.
]

**Shared fixed paths:**
[
P_{\text{fixed}} ;\approx; 2,d_L^2 ;+; 2,d_M^2.
]

**Totals.** Sum across clusters, tiers, and multiplicities from §2.2. Report **total** params in §2 and track **active** params/token via routing logs (top‑1 experts, fixed‑path scales, FiLM and attention projections actually executed). Use the attribution and router telemetry to monitor how active compute evolves as anneal proceeds.

#### 6.1.6 Why this works (HRM sizes & rounding)

* **Small‑tier dominance** via power‑of‑two / 64‑aligned minima lifts average **granularity** (3.5 on Q0.5B; ~3.0–3.1 on Q1.5B) without exploding parameters.
* **Five ratios per band** preserve routing flexibility; a **rare escape tier (>1.0)** gives headroom without dominating compute.
* **Budgeted Q1.5B** lowers **M** band width (4096) and trims **L** multiplicity, cutting quadratic cost while preserving compile invariants and meeting **≥3× granularity**.
* The rounding policy yields **hardware‑friendly** dimensions and **reproducible configs** (`ratio = rounded_width / band_width`), and pre‑tracing all shapes keeps **compile warm‑up** complete and fast.

#### 6.1.7 Fixed‑path anneal scheduler (S‑REG‑driven; fragility‑aware)

Anneal the shared fixed HRM paths using **attribution** and **fragility**:

* **Attribution driver:** increase anneal rate when S‑REG shows routed paths contributing the majority of residual magnitude for the band; slow when fixed paths still carry significant share.
* **Fragility score:** combine (i) donor alignment trend (L_{\text{align}}) and (ii) H‑Net stability signals (e.g., “hockey‑stick” health, chunk drift). High fragility pauses or slows anneal.
* **Floor & safety:** maintain a floor on fixed‑path scale during warm‑start; never hard‑drop without a removal test (see §6.3).

### 6.2 FFN knowledge experts (transplanted)

These are **knowledge / recall banks** attached to HRM-G. They are carved from donor dense models (Qwen by default, optional Mamba). Instead of fully retraining, we carve neurons, bake in donor gates, partially re-initialize slices to add diversity, and wrap them with adapters. Routing is a 3-tier hierarchy.

#### 6.2.1 Families, layout, flattening & **data‑driven donor layer selection** [^5][^6]

**Families.** Each donor family contributes one **flattened global FFN bank** shared across the model. Family interfaces (shared (A_{\text{in}}, A_{\text{out}}) and donor LN stats) align donor activations to native widths.

**Data‑driven capture‑layer selection.** Rather than a fixed list, select donor layers by a calibrated score:

* Compute per‑layer metrics on a calibration slice (10–50M tokens): neuron activation stats, **corr_with_high_token_codes**, domain‑conditioned entropies, and co‑activation structure.
* Score layers by a weighted sum of semantic diversity, stability, and coverage; **choose the top‑S** under a budget (S) (typ. 9–13).
* If instrumentation is unavailable, a **seed set** may be used to bootstrap; the calibrated selector supersedes the seed once metrics are available.

**Flattening across donor layers.** Pool neurons from captured layers, cluster into **K = 8** clusters per family, carve sub‑experts with limited neuron overlap, and bake donor gates into weights via gate‑compensation. Routing remains **family→cluster→expert (top‑1)**.

#### 6.2.2 Calibration & carving (per family)

Step-by-step:

1.  **Calibration pass** (10–50M mixed tokens):
    *   For each donor FFN layer, record gate/up/down activations, per-neuron stats (`freq_on`, `mean_abs`), domain-conditioned entropies, co-activations, and **cluster-conditioned gate means** `s_{j,c}`.
2.  **Feature extraction**:
    *   Build `φ_j = [ freq_on, mean_abs, entropies, PCA(co-act), corr_with_high_token_codes ]` per donor neuron `j`.
3.  **Cluster formation**:
    *   Run k-means++ (or spectral clustering on the co-activation graph) to produce **K = 8 clusters** per family. Each cluster `c` is a semantic / domain / style neighborhood.
4.  **Expert carving inside each cluster**:
    *   Within cluster `c`, run k-means++ again to form centroids for sub-experts `E_{c,i}`.
    *   For centroid `i`, pick the nearest neuron set `J_{c,i}`. Each expert gets width `f_i = |J_{c,i}|`.
    *   Enforce ≤10% overlap between experts to keep them mostly disjoint.
    *   Gate-compensation: for each neuron `j` in cluster `c` we have `s_{j,c}`, its average gate scale (from calibration). For donor `W_up`, `W_down`:
        *   `W1'[:, j] = s_{j,c} · W_up[:, j]`
        *   `W2'[j, :] = W_down[j, :]`
    *   This bakes donor gating into weights and lets us **drop the explicit donor gate at runtime**.

(We can optionally run a cheap least-squares fit on a small calibration slice to refine `s_{j,c}` after transplant.)

#### 6.2.3 Partial re‑initialization (diversity injection) — schedule & guardrails

Initial refresh fraction **(\rho_{\text{applied}}=0.05{-}0.10)** per expert (default **0.10**). Maintain a **default long‑term target (\rho_{\text{target}}=0.20)** and **optionally** raise to **0.25 only if** the *diversity MoE tier is disabled* (§6.2.10) **and** the transplant remains stable under the guardrails below.

**Top‑up rule (in +0.05 steps).** Increase only when **all** hold within the last 5k–10k steps:
(a) alignment improves (downward slope in (L_{\text{align}})),
(b) S‑REG attribution to routed paths is stable (no persistent drift in contribution shares),
(c) dev perplexity spike ≤ **0.2%** over a 2k‑step window,
(d) router overflow ≤ **5%** and dead‑expert ≤ **10%** (tier‑level EC fallback not persistently engaged).

**Floor for tiny experts.** Refresh at least `max(32, ceil(0.05·f_i))` neurons so even small experts gain plasticity.

**Never refresh** the family’s **fixed** FFN expert; that path is reserved for stabilization/repurpose (§6.3) and as the value‑cache anchor (§8.3).

Optimizer grouping: refreshed rows/cols use a higher LR multiplier than retained donor rows/cols; both follow the main schedule’s warm‑up/decay.

#### 6.2.4 Fixed FFN expert (per family)

Each donor family also has a **fixed FFN expert** that is always available as a small residual path:

*   Score donor neurons for broad utility with
    `U_j = α·freq_on + β·mean_abs − γ·entropy_level − γ'·entropy_domain`.
*   Take high-`U_j` neurons to build a single MLP:
    *   `W1' = s̄ ⊙ W_up` with `s̄` = global gate mean.
    *   `W2' = W_down`.
*   This fixed expert starts at small weight in training and gets annealed down as routed experts stabilize.
*   We can **repurpose** it instead of annealing to zero:
    *   Freeze routed experts.
    *   Train tiny adapters / LN gains on the fixed expert via reverse-KL + orthogonality.
    *   Make it a distilled summary expert.
*   Fixed expert is **never** partially re-initialized.

#### 6.2.5 Routing for FFN knowledge (family → cluster → expert) [^8]

We route in three tiers, then optionally add the fixed expert residual:

1.  **Tier 0: family selection**
    *   Score `score_f = q · μ_f + β · prior_f`.
    *   Pick **top-1 family**.
2.  **Tier 1: cluster selection within that family**
    *   Use a **Mixture-of-Routers (MoR)**:
        *   One head scores with the live query `q` from `hG'`.
        *   One head scores with a high-token prior (from v2 / topic frequencies / domain tags).
        *   We learn a mixer `α(hG') ∈ [0,1]` that blends them:
            `ℓ_c = α(hG') · (q · μ_c) + (1 − α(hG')) · prior_c`.
    *   Pick **top-1 cluster** `c`.
3.  **Tier 2: expert selection within that cluster**
    *   Route to **top-1 expert** in that cluster via `q · μ_{c,i}`.
4.  **Fixed FFN expert residual**
    *   Add the chosen family’s fixed FFN expert output as a low-weight residual.
    *   Anneal its scale to 0 unless doing the “repurpose” path.
5.  **Schedule / curriculum**
    *   Warm-up: only 1 active FFN cluster per family.
    *   After ~10% of training tokens, allow routing to 2 clusters per family.
    *   Switch-LBL + z-loss apply at family→cluster and cluster→expert stages to maintain load-balance and avoid collapse.

When innovation-based routing bias is enabled (see §7.2.1), we increase MoR’s reliance on the query head (vs prior) for high-innovation spans.

#### 6.2.6 Adapters & alignment — **gains‑then‑biases** policy [^23]

Adapters (A_{\text{in}}, A_{\text{out}}) and donor LN stats align donor to native widths. If **(L_{\text{align}})** stalls for >1 epoch:

1. **Unfreeze LN gains** (biases remain frozen) to rescale without recentering.
2. If still stalled after another epoch, **unfreeze LN biases** with a strict LR multiplier (0.1×) and an L2 penalty.
3. **Re‑lock** if donor centering drifts beyond **2σ** of the calibration stats.

#### 6.2.7 Summary of transplant pipeline

For each donor family (e.g. Qwen), for each cluster `c`, for each expert `E_{c,i}`:

1.  **Carve neurons** to form `J_{c,i}` with ≤10% overlap.
2.  **Gate-compensate** donor weights: bake per-cluster gate means `s_{j,c}` into `W_up` columns to form `W1'`, reuse donor `W_down` rows to form `W2'`, and drop donor gating at runtime.
3.  **Partial re-initialize** ~10% of neurons (growing toward 25% in M5) with donor-like stats; reapply `s_{j,c}`. Never touch the family's global fixed expert here.
4.  **Attach adapters and donor norms** (per-family shared `A_in` / `A_out`; per-expert wrappers `A_in^{(e)}`/`A_out^{(e)}` if needed for HRM width), and record train masks for optimizer grouping.
5.  **Register routing hooks** with the 3-tier router (family→cluster→expert).
    Routing uses top-1 at every tier plus a low-weight fixed expert residual. Capacity factor is 1.25. All stabilized with Switch-LBL α=0.02→0.01→0.005, z-loss=1e-3, τ 1.2→1.0, router noise 1.0→0.2.

#### 6.2.8 Per-Layer **Site Adapters** & **Layer-ID Scales (S-REG-LID)** for Trunk Taps

Trunk layers can “tap” the global family bank (UMoE-lite; see §7.4) using **site adapters** and **layer-ID scales**:

**Site adapters (per tapped trunk layer ℓ).**

To keep parameters tiny and compile-friendly, we use **diag-scale + bias** adapters around both the **fixed** and **routed** FFN paths:

```text
y_fixed_att(ℓ)  = A_out_fixed(ℓ) · FixedFFN( A_in_fixed(ℓ)  · y_mix )  # diag+bias both sides
y_expert_att(ℓ) = A_out_route(ℓ) ·  E_top1( A_in_route(ℓ)   · y_mix )
y_tap(ℓ)        = p(ℓ) · y_expert_att(ℓ) + y_fixed_att(ℓ)              # p(ℓ)∈[0,1], init≈0.3
```

* `A_in_* (ℓ), A_out_* (ℓ)` are **per-layer diag+bias** (rank-0) parameters; identity-initialized.
* `p(ℓ)` is a per-layer scalar mixing weight (learned).
* Core expert weights are **shared** with the family bank (no duplication).

**Layer-ID scales (S-REG-LID).**

We extend S-REG with **per-layer vectors** that modulate residual magnitudes for (a) **trunk attention** and (b) **attention taps**:

```text
y_att(ℓ)  = x + [ s_type.attn · s_band.attn · s_lid.attn[ℓ] ] · DropPath( Attnℓ(x) )
y_tap(ℓ)  ←      [ s_type.umoe_tap · s_band.attn_tap · s_lid.attn_tap[ℓ] ] · y_tap(ℓ)
```

* `s_type.*` and `s_band.*` are the **type** and **band** scales you already added.
* `s_lid.attn[ℓ]`, `s_lid.attn_tap[ℓ]` are **learned per-layer scalars** (softplus-param) with conservative inits (§7.1.4).
* If UMoE-lite is disabled, set `s_lid.attn_tap[:] = 0.0` (or omit).

**Invariants.**

All adapters and scales are scalar or diagonal; shapes are static; routers and DSA K-buckets are unaffected.

#### 6.2.9 Parameter accounting, defaults, and non‑HRM totals

Let

* `d_n` = native trunk width (MoirAI‑0.9B‑A0.3B: **896**; MoirAI‑2.9B‑A0.9B: **1536**)
* `f_d` = donor FFN expansion for budgeting (**`6·d_n`** default)
* `S` = number of captured donor layers (default **13**; optional 9)
* `η` = carve fraction over captured donor mass (default **0.60**)
* `ρ_ov` = max neuron overlap between carved experts (default **0.15**)
* `L_tap` = number of tapped trunk layers (typ. **2** for MoirAI‑0.9B‑A0.3B; **4** for MoirAI‑2.9B‑A0.9B)
* `L_trunk` = trunk depth (**24** and **28** respectively)

**Donor FFN mass across captured layers (per family):**
[
P_{\text{donor_FFN}} = 2,S,d_n,f_d.
]

**Transplanted family bank (per family):**
[
P_{\text{bank}} ;\approx; \big(\eta \cdot (1+\rho_{ov})\big),P_{\text{donor_FFN}}.
]

**Family interfaces (shared (A_{\text{in}}, A_{\text{out}}) + donor LN):**
[
P_{\text{family_ifaces}} ;\approx; 2,d_n^2.
]

**UMoE‑lite site adapters (diag+bias in/out for fixed+routed) per tapped layer:**
[
P_{\text{site_per_layer}} \approx 8,d_n,\qquad
P_{\text{site_total}} \approx (8,d_n),L_{\text{tap}}.
]

**Routing prototypes (family + clusters):**
[
P_{\text{router_proto}} \approx (1+K),d_n \quad (K=8).
]

**Trunk attention projections (all layers):**
[
P_{\text{trunk}} \approx L_{\text{trunk}},(4,d_n^2).
]

**Non‑HRM subtotal (per family):**
[
P_{\text{nonHRM}} \approx P_{\text{trunk}} + P_{\text{bank}} + P_{\text{family_ifaces}} + P_{\text{site_total}} + P_{\text{router_proto}}.
]

Use these expressions to reproduce the totals reported in §2 and to sanity‑check variant deltas when adjusting `S, η, ρ_ov, L_tap`.

#### 6.2.10 Diversity MoE tier (optional)

As an alternative to heavy partial re‑init, attach **1–2 tiny “diversity experts”** (≈0.1× width) per cluster with a small router bias (~10%). Anneal the bias toward 0 as they specialize. Disable if **Δ step‑time >3%** without measurable perplexity gains.

### 6.3 Fixed experts policy (scaffold → **repurpose** by default) [^7]

Shared fixed paths (HRM‑L/M) and the per‑family fixed FFN expert begin as stabilizers and are **repurposed by default**:

* **Repurpose default.** Train small adapters/LN gains on the fixed module using reverse‑KL and orthogonality to make it a **summary expert**. The fixed FFN expert also serves as the **basis for the value‑cache surrogate** (§8.3).
* **Removal only when unequivocally safe.** If the “fixed share” is <20% for a sustained window and forcing all fixed gates to zero causes ≤0.5% absolute drop on key dev metrics, keep them off; otherwise, keep repurposed.
* **Compile safety.** Enabling/annealing/removing fixed paths never changes shapes; dispatch stays top‑1 everywhere.
* **Emergency bypass.** A debug mode can bypass per‑cluster gates and apply only the type scale (e.g., (s_{\text{type.ffn_fixed}})) to simplify attribution.

### 6.4 Expert & fixed-path scaling policy

**Donor FFN bank (transplanted).**
Scale routed expert and fixed-family outputs before integration at HRM-G:

```text
y_expert_top1 ← (s_type.ffn_retrieve · s_layer.G) · y_expert_top1
y_fixed_fam   ← (s_type.ffn_fixed    · s_layer.G) · y_fixed_fam
```

During **repurpose** (§6.3), freeze `s_type.ffn_fixed` to keep a stable teacher while training adapters/LN gains on the fixed expert.

**HRM reasoning experts (heterogeneous).**
HRM-L/M expert residuals are always wrapped by S-REG as in §5.8. Size-tier compute priors (κ) and routing losses are unchanged and independent of S-REG.

**Fixed paths (HRM-L/M).**
S-REG multiplies the contribution **after** the per-cluster scalar gates `w_fix{L,M}`: it does not alter gate anneal schedules or the remove vs repurpose decision flow.

### 6.5 S‑REG‑guided freezing and pruning policy

Use S‑REG attribution telemetry to optimize capacity:

* **Freeze** modules with consistently low impact (scales near 0 and minimal contribution share) to save training updates.
* **Mark for pruning** at the next static re‑compile window (e.g., M5‑LC) when a module’s contribution remains negligible across evaluation suites.
* **Emergency bypass.** For fixed FFN paths and fixed HRM paths, a **clear‑glass** mode can bypass gates and apply only the type scale (e.g., (s_{\text{type.ffn_fixed}})) to debug attributions.

---

## 7) Attention, Mixing, Stability, and Long-Context Strategy

MoirAI's attention is not monolithic. It is a hierarchical system with different strategies for different bands (H-Net v0/v1/v2) and the main trunk. We use a per-layer backend registry to combine local sliding-window attention, cheap global "bleed," and powerful but sparse global hops (DSA/Power). The system is augmented with optional backends and stability knobs to reduce attention sink, tame activation spikes, and provide uncertainty signals.

### 7.1 Canonical Attention Layouts (24-layer and 28-layer presets)

These presets define the default mix of attention mechanisms across the trunk layers. They provide a reproducible recipe for balancing local processing, cheap global updates, and powerful but sparse global reasoning. They serve as reference starting points for the per-layer backend registry.

#### 7.1.1 Shared runtime knobs and mechanism definitions

```yaml
attn_runtime:
  mods_default:
    silm: false             # enable at M5‑LC on trunk dotprod/DSA per §7.2.2
    ssa:
      enabled: true         # per‑query temperature; clamped; see §7.2.5
      base_tau: 1.0
      min_tau: 0.6
      max_tau: 1.4
    ga: true                # head‑gated attention; defaults per §7.2.3

  verify_bump:
    enabled: true
    conf_threshold: 0.25
    max_bumps: 3
    verify_bump_max_k: 4096
    fallback_if_capped: power

mechanisms:
  sw:         { window: 1024, dilation: 1,  nonlinearity: softmax }
  sw_dilated: { window: 1024, dilation: 2,  nonlinearity: softmax }
  sw_sigmoid: { window: 1024, dilation: 2,  nonlinearity: sigmoid }

  linear:
    feature_map: elu_plus_one
    stable_sum: true        # numerically stable reductions; see §7.2.7

  power:
    rank: 128
    mixer: residual_gate
    gate_init: 0.2

  dsa:
    k_schedule:
      mode: length_scaled_power2
      k_min: 256
      k_max: 2048
      depth_mult: { early: 0.9, mid: 1.0, late: 1.2 }
    indexer: tiny_mlp
    bias_with_silm: true
    sdpa_kernel: flash

positional:
  # Exactly ONE of the following must be true when long‑context is enabled.
  # Selection is frozen at M5‑LC and remains immutable for the run.

  hybrid_rope_nope:
    enable: true            # 3/1 RoPE/NoPE across trunk depth (see §10.1.5)
    qk_norm: { enable: true, eps: 1e-6 }
    theta_scale: { enable: false, factor: 1.0 }

  rotary_fraction:          # Alternative to the 3/1 pattern
    enable: false           # set true ONLY if replacing hybrid_rope_nope
    fraction_qk: 0.25       # apply RoPE to the first 25% of Q/K head dims
    qk_norm: { enable: true, eps: 1e-6 }

guards:
  - rule: "exactly_one(positional.hybrid_rope_nope.enable, positional.rotary_fraction.enable)"
  - rule: "ssa_disabled_where_sigmoid"    # SSA may not run on sigmoid attention layers
  - rule: "immutable_registry_after_M5LC" # backends/positional are frozen post‑compile
```

**Notes.**

* The **rotary_fraction** option (RoPE on the first 25% of Q/K dims) is a *mutually exclusive* alternative to the 3/1 RoPE/NoPE pattern. Select **one** at M5‑LC based on ablations at equalized compute; freeze thereafter.
* QK‑Norm remains enabled in either scheme to stabilize logits under fp16/bf16 (§11.1).

#### 7.1.2 28-Layer Preset (6 DSA Layers)

```yaml
model:
  name: moirai-28
  num_layers: 28
  layers:  # 1-based indexing
    - { id:  1, type: sw,         note: "local lexical features" }
    - { id:  2, type: sw,         note: "local patterns" }
    - { id:  3, type: dsa,        note: "first global anchor (DSA)" }
    - { id:  4, type: sw,         note: "local refinement" }
    - { id:  5, type: sw_sigmoid, note: "soft aggregation to avoid peaky early mix" }
    - { id:  6, type: sw,         note: "pre-anchor local" }
    - { id:  7, type: dsa,        note: "second global anchor (DSA)" }
    - { id:  8, type: sw,         note: "transition to mid" }
    - { id:  9, type: linear,     note: "ultra-cheap global bleed-through" }
    - { id: 10, type: sw_dilated, note: "effective horizon↑ via dilation" }
    - { id: 11, type: sw_dilated, note: "as above" }
    - { id: 12, type: power,      note: "cheap global smoother (pre-keystone)" }
    - { id: 13, type: sw_dilated, note: "prep for keystone" }
    - { id: 14, type: dsa,        note: "mid keystone (DSA)" }
    - { id: 15, type: sw_dilated, note: "post-keystone local+mid" }
    - { id: 16, type: sw_dilated, note: "maintain widened field" }
    - { id: 17, type: power,      note: "cheap global smoother (post-keystone)" }
    - { id: 18, type: sw_sigmoid, note: "support-evidence blending" }
    - { id: 19, type: sw_dilated, note: "final mid-range pass" }
    - { id: 20, type: dsa,        note: "enter late global stack (DSA)" }
    - { id: 21, type: sw,         note: "tighten to local before closure" }
    - { id: 22, type: sw_sigmoid, note: "gentle evidence merge pre-top" }
    - { id: 23, type: sw,         note: "stabilize logits path" }
    - { id: 24, type: sw,         note: "OPTIONAL flip to DSA for 7-DSA variant" }
    - { id: 25, type: linear,     note: "cheap preconditioner for late DSA" }
    - { id: 26, type: dsa,        note: "late global #2 (DSA)" }
    - { id: 27, type: sw,         note: "final local polish" }
    - { id: 28, type: dsa,        note: "last-layer global (DSA)" }
```

#### 7.1.3 24-Layer Preset (5 DSA Layers)

```yaml
model:
  name: moirai-24
  num_layers: 24
  layers:  # 1-based indexing
    - { id:  1, type: sw,         note: "local lexical features" }
    - { id:  2, type: sw,         note: "local patterns" }
    - { id:  3, type: dsa,        note: "early global anchor (DSA)" }
    - { id:  4, type: sw,         note: "local refinement" }
    - { id:  5, type: sw_sigmoid, note: "soft aggregation; anti-peak" }
    - { id:  6, type: sw,         note: "pre-anchor local" }
    - { id:  7, type: dsa,        note: "second early global (DSA)" }
    - { id:  8, type: linear,     note: "cheap global bleed-through" }
    - { id:  9, type: sw_dilated, note: "effective horizon↑ via dilation" }
    - { id: 10, type: power,      note: "cheap global smoother (pre-keystone)" }
    - { id: 11, type: sw_dilated, note: "prep for keystone" }
    - { id: 12, type: dsa,        note: "mid keystone (DSA)" }
    - { id: 13, type: sw_dilated, note: "post-keystone mid-range" }
    - { id: 14, type: sw_dilated, note: "maintain widened field" }
    - { id: 15, type: power,      note: "cheap global smoother (post-keystone)" }
    - { id: 16, type: sw_sigmoid, note: "support-evidence blending" }
    - { id: 17, type: sw_dilated, note: "OPTIONAL flip to DSA for 6-DSA variant" }
    - { id: 18, type: sw_dilated, note: "final mid-range pass" }
    - { id: 19, type: sw,         note: "tighten to local before closure" }
    - { id: 20, type: dsa,        note: "late global #1 (DSA)" }
    - { id: 21, type: sw,         note: "stabilize logits path" }
    - { id: 22, type: linear,     note: "cheap preconditioner for late DSA" }
    - { id: 23, type: sw,         note: "final local polish" }
    - { id: 24, type: dsa,        note: "last-layer global (DSA)" }
```

#### 7.1.4 Layer‑ID Scale Tables (24/28‑layer trunks)

Layer‑ID scale initializations for `s_lid.attn[ℓ]` and `s_lid.attn_tap[ℓ]` are **learned** (softplus‑param) and combine multiplicatively with type and band scales. Indexing is **1‑based**.

**(A) 24‑Layer Variant — MoirAI‑0.9B‑A0.3B (matches §7.1.3)**

```yaml
scales:
  layer_id:
    trunk:
      attn:      [0.50, 0.50, 0.60, 0.50, 0.50, 0.50, 0.60, 0.52, 0.53, 0.55, 0.53, 0.60,
                  0.53, 0.53, 0.55, 0.52, 0.53, 0.53, 0.50, 0.60, 0.50, 0.52, 0.50, 0.60]
      attn_tap:  [0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00,
                  0.00, 0.35, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
```

**(B) 28‑Layer Variant — MoirAI‑2.9B‑A0.9B (matches §7.1.2)**

```yaml
scales:
  layer_id:
    trunk:
      attn:      [0.50, 0.50, 0.60, 0.50, 0.50, 0.50, 0.60, 0.50, 0.53, 0.55, 0.53, 0.55,
                  0.53, 0.60, 0.53, 0.53, 0.55, 0.52, 0.53, 0.60, 0.50, 0.52, 0.50, 0.52,
                  0.55, 0.60, 0.50, 0.60]
      attn_tap:  [0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35,
                  0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35,
                  0.00, 0.00, 0.00, 0.00]
```

#### 7.1.5 Rotary‑fraction option (25% dims) [^13]

An alternative positional scheme applies **RoPE to the first 25% of head dims** per attention head, with the remaining dims unpositioned (NoPE) and **QK‑Norm** enabled.

**Configuration.**

```yaml
positional:
  rotary_fraction:
    enable: true
    frac: 0.25
    qk_norm: {enable: true, eps: 1e-6}
```

**Mutual exclusivity.** This option is **mutually exclusive** with the trunk’s hybrid 3/1 RoPE/NoPE layer pattern (see §10.1.5). Choose one scheme per run.

**Adoption criteria.** Run an ablation at matched compute against the hybrid 3/1 scheme on long‑context evaluations; **adopt only if** length‑generalization and stability **match or exceed** the hybrid pattern. Once selected at the long‑context compile milestone, the positional scheme is **frozen** for the run.

### 7.2 Attention Backends and Stability Knobs

We use a per-layer registry to assign different attention backends. The following are optional, configurable backends and post-processing steps, primarily for HRM-L/M and trunk layers that use dot-product or DSA.

#### 7.2.1 AFA (Uncertainty-Aware Attention) & Innovation Hooks

AFA replaces or blends with dot-product attention using a simple per-head linear dynamics model in a learned eigenbasis:

- Parameters per head (tied across HRM-L layers by default):
  - Orthonormal basis U ∈ R^{d_h×d_h}
  - Diagonal dynamics gain a ∈ (0,1] via a = sigmoid(ā)
  - Diagonal precision s ≥ 0 via s = softplus(ŝ)

Computation (per head):
1) Transform: q' = Uᵀ q, k' = Uᵀ k
2) Residual (innovation): r = q' − (a ⊙ k')
3) AFA logits: ℓ_afa = −0.5 · || r ⊙ √s ||² (elementwise squared Mahalanobis)
4) If in blend mode: ℓ = α·ℓ_afa + (1−α)·(q·k / √d_h)
5) Weights = softmax(ℓ); output = weights·V

Innovation side-info (stop-grad):
- innov_mean = mean_j ||r_j||₁
- innov_pw = mean_j (r_j² ⊙ s)

Hooks:
- Halting: add innovation features to the halter MLP (improves step decisions)
- FiLM: scale FiLM magnitude by σ(w·innov_mean) to avoid over-steering when confident
- H-Net: optionally add innovation as a feature to the v0 boundary head
- Routers: small bias term from innovation to HRM cluster and FFN MoR mixers (favor richer experts for surprising spans)

Milestones:
- Enable on HRM-L first in blend mode during M3 (ramp α from 0.2→0.7); extend to HRM-M in M4 only if M3 gains persist.

#### 7.2.2 SILM (Scale‑Invariant Logit Modulation) — **enablement policy** [^9]

SILM runs as a per‑head logit post‑processor to improve length generalization.

* **Where.** Apply on **trunk** layers that use **dot‑product** or **DSA** attention.
* **When.** **Enable at M5‑LC** (the first long‑context compile) and freeze thereafter.
* **Coupling with DSA.** Add the same distance bias to the DSA indexer score so pre‑selection aligns with post‑gather logits.
* **Tests.** Require ≤0.5% loss delta at ≥16× context and no degradation in DSA top‑K recall.

#### 7.2.3 GA (Gated Attention) — defaults [^25]

**Enable GA on both HRM‑L *and* HRM‑M by default.** Each head output is multiplied by a gate (g_h(q_h)\in(0,1)) after value mixing: (y_h\leftarrow g_h\cdot y_h).

* **Regularization.** L1 on gates (1e‑4) + **gate_floor=0.02** to prevent dead heads.
* **Learning rate.** LR multiplier 0.5 for gate parameters.
* **Compatibility.** GA composes with SSA (per‑query temperature) and with AFA blend mode; S‑REG scales the post‑gate residual as usual.
* **Sink mitigation.** Expect first‑token mass reductions vs softmax‑only baselines (monitored in §12).

#### 7.2.4 Attention Sink Mitigations (Sigmoid Attention & Logit Hygiene) [^20][^21] [^22]

If attention sink remains an issue, we have two additional tools for softmax-based attention layers:
1.  **Sigmoid Attention:** Replace softmax with an elementwise sigmoid, removing the across-key normalization that often creates the sink.
2.  **Logit Hygiene:** Apply pre-softmax cleanup: per-head logit centering, learned key-bias decay, and a smooth `LogitClip` to prevent extreme peaks that cause activation outliers.

#### 7.2.5 SSA (Selective Self‑Attention) — replacement lead [^10] 

**What.** A per‑query temperature `τ` (predicted by a small shared MLP) rescales logits before softmax: `logits' = logits / τ`. We **clamp** `τ ∈ [0.6, 1.4]` (matching §7.1.1) to avoid over‑flattening/spiking. Optionally rescale values with a small gate (usually OFF).


#### 7.2.6 DSA (Dynamic Sparse Attention) [^18]

**What.** Instead of attending to all keys/values, DSA uses a lightweight indexer to select the **top-K** most relevant KVs for each query. `K` is scaled dynamically with sequence length and bucketed to a power of two for compile safety. A `dense_short_circuit` is used for very short sequences. This provides near-linear cost for global attention.

#### 7.2.7 Precision caveats for linear / Power / DSA backends

* **Default precision:** **fp16** with **dynamic loss scaling** and **QK‑Norm**.
* **Stability aids:** explicit activation clamping in linear/Power/DSA; “logit hygiene” (centering, key‑bias decay, smooth clipping) for softmax‑based layers.
* **Fallback:** If a module remains unstable under fp16, use **bf16** with fp32 accumulators for that module only; record the exception.
* **FP8 experiments:** require QAT and explicit activation clamping; keep a softmax temperature floor.

### 7.3 Long-Context Strategy

#### 7.3.1 ESE (External Segment Encoder)

For very long documents, we replace raw byte ingestion with ESE. A small external encoder processes 1-2 KB byte segments into a few compact latent vectors, which are then fed into the MoirAI trunk. This drastically reduces sequence length and can be cached. The verify path can fall back to raw bytes for specific segments if confidence is low.

#### 7.3.2 Power Attention & Per-layer Registry

As an alternative to DSA, **Power Attention** provides linear-cost global mixing with a tunable state size `m`. Our **per-layer registry** allows us to assign different backends (`dotprod`, `power`, `dsa`) to different trunk layers at initialization, enabling a hybrid approach. For example, some layers can be `power` while others are `dsa`.

#### 7.3.3 Hybrid Positional Encoding (Trunk) [^13]

For better long-context extrapolation, trunk layers can use a hybrid positional encoding scheme: a repeating pattern of 3 layers with RoPE and 1 layer with NoPE, with QK-Norm applied to all trunk layers for stability.

#### 7.3.4 Fourier positional embeddings (optional; later)

**Scope.** Experimental **Fourier positional embeddings (FPE)** may be applied to **H‑Net internal encoders** (levels v0/v1/v2) or to **selected trunk layers** in a dedicated ablation branch.

**Constraints & compile safety.**

* Shapes are unchanged (FPE adds no sequence‑length–dependent parameters).
* If enabled on the trunk, FPE replaces that layer’s existing PE for the entire run; switching mid‑run requires a late‑swap distillation shield (§8.4).
* When enabled concurrently with hybrid RoPE/NoPE, FPE substitutes for the **NoPE** layers only (mutually exclusive with the rotary‑fraction option below).

**Configuration (concise).**

```yaml
position:
  fpe:
    enable: false
    where: { hnet_levels: [], trunk_layers: [] }  # e.g., trunk_layers: [9,13]
    spectrum: {bands: 4, min_freq: 1e-4, max_freq: 1.0}
    norm: {type: "layer", eps: 1e-6}             # keeps logits stable under fp16
    gating: {enable: true, init: 0.2, l1: 1e-4}  # learned gate; anneals toward fixed use
    freeze_after_m5lc: true                      # lock choice at the long-context compile
```

**Ablation policy.** Train a short branch with FPE on (a) v0/v1 only, (b) a small trunk subset, then carry forward **only** if long‑context retrieval does not regress and stability/latency remain within budget.

#### 7.3.5 Rotary‑fraction option (25%) (mutually exclusive with 3:1 RoPE/NoPE)

A trunk configuration can apply RoPE to the **first 25% of QK head dims** and use **NoPE + QK‑Norm** on the remainder:

* **Mutual exclusivity:** cannot co‑enable with the 3‑in‑4 RoPE / 1‑in‑4 NoPE layer pattern.
* **Decision point:** run a short M5‑LC ablation at **equal compute** versus the hybrid scheme; promote only if it **matches or improves** length generalization and stability.
* **Freezing:** whichever positional scheme wins at M5‑LC is frozen for the run.

### 7.4 UMoE-lite — Shared FFN Knowledge Experts in Attention

We allow selected trunk layers to **tap the global family bank** used at HRM-G, via tiny per-layer site adapters and S-REG-LID.

**Data flow at tapped trunk layer ℓ.**

```text
# Base attention output (any backend); GA/SSA happen inside.
y_att = AttentionBackendℓ(x, ...)

# Optional tap into the shared family bank (routed top-1 + fixed expert), with per-layer adapters:
y_fixed_att(ℓ)  = A_out_fixed(ℓ) · FixedFFN( A_in_fixed(ℓ)  · y_att )
y_expert_att(ℓ) = A_out_route(ℓ) ·  E_top1(  A_in_route(ℓ)   · y_att )

# Per-layer mixing of routed vs fixed:
y_tap(ℓ) = p(ℓ) · y_expert_att(ℓ) + y_fixed_att(ℓ)      # p(ℓ)∈[0,1], init≈0.3

# S-REG (type · band · layer-ID) scales the tap residual before adding:
y_tap(ℓ) ← [ s_type.umoe_tap · s_band.attn_tap · s_lid.attn_tap[ℓ] ] · y_tap(ℓ)

# Final residual adds (attention first, then tap):
y = x
y = y + [ s_type.attn · s_band.attn · s_lid.attn[ℓ] ] · DropPath( y_att )
y = y + y_tap(ℓ)                                       # already S-REG-scaled
```

**Warm-up & unfreeze.**

* **Warm-up (e.g., 30k steps):** freeze core experts; train only site adapters, `p(ℓ)`, and the MoR query-bias.
* **Unfreeze:** then unfreeze shared expert weights with a low LR multiplier (×0.1).

**Compute budget.** One extra expert MLP (top-1) + one fixed expert per tapped layer. Limit `L_tap` so step-time stays within budget (e.g., 2–4 tapped layers on the 1.5B model).

**Compile invariants.** One backend per layer; static top-1 dispatch; shapes unchanged.

### 7.5 Cross-feature Guards & Config Validator

To manage complexity, our configuration loader enforces these rules:

- SSA vs Sigmoid: For any band/layer, forbid ssa.enable when attention.impl=="sigmoid".
- Verify-only mode: If mode="verify_only", selective confidence may never abstain (only verify).
- Memory/compile guards for long-context: Never request a DSA K that was not compile-warmed; if violated, fall back to dense/windowed and log.
- Backend exclusivity:
  - HRM bands: enforce one backend per layer (no mixing).
  - Trunk: allow hybrid_heads (mixing Power and dotprod within a layer) only when hybrid_heads.enable=true in the registry and the split is fixed at init. Otherwise enforce one backend per layer.

### 7.6 Backend-aware residual scales

For any attention backend (`dotprod`, `sigmoid`, `linear`, `power`, `dsa`) in trunk or HRM bands, we scale the **post-mix** residual before adding it to the skip path:

```text
y_att = AttentionBackend(x, ...)
y     = x + (s_type.attn · s_layer.attn) · DropPath(y_att)
```

**Ordering with other knobs.**

* **GA (Gated Attention):** head gates apply first; S-REG scales the aggregated result.
* **SSA:** temperature affects logits inside the backend; S-REG scales the resulting residual.
* **DSA:** S-REG is orthogonal to K-selection and does not alter compile-warmed K buckets.

**UMoE-lite taps (if enabled).**
Scale the tap residual as a whole: `y_att_tap ← (s_type.umoe_tap · s_layer.attn_tap) · y_att_tap`.

### 7.7 Limited dynamic backend mixer (head‑scoped; dual‑exec)

A small **head‑scoped** mixer may run **both** DSA and Power **on ≤10% of heads in ≤2 trunk layers**, producing an output
[
y = \lambda , y_\text{DSA} + (1-\lambda), y_\text{Power},
]
with a learned (\lambda\in[0,1]) predicted by a tiny MLP over per‑head features. An **entropy/L1 penalty** drives (\lambda) toward **near‑binary**, annealing to one backend.

**Compute guard.** Enforce global **Δ step‑time ≤5%**; auto‑disable if exceeded. **Compile warm‑up** must trace both code paths for the selected layers/heads.

**Stability aid.** An optional mimic loss stabilizes the chosen backend as (\lambda\to{0,1}).

**Default.** **OFF** by default; enable only when ablations show consistent value within the compute guard.

### 7.8 Attention‑only ablation baseline (external + internal) [^26]

A two‑track baseline isolates trunk‑backend effects and, separately, the incremental value of HRM and MoirAI‑specific controls.

**Track A — External attention‑only baseline (no HRM).**
Use the *modded‑nanogpt* speedrun harness to train a 24‑layer attention‑only model (matching the MoirAI‑0.9B‑A0.3B trunk depth) at equalized FLOPs:

* **Backends:** windowed/dilated SDPA, DSA (length‑scaled (K) with power‑of‑two buckets), and Power Attention (fixed state size (m)).
* **Layout:** approximate MoirAI’s early/mid/late global anchor cadence with closest analogues.
* **Learned→fixed selection (later/secondary):** add a per‑layer learned backend selector that **anneals to one** backend per layer (near‑binary entropy penalty). After anneal, re‑evaluate at equalized FLOPs.
* **Outputs:** per‑layer backend preferences, long‑context scaling curves, and stability diagnostics.

**Track B — Internal MoirAI ablations (compile‑safe).**
Run within MoirAI with **UMoE‑lite taps OFF**, **AFA/SSA/GA OFF** (unless explicitly tested), and transplant features disabled:

1. **HRM‑OFF trunk baseline:** set **outer_max=1** and disable HRM dispatch (no L/M recurrence). Compare SDPA vs DSA vs Power at equalized FLOPs using the per‑layer registry.
2. **HRM‑ON trunk baseline:** enable HRM (standard L/M/G caps) while keeping MoirAI extras OFF. Measure the **incremental gain** vs HRM‑OFF for each trunk backend choice.
3. **Limited dynamic mixer (head‑scoped; learned→fixed):** enable dual‑exec on ≤10% heads in ≤2 trunk layers with a near‑binary mixing coefficient (\lambda). Enforce **Δ step‑time ≤5%** and **compile warm‑up** for both paths; once (\lambda) saturates, **freeze** to the winning backend.

**Carry‑forward policy.** Use Track A to shortlist backends; confirm in Track B under MoirAI’s compile constraints. Adopt **fixed** (or **learned→fixed**) backends only; **per‑token backend routing is out‑of‑scope** to preserve compile invariants and stable active‑parameter accounting (see §2.3). Results may promote the learned→fixed mixer to default for the shortlisted layers if cost stays within budget.

#### 7.8.1 Attention‑only ablation — protocol & parity requirements

To ensure fair attribution when comparing backends with the attention‑only baseline:

* **Depth parity:** **24 layers** to match MoirAI‑0.9B‑A0.3B trunk.
* **Compute parity:** equalize **FLOPs and KV memory** per step across SDPA (windowed/dilated), DSA, and Power settings.
* **Data parity:** identical data order, seeds, and token budgets; three seeds minimum.
* **Mechanism set:** begin with the **simplified** backend set; expand only if stable.
* **Learned→fixed trial:** a per‑layer selector may be trained but must **anneal to one** backend per layer prior to final evaluation.
* **Outcome gating:** MoirAI adopts only those backend choices that win under parity on length‑generalization and stability at ≤ baseline step‑time.

### 7.9 Fourier Positional Embeddings (FPE) — optional experiment

**Motivation.** Test whether **Fourier positional embeddings** improve stability or cross‑domain transfer for H‑Net internal encoders or trunk ablations.

**Scopes.**

* **H‑Net internal encoders:** apply FPE channels in the small 1D encoders per level; keep reverse‑lattice semantics unchanged.
* **Trunk ablation branch:** swap RoPE/NoPE in a small layer subset with fixed FPE channels.

**Constraints.** Static shapes; compile‑safe toggles; FPE channels sized from a predeclared set.
**Ablation.** Compare against the hybrid RoPE/NoPE baseline at equal compute; carry forward only if length‑gen or stability improves without latency regression.

---

## 8) Efficiency, Compile Invariants, and Cached Retrieval

MoirAI is designed for `torch.compile(dynamic=False)`. This requires strict adherence to static shapes and control flow. This section details the invariants we enforce and the optimizations we use.

### 8.1 Static-Shape Invariants

*   **Static Top-k=1 Everywhere:** All routers (HRM cluster, HRM-L/M experts, FFN family/cluster/expert) are hard-coded to `top-k=1` dispatch. There is no dynamic K at runtime.
*   **Tensorized Dispatch:** Routing is implemented with tensor gathers/scatters into pre-allocated buffers. There is no Python-side per-sample branching or shape-changing logic in the compiled forward path.
*   **Cluster Count Stability:** HRM always selects exactly one cluster per outer step. The FFN bank's 1→2 cluster schedule is a training-time transition that permits a re-compile at that specific milestone boundary.
*   **Shared Fixed Paths:** The shared `FixedL_shared` / `FixedM_shared` paths are singletons with per-cluster scalar gates. Enabling, annealing, or removing them never changes tensor shapes.

### 8.2 Re-compile Rules and Warm-up

Re-compilation is expensive and avoided. It is permitted **only** at specific, pre-defined training milestones:
1.  When flipping the FFN bank from a 1-cluster-per-family to a 2-cluster-per-family schedule.
2.  When first enabling long-context features (ESE, Power Attention, or DSA in the trunk) during the M5-LC milestone.

We use a **compile warm-up driver** to pre-trace all likely graph variants before training starts, exercising different THPL headers, DSA K-buckets, and verify path hooks to prevent runtime compilation lag.

### 8.3 Expert value cache (FFN retrieval cache) — policy

Each FFN cluster maintains a low‑rank surrogate ((A_c,B_c)) **tied to the repurposed fixed expert**. Serve the surrogate when queries are near the cluster prototype; **always attempt cache** before evaluating a second expert in the verify path.

* **Error budget:** relative L2 ≤ **1%** when hit‑rate ≥70% on flagged queries; auto‑relax to **≤2%** if hit‑rate drops ≥10% vs baseline.
* **Drift handling:** refresh ((A_c,B_c)) when the repurposed fixed expert shifts beyond threshold.
* **Static shapes:** cache lookups and updates are tensorized and shape‑stable.

* **Logging.** Track hit‑rate, error, and drift; auto‑disable on persistent budget violations.

### 8.4 Training Invariants & Late-Swap Shielding

*   **Enable-Early Rule:** Features that fundamentally change the model's architecture, like ESE, Power Attention, or Hybrid Positional Encoding, must be enabled at their designated milestone (M5-LC) and then frozen.
*   **Late-Swap Shielding:** If a backend must be changed mid-project (strongly discouraged), we use a one-epoch distillation shield. The new backend is trained to mimic the attention maps and outputs of the old one, minimizing distribution shift before unfreezing the full model.

### 8.5 Scale-registry invariants

* **Static shapes:** S-REG adds only scalar parameters; no dynamic control flow and no tensor shape changes.
* **Dispatch invariants:** Does **not** affect router top-k=1 policies, capacities, or K-buckets for DSA; compile warm-ups remain valid.
* **Residual locality:** Only residual **contributions** are scaled. Core state updates (GRUs) and logits heads remain functionally identical aside from FiLM magnitude scaling (see §5.3 (REPLACED)).
* **DropPath locality:** S-REG scales the residual **including** DropPath so stochastic depth behavior is preserved.

---

## 9) Task Header Block (THB) & Task Header Policy Layer (THPL)

Every sample is prefixed with a **fixed 64-byte Task Header Block (THB)** that encodes the authoritative runtime policy. The **Task Header Policy Layer (THPL)** is the sole component responsible for building, parsing, and enforcing this policy. All other modules (routers, halters, etc.) read the decoded `Policy` object from THPL.

### 9.1 Header Format (fixed 64 bytes + CRC16)

The Task Header Block (THB) is a compact, **64‑byte** structure that encodes the authoritative runtime policy for each sample. H‑Net must not chunk across the header.

```text
Bytes  Field                 Type        Notes
0–1    version               uint16
2      task_id               uint8       nl_chat, code_chat, sudoku_9x9, arc, …
3      domain_id             uint8       math, logic, general, legal, …
4      mode_flags            bitfield    verify_only, allow_abstain, creative, force_dense_attn
5      halt_kind             uint8       0 = cosine_mlp, 1 = bce_no_extra_pass
6      answer_head           uint8       0 = none, 1 = puzzle_fixed
7      bptt_flag             uint8       0 = off, 1 = on (request truncated BPTT sprinkles)
8–11   answer_len            uint32      e.g., 81 for 9×9 Sudoku; 0 if unused
12–15  ans_vocab_bits        uint32      log2 of vocab for puzzle head (e.g., 4 for ARC colors)
16–17  outer_steps_max       uint16      cap on outer loops
18–19  l_iters_max           uint16      cap for HRM‑L inner loop
20–21  m_iters_max           uint16      cap for HRM‑M inner loop
22     seq_mlp_allowed       uint8       0/1 to enable the sequence‑MLP expert
23     payload_codec         enum (u8)   UTF8_NFC, PPM_P6, WAV_PCM16LE, …
24     num_std               uint8       1 if numeral standardization applied
25     media_type            enum (u8)   0 = none, 1 = IMAGE, 2 = AUDIO (ESE/media flows)
26–31  media_meta            6 bytes     packed (e.g., {w,h} or {sr,mono}); fixed offsets/bit‑packing
32     halt_offset_q2        int8        inference‑time halter logit offset ×2 (step = 0.5); clamp to [+15] ≡ +7.5
33     ffn_verify_k          uint8       allowed FFN top‑k used **inside verify only** (1 or 2; others ignored)
34–61  reserved              bytes       future compact fields (e.g., language id, grid dims, safety tier)
62–63  CRC16(0..61)          uint16      integrity check; CRC fail → conservative fallback
```

**Encoding notes**

* `halt_offset_q2` is a signed 8‑bit fixed‑point value with **0.5** steps (e.g., `+10` ≡ `+5.0` logit offset). At runtime this is mapped to `policy.halt_offset = clamp(halt_offset_q2 / 2.0, 0.0, +7.5)` for compute‑sensitive tasks; the default is `0.0`.
* `ffn_verify_k` applies **only** within the verify path; it leaves the main forward path’s static **top‑1** invariant untouched. Values other than `{1,2}` are ignored and treated as `1`.

**Guards**

* On CRC failure or malformed fields, THPL emits a conservative policy: dense/windowed attention on short puzzles, `halt_offset = 0.0`, and `ffn_verify_k = 1`.
* The header is per‑sample; mixed batches are supported. Shapes remain static.

### 9.2 THPL Runtime Policy & Presets

THPL parses the 64‑byte header into a `Policy` object used by halters, routers, verify logic, and long‑context mechanisms.

**Field mapping**

* `policy.mode` from `mode_flags` (verify_only, allow_abstain, creative, force_dense_attn).
* `policy.halt_kind` from `halt_kind`.
* `policy.answer_head`, `policy.answer_len`, `policy.ans_vocab_bits`.
* `policy.outer_max`, `policy.l_max`, `policy.m_max` from their respective caps.
* `policy.seq_mlp_allowed` (0/1).
* `policy.payload_codec`, `policy.num_std`, `policy.media_type`, `policy.media_meta`.
* `policy.bptt_enabled = (bptt_flag == 1) && cfg.allow_bptt && (task_id ∈ cfg.bptt_tasks)`.
* **Inference bias:** `policy.halt_offset = clamp(halt_offset_q2 / 2.0, 0.0, +7.5)`.
* **Verify FFN fanout:** `policy.ffn_verify_k = (ffn_verify_k in {1,2}) ? ffn_verify_k : 1`.

**Presets**

* **nl_chat**: moderate outer cap, `halt_kind=cosine_mlp`, `verify_only` true, `halt_offset=0.0`, `ffn_verify_k=1`.
* **puzzle_fixed** (Sudoku/ARC): `halt_kind=bce_no_extra_pass`, allow `halt_offset ∈ [0,+7.5]` for calibration; `ffn_verify_k=2` inside verify for low‑confidence spans; `force_dense_attn` set for short contexts.
* **long_doc**: `payload_codec=UTF8_NFC`, ESE enabled by profile, verify path permitted with attention bump, `ffn_verify_k=1`.

**Invariants**

* Policy can **restrict** but not expand global safety limits (e.g., cannot exceed compile‑warmed DSA K buckets or outer caps).
* All policy effects are shape‑stable: top‑k dispatch remains **1** in the main path; verify internals are pre‑allocated.

### 9.3 Header-gated Mechanisms

#### 9.3.1 Puzzle Answer Head with Answer Memory

When `policy.answer_head == "puzzle_fixed"`, a specialized head is used instead of byte logits. It maintains an **answer memory** `y_t`, which is refined each outer step and receives deep supervision. Its halting is controlled by a BCE loss (`halt_kind="bce_no_extra_pass"`) that predicts if the answer is currently correct.

#### 9.3.2 Header-gated Seq-MLP Expert

A sequence-mixing MLP expert is included in the HRM-L expert bank but is only **eligible** for routing when `policy.seq_mlp_allowed == true`. This provides specialized capacity for grid-like tasks (Sudoku, ARC) without affecting general-purpose ones.

### 9.4 HRM control: EMA, loop‑cap auto‑tuning, and inference bias [^28]

**EMA.** Evaluate HRM L/M/G (and their routers/adapters) with EMA weights; trunk attention and FFN bank use raw weights.

**Auto‑tuned caps.** Raise a cap by +1 at the next epoch boundary when
(a) >20% of samples hit the cap for 5k steps **or** (b) **mean usage ≥75%** of the cap over 20k steps.
Lower by −1 if <1% hit the cap for 20k steps **and** mean usage <50%. Never change caps mid‑epoch.

**Halting bias.** THPL may set **halt_offset** (default 0.0; clip to +7.5) to adjust continue logits **at inference**. Effects are profiled per task.

**Tests.** EMA swaps do not regress unexpectedly; cap adjustments do not oscillate; halt_offset changes precision/recall of halting as intended on target tasks.

### 9.5 Canonical Byte Policy (CBP)

THPL enforces a consistent byte policy for H-Net:

- Text: UTF-8 NFC with normalized newlines and numeral standardization (NUM-STD). Header reflects this via payload_codec=UTF8_NFC and num_std=1.
- Media: Prefer ESE (§10.1.1). For samples that include media, THPL sets media_type=IMAGE|AUDIO and packs coarse meta into media_meta (e.g., {w,h} or {sr,mono}). Raw media bytes are not passed to the trunk when ESE is active. If raw-codec debug is requested, payload_codec=PPM_P6 (image) or WAV_PCM16LE (audio) is set, and lengths are validated.

Mixed batches are fine; each sample carries its own header; H-Net respects header boundaries.

### 9.6 Compile warm‑up driver (“Path Exciter”)

Pre‑trace all static shapes and guarded paths before training to avoid first‑use stalls with `torch.compile(dynamic=false)`:

1. **HRM expert shapes (all clusters, both bands):** route once to every recorded ratio for the active variant (see §2.2 and §6.1.0).
2. **FFN bank:** for each **family→cluster**, route once to **each carved expert** and once to the **fixed** expert; include value‑cache path and repurposed‑fixed surrogate projections.
3. **DSA (K) buckets:** for every long‑context DSA layer, execute forwards that hit **all powers‑of‑two** (K) from (k_{\min}) up to **verify_bump_max_k**.
4. **Limited dynamic mixer (head‑scoped):** for each enabled layer, warm up **both** backends (DSA and Power) on the **dual‑exec heads** and the **post‑anneal single‑backend** path.
5. **THPL presets & verify path:** one micro‑batch per header preset (nl_chat, code_chat, sudoku_9x9, arc_30x30, bptt_demo) plus a forced **verify** case (span bunching, prototype two‑shot, **top‑2 FFN in verify**).
6. **Micro‑batching:** keep tiny; routing **top‑1**; shapes static; buffers pre‑allocated for all verified paths.

**Coverage requirement:** report ≥95% of intended paths exercised (including dual‑exec heads and all DSA buckets) before main training starts.

### 9.7 Integration and Tests

*   **H-Net:** Is forbidden from chunking across the THB boundary.
*   **Routers:** Use policy bits to create bias vectors for HRM cluster selection and FFN MoR mixing.
*   **Halting/Answer Heads:** Configured directly from `Policy` fields.
*   **Tests:**
    *   **Round-trip:** Header↔Policy conversions must be exact, including CRC.
    *   **Router Effect:** Toggling policy bits must measurably alter router priors.
    *   **CRC Failure:** A bad CRC must trigger the conservative fallback policy.
    *   **Idempotence:** The Canonical Byte Policy must be idempotent.
    *   **Warm-up Coverage:** Assert that ≥95% of intended paths were exercised by the warm-up driver.

---

## 10) Training Plan: Milestones, Branches, and Invariants

We stage capability so stability comes first, then reasoning depth, then donor knowledge, then long-context scale-out. Each milestone has exit criteria and CI tests.

### M0 — Infra bring-up (≈1 week)

Goal: ensure the scaffolding compiles and routes.

*   **Actions:** ROCm / PyTorch ≥2.3; `torch.compile` harness; ACT halter stub; HRM loop skeletons; reverse lattice; static top-k=1 routing API; THPL header builder.
*   **Exit Criteria:** Reverse-lattice round-trip is exact; ACT extremes and router calls don't break compiled graphs; THPL header round-trip is exact.

### M1 — H-Net Chunker bootstrapping (≈2 weeks)

Goal: learn hierarchical chunking and validate early training health.

*   **Actions:** Train copy-task autoencoder on raw bytes; initial targets v0=64, v1=1024, v2=16384 bits; enable ratio/entropy losses and chunk-aware attention.
*   **Exit Criteria:** Copy exact-match ≥99.9%; bits/chunk means within ±15% of targets; "hockey-stick" pattern detected in early selection rates.

### M2 — HRM-L inner loop + Halting (≈3 weeks)

Goal: get iterative low-level reasoning with controlled stopping.

*   **Actions:** Add HRM-L (k≤4); add 2-layer MLP halter with ACT (λₒ→0.01); train on Sudoku/Maze from bytes.
*   **Exit Criteria:** Sudoku-easy ≥95%; median outer steps ≤1.6.

### M3 — Full HRM L/M/G + FiLM + Attention Controls (≈4 weeks)

Goal: enable hierarchical multi-timescale reasoning and stabilize attention.

- Actions:
  - Add HRM-M/G and FiLM broadcast; anneal H-Net targets to 32/512/8192 bits.
  - Enable GA, SSA, and AFA/blend on HRM-L; connect AFA innovation hooks to halter, FiLM, and router bias.
  - Add “Dump Head” for introspection.
  - Add Klotski (sliding-block) as a dedicated HRM reasoning task:
    - Encoding: grid serialized to bytes; each cell as 4 bits; row/sample separators (same scheme as ARC).
    - Curriculum: start shallow (≤20 moves), then increase depth; mix with existing puzzle suite to avoid overfit.
    - Objective: supervised next-move targets from BFS/IDA* solutions; optional imitation on full move sequences; optional value head for remaining distance (auxiliary).
    - Integration point: late M3 (or early M4) as part of the puzzles bucket.
    - Exit metric: success@k (k move budget), average plan length within +10% of optimal on dev.
    - Rationale: stresses outer-step halting and G→L FiLM control; encourages coherent long-horizon updates.

- Exit Criteria:
  - ARC dev ≥80%; ablating FiLM hurts ≥15% relative.
  - Attention sink reduced (GA on HRM-L; SSA improves per-query control).
  - Klotski success trending up (per above metrics).

### M4 — HRM MoE (heterogeneous experts) (≈4 weeks)

Goal: add specialized reasoning experts and smarter routing.

*   **Actions:** Enable per-cluster HRM expert banks with heterogeneous sizes and size curricula; enable shared fixed HRM paths and anneal their gates; enable tier-level EC fallback; train selective confidence and entity-risk heads in `report_only` mode.
*   **Exit Criteria:** No regression vs M3; router entropy ≥0.6; fixed-path share <20% and can be safely removed or repurposed; confidence/risk heads show sane calibration/AUC.

### M5 — FFN Transplant + Healing LM (≈6 weeks)

Goal: splice in donor knowledge without catastrophic forgetting and enable verification.

*   **Actions:** Transplant Qwen FFN experts using the full pipeline (carve, gate-compensate, partial re-init); follow A/B/C phases for adapter alignment, main LM training, and fixed expert annealing/repurposing; enable selective confidence in `verify_only` mode with the full verify path (extra step, rare top-2 FFN, etc.).
*   **Exit Criteria:** PPL ≤ same-size dense baseline; ARC/Sudoku drop ≤3% abs; verify path overhead within budget.

### M5-LC — Branch Point for Long-Context

This milestone runs concurrently with or immediately after M5, introducing foundational long-context capabilities. This is an “enable-early” milestone: features enabled here are considered core to the architecture going forward. We run explicit ablations and carry forward the winners.

- Actions:
  1. Branch A — Trunk attention backend:
     - Compare trunk backends at equalized compute/memory: periodic full SDPA (baseline) vs. Power Attention vs. DSA.
     - Use the per-layer registry (frozen post-milestone) with compile warm-up for all DSA K buckets needed for verify-bump (§10.1.4).
  2. Branch B — ESE compression ON vs OFF:
     - Run ESE in shadow mode (p(ESE)=0.3 ramping to 1.0) vs ESE OFF on the same data slices.
     - Compare long-context quality vs latency/VRAM; if ESE OFF wins on net utility, leave ESE disabled and keep the raw-byte path; else carry ESE forward and freeze its adapters for this run.
  3. Positional encoding:
     - Enable the hybrid RoPE/NoPE + QK-Norm trunk scheme.
  4. Re-compile & Freeze:
     - After this milestone, freeze the attention registry and positional scheme (immutable going forward for this run).

- Carry-forward Policy:
  - Carry forward one trunk backend (winner of Branch A).
  - Carry forward ESE ON or OFF (winner of Branch B).
  - Archive both checkpoints for reproducibility; any later backend change requires §8.4 late-swap shielding.

- Exit Criteria:
  - ESE ON vs OFF: quality drop ≤0.5% abs with ≥30% latency/VRAM win if ON; else leave OFF.
  - Backend: chosen trunk backend outperforms baseline on long-context retrieval at matched or better latency/VRAM.
  - Compile stability: registry frozen; warm-up driver covers all intended K buckets; no graph breaks.

### M6 — Continual Learning (≈4 weeks)

Goal: add new domain knowledge without forgetting prior skills.
*   **Actions:** Introduce new domains; add experts to under-served FFN clusters to support them; freeze experts with low routing entropy (<0.4); maintain a 5% replay of core puzzle/reasoning tasks.
*   **Exit Criteria:** Forgetting on old domains <5% absolute after 3 new domains are introduced; new domains show ≥5% relative gain in 100k tokens.

### M7 — Stretch: Hybrid Family & Scale-Out

Goal: expand the knowledge bank with a new donor family and scale to multiple nodes.

*   **Actions:** Add a **Mamba family** as a second FFN bank with its own adapters and fixed expert; map families/clusters across GPU nodes for hierarchical all-to-all communication.
*   **Exit Criteria:** 2-node training speedup ≥1.6×; distributed compile is stable; cross-family checks in the verify path are functional.

### Training Invariants and Late-Swap Shielding

*   **Enable-Early Rule:** ESE, Power Attention, DSA, and Hybrid Positional Encoding must be enabled at M5-LC if they are to be used at all. The per-layer attention registry is considered immutable after this point.
*   **Late-Swap Shielding:** If a backend must be changed after M5-LC (strongly discouraged), a **one-epoch distillation shield** is required. The model is frozen except for the new backend, which is trained to mimic the attention maps and outputs of the old one (`Loss = λ_attn * ||A_new - A_old||² + λ_out * ||Y_new - Y_old||² + 0.5 * main_loss`) before unfreezing the full model.

## 10.1 Long-context Capacity & Compression

This section covers how we scale to long contexts without blowing up cost. It unifies:

* ESE (External Segment Encoder) for compression of long text segments,
* Power / DSA / linear attention backends in the trunk and HRM bands,
* the DSA length-scaled top-K rule (with compile warmup),
* dense-attention override for “short puzzle” tasks,
* verify_bump behavior.

All mechanisms in this section are shape-stable and must be compile-friendly.

#### 10.1.1 External Segment Encoder (ESE) — enablement & verify fallbacks [^12]

ESE compresses long documents by segment→latent mapping with a small external encoder.

* **Enablement.** For long‑doc presets, **ESE is ON by default** at M5‑LC and frozen thereafter if it meets latency/quality targets.
* **Training.** Distill from raw‑byte runs (KL on byte logits, pooled (hG), and QA coverage) with a rate–distortion term that targets (\bar{t}\approx 2) latents/segment.
* **Verify fallbacks.** On low confidence, either request a **residual latent** (rank‑8) or re‑encode the span via H‑Net for that window. Bound to ≤5% of segments; preallocate buffers.
* **Cache.** Latents are cached by ((\text{doc_id}, \text{seg_idx}, \text{ESE_ckpt_hash}, \text{adapter_hash})).

### 10.1.2 Power / Linear / Dense attention registry (Trunk & HRM-L)

We maintain a **per-layer attention backend registry**, frozen at init (or when we enable long-context for the first time at M5-LC). Each layer in the trunk, and optionally HRM-L, is tagged with one backend:

* `dotprod` = standard scaled dot-product softmax SDPA (dense),
* `power`   = Power Attention / kernelized linear attention with fixed state size `m` per layer (typical `m=128` shallow, `m=256` deep),
* `linear`  = other kernelized linear variants (e.g. ELU+1/FAVOR),
* `dsa`     = Dynamic Sparse Attention (top-K selection of keys/values before SDPA),
* `sigmoid` (HRM-L only) = elementwise-sigmoid attention without normalization,
* plus head-gated output (post-mix scalar gates per head) and optional SSA temperature scaling / value scaling as described.

Rules:

* A single layer never mixes two different backends for the *same head group*, except that the trunk **can** split heads between `power` and `dotprod` at init time (“hybrid heads”), but the split ratio is fixed and compile-warmed.
* HRM-M and HRM-G generally keep their simpler windowed/dotprod mix unless/until we explicitly enable other backends later.
* The registry is immutable after we finish the first long-context compile warmup. If we *must* swap a layer’s backend later, we fall back to the “late-swap shielding” procedure in §17.

#### 10.1.2.1 Attention Registry Initialization and Freezing

Init (first long-context compile):

Assign each trunk layer one backend: dotprod, power, linear, or dsa.

Optionally define hybrid heads (fixed split between power and dotprod).

Pre-warm all DSA K values needed for verify-bump: powers of two from k_min up to verify_bump_max_k (e.g., 256, 512, 1024, 2048, 4096).

Immutability: after this compile, the registry is immutable. If a backend must change later, use the late-swap shielding procedure in §8.4 and §10 (Training Invariants and Late-Swap Shielding).

SSA/Sigmoid guard: SSA is disabled on any band/layer that uses sigmoid attention (enforced by config guards in §7.5 and §16).

#### 10.1.2.2 Power Attention State-Size Buckets & Kernel Fallback [^19]

Power Attention maintains a fixed state size m per layer. To keep shapes static and compile-safe across different sequence lengths, we select m from a small, predeclared set of “length buckets” at launch time.

- Buckets by max input length
  - Define a per-run bucket table that scales m by a fixed factor when the declared maximum context length increases.
  - Example: shallow trunk layers use m=128 at ≤64k tokens and m=160 (1.25×) at ≤128k; deep layers use m=256 at ≤64k and m=320 (1.25×) at ≤128k.
  - The bucket decision is made once at run start (or the first long-context compile checkpoint) based on the configured max_seq. It is not changed mid-run.

- Registry immutability
  - Once the bucket is selected, the attention registry (backend per layer and each layer’s m) is frozen for the run.
  - If you must change buckets later (strongly discouraged), use §8.4 late-swap shielding.

- Kernel fallback
  - If an optimized kernel is unavailable at the selected m, fall back to Triton/PyTorch with a warning; never refuse to run.
  - This fallback does not change shapes (same m), preserving compile safety.

- Config (concise)
```yaml
long_context:
  attention_registry:
    trunk:
      pattern:
        - {layers: [12,16,20,24], impl: "power", m: 128}    # example base m
        - {layers: [28],          impl: "power", m: 256}
    buckets:
      - {max_seq: 65536,  trunk_m_scale: 1.00}   # ≤64k
      - {max_seq: 131072, trunk_m_scale: 1.25}   # ≤128k
    immutable_after_compile: true
kernels:
  prefer_custom: true          # warning-only fallback to Triton/PyTorch
```

- Guards & tests
  - Assert the bucket selection occurs before compile warm-up and registry freeze.
  - If max_seq changes, require an explicit re-run with a new bucket; otherwise warn and keep the existing registry.
  - Track a warning if an optimized kernel is missing; verify fallback has zero impact on shapes and compile graphs.

#### 10.1.2.3 Periodic full SDPA (baseline preset)

For the “dense baseline” branch at M5‑LC, we use periodic full SDPA in the trunk (every **4th** layer), optional grouped‑query attention (GQA), and the remaining layers as sliding‑window/local variants.

```yaml
long_context:
  attention_registry:
    trunk:
      periodic_full_sdpa:
        enable: true
        period: 4         # every 4th layer dense SDPA
        gqa: true         # optional grouped queries/keys; fixed at init
```

> This preset is mutually exclusive with a per‑layer custom registry. Use it only for the “baseline” A/B branch at M5‑LC, then freeze the chosen winner.

### 10.1.3 DSA (Dynamic Sparse Attention) with length-scaled K

Goal: replace dense SDPA by selecting only the top-K keys/values per query—where K scales with the visible length—and snap K to a power of two for compile safety.

Length → K rule (runtime):

N = number of visible keys.
K_raw = round(ratio · N).
Clamp K_clamped = min(max(K_raw, k_min), k_max).
Snap to nearest power-of-two within [k_min, k_max] or [k_min, verify_bump_max_k] when verify-bump applies → K_snap.

Dense short-circuit: if N ≤ k_min, skip DSA and run dense SDPA.

Indexing + SDPA:

A tiny per-head indexer scores keys (low-rank q⊗k or 1-hidden-layer MLP).
Gather top‑K_snap keys/values into a preallocated buffer sized to that layer’s k_max (or never exceeding `verify_bump_max_k` when verify‑bump applies).
Run flash SDPA on the reduced set; apply GA/SSA, etc.

Compile/warmup:

torch.compile warm-up must pre-trace all power-of-two K in [k_min, verify_bump_max_k], to support verify-bump (§10.1.4).
Puzzle/short-context override:

If THPL marks force_dense_attn=true (see §9.1–§9.2), behave as if K_snap ≥ N (i.e., use dense SDPA/windowed attention). This override is shape-stable: the top-K buffer is still allocated; we just fill it with all keys.

Training schedule:

Warm-up: run dense SDPA and train the indexer with a KL to dense logits.

Sparse: switch to DSA; keep a small KL to maintain calibration.
See §10.1.4 for verify-bump details.

### 10.1.4 Verify-bump ceiling (global reach escalation during verify)

When selective confidence (§14) flags low confidence, we allow one extra outer reasoning step plus a temporary escalation of attention reach in designated global-mixing layers.

- We request the next power-of-two K above the normal K_snap, up to a hard ceiling called verify_bump_max_k (default 4096).
- Because verify-bump can request K values up to verify_bump_max_k, torch.compile warm-up MUST pre-trace all powers-of-two K in [k_min, verify_bump_max_k] rather than [k_min, k_max], e.g., {256, 512, 1024, 2048, 4096} for long-context DSA layers.
- verify-bump is used only in the verify path and only for flagged spans. Latency p95 on flagged spans must remain ≤15%.

#### 10.1.5 Hybrid positional encoding for trunk (RoPE / NoPE / QK‑Norm)

Two **mutually exclusive** long‑context positional presets for the trunk:

1. **Hybrid pattern** — a repeating *3‑of‑4 RoPE + 1‑of‑4 NoPE* scheme across depth, with QK‑Norm on all trunk attention layers.
2. **Rotary fraction** — **RoPE on the first 25% of q–k head dims** (`rotary_fraction=0.25`), with the rest unpositioned (NoPE) but QK‑Normed.

Both are compile‑frozen at the first long‑context compile (M5‑LC). Promote `rotary_fraction=0.25` to default only if it **matches or exceeds** the hybrid pattern on length‑gen at equal compute (checkpointed at M5‑LC). θ‑scaling remains optional.

##### 10.1.5.1 Rotary‑fraction alternative (0.25) — ablation & config

As an alternative to the 3:1 **RoPE/NoPE** hybrid across trunk depth, test **per‑layer partial rotary** that applies RoPE to only the **first 25% of QK head dimensions** while leaving the remaining 75% NoPE/QK‑Normed.

**Rules.**

* **Mutual exclusivity:** do **not** combine with the 3:1 RoPE/NoPE pattern in the same run.
* **Placement:** same trunk layers as the hybrid scheme; only the per‑head rotary fraction changes.
* **Freeze timing:** choose at **M5‑LC** and freeze with the attention registry.
* **Acceptance:** promote to default only if it **matches or exceeds** the hybrid scheme on ≥16× length generalization **at equal compute** and shows no stability regressions.

**Minimal config knob.**

```yaml
long_context:
  hybrid_positional:
    mode: "rope_fraction"    # alternatives: "rope_nope_pattern"
    rope_fraction_qk: 0.25    # RoPE on first 25% QK dims; remainder NoPE with QK-Norm
    qk_norm: {enable: true, eps: 1e-6}
```

### 10.1.6 Config knobs (long-context block)

```yaml
long_context:
  enable: false                 # flipped ON at M5-LC with a re-compile
  ese:
    segment_bytes: 2048
    max_latents_per_segment: 4
    residual:
      enable: true
      rank: 8
    adapter_ln_affine: true
    rate_lambda:
      start: 1.0
      target_latents: 2.0
    cache:
      enable: true
      cache_key: ["doc_id","seg_idx","ESE_ckpt_hash","adapter_hash"]

  attention_registry:
    trunk:
      pattern:
        - {layers: [12,16,20,24], impl: "dsa"}          # or "power"
        - {layers: [4,8],         impl: "power", m: 128}
        - {layers: [28],          impl: "power", m: 256}
      hybrid_heads:
        enable: true
        power_head_fraction: 0.5  # fixed at init
    hrm_l:
      pattern:
        - {layers: "all", impl: "sigmoid"}              # or "afa","dotprod","dsa"
    immutable_after_compile: true

  dsa_defaults:
    ratio: 0.05
    k_min: 256
    k_max: 2048
    verify_bump_max_k: 4096  # global hard ceiling for verify bump
    warmup_all_powers_of_two: true

  puzzle_override:
    force_dense_attn_if_short: true   # if THPL flag force_dense_attn=true
    short_context_gate: true          # treat K_snap >= N, i.e. dense path

  hybrid_positional:
    rope_nope_pattern: {rope_layers: 3, nope_layers: 1}
    qk_norm: {enable: true, eps: 1e-6}
    theta_scale: {enable: false, factor: 1.0}
```

#### 10.1.7 Fourier positional embeddings (FPE) — optional experiments

**Scope.** FPE may be evaluated as an **optional** replacement for positional encoding in two scopes:

1. **Trunk ablation branch** (selected layers only, registry‑frozen per §10.1.2), or
2. **H‑Net internal encoders** (v0/v1/v2 1D blocks) where FPE replaces or augments the within‑level positional signal.

**Compile invariants.** FPE adds only deterministic, shape‑stable features; no dynamic control flow. Kernel availability is not required; fall back to standard ops if a custom kernel is absent.

**Adoption criteria.** Equalize compute vs the incumbent scheme and require **no worse than 0.5%** loss delta on long‑context tests with **no regressions** in H‑Net stability (target‑bits adherence, boundary entropy). Carry forward only if latency/VRAM is neutral or improved.

### 10.2 SFT with reasoning traces (if available) [^29]

When datasets provide **intermediate reasoning traces**, add a light auxiliary loss that supervises HRM global states (hG_t) (and, optionally, pooled L/M states) at each outer step:

* **Signal.** Match to provided intermediate reasoning embeddings via cosine/L2 (weight 0.05–0.10).
* **When.** Early SFT phases (M3/M4) and selected later milestones.
* **Safety.** The auxiliary is purely latent; no exposure of traces in generated text; no change to compile invariants.
* **Ablate.** Confirm improvements on multi‑step puzzles and long‑horizon NL tasks.

#### 10.2.1 Coupling attention sharpness to innovation — defaults & decay

Couple SSA temperature and GA bias to AFA innovation:

* **HRM‑L defaults:** (\kappa_{\tau}=\kappa_{\text{gate}}=0.35).
* **HRM‑M defaults:** (\kappa_{\text{gate}}=0.25). **(\kappa_{\tau}=0.25) applies only if SSA is enabled on HRM‑M; otherwise set (\kappa_{\tau}=0)** (GA‑only coupling).
* **Clamps/EMA:** use a low‑innovation EMA baseline; bound modulation by per‑band clamps.
* **Decay:** apply ≈20% cosine decay of (\kappa) by end of M4.

#### 10.2.2 Verify path QoL improvements

- Prototype two-shot: check runner-up cluster prototype similarity before a full second expert; try the FFN value cache first; only run the full MLP if needed.
- Span bunching: merge flagged tokens into one span (max_gap_tokens=8) to run a single verify pass.

Both are shape-stable and preallocated.

#### 10.2.3 Cross-feature / safety guards

- SSA vs Sigmoid: forbid SSA on layers/bands using sigmoid attention.
- force_dense_attn for short puzzles: if THPL marks force_dense_attn=true, override DSA/Power/hybrids to dense per-sample; no verify-bump.
- Selective-confidence modes: verify_only=true forbids abstention; allow_abstain=true permits abstention; creative=true biases halter/router toward generative modes.
- Memory/compile guards for long-context: never request a K that was not compile-warmed; if violated, fall back to dense/windowed and log.

Note: M6 and M7 are defined once in §10 (above); no additional milestone definitions appear here.

#### 10.2.4 Reasoning traces during SFT (optional) [^29]

When datasets provide latent reasoning signals, add a small auxiliary loss (0.05–0.10) that encourages (hG_t) (and optional pooled L/M states) to match provided intermediate embeddings at each outer step. This supervision is **latent‑only** and does not alter generated text.

---

## 11) Optimizer, precision & schedules

**AdamW:** β=(0.9,0.95), ε=1e‑8, weight decay 0.1, global grad clip 1.0.
**LR schedule:** cosine decay with 5% warm‑up; LR multipliers: ×0.5 for family‑shared adapters at start.

### 11.1 Precision & numerics [^32][^33]

* **Default:** **fp16** with **dynamic loss scaling** across the model.
* **Numerically sensitive:** reverse lattice (H‑Net) computes in **fp32**; QK‑Norm on trunk attention layers.
* **Heads & embeddings:** Byte logits are produced by **ByteHead** over HRM‑L; H‑Net replaces token embeddings.
* **Fallback:** If a module is unstable in fp16, use **bf16** with fp32 accumulators for that module only; retain QK‑Norm and logit hygiene.
* **Linear/Power/DSA:** use activation clamping (and QAT for FP8 experiments). Document any per‑module precision exceptions.

#### 11.1.1 Output head precision (ByteHead & puzzle answer head)

**ByteHead.** Compute the final projection and logit accumulation in **fp32**; upstream activations may remain fp16.
**Puzzle answer head.** When the fixed puzzle head is active, compute its projection/logits in **fp32**.
**Reverse lattice.** Keep in **fp32**.
These heads replace conventional “embedding/lm‑head” components (H‑Net supplies bytes; there is no separate token embedding).

#### 11.1.2 Precision mapping for MoirAI components

The precision policy maps cleanly onto MoirAI’s modules, which lack a traditional “embedding” and “lm‑head” due to H‑Net and the ByteHead:

* **Global default:** **fp16** with **dynamic loss scaling** and **QK‑Norm** on trunk attention. Backends using linear/Power/DSA apply explicit activation clamping; FP8 experiments require QAT and clamping (kept optional and isolated).
* **Byte output head:** **ByteHead** (final logits over bytes) runs in **fp32** to match “lm‑head” stability expectations from dense LMs, then casts to fp16 for loss as needed.
* **H‑Net stages:** the **reverse lattice** (byte reconstruction path) runs in **fp32**; the **boundary heads** may accumulate logits in fp32 with fp16 I/O to maintain numeric headroom around chunk boundaries.
* **Fallback scopes:** if a specific attention layer or H‑Net stage proves unstable under fp16 despite QK‑Norm/logit hygiene, switch **that module** to **bf16 with fp32 accumulators**; record the exception in run metadata.
* **Optimizers:** keep fp32 master weights/accumulators in AdamW for all modules; scale‑registry (S‑REG) parameters follow the module defaults but are cheap enough to compute in fp32 when co‑located with fp32 heads.

### 11.2 Mixed‑BPTT & convergence guards

**Across outer steps.** Training uses **truncated BPTT** across outer steps by default: gradients flow through the **final** outer step only.

**Across inner loops.** Within the final outer step, **full gradients** flow through the HRM‑L and HRM‑M inner loops.

**Prefix “sprinkle” passes.** To supply upstream credit without unbounded memory growth, periodically *sprinkle* non‑truncated passes:

* With small, configurable probabilities per batch, enable gradients through the **first** outer step or the **first + second** outer steps;
* Perform a backward pass, **clear gradients**, then run the final outer step and backward again;
* Require activation checkpointing on outer‑step boundaries and auto‑throttle sprinkle rates on OOM sentinels.
  All sprinkle counters and memory incidents are logged.

**Convergence regularizer.** When outer caps exceed 4, apply a tiny penalty to discourage too‑rapid shrinkage of (\lVert\Delta h\rVert) (or excessive GRU Jacobian contraction). This combines well with the cosine‑veto option to prevent shallow convergence.

**Caps & policy.** Train with **outer_max ≤ 16**; inference may allow higher caps when authorized by THPL policy, with budget logging (§5.4).

### 11.3 Group‑specific regularization & losses

* **GA gates:** L1 on gate activations (1e‑4), **gate_floor=0.02**, LR multiplier 0.5.
* **S‑REG:** soft barrier with (s_{\max}=2.0), LR multiplier 0.5, and **weight decay 1e‑4** on S‑REG parameters.
* **Adapters:** alignment loss on family adapters; orthogonality when repurposing fixed experts.
* **Routers:** Switch‑LBL α schedule 0.02→0.01→0.005; z‑loss=1e‑3; capacity factor 1.25.

### 11.4 ES fine‑tuning (optional, late‑phase) [^27]

Late‑phase **Evolution Strategies** fine‑tuning on small evaluation slices can optimize non‑convex knobs (gates, halter, S‑REG scales). Keep perturbations small, verify neutrality on held‑out sets, and roll back on regressions.

### 11.5 S‑REG priors & (s_{\max}) rationale

**Type priors (softplus‑param, multiplied with band and Layer‑ID scales):**

| Type         | Prior (init) | Rationale                                      |
| ------------ | :----------: | ---------------------------------------------- |
| attn         |     0.45     | moderate initial residual from trunk attention |
| ffn_retrieve |     0.35     | donor recall useful but not dominant early     |
| ffn_fixed    |     0.25     | conservative fixed FFN path                    |
| hrmL_expert  |     0.25     | keep L experts modest at boot                  |
| hrmM_expert  |     0.30     | slightly higher to stabilize mid‑band          |
| hrmL_fixed   |     0.35     | warm‑start stabilizer                          |
| hrmM_fixed   |     0.35     | warm‑start stabilizer                          |
| film         |     0.55     | FiLM must steer logits path reliably           |
| umoe_tap     |     0.25     | small taps until alignment settles             |

**Barrier.** A soft barrier enforces (s \le s_{\max}) with (s_{\max}=2.0) by default to prevent runaway scales. Weight decay **1e‑4** on S‑REG parameters adds gentle shrinkage. Attribution telemetry (share of residual magnitudes) is logged for all types and bands to support freezing/pruning decisions.

### 11.6 S‑REG training schedule and freeze

* **Initial freeze:** S‑REG parameters remain **frozen for the first 2% of tokens** to allow H‑Net and HRM anchors to settle.
* **Learning rate:** S‑REG optimizer group uses an **LR multiplier of 0.5** relative to the trunk default with **weight decay 1e‑4** and a soft barrier (s\le s_{\max}) (default (s_{\max}=2.0)).
* **Unfreeze & train:** after the freeze window, S‑REG trains continuously; attribution histograms are logged each epoch to support freezing/pruning (§6.5).
* **Debug guard:** a compile‑time flag can zero specific type scales for sanity checks; production runs keep all scales enabled.

---

## 12) Monitoring & Guardrails (always-on metrics)

### Chunker / H-Net

* Track mean bits/chunk and histograms for v0/v1/v2.
* Track the ratio loss and boundary entropy. Page an alert if v0/v1/v2 drift >15% from target means week-over-week or if boundary entropy collapses (spiky boundaries).
* Hockey-stick watchdog (§4.6): confirm early “drop → spike → settle” in selection rates; auto-remediation if missing.

### Halting / Outer Steps

* Histogram of outer steps per task (linked to THPL task type).
* Distribution of ‖ΔhG‖₂ across steps and cosine deltas.
* Δ-loss(n±1 outer steps) correlation for halter quality.

### Routers / MoE

- Router entropy for:
  - HRM cluster router,
  - HRM-L/M expert routers,
  - FFN family→cluster→expert routers.
- Overflow and dead-expert rates per tier; trigger Expert-Choice fallback if overflow>5% or dead>10% persists >1k steps in a tier.
- Per-size-tier utilization (HRM-L small/med/large; HRM-M med/large/XL) vs their targets and vs compute prior κ.
- MoR mixer α(hG′) distribution (prior vs query) for FFN families; when innovation-based routing bias is enabled, log how innovation (§7.2.1) shifts α toward the query head on surprising spans.

### Shared fixed paths (HRM & FFN)

* “Fixed share”: fraction of contribution coming from shared fixed HRM paths and from the donor-family fixed FFN expert.
* Anneal gates `w_fix{L,M}_c` toward 0 by end of M4 unless “repurpose” is triggered.
* Removal safety test: forcing fixed gates to 0 should reduce core metrics ≤0.5% absolute; else we switch that fixed path into “repurpose” mode (reverse-KL distillation / orthogonality adapters only).

### Adapters & Alignment

* `L_align` on each donor family’s shared `A_in`/`A_out` + donor LayerNorm stats.
* If `L_align` stalls >1 epoch, we unfreeze donor LN gains (not biases) according to plan.

### Selective Confidence / Abstention / Verify Path

* Calibration (ECE ≤5% for puzzle-style heads).
* Coverage control: actual coverage stays within ±2% of `target_coverage`.
* False abstain / false verify rates on easy splits.
* Latency impact from verify path:

  * overall average <5% overhead,
  * p95 increase ≤15% on flagged spans.
* Rare top-2 FFN usage:

  * trigger rate <3% tokens,
  * win-rate ≥60% when triggered.

### Entity-Risk & Hallucination Control

* Entity-risk AUC (goal ≥0.80 on held-out entity spans).
* Rate at which verify path triggers specifically because of entity-risk + off-manifold signals.
* Reduction in wrong named entities / citations after verify.

### Attention & Sink / Stability

* First-token mass per head/layer (attention sink). Expect large reductions once gated attention (§13.2) / sigmoid attention (§13.4.2) are active.
* 99.9th percentile activation magnitude per layer; expect smoother tails with gated attention / sigmoid attention.
* Attention entropy distribution per band with SSA (§13.3).

### Expert Value Cache (FFN retrieval cache)

* Cache hit-rate per FFN cluster (how often we skip a full carved expert forward).
* Approximation error of cache vs real expert output:

  * rel L2 error target <3%.
* Drift: monitor `‖q−μ_c‖` distribution; page if we see systematic off-manifold drift (queries far from any cluster prototype).
* Verify-path “runner-up cluster” cache usage:

  * success rate of cache vs full second expert,
  * latency savings.

### Throughput / Budget

* p95 latency, memory footprint, and active parameters/token vs budget targets for each milestone phase.
* verify_bump usage stats:

  * how often we escalate to larger K up to `verify_bump_max_k`,
  * confirm we stay under compile-warmed limits (see §13.4.5).

### S-REG Monitoring & Guardrails

**Scale statistics.**
Log per-epoch histograms for `s_type.*` and `s_layer.*`. Page if a median leaves `[0.05, 1.80]` for >2 epochs.

**Attribution shares.**
Track running contribution shares per module type:
`Share(type) = E[‖(s_type·s_layer·Δ_type)‖₂ / ‖Σ_types Δ‖₂]`.
Watch for sudden flips near M5 (transplant) and M5-LC (long-context).

**Transplant health.**
During M5, rising `s_type.ffn_retrieve` should correlate with decreasing adapter alignment loss (`L_align`); alert on sustained anti-correlation (>5k steps).

**Verify-path neutrality.**
S-REG must not change the rate of **verify_bump** requests unrelated to confidence/innovation triggers. Track verify usage stratified by `s_type.*` deciles; differences >10% absolute trigger investigation.

### 12.1 S‑REG attribution quotient monitor

Track a simple **attribution quotient** during fixed→routed handover within a time window:

[
\text{share}(\Delta_{\text{fixed}}) ;+; \text{share}(\Delta_{\text{routed}}) ;\approx; 1 ;(\pm \varepsilon).
]

Page an alert on persistent deviation; this indicates leakage or mis‑attribution during anneal.

---

## 13) Tests by milestone

- M0 (Infra):
  - Reverse-lattice round-trip exact (fuzz ≥1k).
  - ACT/router calls stable under torch.compile(dynamic=false).
  - THPL header round-trip exact (fields ↔ bytes ↔ CRC).

- M1 (Chunker):
  - Copy exact ≥99.9%.
  - Bits/chunk means within bands for v0/v1/v2.
  - Hockey-stick pattern detected: selection rate drop → spike → settle in first 2k steps; auto-remediation triggers if missing.

- M2 (HRM-L):
  - Sudoku-easy ≥95%.
  - Median outer steps ≤1.6.
  - Δ-loss(n±1) sign correct ≥70%.

- M3 (Full HRM):
  - ARC dev ≥80%.
  - FiLM ablation hurts ≥15% relative.
  - Level-2 freeze (during H-Net target switch) reduces prior variance vs. no-freeze baseline (explicitly assert lower prior variance).
  - Klotski success trending up (success@k; plan length within +10% of optimal on dev).
  - Attention sink reduced with GA on HRM-L (first-token mass drops ≥5× vs baseline).

- M4 (HRM MoE):
  - Tier utilization near targets (HRM-L: small/med ≈75%, large ≤10%; HRM-M: med/large dominate, very-large ≤10%).
  - Heterogeneous > homogeneous at equal FLOPs by ≥2% on ARC-hard; median outer steps ↓ by ≥0.2 vs homogeneous.
  - EC fallback (tier-level) resolves overflow<1% and dead<10% within 1k steps when triggered.
  - Weight-tying: FixedL_shared/FixedM_shared parameter hashes identical across clusters; only per-cluster scalar gates differ.
  - Gate anneal: mean(w_fixL_c), mean(w_fixM_c) ≤0.2 by M4 end unless repurposing is enabled.
  - Removal safety: toggling all fixed gates to 0 changes ARC/Sudoku/logic-grid metrics by ≤0.5% abs; else auto-switch to repurpose path (assert reverse-KL + orthogonality active next run).
  - Compile invariants: toggling fixed gates does not change shapes; compiled graph intact under torch.compile(dynamic=false).

- M5 (FFN transplant):
  - Gate-comp recon error ≤5% on calibration slice (expert outputs vs donor).
  - L_align < 0.1 for family-shared adapters.
  - MoR > prior-only and > query-only by ≥2% PPL relative (same compute).
  - Removing fixed FFN expert causes ≤0.5% abs drop; else engage repurpose (reverse-KL + orthogonality) and assert improvement.
  - Verify path (report-only → verify_only transition): latency p95 on flagged spans ≤15%; overall average <5%.

- M5-LC (Long-Context Branch):
  - ESE ON vs OFF: carry-forward decision per §10 (≤0.5% abs delta with ≥30% latency/VRAM win if ON).
  - Trunk backend Power vs DSA vs periodic SDPA: winner shows superior scaling at matched compute.
  - Compile: registry frozen; warm-up covers all DSA K={k_min, …, verify_bump_max_k}; zero graph breaks.

- M6 (Continual):
  - New domain + expert → ≥5% relative gain within 100k tokens.
  - Forgetting on prior domains ≤5% abs.
  - Router entropy remains ≥0.6.

- M7 (Hybrid & scale-out):
  - 2-node speedup ≥1.6×; DDP compile intact.
  - Cross-family check (if hybrid bank): verify path prototype checks work and do not regress compile stability.

### 13.1 S-REG Tests by milestone

* **M1 (Chunker)**

  * Enabling S-REG after 2% tokens maintains hockey-stick (§4.6).
  * PPL delta vs no-S-REG (shadow) ≤ **0.1%**.

* **M3 (Full HRM)**

  * With GA+SSA+S-REG, first-token mass drops ≥**5×** vs baseline.
  * Turning S-REG OFF (shadow) regresses ARC dev by ≥**2%** relative.

* **M4 (HRM MoE)**

  * HRM-L/M tier utilization stays near targets; no expert dead-zones emerge with S-REG ON.
  * Removal test for fixed paths (§6.1.4) unchanged by S-REG presence.

* **M5 (FFN Transplant)**

  * Temporarily zeroing `s_type.ffn_retrieve` for 1k steps increases PPL by ≥**0.5%** abs at equal compute (sanity).
  * `L_align < 0.1` holds while `s_type.ffn_retrieve` rises to ≥**0.6** median.

* **M5-LC (Long-context)**

  * DSA K-bucket selection **identical** with/without S-REG.
  * Length-gen test: ≤**0.5%** loss delta at ≥16× context vs baseline at matched compute.

---

## 14) Selective Confidence, Verification, and Introspection [^11]

MoirAI includes a system for calibrated confidence reporting, controlled abstention, and a low-latency verify path for low-confidence outputs. This is driven by the `Policy` from the THPL and a dedicated "selective abstain head."

### 14.1 Signals for Confidence Modeling

A small MLP head predicts sequence-level confidence based on a rich set of stop-gradient features:
*   **Prototype Distances:** Cosine distance from the HRM-G query to the nearest FFN cluster prototypes.
*   **Adapter OOD Score:** Mahalanobis distance of adapter inputs from their calibration distribution.
*   **Innovation Stats:** The `innov_mean` and `innov_pw` signals from AFA, indicating model surprise.
*   **MoR Mixing Entropy:** The entropy of the FFN cluster router's mixture-of-routers weights.
*   **Token Confidence:** Per-token log-probability, entropy, and margin from the output head.
*   **Entity-Risk Probe:** A token-level MLP that flags likely unsupported factual entities (names, dates, IDs), providing a hallucination-specific risk score.

**Practical thresholds.** Default token flag `≥0.6`, segment flag `≥0.5`; route high‑risk spans to the verify path when (entity‑risk high) ∧ (prototype distance high ∨ adapter‑OOD true).

### 14.2 Policy and Modes (THPL-driven)

The THPL `mode_flags` determine the behavior:
*   `mode="verify_only"`: The model can run an extra verification pass but will **never** replace its output with an abstention message. This is the default for creative tasks.
*   `mode="strict"`: The model may abstain and output "I'm not confident" if its score is below a threshold after verification. This is for safety-critical or puzzle-solving tasks.
*   `mode="report_only"`: The model only reports confidence scores without ever verifying or abstaining.

Training uses a coverage-controlled selective risk objective to calibrate the head.

#### 14.2.1 Entity‑risk routing policy (tightened; verify‑first)

Selective confidence escalates to verify when the **entity‑risk head** and off‑manifold signals agree:

**Condition (per span):**
[
(\textsf{entity_risk} \ge \tau_{\text{risk}})\ \wedge
\Big( \ \textsf{proto_dist} \ge \tau_{\text{proto}} \ \ \ \lor\ \ \ \textsf{adapter_OOD} \ge \tau_{\text{ood}} \ \Big)
]

* **Signals.**

  * `entity_risk`: token‑level risk from the entity‑risk head (names, dates, IDs).
  * `proto_dist`: cosine distance of (q=hG') to the nearest FFN‑cluster prototype (stop‑grad).
  * `adapter_OOD`: Mahalanobis distance of adapter inputs from calibration stats.
* **Default thresholds (per domain; THPL‑configurable).**

  * General text: (\tau_{\text{risk}}=0.60,\ \tau_{\text{proto}}=0.35,\ \tau_{\text{ood}}=0.85)
  * Code/doc QA (stricter abstain): (\tau_{\text{risk}}=0.55,\ \tau_{\text{proto}}=0.30,\ \tau_{\text{ood}}=0.80)
* **Action.** Route the span to the **verify path** (one extra outer step, attention **verify‑bump**, cache‑first, then rare top‑2 FFN if enabled). If `mode=verify_only`, do **not** abstain; otherwise, allow abstention only **after** verification if confidence remains below the abstain threshold.
* **Budget & guards.** Span bunching (max_gap_tokens=8) reduces overhead. Verify path budgets remain: average <5% and p95 ≤15% on flagged spans. Thresholds may be decayed slightly for high‑harm entity classes (e.g., medical/legal) via THPL presets while maintaining compile‑safe invariants.

### 14.3 The verify path & QoL improvements (shape‑stable parallelism)

When sequence‑level confidence is low, the verify path executes compile‑safe steps:

* **+1 outer step** (policy‑capped).
* **Attention bump:** escalate global reach in designated layers (e.g., DSA (K)) in power‑of‑two steps up to **verify_bump_max_k** (pre‑warmed buckets only).
* **Parallel top‑2 FFN (verify‑only):** inside verify, evaluate the runner‑up FFN expert **in parallel** with the top‑1 expert using **pre‑allocated k=2 buffers/kernels** owned by the verify module. This is shape‑stable and never changes the main forward’s top‑1 invariant. A prototype two‑shot cosine check can short‑circuit the full second expert; **always try the value cache first**.
* **Span bunching:** merge adjacent flagged tokens (max_gap_tokens = 8) to run a single verify pass on the span.

**Budgets & guards:** overall average overhead <5%; p95 ≤15% on flagged spans; rare top‑2 FFN trigger rate <3% of tokens with ≥60% win‑rate. If p95 exceeds 15% for two consecutive eval windows, auto‑disable span bunching and revert to a stricter cache‑first policy until tails recover.

#### 14.3.1 Rare top‑2 FFN (verify‑only) — **parallel compute** semantics

When the verify path is active and **policy.ffn_verify_k = 2**, the runner‑up FFN expert (**same family→cluster tier**) is evaluated **in parallel** with the top‑1 expert **inside the verify module only**.

**Semantics & invariants.**

* **Parallel evaluation:** both experts run concurrently using **pre‑allocated k=2 buffers**; the better output (lower loss proxy or higher agreement with cache/teacher) is selected by a fixed reducer.
* **Shape‑stable:** compile warm‑up pre‑traces the k=2 path; the main forward remains **top‑1** elsewhere.
* **Cache‑first:** if the value cache approximates the runner‑up within the current error budget, the cache result is used and the second MLP is skipped.
* **Budgets:** trigger rate <3% tokens; win‑rate ≥60%; p95 latency on flagged spans ≤15%.

**Config (concise).**

```yaml
verify_path:
  ffn:
    policy_ffn_verify_k: 2          # verify-only; main path remains k=1
    reducer: "min_loss_proxy"        # tie-break by cache agreement, then entropy
    preallocate_k2_buffers: true
    cache_first: true
```

### 14.4 Uncertainty-Coupled Attention Controls

To make the model focus more when it is uncertain, we modulate SSA temperature (`τ`) and Gated Attention strength (`g_h`) based on the innovation signal from AFA:
*   **High Innovation (Surprise):** We decrease `τ` (sharpening attention) and increase the pre-sigmoid bias for `g_h` (allowing heads to contribute more strongly).
*   **Low Innovation (Confidence):** We increase `τ` (flattening attention) and decrease the bias for `g_h` (discouraging over-focus).

This is implemented via a small, learned modulation based on the deviation from a running EMA of low innovation, keeping all shapes static.

### 14.5 The "Dump Head" for Introspection

For debugging, we attach a parallel **dump head**. This is a separate byte-distribution head that:
*   Produces a "first guess" set of logits from a pre-FiLM or post-FiLM state.
*   Is completely **detached from the loss** and has no gradient.
*   Is not fed back into any model component.

By comparing its output to the final logits, we can diagnose hallucinations and analyze the impact of FiLM and the verify path.

---

## 15) Implementation Details: Project Layout & Engineering Tasks

### 15.1 Project Layout

This is a preliminary layout, subject to refinement.

```
moirai/
  __init__.py

  hnet/                     # H-Net dynamic chunker
    __init__.py
    embed.py
    boundary_head.py
    soft_pool.py
    attention.py
    chunker.py              # Orchestrates L0/L1/L2, target-bits curriculum, reverse lattice

  hrm/                      # Hierarchical Recurrent Reasoning
    __init__.py
    gru_cells.py            # GRU_L / GRU_M / GRU_G (+ Jacobian clamp hook)
    lm_heads.py             # ByteHead; FiLM (per-channel or grouped); DumpHead
    halting.py              # MLP halter (ACT), BCE halter, cosine veto, one-step-grad switch
    loops.py                # Exact L/M/G step (residual expert wrapping; order of ops)
    micro_experts.py        # Optional L inner-loop MoE cell
    fixed_shared.py         # FixedL_shared / FixedM_shared (shared weights, per-cluster gates)

  moe/                      # Mixture-of-Experts infrastructure
    __init__.py
    routing.py              # Static top-k routers; MoR mixer; EC fallback; innovation bias
    hrm_experts.py          # L/M expert banks (hetero sizes); per-expert adapters
    ffn_family.py           # Family registry (Qwen, Mamba); shared A_in/A_out; donor LN handling
    ffn_clusters.py         # Calibration stats; clustering; expert carving; gate-compensation
    ffn_retrieve.py         # Family→cluster→expert retrieval; pooling; expert value cache

  attention_backends/       # Modular attention implementations (see §7)
    ...

  policy/                   # Task Header Policy Layer (see §9)
    thpl.py
    header_layout.py
    byte_policy.py

  selective_confidence/     # Confidence, abstention, and verification (see §14)
    abstain_head.py
    entity_risk_head.py
    verify_path.py

  transplant/               # Donor model transplantation toolkit
    __init__.py
    calib_dataset.py
    stats.py
    gate_comp.py
    fixed_builder.py
    layer_select.py

  train/
    __init__.py
    dataloader.py           # Byte I/O; ARC 4-bit pack/unpack; THB prepending
    curriculum.py           # H-Net targets & HRM size unlock schedules
    optimizer.py            # AdamW groups (adapters, routers, cores)
    trainer.py              # Phase runner; compile & re-compile at schedule switches
    metrics.py              # All monitoring metrics from §12
    warmup_driver.py        # The "Path Exciter" for pre-compiling graph variants
    checkpoints.py

  configs/
    moirai_q05b.yaml
    moirai_q15b.yaml
    thresholds.yaml

  tests/
    test_m0_infra.py
    test_m1_chunker.py
    # ... tests for each milestone ...

  cli/
    run_phase.py
    eval_bench.py
```

### 15.2 Immediate engineering tasks

1. **Router API.** Family→cluster→expert router with MoR; HRM cluster router with compute priors; tier‑level Expert‑Choice fallback.
2. **H‑Net.** Boundary heads, soft pooling, reverse lattice, target‑bits curriculum, ratio/entropy regularizers, block‑sparse masks.
3. **HRM loops.** Exact L/M/G step order; residual expert wrapping; FiLM broadcast (grouped option).
4. **HRM experts.** Heterogeneous sizes, per‑expert adapters, shared fixed HRM paths with per‑cluster gates, size curricula controllers.
5. **FFN transplant kit.** Calibration stats; gate‑compensation; fixed FFN constructor; family adapters; donor LN handling; alignment loss.
6. **Halter module.** ACT and BCE halters; cosine veto hook; convergence regularizer and one‑step gradient toggles (THPL‑gated).
7. **Expert value cache.** Per‑cluster low‑rank surrogate with EMA updates; tie to repurposed fixed expert; error/hit‑rate guards.
8. **Compile harness.** Enforce static top‑k=1 dispatch; manage re‑compiles at allowed phase boundaries (FFN 1→2 clusters; M5‑LC long‑context enable).
9. **Attention backends.** Per‑layer registry; SDPA/windowed/dilated, AFA, GA, SSA, Sigmoid, DSA, Power; SILM coupling to DSA; precision hygiene.
10. **THPL.** 64‑byte header builder, parser, presets, Canonical Byte Policy.
11. **Selective confidence.** Abstain head, entity‑risk probe, verify path (span bunching, prototype two‑shot, verify‑bump K escalation).
12. **Warm‑up driver.** Pre‑trace all expert shapes, DSA K buckets (up to verify_bump_max_k), THPL presets, and verify triggers; include **dual‑exec heads** for the limited mixer (§7.7).
13. **HP search & logging.** Integrate **Optuna/Ray Tune** for key knobs (SSA bounds, GA regularization, S‑REG (s_{\max}), loop caps); adopt **SwanLab** for lightweight metrics and artifacts.
14. **Project plumbing.** Use **uv** and **pyproject.toml** for reproducible builds, lockfiles, and environment bootstrap.

### 15.3 Hyperparameter search & experiment logging (Optuna + SwanLab)

**Objective.** Automate critical knob selection with **Optuna** (or Ray Tune backend) and ensure experiment hygiene with **SwanLab** logging. Reproducibility is enforced via **uv** and **pyproject.toml**.

**Search spaces (examples).**

* **SSA temperature bounds:** `min_tau ∈ [0.5,0.8]`, `max_tau ∈ [1.2,1.6]`.
* **GA regularization:** `l1_on_gate ∈ [5e-5, 5e-4]`, `gate_floor ∈ [0.01,0.05]`.
* **S‑REG barrier:** `s_max ∈ [1.6, 2.2]`, `λ_barrier ∈ [5e-4, 2e-3]`.
* **Loop caps (policy):** `outer_max ∈ {6,8,12,16}`, `l_max ∈ {2,3,4}`, `m_max ∈ {1,2}`.
* **Verify budgets:** `verify_bump_max_k ∈ {2048,4096}`, `max_gap_tokens ∈ {4,8,12}`.
* **FFN cache tolerance:** `rel_L2_target ∈ {0.01,0.02}`, with hit‑rate guard thresholds.

**Study protocol.**

* **Samplers/Pruners:** TPE sampler, median‑stopping pruner; budget small runs to 10–25 trials per milestone‑critical knob set.
* **Metrics:** primary (PPL or task metric), latency p50/p95, VRAM, stability flags (NaNs/overflow), and coverage deltas (verify/abstain).
* **Constraints:** enforce compile‑warmed paths; abort trial if it requests unsupported K‑buckets or backends.
* **Artifacts:** store THPL presets, exact config YAML, compile warm‑up coverage, and seed in SwanLab; attach git commit hash and study ID.

**Repro & plumbing.**

* Environments are resolved via **uv**; the `pyproject.toml` lock ensures identical wheels across trials.
* Each study run exports a **manifest** (config+warm‑up coverage+study ID) to allow byte‑for‑byte regeneration of the best trial.

---

## 16) Example config (small model excerpt + attention / policy knobs + S-REG)

```yaml
model:
  # See §2 for variants.
  d_model: 896
  hrm_clusters: 4            # Q0.5B; use 6 for Q1.5B
  ffn_families: ["qwen"]

hnet:
  targets_bits: { v0: 64->32, v1: 1024->512, v2: 16384->8192 }
  zero_band: 0.20
  ratio_loss: 0.01
  boundary_entropy: 0.01
  attn_neighbors: { v0: 16, v1: 2 }
  hockeystick:
    enable: true
    init_select_rate: 0.40
    window_steps: 2000
    spike_step_range: [100, 800]
    remediate: {ratio_loss_boost: 1.5, main_lr_scale: 0.5, retries: 1}
  spacing_helpers:
    fba: {enable: true, weight: 0.3}
    anchor_prior: {enable: false, max_coef: 0.05}
  boundary:
    innovation_feature: false
  rechunk_on_demand:
    enable: false
    trigger: {innov_pctl: 97, proto_sim_max: 0.15}
    window_tokens: 1024

# ---------------- HRM: band widths + iteration caps ----------------
# Select ONE of the following per run (comment the other).

# --- Q0.5B HRM bands (high granularity) ---
hrm:
  bands:
    L_width: 1920
    M_width: 2880
    G_width: 3840         # = 2 * L_width
  L: { iter_cap: 4 }
  M: { iter_cap: 2 }
  G: { iter_cap: 1 }
  broadcast:
    film_groups: 8
    innovation_gate: { enable: true, w: 0.5 }
  halting:
    type: mlp
    mlp_widen: 4
    step_penalty_target: 0.01
    cosine_safety: { enable_if_outer_cap: 8, epsilon: 0.05, gamma: 5.0 }
    one_step_gradient: false
    use_innovation: true

# # --- Q1.5B HRM bands (budget-adjusted) ---
# hrm:
#   bands:
#     L_width: 3072
#     M_width: 4096        # override (not 1.5×) to reduce quadratic cost
#     G_width: 6144        # = 2 * L_width
#   L: { iter_cap: 4 }
#   M: { iter_cap: 2 }
#   G: { iter_cap: 1 }
#   broadcast:
#     film_groups: 8
#     innovation_gate: { enable: true, w: 0.5 }
#   halting:
#     type: mlp
#     mlp_widen: 4
#     step_penalty_target: 0.01
#     cosine_safety: { enable_if_outer_cap: 8, epsilon: 0.05, gamma: 5.0 }
#     one_step_gradient: false
#     use_innovation: true

hrm_experts:
  compute_prior_kappa: { L: 0.20, M: 0.10 }
  curricula:
    L: [{at:0.00, sizes:[1.00]}, {at:0.05, sizes:[0.75,1.50]}, {at:0.10, sizes:[0.50]}]
    M: [{at:0.00, sizes:[1.50]}, {at:0.05, sizes:[1.00]}, {at:0.10, sizes:[2.00]}]
  L_per_cluster:
    fixed: { mult: 1.00, weight: 0.4->0.0 }
    sizes: [{mult:0.50,count:2},{mult:0.75,count:2},{mult:1.00,count:2},{mult:1.50,count:1}]
  M_per_cluster:
    fixed: { mult: 1.50, weight: 0.4->0.0 }
    sizes: [{mult:1.00,count:1},{mult:1.50,count:1},{mult:2.00,count:1}]
  routing:
    topk_expert: 1
    switch_lbl_alpha: [0.02,0.01,0.005]
    z_loss: 1.0e-3
    bias_from_innovation:
      hrm_cluster: {enable: true, coef: 0.1}
      ffn_mor:     {enable: true, coef: 0.1}
  rounding_policy:
    pow2_snap_tol: 0.05        # ±5% → snap to power-of-two
    align: 64
    align_tol: 0.02            # ≤2% → prefer 64-aligned
    min_width:
      q05b: 512
      q15b: 1024
    record_ratio_as: "rounded_width / band_width"
  # ---- S-REG acknowledgment for HRM experts ----
  scales:
    compute_prior_unchanged: true   # κ unchanged; S-REG orthogonal
  
  # ---------------- HRM-L/M expert tiers & multiplicities -------------
  # (ratios are pre-round; actual widths come from rounding_policy + bands)
  # Comment unused model settings
  L_per_cluster:
    # Q0.5B: 8 per cluster
    ratios: [0.2667, 0.50, 0.75, 1.00, 1.25]
    counts: { "0.2667": 4, "0.50": 1, "0.75": 1, "1.00": 1, "1.25": 1 }
    # Q1.5B: 6 per cluster
    #ratios: [0.3333, 0.50, 0.75, 1.00, 1.20]
    #counts: { "0.3333": 2, "0.50": 1, "0.75": 1, "1.00": 1, "1.20": 1 }
    fixed:
      mult: 1.00
      weight: "0.4->0.0"       # anneal slightly slower during warm-start

  M_per_cluster:
    # Both variants: 5 per cluster, one of each ratio in the set for that variant
    # Comment unused model settings
    #q05b:
    ratios: [0.2667, 0.50, 0.75, 1.00, 1.25]
    counts: { "0.2667": 1, "0.50": 1, "0.75": 1, "1.00": 1, "1.25": 1 }
    #q15b:
    #ratios: [0.3333, 0.50, 0.75, 1.00, 1.20]
    #counts: { "0.3333": 1, "0.50": 1, "0.75": 1, "1.00": 1, "1.20": 1 }
    fixed:
      mult: 1.00               # fixed path runs at band width
      weight: "0.4->0.0"       # anneal slightly slower during warm-start

  routing:
    topk_expert: 1
    capacity_factor: 1.25
    switch_lbl_alpha: [0.02, 0.01, 0.005]
    z_loss: 1.0e-3
    compute_prior_kappa: { L: 0.34, M: 0.25 }   # raised to bias small tiers
    curricula:
      policy: "small_to_large"
      unlock:
        - { at_frac_tokens: 0.00, tiers: ["smallest"] }
        - { at_frac_tokens: 0.05, tiers: ["0.50","0.75"] }
        - { at_frac_tokens: 0.10, tiers: ["1.00"] }
        - { at_frac_tokens: 0.15, tiers: [">1.0_escape"] }
    tier_level_ec_fallback:
      enable: true
      triggers: { overflow_gt: 0.05, dead_gt: 0.10, min_steps: 1000 }
    bias_from_innovation:
      hrm_cluster: { enable: true, coef: 0.1 }
      ffn_mor:     { enable: true, coef: 0.1 }

ffn_bank:
  families:
    qwen:
      clusters: 8
      adapters: { shared: true }
      donor_ln: { copy: true, unfreeze_gains_if: "align_stalls" }
      routing:
        family_topk: 1
        cluster_mor: { mix: α(hG'), topk: 1 }
        expert_topk: 1
        schedule_clusters: { warmup_tokens_frac: 0.10, start: 1, after: 2 }
        switch_lbl_alpha: [0.02,0.01,0.005]
        z_loss: 1.0e-3
      fixed_expert:
        width: 4096
        weight: 0.6->0.0
      align_loss_weight: 0.01
      partial_reinit:
        rho_applied: 0.10
        rho_target: 0.25
        min_cols: "max(32, ceil(0.05*f_i))"
      # ---- S-REG hooks for donor FFN outputs ----
      scales:
        apply_routed: true   # s_type.ffn_retrieve * s_layer.G
        apply_fixed:  true   # s_type.ffn_fixed    * s_layer.G

attention:
  ssa:
    enable: {hrm_l: true, hrm_m: false}
    temp_mlp: {hidden: 64, share_across_layers: true}
    tau: {min: 0.5, max: 2.0, init_bias: 0.0}
    value_scale: {enable: false}
  gated_attention:
    enable: {hrm_l: true, hrm_m: true}   # GA mandated on both L and M
    granularity: "head"
    condition: "query"
    init: {bias: -1.5, w_std: 0.02}
    regularization: {l1_on_gate: 1e-4, gate_floor: 0.02}
    lr_mult: 0.5
    value_gate_also: false
  silm:
    enable: false
    where: {trunk_layers: "all", hrm_l_layers: []}
    alpha: 1.64872
    beta:  1.64872
    tau: {v0: 10, v1: 2, v2: 1}
    clamp_t: {min: 0, max: null}
  attention_softmax_hygiene:
    enable: {hrm_l: true, hrm_m: false}
    center_logits: true
    key_bias: {enable: true, l2_weight: 1e-5}
    logit_clip: {enable: true, tau: 6.0}
  sigmoid_attention:
    enable: {hrm_l: false}     # if sink persists, set true; SSA auto-disabled on these bands
    head_bias_init: -1.5
    scale_norm: true
  # ---- S-REG hook for all attention residuals ----
  scales:
    apply: true   # residual post-mix scaled by s_type.attn * s_layer.attn

# Long-context registry, ESE, DSA (frozen at M5-LC)
long_context:
  enable: false
  ese:
    segment_bytes: 2048
    max_latents_per_segment: 4
    residual: {enable: true, rank: 8}
    adapter_ln_affine: true
    rate_lambda: {start: 1.0, target_latents: 2.0}
    cache:
      enable: true
      cache_key: ["doc_id","seg_idx","ESE_ckpt_hash","adapter_hash"]
  attention_registry:
    trunk:
      pattern:
        - {layers: [12,16,20,24], impl: "dsa"}
        - {layers: [28],          impl: "power", m: 256}
      hybrid_heads: {enable: false}
    hrm_l:
      pattern:
        - {layers: "all", impl: "dotprod"}
    immutable_after_compile: true
  dsa_defaults:
    ratio: 0.05
    k_min: 256
    k_max: 2048
    verify_bump_max_k: 4096
    warmup_all_powers_of_two: true
  puzzle_override:
    force_dense_attn_if_short: true
    short_context_gate: true
  hybrid_positional:
    rope_nope_pattern: {rope_layers: 3, nope_layers: 1}
    qk_norm: {enable: true, eps: 1e-6}
    theta_scale: {enable: false, factor: 1.0}

selective_confidence:
  enable: true
  mode: "verify_only"
  target_coverage: 0.92
  lambda_seq_vs_tok: 0.6
  thresholds: {abstain: 0.35, verify: 0.45, ok: 0.55, tok_low: 0.20}
  features:
    use_proto_distance: true
    use_adapter_ood:   true
    use_innovation:    true
    use_mor_alpha:     true
    use_mc_dropout:    false
  verify_path:
    extra_outer_step: true
    rare_top2_ffn:    true
    cross_family_check: true
  verify_qol:
    proto_two_shot: {enable: true, cos_thresh: 0.85}
    span_bunching:  {enable: true, max_gap_tokens: 8}

entity_risk:
  enable: true
  head: {hidden: 64}
  features: {byte_patterns: true, cap_run: true, digit_run: true}
  thresholds: {token_flag: 0.6, segment_flag: 0.5}
  route_to:
    selective_confidence: true
    verify_path_on_flag: true

thpl:
  default_payload_codec: "UTF8_NFC"
  default_num_std: true

compile_warmup:
  path_exciter:
    enable: true
    hrm_ratios:
      q05b: [0.2667, 0.50, 0.75, 1.00, 1.25]
      q15b: [0.3333, 0.50, 0.75, 1.00, 1.20]
    cover_bands: ["L","M"]        # route once to each ratio per band per cluster
    cover_dsa_powers_of_two: true  # see §10.1.4
    cover_headers: ["nl_chat","code_chat","sudoku_9x9","arc_30x30","bptt_demo"]
    force_verify_case: true

config_guards:
  - rule: "ssa_on_band_implies_not_sigmoid"
  - rule: "dense_short_circuit_when_N_le_k_min"
  - rule: "one_backend_per_layer"
  - rule: "trunk_full_sdpa_requires_mem_ok"
  - rule: "no_abstain_when_mode_verify_only"

# ---------------- S-REG master block ----------------
scales:
  enable: true
  parameterization: "softplus"      # non-negative; compile-safe
  clamp:
    max: 2.0                        # soft barrier (see optimizer.barrier)
  # These values are softplus-ed, so ~0.54->1.0 and ~0.11->0.75 and ~0.0->0.667
  init:
    type_priors:                    # module-type priors
      attn:         0.45
      ffn_retrieve: 0.35
      ffn_fixed:    0.25
      hrmL_expert:  0.25
      hrmM_expert:  0.30
      hrmL_fixed:   0.35
      hrmM_fixed:   0.35
      film:         0.55
      umoe_tap:     0.25
    per_layer:                      # band/layer priors
      L:         0.35
      M:         0.40
      G:         0.55
      attn:      0.55
      attn_tap:  0.35
  optimizer:
    lr_mult: 0.5
    weight_decay: 0.0
    barrier:
      enable: true
      s_max: 2.0
      lambda: 1.0e-3
  schedule:
    warm_start_freeze_tokens_frac: 0.02

# ---- Layer-ID scale tables (initializations) ----
scales:
  layer_id:

    # 24-layer (0.5B source model)
    trunk:
      attn:      [0.50, 0.50, 0.60, 0.50, 0.50, 0.50, 0.60, 0.52, 0.53, 0.55, 0.53, 0.60,
                  0.53, 0.53, 0.55, 0.52, 0.53, 0.53, 0.50, 0.60, 0.50, 0.52, 0.50, 0.60]
      attn_tap:  [0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00,
                  0.00, 0.35, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]

    # 28-layer (1.5B source model)
    # trunk:
    #   attn:      [0.50, 0.50, 0.60, 0.50, 0.50, 0.50, 0.60, 0.50, 0.53, 0.55, 0.53, 0.55, 0.53, 0.60, 0.53, 0.53, 0.55, 0.52, 0.53, 0.60, 0.50, 0.52, 0.50, 0.52, 0.55, 0.60, 0.50, 0.60]
    #   attn_tap:  [0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00, 0.00, 0.35, 0.00, 0.00, 0.00, 0.00]

# ---- S-REG hooks (already in your config; shown here for clarity) ----
attention:
  scales:
    apply: true                        # s_type.attn * s_band.attn * s_lid.attn[ℓ]

hrm:
  scales:
    apply_film: true

ffn_bank:
  families:
    qwen:
      scales:
        apply_routed: true
        apply_fixed:  true

# ---- Donor capture defaults (family bank carving) ----
ffn_bank:
  families:
    qwen:
      carving:
        # DEFAULT (extended) capture set for higher bank capacity
        capture_layers: [4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]  # S = 13 (default)

        # Optional minimal set (leave commented unless you re-budget)
        # capture_layers: [6, 8, 10, 12, 13, 14, 16, 18, 20]               # S = 9 (optional)

        eta: 0.60                      # carve fraction η (↑ from 0.50)
        overlap_max: 0.15              # ≤15% neuron overlap across carved experts

      # Parameter-budget assumptions used for counts (informational only)
      budget_notes:
        donor_ffn_expansion: "f_d = 6 * d_n (planning uplift; affects counts only)"
        formulas:
          donor_ffn:      "P_donor_FFN = 2 * S * d_n * f_d"
          bank:           "P_bank = (eta * (1 + overlap_max)) * P_donor_FFN"
          interfaces:     "P_family_ifaces ≈ 2 * d_n * d_n"
          site_adapters:  "P_site_total ≈ (8 * d_n) * L_tap"
          trunk:          "P_trunk ≈ L_trunk * (4 * d_n^2)"
          non_hrm_total:  "P_nonHRM_total ≈ P_trunk + P_bank + P_family_ifaces + P_site_total + P_router_proto"
        defaults_q05b:
          d_n: 896
          f_d: "5376  (= 6 * 896)"
          S: 13
          eta: 0.60
          overlap_max: 0.15
          L_trunk: 24
          L_tap: 2
          approx_params_M:
            donor_ffn: 125.24
            bank: 86.42
            interfaces: 1.61
            site_adapters: 0.01
            total_family: 88.04
            trunk: 77.07
            non_hrm_total: 165.11
        # defaults_q15b (commented; enable if training Q1.5B)
        # d_n: 1536
        # f_d: "9216 (= 6 * 1536)"
        # S: 13
        # eta: 0.60
        # overlap_max: 0.15
        # L_trunk: 28
        # L_tap: 4
        # approx_params_M:
        #   donor_ffn: 368.05
        #   bank: 253.95
        #   interfaces: 4.72
        #   site_adapters: 0.05
        #   total_family: 258.74
        #   trunk: 264.24
        #   non_hrm_total: 522.98

# ---- UMoE-lite site taps (explicit 0.5B defaults; 1.5B commented) ----
umoe_lite:
  enable: true
  layers: [10, 14]                 # Q0.5B: two taps explicit
  router_q_bias: 0.25
  site_adapters:
    kind: "diag_bias"              # per-layer diag scale + bias for fixed+routed paths
    init_scale: 1.0
    init_bias: 0.0
    per_layer: true
  expert_fixed_always_on: true
  p_scale_init: 0.3

# Q1.5B taps (enable if using 1.5B)
# umoe_lite:
#   enable: true
#   layers: [6, 12, 18, 24]        # Q1.5B: four taps explicit
#   router_q_bias: 0.25
#   site_adapters:
#     kind: "diag_bias"
#     init_scale: 1.0
#     init_bias: 0.0
#     per_layer: true
#   expert_fixed_always_on: true
#   p_scale_init: 0.3

training:
  umoe_warmup_steps: 30000         # freeze shared experts during warm-up
  unfreeze_core_lr_mult: 0.1
```

> **Monitoring intents (informational):**
> Q0.5B: L ~88–92% mass on 0.2667×; M ~85–90% on 0.2667×.
> Q1.5B: L ≥90% on 0.3333×; M ≥88% on 0.3333×. Escape tiers remain rare.

---

## 17) Token & compute budgets

* **MoirAI‑0.9B‑A0.3B**

  * **Avg active parameters / token:** **≈ 0.30–0.38B**
  * **Feasible pretraining:** **~4–7B tokens** on a single high‑end consumer GPU with checkpointing and truncated BPTT.

* **MoirAI‑2.9B‑A0.9B**

  * **Avg active parameters / token:** **≈ 0.82–0.90B**
  * **Feasible pretraining:** **~2–4B tokens** at similar wall‑time due to donor FFN knowledge and efficient routing.

These budgets guide router capacity factors, verify‑path ceilings, and long‑context backend choices so latency and VRAM remain within target envelopes.

---

## 18) Milestone impacts, ablations, caveats & guards

### 18.1 Carry‑forward choices (M5‑LC)

* **Trunk backend:** pick **one** at matched compute/memory (periodic SDPA vs **Power** vs **DSA**); freeze the per‑layer registry after selection.
* **ESE:** ON vs OFF. Carry ESE forward **only** if long‑context quality drop ≤0.5% abs with ≥30% latency/VRAM win.
* **Positional scheme:** If enabled (RoPE/NoPE + QK‑Norm), freeze with the registry.

### 18.2 Late‑swap shielding (if unavoidable)

If a backend must be changed post M5‑LC:

```
Loss = λ_attn · ||A_new − A_old||² + λ_out · ||Y_new − Y_old||² + 0.5 · main_loss
λ_attn=1.0, λ_out=0.5, duration≈1 epoch (≤2e9 tokens)
```

Freeze everything but the target band; then unfreeze normally after the shield.

### 18.3 Common caveats & mitigations

| Feature             | Caveat                       | Mitigation                                                  |
| ------------------- | ---------------------------- | ----------------------------------------------------------- |
| DSA (top‑K)         | Small K → recall loss        | Length‑scaled K, `k_min`, verify‑bump, SILM indexer bias    |
| DSA kernels         | Optimized kernel unavailable | Triton/PyTorch fallback (shapes unchanged)                  |
| Power vs DSA        | Don’t stack same layer       | Registry enforces one backend per layer                     |
| UMoE‑lite           | Throughput hit on 0.5B       | Limit to few trunk layers; keep OFF if >10% step‑time hit   |
| Attention sink      | Residual even with GA        | Enable sigmoid attention + hygiene; keep `gate_floor` clamp |
| Verify path latency | Tails too heavy              | Prototype two‑shot + span‑bunching; cap verify bump powers  |
| Fixed paths removal | Small regressions            | Switch to “repurpose” (reverse‑KL + orthogonality adapters) |

### 18.4 Guards (recap)

* **SSA vs Sigmoid:** SSA automatically disabled where sigmoid attention is enabled.
* **Dense‑short‑circuit:** For very short contexts or THPL `force_dense_attn`, bypass DSA while keeping shapes static.
* **Verify‑only mode:** Must **never** abstain; verify path only.
* **Immutable registry:** After compile warm‑up at M5‑LC, per‑layer backends are frozen.

#### 18.5 Ablation checklist (controversial knobs)

**Backends & long‑context**

* **SILM** on trunk dotprod/DSA: length‑generalization (≥16×), DSA top‑K recall, NaNs.
* **Rotary 25% vs hybrid 3/1 RoPE/NoPE**: equal compute, long‑context loss delta, stability.
* **Limited dynamic mixer (head‑scoped)**: Δ step‑time (≤5%), entropy of mixing (near‑binary), domain‑wise gains.

**Verify & selective confidence**

* **Verify top‑2 FFN (verify‑only)**: p95 on flagged spans vs sequential top‑2, cache hit‑rate, trigger rate <3% tokens, win‑rate ≥60%.
* **Span bunching**: latency tails vs coverage.

**H‑Net & chunking**

* **Re‑chunk on demand**: p95/p99 tails, first‑byte error reduction, trigger budget compliance.
* **FBA on/off (text)**: first‑byte boundary errors, global perplexity ≤0.1% drift.

**HRM & experts**

* **GA on HRM‑M**: sink metrics, training loss variance, latency impact.
* **HRM‑L micro‑experts**: local pick‑rates, outer‑step reductions, net step‑time (Δ≤5%).
* **Repurpose vs removal of fixed paths**: dev deltas ≤0.5% abs; cache fidelity when repurposed anchors are used.
* **Partial re‑init schedule vs diversity tier**: (L_{\text{align}}) trend, perplexity vs step‑time; choose safer route for milestone freeze.

**Residual & precision**

* **Learned convex aggregator vs hard add**: convergence speed and stability; no regression on variance metrics.
* **Precision fallback (bf16 scopes)**: stability vs throughput; document per‑module exceptions; maintain QK‑Norm and logit hygiene.

**Reasoning**

* **Reasoning traces during SFT**: effect on multi‑step puzzles and long‑horizon NL tasks; ensure no leakage into generated text.
* **MTP head**: quality/latency trade‑off at small (m); disable if cost outweighs gains.

Each ablation specifies: datasets/slices, compute equalization, primary metric(s), latency/VRAM budget, acceptance thresholds, and roll‑forward criteria.

---

## References

[^1]: **H‑Net: Hierarchical Tokenization from Raw Bytes.** arXiv:2507.07955. [https://arxiv.org/abs/2507.07955](https://arxiv.org/abs/2507.07955)

[^2]: **H‑Net, Past & “Attention as a Primitive.”** GoombaLab blog (2025). [https://goombalab.github.io/blog/2025/hnet-past/#attention-as-a-primitive](https://goombalab.github.io/blog/2025/hnet-past/#attention-as-a-primitive)

[^3]: **Hierarchical Reasoning Model (HRM).** arXiv:2506.21734. [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)

[^4]: **HRM Code.** sapientinc/HRM (GitHub). [https://github.com/sapientinc/HRM](https://github.com/sapientinc/HRM)

[^5]: **CMoE: Converting Mixture‑of‑Experts from Dense to MoEs (training‑free).** arXiv:2502.04416. [https://arxiv.org/abs/2502.04416](https://arxiv.org/abs/2502.04416)

[^6]: **Generalized MoEfication for Dense Pretrained Models.** EMNLP 2024. [https://aclanthology.org/2024.emnlp-main.563.pdf](https://aclanthology.org/2024.emnlp-main.563.pdf)

[^7]: **HC‑SMoE: Retraining‑Free Merging of Sparse MoE via Hierarchical Clustering.** arXiv:2410.08589. [https://arxiv.org/abs/2410.08589](https://arxiv.org/abs/2410.08589)

[^8]: **UMoE: Unified/Universal MoE.** arXiv:2505.07260. [https://arxiv.org/abs/2505.07260](https://arxiv.org/abs/2505.07260)

[^9]: **SILM: Scale‑Invariant Logit Modulation.** arXiv:2505.17083. [https://arxiv.org/abs/2505.17083](https://arxiv.org/abs/2505.17083)

[^10]: **Selective Self‑Attention.** arXiv:2411.12892. [https://arxiv.org/abs/2411.12892](https://arxiv.org/abs/2411.12892)

[^11]: **Confidence‑Aware Selective Generation (abstention/verification).** arXiv:2509.03531. [https://arxiv.org/abs/2509.03531](https://arxiv.org/abs/2509.03531)

[^12]: **CompLLM: Semantic Compression with LLMs (segment→embedding).** arXiv:2304.12512. [https://arxiv.org/abs/2304.12512](https://arxiv.org/abs/2304.12512)

[^13]: **Qwen3‑Next (architecture & RoPE/NoPE mixing notes).** Qwen blog (2025). [https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd)

[^14]: **Qwen2/2.5 Technical Details (dims/sizes).** Qwen3 docs. [https://qwenlm.github.io/blog/qwen3/](https://qwenlm.github.io/blog/qwen3/)

[^15]: **Qwen Key Concepts / Model Specs.** Qwen site. [https://qwen.ai/research](https://qwen.ai/research)

[^16]: **H‑Net Router: Practical Notes.** declan.dev. [https://www.deklan.dev/hnet-router](https://www.deklan.dev/hnet-router)

[^17]: **H‑Net Intuitions Gallery.** [https://main-horse.github.io/hnet/intuitions/](https://main-horse.github.io/hnet/intuitions/)

[^18]: **DeepSeek V3.2 – Dynamic Sparse Attention (DSA).** Technical report (internal).

[^19]: **Power Attention: Scaling Context Requires Rethinking Attention.** arXiv:2507.04239. [https://arxiv.org/abs/2507.04239](https://arxiv.org/abs/2507.04239)

[^20]: **Attention Sink & Activation Outliers.** ICLR 2025. [https://openreview.net/forum?id=78Nn4QJTEN](https://openreview.net/forum?id=78Nn4QJTEN)

[^21]: **Quantizable Transformers: Clipped Softmax & Gated Attention.** arXiv:2306.12929. [https://arxiv.org/abs/2306.12929](https://arxiv.org/abs/2306.12929)

[^22]: **Outlier Suppression for Low‑bit LMs.** arXiv:2209.13325. [https://arxiv.org/abs/2209.13325](https://arxiv.org/abs/2209.13325)

[^23]: **LayerNorm’s Role in Attention Expressivity.** arXiv:2305.02582. [https://arxiv.org/abs/2305.02582](https://arxiv.org/abs/2305.02582)

[^24]: **TRM vs HRM (generalization under tiny nets).** alphaXiv 2510.04871. [https://www.emergentmind.com/papers/2510.04871](https://www.emergentmind.com/papers/2510.04871)

[^25]: **Gated Attention for Large Language Models.** arXiv:2505.06708. [https://arxiv.org/abs/2505.06708](https://arxiv.org/abs/2505.06708)

[^26]: **Modded NanoGPT Speedrun.** GitHub: KellerJordan/modded‑nanogpt. [https://github.com/KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)

[^27]: **ES Fine‑tuning.** arXiv:2509.24372v1; GitHub: VsonicV/es‑fine‑tuning‑paper. [https://arxiv.org/abs/2509.24372](https://arxiv.org/abs/2509.24372), [https://github.com/VsonicV/es-fine-tuning-paper](https://github.com/VsonicV/es-fine-tuning-paper)

[^28]: **Halting head inference logit threshold (discussion).** X/Twitter @ritteradam. [https://x.com/ritteradam/status/1983291492685824352](https://x.com/ritteradam/status/1983291492685824352)

[^29]: **Reasoning data during SFT (Synergy).** NVIDIA Research. [https://research.nvidia.com/labs/adlr/Synergy/](https://research.nvidia.com/labs/adlr/Synergy/)

[^30]: **Train with ≤16 outer steps; allow longer at inference (practice).** X/Twitter @huskydogewoof. [https://x.com/huskydogewoof/status/1982503109042831472](https://x.com/huskydogewoof/status/1982503109042831472)

[^31]: **BPTT on full final outer step; training lessons (TRM talk).** YouTube. [https://youtu.be/ETukUNsn_wQ](https://youtu.be/ETukUNsn_wQ)

[^32]: **Using fp16 over bf16 (trade‑offs & evidence).** arXiv:2510.26788. [https://arxiv.org/abs/2510.26788](https://arxiv.org/abs/2510.26788)

[^33]: **PyTorch AMP — Gradient scaling.** Docs. [https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling](https://docs.pytorch.org/docs/stable/amp.html#gradient-scaling)
