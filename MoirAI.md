# MoirAI

**Dynamic hierarchical tokens (H-Net) + hierarchical recurrent reasoning (HRM-L/M/G) + hierarchical MoE experts**
Knowledge bank via *transplanted* FFN experts (Qwen by default; optional Mamba mixers), plus *trained-from-scratch* HRM experts (heterogeneous sizes). Efficient, compile-friendly, and staged for stability.

---

## Introduction (what these subsystems are and why)

Modern large language models are powerful but face fundamental limitations: they rely on static, pre-defined tokenizers that struggle with raw, byte-level data, and their fixed-depth architectures limit multi-step reasoning. **MoirAI** overcomes these by integrating three synergistic subsystems that learn to **reason efficiently and dynamically from bytes**.

### H-Net (Dynamic Hierarchical Chunking)

* **What:** H-Net is a tokenizer-free input layer that learns to segment raw byte streams into **three token bands**: **v0 (low)** for fine details, **v1 (mid)** for clauses/semantic chunks, and **v2 (high)** for global context. The model, not a fixed vocab, decides boundaries.
* **Why:** Static tokenizers decouple representation from reasoning. Learned chunking yields (a) semantically coherent units that adapt to data/task, (b) **fewer, larger high-level tokens** for efficient global passes, and (c) a **reverse lattice** guaranteeing perfect byte-level generation.
* **Targets:** we softly target **v0≈32 bits**, **v1≈512 bits**, **v2≈8 kbits**, with a brief curriculum to reach them.

### HRM (Hierarchical Recurrent Reasoning)

* **What:** HRM is a multi-timescale reasoning stack: **HRM-L** iterates rapidly over v0, **HRM-M** iterates more slowly over v1, and **HRM-G** performs a single, high-level update per outer step using pooled states. A learned **halting** mechanism stops early when extra compute won’t help.
* **Why:** Hard problems need **iteration**, not one pass. HRM lets the model “think” as needed while summarizing easy parts—deep latent reasoning **without external CoT text**. A **global control signal** from HRM-G is **broadcast via FiLM** back to L before byte logits.

### Hierarchical MoE (Conditional Knowledge & Reasoning)

* **What:** Two conditional expert systems:

  1. **FFN Knowledge Experts** transplanted from strong donors (e.g., Qwen), serving as a ready-to-use **knowledge bank**.
  2. **HRM Reasoning Experts** (heterogeneous sizes), trained from scratch, supplying varied **reasoning horsepower**.
* **Why:** Inputs differ in knowledge needs and difficulty. Routing to a **small set of specialists** keeps active compute low; transplantation **slashes pretraining cost**; heterogeneous HRM experts match compute to difficulty.

---

## 0) Model variants (trunks trained from scratch; Qwen-aligned dims)

| Variant          | Layers |  d\_model | Donor FFN width (f) | HRM clusters C\_hrm | FFN clusters K (per family) |
| ---------------- | -----: | --------: | ------------------: | ------------------: | --------------------------: |
| **MoirAI-Q0.5B** |     24 |   **896** |           **4 864** |               **4** |                       **8** |
| **MoirAI-Q1.5B** |     28 | **1 536** |           **8 960** |               **6** |                       **8** |

*(“Family” = donor lineage; default: Qwen. Optional: Mamba mixers as a second family.)*

---

## 1) Data & representation

* **ARC-AGI packing:** 4 bits/cell; reserve two codes for **row** and **sample** separators; 30×30 < 4 KB.
* **Bytes in/out:** Everything is trained from bytes. Reverse lattice (from H-Net) ensures we can map L’s state back to bytes exactly for generation and loss.

---

## 2) H-Net chunker (3 levels; **new targets**)

### 2.1 Targets (bits per chunk)

* **v0:** **32 bits** (\~4 bytes; ≈ BPE-token scale).
* **v1:** **512 bits** (\~64 bytes; ≈ 10–15 words / a clause).
* **v2:** **8 192 bits** (≈ 1 KB; ≈ 150–200 words / \~⅘ of a page).

### 2.2 Level blocks (each level ℓ ∈ {0,1,2})

* Input stream (bytes for v0; vℓ for ℓ>0) → **embed** → **1D Transformer** → **boundary head** σ(b) → **soft pooling** between boundaries to produce vℓ+1 tokens.
* **Reverse lattice:** store (start, end, weights) per v0 segment for byte reconstruction.

### 2.3 Losses & regularizers

* **Target-bits Huber** with zero-penalty band ±15% (±20% during target anneal; see schedule).
* **Ratio term** to discourage overall compression extremes.
* **Boundary entropy** term to prevent spiky, degenerate segmentation.

### 2.4 Chunk-aware attention windows (speed)

* **v0 attention:** within-chunk + **±16 neighbor chunks** (tiny slack window across boundary).
* **v1 attention:** within-chunk + **±2 neighbor chunks**.
* **v2 attention:** full.

### 2.5 Target-bits **curriculum**

* **M1 → early M2:** start **v0=64**, **v1=1 024**, **v2=16 384** bits.
* **Late M2 → M3:** cosine-anneal to **32 / 512 / 8 192**; **freeze Level-2** for one epoch during the switch. Keep ratio/entropy losses active; use ±20% zero band while annealing.

---

## 3) HRM (clusters, loops, halting, broadcast)

We split the HRM reasoning engine into **C\_hrm clusters** to specialize its function. At the beginning of each outer step, a **cluster router** selects the **top-1 cluster** using the current global state `hG` (details in §3.0). The chosen cluster `c*` then runs its own **HRM-G** and activates its private **HRM-L/M expert banks** (banks are *not shared across clusters*). This lets the model pick a complete, specialized reasoning pathway (e.g., “logic-puzzle mode” vs “summarization mode”) for the step while keeping compute tight.

### 3.1 HRM Cluster Routing

**Inputs.** Previous outer step’s global state `hG`.
**Mechanism.** Project to a query `q_hrm = W_hrm_q · hG`. Compare against learned cluster prototypes `{μ_c}` to form logits `ℓ_c = q_hrm · μ_c`.
**Selection.** **Top-1** cluster `argmax_c ℓ_c`.
**Stability.** Train with **Switch-style load-balance** (auxiliary α≈1e-2) and **z-loss** (1e-3) on the cluster router to avoid collapse. Entropy targets can be scheduled (looser early, tighter later).
**Note.** HRM **always** picks **one** cluster per step (no 1→2 schedule here); the 1→2 schedule applies to **FFN clusters per family** only.

### 3.2 Widths & iteration caps

* **HRM-L:** width **d**, iter cap **k ≤ 4**.
* **HRM-M:** width **dₘ = 1.5·d**, iter cap **≤ 2**.
* **HRM-G:** width **2·d**, single step per outer step.

### 3.3 Exact loop (cluster c\* chosen; omit c\* for brevity)

**L inner loop (t=1…k)**

```
ctxL_t = Attn_L(hL_{t-1}, v0)                      # [T0, d]
eL_t   = ExpertL(hL_{t-1})                         # top-1 expert (size-tiered MLP)
hL̃_t   = hL_{t-1} + DropPath(eL_t)
hL_t   = GRU_L(hL̃_t, ctxL_t)
hL_t  += w_fixL · FixedL(hL_{t-1})                 # fixed HRM-L per cluster, small weight
```

**M inner loop (u=1…U≤2)**

```
poolL = Pool(hL_k)                                  # mean/max over routed low tokens
ctxM  = Attn_M(hM_{u-1}, v1, extra_kv=poolL)
eM_u  = ExpertM(hM_{u-1})                           # top-1 expert
hM̃_u  = hM_{u-1} + DropPath(eM_u)
hM_u  = GRU_M(hM̃_u, ctxM)
hM_u += w_fixM · FixedM(hM_{u-1})                   # fixed HRM-M per cluster, small weight
```

**G update + FFN retrieval, then broadcast**

```
poolM = Pool(hM_U);  poolL = Pool(hL_k)
xG    = concat(poolL, poolM, CLS(v2))
hG'   = GRU_G(hG, xG)
q     = Wq · hG'                                     # query vector

# FFN routing (per family; see §4)
y_FFN = RetrieveFFN(q, v0, high_token_prior=v2)      # pooled fixed + top-1 routed expert from selected family/cluster
hG''  = G_update(hG', y_FFN)                         # small MLP/GRU integration

(γ,β)   = FiLM(hG'')                                  # per-channel or grouped (8–16 groups)
hL_mod  = (1 + γ) ⊙ hL_k + β
logits  = ByteHead(hL_mod)
```

### 3.4 Halting (outer steps)

* **Primary:** **2-layer MLP halter** on `hG''` → ACT continue prob. **Step penalty λₒ target = 0.01** (linear warm-up).
* **Optional cosine veto** when `outer_cap > 8`: stop if `‖hG_t − hG_{t-1}‖₂ < ε` (ε=0.05) or subtract γ=5 from continue logit.
* **Convergence regularizer (OFF by default):** small penalty on too-rapid shrinkage of ‖Δh‖ or on GRU Jacobian spectral norm; enable if you raise outer caps.
* **One-step gradient option (OFF by default):** for outer caps >4, backprop only through final states; add light deep supervision on L/M to stabilize.

### 3.5 Optional: HRM-L micro-experts (inner-loop MoE)

Replace the single GRU cell in L with 3–4 tiny GRU cells and a per-token router (**top-1**). Add small Switch-LBL + z-loss *local* to this router. Toggleable.

---

## 4) Hierarchical MoE experts

### 4.1 HRM reasoning experts (trained from scratch; **heterogeneous sizes**; **shared fixed paths**)

Per **HRM cluster** we keep separate **HRM-L** and **HRM-M** expert banks (banks are **not shared across clusters**). Experts are **heterogeneous in size**; routing is **top-1**. To stabilize early training without duplicating parameters, we use **shared fixed HRM paths** whose **weights are tied across clusters**, with **tiny per-cluster scalar gates**.

## HRM-L (operates on v0; state width = d)

* **Expert bank per cluster (top-1):**
  Tiers and counts per cluster: **0.50×d (2)**, **0.75×d (2)**, **1.00×d (2)**, **1.50×d (1)** → **7 experts/cluster**.
  **Adapters per expert:** `In: d → d_e`, core **2-mat MLP** at `d_e`, `Out: d_e → d`. Wrap as a **residual** around the GRU-L update.
* **Shared fixed path (weight-tied across clusters):**
  A single module **FixedL\_shared: d → d** (width = **1.0×d**) used in all clusters.
  At use time in cluster `c`:

  ```
  hL ← hL + w_fixL_c · FixedL_shared(hL_prev)     # w_fixL_c is a scalar gate
  w_fixL_c = a_c · σ(b_c)                          # 2 learned scalars per cluster
  ```

  **Init:** `a_c = 0.4, b_c = 0` for all clusters. **Anneal** `a_c → 0` over M4 unless we repurpose (see below).

## HRM-M (operates on v1; state width = d\_m = 1.5·d)

* **Expert bank per cluster (top-1):**
  Tiers and counts per cluster: **1.00×d (1)**, **1.50×d (1)**, **2.00×d (1)** → **3 experts/cluster**.
  **Adapters per expert (w\.r.t. d\_m):** `In: d_m → d_e`, core **2-mat MLP** at `d_e`, `Out: d_e → d_m`. Residual wrap around GRU-M.
* **Shared fixed path (weight-tied across clusters):**
  **FixedM\_shared: d\_m → d\_m** (width = **1.5×d = d\_m**).
  Per cluster `c`:

  ```
  hM ← hM + w_fixM_c · FixedM_shared(hM_prev)
  w_fixM_c = a'_c · σ(b'_c)                        # 2 scalars per cluster
  ```

  **Init/anneal:** same policy as HRM-L.

## Routing, stability, and curricula (both L and M)

* **Top-1 expert**, capacity factor **1.25**.
* **Stabilizers:** **Switch-LBL** α = 0.02 → 0.01 → 0.005 (warm/main/late); **z-loss** 1e-3; temperature τ 1.2 → 1.0; router logit noise 1.0 → 0.2.
* **Compute prior** (make larger experts earn it): subtract `κ·(d_e/d − 1)` from router logits (**κ\_L = 0.20**, **κ\_M = 0.10**).
* **Size curricula:**

  * **L:** 0–5% tokens enable only **1.0×d**; 5–10% add **0.75×d** and **1.50×d**; ≥10% add **0.50×d**.
  * **M:** 0–5% tokens enable only **1.5×d (= d\_m)**; 5–10% add **1.0×d**; ≥10% add **2.0×d**.
* **Tier-level EC fallback:** if a size tier starves or dominates, flip **that tier** to **Expert-Choice** temporarily (cluster remains top-1).

## Remove vs. repurpose shared fixed paths

* **Default (remove):** monitor fixed-share; if global average `< 20%` for ≥N steps and removing fixed (set all `a_c, a'_c → 0`) causes **≤ 0.5% abs** metric drop, keep them off.
* **Repurpose (keep):** freeze routed experts; **reverse-KL** (routed → fixed) + **orthogonality** on pooled outputs; unfreeze **adapters/LN gains inside FixedL\_shared/FixedM\_shared** only (no widening).
* **Per-variant clusters:** **Q0.5B → 4 clusters**, **Q1.5B → 6 clusters** (keeps HRM params ≳¼ of FFN params).

### 4.2 FFN knowledge experts (transplanted; **family → cluster → expert**)

**Families:**

* **Qwen** (default): copy **LayerNorm + W1/W2** (SwiGLU dropped via gate-compensation).
* **Mamba (optional):** copy mixer FFNs similarly (lives in its own namespace).
  Each family has its own **K=8 FFN clusters** and a **single global fixed FFN expert**.

**Calibration & carving (per family):**

1. **Calibrate** (10–50M mixed tokens): for donor FFNs, compute gate/up/down activations; per-neuron stats (`freq_on`, `mean_abs`, level/domain entropies, co-activations) and **cluster-conditioned gate means** `s_{j,c}`.
2. **Features** φ\_j = \[freq\_on, mean\_abs, entropies, PCA(co-act), corr with high-token codes].
3. **Clusters:** **k-means++** (or spectral on co-activation) into **K=8** clusters.
4. **Experts within a cluster:** k-means++ to E\_c centroids; pick top-K neurons per centroid with **≤10% overlap**; form **two-mat MLP** by **gate-compensation**:
   `W1'_{c}[:,j] = s_{j,c} · W_up[:,j]`, `W2' = W_down[I_e,:]` (optional least-squares fine-tune on calibration).

**Fixed FFN expert (per family):**
Select **broad-utility neurons** by `U_j=α·freq_on + β·mean_abs − γ·entropy_level − γ’·entropy_domain`; build `W1' = s̄⊙W_up`, `W2' = W_down` with global gate mean s̄. Mixed at small weight early, **annealed down** later (or **repurposed** with reverse-KL + orthogonality and adapters/LN gains only).

**Interfaces (shared per family):**

* `A_in: d_native→d_donor` (trainable, shared).
* **Donor LayerNorm** (copy gains/bias; unfreeze gains only if needed).
* Two-mat donor MLP (fixed initially).
* **Pool** (over routed v0 tokens).
* `A_out: d_donor→d_native` (trainable, shared).
  **Alignment loss** on `A_in` outputs (post-donor LN): target |μ|<0.05, |σ−1|<0.05, **L\_align < 0.1**.

**Routing (family → cluster → expert):**

* **Tier 0 (family):** select **top-1 family** by `score_f = q·μ_f + β·prior_f`.
* **Tier 1 (cluster, within family):** **Mixture-of-Routers (MoR):** combine **query head** and **high-token prior** with a learned mixer **α(hG' )**:
  `ℓ_c = α(hG')·(q·μ_c) + (1−α(hG'))·prior_c`.
* **Tier 2 (expert, within cluster):** **top-1 expert** by query only (`q·μ_{c,e}`).
* **Fixed FFN expert:** add from the **selected family only** (default), small weight; anneal down as routed usage rises.
* **Schedule:** one→two FFN clusters per family after 10% tokens; **Switch-LBL + z-loss** active at both levels.

---

## 5) Broadcast FiLM (placement & grouping)

* **Where:** after G integrates FFN output (`hG''`) and **before** the byte head on L’s *final* state of this outer step (or single-step).
* **How:** predict `(γ,β)` either **per channel** or **grouped** (8–16 groups) for higher control bandwidth without large matrices.
* **Equation:** `hL_mod = (1 + γ) ⊙ hL + β; logits = ByteHead(hL_mod)`.

---

## 6) Fixed experts policy (scaffold → fade or repurpose)

* **HRM fixed per cluster (L:1.0×d, M:1.5×d):** always-on at small weight early; **anneal** as routed experts stabilize. If kept, **reverse-KL** (routed→fixed) + **orthogonality** to make it complementary; unfreeze only **adapters/LN gains**.
* **FFN fixed per family:** early **distill** (fixed→routed KL) then anneal weight to 0. If removing incurs >0.5% absolute drop on held-out, **keep** and repurpose as above.

---

## 7) Efficiency & compile rules

* **Compile invariants.** All routers use **static top-k=1** (HRM cluster; HRM L/M experts; FFN family/cluster/expert). Dispatch is tensorized (no Python control flow by batch).
* **Re-compile boundaries.** You **may re-compile only when enabling FFN’s 1→2 cluster schedule** per family. HRM * **does not** change cluster count at runtime (always 1 cluster/step).
* **Shared fixed HRM blocks.** `FixedL_shared`/`FixedM_shared` have **one** weight set used in all clusters; per-cluster gates are **scalars** (don’t alter shapes).
* **Preallocate** per-tier buffers; **tensorize** routing (no Python branches).
* **Expert value cache (optional):** per FFN cluster, maintain low-rank `(A_c,B_c)` (rank 16–32) to approximate pooled expert output from `q`; serve when `‖q−μ_c‖<τ`, EMA-update otherwise.

---

## 8) Transplant specifics (Qwen default)

* **Layer priorities:** **6, 8, 10, 12, 13, 14, 16, 18, 20** (\~247.5 M params). Optional: **17, 11, 15, 4**.
* **Gate compensation:** bake in expected gate via `W1'_{c}[:,j] = s_{j,c}·W_up[:,j]`; drop gate at runtime; optional LS refinement on calibration.
* **Adapters:** **shared per family** (not per expert); A\_in/A\_out are the only trainable mappings across the family initially.
* **Donor LN:** copy gains/bias; unfreeze **gains** later if `L_align` stalls.

*(Mamba family: identical steps on mixer FFNs; lives under its own “family” with its own clusters & fixed.)*

---

## 9) Training plan (milestones & phases)

**M0 — Infra (1 wk)**
ROCm PyTorch≥2.3; ACT; router core; reverse lattice; compile-safe dispatch (static k, tensorized selects). Unit tests (see §12).

**M1 — Chunker (2 wks)**
Copy-task autoencoder + next-byte LM on 10 MB. Targets **v0=64, v1=1 024, v2=16 384**; chunk-aware attention; ratio/entropy losses on. Exit: copy exact-match ≥99.9%; bits/chunk means within ±15% (±20% ok during warm-up).

**M2 — HRM-L & halting (3 wks)**
Sudoku/Maze from bytes. L with k≤4; **MLP halter + ACT** (λₒ→0.01). Optional L micro-experts. Exit: Sudoku-easy ≥95%; median outer steps ≤1.6.

**M3 — Full HRM L/M/G + FiLM (4 wks)**
Add M (width **dₘ=1.5·d**) and **per-cluster G (2·d)**; **FiLM broadcast**. **Anneal H-Net targets to 32/512/8 192**; **freeze Level-2** first epoch. ARC subset, killer Sudoku, logic grids. Exit: ARC dev ≥80%; ablating FiLM hurts ≥15% rel.

**M4 — HRM MoE (heterogeneous sizes + shared fixed HRM) (4 wks)**
Enable per-cluster HRM banks (§4.1) with **heterogeneous sizes** and **shared fixed paths**.

* **Cluster routing:** always **top-1** HRM cluster per step (no 1→2 schedule). Monitor **cluster entropy** and apply aux load-balance (α≈1e-2) + z-loss (1e-3).
* **Expert routing:** **top-1** per bank; capacity 1.25.
* **Size curricula:** enforce unlock schedules (L: 1.0×d → add 0.75×/1.5× → add 0.5×; M: 1.5×d → add 1.0×d → add 2.0×d).
* **Compute prior κ:** start **0.20 (L)** / **0.10 (M)**; adjust ±0.05 if tier imbalance persists >5k steps.
* **Shared fixed HRM gates:** init `w_fix{L,M}_c ≈ 0.4` (via `a_c=0.4,b_c=0`); **anneal to 0** by phase end unless repurposing is triggered.
* **EC fallback:** tier-level Expert-Choice if overflow>5% or dead>10% persists >1k steps.

**Exit criteria:**

* No regression vs M3; **routing entropy ≥0.6**.
* Tier utilization near targets (L: small/med ≈75%, large ≤10%; M: med/large dominate, very-large ≤10%).
* **Fixed share <20%** over a sustained window; setting all fixed gates to 0 causes **≤0.5% abs** drop (else switch to repurpose: reverse-KL + orthogonality; adapters/LN gains only).

**M5 — FFN transplant + healing LM (6 wks)**
Enable **family=Qwen** bank with **MoR** (prior+query) and **one→two** cluster schedule.

* **A (adapter-align 5–10% tokens):** 1 cluster, 1 expert + fixed; train A\_in/A\_out + routers; donor FFN+LN frozen; hit **L\_align < 0.1**.
* **B (main LM + 20% puzzle replay):** 2 clusters; unfreeze donor LN gains if needed; LBL/z-loss.
* **C (steady-state):** anneal **FFN fixed** to 0; if drop >0.5% abs, keep fixed and **reverse-KL + orthogonality**.
  Exit: PPL ≤ same-size dense baseline; ARC/Sudoku ≤3% abs drop vs M3; healthy routed>fixed share.

**M6 — Continual learning (4 wks)**
Add experts to under-served clusters on new domains; freeze experts with routing entropy <0.4; 5% replay. Exit: after 3 domains, forgetting <5% abs.

**M7 — Stretch: hybrid family & scale-out**
Add **Mamba family** as second bank (same routing; its own fixed). Multi-GPU: map **families/clusters ↔ nodes**; hierarchical all-to-all.

*Ablations baked in*: prior-only vs query-only vs MoR; FiLM vs gate+bias; hetero vs homo HRM sizes; expert value cache on/off; L micro-experts on/off.

---

## 10) Optimizer & schedules

* **AdamW** β=(0.9, 0.95), ε=1e-8; weight decay 0.1; grad clip 1.0; **bf16**.
* **LR:** cosine; 5% warm-up; separate small LR for A\_in/A\_out initially (e.g., ×0.5 base).
* **Routers:** τ 1.2→1.0 (10k steps), noise std 1.0→0.2; **Switch-LBL** α=0.02→0.01→0.005; **z-loss** 1e-3; capacity 1.25.
* **Halting:** MLP widen×4; **λₒ target 0.01**; cosine veto auto-enable if outer\_cap>8.
* **Adapters alignment:** loss weight 0.01; unfreeze donor LN gains if `L_align` stalls > 1 epoch.
* **Convergence reg / one-step gradient:** OFF by default; enable when outer caps > 4.

---

## 11) Monitoring & guardrails (always-on metrics)

* **Chunker:** bits/chunk means & histograms; ratio and boundary entropy; divergence alerts if v0/v1/v2 means drift >15% week-over-week.
* **Halting:** outer-step histograms; ‖ΔhG‖₂ distributions; Δ-loss(n±1).
* **Routers:** entropy, overflow/dead rates; tier utilization (vs targets); **MoR α(hG')** distribution.
* **Fixed share:** routed vs fixed contribution over time; trigger removal or repurpose policy.
* **Adapters:** `L_align`; donor LN stats drift.
* **Cache:** hit-rate and approximation error (<3% rel L2) vs true pooled expert output.
* **Throughput:** p95 latency and active-params/token vs budget.

---

## 12) Tests by milestone (targets only; see CI harness for exact asserts)

**M0** Infra: reverse-lattice round-trip (fuzz 1k); ACT extremes (0/1 continue); router k=1 stable; **no graph breaks** under compile.
**M1** Chunker: copy exact ≥99.9%; bits/chunk within bands; block-sparse attention ≥1.2× speedup.
**M2** HRM-L: Sudoku-easy ≥95%; median outer ≤1.6; Δ-loss(n±1) sign correct ≥70%.
**M3** L/M/G + FiLM: ARC ≥80%; FiLM ablation −15% rel; Level-2 freeze → lower prior variance.
**M4** HRM MoE: tier utilization near targets; EC fallback restores overflow<1%; hetero>homo at equal FLOPs by ≥2% on ARC hard; outer steps ↓ by ≥0.2 median.
* **Weight-tying:** `FixedL_shared`/`FixedM_shared` parameter hashes identical regardless of cluster; only per-cluster gates differ.
* **Gate anneal:** mean(`w_fixL_c`), mean(`w_fixM_c`) **≤0.2** by M4 end unless `repurpose_fixed=true`.
* **Removal safety:** with all fixed gates set to 0, ARC/Sudoku/logic grids change **≤0.5% abs**; else auto-enable repurpose path and assert reverse-KL + orthogonality active next run.
* **Compile invariants:** toggling fixed gates doesn’t change shapes; **no graph breaks** under `torch.compile(dynamic=False)`.
**M5** FFN transplant: gate-comp recon error ≤5%; **L\_align<0.1**; MoR > prior-only/query-only by ≥2% PPL rel; removing fixed ≤0.5% abs drop (else repurpose).
**M6** Continual: new domain +expert → ≥5% rel gain in 100k tokens; forgetting ≤5% abs; router entropy healthy.
**M7** Hybrid & scale-out: 2-node speedup ≥1.6×; DDP compile intact.

---

## 13) Immediate engineering tasks (tickets)

1. **Router API**: family→cluster→expert with **MoR** at cluster level; HRM cluster router with compute prior κ; tier EC fallback.
2. **H-Net**: boundary heads; soft pooling; **reverse lattice**; **target-bits curriculum** and **ratio/entropy** regularizers; block-sparse masks.
3. **HRM loops**: exact order & residual expert wrapping; **FiLM broadcast** (grouped option).
4. **HRM experts**: hetero sizes; per-expert adapters (L: d↔d\_e; M: dₘ↔d\_e); fixed HRM paths; curricula/controllers.
5. **FFN transplant kit**: calibration stats; **gate-comp** builder; fixed FFN constructor; **shared family adapters**; donor LN copy; alignment loss.
6. **Halter**: 2-layer MLP; ACT controller; cosine veto hook; convergence reg & one-step gradient toggles.
7. **Expert value cache**: per-cluster low-rank surrogate with EMA updates.
8. **Compile harness**: static k=1; per-tier buffers; phase boundary re-compile for 1→2 cluster schedule.

---

## 14) Example config (small; excerpt)

```yaml
model:
  d_model: 896
  hrm_clusters: 4
  ffn_families: ["qwen"]        # later ["qwen","mamba"]

hnet:
  targets_bits: { v0: 64->32, v1: 1024->512, v2: 16384->8192 }
  zero_band: 0.20               # during anneal, then 0.15
  ratio_loss: 0.01
  boundary_entropy: 0.01
  attn_neighbors: { v0: 16, v1: 2 }

hrm:
  L: { width: d, iter_cap: 4, micro_experts: false }
  M: { width: 1.5*d, iter_cap: 2 }
  G: { width: 2.0*d }

  broadcast:
    film_groups: 8              # per-group FiLM (8 or 16)

  halting:
    type: mlp
    mlp_widen: 4
    step_penalty_target: 0.01
    cosine_safety: { enable_if_outer_cap: 8, epsilon: 0.05, gamma: 5.0 }
    one_step_gradient: false    # enable when outer_cap>4

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
    schedule_clusters: { warmup_tokens_frac: 0.10, start: 1, after: 2 }
    switch_lbl_alpha: [0.02,0.01,0.005]
    z_loss: 1.0e-3

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
```

---

## 15) Token & compute budgets (reminder)

* **Q0.5B:** active ≈ **300–380 M** params/token (with hetero HRM); **\~4–7 B tokens** in ≲2 weeks on a 7900 XTX with checkpointing.
* **Q1.5B:** active ≈ **820–900 M** params/token; **\~2–4 B tokens** similar walltime.
  Feasible because **FFN knowledge is transplanted**, not learned from scratch.

---

# Appendix A — Project layout (**preliminary**, subject to refinement)

```
moirai/
  __init__.py

  hnet/
    __init__.py
    embed.py
    boundary_head.py
    soft_pool.py
    attention.py
    chunker.py              # Orchestrates L0/L1/L2, target-bits curriculum, reverse lattice

  hrm/
    __init__.py
    gru_cells.py            # GRU_L / GRU_M / GRU_G (+ Jacobian clamp hook)
    lm_heads.py             # ByteHead; FiLM (per-channel or grouped)
    halting.py              # MLP halter (ACT), cosine veto, one-step-grad switch
    loops.py                # Exact L/M/G step (residual expert wrapping; order of ops)
    micro_experts.py        # Optional L inner-loop MoE cell
    fixed_shared.py         # FixedL_shared / FixedM_shared (shared weights, per-cluster gates)

  moe/
    __init__.py
    routing.py              # Static top-k routers; MoR mixer; EC fallback
    hrm_experts.py          # L/M expert banks (hetero sizes); per-expert adapters (L: d↔d_e, M: d_m↔d_e)
    ffn_family.py           # Family registry (Qwen, Mamba); shared A_in/A_out; donor LN handling
    ffn_clusters.py         # Calibration stats; clustering; expert carving; gate-compensation
    ffn_retrieve.py         # Family→cluster→expert retrieval; pooling; expert value cache

  transplant/
    __init__.py
    calib_dataset.py
    stats.py
    gate_comp.py
    fixed_builder.py
    layer_select.py

  train/
    __init__.py
    dataloader.py           # Byte I/O; ARC 4-bit pack/unpack
    curriculum.py           # H-Net targets & HRM size unlock schedules
    optimizer.py            # AdamW groups (adapters, routers, cores)
    trainer.py              # Phase runner; compile & re-compile at 1→2 cluster switch
    metrics.py              # bits/chunk, L_align, router entropy/utilization, halting histograms
    checkpoints.py

  configs/
    moirai_q05b.yaml
    moirai_q15b.yaml
    thresholds.yaml

  tests/
    test_m0_infra.py
    test_m1_chunker.py
    test_m2_hrm_l.py
    test_m3_full_hrm.py
    test_m4_moe_hrm.py
    test_m5_ffn_transplant.py
    test_m6_continual.py

  cli/
    run_phase.py
    eval_bench.py

  viz/
    dashboards.py
```
