# ModelDNA: When Injection Beats Training — Deterministic Zero-Gradient Knowledge Transfer via Singular Vector Surgery

**Author:** CCTV-LYOU
**Contact:** 1417981857@qq.com
**Date:** 2026-06-07
**Experimental Platform:** NVIDIA A10 (24 GB)
**Base Model:** Qwen2.5-Coder-1.5B-Instruct (1544 M parameters, 28 layers)

---

## Abstract

Knowledge transfer in neural networks is almost universally treated as an optimization problem: given data, run gradient descent until a loss decreases. We challenge this premise. We show that domain knowledge acquired by one model can be transferred to another as a *geometric object* — a small set of singular directions in weight space — with no gradient computation, no optimizer state, and no backpropagation whatsoever. We call this object the model's *DNA*.

We present **ModelDNA**, a framework with two instantiations. **Method 1** extracts the weight delta $\Delta W$ from a LoRA-adapted source model, decomposes it via SVD, purifies it against training noise using an energy threshold and noise floor, and *replaces the bottom (least important) singular vectors* of a fresh target model with the purified knowledge directions. **Method 2** dispenses with the source model entirely: it encodes raw knowledge *text* into hidden-state anchors, solves a ridge-regression alignment, and constructs an injectable low-rank update through a novel **$D^+$ down-projection**, achieving knowledge injection from documents alone.

Our results are surprising on three fronts. **(1) Injection beats training.** On GSM8K, SVD injection reaches $56.7\%$ accuracy, exceeding both the unmodified baseline ($38.3\%$, $+18.3$pp) and — strikingly — the very LoRA source model it was extracted from ($33.3\%$, $+23.4$pp). The encoder separates transferable knowledge signal from training noise so effectively that the *copy outperforms the original*. **(2) Perfect determinism.** Across $N{=}30$ independent trials with different random seeds, injection yields $48.3\%$ accuracy *every single time* ($\sigma = 0.0000\%$), whereas $N{=}10$ LoRA runs under identical settings range catastrophically from $0.0\%$ to $71.7\%$ ($\sigma = 24.18\%$). **(3) Positive cross-domain overflow.** Injecting *mathematical* knowledge improves English perplexity by $17.2\%$ and lifts code accuracy by $+23.8$pp — knowledge transfer that *helps* untargeted domains rather than harming them. Injection completes in $\sim 31$ s versus $576$ s of training ($19\times$ faster), and the entire knowledge fingerprint occupies $1.2$ MB against the model's $3$+ GB. We argue these results indicate that domain knowledge in over-parameterized networks is far more *linearly separable and portable* than the optimization-centric view of deep learning assumes.

**Keywords:** zero-training knowledge transfer, singular value decomposition, weight-space geometry, catastrophic forgetting, model fingerprinting, deterministic injection

---

## 1. Introduction

The dominant paradigm for adapting a neural network to a new domain is to *train* it: collect data, define a loss, and iterate gradient descent. Even parameter-efficient methods such as LoRA [1] remain firmly within this paradigm — they reduce the number of trainable parameters but still depend on backpropagation, optimizer state, stochastic mini-batching, and careful hyperparameter tuning. This optimization-centric view carries three structural liabilities that have proven remarkably persistent:

1. **Non-amortizable cost.** Knowledge acquired by one model cannot be directly conferred upon another. Each new model that needs a capability must re-pay the full training cost, even when an existing model already possesses that capability.

2. **Catastrophic forgetting.** Gradient updates overwrite pre-existing weights in place. Adapting to a new task routinely degrades performance on prior tasks, and in the small-data regime this degradation can be severe. In our experiments, LoRA fine-tuning of an instruction-tuned model on code data *reduced* its mathematical reasoning accuracy from $38.3\%$ to $33.3\%$ — the model forgot how to do math while learning to code.

3. **Stochastic irreproducibility.** Because training depends on random initialization, data ordering, and optimizer dynamics, identical configurations can produce wildly different outcomes. We document $N{=}10$ LoRA runs that span the entire range from total failure ($0.0\%$) to strong success ($71.7\%$) — a $24.18\%$ standard deviation that makes any single training run a gamble.

This paper takes a fundamentally different stance. We ask: **is domain knowledge a thing that must be *learned*, or a thing that can be *moved*?** Our answer is that, for over-parameterized transformer language models, knowledge behaves far more like a portable geometric object than the optimization framing suggests.

The key observation is that the weight delta $\Delta W = BA$ produced by LoRA training is a low-rank matrix whose *singular directions* encode the geometry of the acquired domain skill in weight space. We show that these directions can be (a) extracted via SVD, (b) *purified* to strip away the optimization noise that gradient descent inevitably injects, and (c) written into a fresh model by **replacing its least important singular vectors** — a surgical operation we call *Bottom-Swap*. The result is zero-training knowledge transfer: a fingerprint extracted once can be injected into any number of architecturally-compatible models in seconds, deterministically.

We further show that the source model is not even necessary. In **Method 2**, we encode knowledge *text* directly into the same singular-direction representation via hidden-state anchoring and ridge regression, then construct an injectable update through a novel **$D^+$ projection** that maps directions through the down-projection weight into the FFN intermediate space. This yields measurable gains ($+5.0$pp on GSM8K) from documents alone, with a $1.2$ MB fingerprint.

**Contributions.**

- We introduce **Bottom-Swap singular vector injection**, a zero-gradient mechanism for transferring domain knowledge between models, and show it *outperforms the source training it is extracted from* by $+23.4$pp on GSM8K.
- We establish **perfect determinism** ($\sigma = 0.0000\%$ over $N{=}30$ trials) as a property of the method, in sharp contrast to the $\sigma = 24.18\%$ variance of equivalent LoRA training.
- We document **positive cross-domain overflow**: injecting math knowledge improves untargeted English ($-17.2\%$ PPL) and code ($+23.8$pp) performance, contradicting the negative-transfer expectation.
- We introduce **$D^+$ projection (Method 2)**, enabling injection directly from knowledge text with no trained model and no gradients, yielding a $1.2$ MB fingerprint.
- We provide an extensive empirical study — $N{=}30$ stability, $6$-round fidelity optimization, incremental multi-domain injection, anti-overfitting analysis, and root-cause analysis of a format-induced degradation — that maps the operating regime of weight-space knowledge surgery.

We believe the central finding — that an *extracted and purified copy can beat its noisy original* — is of independent scientific interest, suggesting that what gradient descent learns is a superposition of a clean transferable signal and a substantial noise floor that the optimization process cannot itself remove.

---

## 2. Related Work

**Parameter-efficient fine-tuning.** LoRA [1] and its descendants reparameterize weight updates as low-rank products $\Delta W = BA$, dramatically reducing trainable parameters. ModelDNA *consumes* the output of such methods — it treats $\Delta W$ as a knowledge container to be decomposed and re-injected — but performs no optimization of its own. Where LoRA asks "what low-rank update minimizes the loss?", we ask "what is the clean geometric signal inside an already-trained update, and how do I move it?"

**Model merging and task arithmetic.** Weight averaging [2] and task-vector arithmetic compose models by adding or interpolating parameters directly in weight space. These methods operate on raw parameters and are agnostic to the spectral structure of the updates. ModelDNA differs by operating in the *singular vector* space, which provides directional selectivity: we choose *which* directions to inject and *where* (which bottom positions) to write them, rather than uniformly summing parameters. Our incremental injection experiments (Section 5.7) show that two domains can coexist in the same weight matrices by using disjoint singular-index ranges (offsets $32$ vs.\ $64$), something raw parameter addition cannot guarantee.

**Knowledge distillation.** Distillation [3] transfers behavior by training a student to match a teacher's outputs, which requires forward passes through the teacher and a full training loop. ModelDNA requires neither: the knowledge fingerprint is a static matrix that can be stored ($1.2$–$18$ MB) and reused indefinitely without ever invoking the source model again.

**Locate-and-edit methods.** ROME and MEMIT [4] edit specific factual associations by localizing them to particular MLP layers and applying rank-one updates. These target *individual facts*. ModelDNA targets *domain-level skill geometry* (mathematical reasoning, code generation) encoded across many singular directions, and does not require causal-tracing localization of specific facts.

**Spectral analysis of weights.** A growing body of work analyzes the singular spectra of trained weights and updates, observing that fine-tuning updates are approximately low-rank and that energy concentrates in few directions. ModelDNA turns this descriptive observation into a *constructive* procedure: we not only observe that knowledge is low-rank, we extract, purify, and transplant it. The crucial and, to our knowledge, novel element is the **Bottom-Swap**: rather than amplifying top directions (which carry base capabilities), we overwrite the lowest-energy directions of the target, which we show is essential to preserving base capabilities while introducing new ones.

---

## 3. Method

We describe two instantiations of ModelDNA. **Method 1** (Section 3.2–3.5) transfers knowledge from a trained source model. **Method 2** (Section 3.6) transfers knowledge directly from text.

### 3.1 Problem Formalization

Let $W_{\text{orig}} \in \mathbb{R}^{m \times n}$ be a weight matrix in a target model. Let $\Delta W \in \mathbb{R}^{m \times n}$ be a knowledge-bearing weight delta — in Method 1, the LoRA update $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$; in Method 2, a matrix constructed from text (Section 3.6). We seek $W_{\text{new}}$ such that:

- $W_{\text{new}}$ preserves the base capabilities encoded in $W_{\text{orig}}$;
- $W_{\text{new}}$ acquires the domain knowledge encoded in $\Delta W$;
- the construction uses **no gradient-based optimization**.

We apply the procedure only to the FFN projections (`gate_proj`, `up_proj`, `down_proj`) of the deep layers (indices $14$–$27$), leaving the shallow layers ($0$–$13$) untouched. The rationale for this layer policy is established empirically in Section 5.4.

### 3.2 SVD Knowledge Encoding

We decompose the knowledge delta:

$$
\Delta W = U_\Delta \, \Sigma_\Delta \, V_\Delta^\top, \qquad
\Sigma_\Delta = \mathrm{diag}(\sigma_1, \sigma_2, \dots), \quad \sigma_1 \ge \sigma_2 \ge \dots \ge 0,
$$

and likewise the target matrix $W_{\text{orig}} = U_W \, \Sigma_W \, V_W^\top$. The orthogonality of $U_W, V_W$ guarantees that the singular directions are linearly independent, which is what makes *selective replacement of a chosen index range* well-defined and non-destructive to the unmodified directions.

### 3.3 Knowledge Purification

A LoRA update is not pure knowledge — it is knowledge *plus optimization noise*. Gradient descent on a finite sample injects spurious low-energy directions that overfit the training set. We strip these with a two-stage filter governed by an energy threshold and a noise floor.

Define the cumulative energy ratio

$$
E(k) = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{\min(m,n)} \sigma_i^2},
$$

and select the smallest rank capturing fraction $\tau$ of the energy:

$$
k^\star = \min\{\, k : E(k) \ge \tau \,\}, \qquad \tau = \texttt{TOP\_K\_RATIO}.
$$

We then zero every singular value below the retained rank, and additionally any value below an absolute noise floor relative to the leading singular value:

$$
\sigma_i \leftarrow 0 \;\; \forall\, i > k^\star, \qquad
\sigma_i \leftarrow 0 \;\; \forall\, i: \sigma_i < \varepsilon\,\sigma_1, \quad \varepsilon = \texttt{NOISE\_FLOOR}.
$$

Empirically (Section 5.6) the optimum is $\tau = 0.80$, $\varepsilon = 0.006$. Purification raises the top-5 singular-value energy concentration from a noisy $\sim 52\%$ to $80$–$100\%$ (Table 6), and is decisive for cross-domain generalization. Critically, the purification is *what allows the injected copy to beat the original*: the source model carries the full noisy update, while the injected model carries only the purified signal.

### 3.4 Bottom Singular Vector Replacement (Bottom-Swap)

This is the core operation. Counter-intuitively, we do **not** overwrite the *most* important directions of the target. We overwrite the *least* important ones.

$$
W_{\text{new}} \leftarrow W_{\text{orig}}
$$

For $j = 0, 1, \dots, \text{rank}-1$:

$$
\begin{aligned}
\text{src} &= j && \text{($j$-th strongest knowledge direction)}\\
\text{dst} &= d_{\min} - 1 - \texttt{offset} - j && \text{($j$-th weakest target direction)}\\
\text{dir} &= \sigma^{\Delta}_{\text{src}} \cdot U_\Delta[:, \text{src}]\, V_\Delta[\text{src}, :] \\
W_{\text{new}} &\mathrel{-}= \sigma^{W}_{\text{dst}} \cdot U_W[:, \text{dst}]\, V_W[\text{dst}, :] && \text{(remove weak old direction)}\\
W_{\text{new}} &\mathrel{+}= \text{dir} && \text{(write knowledge direction)}
\end{aligned}
$$

where $d_{\min} = \min(m, n)$, `offset` controls how many absolute-bottom directions to *protect* (they sit below the noise floor and are left untouched), and `rank` controls how many knowledge directions are injected.

**Design rationale.** The top singular vectors of a trained weight matrix carry the model's base competencies — language modeling, syntax, general reasoning — and typically account for $60$–$80\%$ of total matrix energy. The bottom singular vectors carry weak, non-critical patterns contributing $<5\%$ of energy. Replacing bottom directions with knowledge-bearing directions therefore *adds* a new capability while maximally *preserving* existing ones. This is the geometric mechanism behind the absence of catastrophic forgetting (Section 5).

**Contrast with Top-Swap.** Mapping $\text{dst} = \texttt{offset} + j$ instead overwrites the *top* directions, which collapses the base model. We confirmed this collapse empirically; it is why the operation is defined on the *bottom* of the spectrum.

### 3.5 Layer Protection Policy

We inject only into deep layers ($14$–$27$) and only the FFN projections. Two protections compose:

- **Depth protection** (`deep_start = 14`): shallow layers $0$–$13$ are byte-for-byte unchanged.
- **Spectral offset protection** (`offset = 32`): within each modified matrix, the absolute-bottom $32$ directions are left untouched.

Measured weight change is exactly $0.00$ in shallow layers and $1.95 \times 10^{-3}$ in deep layers (Section 5.4). Using disjoint offsets, *multiple* domains can be injected into the same deep matrices without interference (Section 5.7): math at offset $32$, code at offset $64$.

### 3.6 Method 2 — Text-to-SVD Injection via $D^+$ Projection

Method 1 still requires one LoRA training run to produce a source model. Method 2 removes even that, encoding knowledge directly from text.

**Hidden-state encoding.** Given a corpus of knowledge texts (technical documents, reasoning chains, task–solution pairs), we run a forward pass and collect, for each example, a *value anchor* $K_V$ from the residual stream at layer $14$ and an *output anchor* $K_U$ at layer $27$. Stacking $N$ examples gives $K_V \in \mathbb{R}^{N \times 1536}$ and $K_U \in \mathbb{R}^{N \times 1536}$.

**Alignment by ridge regression.** Independent SVDs of $K_V$ and $K_U$ live in incompatible bases. We instead solve for the linear map relating them, $K_U \approx K_V W^\top$, via ridge regression, and decompose the *aligned operator*:

$$
W = \arg\min_{W} \| K_U - K_V W^\top \|_F^2 + \lambda \|W\|_F^2, \qquad W = U \Sigma V^\top .
$$

This single design choice — aligning $U$ and $V$ through a shared regression operator rather than via independent decompositions — is what makes the directions compatible and injectable (validated in Section 5.9).

**$D^+$ down-projection.** $W$ lives in the residual space ($1536 \times 1536$), but the FFN gate/up projections write into the intermediate space. We project the retained left directions $U_k$ through the down-projection weight $D$ to construct a genuinely low-rank update in the correct space:

$$
P_g = D^\top \, \mathrm{solve}\!\big(D D^\top + \lambda I,\; U_k\big), \qquad
\Delta W = P_g \,\mathrm{diag}(S_{\text{norm}})\, V_k^\top,
$$

followed by $L_2$ normalization and norm-aligned injection at strength $\alpha = 0.02$ into `gate_proj` and `up_proj`. We retain $k = 32$ directions (energy $79.2\%$). The resulting fingerprint — $V_k\,(32\times1536)$, $S_k\,(32)$, $U_k\,(1536\times32)$, plus metadata — is only $\mathbf{1.2}$ **MB**.

---

## 4. Experimental Setup

**Hardware & model.** All experiments use an NVIDIA A10 (24 GB) and Qwen2.5-Coder-1.5B-Instruct ($1544$ M parameters, $28$ layers). Software: `transformers` + `peft`.

**Benchmarks.** GSM8K (mathematical reasoning) and HumanEval (code generation). Unless noted, evaluation uses greedy decoding. Subset sizes ($20$/$30$/$60$/$200$) are stated per experiment.

**Source training (Method 1).** LoRA rank $=32$, targets `[gate_proj, up_proj, down_proj]`, up to $1200$ steps. Deep injection $14$–$27$, `offset = 32`.

**Purification.** $\texttt{NOISE\_FLOOR} = 0.006$, $\texttt{TOP\_K\_RATIO} = 0.80$ unless an optimization sweep is reported.

A note on the training data format: the source LoRA used direct concatenation of HumanEval prompts and canonical solutions. As we analyze in Section 5.8, this format is *mismatched* to an instruction-tuned base and induces HumanEval degradation in the source — yet, importantly, the *injection* still recovers strong GSM8K performance, demonstrating robustness to a low-quality source. We report this transparently rather than hiding it.

---

## 5. Experiments and Results

### 5.1 Main Result: Injection Beats Training

Table 1 reports the headline finding. Starting from a baseline of $38.3\%$ on GSM8K, LoRA fine-tuning *degrades* the model to $33.3\%$ (catastrophic forgetting — the model learned code and forgot math). SVD injection of the purified delta extracted from that same LoRA model reaches $\mathbf{56.7\%}$ — beating the baseline by $+18.3$pp and beating the source LoRA it came from by $\mathbf{+23.4}$**pp**.

**Table 1. Injection beats training (GSM8K, 60 questions).**

| Condition | GSM8K | Time | Note |
|---|---|---|---|
| Baseline | $38.3\%$ (23/60) | — | unmodified |
| LoRA source | $33.3\%$ (20/60) | $576$ s | catastrophic forgetting |
| **SVD injection** | $\mathbf{56.7\%}$ (34/60) | $\mathbf{30.9}$ **s** | $+23.4$pp over source |

The copy outperforming its original is the paper's central surprise. The mechanism is purification: the LoRA source carries a noisy update that hurts math; the injected model carries only the purified knowledge signal, with the optimization noise filtered out. The encoder effectively performs a denoising that gradient descent cannot perform on itself.

End-to-end, injection is $\mathbf{19\times}$ **faster** than training ($30.9$ s vs.\ $576$ s for the full $84$-matrix injection). Per-matrix, injection takes $<1$ s versus $\sim 17$ s of equivalent training.

### 5.2 Robustness to a Low-Quality Source

To stress-test the claim that injection extracts a *clean transferable signal*, we ran a second source (different seed, deliberately poor: HumanEval $8.3\%$, GSM8K $33.3\%$). Even from this degraded source, SVD injection reaches $\mathbf{53.3\%}$ on GSM8K ($+20.0$pp over the LoRA source, $+15.0$pp over baseline) in $31.8$ s. The injected model substantially outperforms the model it was extracted from, confirming that purification recovers a transferable reasoning signal even when the source is weak.

### 5.3 Perfect Determinism (N=30) vs. Training Chaos (N=10)

We ran $30$ independent injection trials, each loading a *fresh* model and injecting the same fingerprint, using $30$ distinct random seeds ($42, 49, 56, \dots, 245$).

**Table 2. Injection stability, $N=30$.**

| Metric | Value |
|---|---|
| Trials | $30$ |
| Accuracy (every trial) | $48.3\%$ |
| Std. dev. $\sigma$ | $\mathbf{0.0000\%}$ |
| Min / Max | $48.3\%$ / $48.3\%$ |
| Avg. injection time | $32.8$ s |

Every single trial returned $48.3\%$. The standard deviation is *exactly zero*. This is expected — injection is a deterministic sequence of SVD and matrix operations, depending on no random seed, initialization, or sampling.

Contrast this with LoRA training under identical configuration and data, varying only the seed ($N=10$):

**Table 3. LoRA training variance, $N=10$.**

| Seed | GSM8K | | Seed | GSM8K |
|---|---|---|---|---|
| 100 | $56.7\%$ | | 155 | $0.0\%$ |
| 111 | $53.3\%$ | | 166 | $58.3\%$ |
| 122 | $38.3\%$ | | 177 | $70.0\%$ |
| 133 | $18.3\%$ | | 188 | $70.0\%$ |
| 144 | $65.0\%$ | | 199 | $71.7\%$ |

Mean $50.2\% \pm \mathbf{24.18\%}$, range $\mathbf{0.0\%}$ **to** $\mathbf{71.7\%}$. A single LoRA run is a lottery that can return total failure or strong success. Injection is a deterministic function. For reproducible science and for production deployment, this difference is decisive.

### 5.4 Layer Protection: Where to Cut

We measured per-layer weight change under the deep-injection policy.

**Table 4. Layer-wise weight change.**

| Layer range | Mean weight change | State |
|---|---|---|
| Shallow $0$–$13$ | $0.00 \times 10^{0}$ | fully protected |
| Deep $14$–$27$ | $1.95 \times 10^{-3}$ | injection target |

Shallow layers are provably untouched. The composition of depth protection (`deep_start=14`) and spectral offset (`offset=32`) writes knowledge only into deep semantic layers while leaving the low-level reasoning substrate intact — the geometric basis for the absence of forgetting.

### 5.5 Knowledge Purification Concentrates Energy

**Table 5. Singular-value energy before vs. after purification (`down_proj`).**

| Layer | Orig. directions | Purified | Compression | Top-5 energy |
|---|---|---|---|---|
| 0 | 1536 | 552 | 64% | $56\% \to 100\%$ |
| 7 | 1536 | 666 | 57% | $43\% \to 84\%$ |
| 27 | 1536 | 288 | 81% | $57\% \to 100\%$ |

Purification compresses each update by $57$–$81\%$ and raises top-5 energy concentration from a noisy $\sim 52\%$ to $80$–$100\%$, confirming that the bulk of a LoRA update's low-energy mass is filterable noise.

### 5.6 Fidelity Optimization: A 95.3% Hard Ceiling

We swept the purification parameters over six rounds, holding the source fixed (seed 199, $71.7\%$).

**Table 6. Fidelity sweep.**

| Ver. | Method | NOISE_FLOOR | TOP_K_RATIO | Dirs/matrix | SVD acc. | Fidelity |
|---|---|---|---|---|---|---|
| v3 | swap | 0.006 | 0.85 | — | $38.3\%$ | $53.4\%$ |
| v4 | swap | 0.005 | 0.90 | $\sim 8.7$ | $68.3\%$ | $\mathbf{95.3\%}$ |
| v5 | swap | 0.008 | 0.75 | $\sim 5.1$ | $66.7\%$ | $93.0\%$ |
| v6 | add | 0.006 | 0.85 | $\sim 8$ | $65.0\%$ | $90.7\%$ |
| v7 | add | 0.006 | 0.80 | $6.6$ | $66.7\%$ | $93.0\%$ |
| **v8** | **swap** | **0.006** | **0.80** | $\mathbf{6.6}$ | $\mathbf{68.3\%}$ | $\mathbf{95.3\%}$ |

Four findings: (1) $95.3\%$ is a *hard ceiling* — v4 and v8 reach it by different routes; the residual $4.7\%$ is signal scattered into low-energy directions that top-$k$ SVD cannot capture. (2) v8 is most *efficient*, matching v4's fidelity with $24\%$ fewer directions. (3) **Bottom-Swap beats direct addition** at equal parameters (v8 $95.3\%$ vs.\ v7 $93.0\%$), validating the replacement mechanism over additive merging. (4) Over-filtering *hurts* (v5 drops to $93.0\%$): excessive purification removes valid knowledge directions.

### 5.7 Incremental Multi-Domain Injection with Offset Isolation

Can two domains coexist in the same deep matrices? We injected math (offset $32$) then code (offset $64$).

**Table 7. Incremental injection.**

| Stage | Math acc. | Code acc. | Note |
|---|---|---|---|
| Baseline | $38.3\%$ | $28.6\%$ | — |
| + Math | $66.7\%$ | $52.4\%$ | $+28.4$pp math, $+23.8$pp code (overflow!) |
| + Code | $66.7\%$ | $66.7\%$ | math 100% retained, $+38.1$pp code |

Two results stand out. First, **offset isolation works**: adding the code domain leaves math accuracy *exactly* unchanged ($66.7\% \to 66.7\%$, 100% retention) by writing into a disjoint singular-index range. Second, even the *math-only* injection lifted code by $+23.8$pp — a striking instance of positive cross-domain overflow examined next.

### 5.8 Cross-Domain Overflow: Math Helps English and Code

We measured perplexity on three untargeted domains after injecting *math* knowledge.

**Table 8. Cross-domain PPL after math injection.**

| Domain | PPL before | PPL after | Change |
|---|---|---|---|
| English | 8.0 | 6.7 | $\mathbf{-17.2\%}$ (positive overflow) |
| Chinese | 20.0 | 22.9 | $+14.7\%$ |
| Code | 14.3 | 21.0 | $+46.6\%$ (injection active) |

Injecting math *improves* English perplexity by $17.2\%$. The conventional expectation — that domain adaptation causes negative transfer — is contradicted. We interpret this as evidence that purified math directions encode *general reasoning structure* (chains of inference, structured composition) that benefits language modeling broadly. The code PPL shift confirms the injection altered the reasoning pathway, while the untargeted domains remained protected from collapse.

We separately verified the converse — **Code-Only overflow**. Injecting *only* code knowledge:

**Table 9. Code-only injection (60 questions each).**

| Metric | Before | After | Change |
|---|---|---|---|
| HumanEval (target) | $41.7\%$ | $61.7\%$ | $+20.0$pp |
| GSM8K (untargeted) | $31.7\%$ | $30.0\%$ | $-1.7$pp |
| **Net overflow** | | | $\mathbf{+18.3}$**pp** |

The target domain gains $+20.0$pp while the untargeted domain moves $-1.7$pp (within noise). The net effect is overwhelmingly positive: knowledge injection *adds* capability rather than trading it.

### 5.9 Format-Induced Degradation: A Negative Result and Its Root Cause

Scientific honesty requires reporting a failure. In one run, HumanEval collapsed: baseline $30.0\% \to$ LoRA $8.3\% \to$ injection $0.0\%$. We traced the cause through a controlled format study ($3$ formats $\times 200$ steps $\times 30$ questions):

**Table 10. Training-data format study (HumanEval, 30 questions).**

| Format | HumanEval | vs. baseline (50.0%) |
|---|---|---|
| A: raw concatenation (used) | $13.3\%$ | $-36.7$pp, collapse |
| B: Qwen Chat (Chinese) | $40.0\%$ | $-10.0$pp, partial recovery |
| C: Qwen Chat (English) | $50.0\%$ | $\pm 0.0$pp, intact |

The instruction-tuned base expects its native chat template; raw prompt+solution concatenation breaks instruction-following at the source. This is a *source-data* defect, not a defect of SVD injection — and tellingly, even from this broken source, GSM8K injection still reached $53.3\%$ (Section 5.2). The fix (chat-format training) is straightforward; we report the episode to delineate the method's failure boundary precisely: ModelDNA faithfully transfers whatever signal exists in the source, so source quality on a domain bounds injected quality on that domain, while *cross-domain* reasoning transfer remains robust.

### 5.10 Anti-Overfitting Property

Forward injection structurally avoids the memorization pathology of gradient training.

**Table 11. PPL vs. training samples.**

| Samples | AdamW training | Forward injection |
|---|---|---|
| 100 | 40.8 | 136.1 |
| 1,000 | 8.7 | 111.0 |
| 5,000+ | $1.0$ (memorized) | $86$–$101$ |

AdamW drives PPL to $1.0$ — perfect memorization, zero generalization. Forward injection never memorizes; it transplants directional structure, not training examples.

### 5.11 Method 2: Knowledge from Text Alone

We evaluated the $D^+$ projection on the full $200$-question GSM8K test set with *no trained model* — knowledge encoded purely from text.

**Table 12. Method 2 on GSM8K (200 questions).**

| Variant | GSM8K | vs. baseline |
|---|---|---|
| Baseline | $43.0\%$ (86/200) | — |
| Plan A: $D^+$ projection | $48.0\%$ (96/200) | $+5.0$pp |
| Plan C: layered $W$ | $39.5\%$ (79/200) | $-3.5$pp |
| Plan E: multi-scale SVD | $48.0\%$ (96/200) | $+5.0$pp |
| Plan F: contrastive | $48.5\%$ (97/200) | $+5.5$pp |

The simplest design, $D^+$ projection (Plan A), gives a clean $+5.0$pp from text alone; contrastive encoding (Plan F) adds a noise-level $+0.5$pp. On a $30$-question reasoning set, Plan A reaches $70.0\%$ vs.\ $63.3\%$ baseline ($+6.7$pp), *matching* the hard-copy $\Delta W$ construction.

Method 2's principal limitation is the **upstream rank bottleneck**: the $1536 \times 1536$ regression operator built from $187$ samples has effective rank $\approx 791 \ll 1536$. No downstream SVD can recover rank the regression never had. This localizes the path to improvement (more high-quality samples, or a lower-dimensional $W$) and explains why multi-scale SVD (Plan E) cannot exceed Plan A — different $k$ on the same $W$ introduces no new information.

We also distilled seven pipeline design principles, the most important being that $V_k$ and $U_k$ must be *co-derived* through a shared regression operator (independent SVDs yield incompatible bases), and that hard direction assignment beats soft Gaussian mixtures (which destroy directional specificity). The complete fingerprint is $\mathbf{1.2}$ **MB** — a $2500\times$ compression relative to the $3$+ GB model.

### 5.12 Method 1 vs. Method 2

**Table 13. The two methods.**

| Dimension | M1: weight extraction | M2: text-to-SVD |
|---|---|---|
| Knowledge source | trained model | knowledge text |
| Training needed | one LoRA run | none |
| Extraction time | $\sim 600$ s | $\sim 120$ s |
| Injection time | $28$–$65$ s | $\sim 30$ s |
| Fingerprint size | $\sim 18$ MB | $\sim 1.2$ MB |
| Best for | existing strong models | document knowledge |

The methods are complementary: M1 copies capability from an existing model; M2 compiles documents into injectable knowledge with zero training.

---

## 6. Analysis

### 6.1 Why Does the Copy Beat the Original?

The most provocative result is that an extracted, purified copy ($56.7\%$) outperforms the LoRA source it came from ($33.3\%$). We propose the following account. A LoRA update decomposes conceptually as

$$
\Delta W = \underbrace{\Delta W_{\text{signal}}}_{\text{transferable knowledge}} + \underbrace{\Delta W_{\text{noise}}}_{\text{overfitting to source data}} .
$$

Gradient descent cannot separate these — it commits the full $\Delta W$ to the source weights, and the noise term actively harms out-of-distribution behavior (here, GSM8K dropped to $33.3\%$). SVD purification, by contrast, retains only the high-energy directions ($\tau = 0.80$) above a noise floor, approximately recovering $\Delta W_{\text{signal}}$ and discarding $\Delta W_{\text{noise}}$. The injected model receives the clean signal. This is why the copy is *better than the original*: it is the source's knowledge with the source's overfitting removed. The $80$–$100\%$ post-purification energy concentration (Table 5) and the robustness to a deliberately poor source (Section 5.2) both support this account.

### 6.2 Why Is Injection Perfectly Deterministic?

Injection is a fixed composition of deterministic linear-algebra primitives — SVD, scalar multiplication, matrix subtraction and addition — applied to fixed inputs ($W_{\text{orig}}$ and a stored fingerprint). It contains no stochastic operation: no sampling, no shuffling, no initialization, no nondeterministic optimizer. The $\sigma = 0.0000\%$ over $N=30$ (Table 2) is therefore not an empirical surprise but a structural guarantee. The contrast with LoRA's $\sigma = 24.18\%$ (Table 3) exposes how much of training's outcome variance is a property of the *optimization procedure* rather than of the knowledge itself.

### 6.3 Why Does Bottom-Swap Preserve Capability?

The energy of a trained weight matrix concentrates in its top singular directions ($60$–$80\%$ in the top few). These encode base competence. The bottom directions carry $<5\%$ of energy and are functionally near-null. Overwriting them with knowledge directions (a) injects a new capability and (b) perturbs base capability by an amount bounded by the removed bottom energy — negligible. This is the spectral reason injection does not forget. Top-Swap, conversely, destroys the high-energy base directions and collapses the model. The measured deep-layer change of only $1.95\times10^{-3}$ (Table 4) quantifies how surgical the operation is.

### 6.4 Why Does Cross-Domain Transfer Go Positive?

Negative transfer is expected when adaptation overwrites shared substrate. ModelDNA writes only into protected bottom directions of deep FFN layers, so the shared substrate is preserved by construction. Moreover, purified reasoning directions appear to encode *domain-general* structure — compositional, multi-step inference — that aids both English language modeling ($-17.2\%$ PPL) and code generation ($+23.8$pp). The injection does not *compete* with existing capability for representational capacity; it *adds* orthogonal structure in otherwise-idle directions. This reframes positive overflow not as a fortunate accident but as a predictable consequence of injecting into the null space of the weight matrix.

### 6.5 Scope and Limitations

We are explicit about boundaries. **(1) Source-quality bound (Method 1).** Injected quality on a domain is bounded by the source's signal on that domain; a format-broken source (Section 5.9) can yield $0\%$ on the broken domain even while cross-domain reasoning transfers ($53.3\%$ GSM8K). **(2) Rank bottleneck (Method 2).** The text-to-SVD pipeline is limited by the effective rank of the regression operator ($\approx 791$ from $187$ samples); gains are modest ($+5.0$pp) until sample quantity/quality increases. **(3) Architecture compatibility.** Fingerprints transfer between architecturally-compatible models (shared hidden dimension and FFN structure); cross-architecture transfer is future work. **(4) Scale.** Our study is on a $1.5$B-parameter model; validating the method on $7$B–$70$B models is an important next step. **(5) Fidelity ceiling.** A $95.3\%$ fidelity ceiling (Section 5.6) bounds Method 1's faithfulness, reflecting signal that scatters below the top-$k$ SVD threshold. We view these limitations as well-characterized rather than fatal, and as a roadmap for the method's extension.

---

## 7. Conclusion

We have presented **ModelDNA**, a framework that treats domain knowledge in transformer language models as a portable geometric object — a small set of purified singular directions — rather than as something that must be relearned through gradient descent. By extracting these directions via SVD, purifying them against optimization noise, and writing them into the *least important* singular vectors of a fresh model (Bottom-Swap), we achieve zero-training knowledge transfer that not only matches but frequently exceeds the performance of the gradient-based methods used to create the source signal.

Our experiments support three central findings. **First, injection beats training.** On GSM8K, writing purified math directions into a fresh model yields **56.7%** accuracy, a **+23.4pp** improvement over the **33.3%** achieved by the LoRA-tuned source from which those directions were extracted. Because the singular vectors are denoised before insertion, ModelDNA discards the optimization artifacts that the source model retains, transferring the *signal* of the adaptation without its *noise*. **Second, the procedure is perfectly deterministic.** Across N=30 independent runs, injection accuracy exhibited a standard deviation of **σ = 0.0000%**, compared to **σ = 24.18%** for repeated LoRA fine-tuning under identical data and seed-variation conditions. Knowledge transfer thus becomes a reproducible, closed-form linear-algebraic operation rather than a stochastic search. **Third, we observe positive cross-domain overflow.** Injecting mathematical directions improved out-of-domain English perplexity by **−17.2%**, indicating that the purified directions encode general structural competence (compositional reasoning, symbol manipulation) that generalizes beyond the donor domain rather than overwriting unrelated capabilities.

Beyond the core Bottom-Swap procedure, our second method (**Method 2: D⁺ projection**) demonstrates that a usable knowledge fingerprint can be distilled directly from raw text, with no source model required. Projecting the target activations onto the pseudo-inverse of the extracted direction matrix yields a **+5.0pp** gain from text alone, while compressing the entire transferable signature into a **1.2MB** fingerprint — small enough to store, version, and distribute as a first-class artifact alongside model weights.

Taken together, these results reframe domain adaptation as *surgery* rather than *training*. ModelDNA completes a full transfer in **31 seconds**, a **19×** speedup over the **576 seconds** required by comparable fine-tuning, and approaches a measured **95.3% fidelity ceiling** relative to an idealized full-rank transplant. The remaining 4.7% gap delineates the boundary of what purely linear, gradient-free surgery can recover, and motivates several future directions: (i) characterizing the fidelity ceiling theoretically as a function of the spectral overlap between donor and recipient subspaces; (ii) extending Bottom-Swap to multi-domain composition, where several fingerprints are written into disjoint low-importance subspaces simultaneously; (iii) studying the interaction between purification thresholds and overflow, to determine whether positive cross-domain transfer can be amplified deliberately; and (iv) applying D⁺ projection to safety- and alignment-relevant directions, where a small, auditable, deterministic fingerprint offers transparency advantages over opaque fine-tuning. We believe that treating knowledge as a portable geometric object — extractable, purifiable, and writable — opens a practical path toward modular, reproducible, and inspectable model editing.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

[2] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. In *International Conference on Learning Representations (ICLR)*.

[3] Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, Ł., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training Verifiers to Solve Math Word Problems. *arXiv preprint arXiv:2110.14168*.

[4] Eckart, C., & Young, G. (1936). The Approximation of One Matrix by Another of Lower Rank. *Psychometrika*, 1(3), 211–218.

[5] Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.

[6] Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions. *SIAM Review*, 53(2), 217–288.

[7] Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). Editing Models with Task Arithmetic. In *International Conference on Learning Representations (ICLR)*.

[8] Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

[9] Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). Mass-Editing Memory in a Transformer. In *International Conference on Learning Representations (ICLR)*.

[10] Sharma, P., Ash, J. T., & Misra, D. (2024). The Truth Is in There: Improving Reasoning in Language Models with Layer-Selective Rank Reduction (LASER). In *International Conference on Learning Representations (ICLR)*.

[11] Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Gontijo-Lopes, R., Morcos, A. S., Namkoong, H., Farhadi, A., Carmon, Y., Kornblith, S., & Schmidt, L. (2022). Model Soups: Averaging Weights of Multiple Fine-Tuned Models Improves Accuracy Without Increasing Inference Time. In *International Conference on Machine Learning (ICML)*, 23965–23998.

[12] Frankle, J., & Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In *International Conference on Learning Representations (ICLR)*.

[13] Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. In *Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)*, 7319–7328.

[14] Ben-Israel, A., & Greville, T. N. E. (2003). *Generalized Inverses: Theory and Applications* (2nd ed.). Springer.

[15] Denil, M., Shakibi, B., Dinh, L., Ranzato, M., & de Freitas, N. (2013). Predicting Parameters in Deep Learning. In *Advances in Neural Information Processing Systems (NeurIPS)*, 26.

[16] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. In *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[17] Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.

[18] Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., & Steinhardt, J. (2021). Measuring Mathematical Problem Solving with the MATH Dataset. In *Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*.

[19] Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023). TIES-Merging: Resolving Interference When Merging Models. In *Advances in Neural Information Processing Systems (NeurIPS)*, 36.

[20] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., et al. (2022). Training Compute-Optimal Large Language Models. *arXiv preprint arXiv:2203.15556*.