# Mathematical Foundations of the BEED SeqBoostNet Pipeline

This document develops the mathematics underlying every stage of the BEED seizure-classification
pipeline, from the raw EEG signal to the final stacked prediction. It is written for readers
comfortable with linear algebra, Fourier analysis, probability, and the standard machinery of
gradient-based and tree-based learning. Each method is presented with its mechanism stated
explicitly — what objective it minimizes, under what constraints, and why that objective is the
right one given the structure of EEG data.

The organizing principle is the non-stationarity and multichannel structure of EEG. A seizure is
not a single feature of the signal; it is a joint change in spectral content (which bands carry
power) and in temporal-spatial dynamics (how the 16 channels co-evolve). No single transform
captures both, which is the reason the pipeline runs two complementary feature extractors — FFT
and UMAP — and fuses their outputs before classification.

## Contents

1. [Problem setup and notation](#1-problem-setup-and-notation)
2. [Per-channel standardization](#2-per-channel-standardization)
3. [Spectral features via the FFT](#3-spectral-features-via-the-fft)
4. [Temporal features via UMAP](#4-temporal-features-via-umap)
5. [SeqBoostNet: a two-level stacking ensemble](#5-seqboostnet-a-two-level-stacking-ensemble)
6. [Why this composition, and its complexity](#6-why-this-composition-and-its-complexity)

## The pipeline at a glance

Each stage below has one job; the sections that follow justify each in turn. Notation: $n = 4{,}000$
samples, $c = 16$ channels, fused feature dimension $19 = 16 + 3$.

| Stage | Map | Shape | Purpose | Section |
|-------|-----|-------|---------|---------|
| Standardize | $X \to \tilde{X}$ | $n \times 16$ | Remove per-channel scale artifacts | [§2](#2-per-channel-standardization) |
| FFT | $\tilde{\mathbf{x}}_i \to \mathbf{s}_i$ | $n \times 16$ | Spectral (spatial-frequency) view | [§3](#3-spectral-features-via-the-fft) |
| UMAP | $\tilde{\mathbf{x}}_i \to \mathbf{u}_i$ | $n \times 3$ | Manifold (global-geometry) view | [§4](#4-temporal-features-via-umap) |
| Fuse | $[\mathbf{s}_i \Vert \mathbf{u}_i]$ | $n \times 19$ | Combine complementary views | [§4.4](#44-feature-fusion) |
| Base learners | $\mathbf{z}_i \to \hat{y}^{(m)}$ | $n \times 3$ | LSTM + XGBoost + GB predictions | [§5.1–5.3](#51-base-learner-i--lstm) |
| Meta learner | $\hat{Y} \to \hat{y}$ | $n \times 1$ | AdaBoost reliability-weighted vote | [§5.4](#54-meta-learner--adaboost) |

---

## 1. Problem setup and notation

The raw data is a matrix $X \in \mathbb{R}^{n \times c}$ with $n = 4{,}000$ samples and $c = 16$
channels, plus a label vector $y \in \{0, 1, 2, 3\}^n$ encoding the four clinical classes
(Healthy, Focal, Generalized, Seizure Events). Row $i$ of $X$, written $\mathbf{x}_i \in
\mathbb{R}^{16}$, is the vector of amplitudes (in μV) read simultaneously across the 16 electrodes
of the 10-20 montage.

The classification task is reduced to six binary problems $A_1, \dots, A_6$, each selecting a pair
of classes (e.g. $A_1$ = Generalized vs. Focal). For a given case the target is a function
$f : \mathbb{R}^{19} \to \{0, 1\}$ acting not on the raw $\mathbf{x}_i$ but on a 19-dimensional
engineered feature vector $\mathbf{z}_i$ constructed in §3–§4. The reduction to binary problems is
deliberate: the clinically meaningful distinctions (seizure vs. healthy, focal vs. generalized
onset) are pairwise, and the difficulty is highly heterogeneous across pairs — seizure-vs-healthy
is near-separable while within-seizure-type discrimination is not, a fact the results bear out and
that a single multiclass boundary would obscure.

Two facts about the raw data drive every preprocessing decision:

- **The CSV is ordered by class.** Any non-stratified split produces train/test folds with
  disjoint label support, which makes reported metrics meaningless. All splits are stratified on
  $y$.
- **Channel amplitudes are on heterogeneous scales.** Electrode impedance, proximity to active
  sources, and the referencing scheme cause per-channel offset and gain differences that are
  artifacts of acquisition, not of brain state. These are removed by per-channel standardization
  before any feature extraction (§2).

---

## 2. Per-channel standardization

Standardization is applied independently to each channel $j$ before feature extraction. For
channel $j$ with training-set mean $\mu_j$ and standard deviation $\sigma_j$,

$$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}.$$

The statistics $\mu_j, \sigma_j$ are estimated on the training fold only and then applied
unchanged to validation and test folds. This is not a cosmetic choice. Fitting the scaler on the
full dataset leaks the test-set location and scale into the training pipeline; under a per-class
shift in mean amplitude — exactly what a seizure produces — that leakage inflates apparent
separability. Estimating $\mu_j, \sigma_j$ on train alone keeps the evaluation honest.

The reason standardization is *per channel* rather than global is that the downstream UMAP step
depends on Euclidean distances in $\mathbb{R}^{16}$ (§4). Without per-channel scaling, a single
high-amplitude electrode dominates the distance metric and the manifold UMAP recovers reflects
that one channel's variance rather than the joint spatial pattern that distinguishes seizure
types.

> [!WARNING]
> Every fitted transform in this pipeline — the scaler here, the UMAP reducer in §4, and the base
> learners in §5 — must be fit on the training fold only and merely *applied* to validation and
> test. Fitting any of them on the full dataset leaks the evaluation distribution into the model
> and inflates the reported metrics. This is the single rule that protects the honesty of the
> results.

---

## 3. Spectral features via the FFT

### 3.1 The discrete Fourier transform

The discrete Fourier transform (DFT) maps a length-$N$ sequence $x_0, \dots, x_{N-1}$ to a
length-$N$ sequence of complex coefficients

$$X_k = \sum_{n=0}^{N-1} x_n \, e^{-2\pi i n k / N}, \qquad k = 0, 1, \dots, N-1.$$

Coefficient $X_k$ is the inner product of the signal with a complex sinusoid of frequency
$k/N$; its modulus $|X_k|$ is the amplitude of that frequency component and its argument is the
phase. The Fast Fourier Transform (FFT) is not a different transform but an $O(N \log N)$ algorithm
for computing the DFT exactly, exploiting the recursive factorization of the DFT matrix (the
Cooley–Tukey decomposition into even- and odd-indexed subsequences). The $O(N \log N)$ cost,
versus $O(N^2)$ for the naive sum, is what makes spectral features cheap enough to compute on every
sample.

### 3.2 What the transform is applied to here

The implementation applies the FFT along `axis=1` — across the 16 channels of a single sample —
not along the time axis across samples:

$$\mathbf{s}_i = \bigl| \mathrm{FFT}(\tilde{\mathbf{x}}_i) \bigr|, \qquad \tilde{\mathbf{x}}_i \in
\mathbb{R}^{16}, \; \mathbf{s}_i \in \mathbb{R}^{16}.$$

This is a deliberate and consequential modeling choice, and it changes the interpretation of the
result. Because each row is one time-point snapshot across electrodes ordered by scalp position,
the transform decomposes the *spatial* pattern of activation across the head into spatial-frequency
components: low-frequency coefficients capture smooth, broadly distributed activation (consistent
with generalized, bilateral discharge), while high-frequency coefficients capture sharp
channel-to-channel contrast (consistent with focal activity localized to a few adjacent
electrodes). The modulus is taken because the absolute spatial phase — which arbitrarily depends on
the chosen electrode ordering — carries no class information, whereas the amplitude spectrum is
what separates broadly-distributed from sharply-localized activation.

The two ends of the spatial-frequency axis map onto the two seizure morphologies the task must
separate:

```text
  Low spatial frequency           High spatial frequency
  ─────────────────────           ──────────────────────
  smooth across electrodes        sharp channel-to-channel
  broad, bilateral activation     contrast over few electrodes
            │                                │
            ▼                                ▼
      Generalized onset                 Focal onset
      (both hemispheres)                (localized region)

   |X_k| keeps amplitude · discards electrode-order phase
```

Taking $|X_k|$ discards phase and so is not invertible, but for classification that is the point:
the map collapses the phase nuisance dimension while preserving the amplitude structure on which
the focal/generalized distinction rests. The output is 16 spectral amplitude features per sample,
named `Fp1_fft, …, T6_fft`, preserving the $(n, 16)$ shape.

> [!NOTE]
> The canonical EEG band decomposition (δ: 0.5–4 Hz, θ: 4–8, α: 8–13, β: 13–30, γ: 30–100) is a
> *temporal*-frequency decomposition and is the usual motivation for FFT on EEG. The pipeline here
> instead computes a spatial-frequency spectrum per time sample. Both are valid feature maps; the
> spatial variant is what this codebase implements, and the band interpretation should be read as
> motivating context rather than as a literal description of these 16 features.

---

## 4. Temporal features via UMAP

FFT gives a per-sample spectral fingerprint but treats each of the 4,000 samples independently. To
capture the *global* geometry of the data — how samples cluster and how the channel-space manifold
is organized — the pipeline reduces the 16 standardized channels jointly to three coordinates with
UMAP (Uniform Manifold Approximation and Projection). The choice of a manifold method over a linear
one (PCA) is justified by the non-linearity of the structure: seizure and non-seizure samples are
not expected to be linearly separable in raw channel space, and PCA, being constrained to
orthogonal linear projections, cannot unfold a curved manifold. UMAP can, at the cost of not
preserving global distances exactly.

### 4.1 High-dimensional fuzzy neighborhood graph

UMAP first models the data as a weighted graph that encodes local neighborhood structure. For each
point $i$, distances to its $k$ nearest neighbors are converted to membership weights through an
exponential kernel,

$$\phi(d_{ij}, \sigma_i) = \exp\!\left(-\frac{\max(0,\, d_{ij} - \rho_i)}{\sigma_i}\right),$$

where $d_{ij} = \lVert \tilde{\mathbf{x}}_i - \tilde{\mathbf{x}}_j \rVert$, $\rho_i$ is the distance
to $i$'s nearest neighbor (guaranteeing each point has at least one full-weight edge), and
$\sigma_i$ is a per-point bandwidth chosen so that

$$\sum_{j} \phi(d_{ij}, \sigma_i) = \log_2 k.$$

This adaptive normalization is the key idea: $\sigma_i$ is calibrated locally so that every point
perceives roughly the same effective number of neighbors regardless of the local density. Dense
regions get a small $\sigma_i$, sparse regions a large one, and the resulting graph is
density-invariant — which is why UMAP recovers cluster structure that a fixed-bandwidth method
would smear out. (The simplified kernel $\exp(-d_{ij}^2 / 2\sigma_i^2)$ quoted in the source paper
conveys the same exponential-decay intuition.)

The directed edge weights are symmetrized into a single undirected fuzzy graph via the
probabilistic t-conorm,

$$p_{ij} = p_{i|j} + p_{j|i} - p_{i|j}\, p_{j|i},$$

so that an edge is "present" if either endpoint considers the other a neighbor. This is the
operational meaning of the mutual-neighbor combination $S_{ij}$ written in the source material.

### 4.2 Low-dimensional embedding by cross-entropy minimization

UMAP then seeks an embedding $\mathbf{e}_1, \dots, \mathbf{e}_n \in \mathbb{R}^3$ whose own
neighborhood graph matches the high-dimensional one. In the low-dimensional space the edge weight
is modeled by a smooth, heavy-tailed curve

$$q_{ij} = \bigl(1 + a \lVert \mathbf{e}_i - \mathbf{e}_j \rVert^{2b}\bigr)^{-1},$$

with $a, b$ fit to the user's `min_dist` setting. The embedding coordinates are found by minimizing
the fuzzy-set cross-entropy between the high-dimensional weights $p_{ij}$ and the low-dimensional
weights $q_{ij}$,

$$\mathcal{L} = \sum_{i \neq j} \Bigl[\, p_{ij} \log \frac{p_{ij}}{q_{ij}} + (1 - p_{ij})
\log \frac{1 - p_{ij}}{1 - q_{ij}} \,\Bigr].$$

The two terms do opposing work. The first is an attractive force: where $p_{ij}$ is large (true
neighbors), it penalizes large embedding distance, pulling neighbors together. The second is a
repulsive force: where $p_{ij}$ is small (non-neighbors), it penalizes small embedding distance,
pushing unrelated points apart. Minimizing their sum by stochastic gradient descent — with negative
sampling to approximate the $O(n^2)$ repulsive sum — yields a layout in which the local graph
structure of $\mathbb{R}^{16}$ is reproduced in $\mathbb{R}^3$. The source paper's objective
$\sum_{ij} S_{ij} \log(S_{ij}/Q_{ij})$ is the attractive half of this cross-entropy.

The embedding is the equilibrium of the two opposing forces:

```text
  Attractive term                  Repulsive term
  (p_ij large → neighbors)         (p_ij small → non-neighbors)
  ────────────────────────         ────────────────────────────
  penalizes large q-distance       penalizes small q-distance
  pulls true neighbors together    pushes unrelated points apart
              │                                 │
              └────────────────┬────────────────┘
                               ▼
                     equilibrium layout
          (local R¹⁶ neighborhoods reproduced in R³)
```

The output is three coordinates per sample, `umap_1, umap_2, umap_3`.

### 4.3 The train/transform discipline and its caveat

The reducer is fit on the training fold and then applied to validation and test folds via
`transform`, which embeds new points against the frozen training manifold. This prevents the test
data from influencing the learned geometry — the same leakage concern as in §2, and a more subtle
one here because UMAP is unsupervised and the temptation to fit it on all data is strong.

One property of UMAP must be stated as a limitation rather than glossed: the embedding preserves
local topology but not global distances, and the absolute coordinate values are not meaningful
across separate fits (the algorithm is stochastic and only fixed here by `random_state = 42`).
Consequently the three UMAP features are useful to a *downstream* classifier as relative-position
coordinates, but they do not admit a direct physical interpretation the way the spectral features
do. Their value is empirical: they encode cluster membership that the tree and recurrent models in
§5 can exploit as split variables.

### 4.4 Feature fusion

The two feature sets are concatenated into the final design matrix

$$\mathbf{z}_i = \bigl[\, \mathbf{s}_i \;\Vert\; \mathbf{u}_i \,\bigr] \in \mathbb{R}^{19},
\qquad \mathbf{s}_i \in \mathbb{R}^{16} \;(\text{FFT}), \quad \mathbf{u}_i \in \mathbb{R}^{3}\;
(\text{UMAP}).$$

The fusion is justified by complementarity. The spectral block answers "what is the distribution of
activation across spatial frequencies in this sample?" and the manifold block answers "where does
this sample sit relative to the global structure of the data?" These are independent views: two
samples with identical spectra can sit in different manifold regions, and vice versa. Concatenating
them gives the classifier access to both, which is the central empirical claim the project tests.

---

## 5. SeqBoostNet: a two-level stacking ensemble

The classifier is a stacked ensemble. Three base learners (LSTM, XGBoost, Gradient Boosting) are
trained on $\mathbf{z}_i$, and a meta-learner (AdaBoost) is trained on their outputs. The
motivation for stacking is bias–variance decorrelation: the three base learners have different
inductive biases — a recurrent network imposes a sequential prior, gradient-boosted trees impose an
axis-aligned piecewise-constant prior — and their errors are therefore only partially correlated.
A meta-learner that combines them can achieve lower error than any single member, provided the
members are individually competent and make *different* mistakes. The results section reports the
ensemble outperforming both constituent and competing stacks, which is the evidence for that
provision holding here.

```text
        z_i (19 features)
              │
   ┌──────────┼──────────┐         LEVEL 0
   ▼          ▼          ▼         (base learners)
 LSTM      XGBoost   GradBoost
   │          │          │
   └── ŷ_LSTM ┴ ŷ_XGB ┴ ŷ_GB ──┐
                                ▼   stacked predictions
                            AdaBoost  LEVEL 1 (meta)
                                │
                                ▼
                          final ŷ
```

### 5.1 Base learner I — LSTM

The Long Short-Term Memory network processes the feature vector as a sequence, maintaining a cell
state $C_t$ and hidden state $h_t$ regulated by three gates. At step $t$, with input $x_t$ and
previous hidden state $h_{t-1}$,

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{(forget gate)} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{(input gate)} \\
\tilde{C}_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{(candidate cell)} \\
C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{(cell update)} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{(output gate)} \\
h_t &= o_t \odot \tanh(C_t) && \text{(hidden state)}
\end{aligned}
$$

where $\sigma$ is the logistic sigmoid and $\odot$ is elementwise product. The architecture exists
to solve the vanishing-gradient problem of plain recurrent networks: the cell-state recurrence
$C_t = f_t \odot C_{t-1} + \dots$ is additive rather than multiplicative, so when the forget gate
$f_t \approx 1$ the gradient $\partial C_t / \partial C_{t-1} \approx 1$ and error signals
propagate across many steps without exponential decay. The forget gate learns *when* to retain or
overwrite the running state; the input and output gates control what enters and what is read out.

The path of information through one cell makes the additive recurrence visible:

```text
  C_{t-1}   (previous cell state)
     │
     ▼  × f_t            forget gate drops stale memory
   (gate)
     │
     ▼  + (i_t × C̃_t)    input gate writes new candidate
   (add) ◄── additive, not multiplicative:
     │        ∂C_t/∂C_{t-1} ≈ f_t  ⇒  gradient does not vanish
     ▼
    C_t     (new cell state)
     │
     ▼  × o_t            output gate reads memory out
   (gate) ──► tanh ──► h_t   (hidden state)
```

The network has 128 units, ReLU activations on the dense head, dropout 0.5 for regularization, and
is trained with Adam against sparse categorical cross-entropy for 100 epochs at batch size 32. The
sequential prior is the LSTM's contribution to the ensemble: it is the only base learner that can
model order-dependent structure across the 19 features, giving it an error profile distinct from
the order-invariant tree models.

### 5.2 Base learner II — XGBoost

XGBoost fits an additive ensemble of $K$ regression trees, $\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{z}_i)$,
by minimizing a regularized objective

$$\mathcal{L} = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k),
\qquad \Omega(f) = \gamma T + \tfrac{1}{2}\lambda \lVert w \rVert^2,$$

where $T$ is the number of leaves in a tree and $w$ its leaf weights. The regularizer $\Omega$ is
what distinguishes XGBoost from ordinary gradient boosting: $\gamma T$ penalizes tree complexity
(pruning leaves whose gain does not exceed $\gamma$) and $\tfrac12 \lambda \lVert w \rVert^2$
shrinks leaf weights toward zero, both controlling overfitting explicitly rather than only through
the learning rate.

Trees are added greedily. At iteration $t$ the objective is expanded to second order in the new
tree's output using gradients $g_i = \partial_{\hat{y}} l$ and Hessians $h_i = \partial^2_{\hat{y}} l$
evaluated at the current prediction:

$$\mathcal{L}^{(t)} \simeq \sum_i \bigl[ g_i f_t(\mathbf{z}_i) + \tfrac12 h_i f_t(\mathbf{z}_i)^2 \bigr]
+ \Omega(f_t).$$

For a fixed tree structure this is a quadratic in the leaf weights with closed-form optimum
$w_j^\star = -\big(\sum_{i \in j} g_i\big) / \big(\sum_{i \in j} h_i + \lambda\big)$, and the
resulting objective value scores candidate splits. The second-order expansion is what makes XGBoost
both fast and accurate: using curvature ($h_i$) rather than gradient alone gives a better local
model of the loss and a principled split-gain criterion. Configured here with 300 estimators, max
depth 6, learning rate 0.05, and a multi-softmax objective, XGBoost is the high-capacity,
heavily-regularized member of the ensemble.

### 5.3 Base learner III — Gradient Boosting

Plain gradient boosting builds its ensemble by the same additive, stage-wise logic but with only a
first-order view of the loss. At stage $m$,

$$F_m(\mathbf{z}) = F_{m-1}(\mathbf{z}) + \alpha\, h_m(\mathbf{z}),$$

where $h_m$ is a regression tree fit to the negative gradient (the pseudo-residual)
$r_{im} = -\,\partial l(y_i, F_{m-1}(\mathbf{z}_i)) / \partial F_{m-1}(\mathbf{z}_i)$ of the loss at
the current ensemble, and $\alpha$ is the learning rate that shrinks each tree's contribution.
Fitting successive trees to residuals is functional gradient descent: each tree is an approximate
step in the steepest-descent direction in function space.

Configured with 100 estimators, learning rate 0.1, and shallow trees of max depth 3, this learner
is intentionally lower-capacity and more strongly biased than the XGBoost member. The redundancy is
deliberate — two tree ensembles with different depths, regularization, and learning rates produce
correlated-but-not-identical errors, and the meta-learner profits from the difference.

### 5.4 Meta learner — AdaBoost

The meta-learner is trained on the base learners' predictions, $\hat{y}^{\text{LSTM}},
\hat{y}^{\text{XGB}}, \hat{y}^{\text{GB}}$, treated as a new 3-dimensional feature representation of
each sample. AdaBoost fits a weighted ensemble of weak classifiers $h_t$ by the reweighting
recursion: starting from uniform sample weights $w_i^{(0)} = 1/n$, at round $t$ it fits $h_t$ to
minimize the weighted error

$$\varepsilon_t = \sum_i w_i^{(t-1)} \mathbb{1}[\,y_i \neq h_t(\mathbf{z}_i)\,],$$

assigns that classifier the weight

$$\alpha_t = \tfrac{1}{2} \log \frac{1 - \varepsilon_t}{\varepsilon_t},$$

and updates the sample weights multiplicatively,

$$w_i^{(t)} \propto w_i^{(t-1)} \exp\bigl(-\alpha_t\, y_i\, h_t(\mathbf{z}_i)\bigr),$$

so that samples the current ensemble misclassifies are up-weighted and dominate the next round. The
final decision is the weighted vote

$$H(\mathbf{z}) = \operatorname{sign}\!\Big( \sum_t \alpha_t\, h_t(\mathbf{z}) \Big).$$

The classifier weight $\alpha_t$ is exactly the quantity that makes AdaBoost a coordinate-descent
minimizer of the exponential loss $\sum_i \exp(-y_i F(\mathbf{z}_i))$: $\alpha_t$ is the closed-form
optimal step size along the direction $h_t$, which is why a more accurate weak learner ($\varepsilon_t$
small) receives exponentially more weight. Placed at the meta level with 50 weak learners and
learning rate 1.0, AdaBoost's role is to learn the reliability-weighted combination of the three
base predictions, concentrating on the samples — the hard within-seizure-type cases — that the base
learners disagree on.

Each round is one turn of a reweighting loop that funnels attention toward the hard samples:

```text
  uniform weights w⁽⁰⁾
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │  round t                                     │
   │    fit hₜ  ──►  εₜ = Σ wᵢ · 1[ yᵢ ≠ hₜ(xᵢ) ] │
   │                      │                       │
   │                      ▼                       │
   │          αₜ = ½ ln( (1 − εₜ) / εₜ )          │
   │                      │                       │
   │                      ▼                       │
   │    reweight  wᵢ ← wᵢ · exp(−αₜ yᵢ hₜ(xᵢ))    │
   │    (misclassified samples up-weighted)       │
   └──────────────────────────────────────────────┘
        │  repeat for T rounds
        ▼
   H(x) = sign( Σ αₜ hₜ(x) )
```

---

## 6. Why this composition, and its complexity

The pipeline is a sequence of maps, each with a stated job:

$$X \xrightarrow{\text{standardize}} \tilde{X}
\xrightarrow{\text{FFT} \,\Vert\, \text{UMAP}} Z \in \mathbb{R}^{n \times 19}
\xrightarrow{\text{LSTM, XGB, GB}} \hat{Y} \in \mathbb{R}^{n \times 3}
\xrightarrow{\text{AdaBoost}} \hat{y}.$$

The total time complexity is the sum of the stage costs,

$$O\big(\underbrace{N \log N}_{\text{FFT}} + \underbrace{N D}_{\text{UMAP}} +
\underbrace{M T}_{\text{boosting}}\big),$$

with $N$ samples, $D$ dimensions, $M$ features, $T$ boosting iterations. Every term is at most
quasi-linear in the data size, which is what makes the full stack — two feature extractors plus a
four-model ensemble — tractable on commodity hardware.

The design coheres around one idea stated at the outset: a seizure is a joint spectral-and-temporal
phenomenon, so the feature stage must extract both views (FFT for spectral content, UMAP for
manifold structure) and the classification stage must combine learners whose inductive biases are
diverse enough that their errors decorrelate (a sequential network and two differently-regularized
tree ensembles, fused by a reliability-weighting meta-learner). Each component is individually
standard; the contribution is the principled composition.

### Limitations of the formulation

- The spatial-FFT interpretation (§3.2) depends on the electrode ordering of the columns; a
  permutation of channels changes the spectrum. The features are well-defined and reproducible
  given a fixed ordering, but they are not invariant to montage relabeling, and the band-power
  reading should not be over-interpreted.
- UMAP coordinates (§4.3) are not physically interpretable and are stable only under a fixed seed;
  the manifold features earn their place empirically, through downstream accuracy, not through a
  mechanistic account.
- Stacking adds variance from the base-learner training and a second layer of hyperparameters. The
  reported gains are credible only to the extent that the meta-learner is evaluated on
  base-learner outputs generated out-of-fold; in-fold stacking would optimistically bias the
  metrics, and the evaluation protocol should guarantee the separation.

---

## References

Najmusseher and Nizar Banu P K, "Feature Engineering for Epileptic Seizure Classification Using
SeqBoostNet," *International Journal of Computing and Digital Systems*, vol. 17, no. 1, pp. 1–15,
2025. DOI: 10.12785/ijcds/1571020131.

McInnes, L., Healy, J., and Melville, J., "UMAP: Uniform Manifold Approximation and Projection for
Dimension Reduction," *arXiv:1802.03426*, 2018.

Chen, T. and Guestrin, C., "XGBoost: A Scalable Tree Boosting System," *Proc. 22nd ACM SIGKDD*,
pp. 785–794, 2016.

Hochreiter, S. and Schmidhuber, J., "Long Short-Term Memory," *Neural Computation*, vol. 9, no. 8,
pp. 1735–1780, 1997.

Freund, Y. and Schapire, R. E., "A Decision-Theoretic Generalization of On-Line Learning and an
Application to Boosting," *Journal of Computer and System Sciences*, vol. 55, no. 1, pp. 119–139,
1997.
