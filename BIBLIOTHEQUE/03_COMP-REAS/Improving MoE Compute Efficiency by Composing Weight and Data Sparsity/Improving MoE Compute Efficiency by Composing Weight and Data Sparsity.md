# IMPROVING MOE COMPUTE EFFICIENCY BY COMPOSING WEIGHT AND DATA SPARSITY

Maciej Kilian

Oleg Mkrtchyan

Luke Zettlemoyer\*

Akshat Shrivastava

Armen Aghajanyan

![](_page_0_Picture_7.jpeg)

#### **ABSTRACT**

Mixture-of-Experts layers achieve compute efficiency through weight sparsity: each token activates only a subset of experts. Data sparsity, where each expert processes only a subset of tokens, offers a complementary axis. Expert-choice routing implements data sparsity directly but violates causality in autoregressive models, creating train-inference mismatch. We recover data sparsity within causal token-choice MoE by leveraging zero-compute (null) experts within the routing pool. When a token routes to null experts, those slots consume no compute. The standard load balancing objective trains the model to uniformly use all experts (real and null) therefore creating data sparsity in expectation without the causality violations. We evaluate on vision-language model training, where data heterogeneity is pronounced: vision encoders produce many low-information tokens while text tokens are denser. At matched expected FLOPs, composing weight and data sparsity yields a more compute-efficient frontier than weight sparsity alone, with gains in training loss and downstream performance. The model learns implicit modality-aware allocation, routing vision tokens to null experts more aggressively than text, without explicit modality routing.

## 1 Introduction

Mixture-of-Experts (MoE) layers [1; 2] have enabled more efficient scaling of Transformers through weight sparsity: they replicate the FFN into many experts and use a router to select a small subset per token, enabling conditional computation with large parameter counts while keeping per-token compute fixed. But data itself is heterogeneous and redundant. Many inputs contain large low-information regions-blank or repetitive patches, punctuation, boilerplate, predictable continuations-suggesting that not every token deserves the same compute budget.

This motivates data sparsity: instead of allocating weights per token, allocate tokens per weight. Weight and data sparsity are dual views of the same routing matrix  $R \in \{0,1\}^{T \times N}$ : weight sparsity constrains columns (each token activates at most k experts), while data sparsity constrains rows (each expert processes a bounded number of tokens). Composing both yields a budget over the full token×expert matrix.

The challenge is implementing data sparsity in autoregressive models. Expert-choice MoE [3] is data-sparse by design-each expert selects its tokens-which achieves weight sparsity in expectation since expert overlap is unlikely. However, this requires access to future tokens, creating non-causal dependencies and train-inference mismatch. Token-choice MoE [1] preserves causality by letting each token select its experts independently, but assigns exactly k experts to every token regardless of information content.

We recover data sparsity within causal token-choice MoE by leveraging zero-compute (null) experts [4–6] to the routing pool. When the router assigns a token to null experts, those slots skip expert computation. The standard load balancing objective [1; 7] trains the model to uniformly use all experts-real and null-creating data sparsity in expectation without causality violations.

Data heterogeneity is particularly pronounced in multimodal training: vision encoders produce many tokens per image, most carrying little information, while text tokens tend to be denser. We therefore focus on vision-language model training as a setting where data sparsity should provide clear benefits.

<sup>\*</sup>University of Washington

![](_page_1_Picture_0.jpeg)

Figure 1: **Top:** Weight and data sparsity as dual budget constraints. Weight sparsity bounds experts per token (row budget  $\leq \rho_w E$ ); data sparsity bounds tokens per expert (column budget  $\leq \rho_d T$ ); composing both yields a budget over the full  $T \times E$  matrix. **Bottom:** Implementations. Token-choice achieves weight sparsity causally; expert-choice achieves data sparsity but requires seeing future tokens; null experts compose both while preserving causality.

At matched expected FLOPs, configurations with data sparsity outperform those without, yielding gains in both training loss and downstream performance. The model learns implicit modality-aware allocation: vision tokens route to null experts more aggressively than text, shifting compute toward task-relevant information without explicit modality routing.

Our main contributions are:

- Demonstration that composing weight and data sparsity improves the compute-efficiency frontier.
- A minimal modification to token-choice MoE that achieves this composition via zero-compute experts while preserving causality.
- Analysis showing modality-aware compute allocation emerges in multimodal training without explicit supervision.

## 2 RELATED WORK

**Mixture-of-Experts.** MoE architectures achieve compute efficiency through weight sparsity, activating a subset of experts per token. GShard [2] and Switch Transformer [1] established token-choice top-K routing; expert-choice routing [3] inverts this by letting experts select tokens. Our work builds on token-choice routing but extends it to support data sparsity.

**Multimodal MoE.** Vision-language models face modality imbalance: images produce many low-information tokens while text tokens are denser. V-MoE [8] handles this via Batch Priority Routing; LiMoE [9] uses an entropy-based regularization scheme and shows modality-specific experts emerge organically; MoMa [10] partitions experts by modality architecturally. Our null expert mechanism achieves modality-aware allocation implicitly through standard load balancing.

**Adaptive Computation.** Several approaches vary compute across tokens. Mixture-of-Depths [11] routes tokens to skip entire transformer blocks. MoE++ [4] introduces zero-computation experts within MoE layers. AdaMoE [5] applies null experts in fine-tuning. LongCat-Flash [6] scales zero-compute experts to 560B parameters at 75% data sparsity. Recent work on MoE router distribution

shaping [12] incentivizes the router outputs to fit a specific probability distribution giving us more control over routing behavior.

We build on these approaches but emphasize a different lens: null experts compose weight sparsity (which experts) with data sparsity (which tokens), and this composition should only help-the solution space of denser configurations is preserved within sparser ones. Our contribution is compute-controlled experiments confirming this intuition: at matched expected FLOPs, weight-and-data-sparse MoE consistently outperforms weight-sparse-only baselines.

#### 3 BACKGROUND

#### 3.1 WEIGHT AND DATA SPARSITY

Standard MoE [1; 2] implements weight sparsity: each token activates k of N experts. Per-token compute is fixed at k expert evaluations regardless of the token's information content.

Data sparsity varies compute across tokens. Some tokens use fewer than k experts; others may use more. At aggregate level, expected compute remains controlled, but allocation adapts to the data.

As shown in Figure 1, these dimensions are orthogonal. Consider the token×expert routing matrix  $R \in \{0,1\}^{T \times N}$  where  $R_{t,e} = 1$  if token t routes to expert e. Weight sparsity constrains columns: each token activates at most k experts. Data sparsity constrains rows: each expert processes a bounded number of tokens. Composing both constrains the total budget  $\sum_{t,e} R_{t,e}$ .

This framing implies a natural experimental comparison. At matched expected compute, configurations with data sparsity ( $\rho < 1$ ) can allocate 0 to k experts per token, while dense configurations ( $\rho = 1$ ) allocate exactly k to every token. If composed sparsity outperforms at iso-compute, data sparsity provides value beyond what weight sparsity alone achieves.

## 3.2 TOKEN-CHOICE MOE

A standard token-choice MoE [1] layer consists of N expert networks  $\mathcal{E} = \{E_1, \dots, E_N\}$ , a shared expert  $E_{\text{shared}}$ , and a router G that activates the top-K experts per token:

$$\boldsymbol{y} = E_{\text{shared}}(\boldsymbol{x}) + \sum_{i=1}^{N} g_i \cdot E_i(\boldsymbol{x}), \quad g_i = \begin{cases} \text{Softmax}(G(\boldsymbol{x}))_i & \text{if } i \in \text{top-K}(\{G(\boldsymbol{x})_j\}_{j=1}^N) \\ 0 & \text{otherwise} \end{cases}$$
(1)

where G(x) = Wx with  $W \in \mathbb{R}^{N \times D}$ .

Token-choice routing preserves causality: each token selects its experts using only its own representation. This is essential for autoregressive models where future tokens are unavailable at inference.

# 3.3 THE CAUSALITY CHALLENGE

Expert-choice routing [3] implements data sparsity directly: each expert selects its top-K tokens from a batch, naturally allowing variable compute per token. However, this requires observing all tokens in a sequence simultaneously, violating causality in autoregressive models [13]. During training the model learns to rely on information from future tokens that will not be available at inference, creating train-inference mismatch.

Token-choice preserves causality but assigns exactly k experts to every token, precluding data sparsity. We want both: the causality of token-choice and the variable allocation of expert-choice.

# 4 METHOD

We extend token-choice MoE with a minimal modification: adding null experts to the routing pool. This composes weight and data sparsity while preserving causality.

#### 4.1 NULL EXPERT MECHANISM

We extend the router to output N+1 logits by expanding  $\mathbf{W} \in \mathbb{R}^{(N+1) \times D}$ . The first N logits correspond to real experts; the (N+1)th corresponds to the null expert, which outputs zero:  $E_{\text{null}}(\mathbf{x}) = \mathbf{0}$ . To control data sparsity, we duplicate the null logit M times before top-K selection (highlighted terms denote changes from token-choice):

$$\tilde{G}(\boldsymbol{x}) = \left[ G(\boldsymbol{x})_{1:N}, \underbrace{G(\boldsymbol{x})_{N+1}, \dots, G(\boldsymbol{x})_{N+1}}_{M \text{ conies}} \right] \in \mathbb{R}^{N+M}$$
 (2)

The gating function becomes:

unction becomes:
$$g_i = \begin{cases} \text{Softmax}(\tilde{\boldsymbol{G}}(\boldsymbol{x}))_i & \text{if } i \in \text{top-K}(\{\tilde{\boldsymbol{G}}(\boldsymbol{x})_j\}_{j=1}^{N+M}) \text{ and } i \leq N \\ 0 & \text{otherwise} \end{cases}$$
(3)

Routing weights are renormalized over only the selected real experts, so output magnitude is unaffected by null routing.

#### 4.2 NOTATION

We denote the number of real experts activated per token as the random variable  $\mathbf{K}_{\rho}^{k_{\max}}$ , where  $k_{\max}$  is the maximum allocation (the top-K value) and  $\rho \in (0,1]$  is the target data sparsity. The expected top-K is:

$$\mathbb{E}[\mathbf{K}_{\rho}^{k_{\max}}] = k_{\max} \cdot \rho \tag{4}$$

For example,  $\mathbf{K}_{0.5}^8$  denotes a configuration with top-8 routing at 50% data sparsity, yielding  $\mathbb{E}[\mathbf{K}_{0.5}^8] = 4$  expected real experts per token. For brevity, we refer to iso-compute families by their shared expectation, e.g., " $\mathbb{E}[K] = 2$  runs."

**Model configurations.** We vary three dimensions: base model scale, maximum allocation  $k_{\text{max}}$ , and target data sparsity  $\rho$ . Configurations are denoted by base scale and routing parameters, e.g., 0.6B  $\mathbf{K}_{0.5}^8$  indicates a 0.6B base model with  $k_{\text{max}}=8$  at  $\rho=0.5$ . When we refer to parameter count alone (e.g., "0.6B" or "1.7B"), we mean the base Qwen3 dense model [14] from which we initialize; when reporting full MoE scale we use the standard [total]-A[active] format, e.g., 5.3B-A1.2B denotes 5.3B total parameters with 1.2B active per token.

# 4.3 THRESHOLDING INTERPRETATION

Duplicating the null logit implements thresholding. For a token to activate exactly r < k real experts, the router learns to set the null logit such that exactly r real expert logits exceed it. The remaining k-r slots go to null copies, contributing nothing. After renormalization, the output matches a standard top-r MoE. A single null logit suffices since all copies are identical.

# 4.4 CONTROLLING DATA SPARSITY

For N real experts and target data sparsity  $\rho \in (0, 1]$ , we set:

$$M = N \cdot \frac{1 - \rho}{\rho} \tag{5}$$

With N=64:  $\rho=0.5$  requires M=64 null copies;  $\rho=0.25$  requires M=192.

#### 4.5 Training Objectives

**Load Balancing Loss.** We apply a global load balancing [7] over all N + M slots:

$$\mathcal{L}_{\text{bal}} = \frac{N+M}{(N+M)} \cdot \sum_{i=1}^{N+M} f_i \cdot P_i \tag{6}$$

where  $f_i$  is the fraction of tokens routed to slot i and  $P_i$  is the average routing probability. Enforcing uniform load across an expanded pool that includes M null copies incentivizes the target data sparsity.

**Z-Loss.** We apply z-loss to stabilize training [15; 16]:

$$\mathcal{L}_z = \frac{1}{T} \sum_{t=1}^{T} \log^2 \left( \sum_{i=1}^{N+M} \exp(\tilde{G}(\boldsymbol{x}_t)_i) \right)$$
 (7)

With null experts, load balancing and task loss conflict more than in standard MoE: the model must route low-information tokens to nulls while achieving good performance on high-information tokens. Z-loss prevents the router from becoming overconfident.

#### 4.6 PROPERTIES

**Solution Space Preservation.** Higher-sparsity configurations can recover lower-sparsity solutions. Consider  $\mathbf{K}_{0.5}^4$  versus  $\mathbf{K}_{1.0}^2$ : both have  $\mathbb{E}[K]=2$ . If the optimal solution uses exactly 2 experts per token, the  $\mathbf{K}_{0.5}^4$  router can learn to always select 2 real experts plus 2 null experts, exactly recovering the  $\mathbf{K}_{1.0}^2$  output after renormalization. Data sparsity should only help: the model can always fall back to denser computation if sparsity does not benefit the data.

**Soft Constraint.** Expert-choice [3] enforces data sparsity as a hard constraint: experts have fixed capacity. Token-choice with nulls enforces it as a soft constraint: load balancing encourages but does not strictly enforce the target sparsity. The model learns to route low-information tokens to null experts as an emergent property of training.

Why Zero (Not Copy) Experts. MoE++ [4] explores several null expert variants: zero experts (output = 0), copy experts (output = x), and constant experts. We use zero experts because only they preserve solution space. With copy experts at sparsity  $\rho$ , the output becomes approximately  $(1-\rho)\cdot x + \rho\cdot y_{\text{dense}}$ —the input dominates regardless of what real experts compute, and the dense solution is no longer recoverable. See Appendix A for empirical analysis.

#### 5 EXPERIMENTS

#### 5.1 Training Details

We train on a multimodal mixture of vision-language data. Following MoMa [10], we begin with a dense warmup phase (20k steps) before enabling MoE routing, allowing the model to develop meaningful representations before routing decisions. After warmup, we upcycle to MoE using a fine-grained expert shape inspired by [17]: 64 experts at  $4\times$  granularity, with 30% of FFN parameters reinitialized randomly. We upcycle from 0.6B and 1.7B Qwen3 [14] dense models, yielding 5.3B-A1.2B and 18.7B-A3.4B model scales at  $\mathbb{E}[K]=4$ .

For optimization, we use AdamW [18] with a peak learning rate of 2e-5 (tuned across all experiments),  $\beta = (0.9, 0.95)$ , and weight decay of 0.1. We apply a WSD [19] learning rate schedule with 500-step warmup, decaying to 10% of peak over the final 10% of training. Load balancing and z-loss weights are set to 2e-2 and 1e-3 respectively.

In Section 5.2 and Section 5.3 we train for 50k steps with batch size 512 and sequence length 2048 ( $\sim$ 52B tokens total). In Section 5.4 we extend training to 200k steps with batch size 128 and sequence length 8192 ( $\sim$ 209B tokens total). Remaining details can be found in [?].

#### 5.2 SCALING DATA SPARSITY

![](_page_5_Figure_1.jpeg)

Figure 2: Effect of data sparsity ( $\rho$ ) on training loss (**Left**) and evaluation score (**Right**) for MoE models upcycled from dense 0.6B and 1.7B models. Lower  $\rho$  indicates more null experts. Training loss decreases monotonically with increased sparsity, while evaluation scores peak at  $\rho \approx 0.5$ .

We first investigate how performance varies across data sparsity levels at fixed expected compute. We train 9 configurations at  $\mathbb{E}[K] = 2$ , varying base scale and sparsity:

• 0.6B: 
$$\mathbf{K}_{1.0}^2$$
,  $\mathbf{K}_{0.67}^3$ ,  $\mathbf{K}_{0.5}^4$ ,  $\mathbf{K}_{0.25}^8$ ,  $\mathbf{K}_{0.17}^{12}$ 

• 1.7B: 
$$\mathbf{K}_{1.0}^2$$
,  $\mathbf{K}_{0.67}^3$ ,  $\mathbf{K}_{0.5}^4$ ,  $\mathbf{K}_{0.25}^8$ 

We measure final training loss (averaged over the last 1k steps) and average eval performance across 10 standard benchmarks (AI2D [20], A-OKVQA [21], BLINK [22], ChartQA [23], Perceptron Grounding [?], DocVQA [24], M3Exam [25], SEED-Bench [26], TextVQA [27], VSR [28]). Results are shown in Figure 2. Three findings emerge.

**Solution space validation.** Data sparsity monotonically improves training loss across both model scales. This confirms solution space preservation: higher-sparsity configurations can recover lower-sparsity solutions, so performance is bounded below by the dense baseline.

**Eval-loss divergence.** Eval gains do not track loss improvements at high sparsity. Performance peaks at  $\rho \approx 0.5$ , then degrades despite continued loss reduction. At  $\mathbf{K}_{0.25}^8$ , evals fall below baseline. We discuss possible explanations in Appendix A.1.

**Scale transfer.** The eval-optimal sparsity region ( $\rho \in [0.5, 0.67]$ ) is consistent across 0.6B and 1.7B base scales. Sparsity configuration can be tuned at smaller scales and transferred, reducing experimental cost.

We constrain subsequent experiments to  $\rho=0.5$ , where eval gains are stable. Extending the stable regime to higher sparsity through alternative routing mechanisms remains future work.

## 5.3 Compute Efficiency Gains

Having established a stable operating regime at  $\geq$ 50% data sparsity, we now ask whether data sparsity improves compute efficiency across compute scales. We train models at two base sizes across multiple top-K values. For each (base model, top-K) pair, we compare three data sparsities: the 1.0 dense baseline, 0.67, and 0.5. Results are shown in Figure 3.

![](_page_6_Figure_0.jpeg)

Figure 3: Training loss (**Left**) and average evaluation score (**Right**) as a function of training FLOPs for data-dense and data-sparse MoE models. Blue circles represent data-dense baselines with fixed top-K routing; red circles represent data-sparse models using null experts, with opacity indicating the sparsity ratio  $\rho$ . The gray line connects data-dense models to form a Pareto frontier, with the shaded region indicating suboptimal performance. Data-sparse models consistently outperform data-dense baselines at equivalent compute budgets, achieving lower training loss and higher evaluation scores. Labels indicate model scale and expected active experts  $\mathbb{E}[K]$ ;  $\rho$  denotes the ratio of real experts to total experts (e.g.,  $\rho = 0.50$  means half of selected experts are real on average).

Data sparsity reveals a more compute-efficient frontier in both training loss and evals, with gains more pronounced at larger compute scales. The effect is clearest in training loss, where data-sparse configurations consistently outperform dense baselines at matched expected FLOPs. Eval gains are present but slightly noisier.

## 5.4 HERO RUN

To confirm these gains hold at larger compute budgets, we extend training of the 1.7B  $\mathbf{K}_{1.0}^4$  and  $\mathbf{K}_{0.5}^8$  models to 200k steps (209B tokens). Table 1 compares our models against Isaac 0.2, a dense 1.7B baseline trained with the same recipe, and InternVL3.5-20B-A4B [29], a similarly sized sparse VLM. Note that InternVL3.5 was trained with more compute across multiple stages, whereas our models use only single-stage SFT; we report their published results for reference. Both MoE configurations consistently outperform the dense baseline across all task categories, with the  $\mathbf{K}_{0.5}^8$  model achieving the best overall results. The improvements are particularly pronounced in OCR and counting tasks. Despite using a simpler training pipeline, our models remain competitive with InternVL and surpass it on several benchmarks.

# 6 ROUTING BEHAVIOR

The previous sections established that data sparsity improves the compute-efficiency frontier. But how does the model use this flexibility? If tokens route to null experts uniformly at random, data sparsity would offer no advantage. The value comes from selective allocation: routing low-information tokens to null experts while preserving compute for tokens that need it.

We find the model learns exactly this. Three patterns emerge from routing analysis. First, null experts shift compute from vision to text without explicit modality routing. Second, this reallocation is task-dependent: the same image receives different compute maps under different prompts. Third, we measure increasing polarization in MoE compute per token at high data sparsity: tokens route entirely to real or null experts rather than mixing.

| Task     | Benchmark             | 1.7B $\mathbf{K}_{1.0}^4$ | 1.7B $\mathbf{K}_{0.5}^{8}$ | Isaac 0.2 | InternVL3.5-20B-A4B |
|----------|-----------------------|---------------------------|-----------------------------|-----------|---------------------|
| Size     | _                     | 18B-A3B                   | 18B-A3B                     | 2B        | 20B-A4B             |
| Pointing | Aerial Grounding      | 80.7                      | 82.2                        | 73.1      | _                   |
| _        | Perceptron Grounding  | 58.8                      | 59.2                        | 51.5      | _                   |
|          | RefCOCO [30]          | 87.6                      | 87.8                        | 87.4      | 91.9                |
|          | Overall               | 75.7                      | 76.4                        | 70.7      | _                   |
| OCR      | ChartQA [23]          | 79.0                      | 80.3                        | 75.1      | 86.6                |
|          | DocVQA [24]           | 92.9                      | 93.8                        | 92.1      | 92.9                |
|          | A-OKVQA (val) [21]    | 89.4                      | 91.8                        | 87.2      | _                   |
|          | TextVQA [27]          | 80.8                      | 82.0                        | 78.1      | 78.5                |
|          | OCRBench [31]         | 873                       | 880                         | 857       | 870                 |
|          | Overall               | 85.9                      | 87.2                        | 83.6      |                     |
| Counting | Aerial Counting       | 53.0                      | 57.0                        | 52.0      |                     |
|          | CVBench [32]          | 72.1                      | 73.8                        | 72.5      | _                   |
|          | PixMoCount [33]       | 68.6                      | 69.4                        | 66.7      | _                   |
|          | CountBench [34]       | 85.5                      | 87.5                        | 84.6      |                     |
|          | Overall               | 69.8                      | 71.9                        | 69.0      | _                   |
| General  | VSR (Zero-Shot) [28]  | 79.6                      | 80.6                        | 78.6      | _                   |
|          | VQA v2 [35]           | 82.6                      | 82.4                        | 80.8      | _                   |
|          | RealWorldQA [36]      | 74.1                      | 75.1                        | 77.9      | 71.2                |
|          | SEED-Bench [26]       | <b>76.8</b>               | 75.8                        | 74.2      | _                   |
|          | M3Exam (English) [25] | 54.8                      | 58.6                        | 55.8      | _                   |
|          | NLVR2 [37]            | 82.6                      | 83.2                        | 76.4      | _ <del>_</del>      |
|          | BLINK [22]            | 53.8                      | 55.6                        | 55.0      | 59.0                |
|          | MathVista [38]        | 73.2                      | 73.9                        | 69.6      | 78.0                |
|          | MME [39]              | 2221                      | 2237                        | 2092      | 2318                |
|          | AI2D [20]             | 79.1                      | 78.9                        | 74.5      | 85.9                |
|          | ERQA [40]             | 40.6                      | 39.4                        | 36.8      | 41.6                |
|          | Overall               | 70.6                      | 71.2                        | 68.6      | _                   |

Table 1: Benchmark performance of extended training runs (200k steps, 209B tokens). Our sparse MoE models (18B-A3B) consistently outperform the dense Isaac 0.2 baseline across pointing, OCR, counting, and general vision-language tasks, while remaining competitive with InternVL3.5-20B-A4B despite its more complex multi-stage training pipeline.

## 6.1 Modality Compute Rebalancing

![](_page_7_Figure_3.jpeg)

Figure 4: MoE compute distribution by modality across sparsity configurations. **Left:** Token distribution (constant at 74% vision, 26% text). **Middle:** Compute distribution. Dense configurations  $(\mathbf{K}_{1.0}^2)$  allocate compute proportionally to token count; data-sparse configurations route vision tokens to null experts, reducing vision's share from 74% to 36%. **Right:** Compute intensity (fraction of top-K slots filled by real experts). Vision intensity drops to 0.04 at  $\mathbf{K}_{0.17}^{12}$  while text remains at 0.19.

Null experts invert the compute distribution between modalities. To quantify this, we measure three quantities: token distribution (each modality's share of total tokens, constant at 78% vision / 22%

text across configurations), compute distribution (each modality's share of total effective compute, where a token routed entirely to null experts contributes 0 and one using only real experts contributes 1), and compute intensity (the average compute score per token, indicating what fraction of expert slots are filled by real versus null experts).

Figure 4 shows these metrics under varying data sparsity. In dense configurations without null experts  $(\mathbf{K}_{1.0}^2)$ , compute distribution mirrors token distribution: vision consumes 78% of MoE compute simply because it produces 78% of tokens. This is inefficient. Vision encoders generate many redundant patches—blank regions, repetitive textures, uninformative background—while text tokens tend to be information-dense.

As we introduce null experts with increasing sparsity ( $\mathbf{K}_{0.67}^3$  through  $\mathbf{K}_{0.17}^{12}$ ), vision tokens route to null experts far more aggressively than text. At  $\mathbf{K}_{0.17}^{12}$ , vision compute intensity drops to 4% compared to 22% for text, producing an inverted compute distribution: text consumes 60% of total compute despite representing only 22% of tokens. The model allocates compute based on information content rather than token count.

Prior work has noted that modality imbalance poses challenges for multimodal MoE training. V-MoE [8] addresses heterogeneous token importance via Batch Prioritized Routing: tokens are ranked by an importance score and, under capacity constraints, the least informative patches are dropped. This enables adaptive per-image compute but requires explicitly scoring and sorting tokens across the batch. LiMoE [9] tackles training instability from modality imbalance through entropy-based auxiliary losses applied per-modality, finding that modality-specific expert specialization emerges organically with proper regularization. MoMa [10] takes an architectural approach, partitioning experts into modality-specific groups where each group exclusively processes its designated modality—text tokens route only to text experts, image tokens only to image experts. Using the null expert mechanism we achieve similar modality-aware rebalancing implicitly: standard load balancing over an expanded routing pool is sufficient to induce differential compute allocation without explicit importance scoring, per-modality regularization, or architectural partitioning. The model receives no signal about which modality deserves more compute; it learns this from task performance alone.

#### 6.2 TOKEN-LEVEL COMPUTE MAPS

![](_page_9_Figure_1.jpeg)

Figure 5: Per-token compute utilization for a sample sequence. **Left:** Original input (text and image patches). **Right:** Compute overlay where brightness indicates fraction of top-K slots filled by real experts—bright regions receive full computation, dark regions route to null experts. Inset values show mean compute per segment (image or text block) and overall sample compute (top-right). Vision tokens receive less compute on average due to redundant patches.

Figure 5 visualizes per-token compute utilization. For each token, we measure the fraction of top-K slots filled by real experts (averaged across MoE layers), producing a score between 0 (complete null routing) and 1 (full computation).

Compute varies substantially within each modality, not just between them. Among vision tokens, salient regions receive more compute than background. Among text tokens, predictable continuations, punctuation, and control sequences route to null experts more aggressively than information-dense tokens. The model allocates compute based on token-level information content, not modality alone. This within-modality variation implies data sparsity benefits text-only training as well-we focus on multimodal settings because vision token heterogeneity is pronounced, but text sequences contain their own redundancy, and the model exploits it. Prior work confirms this [4–6]. See Appendix C for more token-level maps.

![](_page_10_Figure_0.jpeg)

Figure 6: Compute allocation varies with task context. **Left:** Original input. **Middle:** Compute overlay under a general QA prompt—high compute distributed broadly. **Right:** Compute overlay under a targeted segmentation prompt—reduced overall compute, concentrated on task-relevant regions. System prompts shown top-left; mean compute statistics shown top-right. Brightness indicates real expert utilization (dark = null routing). The model routes more aggressively to null experts when the task requires only a subset of the image.

Allocation is also context-dependent. The same image receives different compute maps under different prompts. Figure 6 compares routing for identical visual input with a underspecified prompt (i.e., "Answer with single word.") versus a targeted segmentation prompt. Under the underspecified task, the model distributes compute broadly. Under segmentation, it concentrates compute on task-relevant regions and routes most patches to null experts. Null expert routing is not a fixed property of inputs-the model learns that "relevant" is task-relative and skips computation for tokens that do not serve the current objective.

# 6.3 DATA SPARSITY STRATEGIES

![](_page_10_Figure_4.jpeg)

Figure 7: (a) Zero-compute token ratio versus null expert ratio, grouped by  $\mathbb{E}[K]$  (labels show  $k_{\max}$ ). Configurations with the same  $\mathbb{E}[K]$  achieve similar zero-compute ratios despite different  $k_{\max}$  values—expected active experts determines polarization, not  $k_{\max}$  alone. Zero-compute falls below null expert ratio because tokens not fully skipped can mix real and null experts within their top-K selection. (b, c) Training dynamics for 0.6B  $\mathbf{K}_{0.17}^{12}$  and 1.7B  $\mathbf{K}_{0.5}^{8}$ . Both metrics converge toward theoretical sparsity (dotted); zero-compute consistently lags below null expert ratio.

How do models achieve data sparsity? They could spread null routing uniformly—every token uses slightly fewer real experts—or polarize, with some tokens using full compute and others using none. Figure 7 shows the distribution of real expert utilization across configurations.

At fixed  $\mathbb{E}[K]=2$ , increasing data sparsity (decreasing  $\rho$ ) produces an almost linear increase in the fraction of tokens routed to zero real experts. The  $\mathbf{K}^{12}_{0.17}$  configuration routes over 60% of tokens to zero real experts despite having 12 slots available—most selections fall on null copies. The model makes binary decisions: full computation or none.

Increasing  $\mathbb{E}[K]$  reduces this polarization. Models with higher expected active experts distribute compute more uniformly across tokens rather than concentrating it on a subset. Scaling base dense model size has a similar marginal effect: 1.7B models show lower zero-compute ratios than 0.6B at equivalent configurations, though the effect is modest compared to the sparsity target itself.

#### 7 CONCLUSION AND FUTURE WORK

Weight and data sparsity are orthogonal efficiency axes, but standard MoE exploits only the first: each token activates a fixed subset of experts regardless of its information content. Composing both yields a strictly better compute-efficient frontier, and null experts achieve this composition within token-choice MoE while preserving causality.

We demonstrated these gains on vision-language model training, where data heterogeneity is pronounced. At matched expected FLOPs, data-sparse configurations achieved lower training loss and improved downstream performance across model scales. The model learned to allocate compute by information content rather than token count: vision tokens routed to null experts more aggressively than text, and routing patterns shifted based on task demands, all without explicit supervision. These behaviors emerged from the standard load balancing objective alone.

Our framing of data sparsity as an axis orthogonal to weight sparsity opens several directions. Null experts are one mechanism for achieving data sparsity, but not necessarily the optimal one – the eval breakdown at high sparsity Section 5.2) suggests limitations in jointly modeling both axes through a single softmax, and understanding this failure mode could inform better architectures. More elaborate router distribution shaping approaches [12] offer a promising alternative: rather than encouraging data sparsity through load balancing over an expanded pool, one could directly fit the global routing distribution to a target encoding balanced load, sparsity level, and other desiderata simultaneously. Data sparsity also extends beyond MoE layers to attention modules, where some tokens may warrant less cross token compute than others [41].

## REFERENCES

- [1] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity, 2022. URL https://arxiv.org/abs/2101.03961.
- [2] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding, 2020. URL https://arxiv.org/abs/2006.16668.
- [3] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew Dai, Zhifeng Chen, Quoc Le, and James Laudon. Mixture-of-experts with expert choice routing, 2022. URL https://arxiv.org/abs/2202.09368.
- [4] Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moe++: Accelerating mixture-of-experts methods with zero-computation experts, 2024. URL https://arxiv.org/abs/2410.07348.
- [5] Zihao Zeng, Yibo Miao, Hongcheng Gao, Hao Zhang, and Zhijie Deng. Adamoe: Token-adaptive routing with null experts for mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2406.13233.
- [6] Meituan LongCat Team, Bayan, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, Chengcheng Han, Chenguang Xi, Chi Zhang, Chong Peng, Chuan Qin, Chuyu Zhang, Cong Chen, Congkui Wang, Dan Ma, Daoru Pan, Defei Bu, Dengchang Zhao, Deyang Kong, Dishan Liu, Feiye Huo, Fengcun Li, Fubao Zhang, Gan Dong, Gang Liu, Gang Xu, Ge Li, Guoqiang Tan, Guoyuan Lin, Haihang Jing, Haomin Fu, Haonan Yan, Haoxing Wen, Haozhe Zhao, Hong Liu, Hongmei Shi, Hongyan Hao, Hongyin Tang, Huantian Lv, Hui Su, Jiacheng Li, Jiahao Liu, Jiahuan Li, Jiajun Yang, Jiaming Wang, Jian Yang, Jianchao Tan, Jiaqi Sun, Jiaqi Zhang, Jiawei Fu, Jiawei Yang, Jiaxi Hu, Jiayu Qin, Jingang

Wang, Jiyuan He, Jun Kuang, Junhui Mei, Kai Liang, Ke He, Kefeng Zhang, Keheng Wang, Keqing He, Liang Gao, Liang Shi, Lianhui Ma, Lin Qiu, Lingbin Kong, Lingtong Si, Linkun Lyu, Linsen Guo, Liqi Yang, Lizhi Yan, Mai Xia, Man Gao, Manyuan Zhang, Meng Zhou, Mengxia Shen, Mingxiang Tuo, Mingyang Zhu, Peiguang Li, Peng Pei, Peng Zhao, Pengcheng Jia, Pingwei Sun, Qi Gu, Qianyun Li, Qingyuan Li, Qiong Huang, Qiyuan Duan, Ran Meng, Rongxiang Weng, Ruichen Shao, Rumei Li, Shizhe Wu, Shuai Liang, Shuo Wang, Suogui Dang, Tao Fang, Tao Li, Tefeng Chen, Tianhao Bai, Tianhao Zhou, Tingwen Xie, Wei He, Wei Huang, Wei Liu, Wei Shi, Wei Wang, Wei Wu, Weikang Zhao, Wen Zan, Wenjie Shi, Xi Nan, Xi Su, Xiang Li, Xiang Mei, Xiangyang Ji, Xiangyu Xi, Xiangzhou Huang, Xianpeng Li, Xiao Fu, Xiao Liu, Xiao Wei, Xiaodong Cai, Xiaolong Chen, Xiaoqing Liu, Xiaotong Li, Xiaowei Shi, Xiaoyu Li, Xili Wang, Xin Chen, Xing Hu, Xingyu Miao, Xinyan He, Xuemiao Zhang, Xueyuan Hao, Xuezhi Cao, Xunliang Cai, Xurui Yang, Yan Feng, Yang Bai, Yang Chen, Yang Yang, Yaqi Huo, Yerui Sun, Yifan Lu, Yifan Zhang, Yipeng Zang, Yitao Zhai, Yiyang Li, Yongjing Yin, Yongkang Lv, Yongwei Zhou, Yu Yang, Yuchen Xie, Yueqing Sun, Yuewen Zheng, Yuhuai Wei, Yulei Qian, Yunfan Liang, Yunfang Tai, Yunke Zhao, Zeyang Yu, Zhao Zhang, Zhaohua Yang, Zhenchao Zhang, Zhikang Xia, Zhiye Zou, Zhizhao Zeng, Zhongda Su, Zhuofan Chen, Zijian Zhang, Ziwen Wang, Zixu Jiang, Zizhe Zhao, Zongyu Wang, and Zunhai Su. Longcat-flash technical report, 2025. URL https://arxiv.org/abs/2509.01322.

- [7] Zihan Qiu, Zeyu Huang, Bo Zheng, Kaiyue Wen, Zekun Wang, Rui Men, Ivan Titov, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Demons in the detail: On implementing load balancing loss for training specialized mixture-of-expert models, 2025. URL https://arxiv.org/abs/2501.11873.
- [8] Carlos Riquelme, Joan Puigcerver, Basil Mustafa, Maxim Neumann, Rodolphe Jenatton, André Susano Pinto, Daniel Keysers, and Neil Houlsby. Scaling vision with sparse mixture of experts, 2021. URL https://arxiv.org/abs/2106.05974.
- [9] Basil Mustafa, Carlos Riquelme, Joan Puigcerver, Rodolphe Jenatton, and Neil Houlsby. Multimodal contrastive learning with limoe: the language-image mixture of experts, 2022. URL https://arxiv.org/abs/2206.02770.
- [10] Xi Victoria Lin, Akshat Shrivastava, Liang Luo, Srinivasan Iyer, Mike Lewis, Gargi Ghosh, Luke Zettlemoyer, and Armen Aghajanyan. Moma: Efficient early-fusion pre-training with mixture of modality-aware experts, 2024. URL https://arxiv.org/abs/2407.21770.
- [11] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models, 2024. URL https://arxiv.org/abs/2404.02258.
- [12] Leyla Mirvakhabova, Babak Ehteshami Bejnordi, Gaurav Kumar, Hanxue Liang, Wanru Zhao, and Paul Whatmough. Dirichlet-prior shaping: Guiding expert specialization in upcycled moes, 2025. URL https://arxiv.org/abs/2510.01185.
- [13] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. Auxiliary-loss-free load balancing strategy for mixture-of-experts, 2024. URL https://arxiv.org/abs/2408. 15664.
- [14] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [15] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh,

Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. Palm: Scaling language modeling with pathways, 2022. URL https://arxiv.org/abs/2204.02311.

- [16] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. St-moe: Designing stable and transferable sparse expert models, 2022. URL https://arxiv.org/abs/2202.08906.
- [17] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. Deepseekmoe: Towards ultimate expert specialization in mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2401.06066.
- [18] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https://arxiv.org/abs/1711.05101.
- [19] Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective, 2024. URL https://arxiv.org/abs/2410.05192.
- [20] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *European Conference on Computer Vision (ECCV)*, 2016. URL https://arxiv.org/abs/1603.07396. Introduces the AI2D diagram dataset.
- [21] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In *European Conference on Computer Vision (ECCV)*, 2022. URL https://www.ecva.net/papers/eccv\_2022/papers\_ECCV/papers/136680141.pdf.
- [22] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. In *European Conference on Computer Vision (ECCV)*, 2024. URL https://www.ecva.net/papers/eccv\_2024/papers\_ECCV/papers/03356.pdf.
- [23] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning, 2022. URL https://arxiv.org/abs/2203.10244.
- [24] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2021. URL https://openaccess.thecvf.com/content/WACV2021/papers/Mathew\_DocVQA\_A\_Dataset\_for\_VQA\_on\_Document\_Images\_WACV\_2021\_paper.pdf.
- [25] Wenxuan Zhang, Sharifah Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models, 2023. URL https://arxiv.org/abs/2306.05179. English subset commonly referenced as M3Exam\_English.
- [26] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. URL https://arxiv.org/abs/2307.16125.

- [27] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards VQA models that can read. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019. URL https://openaccess.thecvf.com/content\_CVPR\_2019/papers/Singh\_Towards\_VQA\_Models\_That\_Can\_Read\_CVPR\_2019\_paper.pdf. Introduces the TextVQA dataset.
- [28] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. *Transactions of the Association for Computational Linguistics*, 11:635–651, 2023. doi: 10.1162/tacl\_a\_00566. URL https://aclanthology.org/2023.tacl-1.37/.
- [29] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.
- [30] Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions. In *European Conference on Computer Vision (ECCV)*, pages 69–85. Springer, 2016. doi: 10.1007/978-3-319-46475-6\_5. URL https://link.springer.com/chapter/10.1007/978-3-319-46475-6\_5. Introduces RefCOCO and RefCOCO+.
- [31] Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. *Science China Information Sciences*, 67(12), December 2024. ISSN 1869-1919. doi: 10.1007/s11432-024-4235-6. URL http://dx.doi.org/10.1007/s11432-024-4235-6.
- [32] Nannan Zhu, Yonghao Dong, Teng Wang, Xueqian Li, Shengjun Deng, Yijia Wang, Zheng Hong, Tiantian Geng, Guo Niu, Hanyan Huang, Xiongfei Yao, and Shuaiwei Jiao. Cvbench: Benchmarking cross-video synergies for complex multimodal reasoning, 2026. URL https://arxiv.org/abs/2508.19542.
- [33] Allen Institute for AI. Pixmo-count. https://huggingface.co/datasets/allenai/pixmo-count, 2025. Dataset card; part of the PixMo collection used in Molmo & PixMo (arXiv:2409.17146).
- [34] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023. URL https://openaccess.thecvf.com/content/ICCV2023/papers/Paiss\_Teaching\_CLIP\_to\_Count\_to\_Ten\_ICCV\_2023\_paper.pdf. Introduces Count-Bench.
- [35] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. URL https://openaccess.thecvf.com/content\_cvpr\_2017/papers/Goyal\_Making\_the\_v\_CVPR\_2017\_paper.pdf.
- [36] xAI. Realworldqa: A benchmark for real-world visual question answering. https://huggingface.co/datasets/xai-org/RealworldQA, 2024.

- [37] Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for reasoning about natural language grounded in photographs. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*, pages 6418–6428, Florence, Italy, 2019. doi: 10.18653/v1/P19-1644. URL https://aclanthology.org/P19-1644/.
- [38] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In *International Conference on Learning Representations (ICLR)*, 2024. URL https://openreview.net/forum?id=KUNzEQMWU7.
- [39] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2023. URL https://arxiv.org/abs/2306.13394.
- [40] Google DeepMind Gemini Robotics Team. Erqa: Embodied reasoning question answer benchmark. https://github.com/embodiedreasoning/ERQA, 2025. Released with the Gemini Robotics tech report: arXiv:2503.20020.
- [41] Peng Jin, Bo Zhu, Li Yuan, and Shuicheng Yan. Moh: Multi-head attention as mixture-of-head attention, 2025. URL https://arxiv.org/abs/2410.11842.
- [42] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. Megablocks: Efficient sparse training with mixture-of-experts, 2022. URL https://arxiv.org/abs/2211.15841.
- [43] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems 32, pages 8024-8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf.

#### A NULL EXPERTS

#### A.1 INFERENCE LIMITATIONS

In Figure 2, training loss improves monotonically as we increase data sparsity, but evaluation peaks around  $\rho \approx 0.5$  and degrades at more aggressive sparsity (e.g.,  $\mathbf{K}^8_{0.25}$ ,  $\mathbf{K}^{12}_{0.17}$ ). We view this gap as a limitation of the *single-softmax*, thresholded null-copy construction rather than evidence that data sparsity is inherently harmful. Below we outline three interacting effects that become more pronounced as  $\rho$  decreases.

- 1) Effective router resolution collapses at high sparsity. With M duplicated null copies, the router produces a categorical distribution over N+M slots, and real experts must compete against a large null block in the same softmax. When  $\rho \ll 1$ , the balancing objective encourages substantial probability mass to sit on the null region of the simplex. This can reduce the *effective* resolution of the router among real experts: relative differences between real experts are compressed after normalization, and gradients that refine expert identity decisions can be attenuated because most mass lies outside the real-expert subset. In this regime the router can remain good at the *compute* decision (real vs. null) while becoming worse at the *expert identity* decision (which real expert), plausibly harming downstream metrics even as likelihood improves.
- 2) Thresholding becomes unstable and encourages polarization. Null-copy routing implements a top-K threshold: a token activates r real experts when exactly r real logits exceed the null logit. At aggressive sparsity, the null threshold must sit above most real logits, making intermediate allocations (e.g., r=1,2,3) sensitive to small logit perturbations. A robust alternative for the model is to adopt a bimodal strategy: push the null logit far above all real logits (route to all nulls) or far below many of them (route to mostly real experts). Such polarization can hurt evaluation when tasks benefit from broad but shallow compute (many moderately informative tokens) rather than concentrating compute on a small subset.
- 3) Auxiliary-objective interference grows as  $\rho$  decreases. Standard MoE load balancing assumes every routed slot corresponds to meaningful computation. With null copies, the balancing term implicitly asks the router to distribute tokens uniformly over both (i) semantically specialized experts and (ii) a replicated "do nothing" region. At high sparsity, satisfying balancing can become easier by increasing null usage rather than improving expert specialization, increasing the mismatch between what balancing incentivizes and what evaluation rewards. This interference is amplified by renormalization: once a token selects any real experts, we renormalize over them, so the forward pass is insensitive to how much probability mass the router placed on nulls as long as the top-k set is unchanged. In contrast, the auxiliary losses depend on the full softmax distribution, enabling progress on auxiliary objectives that does not necessarily translate into better expert assignments.

**Empirical signatures.** These mechanisms make predictions that are consistent with the routing behavior we already observe. As  $\rho$  decreases, routing becomes increasingly polarized (Figure 7): a growing fraction of tokens route to zero real experts, while the remainder consume most of the compute budget. This is the expected qualitative signature of an unstable thresholding regime, where intermediate allocations are fragile and the model prefers an all-or-nothing strategy. In the same regime, reduced discrimination among real experts becomes more likely because real experts must compete against a large null block in the softmax; this provides a plausible explanation for why likelihood can continue to improve while downstream metrics degrade.

**Implication.** Taken together, these effects suggest that the main limitation is *coupling* the compute decision (how many real experts) and the expert identity decision (which real experts) inside a single normalized routing distribution whose support is dominated by null copies at low  $\rho$ . Extending stable evaluation gains to more aggressive data sparsity likely requires modifying this coupling or the associated regularization; we leave this to future work and restrict subsequent experiments to  $\rho \geq 0.5$ , where evaluation gains are stable.

#### A.2 COPY EXPERTS

We initially explored copy experts following prior work, but abandoned this direction due to both theoretical and empirical concerns.

Solution space violation. Copy experts break the solution space preservation property that makes null experts attractive. With copy experts at data sparsity  $\rho$ , the MoE output becomes approximately  $(1-\rho)\cdot x + \rho\cdot y_{\text{dense}}$ , where the input x dominates regardless of what real experts compute. The dense baseline  $(\mathbf{K}_{1.0}^k)$  is therefore not recoverable within the data-sparse configuration. This residual dilution effect worsens at higher sparsity—at extreme settings, the MLP output collapses toward the identity function. We suspect this explains why LongCat-Flash investigated only modest data sparsity (75%).

**Polarized routing.** The residual dilution problem incentivizes polarized routing behavior. When a token routes to a mixture of real and copy experts, the output is diluted by the copy expert contributions. The model can avoid this dilution by making binary decisions: route entirely to real experts (preserving full expert output) or entirely to copy experts (clean residual connection). Figure 8 shows this effect in token-level compute maps from copy expert runs, which exhibit far more polarized compute distributions than their zero expert counterparts. With zero experts, this pressure does not exist—routing weights are renormalized over only the selected real experts, so mixed routing incurs no penalty.

**Pathological training dynamics.** The combination of residual connections and load balancing loss creates a strong gradient pathway through copy experts. In some cases, the model learned to rely on copy expert routing as a default strategy rather than using null routing selectively for low-information tokens. This produced pathological behaviors: the model learned inverse mappings that allowed strong SigLIP embeddings to pass through unchanged, and in early training (within the first few hundred steps), learned to drop virtually all vision tokens to satisfy sparsity targets while relying solely on attention and the vision encoder for visual conditioning.

**Mitigation.** For practitioners who wish to use copy experts despite these issues, we found that both dense warmup and null expert warmup are essential to prevent the model from dropping all vision tokens early into training but compute maps remained polarized.

![](_page_18_Figure_0.jpeg)

Figure 8: Copy experts produce polarized routing. Tokens route fully to real experts (bright) or fully to copy experts (dark), with few intermediate values. Mixed routing causes residual dilution, incentivizing all-or-nothing decisions. Compare to zero expert compute maps (Figure 5), which show smoother gradation.

#### B Infrastructure

Our implementation of null experts is a minimal extension of standard token-choice MoE, adding minimal lines of code to the routing logic. This simplicity is possible because token-choice MoE already contains a natural pocket of dynamic computation that we can exploit.

Many early MoE implementations restored static shapes by imposing expert capacity-GShard-style routing [2], for instance, drops or truncate tokens when experts overflow. Others relied on specialized sparse kernels like MegaBlocks [42] to remain dropless while handling variable loads. More recently, implementations have converged on grouped GEMM: tokens are permuted into contiguous per-expert blocks, a single grouped matrix multiplication executes over the resulting operands, and outputs scatter back to the original token order. PyTorch [43] 2.8's native GroupedGEMM handles highly variable tokens-per-expert efficiently with only minimal alignment padding.

This is what makes null experts essentially free. The kernel already accepts variable token counts, so we simply expand the router to include null logits and truncate the sorted token list before the same grouped\_mm call. The kernel itself is unchanged. Algorithm 1 shows the implementation in PyTorch-style pseudocode, with highlighted lines denoting changes from standard token-choice. Because argsort places null expert indices  $(\geq N)$  at the end of the sorted order, we slice  $[:num\_real]$  and proceed as usual.

Asynchronous load-balancing loss. Global load-balancing losses are critical for stable token-choice MoE training [7] and remain our mechanism for controlling data sparsity. However, they require additional collectives in each MoE layer-typically all-reduces over per-expert token counts. To keep these off the forward critical path, we launch them asynchronously inside each layer, continue without waiting, and synchronize only after the forward completes to finalize the auxiliary loss.

# Algorithm 1 Token-Choice MoE with Null Experts

```
Require: x: (T, D), W: (N + 1, D), E: (N, D_h, D), top-K, null copies M
   ▶ Router
1: logits = x \in W.T
                                                                       \triangleright (T, N+1)
2: logits = cat([logits[:, :N], logits[:, N:].expand(-1, M)], dim=1)
3: scores = softmax(logits, dim=1)
4: top_scores, top_idx = top-K(scores, k, dim=1)
                                                                          \triangleright (T, k)
   ▶ Reorder tokens by expert (standard token-choice)
5: order = argsort(top_idx.flatten(), stable=True)
6: sorted_scores = top_scores.flatten()[order]
7: sorted_tokens = order // k
                                                                       \triangleright (N+M,)
8: num_per_expert = histc(top_idx, bins=N+M)
   ▶ Truncate to real experts (null experts sort last)
9: num real = num per expert[:N].sum()
10: sorted_scores[:num_real] /= sorted_scores[:num_real].sum_per_token()
  ⊳ renorm
   ▶ Expert computation (unchanged grouped mm)
11: x_in = x[sorted_tokens [:num_real]] * sorted_scores [:num_real, None]
12: x_out = grouped_mm(x_in, E, num_per_expert[:N])

13: y = zeros(T, D)
14: y.scatter_add_(0, sorted_tokens[:num_real], x_out)
15: return y
```

Adaptive activation checkpointing. With null experts, memory requirements shift during training. Early on, routers utilize the full configured top-K. As load balancing converges, the effective number of active experts drops for most tokens, and a checkpointing configuration tuned for the first  $\sim\!\!100$  steps becomes overly conservative. We address this with an adaptive controller that periodically measures peak memory and headroom, disabling checkpointing when safe to increase throughput and re-enabling it if memory becomes constrained. Since policy updates occur only every N steps, recompilation overhead is negligible.

# C COMPUTE MAPS

![](_page_20_Figure_1.jpeg)

Figure 9: DocVQA samples elicit diverse compute maps due to task specificity and importance of fine-grained features. **Left:** A sample with lots of text which can be relevant to the question. The model allocates a lot of compute to understand each patch. **Right:** A sample with lots of white space which the model can save compute on. The sample on the left uses more than double average compute.

![](_page_21_Figure_0.jpeg)

Figure 10: Compute allocation strategy varies significantly between layers. Earlier layers are much more focused on text and semantic aspects of the image whereas the last layer focuses more on the entirety of the image, allocating less compute to text.