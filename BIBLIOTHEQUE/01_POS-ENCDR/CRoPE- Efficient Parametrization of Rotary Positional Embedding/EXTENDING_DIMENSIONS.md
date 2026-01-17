## 1. Basic Metadata

- Title: "CRoPE: Efficient Parametrization of Rotary Positional Embedding" (Title)
- Authors: "Beicheng Lou*"; "Zifei Xu*" (Title)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

"We argue that complex linear transformation is a more natural parametrization and saves near 50% parameters within the attention block" while "removing such redundancy has negligible impact on the model performance both in sample and out of sample" (Abstract).

## 3. Tasks Evaluated

- Task: WikiText-2 evaluation (training/validation loss); Task type: Other (specify: training/validation loss on text corpus); Dataset(s): WikiText-2; Domain: natural language text; Evidence: "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits." (Section 5.2 Datasets) and "Training losses are shown in the top row of Fig. 4." / "The validation losses are shown in the bottom row in Fig. 4." (Section 6.1 Training and validation Losses)
- Task: Penn Treebank evaluation (training/validation loss); Task type: Other (specify: training/validation loss on text corpus); Dataset(s): Penn Treebank; Domain: natural language text; Evidence: "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits." (Section 5.2 Datasets) and "Table 6.1 shows the final validation loss of the models on different datasets." (Section 6.1 Training and validation Losses)
- Task: PG-19 evaluation (training/validation loss); Task type: Other (specify: training/validation loss on text corpus); Dataset(s): PG-19; Domain: natural language text; Evidence: "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits. For PG-19, only a subset of the data split was used" (Section 5.2 Datasets) and "Table 2: Final validation loss for different model/training configurations on PG-19 Dataset" (Section 6.2 Ablation Studies)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within the same modality (text); Evidence: "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets" (Section 5.2 Datasets).
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| WikiText-2 evaluation | Not specified (reported per-dataset) | Not specified | Not specified | "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits." (Section 5.2 Datasets) |
| Penn Treebank evaluation | Not specified (reported per-dataset) | Not specified | Not specified | "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits." (Section 5.2 Datasets) |
| PG-19 evaluation | Not specified (reported per-dataset) | Not specified | Not specified | "We used WikiText-2 [29], Penn Treebank [30] and PG-19[31] datasets with training and validation on the respective data splits." (Section 5.2 Datasets) |

## 6. Input and Representation Constraints

- Sequence length constraint: "max sequence length of 1024" (Section 5.3 Training setting).
- Model width/depth constraints: "The backbone of our models is the lightweight GPT-2 [28] decoder architecture with L=4 layers, H=4 attention heads per layer, and hidden size  $d_{\rm model}=128$ .." (Section 5.1 Model architecture).
- Projection dimensionality constraint for CRoPE: "CRoPE model: a GPT2 model with the output dimensions of the Q, K, V projection layers halved and the input dimension of the attention output projection layer halved." (Section 5.1 Model architecture).
- Fixed input resolution, patch size, padding, resizing: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: "max sequence length of 1024" (Section 5.3 Training setting).
- Fixed or variable length: Sequence length is a tunable setting; "ablation study on the different choice of batch size, sequence length, hidden dimension and learning rate decay." (Section 6.2 Ablation Studies).
- Attention type (global/windowed/hierarchical/sparse): Not specified; only "GPT-2 [28] decoder architecture" is stated (Section 5.1 Model architecture).
- Mechanisms to manage computational cost (windowing/pooling/pruning): Not specified.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanisms compared: "XPOS model: a typical GPT2 model with absolute positional embedding from the original Transformer paper [1]." / "RoPE model: a typical GPT2 model with standard rotary positional embedding" / "CRoPE model: a GPT2 model with the output dimensions of the Q, K, V projection layers halved and the input dimension of the attention output projection layer halved." (Section 5.1 Model architecture).
- RoPE definition: "In RoPE[14], one has a rotation matrix that performs position-dependent rotations to each 2-by-2 subspace" (Section 2.2 Rotary Positional Embedding).
- Where applied: "The models differ in the usage of CRoPE weights for the Q, K, V projection layers." (Section 5.1 Model architecture). Placement for XPOS/RoPE beyond this is not specified.
- Fixed vs modified vs compared: Multiple positional encodings are explicitly compared across experiments via the three model variants (Section 5.1 Model architecture; Section 6 Results).

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Core variable; "We define the different model architectures used in this paper as follow: - **XPOS** model... - RoPE model... - **CRoPE** model..." (Section 5.1 Model architecture).
- Multiple positional encodings compared: Yes; "Models that rely on absolute positional encodings show markedly higher loss than any variant that employs rotary encodings." (Section 6.1 Training and validation Losses).
- Claim PE choice is not critical or secondary: Not stated; performance is reported as "removing such redundancy has negligible impact on the model performance" (Abstract).

## 10. Evidence of Constraint Masking

- Model sizes reported: "L=4 layers, H=4 attention heads per layer, and hidden size  $d_{\rm model}=128$" (Section 5.1 Model architecture).
- Parameter savings (architecture change): "it saves 50% of parameters compared to typical weights with the same shape." (Section 5.1 Model architecture) and "saving nearly half the parameters in the attention layers" (Section 7 Conclusion).
- Dataset sizes: Not specified.
- Attribution of gains: "removing such redundancy has negligible impact on the model performance both in sample and out of sample." (Abstract) and "the difference between **RoPE** and **CRoPE** losses are within the range of noise." (Section 6.1 Training and validation Losses).

## 11. Architectural Workarounds

- Parameter-reducing projection structure: "CRoPE weight is defined as ... where it saves 50% of parameters compared to typical weights with the same shape." (Section 5.1 Model architecture).
- Dimensionality halving in attention projections: "CRoPE model: a GPT2 model with the output dimensions of the Q, K, V projection layers halved and the input dimension of the attention output projection layer halved." (Section 5.1 Model architecture).
- Lightweight backbone choice: "lightweight GPT-2 [28] decoder architecture with L=4 layers, H=4 attention heads per layer" (Section 5.1 Model architecture).

## 12. Explicit Limitations and Non-Claims

- Limitation on analytical examples: "Note that this example is only for illustration. How well it can extrapolate to deeper networks may be beyond analytical work and invite for empirical study." (Section 4.3 Token-dependent position comparison).
- Scope caveat: "While this toy problem is simplistic, it is a minimal example to illustrate the advantage of RoPE over conventional absolute positional embedding." (Section 4.3 Token-dependent position comparison).
- Uncertainty about expressivity tradeoff: "Whether that expressivity is worth the parameters is a different story, which depends on interpretation, tasks and various other factors." (Section 3.2 Detailed look into function space).
- Explicit statements about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Single modality (text) with multiple text datasets (Section 5.2 Datasets).
- Task structure: Per-dataset training/validation loss evaluation; no joint multi-task setup described (Section 5.2 Datasets; Section 6.1 Training and validation Losses).
- Representation rigidity: Fixed architecture and sequence-length settings per run (L=4, H=4, d_model=128; max sequence length 1024) (Section 5.1 Model architecture; Section 5.3 Training setting).
- Model sharing vs specialization: Separate per-dataset reporting; shared-weight or multi-task training not specified (Section 5.2 Datasets).
- Role of positional encoding: Central variable with explicit comparison of absolute vs rotary vs CRoPE variants (Section 5.1 Model architecture; Section 6.1 Training and validation Losses).

### 14. Final Classification

**Single-task, single-domain.** The experiments report losses on multiple text corpora but use the same model family and do not describe joint multi-task training, only "training and validation on the respective data splits" (Section 5.2 Datasets). The modality and domain remain natural language text, and the paper focuses on positional-encoding variants rather than cross-domain transfer.
