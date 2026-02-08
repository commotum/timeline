# RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING (Year not specified in the paper)
Source: Rethinking Positional Encoding in Language Pre-training (TUPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked language modeling (pre-training prediction) | Sub-word token sequences | 1D (t) | Capped | Static (inferred) | Direct (inferred) | Predicted tokens at masked positions | 1D (t) (inferred) | Capped (inferred) |
| Sentence-level downstream prediction (GLUE classification/regression) | Sentence token sequences (single/pair) | 1D (t) | Capped | Static (inferred) | Direct (inferred) | Sentence-level labels/scores | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers language modeling and sentence-level language understanding tasks in text sequences. Inputs are token sequences with a capped maximum length (512), so the supported input domain is 1D (t), Capped. Output coverage includes token-level sequence prediction for masked language modeling and sentence-level scalar decisions for downstream GLUE evaluation. Attention and state dynamics are not labeled explicitly in the paper; from the described fixed-window Transformer setup, they are Static and Direct (inferred).

## Evidence
### Task: Masked language modeling (pre-training prediction)
- "We use masked language modeling as the objective of pre-training." (Section B EXPERIMENTAL DETAILS)
- "We train the models for 1000k steps where the batch size is 256 and the maximum sequence length is 512." (Section B EXPERIMENTAL DETAILS)
- Inference: `Attention Dynamic = Static`, `State Dynamic = Direct`, `Out Dimension = 1D (t)`, and `Out Dynamics = Capped` are inferred from the fixed-sequence Transformer setup and capped context: "the maximum sequence length is 512" and the self-attention formulation over sequence positions "z_i^l = \sum_{j=1}^n" (Section B EXPERIMENTAL DETAILS; Section 2.1 Attention module).

### Task: Sentence-level downstream prediction (GLUE classification/regression)
- "Its contextual representation will be used to make predictions in the sentence-level downstream tasks after pre-training" (Section 3.2 Untie the [CLS] Symbol from Positions)
- "Particularly, we use nine tasks in GLUE, including CoLA, RTE, MRPC, STS, SST, QNLI, QQP, and MNLI-m/mm." (Section B EXPERIMENTAL DETAILS)
- Inference: `Out Dimension = 0D` and `Out Dynamics = Fixed` are inferred because predictions are sentence-level per example, supported by sentence-level usage and scalar evaluation metrics: "we report Matthews correlation for CoLA, Pearson correlation for STS-B, and accuracy for other tasks." (Section B EXPERIMENTAL DETAILS).
