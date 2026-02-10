# The Lessons of Developing Process Reward Models in Mathematical Reasoning (Not specified in the paper.)
Source: The Lessons of Developing Process Reward Models....md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Step-wise correctness classification for mathematical reasoning | Math problem plus ordered solution steps/paragraphs (text) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Per-step correctness labels/scores | 1D (t) (inferred) | Capped (inferred) |
| First erroneous-step localization | Ordered multi-step solution text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | First erroneous step or all-steps-correct decision | 0D (inferred) | Fixed (inferred) |
| Best-of-N response reranking/selection | N candidate solution responses scored by a PRM | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Highest-scored response among candidates | 0D (inferred) | Fixed (inferred) |
| PRM-guided next-step selection in greedy search | Current partial solution plus N candidate next steps | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Selected next step for subsequent expansion | 0D (inferred) | Fixed (inferred) |

## Summary
The paper focuses on text-based mathematical reasoning trajectories, with PRMs primarily used for step-wise process verification and error detection. It also applies PRM scores to decision tasks that select one item from multiple candidates (Best-of-N response selection and greedy next-step selection). The justified dimensions are mainly 1D (t) for sequential reasoning inputs and 0D for single decision outputs. Dynamics/Attention/State labels are mostly inferred from the described scoring-and-selection procedures, with capped candidate sets (e.g., N=8 or N=64) and predominantly static/direct PRM scoring, plus dynamic/constructed behavior in the iterative greedy-search loop.

## Evidence
### Task: Step-wise correctness classification for mathematical reasoning
- "PRMs provide fine-grained supervision by evaluating the correctness of intermediate reasoning steps." (Section 3.1.1)
- "we employ cross-entropy loss on the tokens at the end of each step to train the binary classification task." (Section 4.1)
- Inference: 1D (t), Capped, Static, and Direct are inferred because the paper describes ordered step/paragraph processing and step-end token scoring, without describing retrieval or external memory state (Sections 2.1, 4.1, C).

### Task: First erroneous-step localization
- "PROCESSBENCH (Zheng et al., 2024) measures the capability of models to identify erroneous steps in mathematical reasoning." (Section 2.2, PROCESSBENCH)
- "Models are required to identify the first step that contains an error or conclude that all steps are correct." (Section 2.2, PROCESSBENCH)
- "Following the evaluation methods for PRMs in PROCESSBENCH, we locate the first erroneous step from predict scores yielded by PRMs." (Section 2.2, PROCESSBENCH)
- Inference: 1D (t) input and 0D Fixed output are inferred because localization is produced as a single decision per ordered step sequence.

### Task: Best-of-N response reranking/selection
- "we employed the Best-of-N (BoN) sampling strategy for evaluation, which selects the highest-scored response from N candidates according to a PRM." (Section 2.2, Best-of-N)
- "we sampled eight responses (i.e., N=8)" (Section 2.2, Best-of-N)
- "Each candidate response is scored using the product of all the individual scores of each step within the response" (Section 2.2, Best-of-N)
- Inference: 1D (t), Capped, Static, and Direct are inferred because PRM scoring is applied over provided response sequences with finite candidate pools, producing one selected response (0D, Fixed).

### Task: PRM-guided next-step selection in greedy search
- "We further integrate PRM with greedy search by generating N candidate steps at each step, evaluating these candidates using PRM scoring, and selecting the highest-scoring step for subsequent expansion." (Section A, PRM Guided Search)
- "We choose the highest-scoring candidate at each step which the score predicted by PRM represents the correctness of this step." (Section A, PRM Guided Search)
- Inference: 1D (t) and Capped are inferred from sequential step candidates with finite N; Dynamic attention and Constructed state are inferred from iterative runtime step selection and trajectory expansion.
