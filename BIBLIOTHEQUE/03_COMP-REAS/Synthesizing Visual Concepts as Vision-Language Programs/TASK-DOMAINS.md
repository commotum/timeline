# Synthesizing Visual Concepts as Vision-Language Programs (Year not specified in the paper)
Source: Synthesizing Visual Concepts as Vision-Language Programs.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Symbol grounding and structured visual description generation | Images from a task (few-shot examples) | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Symbol sets and structured object/property/action representations | 1D (t) (inferred) | Capped (inferred) |
| Visual rule induction via program synthesis | Labeled support images (image, binary label pairs) | 2D (x, y) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Executable boolean visual rule/program | 1D (t) (inferred) | Capped (inferred) |
| Query image binary classification using induced program | Held-out query images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Binary label / boolean prediction | 0D (inferred) | Fixed (inferred) |

## Summary
The paper describes a pipeline that first extracts symbolic visual structure, then induces an executable visual rule, and finally classifies held-out query images. The dominant modality is 2D image input, with binary supervision for support/query labels and symbolic program outputs treated as 1D structures (inferred). Dynamics are bounded in the reported setup (fixed/capped support-query counts and capped program depth), rather than open-ended streaming. Attention is dynamic during grounding and synthesis, while final query prediction uses a fixed synthesized program (static attention, inferred), and state is constructed through grounded symbols and programs.

## Evidence
### Task: Symbol grounding and structured visual description generation
- "The first stage of VLP establishes an interface between continuous visual inputs and discrete symbolic representations." (Section 3.2)
- "This process, which we refer to as *symbol grounding*, maps perceptual information from images into structured, typeconstrained symbols that form the atomic units for subsequent reasoning." (Section 3.2)
- "Each function  $(v \in \mathcal{V})$  takes as input an image (I) together with one or more sets of ground symbols  $(E_i)$  obtained during symbol grounding (Sec. 3.2), and outputs a nested symbolic representation s:" (Section 3.3)
- Inference: `2D (x, y)` is inferred from repeated image-based inputs; `Capped` dynamics are inferred from fixed symbol-request interfaces ("Return exactly {n} objects" in Section G.2 and fixed counts in Table 8); `Dynamic` attention is inferred from "the vocabulary is dynamically adjusted based on the task at hand" (Section 3.2); `Constructed` state is inferred because the system builds structured symbolic representations from raw images.

### Task: Visual rule induction via program synthesis
- "We formulate inductive visual reasoning as the task of discovering a latent visual rule that explains a set of example images (denoted as few-shot examples in the remainder)." (Section 3.1)
- "Given the PCFG, VLP performs program synthesis by searching for an executable program p that best explains the task  $\mathcal{X}$ ." (Section 3.5)
- "The maximum program depth is 4 for Bongard-OpenWorld and Bongard HOI, and 6 for COCOLogic and CLEVR-Hans3." (Section 4, Experimental Setup)
- Inference: `2D (x, y); 0D` input dimension is inferred from image-plus-binary-label task definition (Section 3.1); `Capped` input dynamics are inferred from finite support-set construction (Section 4, Data); `Dynamic` attention is inferred from search over candidate programs (Section 3.5); `Constructed` state is inferred from synthesis of an explicit executable program; `1D (t)` output dimension is inferred by treating the synthesized rule as a symbolic expression; `Capped` output dynamics are supported by explicit maximum program depth.

### Task: Query image binary classification using induced program
- "Each task is additionally associated with a set of held-out query samples for evaluation." (Section 3.1)
- "Each program transforms an input image  $I_i$  into a boolean prediction  $\hat{y}_i = p(I_i)$ , whose correctness is evaluated against the ground-truth label  $y_i$ ." (Section 3.5)
- "Across all evaluations, model performance is measured using balanced accuracy, reflecting each model's ability to correctly classify the query (test) images." (Section 4, Experimental Setup)
- Inference: `2D (x, y)` input dimension is inferred from query-image classification; `Capped` input dynamics are inferred from finite query-set construction per task (Section A dataset descriptions); `Static` attention is inferred because the selected top-ranked program is fixed for query evaluation; `Constructed` state is inferred because predictions depend on the synthesized program; `0D` output and `Fixed` output dynamics are inferred from boolean/binary prediction outputs.
