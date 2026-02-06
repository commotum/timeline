# Adaptive patch selection to improve Vision Transformers through Reinforcement Learning (2025)
Source: Adaptive Patch Selection for ViTs via Reinforcement Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | images | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Direct (inferred) | class labels (inferred) | 0D (inferred) | Fixed (inferred) |
| patch selection | attention values per image patch | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Direct (inferred) | selected image patches | 2D (x, y) (inferred) | Capped (inferred) |

## Summary
AgentViT is evaluated on image classification and uses an RL agent to select patches that the ViT processes. The paper therefore spans classification outputs (class labels) and a patch-selection control step that chooses subsets of image patches. The evidence supports 2D spatial inputs with fixed image sizes in experiments and variable (capped) patch subsets, with dynamic attention selection and direct state mappings.

## Evidence
### Task: classification
- "We tested AgentViT on CIFAR10, FashionMNIST, and Imagenette<sup>+</sup> (which is a subset of ImageNet) in the image classification task" (Abstract)
- "CIFAR10 [24] consists of 60,000 RGB images of size  $32 \times 32$ pixels, categorized into 10 classes." (Section 4.1 Experimental setup)
- Inference: Input treated as 2D (x, y) and Fixed because datasets are fixed-size images; output treated as class labels (0D, Fixed) because images are categorized into classes; attention marked Dynamic and state Direct because the agent selects patches per image within an image classification setup without persistent state. (Abstract; Section 3.1 Schematic workflow of AgentViT; Section 4.1 Experimental setup)

### Task: patch selection
- "an agent that selects the most important patches to improve the learning of a ViT." (Abstract)
- "the state of the environment is represented by attention values obtained from an image processed by the first attention layer of the ViT." (Section 3.2 State)
- "The action space  $\mathcal{A}$  of AgentViT consists of N discrete actions, each corresponding to the selection of a specific patch." (Section 3.3 Action)
- Inference: Treated patch/attention domains as 2D (x, y) and Fixed because they are derived from an input image decomposed into N patches; marked attention Dynamic because the agent chooses which patches to retain; marked state Direct because the state is the attention-value vector; output treated as 2D and Capped because a subset of patches is selected. (Section 3.2 State; Section 3.3 Action)

---

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
classification,images,"2D (x, y) (inferred)","Fixed (inferred)","Dynamic (inferred)","Direct (inferred)","class labels (inferred)","0D (inferred)","Fixed (inferred)"
patch selection,attention values per image patch,"2D (x, y) (inferred)","Fixed (inferred)","Dynamic (inferred)","Direct (inferred)",selected image patches,"2D (x, y) (inferred)","Capped (inferred)"
