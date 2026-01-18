## 1. Basic Metadata
- Title: "TACKLING THE ABSTRACTION AND REASONING COR-PUS WITH VISION TRANSFORMERS: THE IMPORTANCE OF 2D REPRESENTATION, POSITIONS, AND OBJECTS" (Title page)
- Authors: "Anonymous authors" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): "Paper under double-blind review" (Title page)

## 2. One-Sentence Contribution Summary
The paper introduces a ViT-based ARC solver, stating that "we propose VITARC, a ViT-style architecture that unlocks some of the visual reasoning capabilities required by the ARC" (Abstract).

## 3. Tasks Evaluated
- Task: Abstraction and Reasoning Corpus (ARC) tasks (400 public tasks). Task type: Generation; Reasoning / relational. Dataset(s): public ARC tasks with synthetic pairs from RE-ARC. Domain: synthetic 2D grids / small 2D images. Quotes: "Each ARC task involves transforming input grids into output grids by identifying a hidden mapping often requiring significant reasoning beyond mere pattern matching" (Section 1 Introduction); "As seen in Figure 2, ARC tasks are *generative* and require mapping an input image to an output image" (Section 3); "To evaluate ViT's reasoning capabilities comprehensively, we treat each of the 400 public training ARC tasks as an individual AVR problem. We generate a dataset of 1 million input-output pairs per task using the RE-ARC generator (Hodel, 2024)" (Section 3.1 Data); "In its original framing, an ARC task requires solving a program synthesis problem over small 2D images using a few input-output training pairs" (Abstract).

## 4. Domain and Modality Scope
- Single domain? Yes; ARC grids: "Each ARC task involves transforming input grids into output grids" (Section 1 Introduction) and "small 2D images" (Abstract).
- Multiple domains within the same modality? Not indicated; evaluation is within ARC grid tasks only (same evidence as above).
- Multiple modalities? No; "AVR tasks do not include any text or background knowledge. Instead, they focus purely on visual abstraction and pattern recognition" (Section 1 Introduction).
- Domain generalization or cross-domain transfer? Not claimed; "VITARC solves task-specific instances of ARC in a data-driven approach, treating each ARC task independently" (Conclusion) and "This method does not fully solve ARC, which requires the ability to generalize across different tasks" (Conclusion).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ARC task (each of 400 public training tasks) | No; trained separately per task from scratch. | No (trained from scratch per task). | Not specified. | "To evaluate ViT's reasoning capabilities comprehensively, we treat each of the 400 public training ARC tasks as an individual AVR problem. We generate a dataset of 1 million input-output pairs per task using the RE-ARC generator (Hodel, 2024) and train all of our models (the vanilla ViT and ViTARC models) in a supervised manner from scratch" (Section 3.1 Data); "VITARC solves task-specific instances of ARC in a data-driven approach, treating each ARC task independently" (Conclusion). |

## 6. Input and Representation Constraints
- Variable grid sizes: "Because image dimensions may vary across instances of the same task and even between the input and output grids of the same instance" (Section 3).
- Pixel-level tokens and fixed patch size: "To achieve the required pixel-level precision for the ARC task, we employ a patch size of  $1 \times 1$ , effectively treating each pixel as an independent input token" (Section 3).
- Fixed maximum token length: "To handle variable-sized grids, the flattened list of tokens is padded to a fixed maximum length" (Section 3).
- 2D padding before flattening: "we implemented 2D padding, where <pad> tokens are applied to the image first before being flattened in raster order into a sequence for transformer processing" (Section 4).
- Fixed 2D coordinate schema: "our setup now operates on fixed-size, two-dimensional input-output pairs that are aligned with a universal (x,y) coordinate system" (Section 4).
- Explicit padding/border tokens and fixed maximum size: "An ARC input image is first tokenized into pixels and padded with visual tokens including end-of-grid tokens that mark the end of the image grid, newline tokens that indicate the end of one row, and pad tokens which are used to pad the image into a fixed maximum size" (Figure 1 caption).

## 7. Context Window and Attention Structure
- Maximum sequence length: "L is the maximum input length" (Section 3).
- Fixed or variable sequence length: variable inputs but fixed-length processing via padding: "Because image dimensions may vary across instances of the same task and even between the input and output grids of the same instance," and "the flattened list of tokens is padded to a fixed maximum length" (Section 3).
- Attention type: Not specified beyond standard Transformer attention; decoder cross-attention is used: "We introduce a decoder with cross-attention using the same positional encoding and attention mechanisms of the encoder" (Section 3).
- Computational cost management mechanisms (windowing/pooling/pruning): Not specified.

## 8. Positional Encoding (Critical Section)
- Absolute PE (learned 1D) in vanilla ViT, applied at input: "Following the standard ViT implementation of Dosovitskiy et al. (2021), the Absolute Positional Encoding (APE) is calculated as a learnable 1D encoding" (Section 3), and "the absolute positional encoding,  $\mathbf{E}_{pos_i}$ , is directly added to the input embedding,  $\mathbf{E}_{p_i}$ , so that it adjusts the token's representation without overwhelming its semantic content" (Section 4.2).
- 2D sinusoidal APE (non-learned) added to tokens: "we implement a (non-learned) 2D sinusoidal APE for VITARC" (Section 4) and "2D Positional Encodings and Object Positional Encodings are then added to each token before being passed into the transformer" (Figure 1 caption).
- PEmixer (input-level mixing): "we modify Equation (1) by learning weight vectors for the encodings" (Section 5).
- 2D-RPE (attention bias): "we adapt the Relative Positional Encoding (RPE) approach from ALiBi (Press et al., 2021) and extend it to 2D" and "ALiBi introduces additive positional biases to the attention scores based on the relative positions of tokens" (Section 5).
- OPE (object index in positional encoding): "We extend the 2D sinusoidal APE defined in Equation (9) by introducing the object index o as an additional component to the pixel coordinates (x, y)" (Section 5).
- Fixed vs modified/ablated: "The final encoding combines all three components: 2DAPE, 2DRPE, and OPE" (Section 5.1), and "Ablated components are prefixed as - and ablate the full model to the left, i.e., -BorderTokens is an ablation of this component from ViTARC-VT and each of -PEmixer, -2D-RPE, and -OPE ablate these respective components from ViTARC" (Figure 7 caption).

## 9. Positional Encoding as a Variable
- Core research variable: "A key finding of our work is that positional information plays a critical role in visual reasoning tasks" (Conclusion) and "Positional Information further enhances ViT reasoning abilities: We improved ViTARC's spatial awareness by learning to combine absolute, relative, and *object* positional information" (Introduction).
- Multiple positional encodings compared: "All three contribute to the overall improvement, with 2D-RPE providing the largest gain, followed by PEmixer and OPE" (Section 5.1), and ablations are explicit (Figure 7 caption).
- PE not critical/secondary? Not claimed; the paper calls positional information "critical" (Conclusion).

## 10. Evidence of Constraint Masking
- Model size: "The ViT baseline consists of three layers with eight attention heads and a hidden dimension of 128" (Section 3.1 Hyperparameters and training protocol).
- Dataset size: "We generate a dataset of 1 million input-output pairs per task using the RE-ARC generator (Hodel, 2024)" (Section 3.1 Data).
- Data scaling alone insufficient: "This is despite using a training set of one million examples per task" (Introduction).
- Gains attributed to representation/positional architecture: "A 2D visual representation significantly boosts ViT reasoning performance" (Introduction) and "Positional Information further enhances ViT reasoning abilities" (Introduction).

## 11. Architectural Workarounds
- Pixel-level and spatially-aware tokenization: "we use a pixel-level input representation, design a spatially-aware tokenization scheme" (Abstract).
- Encoder-decoder with cross-attention for pixel-wise outputs: "We introduce a decoder with cross-attention using the same positional encoding and attention mechanisms of the encoder" (Section 3) and "we employ a patch size of  $1 \times 1$ , effectively treating each pixel as an independent input token" (Section 3).
- 2D padding and pad tokens to preserve 2D structure: "we implemented 2D padding, where <pad> tokens are applied to the image first before being flattened" (Section 4) and pad tokens create "a fixed maximum size" (Figure 1 caption).
- Border tokens to define grid boundaries: "we introduce *border tokens* to explicitly define the grid boundaries" (Section 4).
- Positional enhancements (2D APE, PEmixer, 2D-RPE, OPE with object indices): "we implement a (non-learned) 2D sinusoidal APE for VITARC" (Section 4); "we modify Equation (1) by learning weight vectors for the encodings" (Section 5); "we adapt the Relative Positional Encoding (RPE) approach from ALiBi (Press et al., 2021) and extend it to 2D" (Section 5); "introducing the object index o as an additional component to the pixel coordinates (x, y)" (Section 5); and "For simplicity, we adopt bounding box segmentation to derive the object index o" (Section 5).

## 12. Explicit Limitations and Non-Claims
- "Specifically, around 10% of ARC tasks have less than 5% of test instances solved, even after training on a large dataset containing one million examples per task" (Section 4.2 Analysis).
- "tasks involving complex visual structures, such as concave shapes, holes, or subgrids, are consistently problematic" (Section 4.2 Analysis).
- "OPE, while effective in specific tasks, is not consistently reliable" (Section 5.1 Results).
- "VITARC solves task-specific instances of ARC in a data-driven approach, treating each ARC task independently" (Conclusion).
- "This method does not fully solve ARC, which requires the ability to generalize across different tasks" (Conclusion).

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single synthetic grid domain (ARC) with image-only inputs.
> - Task structure: Many ARC tasks, each treated as an independent input-output mapping problem.
> - Representation rigidity: Pixel-level tokens with fixed-size 2D grids via padding/border tokens and fixed maximum length.
> - Model sharing vs specialization: Task-specific models trained from scratch per ARC task; no shared weights.
> - Role of positional encoding: Central design variable with multiple PE variants (2D APE, 2D-RPE, OPE, PEmixer) compared.

## 14. Final Classification
**Multi-task, single-domain.** The evaluation spans many ARC tasks: "To evaluate ViT's reasoning capabilities comprehensively, we treat each of the 400 public training ARC tasks as an individual AVR problem" (Section 3.1 Data), and all tasks remain within the same grid-to-grid visual reasoning domain where "Each ARC task involves transforming input grids into output grids" (Section 1 Introduction). The paper also emphasizes task-specific training, "treating each ARC task independently" (Conclusion), indicating a constrained single-domain setting rather than cross-domain or unrestrained multi-task learning.
