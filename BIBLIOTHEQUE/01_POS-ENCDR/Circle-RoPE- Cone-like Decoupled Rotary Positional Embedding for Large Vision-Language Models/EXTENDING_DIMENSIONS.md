## 1. Basic Metadata
- Title: "CIRCLE-ROPE: CONE-LIKE DECOUPLED ROTARY POSI-TIONAL EMBEDDING FOR LARGE VISION-LANGUAGE MODELS" (Title)
- Authors: "Chengcheng Wang<sup>1\*</sup> Jianyuan Guo<sup>2\*</sup> Hongguang Li<sup>1\*</sup> Yuchuan Tian<sup>4</sup> Ying Nie<sup>1</sup> Chang Xu<sup>3†</sup> Kai Han<sup>1†</sup>" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary
The paper proposes "Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases" in RoPE-based VLMs to mitigate cross-modal positional bias between text and image tokens (Abstract).

## 3. Tasks Evaluated

### Task: MMMU (val)
- Task name: MMMU (val)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "MMMU <sub>val</sub> [29]" (Section 5.2, Table 2)
- Domain: Not specified.
- Quote: "MMMU <sub>val</sub> [29]" (Section 5.2, Table 2)

### Task: MMMU-Pro (overall/avg)
- Task name: MMMU-Pro (overall/avg)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "MMMU-Pro <sub>overall</sub> [30]" (Section 5.2, Table 2); "MMMU_Pro-avg" (Section 5.5, Table 5)
- Domain: Not specified.
- Quote: "MMMU-Pro <sub>overall</sub> [30]" (Section 5.2, Table 2)

### Task: MMMU (test)
- Task name: MMMU (test)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "evaluations performed on the MMMU $_{test}$ benchmark [29]." (Section 5.6)
- Domain: Not specified.
- Quote: "evaluations performed on the MMMU $_{test}$ benchmark [29]." (Section 5.6)

### Task: MathVista (mini)
- Task name: MathVista (mini)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "MathVista <sub>mini</sub> [15]" (Section 5.2, Table 2)
- Domain: Not specified.
- Quote: "MathVista <sub>mini</sub> [15]" (Section 5.2, Table 2)

### Task: MMStar
- Task name: MMStar
- Task type: Other (task type not specified in text)
- Dataset(s) used: "MMStar [3]" (Section 5.2, Table 2)
- Domain: Not specified.
- Quote: "MMStar [3]" (Section 5.2, Table 2)

### Task: AI2D (test)
- Task name: AI2D (test)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "AI2D [9]" (Section 5.2, Table 2); "AI2D_TEST" (Section 5.4, Table 4)
- Domain: Not specified.
- Quote: "AI2D [9]" (Section 5.2, Table 2)

### Task: RealWorldQA
- Task name: RealWorldQA
- Task type: Other (task type not specified in text)
- Dataset(s) used: "RealWorldQA [25]" (Section 5.2, Table 2)
- Domain: Not specified.
- Quote: "RealWorldQA [25]" (Section 5.2, Table 2)

### Task: InfoVQA
- Task name: InfoVQA
- Task type: Other (task type not specified in text)
- Dataset(s) used: "InfoVQA [17]" (Section 5.2, Table 2)
- Domain: Not specified.
- Quote: "InfoVQA [17]" (Section 5.2, Table 2)

### Task: ChartQA (test)
- Task name: ChartQA (test)
- Task type: Other (task type not specified in text)
- Dataset(s) used: "ChartQA_TEST" (Section 5.4, Table 4)
- Domain: Not specified.
- Quote: "ChartQA_TEST" (Section 5.4, Table 4)

### Task: MathVision
- Task name: MathVision
- Task type: Other (task type not specified in text)
- Dataset(s) used: "MathVision ↑" (Appendix A.1, Table 6)
- Domain: Not specified.
- Quote: "MathVision ↑" (Appendix A.1, Table 6)

## 4. Domain and Modality Scope
- Single domain? Not explicitly stated; the evaluation is framed around "Vision-Language Models (VLMs)" handling "both textual and visual inputs" (Section 1 Introduction).
- Multiple domains within the same modality? Not specified; the paper only states a "diverse range of datasets" without domain breakdown (Section 5.2).
- Multiple modalities? Yes: "both textual and visual inputs" in VLMs (Section 1 Introduction).
- Domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MMMU (val) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| MMMU-Pro (overall/avg) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| MMMU (test) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| MathVista (mini) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| MMStar | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| AI2D (test) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| RealWorldQA | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| InfoVQA | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| ChartQA (test) | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |
| MathVision | Yes (unified setup) | Yes (SFT on MAmmoTH-VL-Sub) | Not specified | "All experiments are conducted under a unified training setup." (Section 5.1); "For training, we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Section 5.1) |

## 6. Input and Representation Constraints
- Fixed or variable input resolution: "Image Resolution    | 512×512" (Appendix B, Table 7).
- Fixed patch size: Not specified.
- Fixed number of tokens: "We flatten the  $H \times W$  grid into a 1D sequence with  $N = H \times W$  points" (Section 4.1.2).
- Fixed dimensionality (e.g., strictly 2D): "image token indices are represented separately by width and height coordinates" (Section 4.1); "w and h correspond to the width and height of the image after tokenization." (Section 4.1).
- Text token indexing: "text tokens use 1D positional index equivalent to standard RoPE." (Section 4.1).
- Padding or resizing requirements: Not specified.
- Multi-image sequencing: "When the input contains multiple images, we explicitly encode their sequential order" (Section 4.3).

## 7. Context Window and Attention Structure
- Maximum sequence length: "Max Sequence Length | 4096" (Appendix B, Table 7).
- Sequence length fixed or variable: Not specified (only a maximum is given).
- Attention type: Not specified; the paper only mentions extracting "the attention matrix from the final decoder layer" (Section 5.6).
- Computational cost mechanisms (windowing, pooling, pruning): Not specified.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: "Rotary Position Embedding (RoPE)" and the proposed "Circle-RoPE" (Abstract); "we extend the M-RoPE mechanism, which represents image token indices by height-width coordinates" (Section 1 Introduction).
- Where applied: "cyclically switches between the M-RoPE [20] index and the Circle-RoPE index across different Transformer layers" (Section 4.2); "applies different RoPE variants across layers" (Abstract).
- Fixed vs modified/compared: The positional encoding is varied and ablated, e.g., "We conducted ablation studies on the parameters used in Circular Image Token Index Projection (CIP)" (Section 5.3) and "we systematically designed and evaluated four distinct encoding configurations" (Section 5.4).

## 9. Positional Encoding as a Variable
- Core research variable or fixed assumption? Core variable: "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Section 5.1).
- Multiple positional encodings compared? Yes: "Hard embedding," "Unordered embedding," and "Spatial embedding" are contrasted (Section 3 Preliminaries and Problem Analysis); LLaVA variants compare "Llava [1D-RoPE]," "Llava [M-RoPE]," and "Llava [Circle-RoPE]" (Section 5.5).
- PE choice claimed as not critical/secondary? Not stated.

## 10. Evidence of Constraint Masking
- Model sizes: Table 2 lists model scales such as "2B," "4B," "2B," "2.8B," "8B," "4.2B," "3B," and "3B" (Section 5.2, Table 2).
- Dataset sizes: "MAmmoTH-VL Instruct dataset (12M) [8]" and "MAmmoTH-VL-Sub (1M)" (Section 5.1).
- Attribution of gains: Improvements are attributed to positional encoding changes rather than scaling, since "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model" and the method improves "even with this reduced data size" (Section 5.1).
- Training tricks: "we exclusively update the parameters of the LLM component while keeping the parameters of the Vision-Language projection layers and the Vision Encoder frozen" (Section 5.1).

## 11. Architectural Workarounds
- Circular Image Token Index Projection (CIP) to decouple modalities: "project image token indices onto a circle in 3D space whose normal vector is aligned with the text vector" (Section 1 Introduction) to achieve a "fully decoupled encoding of text and image tokens" (Section 1 Introduction).
- Alternating Geometry Encoding (AGE) across layers: "cyclically switches between the M-RoPE [20] index and the Circle-RoPE index across different Transformer layers" to leverage "complementary strengths" (Section 4.2).
- Multi-image sequencing workaround: "translate each image's circular-encoding center along a fixed global axis" to encode order in multi-image input (Section 4.3).
- Fixed grid assumption for image tokens: image indices are based on "a regular grid" with "W = \{0, 1, \dots, w-1\} and H = \{0, 1, \dots, h-1\}" (Section 4.1).

## 12. Explicit Limitations and Non-Claims
- Adaptation cost: "even minor architectural modifications—such as altering the positional encoding—require substantial retraining with large-scale data for the model to adapt to the new positional distribution." (Appendix A.1).
- Compute/data limits: "Under limited compute and a relatively small SFT set, these gains are conservative rather than inflated." (Appendix A.1).
- Architecture scope limit: "adopting a more dissimilar backbone would likely incur a larger adaptation cost that is computationally prohibitive." (Appendix A.1).
- Data scope limit: "exclude all video data" (Section 5.1).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: Multimodal vision-language setting (text + images); no explicit cross-domain transfer claims.
> - Task structure: Multiple benchmark evaluations (MMMU, MMStar, MathVista, etc.) without task descriptions.
> - Representation rigidity: 2D grid image tokens, fixed image resolution 512×512, and a max sequence length of 4096.
> - Model sharing vs specialization: Unified training setup with shared weights across tasks; no task-specific heads described.
> - Role of positional encoding: Central experimental variable (Circle-RoPE vs M-RoPE/1D-RoPE, layer alternation, and ablations).

### 14. Final Classification
**Multi-task, single-domain**

The paper evaluates on "a diverse range of datasets" such as "MMMU <sub>val</sub> [29]" and "MathVista <sub>mini</sub> [15]" (Section 5.2, Table 2), indicating multiple tasks/benchmarks. All experiments are situated in "Vision-Language Models (VLMs)" with "both textual and visual inputs" (Section 1 Introduction), and no cross-domain transfer claims are made, so the scope is multi-task within a single multimodal domain.
