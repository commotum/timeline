# Don't throw the baby out with the bathwater: How and why deep learning for ARC (Not specified in the paper)
Source: Don't throw the baby out with the bathwater- How and why deep learning for ARC.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Grid-to-grid transformation (ARC riddle solving) | ARC grid-pairs (input/output grids) and test input grid | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | output grid | 2D (x, y) | Not specified in the paper. |
| DSL function name/parameter prediction (ARC) | ARC riddle grids | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | DSL function names and parameters | 1D (t) (inferred) | Not specified in the paper. |
| Code generation for ARC riddle solutions | ARC riddle grids | 2D (x, y) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | code/program | 1D (t) (inferred) | Not specified in the paper. |
| Multimodal grid representation translation | grid representations (Base64 images; text; English descriptions; code) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | grid representations (Base64 images; text; English descriptions; code) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. |
| PCFG string-operation tasks with function name generation | strings (few-shot examples) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | strings; function names | 1D (t) (inferred) | Not specified in the paper. |
| Arithmetic and counting tasks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Arithmetic code generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | code | 1D (t) (inferred) | Not specified in the paper. |
| Chained boolean expression evaluation | boolean expressions (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | boolean values (inferred) | 0D (inferred) | Not specified in the paper. |
| Cellular automata and mathematical pattern grid tasks | ARC-style grid riddles (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | output grids (inferred) | 2D (x, y) (inferred) | Not specified in the paper. |
| Solution graph prediction | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | solution graphs | Not specified in the paper. | Not specified in the paper. |
| Language and reasoning tasks (NLP datasets) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Domain-specific scientific tasks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Multimodal and visual reasoning tasks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Programming and code-related tasks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper centers on ARC riddle solving as a grid-to-grid transformation task and adds auxiliary targets like DSL function-name prediction and (experimentally) code generation. It also reports training on additional task domains, including multimodal grid translation, PCFG string operations with function-name generation, arithmetic/counting, boolean expression evaluation, cellular automata/grid pattern tasks, solution-graph prediction, and broad NLP, scientific, multimodal reasoning, and programming datasets. Dimension coverage explicitly includes 2D grids and (inferred) 1D token sequences for strings/code, while dynamics are largely unspecified. Attention and state dynamics are only inferable for the ARC solver from the non-causal attention description and the emphasis on forming novel abstractions.

## Evidence
### Task: Grid-to-grid transformation (ARC riddle solving)
- "A set of training examples  $(x_j^{(i)}, y_j^{(i)})_{j=1}^{n_i}$ , where each  $x_j^{(i)}$  and  $y_j^{(i)}$  are input and output grids" (Section 2.1 Dataset description)
- "The objective is to infer a task-specific function  $f_i$  such that  $f_i(x_j^{(i)}) = y_j^{(i)}$" (Section 2.1 Dataset description)
- "Each grid is a 2D array  $x \in C^{h \times w}$" (Section 2.1 Dataset description)
- "the model should produce the correct transformed grid (toutput1) corresponding to the given test grid (tinput1)." (Figure 2 caption)
- "non-causal (unmasked) attention within the encoder, allowing each token to simultaneously attend to the entire input sequence." (Section 3.2.2 Attention and masking)
- "model's ability to create abstractions on the fly, to solve novel ARC tasks in the forward pass." (Section 1 Introduction)
- Inference: Attention Dynamic marked Static (inferred) because the encoder attends to the entire input sequence; State Dynamic marked Constructed (inferred) because the model is described as creating abstractions on the fly.

### Task: DSL function name/parameter prediction (ARC)
- "training the model to infer these underlying DSL function names and parameters from the input riddle grids, in addition to predicting the final output grids." (Section 3.1.4 Automatic Riddle Generators)
- "dual-prediction strategy, where models learn to predict both the output grid and the DSL function names" (Section 3.1.4 Automatic Riddle Generators)
- "Each grid is a 2D array  $x \in C^{h \times w}$" (Section 2.1 Dataset description)
- "non-causal (unmasked) attention within the encoder, allowing each token to simultaneously attend to the entire input sequence." (Section 3.2.2 Attention and masking)
- "model's ability to create abstractions on the fly, to solve novel ARC tasks in the forward pass." (Section 1 Introduction)
- Inference: Out Dimension set to 1D (t) (inferred) because DSL function names/parameters are token sequences; Attention Dynamic and State Dynamic inferred as Static/Constructed using the same encoder attention and abstraction statements.

### Task: Code generation for ARC riddle solutions
- "produce code as an intermediate output that can then be run to produce the output grid from the input" (Section 3.1.1 Direct output)
- "our experiments found that the added complexity of producing a syntactically correct, general-purpose solution introduced extra challenges" (Section 3.1.1 Direct output)
- Inference: Out Dimension set to 1D (t) (inferred) because the output is code, which is a token sequence.

### Task: Multimodal grid representation translation
- "translation between visual and symbolic representations of grids - including Base64 images, text, English descriptions, and code implementations." (Appendix A Training Dataset Construction and Composition)
- "We developed a specialized dataset for translating between: 1. Base64 images of grids 2. Text representations 3. English descriptions 4. Code implementations" (Appendix A.2 Custom Synthetic Datasets)
- Inference: In/Out Dimensions include 2D (x, y) for grid images and 1D (t) for text/English/code representations.

### Task: PCFG string-operation tasks with function name generation
- "expanded PCFG dataset with 100 distinct string operations" (Appendix A Training Dataset Construction and Composition)
- "Program synthesis components requiring function name generation" (Appendix A.2 Custom Synthetic Datasets)
- Inference: In/Out Dimensions set to 1D (t) (inferred) because the tasks are string operations with function-name outputs.

### Task: Arithmetic and counting tasks
- "We created fine-grained arithmetic and counting datasets that emphasized precise numerical operations and pattern recognition." (Appendix A.2 Custom Synthetic Datasets)

### Task: Arithmetic code generation
- "Arithmetic code generation" (Appendix A.2 Custom Synthetic Datasets, Multi-Task Integration Dataset)
- Inference: Out Dimension set to 1D (t) (inferred) because the task explicitly involves code generation.

### Task: Chained boolean expression evaluation
- "Chained boolean expression evaluation" (Appendix A.2 Custom Synthetic Datasets, Multi-Task Integration Dataset)
- Inference: Input inferred as boolean expressions (1D (t)) and output inferred as boolean values (0D) based on the evaluation task phrasing.

### Task: Cellular automata and mathematical pattern grid tasks
- "Cellular automata tasks within the ARC framework" (Appendix A.2 Custom Synthetic Datasets)
- "ARC riddle boards generated using mathematical equations" (Appendix A.2 Custom Synthetic Datasets)
- Inference: Input/output inferred as ARC-style grids (2D) because these are ARC framework riddle boards and cellular automata grids.

### Task: Solution graph prediction
- "Solution graph prediction tasks" (Appendix A.3 Data Sources and Generation)

### Task: Language and reasoning tasks (NLP datasets)
- "we integrate additional tasks requiring high levels of contextualization and reasoning—drawn from various NLP datasets—alongside the ARC data." (Section 3.1.2 Multi-task Training)

### Task: Domain-specific scientific tasks
- "A.1.2 Domain-Specific Scientific Datasets" (Appendix A.1 Public Datasets)
- "Arxiv Math Instruct (50k examples)" (Appendix A.1.2 Domain-Specific Scientific Datasets)

### Task: Multimodal and visual reasoning tasks
- "We integrated several multimodal reasoning datasets from the M3IT collection, specifically focusing on:" (Appendix A.1.3 Multimodal and Visual Reasoning Datasets)
- "CLEVR" (Appendix A.1.3 Multimodal and Visual Reasoning Datasets)

### Task: Programming and code-related tasks
- "A.1.4 Programming and Code-Related Datasets" (Appendix A.1 Public Datasets)
- "Magicoder Evolution Instruct (110K examples)" (Appendix A.1.4 Programming and Code-Related Datasets)
