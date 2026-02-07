# Neural Module Networks (Not specified in the paper.)
Source: Neural Module Networks (NMN).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Visual question answering (answer classification) | images; natural-language questions | 2D (x, y); 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | answer labels | 0D | Fixed (inferred) |

## Summary
The paper covers visual question answering over images paired with natural-language questions, evaluated on the VQA natural image dataset and the SHAPES synthetic dataset. Inputs span 2D image grids and 1D question sequences, producing 0D answer labels via classification; input size dynamics are not explicitly specified. The model dynamically composes attention-based modules and passes intermediate attentions, indicating Dynamic attention and Constructed state (inferred), with a fixed answer set (inferred) for outputs.

## Evidence
### Task: Visual question answering (answer classification)
- "given an image and an associated question (e.g. where is the dog?), we wish to predict a corresponding answer" (Section 1. Introduction)
- "w is a natural-language question" (Section 4. Neural module networks for visual QA)
- "x is an image" (Section 4. Neural module networks for visual QA)
- "y is an answer" (Section 4. Neural module networks for visual QA)
- "we have treated answer prediction as a pure classification problem: the model selects from the set of answers observed during training" (Section 4.3. Answering natural language questions)
- "uses these structures to dynamically instantiate modular networks" (Abstract)
- "messages passed between modules may be raw image features, attentions, or classification decisions" (Section 1. Introduction)
- Inference: Attention Dynamic = Dynamic and State Dynamic = Constructed because the model "dynamically instantiate[s] modular networks" and passes intermediate "attentions" between modules. Out Dynamics = Fixed because answer prediction is a classification over a fixed set of observed answers.
