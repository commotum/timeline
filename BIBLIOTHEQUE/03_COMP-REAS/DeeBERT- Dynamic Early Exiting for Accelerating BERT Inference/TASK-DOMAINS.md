# DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference (Not specified in the paper)
Source: DeeBERT- Dynamic Early Exiting for Accelerating BERT Inference.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | natural language text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | class label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates DeeBERT on downstream NLP classification tasks (GLUE datasets), so the task intent is classification with label outputs. The input modality is natural language text (inferred), which implies 1D sequence structure, while the output is a single label (0D, inferred). The paper does not specify interface size limits for inputs, but inference uses a dynamic early-exit policy based on entropy and relies on constructed intermediate features.

## Evidence
### Task: classification
- "conduct experiments on six classification datasets from the GLUE benchmark (Wang et al., 2018): SST-2, MRPC, QNLI, RTE, QQP, and MNLI." (Section 4.1 Experimental Setup)
- "Large-scale pre-trained language models such as BERT have brought significant improvements to natural language processing (NLP) applications." (Section 1 Introduction)
- "(x, y) is the feature-label pair of a sample" (Section 3.1 DeeBERT at Fine-Tuning)
- "features provided by the intermediate transformer layers may suffice to classify some input samples." (Section 1 Introduction)
- "If the off-ramp is confident of the prediction, the result is returned; otherwise, the sample is sent to the next transformer layer." (Section 1 Introduction)
- Inference: Treated input as natural language text and 1D (t) based on the paper framing BERT as a language model for NLP applications; treated attention as Dynamic due to entropy-based early exiting; treated state as Constructed because intermediate transformer features are used; treated output as a 0D Fixed label based on feature-label pairs and classification datasets.
