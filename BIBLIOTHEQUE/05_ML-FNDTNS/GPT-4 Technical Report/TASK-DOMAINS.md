# GPT-4 Technical Report (2023)
Source: GPT-4 Technical Report.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (language modeling) | text tokens (document) | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | next token | 0D | Fixed (inferred) |
| Dialogue/text response generation | text prompts or dialogue turns | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | generated text (dialogue/emails) | 1D (t) | Capped (inferred) |
| Machine translation | text (source language) | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | text (target language) | 1D (t) | Capped (inferred) |
| Text summarization | text; document images | 1D (t); 2D (x, y) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | summary text | 1D (t) | Capped (inferred) |
| Text editing / copyediting (formatting) | text drafts | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | edited text | 1D (t) | Capped (inferred) |
| Free-response question answering (exam-style) | question prompts (text) | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | free-response text | 1D (t) | Capped (inferred) |
| Multiple-choice question answering | question text + answer options | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer choice (letter) | 0D | Fixed (inferred) |
| Arithmetic problem solving | math word problems (text) | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | numeric answer | 0D | Fixed (inferred) |
| Text classification (content categories) | text content | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | category label | 0D | Fixed (inferred) |
| Code synthesis (Python functions / coding tasks) | natural language programming prompts | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | code (tokens) | 1D (t) | Capped (inferred) |
| Code vulnerability analysis | source code | 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | vulnerability findings (text) | 1D (t) | Capped (inferred) |
| Multimodal visual question answering / image understanding | images + text | 2D (x, y); 1D (t) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | text answers/descriptions | 1D (t) | Capped (inferred) |

## Summary
GPT-4 is described as a text- and image-conditioned model that supports core language-modeling and a wide range of text tasks such as dialogue generation, question answering, classification, arithmetic, summarization, editing, translation, and coding. The paper also documents multimodal image+text prompting that yields textual answers, including document image reading and visual question answering. Inputs are primarily 1D text sequences with 2D images for vision tasks, while outputs are text sequences or single labels/numbers. Where the report notes a limited context window and response-length limits, the corresponding input/output dynamics are marked as capped (inferred); attention and state dynamics are not explicitly specified.

## Evidence
### Task: Next-token prediction (language modeling)
- "GPT-4 is a Transformer-based model pre-trained to predict the next token in a document." (Abstract)
- Inference: In Dynamics = Capped (inferred) based on "has a limited context window" (1 Introduction). Out Dynamics = Fixed (inferred) because the task is next-token prediction (single token).
### Task: Dialogue/text response generation
- "GPT-4 can generate plausibly realistic and targeted content, including news articles, tweets, dialogue, and emails." (System Card 2.5 Disinformation and Influence Operations)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Machine translation
- "applications, such as dialogue systems, text summarization, and machine translation." (1 Introduction)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Text summarization
- "GPT-4 was used in the following ways: to help us iterate on LaTeX formatting; for text summarization; and as a copyediting tool." (Acknowledgements)
- "User Below is part of the InstuctGPT paper. Could you read and summarize it to me?" (GPT-4 visual input example, Pixel to Paper Summaries)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Text editing / copyediting (formatting)
- "GPT-4 was used in the following ways: to help us iterate on LaTeX formatting; for text summarization; and as a copyediting tool." (Acknowledgements)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Free-response question answering (exam-style)
- "Exam questions included both multiple-choice and free-response questions; we designed separate prompts for each format." (4 Capabilities)
- "For each free-response section, we gave the model the free-response question's prompt as a simple instruction-following-style request" (Appendix A.3 Prompting: free-response)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Multiple-choice question answering
- "For multiple-choice questions, we present all answers (ABCD) to the model and ask it to choose the letter of the answer" (Table 2 caption)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Fixed (inferred) because the output is a single choice letter from a fixed option set (ABCD).
### Task: Arithmetic problem solving
- "including question answering, arithmetic, and classification." (System Card 1 Introduction)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Fixed (inferred) because the output is a single numeric answer.
### Task: Text classification (content categories)
- "Our rule-based reward models (RBRMs) are a set of zero-shot GPT-4 classifiers." (Model-Assisted Safety Pipeline)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Fixed (inferred) because classification outputs are single category labels.
### Task: Code synthesis (Python functions / coding tasks)
- "HumanEval dataset [43], which measures the ability to synthesize Python functions of varying complexity." (3.2 Scaling of Capabilities on HumanEval)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Code vulnerability analysis
- "Below is an example that demonstrates the model's dual-use capability of finding code vulnerabilities:" (2.8 Cybersecurity)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).
### Task: Multimodal visual question answering / image understanding
- "GPT-4 accepts prompts consisting of both images and text" (4.1 Visual Inputs)
- "the model generates text outputs given inputs consisting of arbitrarily interlaced text and images." (4.1 Visual Inputs)
- Inference: In Dynamics = Capped (inferred) from "has a limited context window" (1 Introduction). Out Dynamics = Capped (inferred) from "we discovered a bug that limited response length" (Appendix A.2 Prompting: multiple-choice).