# REALM: Retrieval-Augmented Language Model Pre-Training (Not specified in the paper.)
Source: REALM- Retrieval-Augmented Language Model Pre-Training.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked token prediction (MLM) | masked text passage (tokens); retrieved documents from knowledge corpus | 1D (t) | Capped | Dynamic | Constructed (inferred) | missing tokens (mask predictions) | 1D (t) | Not specified in the paper. |
| open-domain question answering | question text (tokens); retrieved documents from knowledge corpus | 1D (t) | Capped | Dynamic | Constructed (inferred) | answer string (contiguous token span) | 1D (t) | Not specified in the paper. |

## Summary
The paper covers masked language modeling pre-training and open-domain question answering fine-tuning, both operating over textual sequences with retrieval from a large document corpus. Inputs and outputs are 1D token sequences, with input sizes capped by fixed retrieval candidate counts and document chunking limits. Attention is dynamic because the model retrieves documents and attends over them at runtime. State dynamics are marked Constructed (inferred) because retrieval introduces intermediate document state beyond the raw input.

## Evidence
### Task: masked token prediction (MLM)
- "In its basic form, an MLM is trained to predict the missing tokens in an input text passage." (Section 2. Background)
- "The model uses its representation of the masked input x to predict the token that should go in each mask." (Section 2. Background)
- "the language model uses the retriever to retrieve documents<sup>1</sup> from a large corpus such as Wikipedia" (Introduction)
- "and then attends over those documents to help inform its prediction." (Introduction)
- "For each example, we retrieve and marginalize over 8 candidate documents, including the null document  $\varnothing$ ." (Section 4.3 Implementation Details, Pre-training)
- Inference: State Dynamic marked Constructed because the model retrieves documents and then attends to them before predicting, indicating constructed intermediate state beyond the raw input. (Supported by "the language model uses the retriever..." and "and then attends over those documents...")

### Task: open-domain question answering
- "For fine-tuning, the task is Open-QA: x is a question, and y is the answer." (Section 3.1. REALM's generative process)
- "For Open-QA fine-tuning, we wish to produce the answer string y." (Section 3.2. Model architecture)
- "given a question x, retrieve potentially relevant documents z from the corpus  $\mathcal{Z}$ , and then extract an answer y from the documents" (Section 2. Background)
- "we will assume that the answer y can be found as a contiguous sequence of tokens in some document z." (Section 3.2. Model architecture)
- "Documents are greedily split into chunks of up to 288 BERT wordpieces" (Section 4.3 Implementation Details, Fine-tuning)
- "During fine-tuning inference, we consider the top-5 candidates" (Section 4.3 Implementation Details, Fine-tuning)
- Inference: State Dynamic marked Constructed because the model retrieves documents from a corpus and conditions on them to produce the answer. (Supported by "given a question x, retrieve potentially relevant documents...")
