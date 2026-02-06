# Dense Passage Retrieval for Open-Domain Question Answering (Not specified in the paper.)
Source: Dense Passage Retrieval for Open-Domain Question Answering.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Passage retrieval (top-k selection) | Question tokens; passage tokens (corpus) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Top-k passages (text tokens) | 1D (t) (inferred) | Fixed (inferred) |
| Extractive question answering (answer span extraction) | Question tokens; retrieved passages (text tokens) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer span tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper covers dense passage retrieval and extractive open-domain question answering in a two-stage pipeline. Both tasks operate over text sequences (1D (t)), with fixed-length passages and top-k retrieval leading to capped or fixed interface sizes. Retrieval dynamically selects relevant passages using constructed dense representations, while the reader consumes a fixed retrieved set and constructs BERT-based representations to score answer spans.

## Evidence
### Task: Passage retrieval (top-k selection)
- "Formally speaking, a retriever  $R:(q,\mathcal{C})\to\mathcal{C}_{\mathcal{F}}$ is a function that takes as input a question q and a corpus C" (Section 2 Background)
- "retrieve efficiently the top k passages relevant to the input question" (Section 3 Dense Passage Retriever)
- Inference: Dimensions, dynamics, attention, and state are inferred from tokenized passages, fixed-length passages, top-k selection, and dense indexing. Supporting text: "each passage  $p_i$  can be viewed as a sequence of tokens  $w_1^{(i)}, w_2^{(i)}, \cdots, w_{|p_i|}^{(i)}$" (Section 2 Background); "split each article into multiple, disjoint text blocks of 100 words as *passages*" (Section 4.1 Wikipedia Data Pre-processing); "retrieve k passages of which vectors are the closest to the question vector" (Section 3.1 Overview); "maps any text passage to a d-dimensional real-valued vectors and builds an index" (Section 3.1 Overview)

### Task: Extractive question answering (answer span extraction)
- "the extractive QA setting, in which the answer is restricted to a span appearing in one or more passages in the corpus." (Section 2 Background)
- "the task is to find a span  $w_s^{(i)}, w_{s+1}^{(i)}, \cdots, w_e^{(i)}$  from one of the passages  $p_i$" (Section 2 Background)
- "Given the top k retrieved passages (up to 100 in our experiments), the reader assigns a passage selection score to each passage." (Section 6.1 End-to-end QA System)
- "it extracts an answer span from each passage and assigns a span score." (Section 6.1 End-to-end QA System)
- Inference: Dimensions, dynamics, attention, and state are inferred from token-sequence passages, fixed top-k inputs, and BERT-based passage representations. Supporting text: "each passage  $p_i$  can be viewed as a sequence of tokens  $w_1^{(i)}, w_2^{(i)}, \cdots, w_{|p_i|}^{(i)}$" (Section 2 Background); "Given the top k retrieved passages (up to 100 in our experiments)" (Section 6.1 End-to-end QA System); "be a BERT (base, uncased in our experiments) representation for the *i*-th passage" (Section 6.1 End-to-end QA System)
