# PonderNet: Learning to Ponder (Not specified in the paper)
Source: PonderNet- Learning to Ponder.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (parity) | fixed-length numeric vector | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | binary label (odd/even) | 0D | Fixed |
| question answering (bAbI) | text stories and query tokens | 1D (t); 2D (x, y) | Capped | Static (inferred) | Constructed (inferred) | answer label (single word/token) | 0D | Fixed |
| classification (paired associative inference) | memory table of image embeddings; query triple embeddings | 2D (x, y); 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | ImageNet class-ID label | 0D | Fixed |

## Summary
The paper evaluates PonderNet on three supervised tasks: parity classification on fixed-length vectors, bAbI question answering over tokenized stories/queries, and paired associative inference with image-embedding memories. Inputs span 1D sequences and 2D table-like structures, with interface dynamics fixed (parity, PAI) or capped (bAbI), and outputs are 0D labels. Attention and state dynamics are not explicitly specified, but the described RNN/transformer architectures imply static attention and constructed state (inferred).

## Evidence
### Task: classification (parity)
- "input vectors had 64 elements" (Section 3.1 Parity)
- "target was 1 if there was an odd number of ones and 0 if there was an even number of ones." (Section 3.1 Parity)
- "simple RNN with a single hidden layer containing 128 tanh units" (Appendix B.1 Training and evaluation details)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed, inferred from the use of a simple RNN architecture with a hidden state. (Appendix B.1 Training and evaluation details)

### Task: question answering (bAbI)
- "bAbI question answering dataset (Weston et al., 2015), which consists of 20 different tasks." (Section 3.2 bAbI)
- "queries are a matrix of 128 x 11 tokens, and sentences are of size 128 x 320 x 11" (Appendix C.1 Training and evaluation details)
- "320 is the max number of stories, and 11 is the max sentence size." (Appendix C.1 Training and evaluation details)
- "We pad with zeros every query and group of stories that do not reach the max sentence and stories size." (Appendix C.1 Training and evaluation details)
- "every input (consisting of 'query' and 'stories') corresponds to a single answer" (Appendix C.1 Training and evaluation details)
- "We use the same architecture as described in Dehghani et al. (2018)." (Appendix C.2 Transformer architecture and hyperparameters)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed, inferred from the transformer-based architecture described. (Appendix C.2 Transformer architecture and hyperparameters)

### Task: classification (paired associative inference)
- "Paired associative inference task (PAI)" (Section 3.3 Paired associative inference)
- "memory with M=32 rows each one with 2 embeddings of size 1000." (Appendix D.1 PAI - Task details)
- "a concatenation of three image embedding vectors" (Appendix D.1 PAI - Task details)
- "a 3 x 1000 dimensional vector." (Appendix D.1 PAI - Task details)
- "targets represent the ImageNet class-ID of the matches." (Appendix D.1 PAI - Task details)
- "Memory was of size 32 * 2 * 1000" (Appendix D.1 PAI - Task details)
- "Queries were of size 1 * 3 * 1000" (Appendix D.1 PAI - Task details)
- "Target was of size 1" (Appendix D.1 PAI - Task details)
- "we augmented the transformer with a memory" (Appendix D.2 PAI - Architecture details)
- "The initial state  $h_0$  was a learnt embedding of the input." (Appendix D.2 PAI - Architecture details)
- Inference: Attention Dynamic = Static and State Dynamic = Constructed, inferred from the transformer-with-memory architecture and explicit internal state updates. (Appendix D.2 PAI - Architecture details)
