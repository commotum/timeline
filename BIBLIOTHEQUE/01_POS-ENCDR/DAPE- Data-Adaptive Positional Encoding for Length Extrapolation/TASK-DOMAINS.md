# DAPE: Data-Adaptive Positional Encoding for Length Extrapolation (Year not specified in the paper)
Source: DAPE- Data-Adaptive Positional Encoding for Length Extrapolation.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling (Arxiv) | tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| language modeling (Books3) | tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (EVEN PAIRS) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (MODULAR ARITHMETIC (SIMPLE)) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (PARITY CHECK) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (CYCLE NAVIGATION) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (STACK MANIPULATION) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (REVERSE STRING) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (MODULAR ARITHMETIC) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (SOLVE EQUATION) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (DUPLICATE STRING) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (MISSING DUPLICATE) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | label (inferred) | 0D (inferred) | Fixed (inferred) |
| prediction (Odds First) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (BINARY ADDITION) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (COMPUTE SQRT) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |
| prediction (BUCKET SORT) (inferred) | sequences (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | sequence (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates DAPE on language modeling (perplexity) with Arxiv and Books3 and on CHE formal-language tasks such as even pairs, modular arithmetic, string reversal, binary addition, and bucket sort. Across tasks, inputs are 1D token sequences with bounded lengths, and outputs are either single labels (0D, fixed) or sequences (1D, capped) depending on the CHE task. The model uses standard transformer attention (static) and no explicit external state (direct) per the described decoder/encoder-only transformer setups.

## Evidence
### Task: language modeling (Arxiv)
- "Our analysis involves training language models on the Arxiv and Books3 datasets" (Section 4 Experiment)
- "We start our evaluation by comparing the last 256 tokens' zero-shot perplexity across different input lengths." (Section 4 Experiment)
- Inference: Input/output treated as 1D token sequences with capped length; attention static and state direct based on "training lengths of 128, 512, and 1024," "evaluation sequence length 8192," "decoder-only Transformers," and "The attention block was originally designed by applying softmax to the key-query multiplication." (Section 4 Experiment settings; Abstract; Section 1 Introduction)

### Task: language modeling (Books3)
- "Our analysis involves training language models on the Arxiv and Books3 datasets" (Section 4 Experiment)
- "We start our evaluation by comparing the last 256 tokens' zero-shot perplexity across different input lengths." (Section 4 Experiment)
- Inference: Input/output treated as 1D token sequences with capped length; attention static and state direct based on "training lengths of 128, 512, and 1024," "evaluation sequence length 8192," "decoder-only Transformers," and "The attention block was originally designed by applying softmax to the key-query multiplication." (Section 4 Experiment settings; Abstract; Section 1 Introduction)

### Task: prediction (EVEN PAIRS) (inferred)
- "|         | EVEN PAIRS                  | aabba                                                                                           | True                  |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (MODULAR ARITHMETIC (SIMPLE)) (inferred)
- "| Dagular | MODULAR ARITHMETIC (SIMPLE) | 1 + 2 - 4                                                                                       | 4                     |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (PARITY CHECK) (inferred)
- "| Regular | PARITY CHECK†††             | aaabba                                                                                          | True                  |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (CYCLE NAVIGATION) (inferred)
- "|         | CYCLE NAVIGATION†††         | 011210                                                                                          | 2                     |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (STACK MANIPULATION) (inferred)
- "|         | STACK MANIPULATION          | abbaa POP PUSH a POP                                                                            | abba                  |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (REVERSE STRING) (inferred)
- "| DCE     | REVERSE STRING              | aabba                                                                                           | abbaa                 |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (MODULAR ARITHMETIC) (inferred)
- "| DCF     | MODULAR ARITHMETIC          | $-(1-2)\cdot(4-3\cdot(-2))$                                                                     | 0                     |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (SOLVE EQUATION) (inferred)
- "|         | SOLVE EQUATION              | $ \begin{array}{l} -(1-2) \cdot (4-3 \cdot (-2)) \\ -(x-2) \cdot (4-3 \cdot (-2)) \end{array} $ | 1                     |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (DUPLICATE STRING) (inferred)
- "|         | DUPLICATE STRING            | abaab                                                                                           | abaababaab            |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (MISSING DUPLICATE) (inferred)
- "|         | MISSING DUPLICATE           | 10011021                                                                                        | 0                     |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 0D label (fixed) based on the Table 4 example output; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (Odds First) (inferred)
- "|         | Odds First                  | aaabaa                                                                                          | aaaaba                |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (BINARY ADDITION) (inferred)
- "| CS      | BINARY ADDITION             | 10010 + 101                                                                                     | 10111                 |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (COMPUTE SQRT) (inferred)
- "|         | COMPUTE SQRT                | 100010                                                                                          | 110                   |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

### Task: prediction (BUCKET SORT) (inferred)
- "|         | BUCKET SORT†††              | 421302214                                                                                       | 011222344             |" (Appendix D, Table 4)
- Inference: Input treated as 1D sequences with capped length from "Training is conducted on sequences whose lengths are uniformly distributed, sampled from U(1,N), with N set to 40." and "Evaluation is performed on sequences that vary in length from N+1 to M, where M equals 500."; output treated as a 1D sequence (capped) based on the Table 4 example output and "In scenarios that necessitate a multi-token output sequence y, such as the task of string duplication"; attention static and state direct based on "we utilize the encoder-only configuration of the original sequence-to-sequence Transformer model" and "The attention block was originally designed by applying softmax to the key-query multiplication." (Appendix D, Problem Setting; Section 1 Introduction)

---

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/DAPE- Data-Adaptive Positional Encoding for Length Extrapolation/.TASK-DOMAINS.csv.tmp.660b5f71f2404375979363ae3657ca65" with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
