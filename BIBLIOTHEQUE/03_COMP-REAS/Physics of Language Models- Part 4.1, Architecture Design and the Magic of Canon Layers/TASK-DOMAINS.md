# Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers (2025)
Source: Physics of Language Models- Part 4.1, Architecture Design and the Magic of Canon Layers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Mental reasoning depth (Depo) | Token sequence of directed permutation edges and queries | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | k-th successor answers | 1D (t) (inferred) | Capped (inferred) |
| Mental reasoning breadth (Brevo) | Token sequence of DAG edges and query vertex | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Reachable vertices in topological order | 1D (t) (inferred) | Capped (inferred) |
| Knowledge capacity (Capo) | Synthetic biographies with attributes | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Next-token prediction distribution | 1D (t) (inferred) | Not specified in the paper. |
| Knowledge manipulation (Mano) | Prefix-notation modular arithmetic expressions | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Single-token modular arithmetic result | 0D (inferred) | Fixed (inferred) |
| Hierarchical language structure (Lano) | CFG-generated sentences | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | CFG-compliant sentences | 1D (t) (inferred) | Capped (inferred) |
| Copying (Full Copy) | Sequence with random permutation and query token | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | Identical copy of the sequence | 1D (t) (inferred) | Fixed (inferred) |
| Discriminative task (PIQA) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (HellaSwag) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (Wino-Grande) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (ARC-easy/challenge) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (SIQA) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (BoolQ) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (WikiText) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Discriminative task (LAMBADA) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Generative task (SWDE) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Generative task (FDA) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Generative task (SQuAD(v2)) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Generative task (TriviaQA) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Generative task (NQ) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Generative task (DROP) | Context and question prompts (inferred) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Generated answers (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Needle-in-a-Haystack retrieval (NIAH) | Long text with a needle value | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Needle value | 0D (inferred) | Fixed (inferred) |
| Multi-hop reasoning (Babilong) | Long junk-filled passages with embedded bAbi tasks | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Information retrieval (1-hop-L) | Wikipedia passages of length L tokens with five birth year statements | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Birth year | 0D (inferred) | Fixed (inferred) |
| Information retrieval (2-hop-L) | Wikipedia passages of length L tokens with birth year and equivalence statements | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Birth year | 0D (inferred) | Fixed (inferred) |

## Summary
The paper defines five synthetic pretraining tasks (Depo, Brevo, Capo, Mano, Lano) plus a Full Copy toy task, all framed as token-sequence problems with fixed or capped context windows and outputs that are sequences or single-token answers.
Real-life evaluation adds a discriminative benchmark suite (PIQA, HellaSwag, Wino-Grande, ARC-easy/challenge, SIQA, BoolQ, WikiText, LAMBADA), a generative suite (SWDE, FDA, SQuAD(v2), TriviaQA, NQ, DROP), and long-context retrieval/multi-hop tasks (NIAH, Babilong, 1-hop-L, 2-hop-L).
Across tasks, the only consistently justified dimensionality is 1D token sequences with fixed/capped interfaces where stated; attention and state dynamics are largely not specified in the OCR text.

## Evidence
### Task: Mental reasoning depth (Depo)
- "Task DEPO evaluates reasoning depth as k-hop traversal over directed permutations, where models compute the k-th successor for each query q entirely internally." (Section 2.1)
- "Context lengths are fixed to 2048 tokens." (Section 2.1)
- Inference: Input/output are treated as 1D token sequences with capped lengths based on the tokenized context window description above.

### Task: Mental reasoning breadth (Brevo)
- "Task Brevo isolates this capability using recursive traversal of directed acyclic graphs (DAGs), abstracting away natural language or arithmetic complexities." (Section 2.1)
- "Upon receiving a query vertex q, the model outputs all vertices recursively reachable from q, sorted in topological order starting from the leaves." (Section 2.1)
- "Brevol: Each vertex name spans a single token, with N = 70/90/110, fit within 1024 tokens." (Section 2.1)
- Inference: Input/output are treated as 1D token sequences and capped by the stated token-length limits.

### Task: Knowledge capacity (Capo)
- "synthetic datasets of (fake) biographies are constructed to test knowledge retention." (Section 2.1)
- "Capacity is measured using the next-token prediction distribution, accounting for both exact correctness and partial accuracy." (Section 2.1)
- Inference: Biographies imply 1D token sequences, and next-token prediction implies 1D output distributions.

### Task: Knowledge manipulation (Mano)
- "MANO employs synthetic modular arithmetic expressions inspired by human mental computation, particularly small-number arithmetic like the  $9\times9$  multiplication table." (Section 2.1)
- "The dataset is parameterized by a maximum expression length L, with  $\ell$  sampled uniformly from [1, L]." (Section 2.1)
- "outputs are single tokens representing exact modular arithmetic results" (Appendix A.4)
- Inference: Expressions are token sequences with capped length L, and outputs are single-token labels (0D) with fixed size.

### Task: Hierarchical language structure (Lano)
- "Task Lano evaluates structural reasoning over hierarchical relationships and long-range dependencies." (Section 2.1)
- "Lano leverages synthetic datasets built from context-free grammars (CFGs)." (Section 2.1)
- "models are prompted with a single **<bos>** token and tasked to generate CFG-compliant sentences" (Appendix A.5)
- "we use a context length of 512" (Appendix A.5)
- Inference: Input/output are treated as 1D token sequences with capped lengths based on stated context lengths and CFG sequence generation.

### Task: Copying (Full Copy)
- "This task involves choosing N=500 and generating a sequence starting with  $\langle bos \rangle$ , followed by a random permutation of N tokens between 1 and N." (Appendix B)
- "then appending  $\langle query \rangle$  and an identical copy of the sequence." (Appendix B)
- Inference: Fixed-length 1D sequences are inferred from N=500 and the explicit copy requirement.

### Task: Discriminative task (PIQA)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (HellaSwag)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (Wino-Grande)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (ARC-easy/challenge)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (SIQA)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (BoolQ)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (WikiText)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Discriminative task (LAMBADA)
- "covers discriminative tasks: PIQA [12], HellaSwag [79], Wino-Grande [51], ARC-easy/challenge [18], SIQA [52], BoolQ [17], WikiText, and LAMBADA [41]." (Section 8)

### Task: Generative task (SWDE)
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Generative task (FDA)
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Generative task (SQuAD(v2))
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Generative task (TriviaQA)
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Generative task (NQ)
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Generative task (DROP)
- "Tasks include SWDE, FDA, SQuAD(v2) [49], TriviaQA [34], NQ [38], and DROP [21], plus their JRT-enhanced variants (denoted as FDA2, SWDE2, etc.)" (Section 8)
- "JRT addresses this by repeating the context and question twice" (Section 8, Footnote 25)
- "Generative task prompts are capped at 1024–2048 tokens (per original codebase), while training used 4096." (Section 8, Footnote 26)
- Inference: Input is treated as context+question token prompts with capped length, and outputs as generated answer tokens.

### Task: Needle-in-a-Haystack retrieval (NIAH)
- "The Needle-in-a-Haystack (NIAH) task from RULER [29] tests recall of a "needle" value (e.g., a magic number) in long text." (Section 8)
- Inference: Input is treated as a 1D token sequence and output as a single value (0D).

### Task: Multi-hop reasoning (Babilong)
- "The Babilong dataset [37] embeds bAbi [69] tasks in long junk-filled passages to test multi-hop reasoning." (Section 8)
- Inference: Passages imply a 1D token sequence input; other fields are not specified.

### Task: Information retrieval (1-hop-L)
- "we evaluated models' performance on extremely simple 1-hop and 2-hop information retrieval tasks." (Appendix B)
- "The five sentences were embedded into random Wikipedia documents of length L tokens" (Appendix B)
- "to test its ability to retrieve the birth year." (Appendix B)
- Inference: Input is a capped 1D token sequence (length L), and output is a single birth-year value (0D).

### Task: Information retrieval (2-hop-L)
- "we evaluated models' performance on extremely simple 1-hop and 2-hop information retrieval tasks." (Appendix B)
- "For the 2-HOP-L task, three random birth year statements were prepared in the same format as above." (Appendix B)
- "This was followed by three equivalence statements" (Appendix B)
- "to test its ability to infer and retrieve the correct birth year." (Appendix B)
- Inference: Input is a capped 1D token sequence (length L), and output is a single birth-year value (0D).
