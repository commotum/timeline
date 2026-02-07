# Recursive Language Models (Not specified in the paper)
Source: Recursive Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-NIAH (single needle-in-the-haystack retrieval) | long unrelated text prompt | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | specific phrase or number | 1D (t) (inferred) | Fixed (inferred) |
| BrowseComp-Plus (1K documents) multi-hop QA | collection of 1000 documents | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer to multi-hop question | 1D (t) (inferred) | Fixed (inferred) |
| OOLONG (trec_coarse) semantic aggregation QA | dataset of questions/chunks with semantic labels | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | final aggregated answer | 1D (t) (inferred) | Fixed (inferred) |
| OOLONG-Pairs pairwise aggregation / list generation | dataset of question entries with user IDs | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | list of user ID pairs satisfying constraints | 1D (t) (inferred) | Capped (inferred) |
| LongBench-v2 CodeQA (multi-choice code repository understanding) | code repository files | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | multiple-choice answer | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates RLMs on five long-context text and code benchmarks: needle-in-haystack retrieval, multi-hop document QA, semantic aggregation QA (OOLONG), pairwise aggregation/list generation (OOLONG-Pairs), and multiple-choice code repository QA. Inputs are long textual sequences (documents, question lists, or code files), so the addressed dimensions are 1D (t) and the benchmark setups imply capped or fixed input sizes. Outputs are mostly single answers, except OOLONG-Pairs which requires variable-length lists of pairs, yielding capped but non-fixed outputs. Dynamic attention and constructed state are inferred from the RLM interface that operates over a REPL environment and programmatically inspects the prompt.

## Evidence
### Task: S-NIAH (single needle-in-the-haystack retrieval)
- "unrelated text" (§2.1 Tasks)
- "specific phrase" (§2.1 Tasks)
- Inference: Marked 1D (t) dimensions and Dynamic/Constructed attention-state from the RLM interface description in §1 Introduction, and set input dynamics to Capped and output dynamics to Fixed based on the benchmark setup and single-answer requirement in §2.1.

### Task: BrowseComp-Plus (1K documents) multi-hop QA
- "different documents" (§2.1 Tasks)
- "correct answers" (§2.1 Tasks)
- Inference: Used §1 Introduction to infer 1D (t) inputs/outputs and Dynamic/Constructed attention-state, and marked input dynamics as Capped from the 1K-document setup with Fixed output dynamics because each task yields one answer (§2.1).

### Task: OOLONG (trec_coarse) semantic aggregation QA
- "semantic labels" (§2.1 Tasks)
- "final answer" (§2.1 Tasks)
- Inference: Used §1 Introduction to infer 1D (t) inputs/outputs and Dynamic/Constructed attention-state, and marked input dynamics as Capped and output dynamics as Fixed from the dataset-based benchmark description in §2.1.

### Task: OOLONG-Pairs pairwise aggregation / list generation
- "input contexts" (Appendix E.1 OOLONG-PAIRS BENCHMARK)
- "user IDs" (Appendix E.1 OOLONG-PAIRS BENCHMARK)
- Inference: Used §1 Introduction to infer 1D (t) inputs/outputs and Dynamic/Constructed attention-state, set input dynamics to Capped from the stated input-length ranges, and set output dynamics to Capped because tasks require listing variable-length pairs (Appendix E.1; §2.1).

### Task: LongBench-v2 CodeQA (multi-choice code repository understanding)
- "code repository" (§2.1 Tasks)
- "right answer" (§2.1 Tasks)
- Inference: Used §1 Introduction to infer 1D (t) inputs and Dynamic/Constructed attention-state, set input dynamics to Fixed from the fixed-number-of-files statement, and set output as a 0D multiple-choice label with Fixed dynamics because each task seeks one right answer (§2.1).
