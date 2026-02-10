# Self-Consistency Improves Chain of Thought Reasoning in Language Models (2022)
Source: Self-Consistency Improves Chain of Thought Reasoning in Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Arithmetic reasoning | Math word problem question tokens with few-shot chain-of-thought exemplars | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Final numeric answer or option label | 0D (inferred) | Fixed (inferred) |
| Commonsense reasoning | Commonsense question tokens with few-shot exemplars | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Final answer label or yes/no decision | 0D (inferred) | Fixed (inferred) |
| Symbolic reasoning (last letter concatenation) | Name/string tokens (e.g., "Elon Musk") | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Concatenated letter string (e.g., "nk") | 1D (t) (inferred) | Fixed (inferred) |
| Symbolic reasoning (coinflip state tracking) | Tokens describing initial coin state and flips | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Yes/no answer about final coin state | 0D (inferred) | Fixed (inferred) |
| Closed-book question answering | Question tokens (BoolQ, HotpotQA) with few-shot exemplars | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Yes/no label (BoolQ) or short text answer (HotpotQA) | 0D (inferred); 1D (t) (inferred) | Fixed (inferred); Capped (inferred) |
| Natural language inference | Premise-hypothesis token pairs (ANLI, e-SNLI, RTE) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Entailment/contradiction/unknown label | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates text-based reasoning tasks across arithmetic reasoning, commonsense reasoning, symbolic reasoning, closed-book QA, and natural language inference. Across these tasks, the input modality is token sequences, so input dimension is 1D (t), while outputs are mainly point-like final answers (0D), with sequence-style outputs also present for last-letter concatenation and HotpotQA answers. The paper explicitly states that self-consistency is applied to problems with a fixed final answer set, but input dynamics are not explicitly specified. Attention and state labels are inferred as Static and Direct from the fixed prompt-plus-decoding setup without explicit retrieval modules or persistent external state.

## Evidence
### Task: Arithmetic reasoning
- "Arithmetic reasoning. For these tasks, we used the Math Word Problem Repository (Koncel-Kedziorski et al., 2016), including AddSub (Hosseini et al., 2014), MultiArith (Roy & Roth, 2015), and ASDiv (Miao et al., 2020). We also included AQUA-RAT (Ling et al., 2017), a recently published benchmark of grade-school-math problems (GSM8K; Cobbe et al., 2021), and a challenge dataset over math word problems (SVAMP; Patel et al., 2021)." (Section 3.1, Tasks and datasets)
- "Table 2: Arithmetic reasoning accuracy by self-consistency compared to chain-of-thought prompting (Wei et al., 2022)." (Section 3.2, Table 2)
- Inference: 1D (t) input, Static attention, and Direct state are inferred from prompt-and-decode text generation ("a language model is prompted with a set of manually written chain-of-thought exemplars" and "we sample a set of candidate outputs from the language model's decoder"); 0D output and Fixed output dynamics are inferred from "the generated answers  $\mathbf{a}_i$  are from a fixed answer set" and final answer aggregation. (Section 2)

### Task: Commonsense reasoning
- "Commonsense reasoning. For these tasks, we used CommonsenseQA (Talmor et al., 2019), StrategyQA (Geva et al., 2021), and the AI2 Reasoning Challenge (ARC) (Clark et al., 2018)." (Section 3.1, Tasks and datasets)
- "Commonsense and Symbolic Reasoning Table 3 shows the results on commonsense and symbolic reasoning tasks." (Section 3.2)
- Inference: 1D (t) input, Static attention, and Direct state are inferred from the same prompt-and-decoder procedure in Section 2; 0D output and Fixed output dynamics are inferred from the fixed-answer aggregation setup (majority vote over final answers from a fixed answer set). (Section 2)

### Task: Symbolic reasoning (last letter concatenation)
- "Symbolic Reasoning. We evaluate two symbolic reasoning tasks: last letter concatenation (e.g., the input is \"Elon Musk\" and the output should be \"nk\"), and Coinflip (e.g., a coin is heads-up, after a few flips is the coin still heads-up?) from Wei et al. (2022)." (Section 3.1, Tasks and datasets)
- "For symbolic reasoning, we test the out-of-distribution (OOD) setting where the input prompt contains examples of 2-letters or 2-flips but we test examples of 4-letters and 4-flips (this setting is more challenging as PaLM-540B or GPT-3 can already achieve perfect in-distribution accuracy)." (Section 3.2)
- Inference: 1D (t) input and 1D (t) output are inferred because the task maps one text string to another short text string (explicit "the input is \"Elon Musk\" and the output should be \"nk\""); Static attention and Direct state are inferred from the same fixed prompt-decoder process in Section 2; Fixed output dynamics are inferred from the benchmark framing with fixed letter-count targets per setting (2-letter/4-letter). (Section 3.1; Section 3.2)

### Task: Symbolic reasoning (coinflip state tracking)
- "Symbolic Reasoning. We evaluate two symbolic reasoning tasks: last letter concatenation (e.g., the input is \"Elon Musk\" and the output should be \"nk\"), and Coinflip (e.g., a coin is heads-up, after a few flips is the coin still heads-up?) from Wei et al. (2022)." (Section 3.1, Tasks and datasets)
- "For symbolic reasoning, we test the out-of-distribution (OOD) setting where the input prompt contains examples of 2-letters or 2-flips but we test examples of 4-letters and 4-flips (this setting is more challenging as PaLM-540B or GPT-3 can already achieve perfect in-distribution accuracy)." (Section 3.2)
- Inference: 1D (t) input is inferred because the coinflip state sequence is textually described; 0D output and Fixed output dynamics are inferred because the task asks a single yes/no-style final state decision ("is the coin still heads-up?"); Static attention and Direct state are inferred from the same fixed prompt-and-decoder mechanism in Section 2. (Section 2; Section 3.1)

### Task: Closed-book question answering
- "Here we perform a study using self-consistency to see if it can help fill in the gap, over a set of common NLP tasks, including (1) Closed-Book Question Answering: BoolQ (Clark et al., 2019), HotpotQA (Yang et al., 2018), and (2) Natural Language Inference: e-SNLI (Camburu et al., 2018), ANLI (Nie et al., 2020) and RTE (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009)." (Section 3.3)
- "Table 16: Few-shot exemplars for HotpotQA (closed-book setting)." (Appendix A.3, Table 16)
- "Table 21: Few-shot exemplars for BoolQ (closed-book setting)." (Appendix A.3, Table 21)
- Inference: 1D (t) input is inferred from text-question prompts; output is mixed 0D and 1D (t) because BoolQ uses yes/no labels (e.g., "The answer is yes.") while HotpotQA returns short text answers (e.g., "The answer is Arthur's Magazine."); Fixed/Capped output dynamics are inferred from fixed-label QA in BoolQ and token-sequence answer generation with decoding limits ("For GPT-3 models, we use 128 max tokens for all methods"). Static attention and Direct state are inferred from the prompt-and-decoder setup. (Section 3.3; Appendix A.2; Appendix A.3)

### Task: Natural language inference
- "(2) Natural Language Inference: e-SNLI (Camburu et al., 2018), ANLI (Nie et al., 2020) and RTE (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009)." (Section 3.3)
- "A: \"one of\" means the same as \"a member of\", \"carry out\" means the same as \"execute\", and \"minutely\" means the same as \"immense precision\". The answer is yes." (Appendix A.3, Table 18)
- Inference: 1D (t) input is inferred from premise-hypothesis text pairs; 0D output and Fixed dynamics are inferred from the fixed NLI label set (yes/no/not possible to tell). Static attention and Direct state are inferred from the same fixed prompt-and-decoder architecture used throughout. (Section 2; Section 3.3; Appendix A.3)
