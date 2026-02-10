# ZEPHYR: DIRECT DISTILLATION OF LM ALIGNMENT (Not specified in the paper.)
Source: Zephyr- Direct Distillation of LM Alignment.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Instruction-following chat generation | User prompts and dialogue text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Model response text | 1D (t) (inferred) | Capped (inferred) |
| Preference ranking for alignment (dDPO) | Prompt with chosen/rejected responses $(x, y_w, y_l)$ | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Preferred-over-rejected ranking signal | 0D (inferred) | Fixed (inferred) |
| Multiclass classification | Benchmark question/context text (ARC, HellaSwag, MMLU, Truthful QA) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Class label selection | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers text-based instruction-following chat generation, pairwise preference ranking for alignment during dDPO training, and multiclass classification evaluation tasks. Across these tasks, inputs are inferred as 1D (t) token sequences with capped dynamics, supported by the reported token-length-limited training setup. Outputs include generated text sequences for chat and fixed discrete selections for ranking/classification. Attention and state are inferred as Static and Direct from the prompted autoregressive LM setup described in the method.

## Evidence
### Task: Instruction-following chat generation
- "Starting with a raw LLM, we first need to train it to respond to user prompts." (Section 3, Distilled Supervised Fine-Tuning (dSFT))
- "Our main evaluations are on single-turn and multi-turn chat benchmarks that measure a model's ability to follow instructions and respond to challenging prompts across a diverse range of domains:" (Section 4.2)
- "AlpacaEval (Li et al., 2023) is a single-turn benchmark where a model must generate a response to 805 questions on different topics, mostly focused on helpfulness." (Section 4.2)
- Inference: In Dimension and Out Dimension are inferred as 1D (t) because prompts and responses are text sequences; In/Out Dynamics are inferred as Capped based on "use packing with a sequence length of 2048 tokens." (Section 4.3). Attention Dynamic is inferred as Static and State Dynamic as Direct from the prompted autoregressive setup (Sections 3 and 4), with no runtime retrieval/controller or explicit constructed external state described.

### Task: Preference ranking for alignment (dDPO)
- "The goal of the final step is to refine the  $\pi_{\text{dSFT}}$  by maximizing the likelihood of ranking the preferred  $y_w$  over  $y_l$  in a preference model." (Section 3, Distilled Direct Preference Optimization (dDPO))
- "The final feedback dataset  $\mathcal D$  consists of a set of these triples  $(x,y_w,y_l)$." (Section 3, AI Feedback through Preferences (AIF))
- "By plugging this function of the reward into the preference model, the authors show that the objective can be written as" (Section 3, Distilled Direct Preference Optimization (dDPO))
- Inference: In Dimension is inferred as 1D (t) because $x$, $y_w$, and $y_l$ are text prompts/responses; Out Dimension is inferred as 0D and Out Dynamics as Fixed because the training signal is a binary preferred-vs-rejected comparison. In Dynamics is inferred as Capped from token-length-limited training (Section 4.3). Attention Dynamic is inferred as Static and State Dynamic as Direct because the method describes probability computation over fixed prompt/response triples without dynamic retrieval or explicit constructed state.

### Task: Multiclass classification
- "We also evaluate ZEPHYR-7B on the Open LLM Leaderboard (Beeching et al., 2023), which measures the performance of LMs across four multiclass classification tasks: ARC (Clark et al., 2018), HellaSwag (Zellers et al., 2019), MMLU (Hendrycks et al., 2021), and Truthful QA(Lin et al., 2022)." (Section 4.2)
- "Although this leaderboard does not directly measure the conversational quality of chat models, it does provide a useful signal to validate whether fine-tuning has introduced regressions on the base model's reasoning and truthfulness capabilities." (Section 4.2)
- Inference: In Dimension is inferred as 1D (t) from text benchmark inputs; Out Dimension is inferred as 0D and Out Dynamics as Fixed from the explicit multiclass classification framing. In Dynamics is inferred as Capped from the reported sequence-length-limited training setup (Section 4.3). Attention Dynamic is inferred as Static and State Dynamic as Direct from the same prompted LM setup without explicit runtime retrieval or constructed external state.
