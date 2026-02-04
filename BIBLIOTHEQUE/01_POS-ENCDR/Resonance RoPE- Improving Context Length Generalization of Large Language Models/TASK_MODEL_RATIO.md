1. Number of distinct tasks evaluated: 11 (POSGEN subtasks: Recursive, Chain-of-Thought (CoT), Semi-recursive; language modeling perplexity on GovReport and Proofpile; L-Eval closed-ended tasks: Coursera, GSM, QuALITY, TOEFL, CodeU, SFiction). Quote (Section 5): "Our PosGeN framework comprises three subtasks, with each extracting the general token dependency pattern of a different type of reasoning task." "The three subtasks of PosGEN are as follows:" "- 1. **Recursive.**" "- 2. Chain-of-Thought (CoT)." "- 3. **Semi-recursive.**" Quote (Section 6.2.2): "We evaluate the model's language modeling performance on GovReport (Huang et al., 2021) and Proofpile (Azerbayev, 2022)." Quote (Table 2): "| Setting                                        | Ctx Len. | Coursera     | GSM          | QuALITY | TOEFL        | CodeU       | SFiction     | Avg.  |  |  |"
2. Number of trained model instances required to cover all tasks: 4 (three separately trained two-layer Transformers for the three POSGEN subtasks; one fine-tuned LLaMA2-Chat model used for the LLM evaluations). Quote (Appendix C.1): "For the synthetic task experiments in Section 6.1.1, we train a two-layer Transformer on each of the subtasks, with each layer following the configuration of a T5-Small model (Raffel et al., 2020)." Quote (Appendix C.2): "For the LLM-based evaluations in Section 6.2, we fine-tune LLaMA2-Chat 7B or LLaMA2-Chat 13B (Touvron et al., 2023b) after replacing its original RoPE position embedding with RoPE scaled with different strategies:" Quote (Section 6.2): "We test the model's performance on two TSTL scenarios: language modeling evaluation on long-text sequences and long-text downstream application performance."
3. Task–Model Ratio:

$$
\boxed{
\frac{11\ \text{tasks}}{4\ \text{models}} = 2.75
}
$$
