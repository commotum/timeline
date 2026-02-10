1. **Number of distinct tasks evaluated:** **15**

   - Verbatim evidence: "To summarize, we can divide our quantitative evaluations into two separate parts:" followed by "**Evaluations on API distribution.**" and "**Evaluations on public NLP datasets.**" (Section 3.6, *Evaluation*).
   - Verbatim evidence: "We perform automatic evaluations on the following benchmark datasets: Winogender (Rudinger et al., 2018), CrowS-Pairs (Nangia et al., 2020), RealToxicityPrompts (Gehman et al., 2020), TruthfulQA (Lin et al., 2021), DROP (Dua et al., 2019), QuAC (Choi et al., 2018), SquadV2 (Rajpurkar et al., 2018), Hellaswag (Zellers et al., 2019), SST (Socher et al., 2013), RTE and WSC (both part of Super-GLUE (Wang et al., 2019)), WMT 15 Fr  $\rightarrow$  En (Bojar et al., 2015), CNN/Daily Mail Summarization (Nallapati et al., 2016), and Reddit TLDR Summarization (Völske et al., 2017)." (Section D, *Automatic evaluation details*).
   - Count used: 1 API-distribution evaluation block + 14 explicitly listed public benchmark tasks = 15.

2. **Number of trained model instances required to cover all tasks:** **1**

   - Verbatim evidence: "All tasks take a similar form: they (optionally) begin with an instruction that is common to all queries in the task; they then contain context for each query; and they end with a completion that is either sampled from the model or chosen from one of multiple choices." (Section D, *Automatic evaluation details*).
   - Verbatim evidence: "Unless otherwise specified, in this paper InstructGPT refers to the PPO-ptx models." (Section 3.5, *Models*).
   - Explicit capability-to-instance total is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{15\ \text{tasks}}{1\ \text{model}} = 15
}
$$
