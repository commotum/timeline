1. **Number of distinct tasks evaluated:** **32**.

   - "The bAbi question answering dataset (Weston et al., 2015) consists of 20 different tasks" (Section 3.1).
   - "Next, we consider the task of predicting number-agreement between subjects and verbs in English sentences" (Section 3.2).
   - "The LAMBADA task (Paperno et al., 2016) is a language modeling task" (Section 3.3).
   - "The task is evaluated in two settings: as *language modeling* (the standard setup) and as *reading comprehension*." (Section 3.3).
   - "We trained UTs on three algorithmic tasks, namely Copy, Reverse, and (integer) Addition" (Section 3.4).
   - "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5).
   - "We trained a UT on the WMT 2014 English-German translation task" (Section 3.6).

2. **Number of trained model instances required to cover all tasks:** **13 models**.

   - For bAbI, one joint model is explicitly supported: "models can either be trained on each task separately (\"train single\") or jointly on all tasks (\"train joint\")." (Section 3.1), and "the original idea is that a single model should be evaluated across all the tasks (not tuning per task), which is the *train joint* setup" (Appendix D.1).
   - For the other evaluated tasks, task-specific training/evaluation setups are described (e.g., "follow their experimental protocol of solving the task using a language modeling training setup" (Section 3.2); "a model is simply trained for next-word prediction" (Section 3.3); "We trained UTs on three algorithmic tasks" (Section 3.4); "We trained a UT on the WMT 2014 English-German translation task" (Section 3.6)).
   - A single jointly trained model spanning all non-bAbI tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{32\ \text{tasks}}{13\ \text{models}} = 2.46
}
$$
