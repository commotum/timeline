1. **Number of distinct tasks evaluated: 7**

   > "The pre-trained models are then fine-tuned on GLUE (Wang et al., 2019), SQuAD (Rajpurkar et al., 2016, 2018), and SWAG (Zellers et al., 2018) to assess the pre-training performance." (Section 4.1, *Pre-training Data and Fine-tuning Tasks*)

   > "We follow this trend and evaluate on the four largest datasets (i.e., SST-2 (Socher et al., 2013), QNLI (Rajpurkar et al., 2016), QQP (Iyer et al., 2017), MNLI (Williams et al., 2018))." (Section 4.1, *Pre-training Data and Fine-tuning Tasks*)

   > "| Method                                  | SST-2 | QNLI | QQP         | MNLI | SQuAD v1.1 | SQuAD v2.0 | SWAG | Avg. |" (Section 4.3, Table 2)

2. **Number of trained model instances required to cover all tasks: 7**

   > "As reported in Table 2, we fine-tune the pre-trained models on different natural-language tasks." (Section 4.3, *Results*)

   > "For fine-tuning tasks, instead of high-cost hyper-parameter sweeping in BERT (Devlin et al., 2019), we train 3 epochs with a learning rate of 1*e*-4 and a batch-size of 32 for all tasks in GLUE." (Appendix A.1, *Full Implementation Details*)

   > "On SQuAD v1.1, we fine-tune for 3 epochs with a learning rate of 5*e*-5 and a batch size of 32. On SQuAD v2.0, we fine-tune for 2 epochs with a learning rate of 5*e*-5 and a batch size of 48. On SWAG, we fine-tune for 3 epochs with a learning rate of 2*e*-5 and a batch size of 16." (Appendix A.1, *Full Implementation Details*)

   A single jointly fine-tuned model that serves all evaluated tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{7\ \text{tasks}}{7\ \text{models}} = 1
}
$$
