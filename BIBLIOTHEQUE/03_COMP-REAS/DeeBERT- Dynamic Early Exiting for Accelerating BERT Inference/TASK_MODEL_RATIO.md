1. **Number of distinct tasks evaluated:** 6

   "We apply DeeBERT to both BERT and RoBERTa, and conduct experiments on six classification datasets from the GLUE benchmark (Wang et al., 2018): SST-2, MRPC, QNLI, RTE, QQP, and MNLI." (Section 4.1, *Experimental Setup*)

2. **Number of trained model instances required to cover all tasks:** 6

   "All transformer layers and off-ramps are jointly fine-tuned on a given downstream dataset." (Section 1, *Introduction*)

   "For fine-tuning on a downstream task, the loss function of the i<sup>th</sup> off-ramp is" (Section 3.1, *DeeBERT at Fine-Tuning*)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
