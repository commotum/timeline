1. **Number of distinct tasks evaluated:** 13.

   "Specifically, we evaluate on the GLUE (Wang et al., 2019) benchmark for RoBERTa and DeBERTa. We follow the setup of Li & Liang (2021) on GPT-2 for a direct comparison and add WikiSQL (Zhong et al., 2017) (NL to SQL queries) and SAMSum (Gliwa et al., 2019) (conversation summarization) for large-scale experiments on GPT-3." (Section 5 EMPIRICAL EXPERIMENTS)

   "Due to space constraint, we only present our result on E2E NLG Challenge (Table 3) in this section. See Section F.1 for results on WebNLG (Gardent et al., 2017) and DART (Nan et al., 2020)." (Section 5.4 GPT-2 MEDIUM/LARGE)

   "It includes MNLI (inference, Williams et al. (2018)), SST-2 (sentiment analysis, Socher et al. (2013)), MRPC (paraphrase detection, Dolan & Brockett (2005)), CoLA (linguistic acceptability, Warstadt et al. (2018)), QNLI (inference, Rajpurkar et al. (2018)), QQP<sup>8</sup> (question-answering), RTE (inference),

<sup>&</sup>lt;sup>8</sup>https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

and STS-B (textual similarity, Cer et al. (2017))." (Section C DATASET DETAILS)

2. **Number of trained model instances required to cover all tasks:** 13.

   "One of the main drawbacks for full fine-tuning is that for *each* downstream task, we learn a *different* set of parameters  $\Delta\Phi$  whose dimension  $|\Delta\Phi|$  equals  $|\Phi_0|$ ." (Section 2 Problem Statement)

   "A pre-trained model can be shared and used to build many small LoRA modules for different tasks. We can freeze the shared model and efficiently switch tasks by replacing the matrices A and B in Figure 1, reducing the storage requirement and task-switching overhead significantly." (Section 1 Introduction)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{13\ \text{tasks}}{13\ \text{models}} = 1
}
$$
