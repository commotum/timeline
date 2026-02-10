1. **Number of distinct tasks evaluated:** 501

   (Section 3.1) "The toy task we will construct consists of many subtasks – distinct types of inputs which each require corresponding distinct computations (quanta)."  
   (Section 3.1) "multitask sparse parity adds an additional parameter  $n_{\text{tasks}}$ , the number of subtasks (number of distinct versions of sparse parity) present in the dataset."  
   (Section 3.2) "For the results shown, we used  $n_{\rm tasks}=500,\,n=100,\,k=3,\,\alpha=0.4$ , and a batch size of 20000."  
   (Section 4) "We now study how scaling curves for large language models decompose."

2. **Number of trained model instances required to cover all tasks:** 2

   (Section 3.2) "We train ReLU MLPs with a single hidden layer to solve this task with cross-entropy loss."  
   (Section 3.1) "The control bits 1-hot encode the task number: on a given input, only one control bit is set to 1 at a time – the rest are zero."  
   (Section 4) "For our experiments, we use the Pythia model suite from Eleuther AI [29], a set of decoder-only transformers of varying size trained on approximately 300 billion tokens of The Pile [30]."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{501\ \text{tasks}}{2\ \text{models}} = 250.5
}
$$
