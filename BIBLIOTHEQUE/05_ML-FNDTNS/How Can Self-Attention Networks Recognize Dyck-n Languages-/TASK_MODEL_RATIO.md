1. **Number of distinct tasks evaluated:** 4

   - Quote (Section 3, Experiments): "We perform experiments on  $\mathcal{D}_1$ ,  $\mathcal{D}_2$ ,  $\mathcal{D}_3$ , and  $\mathcal{D}_4$  languages."
   - Quote (Section 3, Experiments): "We follow prior works (Gers and Schmidhuber, 2001; Suzgun et al., 2019), and formulate the recognition of  $\mathcal{D}_n$  languages as a transduction task: Given a valid string, we ask the model to predict the next possible symbols auto-regressively."

2. **Number of trained model instances required to cover all tasks:** 4

   - Quote (Section 3, Experiments): "For each  $\mathcal{D}_n$  language, we train on 32k sequences of length 2-50, validate on 3.2k sequences of length 52-74, and evaluate on 10k sequences divided equally over the length intervals 76-100 and 102-126."
   - Quote (Section 3, Experiments): "where  $|V_n^o|$  is the output vocabulary size (2 for  $\mathcal{D}_1$ ,

4 for  $\mathcal{D}_2$ , 6 for  $\mathcal{D}_3$ , 8 for  $\mathcal{D}_4$ )"

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
