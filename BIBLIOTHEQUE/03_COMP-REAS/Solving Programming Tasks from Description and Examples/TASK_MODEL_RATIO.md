1. **Number of distinct tasks evaluated:** 1

   “We specifically consider a problem of synthesizing programs from a short description and several input / output pairs.” (Section 1 Introduction)

   “In this section we describe a new dataset we prepared to train and evaluate models that learn to synthesize simple data processing programs.” (Section 4 ALGOLISP)

2. **Number of trained model instances required to cover all tasks:** 1 model

   “The model is trained using back-propagation.” (Section 3.2 SeQ2Tree)

   “we run the search for *batch_size* tasks simultaneously, and on each step pop the single most likely incomplete tree for each task, identify the empty node in each of them, and compute the probabilities of the symbols in all of them at once.” (Section 3.3 Search)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{tasks}}{1\ \text{model}} = 1
}
$$
