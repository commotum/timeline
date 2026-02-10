1. **Number of distinct tasks evaluated:** 3  
   - "Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches." (Abstract)  
   - "To make sure we do not overfit on the original ARC data, we further evaluate our method on ConceptARC (Moskvichev et al., 2023) - an ARC-like dataset containing tasks sorted into specific conceptual categories." (Section 5.5, ConceptARC)  
   - "We further test our approach on the Sudoku 3M dataset (Radcliffe, 2020) to evaluate generalizability of the method to different domains." (Section 5.6, Sudoku)

2. **Number of trained model instances required to cover all tasks:** 2  
   - "After evaluating various models, we identified **Mistral-NeMo-Minitron-8B-Base** (Sreenivas et al., 2024) as exhibiting the strongest performance in our experiments." (Section 5.2, Training the models)  
   - "73.3% 2-guess accuracy on ConceptARC (using the exact same hyperparameters as DFS T=9%), showing that we generalize well to other ARC-like datasets of similar difficulty." (Section 5.5, ConceptARC)  
   - "Instead, we start out with our Llama 3B model pre-trained on ARC, which we then finetune again on 128000 Sudoku tasks." (Section 5.6, Sudoku)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{2\ \text{models}} = 1.5
}
$$
