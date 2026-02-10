1. **Number of distinct tasks evaluated:** 2

> "Given an unknown sub-sequence  $y^*$  consisting of tokens from vocabulary  $\mathcal{V} := \{1, 2, \dots, 15\}$ , we train a policy  $\pi$  to produce a response which contains this sub-sequence. The task completion reward is terminal and sparse, *i.e.*,  $r(y, y^*) = 1$  for a y if and only if  $y^*$  appears in y." (Section 3.3, "Analysis in a Didactic Setting: Learning a Planted Sub-sequence")

> "We finetune Gemma 2B, 9B, and 27B (Gemma Team et al., 2024) on MATH (Hendrycks et al., 2021) via supervised fine-tuning (SFT) to get three base policies." (Section 4, Setup)

> "...resulting in a higher accuracy on the math reasoning task." (Section 7, "Discussion and Conclusion")

2. **Number of trained model instances required to cover all tasks:** 2

> "The policy  $\pi$  in our experiments is represented by a multi-layer neural network, similar to the MADE architecture (Germain et al., 2015)." (Appendix B, "Didactic Analysis")

> "We finetune Gemma 2B, 9B, and 27B (Gemma Team et al., 2024) on MATH (Hendrycks et al., 2021) via supervised fine-tuning (SFT) to get three base policies." (Section 4, Setup)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
