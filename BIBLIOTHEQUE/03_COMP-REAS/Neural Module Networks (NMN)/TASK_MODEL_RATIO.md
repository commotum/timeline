1. **Number of distinct tasks evaluated:** 2

> "We evaluate our approach on two visual question answering tasks." (Section 1, Introduction)
>
> "On the recently-released VQA [3] dataset we achieve results comparable to or better than existing approaches. However, that many of the questions in the VQA dataset are quite simple, with little composition or reasoning required. To test our approach's ability to handle harder questions, we introduce a new dataset of synthetic images paired with complex questions involving spatial relations, set-theoretic reasoning, and shape and attribute recognition. On this dataset we outperform baseline approaches by as much as 25% absolute accuracy." (Section 1, Introduction)

2. **Number of trained model instances required to cover all tasks:** 2

> "As noted above, parts of this conversion process are task-specific—we found that relatively simple expressions were best for the natural image questions, while the synthetic data (by design) required deeper structures." (Section 4.2, From strings to networks)
>
> "| VQA    | find, combine, describe           | 877         | 51138     | 3         | 4        |"
>
> "| SHAPES | find, transform, combine, measure | 8           | 164       | 5         | 6        |" (Table 1)
>
> "To produce an initial set of image features, we pass the input image through the convolutional portion of a LeNet [22] which is jointly trained with the question-answering part of the model." (Section 6, Experiments: compositionality)
>
> "Here we evaluate on the VQA dataset [3]." (Section 7, Experiments: natural images)
>
> "The visual input to the NMN is the conv5 layer of a 16-layer VGGNet [35] after max-pooling, with features normalized to have mean 0 and standard deviation 1." (Section 7, Experiments: natural images)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
