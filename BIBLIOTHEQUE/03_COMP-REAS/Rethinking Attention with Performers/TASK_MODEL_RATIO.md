1. **Number of distinct tasks evaluated:** 3

> "We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling." (ABSTRACT)

2. **Number of trained model instances required to cover all tasks:** 3

> "1. Backwards compatibility with pretrained models is available as a benefit from softmax approximation, via small finetuning (required due to error propagation) even for trigonometric features (Fig. 5, left) on the LM1B dataset (Chelba et al., 2014). However, when on larger dataset PG-19, 2. Positive (POS) softmax features (with redrawing) become crucial for achieving performance matching regular Transformers (Fig. 5, right)." (Section 4.3: SOFTMAX APPROXIMATION ON TRANSFORMERS)

> "We further benchmark the Performer on both (U) and (B) cases by training a 36-layer model using protein sequences from the Jan. 2019 release of TrEMBL (Consortium, 2019), similar to (Madani et al., 2020)." (Section 4.4: MULTIPLE LAYER TRAINING FOR PROTEINS)

> "On the standard (U) ImageNet64 benchmark from (Parmar et al., 2018) with L=12288 which is unfeasible for regular Transformers, we set all models to use the same  $(n_{heads}, d_{ff}, d)$  but varying  $n_{layers}$ ." (Section 4.5: Large length training - Common datasets)

> "Batch sizes were maximized for each separate run given the compute constraints." (Figure 6 caption, Section 4.4)

A single jointly trained model instance that covers all tasks without task-specific training: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
