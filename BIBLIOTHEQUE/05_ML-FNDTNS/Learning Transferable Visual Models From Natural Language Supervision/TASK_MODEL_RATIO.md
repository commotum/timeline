1. **Number of distinct tasks evaluated:** 36

   "We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification." (Abstract)

   "Across 39 evals on 36 different datasets, average zero-shot error is well modeled by a log-log linear trend across a 44x range of compute spanning 5 different CLIP models." (Figure 9, Section 3.1.5)

   "To provide a qualitative summary / overview of CLIP's zero-shot performance we visualize a randomly selected prediction for 36 different zero-shot CLIP classifiers in Figure 21." (Appendix B)

2. **Number of trained model instances required to cover all tasks:** 1

   "The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training." (Abstract)

   "For each dataset, we use the names of all the classes in the dataset as the set of potential text pairings and predict the most probable (image, text) pair according to CLIP." (Section 3.1.2)

   "Unless otherwise specified, all results reported in this paper as "CLIP" use this model which we found to perform best." (Section 2.5)

3. **Task–Model Ratio = (1) / (2):** 36

$$
\boxed{
\frac{36\ \text{tasks}}{1\ \text{model}} = 36
}
$$
