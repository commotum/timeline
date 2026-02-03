1. Number of distinct tasks evaluated: 3. Evidence: "We use three datasets for evaluating our LXMERT framework: VQA v2.0 dataset (Goyal et al., 2017), GQA (Hudson and Manning, 2019), and NLVR<sup>2</sup>." (Section 4.1 Evaluated Datasets)
2. Number of trained model instances required to cover all tasks: 3. Evidence: "On VQA and GQA, we fine-tune our model from the pre-trained snapshot without data augmentation (analysis in Sec. 5.2)." and "Since each datum in NLVR<sup>2</sup> has two natural images  $img_0$ ,  $img_1$  and one language statement s, we use LXMERT to encode the two image-statement pairs ( $img_0$ , s) and ( $img_1$ , s), then train a classifier based on the concatenation of the two cross-modality outputs." (Section 4.2 Implementation Details)
3. Task–Model Ratio = 3 / 3 = 1.

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
