1. **Number of distinct tasks evaluated:** 7

> "Table 1. Performance on downstream training tasks, tested on data from a possibly different distribution from the downstream task dataset(s) under matched learner update budgets. Language: GSM8K test Pass@1. Vision: ADE20K mIoU, NYUv2 RMSE, and ImageNet linear accuracy." (Section 4.2, Table 1)
>
> "*Table 2.* **Evaluation on tasks not used for feedback.** Language: value-adjacent transfer under distribution shift (OMEGA) and value-extrapolative evaluation (MMLU). Vision: instance retrieval transfer on Revisited Oxford/Paris." (Section 4.2, Table 2)

2. **Number of trained model instances required to cover all tasks:** 4

> "**Learner models.** Our learner is an off-the-shelf causal LM initialized from pretrained checkpoints (Qwen family in the main paper)." (Appendix A.1)
>
> "**Meta step details (vision).** Each meta step consists of: (1) updating segmentation/depth heads on labeled *train* minibatches with the backbone frozen, (2) computing  $g_{\text{down}}$  on labeled *meta* mini-batches w.r.t. a subset of backbone parameters (last k ViT blocks), (3) computing  $g_{\text{ssl}}$  on an unlabeled meta-SSL batch with masks applied (with create\_graph=True), (4) updating the designer by minimizing:" (Appendix A.2)
>
> "**Evaluation protocols.** We evaluate representation quality using: (i) ADE20K mIoU with standard label remapping (ignore void) and either a linear-BN probe or a small conv decoder, (ii) NYUv2 depth using RMSE (and auxiliary metrics such as AbsRel and  $\delta_1$ ), with standard min/max depth clipping and optional Eigen crop, (iii) ImageNet-1K linear evaluation with a linear-BN head trained on frozen features (or partial finetuning of the last k blocks in ablations)." (Appendix A.2)
>
> "**Instance retrieval transfer.** To test whether dense-task feedback harms transfer to a distinct vision capability, we evaluate frozen ViT-L representations on Revisited Oxford (R-Oxford5k) and Revisited Paris (R-Paris6k) instance retrieval (Radenović et al., 2018). We extract a single global descriptor per image by mean-pooling patch tokens,  $\ell_2$ -normalize features, and rank database images by cosine similarity." (Section 4.3)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{7\ \text{tasks}}{4\ \text{models}} = 1.75
}
$$
