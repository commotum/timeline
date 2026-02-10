1. **Number of distinct tasks evaluated:** 2

Section 4.2 ("Transferring to Image-Text Matching & Retrieval"): "We evaluate ALIGN models on image-to-text and text-toimage retrieval tasks, with and without finetuning."

Section 4.3 ("Transferring to Visual Classification"): "We first apply zero-shot transfer of ALIGN to visual classification tasks on ImageNet ILSVRC-2012 benchmark (Deng et al., 2009) and its variants including ImageNet-R(endition) (Hendrycks et al., 2020) (non-natural images such as art, cartoons, sketches), ImageNet-A(dversarial) (Hendrycks et al., 2021) (more challenging images for ML models), and ImageNet-V2 (Recht et al., 2019)."

2. **Number of trained model instances required to cover all tasks:** 1

Figure 1: "Without any fine-tuning, ALIGN powers zero-shot visual classification and cross-modal search including image-to-text search, text-to-image search and even search with joint image+text queries."

Section 4.1 ("Pre-training on Noisy Image-Text Pairs"): "We pre-train ALIGN using a dual-encoder architecture."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$
