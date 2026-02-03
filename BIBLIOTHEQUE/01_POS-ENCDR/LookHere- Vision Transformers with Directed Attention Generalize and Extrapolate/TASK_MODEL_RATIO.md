“We demonstrate that LookHere: a improves classification, segmentation, adversarial robustness, and model calibration” (Section 1 Introduction).

“Adversarial Attacks. We perform Fast Gradient Sign Method (FGSM [82]) adversarial attacks with two strengths  $(\frac{1}{255}, \frac{3}{255})$  on all models using Val images.” (Section 4.1 Setup).

“Calibration Estimates. We calculate the Expected Calibration Error (ECE [83]) with 15 bins of all models using Val images.” (Section 4.1 Setup).

“Segmentation. With the best model per method, we finetune following the Segmenter protocol with a linear decoder [84]. Additionally, we probe the patches by only training a linear layer to produce a low-resolution logit map which is upsampled to obtain a full resolution segmentation map, following [85].” (Section 4.1 Setup).

$$
\boxed{
\frac{4\ \text{tasks}}{2\ \text{models}} = 2
}
$$
