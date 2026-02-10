1. **Number of distinct tasks evaluated: 3**
"Third, we show across three datasets that, compared to SLATE, our method for training achieves much lower validation loss in training, as well as lower Fréchet inception distance (FID) [29] and mean squared error (MSE) in image reconstruction." (# 1 Introduction)
"Fifth, when integrated with the original slot attention encoders and decoders from Locatello et al. [39], implicit differentiation substantially improves object property prediction and continues to predict intuitive segmentation masks as the vanilla slot attention." (# 1 Introduction)

2. **Number of trained model instances required to cover all tasks: 2**
"SLATE uses a discrete VAE [50] to compress an input image into a grid of discrete tokens. These tokens index into a codebook of latent code-vectors, which, after applying a learned position encoding, serve as the input to the slot attention module. An Image GPT decoder [10] is trained with a cross-entropy loss to autoregressively reconstruct the latent code-vectors, using the outputted slots from slot attention as queries and the latent code-vectors as keys/values. Gradients are blocked from flowing between the discrete VAE and the rest of the network (i.e. the slot attention module and the Image GPT decoder), but the entire system is trained simultaneously." (#### 5.1 Experimental setup)
"Futhermore, Fig. 7 shows that using only a single iteration is not enough to improve optimization of vanilla slot attention for the object property prediction task used by Locatello et al. 39. We directly modified the released code from Locatello et al. 39, which used three iterations, to use implicit differentiation." (## 5.5 Does this mean that iterating slot attention for one iteration is enough?)
"We sought to check whether implicit differentiation still preserves the quality of the segmentation masks produced by the original slot attention architecture by Locatello et al. [39], which uses a spatial broadcast decoder [63] rather than a transformer decoder as SLATE does." (#### 5.6 Does implicit slot attention still produce intuitive masks with a different architecture?)
Whether object property prediction and segmentation masks were evaluated with separate trained instances is Not specified in the paper.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{2\ \text{models}} = 1.5
}
$$
