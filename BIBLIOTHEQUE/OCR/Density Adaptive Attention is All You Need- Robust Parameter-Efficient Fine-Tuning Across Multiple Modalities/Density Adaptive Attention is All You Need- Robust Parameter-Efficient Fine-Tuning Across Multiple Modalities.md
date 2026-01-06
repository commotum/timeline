# Density Adaptive Attention is All You Need: Robust Parameter-Efficient Fine-Tuning Across Multiple Modalities

#### **Georgios Ioannides**

James Silberrad Brown Center for Artificial Intelligence Carnegie Mellon University Amazon GenAI\* gioannid@alumni.cmu.edu

#### **Aman Chadha**

James Silberrad Brown Center for Artificial Intelligence Stanford University Amazon GenAI\* hi@aman.ai

#### **Aaron Elkins**

James Silberrad Brown Center for Artificial Intelligence aelkins@sdsu.edu

## **Abstract**

We propose the Multi-Head Density Adaptive Attention Mechanism (DAAM), a novel probabilistic attention framework that can be used for Parameter-Efficient Fine-tuning (PEFT), and the Density Adaptive Transformer (DAT), designed to enhance information aggregation across multiple modalities, including Speech, Text, and Vision. DAAM integrates learnable mean and variance into its attention mechanism, implemented in a multi-head framework, enabling it to collectively model any probability distribution for dynamic recalibration of feature significance. This method demonstrates significant improvements, especially with highly non-stationary data, surpassing the state-of-the-art attention techniques in model performance, up to approximately +20% (abs.) in accuracy. Empirically, DAAM exhibits superior adaptability and efficacy across a diverse range of tasks, including emotion recognition in speech, image classification, and text classification, thereby establishing its robustness and versatility in handling data across multiple modalities. Furthermore, we introduce the Importance Factor, a new learning-based metric that enhances the explainability of models trained with DAAM-based methods<sup>†</sup>.

Attention mechanisms, as exemplified in the Transformer model [1], have significantly advanced the field of sequence modeling, particularly in Natural Language Processing (NLP) and various branches of Signal Processing such as Speech Processing and Digital Image Processing. These mechanisms are adept at capturing dependencies within the context length, although their effectiveness can vary based on the relative placement of tokens and the inherent limitations in handling long-range dependencies due to quadratic complexity [2, 3, 4]. Ongoing research continues to address these challenges, seeking more efficient ways to model long sequences and capture global context dependencies.

In recent years, the self-attention mechanism, or its variations that are based on dot-product, has become central to the encoders of Pre-Trained Models (PTMs) developed using Self-Supervised Learning (SSL) methods. Notable examples include WavLM [5] and HuBERT [6] for speech processing, Llama 2 [7] for text processing, and BEiT [8] for image processing. These PTMs

<sup>\*</sup>Work does not relate to position at Amazon.

<sup>†</sup>https://github.com/gioannides/DAAM-PEFT-paper-code

are highly effective at generating contextualized embeddings, outperforming traditional feature engineering methods [9], and they are adaptable to a wide range of downstream tasks specific to their training modalities.

Section 1 reviews relevant previous work and explains DAAM and its integration with dot-product attention (Grouped Query Attention); Section 3 discusses experimental findings, limitations and future research; Section 5 concludes. Our contributions can be summarized as follows:

- 1. We propose the Multi-Head DAAM and the Density Adaptive Transformer (DAT), featuring fully learnable mean and variance parameters within a multi-headed parameter-efficient framework. This enables dynamic recalibration of feature importance to fit any probability distribution best suited to attend the data it is trained on.
- 2. Introduction of IF, a new metric to enhance explainability in models using DAAM, quantitatively assessing feature significance for improved interpretability.
- 3. Utilization of PTMs as an embedding extractors across Speech, Text, and Vision modalities, demonstrating DAAM's superiority over conventional dot-product attention and other Parameter-Efficient Fine-tuning methods in handling non-stationary data.
- 4. Integration of DAAM with Grouped Query Attention, optimizing computational efficiency and showcasing compatibility with dot-product attention in popular PTM models (e.g., WavLM, HuBERT, Llama), enhancing performance with minimal parameter increase.

## 1 Methods

# Advantages of Learning with Multi-Head DAAM

DAAM leverages additive and multiplicative Gaussian parameters – mean offsets and variance scaling factors, respectively – to dynamically adjust attention. The mean offset shifts the Gaussian's focus based on input context, enhancing responsiveness to deviations. In parallel, the variance scaling adapts the distribution's spread, ensuring attention is not only accurately centered but also suitably scaled to the task's specificity. The multi-head design allows each head to address different data aspects, enhancing the model's adaptability to non-Gaussian traits [10].

Below, we analyze the theoretical aspects of entropy for both traditional self-attention mechanisms and DAAM.

## Density Adaptive Attention with N Gaussians per Head, h

Each head, h, in Density Adaptive Attention Mechanism (DAAM) processes input using Gaussian normalization, which is controlled by learnable parameters  $\mu_{i,h}$  and  $\sigma_{i,h}$ . The transformation is defined by the formula  $y_{\text{norm}} = \frac{y - (\text{mean} + \text{mean} - \text{offset})}{\sqrt{\text{var} + \epsilon}}$ , where  $\epsilon$  is a small constant ensuring numerical stability. This normalized input is then applied to a Gaussian function,  $f^{(h)}(x) = \exp\left(-\frac{y_{\text{norm}}^2}{2c^2}\right)$ , with c as a learnable parameter that controls the spread of the Gaussian function. The overall transformation for each head approximates a Gaussian distribution, where the variance  $\sigma_{\text{prod}}^2$  is a function of the aggregated variances and mean adjustments within the head, represented by  $\sigma_{\text{prod}}^2 = \left(\sum_{i=1}^N \frac{1}{\sigma_{i,h}^2}\right)^{-1}$ . Additionally, the effective mean is given by  $\mu_{\text{prod}} = \sigma_{\text{prod}}^2 \left(\sum_{i=1}^N \frac{\mu_{i,h}}{\sigma_{i,h}^2}\right)$ .

The entropy for each head, denoted as  $H(X_h)$ , is calculated using the formula  $\frac{1}{2}\log(2\pi e\sigma_{\mathrm{prod}}^2)$ . This entropy value reflects how the data is spread, influenced by parameters such as c, the mean offset, and the computed variance from the downsampled data. To capture the overall system entropy, including potential interactions among multiple heads, it is represented by the formula  $H(X) = \sum_{h=1}^H H(X_h) + \Delta$ . Here,  $\Delta$  accounts for additional entropy arising from the diversity and interactions across different heads, highlighting the ensemble effect of the multi-head Gaussian transformations. This approach allows DAAM to modulate attention distribution adaptively, balancing between broad and focused attention based on input characteristics.

#### **Traditional Self-Attention**

Traditional self-attention mechanisms are mathematically represented as Attention (Q,K,V)= softmax  $\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ . In this framework, the softmax function is applied to the scaled dot products of queries (Q) and keys (K), producing attention weights. Consider a vector  $z=\{z_1,z_2,\ldots,z_n\}$ , derived from these scaled dot products. Let  $S=\sum_{j=1}^n e^{z_j}$  denote the sum of exponential terms. The softmax values are represented as  $\left\{\frac{e^{z_1}}{S},\frac{e^{z_2}}{S},\ldots,\frac{e^{z_n}}{S}\right\}$ , with entropy  $H(\operatorname{softmax}(z))=-\sum_{i=1}^n\left(\frac{e^{z_i}}{S}\log\frac{e^{z_i}}{S}\right)$ . Typically, this entropy is low unless the z values are nearly identical, leading to a uniform softmax output. This low entropy results from the exponential nature of the softmax function, which tends to emphasize larger dot product values, thereby focusing attention on specific parts of the input.

In practical terms, the output of traditional self-attention often skews towards dominant features, concentrating attention and leading to lower entropy. Higher entropy, indicating a more uniform attention distribution, can be beneficial for tasks requiring a comprehensive view of all input data. Achieving higher entropy theoretically demands near-uniformity in the elements of z (the inputs to the softmax function). However, without modifications to the architecture—such as designing or constraining the weight matrices  $W^Q$  and  $W^K$  to produce similar outputs across different inputs—traditional self-attention mechanisms inherently produce lower entropy. This characteristic makes them less adaptable in scenarios demanding sensitivity to diverse and dynamic data elements, highlighting a limitation in their design for certain applications.

Higher entropy in an attention mechanism signifies a more balanced distribution of attention across various parts of the input data. This balance is crucial for ensuring that the model does not overly focus on a few features but instead considers a broader array of information. Such capability is vital in complex tasks where the input data is highly non-stationary. In contrast to traditional self-attention, the DAAM architecture dynamically adjusts its entropy in response to input characteristics. It provides both broad (high entropy) and focused (low entropy) attention distributions as needed, which is essential for effectively handling both highly non-stationary and stationary data environments.

# 1.1 Attention Mechanisms & Parameter Efficient Fine-Tuning

The Multi-Head Attention (MHA) mechanism in Transformer architectures uses parallel attention heads to enhance sequence modeling. Each head computes attention scores independently using the scaled dot-product attention formula:

$$\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V,$$

where  $d_k$  is the dimensionality of the keys. This enables each attention head to focus on different parts of the input, capturing diverse relationships within the data. The overall MHA is expressed as:

$$MHA(Q, K, V) = Concat(head_1, ..., head_H)W^O$$
,

with each head calculated as:

$$head = Attention(QW_i^Q, KW_i^K, VW_i^V).$$

This architecture allows the model to capture complex patterns and dependencies, improving performance in various tasks, such as machine translation and text summarization.

The Grouped Query Attention (GQA) mechanism serves as an intermediary between MHA and Multi-Query Attention (MQA). In GQA, query heads are grouped into G groups, with each group sharing a single key and value. This can be formulated as:

$$\mathrm{GQA}(Q,K,V,G) = \mathrm{Concat}(\mathrm{group}_1,\dots,\mathrm{group}_G)W^O.$$

GQA provides a balance between computational efficiency and the model's capacity to learn complex relationships, making it a valuable tool for scaling attention mechanisms in large models.

Low-Rank Matrix Adaptation (LoRA) is a technique designed for efficient model adaptation, commonly used in the Natural Language Processing domain. LoRA leverages low-rank decomposition to fine-tune pre-trained models by introducing low-rank updates to the weight matrices in neural networks. This is parameterized as  $\Delta W = AB^T$ , where  $A \in \mathbb{R}^{d \times r}$  and  $B \in \mathbb{R}^{d \times r}$  with  $r \ll d$ . This approach significantly reduces the number of parameters while maintaining the model's expressive power. The overall weight update in the model is represented as  $W' = W + \Delta W = W + AB^T$ , where W is the original weight matrix.

By adjusting the rank r, LoRA balances between model complexity and computational efficiency, proving particularly effective for scenarios requiring rapid adaptation to new tasks or domains. This method has demonstrated competitive performance with significantly reduced resource requirements, as noted in recent research [11, 12, 13].

Additionally, LoRA+ extends the original LoRA technique by introducing separate learning rates for the matrices A and B, thereby allowing more fine-grained control over the model adaptation process. This modification enhances the flexibility of LoRA by decoupling the optimization dynamics of the two matrices. Specifically, instead of a shared learning rate for both A and B, LoRA+ applies different learning rates,  $\eta_A$  and  $\eta_B$ , to each matrix during the training process, enabling improved convergence behavior and potentially better adaptation to the target task. This innovation has been explored in more recent literature [14], demonstrating that separate learning rates can further optimize performance while maintaining LoRA's benefits of parameter efficiency and adaptability.

Despite these efficiencies, LoRA still requires a relatively higher number of parameters—at least 24 times more than DAAM across the models used and at least 2 times for than DAT in this study, even when assuming the lowest rank possible, r=4. This parameter difference highlights the efficiency of DAAM and DAT in comparison, as illustrated in Table 1.

#### 1.2 General Purpose Pre-Trained Models

WavLM [5] is a large-scale pre-trained model designed to enhance speech processing capabilities by utilizing 94,000 hours of diverse audio inputs. Building upon the Hidden Unit BERT (HuBERT) framework [6], which primarily focuses on masked speech prediction, WavLM incorporates an additional denoising modeling component. This dual approach allows WavLM to handle a broader range of speech-related tasks effectively. At its core, WavLM utilizes self-supervised learning techniques, enabling the model to predict masked portions of audio inputs. Through this process, the model acquires a deeper understanding of speech patterns, nuances, and contextual information, thereby improving its performance in various speech processing applications.

Llama family of models [7] mark a significant advancement in Large Language Models (LLMs), leveraging an optimized auto-regressive transformer architecture.

Bidirectional Encoder Representations from Image Transformers (BEiT) [8] represents a breakthrough in self-supervised learning for vision tasks, drawing inspiration from the BERT [15] approach in natural language processing. BEiT utilizes Masked Image Modeling (MIM) as a pre-training strategy for vision transformers. In this approach, images are tokenized into discrete visual tokens, and a blockwise masking strategy is applied. The model then predicts the original visual tokens from these masked patches, focusing on learning high-level semantic representations directly from raw pixel data. This methodology allows BEiT to excel in downstream tasks such as image classification and semantic segmentation, surpassing other pre-training methods in both performance and fine-tuning stability.

#### 1.3 Datasets

The frozen encoder of each of the three PTM implementations (as described in Section 1) is used to train and evaluate a decoder on the *IEMOCAP* [16], *AG News* [17] and *CIFAR100* [18] datasets to assess the applicability of the newly proposed attention mechanisms across speech, text and image modalities. For the *IEMOCAP* dataset, we employ 5-fold cross-validation, training on 4 sessions and validating on 1. We focus on the emotion categories *neutral*, *happiness* (merging *happiness* and *excited*), *anger*, and *sadness*. *AG News* dataset is employed in our study. Our dataset construction

![](_page_4_Figure_0.jpeg)

Figure 1: Proposed model architecture showcasing a pre-trained model (i.e., the encoder) for feature extraction (i.e., embeddings) via its N transformer layers, followed by the attention module within the decoder network for selective emphasis, and concluding with probability output. The process flow is marked with the trainable and frozen states.

focuses solely on the title and description fields of these articles. In terms of data distribution, each category (out of four) contributed 30,000 articles to the training set and 1,900 articles to the validation set. For our analysis, we use the following division of the CIFAR-100 dataset: 50,000 images for training and 10,000 for validation. In the Multi-head DAAM, as outlined in Algorithm 1, the attention weights are determined utilizing multiple Gaussian probability density functions. In this formulation, the scaled variances are treated as learnable parameters, while the means are adjusted through learnable offsets. This methodology enables the model to dynamically adapt its attention focus in response to the distribution of the input data.

In a multi-head configuration, the DAAM process is independently applied across different heads, each focusing on distinct subspaces of the original input features. The final output (of all heads combined via stacking) is computed as an element-wise multiplication (Hadamard product) of the original input features and the Density attention weights.

This process enhances the model's ability to focus on contextually relevant information in the input sequence. All head outputs are stacked *vertically*, forming the Density Attention modulated Tensor (X'). An even more parameter-efficient solution of the DAAM – termed as the *Mixture of Densities Adaptive Attention Mechanism*, can be found in the Appendix (see Section B). Following the integration of Multi-Head DAAM, we investigate its compatibility with dot-product-based attention mechanisms (e.g., MHA, MQA, and GQA). Our focus on GQA is driven by its comparable performance to MHA, superior computational efficiency [19] and advantages of its hierarchical learning structure [20]. We refer to this method as the Grouped Query Density Adaptive Attention Mechanism (GQDAAM). The objective here is to showcase that DAAM can benefit PTMs across multiple modalities as a parameter-efficient fine-tuning method. The Computational complexity of DAAM can be analyzed as follows:  $O(n \cdot m)$ , where n is the batch size and m is the dimension size along normAxis.  $O(h \cdot n \cdot m)$ , with h as numHeads, allowing for parallelization.

# 1.4 Encoder and Decoder Models

We apply different attention mechanisms on SSL-based PTMs acting as PTMs to extract embeddings from. Specifically, we utilize the pre-trained model weights from three distinct PTMs: (i) WavLM-Large, (ii) Llama2-13B, and (iii) BEiT-Large.

Libri-light [21] GigaSpeech [22] and English parts of VoxPopuli [23] have been used for pre-training (i). ImageNet-1k [24] has been used for pre-training (iii). Conversely, while (ii) has undergone pre-training on undisclosed, publicly sourced textual data, this aspect does not impact our research. The downstream application we employ is different from the model's original pre-training task. The specific datasets used in this work are described in further detail in Section 1.3.

The role of PTMs within the proposed model architecture, as depicted in Figure 1, is crucial during the inference phase (post-training). It is important to note that in this study, PTMs are utilized in their original pre-trained state, eschewing any further re-training during the preprocessing stage. For each PTM under consideration, the encoder component remains static (frozen), allowing the focus

# Algorithm 1 Density Adaptive Attention Mechanism

```
Require: x (input tensor), normDimSize, normAxis, c, eps
Ensure: Attention-modified tensor
 1: Initialize c to a tensor of size (1, normDimSize) filled with c
 2: Initialize meanOffset to a tensor of size (1, normDimSize) filled with zeros
 3: for each batch in x do
       mean \leftarrow mean(x, dim = normAxis)
       \mathbf{var} \leftarrow \text{mean}(x^2, \text{dim} = \text{normAxis}) - \mathbf{mean}^2
 5:
       \mathbf{var} \leftarrow |\mathbf{var}| + 10^{-8}
 6:
       adjusted Mean \leftarrow mean + mean Offset
 7:
       yNorm \leftarrow (x - adjustedMean)/\sqrt{var + 10^{-5}}
 8:
       \mathbf{vTransform} \leftarrow \exp(-(\mathbf{vNorm}^2/(2 \cdot \mathbf{c})))
 9:
       x \leftarrow x \cdot \mathbf{vTransform}
11: end for
12: return x
```

| Mechanism                           | Heads             | N Gaussians | Parameters (Millions)              |
|-------------------------------------|-------------------|-------------|------------------------------------|
| GQDAAM                              | g: 8, q: 8, kv: 2 | 1024 - 5120 | 1.21 - 3.55 (DAAM – 0.016 - 0.082) |
| GQA                                 | q: 8, kv: 2       | N/A         | 1.19 - 3.47                        |
| LoRA+ $(r = \{4, 8\}, \alpha = 16)$ | N/A               | N/A         | 0.39 - 3.28                        |
| LoRA $(r = \{4, 8\}, \alpha = 16)$  | N/A               | N/A         | 0.39 - 3.28                        |
| DAAMv1 (with 2 conv. layers)        | g:8               | 1024 - 5120 | 0.22 - 0.45 (DAAM – 0.016 - 0.082) |
| DAAMv2 (with 2 conv. layers)        | g:1               | 1024 - 5120 | 0.22 - 0.45 (DAAM – 0.002 - 0.010) |

Table 1: Comparison of min and max learnable parameters (in millions) for various PEFT methods.

to be on training and subsequently evaluating the performance of the newly proposed decoder on the designated downstream task. This approach ensures that the intrinsic properties and learned representations of the PTMs are preserved, while the decoder adapts and fine-tunes to the specific requirements of the task at hand [25]. The output from each transformer layer (in the encoder) undergoes mean pooling across the time dimension (sequence length), followed by concatenation of these pooled outputs. These concatenated outputs then serve as input embeddings for the Attention Module, which employs either (i) Multi-Head Self-Attention, (ii) Multi-Head Density Adaptive Attention, or (iii) Multi-Head Grouped Query Density Adaptive Attention where (ii) and (iii) are contributions of this work. When (ii) or (iii) are used, the decoder network is termed as DAT.

In mathematical terms, the embeddings are represented as  $X \in \mathbb{R}^{N \times d}$ , where each  $x_i$  is a vector in a d-dimensional space, with d taking values in the set  $\{1024, 5120\}$ . Here, N signifies the total count of transformer layers in the encoder, which are kept in a static (frozen) state. The attention mechanism of the module then produces a new, contextualized representation  $C \in \mathbb{R}^{N \times d}$  for the input sequence. Subsequently, convolutional layers are utilized to distill features (pertaining to speech, text, or image data) from the context matrix generated by the attention mechanism. By employing 2-dimensional convolution layers (with kernel\_size = (3,3), stride = 1, and padding = 1), the model adeptly processes the array of context tensor outputs from each transformer layer.

Table 1 lists the attention mechanism parameters for the proposed DAAM-based decoders of WavLM-Large, Llama2-13B, and BEiT-Large encoders. Here, g denotes DAAM-based head count, with higher values indicating a higher number of learned Gaussian Distributions. q and kv are the counts of query and key-value heads, respectively. The embedding dimensions are 1024 for WavLM and BEiT, and 5120 for Llama2. We also benchmark against LoRA-based methods downstream task by fine-tuning the query and key projections modules.

All decoder network models are trained for 35 epochs and their layer weights (except their respective attention module) are initialized using Xavier initialization [26]. Adam [27] is used as the optimizer, with both weight decay factor of 0.1 and an initial learning rate of  $10^{-4}$  (except for when Llama 2 is used as an encoder, in which case it is  $5 \times 10^{-5}$ ). For the SER experiments, Focal Loss [28] is used, where  $\gamma = 2.5$ . For the text and image classification experiments the Cross-Entropy Loss is used. All DAAM-based attention modules are initialized with a mean offset,  $\delta = 0$ , where  $\delta \in 1 \times d$  and scaled variance,  $\xi = 2$ , where  $\xi \in 1 \times d$ . A batch size of 8 is used for SER and a batch size

of 32 for Text and Image Classification. Across all downstream tasks, mixed precision training is utilized. Regarding the SER downstream task – during training and evaluation, audio files are split to a maximum of 5 second clips. If an audio file exceeds 5 seconds in duration, a new audio file will be generated containing the excess audio. Each audio file is passed through the trained Encoder model. For the text classification downstream task, text is tokenized with maximum context length of 4096 during both training and evaluation. For the image classification downstream task, images are resized to  $224 \times 224$  during both training and evaluation.

#### 1.5 Evaluation Metrics

In this study, the primary evaluation metric is Accuracy (Acc.), calculated as the percentage of correct predictions to total predictions. Additionally, the Importance Factor (IF) is introduced, calculated using Density Attention weights (DA) from the attention module. IF is  $\frac{DA_{ij}-\min(DA)}{\max(DA)-\min(DA)}$  with IF  $\in$  [0, 1], indicating the relative importance of features in the model's decision process. Higher IF values indicate more significant features and vice versa. IF-based heatmaps are created by taking the arithmetic average of the generated Density Attention maps during validation and then applying the IF formula. They visually depict feature importance. Unlike traditional self-attention, where attention might skew towards a few dominant features, DAAM's Gaussian-based attention provides a more balanced spread. This helps in capturing a broader range of features, reducing the bias towards overly frequent features and focusing more on features that contribute meaningfully to the task. All experiments have been carried out on two A100 80GB NVIDIA Graphical Processing Units (GPUs).

## 2 Discussion

The motivation for this research arises from the broad range of downstream applications that could benefit from an improved attention mechanism, addressing the limitations inherent in the self-attention mechanism (or other dot-product attention mechanism variations) found in Transformer models, which rely on normalized dot-products. This presents an opportunity to explore more robust and explainable approaches.

Despite their widespread adoption, self-attention mechanisms face several limitations that can impact their performance. One significant challenge is the fixed-length context window, which can lead to sub-optimal outcomes, particularly for long sequences where distant elements may lack relevance [29]. Additionally, without inductive biases like locality [30], self-attention layers may require more data to learn patterns that other methods can capture more efficiently. Although self-attention is theoretically capable of capturing long-term dependencies, it can struggle in practice as sequence length increases [31]. Furthermore, the interpretability of self-attention mechanisms is limited; it primarily relies on correlation-based activations, which focus on pairwise similarities and may not effectively capture the most relevant context [32]. This makes it difficult to understand why certain parts of the input are prioritized, underscoring the need for more transparent and interpretable attention mechanisms/frameworks.

In our work, we introduce a significant enhancement to the Transformer model's attention mechanism: the (Multi-Head) Density Adaptive Attention Mechanism (DAAM). DAAM is designed to improve upon the standard self-attention mechanism in Transformers. Unlike conventional attention in the Transformer, which calculates weights based on dot-product between different weight matrices, DAAM employs a Gaussian-based modulation of input features instead. This approach enables the model to concentrate on the most pertinent features in a context-sensitive manner, thereby improving its capability to interpret and process sequential and spatial data.

DAAM's attention mechanism, can be applied in various domains like multimedia recommendation (as in [33]), image classification (aligning with Patrick et al.'s [34] robustness strategies), and text classification (enhancing accuracy in contexts like e-commerce as shown by Yıldırım et al. [35]), and can significantly enhance model performance. Its ability to dynamically recalibrate feature significance based on Gaussian parameters proves particularly beneficial, offering improved accuracy, robustness, and user experience across diverse and challenging real-world applications. Furthermore, DAAM's Gaussian-based modulation offers a more interpretable framework for Artificial Intelligence (AI), addressing the critical need for transparency and trustworthiness in real-world AI systems [36].

Our proposed DAAM mechanism learns multiple means and variances of input features in a Multi-Headed setting. This mechanism operates independently across different heads, each focusing on distinct, non-overlapping subspaces of the input features. By employing Gaussian modulation, DAAM assigns varying levels of importance to each feature, effectively generating local attention outputs from each head. These outputs are then combined to construct a comprehensive Global Attention map. Each head independently adjusts its means and variances, allowing for a focused approach to different skewness aspects in data subsets capturing a broader range of data characteristics, including asymmetries, and collectively, non-Gaussian traits. Unlike other approaches in the literature wherein no parameters of the Gaussian distribution are learned, and are thus hard-coded making them non-specific to the data they are used on [37, 38], only multiplicative parameters like the scaled variance are learned [39, 40], a pre-defined amplitude that is updated during training [41] which are all approaches that are limited because their attention framework can only explicitly model Gaussian traits behavior [42, 43].

#### 3 Results

# 3.1 Current Benchmarks & Analysis of Encoder Layer Contribution

In Table 2, we compare DAAM-based methods with those utilizing Multi-Head Self-Attention with and without Batch Normalization (BN); LoRA-based methods (rank of 4 and 8). BN assumes independent and identical distribution across mini-batches [44] and is applied immediately after MHA to maintain the normalization order consistent with the original Transformer architecture. DAAM, due to its multi-head structure, effectively handles variations in feature distribution, including shifts in mean offset and variance, as well as the ability to model a mixture of density distributions with varying attention weights. This capability is demonstrated in Table 2 and Figure 2. Such adaptability enables DAAM to outperform methods dependent on static feature distributions, such as BN. Unlike methods that assume data follows a single Gaussian distribution, DAAM can model multiple Gaussian distributions with varying parameters, allowing it to approximate any probability distribution. The results indicate that DAAM demonstrates superior performance in scenarios characterized by significant variations in Gaussian parameters, which suggests a need for modeling non-stationary data and potentially non-Gaussian characteristics. Conversely, in cases where such parameter variations are minimal, DAAM outperforms MHA.

Observing the data in Table 2, we note that speech data exhibits high variability in both central tendency (mean offset,  $\mu$ ) and spread (scaled variance,  $\sigma^2$ ). This variability reflects the highly non-stationary nature of speech [45], necessitating attention mechanisms that can dynamically adjust both focus ( $\mu$ ) and width ( $\sigma$ ) to capture the rapidly changing features essential for tasks such as emotion recognition. Text data, on the other hand, shows high mean variation, possibly due to changing semantic contexts, while the variance remains moderately stable, which aligns with the structured nature of language. In text processing, attention mechanisms need to focus primarily on tracking the shifting mean to match the changing semantic and syntactic focal points. In contrast, vision tasks show low variations in both mean and variance, indicating that features are relatively stable and consistent in their locations and spreads. This stability suggests that simpler attention mechanisms can be effective, maintaining a consistent focus and width suitable for tasks with minimal feature variability.

To understand the contribution of encoder layers, we employ heatmap visualizations of the Importance Factor (IF) to reveal how features within the frozen pre-trained models drive decision-making. We analyze the IF heatmap of the best-performing multi-head attention mechanism across each data modality, as indicated in Tables 2. For instance, in the context of *Speech Processing with WavLM-Large using DAAM*, Figure 2a shows a dense population of higher IF values at the lower layers, indicating these layers' significant role in modulating the input sequence. This suggests that foundational speech features are captured early on, with upper layers refining these features into more abstract representations. Conversely, *Text Processing with Llama2-13B using GQDAAM*, illustrated in Figure 2b, displays a more uniform distribution of IF across all layers, with a slight concentration in earlier layers. This pattern reflects a balanced approach to hierarchical feature extraction, where both lower and higher-level features are crucial, particularly those derived from the early to middle layers.

Similarly, *Digital Image Processing with BEiT-Large using GQDAAM* in Figure 2c emphasizes the importance of lower layer features, which is consistent with the need for early-stage feature extraction in visual tasks, such as identifying edges and textures. These variations in IF distribution highlight the unique information processing needs of each modality. While speech and image processing rely heavily on primary feature extraction, text processing requires a combination of fundamental and more complex feature identification. The insights gained from IF analysis not only enhance the explainability of the models but also provide a quantifiable measure of feature significance, facilitating more informed decisions in model refinement and adaptation.

![](_page_8_Figure_1.jpeg)

(a) Speech Processing with WavLM-Large using DAAM.

![](_page_8_Figure_3.jpeg)

(b) Text Processing with Llama2-13B using DAAM.

![](_page_8_Figure_5.jpeg)

(c) Digital Image Processing with BEiT-Large using DAAM.

Figure 2: IF values for different processing tasks with their respective models (with output feature number on the X-axis and layer number on the Y-axis).

| IEMOCAP 5-fold validation (F1-F5) using WavLM-Large                 |                                      |                  |                  |                  |                           |                     |  |  |  |  |
|---------------------------------------------------------------------|--------------------------------------|------------------|------------------|------------------|---------------------------|---------------------|--|--|--|--|
| Method                                                              | F1                                   | F2               | F3               | F4               | F5                        | $\mu \pm \sigma$    |  |  |  |  |
| LoRA+ (r=4)                                                         | 27.6                                 | 25.7             | 31.7             | 25.1             | 16.8                      | $25.4 \pm 4.87$     |  |  |  |  |
| LoRA+(r=8)                                                          | 27.6                                 | 28.3             | 20.5             | 20.6             | 24.6                      | $24.3 \pm 3.32$     |  |  |  |  |
| LoRA (r=4)                                                          | 49.9                                 | 51.5             | 58.2             | 52.6             | 52.7                      | $53.0 \pm 2.79$     |  |  |  |  |
| LoRA (r=8)                                                          | 49.4                                 | 51.8             | 61.5             | 48.7             | 55.1                      | $53.3 \pm 4.66$     |  |  |  |  |
| MHA                                                                 | 62.7                                 | 59.9             | 61.7             | 61.3             | 65.7                      | $62.3 \pm 2.00$     |  |  |  |  |
| $MHA \to BN$                                                        | 62.7                                 | 59.9             | 62.9             | 64.8             |                           |                     |  |  |  |  |
| DAAMv2                                                              | 66.1                                 | 60.0             | 66.3             | 65.2             | 65.2 65.4 $64.6 \pm 2.47$ |                     |  |  |  |  |
| GQDAAM                                                              | 66.5                                 | 65.4             | 68.7             | 65.9             | 66.8                      | $66.7 \pm 1.18$     |  |  |  |  |
| DAAMv1                                                              | 67.2                                 | 64.6             | 68.1             | 67.9             | 69.0                      | $67.4 \pm 1.49$     |  |  |  |  |
|                                                                     |                                      |                  |                  | `                |                           | BEiT-Large          |  |  |  |  |
| Method                                                              | R1                                   | R2               | R3               | R4               | R5                        | $\mu \pm \sigma$    |  |  |  |  |
| LoRA+(r=4)                                                          | 20.2                                 | 21.1             | 26.8             | 17.9             | 24.5                      | $22.1 \pm 3.17$     |  |  |  |  |
| LoRA+(r=8)                                                          | 25.0                                 | 32.9             | 22.9             | 29.1             | 27.5                      | $27.5 \pm 3.44$     |  |  |  |  |
| LoRA (r=4)                                                          | 35.7                                 | 32.3             | 31.5             | 36.2             | 40.1                      | $35.2 \pm 3.08$     |  |  |  |  |
| LoRA (r=8)                                                          | 38.1                                 | 40.0             | 42.3             | 41.6             | 39.6                      | $40.3 \pm 1.49$     |  |  |  |  |
| MHA                                                                 | 60.4                                 | 61.9             | 62.1             | 62.0             | 62.1                      | $61.7 \pm 0.75$     |  |  |  |  |
| $MHA \rightarrow BN$                                                | 63.0                                 | 67.1             | 69.5             | 63.9             | 67.0                      | $66.1 \pm 2.25$     |  |  |  |  |
| GQDAAM                                                              | 80.0                                 | 80.1             | 80.1             | 80.6             | 80.0                      | $80.1 \pm 0.24$     |  |  |  |  |
| DAAMv1                                                              | 79.9                                 | 80.2             | 80.2             | 80.7             | 80.7                      | $80.3 \pm 0.32$     |  |  |  |  |
| DAAMv2                                                              | 80.2                                 | 80.4             | 81.0             |                  |                           |                     |  |  |  |  |
| Modality Summary: Gaussian Parameters for Normalized Features       |                                      |                  |                  |                  |                           |                     |  |  |  |  |
| Modality                                                            | Modality Mean Offset Scaled Variance |                  |                  |                  |                           |                     |  |  |  |  |
| Speech                                                              |                                      | 0.06, 0.1        |                  | [1.88, 2.06]     |                           |                     |  |  |  |  |
| Text                                                                |                                      | [-0.05, 0.07]    |                  |                  | [1.94, 2.02]              |                     |  |  |  |  |
| Vision                                                              | [-(                                  | 0.02, 0.0        | )2]              | [1.98, 2.03]     |                           |                     |  |  |  |  |
|                                                                     |                                      |                  |                  | n (R1-R          | (3) using                 | Llama2-13B          |  |  |  |  |
| Method                                                              | R1                                   | R2               | R3               | $\mu \pm \sigma$ |                           |                     |  |  |  |  |
| LoRA+(r=4)                                                          | 93.4                                 | 65.9             | 92.8             | $84.0 \pm 12.8$  |                           |                     |  |  |  |  |
| LoRA+(r=8)                                                          | 95.0                                 | 69.8             | 94.6             | $86.5 \pm 11.8$  |                           |                     |  |  |  |  |
| DAAMv2                                                              | 94.4                                 | 94.5             | 94.6             |                  |                           | $4.5\pm0.08$        |  |  |  |  |
| $MHA \rightarrow BN$                                                | 94.5                                 | 94.5             | 94.7             |                  |                           | $4.6 \pm 0.11$      |  |  |  |  |
| MHA                                                                 | 94.4                                 | 94.5             | 94.8             | $94.6 \pm 0.16$  |                           |                     |  |  |  |  |
| DAAMv1                                                              | 94.5                                 | 94.5             | 94.7             | $94.6 \pm 0.11$  |                           |                     |  |  |  |  |
| LoRA (r=8)                                                          | 94.9                                 | 94.6             | 94.9             |                  |                           | $4.8 \pm 0.14$      |  |  |  |  |
| GQDAAM                                                              | 94.8                                 | 94.9             | 94.9             |                  |                           | $4.9 \pm 0.06$      |  |  |  |  |
| LoRA (r=4)                                                          | 95.1                                 | 94.5             | 95.3             |                  |                           | $5.0 \pm 0.30$      |  |  |  |  |
| High vs. Low IF Scores for IEMOCAP Validation using WavLM           |                                      |                  |                  |                  |                           |                     |  |  |  |  |
| Layer                                                               | F1                                   | F2               | F3               | F4               | F5                        | Avg.                |  |  |  |  |
| 9 (High)                                                            | <b>65.9</b> 62.8                     | <b>60.1</b> 58.9 | <b>64.4</b> 63.2 | <b>62.7</b> 62.0 | <b>67.0</b> 64.5          | <b>64.0</b><br>62.3 |  |  |  |  |
| 23 (Low)                                                            |                                      |                  |                  |                  |                           |                     |  |  |  |  |
| Accuracy for High and Low IF Layers using Llama2-13B and BEiT-Large |                                      |                  |                  |                  |                           |                     |  |  |  |  |
| Dataset                                                             |                                      | h IF La          |                  |                  |                           | v IF Layers         |  |  |  |  |
|                                                                     | 0.4                                  | 0 (10 0          | 11)              |                  | 0/                        | l.7 (37-39)         |  |  |  |  |
| AGNews                                                              |                                      | <b>.9</b> (19-2  |                  |                  |                           |                     |  |  |  |  |
| AGNews<br>CIFAR100                                                  |                                      | <b>.6</b> (10-1  |                  |                  |                           | 1.7 (22-24)         |  |  |  |  |

Table 2: Comparison of results from IEMOCAP, CIFAR100, modality summary, and AG News datasets, including IF layers for different methods. Best performance is indicated in bold.

# 3.2 Ablation Studies

We validate that the IF from DAAM and GQDAAM attention weights accurately identifies key feature extraction regions affecting model performance. This is achieved by reassessing experiments focused on layers with low and high IF scores, aiming to understand the link between IF scores and the significance of the highlighted features.

Across Speech, Text, and Vision modalities, higher IF scores consistently align with improved model performance and vice versa. For the Speech downstream task, High IF layer (Layer 9) consistently outperforms low IF layer (Layer 23) across all folds (Table 2). For the Text and Vision downstream tasks (Table 2), layers with higher IF achieve better performance, especially in Vision. In MHA,

![](_page_10_Figure_0.jpeg)

(a) Speech Processing with WavLM- (b) Text Processing with Llama2- (c) Image Processing with BEiT-Large using DAAM. 13B using DAAM. Large using DAAM.

Figure 3: Percentage contribution of each layer to attention weights in different downstream tasks (for best performing DAAM-based models using g:8).

attention weights primarily indicate the level of correlation between different parts of the input sequence [46]. Each element's weight reflects its relevance to every other element within the same sequence. However, this approach does not directly translate to the performance on downstream tasks. For instance, the authors in [25] derive normalized self-attention weights for SER on IEMOCAP using WavLM-Large, identifying layer 23 as pivotal. While useful for their use case, this only indicates inter-layer correlation, and not a direct link to better or worse performance and would be misleading to use these weights for that use case (as shown in Table 2 using DAAM for the same task). In contrast, DAAM-based learning dynamically adjusts attention weights tailoring attention to improve feature representation aligned with the model's end goal. Analysis of Figure 3 indicates earlier layers, exhibit more meaningful features, and contribute more to model performance, suggesting potential overparameterization in later layers [47].

## 4 Limitations & Future Work

DAAM's fixed number of Gaussians can limit its adaptability across different datasets and tasks. This can be improved by adopting a Bayesian approach to dynamically select the optimal number of Gaussians. Utilizing criteria like the Bayesian Information Criterion (BIC) or a neural network to predict the number based on input characteristics can enhance performance and efficiency, allowing the model to better adapt to varying data distributions and complexities. Future work should explore DAAM in additional tasks, datasets, grounding experiments [48], and beyond feature extraction, including model compression using attention weights during training (crucial for resource-limited applications) [49].

# 5 Conclusion

In this work, we introduce Multi-Head DAAM and the Density Adaptive Transformer. We demonstrate their effectiveness in enhancing model performance, particularly with highly non-stationary data such as Speech and Vision. Results show that combining learnable mean and variance for every Gaussian Distribution enables dynamic feature significance recalibration and approximation of any Probability Distribution across multiple modalities. Combining this mechanism with the dot-product attention mechanism enhances performance with a minimal increase in parameters (0.016%-0.08% compared to GQA models) and at least 44% less total parameters than LoRA. Finally, we introduce the Importance Factor for improved model explainability.

# References

- [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," *Advances in neural information processing systems*, vol. 30, 2017.
- [2] L. Wang, "Rrwkv: Capturing long-range dependencies in rwkv," arXiv preprint arXiv:2306.05176, 2023.
- [3] Y. Zhuang, J. Zhang, and M. Tu, "Long-range sequence modeling with predictable sparse attention," in *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 2022, pp. 271–281.
- [4] H. He, "A unified view of long-sequence models towards million-scale dependencies," *arXiv* preprint arXiv:2307.03172, 2023.
- [5] S. Chen and C. e. a. Wang, "Wavlm: Large-scale self-supervised pre-training for full stack speech processing," *IEEE Journal of Selected Topics in Signal Processing*, vol. 16, no. 6, pp. 1505–1518, 2022.
- [6] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "Hubert: Self-supervised speech representation learning by masked prediction of hidden units," 2021.
- [7] H. Touvron and L. M. et al., "Llama 2: Open foundation and fine-tuned chat models," 2023.
- [8] H. Bao, L. Dong, S. Piao, and F. Wei, "Beit: Bert pre-training of image transformers," 2022.
- [9] S. Chen, J. Xie, and J. H. L. Hansen, "Fearless: Feature refinement loss for ensembling self-supervised learning features in robust end-to-end speech recognition," in *Interspeech 2022, 23rd Annual Conference of the International Speech Communication Association, Incheon, Korea, 18-22 September 2022*, H. Ko and J. H. L. Hansen, Eds. ISCA, 2022.
- [10] J. Fluri, T. Kacprzak, A. Lucchi, A. Schneider, A. Réfrégier, and T. Hofmann, "Full wcdm analysis of kids-1000 weak lensing maps using deep learning," *Physical Review D*, 2022.
- [11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, L. Wang, and W. Chen, "Lora: Low-rank adaptation of large language models," *arXiv preprint arXiv:2106.09685*, 2021.
- [12] M. Ding, W. Zheng, X. Liu, W. Hong, X. Tian, and J. Tang, "Parameter-efficient fine-tuning by low-rank adaptation," *arXiv preprint arXiv:2203.08275*, 2022.
- [13] X. Li, M. Liang, Y. Shen, and L. Wang, "Towards parameter-efficient transfer learning for natural language processing," in *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*, 2021.
- [14] S. Hayou, N. Ghosh, and B. Yu, "Lora+: Efficient low rank adaptation of large models," 2024. [Online]. Available: https://arxiv.org/abs/2402.12354
- [15] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," in *North American Chapter of the Association for Computational Linguistics*, 2019. [Online]. Available: https://api.semanticscholar.org/CorpusID:52967399
- [16] C. Busso and M. e. a. Bulut, "Iemocap: Interactive emotional dyadic motion capture database," *Language Resources and Evaluation*, 2008.
- [17] X. Zhang, J. Zhao, and Y. LeCun, "Character-level convolutional networks for text classification," in *Advances in Neural Information Processing Systems*, C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, Eds., vol. 28. Curran Associates, Inc., 2015. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf
- [18] A. Krizhevsky, "Learning multiple layers of features from tiny images," Canadian Institute For Advanced Research, Tech. Rep., 2009.
- [19] J. Ainslie and L.-T. et al., "GQA: Training generalized multi-query transformer models from multi-head checkpoints," in *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, H. Bouamor, J. Pino, and K. Bali, Eds. Singapore: Association for Computational Linguistics, Dec. 2023, pp. 4895–4901. [Online]. Available: https://aclanthology.org/2023.emnlp-main.298

- [20] M. Chen and Y. e. a. Bai, *Towards Understanding Hierarchical Learning: Benefits of Neural Representations*. Curran Associates Inc., 2020.
- [21] J. Kahn and M. R. et al., "Libri-light: A benchmark for asr with limited or no supervision," in *International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020.
- [22] G. Chen and S. C. et al., "Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio," in *Interspeech*, 2021.
- [23] C. Wang, M. Riviere, A. Lee, A. Wu, C. Talnikar, D. Haziza, M. Williamson, J. Pino, and E. Dupoux, "VoxPopuli: A large-scale multilingual speech corpus for representation learning, semi-supervised learning and interpretation," in *International Joint Conference on Natural Language Processing*, 2021.
- [24] O. Russakovsky and J. D. et al., "Imagenet large scale visual recognition challenge," *International Journal of Computer Vision*, vol. 115, no. 3, pp. 211–252, 12 2015. [Online]. Available: https://doi.org/10.1007/s11263-015-0816-y
- [25] G. Ioannides, M. Owen, A. Fletcher, V. Rozgic, and C. Wang, "Towards Paralinguistic-Only Speech Representations for End-to-End Speech Emotion Recognition," in *Proc. INTERSPEECH* 2023, 2023, pp. 1853–1857.
- [26] X. Glorot and Y. Bengio, "Understanding the difficulty of training deep feedforward neural networks," in *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics*, ser. Proceedings of Machine Learning Research, Y. W. Teh and M. Titterington, Eds., vol. 9. Chia Laguna Resort, Sardinia, Italy: PMLR, 13–15 May 2010, pp. 249–256. [Online]. Available: https://proceedings.mlr.press/v9/glorot10a.html
- [27] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," 2017.
- [28] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," 2018.
- [29] Y. Li, "Unlocking context constraints of llms: Enhancing context efficiency of llms with self-information-based content filtering," *arXiv preprint arXiv:2304.12102*, 2023. [Online]. Available: https://dx.doi.org/10.48550/arXiv.2304.12102
- [30] B. L. Edelman, S. Goel, S. Kakade, and C. Zhang, "Inductive biases and variable creation in self-attention mechanisms," *International Conference on Machine Learning (ICML)*, 2022. [Online]. Available: https://arxiv.org/abs/2110.10090
- [31] M. Hahn, "Theoretical Limitations of Self-Attention in Neural Sequence Models," *Transactions of the Association for Computational Linguistics*, vol. 8, pp. 156–171, 2020. [Online]. Available: https://doi.org/10.1162/tacl\_a\_00306
- [32] M. Bhan, N. Achache, V. Legrand, A. Blangero, and N. Chesneau, "Evaluating self-attention interpretability through human-grounded experimental protocol," in *Explainable Artificial Intelligence*, L. Longo, Ed. Cham: Springer Nature Switzerland, 2023, pp. 26–46.
- [33] Z. Tao, X. Liu, Y. Xia, X. Wang, L. Yang, X. Huang, and T.-S. Chua, "Self-supervised learning for multimedia recommendation," *IEEE Transactions on Multimedia*, 2022. [Online]. Available: https://dx.doi.org/10.1109/TMM.2022.3187556
- [34] D. Patrick, M. Geyer, R. Tran, and A. Fernandez, "Reconstructive training for real-world robustness in image classification," in *IEEE Winter Conference on Applications of Computer Vision Workshops (WACVW)*, 2022. [Online]. Available: https://dx.doi.org/10.1109/WACVW54805.2022.00031
- [35] F. M. Yıldırım, A. Kaya, S. Öztürk, and D. Kılınç, "A real-world text classification application for an e-commerce platform," in *International Symposium on Advanced Electrical and Communication Technologies (ISAECT)*, 2019. [Online]. Available: https://dx.doi.org/10.1109/ASYU48272.2019.8946337
- [36] W. Jin, X. Li, and G. Hamarneh, "Rethinking ai explainability and plausibility," *arXiv preprint arXiv:2303.17707*, 2023. [Online]. Available: http://arxiv.org/pdf/2303.17707
- [37] W. You, S. Sun, and M. Iyyer, "Hard-coded gaussian attention for neural machine translation," in *Annual Meeting of the Association for Computational Linguistics*, 2020. [Online]. Available: https://api.semanticscholar.org/CorpusID:218487704

- [38] M. Guo, Y. Zhang, and T. Liu, "Gaussian transformer: A lightweight approach for natural language inference," *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 33, no. 01, pp. 6489–6496, Jul. 2019. [Online]. Available: https://ojs.aaai.org/index.php/AAAI/article/view/4614
- [39] D. Ruan, D. Wang, Y. Zheng, N. Zheng, and M. Zheng, "Gaussian context transformer," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, June 2021, pp. 15129–15138.
- [40] J. Kim, M. El-Khamy, and J. Lee, "T-gsa: Transformer with gaussian-weighted self-attention for speech enhancement," in *ICASSP 2020 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2020, pp. 6649–6653.
- [41] A. Luo, F. Yang, X. Li, L. Nie, C. Lin, H. Fan, and S. Liu, "Gaflow: Incorporating gaussian attention into optical flow," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, October 2023, pp. 9642–9651.
- [42] C. Chen and B. Li, "An interpretable channelwise attention mechanism based on asymmetric and skewed gaussian distribution," *Pattern Recognition*, vol. 139, p. 109467, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S003132032300167X
- [43] J. Xie, Z. Ma, D. Chang, G. Zhang, and J. Guo, "Gpca: A probabilistic framework for gaussian process embedded channel attention," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 11, pp. 8230–8248, 2022.
- [44] Y. Wang, Q. Shi, and T.-H. Chang, "Batch normalization damages federated learning on noniid data: Analysis and remedy," in *ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, 2023, pp. 1–5.
- [45] R. Jaiswal and D. Romero, "Implicit wiener filtering for speech enhancement in non-stationary noise," in 2021 11th International Conference on Information Science and Technology (ICIST), 2021, pp. 39–47.
- [46] G. Lovisotto, N. Finnie, M. Muñoz, C. K. Mummadi, and J. H. Metzen, "Give me your attention: Dot-product attention considered harmful for adversarial patch robustness," 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 15 213–15 222, 2022. [Online]. Available: https://api.semanticscholar.org/CorpusID:247748735
- [47] Q. Zhang, S. Zuo, C. Liang, A. Bukharin, P. He, W. Chen, and T. Zhao, "Platon: Pruning large transformer models with upper confidence bound of weight importance," 2022.
- [48] K. Li, J. Li, D. Guo, X. Yang, and M. Wang, "Transformer-based visual grounding with cross-modality interaction," *ACM Trans. Multimedia Comput. Commun. Appl.*, vol. 19, no. 6, may 2023. [Online]. Available: https://doi.org/10.1145/3587251
- [49] J. Back, N. Ahn, and J. Kim, "Magnitude attention-based dynamic pruning," 2023.

# A Reproducibility

The source code has been provided in the following GitHub repository: https://github.com/ gioannides/DAAM-PEFT-paper-code

# Multi-Head Mixture of Densities Adaptive Attention Mechanism Extension

This section presents an extension of the Multi-Head Density Adaptive Attention Mechanism (DAAM), focusing on enhancing the stability of the training process and the model's efficiency by **significantly** reducing the number of learnable parameters even further. The proposed method integrates multi-head attention mechanisms with Gaussian mixtures and skip connections to provide a more refined and adaptable approach to handling complex datasets.

The extended DAAM incorporates multiple attention heads, each with its Gaussian mixture model, to process different segments of the input tensor in parallel. This approach allows for a more diverse and comprehensive understanding of the data, leading to increased model robustness and efficiency.

Additionally, as illustrated in Algorithm 4 then add the original input features (X) to the augmented one (X') for enhanced stability during training (i.e.  $X' \leftarrow X' + X$ ).

# **B.1** Algorithmic Details

This algorithm which forms the core of the extended DAAM, implementing the Gaussian mixture model within each attention head can be found in Algorithm 2. Key elements include initialization of Gaussian parameters and mean offsets, and a forward pass handling.

## **Algorithm 2** Mixture of Densities Adaptive Attention Mechanism

**Require:** Input tensor x, norm axis normAxis, N Gaussians,  $\epsilon$ 

**Ensure:** Attention-modified x

- 1: Initialize m, c of size  $N, \mu \leftarrow \text{mean}(\mathbf{x}, \text{axis} = \text{normAxis}), \sigma^2 \leftarrow \text{var}(\mathbf{x}, \text{axis} = \text{normAxis}) + \epsilon$ ,  $mixture \leftarrow 1$
- 2: for i=0 to N-1 do 3:  $\mu_i^{\mathrm{adj}} \leftarrow \mu + \mathbf{m}[i]$

- 7: end for
- 8: Normalize mixture across normAxis
- 9:  $\mathbf{x}' \leftarrow \mathbf{x} \cdot \text{mixture}$
- 10: return  $\mathbf{x}'$

## **Algorithm 4** Density Block

**Require:** x (input tensor), normAxes, numHeads, numGaussians, paddingValue, eps

Ensure: Final modified tensor

- 1: **for** each layer in MultiHeadDensityAdaptiveAttention **do**
- $x \leftarrow \text{layer}(x) + x$
- 3: end for
- 4: return x

#### **Extended Results**

We repeat all experiments previously carried out but with the Mixture of DAAM instead (see Table 2). It is evident that Mixture of DAAM not only outperforms DAAM but it also reduces its overall

## Algorithm 3 Multi-Head Density Adaptive Attention

Require: x (input tensor), normDimSize, numHeads, normAxis, c

Ensure: Concatenated attention output tensor

- 1: Initialize an array attentionHeads of size numHeads
- 2: **for** i = 1 to numHeads **do**
- 3:  $attentionHeads[i] \leftarrow GAAM(normDimSize, normAxis, c, eps)$
- 4: end for
- 5: Initialize an empty list **outputs**
- 6: for each head in attentionHeads do
- 7:  $\mathbf{headOutput} \leftarrow \mathbf{apply}(\mathbf{head}, x)$
- 8: Append headOutput to outputs
- 9: end for
- 10: **output** ← concatenate(**outputs**, dim = normAxis)
- 11: return output

trainable parameter count significantly (see Table 2 inside the parentheses where only the parameters relevant to the attention mechanism are provided).

| Mechanism       | Heads | N Gaussians | Parameters |  |
|-----------------|-------|-------------|------------|--|
| Mixture of DAAM | g:8   | 4           | 64         |  |

Table 3: Number of learnable parameters for Mixture of Densities Adaptive Attention Mechanism.

| Method                                | F1   | F2   | F3   | F4   | F5   | $\mu \pm \sigma$ |
|---------------------------------------|------|------|------|------|------|------------------|
| Mixture of DAAM (with 2 conv. layers) | 67.8 | 69.5 | 65.1 | 68.7 | 67.8 | $67.9 \pm 1.35$  |

Table 4: IEMOCAP 5-fold validation (F1-F5) using WavLM-Large.

| Method                                | R1   | R2   | R3   | R4   | R5   | $\mu \pm \sigma$ |
|---------------------------------------|------|------|------|------|------|------------------|
| Mixture of DAAM (with 2 conv. layers) | 80.6 | 79.7 | 80.5 | 80.3 | 80.3 | $80.3 \pm 0.3$   |

Table 5: CIFAR100 5 Run validation (R1-R5) using BEiT-Large.

| Method                                | R1   | R2   | R3   | $\mu \pm \sigma$ |
|---------------------------------------|------|------|------|------------------|
| Mixture of DAAM (with 2 conv. layers) | 94.5 | 94.6 | 94.6 | $94.6 \pm 0.05$  |

Table 6: AG News 3 Run validation (R1-R3) using Llama2-13B.