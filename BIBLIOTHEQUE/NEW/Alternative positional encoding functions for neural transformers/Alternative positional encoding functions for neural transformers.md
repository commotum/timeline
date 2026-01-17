# ALTERNATIVE POSITIONAL ENCODING FUNCTIONS FOR NEURAL TRANSFORMERS

#### A PREPRINT

## © Ezequiel López-Rubio\*

Department of Computer Languages and Computer Science University of Málaga Bulevar Louis Pasteur, 35 29071 Málaga, Spain ezeqlr@lcc.uma.es

#### Macorís Decena-Giménez

Department of Systems Engineering and Automation University of Málaga Bulevar Louis Pasteur, 35 29071 Málaga, Spain macorisd@uma.es

#### Rafael Marcos Luque-Baena

Department of Computer Languages and Computer Science University of Málaga Bulevar Louis Pasteur, 35 29071 Málaga, Spain rmluque@uma.es

December 23, 2025

#### **ABSTRACT**

A key module in neural transformer-based deep architectures is positional encoding. This module enables a suitable way to encode positional information as input for transformer neural layers. This success has been rooted in the use of sinusoidal functions of various frequencies, in order to capture recurrent patterns of differing typical periods. In this work, an alternative set of periodic functions is proposed for positional encoding. These functions preserve some key properties of sinusoidal ones, while they depart from them in fundamental ways. Some tentative experiments are reported, where the original sinusoidal version is substantially outperformed. This strongly suggests that the alternative functions may have a wider use in other transformer architectures.

Keywords neural transformers · positional encoding · language models · periodic functions

# 1 Introduction

Transformer architectures are now the dominant models for sequence processing in natural language and other modalities, yet the underlying self-attention operation is permutation-invariant and thus has no intrinsic notion of order [Dufter et al., 2022]. To make Transformers sensitive to word order or temporal structure, positional encoding (PE) mechanisms inject information about token positions, either as absolute indices or as relative distances between tokens [Dufter et al., 2022]. The design of positional encoding has emerged as a central inductive bias that strongly affects performance, robustness, and length generalization [Dufter et al., 2022, Kazemnejad et al., 2023].

A standard Transformer layer applies content-based self-attention followed by position-wise feed-forward networks, which means that, without additional signals, reordering the input tokens leaves the output unchanged [Dufter et al., 2022, Vaswani et al., 2017]. Position information can be introduced at the input level, within attention matrices, or

<sup>\*</sup>Corresponding author: Ezequiel López-Rubio. Ezequiel López-Rubio and Rafael Marcos Luque-Baena are also with ITIS Software. Universidad de Málaga. C/ Arquitecto Francisco Peñalosa 18, 29010, Málaga, Spain

before the output, corresponding broadly to position embeddings, attention manipulation, and hybrid schemes [Dufter et al., 2022, Shaw et al., 2018]. An extensive survey shows that these approaches can be clustered into absolute and relative methods, with further distinctions by injection point and functional form [Dufter et al., 2022, Kazemnejad et al., 2023, Chi et al., 2022].

Absolute positional encodings assign each sequence index a dedicated vector that is combined with token embeddings, for instance via elementwise addition [Dufter et al., 2022, Vaswani et al., 2017]. The original Transformer uses fixed sinusoidal encodings, while many subsequent variants employ learned absolute embeddings whose parameters are optimized along with token embeddings [Dufter et al., 2022]. Absolute schemes are simple and effective on moderate sequence lengths but tend to generalize poorly when models are evaluated on contexts significantly longer than those seen during training [Kazemnejad et al., 2023].

Relative positional encodings instead represent the distance between token pairs and inject this information directly into the attention computation [Dufter et al., 2022, Shaw et al., 2018]. In these models, attention logits are augmented with terms that depend on the relative offset of query and key positions, allowing the model to learn distance-sensitive patterns that are invariant under global shifts of the sequence [Dufter et al., 2022, Shaw et al., 2018]. This perspective unifies a large family of methods, including those that use learned relative embeddings or structured functions of the distance [Dufter et al., 2022, Chi et al., 2022].

A prominent recent line of work develops kernelized relative positional embeddings for length extrapolation [Chi et al., 2022]. In this framework, distances between positions are mapped through conditionally positive definite kernels, which are then incorporated into the attention scores in a way that preserves the probabilistic interpretation of self-attention [Chi et al., 2022]. Empirical results show that appropriate kernel choices, such as logarithmic variants, can yield strong extrapolation to much longer sequences than those used during training, often outperforming standard relative schemes on language modeling benchmarks [Chi et al., 2022].

# 2 Methodology

Next, the proposed alternative positional encoding functions are detailed. Transformer architectures rely on positional encodings to inject order information into sequences processed by permutation-invariant self-attention layers [Vaswani et al., 2017].

The original Transformer uses a deterministic periodic encoding that assigns to each position  $m \in \{0, \dots, L-1\}$  a vector  $\text{PE}(m) \in \mathbb{R}^{d_{\text{model}}}$  [Vaswani et al., 2017, Kazemnejad, 2019]:

$$PE(m, 2i) = \varphi\left(\frac{m}{10000^{2i/d_{\text{model}}}}\right) \tag{1}$$

$$PE(m, 2i + 1) = \psi\left(\frac{m}{10000^{2i/d_{\text{model}}}}\right)$$
 (2)

where  $0 \le i < d_{\text{model}}/2$ . The standard choice for the periodic functions is sinusoidal, i.e.,  $\varphi = \sin \psi = \cos$ .

The encoding is then added to the token embeddings  $x_m$ ,

$$\tilde{\boldsymbol{x}}_m = \boldsymbol{x}_m + \mathrm{PE}(m), \tag{3}$$

before being fed to the first self-attention layer. The choice of exponentially spaced frequencies allows the model to represent relative offsets as approximately linear functions of the encodings and to extrapolate to longer sequences.

Rotary Positional Embedding (RoPE) encodes positions by rotating query and key vectors in a shared complex (or 2D) subspace, so that their inner product depends on relative position [Su et al., 2021, EleutherAI, 2024]. Consider a per-head query  $q_m \in \mathbb{R}^{d_k}$  and key  $k_n \in \mathbb{R}^{d_k}$  at positions m and n. Split each into  $d_k/2$  two-dimensional components and define a rotation for each pair:

$$R_{\theta}(m) = \begin{bmatrix} \psi(m\theta) & -\varphi(m\theta) \\ \varphi(m\theta) & \psi(m\theta) \end{bmatrix}$$
(4)

Again, the standard choice for the periodic functions is sinusoidal, i.e.,  $\varphi = \sin \psi = \cos \omega$ 

RoPE applies  $R_{\theta_j}(m)$  and  $R_{\theta_j}(n)$  to the j-th 2D component of  $q_m$  and  $k_n$ , respectively:

$$\boldsymbol{q}_{m}^{\prime}=R\left( m\right) \boldsymbol{q}_{m},\tag{5}$$

$$\mathbf{k}_{n}^{\prime} = R\left(n\right)\mathbf{k}_{n},\tag{6}$$

where R(m) is block-diagonal with 2D rotations along the diagonal. The attention is then computed using the rotated vectors:

$$e_{mn} = \frac{\mathbf{q}_m' \cdot \mathbf{k}_n'}{\sqrt{d_k}}. (7)$$

Because  $q'_m \cdot k'_n$  depends on m-n through the rotation angles, RoPE effectively injects relative position information while preserving absolute encodings.

Now, we propose to employ non sinusoidal functions for  $\varphi$  and  $\psi$ . Two restrictions are imposed to keep the key features of the original method:

- 1. Periodicity. The functions  $\varphi: \mathbb{R} \to \mathbb{R}$  and  $\psi: \mathbb{R} \to \mathbb{R}$  must be periodic real valued functions with period  $[0, 2\pi]$ .
- 2. Phase shift. The functions  $\varphi$  and  $\psi$  must have the same shape but different phase:

$$\psi\left(m\right) = \varphi\left(\frac{\pi}{2} - m\right) \tag{8}$$

Given (8), only the function  $\varphi$  must be specified, because  $\psi$  is readily obtained.

The periodic continuous piecewise linear function  $\varphi=\mathrm{tri}$ , the square wave function  $\varphi=\mathrm{sqw}$ , and the sawtooth function  $\varphi=\mathrm{saw}$  are proposed, where:

$$\operatorname{tri}(m) = \begin{cases} \frac{2m}{\pi} & \text{if } m \in \left[0, \frac{\pi}{2}\right] \\ -\frac{2}{\pi}m + 2 & \text{if } m \in \left[\frac{\pi}{2}, \frac{3\pi}{2}\right] \\ \frac{2m}{\pi} - 4 & \text{if } m \in \left[\frac{3\pi}{2}, 2\pi\right] \\ \operatorname{tri}\left(\operatorname{mod}\left(m, 2\pi\right)\right) & \text{otherwise} \end{cases}$$

$$(9)$$

$$\operatorname{sqw}(m) = \begin{cases} -1 & \text{if } m \in [0, \pi) \\ 1 & \text{if } m \in [\pi, 2\pi) \\ \operatorname{sqw}\left(\operatorname{mod}\left(m, 2\pi\right)\right) & \text{otherwise} \end{cases}$$
(10)

$$\operatorname{saw}(m) = \begin{cases} m & \text{if } m \in [0, \pi) \\ m - 2\pi & \text{if } m \in [\pi, 2\pi) \\ \operatorname{saw}(\operatorname{mod}(m, 2\pi)) & \text{otherwise} \end{cases}$$
(11)

where mod stands for the modulus of the floating point division

## 3 Experiments

All experiments were carried out on a single NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM) running Ubuntu 22.04, Python 3.10, and PyTorch 2.9.1. The code was based on the public implementation of the original Transformer [Vaswani et al., 2017] by Ko [2019]. More specifically, we used the fork by Zhao [2025] that already contained the pre-processed data required for training. Our own fork, available at https://github.com/macorisd/alt-positional-encoding-transformer, added four interchangeable positional encoding functions:

- 1. **Sinusoidal**, i.e. the original fixed encoding;
- 2. **Triangular**, equation (9);
- 3. **Square**, equation (10);
- 4. Sawtooth, equation (11).

The model architecture followed the Transformer base configuration [Vaswani et al., 2017], with  $d_{\rm model} = 512$ , N = 6 layers in both encoder and decoder, h = 8 attention heads, a feed–forward dimension of  $d_{\rm ff} = 2048$ , and a dropout probability of 0.1 applied to all sub–layers.

We trained and evaluated the model on the Multi30K English–German image–description dataset [Elliott et al., 2016], which provided parallel English–German captions aligned at the sentence level. All text was lowercased and tokenized

![](_page_3_Figure_2.jpeg)

Figure 1: The standard sinusoidal function and the proposed alternative functions.

at the word level using language–specific tokenizers. Special tokens were added to each sequence, including <sos> and <eos> to mark sentence boundaries, <unk> for out–of–vocabulary words, and <pad> for sequence padding. Separate vocabularies were built for the source (English) and target (German) languages using only the training split, discarding tokens with a frequency lower than two. Sentences were converted to sequences of token indices, with unseen tokens mapped to <unk>. During training, batches were dynamically padded to the maximum sequence length within each batch, with a maximum allowed length of 256 tokens. A batch size of 512 sentence pairs was used for all experiments.

To systematically compare the four positional encoding variants, we employed 10–fold cross–validation using only the Multi30K training split, which consists of 29,001 sentence pairs. The training data was randomly shuffled with a fixed seed and partitioned into 10 folds. For each positional encoding variant, we trained 10 models using 9 folds for training and 1 fold for validation, rotating the held-out fold across runs. Models were trained using the Adam optimizer [Kingma, 2014] with an initial learning rate of  $10^{-5}$  and weight decay of  $5 \times 10^{-4}$ . The learning rate was automatically decayed based on validation performance. Gradients were clipped to a maximum norm of 1.0, and training used cross-entropy loss with padding tokens ignored.

For each encoding function, we report the final training/validation loss and BLEU-4 after the last epoch, as well as the best validation BLEU-4 observed during training. Table 1 summarizes the mean performance across the 10 cross-validation folds.

Table 1: Average performance over 10-fold cross-validation (1000 epochs per fold). Values are reported as mean  $\pm$  standard deviation, and the best results are highlighted in bold.

| <b>Encoding Function</b> | Loss                               |                                    | BLEU-4                            |                                      |
|--------------------------|------------------------------------|------------------------------------|-----------------------------------|--------------------------------------|
|                          | Final Train                        | Final Val                          | Final                             | Best                                 |
| Sinusoidal               | $3.05 \pm 0.03$                    | $3.12 \pm 0.03$                    | $29.48 \pm 0.76$                  | $29.63 \pm 0.77$                     |
| Triangular               | $2.41 \pm 0.01$                    | $2.57 \pm 0.02$                    | $40.68 \pm 0.36$                  | $40.78 \pm 0.37$                     |
| Square<br>Sawtooth       | $2.64 \pm 0.07$<br>$2.41 \pm 0.08$ | $2.74 \pm 0.06$<br>$2.53 \pm 0.10$ | $34.54 \pm 1.54$ $40.77 \pm 2.65$ | $34.93 \pm 1.72$<br>$41.03 \pm 2.60$ |

Figure 2 and 3 provide a detailed view of the training dynamics underlying the summary statistics in Table 1. Figure 2 shows the evolution of the average training loss and validation loss across epochs for each positional encoding variant. Figure 3 shows the average validation BLEU-4 score across epochs for the same variants. All curves are averaged over the 10 cross-validation folds.

![](_page_4_Figure_2.jpeg)

Figure 2: Training dynamics of the loss across the four positional encoding variants. The plot shows the average training loss (solid lines) and validation loss (dashed lines) as a function of the training epoch. All curves are averaged over the 10 cross-validation folds.

#### 4 Discussion

The key features of our proposal and the main implications of the experimental results are discussed next. The sinusoidal functions have been the standard choice for most of the work done on fixed positional encoding for neural transformers. Our present work proposes three alternative periodic functions, each with its completely different features, that can also be employed for all kinds of neural transformers. Therefore, a suitable function can be chosen to fit the problem at hand. In particular, each of the proposed functions exhibits essential features that sinusoidal functions do not have:

- The piecewise linear function  $\varphi=$  tri, equation (9), has a piecewise constant slope that distributes its outputs more uniformly over the range of the function, as compared to the sinusoidal function that compresses the output values associated with input values close to integer multiples of  $\frac{\pi}{2}$  into smaller intervals of the output range.
- The square wave function  $\varphi = \text{sqw}$ , equation (10), quantizes the input values into a discrete set of possible output values. This way, the function is a quantizer of its input domain.
- The sawtooth function  $\varphi = \text{saw}$ , equation (11), has the same slope for all points where it is differentiable. Moreover it distributes its outputs uniformly over its range, like the piecewise linear function.

Given the wide range of applications of neural transformers, it is envisaged that each function may be the best performer for a particular set of problems.

The results of the experiments reported in Section 3 give some relevant information. For the benchmark problem considered, all three alternative functions clearly outperform the standard sinusoidal function, both in cross-entropy loss and BLEU score terms. The evolution of the learning process is stable in all cases, reaching a steady state. The best performing functions are triangular and sawtooth, although the triangular function is faster to learn. The number of epochs that the learning process takes to reach the steady state is critical since it is directly related to the energy consumption of such a process. Therefore, it may be more advantageous to choose a fast learning function (the triangular one in this case), rather than the best performer (the sawtooth function).

![](_page_5_Figure_2.jpeg)

Figure 3: Training dynamics of validation BLEU-4 across the four positional encoding variants. The plot shows the average validation BLEU-4 score as a function of the training epoch. All curves are averaged over the 10 cross-validation folds.

Future work includes more experimentation with a wider range of problems so that the preliminary results presented here are further validated. Other alternative periodic functions may also be considered.

# 5 Conclusions

A new set of functions for the positional encoding module of neural transformers has been proposed. These functions differ from the standard sinusoidal functions in several relevant ways. The proposal has been tested on a well-known benchmark task. It has been found that the alternative functions clearly surpass the original sinusoidal approach. These results open the path for more successful applications of these encoding functions. In particular, significant performance enhancements and energy savings may be obtained.

## **Author contributions**

Conceptualization, E.L.-R.; methodology, E.L.-R.; software, M.D.-G.; validation, E.L.-R. and R.M.L.-B.; formal analysis, E.L.-R.; investigation, E.L.-R.; resources, M.D.-G.; data curation, M.D.-G.; writing—original draft preparation, E.L.-R. and M.D.-G.; writing—review and editing, E.L.-R., M.D.-G. and R.M.L.-B.; visualization, M.D.-G.; supervision, E.L.-R.; project administration, E.L.-R.; funding acquisition, E.L.-R. and R.M.L.-B. All authors have read and agreed to the published version of the manuscript.

### References

Philipp Dufter, Martin Schmitt, and Hinrich Schütze. Position information in transformers: An overview. *Computational Linguistics*, 48(3):733–763, 2022.

Amir Kazemnejad, Shuai Zhang, and Yonatan Belinkov. The impact of positional encoding on length generalization in transformers. In *Advances in Neural Information Processing Systems*. Neural Information Processing Systems Foundation, 2023.

- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, 2017.
- Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In *Proceedings* of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2018.
- Ta-Chung Chi, Linxi Fan, and Alexander I. Rudnicky. Kerple: Kernelized relative positional embedding for length extrapolation. In *Advances in Neural Information Processing Systems*, pages 21438–21451. Neural Information Processing Systems Foundation, 2022.
- Sina Kazemnejad. Transformer architecture: The positional encoding. https://kazemnejad.com/blog/transformer\_architecture\_positional\_encoding/, 2019. Accessed 2025-12-14.
- Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 683–693, 2021.
- Eleuther AI. Rotary embeddings: A relative revolution. https://blog.eleuther.ai/rotary-embeddings/, 2024. Accessed 2025-12-14.
- Hyunwoong Ko. Transformer: Pytorch implementation of "attention is all you need". https://github.com/hyunwoongko/transformer, 2019. GitHub repository, accessed 27 Nov 2025.
- Zilin Zhao. transformer-translation: Pytorch implementation of "attention is all you need". https://github.com/sssn-tech/transformer-translation, 2025. GitHub repository, accessed 27 Nov 2025.
- Desmond Elliott, Stella Frank, Khalil Sima'an, and Lucia Specia. Multi30k: Multilingual english-german image descriptions. In *Proceedings of the 5th Workshop on Vision and Language*, pages 70–74, 2016.
- Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.