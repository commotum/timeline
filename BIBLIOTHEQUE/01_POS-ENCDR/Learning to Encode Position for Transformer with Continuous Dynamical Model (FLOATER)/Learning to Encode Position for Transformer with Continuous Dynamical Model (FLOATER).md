# Learning to Encode Position for Transformer with Continuous Dynamical Model

Xuanqing Liu<sup>†</sup>, Hsiang-Fu Yu<sup>‡</sup>, Inderjit Dhillon<sup>§‡</sup>, Cho-Jui Hsieh<sup>†</sup>

<sup>†</sup> UCLA § UT Austin ‡ Amazon Inc.

xqliu@cs.ucla.edu rofu.yu@gmail.com

inderjit@cs.utexas.edu chohsieh@cs.ucla.edu

#### **Abstract**

We introduce a new way of learning to encode position information for non-recurrent models, such as Transformer models. Unlike RNN and LSTM, which contain inductive bias by loading the input tokens sequentially, non-recurrent models are less sensitive to position. The main reason is that position information among input units is not inherently encoded, i.e., the models are permutation equivalent; this problem justifies why all of the existing models are accompanied by a sinusoidal encoding/embedding layer at the input. However, this solution has clear limitations: the sinusoidal encoding is not flexible enough as it is manually designed and does not contain any learnable parameters, whereas the position embedding restricts the maximum length of input sequences. It is thus desirable to design a new position layer that contains learnable parameters to adjust to different datasets and different architectures. At the same time, we would also like the encodings to extrapolate in accordance with the variable length of inputs. In our proposed solution, we borrow from the recent Neural ODE approach, which may be viewed as a versatile continuous version of a ResNet. This model is capable of modeling many kinds of dynamical systems. We model the evolution of encoded results along position index by such a dynamical system, thereby overcoming the above limitations of existing methods. We evaluate our new position layers on a variety of neural machine translation and language understanding tasks, the experimental results show consistent improvements over the baselines.

#### 1 Introduction

Transformer based models [1, 2, 3, 4, 5, 6] have become one of the most effective approaches to model sequence data of variable lengths. Transformers have shown wide applicability to many natural language processing (NLP) tasks such as language modeling [4], neural machine translation (NMT) [1], and language understanding [2]. Unlike traditional recurrent-based models (e.g., RNN or LSTM), Transformer utilizes a non-recurrent but self-attentive neural architecture to model the dependency among elements at different positions in the sequence, which leads to better parallelization using modern hardware and alleviates the vanishing/exploding gradient problem in traditional recurrent models

[7] prove that the design of self-attentive architecture leads to a family of permutation equivalence functions. Thus, for applications where the ordering of the elements matters, how to properly encode position information is crucial for Transformer based models. There have been many attempts to encode position information for the Transformer. In the original Transformer paper [1], a family of

pre-defined sinusoidal functions was adapted to construct a set of embeddings for each position. These fixed position embeddings are then added to the word embeddings of the input sequence accordingly. To further construct these position embeddings in a more data-driven way, many recent Transformer variants such as [2, 8] include these embeddings as learnable model parameters in the training stage. This data-driven approach comes at the cost of the limitation of a fixed maximum length of input sequence  $L_{\rm max}$  and the computational/memory overhead of additional  $L_{\rm max} \times d$  parameters, where  $L_{\rm max}$  is usually set to 512 in many applications, and d is the dimension of the embeddings. [9] propose a relative position representation to reduce the number of parameters to (2K+1)d by dropping the interactions between tokens with a distance greater than K. In addition to just the input layer, [10] and [5] suggest that the injection of position information to every layer leads to even better performance for the Transformer.

An ideal position encoding approach should satisfy the following three properties:

- 1. **Inductive**: the ability to handle sequences longer than any sequence seen in the training time.
- 2. **Data-Driven**: the position encoding should be learnable from the data.
- 3. **Parameter Efficient**: number of trainable parameters introduced by the encoding should be limited to avoid increased model size, which could hurt generalization.

In Table 1, we summarize some of the existing position encoding approaches in terms of these three properties.

In this paper, we propose a new method to encode position with minimum cost. The main idea is to model position encoding as a continuous dynamical system, so we only need to learn the system dynamics instead of learning the embeddings for each position independently. By doing so, our method enjoys the best of both worlds – we bring back the inductive bias, and the encoding method is freely trainable while being parameter efficient. To enable training of this dynamical system with backpropagation, we adopt the recent progress in continuous neural network [11], officially called Neural ODE. In some generative modeling literature, it is also called the free-form flow model [12], so we call our model **FLOw-bAsed TransformER** (**FLOATER**). We highlight our contributions as follows:

- We propose FLOATER, a new position encoder for Transformer, which models the position information via a continuous dynamical model in a data-driven and parameter-efficient manner.
- Due to the use of a continuous dynamic model, FLOATER can handle sequences of any length. This property makes inference more flexible.
- With careful design, our position encoder is **compatible** with the original Transformer; *i.e.*, the original Transformer can be regarded as a special case of our proposed position encoding approach. As a result, we are not only able to train a Transformer model with FLOATER from scratch but also plug FLOATER into most existing pre-trained Transformer models such as BERT, RoBERTa, *etc*.
- We demonstrate that FLOATER consistent improvements over baseline models across a variety of NLP tasks ranging from machine translations, language understanding, and question answering.

# 2 Background and Related Work

#### 2.1 Importance of Position Encoding for Transformer

We use a simplified self-attentive sequence encoder to illustrate the importance of position encoding in the Transformer. Without position encoding, the Transformer architecture can be viewed as a stack of N blocks  $B_n: n=1,\ldots,N$  containing a self-attentive  $A_n$  and a feed-forward layer  $F_n$ . By dropping the residual connections and layer normalization, the architecture of a simplified Transformer encoder

Table 1: Comparing position representation methods

| Methods        | Inductive    | Data-Driven | Parameter Efficient |
|----------------|--------------|-------------|---------------------|
| Sinusoidal [1] | ✓            | X           | ✓                   |
| Embedding [2]  | X            | ✓           | X                   |
| Relative [9]   | X            | ✓           | ✓                   |
| This paper     | $\checkmark$ | ✓           | ✓                   |

can be represented as follows.

$$\operatorname{Encode}(\boldsymbol{x}) = B_N \circ B_{N-1} \circ \cdots \circ B_1(\boldsymbol{x}), \tag{1}$$

$$B_n(\mathbf{x}) = F_n \circ A_n(\mathbf{x}), \qquad (2)$$

where  $\boldsymbol{x} = [\boldsymbol{x}_1, \boldsymbol{x}_2, \dots, \boldsymbol{x}_L]^{\top} \in \mathbb{R}^{L \times d}$ , L is the length of the sequence and d is the dimension of the word embedding.  $A_n(\cdot)$  and  $F_n(\cdot)$  are the self-attentive and feed-forward layer in the n-th block  $B_n(\cdot)$ , respectively.

Each row of  $A_1(\boldsymbol{x})$  can be regarded as a weighted sum of the value matrix  $\boldsymbol{V} \in \mathbb{R}^{L \times d}$ , with the weights determined by similarity scores between the key matrix  $\boldsymbol{K} \in \mathbb{R}^{L \times d}$  and query matrix  $\boldsymbol{Q} \in \mathbb{R}^{L \times d}$  as follows:

$$A_1(\boldsymbol{x}) = \operatorname{Softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^{\top}}{\sqrt{d}}\right)\boldsymbol{V},$$

$$\boldsymbol{Q} = [\boldsymbol{q}_1, \boldsymbol{q}_2, ..., \boldsymbol{q}_L]^{\top}, \qquad \boldsymbol{q}_i = \boldsymbol{W}_q\boldsymbol{x}_i + \boldsymbol{b}_q,$$

$$\boldsymbol{K} = [\boldsymbol{k}_1, \boldsymbol{k}_2, ..., \boldsymbol{k}_L]^{\top}, \qquad \boldsymbol{k}_i = \boldsymbol{W}_k\boldsymbol{x}_i + \boldsymbol{b}_k,$$

$$\boldsymbol{V} = [\boldsymbol{v}_1, \boldsymbol{v}_2, ..., \boldsymbol{v}_L]^{\top}, \qquad \boldsymbol{v}_i = \boldsymbol{W}_v\boldsymbol{x}_i + \boldsymbol{b}_v,$$
(3)

 $W_{q/k/v}$  and  $b_{q/k/v}$  are the weight and bias parameters introduced in the self-attentive function  $A_1(\cdot)$ . The output of the feed-forward function  $F_1(\cdot)$  used in the Transformer is also a matrix with L rows. In particular, the i-th row is obtained as follows.

the *i*-th row of 
$$F_1(\mathbf{x}) = \mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{x}_i + \mathbf{b}_1) + \mathbf{b}_2,$$
 (4)

where  $W_{1,2}$  and  $b_{1,2}$  are the weights and biases of linear transforms, and  $\sigma(\cdot)$  is the activation function. It is not hard to see from (3) and (4) that both  $A_1(\cdot)$  and  $F_1(\cdot)$  are permutation equivalent. Thus, we can conclude that the entire function defined in (1) is also permutation equivalent, i.e.,  $\Pi \times \operatorname{Encode}(\boldsymbol{x}) = \operatorname{Encode}(\Pi \times \boldsymbol{x})$  for any  $L \times L$  permutation matrix  $\Pi$ . This permutation equivalence property restricts the Transformer without position information from modeling sequences where the ordering of elements matters.

#### 2.2 Position Encoding in Transformer

As mentioned in Section 1, there are many attempts to inject position information in self-attentive components. Most of them can be described in the following form:

$$B_n(\mathbf{x}) = F_n \circ A_n \circ \Phi_n(\mathbf{x}), \ n \in \{1, ..., N\},$$
 (5)

![](_page_3_Figure_0.jpeg)

Figure 1: The architecture of our model (FLOATER). The main differences between FLOATER and the original Transformer model are: 1) the position representation is integrated into each block in the hierarchy (there are N blocks in total); and 2) there is a dynamical model (see (8)) that generates position encoding vectors for each block. The dynamics are solved with a black-box ODE solver detailed in the supplementary material.

where  $\Phi_n(x)$  is a position encoding function.

[1] propose to keep  $\Phi_n(\boldsymbol{x}) = \boldsymbol{x}, \forall n \geq 2$  and inject position information only at the input block with a family of pre-defined sinusoidal functions:  $\Phi_1(\boldsymbol{x}) = \boldsymbol{x} + \boldsymbol{p}^{(1)}$ , where  $\boldsymbol{p}^{(1)} = [\boldsymbol{p}_1^{(1)}, \boldsymbol{p}_2^{(1)}, ..., \boldsymbol{p}_L^{(1)}]$  is a position embedding matrix with the *i*-th row corresponding to the *i*-th position in the input sequence. In particular, the *j*-th dimension of the *i*-th row is defined as follows.

$$\mathbf{p}_{i}^{(1)}[j] = \begin{cases} \sin(i \cdot c^{\frac{j}{d}}) & \text{if } j \text{ is even,} \\ \cos(i \cdot c^{\frac{j-1}{d}}) & \text{if } j \text{ is odd,} \end{cases}$$
 (6)

where  $c=10^{-4}$ . [10] and [5] observe better performance by further injecting the position information at each block, i.e.,  $\Phi_n(x) = x + p^{(n)}$  as follows:

$$\boldsymbol{p}_{i}^{(n)}[j] = \begin{cases} \sin(i \cdot c^{\frac{j}{d}}) + \sin(n \cdot c^{\frac{j}{d}}) & \text{if } j \text{ is even,} \\ \cos(i \cdot c^{\frac{j-1}{d}}) + \cos(n \cdot c^{\frac{j-1}{d}}) & \text{if } j \text{ is odd.} \end{cases}$$
 (7)

Note that for the above two approaches, position encoding functions  $\Phi_n(\cdot)$  are fixed for all the applications. Although no additional parameters are introduced in the model, both approaches are inductive and can handle input sequences of variable length.

Many successful variants of pre-trained Transformer models, such as BERT [2] and RoBERTa [8], include the entire embedding matrix  $p^{(1)} \in \mathbb{R}^{L \times d}$  in  $\Phi_1(x)$  as training parameters. As the number of training parameters needs to be fixed, the maximum length of a sequence,  $L_{\max}$ , is required to be determined before the training. Although it lacks the inductive property, this data-driven approach is found to be effective for many NLP tasks. Note that, unlike the fixed sinusoidal position encoding, there is no attempt to inject a learnable position embedding matrix at each block for Transformer due to a large number of additional parameters  $(NL_{\max}d)$ .

## 3 FLOATER: Our Proposed Position Encoder

We introduce our method in three steps. In the first step, we only look at one Transformer block, and describe how to learn the position representation driven by a dynamical system; in the second step, we show how to save parameters if we add position signals to every layer; lastly, we slightly change the architecture to save trainable parameters further and make FLOATER "compatible" with the original Transformer [1]. The compatibility means our model is a strict superset of the vanilla Transformer so that it can be initialized from the Transformer.

#### 3.1 Position Encoding with Dynamical Systems

Position representations in Transformer models are a sequence of vectors  $\{p_i \in \mathbb{R}^d : i = 1, ..., L\}$  to be added to the sequence of the input representations  $\{x_i : i = 1, ..., L\}$ . Existing position encoding approaches either apply a fixed sinusoidal function to obtain  $\{p_i\}$ , or include them as uncorrelated learnable parameters. Both of them fail to capture the dependency or dynamics among these position representations  $\{p_i\}$ . In this paper, we propose to use a dynamical system to model these position representations; that is, there is a "latent force" denoted by  $h_i$  that drives the changes from  $p_i$  to  $p_{i+1}$ . To encourage smoothness, we consider  $p(t) : \mathbb{R}_+ \to \mathbb{R}^d$  as the continuous version of the discrete sequence  $\{p_i\}$ . In particular, our proposed continuous dynamical system is characterized as follows:

$$p(t) = p(s) + \int_{s}^{t} h(\tau, p(\tau); \boldsymbol{\theta}_h) d\tau, \ 0 \le s \le t < \infty,$$
 (8)

together with an initial vector p(0), where  $h(\tau, p(\tau); \theta_h)$  is a neural network parameterized by  $\theta_h$  and takes the previous state  $(\tau, p(\tau))$ . Notice that the domain of  $p(\cdot)$  is  $\mathbb{R}_+$ . The position sequence  $\{p_i\}$  can be obtained by taking  $p(\cdot)$  on a series of points  $\{t_i: 0 \leq t_1 < \cdots \leq t_L\}$ :  $p_i = p(t_i)$ . One simple strategy is to set  $t_i = i \cdot \Delta t$  so that the points are equidistant, where  $\Delta$  is a hyperparameter (e.g.,  $\Delta = 0.1$ ). With this strategy, we are implicitly assuming the position signals evolve steadily as we go through each token in a sentence. In general,  $\{t_i\}$  can be any monotonically increasing series, which allows us to extend our work to more applications where the elements in the sequence are not always observed with the same interval. More discussions about the applicability for this general setting is included in the Supplementary material. For the NLP applications discussed in this paper, we choose  $t_i = i \cdot \Delta t$ .

Eq. (8) is equivalent to an ODE problem  $\frac{d\mathbf{p}(t)}{dt} = \mathbf{h}(t, \mathbf{p}(t); \boldsymbol{\theta}_h)$ , which is guaranteed to have a unique solution under mild conditions [13]. We follow the efficient approach by [11] to calculate the gradients of  $\boldsymbol{\theta}_h$  with respect to the overall training loss, which allows us to include this parameterized dynamical position encoder into the end-to-end training of Transformer models. More details can be found in the Supplementary material.

Our dynamical system (8) is quite flexible to admit the standard sinusoidal position encoding (6) as a special case:

$$\begin{aligned} & \boldsymbol{p}_{i+1}[j] - \boldsymbol{p}_{i}[j] \\ &= \begin{cases} \sin\left((i+1) \cdot c^{\frac{j}{d}}\right) - \sin\left(i \cdot c^{\frac{j}{d}}\right) & \text{if } j \text{ is even} \\ \cos\left((i+1) \cdot c^{\frac{j-1}{d}}\right) - \cos\left(i \cdot c^{\frac{j-1}{d}}\right) & \text{if } j \text{ is odd} \end{cases} \\ &= \begin{cases} \int_{i}^{i+1} c^{-\frac{j}{d}} \cos(\tau \cdot c^{\frac{j}{d}}) \, \mathrm{d}\tau & \text{if } j \text{ is even} \\ \int_{i}^{i+1} - c^{-\frac{j-1}{d}} \sin(\tau \cdot c^{\frac{j-1}{d}}) \, \mathrm{d}\tau & \text{if } j \text{ is odd,} \end{cases} \end{aligned}$$
(9)

This indicates that for simple sinusoidal encoding, there exists a dynamical system  $h(\cdot)$  which is also sinusoidal function.

#### 3.2 Parameter Sharing among Blocks

As mentioned in Section 2, injecting position information to each block for Transformer leads to better performance [10, 5] in some language understanding tasks. Our proposed position encoder FLOATER (8) can also be injected into each block. The idea is illustrated in Figure 1. Typically there are 6 blocks in sequence-to-sequence Transformer and 12 or 24 blocks in BERT. We add a superscript (n) to denote dynamics at n-th block:

$$p^{(n)}(t) = p^{(n)}(s) + \int_{s}^{t} h^{(n)}(\tau, p^{(n)}(\tau); \theta_{h}^{(n)}) d\tau.$$

As we can imagine, having N different dynamical models  $\boldsymbol{h}^{(n)}(\cdot;\boldsymbol{\theta}_h^{(n)})$  for each block can introduce too many parameters and cause significant training overhead. Instead, we address this issue by sharing parameters across all the blocks, namely

$$\boldsymbol{\theta}_h^{(1)} = \boldsymbol{\theta}_h^{(2)} = \dots = \boldsymbol{\theta}_h^{(N)}.$$
 (10)

Note that (10) does not imply that all the  $p_t^{(n)}$  are the same, as we will assign different initial values for each block, that is  $p^{(n_1)}(0) \neq p^{(n_2)}(0)$  for  $n_1 \neq n_2$ .

|                                         | Transformer-Base |       | Transformer-Large |       |
|-----------------------------------------|------------------|-------|-------------------|-------|
|                                         | En-De            | En-Fr | En-De             | En-Fr |
| Position encoders at all blocks         |                  |       |                   |       |
| FLOATER                                 | 28.6             | 41.6  | 29.2              | 42.7  |
| Pre-defined Sinusoidal Position Encoder | 28.2             | 40.6  | 28.4              | 42.0  |
| Fixed-length Position Embedding         | 26.9             | 40.9  | 28.3              | 42.0  |
| Position encoder only at input block    |                  |       |                   |       |
| FLOATER                                 | 28.3             | 41.1  | 29.1              | 42.4  |
| Pre-defined Sinusoidal Position Encoder | 27.9             | 40.4  | 28.4              | 41.8  |
| Fixed-length Position Embedding         | 27.8             | 40.9  | 28.5              | 42.4  |

Table 2: Experimental results of various position encoders on the machine translation task.

#### 3.3 Compatibility and Warm-start Training

In this section, we change the way to add position encoding so that our FLOATER can be directly initialized from Transformer. As an example, we use the standard Transformer model, which has a fixed sinusoidal encoding at the input block and no position encoding at deeper levels. Note that this technique can be extended to other variants of Transformers with different position encoding methods,

such as embedding matrix. We first examine the standard Transformer model, the query matrix  $\mathbf{Q}^{(n)}$  at block-n is

$$\widetilde{\boldsymbol{q}}_{i}^{(n)} = \boldsymbol{W}_{q}^{(n)} \left( \boldsymbol{x}_{i} + \widetilde{\boldsymbol{p}}_{i}^{(n)} \right) + \boldsymbol{b}_{q}^{(n)}, \tag{11}$$

where  $\boldsymbol{W}_{q}^{(n)}$  and  $\boldsymbol{b}_{q}^{(n)}$  are parameters in  $A_{n}$  (3);  $\boldsymbol{\widetilde{p}}^{(n)}$  is the sinusoidal encoding;  $\boldsymbol{\widetilde{q}}_{i}^{(n)}$  is the *i*-th row of  $\boldsymbol{Q}^{(n)}$ . Here we add a tilde sign to indicate the sinusoidal vectors. Formulas for  $\boldsymbol{\widetilde{k}}_{i}^{(n)}$  and  $\boldsymbol{\widetilde{v}}_{i}^{(n)}$  have a very similar form and are omitted for brevity.

Now we consider the case of FLOATER, where new position encodings  $p_i$  are added

$$q_{i}^{(n)} = \boldsymbol{W}_{q}^{(n)} (\boldsymbol{x}_{i} + \boldsymbol{p}_{i}) + \boldsymbol{b}_{q}^{(n)}$$

$$= \underbrace{\boldsymbol{W}_{q}^{(n)} (\boldsymbol{x}_{i} + \widetilde{\boldsymbol{p}}_{i}^{(n)}) + \boldsymbol{b}_{q}^{(n)}}_{\text{Eq. (11)}} + \underbrace{\boldsymbol{W}_{q}^{(n)} (\boldsymbol{p}_{i} - \widetilde{\boldsymbol{p}}_{i}^{(n)})}_{\text{Extra bias term depends on } i}$$

$$= \widetilde{\boldsymbol{q}}_{i}^{(n)} + \boldsymbol{b}_{q,i}^{(n)}.$$
(12)

It is easy to see that the changing the position embedding from  $\{ \widecheck{p}_i^{(n)} \}$  to  $\{ p_i^{(n)} \}$  is equivalent to adding a position-aware bias vector  $\boldsymbol{b}_{q,i}^{(n)}$  into each self-attentive layers  $\{ A_n(\cdot) \}$ . As a result, we can instead apply (8) to model the dynamics of  $\boldsymbol{b}_q^{(n)}$ . In particular, we have the following dynamical system:

$$\boldsymbol{b}_{q}^{(n)}(t) = \boldsymbol{b}_{q}^{(n)}(0) + \int_{0}^{t} \boldsymbol{h}^{(n)}(\tau, \boldsymbol{b}_{q}^{(n)}(\tau); \boldsymbol{\theta}_{h}) d\tau.$$
 (13)

After that, we set  $b_{q,i}^{(n)} = b_q^{(n)}(i \cdot \Delta t)$ . We can see that if  $h(\cdot) = 0$  and  $b_q^{(n)}(0) = 0$ , then  $b_q^{(n)} \equiv 0$ . This implies (12) degenerates to (11). Note that (13) has the same form as (8), except that we are now modeling the bias terms  $b_{q,i}$  in (3). We will apply the same technique to K and V.

To summarize, our model has a tight connection to the original Transformer: if we set all dynamical models to zero, which means  $h(\tau, p(\tau); \theta_h) \equiv 0$ , then our FLOATER model will be equivalent to the original Transformer with the sinusoidal encoding. The same trick also works for Transformer with position embedding such as BERT [2].

We strive to make our model compatible with the original Transformer due to the following reasons. First of all, the original Transformer is faster to train as it does not contain any recurrent computation; this is in contrast to our dynamical model (8), where the next position  $p_{i+1}$  depends on the previous one  $p_i$ . By leveraging the compatibility of model architecture, we can directly initialize FLOATER model from a pre-trained Transformer model checkpoint and then fine-tune for the downstream task for a few more epochs. By doing so, we enjoy all the benefits of our FLOATER model but still maintain an acceptable training budget. Likewise, for models such as BERT or Transformer-XL, we already have well-organized checkpoints out of the box for downstream tasks. These models are costly to train from scratch, and since our goal is to examine whether our proposed position representation method can improve over the original one, we decided to copy the weights layer by layer for attention as well as FFN layers, and randomly initialize the dynamical model  $h(\tau, p(\tau); \theta_h)$ .

# 4 Experimental Results

In this section, we perform experiments to see if FLOATER can improve over the existing position encoding approaches for a given Transformer model on various NLP tasks. Thus, all the metrics

reported in this paper are computed from a single (not ensemble) Transformer model over each evaluation NLP task. Albeit lower than top scores on the leaderboard, these metrics are able to reveal more clear signal to judge the effectiveness of the proposed position encoder.

All our codes to perform experiments in this paper are based on the Transformer implementations in the fairseq [14] package. Implementation details can be found in the Supplementary material. Our experimental codes will be made publicly available.

Table 3: Experimental results on GLUE benchmark

| Model     | Single Sentence |       | Similarity and Paraphrase |      | Natural Language Inference |      |      |      |
|-----------|-----------------|-------|---------------------------|------|----------------------------|------|------|------|
| Model     | CoLA            | SST-2 | MRPC                      | QQP  | STS-B                      | MNLI | QNLI | RTE  |
| Base mode | rl              |       |                           |      |                            |      |      |      |
| RoBERTa   | 63.6            | 94.8  | 88.2                      | 91.9 | 91.2                       | 87.6 | 92.8 | 78.7 |
| FLOATER   | 63.4            | 95.1  | 89.0                      | 91.7 | 91.5                       | 87.7 | 93.1 | 80.5 |
| Large mod | 'el             |       |                           |      |                            |      |      |      |
| RoBERTa   | 68.0            | 96.4  | 90.9                      | 92.2 | 92.4                       | 90.2 | 94.7 | 86.6 |
| FLOATER   | 69.0            | 96.7  | 91.4                      | 92.2 | 92.5                       | 90.4 | 94.8 | 87.0 |

Table 4: Experiment results on RACE benchmark. "Middle" means middle school level English exams, "High" means high school exams. Other details can be found in [15].

| Model       | Accuracy       | Middle   | High |
|-------------|----------------|----------|------|
| Single mode | l on test, lar | ge model |      |
| RoBERTa     | 82.8           | 86.5     | 81.3 |
| FLOATER     | 83.3           | 87.1     | 81.7 |

#### 4.1 Neural Machine Translation

Neural Machine Translation (NMT) is the first application that demonstrates the superiority of a sequence-to-sequence Transformer model over conventional recurrent sequence models. We include the following three additive position encoders:  $\Phi^{(n)}(x) = x + p^{(n)}$ .

- **Data-driven FLOATER**:  $p^{(n)}$  is generated by our proposed continuous dynamical models with data-driven parameters described in (8).
- **Pre-defined sinusoidal position encoder**:  $p^{(n)}$  is constructed by a pre-defined function described in (7), which is proposed by [1] and extended by [10].
- Length-fixed position embedding:  $p^{(n)}$  is included as learnable training parameters. This is first introduced by [1] and adopted in many variants of Transformer [2, 8].

To better demonstrate the parameter efficiency brought by FLOATER, for each above encoder, we also include two experimental settings: position encoder at all blocks or only at the input block (i.e.,  $p^{(n)} = 0, \forall n \geq 2$ ).

In Table 2, we present the BLEU scores on WMT14 Ee-De and En-Fr datasets with both *Transformer-base* and *Transformer-large* models described in [1]. Among all the data/model combinations, our proposed FLOATER at all blocks outperforms two other position encoders.

On the other hand, we also observe that adding position encoders at all blocks yields better performance than only at the input block. While there is an exception in the fixed-length position embedding approach. We suspect that this phenomenon is due to over-fitting cased by  $L_{\rm max}dN$  learnable parameters introduced by this approach. In contrast, our proposed FLOATER is parameter efficient (more discussions in Section 4.3), so the performance can be improved by injecting the position encoder at all the blocks of Transformer without much additional overhead.

#### 4.2 Language Understanding and Question Answering

Table 5: Experiment results on SQuAD benchmark. All results are obtained from RoBERTa-large model.

| Model                                       | SQuA | D 1.1 | SQuAD 2.0 |      |  |
|---------------------------------------------|------|-------|-----------|------|--|
| Wiodei                                      | EM   | F1    | EM        | F1   |  |
| Single models on dev, w/o data augmentation |      |       |           |      |  |
| RoBERTa                                     | 88.9 | 94.6  | 86.5      | 89.4 |  |
| FLOATER                                     | 88.9 | 94.6  | 86.6      | 89.5 |  |

Pretrained Transformer models such as BERT and RoBERTa have become the key to achieving the state-of-the-art performance for various language understanding and question answering tasks. In this section, we want to evaluate the effectiveness of the proposed FLOATER on these tasks. In particular, we focus on three language understanding benchmark sets, GLUE [16], RACE [15] and SQuAD [17]. As mentioned in Section 3.3, FLOATER is carefully designed to be compatible with the existing Transformer models. Thus, we can utilize pretrained Transformer models to warm-start a FLOATER model easily to be used to finetune on these NLP tasks. In this paper, we download the same pre-trained RoBERTa model from the official repository as our pretrained Transformer model for all NLP tasks discussed in this section. **GLUE Benchmark.** This benchmark is commonly used to evaluate the language understanding skills of NLP models. Experimental results in Table 3 show that our FLOATER model outperforms RoBERTa in most datasets, even though the only difference is the choice of positional encoding. **RACE benchmark** Similar to the GLUE benchmark, the RACE benchmark is another widely used test suit for language understanding. Compared with GLUE, each item in RACE contains a significantly longer context, which we believe requires more important to grasp the accurate position information. Like in GLUE benchmark, we finetune the model from the same pretrained RoBERTa checkpoint. We keep the hyperparameters, such as batch size and learning rate, to also be the same. Table 4 shows the experimental results. We again see consistent improvement of FLOATER across all subtasks.

**SQuAD benchmark** SQuAD benchmark [17, 18] is another challenging task to evaluate the question answering skills of NLP models. In this dataset, each item contains a lengthy paragraph containing facts and several questions related to the paragraph. The model needs to predict the range of characters that answer the questions. In SQuAD-v2, the problem becomes more challenging that the questions might be unanswerable by the context. We follow the same data processing script as BERT/RoBERTa for fair

![](_page_9_Figure_0.jpeg)

Figure 2: Comparing BLEU scores of different encoding methods.

Table 6: Performance comparison on WMT14 En-De data and Transformer-base architecture. Both BLUE scores and the number of trainable parameters inside each position encoder are included.

|                      | BLEU (†) | #Parameters (↓) |
|----------------------|----------|-----------------|
| FLOATER              | 28.57    | 526.3K          |
| 1-layer RNN + scalar | 27.99    | 263.2K          |
| 2-layer RNN + scalar | 28.16    | 526.3K          |
| 1-layer RNN + vector | 27.99    | 1,050.0K        |

comparison; more details about the training process are described in the Supplementary material. The experiment results are presented in Table 5. As we can see, the FLOATER model beats the baseline RoBERTa model consistently across most datasets. The improvement is significant, considering that both models are finetuned from the same pretrained checkpoint.

#### 4.3 More Discussions and Analysis

How inductive is FLOATER? FLOATER is designed to be inductive by a data-driven dynamical model (8). To see how inductive FLOATER is when comparing to existing approaches, we design the following experiment. We first notice that in WMT14 En-De dataset, 98.6% of the training sentences are shorter than 80 tokens. Based on that, we make a new dataset called *En-De short to long* (or S2L for brevity): this dataset takes all the short sentences (< 80 tokens) as the training split and all the long sentences ( $\ge 80$  tokens) as the testing split. We further divide the testing split to four bins according to the source length fallen in [80, 100), [100, 120), [120, 140),  $[140, +\infty)$ . BLEU scores are calculated in each bin, and the results are presented in Figure 2.

Our FLOATER model performs particularly well on long sentences, even though only short sentences are seen by the model during training. This empirical observation supports our conjecture that FLOATER model is inductive: the dynamics learned from shorter sequences can be appropriately generalized to longer sequences.

**Is RNN a good alternative to model the dynamics?** Recurrent neural network (RNN) is commonly used to perform sequential modeling. RNN and our continuous dynamical model (8) indeed share some

![](_page_10_Figure_0.jpeg)

Figure 3: Visualizing the four different position methods. All models are trained using the Transformer-base architecture and En-De dataset. For better visualization, dimension indices are permuted in Figure 3b-3d.

commonality. Computing the value at the i-th step relies on the results at the (i-1)-st step. Further, they all contain trainable parameters, allowing them to adapt to each particular task. Lastly, they can be extrapolated to any length as needed. To see if RNN works equally well, we model the sequence  $\{p_i\}_{i\in\{1,2,\dots\}}$  with RNN models:

$$\boldsymbol{p}_{i+1} = \text{RNN}(\boldsymbol{z}_i, \boldsymbol{p}_i), \tag{14}$$

where  $z_i \in \mathbb{R}^{d_{\text{in}}}$  is the input to the RNN model at index i. Recall in RNN language models,  $z_i$  is the word embedding or hidden feature of the i-th token. In our case, since we apply RNN to learn the encodings as opposed to hidden features, sensible inputs can be scalar value i or vectorized value Vectorize(i) by sinusoidal encoding. We tried both choices on WMT14 En-De data and found that vectorized value generally works better, though not as good as our FLOATER model. Detailed results can be found in Table 6.

What does each position encoding look like? To better understand how different position encodings affect the sequence modeling, in Figure 3, we visualize the position embedding matrix p obtained from four different position encoding approaches for the Transformer-base backbone on WMT14 En-De dataset. We can see that sinusoidal encoding (3a) is the most structural, while position embedding (3b) is quite chaotic. Our FLOATER model learns position representation completely from data, but still exhibits some regularities (3c). Finally, the RNN model (3d) fails to extract sufficient positional information, probably due to the vanishing gradient problem. Another finding is that by looking at (3b), we observe that the vectors are nearly constant among different large positions (near the bottom of Figure 3b, we see patterns of vertical lines with the same color). This phenomenon is due to long sentences in the dataset being scarce, and so the positional information carried by lower indices cannot be extrapolated to higher indices. On the contrary, the dynamical model proposed in this paper enjoys

the best of both worlds – it is adaptive to dataset distribution, and it is inductive to handle sequences with lengths longer than the training split.

#### 4.4 Remarks on Training and Testing Efficiency

It is not surprising that during the training time, our flow-based method adds a non-negligible time and memory overhead; this is because solving the Neural ODE precisely involves  $\sim \! 100$  times forward and backward propagations of the *flow model*. Even though we deliberately designed a small flow model (consisting of only two FFN and one nonlinearity layers), stacking them together still increases training time substantially. To make it possible to train big models, we use the following optimizations:

- Initialize with pretrained models that do not contain flow-based dynamics, as discussed in Section 3.3.
- From (8), we know that if  $h(\cdot)$  is close to zero, then the position information diminishes (derived in appendix). In this way, our model degenerates to the original Transformer. Inspired by this property, we can initialize the FLOATER with smaller weights. Combining with the previous trick, we obtain an informed initialization that incurs lower training loss at the beginning.
- We observed that weights in  $h(\cdot)$  are more stable and easy to train. Thus, we can separate the weights of  $h(\cdot)$  from the remaining parts of the Transformer model. Concretely, we can 1) cache the positional bias vectors for some iterations without re-computing, 2) update the weights of flow models less frequently than other parts of the Transformer, and 3) update the flow models with a larger learning rate to accelerate convergence.
- For the RoBERTa model, we adopt an even more straightforward strategy: we first download a pretrained RoBERTa model, plug in some flow-based encoding layers, and re-train the encoding layers on WikiText-103 dataset for one epoch. When finetuning on GLUE datasets, we can choose to freeze the encoding layers.

Combining those tricks, we successfully train our proposed models with only 20-30% overhead compared to traditional models, and virtually no overhead when finetuning RoBERTa model on GLUE benchmarks. Moreover, there is no overhead during the inference stage if we store the pre-calculated positional bias vectors in the checkpoints.

#### 5 Conclusions

In this paper, we have shown that learning position encoding with a dynamical model can be an advantageous approach to improve Transformer models. Our proposed position encoding approach is inductive, data-driven, and parameter efficient. We have also demonstrated the superiority of our proposed model over existing position encoding approaches on various natural language processing tasks such as neural machine translation, language understanding, and question answering tasks.

#### References

- [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008, 2017.
- [2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*, 2018.

- [3] Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, and Quoc V Le. Xlnet: Generalized autoregressive pretraining for language understanding. *arXiv* preprint *arXiv*:1906.08237, 2019.
- [4] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.
- [5] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. Albert: A lite bert for self-supervised learning of language representations. *arXiv* preprint *arXiv*:1909.11942, 2019.
- [6] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. *arXiv* preprint arXiv:1910.10683, 2019.
- [7] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? *arXiv* preprint *arXiv*:1912.10077, 2019.
- [8] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*, 2019.
- [9] Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Self-attention with relative position representations. In *Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)*, pages 464–468, 2018.
- [10] Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. *arXiv preprint arXiv:1807.03819*, 2018.
- [11] Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In *Advances in Neural Information Processing Systems*, pages 6571–6583, 2018.
- [12] Will Grathwohl, Ricky TQ Chen, Jesse Betterncourt, Ilya Sutskever, and David Duvenaud. Ffjord: Free-form continuous dynamics for scalable reversible generative models. *arXiv preprint arXiv:1810.01367*, 2018.
- [13] M. Tenenbaum and H. Pollard. *Ordinary Differential Equations: An Elementary Textbook for Students of Mathematics, Engineering, and the Sciences*. Dover Books on Mathematics. Dover Publications, 1985.
- [14] Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. fairseq: A fast, extensible toolkit for sequence modeling. *arXiv preprint arXiv:1904.01038*, 2019.
- [15] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. Race: Large-scale reading comprehension dataset from examinations. *arXiv preprint arXiv:1704.04683*, 2017.

- [16] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. *arXiv* preprint arXiv:1804.07461, 2018.
- [17] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. *arXiv preprint arXiv:1606.05250*, 2016.
- [18] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable questions for squad. *arXiv preprint arXiv:1806.03822*, 2018.
- [19] William H Press, Saul A Teukolsky, William T Vetterling, and Brian P Flannery. Numerical recipes in c++. *The art of scientific computing*, 2:1002, 1992.
- [20] Myle Ott, Sergey Edunov, David Grangier, and Michael Auli. Scaling neural machine translation. *arXiv preprint arXiv:1806.00187*, 2018.
- [21] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. *arXiv preprint arXiv:1609.07843*, 2016.
- [22] Yang Liu and Mirella Lapata. Hierarchical transformers for multi-document summarization. *arXiv preprint arXiv:1905.13164*, 2019.
- [23] Xingxing Zhang, Furu Wei, and Ming Zhou. HIBERT: document level pre-training of hierarchical bidirectional transformers for document summarization. *CoRR*, abs/1905.06566, 2019.
- [24] Tian Qi Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, *Advances in Neural Information Processing Systems 31*, pages 6571–6583. Curran Associates, Inc., 2018.

# A Training a Neural ODE model in Transformer

We discuss the details of training the dynamical model  $h(\tau, p_{\tau}; w_h)$ , recall in our FLOWER model, function h joins in the computational graph implicitly by generating a sequence of position encoding vectors  $\{p_1, p_2, \dots, p_N\}$ , conditioning on a freely initialized vector  $p_0$ . The generation steps are computed iteratively as follows (suppose we choose the interval between two consecutive tokens to be  $\Delta$ )

$$p_{1} = p_{0} + \int_{0}^{\Delta} \boldsymbol{h}(\tau, \boldsymbol{p}_{\tau}; \boldsymbol{w}_{h}) d\tau,$$

$$p_{2} = \boldsymbol{p}_{1} + \int_{\Delta}^{2\Delta} \boldsymbol{h}(\tau, \boldsymbol{p}_{\tau}; \boldsymbol{w}_{h}) d\tau,$$

$$\vdots$$

$$p_{N} = \boldsymbol{p}_{N-1} + \int_{(N-1)\Delta}^{N\Delta} \boldsymbol{h}(\tau, \boldsymbol{p}_{\tau}; \boldsymbol{w}_{h}) d\tau.$$
(15)

Finally, the loss L of this sequence is going to be a function of all position encoding results  $L = L(\mathbf{p}_0, \mathbf{p}_1, \dots, \mathbf{p}_N)$ , which is further a function of model parameters  $\mathbf{w}_h$ . The question is how to

calculate the gradient  $\frac{dL}{dw_h}$  through backpropagation. This question is fully solved in Neural ODE method [11] with an efficient adjoint ODE solver. To illustrate the principle, we draw a diagram showing the forward and backward propagation in Figure 4.

![](_page_14_Figure_1.jpeg)

Figure 4: Direction of forward and backward propagation. Here we consider a simplified version where only position encodings  $p_s$  and  $p_t$  are in the computational graph.

From [11], we know that the gradients  $\frac{\mathrm{d}}{\mathrm{d}\boldsymbol{w}_h}L\Big(\boldsymbol{p}_s+\int_s^t\boldsymbol{h}(\tau,\boldsymbol{p}_\tau;\boldsymbol{w}_h)\,\mathrm{d}\tau\Big)\triangleq\frac{\mathrm{d}L}{\mathrm{d}\boldsymbol{w}_h}$  can be computed by

$$\frac{\mathrm{d}L}{\mathrm{d}\boldsymbol{w}_h} = -\int_t^s \boldsymbol{a}(\tau)^{\mathsf{T}} \frac{\partial \boldsymbol{h}(\tau, \boldsymbol{p}_\tau; \boldsymbol{w}_h)}{\partial \boldsymbol{w}_h} \,\mathrm{d}\tau, \tag{16}$$

where  $a(\tau)$  defined in  $\tau \in [s,t]$  is called the "adjoint state" of ODE, which can be computed by solving another ODE

$$\frac{\mathrm{d}\boldsymbol{a}(\tau)}{\mathrm{d}\tau} = -\boldsymbol{a}(\tau)^{\mathsf{T}} \frac{\partial \boldsymbol{h}(\tau, \boldsymbol{p}_{\tau}; \boldsymbol{w}_{h})}{\partial \boldsymbol{p}_{\tau}}.$$
(17)

Note that the computation of (17) only involves Jacobian-vector product so it can be efficiently calculated by automatic differentiation.

# **B** Implementation details

### **B.1** Settings of ODE solver

To setup the ODE server, we need to first choose the numerical algorithms [19]. We have different setups for different datasets. For neural machine translation problems (WMT14 En-De and En-Fr), we use the more accurate Runge-Kutta scheme with discretization step  $\frac{\Delta}{5.0}$  to solve the adjoint equation (recall that we set the interval of two neighboring tokens to be  $\Delta=0.1$  globally). While for datasets with long sentences such as GLUE and RACE benchmarks, we found that solving the adjoint equation with high order scheme is too slow, in such case we adopt simple midpoint method with discretization step  $\frac{\Delta}{5.0}$ , and the gradients are calculated by automatic differentiation rather than adjoint method. The third party implementation of ODE solver can be found at https://github.com/rtqichen/torchdiffeq.

#### **B.2** Training NMT tasks

We run the same preprocessing script provided by fairseq [14], which is also used in ScalingNMT [20]. With the standard training script, we first successfully reproduce all the results in Transformer paper [1]. Based on that we execute the following protocol to get our results:

- 1. Train the original Transformer model for 30 epochs.
- 2. Random initialize FLOWER model of same shape configuration.
- 3. Copy tensors from the best performing checkpoint (validation set) to initialize FLOWER model. Initialize weights in the dynamical model with small values.
- 4. Half the peak learning rate (e.g. in Transformer-base + En-De, the peak learning rate is changed from  $7.0 \times 10^{-4}$  to  $3.5 \times 10^{-4}$ ).
- 5. With the warm-initialized FLOWER checkpoint, retrain on the same dataset for 10 epochs (En-De) or 1 epoch (En-Fr).
- 6. Averaging last 5 checkpoints and compute BLEU score on test split.

## **B.3** Training language understanding tasks

For GLUE/SQuAD/RACE benchmarks, our experiments are all conducted upon RoBERTa, in which both base and large configurations are available. Due to resource constraint (and to show the compatibility to existing models), we initialize our FLOWER model with pretrained RoBERTa, which is similar to NMT task. However, the weights  $\boldsymbol{w}_h$  in dynamic function  $\boldsymbol{h}(\tau, \boldsymbol{p}_\tau; \boldsymbol{w}_h)$  are not trained in large corpus, given that GLUE/SQuAD/RACE datasets are too small to train dynamics from scratch, we decided to pretrain  $\boldsymbol{h}$  alone in WikiText103 [21] data using masked language modeling loss. We have found that when we train  $\boldsymbol{w}_h$  alone, it only takes a few hours (2x Titan V100) and one epoch to convergence.

Once having the pretrained FLOWER model, we can run following downstream tasks and compare with RoBERTa under the same setting:

**GLUE benchmark** consists of eight datasets and each have different hyperparameter settings. For hyperparameters such as learning rate, batch size, training iterations, warm-up iterations, etc., we use the same values recommended by official repository of RoBERTa<sup>1</sup>.

**SQuAD benchmark.** For this benchmark we wrote our own finetuning code because currently there is no official code available. During the implementation process, we mainly refer to the third-party repositories<sup>2</sup>. We are not able to exactly match the official result reported in RoBERTa paper but quite close ( $\sim$ 0.1 difference in F1). For our FLOWER model, we use the same hyperparameters as RoBERTa.

**RACE benchmark.** This benchmark has the longest context and sequence length. We follow the official training script<sup>3</sup> and reproduce the result. Similar to other benchmarks, we then repeat the training process using exactly the same training hyperparameters to make a fair comparison. In this benchmark we freeze the weights  $w_h$  and only finetune the weights of RoBERTa.

 $<sup>^{1}</sup>Available \ at: \ \texttt{https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.} \\ glue.md$ 

<sup>&</sup>lt;sup>2</sup> Mainly https://github.com/ecchochan/roberta-squad and https://github.com/huggingface/transformers

<sup>3</sup>https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.race.md

## C Cases suitable for non-equidistant discritization

Although our model allows continuous values of s and t in (8), limiting the scope to text modeling tasks, positions are discrete values as  $\{0,1,2,\dots\}$ . Once the continuous version of position representation  $p_t$  is obtained, we simply take the discritized  $\{p_0,p_\Delta,p_{2\Delta},\dots,\}$  as the actual values to feed into Transformer model, where  $\Delta$  is a hyperparameter (e.g.  $\Delta=0.1$ ). By choosing positions t equidistantly, we are implicitly assuming the position signal evolves steadily as we go through each token in a sentence. More generally, the dynamics in (8) can deal with the case in which positions are not integers  $0,1,2,\dots$  etc., but arbitrary monotone increasing series  $t_0 < t_1 < t_2 < \dots$  which may not be equidistant. In appendix, we exemplify this general situation with several widely deployed tasks; we regard this as a interesting future direction. This makes our model particularly suitable for following scenarios yet traditional position representation may not be good at:

- Hierarchical Transformer model [22, 23]. The model is a direct extension of hierarchical RNN and is often used in long document processing. It works by first running a word-level Transformer model on each sentence to extract the sentence embedding, and then applying a sentence-level Transformer scanning through each sentence embedding sequentially. We argue that when processing at the sentence level, it could be better to set the increment of position index  $t_{i+1} t_i$  proportional to the length of the *i*-th sentence. This is because longer sentences tend to carry more information, so  $p_{i+1}$  is likely to move farther from  $p_i$ .
- Transformer for time-series events. As measurement time is continuous, time-series data is another scenario when a continuous position makes more sense than a discrete counterpart. More importantly, to predict the future values by modeling historical values observed at irregular time grids, it is better to consider the length of time horizon between two consecutive measures. A successful previous work is the Latent ODE [24], except that they use RNN as the backbone, and they model the hidden states rather than position representations with Neural ODE (because RNN itself provides positional bias).

In this paper, we are not going to explore the more general cases discussed above. Instead, we decided to leave them as interesting future work.