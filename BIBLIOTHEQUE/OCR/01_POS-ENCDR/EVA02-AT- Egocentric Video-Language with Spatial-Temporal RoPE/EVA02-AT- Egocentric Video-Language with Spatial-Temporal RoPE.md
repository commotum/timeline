# EVA02-AT: Egocentric Video-Language Understanding with Spatial-Temporal Rotary Positional Embeddings and Symmetric Optimization

Xiaoqi Wang, Student Member, IEEE, Yi Wang, Member, IEEE, Lap-Pui Chau, Fellow, IEEE

Abstract—Egocentric video-language understanding demands both high efficiency and accurate spatial-temporal modeling. Existing approaches face three key challenges: 1) Excessive pretraining cost arising from multi-stage pre-training pipelines, 2) Ineffective spatial-temporal encoding due to manually split 3D rotary positional embeddings that hinder feature interactions, and 3) Imprecise learning objectives in soft-label multi-instance retrieval, which neglect negative pair correlations. In this paper, we introduce EVA02-AT, a suite of EVA02-based video-language foundation models tailored to egocentric video understanding tasks. EVA02-AT first efficiently transfers an image-based CLIP model into a unified video encoder via a single-stage pretraining. Second, instead of applying rotary positional embeddings to isolated dimensions, we introduce spatial-temporal rotary positional embeddings along with joint attention, which can effectively encode both spatial and temporal information on the entire hidden dimension. This joint encoding of spatial-temporal features enables the model to learn cross-axis relationships, which are crucial for accurately modeling motion and interaction in videos. Third, focusing on multi-instance video-language retrieval tasks, we introduce the Symmetric Multi-Similarity (SMS) loss and a novel training framework that advances all soft labels for both positive and negative pairs, providing a more precise learning objective. Extensive experiments on Ego4D, EPIC-Kitchens-100, and Charades-Ego under zero-shot and finetuning settings demonstrate that EVA02-AT achieves state-ofthe-art performance across diverse egocentric video-language tasks with fewer parameters. Models with our SMS loss also show significant performance gains on multi-instance retrieval benchmarks. Our code and models are publicly available at https://github.com/xqwang14/EVA02-AT.

Index Terms—Vision-Language Model, Video-Text Retrieval, Cross-Modal Retrieval, Loss Function.

## I. Introduction

THE research community has witnessed rapid development of egocentric video understanding, driven by improvements in foundation models [1]–[4], pretraining strategies [5]–[7], loss functions [8], [9], and data augmentations [10]. Despite significant performance gains, the increasing scale of models, prolonged training pipelines, and ever-larger datasets have led to an exponential rise in training costs.

Current state-of-the-art pretraining solutions [6], [11] generally adopt a pretraining pipeline that involves three stages:

The research work was conducted in the JC STEM Lab of Machine Learning and Computer Vision funded by The Hong Kong Jockey Club Charities Trust. Xiaoqi Wang, Yi Wang, and Lap-Pui Chau are with the Department of Electrical and Electronic Engineering, The Hong Kong Polytechnic University, Hong Kong SAR (e-mail: xiaoqi.wang@connect.polyu.hk; yieie.wang@polyu.edu.hk; lap-pui.chau@polyu.edu.hk).

![](_page_0_Figure_10.jpeg)

Fig. 1. Our EVA02-AT-L model outperforms the previous state-of-the-art methods on three egocentric benchmarks: EgoMCQ, EK-100 MIR, and CharadesEgo in both zero-shot and fine-tune settings by adopting joint attention blocks with integrated spatial-temporal RoPE.

1) capturing the spatial-temporal structure through video reconstruction tasks [12], 2) image-text alignment, and 3) videotext alignment via contrastive learning. During the pretraining process, large image and video datasets such as LAION [13] and InternVid [14], which contain hundreds of millions of vision-text pairs, make the training process prohibitively expensive.

Besides the training cost, Rotary Positional Embeddings (RoPE) are now widely used in state-of-the-art vision models [15], [16]. CogvideoX [17] first proposes 3D-RoPE, which extends the RoPE to a spatial-temporal approach. Specifically, video tensors in latent space are treated as (x, y, t) coordinates, and CogVideoX applies 1D-RoPE independently at these three coordinates. In practice, the feature dimension is divided into slices of 3/8, 3/8, and 1/4 corresponding to the x, y, and t coordinates, respectively. Although the effectiveness of this approach has been demonstrated, there are two key issues with the manual division of hidden feature dimensions:

 Separation of spatial and temporal embeddings. The isolation in 3D-RoPE proposed in CogVideoX fails to model cross-axis relationships. Temporal embeddings, which represent motion between frames in video sequences, should ideally reflect changes in the spatial axis over time. In 3D-RoPE, since the dimensions are independent, the time changes  $xy + \Delta t$  lack geometric meaning in spatial dimension, preventing the fusion of relative positions across temporal and spatial axes.

• **Uneven dimension division**. Dividing the hidden dimensions of vision transformer architectures into three parts is not always feasible (e.g., 1024 for ViT-L). In the case of 3D-RoPE, the dimensions of the *t* coordinate are smaller than those of the *x* and *y* coordinates, which may be beneficial for spatially sensitive tasks, but reduce the ability to model long video sequences.

Moreover, we identified an issue with the current loss functions used in egocentric retrieval tasks. Specifically, EgoVLP [8] introduces the adaptive Multi-Instance Max Margin (MI-MM) loss, which employs a hard mining strategy. This strategy allows the dataloader to select samples where the soft label values exceed a threshold, rather than always selecting the most relevant ones. However, this could lead to negative pairs that are more strongly related to the textual descriptions than the positive pairs, steering the model in the wrong direction. However, simply removing the hard mining strategy would significantly reduce model performance.

To address these issues, we propose EVA-02 with spAtial-**Temporal attention** (EVA02-AT), a training-efficient solution for egocentric video understanding tasks. The EVA02-AT leverages the image-based pretraining CLIP model of EVA-02 [16], [18], simplifying the pretraining pipeline to a single stage by directly transferring the image-based CLIP model to a video-based one through video-text alignment. To achieve this, we extend the Rotary Positional Embedding (RoPE) to a spatial-temporal approach that is compatible with the original 2D-RoPE. Concretely, RoPE can be treated as a rotation matrix, which is multiplicative, meaning the inner product of two RoPEs equals the sum of their respective positional angles. Therefore, we first generate a 1D-RoPE for the temporal embeddings and a 2D-RoPE for the spatial embeddings, where the dimension of both embeddings corresponds to the whole feature dimension. Then, we conduct an inner product of the temporal and spatial RoPEs to obtain the final representations of our spatial-temporal RoPE. This approach combines the RoPE with learnable temporal and spatial positional embeddings, forming a final positional embedding. Our spatial-temporal RoPE enables each subspace to jointly encode spatiotemporal information, naturally supporting crossaxis relative positions.

To provide a more precise learning objective, we propose the **Symmetric Multi-Similarity** (**SMS**) loss to soft label multi-instance retrieval tasks. Inspired by Multi-Similarity loss [19] and RANP [9], our SMS loss collects not only the correlation values of positive pairs but also the negative pairs, optimizing the model from both sides. Therefore, the SMS loss redefines the relationship between positive and negative pairs and possibly converts certain negative pairs into positive ones under specific conditions, which enables the symmetric optimization of positive and negative pairs. Additionally, we introduce a relaxation factor to SMS loss to avoid the loss from falling into optimizing minor, unimportant samples.

We evaluate our framework on three widely-used egocentric

video datasets: Ego4D [20], EPIC-Kitchen-100(EK-100) [21], [22], and Charades-Ego [23]. The experiment results demonstrate both the effectiveness of our EVA02-AT models and the SMS loss. Our method is able to achieve state-of-the-art performance on these benchmarks in both zero-shot and fine-tuned settings, and the partial results are shown in Fig.1.

#### II. RELATED WORKS

# A. Video Foundation Models

Video foundation models can be grouped by their pretraining pipeline, which is often highly related to their architectural design. The foundation models based on video-text contrastive learning generally extend image—text models by adding temporal modules to capture temporal features. Early work like I3D [1] augments spatial 2D-CNNs with an LSTM [24] for temporal feature aggregation. More recent approaches like LaViLa [10] and EgoVLP [7], [8] utilize TSF [25] and FiT [26] as backbone networks, which add temporal-attention blocks into the ViT backbone, while AVION [4] treats each video as a flattened spatial-temporal sequence, processing end-to-end by ViT, greatly reducing overall training costs.

In contrast, models utilizing reconstruction-based pretraining pipelines can learn video representations via selfsupervised objectives such as masked video reconstruction [12], [27] and next-frame prediction [28]. This pretraining pipeline trains the model from the beginning, thus facilitating a more flexible architecture. Specifically, Internvideo [5], [6] adopts a 3D-CNN in the patchify process to form spatialtemporal cubes before feeding a ViT, such that the patches contain temporal information, while Flamingo [28] interleaves cross-attention layers to jointly encode video and text features.

RoPE [29] has driven recent advances in vision–language models [15], [30] by providing continuous, unbounded position encoding. However, transferring RoPE to videos remains challenging. As shown in Fig. 3, existing solutions like 3D-RoPE [17], M-RoPE [15], and VideoRoPE [31] provide different solutions for video RoPE. 3D-RoPE divides the feature dimension into uneven dimensions and applies three 1D-RoPEs on the entire dimension, so that the three 1D-RoPEs represent *x*-axis, *y*-axis, and *t*-axis individually. VideoRoPE further improves the 3D-RoPE by combining spatial axes, *x* and *y*, into a uniform 2D-RoPE. However, these methods manually split the embedding dimensions into spatial and temporal parts, such that they preclude a direct transfer of image-based encoders to video domains, and the uneven dimension division may cause a lack of ability to capture temporal information.

## B. Loss Functions for Contrastive Learning

Contrastive learning is a widely adopted paradigm for learning cross-modal representations by aligning paired samples while repelling mismatched ones [32]–[34]. In video–text pretraining, a common choice is the InfoNCE loss [35], which treats video–text pairs as positives and all other pairings within a minibatch as negatives. To better handle noisy alignments, MIL-NCE [36] relaxes the assumption of perfect correspondence by calculating the summation of multi-instance scores between all positive candidates, while EgoNCE [8] explicitly

parses verbs and nouns in captions to weight pairwise affinities according to semantic overlap within each batch.

Beyond batch-wide negatives, several works emphasize the importance of hard negatives or fine-grained similarity metrics. For example, RANP [9] mines semantic hard negative pairs and trains in a triplet manner [37], improving the discrimination of closely related but non-matching pairs. Circle loss [38] and Multi-Similarity (MS) loss [19] further generalize this idea by weighting each positive and negative pair according to its difficulty, enabling the model to focus more on challenging examples.

Recent advances in soft labeling and adaptive margin strategies have also been shown to improve performance. The adaptive MI-MM loss in EgoVLP [8] incorporates soft labels from EK-100 MIR annotations, achieving a substantial improvement. The relevancy margin loss [39] adds the correlation value on negatives, providing a more accurate learning objective. Inspired by this, we propose the SMS loss, which extends the soft label to both the positive and negative pairs.

#### III. Preliminary

**Rotary Positional Embedding.** RoPE [29] is known as an effective relative positional embedding approach that has shown extraordinary performance in many state-of-the-art video network architectures [6], [17], [40]. Originally, the vanilla 1D-RoPE was designed for word embeddings. In transformer-based models that use self-attention mechanisms, RoPE incorporates relative positional information into the attention mechanism. Specifically, the goal of RoPE is to embed the relative position information between the query  $\mathbf{x_m}$  at position  $m^{th}$  and the key  $\mathbf{x_n}$  at position  $n_{th}$  within the attention blocks. It should be a function  $f(\cdot)$  that satisfies the following condition:

$$\left\langle f_{q}\left(\boldsymbol{x}_{m},m\right),f_{k}\left(\boldsymbol{x}_{n},n\right)\right\rangle =g\left(\boldsymbol{x}_{m},\boldsymbol{x}_{n},m-n\right),$$
 (1)

where  $g(\cdot)$  denotes the real part of the inner product between  $f_q(x_m, m)$  and  $f_k(x_n, n)$ . In other words, the inner product between the projected query and key vectors at positions m and n is a function of both the input vectors and their relative position m-n. This property indicates that RoPE is a multiplicative positional embedding, meaning the inner product between two RoPE embeddings is equivalent to the subtraction of their corresponding absolute positional embeddings.

**Learning objective.** Given a triplet set  $\mathcal{D} = \{V, \mathcal{T}, C\}$ , the objective of the video text retrieval task is to learn a similarity calculation function  $S(\cdot)$  that satisfies  $S(\mathcal{V}, \mathcal{T}) = C$ . Here,  $\mathcal{V} = \{\mathbf{v}_i\}_{i=1}^{N_t}$  and  $\mathcal{T} = \{\mathbf{t}_j\}_{j=1}^{N_t}$  represent the video and narration sets with  $\mathcal{N}_v$  and  $\mathcal{N}_t$  samples, respectively. The label  $C = \{\mathbf{c}_{ij} \in \{0,1\} \mid i=1,2,\ldots,\mathcal{N}_v, j=1,2,\ldots,\mathcal{N}_t\}$  denotes whether a visual-text pair matches, that is,  $c_{ij} = 1$  signifies that  $(\mathbf{v}_i, \mathbf{t}_i)$  is a corresponding visual-text pair, and vice versa.

In deep metric learning, it is challenging to optimize every sample to its exact position. Alternatively, a general approach is to take advantage of a margin  $\gamma$  to separate the positive and negative pairs. Therefore, in the typical visual-to-text retrieval task, the most instinctive learning objective is to ensure that

![](_page_2_Figure_10.jpeg)

3

(a) Hard Mining Strategy of adaptive MI-MM Loss

![](_page_2_Figure_12.jpeg)

(b) Corner case where negative pair could be positive.

Fig. 2. Illustration of the label collection mechanism of adaptive MI-MM loss. Sub-figure (a) indicates that the soft labels collected by the previous dataloader during the training process differ from the actual soft labels, since they only capture correlation values for positive pairs (i.e., the diagonal values). Subfigure (b) illustrates a case where negative pairs can have higher correlation values than positive pairs.

the distance between positive and negative pairs is larger than the margin, which can be formulated as:

$$O_{v2t} := S(\mathcal{V}, \mathcal{T}_p) - S(\mathcal{V}, \mathcal{T}_n) \ge C \cdot \gamma, \tag{2}$$

where  $S(\cdot)$  denotes the similarity calculation function,  $\mathcal{T}_p$  and  $\mathcal{T}_n$  are the matching narrations and mismatching narrations to the corresponding video clips.

Given that C is the hard label set, where the values can only be 0 or 1, the target distance between the positive and negative pairs for every batch becomes:  $(\mathbf{c}_p - \mathbf{c}_n)\gamma = \gamma$ . Consider that cosine similarity is used for similarity calculations, where the matrix product of L2-normalized features will represent their similarity, the learning objective becomes:

$$O_{v2t} := S(\mathbf{v}_i, \mathbf{t}_j) - S(\mathbf{v}_i, \mathbf{t}_k) \ge \gamma$$
  
:=  $\gamma - \mathbf{v}_i^T \mathbf{t}_j + \mathbf{v}_i^T \mathbf{t}_k \le 0.$  (3)

Here, j and k are the samples in positive and negative sets, respectively. Since our task is bidirectional, that is, we need to conduct both the video-to-text and text-to-video retrieval, thus the loss function can be formulated as:

$$\mathcal{L} = \sum_{(i,j,k)\in\mathcal{N}} \left[ \gamma - \mathbf{v}_i^T \mathbf{t}_j + \mathbf{v}_i^T \mathbf{t}_k \right]_+ + \left[ \gamma - \mathbf{t}_i^T \mathbf{v}_j + \mathbf{t}_i^T \mathbf{v}_k \right]_+. \quad (4)$$

This is a commonly used loss function in the video-text retrieval task, called hinge loss or Multi-Instance Max-Margin (MI-MM) loss [41], and  $[\cdot]_+$  denotes the ReLU function here.

Meanwhile, consider a special scenario when soft labels are introduced. In the Epic-Kitchen-100 multi-instance retrieval task, a semantic-based soft label generation method is proposed [42]. Specifically, since narrations are used to describe actions, which can be simplified as the combination

of verbs and their corresponding objects(nouns). Consequently, the generation method can be formulated as follows.

$$S_{PoS}(y_i, y_j) = \sum_{p \in P} \alpha^p \frac{\left| w_i^p \cap w_j^p \right|}{\left| w_i^p \cup w_j^p \right|}, \tag{5}$$

where p denotes parts of speech, e.g., verb and noun;  $\alpha^p$  denotes the weights for every part of speech, commonly 0.5 for both verb and noun. Therefore, the equation means that the relevancy value, or the soft label values  $\mathbf{c}_{ij} \in [0,1]$  between the i-th and j-th narrations equals the IOU of the words in the selected part of the speech. In this scenario, the relevance matrix becomes  $C = {\mathbf{c}_{ij} \in [0,1]|i=1,2,...,N_v,j=1,2,...,N_t}$ . To take advantage of this prior information, the adaptive MI-MM loss [8], [43] is proposed, formulated as:

$$\mathcal{L} = \sum_{(i,j,k)\in\mathcal{N}} [\mathbf{c}_{ij}\gamma - \mathbf{v}_i^T \mathbf{t}_j + \mathbf{v}_i^T \mathbf{t}_k]_+ + [\mathbf{c}_{ij}\gamma - \mathbf{t}_i^T \mathbf{v}_j + \mathbf{t}_i^T \mathbf{v}_k]_+.$$
(6)

The learning objective of adaptive MI-MM Loss is similar to MI-MM Loss, but introduces the relevancy matrix C to the learning objective. However, the adaptive MI-MM loss only considers the correlations of positive pairs, treating the correlation between video clips and their corresponding negative pairs as 0. As shown in Fig. 2(a), the correlation between negative pairs,  $c_{ik}$ , is not always 0. This makes the learning objective less precise for soft-label-based multi-instance retrieval tasks. Moreover, EgoVLP [8] employs a hard mining strategy that defines the positive set as  $i^+ = \{j | \mathbf{c}_{ij} \geq 0.1\}$ , that is, the partially matched video-text pairs could be treated as positive samples. As illustrated in Fig. 2, since adaptive MI-MM loss ignores the correlation values of negative pairs, this can be problematic when  $\mathbf{c}_{ij} < \mathbf{c}_{ik}$ , leading the learning objective in the opposite direction to the correct one.

# IV. THE PROPOSED METHOD

## A. EVA-02 AT Transformer

In this subsection, we introduce the design choices in the EVA-02 transformer, including the patchify process, spatial-temporal RoPE embedding, and the theory of joint attention blocks.

**Patchify.** Inspired by the framework of AVION [4], we integrate a spatial-temporal attention block into a vanilla EVA-02 [18], [44]. For patch embedding, an input video sequence  $\mathbf{v} \in \mathbb{R}^{C \times T \times H \times W}$ , where C, T, H, W represents channels, number of frames, height, and length, is processed in the spatial domain only. This approach ensures compatibility with the original image encoder, yielding a patchified feature of dimension  $\mathbb{R}^{B \times (T \times P^2) \times D}$ , where  $D = \frac{CHW}{p^2}$ .

We introduce two distinct learnable positional embeddings: a temporal positional embedding  $P_t \in \mathbb{R}^{T \times D}$  and a spatial positional embedding  $P_{xy} \in \mathbb{R}^{p^2 \times D}$ . Each temporal positional embedding is replicated  $p^2$  times across the patches of a frame, while each spatial positional embedding is replicated T times to cover all frames. Therefore, the initial representation  $z^{(0)}$  after patch embedding is formulated as:

![](_page_3_Picture_11.jpeg)

Fig. 3. Illustration of different video RoPEs. Our method conducts both spatial and temporal RoPE on the entire feature dimension, forming an integrated spatial-temporal RoPE by leveraging its multiplicative property.

$$z^{(0)} = P_{xy}^{T} + P_{t}^{S} + x^{(0)},$$

$$s.t.P_{xy}^{T} = \{P_{xy}^{i} \in \mathbb{R}^{p^{2} \times T \times D} \mid i = 1, 2, ..., t\},$$

$$P_{t}^{S} = \{P_{xy}^{j} \in \mathbb{R}^{p^{2} \times T \times D} \mid j = 1, 2, ..., xy\}.$$

$$(7)$$

Here,  $P_t^S$  and  $P_{xy}^T$  denote the final spatial and temporal positional embeddings before the transformer blocks.  $x^{(0)}$  denotes the initial feature of the video clip after passing through the first convolutional layer in the patch embedding block. In this case, we employ a 3D convolution, also known as tube convolution [12], with a convolution kernel of  $1 \times p \times p$ . This convolutional operation effectively captures both the spatial and temporal information of the video during the patch embedding phase. The inclusion of temporal dimensions allows the image encoder to act as a video encoder.

**Joint Spatial-Temporal Attention.** The learnable spatial-temporal positional embedding in EVA02-AT enables the joint spatial-temporal attention. In EVA02-AT, joint attention blocks that process both spatial and temporal information simultaneously are adopted, rather than the divided spatial and temporal attention used in typical video encoders such as Timesformer and Frozen-in-Time [25], [45].

To cooperate with the joint attention, we need to apply an integrated spatial-temporal RoPE to capture the joint features. Fig. 3 illustrates how our spatial-temporal RoPE works. Specifically, since the RoPE is a multiplicative positional embedding where the inner product of two RoPEs is equivalent to the addition of rotation angles, to describe a time change in the spatial domain,  $xy + \Delta t$ , it obeys the following equation:

$$R_{(xy+\Delta t)} = R_{xy} \cdot R_{\Delta t}. \tag{8}$$

Therefore, we initialize a 2D-RoPE  $R_{xy} \in \mathbb{R}^{p^2 \times D}$  on the spatial domain, where the dimension is evenly divided for height and width, and a 1D temporal RoPE  $R_t \in \mathbb{R}^{T \times D}$  on the entire dimension. By calculating the inner product of spatial and temporal RoPE, we obtain an addition of spatial and temporal rotation angles. Similar to the learnable positional

![](_page_4_Figure_2.jpeg)

Fig. 4. Training framework of EVA02-AT. Given an input video clip  $v_i$ , a hard mining strategy is applied to find a partially matching narration  $t_j$  from the pre-build relevancy matrix. Then the dataloader would randomly select a narration as the positive pair to the input video clip from the candidates where the correlation value between  $v_i$  and narration  $\{t_j | j = 1, 2, ..., \mathcal{N}_{\sqcup}\}$   $\mathbf{c}_{ij}$  is greater than a predefined threshold  $\epsilon$ . Meanwhile, the dataloader will record the serial number of the video clip in pre-build relevancy matrix, thus to rebuild a  $B \times B$  correlation matrix during the loss calculation.

embeddings, the spatial RoPE is replicated T times for T frames in the batch, and the temporal RoPE is replicated  $p^2$  times for patches in every frame in order to align our 3D-RoPE with the positional embedding. This operation can be expressed as:

$$R_{(xy+t)} = R_{xy}^{T} \cdot R_{t}^{S},$$

$$s.t.R_{xy}^{T} = \{R_{xy}^{i} \in \mathbb{R}^{p^{2} \times T \times D} \mid i = 1, 2, ..., t\},$$

$$R_{t}^{S} = \{R_{xy}^{i} \in \mathbb{R}^{p^{2} \times T \times D} \mid j = 1, 2, ..., xy\}.$$
(9)

In this way, we thus apply the spatial RoPE and temporal RoPE on the entire dimension instead of manually dividing the dimension into uneven slides. Since we use the standard QK-RoPE, the output of our joint spatial-temporal attention at k-th layer can be expressed as:

$$z^{k} = SPACE - TIME(z^{k-1})$$

$$= Attn\left(R_{(xy+t)}W_{q}z^{k-1}, R_{(xy+t)}W_{k}z^{k-1}, W_{v}z^{k-1}\right).$$
(10)

The  $z^{k-1}$  denotes the output of the k-1th layer. In this way, the attention score between query and key becomes a global attention among all the patches in the video clip instead of the spatial attention on a single frame. Meanwhile, the model can still be trained on the basis of an image encoder, which simplifies the pretraining process.

### B. Symmetric Multi-Similarity Loss

As aforementioned, the adaptive MI-MM [8] is not an accurate loss function since the correlation values of negative pairs are not considered. Therefore, to provide a more accurate learning objective, we introduce a novel training framework,

which is shown in Fig. 4. Building on the hard-mining strategy of EgoVLP [8], which treats partially matched pairs as positives, the training framework can learn verbs and nouns in natural languages independently. Beyond this, we further refine it by incorporating correlations from both positive and negative samples.

Specifically, we compute the relevance matrix via Eqn. 5. During training, the dataloader collects not only matched video-text pairs but also sequences of video  $v_i$  and partially matched text  $t_j$ . For each batch, we reconstruct a  $B \times B$  relevance matrix from these sequences, where B represents the batch size. Thus, the relevancy value between arbitrary video and text within the batch can be found in this batch-wise relevancy matrix, so that negative pair entries reflect their true correlation scores rather than defaulting to zero. This enriched matrix serves as the foundation for our SMS loss.

Given the correlation values for both positive and negative pairs, we aim to create a loss function that can optimize the model from both directions. The Multi-Similarity Loss [19] provides us a good example and demonstrates its effectiveness on metric learning tasks, which is formulated as:

$$\mathcal{L}_{MS} = \frac{1}{N} \sum_{i=1}^{N} \left\{ \frac{1}{\alpha} \log \left[ 1 + \sum_{j \in P_i} e^{-\alpha (S_{ij} - \gamma)} \right] + \frac{1}{\beta} \log \left[ 1 + \sum_{k \in N_i} e^{\beta (S_{ik} - \gamma)} \right] \right\},$$
(11)

where  $\mathcal{P}_i$  and  $\mathcal{N}_i$  refer to the positive and negative sets corresponding to the *i*-th video clip,  $\alpha$  and  $\beta$  are the scale factors for positive and negative pairs, respectively. To simplify this loss function, we consider a special case when  $\alpha, \beta \to \infty$ :

$$\mathcal{L}_{MS}^{'} = \sum_{(i,j,k)\in\mathcal{N}} \left[ \gamma - \mathbf{S}_{ij} \right]_{+} + \left[ \mathbf{S}_{ik} - \gamma \right]_{+}. \tag{12}$$

This reveals that the learning objective for Multi-Similarity Loss is to push positive pairs closer to the margin while pulling negative pairs away from it. This inspires us to define a symmetric loss function for positive and negative pairs. However, as previously illustrated, it is challenging to determine if  $\mathbf{t}_j$  and  $\mathbf{t}_k$  are relatively more positive to the video clip  $\mathbf{v}_i$ . Therefore, directly applying Multi-Similarity Loss to this multi-instance retrieval task is still far from satisfactory.

Therefore, we need to define the positive and negative pairs in our training pipeline. Given two narrations j and k corresponding to i-th video clip, we formulate the correlation  $\mathcal{R}$  between  $S_{ij}$  and  $S_{ik}$  as follows:

$$\mathcal{R} = \sum_{(i,j,k)\in\mathcal{N}} \mathbf{c}_{ij} - \mathbf{c}_{ik}.$$
 (13)

In this way, when the correlation factors  $\mathcal{R} > 0$ ,  $\mathbf{v}_i$ , and  $\mathbf{t}_j$  are the relatively more positive pair compared to  $\mathbf{v}_i$  and  $\mathbf{t}_k$ , and vice versa. Following the concept of multi-similarity loss, we extend the adaptive MI-MM loss to a bi-directional and symmetric form:

$$\mathcal{L} = \sum_{(i,j,k)\in\mathcal{N}} \left\{ \begin{bmatrix} \mathcal{R}\gamma - S_{ij} + S_{ik} \end{bmatrix}_{+} & \mathcal{R} > 0 \\ [-\mathcal{R}\gamma + S_{ij} - S_{ik}]_{+} & \mathcal{R} < 0 \end{bmatrix}$$
(14)

However, a special case happens when  $\mathcal{R}=0$ , where the distance between  $S_{ij}$  and  $S_{ik}$  should be optimized to 0. However, in practice, two factors are preventing us from doing so. First, two descriptions with different verbs and nouns could have the same corresponding values. e.g., the current action label is "eat banana", while the partially matched positive pair is "eat apple", and the negative pair is "grab banana". In this case,  $\mathbf{c}_{ij}$  and  $\mathbf{c}_{ik}$  are the same, but the distance between them should not be optimized to 0.

Meanwhile, we find that the loss at  $\mathcal{R}=0$  tends to be the dominant loss since the value of  $\mathcal{R}$  is very small. To mitigate this, we introduce a relaxation factor,  $\tau$ , such that when the Euclidean distance between  $S_{ij}$  and  $S_{ik}$  is smaller than  $\tau$ , we cease optimizing this part. This adjustment allows us to maintain the major learning objective, i.e.,  $O:=S_p-S_n>\mathcal{R}\gamma$ . Thus, we obtain a symmetric loss regarding the distance between positive and negative pairs:

$$\mathcal{L}_{SMS} = \sum_{(i,j,k)\in\mathcal{N}} \begin{cases} \left[ \mathcal{R}\gamma - S_{ij} + S_{ik} \right]_{+} & \mathcal{R} > 0 \\ \left[ -\mathcal{R}\gamma + S_{ij} - S_{ik} \right]_{+} & \mathcal{R} < 0 \\ \left[ \|S_{ij} - S_{ik}\|_{1} - \tau \right]_{+} & \mathcal{R} = 0 \end{cases}$$
(15)

Here,  $S_*$  denotes both the similarity of video-to-text and text-to-video. Additionally, we add a threshold  $\lambda$  to constrain the edge conditions, of which the value equals the threshold for selecting positive pairs. Thus, the final loss function becomes:

$$\mathcal{L}_{SMS} = \sum_{(i,j,k)\in\mathcal{N}} \begin{cases} \left[ \mathcal{R}\gamma - S_{ij} + S_{ik} \right]_{+} & \mathcal{R} \geqslant \lambda \\ \left[ -\mathcal{R}\gamma + S_{ij} - S_{ik} \right]_{+} & \mathcal{R} \leqslant -\lambda \\ \left[ \left| S_{ij} - S_{ik} \right| - \tau \right]_{+} & |\mathcal{R}| < \lambda \end{cases}$$
(16)

Theoretically, the relaxation factor  $\tau$  should be less than the minimum value of C for C>0. This ensures that the optimization process remains effective and balanced across different correlation scenarios. However, in practice, we sometimes need a larger  $\tau$  to prevent the model from focusing on similar pairs. Therefore, we obtain the final representation of SMS loss, which optimizes the model symmetrically according to the difference in correlation values.

#### V. Experiments

# A. Datasets and Implementation Details

Datasets. We conduct the experiments on three egocentric datasets: Ego4D, Epic-Kitchens-100 (EK-100), and Charades-Ego. We first pretrain our models on the EgoClip and Ego-Clip+ versions of the Ego4D dataset, where the EgoClip is proposed by EgoVLP [8], which contains 3.8 million videotext pairs for training, and the average length for each video clip is about 1 second. The EgoClip+ is proposed by LaViLa [10], which has a 35-million corpus that is augmented by GPT-2 XL [46]. After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark. Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge, where the performance will be treated as zero-shot results. After that, we fine-tune the pretrained model on the training set of these two benchmarks, respectively, and evaluate their fine-tuned results.

**Implementation Details.** We build our EVA02-AT models based on the AVION framework [4], a vanilla ViT-CLIP backbone, and our EVA02-AT-CLIP variants retain the same architecture as EVA02-CLIP except for the modified positional embeddings described in Section 4. We train on  $4 \times \text{NVIDIA}$  RTX 6000 Ada GPUs. During both pretraining and fine-tuning, frames are sampled uniformly from each clip at a resolution of  $3 \times 224 \times 224$ , and the dimension of the feature space is set to 256. For our SMS loss, unless specified, we set the SMS-loss margin  $\gamma$  to 0.6, and the relaxation factor  $\tau$  to 0.1.

**Ego4D pretraining.** For Ego4D pretraining, we optimize a bi-directional InfoNCE loss [35] with the temperature 0.05. We evenly sample 4 frames for each video clip. The batch size for our base-size model is set to 256 per GPU, resulting in a total batch size of 1024, while the batch size is set to 128 for the large model, resulting in a total batch size of 512. We train for five epochs using AdamW [52] with a fixed learning rate of  $3 \times 10^{-5}$ . The pretraining process takes approximately 40 hours for our base model.

**EK-100 MIR.** When fine-tuning on the EK-100 dataset, we employ our SMS loss to fine-tune the model pretrained on the Ego4D dataset for 100 epochs. We warm up the learning rate from  $10^{-6}$  to a peak of  $2 \times 10^{-5}$  over the first epoch. During fine-tuning, 16 frames are sampled for each video clip, and the batch size is set to 64 for each GPU. Fine-tuning the base model under these settings requires about 20 hours.

**Charades-Ego Fine-tuning.** The Charades-Ego dataset only contains hard labels, but there could be multiple different hard labels for each video clip. In order to be compatible with the Charades-Ego dataset, we refine our SMS loss as follows:

TABLE I

The main results on EK-100 multi-instance retrieval task. 'PT Dataset' identifies the pretraining dataset, 'Vis Enc.' indicates the visual encoder the models are using. The symbol '\*' indicates reproduced results, "†" denotes that three input modalities are used: RGB, Flow, and Audio. The base-size models are in white rows and large-size models are in gray.

| Methods         | PT Dataset     | Vis Enc.   | # Frames | mAP (%)     |                   |             | nDCG (%)          |                   |             |
|-----------------|----------------|------------|----------|-------------|-------------------|-------------|-------------------|-------------------|-------------|
|                 |                |            |          | V→T         | $T{\rightarrow}V$ | Avg.        | $V \rightarrow T$ | $T{\rightarrow}V$ | Avg.        |
| MI-MM           | HowTo100M [47] | S3D [48]   | 32       | 34.8        | 23.6              | 29.2        | 47.1              | 42.4              | 44.7        |
| MME [41]        | -              | TBN [45]   | 25†      | 43.0        | 34.0              | 38.5        | 50.1              | 46.9              | 48.5        |
| JPoSE [41]      | -              | TBN        | 25†      | 49.9        | 38.1              | 44.0        | 55.5              | 51.6              | 53.5        |
| AVION* [4]      | WIT [49]       | ViT-B      | 16       | 46.8        | 39.9              | 43.4        | 60.0              | 58.0              | 59.0        |
| AVION + SMS [4] | WIT            | ViT-B      | 16       | 53.8        | <u>41.4</u>       | <u>47.6</u> | 63.2              | <u>59.2</u>       | 61.2        |
| EVA02-AT + SMS  | Merged2B       | EVA02-AT-B | 16       | 57.6        | 45.0              | 51.3        | 67.1              | 63.0              | 65.0        |
| AVION*          | WIT            | ViT-L      | 16       | 51.0        | 44.9              | 47.9        | 64.7              | 62.5              | 63.6        |
| AVION + SMS     | WIT            | ViT-L      | 16       | 60.0        | <u>47.8</u>       | <u>53.9</u> | <u>68.7</u>       | 64.5              | 66.6        |
| EVA02-AT + SMS  | Merged2B       | EVA02-AT-L | 16       | 63.8        | 52.4              | 58.1        | 71.9              | 67.9              | 69.9        |
| EgoVLP [8]      | EgoClip        | TSF-B [25] | 16       | 49.9        | 40.1              | 45.0        | 60.9              | 57.9              | 59.4        |
| HierVL-SA [50]  | EgoClip        | FiT-B [26] | 16       | -           | -                 | 46.7        | -                 | -                 | 61.1        |
| EgoVLPv2 [7]    | EgoClip        | TSF-B      | 16       | -           | -                 | 47.3        | -                 | -                 | 61.9        |
| AVION*          | EgoClip        | ViT-B      | 16       | 53.3        | 46.6              | 50.0        | 66.3              | 64.0              | 65.1        |
| AVION + SMS     | EgoClip        | ViT-B      | 16       | 60.9        | <u>48.8</u>       | <u>54.9</u> | <u>69.2</u>       | <u>65.5</u>       | <u>67.3</u> |
| EVA02-AT + SMS  | EgoClip        | ViT-B      | 16       | 63.2        | 51.3              | 57.3        | 71.0              | 67.0              | 69.0        |
| LaViLa [10]     | EgoClip+       | TSF-B      | 16       | 55.2        | 45.7              | 50.5        | 66.5              | 63.4              | 65.0        |
| AVION           | EgoClip+       | ViT-B      | 16       | 55.9        | 47.8              | 51.8        | 68.2              | 65.4              | 66.8        |
| AVION + SMS     | EgoClip+       | ViT-B      | 16       | 62.9        | <u>51.1</u>       | <u>57.0</u> | <u>71.2</u>       | <u>67.3</u>       | 69.2        |
| EVA02-AT + SMS  | EgoClip+       | ViT-B      | 16       | 64.6        | 53.4              | <b>59.0</b> | <b>72.6</b>       | 69.0              | 70.8        |
| LaViLa          | EgoClip+       | TSF-L      | 16       | 54.7        | 47.1              | 50.9        | 68.1              | 64.9              | 66.5        |
| AVION           | EgoClip+       | ViT-L      | 16       | 57.9        | 51.1              | 54.5        | 70.4              | 67.6              | 69.0        |
| AVION + SMS     | EgoClip+       | ViT-L      | 16       | <u>67.3</u> | <u>56.9</u>       | <u>62.1</u> | <u>74.7</u>       | <u>71.2</u>       | <u>73.0</u> |
| EVA02-AT + SMS  | EgoClip+       | EVA02-AT-L | 16       | 68.7        | 58.3              | 63.5        | 76.1              | 72.3              | 74.2        |

TABLE II

ZERO-SHOT AND FINE-TUNED RESULTS FOR VIDEO-TO-TEXT RETRIEVAL TASK ON CHARADES-EGO DATASET AND EGOMCQ BENCHMARK. THE BASE-SIZE MODELS ARE IN WHITE ROWS AND LARGE-SIZE MODELS ARE IN GRAY.

| Method         | Chara       | desEgo      | EgoMCQ Acc. |             |  |
|----------------|-------------|-------------|-------------|-------------|--|
| Method         | mAP (ZS)    | mAP (FT)    | Inter-vid.  | Intra-vid.  |  |
| (EgoClip)      |             |             |             |             |  |
| EgoVLP         | 25.0        | 32.1        | 90.6        | 57.2        |  |
| HierVL-SA      | 26.0        | 33.8        | 90.5        | 52.4        |  |
| EgoVLPv2       | 26.2        | 34.1        | 91.0        | 60.9        |  |
| SViTT-Ego [51] | -           | -           | 92.9        | <u>65.9</u> |  |
| EVA02-AT       | -           | -           | 94.8        | 62.0        |  |
| EVA02-AT       | -           | -           | 96.2        | 66.0        |  |
| (EgoClip+)     |             |             |             |             |  |
| LaViLa         | 26.8        | 33.7        | 93.8        | 59.9        |  |
| AVION*         | <u>27.4</u> | 34.8        | 94.5        | 61.4        |  |
| EVA02-AT       | 27.8        | <u>36.1</u> | 95.0        | 63.2        |  |
| EVA02-AT+SMS   | -           | 38.0        | -           | -           |  |
| LaViLa         | 28.9        | 36.1        | 94.5        | 63.1        |  |
| AVION*         | <u>29.9</u> | 39.7        | <u>95.4</u> | 64.5        |  |
| EVA02-AT       | 30.9        | <u>42.2</u> | 95.9        | 66.5        |  |
| EVA02-AT+SMS   | -           | 42.5        | -           | -           |  |

$$\mathcal{L}_{SMS} = \sum_{(i,j,k) \in \mathcal{N}} \left\{ \begin{array}{l} \left[ \mathcal{R}\gamma - S_{ij} + S_{ik} \right]_{+} & \mathcal{R} = 1 \\ \left[ |S_{ij} - S_{ik}| - \tau \right]_{+} & \mathcal{R} = 0 \end{array} \right.$$
 (17)

We fine-tune the model for 10 epochs using the Lamb optimizer, warming up from  $10^{-6}$  to  $3\times10^{-5}$  in the first epoch. And we sample 16 frames per video clip per GPU. The margin  $\gamma$  of SMS loss is set to 0.3.

### B. Compare with State-of-the-Arts

**EK100 MIR.** The choice of pretraining data critically affects performance on the EK-100 multi-instance retrieval task. Currently, the state-of-the-art methods are using different pretraining settings, leading to a variety of results. To ensure fair comparisons, we group existing methods by their public pretraining datasets: (a) Image or non-egocentric video dataset; (b) EgoClip [8], [20]; and (c) EgoClip with LLM-augmented corpus (EgoClip+) proposed by LaViLa [10].

Table I shows the comparison between the state-of-theart results and our methods on the EK-100 multi-instance

TABLE III

Zero-shot performance of various network architectures on the EK-100 multi-instance retrieval task. The symbol '\*' indicates reproduced results. The 'Params (M)' column lists the number of parameters for video encoders, text encoders, and additional blocks (if any), in that order. The base-size models are in white rows and large-size models are in gray.

| Methods   | PT Dataset | Backbone                   | Params (M)    | Zero-Shot mAP (%) |                   |             | Zero-Shot nDCG (%) |                   |             |
|-----------|------------|----------------------------|---------------|-------------------|-------------------|-------------|--------------------|-------------------|-------------|
| wiethous  |            | Vis-Text Enc.              | Vis-Text Enc. | V→T               | $T{\rightarrow}V$ | Avg.        | $V \rightarrow T$  | $T{\rightarrow}V$ | Avg.        |
| Random    | -          | -                          | -             | 5.7               | 5.6               | 5.7         | 10.8               | 10.9              | 10.9        |
| EgoVLP    | EgoClip    | TSF-B + DistillBERT-B [53] | 114 + 66      | 19.4              | 13.9              | 16.6        | 24.1               | 22.0              | 23.1        |
| HierVL-SA | EgoClip    | FiT-B + DistillBERT-B      | 114 + 66 + 7  | -                 | -                 | 18.9        | -                  | -                 | 24.7        |
| EgoVLPv2  | EgoClip    | TSF-B + RoBERT-B [54]      | 129 + 138     | _                 | -                 | 26.7        | -                  | -                 | 29.1        |
| AVION     | EgoClip    | CLIP-ViT-B                 | 86 + 63       | 31.7              | <u>25.1</u>       | 28.4        | 31.0               | 28.1              | <u>29.5</u> |
| EVA02-AT  | EgoClip    | CLIP-EVA02-AT-B            | 86 + 63       | 33.5              | 26.8              | 30.2        | 32.8               | 29.4              | 31.1        |
| AVION*    | EgoClip    | CLIP-ViT-L                 | 304 + 124     | 33.6              | <u>27.1</u>       | <u>30.4</u> | 31.8               | <u>29.0</u>       | <u>30.4</u> |
| EVA02-AT  | EgoClip    | CLIP-EVA02-AT-L            | 304 + 124     | 42.1              | 35.0              | 38.5        | 37.2               | 33.9              | 35.5        |
| EgoVLP    | EgoClip+   | TSF-B + DistillBERT-B      | 114 + 66      | 26.0              | 20.6              | 23.3        | 28.8               | 27.0              | 27.9        |
| LaViLa    | EgoClip+   | TSF-B + Text-CLIP-B        | 114 + 63      | 35.1              | 26.6              | 30.9        | 33.7               | 30.4              | 32.0        |
| AVION     | EgoClip+   | CLIP-ViT-B                 | 86 + 63       | <u>37.1</u>       | <u>28.7</u>       | 32.9        | <u>34.4</u>        | 31.0              | <u>32.7</u> |
| EVA02-AT  | EgoClip+   | CLIP-EVA02-AT-B            | 86 + 63       | 38.3              | 30.3              | 34.3        | 36.0               | 32.2              | 34.1        |
| LaViLa    | EgoClip+   | TSF-L + DistillBERT-B      | 404 + 66      | 40.0              | 32.2              | 36.1        | 36.1               | 33.2              | 34.6        |
| AVION     | EgoClip+   | CLIP-ViT-L                 | 304 + 124     | <u>41.7</u>       | <u>33.5</u>       | <u>37.6</u> | 36.8               | <u>33.9</u>       | <u>35.3</u> |
| EVA02-AT  | EgoClip+   | CLIP-EVA02-AT-L            | 304 + 124     | 43.2              | 34.5              | 38.9        | 38.4               | 33.9              | 36.2        |

retrieval task, with base-size models in white rows and large-size models in gray. Across all three dataset categories, our models lead both base and large configurations. For base-size models, we improve average mAP by 7.2% (59.0 vs. 51.8) and average nDCG by 4.0% over the previous state-of-the-art method, AVION. Scaling to a large-size model, the gain boosts to 9.0% in average mAP (63.5 vs. 54.5), and 5.2% (74.2 vs. 69.0) in average nDCG.

We can also observe from the table that our SMS loss drives much of this improvement. Simply replacing AVION's MI-MM loss with SMS yields a 7.6% improvement in average mAP and a 4.0% improvement in average nDCG. Furthermore, EVA02-AT architectures consistently outperform vanilla ViTs: when training on EgoClip+, our base-size and large-size models improve the performance by 2.0% and 1.4% in average mAP, respectively.

CharadesEgo Action Recognition. Table II provides the comparison results on CharadesEgo Video-to-Text action recognition task. Notably, with our SMS loss, our model outperforms the previous state-of-the-art results by 3.2% on the base model and 2.8% on the large model in V2T mAP in the fine-tune setup. We also evaluate our EVA02-AT model in a zero-shot setup, and the experiments show that our EVA02-AT outperforms the ViT models by 0.4% on the base model and 1.0% on the large model, respectively.

**EgoMCQ.** We directly evaluate the EgoMCQ performance after pretraining the model on the Ego4D dataset. On EgoMCQ, our base model achieves 95.0% inter-video accuracy and 63.2% intra-video accuracy, while our large

model achieves 95.9% inter-video accuracy and 66.5% intravideo accuracy, which surpasses the previous state-of-the-art results.

#### C. Ablation Study

To evaluate the effectiveness of both our EVA02-AT network and the SMS loss function, we conduct the ablation experiments from three aspects: (1) the zero-shot performance across different network architectures; (2) the EVA02-AT model with different temporal positional embedding choices; (3) the fine-tuned performance across different loss functions.

TABLE IV Comparison between different temporal embeddings on the zero-shot EK-100 MIR benchmark.

| Backbone | Temporal PE.   | mAP  | nDCG |  |
|----------|----------------|------|------|--|
| ViT      | Learnable      | 28.4 | 29.5 |  |
| EVA02-AT | Learnable      | 28.2 | 29.9 |  |
| EVA02-AT | RoPE           | 28.8 | 30.1 |  |
| EVA02-AT | Learnable+RoPE | 30.2 | 31.1 |  |

Effect of EVA02-AT. In the zero-shot setting, we evaluate models pretrained on the EgoClip and EgoClip+ datasets, respectively. As Table III, our model consistently achieves the state-of-the-art results on both pretraining datasets without increasing the number of parameters. In contrast, backbone models like TSF and FiT, which introduce an external temporal attention block, inevitably increase the model's parameters but

![](_page_8_Figure_2.jpeg)

Fig. 5. Training curves for different loss functions. Figure (a) shows the loss value during the training process, and Figure (b) shows the validation mAP during the training process on the EK-100 MIR task. By providing an accurate learning objective, SMS decades more sharply than the other two losses.

TABLE V The performance comparison of different loss functions pretrained on the EgoClip+ dataset and fine-tuned on the EK-100 multi-instance retrieval task.

|          | Methods                                               | $V \to T$                            | mAP (%)<br>T→ V                             | Avg.                                 | V→ T                                 | DCG (%)<br>T→ V                      | Avg.                                        |
|----------|-------------------------------------------------------|--------------------------------------|---------------------------------------------|--------------------------------------|--------------------------------------|--------------------------------------|---------------------------------------------|
| ViT-B-16 | AVION MI-MM* Adaptive MI-MM SMS w/o $\tau$ SMS (ours) | 55.9<br>55.5<br>60.5<br>62.2<br>62.9 | 47.8<br>48.8<br>49.6<br>48.1<br><b>51.1</b> | 51.8<br>52.1<br>55.1<br>55.2<br>57.0 | 68.2<br>68.4<br>69.7<br>70.8<br>71.2 | 65.4<br>66.3<br>66.5<br>66.5<br>67.3 | 66.8<br>67.3<br>68.1<br>68.6<br><b>69.2</b> |
| ViT-L-14 | AVION                                                 | 57.9                                 | 51.1                                        | 54.5                                 | 70.4                                 | 67.6                                 | 69.0                                        |
|          | MI-MM*                                                | 58.7                                 | 52.7                                        | 55.7                                 | 71.9                                 | 69.4                                 | 70.6                                        |
|          | Adaptive MI-MM                                        | 65.0                                 | 54.6                                        | 59.8                                 | 73.3                                 | 70.0                                 | 71.6                                        |
|          | SMS (ours)                                            | 67.3                                 | 56.9                                        | <b>62.1</b>                          | 74.7                                 | 71.2                                 | <b>73.0</b>                                 |
|          | w/ EVA02-AT-B                                         | 64.6                                 | 53.4                                        | 59.0                                 | 72.6                                 | 69.0                                 | 70.8                                        |
|          | Δ - ViT-B-16                                          | +1.7                                 | +2.3                                        | +2.0                                 | +1.4                                 | +1.7                                 | +1.5                                        |
|          | w/ EVA02-AT-L                                         | 68.7                                 | 58.3                                        | 63.5                                 | 76.1                                 | 72.3                                 | 74.2                                        |
|          | Δ - ViT-L-14                                          | +1.4                                 | +1.4                                        | +1.4                                 | +1.4                                 | +1.1                                 | +1.2                                        |

fail to provide improved performance. i.e., the EVA02-AT outperforms LaViLa by 2.8% in average mAP for the large model. Meanwhile, compared with the architectures with joint attention, our model also achieves a better result with the help of spatial-temporal RoPE. i.e., the EVA02-AT beats ViT-B and ViT-L by 1.4% and 1.3% in average mAP, respectively.

Effect of 3D-RoPE. In table IV, we change the temporal positional embedding to (a) the learnable positional embedding, (b) 1D-RoPE embedding, and (c) learnable positional embedding with RoPE embedding. From the table, we can find that changing the temporal positional embedding will not influence the performance significantly, but (c) still outperforms all the other settings. Concretely, compared to the learnable temporal positional embedding, RoPE improves the model by 2.0% in average mAP. And compared to the case that only uses temporal RoPE, a learnable positional embedding can provide a 1.4\$ gain in average mAP. Additionally, the experiment suggests that preserving the model's extrapolation ability, i.e., using RoPE as the only temporal positional embedding, does not lead to a noticeable performance drop.

**Effect of SMS Loss.** To verify the effectiveness and robustness of our SMS loss, we conduct an ablation study on both our ViT-B-based and ViT-L-based models. All experiments across

![](_page_8_Figure_9.jpeg)

Fig. 6. Training curves for different hyper-parameter choices in SMS loss. Figures (a) and (b) show the average mAP and nDCG performances when  $\gamma$  changes.

different loss functions are conducted under the same learning rate and optimizer settings. We choose the best-performing hyperparameters for each loss function, i.e., a margin of 0.2 for the MI-MM loss and 0.4 for the adaptive MI-MM loss. The experiment results are presented in Table V.

For both the ViT-B-based and ViT-L-based models, our SMS loss demonstrates superior performance compared to its counterparts. Specifically, for the ViT-B-based model, our SMS loss improves the average mAP by 1.9% and the average nDCG by 2.4% compared to the adaptive MI-MM loss. Similarly, for the ViT-L-based model, our SMS loss also improves the model in average mAP by 2.3% and in average nDCG by 2.4%.

We also conducted an experiment using a ViT-B-based model with  $\tau=0$  to evaluate the impact of the relaxation factor. This factor helps prevent over-optimization when the correlation values between positive and negative samples are similar. Our results show a performance drop of 1.8% in average mAP and 0.8% in average nDCG compared to the case where  $\tau=0.1$ . This demonstrates the crucial role of the relaxation factor in ensuring optimal model performance.

We next examine the impact of the SMS loss hyperparameter,  $\gamma$ . Fig. 6 plots average mAP as  $\gamma$  varies from 0.3 to 0.8. Our results show that the model achieves its highest mAP at  $\gamma = 0.6$ . Moreover, performance remains stable across the entire range, since the mAP difference between the best and worst settings is only 1.2%.

The training curves for different loss functions are presented in Fig. 5. Notably, a performance gap emerges as early as 20 epochs, with the SMS loss continuing to decrease exponentially, while the other two loss functions show slower declines over the next 80 epochs. Although the absolute value of SMS loss is naturally lower than that of MI-MM and adaptive MI-MM losses, the results highlight that an accurate learning objective significantly helps the fine-tuning process.

# VI. Conclusion

This paper proposes the EVA02-AT suite, a strong and training-efficient video-text CLIP model. Instead of divided spatial and temporal attention blocks used in typical video encoders, we adopt a joint attention block along with an

integrated Spatial-Temporal RoPE, which conducts both spatial and temporal RoPE on the entire feature dimension. This approach enables global attention across all the patches within video clips, and avoids an uneven manual division of the feature dimension. The EVA02-AT can be trained on the basis of the image CLIP model, achieves the generalized egocentric video representations without increasing the number of parameters, and surpasses all the previous state-of-the-art results on major egocentric benchmarks. Additionally, we propose the SMS loss, which significantly advances the state-of-the-art performance on the EK-100 MIR task.

#### REFERENCES

- [1] J. Carreira and A. Zisserman, "Quo vadis, action recognition? a new model and the kinetics dataset," in CVPR, 2017.
- [2] T. Liu, Q. Meng, J.-J. Huang, A. Vlontzos, D. Rueckert, and B. Kainz, "Video summarization through reinforcement learning with a 3d spatiotemporal u-net," *IEEE Transactions on Image Processing*, vol. 31, pp. 1573–1586, 2022.
- [3] M. Lu, Z.-N. Li, Y. Wang, and G. Pan, "Deep attention network for egocentric action recognition," *IEEE Transactions on Image Processing*, vol. 28, no. 8, pp. 3703–3713, 2019.
- [4] Y. Zhao and P. Krähenbühl, "Training a large video model on a single machine in a day," arXiv preprint arXiv:2309.16669, 2023.
- [5] Y. Wang, K. Li, Y. Li, Y. He, B. Huang, Z. Zhao, H. Zhang, J. Xu, Y. Liu, Z. Wang *et al.*, "Internvideo: General video foundation models via generative and discriminative learning," *arXiv preprint arXiv:2212.03191*, 2022.
- [6] Y. Wang, K. Li, X. Li, J. Yu, Y. He, C. Wang, G. Chen, B. Pei, Z. Yan, R. Zheng, J. Xu, Z. Wang, Y. Shi, T. Jiang, S. Li, H. Zhang, Y. Huang, Y. Qiao, Y. Wang, and L. Wang, "InternVideo2: Scaling foundation models for multimodal video understanding," arXiv preprint arXiv:2403.15377, 2024.
- [7] S. Pramanick, Y. Song, S. Nag, K. Q. Lin, H. Shah, M. Z. Shou, R. Chellappa, and P. Zhang, "EgoVLPv2: Egocentric video-language pre-training with fusion in the backbone," in *ICCV*, 2023, pp. 5285– 5297.
- [8] K. Q. Lin, J. Wang, M. Soldan, M. Wray, R. Yan, E. Z. XU, D. Gao, R.-C. Tu, W. Zhao, W. Kong, C. Cai, W. HongFa, D. Damen, B. Ghanem, W. Liu, and M. Z. Shou, "Egocentric video-language pretraining," in *NeurIPS*, vol. 35, 2022, pp. 7575–7586.
- [9] A. Falcon, G. Serra, and O. Lanz, "Improving semantic video retrieval models by training with a relevance-aware online mining strategy," *Computer Vision and Image Understanding*, 2024.
- [10] Y. Zhao, I. Misra, P. Krähenbühl, and R. Girdhar, "Learning video representations from large language models," in CVPR, 2023, pp. 6586– 6597.
- [11] B. Pei, G. Chen, J. Xu, Y. He, Y. Liu, K. Pan, Y. Huang, Y. Wang, T. Lu, L. Wang et al., "Egovideo: Exploring egocentric foundation model and downstream adaptation," arXiv preprint arXiv:2406.18070, 2024.
- [12] Z. Tong, Y. Song, J. Wang, and L. Wang, "Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training," *NeurIPS*, vol. 35, pp. 10078–10093, 2022.
- [13] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki, "Laion-400m: Open dataset of clip-filtered 400 million image-text pairs," arXiv preprint arXiv:2111.02114, 2021.
- [14] Y. Wang, Y. He, Y. Li, K. Li, J. Yu, X. Ma, X. Li, G. Chen, X. Chen, Y. Wang et al., "Internvid: A large-scale video-text dataset for multimodal understanding and generation," in ICLR, 2023.
- [15] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge et al., "Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution," arXiv preprint arXiv:2409.12191, 2024.
- [16] Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, "EVA-CLIP: Improved training techniques for clip at scale," arXiv preprint arXiv:2303.15389, 2023.
- [17] Z. Yang, J. Teng, W. Zheng, M. Ding, S. Huang, J. Xu, Y. Yang, W. Hong, X. Zhang, G. Feng, D. Yin, X. Gu, Y. Zhang, W. Wang, Y. Cheng, T. Liu, B. Xu, Y. Dong, and J. Tang, "CogVideoX: Text-to-video diffusion models with an expert transformer," arXiv preprint arXiv:2408.06072, 2024.

- [18] Y. Fang, Q. Sun, X. Wang, T. Huang, X. Wang, and Y. Cao, "EVA-02: A visual representation for neon genesis," arXiv preprint arXiv:2303.11331, 2023.
- [19] X. Wang, X. Han, W. Huang, D. Dong, and M. R. Scott, "Multi-similarity loss with general pair weighting for deep metric learning," in CVPR, 2019, pp. 5022–5030.
- [20] K. Grauman, A. Westbury, E. Byrne, Z. Chavis, A. Furnari, R. Girdhar, J. Hamburger, H. Jiang, M. Liu, X. Liu et al., "Ego4d: Around the world in 3,000 hours of egocentric video," in CVPR, 2022, pp. 18995–19012.
- [21] D. Damen, H. Doughty, G. M. Farinella, A. Furnari, J. Ma, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray, "Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100," *IJCV*, vol. 130, p. 33–55, 2022.
- [22] D. Damen, H. Doughty, G. M. Farinella, S. Fidler, A. Furnari, E. Kazakos, D. Moltisanti, J. Munro, T. Perrett, W. Price, and M. Wray, "Scaling egocentric vision: The epic-kitchens dataset," in ECCV, 2018.
- [23] G. A. Sigurdsson, A. Gupta, C. Schmid, A. Farhadi, and K. Alahari, "Charades-ego: A large-scale dataset of paired third and first person videos," arXiv preprint arXiv:1804.09626, 2018.
- [24] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
- [25] G. Bertasius, H. Wang, and L. Torresani, "Is space-time attention all you need for video understanding?" in *ICML*, 2021.
- [26] M. Bain, A. Nagrani, G. Varol, and A. Zisserman, "Frozen in time: A joint video and image encoder for end-to-end retrieval," in *ICCV*, 2021, pp. 1728–1738.
- [27] L. Wang, B. Huang, Z. Zhao, Z. Tong, Y. He, Y. Wang, Y. Wang, and Y. Qiao, "VideoMAE V2: Scaling video masked autoencoders with dual masking," in CVPR, 2023, pp. 14549–14560.
- [28] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, R. Ring, E. Rutherford, S. Cabi, T. Han, Z. Gong, S. Samangooei, M. Monteiro, J. L. Menick, S. Borgeaud, A. Brock, A. Nematzadeh, S. Sharifzadeh, M. a. Bińkowski, R. Barreira, O. Vinyals, A. Zisserman, and K. Simonyan, "Flamingo: a visual language model for few-shot learning," in *NeurIPS*, vol. 35, 2022, pp. 23716–23736.
- [29] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, "Roformer: Enhanced transformer with rotary position embedding," *Neurocomputing*, vol. 568, p. 127063, 2024.
- [30] H. Zhang, X. Li, and L. Bing, "Video-LLaMA: An instruction-tuned audio-visual language model for video understanding," https://arxiv.org/abs/2306.02858, 2023.
- [31] X. Wei, X. Liu, Y. Zang, X. Dong, P. Zhang, Y. Cao, J. Tong, H. Duan, Q. Guo, J. Wang, X. Qiu, and D. Lin, "VideoRoPE: What makes for good video rotary position embedding?" https://arxiv.org/abs/2502.05173, 2025.
- [32] P. Khosla, P. Teterwak, C. Wang, A. Sarna, Y. Tian, P. Isola, A. Maschinot, C. Liu, and D. Krishnan, "Supervised contrastive learning," in *NeurIPS*, vol. 33, 2020, pp. 18 661–18 673.
- [33] Z. Lou, H. Xue, Y. Wang, C. Zhang, X. Yang, and S. Hu, "Parameter-free deep multi-modal clustering with reliable contrastive learning," *IEEE Transactions on Image Processing*, vol. 34, pp. 2628–2640, 2025.
- [34] X. Wang, Y. Yan, H.-M. Hu, B. Li, and H. Wang, "Cross-modal contrastive learning network for few-shot action recognition," *IEEE Transactions on Image Processing*, vol. 33, pp. 1257–1271, 2024.
- [35] A. Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," arXiv preprint arXiv:1807.03748, 2018.
- [36] A. Miech, J.-B. Alayrac, L. Smaira, I. Laptev, J. Sivic, and A. Zisserman, "End-to-end learning of visual representations from uncurated instructional videos," in CVPR, 2020, pp. 9879–9889.
- [37] F. Schroff, D. Kalenichenko, and J. Philbin, "Facenet: A unified embedding for face recognition and clustering," in CVPR, 2015.
- [38] Y. Sun, C. Cheng, Y. Zhang, C. Zhang, L. Zheng, Z. Wang, and Y. Wei, "Circle loss: A unified perspective of pair similarity optimization," in CVPR, 2020, pp. 6398–6407.
- [39] A. Falcon, S. Sudhakaran, G. Serra, S. Escalera, and O. Lanz, "Relevance-based margin for contrastively-trained video retrieval models," in *ICMR*, 2022, pp. 146–157.
- [40] R. Wang, D. Chen, Z. Wu, Y. Chen, X. Dai, M. Liu, L. Yuan, and Y. Jiang, "Masked video distillation: Rethinking masked feature modeling for self-supervised video representation learning," in CVPR, 2023, pp. 6312–6322.
- [41] M. Wray, D. Larlus, G. Csurka, and D. Damen, "Fine-grained action retrieval through multiple parts-of-speech embeddings," in *ICCV*, 2019, pp. 450–459.
- [42] M. Wray, H. Doughty, and D. Damen, "On semantic similarity in video retrieval," in CVPR, 2021, pp. 3650–3660.

- [43] K. Q. Lin, A. J. Wang, R. Yan, E. Z. Xu, R. Tu, Y. Zhu, W. Zhao, W. Kong, C. Cai, H. Wang, W. Liu, and M. Z. Shou, "Egocentric video-language pretraining @ epic-kitchens-100 multi-instance retrieval challenge 2022," arXiv preprint arXiv:2207.01334, 2022.
- [44] D. Chen and W. B. Dolan, "Collecting highly parallel data for paraphrase evaluation," in *ACL*, 2011, pp. 190–200.
- [45] E. Kazakos, A. Nagrani, A. Zisserman, and D. Damen, "Epic-fusion: Audio-visual temporal binding for egocentric action recognition," in *ICCV*, 2019, pp. 5492–5501.
- [46] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, "Language models are unsupervised multitask learners," *OpenAI blog*, vol. 1, no. 8, p. 9, 2019.
- [47] A. Miech, D. Zhukov, J.-B. Alayrac, M. Tapaswi, I. Laptev, and J. Sivic, "HowTo100M: Learning a text-video embedding by watching hundred million narrated video clips," in *ICCV*, 2019, pp. 2630–2640.
- [48] S. Xie, C. Sun, J. Huang, Z. Tu, and K. Murphy, "Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification," in ECCV, 2018, pp. 305–321.
- [49] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning transferable visual models from natural language supervision," in *ICML*, vol. 139, 2021, pp. 8748–8763.
- [50] K. Ashutosh, R. Girdhar, L. Torresani, and K. Grauman, "Hiervl: Learning hierarchical video-language embeddings," in CVPR, 2023, pp. 23 066–23 078.
- [51] H. A. Valdez, K. Min, and S. Tripathi, "SViTT-Ego: A sparse videotext transformer for egocentric video," arXiv preprint arXiv:2406.09462, 2024
- [52] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization." in ICLR, 2019.
- [53] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of bert: smaller, faster, cheaper and lighter," arXiv preprint arXiv:1910.01108, 2019.
- [54] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, "RoBERTA: A robustly optimized bert pretraining approach," arXiv preprint arXiv:1907.11692, 2019.