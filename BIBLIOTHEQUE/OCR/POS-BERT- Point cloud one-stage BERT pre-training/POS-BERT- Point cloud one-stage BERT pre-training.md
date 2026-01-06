# POS-BERT: Point Cloud One-Stage BERT Pre-Training

 $\begin{array}{cccccccccccccccccccccccccccccccccccc$ 

1 Digital Medical Research Center, School of Basic Medical Sciences, Fudan University 2 Shanghai AI Lab

3 Shanghai Key Laboratory of Medical Image Computing and Computer Assisted Intervention {fukexue, mnwang}@fudan.edu.cn, gaopeng@pjlab.org.cn

# **Abstract**

Recently, the pre-training paradigm combining Transformer and masked language modeling has achieved tremendous success in NLP, images, and point clouds, such as BERT. However, directly extending BERT from NLP to point clouds requires training a fixed discrete Variational AutoEncoder (dVAE) before pre-training, which results in a complex two-stage method called Point-BERT. Inspired by BERT and MoCo, we propose POS-BERT, a one-stage BERT pre-training method for point clouds. Specifically, we use the mask patch modeling (MPM) task to perform point cloud pre-training, which aims to recover masked patches information under the supervision of the corresponding tokenizer output. Unlike Point-BERT, its tokenizer is extra-trained and frozen. We propose to use the dynamically updated momentum encoder as the tokenizer, which is updated and outputs the dynamic supervision signal along with the training process. Further, in order to learn high-level semantic representation, we combine contrastive learning to maximize the class token consistency between different transformation point clouds. Extensive experiments have demonstrated that POS-BERT can extract high-quality pre-training features and promote downstream tasks to improve performance. Using the pretraining model without any fine-tuning to extract features and train linear SVM on ModelNet40, POS-BERT achieves the state-of-the-art classification accuracy, which exceeds Point-BERT by 3.5%. In addition, our approach has significantly improved many downstream tasks, such as fine-tuned classification, few-shot classification, part segmentation. The code and trained-models will be available at: https://github.com/fukexue/POS-BERT.

# 1 Introduction

Point cloud is an intuitive, flexible and memory-efficient 3D data representation and has become indispensable in 3D vision. Learning powerful point cloud representation is very crucial for facilitating machines to understand the 3D world, which is beneficial for promoting the development of many important real-world applications, such as autonomous driving [1], augmented reality [2] and robotics [3]. With the rapid development of deep learning in these years [4, 5], supervised 3D point cloud analysis methods have made great progress [6–9]. However, both exponentially increasing demand for data and expensive 3D data annotation hinder further performance improvement of supervised methods. On the contrary, due to the widespread popularity of 3D sensors (Lidar, ToF camera, RGB-D sensor or camera stereo-pair), a large number of unlabeled point cloud data are available for self-supervised point cloud representation learning.

<sup>\*</sup>Corresponding author

Unsupervised or self-supervised learning methods have shown their effectiveness in different fields [10–14]. Recent work [15, 11, 16, 13, 17] has achieved good performance by combining point clouds with self-supervised learning techniques, such as generative adversarial networks (GAN) [11], variational autoencoders (VAE) [10], and Gaussian mixture models (GMM) [12]. These methods usually rely on tasks such as distribution estimation or reconstruction to provide supervisory signals, and can learn good local detail features, but it is difficult to capture higher-level semantic features. To learn higher-level semantic features, some methods learn point cloud representations, such as orientation estimation, by constructing a series of transformation prediction tasks [18–20]. Inspired by unsupervised learning of 2D images [21–23], point cloud representation is learned by constructing a series of contrast views [24–27] and combining the most advanced comparative learning methods. However, these methods rely on network structures with specific inductive bias to achieve good performance, such as PointNet++, DGCNN, and so on. In addition, previous methods have never studied the performance of standard transformers in point cloud analysis tasks.

![](_page_1_Figure_1.jpeg)

Figure 1: The difference between our approach and Point-BERT. (a) Point-BERT uses an additional pre-trained dVAE as the Tokenizer, which is frozen during training and the output is discrete. (b) Our approach eliminates the need for extra processing stages, and Tokenizer is derived from Encoder through momentum updates, which are dynamic during the training process and the output is continuous.

Recently, Transformer has achieved impressive results in language and image tasks through extensive unlabeled data learning and is becoming increasingly popular. Inspired by NLP, Point-BERT devise a mask patch modeling (MPM) task to pre-train point cloud Transformers. To generate meaningful representations for masked patches to guide point cloud Transformers learning, Point-BERT additionally trains a discrete Variational AutoEncoder (dVAE) based on DGCNN as a tokenizer, as shown in Fig.1 (a). As a result, Point-BERT is a two-stage approach, in which the weight of tokenizer is frozen, and its feature extraction capabilities directly affect the learning of point cloud Transformers. Unlike Point-BERT, we extract meaningful representations of masked patches by replacing the frozen tokenizer with momentum encoder, which is dynamically updated, as shown in Fig.1 (b). Therefore, our approach is one-stage, and the meaningful representation of mask patches will become better as the training progresses. In this article, we propose a one-stage BERT point cloud pre-training method named POS-BERT. Inspired by BERT and MoCo, we used MPM task to pre-train on point cloud and chose standard Transformer without specific inductive biases as backbone. Specifically, we first divide the point cloud into a series of patches, then randomly mask out some patches and feed them into an encoder based on standard transformer. Then, we use a dynamically updated momentum encoder as the tokenizer. The Momentum Encoder has the same network structure as the Encoder, but it does not have gradient backward. Its weight is jointly optimized with MPM through momentum update during the pre-training stage. This greatly simplifies the pre-training step. Next, the point cloud patches before masked are fed to the Momentum Encoder. The objective of MPM is to make the Encoder recover output consistent with the Momentum Encoder output at the masked patches position as much as possible. However, recovering the masked patch information independently leads to limited ability of point cloud transformer's class token to extract high-level semantic information. To address this problem, we perform contrastive learning to maximize the class token consistency between different augmentation (for example, cropping) point cloud pairs. The main contributions are summarized as follows:

1) We propose a Point Cloud One-Stage BERT pre-training method, and named POS-BERT. We use momentum encoder to provide continuous and dynamic supervision signals for masked patches in

- mask patch modeling pretext task. The Momentum Encoder is updated dynamically during the pre-training stage and does not require extra pre-training processing.
- 2) We introduce a contrastive learning strategy on transformer's class token between different augmentation point cloud pairs, which can help point cloud transformer's class token obtain a better high-level semantic representation.
- 3) Experiments demonstrate that POS-BERT achieves state-of-the-art performance in linear SVM classification task and downstream tasks, such as classification and segmentation.

# 2 Related work

Point Cloud Self-Supervised Learning The goal of self-supervised learning is to learn good feature representations from unlabeled raw data so that they can be well adapted to various downstream tasks [28]. Currently, self-supervised learning has been extensively studied in point cloud representation learning, and they focus on constructing a pretext task to help the network better learn 3D point cloud representations. A commonly adopted pretext task is to reconstruct the input point cloud from the latent encoding space, which can be implemented through Variational AutoEncoders [29–34], Generative Adversarial Learning (GANs) [35, 36], Gaussian Mixed Model [12, 37], etc. However, these methods are computationally expensive, and rely excessively on reconstructing local details, making it difficult to learn high-level semantic features. Hence, some researchers employed Transformation Prediction as a prediction pseudo-task. Sauder et al. [18] proposed to use jigsaw puzzle as a pretext task for 3D point cloud representation learning. Wang et al. [19] destroyed the point cloud and then pretrained the network by a self-supervised manner with the help of point cloud complementation task. Poursaeed et al. [20] used orientation estimation as a pretext task by randomly rotating the point cloud and then allowing the network to predict the rotation. As contrastive learning becomes increasingly popular, Jing and Afham et al. [24, 25], proposed a task training network for finding cross-modality correspondences. Specifically, they obtain the corresponding 2D view by rendering the 3D model, and then extracts 2D view features and 3D point cloud features using 2D convolutional networks and graph convolutional networks. Finally, the instance correspondence between the two modalities is estimated based on these features. Qi et al. [38] calculated the contrastive loss on matched point pairs by rigidly transforming the point clouds with feature vectors for each point of the two point clouds before and after the transformation. Wang et al. [26] designed a multi-resolution contrastive learning training strategy that can train point-by-point and shape feature vectors simultaneously. Inspired by BYOL [39], Huang et al. [27] constructed point cloud pairs that undergo spatio-temporal transformations, and forced the network to learn the consistency between different augmented views. However, all previous studies resort to point cloud domain-specific network architectures to achieve promising performance, which would greatly hinder the development of deep learning towards a generalized model. More importantly, these studies have never investigated self-supervised representation learning using a transformer-based point cloud processing network. Recently, Point-BERT [40] has proposed a modeling approach using standard transformer network combined with mask language modeling for the first time to achieve self-supervised representation learning of point clouds, which is a direct extension of BERT [41] (popular in the field of NLP) on point clouds. However, there is no mature BPE [42] algorithm in the point cloud domain as in NLP, leading to a lack of an effective vocabulary to guide the learning of mask language modeling. For this reason, Point-BERT [40] pre-trained a discrete Variational AutoEncoder (dVAE) [43] as tokenizer through an additional point cloud network DGCNN to construct vocabularies for point cloud patches. This directly brings about two problems: First, the whole method becomes a complex two-stage solution; Second, the weights of the pre-trained tokenizer are frozen and cannot change adaptively with the network training process, and the performance of the fixed tokenizer will directly doom the performance of the pre-trained model. Unlike Point-BERT, we use dynamically updated momentum encoder instead of a frozen tokenizer to extract features from point cloud patches. Additionally, our solution is one-stage, and the Momentum Encoder can be continuously updated as the network training progresses, providing the network with a suitable feature representation of point cloud patches for the current training stage.

**Transformers** Transformer has made great advances in the field of machine translation and natural language processing with its long-range modeling capability brought by the attention mechanism. Inspired by the successful applications of Transformer in NLP field, it has also been introduced into the image field [44–46], leading to backbone networks such as ViT [44], SWin [45], Container[5],

etc., which surpassed CNN-based ResNet and showed excellent performance in downstream tasks such as classification [44], segmentation [47], object detection [48]. Although there is a trend of grand unification of transformer in the field of NLP and image, the development of transformer in the field of point cloud is highly slow. PCT [49] and PointTransformer [50] have modified the transformer layer in standard transformer and combined with layer aggregation operation to achieve point cloud classification and segmentation. Unlike these approaches, Point-BERT [40] achieves comparable performance with a standard transformer without introducing a bias structure, but it requires a specific point cloud network DGCNN to provide supervised signals for pre-training. By comparison, our proposed method completely rejects the introduction of other networks and uses only the standard transformer-based network to learn point cloud representations.

Mask Language Modeling Paradigm Mask language modeling was proposed in BERT [51], which revolutionized the pre-training paradigm for natural language. Inspired by BERT, Bao et al. proposed BEiT [52] for pre-training a standard transformer applicable to images. It maps the input image patches into meaningful discrete tokens by dVAE [43], then randomly masks some of the image patches, and feeds the masked image patches and the remaining images into the standard transformer to reconstruct the tokens of these masked image patches. Following BEiT, Zhou et al. [9] perform masked prediction with an online tokenizer. Unlike BEiT, He et al. [53] trained the network by directly reconstructing the original image patches. Inspired by BEiT, Yu et al. [40] proposed Point-BERT for point cloud pre-training and demonstrated that the MLM paradigm is feasible for point cloud pre-training. We inherit the idea of Yu et al. and also adopt the MLM approach for point cloud pre-training.

Contrastive learning Contrastive learning is a branch of self-supervised learning, which learns knowledge from the data itself without the demand of data annotation. The main idea of contrastive learning is to maximize the consistency between positive sample pairs and the differences between negative sample pairs. Representative methods of contrastive learning include MoCo series [54, 21, 22] and SimCLR [55]. Recently, BYOL [23] and Barlow Twins [56] pointed out that only using positive samples can still obtain powerful features. In this paper, we introduce the idea of contrastive learning to help point cloud Transformer learn the high-level semantic representation.

## 3 Method

We propose a Point Cloud One-Stage BERT pre-training approach POS-BERT, which is simple and efficient. Fig.2 illustrates the overall framework of POS-BERT. Firstly, the global point cloud set  $P_g$  and the local point cloud set  $P_l$  are obtained by cropping the raw point clouds  $P \in \mathbb{R}^{N \times 3}$ with different cropping ratios. Then, we use the PGE module to divide both global and local point clouds into smaller patches with fixed number of points and embed the patches into high-dimensional representation (patch token) though standard Transformer-based encoders. Because local point clouds do not represent complete objects very well, only global point clouds are input into the Momentum Encoder, which is dynamically updated to encode meaningful representations to provide learning objectives for the Encoder. The Encoder is trained using the mask patch modeling task to match the Momentum Encoder outputs. Some patches of the global point clouds are randomly masked out and position information is added to the corresponding masked patches, and then they are input into the Encoder together with the local point cloud set. Finally, we calculate the mask patch modeling loss  $\mathcal{L}_{MPM}$  between the Encoder outputs' patch tokens and the Momentum Encoder outputs' patch token, and the global feature loss loss  $\mathcal{L}_{GFC}$  between the Encoder outputs' class token and the Momentum Encoder outputs' class token. Overall, our framework consists of four key components: Encoder, Momentum Encoder, Mask Patch Modeling and Loss Function and they will be introduced in detail the following part of this section. We will start with Section 3.1 on how to transform point into patch embedding with the Encoder. Next, mask patch modeling is described in section 3.2. Then we introduce the dynamic tokenizer implemented by the Momentum Encoder for providing supervision for the MPM tasks in section 3.3. Finally, we describe our loss function in section 3.4.

# 3.1 Point2Patch Embedding and Encoder Architecture

The simplest way to extract point cloud features is to input each point into the transformer as one token. Because the complexity of transformer is  $O(N^2)$ , where N is the length of the input token, extracting feature of each point directly will result in memory explosion. Fig.3 describes the overall

![](_page_4_Figure_0.jpeg)

Figure 2: The overall framework of POS-BERT. PGE represents patch generation and embedding module, CLS represents class token, EMA represents exponential moving average, solid line represents gradient back-propagation, and dotted line represents stop-gradient operator.

pipeline of the Transformer-based feature extraction in this paper. Following Point-BERT, we divide a given global/local point cloud P into local patches with a fixed number of K points. In order to minimize overlap between patches  $\{p_i \in \mathbb{C} \mid i=1\dots Q\}$ , we first calculate the number of patches  $Q = \operatorname{ceil}(N/K)$ , then use farthest point sampling (FPS) algorithm to sample the center point  $c_i$  of each patch. The k-nearest neighbor algorithm is used to obtain K neighbors for each center point, and the center point and corresponding neighbor points form a local patch  $p_i$ . Next, Using the PointNet and maxpooling operations to map point coordinates of each patch to a high-dimensional embedding as patch tokens. Finally, these patch tokens are fed into the standard transformer with a learnable class token.

We used a standard transformer as the Encoder backbone, which consists of a series of stacked multihead self-attention layers and fully connected feed-forward network. As mentioned earlier, class tokens  $\{t_0\}$  and a series of patch tokens  $\{t_1,\ldots,t_M\}$  are concatenated along the patch dimension to get the transformer's input  $T_0=\{t_0,t_1,\ldots,t_Q\}\in R^{(Q+1)\times D}$ . After  $T_0$  passes through the h-layer transformer block, we get the feature of each patch  $T_{\rm h}=\{t_0^h,t_1^h,\ldots,t_Q^h\}$  with global receptive field. Finally, we map the features of each patch to the loss space, where the projector is composed of multiple layers of MLP. In the inference stage and downstream tasks, we do not need the projector. Decoupling the feature representation and loss function can make the learned patch's features more general.

![](_page_4_Figure_4.jpeg)

Figure 3: The architecture of standard transformer-based Encoder.

# 3.2 Mask Patch Modeling

Inspired by Point-Bert, we also use a mask patch modeling task to pretrain the point cloud Transformer. As described in Section 3.1, we have obtained the transformer's input  $T_0 = \{t_0, t_1, \ldots, t_M\}$ . Masked patch tokens  $\mathcal{M}$  is obtained by randomly masking the tokens of some patches in  $T_0$ , except  $t_0$ . Next, we randomly mask/replace [20%, 40%] patch tokens with a learnable mask token  $E[m] \in \mathbb{R}^D$ , where masked tokens are defined as  $m_t$ . Then, the center point position embedding  $pos = mlp(c_i)$  corresponding to patch tokens is added to  $m_t$ ,  $c_i$  represents the xyz coordinate of the patch center

point. Finally, the transformer's input tokens obtained by high-dimensional embedding after masking can be expressed as  $\widehat{T}_0 = \{t_0\} \cup \{t_i \mid i \notin \mathcal{M}\}_{i=1}^Q \cup \{E[m] + \mathrm{pos}_i \mid i \in \mathcal{M}\}_{i=1}^Q$ , and the lost information of masked tokens is recovered from  $\widehat{T}_0$  through Encoder.

# 3.3 Dynamic Tokenizer by Momentum Encoder

Momentum Encoder is often used in contrastive learning to provide a global semantic supervision for target network. Inspired by MoCo, we propose a dynamically updated tokenizer, which is implemented by momentum Encoder. Grill's preliminary experiments show that even using the output of random initialization network as supervision, target network can also learn a better output representation than random initialization network [23]. This result provides a strong support for the replacement of dVAE by the dynamically updated momentum encoder during early training. Therefore, we use a random network to initialize the Momentum Encoder. Although randomly initialized networks can help Encoder get better representation in the early stages of training, if the performance of tokenizer is not continuously improved, the ability of Encoder will stop as tokenizer stops. Accordingly, we need a tokenizer that can dynamically update and improve its quality while at the same time its output does not change rapidly before and after each update. The momentum encoder in contrastive learning solves these two concerns well, and its update formula is as follows:

$$\theta_m = \lambda \theta_m + (1 - \lambda)\theta_e \tag{1}$$

where,  $\theta_m$  represents the weight of Momentum Encoder,  $\theta_e$  represents the weight of Encoder.  $\lambda \in [0,1)$  is a momentum coefficient, which follows a cosine schedule from 0.996 to 1 during training.

Momentum Encoder enhances itself by constantly introducing new knowledge learned from Encoder, so Momentum Encoder also has the ability to recover lost information. Moreover, it dynamically integrates the Encoder weights of multiple training stages, and has better feature extraction ability than the Encoder. Therefore, our final pre-training model weights come from Momentum Encoder.

### 3.4 Loss Function

We hope that the pre-training model can not only recover the lost information, but also learn the high-level semantic representation. Therefore, our loss function consists of two parts: mask patch modeling loss  $\mathcal{L}_{MPM}$  and global feature contrastive loss  $\mathcal{L}_{GFC}$ .

For mask patch model loss  $\mathcal{L}_{MPM}$ , we encourage the Encoder to recover the information lost by masked patch under the supervision of meaningful representations, which is generated by Momentum Encoder. The formula of mask patch model loss is as follows:

$$\mathcal{L}_{MPM} = \min_{\theta_e} \sum_{i \notin \mathcal{M}} -O_m^i \cdot \log\left(O_e^i\right) \tag{2}$$

where,  $O_m^i$  represents the output of the Momentum Encoder corresponding to the *i*-th patch,  $O_e^i$  represents the output of the Encoder corresponding to the *i*-th patch.

Although the idea of contrastive learning was also used in Point-BERT to achieve high-level semantic features, the results were not ideal, which can be observed from Tab .1. In addition, it needs to maintain a memory bank to store a large number of negative samples, which takes up a large amount of storage space. In contrast, we utilize different cropping rate to obtain different augmentation state point clouds: global point clouds and local point clouds with the following formula:

$$P_g^i = \text{crop}(P, \text{rand}(r_{g1}, r_{g2})), \quad i = 1 \cdots I$$
  
 $P_l^j = \text{crop}(P, \text{rand}(r_{l1}, r_{l2})), \quad j = 1 \cdots J$ 
(3)

where  $\operatorname{crop}(\cdot\,,\cdot)$  represents cropping an area at a fixed ratio, represented by the second parameter.  $\operatorname{rand}(\cdot\,,\cdot)$  generates a random value between the maximum and the minimum values. Here,  $r_{g1}$  and  $r_{g2}$  are the minimum and maximum cropping ratio for generating the global point cloud set, respectively. Similarly,  $r_{l1}$  and  $r_{l2}$  are the minimum and maximum cropping ratios for generating

of the local point cloud set, respectively. I and J are the number of point clouds in  $P_g$  and  $P_l$ , respectively. During training phase, the Encoder encodes masked global point clouds and local point clouds, while the Momentum Encoder only encodes global point clouds.

$$\mathcal{L}_{GFC} = \min_{\theta_e} \sum_{i=1}^{I} \sum_{j \neq i}^{J} - \left(O_m^{cls}\right)_i \cdot \log\left(\left(O_e^{cls}\right)_j\right) \tag{4}$$

Finally, we combine all the above-mentioned loss function as our final self-supervised objectives:

$$\mathcal{L} = \omega_1 * \mathcal{L}_{MPM} + \omega_2 * \mathcal{L}_{GFC} \tag{5}$$

where, the hyperparameters  $\omega$  control the balance between loss functions, for all the experiments in this paper, we set  $\omega_1=0.5,\,\omega_2=1.0.$ 

# 4 Implementation and Dataset

## 4.1 Implementation

**Pre-training** We use Adamw optimizer [57] to train the network with the initial learning rate 0.0001. The learning rate increases linearly for the first 10 epochs and then decays with a cosine schedule. We train the pre-training model with the batch size 64 and 200 epochs, and the whole pre-training is implemented on NVIDIA A100. For the exponential moving average weight  $\lambda$  of the target network, the starting value is set to 0.996 and then gradually increases to 1. The dimension K of the final features used to calculate the loss is set to 512. When cropping the global point cloud, the crop ratios  $\gamma_{g1}$ ,  $\gamma_{g2}$  are set to 0.7 and 1.0, respectively, and the number of crops I is 2. When cropping local point clouds, the crop ratios  $\gamma_{l1}$ ,  $\gamma_{l2}$  are set to 0.2 and 0.5, respectively, and the number of crops J is 8. Additionally, we use the FPS sample half of the original point cloud as different resolution point clouds and add them to local point cloud set. The number of different resolution point clouds is 2.

**Classification** We use a fully connected MLP network that combines ReLU, BN, and Dropout operations as the classification head. The SGD is used as the optimizer to fine tune the classification network with cosine schedule. We set the batch size to 32.

**Segmentation** Different from the classification task, the segmentation task needs to predict pre-point labels. We first select multiple stage features of network, including the initial input feature of standard transformer and the output features of layer 3 and layer 7. We cascade the features of these different layers, and then use the point feature propagation in PointNet++ to propagate the features of the 256 down sampled points to the 2048 raw input points. Finally, MLP is used to map the features to the segmentation label space. Our batch size is 16 with a learning rate initialized to 0.0002 and decayed via the cosine schedule. We use the Adamw optimizer to train the segmentation network.

### 4.2 Dataset

In the experiments of this paper, four datasets (ShapeNet [58], ModelNet40 [59], SacnObjectNN [60], and ShapeNetPart [61]) are used.

**ShapeNet** contains 57448 CAD models, with a total of 55 categories. For the acquisition of point cloud data, we follow the processing method of Yang et al., and sample 2048 points from each CAD model surface. We use ShapeNet dataset as pre-training dataset. In the pre-training stage, we use the farthest point sampling algorithm to select 64 group center points, and divide 2048 points into 64 groups, where each group contains 32 points.

**ModelNet40** contains 12,331 handmade CAD models of from 40 categories and is widely used for point cloud classification tasks. We follow Yu et al. to sample 8192 points from each CAD model surface. According to the official split, 9,843 are used for training and 2,468 for testing. Following the work of Yu et al. [40], we generated a Fewshot-ModelNet40 dataset based on ModelNet40. "M-way N-shot" represents the data under different settings, where M-way represents the number of categories selected for training, N-shot represents the number of samples for each category, and the

Table 1: Classification results with linear SVM on ModelNet40. These models are trained in ShapeNet.

| Method               | Year | Input | Accuracy      |
|----------------------|------|-------|---------------|
| SPH [62]             | 2003 | voxel | 68.2%         |
| LFD [63]             | 2003 | view  | 75.5%         |
| T-L [64]             | 2016 | view  | 74.4%         |
| VConv-DAE [65]       | 2016 | voxel | 75.5%         |
| 3D-GAN [66]          | 2016 | voxel | 83.3%         |
| Latent-GAN [16]      | 2018 | point | 85.7%         |
| MRTNet [15]          | 2018 | point | 86.4%         |
| SO-Net [67]          | 2018 | point | 87.3%         |
| FoldingNet [68]      | 2018 | point | 88.4%         |
| MAP-VAE [69]         | 2019 | point | 88.4%         |
| VIP-GAN [70]         | 2019 | view  | 90.2%         |
| 3D-PointCapsNet [71] | 2019 | point | 88.9%         |
| Jigsaw3D [72]        | 2019 | point | 90.6%         |
| Rotation3D [73]      | 2020 | point | 90.7%         |
| CMCV [74]            | 2021 | point | 89.8%         |
| MID-FC [75]          | 2021 | point | 90.3%         |
| GSIR [76]            | 2021 | point | 90.4%         |
| PSG-Net [77]         | 2021 | point | 90.9%         |
| STRL [17]            | 2021 | point | 90.9%         |
| ParAE [78]           | 2021 | point | 91.6%         |
| Point-BERT [79]      | 2022 | point | 88.6%         |
| CrossPoint [80]      | 2022 | point | 91.2%         |
| POS-BERT (our)       | 2022 | point | <b>92.1</b> % |

number of samples used for testing is 20. M is selected from 5 and 10, and N is selected from 10 and 20.

**SacnObjectNN** is a 3D point cloud classification dataset derived from real-world scanned data. It contains 2902 point clouds from 15 categories. Due to the noise of occlusion, rotation and background, it is more difficult to classify. Following Yu et al. [40], we selected three variant datasets to conduct experiments, including OBJ-BG, OBJ-ONLY, and PB-T50-RS.

**ShapeNetPart** contains 16811 objects from 16 categories. Each object consists of 2 to 6 parts with total of 50 distinct parts among all categories. Following Yu et al. [40], we randomly select 2048 points as input.

![](_page_7_Figure_5.jpeg)

Figure 4: Visualization of self-supervised features on ModelNet40.

# 5 Experiment

## 5.1 Linear SVM Classification

Linear SVM classification task has become a classic task to evaluate self-supervised point cloud representation learning. This experiment was designed to directly verify that our POS-BERT has learned better representation. To make a fair comparison with previous studies, we followed the common settings used in previous work [24–26, 38], pre-trained the model on ShapeNet and tested it on the ModelNet40. We used our pre-training model to extract the features of each point cloud, then trained a simple linear Support Vector Machine (SVM) on the training set of ModelNet40, and finally tested the SVM on the ModelNet40 test set. We compared a series of competitive methods, including handcrafted descriptor methods, generation-based method, contrastive learning method,

Table 2: **Shape classification results fine-tuned on ModelNet40.** We report the classification accuracy (%).

| Category     | Method                | Input     | Acc(%) |
|--------------|-----------------------|-----------|--------|
|              | PointNet [6]          | point     | 89.2   |
|              | PointNet++ [21]       | point     | 90.5   |
|              | SO-Net [67]           | point     | 92.5   |
|              | PointCNN [7]          | point     | 92.2   |
|              | DGCNN [81]            | point     | 92.9   |
| From scratch | DensePoint [82]       | point     | 92.8   |
|              | RSCNN [83]            | point     | 92.9   |
|              | PTC [49]              | point     | 93.2   |
|              | PointTransformer [84] | point+nor | 93.7   |
|              | NPTC [49]             | point     | 91.0   |
|              | Transformer [85]      | point     | 91.4   |
|              | Transformer-OcCo [86] | point     | 92.10  |
|              | Point-BERT [85]       | point     | 93.16  |
| Pretrain     | POS-BERT              | point     | 93.56  |
| _            | Point-BERT* [85]      | point     | 93.76  |
|              | POS-BERT*             | point     | 93.80  |

and the method based on mask patch modeling. The results of all methods are summarized in Tab.1. The results of the comparison methods we reported adopt the best results in the original papers. As shown in Tab.1, our method outperforms all other methods by a large margin, including the latest method CrossPoint based on contrastive learning and ParAE based on generation model. More importantly, it can surpass Point-BERT, which is also based on MPM paradigm, by 3.5%. This result fully shows that our Momentum Encoder can provide more meaningful supervision representation for masked patches. Finally, it is worth mentioning that our linear classification results exceed some supervised point cloud networks, such as PointNet (89.7%) and PointNet++ (91.9%). For a more intuitive understanding of the performance of our model, we use t-SNE to map the self-supervised learn features to a 2D space, as shown in Fig.4. It can be observed that different categories are separated from each other. These experimental results demonstrate that our method can learn a better representation.

#### 5.2 Downstream Tasks

**3D Object Classification on Synthetic Data** To test whether POS-BERT can help boost downstream tasks. We first performed fine-tuning experiments on point cloud classification tasks using a pretraining model. Here, **From scratch** stands for training the model on ModelNet40 from randomly initialized network and **Pretrain** stands for pre-training the model on ShapeNet and then fine-tune the network on ModelNet40. We fine-tuned the classification network weights using different initialization methods on ModelNet40, and the final classification results were summarized in Tab.2. Tab.2 shows that the original transformer's accuracy in point cloud classification task is just 91.4 percent. The transformer's classification accuracy was greatly increased to 93.56 percent using our pre-training weights to initialize the network. To achieve a fair comparison with Point-BERT, we also use voting strategy during the test, and the voting results are annotated with \*. By comparison, we can see that our method outperforms OcCo and Point-BERT without voting by 1.4% and 0.4%, respectively. When using the voting strategy, even if the accuracy is already high, our method is slightly better than Point-BERT.

**Few-shot Classification** To demonstrate that our pre-training model can learn quickly from few-shot samples, we conduct experiment on the Few-shot ModelNet40 dataset. We experimented with four different settings, including, "5-way 10-shot", "5-way 20-shot", "10-way 10-shot" and "10-way 20-shot", way represents the number of categories and shot represents the number of samples per category. During the test, 20 samples not in the training set were selected for evaluation. We conducted 10 independent experiments under each different setting, and reported the mean and variance of 10 experiments. We compared with the current SOTA methods OcCo and Point-BERT, and the results are summarized in Tab.3. Our approach produces the best results on the Few-shot Classification task. Compared with baseline, the mean was increased by 8.6%, 3.7%, 8%, and 5.5%, respectively. The variance is almost halved. Compared with point-Bert, the mean increased by 1.8%, 0.8%, 1.6% and 2.2% respectively, and the variance was smaller. This completely demonstrates that POS-BERT has learned a universal representation suitable for quick knowledge transfer with limited data.

Table 3: **Few-shot classification results on ModelNet40.** We report the average accuracy (%) as well as the standard deviation over 10 independent experiments.

|                       | 5-v            | vay            | 10-way         |                |  |
|-----------------------|----------------|----------------|----------------|----------------|--|
|                       | 10-shot        | 20-shot        | 10-shot        | 20-shot        |  |
| DGCNN-rand [86]       | $31.6 \pm 2.8$ | $40.8 \pm 4.6$ | $19.9 \pm 2.1$ | $16.9 \pm 1.5$ |  |
| DGCNN-OcCo [86]       | $90.6 \pm 2.8$ | $92.5 \pm 1.9$ | $82.9 \pm 1.3$ | $86.5 \pm 2.2$ |  |
| Transformer-rand [85] | $87.8 \pm 5.2$ | $93.3 \pm 4.3$ | $84.6 \pm 5.5$ | $89.4 \pm 6.3$ |  |
| Transformer-OcCo [86] | $94.0 \pm 3.6$ | $95.9 \pm 2.3$ | $89.4 \pm 5.1$ | $92.4 \pm 4.6$ |  |
| Point-BERT [85]       | $94.6 \pm 3.1$ | $96.3 \pm 2.7$ | $91.0 \pm 5.4$ | $92.7 \pm 5.1$ |  |
| POS-BERT              | $96.4 \pm 1.9$ | $97.0 \pm 2.2$ | $92.6 \pm 4.0$ | $94.9 \pm 2.9$ |  |

Table 4: Classification results on the ScanObjectNN dataset. We report the accuracy (%) of three different settings.

| Methods               | OBJ-BG | OBJ-ONLY | PB-T50-RS |
|-----------------------|--------|----------|-----------|
| PointNet [6]          | 73.3   | 79.2     | 68.0      |
| SpiderCNN [87]        | 77.1   | 79.5     | 73.7      |
| PointNet++ [88]       | 82.3   | 84.3     | 77.9      |
| PointCNN [7]          | 86.1   | 85.5     | 78.5      |
| DGCNN [81]            | 82.8   | 86.2     | 78.1      |
| BGA-DGCNN [89]        | _      | _        | 79.7      |
| BGA-PN++ [89]         | _      | _        | 80.2      |
| SimpleView [90]       | _      | _        | 80.5      |
| Transformer [85]      | 79.86  | 80.55    | 77.24     |
| Transformer-OcCo [86] | 84.85  | 85.54    | 78.79     |
| Point-BERT [85]       | 87.43  | 88.12    | 83.07     |
| POS-BERT              | 90.88  | 90.88    | 83.21     |

**3D Object Classification on Real-world Data** In this experiment, we aim to explore whether the knowledge POS-BERT learns from ShapNet can be transferred to real-world data. We conduct experiments on three variants of ScanObjectNN [60] dataset, including OBJ-BG, OBJ-ONLY, and PB-T50-RS. We compare to several methods, including supervised methods using specific point cloud networks: PointNet, BGA-PN++, SimpleView, et al., as well as pre-training methods: OcCo, Point-BERT. The experimental results are summarized in Tab.4. It can be found from the table that our method obtains the best results. With OBG-BG and OBJ-ONLY, we have surpassed Point-BERT by 3.45% and 2.76%, respectively. We also outperform Point-BERT with the PB-T50-RS settings. The results of the experiments suggest that the knowledge learned by POS-BERT can easily transfer into real-world data.

**Part Segmentation** In this section, we explore how the pre-training model performs in the pre-point classification. We experimented on ShapeNetPart, a benchmark dataset commonly used in point cloud segmentation tasks. Compared with the classification task, the segmentation task needs to obtain the label of each point intensively. We compare it with the commonly used point cloud analysis networks and the most advanced self-supervised methods. The mean Intersection Over Union (mIOU) metric of various methods is reported in Tab.5. From the table, our method is significantly better than the most advanced method Point-BERT on  $mIoU_C$ . From a category perspective, we have exceeded other methods in most categories. These results show that our methods can also learn to distinguish details very well.

# 5.3 Ablation study

To demonstrate the effectiveness of our key modules, we conducted ablation study on the ModelNet40 Linear SVM classification task. We have designed four variants. The first variant uses a randomly initialized Transformer network to extract features directly without any pre-training, and then classifies them using SVM, which is defined as POS-BERT-Var1. The second variant, defined as POS-BERT-Var2, uses only masking patch modeling's pretext task for pre-training. The third variant uses the randomly initialized momentum encoder as the tokenizer to pre-training, which is defined as POS-BERT-Var3. The fourth variant, which uses only contrastive loss to train the point cloud transformer, is defined as POS-BERT-Var4. The results are summarized in Tab.6. From the table we can see that a fixed Momentum Encoder does not help the network train well. Pre-training with masking patch modeling alone is difficult to obtain high-level semantic information. The best results are obtained when masking patch modeling and contrastive learning work together.

Table 5: **Part segmentation results on the ShapeNetPart dataset.** We report the mean IoU across all instances  $mIoU_I$  (%), as well as the IoU (%) for each categories and mean IoU across all categories  $mIoU_C$  (%).

|                       |                   |           | aero   | bag   | cap  | car    | chair  | e-phone    | guitar | knife |
|-----------------------|-------------------|-----------|--------|-------|------|--------|--------|------------|--------|-------|
| Methods $mIoU_C$      | $\mathrm{mIoU}_I$ | lamp      | laptop | motor | mug  | pistol | rocket | skateboard | table  |       |
| PointNet [6]          | 80.4              | 83.7      | 83.4   | 78.7  | 82.5 | 74.9   | 89.6   | 73.0       | 91.5   | 85.9  |
| romunet [0]           | 00.4              |           | 80.8   | 95.3  | 65.2 | 93.0   | 81.2   | 57.9       | 72.8   | 80.6  |
| PointNet++ [88]       | 81.9              | 85.1      | 82.4   | 79.0  | 87.7 | 77.3   | 90.8   | 71.8       | 91.0   | 85.9  |
| Tomuset++ [66]        | 01.3              |           | 83.7   | 95.3  | 71.6 | 94.1   | 81.3   | 58.7       | 76.4   | 82.6  |
| DGCNN [81] 82.3       | 99.9              | 85.2      | 84.0   | 83.4  | 86.7 | 77.8   | 90.6   | 74.7       | 91.2   | 87.5  |
|                       | 02.3              |           | 82.8   | 95.7  | 66.3 | 94.9   | 81.1   | 63.5       | 74.5   | 82.6  |
| Transformer [85]      | 83.4              | 85.1      | 82.9   | 85.4  | 87.7 | 78.8   | 90.5   | 80.8       | 91.1   | 87.7  |
| Transformer [83]      | 03.4              |           | 85.3   | 95.6  | 73.9 | 94.9   | 83.5   | 61.2       | 74.9   | 80.6  |
| Transformer-OcCo [86] | 83.4              | 85.1      | 83.3   | 85.2  | 88.3 | 79.9   | 90.7   | 74.1       | 91.9   | 87.6  |
| Transformer-occo [60] | 00.4              |           | 84.7   | 95.4  | 75.5 | 94.4   | 84.1   | 63.1       | 75.7   | 80.8  |
| Point-BERT [85] 84.1  | 85.6              | 84.3      | 84.8   | 88.0  | 79.8 | 91.0   | 81.7   | 91.6       | 87.9   |       |
| Tollic-BERT [63]      | 04.1              | 65.0      | 85.2   | 95.6  | 75.6 | 94.7   | 84.3   | 63.4       | 76.3   | 81.5  |
| POS-BERT              | 84.2              | 84.2 86.0 | 84.9   | 86.4  | 87.4 | 81.0   | 91.3   | 78.4       | 92.0   | 88.2  |
| 103-DER1 54           | 04.2              |           | 85.0   | 95.5  | 76.0 | 94.9   | 84.7   | 63.9       | 75.9   | 82.1  |

Table 6: Ablation study. .

| Model Name    | MPM | GFC | Momentum Encoder | Acc(%). |
|---------------|-----|-----|------------------|---------|
| POS-BERT-Var1 |     |     |                  | 53.61   |
| POS-BERT-Var2 | ✓   |     |                  | 80.43   |
| POS-BERT-Var3 | ✓   | ✓   |                  | 79.05   |
| POS-BERT-Var4 |     | ✓   | ✓                | 91.29   |
| POS-BERT      | ✓   | ✓   | ✓                | 92.14   |

# 6 Conclusion

In this paper, we propose a one-stage point cloud pre-training method POS-BERT, which is simple, flexible and efficient. It uses momentum encoder as tokenizer to provide supervision for mask patch model pretext tasks, and joint training of momentum encoder and MPM tasks greatly simplifies the training steps and saves training costs. Experiments show that our method has the best ability to extract high-level semantic information in the Linear SVM classification task, and it improves significantly compared with Point-BERT. At the same time, many downstream tasks, including 3D object classification, few-shot classification, part segmentation, have achieved state-of-the-art performance.

# References

- [1] Y. Cui, R. Chen, W. Chu, L. Chen, D. Tian, Y. Li, and D. Cao, "Deep learning for image and point cloud fusion in autonomous driving: A review," *IEEE Transactions on Intelligent Transportation Systems*, pp. 1–18, 2021.
- [2] W. Liu, B. Lai, C. Wang, X. Bian, W. Yang, Y. Xia, X. Lin, S.-H. Lai, D. Weng, and J. Li, "Learning to match 2d images and 3d lidar point clouds for outdoor augmented reality," in *IEEE Conference on Virtual Reality and 3D User Interfaces Abstracts and Workshops (VRW)*, 2020, pp. 654–655.
- [3] Z. Wang, Y. Xu, Q. He, Z. Fang, G. Xu, and J. Fu, "Grasping pose estimation for scara robot based on deep learning of point cloud," *The International Journal of Advanced Manufacturing Technology*, vol. 108, no. 4, pp. 1217–1231, 2020.
- [4] P. Gao, M. Zheng, X. Wang, J. Dai, and H. Li, "Fast convergence of detr with spatially modulated co-attention," in *IEEE/CVF International Conference on Computer Vision*, 2021, pp. 3621–3630.
- [5] P. Gao, J. Lu, H. Li, R. Mottaghi et al., "Container: Context aggregation networks," Advances in Neural Information Processing Systems (NeurIPS), vol. 34, 2021.
- [6] C. R. Qi, H. Su, K. Mo, and L. J. Guibas, "Pointnet: Deep learning on point sets for 3d classification and segmentation," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 652–660.

- [7] Y. Li, R. Bu, M. Sun, W. Wu, X. Di, and B. Chen, "Pointenn: Convolution on x-transformed points," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, pp. 820–830, 2018
- [8] T. Xiang, C. Zhang, Y. Song, J. Yu, and W. Cai, "Walk in the cloud: Learning curves for point clouds shape analysis," *arXiv preprint arXiv:2105.01288*, 2021.
- [9] K. Fu, S. Liu, X. Luo, and M. Wang, "Robust point cloud registration framework based on deep graph matching," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8893–8902.
- [10] M. Gadelha, R. Wang, and S. Maji, "Multiresolution tree networks for 3d point cloud processing," in *European Conference on Computer Vision (ECCV)*, 2018, pp. 103–118.
- [11] I. J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial nets," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2014, pp. 2672–2680.
- [12] P. Achlioptas, O. Diamanti, I. Mitliagkas, and L. Guibas, "Learning representations and generative models for 3d point clouds," in *International Conference on Machine Learning (ICML)*, 2018, pp. 40–49.
- [13] Y. Rao, J. Lu, and J. Zhou, "Global-local bidirectional reasoning for unsupervised representation learning of 3d point clouds," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 5376–5385.
- [14] S. Huang, Y. Xie, S.-C. Zhu, and Y. Zhu, "Spatio-temporal self-supervised representation learning for 3d point clouds," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 6535–6545.
- [15] M. Gadelha, R. Wang, and S. Maji, "Multiresolution tree networks for 3d point cloud processing," in *European Conference on Computer Vision (ECCV)*, 2018, pp. 103–118.
- [16] P. Achlioptas, O. Diamanti, I. Mitliagkas, and L. Guibas, "Learning representations and generative models for 3d point clouds," in *International Conference on Machine Learning (ICML)*, 2018, pp. 40–49.
- [17] S. Huang, Y. Xie, S.-C. Zhu, and Y. Zhu, "Spatio-temporal self-supervised representation learning for 3d point clouds," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 6535–6545.
- [18] J. Sauder and B. Sievers, "Self-supervised deep learning on point clouds by reconstructing space," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 32, 2019.
- [19] H. Wang, Q. Liu, X. Yue, J. Lasenby, and M. Kusner, "Pre-training by completing point clouds," 2020.
- [20] O. Poursaeed, T. Jiang, H. Qiao, N. Xu, and V. G. Kim, "Self-supervised learning of point clouds via orientation estimation," in *International Conference on 3D Vision (3DV)*. IEEE, 2020, pp. 1018–1028.
- [21] X. Chen, H. Fan, R. Girshick, and K. He, "Improved baselines with momentum contrastive learning," *arXiv preprint arXiv:2003.04297*, 2020.
- [22] X. Chen, S. Xie, and K. He, "An empirical study of training self-supervised visual transformers," *arXiv e-prints*, 2021.
- [23] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. H. Richemond, E. Buchatskaya, C. Doersch, B. A. Pires, Z. D. Guo, M. G. Azar *et al.*, "Bootstrap your own latent: A new approach to self-supervised learning," *arXiv preprint arXiv:2006.07733*, 2020.
- [24] L. Jing, L. Zhang, and Y. Tian, "Self-supervised feature learning by cross-modality and cross-view correspondences," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR), 2021, pp. 1581–1591.

- [25] M. Afham, I. Dissanayake, D. Dissanayake, A. Dharmasiri, K. Thilakarathna, and R. Rodrigo, "Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding," IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [26] P.-S. Wang, Y.-Q. Yang, Q.-F. Zou, Z. Wu, Y. Liu, and X. Tong, "Unsupervised 3d learning for shape analysis via multiresolution instance discrimination," *ACM Trans. Graphic*, 2020.
- [27] S. Huang, Y. Xie, S.-C. Zhu, and Y. Zhu, "Spatio-temporal self-supervised representation learning for 3d point clouds," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 6535–6545.
- [28] K. Fukushima and S. Miyake, "Neocognitron: A self-organizing neural network model for a mechanism of visual pattern recognition," in *Competition and Cooperation in Neural Nets*. Springer, 1982, pp. 267–285.
- [29] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," *arXiv preprint arXiv:1312.6114*, 2013.
- [30] Y. Bengio, L. Yao, G. Alain, and P. Vincent, "Generalized denoising auto-encoders as generative models," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 26, 2013.
- [31] A. Sharma, O. Grau, and M. Fritz, "Vconv-dae: Deep volumetric shape learning without object labels," in *European Conference on Computer Vision (ECCV)*. Springer, 2016, pp. 236–250.
- [32] Y. Yang, C. Feng, Y. Shen, and D. Tian, "Foldingnet: Point cloud auto-encoder via deep grid deformation," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 206–215.
- [33] J. Li, B. M. Chen, and G. H. Lee, "So-net: Self-organizing network for point cloud analysis," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 9397–9406.
- [34] J. Yang, P. Ahn, D. Kim, H. Lee, and J. Kim, "Progressive seed generation auto-encoder for unsupervised point cloud learning," in *IEEE/CVF International Conference on Computer Vision* (*ICCV*), 2021, pp. 6413–6422.
- [35] J. Wu, C. Zhang, T. Xue, B. Freeman, and J. Tenenbaum, "Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 29, 2016.
- [36] Z. Han, M. Shang, Y.-S. Liu, and M. Zwicker, "View inter-prediction gan: Unsupervised representation learning for 3d shapes by learning global shape memories to support local view predictions," in *Conference on Artificial Intelligence (AAAI)*, vol. 33, no. 01, 2019, pp. 8376–8384.
- [37] B. Eckart, W. Yuan, C. Liu, and J. Kautz, "Self-supervised learning on 3d point clouds by learning discrete generative models," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8248–8257.
- [38] S. Xie, J. Gu, D. Guo, C. R. Qi, L. Guibas, and O. Litany, "Pointcontrast: Unsupervised pre-training for 3d point cloud understanding," in *European Conference on Computer Vision (ECCV)*. Springer, 2020, pp. 574–591.
- [39] J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar *et al.*, "Bootstrap your own latent-a new approach to self-supervised learning," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 33, pp. 21 271–21 284, 2020.
- [40] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, "Point-bert: Pre-training 3d point cloud transformers with masked point modeling," *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.
- [41] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.

- [42] R. Sennrich, B. Haddow, and A. Birch, "Neural machine translation of rare words with subword units," *arXiv preprint arXiv:1508.07909*, 2015.
- [43] J. T. Rolfe, "Discrete variational autoencoders," arXiv preprint arXiv:1609.02200, 2016.
- [44] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," *arXiv preprint arXiv:2010.11929*, 2020.
- [45] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, "Swin transformer: Hierarchical vision transformer using shifted windows," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 10012–10022.
- [46] P. Gao, Z. Jiang, H. You, P. Lu, S. C. Hoi, X. Wang, and H. Li, "Dynamic fusion with intra-and inter-modality attention flow for visual question answering," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 6639–6648.
- [47] R. Strudel, R. Garcia, I. Laptev, and C. Schmid, "Segmenter: Transformer for semantic segmentation," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 7262–7272.
- [48] Z. Zhang, X. Lu, G. Cao, Y. Yang, L. Jiao, and F. Liu, "Vit-yolo: Transformer-based yolo for object detection," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 2799–2808.
- [49] M.-H. Guo, J.-X. Cai, Z.-N. Liu, T.-J. Mu, R. R. Martin, and S.-M. Hu, "Pct: Point cloud transformer," *arXiv preprint arXiv:2012.09688*, 2020.
- [50] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, "Point transformer," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 16259–16268.
- [51] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "Bert: Pre-training of deep bidirectional transformers for language understanding," *arXiv preprint arXiv:1810.04805*, 2018.
- [52] H. Bao, L. Dong, and F. Wei, "Beit: Bert pre-training of image transformers," *International Conference on Learning Representations (ICLR)*, 2022.
- [53] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, "Masked autoencoders are scalable vision learners," *arXiv preprint arXiv:2111.06377*, 2021.
- [54] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum contrast for unsupervised visual representation learning," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 9729–9738.
- [55] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *International Conference on Machine Learning (ICML)*, 2020, pp. 1597–1607.
- [56] J. Zbontar, L. Jing, I. Misra, Y. LeCun, and S. Deny, "Barlow twins: Self-supervised learning via redundancy reduction," *arXiv preprint arXiv:2103.03230*, 2021.
- [57] I. Loshchilov and F. Hutter, "Fixing weight decay regularization in adam," arXiv preprint arXiv:1711.05101, 2017.
- [58] A. X. Chang, T. Funkhouser, L. Guibas, P. Hanrahan, Q. Huang, Z. Li, S. Savarese, M. Savva, S. Song, H. Su *et al.*, "Shapenet: An information-rich 3d model repository," *arXiv preprint arXiv:1512.03012*, 2015.
- [59] Z. Wu, S. Song, A. Khosla, F. Yu, L. Zhang, X. Tang, and J. Xiao, "3d shapenets: A deep representation for volumetric shapes," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1912–1920.
- [60] M. A. Uy, Q.-H. Pham, B.-S. Hua, T. Nguyen, and S.-K. Yeung, "Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1588–1597.

- [61] I. Armeni, O. Sener, A. R. Zamir, H. Jiang, I. Brilakis, M. Fischer, and S. Savarese, "3d semantic parsing of large-scale indoor spaces," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 1534–1543.
- [62] M. Kazhdan, T. Funkhouser, and S. Rusinkiewicz, "Rotation invariant spherical harmonic representation of 3 d shape descriptors," in *Symposium on Geometry Processing*, vol. 6, 2003, pp. 156–164.
- [63] D.-Y. Chen, X.-P. Tian, Y.-T. Shen, and M. Ouhyoung, "On visual similarity based 3d model retrieval," in *Computer Graphics Forum*, vol. 22, no. 3, 2003, pp. 223–232.
- [64] R. Girdhar, D. F. Fouhey, M. Rodriguez, and A. Gupta, "Learning a predictable and generative vector representation for objects," in *European Conference on Computer Vision (ECCV)*, 2016, pp. 484–499.
- [65] A. Sharma, O. Grau, and M. Fritz, "Vconv-dae: Deep volumetric shape learning without object labels," in *European Conference on Computer Vision (ECCV)*. Springer, 2016, pp. 236–250.
- [66] J. Wu, C. Zhang, T. Xue, W. T. Freeman, and J. B. Tenenbaum, "Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2016, pp. 82–90.
- [67] J. Li, B. M. Chen, and G. H. Lee, "So-net: Self-organizing network for point cloud analysis," in *IEEE/CVF conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 9397–9406.
- [68] Y. Yang, C. Feng, Y. Shen, and D. Tian, "Foldingnet: Point cloud auto-encoder via deep grid deformation," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 206–215.
- [69] Z. Han, X. Wang, Y.-S. Liu, and M. Zwicker, "Multi-angle point cloud-vae: Unsupervised feature learning for 3d point clouds from multiple angles by joint self-reconstruction and halfto-half prediction," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 10441–10450.
- [70] Z. Han, M. Shang, Y.-S. Liu, and M. Zwicker, "View inter-prediction gan: Unsupervised representation learning for 3d shapes by learning global shape memories to support local view predictions," in *Conference on Artificial Intelligence (AAAI)*, vol. 33, no. 01, 2019, pp. 8376–8384.
- [71] Y. Zhao, T. Birdal, H. Deng, and F. Tombari, "3d point capsule networks," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 1009–1018.
- [72] J. Sauder and B. Sievers, "Self-supervised deep learning on point clouds by reconstructing space," Advances in Neural Information Processing Systems (NeurIPS), vol. 32, pp. 12962–12972, 2019.
- [73] O. Poursaeed, T. Jiang, H. Qiao, N. Xu, and V. G. Kim, "Self-supervised learning of point clouds via orientation estimation," in *International Conference on 3D Vision (3DV)*, 2020, pp. 1018–1028.
- [74] L. Jing, L. Zhang, and Y. Tian, "Self-supervised feature learning by cross-modality and cross-view correspondences," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition* (CVPR), 2021, pp. 1581–1591.
- [75] P.-S. Wang, Y.-Q. Yang, Q.-F. Zou, Z. Wu, Y. Liu, and X. Tong, "Unsupervised 3d learning for shape analysis via multiresolution instance discrimination," in *Conference on Artificial Intelligence (AAAI)*, vol. 35, no. 4, 2021, pp. 2773–2781.
- [76] H. Chen, S. Luo, X. Gao, and W. Hu, "Unsupervised learning of geometric sampling invariant representations for 3d point clouds," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 893–903.

- [77] J. Yang, P. Ahn, D. Kim, H. Lee, and J. Kim, "Progressive seed generation auto-encoder for unsupervised point cloud learning," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 6413–6422.
- [78] B. Eckart, W. Yuan, C. Liu, and J. Kautz, "Self-supervised learning on 3d point clouds by learning discrete generative models," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8248–8257.
- [79] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, "Point-bert: Pre-training 3d point cloud transformers with masked point modeling," 2022.
- [80] M. Afham, I. Dissanayake, D. Dissanayake, A. Dharmasiri, K. Thilakarathna, and R. Rodrigo, "Crosspoint: Self-supervised cross-modal contrastive learning for 3d point cloud understanding," 2022.
- [81] Y. Wang, Y. Sun, Z. Liu, S. E. Sarma, M. M. Bronstein, and J. M. Solomon, "Dynamic graph cnn for learning on point clouds," *Acm Transactions On Graphics (TOG)*, vol. 38, no. 5, pp. 1–12, 2019.
- [82] Y. Liu, B. Fan, G. Meng, J. Lu, S. Xiang, and C. Pan, "Densepoint: Learning densely contextual representation for efficient point cloud processing," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 5239–5248.
- [83] Y. Liu, B. Fan, S. Xiang, and C. Pan, "Relation-shape convolutional neural network for point cloud analysis," in *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2019, pp. 8895–8904.
- [84] H. Zhao, L. Jiang, J. Jia, P. H. Torr, and V. Koltun, "Point transformer," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 16259–16268.
- [85] X. Yu, L. Tang, Y. Rao, T. Huang, J. Zhou, and J. Lu, "Point-bert: Pre-training 3d point cloud transformers with masked point modeling," 2022.
- [86] H. Wang, Q. Liu, X. Yue, J. Lasenby, and M. J. Kusner, "Unsupervised point cloud pre-training via occlusion completion," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 9782–9792.
- [87] Y. Xu, T. Fan, M. Xu, L. Zeng, and Y. Qiao, "Spidercnn: Deep learning on point sets with parameterized convolutional filters," in *European Conference on Computer Vision (ECCV)*, 2018, pp. 87–102.
- [88] C. R. Qi, L. Yi, H. Su, and L. J. Guibas, "Pointnet++: Deep hierarchical feature learning on point sets in a metric space," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017.
- [89] M. A. Uy, Q.-H. Pham, B.-S. Hua, T. Nguyen, and S.-K. Yeung, "Revisiting point cloud classification: A new benchmark dataset and classification model on real-world data," in *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019, pp. 1588–1597.
- [90] A. Goyal, H. Law, B. Liu, A. Newell, and J. Deng, "Revisiting point cloud shape classification with a simple and effective baseline," in *International Conference on Machine Learning (ICML)*. PMLR, 2021, pp. 3809–3820.