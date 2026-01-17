# **Towards VQA Models That Can Read**

Amanpreet Singh<sup>1</sup>, Vivek Natarajan, Meet Shah<sup>1</sup>, Yu Jiang<sup>1</sup>, Xinlei Chen<sup>1</sup>, Dhruv Batra<sup>1,2</sup>, Devi Parikh<sup>1,2</sup>, Marcus Rohrbach<sup>1</sup>

<sup>1</sup>Facebook AI Research, <sup>2</sup>Georgia Institute of Technology

https://textvqa.org

#### **Abstract**

Studies have shown that a dominant class of questions asked by visually impaired users on images of their surroundings involves reading text in the image. But today's VQA models can not read! Our paper takes a first step towards addressing this problem. First, we introduce a new "TextVQA" dataset to facilitate progress on this important problem. Existing datasets either have a small proportion of questions about text (e.g., the VQA dataset) or are too small (e.g., the VizWiz dataset). TextVOA contains 45,336 questions on 28,408 images that require reasoning about text to answer. Second, we introduce a novel model architecture that reads text in the image, reasons about it in the context of the image and the question, and predicts an answer which might be a deduction based on the text and the image or is composed of the strings found in the image. Consequently, we call our approach Look, Read, Reason & Answer (LoRRA)<sup>1</sup>. We show that LoRRA outperforms existing state-of-the-art VQA models on our TextVQA dataset. We find that the gap between human performance and machine performance is significantly larger on TextVQA than on VQA 2.0, suggesting that TextVQA is well-suited to benchmark progress along directions complementary to VQA 2.0.

#### 1. Introduction

The focus of this paper is endowing Visual Question Answering (VQA) models a new capability – the ability to *read text in images and answer questions* by reasoning over the text and other visual content.

VQA has witnessed tremendous progress. But today's VQA models fail catastrophically on questions requiring reading!<sup>2</sup> This is ironic because these are *exactly* the ques-

![](_page_0_Figure_12.jpeg)

Figure 1: Examples from our TextVQA dataset. TextVQA questions require VQA models to understand text embedded in the images to answer them correctly. Ground truth answers are shown in green and the answers predicted by a state-of-the-art VQA model (Pythia [17]) are shown in red. Clearly, today's VQA models fail at answering questions that involve reading and reasoning about text in images.

tions visually-impaired users frequently ask of their assistive devices. Specifically, the VizWiz study [5] found that up to 21% of these questions involve reading and reasoning about the text captured in the images of a user's surroundings – 'what temperature is my oven set to?', 'what denomination is this bill?'.

Consider the question in Fig. 1(a) – 'What does it say near the star on the tail of the plane?' from the TextVQA dataset. With a few notable exceptions, today's state-of-

<sup>&</sup>lt;sup>1</sup>Code is available at https://github.com/facebookresearch/pythia

<sup>&</sup>lt;sup>2</sup>All top entries in the CVPR VQA Challenges (2016-18) struggle to answer questions in category requiring reading correctly.

art VQA models are predominantly monolithic deep neural networks (without any specialized components). Consider what we are asking such models to learn; for answering these questions, the model must learn to

- realize when the question is about text ('What ... say?'),
- detect image regions containing text ('15:20', '500'),
- convert pixel representations of these regions (convolutional features) to symbols ('15:20') or textual representations (semantic word-embeddings),
- jointly reason about detected text and visual content, *e.g.* resolving spatial or other visual reference relations (*'tail of the plane ... on the back'*) to focus on the correct regions.
- finally, decide if the detected text needs to be 'copypasted' as the answer (e.g. '16' in Fig. 1 (c)) or if the detected text informs the model about an answer in the answer space (e.g. answering 'jet', in Fig. 1(a)).

When laid out like that, it is perhaps unsurprising why today's models have not been able to make progress on questions requiring reading and reasoning about text in the images – simply put, despite all the strengths of deep learning, it seems hopelessly implausible that all of the above skills will simply *emerge* in a monolithic network all from the distant supervision of VQA accuracy.

Fortunately, we can do more than just hope. Optical Character Recognition (OCR) is a mature sub-field of computer vision. A key thesis of our work is the following – we should bake in inductive biases and specialized components (*e.g.* OCR) into models to endow them with the different skills (*e.g.* reading, reasoning) required by the allencompassing task of VQA.

Specifically, we propose a new VOA model that includes OCR as a module. We call it Look, Read, Reason & Answer (LoRRA). Our model architecture incorporates the regions (bounding boxes) in the image containing text as entities to attend over (in addition to object proposals). It also incorporates the actual text recognized in these regions (e.g. '15:20') as information (in addition to visual features) that the model learns to reason over. Finally, our model includes a mechanism to decide if the answer produced should be 'copied' over from the OCR output (in more of a generation or slot-filling flavor), or should be deduced from the text (as in a standard discriminative prediction paradigm popular among existing VQA models). Our model learns this mechanism end-to-end. While currently limited in scope to OCR, our model is as an initial step towards endowing VQA models with the ability to reason over unstructured sources of external knowledge (in this case text found in a test image) and accommodate multiple streams of information flow (in this case predicting an answer from a predetermined vocabulary or generating an answer via copy).

One reason why there has been limited progress on VQA models that can read and reason about text in images is because such questions, while being a dominant category in

real applications for aiding visually impaired users [5], are infrequent in the standard VQA datasets [3, 10, 51] because they were not collected in the settings that mimic those of visually impaired users. While the VizWiz dataset [13] does contain data collected from visually impaired users, the effective size of the dataset is small due to 58% of the questions being "unanswerable". This makes it challenging to study the problem systematically, train effective models, or even draw sufficient attention to this important skill that current VQA models lack.

To this end, we introduce the TextVQA dataset. It contains 45,336 questions asked by (sighted) humans on 28,408 images from the Open Images dataset [27] from categories that tend to contain text *e.g.* "billboard", "traffic sign", "whiteboard". Questions in the dataset require reading and reasoning about text in the image. Each question-image pair has 10 ground truth answers provided by humans.

Models that do well on this dataset will not only need to parse the image and the question as in traditional VQA, but also read the text in the image, identify which of the text might be relevant to the question, and recognize whether a subset of the detected text can directly be the answer (e.g., in the case of 'what temperature is my oven set to?') or additional reasoning is required on the detected text to answer the question (e.g., 'which team is winning?').

Overall, our contributions are:

- We introduce a novel dataset (TextVQA) containing questions which require the model to read and reason about the text in the image to be answered.
- We propose *Look*, *Read*, *Reason & Answer* (LoRRA): a novel model architecture which explicitly reasons over the outputs from an OCR system when answering questions.
- LoRRA outperforms existing state-of-the-art VQA models on our TextVQA as well as VQA 2.0 dataset.

## 2. Related work

Visual Question Answering. VQA has seen numerous advances and new datasets since the first large-scale VQA dataset was introduced by Antol et al. [3]. This dataset was larger, more natural, and more varied than earlier VQA datasets such as DAQUAR [31] or COCO-QA [38] but had linguistic priors which were exploited by models to answer questions without sufficient visual grounding. This issue was addressed by Goyal et al. [10] by adding complementary triplets  $(I_c, q, a_c)$  for each original triplet  $(I_o, q, a_o)$ where image  $I_c$  is similar to image  $I_o$  but the answer for the given question q changes from  $a_o$  to  $a_c$ . To study visual reasoning independent of language, non-photo-realistic VQA datasets have been introduced such as CLEVR [18], NLVR [42] and FigureQA [21]. Wang et al. [45] introduced a Fact-Based VQA dataset which explicitly requires external knowledge to answer a question.

**Text based VQA.** Several existing datasets study text detection and/or parsing in natural everyday scenes: COCO-

![](_page_2_Figure_0.jpeg)

Figure 2: **Overview of our approach** *Look*, *Read*, *Reason* & *Answer* (**LoRRA**). Our approach looks at the image, reads its text, reasons about the image and text content and then answers, either with an answer a from the fixed answer vocabulary or by selecting one of the OCR strings s. Dashed lines indicate components that are not jointly-trained. The answer cubes on the right with darker color have more attention weight. The OCR token "20" has the highest attention weight in the example.

Text [43], Street-View text [44] IIIT-5k [33] and ICDAR 2015 [22]. These do not involve answering questions about the images or reasoning about the text. DVOA [20] assesses automatic bar-chart understanding by training models to answer questions about graphs and plots. The Multi-Output Model (MOM) introduced in DVQA uses an OCR module to read chart specific content. Textbook QA (TQA) [24] considers the task of answering questions from middle-school textbooks, which often require understanding and reasoning about text and diagrams. Similarly, AI2D [23] contains diagram based multiple-choice questions. MemexQA [16] introduces a VQA task which involves reasoning about the time and date at which a photo/video was taken, but this information is structured and is part of the meta data. Note that these works all require reasoning about text to answer questions, but in narrow domains (bar charts, textbook diagrams, etc.). The focus of our work is to reason and answer questions about text in natural everyday scenes.

Visual Representations for VQA Models. VQA models typically use some variant of attention to get a representation of the image that is relevant for answering the given question [2, 7, 30, 47, 48, 51, 17]. The object region proposals and the associated features are generated by using a detection network which are then spatially attended to and conditioned on a question representation. In this work, we extend the representations that a VQA model reasons over. Specifically, in addition to attending over object proposals, our model also attends over the regions where text is detected

**Copy Mechanism.** A core component of our proposed model is its ability to decide whether the answer to a question should be an OCR token detected in the image, or if the OCR tokens should only inform about the answer to the question. The former is implemented as a "copy mecha-

nism" – a learned slot filling approach. Our copy mechanism is based on a series of works on the pointer generator networks [11, 39, 32, 12, 34]. A copy mechanism provides networks the ability to generate out-of-vocabulary words by pointing at a word in context and then copying it as the answer. This approach has been used for a variety of tasks in NLP such as summarization [11, 34, 39], question answering [46], language modelling [32], neural machine translation [12], and dialog [37].

# 3. LoRRA: Look, Read, Reason & Answer

In this section, we introduce our novel model architecture to answer questions which require reading text in the image.

We assume we get an image v and a question q as the input, where the question consists of L words  $w_1, w_2, \ldots, w_L$ . At a high level, our model contains three components: (i) a **VQA component** to reason and infer about the answer based on the image v and the question q (Sec 3.3); (ii) a **reading component** which allows our model to read the text in the image (Sec 3.2); and (iii) an **answering module** which either predicts from an answer space or points to the text read by the *reading component* (Sec. 3.3). The overall model is shown in Fig. 2. Note that, the OCR module and backbone VQA model can be any OCR model and any recent attention-based VQA model. Our approach is agnostic to the internal details of these components. We detail our exact implementation choices and hyper parameters in Sec. 3.4.

## 3.1. VQA Component

Similar to many VQA models [7, 17], we first embed the question words  $w_1, w_2, \ldots, w_L$  of the question q with a pre-trained embedding function (e.g. GloVe [36]) and then encode the resultant word embeddings iteratively with a re-

current network (e.g. LSTM [15]) to produce a question embedding  $f_Q(q)$ . For images, the visual features are represented as spatial features, either in the form of grid-based convolutions and/or features extracted from the bounding box proposals [1]. We refer to these features as  $f_I(v)$  where  $f_I$  is the network which extracts the image representation. We use an attention mechanism  $f_A$  over the spatial features [4, 7], which predicts attentions based on the  $f_I(v)$  and  $f_Q(q)$  and gives a weighted average over the spatial features as the output.

We then combine the output with the question embedding. At a high level, the calculation of our VQA features  $f_{VQA}(v,q)$  can be written as:

$$f_{VQA}(v,q) = f_{comb}(f_A(f_I(v), f_Q(q)), f_Q(q))$$
 (1)

where  $f_{comb}$  is the combination module ( $\bigotimes$ ) in Fig. 2.

Assuming that we have a fixed answer space of  $a_1, \ldots, a_N$ , we use a feed-forward MLP  $f_c$  on the combined embedding  $f_{VQA}(v,q)$  to predict probabilities  $p_1, \ldots, p_N$  where the probability of  $a_i$  being the correct answer is  $p_i$ .

#### 3.2. Reading Component

To add the capability of reading text from an image, we rely on an OCR model which is not jointly trained with our system. We assume that the OCR model can read and return word tokens from an image, e.g. [6, 41]. The OCR model extracts M words  $s = s_1, s_2, ..., s_M$  from the image which are then embedded with a pre-trained word embedding,  $f_O$ . Finally, we use the same architecture as VQA component to get combined OCR-question features,  $f_{OCR}$ . Specifically,

$$f_{OCR}(s,q) = f_{comb}(f_A(f_O(s), f_O(q)), f_O(q))$$
 (2)

This is visualized in Fig. 2. Note that the parameters of the functions  $f_A$  and  $f_{comb}$  are not shared with the VQA model component above but they have the same architecture, just with different input dimensions.

During weighted attention because the features are multiplied by weights and then averaged, the ordering information gets lost. To provide the answer module with the ordering information of the original OCR tokens, we concatenate the attention weights with the final weight-averaged features. This allows the answer module to know the original attention weights for each token in order.

#### 3.3. Answer Module

With a fixed answer space, the current VQA models are only able to predict fixed tokens which limits the generalization to out-of-vocabulary (OOV) words. As the text in images frequently contains words not seen at training time, it is hard to answer text-based questions based on a pre-defined answer space alone. To generalize to arbitrary text, we take

| VQA 2.0 Accuracy           |          |
|----------------------------|----------|
| Model                      | test-dev |
| BUTD [1]                   | 65.32    |
| Counter [50]               | 68.09    |
| BAN [25]                   | 69.08    |
| Pythia v0.1 [17]           | 68.49    |
| Pythia v0.3 (Ours)         | 68.71    |
| Pythia v0.3 + LoRRA (Ours) | 69.21    |
| VizWiz Accuracy            |          |
| Model                      | test     |
| BAN[25]                    | 51.40    |
| Pythia v0.3 (Ours)         | 54.72    |

Table 1: Single model VQA 2.0 and VizWiz performance in %. Our revised implementation of Pythia, v0.3, with LoRRA outperforms or is comparable to state-of-the-art on VQA 2.0.

inspiration from pointer networks which allow pointing to OOV words in context [11, 39, 32, 12, 34]. We extend our answer space through addition of a dynamic component which corresponds to M OCR tokens. The model now has to predict probabilities  $(p_1,\ldots,p_N,\ldots,p_{N+M})$  for N+M items in the answer space instead of the original N items.

We pick the index with the highest probability  $p_i$  as the index of our predicted answer. If the model predicts an index larger than N (i.e., among the last M tokens in answer space), we directly "copy" the corresponding OCR token as the predicted answer. Hence, our answering module can be thought of as "copy if you need" module which allows answering from the OOV words using the OCR tokens.

With all of the components, the final equation  $f_{LoRRA}$  for predicting the answer probabilities can be written as:

$$f_{LoRRA}(v, s, q) = f_{MLP}([f_{VQA}(v, q); f_{OCR}(s, q)])$$
 (3)

where [;] refers to concatenation and  $f_{MLP}$  is a two-layer feed-forward network which predicts the binary probabilities as logits for each answer. We opt for binary cross entropy using logits instead of calculating the probabilities through softmax as it allows us to handle cases where the answer can be in both the actual answer space and the OCR tokens without penalizing for predicting either one (the likelihood of logits is independent of each other). Note that if the model chooses to copy, it can only produce one of the OCR tokens as the predicted answer. 8.9% of the TextVQA questions can only be answered by combining multiple OCR tokens; we leave this as future work.

#### 3.4. Implementation Details

Our VQA component is based on the VQA 2018 challenge winner entry, Pythia v0.1 [17]. Our revised implementation, Pythia v0.3 [40], with slight changes in hyperparameters (*e.g.* size of question vocabulary, hidden dimensions) achieves state-of-the-art VQA accuracy for a single

![](_page_4_Picture_0.jpeg)

(a) Question: which processor Brand is featured on the top left? Answer: intel

![](_page_4_Picture_2.jpeg)

cravons? Answer: crayola

![](_page_4_Picture_4.jpeg)

![](_page_4_Picture_5.jpeg)

(c) Question: what is the name of the bose speaker style in these boxes? Answer: freestyle

![](_page_4_Picture_7.jpeg)

(d) Question: what is the license

number? Answer: cu58 ckk

Figure 3: Examples from TextVQA. Questions require inferring hidden characters ("intel"), handling rotated text ("crayola"), reasoning ("bose" versus "freestyle") and selecting among multiple texts in image "cu58 ckk" versus "western power distribution").

model (i.e. w/o ensemble) as shown in Tab. 1 on both VOA v2.0 dataset [9] and VizWiz dataset [13]. The revised design choices are discussed in [40].

Pythia [17, 40] is inspired from the detector-based bounding box prediction approach of the bottom-up topdown attention network [1] (VQA winner 2017), which in turn has a multi-modal attention mechanism similar to the VQA 2016 winner [7], which relied on grid-based features.

In Pythia, for spatial features  $f_I(v)$ , we rely on both grid and region based features for an image. The grid based features are obtained by average pooling 2048 $\mathcal{D}$  features from the res-5c block of a pre-trained ResNet-152 [14]. The region based features are extracted from the fc6 layer of an improved Faster-RCNN model [8] trained on the Visual Genome [28] objects and attributes as provided in [1]. During training, we fine-tune the fc7 weights as in [17].

We use pre-trained GloVe embeddings with a custom vocabulary (top  $\sim$ 77k question words in the VQA 2.0) for the question embedding [36]. The  $f_Q$  module passes GloVe embeddings to an LSTM [15] with self-attention [49] to generate question's sentence embedding. For OCR, we run the Rosetta OCR system [6] to provide us word strings  $s_1, ..., s_N$ . OCR tokens are first embedded using pretrained FastText embeddings  $(f_O)$  [19], which can generate word embeddings even for OOV tokens as explained in [19].

In  $f_A$ , the question embedding  $f_Q(q)$  is used to obtain the top-down *i.e.* task-specific attention on both  $f_O(s)$  OCR tokens features and  $f_I(v)$  image features. The features are then averaged based on the attention weights to get a final feature representation for both the OCR tokens and the image features. The final grid-level and region-based features are concatenated in case of the image features. For the OCR tokens, attention weights are concatenated to the final attended features as explained in Sec. 3.1. Finally, in  $f_{comb}(x, y)$ , the two feature embeddings in consideration are fused using element-wise/hadamard product, ⊗, of the features. The fused features from  $f_{OCR}(s,q)$  and  $f_{VQA}(v,q)$  are concatenated and passed through an MLP to produce logits from which word corresponding to maximum logit's index is selected as the answer.

![](_page_4_Picture_16.jpeg)

Figure 4: Distribution of first four words in questions in TextVQA. Most questions start with "what".

# **TextVQA**

To study the task of answering questions that require reading text in images, we collect a new dataset called TextVQA which is publicly available at https://textvqa.org. In this section, we start by describing how we selected the images that we use in TextVQA. We then explain our data collection pipeline for collecting the questions and the answers. Finally, we provide statistics and an analysis of the dataset. Snapshots of the annotation interface and detailed instructions can be found in the Appendix A.

#### 4.1. Images

We use Open Images v3 dataset [27] as the source of our images. In line with the goal of developing and studying VQA models that can reason about text, we are most interested in the images that contain text in them. Several categories in Open Images fit this criterion (e.g., billboard,

![](_page_5_Figure_0.jpeg)

![](_page_5_Figure_1.jpeg)

![](_page_5_Figure_2.jpeg)

(a) Number of questions with a particular question length. We see that the average question length (7.16) is higher in TextVQA compared to others.

(b) Number of majority answers with a particular length. Average answer length (1.7) is high and answer can contain long paragraph and quotes.

(c) Number of images with a particular number of OCR tokens. Average number of tokens is around 3.14. In TextVOA, 10x more images contain OCR text than the others.

![](_page_5_Figure_6.jpeg)

![](_page_5_Figure_7.jpeg)

![](_page_5_Figure_8.jpeg)

(d) Top 15 most occurring questions in (e) Total occurrences for 500 most common TextVQA. Most of the top questions start with "what".

questions among 23184 unique questions with markers for particular ranks.

(f) Similar to 5e, plot shows total occurrences for 500 most common majority answers with markers for particular ranks.

Figure 5: Question, Answer and OCR statistics for TextVQA. We show comparisons with VQA 2.0 [10] and VizWiz [13].

![](_page_5_Picture_13.jpeg)

![](_page_5_Picture_14.jpeg)

Figure 6: (Left) Wordcloud for majority answers in TextVQA. Frequently occurring answers include yes, brand names, "stop" and city names. (Right) Wordcloud for OCR tokens predicted **by Rosetta.** Note the overlap with answers on brand names (lg), cities (london) and verbs (stop).

traffic sign, whiteboard). To automate this process of identifying categories that tend to have images with text in them, we select 100 random images from each category (or all images if max images for that category is less than 100). We run a state-of-the-art OCR model Rosetta [6] on these images and compute the average number of OCR boxes in a category. The average number of OCR boxes per-category were normalized and used as per-category weights for sampling the images from the categories.

We collect TextVOA's training and validation set from Open Images' training set while test set is collected from Open Images' test set. We set up a three stage pipeline for crowd-sourcing our data. In the first stage, annotators were asked to identify images that did not contain text (using a forced-choice "yes"/"no" flag). Filtering those (and noisy data from annotators) out resulted in 28,408 images, which form the basis of our TextVQA dataset.

#### 4.2. Questions and Answers

In the second stage, we collect 1-2 questions for each image. For the first question, we show annotators an image and ask them to provide a question which requires reading the text in the image to answer. Specifically, they were told to 'Please ensure that answering the question requires reading of the text in the image. It is OK if the answer cannot be directly copied from the text but needs to be inferred or paraphrased.'

To collect a second question that is different from the first, we show annotators the first question and ask them to come up with a question that requires reasoning about the text in the image and has a different answer. Following VQA [3, 10] and VizWiz [13] datasets, we collect 10 answers for each question.

To ensure answer quality, we gave annotators instructions similar to those used in [3, 13] when collecting the VQA and VizWiz datasets. In addition, to catch any poor quality data from earlier steps, we give annotators these four options: (i) no text in image; (ii) not a question; (iii) answering the question doesn't require reading any text in image; and (iv) unanswerable, e.g. questions involving speculation about the meaning of text. We remove the questions where a majority of workers marked any of these flags. Additionally, we use hand-crafted questions for which we know the correct answers to identify and filter out bad annotators.

## 4.3. Statistics and Analysis

We first analyze the diversity of the questions that we have in the dataset. TextVQA contains 45,336 questions of which 37,912 (83.6%) are unique. Fig. 5a shows the distribution of question length along with the same statistics for the VQA 2.0 and the VizWiz datasets for reference. The average question length in TextVQA is 7.18 words which is higher than in VQA 2.0 (6.29) and VizWiz (6.68). We also note that the minimum question length is 3 words. Workers often form questions which are longer to disambiguate the response (e.g. specifying where exactly the text is in the image, see Fig. 3). Fig. 5d shows top 15 most occurring questions in the dataset with their count while Fig. 5e shows top 500 most occurring questions with their counts. We can see the uniform shift from common questions about "time" to questions occurring in specific situations like "team names". Fig. 4 shows sunburst for first 4 words in questions. We also observe that most questions involve reasoning about common things (e.g. figuring out brand names, cities and temperature). Questions often start with "what", frequently inquiring about "time", "names", "brands" or "authors".

In total there are 26,263 (49.2%) unique majority answers in TextVQA. The percentage of unique answers in TextVQA is quite high compared to VQA 2.0 (3.4%) and VizWiz (22.8%). All 10 annotators agree on the most common answer for 22.8% questions, while 3 or more annotators agree on most common answer for 97.9% questions. Fig. 6 (left) shows a word cloud plot for the majority answers in the dataset. The answer space is diverse and involves brand names, cities, people's names, time, and countries. Note that this diversity makes it difficult to have a fixed answer space - a challenge that most existing VQA datasets do not typically pose. The most common answer ("yes") is the majority answer for only 4.71% of the dataset and "yes/no" (majority answer) questions in total only make up 5.55% of the dataset. The average answer length is 1.58 (Fig. 5b). In a few occurrences where the text in the image is long (e.g., a quote or a paragraph), the answer length is high. Fig. 5f shows the frequency of top 500 most common answers. The gradual shift from brands to rare cities is depicted. We also note that the drop in TextVQA for number of answers of a particular answer length is more gradual than in VQA 2.0 which drops sharply after answer length 3.

Finally, we analyze the OCR tokens produced by the Rosetta OCR system [6]. In Fig. 5c, we plot number of images containing "x" number of OCR tokens. The peak between 4 and 5 shows that a lot of images in our dataset contain a good number of OCR tokens. In some cases, when the system is unable to detect text we get 0 tokens but those cases are restricted to  $\sim$ 1.5k images and we manually ver-

| Model        | Accuracy(%) |       |              |       | Accuracy(%) |       |
|--------------|-------------|-------|--------------|-------|-------------|-------|
|              | Val         | Test  | Model        | Vocab | Val         | Test  |
| Human        | 85.01       | 86.79 | Q            | LA    | 8.09        | 8.70  |
| OCR UB       | 37.12       | 36.52 | I            | LA    | 6.29        | 5.58  |
| LA UB        | 48.46       | 48.16 | Pythia (I+Q) | LA    | 13.04       | 14.0  |
| LA+OCR UB    | 67.56       | 68.24 | +0           | LA    | 18.35       | _     |
| Rand 100     | 0.22        | 0.20  | +O+C         | n/a   | 20.06       | -     |
| Wt. Rand 100 | 0.27        | 0.26  | +LoRRA       | LA    | 26.23       | _     |
| Majority Ans | 4.48        | 2.63  | +LoRRA       | SA    | 26.56       | 27.63 |
| Random OCI   | 7.72        | 9.12  | BAN (I+Q)    | LA    | 12.30       | _     |
| OCR Max      | 9.76        | 11.60 | +LoRRA       | SA    | 18.41       | _     |

Table 2: Evaluation on TextVQA. (Left) Accuracies for various heuristics baselines, which show that using OCR can help in achieving a good accuracy on TextVQA. LA+OCR UB refers to maximum accuracy achievable by models using LoRRA with our OCR tokens. (Right) Accuracies of our trained baselines and ablations in comparison with our model LoRRA. I denotes usage of image features, Q question features, O OCR tokens' features, and C copy mechanism. LA and SA refer to use of large and short vocabulary, respectively. Models with LoRRA outperform VQA SoTA (Pythia, BAN) and other baselines.

ified that the images actually do contain text. Fig. 6 (right) shows a word cloud of OCR tokens which shows they do contain common answers such as brand names and cities.

# 5. Experiments

We start by explaining our baselines including both heuristics and end-to-end trained models which we compare with LoRRA. We divide TextVQA into train, validation and test splits with size 34,602, 5,000, and 5,734, respectively. The TextVQA questions collected from Open Images v3's training set were randomly split into training and validation sets. There is no image overlap between the sets. For our approach, we use a vocabulary SA of size 3996, which contains answers which appear at least twice in the training set. For the baselines that don't use the copy mechanism, this vocabulary turns out to be too limited. To give them a fair shot, we also create a larger vocabulary (LA), containing the 8000 most frequent answers.

Upper Bounds and Heuristics. These mainly evaluate the upper bounds of what can be achieved using the OCR tokens detected by our OCR module and benchmark biases in the dataset. We test (i) OCR UB: the upper bound accuracy one can get if the answer can be build directly from OCR tokens (and can always be predicted correctly). OCR UB considers combinations of OCR tokens upto 4-grams. (ii) LA UB: the upper bound accuracy by always predicting the correct answer if it is present in LA. (iii) LA+OCR UB: (i) + (ii) - the upper bound accuracy one can get by predicting the correct answer if it is present in either LA or OCR tokens. (iv) Rand 100: the accuracy one can get by selecting a random answer from top 100

most frequent answers (v) **Wt. Rand 100:** the accuracy of baseline (iv) but with weighted random sampling using 100 most occurring tokens' frequencies as weights. (vi) **Majority Ans:** the accuracy of always predicting the majority answer "yes" (vii) **Random OCR token:** the accuracy of predicting a random OCR token from the OCR tokens detected in an image (viii) **OCR Max:** accuracy of always predicting the OCR token that is detected maximum times in the image (e.g., "crayola" in Fig. 3 (b)).

**Baselines.**<sup>3</sup> We make modifications to the implementation discussed in Sec. 3.4 for our baselines which include (i) **Question Only (Q):** we only use the  $f_Q(q)$  module of LoRRA to predict the answer and the rest of the features are zeroed out. (ii) **Image Only (I):** similar to **Q**, we only use image features  $f_I(v)$  to predict answers. **Q** and **I** do not have access to OCR tokens and predict from **LA**.

**Ablations.** We create several ablations of our approach LoRRA by using the reading component and answering module in conjunction and alternatively. (i) **I+Q:** This ablation is state-of-the-art for VQA 2.0 and doesn't use any kind of OCR features; we provide results on Pythia v0.3 and BAN [25] in Tab. 1; (ii) **Pythia+O:** Pythia with OCR features as input but no copy module or dynamic answer space; (iii) **Pythia+O+C:** (ii) with the copy mechanism but no fixed answer space *i.e.* the model can only predict from the OCR tokens. Abbreviation **C** is used when we add the copy module and dynamic answer space to a model.

Our full model corresponds to **LoRRA** attached to Pythia. We also compare **Pythia+LoRRA** with small answer space (**SA**) to a version with large answer space (**LA**). We also provide results on **LoRRA** attached to BAN [25].

Experimental Setup. We develop our model in PyTorch [35]. We use AdaMax optimizer [26] to perform backpropagation [29]. We predict logits and train using binary cross-entropy loss. We train all of our models for 24000 iterations with a batch size of 128 on 8 GPUs. We set the maximum question length to 14 and maximum number of OCR tokens to 50. We pad rest of the sequence if it is less than the maximum length. We use a learning rate of 5e-2 for all layers except the fc7 layers used for fine-tuning which are trained with 5e-3. We uniformly decrease the learning rate to 5e-4 after 14k iterations. We calculate val accuracy using VQA accuracy metric [10] at every 1000th iteration and use the model with the best validation accuracy to calculate the test accuracy. All validation accuracies are averaged over 5 runs with different seeds.

Results. Tab. 2 shows accuracies on both heuristics (left)

and trained baselines and models (right). Despite collecting open-ended answers from annotators, we find that human accuracy is 85.01%, consistent with that on VQA 2.0 [10] and VizWiz [13]. While the OCR system we used is not perfect, the upper-bound on the validation set that one can achieve by correctly predicting the answer using these OCR tokens is 37.12%. This is higher than our best model, suggesting room for improvement to reason about the OCR tokens. LA UB is quite high as they contain most commonly occurring questions. This accuracy on VQA 2.0 validation set with 3129 most common answers is 88.9% which suggests uniqueness of answers in TextVQA and limits of a fixed answer space. The difference between LoRRA and **LA+OCR UB** of 41% represents the room for improvement in modelling with current OCR tokens and LA. Majority answer ("yes") gets only 4.48% on test set. Random baselines, even the weighted one, are rarely correct. Random OCR token selection and maximum occurring OCR token selection (OCR Max) yields better accuracies compared to other heuristics baselines. Question only (**Q**) and Image only (I) baseline get 8.09% and 6.29% validation accuracies, respectively, which shows that the dataset does not have significant biases w.r.t. images and questions. I+Q models - Pythia v0.3 [40] and BAN [25], which are stateof-the-art on VQA 2.0 and VizWiz only achieve 13.04% and 12.3% validation accuracy on TextVQA, respectively. This demonstrates the inability of current VQA models to read and reason about text in images. A jump in accuracy to 18.35% is observed by feeding OCR tokens (**Pythia+O**) into the model; this supports the hypothesis that OCR tokens do help in predicting correct answers. Validation accuracy of 20.06 achieved by Pythia+O+C by only predicting answers from OCR tokens, further bolsters OCR importance as it is quite high compared to our Pythia v0.3 [40].

Our **LoRRA** (**LA**) with Pythia model outperforms all of the ablations. Finally, a slight modification which allows the model to predict from the OCR tokens more often by changing the fixed answer space **LA** to **SA** further improves performance. Validation accuracy for BAN [25] also improves to 18.41% by adding LoRRA. This suggests that LoRRA can help state-of-the-art VQA models to perform better on TextVQA.

While LoRRA can reach up to 26.56% accuracy on the TextVQA's validation set, there is a large gap to human performance of 85.01% and LA+OCR UB of 67.56%.

Interestingly, when adding LoRRA to Pythia it improves accuracy from 68.71 to 69.21 on VQA 2.0 [9] (see Tab. 1), indicating the ability of our model to also exploit reading and reasoning in this more general VQA benchmark.

# 6. Conclusion

We explore a specific skill in Visual Question Answering that is important for the applications involving aiding

<sup>&</sup>lt;sup>3</sup>Code is available at https://github.com/facebookresearch/pythia

visually impaired users – answering questions about everyday images that involve reading and reasoning about text in these images. We find that existing datasets do not support a systematic exploration of the research efforts towards this goal. To this end, we introduce the TextVQA dataset which contains questions which can only be answered by reading and reasoning about text in images. We also introduce Look, Read, Reason & Answer (LoRRA), a novel model architecture for answering questions based on text in images. LoRRA reads the text in images, reasons about it based on the provided question, and predicts an answer from a fixed vocabulary or the text found in the image. LoRRA is agnostic to the specifics of the underlying OCR and VQA modules. LoRRA significantly outperforms the current state-ofthe-art VQA models on TextVQA. Our OCR model, while mature, still fails at detecting text that is rotated, a bit unstructured (e.g., a scribble) or partially occluded. We believe TextVOA will encourage research both on improving text detection and recognition in unconstrained environments as well as on enabling the VQA models to read and reason about text in images.

#### References

- [1] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In *Computer Vision and Pattern Recognition (CVPR)*, 2018.
- [2] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In *Computer Vision and Pattern Recognition (CVPR)*, 2016.
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In *ICCV*, 2015.
- [4] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. In *International Conference on Learning Repre*sentations (ICLR), 2015.
- [5] Jeffrey P Bigham, Chandrika Jayant, Hanjie Ji, Greg Little, Andrew Miller, Robert C Miller, Robin Miller, Aubrey Tatarowicz, Brandyn White, Samual White, et al. Vizwiz: nearly real-time answers to visual questions. In *Proceedings of the 23nd annual ACM symposium on User interface software and technology*, pages 333–342. ACM, 2010.
- [6] Fedor Borisyuk, Albert Gordo, and Viswanath Sivakumar. Rosetta: Large scale system for text detection and recognition in images. In *Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*, pages 71–79. ACM, 2018.
- [7] Akira Fukui, Dong Huk Park, Daylen Yang, Anna Rohrbach, Trevor Darrell, and Marcus Rohrbach. Multimodal compact bilinear pooling for visual question answering and visual grounding. In EMNLP, 2016.
- [8] Ross Girshick, Ilija Radosavovic, Georgia Gkioxari, Piotr Dollár, and Kaiming He. Detectron. https://github.com/facebookresearch/detectron, 2018.

- [9] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.
- [10] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in Visual Question Answering. In *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017.
- [11] J Gu, Z Lu, H Li, and VOK Li. Incorporating copying mechanism in sequence-to-sequence learning. In *Annual Meeting of the Association for Computational Linguistics (ACL)*, 2016. Association for Computational Linguistics., 2016.
- [12] Caglar Gulcehre, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. Pointing the unknown words. In ACL, 2016.
- [13] Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. Vizwiz grand challenge: Answering visual questions from blind people. In Conference on Computer Vision and PatternRecognition (CVPR), 2017.
- [14] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 770–778, 2016.
- [15] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. *Neural computation*, 9(8):1735–1780, 1997.
- [16] Lu Jiang, Junwei Liang, Liangliang Cao, Yannis Kalantidis, Sachin Farfade, and Alexander G Hauptmann. Memexqa: Visual memex question answering. arXiv:1708.01336, 2017.
- [17] Yu Jiang, Vivek Natarajan, Xinlei Chen, Marcus Rohrbach, Dhruv Batra, and Devi Parikh. Pythia v0. 1: the winning entry to the vqa challenge 2018. arXiv preprint arXiv:1807.09956, 2018.
- [18] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pages 1988–1997. IEEE, 2017.
- [19] Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov. Bag of tricks for efficient text classification. In European Chapter of the Association for Computational Linguistics, 2017.
- [20] Kushal Kafle, Scott Cohen, Brian Price, and Christopher Kanan. Dvqa: Understanding data visualizations via question answering. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 5648– 5656, 2018.
- [21] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkinson, Akos Kadar, Adam Trischler, and Yoshua Bengio. Figureqa: An annotated figure dataset for visual reasoning. In *ICLR workshop track*, 2018.
- [22] Dimosthenis Karatzas, Lluis Gomez-Bigorda, Anguelos Nicolaou, Suman Ghosh, Andrew Bagdanov, Masakazu Iwamura, Jiri Matas, Lukas Neumann, Vijay Ramaseshan Chandrasekhar, Shijian Lu, et al. Icdar 2015 competition on robust reading. In *Document Analysis and Recognition (ICDAR)*,

- 2015 13th International Conference on, pages 1156–1160. IEEE, 2015.
- [23] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In *European Conference on Computer Vision*, pages 235–251. Springer, 2016.
- [24] Aniruddha Kembhavi, Min Joon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In *Computer Vision and Pattern Recognition (CVPR)*, volume 2, page 3, 2017.
- [25] Jin-Hwa Kim, Jaehyun Jun, and Byoung-Tak Zhang. Bilinear attention networks. In *Neural Information Processing* Systems, 2018.
- [26] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*, 2015.
- [27] Ivan Krasin, Tom Duerig, Neil Alldrin, Andreas Veit, Sami Abu-El-Haija, Serge Belongie, David Cai, Zheyun Feng, Vittorio Ferrari, Victor Gomes, et al. Openimages: A public dataset for large-scale multi-label and multi-class image classification. *Dataset available from https://github.* com/openimages, 2(6):7, 2016.
- [28] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. *IJCV*, 2017.
- [29] Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition. *Neural computation*, 1(4):541–551, 1989.
- [30] Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh. Hierarchical question-image co-attention for visual question answering. In *Advances In Neural Information Processing Systems*, pages 289–297, 2016.
- [31] Mateusz Malinowski and Mario Fritz. A multi-world approach to question answering about real-world scenes based on uncertain input. In *Advances in neural information processing systems*, pages 1682–1690, 2014.
- [32] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations (ICLR), 2017
- [33] Anand Mishra, Karteek Alahari, and CV Jawahar. Scene text recognition using higher order language priors. In BMVC-British Machine Vision Conference. BMVA, 2012.
- [34] Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. In *The SIGNLL Conference on Computational Natural Language Learning (CoNLL)*, 2016.
- [35] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer. Automatic differentiation in pytorch. NIPS AutoDiff Workshop, 2017.
- [36] Jeffrey Pennington, Richard Socher, and Christopher Manning. Glove: Global vectors for word representation. In

- EMNLP, 2014.
- [37] Dinesh Raghu, Nikhil Gupta, et al. Hierarchical pointer memory network for task oriented dialogue. arXiv preprint arXiv:1805.01216, 2018.
- [38] Mengye Ren, Ryan Kiros, and Richard Zemel. Exploring models and data for image question answering. In Advances in neural information processing systems, pages 2953–2961, 2015.
- [39] Abigail See, Peter J Liu, and Christopher D Manning. Get to the point: Summarization with pointer-generator networks. In Association for Computational Linguistics, 2017.
- [40] Amanpreet Singh, Vivek Natarajan, Yu Jiang, Xinlei Chen, Meet Shah, Marcus Rohrbach, Dhruv Batra, and Devi Parikh. Pythia-a platform for vision & language research. SysML Workshop, NeurIPS 2019, 2018.
- [41] Ray Smith. An overview of the tesseract ocr engine. In Document Analysis and Recognition, 2007. ICDAR 2007. Ninth International Conference on, volume 2, pages 629– 633. IEEE, 2007.
- [42] Alane Suhr, Mike Lewis, James Yeh, and Yoav Artzi. A corpus of natural language for visual reasoning. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, volume 2, pages 217–223, 2017.
- [43] Andreas Veit, Tomas Matera, Lukas Neumann, Jiri Matas, and Serge Belongie. Coco-text: Dataset and benchmark for text detection and recognition in natural images. arXiv preprint arXiv:1601.07140, 2016.
- [44] Kai Wang and Serge Belongie. Word spotting in the wild. In European Conference on Computer Vision, pages 591–604. Springer, 2010.
- [45] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton van den Hengel. Fvqa: Fact-based visual question answering. *IEEE transactions on pattern analysis and machine intelligence*, 2018.
- [46] Caiming Xiong, Victor Zhong, and Richard Socher. Dynamic coattention networks for question answering. In International Conference on Learning Representations (ICLR), 2016
- [47] Huijuan Xu and Kate Saenko. Ask, attend and answer: Exploring question-guided spatial attention for visual question answering. In *European Conference on Computer Vision*, pages 451–466. Springer, 2016.
- [48] Zichao Yang, Xiaodong He, Jianfeng Gao, Li Deng, and Alex Smola. Stacked attention networks for image question answering. In CVPR, 2016.
- [49] Zhou Yu, Jun Yu, Chenchao Xiang, Jianping Fan, and Dacheng Tao. Beyond bilinear: Generalized multimodal factorized high-order pooling for visual question answering. *IEEE Transactions on Neural Networks and Learning Systems*, 2018.
- [50] Yan Zhang, Jonathon Hare, and Adam Prügel-Bennett. Learning to count objects in natural images for visual question answering. In *International Conference on Learning Representations (ICLR)*, 2018.
- [51] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. Visual7w: Grounded question answering in images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4995–5004, 2016.

# A. OCR and Answer Space Analysis

We perform the following analysis on TextVQA's validation set. We find that 44.9% of LoRRA's predicted answers are from OCR tokens (i.e., using the copy mechanism). The remaining 55.1% of predicted answers are from the predetermined (short) answer vocabulary (SA). This shows that our approach does in fact rely heavily on what it reads in the image, and relies on its copy mechanism to generalize and produce answers that have never been seen or are rare in the training data. While predicting answers from OCR tokens, the model gets the entire answer string correct 27% of the time, and partially correct (i.e., matches one word in answer) 11% of the time. The percentage of partially correct answers indicates the possibility of getting better results by using n-grams of OCR tokens or spelling correction for improving OCR predictions. When predicting from the answer space, the model gets the answer correct 22.4% of the

We find that 30.6% of questions have their answers in OCR tokens. For these questions, LoRRA chooses to predict from OCR tokens 68% of the times and answers 57.5% of these correct. Similarly, 48% of questions have their answers in SA. For these questions, LoRRA chooses to predict from LA 66.75% of the times and gets 38% of these correct.

81% of the questions in TextVQA's validation set have images with 2 or more OCR tokens. Among these 4,645 questions, LoRRA chooses to copy from OCR tokens 49.7% of the time and gets 24.3% of these correct. This suggests that LoRRA doesn't randomly copy OCR token from a list of available tokens.

# B. TextVQA Examples and LoRRA Predictions

In Fig. 7, we show representative examples from our TextVQA dataset along with the predictions from Pythia+LoRRA. Each example shows the ground truth answer, the predictions from LoRRA, whether the answer prediction was from OCR tokens or the pre-determined answer space, and attention weights for each of the OCR tokens. The examples indicate the following points:

- The model is able to successfully answer questions about times, dates, brands, cities and places, and is often able to correctly spell them even if the OCR tokens had them misspelled (by picking an answer from the pre-determined answer space). See Fig. 7k (short hand's hour), Fig. 7g (birthday date), Fig. 7s (picking out city "london" from the large amount of text), Fig. 7o (samsung).
- The model is able to successfully answer questions involving colors and spatial reasoning. See Fig. 7e (player on the right), Fig. 7f (location of coin), Fig. 7c (location of banner). See Fig. 7q where the model needs to identify the

correct sign based on multiple colors, or Fig. 7r where the model needs to identify the correct sign in the red circle. Note that unlike most existing VQA models, the model does not seem to be biased toward "stop" for red signs. In Fig. 7a the model needs to predict the correct number based on spatial reasoning between the two choices 7 and 14.

- The model is also able to reason about basic sizes (less, greater, smallest) and shapes (circle). See Fig. 7k where the model needs to figure out which one is the shorter hand, or Fig. 7q where the model needs to figure out which one is the lowest measurement among four.
- The model often predicts an answer from the answer space as informed by OCR tokens. See Fig. 7k where the Pythia model (which doesn't use OCR) predicts 3, but our approach predicts 4 which is the correct answer.
- The model often answers questions about cities with "new york". See Fig. 7j where the model predicts New York instead of San Francisco. We have observed this bias in other city related questions as well.
- For yes/no questions, even though "yes" is the more common answer, the model does predict "no" frequently. See Fig. 7m, Fig. 7l.
- Sometimes when the answer is not in the answer space, but the partial answer is in OCR tokens, the model predicts the partial answer which is closest to the actual answer. See Fig. 7e where the model predicts "fly" instead of "fly emirates", or Fig. 7g where the model predicts only the birthday date "19", instead of "may 19". By construction our model can only copy a single OCR token, but our TextVQA dataset contains Q/A pairs which require copying multiple OCR tokens in the right order. Exploring this is an interesting direction for future work.
- The model sometimes gets seemingly simple questions wrong by predicting generic answers. See Fig. 7h where the model can't predict "embossed" even though it is in the detected OCR tokens, or see Fig. 7b where the model predicts most common letter "g" in the answer space instead of predicting based on "a-2" in the OCR tokens.
- The model has a strong dependency on the quality of OCR tokens produced. If the OCR module missed some text in the image, the model's output can be wrong. See Fig. 7i or Fig. 7p where the OCR tokens do not contain the ground truth answer or see Fig. 7u where the OCR system is unable to correctly read "irig" the second time.

#### C. Interface Screenshots

We show the three stages of the data collection pipeline in Fig. 8, Fig. 9, Fig. 10 and Fig. 11. Fig. 8 and Fig. 9

shows the introduction and first stage of our pipeline which is used to identify and remove images without text in them. Fig 10 shows the second stage of our pipeline which is used to collect questions on images with text. Finally, the third stage interface is shown in Fig. 11 which is used to collect the answer for a question about an image.

![](_page_12_Figure_0.jpeg)

![](_page_13_Figure_0.jpeg)

Figure 7: **TextVQA Examples and LoRRA's predictions on them.** We show multiple examples from TextVQA, ground truth answers, along with predictions from LoRRA, attention maps on OCR tokens and whether LoRRA predicted the answer from the OCR tokens or pre-determined answer space. Green, red, and blue boxes correspond to correct, incorrect, and partially correct answers, respectively. On the right side of each image, we show attention bars which depict attention weights (0-1) for each of the OCR tokens.

![](_page_13_Figure_2.jpeg)

Figure 8: Introduction page for our task.

![](_page_14_Figure_0.jpeg)

Figure 9: Text detection task. First stage of our data collection pipeline involves identifying and removing images without text.

![](_page_14_Figure_2.jpeg)

Figure 10: **Question task.** In the second stage, we ask workers to ask a question about an image whose answer requires reading text in the image. We provide instructions and rules to ensure that we get high quality questions.

![](_page_15_Figure_0.jpeg)

Figure 11: Answer task. In the third stage, we ask workers to answer a question about the image.