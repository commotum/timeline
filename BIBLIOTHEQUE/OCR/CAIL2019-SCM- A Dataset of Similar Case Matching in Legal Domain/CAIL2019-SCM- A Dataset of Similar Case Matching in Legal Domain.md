# CAIL2019-SCM: A Dataset of Similar Case Matching in Legal Domain

Chaojun Xiao<sup>1\*</sup> Haoxi Zhong<sup>1\*</sup> Zhipeng Guo<sup>1</sup> Cunchao Tu<sup>1</sup> Zhiyuan Liu<sup>1</sup> Maosong Sun<sup>1</sup> Tianyang Zhang<sup>2</sup> Xianpei Han<sup>3</sup> Zhen Hu<sup>4</sup> Heng Wang<sup>4</sup> Jianfeng Xu<sup>5</sup>

<sup>1</sup>Department of Computer Science and Technology, Tsinghua University, China
<sup>2</sup>Beijing Powerlaw Intelligent Technology Co., Ltd., China
<sup>3</sup>Institute of Software, Chinese Academy of Sciences, China
<sup>4</sup>China Justice Big Data Institute
<sup>5</sup>Supreme People Court, China

### **Abstract**

In this paper, we introduce CAIL2019-SCM. Chinese AI and Law 2019 Similar Case Matching dataset. CAIL2019-SCM contains 8,964 triplets of cases published by the Supreme People's Court of China. CAIL2019-SCM focuses on detecting similar cases, and the participants are required to check which two cases are more similar in the There are 711 teams who partictriplets. ipated in this year's competition, and the best team has reached a score of 71.88. We have also implemented several baselines to help researchers better understand this The dataset and more details can task. be found from https://github.com/china-ai-lawchallenge/CAIL2019/tree/master/scm.

#### 1 Introduction

Similar Case Matching (SCM) plays a major role in legal system, especially in common law legal system. The most similar cases in the past determine the judgment results of cases in common law systems. As a result, legal professionals often spend much time finding and judging similar cases to prove fairness in judgment. As automatically finding similar cases can benefit to the legal system, we select SCM as one of the tasks of CAIL2019.

Chinese AI and Law Challenge (CAIL) is a competition of applying artificial intelligence technology to legal tasks. The goal of the competition is to use AI to help the legal system. CAIL was first held in 2018, and the main task of CAIL2018 (Xiao et al., 2018; Zhong et al., 2018b) is predicting the judgment results from the fact description. The judgment results include the accusation, applicable articles, and the term of penalty. CAIL2019 contains three different tasks, including Legal Question-Answering, Legal Case Ele-

ment Prediction, and Similar Case Matching. Furthermore, we will focus on SCM in this paper.

More specifically, CAIL2019-SCM contains 8,964 triplets of legal documents. Every legal documents is collected from China Judgments Online<sup>1</sup>. In order to ensure the similarity of the cases in one triplet, all selected documents are related to Private Lending. Every document in the triplet contains the fact description. CAIL2019-SCM requires researchers to decide which two cases are more similar in a triplet. By detecting similar cases in triplets, we can apply this algorithm for ranking all documents to find the most similar document in the database. There are 247 teams who have participated CAIL2019-SCM, and the best team has reached a score of 71.88, which is about 20 points higher than the baseline. The results show that the existing methods have made great progress on this task, but there is still much room for improvement.

In other words, CAIL2019-SCM can benefit the research of legal case matching. Furthermore, there are several main challenges of CAIL2019-SCM: (1) The difference between documents may be small, and then it is hard to decide which two documents are more similar. Moreover, the similarity is defined by legal workers. We must utilize legal knowledge into this task rather than calculate similarity on the lexical level. (2) The length of the documents is quite long. Most documents contain more than 512 characters, and then it is hard for existing methods to capture document level information.

In the following parts, we will give more details about CAIL2019-SCM, including related works about SCM, the task definition, the construction of the dataset, and several experiments on the dataset.

<sup>\*</sup> indicates equal contribution.

<sup>1</sup>http://wenshu.court.gov.cn/

### 2 Related Work

### 2.1 Semantic Text Matching

SCM aims to measure the similarity between legal case documents. Essentially, it is an application of semantic text matching, which is central for many tasks in natural language processing, such as question answering, information retrieval, and natural language inference. Take information retrieval as an example, given a query and a database, a semantic matching model is required to judge the semantic similarity between the query and documents in the database. Moreover, the tasks related to semantic matching have attracted the attention of many researchers in recent decades.

Intuitively traditional approaches calculate word-to-word similarity with vector space model, e.g. term frequency-inverse document frequency (Wu et al., 2008), bag-of-words (Bilotti et al., 2007). However, due to the variety of words in different texts, these approaches achieve limited success in the task.

Recently, with the development of deep learning in natural language processing, researchers attempt to apply neural models to encode text into distributed representation. The Siamese structure (Bromley et al., 1994) for metric learning achieve great success and is widely applied (Amiri et al., 2016; Liu et al., 2018; Mueller and Thyagarajan, 2016; Neculoiu et al., 2016; Wan et al., 2016; He et al., 2015). Besides, there are many researchers put emphasis on integrating syntactic structure into semantic matching (Liu et al., 2018; Chen et al., 2017) and multi-level text matching with attention-aware representation (Duan et al., 2018; Tan et al., 2018; Yin et al., 2016).

Nevertheless, most previous studies are designed for identifying the relationship between two sentences with limited length.

# 2.2 Legal Intelligence

Researchers widely concern tasks for legal intelligence. Applying NLP techniques to solve a legal problem becomes more and more popular in recent years. Previous works (Kort, 1957; Keown, 1980; Lauderdale and Clark, 2012) focus on analyzing existing cases with mathematical tools. With the development of deep learning, more researchers pay much efforts on predicting the judgment result of legal cases (Luo et al., 2017; Hu et al., 2018; Zhong et al., 2018a; Chalkidis et al., 2019;

Jiang et al., 2018; Yang et al., 2019). Furthermore, there are many works on generating court views to interpret charge results (Ye et al., 2018), information extraction from legal text (Vacek and Schilder, 2017; Vacek et al., 2019), legal event detection (Yan et al., 2017), identifying applicable law articles (Liu et al., 2015; Liu and Hsieh, 2006) and legal question answering (Kim et al., 2015; Fawei et al., 2018).

Meanwhile, retrieving related legal documents with a query has been studied for decades and is a critical issue in applications of legal intelligence. Raghav et al. (2016) emphasize exploiting paragraph-level and citation information. Kano et al. (2017) and Zhong et al. (2018b) held a legal information extraction and entailment competition to promote progress in legal case retrieval.

# 3 Overview of Dataset

### 3.1 Task Definition

We first define the task of CAIL2019-SCM here. The input of CAIL2019-SCM is a triplet (A,B,C), where A,B,C are fact descriptions of three cases. Here we define a function sim which is used for measuring the similarity between two cases. Then the task of CAIL2019-SCM is to predict whether sim(A,B) > sim(A,C) or sim(A,C) > sim(A,B).

#### 3.2 Dataset Construction and Details

To ensure the quality of the dataset, we have several steps of constructing the dataset. First, we select many documents within the range of Private Lending. However, although all cases are related to Private Lending, they are still various so that many cases are not similar at all. If the cases in the triplets are not similar, it does not make sense to compare their similarities. To produce qualified triplets, we first annotated some crucial elements in Private Lending for each document. The elements include:

- The properties of lender and borrower, whether they are a natural person, a legal person, or some other organization.
- The type of guarantee, including no guarantee, guarantee, mortgage, pledge, and others.
- The usage of the loan, including personal life, family life, enterprise production and operation, crime, and others.

- The lending intention, including regular lending, transfer loan, and others.
- Conventional interest rate method, including no interest, simple interest, compound interest, unclear agreement, and others.
- Interest during the agreed period, including [0%, 24%], (24%, 36%],  $(36\%, \infty)$ , and others.
- Borrowing delivery form, including no lending, cash, bank transfer, online electronic remittance, bill, online loan platform, authorization to control a specific fund account, unknown or fuzzy, and others.
- Repayment form, including unpaid, partial repayment, cash, bank transfer, online electronic remittance, bill, unknown or fuzzy, and others.
- Loan agreement, including loan contract, or borrowing, "WeChat, SMS, phone or other chat records", receipt, irrigation, repayment commitment, guarantee, unknown or fuzzy and others.

After annotating these elements, we can assume that cases with similar elements are quite similar. So when we construct the triplets, we calculate the tf-idf similarity and elemental similarity between cases and select those similar cases to construct our dataset. We have constructed 8,964 triples in total by these methods, and the statistics can be found from Table 1. Then, legal professionals will annotate every triplet to see whether sim(A,B) > sim(A,C) or sim(A,B) < sim(A,C). Furthermore, to ensure the quality of annotation, every document and triplet is annotated by at least three legal professionals to reach an agreement.

| Туре        | Count |  |
|-------------|-------|--|
| Small Train | 500   |  |
| Small Test  | 326   |  |
| Large Train | 5,102 |  |
| Large Valid | 1,500 |  |
| Large Test  | 1,536 |  |
| Total       | 8,964 |  |

Table 1: The number of triplets in different stages of CAIL2019-SCM.

## 4 Experiments

To access the challenge of the similar cases matching task, we evaluate several baselines on our dataset. The experiment results show that even the state-of-the-art systems perform poorly in evaluating the similarity between different cases.

**Baselines.** All the baseline models are trained on *Large Train* and tested on *Large Valid* and *Large Test*. We adapt the Siamese framework (Bromley et al., 1994) to our scenario with different encoder, e.g. CNN (Kim, 2014), LSTM (Hochreiter and Schmidhuber, 1997), Bert (Devlin et al., 2019), used for encoding the legal documents. We will elaborate on the details of the framework in the following part.

Given the triplet of fact description,  $(D_A, D_B, D_C)$ , we first encode them into distributed vectors with the same encoder and then compute the similarity scores between the query case  $D_A$  and the candidate cases  $D_B$ ,  $D_C$  with a linear layer. Assume that each document D consisting of n words, i.e.  $D = \{w_1, w_2, ..., w_n\}$ .

For CNN/LSTM encoder, we first employ THU-LAC (Sun et al., 2016) for word segmentation and then transform each word into distributed representation  $X = \{x_1, x_2, ..., x_n\}$  with Glove (Pennington et al., 2014), where  $x_i \in \mathbb{R}^d, i = 1, 2, ..., n$  and d is the dimension of word embeddings. Next, the encoder layer and max pooling layer transform the embedding sequence X into features  $h \in \mathbb{R}^{d_h}$ , where  $d_h$  is the dimension of hidden vector. While for Bert encoder, we feed the document in character-level into the  $bert\_base\_chinese$  model to get the features h.

$$h_A = Encoder(D_A)$$

$$h_B = Encoder(D_B)$$

$$h_C = Encoder(D_C)$$
(1)

Afterward, we calculate the similarity with a linear layer with softmax activation.  $W \in \mathbb{R}^{d_h \times d_h}$  is a weight matrix to be learned.

$$s_{Aj} = \operatorname{softmax}(exp(h_A W h_j))$$
  

$$j = B, C$$
(2)

For the learning objective, we apply the binary cross-entropy loss function with ground-truth label *p*:

$$L(\theta) = \mathbb{E}[p\ln(s_{AB}) + (1-p)\ln(s_{AC})] \tag{3}$$

|           | Method                              | Valid                    | Test                     |
|-----------|-------------------------------------|--------------------------|--------------------------|
| baselines | CNN<br>LSTM<br>BERT                 | 62.27<br>62.00<br>61.93  | 69.53<br>68.00<br>67.32  |
| Teams     | AlphaCourt<br>backward<br>11.2 yuan | <b>70.07</b> 67.73 66.73 | <b>72.66</b> 71.81 72.07 |

Table 2: Results of baselines and scores of top 3 participants on valid and test datasets.

Model Performance. We use the accuracy metric in our experiments. Table 2 shows the results of baselines and top 3 participant teams on Large Valid and Large Test dataset, from which we get the following conclusion: 1) The participants achieve promising progress compared to baseline models. 2) Both the baselines systems and participant teams perform poorly on the dataset, due to the lack of utilization of prior legal knowledge. It's still challenging to utilize legal knowledge and simulate legal reasoning for the dataset.

### 5 Conclusion

In this paper, we propose a new dataset, CAIL2019-SCM, which focuses on the task of similar case matching in the legal domain. Compared with existing datasets, CAIL2019-SCM can benefit the case matching in the legal domain to help the legal partitioners work better. Experimental results also show that there is still plenty of room for improvement.

# References

- Hadi Amiri, Philip Resnik, Jordan Boyd-Graber, and Hal Daumé III. 2016. Learning text pair similarity with context-sensitive autoencoders. In *Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1882–1892.
- Matthew W Bilotti, Paul Ogilvie, Jamie Callan, and Eric Nyberg. 2007. Structured retrieval for question answering. In *Proceedings of the 30th annual international ACM SIGIR conference on Research and development in information retrieval*, pages 351–358. ACM.
- Jane Bromley, Isabelle Guyon, Yann LeCun, Eduard Säckinger, and Roopak Shah. 1994. Signature verification using a" siamese" time delay neural network. In *Advances in neural information processing systems*, pages 737–744.

- Ilias Chalkidis, Ion Androutsopoulos, and Nikolaos Aletras. 2019. Neural legal judgment prediction in english. *In Proceedings of ACL*.
- Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, and Diana Inkpen. 2017. Enhanced lstm for natural language inference. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1657–1668.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186.
- Chaoqun Duan, Lei Cui, Xinchi Chen, Furu Wei, Conghui Zhu, and Tiejun Zhao. 2018. Attention-fused deep matching network for natural language inference. In *IJCAI*, pages 4033–4040.
- Biralatei Fawei, Jeff Z Pan, Martin Kollingbaum, and Adam Z Wyner. 2018. A methodology for a criminal law and procedure ontology for legal question answering. In *In Proceddings of JIST*. Springer Verlag.
- Hua He, Kevin Gimpel, and Jimmy Lin. 2015. Multiperspective sentence similarity modeling with convolutional neural networks. In *Proceedings of the* 2015 Conference on Empirical Methods in Natural Language Processing, pages 1576–1586.
- Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. *Neural computation*, 9(8):1735–1780.
- Zikun Hu, Xiang Li, Cunchao Tu Zhiyuan Liu, and Maosong Sun. 2018. Few-shot charge prediction with discriminative legal attributes.
- Xin Jiang, Hai Ye, Zhunchen Luo, WenHan Chao, and Wenjia Ma. 2018. Interpretable rationale augmented charge prediction system. In *In Proceedings of COL-ING*.
- Yoshinobu Kano, Mi-Young Kim, Randy Goebel, and Ken Satoh. 2017. Overview of coliee 2017. In *COL-IEE*@ *ICAIL*, pages 1–8.
- R Keown. 1980. Mathematical models for legal prediction. *Computer/lj*, 2:829.
- Mi-Young Kim, Randy Goebel, and S Ken. 2015. Coliee-2015: evaluation of legal question answering. In *In Proceedings of JURISIN*.
- Yoon Kim. 2014. Convolutional neural networks for sentence classification. In *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 1746–1751.

- Fred Kort. 1957. Predicting supreme court decisions mathematically: A quantitative analysis of the" right to counsel" cases. *The American Political Science Review*, 51(1):1–12.
- Benjamin E Lauderdale and Tom S Clark. 2012. The supreme court's many median justices. *American Political Science Review*, 106(4):847–866.
- Bang Liu, Ting Zhang, Fred X Han, Di Niu, Kunfeng Lai, and Yu Xu. 2018. Matching natural language sentences with hierarchical sentence factorization. In *Proceedings of the 2018 World Wide Web Conference*, pages 1237–1246. International World Wide Web Conferences Steering Committee.
- Chao-Lin Liu and Chwen-Dar Hsieh. 2006. Exploring phrase-based classification of judicial documents for criminal charges in chinese. In *Proceedings of the 16th international conference on Foundations of Intelligent Systems*. Springer-Verlag.
- Yi-Hung Liu, Yen-Liang Chen, and Wu-Liang Ho. 2015. Predicting associated statutes for legal problems. *Information Processing & Management*, 51(1):194–211.
- Bingfeng Luo, Yansong Feng, Jianbo Xu, Xiang Zhang, and Dongyan Zhao. 2017. Learning to predict charges for criminal cases with legal basis. In *In Proceedings of EMNLP*.
- Jonas Mueller and Aditya Thyagarajan. 2016. Siamese recurrent architectures for learning sentence similarity. In *Thirtieth AAAI Conference on Artificial Intelligence*.
- Paul Neculoiu, Maarten Versteegh, and Mihai Rotaru. 2016. Learning text similarity with siamese recurrent networks. In *Proceedings of the 1st Workshop on Representation Learning for NLP*, pages 148–157.
- Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In *Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP)*, pages 1532–1543.
- K Raghav, P Krishna Reddy, and V Balakista Reddy. 2016. Analyzing the extraction of relevant legal judgments using paragraph-level and citation information. *AI4JCArtificial Intelligence for Justice*, page 30.
- Maosong Sun, Xinxiong Chen, Kaixu Zhang, Zhipeng Guo, and Zhiyuan Liu. 2016. Thulac: An efficient lexical analyzer for chinese. Technical report, Technical Report. Technical Report.
- Chuanqi Tan, Furu Wei, Wenhui Wang, Weifeng Lv, and Ming Zhou. 2018. Multiway attention networks for modeling sentence pairs. In *IJCAI*, pages 4411–4417.

- Thomas Vacek, Ronald Teo, Dezhao Song, Conner Cowling, Frank Schilder, Timothy Nugent, and Canary Wharf. 2019. Litigation analytics: Case outcomes extracted from us federal court dockets. *In Proceedings of NAACL-HLT*.
- Tom Vacek and Frank Schilder. 2017. A sequence approach to case outcome detection. In *In Proceedings of ICAIL*. ACM.
- Shengxian Wan, Yanyan Lan, Jiafeng Guo, Jun Xu, Liang Pang, and Xueqi Cheng. 2016. A deep architecture for semantic matching with multiple positional sentence representations. In *Thirtieth AAAI Conference on Artificial Intelligence*.
- Ho Chung Wu, Robert Wing Pong Luk, Kam Fai Wong, and Kui Lam Kwok. 2008. Interpreting tf-idf term weights as making relevance decisions. *ACM Transactions on Information Systems (TOIS)*, 26(3):13.
- Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, et al. 2018. Cail2018: A large-scale legal dataset for judgment prediction. *arXiv preprint arXiv:1807.02478*.
- Yukun Yan, Daqi Zheng, Zhengdong Lu, and Sen Song. 2017. Event identification as a decision process with non-linear representation of text. *arXiv preprint arXiv:1710.00969*.
- Wenmian Yang, Weijia Jia, Xiaojie Zhou, and Yutao Luo. 2019. Legal judgment prediction via multiperspective bi-feedback network.
- Hai Ye, Xin Jiang, Zhunchen Luo, and Wenhan Chao. 2018. Interpretable charge predictions for criminal cases: Learning to generate court views from fact descriptions. In *In Proceedings of NAACL*.
- Wenpeng Yin, Hinrich Schütze, Bing Xiang, and Bowen Zhou. 2016. Abcnn: Attention-based convolutional neural network for modeling sentence pairs. *Transactions of the Association for Computational Linguistics*, 4:259–272.
- Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Chaojun Xiao, Zhiyuan Liu, and Maosong Sun. 2018a. Legal judgment prediction via topological learning. In *In Proceedings of the EMNLP*.
- Haoxi Zhong, Chaojun Xiao, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, et al. 2018b. Overview of cail2018: Legal judgment prediction competition. *arXiv* preprint arXiv:1810.05851.