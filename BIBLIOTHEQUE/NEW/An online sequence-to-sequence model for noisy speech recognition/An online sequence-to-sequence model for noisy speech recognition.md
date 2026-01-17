# An online sequence-to-sequence model for noisy speech recognition

Chung-Cheng Chiu\*, Dieterich Lawson\*, Yuping Luo, George Tucker, Kevin Swersky, Ilya Sutskever, Navdeep Jaitly

Abstract—Generative models have long been the dominant approach for speech recognition. The success of these models however relies on the use of sophisticated recipes and complicated machinery that is not easily accessible to non-practitioners. Recent innovations in Deep Learning have given rise to an alternative - discriminative models called Sequence-to-Sequence models, that can almost match the accuracy of state of the art generative models. While these models are easy to train as they can be trained end-to-end in a single step, they have a practical limitation that they can only be used for offline recognition. This is because the models require that the entirety of the input sequence be available at the beginning of inference, an assumption that is not valid for instantaneous speech recognition. To address this problem, online sequence-to-sequence models were recently introduced. These models are able to start producing outputs as data arrives, and the model feels confident enough to output partial transcripts. These models, like sequence-to-sequence are causal – the output produced by the model until any time, t, affects the features that are computed subsequently. This makes the model inherently more powerful than generative models that are unable to change features that are computed from the data. This paper highlights two main contributions - an improvement to online sequence-to-sequence model training, and its application to noisy settings with mixed speech from two speakers.

Index Terms—Automatic Speech Recognition, End-to-End Speech Recognition, Very Deep Convolutional Neural Networks

## I. INTRODUCTION

Enerative models have long been the bread and butter of traditional speech recognition techniques. Using these models, transcription is typically performed by Maximum-aposteriori (MAP) estimation of the word sequence, given a trained generative model and an acoustic observation. Gaussian Mixture Models (GMMs) were the dominant models for instantaneous emission distributions and were coupled to Hidden Markov Models (HMMs) to model the dynamics. While posteriors from the GMM's have been supplanted by Deep Neural Networks (DNN) lately, the recognition model essentially retains its generative interpretation.

Recent developments in deep learning have given rise to a powerful alternative – discriminative models called sequence-to-sequence models, can be trained to model the conditional probability distribution of the output transcript sequence given the input acoustic sequence, directly without inverting a generative model. Sequence-to-sequence models [1], [2] are a general model family for solving supervised learning problems where both the inputs and the outputs are sequences. The performance of the original sequence-to-sequence model has

been greatly improved by the invention of *soft attention* [3], which made it possible for sequence-to-sequence models to generalize better and achieve excellent results using much smaller networks on long sequences. The sequence-to-sequence model with attention had considerable empirical success on machine translation [3], speech recognition [4], [5], image caption generation [6], [7], and question answering [8].

Although remarkably successful, the sequence-to-sequence model with attention must process the entire input sequence before producing an output. However, there are tasks where it is useful to start producing outputs before the entire input is processed. These tasks include speech recognition, machine translation and simultaneous speech recognition and translation with one model [9].

Recently new models have been developed that overcome these shortcomings. These models, which we call online sequence-to-sequence models have the property that they produce outputs as inputs are received [10], [11], while retaining the causal nature of sequence-to-sequence models. In this paper, we use the model that we previously introduced in [11]<sup>1</sup>. This model uses binary stochastic variables to select the timesteps at which to produce outputs. We call this model the Neural Autoregressive Transducer (NAT). The stochastic variables are trained with a policy gradient method. However unlike the work by Luo et. al [11] we use a modified method of training that improves our training results. Further, we explore the use of this model for noisy input where we present single channel mixed speech from two speakers at different mixing proportions as input to the model. This models is uniquely suited for this task as it is a causal model, and as it is trained discriminatively. We show results of this model on a task we call MultiTIMIT it shows that the model is able to handle noisy speech quite well. We speculate that the use of this model with a multiple microphone arrangement should lead to strong results on mixed and noisy speech.

# A. Relation To Prior Work

Sequence to sequence models have been recently applied to phoneme recognition [12] and speech recognition [5]. In these models, the input acoustics, in the form of log Mel filter banks are processed with an encoder neural network that is usually a bidirectional neural network. A decoder then produces output tokens one symbol at a time, using next step prediction. At each step, the decoder uses "soft attention" over the encoder

 $<sup>^{\</sup>rm 1}$  We borrow text heavily from this prior paper to explain the motivation and several details about the model

![](_page_1_Picture_1.jpeg)

Fig. 1: Overall Architecture of the model used in this paper.

time steps to create a "context vector" that is a summary of features of the encoder. The context vector is fed into the decoder and is used to make the prediction at any time step.

While the idea of soft attention as it is currently understood was first introduced by Graves [13], the first truly successful formulation of soft attention is due to Bahdanau et al. [3]. It used a neural architecture that implements a "search query" that finds the most relevant element in the input, which it then picks out. Soft attention has quickly become the method of choice in various settings because it is easy to implement and it has led to state of the art results on various tasks. For example, the Neural Turing Machine [14] and the Memory Network [15] both use an attention mechanism similar to that of Bahdanau et al. [3] to implement models for learning algorithms and for question answering.

While soft attention is immensely flexible and easy to use, it assumes that the test sequence is provided in its entirety at test time. It is an inconvenient assumption whenever we wish to produce the relevant output as soon as possible, without processing the input sequence in its entirety first. Doing so is useful in the context of a speech recognition system that runs on a smartphone, and it is especially useful in a combined speech recognition and a machine translation system.

This model can be thought of extending two previous models – the Connectionist Temporal Classification (CTC) [16] and the Sequence Transducer [17] models that have been used for speech recognition previously. However, neither CTC nor the Sequence Transducer are causal models – both models compute features from the data independently at each time step, and this feature computation is unaffected by the tokens output previously. Note that while the language model RNN in the sequence transducer computes predictions causally, these do not impact the local class predictions made by the acoustics which are independent of each others and not causal.

There exists prior work that investigated causal models

for producing an output without consuming the input in its entirety. These include the work by Mnih [18] and Zaremba and Sutskever [19] who used the Reinforce algorithm to learn the location in which to consume the input and when to emit an output. Finally, Jaitly et al. [10] used an online sequence-to-sequence method with conditioning on partial inputs, which yielded encouraging results on the TIMIT dataset.

This work is technically an extension of our prior work in [11] where policy gradients with continuous rewards was used to train the model. In this paper, we use similar ideas, but instead of using a single sample REINFORCE model with a parameteric baseline for centering the training of the stochastic model, we use a multi-sample training, with a baseline that is an average over leave-one-out samples.

Further, in this paper, we explore the use of this model for noisy data – specifically noisy data that corresponds to speech from two different speakers mixed in at different levels.

#### II. METHODS

In this section we describe the details of the Autoregressive Sequence Transducer. This includes the recurrent neural network architecture, the reward function, and the training and inference procedure. Much of the description is borrowed heavily from our description in [11]. We refer the reader to figure 1 for the details of the model.

We begin by describing the probabilistic model we used in this work. At each time step, i, a recurrent neural network (represented in figure 1) decides whether to emit an output token. The decision is made by a stochastic binary logistic unit  $b_i$ . Let  $\tilde{b}_i \sim \text{Bernoulli}(b_i)$  be a Bernoulli distribution such that if  $\tilde{b}_i$  is 1, then the model outputs the vector  $d_i$ , a softmax distribution over the set of possible tokens. The current position in the output sequence y can be written  $\tilde{p}_i = \sum_{j=1}^i \tilde{b}_j$ , which is incremented by 1 every time the model chooses to emit. Then the model's goal is to predict

![](_page_2_Figure_1.jpeg)

Fig. 2: The impact of entropy regularization on emission locations. Each line shows the emission predictions made for an example input utterance, with each symbol representing 3 input time steps. 'x' indicates that the model chooses to emit output at the time steps, whereas '-' indicates otherwise. Top line - without entropy penalty the model emits symbols either at the start or at the end of the input, and is unable to get meaningful gradients to learn a model. Middle line - with entropy regularization, the model avoids clustering emission predictions in time and learns to spread the emissions meaningfully and learn a model. Bottom line - using KL divergence regularization of emission probability also mitigates the clustering problem, albeit not as effectively as with entropy regularization.

the desired output  $y_{\tilde{p}_i}$ ; thus whenever  $\tilde{b}_i=1$ , the model experiences a loss given by

$$\operatorname{softmax\_logprob}(d_i; y_{\tilde{p}_i}) = -\sum_{c} \log(d_{ic}) y_{\tilde{p}_i c}$$

where c ranges over the number of possible output tokens.

At each step of the RNN, the binary decision of the previous timestep,  $\tilde{b}_{i-1}$  and the corresponding previous target  $t_{i-1} = y_{\tilde{p}_{i-1}}$  are fed into the model as input. This feedback ensures that the model's outputs are causally dependent on the model's previous outputs, and thus the model is from the sequence to sequence family.

We train this model by estimating the gradient of the log probability of the target sequence with respect to the parameters of the model. While this model is not fully differentiable because it uses non-diffentiable binary stochastic units, we can estimate the gradients with respect to model parameters by using a policy gradient method, which has been discussed in detail by Schulman et al. [20] and used by Zaremba and Sutskever [19].

In more detail, we use supervised learning to train the network to make the correct output predictions, and reinforcement learning to train the network to decide on when to emit the various outputs. Let us assume that the input sequence is given by  $(x_1,\ldots,x_{T_1})$  and let the desired sequence be  $(y_1,\ldots,y_{T_2})$ , where  $y_{T_2}$  is a special end-of-sequence token, and where we assume that  $T_2 \leq T_1$ . Then the log probability of the model is given by the following equations:

$$h_i = LSTM(h_{i-1}, concat(x_i, \tilde{b}_{i-1}, \tilde{y}_{i-1}))$$
 (1)

$$b_i = \operatorname{sigmoid}(W_b \cdot h_i)$$
 (2)

$$\tilde{b}_i \sim \text{Bernoulli}(b_i)$$
 (3)

$$\tilde{p}_i = \sum_{j=1}^i \tilde{b}_j \tag{4}$$

$$\tilde{y}_i = y_{\tilde{p}_i} \tag{5}$$

$$d_i = \operatorname{softmax}(W_o h_i) \tag{6}$$

$$\mathcal{R} = \mathcal{R} + \tilde{b}_i \cdot \operatorname{softmax\_logprob}(d_i; \tilde{y}_i)$$

In the above equations,  $\tilde{p}_i$  is the "position" of the model in the output, which is always equal to  $\sum_{k=1}^i \tilde{b}_i$ : the position advances if and only if the model makes a prediction. Note that we define  $y_0$  to be a special beginning-of-sequence symbol. The above equations also suggest that our model can easily be implemented within a static graph in a neural net library such as TensorFlow [21], even though the model has, conceptually, a dynamic neural network architecture.

Following Zaremba and Sutskever [19], we modify the model from the above equations by forcing  $\tilde{b}_i$  to be equal to 1 whenever  $T_1-i \leq T_2-\tilde{p}_i$ . Doing so ensures that the model will be forced to predict the entire target sequence  $(y_1,\ldots,y_{T_2})$ , and that it will not be able to learn the degenerate solution where it chooses to never make any prediction and therefore never experience any prediction error.

We now elaborate on the manner in which the gradient is computed. It is clear that for a given value of the binary decisions  $\tilde{b}_i$ , we can compute  $\partial \mathcal{R}/\partial \theta$  using the backpropagation algorithm. Figuring out how to learn  $\tilde{b}_i$  is slightly more challenging. To understand it, we will factor the reward  $\mathcal{R}$  into an expression  $\mathcal{R}(\tilde{\mathbf{b}})$  and a distribution  $\rho(\tilde{\mathbf{b}})$  over the binary vectors, and derive a gradient estimate with respect to the parameters of the model:

$$\mathcal{R} = \mathbb{E}_{\tilde{\mathbf{b}}} \left[ R(\tilde{\mathbf{b}}) \right] \tag{8}$$

Differentiating, we get

$$\nabla \mathcal{R} = \mathbb{E}_{\tilde{\mathbf{b}}} \left[ \nabla R(\tilde{\mathbf{b}}) + R(\tilde{\mathbf{b}}) \nabla \log \rho(\tilde{\mathbf{b}}) \right]$$
(9)

where  $\rho(\tilde{\mathbf{b}})$  is the probability of a binary sequence of the  $\tilde{b}_i$  decision variables. In our model,  $\rho(\tilde{\mathbf{b}})$  is computed using the chain rule over the  $b_i$  probabilities:

$$\log \rho(\tilde{\mathbf{b}}) = \sum_{i=1}^{T} \tilde{b}_i \log b_i + (1 - \tilde{b}_i) \log(1 - b_i)$$
 (10)

Since the gradient in equation 9 is a policy gradient, it has very high variance, and variance reduction techniques must be applied. As is common in such problems we use *centering* (also known as baselines) and Rao-Blackwellization to reduce the variance of such models. See Mnih and Gregor [22] for an

![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

![](_page_3_Figure_3.jpeg)

(b) Multi-speaker - with 25% mixing

(c) Multi-speaker - with 50% mixing

Fig. 3: Example training run on TIMIT.

example of the use of such techniques in training generative models with stochastic units.

Baselines are commonly used in the reinforcement learning literature to reduce the variance of estimators, by relying on the identity  $\mathbb{E}_{\tilde{\mathbf{b}}} \left| \nabla \log \rho(\tilde{\mathbf{b}}) \right| = 0$ . Thus the gradient in 9 can be better estimated by the following, through the use of a well chosen baseline function,  $\Omega(\mathbf{x})$ , where  $\mathbf{x}$  is a vector of side information which happens to be the input and all the outputs up to timestep  $\tilde{p}_i$ :

$$\nabla \mathcal{R} = \mathbb{E}_{\tilde{\mathbf{b}}} \left[ \nabla R(\tilde{\mathbf{b}}) + \left( R(\tilde{\mathbf{b}}) - \Omega(\mathbf{x}) \right) \nabla \log \rho(\tilde{\mathbf{b}}) \right] \quad (11)$$

The variance of this estimator itself can be further reduced by Rao-Blackwellization, giving:

$$\mathbb{E}_{\tilde{\mathbf{b}}} \left[ \left( R(\tilde{\mathbf{b}}) - \Omega(\mathbf{x}) \right) \nabla \log \rho(\tilde{\mathbf{b}}) \right] = \sum_{j=1}^{T} \mathbb{E}_{\tilde{\mathbf{b}}} \left[ \left( \sum_{i=j}^{T} R_i - \Omega_j \right) \nabla \log p(b_j | b_{< j}, \mathbf{x}_{\leq j}, \mathbf{y}_{\leq \tilde{p}_{j-1}}) \right]$$
(12)

This above term, while not computable analytically, can be estimated numerically by drawing K sample trajectories, indexed by k. Thus we have an estimate of the gradient as follows:

$$\nabla \mathcal{R} \approx \frac{1}{K} \sum_{k=1}^{K} \sum_{j=1}^{T} \left[ \left( \sum_{i=j}^{T} R_i^k - \Omega_j^k \right) \nabla \log p(b_j^k | b_{< j}^k, \mathbf{x}_{\leq j}, \mathbf{y}_{\leq \tilde{p}_{j-1}}) \right]$$
(13)

where, the superscript of k indicates the sample index. In previous work [11] we used a single sample estimate (i.e. K=1) and a neural network as a parametric baseline to estimate  $\Omega_i^k$ . This was computed using a linear projection of the hidden state  $h_j$  of the top LSTM layer of the RNN, i.e.  $\Omega_i^k = W' h_i^k + o$ , where W is a vector and o is a bias.

Recent work in reinforcement learning and variational methods has shown the advantage of multi-sample estimates [23], [24]. In this paper, we thus explore the use of a multi-sample estimate, with K = 16. Further, as in [24] we used a baseline with a leave one out average, which we explain next.

A straightforward choice of this baseline,  $\Omega_i^k$  is the average sum of future rewards from the other samples

$$\Omega_j^k = \frac{1}{K - 1} \sum_{k' \neq k} \sum_{i \ge j}^T R_i^{k'},$$

however, this ignores the fact that the internal state of the different samples are not the same. Ideally we would average over multiple trajectories starting from the same state (i.e. number of inputs consumed and outputs produced), but this is computationally expensive. As a result there is an imbalance where some of the samples have emitted more symbols than the others, and thus the future rewards may not be directly comparable. We add a residual term to address this,

$$\Omega_j^k = \frac{1}{K-1} \sum_{k' \neq k} \sum_{i>j}^T R_i^{k'} + \frac{1}{K-1} \sum_{k' \neq k} \sum_{i (14)$$

We call this the *leave-one-out* baseline.

Finally, we note that reinforcement learning models are often trained with augmented objectives that add an entropy penalty for actions are the too confident [25], [26]. We found this to be crucial for our models to train successfully. In light of the regularization term, the augmented reward at any time steps, i, is:

$$R_{i} = \tilde{b}_{i} \log p(d_{i} = t_{i} | \mathbf{x}_{\leq i}, \tilde{\mathbf{b}}_{< i}, \mathbf{t}_{< i})$$

$$- \lambda \tilde{b}_{i} \log p(b_{i} = 1 | b_{< i}, \mathbf{x}_{\leq i})$$

$$+ \lambda (1 - \tilde{b}_{i}) \log(p(b_{i} = 0 | b_{< i}, \mathbf{x}_{\leq i})) \quad (15)$$

Without the use of this regularization in the model, the RNN emits all the symbols clustered in time, either at very start of the input sequence, or at the end. The model has a difficult time recovering from this configuration, since the gradients are too noisy and biased. However, with the use of this penalty, the model successfully navigates away from parameters that lead to very clustered predictions and eventually learns sensible parameters. An alternative we explored was to use the the KL divergence of the predictions from a target Bernouilli rate of emission at every step. However, while this helped the model, it was not as successful as entropy regularization. See figure 2 for an example of this clustering problem and how regularization ameliorates it.

![](_page_4_Figure_2.jpeg)

Fig. 4: This figure shows the model emission distributions, the probability of emitting tokens as the input is received. The target phonemes are "pau f er s pau t pau ae pau m ih l pau t ah dh ih sh r eh dx ih pau ch iy s pau"

### III. EXPERIMENTS AND RESULTS

We conducted experiments on two different speech corpora using this model. Initial experiments were conducted on TIMIT to assess hyperparameters that could lead to stable behavior of the model. The second set of experiments were conducted on speech mixed in from two different speakers – a male speaker and a female speaker – at different mixing proportions. We call these experiments Multi-TIMIT.

### A. TIMIT

The TIMIT data set is a phoneme recognition task in which phoneme sequences have to be inferred from input audio utterances. The training dataset contains 3696 different audio clips and the target is one of 60 phonemes. Before scoring, these are collapsed to a standard 39 phoneme set, and then the Levenshtein edit distance is computed to get the phoneme error rate (PER).

The models we trained on TIMIT had two layers with 256 units per layer. Each model was trained with Adam( [27]) and used a learning rate of  $7\times 10^{-5}$ . We used asynchronous SGD with 16 replicas in tensorflow as the neural network framework for training the models [21], [28]. No GPUs were used in the training.

Entropy regularization was crucial to produce the best results, with emissions clumping at the end or beginning of the utterances when an entropy penalty was not used. We started the weight of the entropy penalty at 1 and decayed it linearly to 0.1. We began decaying the entropy penalty at 10,000 steps and experimented with ending the decay at {100,000, 200,000, 300,000, 400,000} steps, finding that step 200,000 worked best. After step 200,000 the entropy penalty weight was kept at 0.1.

We also regularized our models with variational weight noise [29]. We tested the values {0.075, 0.1, 0.15} for the standard deviation of the noise and found that 0.15 worked best. We started the standard deviation of the variational noise at 0, and increased it linearly from step 10,000 to a value of 0.15 at step 200,000. In each experiment the entropy penalty stopped decaying on the same step that the variational noise finished increasing.

We also used L2-norm weight regularization to encourage small weights. We found that a weight of 0.001 worked best after trying weights  $\{10^{-5}, 10^{-4}, 10^{-3}\}$ .

Lastly, we note that the input filterbanks were processed such that three continuous frames of filterbanks, representing a total of 30ms of speech were concatenated and input to the model. This results in a smaller number of input steps and allows the model to learn hard alignments much faster than it would otherwise.

See Figure 3 for an example of a training curve. It can be seen that the model requires a larger number of updates (> 100K) before meaningful models are learnt. However, once learning starts, steady process is achieved, even though the model is trained by policy gradient.

Table I shows a summary of the results achieved on TIMIT by our method and other, more mature models. As can be seen our model compares favorably with other unidirectional models, such as CTC, DNN-HMM's etc. Combining with more sophisticated features such as convolutional models should produce better results. Moreover, this model has the capacity to absorb language models, and as a result, should be more suited to end to end training than CTC and DNN-HMM based models that cannot inherently capture language models because they predict all tokens independently of each other.

TABLE I: Results on TIMIT using Unidirectional LSTMs for various models.

| Method                                                                                                                                              | PER                              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------|
| CTC[30] DNN-HMM[31] seq2seq with attention (our implementation) neural transducer[10]                                                               | 19.6%<br>20.7%<br>24.5%<br>19.8% |
| NAT (Stacked LSTM) + Parameteric Baseline[11]<br>NAT (Grid LSTM) + Parameteric Baseline[11]<br>NAT (Stacked LSTM) + Averaging Baseline (this paper) | 21.5%<br>20.5%<br>20.0%          |

## B. Multi-TIMIT

We generate a new data set by mixing a male voice with a female voice from the original TIMIT data. Each utterance in the original TIMIT data pairs with an utterance coming from the opposite gender. The wave signal of both utterances are

![](_page_5_Figure_2.jpeg)

Fig. 5: Emission distributions for Multi-TIMIT: This figure shows the probability of emitting tokens for the case of a clean utterance from TIMIT and a corresponding noisy utterance in Multi-TIMIT. It can be seen that for the Multi-TIMIT utterances, the model chooses to emit symbols slightly later than it would have for TIMIT utterances.

first scaled to the same range, and then the signal scale of the second utterance is reduced to a smaller volume when mixing the two utterances. We explored different scale for mixing the second utterance, 50%, 25%, and 10%, and created three sets of experiments. The same feature generation method that was described above was used, resulting in a 123 dimensional input per frame. The transcript of the speaker 1 was used as the ground truth transcript for this new utterance. This data follow the same train, dev, and test specification as TIMIT. As a result the mixed data has the same number of train, dev, and test utterances as the original TIMIT, and they also have the same sets of target phonemes.

Our model was a 2-layer LSTM with 256 units in each layer. The same hyper-parameter search strategy that was used for clean TIMIT (section III-A) was applied here.

TABLE II: Results on Multi-TIMIT: This table show the phoneme error rate (PER) achieved by our models at different proportions of mixing in for the distracting speech. Also shown are results from CTC with deep LSTMs [16] and RNN-Transducer [17] using an implementation provided by Alex Graves.

| Mixing Proportion | NAT   | CTC   | RNN-Transducer |
|-------------------|-------|-------|----------------|
| 0.1               | 25.9% | 27.3% | 25.7%          |
| 0.25              | 32.5% | 33.3% | 32.2%          |
| 0.5               | 42.9% | 43.8% | 48.9%          |

Figures 3b and 3c show examples of training curves for two cases with mixing proportions of 0.25 and 0.5 respectively. In both cases it can be seen that the model learns to overfit the data.

Table II shows results from using different mixing proportions of confounding speaker. It can be seen that with increasing mixing proportion, the model's results get worse as expected. For the experiments, each audio input is always paired with the same confounding audio input. Interestingly we

found that pairing the same audio with multiple confounding audio inputs produced worse results, because of much worse overfitting. This presumably happens because our model is powerful enough to memorize the entire transcripts.

Figure 5 shows an example of where the model emits symbols for an example Multi-TIMIT utterance. It also shows a comparison with the emissions from a clean model. Generally speaking the model chooses to emit later for Multi-TIMIT compared to when it emits for TIMIT.

## IV. DISCUSSION

In this paper we have introduced a new way to train online sequence-to-sequence models and showed its application to noisy input. These models, as a result of being causal models, can incorporate language models, and can also generate multiple different transcripts for the same audio input. This makes it a very powerful class of models. Even on a dataset as small as TIMIT the model is able to adapt to mixed speech. For our experiments each speaker was only coupled to one distracting speaker and hence the dataset size was limited. By pairing each speaker with multiple other speakers, and predicting each one as outputs, we should be able to achieve greater robustness. Because of this capability, we would like to apply these models to multi-channel, multi-speaker recognition in the future.

## V. CONCLUSIONS

In this work, we presented a new way of training an online sequence to sequence model. This model allows us to exploit the modelling power of sequence-to-sequence problems without the need to process the entire input sequence first. We show the results of training this model on the TIMIT corpus and acheived results comparable to state of the art results with uni-directional models.

We also applied this model to the task of mixed speech from two speakers, producing the output for the louder speaker. We show that the model is able to achieve reasonble accuracy, even with single channel input. In the future, we will apply this work to multi-speaker recognition.

#### REFERENCES

- I. Sutskever, O. Vinyals, and Q. V. Le, "Sequence to Sequence Learning with Neural Networks," in *Neural Information Processing Systems*, 2014.
- [2] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwen, and Y. Bengio, "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation," in Conference on Empirical Methods in Natural Language Processing, 2014.
- [3] D. Bahdanau, K. Cho, and Y. Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate," in *International Conference* on Learning Representations, 2015.
- [4] J. Chorowski, D. Bahdanau, K. Cho, and Y. Bengio, "End-to-end Continuous Speech Recognition using Attention-based Recurrent NN: First Results," in Neural Information Processing Systems: Workshop Deep Learning and Representation Learning Workshop, 2014.
- [5] W. Chan, N. Jaitly, Q. V. Le, and O. Vinyals, "Listen, attend and spell," arXiv preprint arXiv:1508.01211, 2015.
- [6] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio, "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention," in *International Conference* on Machine Learning, 2015.
- [7] O. Vinyals, A. Toshev, S. Bengio, and D. Erhan, "Show and Tell: A Neural Image Caption Generator," in *IEEE Conference on Computer Vision and Pattern Recognition*, 2015.
- [8] J. Weston, S. Chopra, and A. Bordes, "Memory networks," arXiv preprint arXiv:1410.3916, 2014.
- [9] R. J. Weiss, J. Chorowski, N. Jaitly, Y. Wu, and Z. Chen, "Sequence-to-sequence models can directly transcribe foreign speech," *CoRR*, vol. abs/1703.08581, 2017. [Online]. Available: http://arxiv.org/abs/1703.08581
- [10] N. Jaitly, D. Sussillo, Q. V. Le, O. Vinyals, I. Sutskever, and S. Bengio, "A neural transducer," in *Advances in Neural Information Processing Systems*, 2016. [Online]. Available: https://arxiv.org/abs/1511.04868
- [11] Y. Luo, C.-C. Chiu, N. Jaitly, and I. Sutskever, "Learning online alignments with continuous rewards policy gradient," arXiv preprint arXiv:1608.01281, 2016.
- [12] J. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Bengio, "Attention-Based Models for Speech Recognition," in *Neural Information Processing Systems*, 2015.
- [13] A. Graves, "Generating sequences with recurrent neural networks," arXiv preprint arXiv:1308.0850, 2013.
- [14] A. Graves, G. Wayne, and I. Danihelka, "Neural turing machines," arXiv preprint arXiv:1410.5401, 2014.
- [15] S. Sukhbaatar, J. Weston, R. Fergus et al., "End-to-end memory networks," in Advances in Neural Information Processing Systems, 2015, pp. 2431–2439.
- [16] A. Graves and N. Jaitly, "Towards End-to-End Speech Recognition with Recurrent Neural Networks," in *International Conference on Machine Learning*, 2014.
- [17] A. Graves, "Sequence Transduction with Recurrent Neural Networks," in International Conference on Machine Learning: Representation Learning Workshop, 2012.
- [18] V. Mnih, N. Heess, A. Graves et al., "Recurrent models of visual attention," in Advances in Neural Information Processing Systems, 2014, pp. 2204–2212.
- [19] W. Zaremba and I. Sutskever, "Reinforcement learning neural turing machines," arXiv preprint arXiv:1505.00521, 2015.
- [20] J. Schulman, N. Heess, T. Weber, and P. Abbeel, "Gradient estimation using stochastic computation graphs," in *Advances in Neural Information Processing Systems*, 2015, pp. 3510–3522.
- [21] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin *et al.*, "Tensorflow: Large-scale machine learning on heterogeneous distributed systems," *arXiv preprint arXiv:1603.04467*, 2016.
- [22] A. Mnih and K. Gregor, "Neural variational inference and learning in belief networks," *CoRR*, vol. abs/1402.0030, 2014. [Online]. Available: http://arxiv.org/abs/1402.0030
- [23] Y. Burda, R. Grosse, and R. Salakhutdinov, "Importance weighted autoencoders," arXiv preprint arXiv:1509.00519, 2015.
- [24] A. Mnih and D. J. Rezende, "Variational inference for monte carlo objectives," arXiv preprint arXiv:1602.06725, 2016.

- [25] S. Levine, "Motor skill learning with local trajectory methods," Ph.D. dissertation, Stanford University, 2014.
- [26] R. J. Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine learning*, vol. 8, no. 3-4, pp. 229–256, 1992.
- [27] D. Kingma and J. Ba, "Adam: A method for stochastic optimization," arXiv preprint arXiv:1412.6980, 2014.
- [28] J. Dean, G. S. Corrado, R. Monga, K. Chen, M. Devin, Q. V. Le, M. Z. Mao, M. Ranzato, A. Senior, P. Tucker, K. Yang, and A. Y. Ng, "Large Scale Distributed Deep Networks," in *Neural Information Processing* Systems, 2012.
- [29] A. Graves, "Practical variational inference for neural networks," in Advances in Neural Information Processing Systems, 2011, pp. 2348– 2356.
- [30] A. Graves, A. Mohamed, and G. Hinton, "Speech recognition with deep recurrent neural networks," in Acoustics, Speech and Signal Processing (ICASSP), 2013 IEEE International Conference on. IEEE, 2013, pp. 6645–6649.
- [31] A. Mohamed, G. E. Dahl, and G. Hinton, "Acoustic modeling using deep belief networks," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 20, no. 1, pp. 14–22, 2012.

**Chung-Cheng Chiu** Chung-Cheng Chiu is a Software Engineer at Google Brain working on deep learning models. He received his PhD from the University of Southern California under the supervision of Stacy Marsella. His interests lie in Deep learning, speech recognition, and reinforcement learning.

**Dieterich Lawson** Dieterich Lawson is a Brain Resident at Google working on sequential latent variable models and methods for variational inference. He did his undergrad and masters at Stanford in computer science and computational math, respectively. His interests include deep learning, reinforcement learning, and optimization.

**Yuping Luo** Yuping Luo is a student at Tsinghua University majoring in Computer Science. His interests are Deep Learning, Optimization, and related theory. He worked on Streaming Algorithms and Online Learning at Tsinghua University under the supervision of Periklis Papakonstantinou. He interned at Google Brain working on Speech Recognition and was supervised by Ilya Sutskever and Navdeep Jaitly.

George Tucker George Tucker is a research software engineer at Google Brain working on deep learning models for sequences. Prior to joining Google, he was a research scientist at Amazon working on deep acoustic models for small-footprint keyword spotting. He received his PhD from MIT under the supervision of Bonnie Berger. His interests lie in Deep Learning, Variational Inference, and Reinforcement Learning.

**Kevin Swersky** Kevin Swersky is a research scientist at Google brain. He received his PhD from the University of Toronto under the supervision of Richard Zemel. His research interests include deep learning, graphical models, generative models, and meta-learning. During his PhD, Kevin co-founded Whetlab, an online hyperparameter tuning service, which was subsequently acquired by Twitter.

**Ilya Sutskever** Ilya Sutskever is the research director of OpenAI. Previously, he was a research scientist at Google Brain. He was also a co-founder of DNNResearch which was acquired by Google. Sutskever has made many contributions to the field of Deep Learning, including the first large scale convolutional neural network that convincingly outperformed all previous vision systems by winning the 2012 ImageNet competition. He was listed in MIT Technology Reviews 35 innovators under 35.

**Navdeep Jaitly** Navdeep Jaitly is a Research Scientist at NVIDIA Research. Previously he was a research scientist at Google Brain. His interests lie in Endto-end models for speech recognition and speech synthesis, Reinforcement Learning and new models for sequences using Deep Learning.