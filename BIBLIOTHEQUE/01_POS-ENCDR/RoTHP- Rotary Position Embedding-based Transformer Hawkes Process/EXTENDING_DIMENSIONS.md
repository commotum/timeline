## 1. Basic Metadata

- Title: "ROTHP: ROTARY POSITION EMBEDDING-BASED TRANSFORMER HAWKES PROCESS" (Title)
- Authors: "Anningzhe Gao\*, Shan Dai \*" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.


## 2. One-Sentence Contribution Summary

The paper proposes "a new Rotary Position Embedding-based THP (RoTHP) architecture" to address the "timestamp noise sensitivity problem and sequence prediction issue" in attention-based Hawkes processes (Abstract; Section 1 Introduction).


## 3. Tasks Evaluated

- Task name: Log-likelihood modeling of event sequences; Task type: Other (specify: temporal point process likelihood modeling); Dataset(s) used: Financial Transactions, StackOverFlow, Synthetic, Retweet, Memetrack, Mimic-II; Domain: temporal event sequences (financial, social/web, medical, synthetic); Evidence: "We evaluated the performance of various models, including RMTPP, NHP, SAHP, THP, and our proposed model. The evaluation metrics employed were log-likelihood and accuracy." (Section 4.4 Result)
- Task name: Next event type prediction; Task type: Classification; Dataset(s) used: Financial, Mimic-II, SO (StackOverFlow); Domain: temporal event sequences; Evidence: "For the prediction of next event type and timestamp, we train two linear layers  $W^e, W^t$" and "By definition,  $\mathcal{L}_{event}$  measures the accuracy of the event type prediction" (Section 3.1.2 Training)
- Task name: Next event timestamp prediction; Task type: Other (specify: time prediction / regression); Dataset(s) used: Financial, Mimic-II, SO (StackOverFlow); Domain: temporal event sequences; Evidence: "For the prediction of next event type and timestamp, we train two linear layers  $W^e, W^t$" and " $\mathcal{L}_{time}$  measures the mean square loss the of time prediction." (Section 3.1.2 Training)
- Task name: Future prediction (predict future events from past data); Task type: Classification + Other (specify: time prediction / regression); Dataset(s) used: financial transaction, synthetic, StackOverFlow; Domain: temporal event sequences; Evidence: "In this subsection, we consider the case where we use the previous information to predict the future ones." and "We do the test on financial transaction, synthetic and StackOverflow dataset, and Table 6 shows the result." (Section 4.6 Predict the future features)


## 4. Domain and Modality Scope

- Evaluation performed on multiple domains within the same modality (temporal event sequences). Evidence: "This dataset comprises stock transaction records from a single trading day." (Section 4.1 Dataset); "This dataset is a collection of data from the question-answer website, Stacoverflow." (Section 4.1 Dataset); "The MIMIC-II medical dataset compiles data from patients' admissions to an ICU over a span of seven years." (Section 4.1 Dataset)
- Multiple modalities: Not stated; all described inputs are temporal event sequences.
- Domain generalization or cross-domain transfer: The paper claims generalization to timestamp translations and sequence prediction, not cross-domain transfer. Evidence: "our RoTHP can be better generalized in sequence data scenarios with timestamp translations and in sequence prediction tasks." (Abstract)


## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Log-likelihood modeling of event sequences | Yes | Not specified | No | "Denote the log-likelihood of  $\mathcal{S}$  as  $\mathcal{L}$ , then the training loss can be defined by  $\mathcal{L}(\mathcal{S}) = -\mathcal{L} + \beta_1 \mathcal{L}_{event}(\mathcal{S}) + \beta_2 \mathcal{L}_{time}(\mathcal{S})$" (Section 3.1.2 Training) |
| Next event type prediction | Yes | Not specified | Yes | "For the prediction of next event type and timestamp, we train two linear layers  $W^e, W^t$" and " $\hat{k}_{j+1} = argmax(Softmax(W^e \mathbf{h}(t_j)))$" (Section 3.1.2 Training) |
| Next event timestamp prediction | Yes | Not specified | Yes | " $\hat{t}_{j+1} = W^t \mathbf{h}(t_j).$" and " $\mathcal{L}_{time}$  measures the mean square loss the of time prediction." (Section 3.1.2 Training) |
| Future prediction (predict future events) | Not specified | Not specified | Not specified | "In this subsection, we consider the case where we use the previous information to predict the future ones." (Section 4.6 Predict the future features) |


## 6. Input and Representation Constraints

- Input is a marked temporal event sequence of timestamps and event types: "We indicate with  $S = \{(t_i, k_i)\}_{i=1}^n$  an event sequence, where the tuple  $(t_i, k_i)$  is the i-th event of the sequence S,  $t_i$  is its timestamp, and  $k_i \in \mathcal{U}$  is its event type." (Section 1 Introduction)
- Fixed event-type vocabulary with one-hot encoding: "We use  $\mathbf{X}$  to denote the matrix representing the one-hot vector corresponding to the event sequence.  $\mathbf{X} \in \mathbb{R}^{K \times L}$ , the ith column of  $\mathbf{X}$  is a one-hot vector where the jth entry is non-zero if and only if  $k_i = j$ ." (Section 3.1.1 Model architecture)
- Embedding dimensionality is fixed and query dimension is even: "Let M be the embedding dimension, K be the number of events." and " $M_K$  in the dimension of the query embedding which is an even number" (Section 3.1.1 Model architecture)
- Sequence length is variable (no fixed token count stated): "Let  $S = \{(t_i, k_i)\}_{i=1}^n$  be a sequence of Hawkes process." (Section 3.1.1 Model architecture)
- Fixed/variable input resolution, patch size, padding/resizing: Not specified.


## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified as a model limit; dataset maxima include "The minimal length is 20 and the maximal length is 100." (Synthetic) and "minimum 41 and maximum 736" (StackOverFlow) and "minimum 50 and maximum 264" (Retweet). (Section 4.1 Dataset)
- Sequence length fixed or variable: Variable (dataset min/max ranges and variable n). Evidence: "The average length of the sequences in the dataset is 72, with minimum 41 and maximum 736" (StackOverFlow) and "Let  $S = \{(t_i, k_i)\}_{i=1}^n$  be a sequence" (Section 3.1.1 Model architecture)
- Attention type: Not specified (self-attention formulation shown). Evidence: "The attention output is given by

$$O = Softmax(\frac{A}{\sqrt{D_K}})V \tag{19}$$" (Section 3.1.1 Model architecture)
- Computational cost mechanisms (windowing/pooling/sparsity): Not stated.
- Sequence length flexibility claim: "We show the translation invariance property and sequence length flexibility of our proposed RoTHP" (Section 1 Introduction) and "RoPE has shown the extension property, which means that it can deal with longer sequences." (Section 3.3 Sequence Prediction Flexibility)


## 8. Positional Encoding (Critical Section)

- Mechanism: Rotary positional encoding. Evidence: "The key idea in our model design is to apply the rotary position embedding method [22] into temporal process." and "Unlike the absolute positional embedding used in THP, we consider the **Rotary Temporal Positional Embedding** (RoTPE)." (Section 3.1.1 Model architecture)
- Where applied: In attention via rotation of Q/K representations. Evidence: "Let  $W^Q, W^K, W^V$  be the linear transformations corresponds to the  $\mathbf{Q}, \mathbf{K}, \mathbf{V}$  vectors" and "The attention matrix A is given by

$$q_{i}^{T} R_{t_{i}}^{T} R_{t_{j}} k_{j}$$" (Section 3.1.1 Model architecture)
- Fixed vs modified per task; ablations/comparisons: Positional encoding is compared against THP's absolute sinusoid. Evidence: "The primary focus, however, is concentrated on comparing our model and the THP, in an attempt to comprehend the influence of the rotary temporal positional encoding." (Section 4.2 Baselines)


## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Core research variable. Evidence: "The primary focus, however, is concentrated on comparing our model and the THP, in an attempt to comprehend the influence of the rotary temporal positional encoding." (Section 4.2 Baselines)
- Multiple positional encodings compared: Yes (RoTPE vs absolute sinusoid in THP). Evidence: "Unlike the absolute positional embedding used in THP, we consider the **Rotary Temporal Positional Embedding** (RoTPE)." (Section 3.1.1 Model architecture)
- Claim that PE choice is not critical or secondary: Not stated.


## 10. Evidence of Constraint Masking

- Model size(s): Not specified.
- Dataset size(s): "Our synthetic dataset admits 5 event types, with average length 60. The minimal length is 20 and the maximal length is 100."; "The average length of the dataset is 2074" (Financial Transactions); "The average length of the sequences in the dataset is 72, with minimum 41 and maximum 736" (StackOverFlow); "The average length of the sequences is 109, with minimum 50 and maximum 264" (Retweet); "This dataset comprises references to 42,000 distinct memes over a period of ten months. It encompasses data from more than 1.5 million documents, including blogs and web articles, sourced from over 5,000 websites." (Memetrack) (Section 4.1 Dataset)
- Attribution of gains: Performance gains are attributed to rotary embedding rather than scaling. Evidence: "RoTHP consistently outperforms THP, underscoring the benefits of rotary embedding in the Hawkes process." (Section 4.4 Result)
- Scaling claims (model size or data scale as primary driver): Not stated.


## 11. Architectural Workarounds

- Rotary temporal positional embedding to enforce relative-time structure: "The key idea in our model design is to apply the rotary position embedding method [22] into temporal process." and "Unlike the absolute positional embedding used in THP, we consider the **Rotary Temporal Positional Embedding** (RoTPE)." (Section 3.1.1 Model architecture)
- Modified intensity input using time differences (translation invariance): "Different from the conditional intensity function setting in the THP [17], here we directly adopt the time difference  $t - t_j$  without the normalization by  $t_j$ ." (Section 3.1.2 Training)
- Separate task-specific heads for event type and time prediction: "For the prediction of next event type and timestamp, we train two linear layers  $W^e, W^t$" (Section 3.1.2 Training)


## 12. Explicit Limitations and Non-Claims

- Limitation in robustness study dataset choice: "We pick the Synthetic and SO datasets to see the influence of the Gaussian noise. The reason we pick these two datasets is because we need a long sequence length so that the temporal information will be more important. The Financial dataset has too small time stamp gaps, and the Retweet dataset has integer time stamps, which are not appropriate for this case." (Section 4.5 Robustness study)
- Other explicit limitations or non-claims (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not stated.


### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: multiple real-world domains plus synthetic data, all as temporal event sequences.
> - Task structure: likelihood modeling and next-event type/time prediction, plus a future-prediction setting.
> - Representation rigidity: fixed event-type vocabulary and embedding dimensions; inputs are (timestamp, event type) sequences with variable length.
> - Model sharing vs specialization: shared transformer trunk with separate linear heads for event type vs time; trained per-dataset (no joint multi-domain training stated).
> - Role of positional encoding: central research variable (RoTPE vs absolute positional encoding) tied to translation invariance.


### 14. Final Classification

**Final Classification:** Multi-task, multi-domain (constrained)

The paper evaluates multiple tasks including "prediction of next event type and timestamp" and reports "log-likelihood and accuracy" (Section 3.1.2 Training; Section 4.4 Result). It also spans multiple domains, e.g., "stock transaction records" (Financial Transactions), "question-answer website, Stacoverflow" (StackOverFlow), and "patients' admissions to an ICU" (Mimic-II) (Section 4.1 Dataset). All datasets are temporal point-process event sequences, so the multi-domain scope remains constrained to a single modality.
