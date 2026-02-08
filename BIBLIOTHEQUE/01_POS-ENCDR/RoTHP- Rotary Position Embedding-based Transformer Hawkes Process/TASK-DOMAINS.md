# ROTHP: ROTARY POSITION EMBEDDING-BASED TRANSFORMER HAWKES PROCESS (Year not specified in the paper)
Source: RoTHP- Rotary Position Embedding-based Transformer Hawkes Process.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Temporal point process likelihood modeling | Marked event sequence (timestamps and event types) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Sequence log-likelihood score | 0D (inferred) | Fixed (inferred) |
| Next event type prediction (classification) | Marked event sequence (timestamps and event types) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Next event type label | 0D (inferred) | Fixed (inferred) |
| Next event timestamp prediction (regression) | Marked event sequence (timestamps and event types) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Next event timestamp value | 0D (inferred) | Fixed (inferred) |

## Summary
The paper covers marked temporal event-sequence modeling and prediction in a single modality: Hawkes-process sequences of timestamps and event types. It explicitly trains/evaluates log-likelihood modeling plus next event type and next event timestamp prediction. The task inputs are temporal sequences, mapped to 1D (t), with variable sequence length and sequence-extension claims supporting Open input dynamics (inferred). The architecture uses standard self-attention over the provided sequence (Static, inferred) and hidden representations h(t_j) for downstream heads (Constructed state, inferred), while outputs are scalar/label-style 0D predictions (inferred).

## Evidence
### Task: Temporal point process likelihood modeling
- "To learn the parameters of Hawkes processes, it is common to use Maximum Likelihood Estimation (MLE)." (Section 1 Introduction)
- "The log-likelihood of an event sequence S over a time interval [0,T] is given by:" (Section 1 Introduction)
- "The evaluation metrics employed were log-likelihood and accuracy." (Section 4.4 Result)
- Inference: Input is classified as 1D (t) from the marked timestamped sequence "We indicate with  $S = \{(t_i, k_i)\}_{i=1}^n$  an event sequence, where the tuple  $(t_i, k_i)$  is the i-th event of the sequence S,  $t_i$  is its timestamp, and  $k_i \in \mathcal{U}$  is its event type." (Section 1 Introduction). In Dynamics is Open (inferred) from variable-length definition "Let  $S = \{(t_i, k_i)\}_{i=1}^n$  be a sequence of Hawkes process." and sequence extension claim "In multiple Natural Language Processing (NLP) tasks, RoPE has shown the extension property, which means that it can deal with longer sequences." (Section 3.1.1 Model architecture; Section 3.3 Sequence Prediction Flexibility). Attention is Static (inferred) from fixed self-attention computation "The attention output is given by" over provided Q/K/V (Section 3.1.1 Model architecture). State is Constructed (inferred) from "Then we apply a feed-forward neural network to get the hidden representation  $\mathbf{h}(t_j)$  for  $1 \leq j \leq n$ ." (Section 3.1.1 Model architecture). Output is treated as 0D Fixed (inferred) because the reported task metric output is a scalar log-likelihood score (Section 4.4 Result).

### Task: Next event type prediction (classification)
- "For the prediction of next event type and timestamp, we train two linear layers  W^e, W^t" (Section 3.1.2 Training)
- "$$\hat{k}_{j+1} = argmax(Softmax(W^e \mathbf{h}(t_j))),$$" (Section 3.1.2 Training)
- "By definition,  $\mathcal{L}_{event}$  measures the accuracy of the event type prediction and  $\mathcal{L}_{time}$  measures the mean square loss the of time prediction." (Section 3.1.2 Training)
- Inference: Input/Attention/State follow the same architecture evidence as above: temporal sequence input implies 1D (t) (inferred), attention is Static (inferred), and hidden representation h(t_j) implies Constructed state (inferred). In Dynamics is Open (inferred) from variable-length sequence formulation and sequence flexibility statements (Section 3.1.1; Section 3.3). Output is 0D Fixed (inferred) because the head predicts a single next-event class label per prediction step.

### Task: Next event timestamp prediction (regression)
- "For the prediction of next event type and timestamp, we train two linear layers  W^e, W^t" (Section 3.1.2 Training)
- "$$\hat{t}_{j+1} = W^t \mathbf{h}(t_j).$$" (Section 3.1.2 Training)
- "By definition,  $\mathcal{L}_{event}$  measures the accuracy of the event type prediction and  $\mathcal{L}_{time}$  measures the mean square loss the of time prediction." (Section 3.1.2 Training)
- Inference: Input/Attention/State are inferred identically from the same sequence-attention-hidden-state architecture evidence (Section 3.1.1 and Section 3.1.2). In Dynamics is Open (inferred) from variable-length sequences and sequence-extension claims (Section 3.1.1; Section 3.3). Output is 0D Fixed (inferred) because the prediction head outputs one timestamp value per next-event prediction step.
