# Temporal-Difference Networks (Not specified in the paper.)
Source: Temporal-Difference Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| n-step unconditional prediction | Action-observation experience in the random walk (including state-indicator bits and a special terminal bit) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Predicted value/probability of the special observation bit n steps ahead | 1D (t) (inferred) | Fixed (inferred) |
| n-step action-conditional prediction | Action-observation experience with left/right action selections | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Predicted value/probability of the special observation bit conditional on future action sequences | 1D (t) (inferred) | Fixed (inferred) |
| Learning a predictive state representation for non-Markov prediction | Partially observable action-observation experience (only special observation bit visible) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Action-conditional n-step predictions used as predictive state representation | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper focuses on temporal prediction tasks over sequential agent-environment interaction data, including unconditional and action-conditional future-observation prediction. Across experiments, the task domain is temporal sequence processing (1D (t)) over ongoing experience streams, with fixed-size prediction heads determined by the question-network depth/length in each setup. The model behavior described in the OCR supports static attention over predefined features, while state is direct in fully observable settings and constructed in the non-Markov setting where prior predictions are fed back as features.

## Evidence
### Task: n-step unconditional prediction
- "In this experiment we sought to predict the observation bit precisely n steps in advance, for n=1, 2, 5, 10, and 25." (Section 3, Experiment 1: n-step Unconditional Prediction)
- "This is specified by a TD network consisting of a single chain of predictions like the left column of Figure 1a, but of length 25 rather than 3." (Section 3, Experiment 1: n-step Unconditional Prediction)
- Inference: 1D (t), Open, Static, Direct, and output/Out Dynamics labels are inferred from the described discrete-time experience stream ("At each of a series of discrete time steps t ..." in Section 1), continuing interaction ("This is a continuing task: reaching an end state does not end or interrupt experience." in Section 1), and fixed feature design for this experiment ("Both algorithms used feature vectors of 7 binary components..." in Section 3), which indicates predefined runtime inputs and no constructed latent state in this setting.

### Task: n-step action-conditional prediction
- "In a second experiment we sought to learn n-step-ahead predictions conditional on action selections." (Section 4, Experiment 2: Action-conditional Prediction)
- "The lower four nodes correspond to the two-step predictions, e.g., the second lower node is the prediction of what the observation bit will be if an L action is taken followed by an R action." (Section 4, Experiment 2: Action-conditional Prediction)
- Inference: 1D (t), Open, Static, Direct, and output/Out Dynamics labels are inferred from the same discrete-time ongoing experience framing in Section 1 and the fixed-depth, fixed-node question network in this experiment ("...except of depth four, consisting of 30 (2+4+8+16) nodes." in Section 4), with predefined conditions rather than adaptive information selection.

### Task: Learning a predictive state representation for non-Markov prediction
- "In Experiment 3, on the other hand, we applied TD networks to a non-Markov version of the random-walk example, in particular, in which only the special observation bit was visible and not the state number." (Section 5, Experiment 3: Learning a Predictive State Representation)
- "In this case it is not possible to make accurate predictions based solely on the current action and observation; the previous time step's predictions must be used as well." (Section 5, Experiment 3: Learning a Predictive State Representation)
- Inference: 1D (t), Open, Static, Constructed, and output/Out Dynamics labels are inferred from the sequential formulation in Section 1 and the explicit reuse of prior predictions as input features in Section 5 ("...and n more features corresponding to the components of y_{t-1}."), which supports a constructed-state interpretation with fixed per-network output size.
