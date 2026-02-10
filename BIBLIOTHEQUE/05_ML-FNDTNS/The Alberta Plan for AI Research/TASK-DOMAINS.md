# The Alberta Plan for AI Research (Not specified in the paper)
Source: The Alberta Plan for AI Research.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continual supervised regression with given features | Infinite sequence of real-valued input vectors with fixed features | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Real-valued predicted outputs approximating desired outputs | 1D (t) (inferred) | Open (inferred) |
| Supervised feature finding in continual multi-task learning | Existing features and continual supervised target vectors | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Output vectors and newly constructed/ranked features | 1D (t) (inferred) | Open (inferred) |
| Continual GVF prediction learning | Sequential real-time process data with state, older signals, and traces | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed | Generalized value function predictions (including average-reward variants) | 1D (t) (inferred) | Open (inferred) |
| Continual actor-critic control | Bandit/contextual/sequential observations, rewards, and features | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | Actions/policies for control with continual learning | 1D (t) (inferred) | Open (inferred) |
| Model-based planning and search control for average-reward RL | States, actions/options, rewards, and learned environment models | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed | Planned policy/value improvements and predicted next-state/next-reward outcomes | 1D (t) (inferred) | Open (inferred) |
| Intelligence amplification of a partnered agent | Prediction/feature/policy signals from a Prototype-AI agent in interaction with another agent | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | Signals and policies that increase a second agent's speed and decision-making capacity | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper defines a reinforcement-learning-centered program spanning continual supervised learning, feature construction, GVF prediction, actor-critic control, model-based planning/search, and intelligence amplification. Across these tasks, the interaction is described as ongoing over time, which supports 1D (t) with Open dynamics. State treatment ranges from direct mappings in fixed-feature supervised learning to explicitly constructed state and abstractions (features, subtasks, options, models) in later steps. Dynamic attention is explicitly supported for planning/search-control updates, while several other tasks do not specify attention policy.

## Evidence
### Task: Continual supervised regression with given features
- "Step 1. Representation I: Continual supervised learning with given features." (Section Roadmap to an AI Prototype, Step 1)
- "In particular, we consider an infinite sequence of examples of desired behavior, each consisting of a real-valued input vector paired with a real-valued desired output." (Section Roadmap to an AI Prototype, Step 1)
- "The learner seeks to find an affine mapping from each input vector  $x_t$  to an output  $y_t$  that closely approximates the desired output  $y_t^*$ ." (Section Roadmap to an AI Prototype, Step 1)
- Inference: `1D (t)` and `Open` are inferred from the explicit time-indexed "infinite sequence" formulation; `Static` attention and `Direct` state are inferred from "static, given features" and affine input-output mapping without explicit state-construction mechanisms in Step 1.

### Task: Supervised feature finding in continual multi-task learning
- "Step 2. Representation II: Supervised feature finding." (Section Roadmap to an AI Prototype, Step 2)
- "This step is focused on creating and introducing new features (made by combining existing features) in the context of continual supervised learning as in Step 1, except now targets will be vectors  $\mathbf{y}_t^*$  approximated by output vectors  $\mathbf{y}_t$ ." (Section Roadmap to an AI Prototype, Step 2)
- "Getting each component of  $\mathbf{y}_t$  to match  $\mathbf{y}_t^*$  is referred to as a separate task." (Section Roadmap to an AI Prototype, Step 2)
- "Solution methods would presumably be, broadly speaking, some form of smart generation of promising features and then smart testing to rank and replace them." (Section Roadmap to an AI Prototype, Step 2)
- Inference: `1D (t)` and `Open` follow the continual time-indexed setup inherited from Step 1; `Dynamic` attention and `Constructed` state are inferred from runtime feature generation/testing/ranking/replacement that changes what representations are considered.

### Task: Continual GVF prediction learning
- "Step 3. Prediction I: Continual GVF prediction learning." (Section Roadmap to an AI Prototype, Step 3)
- "Repeat the above two steps for sequential, real-time settings where the data is not i.i.d., but rather is from a process with state and the task is generalized value function (GVF) prediction." (Section Roadmap to an AI Prototype, Step 3)
- "Here we explicitly address the question of constructing state, the *perception* part of the standard agent model" (Section Roadmap to an AI Prototype, Step 3)
- "Step 5. Prediction II: Average-reward GVF learning." (Section Roadmap to an AI Prototype, Step 5)
- Inference: `1D (t)` and `Open` are inferred from "sequential, real-time" and continuing prediction framing; output dimension/dynamics are inferred to match continual GVF prediction streams. Attention behavior is left as `Not specified in the paper.` because no explicit runtime information-selection mechanism is defined for this task.

### Task: Continual actor-critic control
- "Step 4. Control I: Continual actor-critic control." (Section Roadmap to an AI Prototype, Step 4)
- "First in a conventional k-arm bandit setting, then in a contextual bandit setting with discrete softmax actions, then in a sequential setting with given features, and finally in a sequential setting with feature finding." (Section Roadmap to an AI Prototype, Step 4)
- "Step 6. Control II: Continuing control problems." (Section Roadmap to an AI Prototype, Step 6)
- "The Alberta Plan characterizes the problem of AI as the online maximization of reward via continual sensing and acting, with limited computation, and potentially in the presence of other agents." (Section Research Vision: Intelligence as signal processing over time)
- Inference: `1D (t)` and `Open` are inferred from continual/continuing control over time; `Constructed` state is inferred because this control step explicitly builds on earlier state-construction/feature-finding steps. Attention is `Not specified in the paper.` for this task.

### Task: Model-based planning and search control for average-reward RL
- "Step 7. Planning I: Planning with average reward." (Section Roadmap to an AI Prototype, Step 7)
- "Develop incremental planning methods based on asynchronous dynamic programming for the average-reward criteria." (Section Roadmap to an AI Prototype, Step 7)
- "Prototype-AI 1 will include a) a recursive state-update (perception) process, b) a one-step environment model, presumably an expectation model or a sample model or something in-between, c) feature finding as in Step 2, utilizing importance feedback from the model, d) a ranking of features used both for feature finding and to determine which features are included in the environment model, e) an influence of model learning and planning on the feature ranking (a cycle), and f) some form of search control, possibly including something like MCTS or prioritized sweeping." (Section Roadmap to an AI Prototype, Step 8)
- "Viewed most generally, search control (varying the order of state updates) enables planning to radically change—from Monte Carlo Tree Search to classical heuristic search, for example." (Section Roadmap to an AI Prototype, Step 9)
- Inference: `1D (t)` and `Open` are inferred from temporally uniform continuing planning; `Dynamic` attention is inferred from explicit runtime control over state-update order in search control.

### Task: Intelligence amplification of a partnered agent
- "Step 12. Prototype-IA: Intelligence amplification." (Section Roadmap to an AI Prototype, Step 12)
- "A demonstration of intelligence applification (IA), wherein a Prototype-AI II agent is shown to increase the speed and overall decision-making capacity of a second agent in non-trivial ways." (Section Roadmap to an AI Prototype, Step 12)
- "We then see a second version that might be best thought of as a computational exo-cortex that fully manifests the ability of an IA agent to form policies and use planning to multiplicatively enhance the intelligence of another, partnered agent or part of a single agent." (Section Roadmap to an AI Prototype, Step 12)
- "We see a first version of this IA agent as what might be best described as a computational exo-cerebellum (a system built mainly on the prediction and continual feature construction elements of Oak and the steps above)." (Section Roadmap to an AI Prototype, Step 12)
- Inference: `1D (t)` and `Open` are inferred from ongoing agent-agent or human-agent interaction settings; `Constructed` state is inferred from explicit dependence on Oak's prediction and continual feature construction. Attention remains `Not specified in the paper.`
