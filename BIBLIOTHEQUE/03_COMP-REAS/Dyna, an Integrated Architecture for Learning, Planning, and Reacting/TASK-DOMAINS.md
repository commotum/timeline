# Dyna, an Integrated Architecture for Learning, Planning, and Reacting (Not specified in the paper)
Source: Dyna, an Integrated Architecture for Learning, Planning, and Reacting.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control / action selection | situation/state | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action | 0D (inferred) | Fixed (inferred) |
| transition & reward prediction (action model) | situation/state + action | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | predicted next situation/state + reward | 0D (inferred) | Fixed (inferred) |
| action-value prediction (Q-function) | state + action | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | action-value / Q(x,a) | 0D (inferred) | Fixed (inferred) |

## Summary
Dyna is presented as a reinforcement-learning architecture that learns a reactive policy, an action model, and action-value estimates. Tasks are defined over abstract states, actions, and rewards, without any specified sensory modality or spatial structure. The description implies per-step, point-like interfaces with fixed arity and reactive (static) attention, and it explicitly relies on learned internal models/value functions (constructed state).

## Evidence
### Task: control / action selection
- "Trial-and-error learning of an optimal reactive policy, a mapping from situations to actions;" (Section 1 Introduction to Dyna)
- "Observe the world's state and reactively choose an action based on it;" (Figure 2)
- Inference: 0D/Fixed/Static inputs and outputs are inferred because the interaction is described per discrete time step with a single situation and action. Constructed state is inferred because Dyna learns internal models/value functions. Supporting text: "At each discrete time interval, the agent observes a situation, takes an action based on it." (Section 1 Introduction to Dyna) and "learn an evaluation function that gives the value of performing each action in each state." (Section 2 Components of Dyna)

### Task: transition & reward prediction (action model)
- "an action model, a black box that takes as input a situation and action and outputs a prediction of the immediate next situation;" (Section 1 Introduction to Dyna)
- "it takes in a description of a state and an action and emits a prediction of the immediate resulting state and reward." (Section 2 Components of Dyna)
- Inference: 0D/Fixed/Static inputs and outputs are inferred because the model operates on a single state-action pair at each step. Constructed state is inferred because the action model is learned internal domain knowledge. Supporting text: "At each discrete time interval, the agent observes a situation, takes an action based on it." (Section 1 Introduction to Dyna) and "Learning of domain knowledge in the form of an action model" (Section 1 Introduction to Dyna)

### Task: action-value prediction (Q-function)
- "learn an evaluation function that gives the value of performing each action in each state." (Section 2 Components of Dyna)
- "This function is usually denoted Q(x, a), where x is a state and a is an action." (Section 2 Components of Dyna)
- Inference: 0D/Fixed/Static inputs and outputs are inferred because Q-learning is defined over individual state-action pairs per step. Constructed state is inferred because the evaluation function is learned internal state. Supporting text: "At each discrete time interval, the agent observes a situation, takes an action based on it." (Section 1 Introduction to Dyna) and "learn an evaluation function that gives the value of performing each action in each state." (Section 2 Components of Dyna)
