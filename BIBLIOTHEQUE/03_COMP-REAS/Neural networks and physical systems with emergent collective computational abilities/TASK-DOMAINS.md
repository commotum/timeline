# Neural networks and physical systems with emergent collective computational abilities (1982)
Source: Neural networks and physical systems with emergent collective computational abilities.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Associative recall (content-addressable memory) | Partial memory pattern / binary state vector | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Full memory pattern / binary state vector | 1D (t) (inferred) | Fixed (inferred) |
| Error correction (denoising recall) | Noisy memory cue / binary state vector | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Corrected memory pattern / binary state vector | 1D (t) (inferred) | Fixed (inferred) |
| Categorization (forced categorizer) | Initial state pattern / binary state vector | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Selected memory state / category (stable state) | 1D (t) (inferred) | Fixed (inferred) |
| Familiarity recognition (novelty detection) | Initial state pattern / binary state vector | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Unfamiliarity-indicating stable state (e.g. 0000...) | 1D (t) (inferred) | Fixed (inferred) |
| Generalization (completion from partial correlated state) | Partial new state X using k neurons | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Completed full state for all N neurons | 1D (t) (inferred) | Fixed (inferred) |
| Time sequence retention (sequence generation) | Current memory state (V_s) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Sequence of memory states (V_s -> V_{s+1}) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper describes a recurrent binary network used for content-addressable memory, including error-correcting recall, categorization, familiarity recognition, and generalization from partial cues. Inputs and outputs are fixed-length binary state vectors with asynchronous updates and stored memory vectors, implying static attention and constructed state (inferred). Time sequence retention is demonstrated by adding asymmetric terms that yield short, capped sequences.

## Evidence
### Task: Associative recall (content-addressable memory)
- "produce a content-addressable memory which correctly yields an entire memory from any subpart of sufficient size." (Abstract)
- "capable of retrieving this entire memory item on the basis of sufficient partial information." (Section The general content-addressable memory of a physical system)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed, attention as Static, and state as Constructed based on the fixed N-bit state representation, fixed update rule, and stored memory vectors (see quotes above).

### Task: Error correction (denoising recall)
- "An ideal memory could deal with errors and retrieve this reference even from the input \"Vannier, (1941)\"." (Section The general content-addressable memory of a physical system)
- "general (and error-correcting) content-addressable memory." (Section The general content-addressable memory of a physical system)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed, attention as Static, and state as Constructed based on the fixed N-bit state representation, fixed update rule, and stored memory vectors (see quotes above).

### Task: Categorization (forced categorizer)
- "The algorithm categorizes initial states according to the similarity to memory states." (Section Studies of the collective behaviors of the model)
- "will use its strong nonlinearity to make choices, produce categories, and regenerate information" (Section The information storage algorithm)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed, attention as Static, and state as Constructed based on the fixed N-bit state representation, fixed update rule, and stored memory vectors (see quotes above).

### Task: Familiarity recognition (novelty detection)
- "The 0000 state is then generated by any initial state that does not resemble adequately closely one of the assigned memories" (Section Studies of the collective behaviors of the model)
- "Familiar and unfamiliar states were distinguishable most of the time at this level of overload" (Section Studies of the collective behaviors of the model)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed, attention as Static, and state as Constructed based on the fixed N-bit state representation, fixed update rule, and stored memory vectors (see quotes above).

### Task: Generalization (completion from partial correlated state)
- "If now a partial new state X is stored" (Section Studies of the collective behaviors of the model)
- "using only k of the neurons rather than N, an attempt to reconstruct it will generate a stable point for all N neurons." (Section Studies of the collective behaviors of the model)
- "Some capacity for generalization is present" (Discussion)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed, attention as Static, and state as Constructed based on the fixed N-bit state representation, fixed update rule, and stored memory vectors (see quotes above).

### Task: Time sequence retention (sequence generation)
- "the system would spend a while near V, and then leave and go to a point near" (Section Studies of the collective behaviors of the model)
- "sequences longer than four states proved impossible to generate" (Section Studies of the collective behaviors of the model)
- "time ordering of memories can also be encoded." (Discussion)
- "represented by a binary word of N bits." (Section The model system)
- "each neuron randomly and asynchronously evaluates whether it is above or below threshold and readjusts accordingly." (Section The model system)
- "We can regard the information stored in the system as the vectors" (Section The general content-addressable memory of a physical system)
- Inference: Labeled dimensions/dynamics as 1D (t)/Fixed and attention as Static because the system is a fixed N-bit state with a fixed update rule; labeled state as Constructed due to stored memory vectors; labeled output dynamics as Capped because sequences longer than four states were impossible to generate (see quotes above).
