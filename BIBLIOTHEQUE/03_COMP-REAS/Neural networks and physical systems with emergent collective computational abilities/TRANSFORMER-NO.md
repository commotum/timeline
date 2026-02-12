# Neural networks and physical systems with emergent collective computational abilities (1982)
Source: Neural networks and physical systems with emergent collective computational abilities.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract describes an asynchronous neuron-state update procedure for content-addressable memory, not a Transformer/self-attention block architecture.
- Auxiliary analyses characterize the method as a recurrent binary network with static attention dynamics rather than learned self-attention.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "The algorithm for the time evolution of the state of the system is based on asynchronous parallel processing." (Abstract, `Neural networks and physical systems with emergent collective computational abilities.md`)
- "The paper describes a recurrent binary network used for content-addressable memory, including error-correcting recall, categorization, familiarity recognition, and generalization from partial cues." (`TASK-DOMAINS.md`, Summary)
- "Associative recall (content-addressable memory),Partial memory pattern / binary state vector,1D (t) (inferred),Fixed (inferred),Static (inferred),Constructed (inferred),Full memory pattern / binary state vector,1D (t) (inferred),Fixed (inferred)" (`TASK-DOMAINS.csv`, first row)
- "2. **Number of trained model instances required to cover all tasks:** 1" (`TASK_MODEL_RATIO.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for high-confidence TRANSFORMER-NO; no central Transformer/self-attention architecture signal in the abstract or auxiliary analyses.
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient; extending-dimensions file was unavailable (`MISSING`).
