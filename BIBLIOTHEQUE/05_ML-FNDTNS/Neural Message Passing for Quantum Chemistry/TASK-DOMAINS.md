# Neural Message Passing for Quantum Chemistry (2017)
Source: Neural Message Passing for Quantum Chemistry.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Molecular property prediction (regression of QM9 quantum properties) | molecular graphs (atom and bond information); optional molecular geometry (atomic distances, bond angles) | 3D (x, y, z) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | quantum mechanical molecular properties (13 targets / regression values) | 0D (inferred) | Fixed |

## Summary
The paper focuses on supervised molecular property prediction on the QM9 benchmark, framed as 13 regression targets per molecule. Inputs are molecular graphs, and the authors consider both full molecular geometry (atomic distances, bond angles) and topology-only graphs, which supports a 3D spatial input dimension (inferred) while input dynamics are not explicitly bounded. Outputs are scalar quantum property values (0D, fixed), and the model constructs internal node states via message passing with fixed neighbor aggregation (constructed state and static attention, inferred).

## Evidence
### Task: Molecular property prediction (regression of QM9 quantum properties)
- "predicting the quantum mechanical properties of small organic molecules" (Section 1. Introduction)
- "QM9 consists of 130k molecules with 13 properties for each molecule which are approximated by an expensive quantum mechanical simulation method (DFT), to yield 13 corresponding regression tasks." (Section 1. Introduction)
- "QM9 therefore lets us consider both the setting where the complete molecular geometry is known (atomic distances, bond angles, etc.) and the setting where we need to compute properties that might still be defined in terms of the spatial positions of atoms, but where only the atom and bond information (i.e. graph) is available as input." (Section 1. Introduction)
- "During the message passing phase, hidden states  $h_v^t$  at each node in the graph are updated based on messages  $m_v^{t+1}$  according to" (Section 2. Message Passing Neural Networks)
- Inference: Labeled input dimension as 3D (x, y, z) because the paper describes molecular geometry and spatial positions (atomic distances, bond angles) as available input; labeled attention as Static and state as Constructed because message passing aggregates over fixed graph neighbors and updates hidden node states; labeled output dimension as 0D because the targets are scalar molecular properties in the QM9 regression tasks. (Section 1. Introduction; Section 2. Message Passing Neural Networks)
