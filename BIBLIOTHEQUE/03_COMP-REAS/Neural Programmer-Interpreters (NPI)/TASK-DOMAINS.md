# NEURAL PROGRAMMER-INTERPRETERS (Not specified in the paper)
Source: Neural Programmer-Interpreters (NPI).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Addition | digits of two base-10 numbers on a scratch pad | 2D (x, y) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | digits of the answer (scratch pad) | 2D (x, y) (inferred) | Capped (inferred) |
| Sorting (bubblesort) | array of numbers (scratch pad) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | sorted array (ascending order) | 1D (t) (inferred) | Capped (inferred) |
| Canonicalizing 3D models | car rendering pixels + target pose (azimuth, elevation) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | trajectory of camera actions to reach target view | 1D (t) (inferred) | Capped (inferred) |
| Maximum-finding (MAX) | array of numbers (scratch pad) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | maximum element (rightmost after sort) | 0D (inferred) | Fixed (inferred) |

## Summary
Across experiments, NPI is evaluated on algorithmic manipulation tasks (addition, bubblesort sorting, and maximum-finding) over numeric arrays, and a vision-based canonicalization task for 3D car pose. Inputs span 1D arrays on scratch pads and 2D images plus pose targets, while outputs are either array values or action trajectories. The paper supports variable-length array interfaces with pointer-driven dynamic attention, and fixed-size image observations with static attention to a read-only pose pad. All tasks use a recurrent core with program memory, implying constructed state.

## Evidence
### Task: Addition
- "The task in this environment is to read in the digits of two base-10 numbers and produce the digits of the answer." (Section 4.1 ADDITION)
- "There are four pointers; one for each of the two input numbers, one for the carry, and another to write the output." (Section 4.1 ADDITION)
- "The first dimension of Q corresponds to scratch pad rows, N is the number of columns (digits) and K is the one-hot encoding dimension." (Section 4.1 ADDITION)
- "At each time step, a pointer can be moved left or right, or it can record a value to the pad." (Section 4.1 ADDITION)
- "problem lengths 1,...,20" (Section 6.3 Additional experiment on addition generalization)
- "NPI has three learnable components: a task-agnostic recurrent core, a persistent key-value program memory, and domain-specific encoders." (Abstract)
- Inference: In/Out Dimension and Dynamics are inferred from the scratch pad having rows and columns and the stated problem lengths (Sections 4.1, 6.3). Attention Dynamic is inferred from pointer movement (Section 4.1 ADDITION). State Dynamic is inferred from persistent program memory (Abstract).

### Task: Sorting (bubblesort)
- "sorting an array of numbers using bubblesort." (Section 4.1 SORTING)
- "N is the array length" (Section 4.1 SORTING)
- "arrays of single-digit numbers from length 2 to length 20." (Section 4.2 Sample complexity and generalization)
- "a 1-D array with read-only pointers and a swap action" (Section 3 Model)
- "BUBBLESORT | Perform bubble sort (ascending order)" (Section 6.1 Listing of learned programs)
- "NPI has three learnable components: a task-agnostic recurrent core, a persistent key-value program memory, and domain-specific encoders." (Abstract)
- Inference: In/Out Dimension and Dynamics are inferred from the array length/variable-length description and 1-D array environment (Sections 3, 4.1, 4.2). Attention Dynamic is inferred from pointer-based access (Section 3 Model). State Dynamic is inferred from persistent program memory (Abstract).

### Task: Canonicalizing 3D models
- "Given a rendering of a 3D car, we would like to learn a visual program that "canonicalizes" the model with respect to its pose." (Section 4.1 CANONICALIZING 3D MODELS)
- "the program should generate a trajectory of actions that delivers the camera to the target view" (Section 4.1 CANONICALIZING 3D MODELS)
- "a very simple read-only pad that only contains a target camera elevation and azimuth" (Section 4.1 CANONICALIZING 3D MODELS)
- "renderings of size  $128 \times 128$ ." (Section 4.2 Sample complexity and generalization)
- "up to four-step trajectories for canonicalization." (Section 4.4 SOLVING MULTIPLE TASKS WITH A SINGLE NETWORK)
- "i_1, i_2  are the (fixed at 1) pointer locations" (Section 4.1 CANONICALIZING 3D MODELS)
- "NPI has three learnable components: a task-agnostic recurrent core, a persistent key-value program memory, and domain-specific encoders." (Abstract)
- Inference: Input Dimension and Fixed In Dynamics are inferred from image pixels and fixed rendering size (Sections 4.1, 4.2). Attention Dynamic is inferred as Static because observation uses full images and fixed pad pointers (Section 4.1). Output Dimension/Dynamics are inferred from the trajectory-of-actions description and capped trajectory length (Sections 4.1, 4.4). State Dynamic is inferred from persistent program memory (Abstract).

### Task: Maximum-finding (MAX)
- "adding a maximum-finding program MAX to a multitask NPI trained on addition, sorting and canonicalization." (Section 4.3 Learning New Programs with a fixed core)
- "RJMP, which moves pointers to the right of the sorted array, where the max element can be read." (Section 4.3 Learning New Programs with a fixed core)
- "MAX        | Find maximum element of an array" (Section 6.1 Listing of learned programs)
- "a 1-D array with read-only pointers and a swap action" (Section 3 Model)
- "NPI has three learnable components: a task-agnostic recurrent core, a persistent key-value program memory, and domain-specific encoders." (Abstract)
- Inference: Input Dimension and Dynamics are inferred from array-based environment and pointer access (Section 3 Model; Section 4.3). Attention Dynamic is inferred from pointer-based access (Section 3 Model). Output Dimension/Dynamics are inferred from "Find maximum element of an array" as a single value (Section 6.1). State Dynamic is inferred from persistent program memory (Abstract).
