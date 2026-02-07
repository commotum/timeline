# Neural Program Meta-Induction (Not specified in the paper.)
Source: Neural Program Meta-Induction.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Program induction (input grid to output grid) | input grid (Karel world) | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | output grid | 2D (x, y) | Capped (inferred) |
| Meta program induction (I/O examples + eval input to output grid) | demonstration I/O grid pairs; eval input grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | output grid | 2D (x, y) | Capped (inferred) |

## Summary
The paper studies example-driven program induction in the Karel domain, predicting an output 2D grid from an input 2D grid. It also covers k-shot meta induction that conditions on a small set of I/O grid demonstrations plus a new input to produce the output grid. The benchmark uses variable-size grids within an explicit size range, and the described architectures use static processing; meta induction constructs a task representation (inferred) while plain induction is a direct mapping (inferred).

## Evidence
### Task: Program induction (input grid to output grid)
- "the model can take some new  $\hat{I}$  as input and emit the corresponding  $\hat{O}$ ." (Section 3 Plain Program Induction)
- "we attempt to directly generate the output grid  $\hat{O}$  from a corresponding input grid  $\hat{I}$ ." (Section 2 Karel Domain)
- "Karel the Robot moves around a 2D grid world" (Section 2 Karel Domain)
- "We sample I/O grids of size  $n \times m$ , where n and m are integers sampled uniformly from the range 2 to 20." (Section 8 Experimental Results)
- "The input encoder is a 3-layer CNN with a FC+relu layer on top." (Section 8 Experimental Results)
- Inference: In/Out Dynamics labeled Capped because grids are sampled from sizes 2 to 20. (Section 8 Experimental Results)
- Inference: Attention Dynamic labeled Static because the architecture processes the full grid with a CNN/LSTM and no runtime selection is described. (Section 8 Experimental Results)
- Inference: State Dynamic labeled Direct because the task is described as mapping input  $\hat{I}$  directly to output  $\hat{O}$  without an explicit task representation. (Section 3 Plain Program Induction)

### Task: Meta program induction (I/O examples + eval input to output grid)
- "our meta induction architecture takes as input a set of demonstration examples  $(I_1,O_1),...,(I_k,O_k)$  and an additional eval input  $\hat{I}$" (Section 5 Meta Program Induction)
- "and emits the corresponding output  $\hat{O}$ ." (Section 5 Meta Program Induction)
- "The number of demonstration examples k is typically small, e.g., 1 to 5." (Section 5 Meta Program Induction)
- "Karel the Robot moves around a 2D grid world" (Section 2 Karel Domain)
- "We sample I/O grids of size  $n \times m$ , where n and m are integers sampled uniformly from the range 2 to 20." (Section 8 Experimental Results)
- "the latent representation of a particular task is represented by conditioning on the training I/O examples for that task." (Section 5 Meta Program Induction)
- "Multiple I/O examples were combined with max-pooling on the final vector." (Section 8 Experimental Results)
- Inference: In/Out Dynamics labeled Capped because k is small (1 to 5) and grids are sampled from sizes 2 to 20. (Section 5 Meta Program Induction; Section 8 Experimental Results)
- Inference: Attention Dynamic labeled Static because demonstration examples are combined with max-pooling and no dynamic selection is described. (Section 8 Experimental Results)
- Inference: State Dynamic labeled Constructed because the model builds a latent task representation by conditioning on training I/O examples. (Section 5 Meta Program Induction)
