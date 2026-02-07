# OptNet: Differentiable Optimization as a Layer in Neural Networks (2017)
Source: OptNet- Differentiable Optimization as a Layer in Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Denoising | noisy 1D signal | 1D (t) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | denoised signal (clean signal) | 1D (t) (inferred) | Not specified in the paper. |
| Sudoku solving (constraint satisfaction) | 4x4 grid (4x4x4 one-hot tensor) | 3D (x, y, z) | Fixed | Not specified in the paper. | Not specified in the paper. | 4x4x4 one-hot tensor solution | 3D (x, y, z) | Fixed |
| MNIST classification (inferred) | MNIST examples (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class prediction (inferred) | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper demonstrates OptNet on 1D signal denoising and 4x4 Sudoku solving, with an additional MNIST experiment described but without explicit input/output modality details. The denoising task operates on 1D signals, while Sudoku uses fixed-size 4x4x4 tensors; dynamics are otherwise mostly unspecified. Attention and state dynamics are not explicitly characterized in the paper.

## Evidence
### Task: Denoising
- "denoise a noisy 1D signal" (Section 4.2. Total variation denoising)
- "piecewise constant signals (which are the desired outputs of the learning algorithm)" (Section 4.2. Total variation denoising)
- "independent Gaussian noise (which form the inputs to the learning algorithm)" (Section 4.2. Total variation denoising)
- Inference: Labeled the output dimension as 1D (t) because the outputs are signals in the same denoising setup as the 1D input signal.

### Task: Sudoku solving (constraint satisfaction)
- "the task of learning the game of Sudoku." (Section 4.4. Sudoku)
- "Sudoku is fundamentally a constraint satisfaction problem," (Section 4.4. Sudoku)
- "input to the algorithm consists of a 4x4 grid (really a 4x4x4 tensor" (Section 4.4. Sudoku)
- "desired output is a 4x4x4 tensor of the one-hot encoding of the solution." (Section 4.4. Sudoku)

### Task: MNIST classification (inferred)
- "integration of QP OptNet layers into a traditional fully connected network for the MNIST problem." (Section A. MNIST Experiment)
- "FC600-FC10-SoftMax fully connected network" (Section A. MNIST Experiment)
- Inference: Treated the MNIST experiment as classification with class predictions because the architecture ends with a SoftMax layer and a 10-unit output.
