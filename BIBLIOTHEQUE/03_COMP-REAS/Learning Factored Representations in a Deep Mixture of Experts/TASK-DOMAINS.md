# Learning Factored Representations in a Deep Mixture of Experts (Not specified in the paper.)
Source: Learning Factored Representations in a Deep Mixture of Experts.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Digit classification (jittered MNIST) | grayscale images (36 x 36) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | digit class labels (10 classes) | 0D (inferred) | Fixed (inferred) |
| Monophone phoneme classification | speech samples (11 frames, 40 frequency bins) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | phoneme class labels (40 classes) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates a deep mixture-of-experts model on two classification tasks: digit classification from jittered MNIST images and monophone phoneme classification from time-frequency speech samples. Inputs are fixed-size 2D grids (image pixels; time-by-frequency bins), while outputs are fixed sets of class labels, implying 0D outputs. Attention and state dynamics are not explicitly defined, but the fixed-size inputs suggest static attention, and the multi-layer expert/gating architecture implies constructed internal representations.

## Evidence
### Task: Digit classification (jittered MNIST)
- "grayscale images of size  $36 \times 36$ ." (Section 4.1 Jittered MNIST)
- "the model was trained to classify digits into ten classes." (Section 4.1 Jittered MNIST)
- Inference: In/Out Dimension and Dynamics inferred from fixed image size and fixed class set ("grayscale images of size  $36 \times 36$ ."; "classify digits into ten classes"). Attention Dynamic set to Static (inferred) because no runtime input selection is described for these fixed-size inputs. State Dynamic set to Constructed (inferred) because the model uses layered expert networks ("We set each  $f_i^l$  to a single linear map with rectification"). (Section 3 Approach; Section 4.1 Jittered MNIST)

### Task: Monophone phoneme classification
- "each sample was limited to 11 frames spaced 10ms apart, and had 40 frequency bins." (Section 4.2 Monophone Speech)
- "There were 40 possible output phoneme classes." (Section 4.2 Monophone Speech)
- Inference: In Dimension and In Dynamics inferred from fixed 11-frame, 40-bin inputs ("each sample was limited to 11 frames spaced 10ms apart, and had 40 frequency bins."); Out Dimension and Out Dynamics inferred from fixed class set ("There were 40 possible output phoneme classes."). Attention Dynamic set to Static (inferred) because no runtime input selection is described for these fixed-size inputs. State Dynamic set to Constructed (inferred) because the model uses layered expert networks ("We set each  $f_i^l$  to a single linear map with rectification"). (Section 3 Approach; Section 4.2 Monophone Speech)
