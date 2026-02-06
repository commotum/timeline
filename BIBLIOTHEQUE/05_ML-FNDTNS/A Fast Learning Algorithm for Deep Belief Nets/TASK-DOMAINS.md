# A Fast Learning Algorithm for Deep Belief Nets (Not specified in the paper.)
Source: A Fast Learning Algorithm for Deep Belief Nets.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| generation (digit images and labels) | associative memory state | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | digit image (pixels); digit label (class) | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred); Fixed (inferred) |
| classification (digit recognition) | handwritten digit image (pixels) | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | digit label (class) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper describes a generative model over handwritten digit images and labels and reports digit classification performance on MNIST. The covered modalities are images and class labels, with fixed-size inputs/outputs implied by the fixed 784 visible units and 10-way label representation. The model relies on multi-layer hidden representations, and no dynamic or sequential attention mechanism is described.

## Evidence
### Task: generation (digit images and labels)
- "generative model of the joint distribution" (Opening paragraph)
- "generate an image" (Section 7)
- Inference: Inferred a fixed 1D latent input from the top-level associative-memory sampling used as input to lower layers and the fixed architecture sizes (Section 6, model description/table). Inferred 2D image and 0D label dimensions and fixed output sizes from the image/pixel domain and the 10-way label representation (Section 6). Inferred static attention and constructed state from the conclusion noting the lack of sequential attention and from the multi-layer/associative-memory description (Introduction, Figure 1 caption).

### Task: classification (digit recognition)
- "digit classification" (Opening paragraph)
- "pattern recognition performance" (Introduction)
- Inference: Inferred 2D fixed-size image inputs and 0D fixed label outputs from the MNIST image description, the 784-visible-unit architecture, and 10-way softmax labels (Section 6). Inferred static attention and constructed state from the same conclusion statement about no sequential attention and the multi-layer hidden/associative-memory description (Introduction, Figure 1 caption).
