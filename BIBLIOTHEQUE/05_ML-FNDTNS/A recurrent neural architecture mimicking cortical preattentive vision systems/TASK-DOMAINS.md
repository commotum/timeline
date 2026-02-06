# A recurrent neural architecture mimicking cortical preattentive vision systems (1996)
Source: A recurrent neural architecture mimicking cortical preattentive vision systems.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| texture segregation (texture discrimination) | texture images (pixel intensity images) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | texture segregation response map / feature-activity images | 2D (x, y) | Not specified in the paper. |

## Summary
The paper applies a recurrent cortical-inspired architecture to preattentive texture segregation/discrimination in images, demonstrated on natural and artificial textures. Inputs are 2D images, and outputs are 2D activity/segregation maps from the model layers. The description supports Static attention and Constructed state as inferred properties, while input/output size dynamics are not specified.

## Evidence
### Task: texture segregation (texture discrimination)
- "Applications to texture discrimination in both natural and artificial textures are presented." (Abstract)
- "In this paper, we present a three layer hierarchical neural network architecture and describe its application to texture segregation." (Section 1 Introduction)
- "where  $I(\mathbf{x})$  represents the intensity of the pixel at the point  $\mathbf{x} = (x_1, x_2)$  in the image plane  $\mathcal{I}$" (Section 3.2 The input layer)
- "The four orientation-channel outputs can be averaged to obtain an overall textural segregation response of the architecture (Fig. 6(b))." (Section 4 Results)
- "Each image represents the activity of neurons selective to a specific orientation, the luminous intensity of the pixel codes the activity of the corresponding neuron." (Section 4 Results)
- Inference: Attention Dynamic is Static and State Dynamic is Constructed because the model uses fixed receptive fields and intermediate representations: "Each neuron acts as a local operator for the features present in its receptive field" and "In each layer, there is a specific intermediate representation of the image:" (Section 3.1 General features; Abstract)
