# A Prototype for Data-Driven Visual Attention (Not specified in the paper.)
Source: A prototype for data-driven visual attention.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Salient region selection (visual attention) | digitized images / oriented edge or brightness response maps | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | salient regions / winning receptive field (pass zone) | 2D (x, y) (inferred) | Capped (inferred) |
| Priority ordering / scan path generation | digitized images / oriented edge or brightness response maps | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | ordered scan path of attended regions | 1D (t); 2D (x, y) (inferred) | Open (inferred) |

## Summary
The paper presents an early visual attention prototype that operates over spatial image inputs (e.g., digitized images with oriented edge or brightness responses) to select salient regions and to order them into scan paths. The task coverage is limited to bottom-up, data-driven attentional selection; no downstream recognition or classification tasks are specified. From the described hierarchy, WTA selection, and iterative beam shifts, the model supports dynamic attention and constructed internal state, with 2D spatial inputs/outputs and bounded receptive-field sizes plus potentially open-ended scan sequences.

## Evidence
### Task: Salient region selection (visual attention)
- "The goal of the prototype is to select salient or \"interesting\" regions in the input." (1 Introduction)
- "Simulations using high-resolution digitized images were conducted, with oriented edge information as the input to the model." (Abstract)
- "An inhibit zone and a pass zone are delineated for a beam that \"shines\" through all levels of the hierarchy." (3 The Attention Prototype)
- Inference: Labeled the input/output dimension as 2D (x, y) and input dynamics as Fixed because the experiments use "256 × 256 8-bit gray-scale images" and the model operates over pixel-based receptive fields; labeled output dynamics as Capped because "a minimum RF (minRF) and a maximum RF (maxRF) are specified" and "All rectangular RFs from minRF × minRF to maxRF × maxRF are computed"; labeled attention as Dynamic and state as Constructed because "an attention beam that traverses the hierarchy" uses WTA selection and "Each unit computes a weighted-sum of the responses from its input at the level below." (4 Experimental Results; 3 The Attention Prototype; Abstract)

### Task: Priority ordering / scan path generation
- "Aspects of attention such as localizing spatial regions of interest and ordering their importance are addressed." (Abstract)
- "Following the movement of the pass zone on the input layer for successive fixations produces scan paths... [that] prioritize the order in which parts of the image are assessed." (4 Experimental Results, Figure 3)
- Inference: Labeled the output dimension as 1D (t); 2D (x, y) because scan paths are an ordered sequence of spatial fixations ("Following the movement of the pass zone on the input layer for successive fixations produces scan paths"); labeled output dynamics as Open because Algorithm 1 specifies to "Do 3 through 8 forever." Also labeled attention as Dynamic and state as Constructed based on "Run WTA process at the current level" and "Each unit computes a weighted-sum of the responses from its input at the level below." (4 Experimental Results; 3 The Attention Prototype; Algorithm 1)
