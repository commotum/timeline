# A simple neural network module for relational reasoning (Not specified in the paper.)
Source: A simple neural network module for relational reasoning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Visual question answering (CLEVR, pixels) | Pixel images; natural-language questions | 2D (x, y) (inferred); 1D (t) (inferred) | Fixed (inferred); Not specified in the paper. | Static (inferred) | Direct (inferred) | Answer (from answer vocabulary) | 0D (inferred) | Fixed (inferred) |
| Visual question answering (CLEVR, state descriptions) | State description matrix of objects (3D coordinates, color, shape, material, size); natural-language questions | 3D (x, y, z) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Answer (from answer vocabulary) | 0D (inferred) | Fixed (inferred) |
| Visual question answering (Sort-of-CLEVR) | Images of 2D colored shapes; fixed-length binary questions | 2D (x, y) (inferred); 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Answer (shape/count/position categories) | 0D (inferred) | Fixed (inferred) |
| Text-based question answering (bAbI) | Text stories (supporting sentences) and question | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer (from answer vocabulary) | 0D (inferred) | Fixed (inferred) |
| Connection inference in dynamic physical systems | State description matrices of balls with color and (x,y) coordinates across 16 time steps | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Connection existence between ball pairs (binary vector length 10^2) | 1D (t) (inferred) | Fixed (inferred) |
| Counting connected systems in dynamic physical systems | State description matrices of balls with color and (x,y) coordinates across 16 time steps | 3D (x, y, t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Count of connected systems (one-hot length 10) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Relation Networks on visual question answering tasks (CLEVR pixels, CLEVR state descriptions, and Sort-of-CLEVR), text-based QA (bAbI), and dynamic physical-system reasoning with connection inference and counting. Inputs span 2D images with 1D question sequences, 3D coordinate-based state descriptions (including spatiotemporal x,y,t trajectories), and 1D text sequences, while outputs are fixed-vocabulary answers or fixed-length classification targets (connection vectors and counts). Where specified, input sizes are fixed or capped (e.g., fixed image sizes, fixed-length binary questions, up to 20 support sentences), and RN computation considers all object pairs, supporting static attention; state dynamics are treated as direct mappings in this analysis.

## Evidence
### Task: Visual question answering (CLEVR, pixels)
- "In visual QA a model must learn to answer questions about an image (Figure 1)." (Section 3.1 CLEVR)
- "the pixel version, in which images were represented in standard 2D pixel form" (Section 3.1 CLEVR)
- "The CNN took images of size  $128 \times 128$ " (Section 4 Models)
- "At each time-step, the LSTM received a single word embedding as input" (Section 4 Models)
- "The final layer was a linear layer that produced logits for a softmax over the answer vocabulary." (Section 4 Models)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension uses 2D for pixel images and 1D for word sequences; In Dynamics fixed for images from the 128x128 size statement; Attention Static and State Direct inferred from RN all-pairs computation. (Sections 2, 4)

### Task: Visual question answering (CLEVR, state descriptions)
- "a state description version, in which images were explicitly represented by state description matrices containing factored object descriptions." (Section 3.1 CLEVR)
- "3D coordinates (x, y, z); color (r, g, b); shape (cube, cylinder, etc.); material (rubber, metal, etc.); size (small, large, etc.)." (Section 3.1 CLEVR)
- "In visual QA a model must learn to answer questions about an image (Figure 1)." (Section 3.1 CLEVR)
- "At each time-step, the LSTM received a single word embedding as input" (Section 4 Models)
- "The final layer was a linear layer that produced logits for a softmax over the answer vocabulary." (Section 4 Models)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension uses 3D for object coordinates and 1D for word sequences; Attention Static and State Direct inferred from RN all-pairs computation; output treated as fixed-vocabulary label. (Sections 2, 3.1, 4)

### Task: Visual question answering (Sort-of-CLEVR)
- "Sort-of-CLEVR consists of images of 2D colored shapes along with questions and answers about the images." (Section 3.2 Sort-of-CLEVR)
- "Questions are hard-coded as fixed-length binary strings" (Section 3.2 Sort-of-CLEVR)
- "The Sort-of-CLEVR dataset contains 10000 images of size  $75 \times 75$ " (Supplementary D Sort-of-CLEVR)
- "An additional final linear layer produced logits for a softmax over the possible answers." (Supplementary D Sort-of-CLEVR)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension uses 2D for images and 1D for binary question strings; In Dynamics fixed from 75x75 images and fixed-length questions; Attention Static and State Direct inferred from RN all-pairs computation. (Sections 2, 3.2, Supplementary D)

### Task: Text-based question answering (bAbI)
- "bAbI is a pure text-based QA dataset" (Section 3.3 bAbI)
- "Each question is associated with a set of supporting facts." (Section 3.3 bAbI)
- "we first identified up to 20 sentences in the support set" (Section 4 Models)
- "processed each sentence word-by-word with an LSTM" (Section 4 Models)
- "The final layer was a linear layer that produced logits for a softmax over the answer vocabulary." (Supplementary E bAbI model for language understanding)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension is 1D from word-by-word sentence processing; In Dynamics capped from "up to 20 sentences"; Attention Static and State Direct inferred from RN all-pairs computation. (Sections 2, 4)

### Task: Connection inference in dynamic physical systems
- "Each scene contained 10 colored balls moving on a table-top surface." (Section 3.4 Dynamic physical systems)
- "Input data consisted of state descriptions matrices, where each ball was represented as a row in a matrix" (Section 3.4 Dynamic physical systems)
- "their spatial coordinates (x,y) across 16 sequential time steps." (Section 3.4 Dynamic physical systems)
- "We defined two separate tasks: 1) infer the existence or absence of connections between balls" (Section 3.4 Dynamic physical systems)
- "For the connection inference task the targets were binary vectors representing the existence (or non-existence) of a connection between each ball pair." (Supplementary F Dynamic physical system reasoning)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension set to 3D (x,y,t) from spatial coordinates across time; In Dynamics fixed from 10 balls and 16 time steps; Out Dimension 1D from binary vectors; Attention Static and State Direct inferred from RN all-pairs computation. (Sections 2, 3.4, Supplementary F)

### Task: Counting connected systems in dynamic physical systems
- "Each scene contained 10 colored balls moving on a table-top surface." (Section 3.4 Dynamic physical systems)
- "Input data consisted of state descriptions matrices, where each ball was represented as a row in a matrix" (Section 3.4 Dynamic physical systems)
- "their spatial coordinates (x,y) across 16 sequential time steps." (Section 3.4 Dynamic physical systems)
- "2) count the number of systems on the table-top" (Section 3.4 Dynamic physical systems)
- "For the counting task, the targets were one-hot vectors (of length 10) indicating the number of systems of connected balls." (Supplementary F Dynamic physical system reasoning)
- "The functional form in Equation 1 dictates that an RN should consider the potential relations between all object pairs." (Section 2 Relation Networks)
- Inference: In Dimension set to 3D (x,y,t) from spatial coordinates across time; In Dynamics fixed from 10 balls and 16 time steps; Output treated as scalar count (0D) from one-hot targets; Attention Static and State Direct inferred from RN all-pairs computation. (Sections 2, 3.4, Supplementary F)
