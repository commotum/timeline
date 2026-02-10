# RETHINKING VISUAL INTELLIGENCE: INSIGHTS FROM VIDEO PRETRAINING (Not specified in the paper)
Source: Rethinking Visual Intelligence- Insights from Video Pretraining.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ARC-AGI abstract visual pattern transformation | 2D grid demonstrations and test grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted output grid | 2D (x, y) | Capped (inferred) |
| ConceptARC visual concept reasoning | 2D grid demonstrations and test grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted output grid | 2D (x, y) | Capped (inferred) |
| Hitori solving | 5x5 number grid | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | Solved Hitori grid | 2D (x, y) | Fixed |
| Sudoku solving (Mini + standard) | Partially filled Sudoku grid | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Completed Sudoku grid | 2D (x, y) | Capped (inferred) |
| Connect 4 winning-move prediction | Connect 4 board configuration | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Winning move in board-form output (inferred) | 2D (x, y) | Fixed (inferred) |
| Chess mate-in-1 prediction | Chess board configuration | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Checkmating move in board-form output (inferred) | 2D (x, y) | Fixed (inferred) |
| Maze path planning | Maze grid with start/goal cells | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Valid path from start to goal | 2D (x, y) | Capped (inferred) |
| Shortest Path planning | Grid with arbitrary start/goal cells | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Shortest valid path | 2D (x, y) | Capped (inferred) |
| Elementary Cellular Automata state prediction | 1D binary cell state with rule | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted binary cell state after evolution | 1D (t) | Capped (inferred) |
| Life-like Cellular Automata state prediction | 2D binary cell grid with B/S rule | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted binary grid state after evolution | 2D (x, y) | Capped (inferred) |
| Langton's ant state prediction | Binary grid with agent state | 2D (x, y) | Capped (inferred) | Static (inferred) | Direct (inferred) | Predicted complete grid state after n steps | 2D (x, y) | Capped (inferred) |
| Geometric transformation | Image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Geometrically transformed image | 2D (x, y) | Not specified in the paper. |
| Style transfer | Image with one reference style example | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Stylized image | 2D (x, y) | Not specified in the paper. |
| Inpainting | Masked/occluded image (inferred) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Completed image (inferred) | 2D (x, y) | Not specified in the paper. |
| Colorization | Grayscale image (inferred) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Color image (inferred) | 2D (x, y) | Not specified in the paper. |
| Jigsaw reconstruction | Shuffled image puzzle (inferred) | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Reconstructed image (inferred) | 2D (x, y) | Not specified in the paper. |
| Binary segmentation | Image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Binary segmentation mask (inferred) | 2D (x, y) | Not specified in the paper. |
| Pose estimation | Image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Pose estimate representation (inferred) | 2D (x, y) | Not specified in the paper. |
| Depth estimation | Image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Depth map (inferred) | 2D (x, y) | Not specified in the paper. |
| Image-to-segmentation (Chamber) | Image | 2D (x, y) | Not specified in the paper. | Static (inferred) | Direct (inferred) | Segmentation map (inferred) | 2D (x, y) | Not specified in the paper. |

## Summary
The paper covers a broad set of visually grounded tasks centered on grid transformations, games, route planning, and cellular automata, and additionally reports qualitative extension to classical image-to-image computer vision tasks. The task space is dominated by 2D (x, y) inputs/outputs, with an explicit 1D (t) case for Elementary Cellular Automata. Dynamics are mostly Fixed or Capped where grid/task-size constraints are explicit, while several appendix computer-vision tasks leave size constraints unspecified. Based on the described inference procedures, attention is static and state is direct across tasks (inferred).

## Evidence
### Task: ARC-AGI abstract visual pattern transformation
- "The ARC-AGI benchmark Chollet (2019) evaluates an agent's ability to infer and apply abstract patterns through compositional understanding, few-shot learning, and inductive generalization." (Section 4.1 ARC FAMILY)
- "Each ARC task provides only a handful of input-output examples (typically 2-5), requiring the model to discover the underlying transformation rule and apply it to novel test inputs." (Section 4.1 ARC FAMILY)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from variable but bounded few-shot setup ("2-5" examples) over benchmark grids; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures without retrieval/external state (Algorithm 1 and Algorithm 2).

### Task: ConceptARC visual concept reasoning
- "We evaluate models on ConceptARC Moskvichev et al. (2023), a curated variant of ARC designed to systematically measure visual concept understanding and generalization." (Section 4.1 ARC FAMILY)
- "ConceptARC groups tasks into 16 concept categories (for example, Above and Below, Center, Count), with each category containing 10 tasks." (Section 4.1 ARC FAMILY)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from finite benchmark task structure; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from the same fixed inference setup used across tasks.

### Task: Hitori solving
- "**Objective:** Eliminate cells so that each number appears at most once per row and column." (Section B.1.1 HITORI 5x5)
- "Figure 11: Example input-output pair for task *Hitori*." (Section B.1.1 HITORI 5x5)
- Inference: `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from the fixed I2I/JSON prediction pipeline with no runtime retrieval policy described.

### Task: Sudoku solving (Mini + standard)
- "**Objective:** Fill the grid so that all constraints are satisfied." (Section B.1.2 SUDOKU)
- "We evaluate two variants: *Mini Sudoku* (4x4 with 2x2 subgrids, see Figure 12) and *Sudoku* (9x9 with 3x3 subgrids, see Figure 13)." (Section B.1.2 SUDOKU)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` because the same task intent is evaluated at two explicit grid sizes (4x4 and 9x9); `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from the fixed prediction pipeline.

### Task: Connect 4 winning-move prediction
- "**Objective:** Place tokens to align four in a row." (Section B.1.3 CONNECT 4)
- "The board games, *Connect 4* and *Chess Mate-in-1*, shift attention to game scenarios where the goal is to identify the winning move in a given configuration." (Section 4.2.1 VISUAL GAMES)
- Inference: `In Dynamics = Fixed (inferred)` and `Out Dynamics = Fixed (inferred)` from standard fixed-board game formulation; `Output = Winning move in board-form output (inferred)` because all tasks are modeled as input-output grid pairs; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Chess mate-in-1 prediction
- "Objective: Deliver checkmate in a single move. Rules:" (Section B.1.4 CHESS MATE-IN-1)
- "A move is correct only if it results in an immediate checkmate of the opposing king." (Section B.1.4 CHESS MATE-IN-1)
- Inference: `In Dynamics = Fixed (inferred)` and `Out Dynamics = Fixed (inferred)` from standard fixed-board chess configuration; `Output = Checkmating move in board-form output (inferred)` because the paper uses grid input-output task formatting; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Maze path planning
- "In *Maze*, the model must navigate from the top-left to the bottom-right corner of a grid." (Section 4.2.2 ROUTE PLANNING)
- "**Objective:** Navigate from the start cell to the goal cell through a grid containing blocked and open positions." (Section B.2.1 MAZE)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from explicit 13x13/21x21 settings; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed prediction procedures.

### Task: Shortest Path planning
- "In *Shortest Path*, the objective is to connect two arbitrary points with the shortest possible route." (Section 4.2.2 ROUTE PLANNING)
- "**Objective:** Connect two arbitrary points with the shortest possible route." (Section B.2.2 SHORTEST PATH)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from finite grid benchmark setup; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Elementary Cellular Automata state prediction
- "Elementary Cellular Automata (ECA) are one-dimensional binary-state automata defined on a line of cells." (Section B.3.1 ELEMENTARY CELLULAR AUTOMATA)
- "Each cell  $c_i^t \in \{0,1\}$  at time t updates based on itself and its two neighbors:" (Section B.3.1 ELEMENTARY CELLULAR AUTOMATA)
- Inference: `Out Dimension = 1D (t)` and output future state prediction from the update equation and CA prediction framing; `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from finite evaluated rules/datasets; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Life-like Cellular Automata state prediction
- "Life-like CA generalize Conway's Game of Life Gardner (1970), using binary cells on a twodimensional grid." (Section B.3.2 LIFE-LIKE CELLULAR AUTOMATA)
- "For Life-like cellular automata, the VDM reaches threshold accuracy with far fewer examples" (Section 4.2.3 CELLULAR AUTOMATA)
- Inference: output future-state prediction is inherited from the CA evaluation framing in Section 4.2.3; `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from finite task settings; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Langton's ant state prediction
- "Additionally, we consider Langton's ant Langton (1986), a deterministic agent-based system, where the task is to predict the complete grid state after n steps of evolution." (Section 4.2.3 CELLULAR AUTOMATA)
- "Langton's ant Langton (1986) is an agent-based CA where a single agent moves on a binary grid." (Section B.3.3 LANGTON'S ANT)
- Inference: `In Dynamics = Capped (inferred)` and `Out Dynamics = Capped (inferred)` from explicit finite horizons ("prediction horizon of 2,3,5 and 10"); `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from fixed inference procedures.

### Task: Geometric transformation
- "Figure 31 illustrates that the model can capture geometric transformations under extreme few-shot conditions." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "Figure 31: Geometric transformations learned in few-shot setting." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `In Dimension = 2D (x, y)` and `Out Dimension = 2D (x, y)` from image-to-image framing; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged architecture/protocol in Section G.

### Task: Style transfer
- "We further show one-shot style transfer in Figure 32." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "Figure 32: 1-shot style transfer results. The model adapts the input images to distinct artistic styles (*Starry Night, Pixel Art, Cubism*, and *Ukiyo-e*) using only a single reference example." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Input = Image with one reference style example` from "using only a single reference example"; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from the same fixed I2I protocol.

### Task: Inpainting
- "Figure 33: Qualitative results for different tasks (*Inpainting*, *Colorization*, *Jigsaw*) with different numbers of training examples." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "While the main text emphasizes grid-structured visual prediction tasks, our framework extends naturally to a broad range of image-to-image problems." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Input = Masked/occluded image (inferred)` and `Output = Completed image (inferred)` from task name and I2I setup; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Colorization
- "Figure 33: Qualitative results for different tasks (*Inpainting*, *Colorization*, *Jigsaw*) with different numbers of training examples." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "our framework extends naturally to a broad range of image-to-image problems." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Input = Grayscale image (inferred)` and `Output = Color image (inferred)` from task name and I2I setup; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Jigsaw reconstruction
- "Figure 33: Qualitative results for different tasks (*Inpainting*, *Colorization*, *Jigsaw*) with different numbers of training examples." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "our framework extends naturally to a broad range of image-to-image problems." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Input = Shuffled image puzzle (inferred)` and `Output = Reconstructed image (inferred)` from task name and I2I setup; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Binary segmentation
- "In Figure 34 we show examples after training with only n=30 samples for *Binary Segmentation* for dogs and *Pose* estimation for humans." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "Figure 34: Predictions after finetuning with n=30 samples for *Binary Segmentation* and *Pose*." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Output = Binary segmentation mask (inferred)` from task name; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Pose estimation
- "In Figure 34 we show examples after training with only n=30 samples for *Binary Segmentation* for dogs and *Pose* estimation for humans." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "Figure 34: Predictions after finetuning with n=30 samples for *Binary Segmentation* and *Pose*." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Output = Pose estimate representation (inferred)` from task name; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Depth estimation
- "Figure 35: Predictions after finetuning with n=30 samples for *Depth*." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "These benchmarks cover a wide range of classical computer vision problems, from structured scene understanding to generative image transformation." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Output = Depth map (inferred)` from task name; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.

### Task: Image-to-segmentation (Chamber)
- "Figure 36: Examples from the  $Image \rightarrow Segmentation$  in 1-shot setting for Chamber." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- "our framework extends naturally to a broad range of image-to-image problems." (Section G EXPLORING GENERALIZATION OF I2I-TUNED VDMs)
- Inference: `Output = Segmentation map (inferred)` from the task label "Image \rightarrow Segmentation"; `Attention Dynamic = Static (inferred)` and `State Dynamic = Direct (inferred)` from unchanged protocol.
