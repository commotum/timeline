## 1. Basic Metadata

- Title: "A 2D nGPT Model For Arc Prize" (Title)
- Authors: "Jean-Francois Puget, NVIDIA" (Title)
- Year: "November 2024" (Title)
- Venue (conference/journal/arXiv): "ARC Prize 2024 competition on Kaggle" (Abstract)

---

## 2. One-Sentence Contribution Summary

The paper presents a solution to the "ARC Prize 2024 competition on Kaggle" using "a tiny 2D transformer trained from scratch" with "task specific test time training" to predict ARC grid outputs from inputs. (Abstract, Section 1 Introduction)

---

## 3. Tasks Evaluated

### Task 1: ARC constant-size grid-to-grid transformation (ARC Prize 2024 / ARC-AGI)

- Task type: Generation; Other (grid-to-grid transformation)
- Dataset(s) used: ARC Prize 2024 competition tasks (public train tasks, public evaluation tasks, hidden test tasks); augmented with re-arc generator pairs.
- Domain: Synthetic colored grids (up to 30x30)

Evidence (task definition and evaluation context):
- "The competition is about learning how to solve graphical tasks like the one in figure 1. For each task, a number of training examples are provided. For each training sample we are given an input and a corresponding output." (Section 1 Introduction)
- "Inputs and outputs are colored grids of dimension up to 30x30. There are 10 colors." (Section 1 Introduction)
- "The problem is to compute the output of the test samples based on the transformation that was used in the training samples." (Section 1 Introduction)
- "About two thirds of the ARC tasks have constant size grids. In these tasks, the size of the output is always the same as the size of the input." (Section 1 Introduction)
- "We decided to focus on solving constant size tasks." (Section 1 Introduction)
- "We can also evaluate its effectiveness using the public evaluation tasks." (Section 5 TTT)
- "We generated 10k input/output pairs for each training tasks by using Michael Hodel's re-arc generator." (Section 4 Invertible Transformations)

---

## 4. Domain and Modality Scope

- Evaluation performed on a single domain (synthetic colored grids). Evidence: "Inputs and outputs are colored grids of dimension up to 30x30. There are 10 colors." (Section 1 Introduction)
- Multiple domains within the same modality: Not indicated.
- Multiple modalities: Not indicated.
- Domain generalization or cross-domain transfer claimed: Not claimed. The paper notes distribution shift within the same domain: "The hidden test tasks are different from all the public tasks, and using models pretrained on public tasks isn't effective." (Section 1 Introduction)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ARC constant-size grid-to-grid transformation | Yes | Yes (TTT retraining) | Not specified (single decoding layer described) | "The model has an embedding layer that turns tasks ids into vectors that are added to all color embeddings." (Section 3 Task Specific Modeling) "We therefore reinitialize the task embedding layer with random weights before performing TTT." (Section 5 TTT) "The architecture is the same as a LLM: ... a decoding layer producing color logits." (Section 2 A 2D nGPT Model for Constant Size Tasks) |

---

## 6. Input and Representation Constraints

- Grid-based inputs/outputs: "Inputs and outputs are colored grids of dimension up to 30x30. There are 10 colors." (Section 1 Introduction)
- Constant-size focus (output size equals input size for this subset): "About two thirds of the ARC tasks have constant size grids. In these tasks, the size of the output is always the same as the size of the input." (Section 1 Introduction)
- 2D representation: "it works with 2D grids rather than 1D sequences." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Token definition: "Tokens are replaced by grid cell color indices." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Padding requirement: "There are 10 colors in ARC tasks, plus an 11th color for padding grids to the same dimension when batching." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Padding masked in attention: "Padded areas are masked." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified (only grid size up to 30x30 is stated).

---

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified; inputs are "grids of dimension up to 30x30." (Section 1 Introduction)
- Sequence length fixed or variable: Not explicitly stated; constant-size tasks have fixed input/output sizes within each task: "the size of the output is always the same as the size of the input." (Section 1 Introduction)
- Attention type: Sparse/axial 2D attention (row and column). Evidence: "The attention layers are 2D attention. A grid cell attends to all cells in same row, and it attends to all cell in same column." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Additional attention variant attempted: "We also implemented a single attention where every grid cell can attend every other grid cell, with a bias that depends on the distance between cells." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Mechanisms for computational cost: Row/column attention and masking padded areas. Evidence: "Each of these attention is a 1D attention and is implemented using pytorch masked scaled dot product attention. Padded areas are masked." (Section 2 A 2D nGPT Model for Constant Size Tasks)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: RoPE. Evidence: "We also use rotary positional embeddings (ROPE) on each row and on each columns, reusing the ROPE implementation for 1D sequence." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Where applied: On each row and column in the attention layers. Evidence: "We also use rotary positional embeddings (ROPE) on each row and on each columns." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Fixed across all experiments / modified per task / ablated: Not specified (no comparisons reported).

---

## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: Fixed architectural assumption; only RoPE is described. Evidence: "We also use rotary positional embeddings (ROPE) on each row and on each columns." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Multiple positional encodings compared: Not specified.
- Claim that PE choice is not critical or secondary: Not specified.

---

## 10. Evidence of Constraint Masking

- Model size(s): "a tiny (42M parameters) transformer trained from scratch" (Abstract); "42.5M trainable parameters." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Dataset size(s): "there are only 262 constant size tasks, and about 1000 input/output pairs." (Section 3 Task Specific Modeling)
- Data scaling via generated pairs: "We generated 10k input/output pairs for each training tasks... The resulting 2.6M pairs." (Section 4 Invertible Transformations)
- Data scaling via invertible transformations: "This is about 335 million samples for N=16." (Section 4 Invertible Transformations)
- Performance gains attributed to training tricks/augmentation: "We got about 55 percents accuracy on the original tasks." (Section 4 Invertible Transformations); "Merging logits moved accuracy on original tasks to about 80 percents." (Section 4 Invertible Transformations); "Using testing time training it could reach 26 percent accuracy on the constant size evaluation tasks." (Section 6 Conclusion)
- Emphasis on augmentation for TTT: "Invertible transformations are key for TTT as well." (Section 5 TTT)

---

## 11. Architectural Workarounds

- Row/column attention to adapt transformers to 2D grids: "The attention layers are 2D attention. A grid cell attends to all cells in same row, and it attends to all cell in same column." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Task embedding to handle task-dependent mappings: "The model has an embedding layer that turns tasks ids into vectors that are added to all color embeddings." (Section 3 Task Specific Modeling)
- Padding plus masking for batching: "an 11th color for padding grids to the same dimension when batching." / "Padded areas are masked." (Section 2 A 2D nGPT Model for Constant Size Tasks)
- Fixed grid assumption (constant-size tasks): "We decided to focus on solving constant size tasks." (Section 1 Introduction)
- Alternative global attention with distance bias (attempted): "We also implemented a single attention where every grid cell can attend every other grid cell, with a bias that depends on the distance between cells." (Section 2 A 2D nGPT Model for Constant Size Tasks)

---

## 12. Explicit Limitations and Non-Claims

- Not competitive / not submitted: "Not only this model is not competitive with best models for the competition, but due to lack of time it was not possible to successfully submit it to the competition." (Abstract)
- Scope limited to constant-size tasks: "We decided to focus on solving constant size tasks. It remains to be seen if our method can be adapted to tasks where output size differs from input size." (Section 1 Introduction)
- Failure to report hidden test results: "We could submit the TTT code only the last afternoon of the competition and it failed." / "As a result we cannot report figures on the hidden test data." (Section 5 TTT)
- High variance in TTT accuracy: "accuracy ranged from 15 to 26 percents in our case." (Section 5 TTT)
- Future work (explicit): "there is a lot we could explore from where we are. For instance, train larger models. Or study where the model works well and where it doesn't." (Section 6 Conclusion)

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single domain of colored 2D grids (ARC).
> - Task structure: Multiple ARC tasks, focused on constant-size grid-to-grid transformations.
> - Representation rigidity: 2D grid tokens (10 colors + padding), output same size as input.
> - Model sharing vs specialization: Shared model with task embeddings; TTT retrains with new task embeddings.
> - Role of positional encoding: Fixed RoPE on rows/columns; no comparisons reported.

---

### 14. Final Classification

**Multi-task, single-domain.** The work trains across many ARC tasks ("there are only 262 constant size tasks") and uses a task embedding so a single model handles multiple tasks ("embedding layer that turns tasks ids into vectors"), but all evaluation stays within a single grid-based domain ("Inputs and outputs are colored grids of dimension up to 30x30"). This indicates multiple tasks within one modality/domain rather than cross-domain or multi-modal evaluation.
