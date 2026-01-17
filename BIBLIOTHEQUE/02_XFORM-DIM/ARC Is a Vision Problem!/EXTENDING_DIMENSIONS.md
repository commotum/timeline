## 1. Basic Metadata

- Title: "ARC Is a Vision Problem!" (Title)
- Authors: "Keya Hu Ali Cy Linlu Qiu Xiaoman Delores Ding Runqian Wang Yeyin Eva Zhu Jacob Andreas Kaiming He MIT" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper claims it "formulate[s] ARC within a vision paradigm, framing it as an image-to-image translation problem" and introduces "Vision ARC (VARC)" to solve ARC tasks with vision models (Abstract).

---

## 3. Tasks Evaluated

### Task 1
- Task name: ARC-1 benchmark (ARC-1 evaluation set)
- Task type: Reasoning / relational; Other (image-to-image translation / per-pixel classification)
- Dataset(s) used: ARC-1 training set; ARC-1 evaluation set; RE-ARC (training augmentation)
- Domain: Synthetic 2D grids
- Quotes:
  - "The Abstraction and Reasoning Corpus (ARC) benchmark [12] was designed to incentivize machine learning research aimed at improving these capabilities. ARC consists of a collection of puzzle-like tasks (Fig. 1, top), each containing only a few examples governed by a unique underlying transformation rule." (Section 1. Introduction)
  - "The ARC benchmark consists of several hundred very few-shot (e.g., 2 to 4-shot) reasoning tasks." (Section 3.1. ARC Problem Definition)
  - "Here, x and y are both 2D grids with maximum size  $30 \times 30$ , in which each location has one of C different color indexes (e.g., C=10)." (Section 3.1. ARC Problem Definition)
  - "With these definitions, we formulate reasoning on each task as an image-to-image translation problem. We frame the problem as per-pixel classification, analogous to the semantic segmentation problem [38]." (Section 3.2. Image-to-Image Translation)
  - "Our experiments are primarily conducted on the benchmark of ARC-1 [12]." (Section 5. Experimental Results)
  - "We evaluate our model on the ARC-1 evaluation set (i.e.,  $\mathcal{T}_{\text{eval}}$ )." (Section 5. Experimental Results)
  - "We use the standard ARC-1 training set  $\mathcal{T}_{train}$  for training: it has 400 tasks with 2-4 demo pairs each." (Section 4. Implementation Details)
  - "Following common practice on ARC, we also expand our training set with the RE-ARC set [22], from which we sample 1,000 additional demo pairs per task." (Section 4. Implementation Details)

### Task 2
- Task name: ARC-2 benchmark
- Task type: Reasoning / relational; Other (image-to-image translation / per-pixel classification)
- Dataset(s) used: ARC-2 evaluation (test-time training and inference on ARC-2)
- Domain: Synthetic 2D grids
- Quotes:
  - "We also report final results on ARC-2 [14]." (Section 5. Experimental Results)
  - "Our ARC-2 models are trained only on the ARC-1 dataset, with test-time training and inference on the ARC-2 set." (Section 5.3. System-level Comparisons)
  - "Here, x and y are both 2D grids with maximum size  $30 \times 30$ , in which each location has one of C different color indexes (e.g., C=10)." (Section 3.1. ARC Problem Definition)
  - "With these definitions, we formulate reasoning on each task as an image-to-image translation problem. We frame the problem as per-pixel classification, analogous to the semantic segmentation problem [38]." (Section 3.2. Image-to-Image Translation)

---

## 4. Domain and Modality Scope

- Evaluation domain scope: Single domain (synthetic 2D grids). Evidence: "Here, x and y are both 2D grids with maximum size  $30 \times 30$ , in which each location has one of C different color indexes (e.g., C=10)." (Section 3.1. ARC Problem Definition)
- Multiple domains within the same modality? Not indicated; the tasks are all ARC grids. Evidence: "ARC consists of a collection of puzzle-like tasks (Fig. 1, top), each containing only a few examples governed by a unique underlying transformation rule." (Section 1. Introduction)
- Multiple modalities? Not indicated. Evidence: "We frame each puzzle as an image-to-image translation problem." (Section 1. Introduction)
- Domain generalization or cross-domain transfer claimed? Not claimed. (The paper states it "generalizes to unseen tasks" within ARC: "Our model is trained from scratch solely on ARC data and generalizes to unseen tasks through test-time training." (Abstract))

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ARC-1 benchmark | Yes | Yes (test-time training per task) | Task token only (task-conditional embedding) | "We train one model  $f_{\theta}$  jointly for all k training tasks (e.g., k=400)... All tasks share the same parameters, only except that each task has its own task-conditional token." (Section 3.4. Two-stage Training) "We perform test-time training for each new task T independently. It has a new task token whose parameters are randomly initialized." (Section 3.4. Two-stage Training) |
| ARC-2 benchmark | Yes | Yes (test-time training per task) | Task token only (task-conditional embedding) | "We train one model  $f_{\theta}$  jointly for all k training tasks (e.g., k=400)... All tasks share the same parameters, only except that each task has its own task-conditional token." (Section 3.4. Two-stage Training) "We perform test-time training for each new task T independently. It has a new task token whose parameters are randomly initialized." (Section 3.4. Two-stage Training) |

---

## 6. Input and Representation Constraints

- Grid size and dimensionality: "Here, x and y are both 2D grids with maximum size  $30 \times 30$ , in which each location has one of C different color indexes (e.g., C=10)." (Section 3.1. ARC Problem Definition)
- Fixed canvas size: "A canvas has a predefined and sufficiently large size, e.g.,  $64 \times 64$ ." (Section 3.3. Visual Modeling, Canvas)
- Default canvas size: "In our best-performing model, the canvas size is  $64 \times 64$ ." (Section 4. Implementation Details)
- Patch size and tokenization: "The input canvas is divided into non-overlapping patches (e.g.,  $2\times2$ ), projected by a linear embedding, added with positional embedding [52], and processed by a stack of Transformer blocks [52]." (Section 3.3. Visual Modeling, Vision Transformer)
- Fixed token count in default setting: "In the case of ViT, the patch size is  $2 \times 2$ , resulting in a sequence length of  $32^2$ ." (Section 4. Implementation Details)
- Padding/background and output shape handling: "the input/output canvas *always* has a fixed size and is filled with a background token [BG]." (Section A.3. Shape Handling)
- Border token and cropping: "we always use an extra border token, [BD], to indicate the right and bottom edges... During inference, we locate the rightmost and bottommost [BD] tokens and crop the output accordingly to recover the final predicted shape." (Section A.3. Shape Handling)
- Resizing/translation constraints: "Scale augmentation: Given a raw input, we randomly resize it by an integer scaling ratio s, duplicating each raw pixel into  $s \times s$ ." (Section 3.3. Visual Modeling, Translation and scale invariance)
- Placement constraint: "Translation augmentation: given the scaled grid, we randomly place it on the fixed-size canvas. We ensure all pixels are visibile." (Section 3.3. Visual Modeling, Translation and scale invariance)

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "In the case of ViT, the patch size is  $2 \times 2$ , resulting in a sequence length of  $32^2$ ." (Section 4. Implementation Details)
- Fixed or variable length: Not explicitly stated; default uses "canvas size is  $64 \times 64$ " and "patch size is  $2 \times 2$ , resulting in a sequence length of  $32^2$ ." (Section 4. Implementation Details)
- Attention type: Not explicitly labeled as global/windowed/sparse; the model is "processed by a stack of Transformer blocks [52]" with self-attention masks applied to background tokens: "The attention masks are applied after the query-key dot-product computation, adding a large negative value to the keys corresponding to background inputs." (Section 3.3. Visual Modeling, Vision Transformer; Section A.3. Shape Handling)
- Mechanisms to manage computational cost: Patchification is used: "The input canvas is divided into non-overlapping patches (e.g.,  $2\times2$ )." (Section 3.3. Visual Modeling, Vision Transformer). Additional cost control mechanisms (windowing, pooling, token pruning) are not stated.

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: "we adopt *separable* 2D positional embeddings" and "use the first half of the channels to embed the horizontal coordinate and the second half to embed the vertical coordinate." (Section 3.3. Visual Modeling, 2D positional embedding)
- Absolute vs relative: "This can be applied both to additive positional embeddings for encoding absolute positions and to the encoding of relative positions (e.g., RoPE [48])." (Section 3.3. Visual Modeling, 2D positional embedding)
- Where applied: "The input canvas is divided into non-overlapping patches (e.g.,  $2\times2$ ), projected by a linear embedding, added with positional embedding [52], and processed by a stack of Transformer blocks [52]." (Section 3.3. Visual Modeling, Vision Transformer)
- Fixed or modified per task / ablated: "Extending from 1D positional embedding to its 2D counterpart is beneficial: see Fig. 7(b)(c). This is observed in both (b) absolute and (c) relative positional embeddings." (Section 5.1. Visual Priors) and "we replace the 2D RoPE in Fig. 7(f) with a 1D RoPE and observe a degradation of 3.5 points." (Section 5.1. Visual Priors)

---

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Treated as a variable; the paper states "we empirically show that explicitly modeling positions in 2D is essential." (Section 3.3. Visual Modeling, 2D positional embedding)
- Multiple positional encodings compared: "Extending from 1D positional embedding to its 2D counterpart is beneficial: see Fig. 7(b)(c). This is observed in both (b) absolute and (c) relative positional embeddings." (Section 5.1. Visual Priors)
- PE claimed not critical? Not claimed; instead, the paper calls 2D modeling "essential." (Section 3.3. Visual Modeling, 2D positional embedding)

---

## 10. Evidence of Constraint Masking

- Model sizes: "We compare variants of ViTs and U-Nets of similar sizes" with ViT sizes "6M" "18M" "66M" and U-Net sizes "7M" "17M" "55M" (Section 5.1. Visual Priors, Table 1). The paper also notes "using a small model with only 18 million parameters." (Section 1. Introduction)
- Dataset sizes: "We use the standard ARC-1 training set  $\mathcal{T}_{train}$  for training: it has 400 tasks with 2-4 demo pairs each." and "we also expand our training set with the RE-ARC set [22], from which we sample 1,000 additional demo pairs per task. Put together, our full training set has about 400k sample pairs." (Section 4. Implementation Details)
- Performance gains attributed to scaling model size: "increasing depth and/or width leads to higher accuracy as a result of better fitting." (Section 5.2. Other Ablation Experiments, Scalability)
- Performance gains attributed to scaling data: "Increasing the amount of offline training data is beneficial" and "Increasing task diversity is beneficial." (Section B.1. Offline Training Data Scaling)
- Performance gains attributed to architectural priors / training tricks: "These priors cumulatively yield **27.7** improvement... the canvas-based designs (c -> f) contribute an **11.5** gain." (Section 5.1. Visual Priors) and "Scale augmentation yields a substantial gain of 6.2 points." (Section 5.1. Visual Priors)

---

## 11. Architectural Workarounds

- Canvas representation to enable vision priors: "we represent the inputs on a \"canvas\" that can be processed like natural images" (Abstract) and "A canvas has a predefined and sufficiently large size, e.g.,  $64 \times 64$ . The raw input is transformed and placed onto this canvas." (Section 3.3. Visual Modeling, Canvas)
- Translation and scale augmentations: "The \"canvas\" concept enables us to flexibly apply translation and scale augmentations" (Section 3.3. Visual Modeling, Translation and scale invariance)
- Patchification: "The input canvas is divided into non-overlapping patches (e.g.,  $2\times2$ )" and patchification "incorporates several critical inductive biases in vision: most notably, locality... and translation invariance." (Section 3.3. Visual Modeling, Vision Transformer)
- Task conditioning token: "The network  $f_{\theta}$  takes an image  $x_i$  as input, conditioned on a task token associated with the task T. The task token is represented as a learnable embedding dependent on T." (Section 3.2. Image-to-Image Translation)
- Fixed-size canvas with background and border tokens: "the input/output canvas *always* has a fixed size and is filled with a background token [BG]" and "we always use an extra border token, [BD], to indicate the right and bottom edges." (Section A.3. Shape Handling)
- Attention masks to focus on foreground: "we apply attention masks in the self-attention blocks to encourage the model to focus on the foreground pixels." (Section A.3. Shape Handling)
- Test-time training per task: "we perform test-time training for each new task T independently." (Section 3.4. Two-stage Training)
- Multi-view inference for accuracy: "we adopt multi-view inference to improve accuracy, where the views are sampled with different augmentations." (Section 3.5. Inference)

---

## 12. Explicit Limitations and Non-Claims

- Overfitting with larger models: "Going beyond this regime can lead to overfitting in our current setting, as shown in Tab. 1 for the 66M ViT model." (Section 5.2. Other Ablation Experiments, Scalability)
- Need for better generalization: "We observe that this larger model achieves higher training accuracy, suggesting that future research should focus on generalization." (Section 5.2. Other Ablation Experiments, Scalability)
- Joint test-task assumption not valid: "In general, it cannot be assumed that multiple unseen tasks will be presented all at once." (Section 5.2. Other Ablation Experiments, Test-time training strategies)
- Future work directions: "Future research may extend this direction through more expressive architectures, richer visual priors, or larger-scale image pre-training." (Section 7. Conclusion)
- Explicit non-claims about open-world learning or unrestrained multi-task learning: Not stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Single synthetic-grid domain (ARC), with inputs as 2D grids and fixed-size canvases.
> - Task structure: Many distinct ARC tasks with few-shot demonstrations, each with a unique transformation rule.
> - Representation rigidity: Fixed-size canvas, patchified ViT inputs, and BG/BD tokens for shape handling.
> - Model sharing vs specialization: One shared model across tasks with task tokens and per-task test-time fine-tuning.
> - Role of positional encoding: Explicit 2D positional encoding is treated as essential and ablated (1D vs 2D, absolute vs relative).

---

## 14. Final Classification

**Multi-task, single-domain.** The paper evaluates on ARC-1 and ARC-2, where "ARC consists of a collection of puzzle-like tasks" and the inputs are "2D grids" (Section 1. Introduction; Section 3.1. ARC Problem Definition), indicating many tasks within a single grid-based domain. It also "train[s] one model... jointly for all k training tasks (e.g., k=400)" (Section 3.4. Two-stage Training), reinforcing multi-task training within one modality and domain.
