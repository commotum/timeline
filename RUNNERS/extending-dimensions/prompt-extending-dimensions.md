## 📄 ViT Paper Review Prompt (Survey-Grade)

You are reviewing an academic paper on Vision Transformers (ViTs).
Your goal is **not** to summarize the paper for a general audience, but to extract **structural and empirical evidence** relevant to evaluating whether the paper studies *constrained single-domain tasks* versus *unrestrained multi-task or multi-domain learning*.

Follow the instructions precisely.
When asked to cite evidence, **quote the paper verbatim and give the section name or page number**.
If information is not explicitly stated, say **“Not specified in the paper.”** Do not infer.

---

### 1. Basic Metadata

Provide:

* **Title**
* **Authors**
* **Year**
* **Venue (conference/journal/arXiv)**

---

### 2. One-Sentence Contribution Summary

In **one sentence**, describe the paper’s primary contribution, focusing on *what problem it claims to solve*.

---

### 3. Tasks Evaluated

List **all tasks** the paper evaluates on.

For each task, provide:

* Task name
* Task type (choose one or more):

  * Classification
  * Detection
  * Segmentation
  * Generation
  * Tracking
  * Reconstruction
  * Reasoning / relational
  * Other (specify)
* Dataset(s) used
* Domain (e.g., natural images, medical images, video, synthetic grids)

Include **exact quotes** from the paper that define or describe each task.

---

### 4. Domain and Modality Scope

Answer the following explicitly:

* Is evaluation performed on:

  * A **single domain**?
  * Multiple domains within the same modality?
  * Multiple modalities?
* Does the paper claim *domain generalization* or *cross-domain transfer*?

Provide supporting quotes or state “Not claimed.”

---

### 5. Model Sharing Across Tasks

For each task listed in Section 3, indicate:

* Are **the same model weights** used across tasks?
* Is the model:

  * Trained separately per task?
  * Pretrained once and fine-tuned per task?
  * Trained jointly on multiple tasks?

Summarize this in a table:

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |

---

### 6. Input and Representation Constraints

Extract all **explicit assumptions** about the input representation:

* Fixed or variable input resolution?
* Fixed patch size?
* Fixed number of tokens?
* Fixed dimensionality (e.g., strictly 2D)?
* Any padding or resizing requirements?

Quote the paper wherever these constraints are stated or implied.

---

### 7. Context Window and Attention Structure

Provide:

* Maximum sequence length
* Whether sequence length is fixed or variable
* Attention type:

  * Global
  * Windowed
  * Hierarchical
  * Sparse
* Any mechanisms introduced to manage computational cost (e.g., windowing, pooling, token pruning)

Include citations.

---

### 8. Positional Encoding (Critical Section)

Describe **exactly**:

* Positional encoding mechanism used

  * Absolute
  * Relative
  * RoPE
  * Axial
  * Bias-based
  * Implicit / none
* Where it is applied:

  * Input only
  * Every layer
  * Attention bias
* Whether positional encoding is:

  * Fixed across all experiments
  * Modified per task
  * Ablated or compared against alternatives

Provide **verbatim quotes** identifying the positional encoding choice.

---

### 9. Positional Encoding as a Variable

Answer explicitly:

* Does the paper treat positional encoding as:

  * A core research variable?
  * A fixed architectural assumption?
* Are multiple positional encodings compared?
* Does the paper claim PE choice is “not critical” or secondary?

Quote any relevant statements.

---

### 10. Evidence of Constraint Masking

Extract evidence related to **scale compensating for structure**, including:

* Model size(s)
* Dataset size(s)
* Whether performance gains are primarily attributed to:

  * Scaling model size
  * Scaling data
  * Architectural hierarchy
  * Training tricks

Cite claims or ablation results.

---

### 11. Architectural Workarounds

List any architectural techniques introduced to manage complexity or scale, such as:

* Windowed attention
* Hierarchical stages
* Token pooling / merging
* Task-specific heads
* Fixed grid assumptions

Briefly describe their purpose, with citations.

---

### 12. Explicit Limitations and Non-Claims

Extract:

* Any stated limitations or “future work”
* Any explicit statements about what the model does **not** attempt to do (e.g., open-world learning, unrestrained multi-task learning, meta-learning)

Quote directly.

---

### 13. Constraint Profile (Synthesis)

Provide a short synthesis (3–5 bullet points):

> **Constraint Profile:**
> – Domain scope
> – Task structure
> – Representation rigidity
> – Model sharing vs specialization
> – Role of positional encoding

This should describe **how constrained the experimental setup is**, not whether results are good.

---

### 14. Final Classification

Classify the paper as one of the following (choose one):

* **Single-task, single-domain**
* **Multi-task, single-domain**
* **Multi-task, multi-domain (constrained)**
* **Unrestrained multi-task / multi-domain**

Justify the classification in 2–3 sentences using evidence from above.

---

## Important Rules

* Do **not** speculate.
* Do **not** generalize beyond what is tested.
* If something is unclear or unstated, say so explicitly.
* Favor **direct quotations over paraphrase** wherever possible.