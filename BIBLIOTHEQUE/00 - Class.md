# Research Paper Classifications

1. **Positional Encoding Improvement Proposal**
2. **Increasing Transformer’s Dimensions**
3. **Computation & Reasoning Mechanism Proposal**
4. **Data, Benchmarks & Measurement**
5. **ML Foundations & Principles**
6. **External Foundations & Cross-Disciplinary Context**
7. **Misclassifications**

Code map:

Positional Encoding Improvement Proposal ↦ POS-ENCDR  
Increasing Transformer’s Dimensions ↦ XFORM-DIM  
Computation & Reasoning Mechanism Proposal ↦ COMP-REAS  
Data, Benchmarks & Measurement ↦ DATA-BNCH  
ML Foundations & Principles ↦ ML-FNDTNS  
External Foundations & Cross-Disciplinary Context ↦ X-CONTEXT  
Misclassifications ↦ MIS-CLASS

---

# Positional Encoding Improvement Proposal

**Criteria:**

1. Works on transformer-based models, where **attention is the core mechanism** and the primary setting for the contribution.
2. The paper’s *main perspective* includes a **critique of existing positional encoding methods**, identifying a limitation or deficiency in prior approaches.
3. The paper’s *core contribution* is a **modification, change, or innovation in positional encoding itself**, intended to improve how position is handled (and **not** primarily by increasing the number of encoded dimensions).

**Examples:**

- *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
  **Critique:** absolute positional embeddings
  **Improvement:** fused relative + absolute encoding; linear-attention compatible

- *YaRN: Efficient Context Window Extension of Large Language Models* (2023)
  **Critique:** poor context length extrapolation
  **Improvement:** frequency-aware RoPE interpolation with attention temperature scaling

- *LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate* (2024)
  **Critique:** resolution and patch-count dependence
  **Improvement:** directional attention heads with distance-penalized bias

- *Decoupling the “What” and “Where” With Polar Coordinate Positional Embeddings (PoPE)* (2025)
  **Critique:** entangled content and position
  **Improvement:** polar-coordinate positional encoding

---

# Increasing Transformer's Dimensions

**Criteria:**

1. Works on transformer based models, where **attention is the core mechanism** and the primary setting for the contribution.
2. Highlights the success of transformer based models in modeling language and other **1-dimensional tasks** **AND/OR** notes the **need for 2D, 3D, 4D, nD, graph-based, or other type of positional encodings** for tasks in other domains.
3. The paper’s *core presentation* is a **Transformer adaptation** that enables modeling of a **higher-dimensional domain** (e.g., images/video/3D/graphs). This adaptation may be *architectural* (attention structure, encoder/decoder changes) or *representational* (tokenization/serialization/ordering/patching and associated positional treatment), as long as it is **central to the paper’s contribution**. (**This SHOULD BE distinct from modifying or improving the positional encoding mechanism.**)


**Examples:**

- **Generative Pretraining from Pixels** (2020)  
  **Improves On:** 1D autoregressive Transformers for text  
  **Adaptation:** serialize 2D images into a 1D pixel sequence for causal attention–based generation

- **An Image Is Worth 16×16 Words (ViT)** (2020)  
  **Improves On:** Transformer encoders for sequence classification  
  **Adaptation:** tokenize images into fixed-size patches with learned absolute positional embeddings

- **End-to-End Object Detection with Transformers (DETR)** (2020)  
  **Improves On:** CNN-based object detection pipelines  
  **Adaptation:** use global attention with 2D positional encoding and learned object queries for set prediction

- **Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)** (2021)  
  **Improves On:** Vision Transformers for static images  
  **Adaptation:** extend patch tokens across time with factorized spatial–temporal attention

---

# Computation & Reasoning Mechanism Proposal

**Criteria:**

1. Works on (or is meant to plug into) neural models used for sequence/structure reasoning (often Transformers, but not required).
2. The paper’s *main POV* is: “standard feedforward inference is missing a capability” (e.g., **memory**, **iteration**, **variable compute**, **search**, **constraint solving**, **planning**, **program induction**, **equilibrium dynamics**, **test-time adaptation/compute**).
3. The paper’s *core offering* is a **mechanism or algorithmic framing** that changes *how computation happens* (not primarily “how positions are encoded,” and not primarily “how to lift Transformers into 2D/3D/4D domains”).

**Examples:**

- *Neural Turing Machines* (2014)  
  **Missing capability:** persistent memory  
  **Mechanism:** differentiable read/write access to external memory

- *Recurrent Relational Networks* (2018)  
  **Missing capability:** iterative constraint satisfaction  
  **Mechanism:** recurrent message passing over relational graphs

- *Adaptive Computation Time for Recurrent Neural Networks* (2016)  
  **Missing capability:** adaptive depth  
  **Mechanism:** learned halting for variable per-input computation

- *Hierarchical Reasoning Model* (2025)  
  **Missing capability:** deep, stable latent reasoning with variable compute  
  **Mechanism:** hierarchical recurrent computation with fast–slow modules, adaptive halting (ACT), and one-step equilibrium-style training

- *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* (2025)  
  **Missing capability:** strong multi-step reasoning emerging from training (not just prompting)  
  **Mechanism:** large-scale RL fine-tuning of a base LLM to reinforce reasoning behaviors (with an R1-Zero variant trained via pure RL without SFT)

---

# Data, Benchmarks & Measurement

**Criteria:**

1. The paper’s main contribution is **infrastructure** for progress: a **dataset**, a **benchmark**, an **evaluation protocol**, or a **measurement critique/taxonomy**.
2. The *POV* is usually: “we’re bottlenecked by data,” or “we’re not measuring the right thing / models fail in this way,” or “here’s a standardized way to compare systems.”
3. Any modeling in the paper is **secondary** to providing the resource, diagnostic, or measurement lens.

**Examples:**

- *On the Measure of Intelligence* (2019)  
  **Target Domain:** abstract and spatial reasoning 
  **Resource:** the *Abstraction and Reasoning Corpus (ARC)*, 800 symbolic grid-based input–output tasks (400 training, 400 evaluation)

- *Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering* (2022)  
  **Target Domain:** multimodal science reasoning and explanation-based question answering  
  **Resource:** the *ScienceQA* benchmark, 21,208 multiple-choice science questions with images, text context, and annotated lectures and explanations

- *LAION-5B: An open large-scale dataset for training next generation image-text models* (2022)  
  **Target Domain:** large-scale vision–language pretraining (image–text representation learning + text-to-image generation)  
  **Resource:** *LAION-5B*, 5.85B CLIP-filtered image–text pairs (2.32B English)

---

# ML Foundations & Principles

**Criteria:**

1. The paper makes a **substantive contribution** to machine learning or neural modeling that does **not cleanly fall into categories 1–4**.
2. The contribution is *not primarily about* positional encoding, dimensional lifting, computation mechanisms, or datasets/benchmarks.
3. The paper may focus on **training methods**, **optimization**, **scaling laws**, **theory**, **interpretability**, **robustness**, **alignment**, **regularization**, or **system-level concerns**.

**Examples:**

- *Attention Is All You Need* (2017)  
  **Contribution Type:** architectural unification  
  **Reason:** introduces the Transformer itself rather than refining an existing class

- *Scaling Laws for Neural Language Models* (2020)  
  **Contribution Type:** empirical theory  
  **Reason:** characterizes performance–compute–data relationships, not a modeling mechanism

- *LoRA: Low-Rank Adaptation of Large Language Models* (2021)  
  **Contribution Type:** parameter-efficient training  
  **Reason:** focuses on fine-tuning efficiency rather than model structure or representation

- *RLHF: Training Language Models to Follow Instructions* (2022)  
  **Contribution Type:** alignment & training methodology  
  **Reason:** centers on optimization objectives and human feedback pipelines

---

# External Foundations & Cross-Disciplinary Context

**Criteria:**

1. The paper does **not cleanly fall** into any of the primary classifications (1–5).
2. The paper originates from a **different discipline** (e.g., cognitive science, neuroscience, physics, mathematics, biology).
3. The paper is included because it provides **theoretical grounding**, **conceptual motivation**, or **explanatory context** relevant to machine learning research.

**Common Source Domains:**

- Cognitive science and psychology
- Neuroscience and biology
- Physics (e.g., statistical mechanics, dynamical systems)
- Mathematics (e.g., information theory, optimization, geometry)
- Philosophy of mind and epistemology (selectively)

**Interpretation:**

These papers are **out-of-scope by design, not by error**.  
They inform *why* learning systems behave as they do or *what* they may be approximating, rather than contributing directly to model design.

---

# Misclassifications

**Criteria:**

1. The paper does **not satisfy the criteria** of any classification in this taxonomy.
2. The paper was **not intentionally included** for cross-disciplinary context or conceptual grounding.
3. Inclusion is best explained by **curation error**, **surface-level similarity**, or **misinterpretation** of relevance.

**Common Causes:**

- Keyword overlap without substantive contribution
- Tangential or incidental use of attention or Transformers
- Purely applied results without methodological insight
- Accidental inclusion during broad literature collection

**Interpretation:**

Papers in this category are candidates for **removal or reassignment**.  
This section exists to support **taxonomy hygiene**, not qualitative judgment.
