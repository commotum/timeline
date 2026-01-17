## 1. Basic Metadata
- Title: "**BROS: A Pre-trained Language Model Focusing on Text and Layout** for Better Key Information Extraction from Documents" (Title)
- Authors: "Teakgyu Hong<sup>1</sup>, Donghyun Kim<sup>1</sup>, Mingi Ji<sup>2</sup>, Wonseok Hwang<sup>3</sup>, Daehyun Nam<sup>4</sup>, Sungrae Park<sup>4</sup>" (Title block)
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary
To address document KIE where "Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space," the paper proposes BROS that "encodes relative positions of texts in 2D space and learns from unlabeled documents with area-masking strategy" (Abstract).

## 3. Tasks Evaluated
- Task name: Entity Extraction (EE). Task type: Classification; Other (sequence labeling). Dataset(s) used: FUNSD, SROIE*, CORD. Domain: document images (forms, receipts). Evidence: "We solve two categories of KIE tasks, entity extraction (EE) and entity linking (EL)." (Key Information Extraction Tasks) "The EE task identifies sequences of text blocks that represent desired target texts." (Key Information Extraction Tasks) "Form Understanding in Noisy Scanned Documents (FUNSD) (Jaume, Ekenel, and Thiran 2019) is a set of documents with various forms." (KIE Benchmark Datasets) "SROIE\* is a variant of Task 3 of \"Scanned Receipts OCR and Information Extraction\" (SROIE)<sup>5</sup> that consists of a set of store receipts." (KIE Benchmark Datasets) "Consolidated Receipt Dataset (CORD) (Park et al. 2019) is a set of store receipts with 800 training, 100 validation, and 100 testing examples." (KIE Benchmark Datasets)
- Task name: Entity Linking (EL). Task type: Reasoning / relational; Other (link prediction). Dataset(s) used: FUNSD, CORD, SciTSR. Domain: document images (forms, receipts, tables). Evidence: "The EL task connects key entities through their hierarchical or semantic relations." (Key Information Extraction Tasks) "FUNSD has both EE and EL tasks." (KIE Benchmark Datasets) "CORD consists of both EE and EL tasks." (KIE Benchmark Datasets) "Complicated Table Structure Recognition (SciTSR) (Chi et al. 2019) is an EL task that connects cells in a table to recognize the table structure." (KIE Benchmark Datasets)

## 4. Domain and Modality Scope
- Is evaluation performed on a single domain? No; datasets span multiple document types within document images, e.g., "documents with various forms" and "store receipts" and "cells in a table" (KIE Benchmark Datasets).
- Is evaluation performed on multiple domains within the same modality? Yes; the datasets include forms, receipts, and tables: "documents with various forms," "store receipts," and "cells in a table" (KIE Benchmark Datasets).
- Is evaluation performed on multiple modalities? No; results are reported "without relying on visual features" (Abstract).
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Entity Extraction (EE) | Pretrained backbone shared; task-specific fine-tuning per dataset | Yes | Yes (BIO tagger or SPADE decoder) | "For pre-training, IIT-CDIP Test Collection 1.0<sup>1</sup> (Lewis et al. 2006), which consists of approximately 11M document images, is used but 400K of RVL-CDIP dataset<sup>2</sup> (Harley, Ufkes, and Derpanis 2015) are excluded following LayoutLM." (Experiment Settings); "During fine-tuning, the learning rate is set to 5e-5. The batch size is set to 16 for all tasks. The number of training epochs or steps is as follows: 100 epochs for FUNSD, 1K steps for SROIE\* and CORD, and 7.5 epochs for SciTSR." (Experiment Settings); "For EE tasks, all models utilize BIO tagger that captures spans of text blocks to represent key entities in documents." (Experiment Results); "To solve EE and EL tasks without the order information, we employ the SPADE decoder for all tasks." (Experiment Results) |
| Entity Linking (EL) | Pretrained backbone shared; task-specific fine-tuning per dataset | Yes | Yes (SPADE decoder) | "For pre-training, IIT-CDIP Test Collection 1.0<sup>1</sup> (Lewis et al. 2006), which consists of approximately 11M document images, is used but 400K of RVL-CDIP dataset<sup>2</sup> (Harley, Ufkes, and Derpanis 2015) are excluded following LayoutLM." (Experiment Settings); "During fine-tuning, the learning rate is set to 5e-5. The batch size is set to 16 for all tasks. The number of training epochs or steps is as follows: 100 epochs for FUNSD, 1K steps for SROIE\* and CORD, and 7.5 epochs for SciTSR." (Experiment Settings); "For EL tasks, SPADE decoder is used to identify relationships between entities not placed sequentially in a series of text blocks." (Experiment Results) |

## 6. Input and Representation Constraints
- 2D layout assumption: "Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space." (Abstract)
- Text blocks as bounding boxes with four vertices: "a bounding box of a text block consists of four vertices, such as  $p^{tl}$ ,  $p^{tr}$ ,  $p^{br}$ , and  $p^{bl}$ , that indicate top-left, top-right, bottom-right, and bottom-left points, respectively." (Encoding Spatial Information into BERT)
- Coordinate normalization: "BROS first normalizes all the 2D points of the text blocks using the size of the image." (Encoding Spatial Information into BERT)
- Maximum token count: "N is the maximum number of tokens." (SPADE Decoder)
- Fixed or variable input resolution: Not specified (only normalization by image size is stated).
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified beyond "N is the maximum number of tokens." (SPADE Decoder)
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; the only explicit mention is "N is the maximum number of tokens." (SPADE Decoder)
- Fixed or variable sequence length: Not specified.
- Attention type: Global all-pairs attention with spatial encoding; "BROS directly encodes the spatial relations to the contextualization of text blocks. In detail, it calculates an attention logit combining both semantic and spatial features as follows;" (Encoding Spatial Information into BERT) and "Since BROS considers relative positions for all text block pairs, it is slower than LayoutLM, but faster than LayoutLMv2 using image features." (Compare the Inference Speed of the Models)
- Mechanisms to manage computational cost: Not specified; instead it "considers relative positions for all text block pairs" (Compare the Inference Speed of the Models).

## 8. Positional Encoding (Critical Section)
- Mechanism: Relative 2D positional encoding with sinusoidal functions of coordinate differences; "BROS employs relative positions between text blocks to explicitly encode spatial relations." (Encoding Spatial Information into BERT) and "BROS calculates relative positions of the vertices from the same vertices of the other bounding boxes of text blocks and applies sinusoidal functions" (Encoding Spatial Information into BERT).
- Where applied: Attention mechanism/logits; "The position difference between text blocks is encoded directly to the attention mechanism in Transformer." (Figure 2 caption) and "BROS directly encodes the spatial relations to the contextualization of text blocks. In detail, it calculates an attention logit combining both semantic and spatial features" (Encoding Spatial Information into BERT).
- Shared across heads: "the multi-head attention modules in Transformer share the same relative positional embeddings" (Encoding Spatial Information into BERT).
- Fixed across experiments or modified per task: Compared/ablated; "Table 9 compares three positional encoding methods: absolute position in LayoutLM, relative position in LayoutLMv2, and ours." (Ablation Study)
- Layer scope: Not specified.

## 9. Positional Encoding as a Variable
- Core research variable: Yes; "We conduct ablation studies to investigate which component contributes the performance improvement." (Ablation Study) and "When applying our proposed positional encoding to LayoutLM, the performances consistently increase" (Ablation Study).
- Multiple positional encodings compared: Yes; "Table 9 compares three positional encoding methods: absolute position in LayoutLM, relative position in LayoutLMv2, and ours." (Ablation Study).
- Claim that positional encoding is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking
- Model size(s): "| LayoutLM <sub>BASE</sub><br>LayoutLMv2 <sub>BASE</sub><br>BROS <sub>BASE</sub>    | 0   | 113M<br>200M<br><b>110M</b> | 82.76 | 33.89<br>40.77<br><b>76.94</b> | 69.92 |" and "| LayoutLM <sub>LARGE</sub><br>LayoutLMv2 <sub>LARGE</sub><br>BROS <sub>LARGE</sub> | 0   | 343M<br>426M<br><b>340M</b> | 84.20 | 33.11<br>62.53<br><b>79.42</b> | 72.12 |" (Table 1)
- Dataset size(s): "For pre-training, IIT-CDIP Test Collection 1.0<sup>1</sup> (Lewis et al. 2006), which consists of approximately 11M document images, is used" (Experiment Settings); "The dataset consists of 149 training and 50 testing examples." (KIE Benchmark Datasets, FUNSD); "We also split the original training set into 526 training and 100 testing examples" (KIE Benchmark Datasets, SROIE*); "Consolidated Receipt Dataset (CORD) (Park et al. 2019) is a set of store receipts with 800 training, 100 validation, and 100 testing examples." (KIE Benchmark Datasets); "The dataset consists of 12,000 training images and 3,000 test images." (KIE Benchmark Datasets, SciTSR)
- Attribution of gains: The paper attributes improvements to architecture and training objectives, e.g., "We propose an effective spatial layout encoding method by accounting for relative positions of text blocks." and "We also propose a novel area-masking self-supervision strategy that reflects 2D natures of text blocks." (Contributions); "This ablation study proves that each component of BROS solely contributes to performance improvements" (Ablation Study).
- Evidence against scale-only gains: "It should be noted that the BROS<sub>BASE</sub> show better performance than that of LayoutLM\*<sub>LARGE</sub>, even though it uses three times lower number of parameters (110M vs 343M)." (Experiment Results)

## 11. Architectural Workarounds
- Relative spatial encoding in attention to model spatial relations: "BROS employs relative positions between text blocks to explicitly encode spatial relations." (Encoding Spatial Information into BERT) and "The position difference between text blocks is encoded directly to the attention mechanism in Transformer." (Figure 2 caption)
- Area-masked LM to pretrain 2D spans: "AMLM masks all text blocks allocated in a randomly chosen area." (Area-masked Language Model)
- SPADE decoder to remove dependence on reading order: "we utilize SPADE (Hwang et al. 2021) decoder to extract key information without any information about the order." (SPADE Decoder)
- Task-specific parsers/heads: "For EE tasks, all models utilize BIO tagger that captures spans of text blocks to represent key entities in documents." (Experiment Results) and "For EL tasks, SPADE decoder is used to identify relationships between entities not placed sequentially in a series of text blocks." (Experiment Results)

## 12. Explicit Limitations and Non-Claims
- Dataset limitation: "Although these four datasets provide testbeds for the EE and EL tasks, they represent the subset of real problems as the order information of text blocks is given." (Key Information Extraction Tasks)
- Other limitations or non-claims: Not stated.

### 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: Document images across forms, receipts, and tables; no multi-modal evaluation beyond text and layout (KIE Benchmark Datasets; Abstract).
- Task structure: Two supervised KIE tasks (EE and EL) on OCR text blocks with task-specific parsers (Key Information Extraction Tasks; Experiment Results).
- Representation rigidity: 2D bounding boxes with normalized coordinates and relative positions; a maximum token count is assumed (Abstract; Encoding Spatial Information into BERT; SPADE Decoder).
- Model sharing vs specialization: Single pretrained backbone on IIT-CDIP with per-dataset fine-tuning and distinct EE/EL heads (Experiment Settings; Experiment Results).
- Role of positional encoding: Central variable, with relative spatial encoding compared in ablations (Encoding Spatial Information into BERT; Ablation Study).

### 14. Final Classification
Classification: **Multi-task, multi-domain (constrained)**. The paper evaluates two tasks, stating "We solve two categories of KIE tasks, entity extraction (EE) and entity linking (EL)," and covers multiple document domains including "documents with various forms," "store receipts," and "cells in a table" (Key Information Extraction Tasks; KIE Benchmark Datasets). At the same time, evaluation is confined to document images and text-layout processing, with results reported "without relying on visual features" (Abstract).
