# SegPoint: Segment Any Point Cloud via Large Language Model (2024)
Source: SegPoint- Segment Any Point Cloud via Large Language Model.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D instruction segmentation | Point clouds; instructional text queries (implicit instructions) | 3D (x, y, z); 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Point-wise segmentation mask(s) for instructed target(s) | 3D (x, y, z) | Capped (inferred) |
| 3D referring segmentation | Point clouds; referring expressions (explicit descriptions) | 3D (x, y, z); 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Point-wise segmentation mask(s) for referred instance(s) | 3D (x, y, z) | Capped (inferred) |
| 3D semantic segmentation | Point clouds; semantic category prompt(s) | 3D (x, y, z); 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Point-wise semantic mask(s) with category labels | 3D (x, y, z) | Capped (inferred) |
| 3D open-vocabulary semantic segmentation | Point clouds; open-vocabulary category text queries | 3D (x, y, z); 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Point-wise mask(s) for unseen/open-vocabulary categories | 3D (x, y, z) | Capped (inferred) |

## Summary
The paper presents a unified model that handles four 3D point-cloud segmentation tasks: instruction, referring, semantic, and open-vocabulary semantic segmentation. Across these tasks, inputs combine 3D point clouds with text queries/prompts, and outputs are point-wise segmentation masks over 3D scenes. The OCR explicitly supports 3D point-cloud and text-token inputs, while dynamics and control-state labels are not explicitly named; from the described finite point/token processing pipeline, Capped dynamics, Static attention, and Direct state are inferred.

## Evidence
### Task: 3D instruction segmentation
- "In this work, we propose a model, called SegPoint, that leverages the reasoning capabilities of a multi-modal Large Language Model (LLM) to produce point-wise segmentation masks across a diverse range of tasks: 1) 3D instruction segmentation, 2) 3D referring segmentation, 3) 3D semantic segmentation, and 4) 3D open-vocabulary semantic segmentation." (Section Abstract)
- "To advance 3D instruction research, we introduce a new benchmark, Instruct3D, designed to evaluate segmentation performance from complex and implicit instructional texts, featuring 2,565 point cloud-instruction pairs." (Section Abstract)
- Inference: `In Dynamics`, `Attention Dynamic`, `State Dynamic`, and `Out Dynamics` are inferred from the architecture description: "The input of the framework is the text instructions  $i_{txt}$  and point cloud  $i_{point} \in \mathbb{R}^{N \times (3+F)}$ ." and "Given input point cloud and text query, the multimodal LLM  $\mathcal F$  generates text output." and "Per-point embeddings  $\boldsymbol f_{\mathcal P}$ ... yield the final segmentation masks." These indicate finite per-example point/token processing with no runtime retrieval/controller.

### Task: 3D referring segmentation
- "Lately, more practical tasks have emerged, such as 3D referring segmentation [1,4,17,23,67,79], which extends referring expression segmentation [8,10,11,33–35] to 3D and segments a target instance based on explicit linguistic descriptions..." (Section 2.2 3D Point Cloud Segmentation)
- "Referring Segmentation Dataset. We use template prompts: \"USER: <POINT> Can you segment the <u>object</u> {description} in this point cloud? ASSISTANT: {category} <SEG>.\", where {description} is the given explicit description from referring segmentation dataset." (Section 4.1 Datasets and Evaluation Metrics)
- Inference: `In Dynamics`, `Attention Dynamic`, `State Dynamic`, and `Out Dynamics` are inferred from the same model pipeline statements: finite N-point and tokenized text input, with direct mask generation from per-point embeddings and `<SEG>` outputs (Sections 3.2 and Fig. 2 caption).

### Task: 3D semantic segmentation
- "3D semantic segmentation [3,7,19,45,53,74,80] assigns each point in a 3D space to specific, predefined classes." (Section 2.2 3D Point Cloud Segmentation)
- "Semantic Segmentation Dataset. We use two strategies to generate templates. 1) segment the specific category: \"USER: <POINT> Can you segment the {category} category in this point cloud? ASSISTANT: {category} <SEG>.\" ... 2) segment all the categories: \"USER: <POINT> Can you segment all the semantic masks in this point cloud and output separate masks for each category...\"" (Section 4.1 Datasets and Evaluation Metrics)
- Inference: `In Dynamics`, `Attention Dynamic`, `State Dynamic`, and `Out Dynamics` are inferred from the described single-pass point+text pipeline and per-point mask prediction mechanism (Sections 3.2 and Fig. 2 caption).

### Task: 3D open-vocabulary semantic segmentation
- "...and 3D open-vocabulary segmentation [12,40,42,57], designed to identify and segment unseen objects beyond a fixed set of known categories." (Section 2.2 3D Point Cloud Segmentation)
- "Table 4 shows our method's open-vocabulary segmentation performance which is directly evaluated on ScanNet++ [74]..." (Section 4.6 Results on Open-vocabulary Semantic Segmentation)
- Inference: `In Dynamics`, `Attention Dynamic`, `State Dynamic`, and `Out Dynamics` are inferred from the same finite point+text processing and direct mask-generation pipeline used across tasks (Sections 3.2, 4.1, and Fig. 2 caption).
