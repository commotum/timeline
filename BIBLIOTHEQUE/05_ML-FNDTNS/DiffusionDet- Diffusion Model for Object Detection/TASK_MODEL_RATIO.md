1. **Number of distinct tasks evaluated:** 1 task (object detection).

- "In this work, we propose DiffusionDet, which tackles the object detection task with a diffusion model..." (Section 1. Introduction)
- "In this work, we aim to solve the object detection task via the diffusion model." (Section 3.1. Preliminaries)
- "We report box average precision over multiple IoU thresholds (AP)..." (Section 4. Experiments, COCO)
- "We adopt MS-COCO style box metric AP, AP $_{50}$ and AP $_{75}$ in LVIS evaluation." (Section 4. Experiments, LVIS v1.0)
- "Following previous settings [54, 90, 109, 113], we adopt evaluation metrics as AP under IoU threshold 0.5." (Section 4. Experiments, CrowdHuman)

2. **Number of trained model instances required to cover all tasks:** 1 model.

- "As a probabilistic model, DiffusionDet has an attractive superiority of flexibility, *i.e.*, we can train the network once and use the same network parameters under diverse settings in the inference stage..." (Section 1. Introduction)
- "The main properties of DiffusionDet lie on *once training for all inference cases*." (Section 4.2. Main Properties)
- "Once the model is trained, it can be used with changing the number of boxes and the number of iteration steps in inference..." (Section 4.2. Main Properties)
- "Therefore, we can deploy a single DiffusionDet to multiple scenarios..." (Section 4.2. Main Properties)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
