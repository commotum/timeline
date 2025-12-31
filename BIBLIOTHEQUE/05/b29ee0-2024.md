# SegPoint: Segment Any Point Cloud via Large Language Model (2024)
Source: b29ee0-2024.pdf

## Core reasons
- Introduces SegPoint, a unified 3D point cloud segmentation model that leverages LLM reasoning across multiple segmentation tasks.
- Presents concrete architectural components (point encoder, LLM, and geometric modules) to perform point-wise segmentation, indicating a modeling/architecture contribution rather than a positional encoding or benchmark-only focus.

## Evidence extracts
- "In this work,
we propose a model, called SegPoint, that leverages the reasoning ca-
pabilities of a multi-modal Large Language Model (LLM) to produce
point-wise segmentation masks across a diverse range of tasks: 1) 3D
instruction segmentation, 2) 3D referring segmentation, 3) 3D seman-
tic segmentation, and 4) 3D open-vocabulary semantic segmentation." (p. 1)
- "The overall architecture of SegPoint is presented in Fig. 2. SegPoint mainly
comprises four parts: i) a pre-trained point encoder E tailored for aligning with
textual data; ii) a large language model F endowed with advanced reasoning
capabilities; iii) a Geometric Enhancer Module G responsible for extracting geo-
metric representation from input point clouds and infusing these priors into the
point encoder; and iv) a Geometric-guided Feature Propagation P which is key
to achieving precise mask generation." (p. 5)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
