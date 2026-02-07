# Intriguing properties of neural networks (Not specified in the paper.)
Source: Intriguing properties of neural networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (visual recognition) | images (pixel value vectors) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | discrete labels (class labels) | 0D (inferred) | Fixed (inferred) |
| adversarial example generation (targeted perturbation) | image + target label | 2D (x, y) (inferred); 0D (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | perturbed image (adversarial example) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper's concrete tasks are visual image classification and generating adversarially perturbed images for a target label. Inputs are fixed-size images (2D) and fixed label sets; outputs are discrete labels (0D) or perturbed images (2D). Attention and state dynamics are not specified in the OCR text.

## Evidence
### Task: classification (visual recognition)
- "We denote by  $x \in \mathbb{R}^m$  an input image" (Section 2 Framework)
- "a classifier mapping image pixel value vectors to a discrete label set." (Section 4.1 Formal description)
- "Consider a state-of-the-art deep neural network that generalizes well on an object recognition task." (Section 1 Introduction)
- Inference: 2D (x, y) and Fixed input are inferred from "input image" and x in R^m; 0D and Fixed outputs are inferred from a discrete label set.

### Task: adversarial example generation (targeted perturbation)
- "we are able to find adversarial examples, which are obtained by imperceptibly small perturbations to a correctly classified input image" (Section 4 Blind Spots in Neural Networks)
- "For a given  $x\in\mathbb{R}^m$  image and target label  $l\in\{1\dots k\}$ , we aim to solve the following box-constrained optimization problem:" (Section 4.1 Formal description)
- "x+r is the closest image to x classified as l by f." (Section 4.1 Formal description)
- Inference: 2D (x, y) and Fixed input/output are inferred from "image" and x in R^m; the 0D target label input is inferred from l in {1..k}.

---

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/Intriguing properties of neural networks/.TASK-DOMAINS.csv.tmp.68bc3d56b37942318b87b318e756b513" with the same rows and columns as the Task Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
