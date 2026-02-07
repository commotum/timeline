# Learning Transferable Visual Models From Natural Language Supervision (Not specified in the paper)
Source: Learning Transferable Visual Models From Natural Language Supervision.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matching (image-text pairing) | images; text snippets | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | paired/not paired or correct pairing | 0D (inferred) | Not specified in the paper. |
| retrieval (image-text) | text queries or images | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | retrieved images or texts | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. |
| classification (image objects) | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | object class labels (including fine-grained) | 0D (inferred) | Not specified in the paper. |
| recognition (OCR) | images of text | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | characters/words | 1D (t) (inferred) | Not specified in the paper. |
| classification (OCR-based semantics) | images of rendered text | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | semantic task labels | 0D (inferred) | Not specified in the paper. |
| recognition (actions in videos) | videos | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | action labels | 0D (inferred) | Not specified in the paper. |
| classification (geo-localization/scene) | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | location/scene labels | 0D (inferred) | Not specified in the paper. |
| prediction (geo-coordinates) | images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | GPS coordinates | 2D (x, y) (inferred) | Not specified in the paper. |
| recognition (facial emotion) | face images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | emotion labels | 0D (inferred) | Not specified in the paper. |
| classification (face attributes) | face images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | race/gender/age labels | 0D (inferred) | Not specified in the paper. |
| classification (satellite imagery) | satellite images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels | 0D (inferred) | Not specified in the paper. |
| detection (lymph node tumors) | medical images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | tumor present/absent label | 0D (inferred) | Not specified in the paper. |
| counting (objects) | synthetic scene images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | object count | 0D (inferred) | Not specified in the paper. |
| classification (traffic signs) | traffic sign images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | traffic sign class labels | 0D (inferred) | Not specified in the paper. |
| prediction (distance to nearest car) | driving images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | distance to nearest car | 0D (inferred) | Not specified in the paper. |
| classification (surveillance scenes) | CCTV images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | scene/subject labels | 0D (inferred) | Not specified in the paper. |
| detection (surveillance presence/absence) | CCTV images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | presence/absence labels for small objects | 0D (inferred) | Not specified in the paper. |
| recognition (celebrity identity) | face images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | identity labels | 0D (inferred) | Not specified in the paper. |

## Summary
The paper describes CLIP handling image-text pairing and retrieval, broad image classification (including fine-grained categories), OCR, action recognition in videos, and geo-localization, plus specialized tasks like satellite classification, tumor detection, counting, surveillance, and identity recognition. The task domains span text, images, and videos, implying 1D, 2D, and 3D inputs and 0D/2D outputs, with these dimensions inferred from task descriptions. Input/output dynamics and attention/state dynamics are not explicitly specified for these tasks in the paper.

## Evidence
### Task: matching (image-text pairing)
- "pre-training task of predicting which caption goes with which image" (Abstract)
- Inference: Input/output dimensions inferred from images and text snippets with a pairing decision output.

### Task: retrieval (image-text)
- "CLIP pre-trains for the task of image-text retrieval" (Section E.1)
- "find relevant images in a database given text, or relevant text given an image." (Section 7)
- Inference: In/out dimensions inferred from text queries or images and retrieved images or texts.

### Task: classification (image objects)
- "carry out arbitrary image classification tasks." (Section 7)
- "many types of fine-grained object classification." (Abstract)
- Inference: Input/output dimensions inferred from image classification producing class labels.

### Task: recognition (OCR)
- "task of optical character recognition (OCR)." (Section E.2)
- "perform low-level character and word recognition" (Section E.2)
- Inference: Input/output dimensions inferred from images of text and character/word outputs.

### Task: classification (OCR-based semantics)
- "check the ability of a model to use OCR to perform a semantic task." (Section E.2)
- Inference: Input/output dimensions inferred from images of rendered text and semantic task labels.

### Task: recognition (actions in videos)
- "action recognition in videos" (Abstract)
- "action classification datasets which measure the ability of a model to recognize verbs." (Section E.3)
- Inference: Input dimension inferred as spatiotemporal video (3D) and output as action labels.

### Task: classification (geo-localization/scene)
- "geo-localization and scene recognition (Country211, SUN397)" (Section 3.2)
- Inference: Input/output dimensions inferred from images and location/scene labels.

### Task: prediction (geo-coordinates)
- "IM2GPS is a regression benchmark, we guess the GPS coordinates" (Section E.4)
- Inference: Output dimension inferred as 2D coordinates from the GPS coordinate prediction task.

### Task: recognition (facial emotion)
- "facial emotion recognition" (Section 3.2)
- Inference: Input/output dimensions inferred from face images and emotion labels.

### Task: classification (face attributes)
- "Race, Gender, and Age classification of images in FairFace" (Table 3)
- Inference: Input/output dimensions inferred from face images and demographic labels.

### Task: classification (satellite imagery)
- "satellite image classification (EuroSAT and RESISC45)" (Section 3.1.5)
- Inference: Input/output dimensions inferred from satellite images and class labels.

### Task: detection (lymph node tumors)
- "lymph node tumor detection (PatchCamelyon)" (Section 3.1.5)
- Inference: Input/output dimensions inferred from medical images and a detection label.

### Task: counting (objects)
- "counting objects in synthetic scenes (CLEVRCounts)" (Section 3.1.5)
- Inference: Output dimension inferred as a scalar count.

### Task: classification (traffic signs)
- "German traffic sign recognition (GTSRB)" (Section 3.1.5)
- Inference: Input/output dimensions inferred from traffic sign images and class labels.

### Task: prediction (distance to nearest car)
- "recognizing distance to the nearest car (KITTI Distance)" (Section 3.1.5)
- Inference: Output dimension inferred as a scalar distance estimate.

### Task: classification (surveillance scenes)
- "classification of images from CCTV cameras" (Section 7.2)
- Inference: Input/output dimensions inferred from CCTV images and scene/subject labels.

### Task: detection (surveillance presence/absence)
- "detecting the presence or absence of small objects" (Section 7.2)
- Inference: Input/output dimensions inferred from CCTV images and presence/absence labels.

### Task: recognition (celebrity identity)
- "zero-shot celebrity identification" (Section 7.2)
- Inference: Input/output dimensions inferred from face images and identity labels.

---

## CSV Output (required)
CSV file: `/home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/Learning Transferable Visual Models From Natural Language Supervision/.TASK-DOMAINS.csv.tmp.4c5f449bec554878b80c18d382c88c1d`
