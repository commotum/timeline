# Multiscale Vision Transformers (Year not specified)
Source: Multiscale Vision Transformers (MViT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Video action classification/recognition | video clips (T frames) | 3D (x, y, t) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action class label(s) | 0D | Fixed (inferred) |
| Spatiotemporal action detection/localization | video clips (T frames) | 3D (x, y, t) | Fixed (inferred) | Static (inferred) | Direct (inferred) | action labels with spatiotemporal bounding boxes | 3D (x, y, t); 0D | Not specified in the paper. |
| Image classification | images (single frame, T=1) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class label | 0D | Fixed (inferred) |

## Summary
MViT is evaluated on visual recognition tasks spanning video action classification (Kinetics/SSv2/Charades), spatiotemporal action detection on AVA, and image classification on ImageNet-1K. Inputs are 3D spatiotemporal clips for video tasks and 2D images for ImageNet, with fixed clip lengths/crops and fixed class sets in the reported experiments, while outputs are class labels or spatiotemporal boxes with labels. Attention and state dynamics are not explicitly labeled; based on the described attention over provided sequences and feedforward classifiers, they are treated as static and direct (inferred).

## Evidence
### Task: Video action classification/recognition
- "We present Multiscale Vision Transformers (MViT) for video and image recognition." (Abstract)
- "We use Kinetics-400 [59] (K400) (~240k training videos in 400 classes) and Kinetics-600 [11]." (Section 4. Experiments: Video Recognition)
- "We report top-1 and top-5 classification accuracy (%) on the validation set" (Section 4. Experiments: Video Recognition)
- "the input to the network are T frames with a temporal stride of τ" (Section 4. Experiments: Video Recognition)
- Inference: In/Out Dynamics marked Fixed and Attention/State marked Static/Direct because inputs are fixed-length clips and attention/classification are computed over the provided sequence ("the input to the network are T frames with a temporal stride of τ"; "Attention is now computed on these shortened vectors"; "the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)."). (Sections 4. Experiments: Video Recognition; 3.1; 3.2)

### Task: Spatiotemporal action detection/localization
- "AVA [39] is a dataset with for spatiotemporal-localization of human actions. We validate our MViT on this detection task." (Section 4.1)
- "The AVA dataset [39] has bounding box annotations for spatiotemporal localization of (possibly multiple) human actions." (Section D.2. Details: AVA Action Detection)
- "The RoI features are then max-pooled and fed to a per-class, sigmoid classifier for prediction." (Section D.2. Details: AVA Action Detection)
- Inference: In Dynamics marked Fixed and Attention/State marked Static/Direct because inputs are fixed-length clips and attention/classification are computed over provided sequences and RoI features ("the input to the network are T frames with a temporal stride of τ"; "Attention is now computed on these shortened vectors"; "The RoI features are then max-pooled and fed to a per-class, sigmoid classifier for prediction."). (Sections 4. Experiments: Video Recognition; 3.1; D.2)

### Task: Image classification
- "We apply our video models on static image recognition by using them with single frame, T=1, on ImageNet-1K [22]." (Section 5. Experiments: Image Recognition)
- "ImageNet-1K [22] dataset that has ~1.28M images in 1000 classes." (Section D.5. Details: ImageNet)
- "We train models on the train set and report top-1 and top-5 classification accuracy (%) on the val set." (Section D.5. Details: ImageNet)
- Inference: In/Out Dynamics marked Fixed and Attention/State marked Static/Direct because inputs are single-frame crops and attention/classification are computed over provided sequences with a linear head ("single frame, T=1"; "single center-crop with resolution of 224^2"; "Attention is now computed on these shortened vectors"; "the class embedding is extracted and passed through a linear layer to predict the desired output (e.g. class)."). (Sections 5. Experiments: Image Recognition; D.5; 3.1; 3.2)
