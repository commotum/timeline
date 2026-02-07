# Maximizing the Position Embedding for Vision Transformers with Global Average Pooling (2025)
Source: Maximizing the Position Embedding for Vision Transformers (MPVG).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| image classification | images | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | class predictions | 0D | Fixed |
| object detection | images | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | bounding boxes and masks | 2D (x, y) (inferred) | Not specified in the paper. |
| semantic segmentation | images | 2D (x, y) | Fixed | Static (inferred) | Direct (inferred) | segmentation map (inferred) | 2D (x, y) (inferred) | Fixed (inferred) |

## Summary
The paper evaluates MPVG on three computer vision tasks: image classification, object detection, and semantic segmentation, all operating on image inputs. The inputs are 2D (x, y) images with fixed training or crop sizes, while outputs range from 0D class labels to 2D spatial outputs (boxes/masks or segmentation maps). Attention and state dynamics are not explicitly discussed; based on the described MSA/MLP transformer stack they are treated as Static and Direct (inferred), and detection output dynamics are not specified.

## Evidence
### Task: image classification
- "surpassing CNNs in various tasks such as image classification, object detection, and semantic segmentation." (Section: Introduction)
- "H and W are the height and width of the image" (Section: Preliminary: Absolute Position Embedding)
- "All vision transformers are trained on 224×224 resolution images for 300 epochs" (Section: Image Classification)
- "the output of this token is then used to make class predictions via Multi-Layer Perceptron (MLP)" (Section: Introduction)
- Inference: Attention Dynamic = Static and State Dynamic = Direct (inferred) because the model is a fixed-token MSA+MLP stack ("x'_l = MSA(LN_l(x_l)) + x_l", Section: Preliminary: Absolute Position Embedding).

### Task: object detection
- "On object detection, we evaluate our methods on COCO 2017 (Lin et al. 2014)." (Section: Object Detection)
- "AP box / AP mask" (Table 3: Performance comparison of Object Detection on COCO2017)
- "COCO 2017 | ViT-Adapter-Ti | Mask R-CNN | DeiT-Ti                | 1024" (Table 8: Hyperparameter settings for object detection on COCO 2017)
- "H and W are the height and width of the image" (Section: Preliminary: Absolute Position Embedding)
- Inference: Out Dimension = 2D (x, y) (inferred) because the outputs are box/mask metrics ("AP box / AP mask"); Attention Dynamic = Static and State Dynamic = Direct (inferred) based on the MSA stack ("x'_l = MSA(LN_l(x_l)) + x_l").

### Task: semantic segmentation
- "On semantic segmentation, we evaluate our methods on ADE20K (Zhou et al. 2019)." (Section: Semantic Segmentation)
- "Performance comparison of Semantic Segmentation on ADE20K." (Table 4 caption)
- "ADE20K | ViT-Adapter-Ti | UperNet   | DeiT-Ti                | 512" (Table 9: Hyperparameter settings for semantic segmentation on ADE20K)
- "H and W are the height and width of the image" (Section: Preliminary: Absolute Position Embedding)
- Inference: Output = segmentation map and Out Dimension = 2D (x, y) (inferred) because the task is semantic segmentation; Out Dynamics = Fixed (inferred) because the crop size is fixed ("... | 512"); Attention Dynamic = Static and State Dynamic = Direct (inferred) based on the MSA stack ("x'_l = MSA(LN_l(x_l)) + x_l").
