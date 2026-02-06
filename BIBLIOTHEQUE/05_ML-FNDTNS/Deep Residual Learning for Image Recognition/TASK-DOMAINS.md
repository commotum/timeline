# Deep Residual Learning for Image Recognition (2015)
Source: Deep Residual Learning for Image Recognition.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image classification | Images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Class labels (softmax) (inferred) | 0D (inferred) | Fixed (inferred) |
| Object detection | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Class labels + bounding boxes (inferred) | 0D; 2D (x, y) (inferred) | Capped (inferred) |
| Object localization | Images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Class labels + bounding boxes (inferred) | 0D; 2D (x, y) (inferred) | Fixed (inferred) |
| Segmentation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper covers image classification, object detection, and ImageNet localization on 2D image inputs, and it mentions COCO segmentation only as a competition task without methodological detail. Classification is described with fixed-size image crops and softmax classifiers, supporting fixed 0D outputs, while detection/localization add bounding-box regression with capped or fixed outputs. Attention and state dynamics are inferred as static and constructed based on the described feedforward convolutional and RPN-based pipelines.

## Evidence
### Task: Image classification
- "We evaluate our method on the ImageNet 2012 classification dataset [36] that consists of 1000 classes." (Section 4.1 ImageNet Classification)
- "The network inputs are  $32 \times 32$  images, with the per-pixel mean subtracted." (Section 4.2 CIFAR-10 and Analysis)
- "The first layer is  $3 \times 3$  convolutions." (Section 4.2 CIFAR-10 and Analysis)
- "A  $224 \times 224$  crop is randomly sampled from an image or its horizontal flip." (Section 3.4 Implementation)
- "The network ends with a global average pooling layer and a 1000-way fully-connected layer with softmax." (Section 3.3 Network Architectures)
- "The network ends with a global average pooling, a 10-way fully-connected layer, and softmax." (Section 4.2 CIFAR-10 and Analysis)
- Inference: Inferred 2D fixed inputs, static attention, and constructed state from fixed-size crops and convolutional processing ("224 \times 224 crop"; "32 \times 32 images"; "3 \times 3 convolutions"). Inferred 0D fixed outputs from softmax classifiers ("1000-way fully-connected layer with softmax"; "10-way fully-connected layer, and softmax").

### Task: Object detection
- "Table 7 and 8 show the object detection baseline results on PASCAL VOC 2007 and 2012 [5] and COCO [26]." (Section 4.3 Object Detection on PASCAL and MS COCO)
- "We adopt *Faster R-CNN* [32] as the detection method." (Section 4.3 Object Detection on PASCAL and MS COCO)
- "These layers are shared by a region proposal network (RPN, generating 300 proposals) [32] and a Fast R-CNN detection network [7]." (Appendix A. Object Detection Baselines)
- "The final classification layer is replaced by two sibling layers (classification and box regression [7])." (Appendix A. Object Detection Baselines)
- "the image's shorter side is s=600 pixels." (Appendix B. Object Detection Improvements)
- "We compute the full-image shared conv feature maps using those layers whose strides on the image are no greater than 16 pixels" (Appendix A. Object Detection Baselines)
- Inference: Inferred 2D capped inputs from fixed-scale testing ("shorter side is s=600 pixels"). Inferred outputs as class labels plus bounding boxes and capped output dynamics from box regression and RPN proposals ("classification and box regression"; "RPN, generating 300 proposals"). Static attention and constructed state are inferred from conv feature maps and the RPN/region-based pipeline.

### Task: Object localization
- "The ImageNet Localization (LOC) task [36] requires to classify and localize the objects." (Section C. ImageNet Localization)
- "the *cls* layer has a 1000-d output" (Section C. ImageNet Localization)
- "the *reg* layer has a  $1000\times4$ -d output consisting of box regressors for 1000 classes." (Section C. ImageNet Localization)
- "Our localization algorithm is based on the RPN framework of [32] with a few modifications." (Section C. ImageNet Localization)
- "As in our ImageNet classification training (Sec. 3.4), we randomly sample  $224 \times 224$  crops for data augmentation." (Section C. ImageNet Localization)
- "For testing, the network is applied on the image fully-convolutionally." (Section C. ImageNet Localization)
- Inference: Inferred 2D capped inputs from fixed-size crops and fully-convolutional testing ("224 \times 224 crops"; "applied on the image fully-convolutionally"). Inferred output structure and fixed dynamics from per-class classification/regression outputs ("1000-d output"; "1000\times4-d output"). Static attention and constructed state are inferred from the RPN-based localization pipeline ("based on the RPN framework").

### Task: Segmentation
- "we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation." (Abstract)

---

## CSV Output (required)
