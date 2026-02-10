1. Number of distinct tasks evaluated: 4

> "It set new performance records on 4 representative vision tasks, including ImageNet-V2 image classification, COCO object detection, ADE20K semantic segmentation, and Kinetics-400 video action classification." (Abstract)

> "We conduct experiments on ImageNet-1K image classification (V1 and V2) [18, 55], COCO object detection [44], and ADE20K semantic segmentation [85]. For the 3B model experiments, we also report the accuracy on Kinetics-400 video action recognition [37]." (Section 4.1. Tasks and Datasets)

2. Number of trained model instances required to cover all tasks: 4

> "For these experiments, we first conduct ImageNet-22K pre-training, and then fine-tune the pre-trained models on individual down-stream recognition tasks." (Section A2.1. SwinV2-B and SwinV2-L Settings)

> "Fine-tuning on ImageNet-1K image classification We adopt an input image size of  $640 \times 640$  for experiments." (Section A2.2. SwinV2-G Settings)

> "**Fine-tuning on COCO object detection** We first conduct inter-mediate fine-tuning using the Objects-365 V2 dataset." (Section A2.2. SwinV2-G Settings)

> "Fine-tuning on ADE20K semantic segmentation The input image size (window size) is set  $640 \times 640$  ( $40 \times 40$ )." (Section A2.2. SwinV2-G Settings)

> "#### Fine-tuning on Kinetics-400 video action recognition" (Section A2.2. SwinV2-G Settings)

3. Task-Model Ratio

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
