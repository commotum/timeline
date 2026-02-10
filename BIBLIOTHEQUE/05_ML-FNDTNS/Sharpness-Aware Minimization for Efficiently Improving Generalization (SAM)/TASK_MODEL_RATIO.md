1. **Number of distinct tasks evaluated:** 11

Verbatim evidence:
- "including image classification from scratch (including on CIFAR-10, CIFAR-100, and ImageNet), finetuning pretrained models, and learning with noisy labels." (Section 3: EMPIRICAL EVALUATION)
- "Beyond CIFAR-{10, 100}, we have also evaluated SAM on the SVHN (Netzer et al., 2011) and Fashion-MNIST datasets (Xiao et al., 2017)." (Section 3.1: IMAGE CLASSIFICATION FROM SCRATCH)
- "FGVC_Aircraft", "Flowers", "Oxford_IIIT_Pets", "Stanford_Cars", "CIFAR-10", "CIFAR-100", "Birdsnap", "Food101", "ImageNet" (Table 3, Section 3.2: FINETUNING)
- "In particular, we measure the effect of applying SAM in the classical noisy-label setting for CIFAR-10" (Section 3.3: ROBUSTNESS TO LABEL NOISE)

An explicit total count of distinct capability tasks is Not specified in the paper.
Counted distinct capability tasks (unique dataset-level classification tasks): CIFAR-10, CIFAR-100, ImageNet, SVHN, Fashion-MNIST, FGVC_Aircraft, Flowers, Oxford_IIIT_Pets, Stanford_Cars, Birdsnap, Food101.

2. **Number of trained model instances required to cover all tasks:** 11 models

Verbatim evidence:
- "We finetune these models on each of several target datasets by training each model starting from the aforementioned checkpoint" (Section 3.2: FINETUNING)
- "Weights are initialized to the values provided by the publicly available checkpoints, except the last dense layer, which change size to accomodate the new number of classes, that is randomly initialized." (Section C.2: FINETUNING DETAILS)

An explicit total count of required trained model instances is Not specified in the paper.
These statements indicate task-specific training (and task-specific classification heads), so one separately trained model instance is required per distinct task capability.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
