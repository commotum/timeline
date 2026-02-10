1. **Number of distinct tasks evaluated:** 5

- "In this section, we present *zero-shot transfer* experiments with SAM, the Segment Anything Model. We consider five tasks, four of which differ significantly from the promptable segmentation task used to train SAM." (§7. Zero-Shot Transfer Experiments)
- "Our experiments begin by testing the core goal of promptable segmentation: producing a valid mask from any prompt. We emphasize the challenging scenario of a *single* foreground point prompt, since it is more likely to be ambiguous than other more specific prompts. Next, we present a sequence of experiments that traverse low, mid, and highlevel image understanding and roughly parallel the historical development of the field. Specifically, we prompt SAM to (1) perform edge detection, (2) segment everything, *i.e.* object proposal generation, (3) segment detected objects, *i.e.* instance segmentation, and (4), as a proof-of-concept, to segment objects from free-form text." (§7. Zero-Shot Transfer Experiments)

2. **Number of trained model instances required to cover all tasks:** 3 models

- "**Implementation.** Unless otherwise specified: (1) SAM uses an MAE [47] pre-trained ViT-H [33] image encoder and (2) SAM was trained on SA-1B, noting that this dataset includes only automatically generated masks from the final stage of our data engine." (§7. Zero-Shot Transfer Experiments)
- "**Approach.** Moving to higher-level vision, we use SAM as the segmentation module of an instance segmenter. The implementation is simple: we run a object detector (the ViTDet used before) and prompt SAM with its output boxes." (§7.4. Zero-Shot Instance Segmentation)
- "**Approach.** Finally, we consider an even higher-level task: segmenting objects from free-form text. This experiment is a proof-of-concept of SAM's ability to process text prompts. While we used the exact same SAM in all prior experiments, for this one SAM's training procedure is modified to make it text-aware, but in a way that does not require new text annotations." (§7.5. Zero-Shot Text-to-Mask)
- Not specified in the paper: whether the text-aware SAM can replace the default SAM for all other tasks at the same reported performance.

3. **Task–Model Ratio**

$$
\boxed{
\frac{5\ \text{tasks}}{3\ \text{models}} = 1.67
}
$$
