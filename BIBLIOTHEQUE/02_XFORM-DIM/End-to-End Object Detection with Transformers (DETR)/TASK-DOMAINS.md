# End-to-End Object Detection with Transformers (DETR) (Not specified in the paper.)
Source: End-to-End Object Detection with Transformers (DETR).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Object detection | images | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | set of bounding boxes and category labels | 2D (x, y); 0D | Fixed |
| Panoptic segmentation | images | 2D (x, y) | Capped (inferred) | Static (inferred) | Constructed (inferred) | panoptic segmentation masks (per-pixel categories) | 2D (x, y) | Capped (inferred) |

## Summary
The paper covers two vision tasks: object detection and panoptic segmentation, both operating on 2D images. Object detection outputs fixed-size sets of bounding boxes and labels, while panoptic segmentation outputs per-pixel masks; both outputs remain within the bounded image sizes described in the paper (capped, inferred). The transformer uses global attention over the full input sequence and learned object queries, which supports Static attention and Constructed state (both inferred).

## Evidence
### Task: Object detection
- "The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest." (Introduction)
- "DETR uses a conventional CNN backbone to learn a 2D representation of an input image." (Fig. 2 caption)
- "DETR infers a fixed-size set of N predictions, in a single pass through the decoder." (Section 3.1 Object detection set prediction loss)
- Inference: In Dynamics = Capped (inferred) because input size is bounded ("shortest side is at least 480 and at most 800 pixels while the longest at most 1333" (Section 4 Experiments, Technical details)); Attention Dynamic = Static (inferred) because attention aggregates "information from the entire input sequence" (Section 2.2 Transformers and Parallel Decoding); State Dynamic = Constructed (inferred) because the model uses "learned object queries" to reason about objects ("Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context" (Abstract)).

### Task: Panoptic segmentation
- "DETR can be naturally extended by adding a mask head on top of the decoder outputs." (Section 4.4 DETR for panoptic segmentation)
- "A binary mask is generated in parallel for each detected object, then the masks are merged using pixel-wise argmax." (Fig. 8 caption)
- "To predict the final panoptic segmentation we simply use an argmax over the mask scores at each pixel" (Section 4.4 DETR for panoptic segmentation)
- Inference: In Dynamics = Capped (inferred) because input size is bounded ("shortest side is at least 480 and at most 800 pixels while the longest at most 1333" (Section 4 Experiments, Technical details)); Out Dynamics = Capped (inferred) because mask outputs are per-pixel over the bounded image size ("argmax over the mask scores at each pixel" (Section 4.4 DETR for panoptic segmentation)); Attention Dynamic = Static (inferred) because attention aggregates "information from the entire input sequence" (Section 2.2 Transformers and Parallel Decoding); State Dynamic = Constructed (inferred) because the model uses "learned object queries" to reason about objects ("Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context" (Abstract)).
