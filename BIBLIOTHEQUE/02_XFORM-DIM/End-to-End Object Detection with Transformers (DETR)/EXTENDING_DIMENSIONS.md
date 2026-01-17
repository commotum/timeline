## 1. Basic Metadata
- Title: Title not specified.
- Authors: "Nicolas Carion\*, Francisco Massa\*, Gabriel Synnaeve, Nicolas Usunier,"; "Alexander Kirillov, and Sergey Zagoruyko" (Front matter)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper presents DETR, which "views object detection as a direct set prediction problem." and uses "The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture." to "directly output the final set of predictions in parallel." (Abstract).

## 3. Tasks Evaluated
- Task name: Object detection
  - Task type: Detection
  - Dataset(s) used: COCO 2017 detection dataset
  - Domain: Images (RGB; COCO)
  - Quotes: "The goal of object detection is to predict a set of bounding boxes and category labels for each object of interest." (Pre-Section 2 text); "We evaluate DETR on one of the most popular object detection datasets, COCO [24], against a very competitive Faster R-CNN baseline [37]." (Pre-Section 2 text); "Dataset. We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images." (Section 4 Experiments); "with 3 color channels<sup>2</sup>" (Section 3.2 DETR architecture, Backbone)

- Task name: Panoptic segmentation
  - Task type: Segmentation
  - Dataset(s) used: COCO panoptic annotations / COCO 2017 panoptic segmentation dataset
  - Domain: Images (RGB; COCO)
  - Quotes: "The design ethos of DETR easily extend to more complex tasks. In our experiments, we show that a simple segmentation head trained on top of a pre-trained DETR outperforms competitive baselines on Panoptic Segmentation [19], a challenging pixel-level recognition task that has recently gained popularity." (Pre-Section 2 text); "We perform our experiments on the panoptic annotations of the COCO dataset that has 53 stuff categories in addition to 80 things categories." (Section 4.4); "We train DETR to predict boxes around both stuff and things classes on COCO, using the same recipe." (Section 4.4)

## 4. Domain and Modality Scope
- Domain scope: Single domain (COCO images) with detection and panoptic segmentation annotations. Evidence: "Dataset. We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images." (Section 4 Experiments); "Each image is annotated with bounding boxes and panoptic segmentation." (Section 4 Experiments)
- Modality: Single modality (RGB images). Evidence: "with 3 color channels<sup>2</sup>" (Section 3.2 DETR architecture, Backbone)
- Multiple domains within the same modality? Not stated.
- Multiple modalities? Not stated.
- Domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Object detection | Yes (DETR weights later reused for panoptic head) | Not stated | Yes (FFN detection head) | "a simple segmentation head trained on top of a pre-trained DETR" (Pre-Section 2 text); 'We pass each output embedding of the decoder to a shared feed forward network (FFN) that predicts either a detection (class and bounding box) or a "no object" class.' (Fig. 2 caption) |
| Panoptic segmentation | Yes (DETR + added mask head) | Optional; two-step or joint training | Yes (mask head) | "DETR can be naturally extended by adding a mask head on top of the decoder outputs." (Section 4.4); "The mask head can be trained either jointly, or in a two steps process, where we train DETR for boxes only, then freeze all the weights and train only the mask head for 25 epochs." (Section 4.4) |

## 6. Input and Representation Constraints
- Input modality and dimensionality: "DETR uses a conventional CNN backbone to learn a 2D representation of an input image." (Fig. 2 caption); "with 3 color channels<sup>2</sup>" (Section 3.2 DETR architecture, Backbone)
- Downsampled spatial grid: "Typical values we use are C = 2048 and  $H, W = \frac{H_0}{32}, \frac{W_0}{32}$ ." (Section 3.2 DETR architecture, Backbone)
- Padding for batching: "The input images are batched together, applying 0-padding adequately to ensure they all have the same dimensions  $(H_0, W_0)$  as the largest image of the batch." (Section 3.2 DETR architecture, Backbone footnote)
- Resizing constraints: "We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333 [50]." (Section 4 Experiments)
- Fixed number of output queries: "DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image." (Section 3.1); "All models were trained with N=100 decoder query slots." (Section A.4)
- Fixed patch size: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified; encoder sequence length is HW and decoder length is fixed N. Evidence: "The encoder expects a sequence as input, hence we collapse the spatial dimensions of  $z_0$  into one dimension, resulting in a  $d \times HW$  feature map." (Section 3.2 DETR architecture); "DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image." (Section 3.1); "All models were trained with N=100 decoder query slots." (Section A.4)
- Sequence length fixed or variable: Input length varies with resizing and padding; output length fixed N. Evidence: "We use scale augmentation, resizing the input images such that the shortest side is at least 480 and at most 800 pixels while the longest at most 1333 [50]." (Section 4 Experiments); "The input images are batched together, applying 0-padding adequately to ensure they all have the same dimensions  $(H_0, W_0)$  as the largest image of the batch." (Section 3.2 DETR architecture, Backbone footnote)
- Attention type: Global self-attention. Evidence: "The self-attention mechanisms of transformers, which explicitly model all pairwise interactions between elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions." (Pre-Section 2 text); "Attention mechanisms [2] are neural network layers that aggregate information from the entire input sequence." (Section 2.2)
- Mechanisms to manage computational cost: No windowing or sparse attention described; higher-resolution variant increases attention cost. Evidence: "This modification increases the resolution by a factor of two, thus improving performance for small objects, at the cost of a 16x higher cost in the self-attentions of the encoder, leading to an overall 2x increase in computational cost." (Technical details)

## 8. Positional Encoding (Critical Section)
- Mechanism: Fixed absolute 2D sine/cosine spatial positional encoding. Evidence: "In our model we use a fixed absolute encoding to represent these spatial positions." (Section A.4 Spatial positional encoding); "Specifically, for both spatial coordinates of each embedding we independently use  $\frac{d}{2}$  sine and cosine functions with different frequencies." (Section A.4 Spatial positional encoding)
- Output positional encodings (object queries): Learned. Evidence: "These input embeddings are learnt positional encodings that we refer to as *object queries*, and similarly to the encoder, we add them to the input of each attention layer." (Section 3.2 DETR architecture, Transformer decoder); "When we pass the encodings to the attentions, they are shared across all layers, and the output encodings (object queries) are always learned." (Section 4.2)
- Where applied: Added to inputs/queries at every attention layer in encoder and decoder. Evidence: "Since the transformer architecture is permutation-invariant, we supplement it with fixed positional encodings [31,3] that are added to the input of each attention layer." (Section 3.2 DETR architecture, Transformer encoder); "Image features from the CNN backbone are passed through the transformer encoder, together with spatial positional encoding that are added to queries and keys at every multihead self-attention layer." (Section A.3 Detailed architecture)
- Fixed vs modified/ablated: Fixed absolute spatial PE in the model, and ablations compare fixed vs learned. Evidence: "In our model we use a fixed absolute encoding to represent these spatial positions." (Section A.4 Spatial positional encoding); "We experiment with various combinations of fixed and learned encodings, results can be found in table 3." (Section 4.2)

## 9. Positional Encoding as a Variable
- Core research variable? Yes; ablated and compared. Evidence: "We experiment with various combinations of fixed and learned encodings, results can be found in table 3." (Section 4.2)
- Multiple positional encodings compared? Yes. Evidence: "We experiment with various combinations of fixed and learned encodings, results can be found in table 3." (Section 4.2)
- Claim PE choice is not critical? Not claimed; they state it contributes. Evidence: "Given these ablations, we conclude that transformer components: the global self-attention in encoder, FFN, multiple decoder layers, and positional encodings, all significantly contribute to the final object detection performance." (Section 4.2)

## 10. Evidence of Constraint Masking
- Model sizes: "Like Faster R-CNN with FPN this model has 41.3M parameters, out of which 23.5M are in ResNet-50, and 17.8M are in the transformer." (Section 4.1)
- Dataset size: "Dataset. We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images." (Section 4 Experiments)
- Scaling model depth: "Table 2: Effect of encoder size. Each row corresponds to a model with varied number of encoder layers and fixed number of decoder layers. Performance gradually improves with more encoder layers." (Table 2 caption)
- Architectural scaling for small objects: "This modification increases the resolution by a factor of two, thus improving performance for small objects, at the cost of a 16x higher cost in the self-attentions of the encoder, leading to an overall 2x increase in computational cost." (Technical details)
- Training schedule effects: "This schedule adds 1.5 AP compared to the shorter schedule." (Section 4 Experiments)

## 11. Architectural Workarounds
- Fixed set prediction slots: "DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image." (Section 3.1); "All models were trained with N=100 decoder query slots." (Section A.4)
- Auxiliary decoding losses: "We add prediction FFNs and Hungarian loss after each decoder layer." (Section 3.2 DETR architecture, Auxiliary decoding losses)
- Resolution-increasing backbone change: "Following [21], we also increase the feature resolution by adding a dilation to the last stage of the backbone and removing a stride from the first convolution of this stage." (Technical details)
- Task-specific panoptic head: "DETR can be naturally extended by adding a mask head on top of the decoder outputs." (Section 4.4); "To make the final prediction and increase the resolution, an FPN-like architecture is used." (Section 4.4)
- Padding for variable image sizes: "The input images are batched together, applying 0-padding adequately to ensure they all have the same dimensions  $(H_0, W_0)$  as the largest image of the batch." (Section 3.2 DETR architecture, Backbone footnote)

## 12. Explicit Limitations and Non-Claims
- Small object performance limitation: "It obtains, however, lower performances on small objects." (Pre-Section 2 text); "This new design for detectors also comes with new challenges, in particular regarding training, optimization and performances on small objects." (Section 5 Conclusion)
- Fixed maximum number of instances: "By design, DETR cannot predict more objects than it has query slots, i.e. 100 in our experiments." (Section A.5)
- Saturation at high instance counts: "Notably, when the image contains all 100 instances, the model only detects 30 on average, which is less than if the image contains only 50 instances that are all detected." (Section A.5)
- Explicit non-claims about open-world or multi-domain learning: Not stated.

### 13. Constraint Profile (Synthesis)
**Constraint Profile:**
- Domain scope: COCO images only, with detection and panoptic segmentation annotations in the same dataset.
- Task structure: Fixed-set prediction for object detection, plus a panoptic mask head added on top of DETR outputs.
- Representation rigidity: RGB images resized/padded to fixed batch sizes, CNN downsampled grid (H0/32, W0/32), fixed N=100 query slots.
- Model sharing vs specialization: Panoptic segmentation reuses DETR weights with an added mask head; detection head is shared FFN.
- Role of positional encoding: Fixed absolute 2D sine/cosine spatial PE and learned object queries; PE is ablated and shown to matter.

### 14. Final Classification
**Multi-task, single-domain.** The paper evaluates object detection and panoptic segmentation on COCO images ("Dataset. We perform experiments on COCO 2017 detection and panoptic segmentation datasets [24,18], containing 118k training images and 5k validation images." and "Each image is annotated with bounding boxes and panoptic segmentation." in Section 4 Experiments). The panoptic model reuses DETR with an added mask head rather than introducing new domains or modalities ("DETR can be naturally extended by adding a mask head on top of the decoder outputs." in Section 4.4), so the tasks stay within a single visual domain.
