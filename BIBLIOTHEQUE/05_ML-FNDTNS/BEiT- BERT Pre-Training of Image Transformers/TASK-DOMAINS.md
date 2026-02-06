# BEIT: BERT Pre-Training of Image Transformers (Not specified in the paper.)
Source: BEiT- BERT Pre-Training of Image Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Masked image modeling (visual token prediction) | corrupted image patches (masked image patches) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | visual tokens (for masked patches) | 2D (x, y) | Fixed (inferred) |
| Image reconstruction (autoencoding tokenizer training) | images (pixels) | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | reconstructed images | 2D (x, y) | Fixed (inferred) |
| Image classification | images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | class labels | 0D (inferred) | Fixed (inferred) |
| Semantic segmentation | images | 2D (x, y) | Fixed (inferred) | Static (inferred) | Direct (inferred) | per-pixel semantic classes (segmentation map) | 2D (x, y) | Fixed (inferred) |

## Summary
BEIT spans self-supervised masked image modeling, a tokenizer pretraining autoencoding reconstruction stage, and downstream image classification and semantic segmentation. All tasks operate on 2D image inputs, with outputs as 2D grids (visual tokens or segmentation maps) or 0D class labels. The paper's setups use fixed-resolution patch grids and a standard Transformer encoder, so dynamics are treated as fixed and attention/state as static/direct where inferred.

## Evidence
### Task: Masked image modeling (visual token prediction)
- "The pre-training objective is to recover the original visual tokens based on the corrupted image patches." (Abstract)
- "We propose a masked image modeling (MIM) task." (Section 2.3 Pre-Training BEIT: Masked Image Modeling)
- "We randomly mask some percentage of image patches, and then predict the visual tokens that are corresponding to the masked patches." (Section 2.3 Pre-Training BEIT: Masked Image Modeling)
- Inference: In/Out Dynamics labeled Fixed and Attention/State labeled Static/Direct based on fixed patch grid and full-sequence Transformer ("In our experiments, we split each  $224 \times 224$  image into a  $14 \times 14$  grid of image patches." Section 2.1.1 Image Patch; "The input of Transformer is a sequence of image patches." Section 2.2 Backbone Network: Image Transformer).

### Task: Image reconstruction (autoencoding tokenizer training)
- "Before pre-training, we learn an \"image tokenizer\" via autoencoding-style reconstruction." (Figure 1 caption)
- "The tokenizer  $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$  maps image pixels  $\boldsymbol{x}$  into discrete tokens  $\boldsymbol{z}$  according to a visual codebook (i.e., vocabulary)." (Section 2.1.2 Visual Token)
- "The decoder  $p_{\psi}(\boldsymbol{x}|\boldsymbol{z})$  learns to reconstruct the input image  $\boldsymbol{x}$  based on the visual tokens  $\boldsymbol{z}$." (Section 2.1.2 Visual Token)
- Inference: In/Out Dynamics labeled Fixed and Attention/State labeled Static/Direct based on fixed token grid and deterministic tokenizer/decoder mappings ("We tokenize each image to a  $14 \times 14$  grid of visual tokens." Section 2.1.2 Visual Token).

### Task: Image classification
- "We perform self-supervised learning and then fine-tune the pretrained BEIT on two downstream tasks, i.e., image classification, and semantic segmentation." (Introduction)
- "The image classification task classifies input images to various categories." (Section 3.1 Image Classification)
- "For image classification tasks, we directly employ a simple linear classifier as the task layer." (Section 2.6 Fine-Tuning BEIT on Downstream Vision Tasks)
- Inference: Out Dimension labeled 0D because the task outputs a single category per image ("The image classification task classifies input images to various categories." Section 3.1). In/Out Dynamics and Attention/State labeled Fixed/Static/Direct based on fixed patch grid and full-sequence Transformer ("In our experiments, we split each  $224 \times 224$  image into a  $14 \times 14$  grid of image patches." Section 2.1.1 Image Patch; "The input of Transformer is a sequence of image patches." Section 2.2 Backbone Network: Image Transformer).

### Task: Semantic segmentation
- "We perform self-supervised learning and then fine-tune the pretrained BEIT on two downstream tasks, i.e., image classification, and semantic segmentation." (Introduction)
- "Semantic segmentation aims to predict a corresponding class for each pixel of the input image." (Section 3.2 Semantic Segmentation)
- "For semantic segmentation, we follow the task layer used in SETR-PUP." (Section 2.6 Fine-Tuning BEIT on Downstream Vision Tasks)
- Inference: In/Out Dynamics and Attention/State labeled Fixed/Static/Direct based on fixed patch grid and full-sequence Transformer ("In our experiments, we split each  $224 \times 224$  image into a  $14 \times 14$  grid of image patches." Section 2.1.1 Image Patch; "The input of Transformer is a sequence of image patches." Section 2.2 Backbone Network: Image Transformer).
