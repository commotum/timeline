## 1. Basic Metadata

- Title: "Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet" (Title page)
- Authors: "Li Yuan<sup>1</sup>\*, Yunpeng Chen<sup>2</sup>, Tao Wang<sup>1,3</sup>\*, Weihao Yu<sup>1</sup>, Yujun Shi<sup>1</sup>, Zihang Jiang<sup>1</sup>, Francis E.H. Tay<sup>1</sup>, Jiashi Feng<sup>1</sup>, Shuicheng Yan<sup>1</sup>" (Title page)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

"To overcome such limitations, we propose a new Tokens-To-Token Vision Transformer (T2T-ViT), which incorporates 1) a layerwise Tokens-to-Token (T2T) transformation to progressively structurize the image to tokens by recursively aggregating neighboring Tokens into one Token (Tokens-to-Token), such that local structure represented by surrounding tokens can be modeled and tokens length can be reduced; 2) an efficient backbone with a deep-narrow structure for vision transformer motivated by CNN architecture design after empirical study." (Abstract)

## 3. Tasks Evaluated

- Task name: Image classification on ImageNet
  - Task type: Classification
  - Dataset(s) used: ImageNet
  - Domain: Natural images
  - Quotes: "We conduct the following experiments with T2T-ViT for image classification on ImageNet." (Section 4); "All experiments are conducted on ImageNet dataset [9], with around 1.3 million images in training set and 50k images in validation set." (Section 4.1)

- Task name: Image classification (transfer learning) on CIFAR10
  - Task type: Classification
  - Dataset(s) used: CIFAR10
  - Domain: Natural images
  - Quotes: "we also transfer the pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100 (Sec. 4.1)." (Section 4); "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1)

- Task name: Image classification (transfer learning) on CIFAR100
  - Task type: Classification
  - Dataset(s) used: CIFAR100
  - Domain: Natural images
  - Quotes: "we also transfer the pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100 (Sec. 4.1)." (Section 4); "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1)

## 4. Domain and Modality Scope

- Evaluation scope: Multiple datasets within a single modality (natural images). Evidence: "We conduct the following experiments with T2T-ViT for image classification on ImageNet." (Section 4); "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1)
- Multiple modalities? Not stated; no multimodal evaluation described. (Not stated.)
- Domain generalization or cross-domain transfer claimed? Not claimed; transfer is within natural-image classification datasets. Evidence: "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification on ImageNet | Not shared across tasks; trained from scratch for this task | No | Not specified | "We conduct the following experiments with T2T-ViT for image classification on ImageNet." (Section 4); "We validate the T2T-ViT by training from scratch on ImageNet" (Section 4)
| Image classification on CIFAR10 (transfer) | Pretrained on ImageNet, then reused for CIFAR10 | Yes | Not specified | "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1); "finetune the pretrained T2T-ViT-14/19 with 60 epochs" (Section 4.1)
| Image classification on CIFAR100 (transfer) | Pretrained on ImageNet, then reused for CIFAR100 | Yes | Not specified | "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1); "finetune the pretrained T2T-ViT-14/19 with 60 epochs" (Section 4.1)

## 6. Input and Representation Constraints

- Input resolution (fixed per experiment): "Throughout the experiments on ImageNet, we set default image size as  $224 \times 224$  except for some specific cases on  $384 \times 384$ ." (Section 4.1)
- Fixed patch sizes and overlaps in T2T: "The patch size for the three soft splits is P=[7,3,3], and the overlapping is S=[3,1,1], which reduces size of the input image from  $224\times224$  to  $14\times14$  according to Eqn. (3)." (Section 3.3)
- Fixed token length after T2T: "After the final iteration, the output tokens  $T_f$  of the T2T module has fixed length" (Section 3.1)
- 2D image grid assumption: "Here \"Reshape\" re-organizes tokens  $T' \in \mathbb{R}^{l \times c}$  to  $I \in \mathbb{R}^{h \times w \times c}$ , where l is the length of T', h, w, c are height, width and channel respectively, and  $l = h \times w$ ." (Section 3.1)
- Padding in tokenization: "When conducting the soft split, the size of each patch is  $k \times k$  with s overlapping and p padding on the image" (Section 3.1)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not explicitly stated; final token grid example is "$14\times14$" for 224x224 input. Evidence: "The patch size for the three soft splits is P=[7,3,3], and the overlapping is S=[3,1,1], which reduces size of the input image from  $224\times224$  to  $14\times14$  according to Eqn. (3)." (Section 3.3)
- Fixed or variable sequence length: Fixed after T2T module. Evidence: "After the final iteration, the output tokens  $T_f$  of the T2T module has fixed length" (Section 3.1)
- Attention type: Global multi-head self-attention. Evidence: "ViT applies transformer layers to model the global relation among these tokens for classification." (Section 1); "where MSA denotes the multihead self-attention operation" (Section 3.1)
- Mechanisms for computational cost: Token length reduction and efficient attention variants. Evidence: "the length of tokens can be reduced by the aggregation process." (Section 1); "we set the channel dimension of the T2T layer small (32 or 64) to reduce MACs, and optionally adopt an efficient Transformer such as Performer [7] layer to reduce memory usage at limited GPU memory." (Section 3.1)

## 8. Positional Encoding (Critical Section)

- Mechanism: Absolute sinusoidal positional embedding. Evidence: "we concatenate a class token to it and then add Sinusoidal Position Embedding (PE) to it, the same as ViT to do classification" (Section 3.2)
- Where applied: Input to the backbone (added to tokens with class token). Evidence: "$$T_{f_0} = [t_{cls}; T_f] + E, \qquad E \in \mathbb{R}^{(l+1) \times d}$$" (Section 3.2)
- Fixed across experiments vs modified/ablated: Not stated; no alternative PE comparisons described. (Not stated.)

## 9. Positional Encoding as a Variable

- Treated as a core research variable or fixed assumption: Fixed architectural assumption (no variations described). Evidence: "add Sinusoidal Position Embedding (PE) to it, the same as ViT to do classification" (Section 3.2)
- Multiple positional encodings compared? Not stated.
- Any claims that PE choice is not critical or secondary? Not stated.

## 10. Evidence of Constraint Masking

- Model sizes and compute emphasis: "T2T-ViT reduces the parameter count and MACs of vanilla ViT by half, while achieving more than 3.0% improvement when trained from scratch on ImageNet." (Abstract)
- Dataset size: "All experiments are conducted on ImageNet dataset [9], with around 1.3 million images in training set and 50k images in validation set." (Section 4.1)
- Performance gains attributed to architectural hierarchy/tokenization: "we propose a progressive tokenization module to aggregate neighboring Tokens to one Token (named Tokens-to-Token module), which can model the local structure information of surrounding tokens and reduce the length of tokens iteratively." (Section 1)
- Ablation evidence for architectural contribution: "the T2T module can improve model performance by 2.0%-2.2% on ImageNet." (Section 4.3)

## 11. Architectural Workarounds

- Progressive tokenization with overlapping soft split to encode local structure and reduce token length: "we propose a progressive tokenization module to aggregate neighboring Tokens to one Token (named Tokens-to-Token module), which can model the local structure information of surrounding tokens and reduce the length of tokens iteratively." (Section 1); "we split it into patches with overlapping" (Section 3.1)
- Deep-narrow backbone to reduce redundancy and improve feature richness: "we find \"deepnarrow\" architecture design with fewer channels but more layers in ViT brings much better performance at comparable model size and MACs" (Section 1); "Based on these findings, we design a deep-narrow architecture for our T2T-ViT backbone." (Section 3.2)
- Efficient attention to reduce memory: "optionally adopt an efficient Transformer such as Performer [7] layer to reduce memory usage at limited GPU memory." (Section 3.1)
- Small channel dimension in T2T module for compute reduction: "we set the channel dimension of the T2T layer small (32 or 64) to reduce MACs" (Section 3.1)

## 12. Explicit Limitations and Non-Claims

- Limitation stated: "But we also note the MACs of our T2T-ViT are still larger than MobileNets because of the dense operations in Transformers." (Section 4.1)
- Explicit non-claims about unrestrained multi-task learning, open-world learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Natural-image classification datasets only (ImageNet, CIFAR10/100). Evidence: "image classification on ImageNet" (Section 4); "transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100" (Section 4.1).
> - Task structure: Single task type (classification) across datasets. Evidence: "image classification on ImageNet" (Section 4).
> - Representation rigidity: Fixed input sizes per experiment and fixed-length tokens after T2T. Evidence: "default image size as  $224 \times 224$  except for some specific cases on  $384 \times 384$" (Section 4.1); "output tokens  $T_f$  of the T2T module has fixed length" (Section 3.1).
> - Model sharing vs specialization: Pretrain on ImageNet, then fine-tune for CIFAR tasks. Evidence: "We transfer our pretrained T2T-ViT to downstream datasets such as CIFAR10 and CIFAR100." (Section 4.1); "finetune the pretrained T2T-ViT-14/19" (Section 4.1).
> - Role of positional encoding: Fixed sinusoidal PE added at backbone input. Evidence: "add Sinusoidal Position Embedding (PE) to it, the same as ViT to do classification" (Section 3.2).

### 14. Final Classification

**Multi-task, single-domain.** The paper evaluates multiple classification tasks on different natural-image datasets: "image classification on ImageNet" (Section 4) and transfer to "CIFAR10 and CIFAR100" (Section 4.1). All evaluations stay within the same modality/domain of natural images, and there is no evidence of cross-modality or cross-domain evaluation.
