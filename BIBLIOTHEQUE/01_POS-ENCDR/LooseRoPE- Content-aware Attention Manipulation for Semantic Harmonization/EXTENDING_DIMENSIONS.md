## 1. Basic Metadata

- Title: "LooseRoPE: Content-aware Attention Manipulation for Semantic Harmonization" (Title)
- Authors: Authors not specified.
- Year: Year not specified.
- Venue: Venue not specified.

## 2. One-Sentence Contribution Summary

LooseRoPE introduces "a saliency-guided modulation of rotational positional encoding (RoPE) that loosens the positional constraints to continuously control the attention field of view" to address the challenge of "preserving the identity of the pasted object while harmonizing it with its new context" in prompt-free crop-and-paste editing. (Abstract)

## 3. Tasks Evaluated

Task 1: Crop-and-paste semantic harmonization (compositional editing)
Task type: Generation; Other (image editing / harmonization)
Dataset(s) used: "Our benchmark consists of 150 examples in total, spanning a wide variety of settings, styles and compositions, each defined by a base image and a crudely edited version of it." (7.3. Benchmark)
Domain: "60% of base images were synthesized and 40% taken from the web." (7.3. Benchmark)
Evidence: "explicit, prompt-free editing, where the user directly specifies the modification by cropping and pasting an object or sub-object into a chosen location within an image." (Abstract) "The pasted region may originate either from another image or from the same image, in which case its removal often leaves a visible hole in the source image." (3.2. LooseRoPE) "The goal is to produce a harmonized image in which the pasted object or sub-object is seamlessly integrated, without requiring any textual guidance describing the scene or desired edit." (3.2. LooseRoPE) "Some of the questions involve a translation task, in which we cut a region and move it to another location in the image." (7.2.2. Metrics) "Figure 10. Compound Editing. We showcase our method's ability to make iterative compound edits." (Figure 10)

## 4. Domain and Modality Scope

- Evaluation domain scope: Multiple domains within the same modality (synthesized vs. web images). Evidence: "60% of base images were synthesized and 40% taken from the web." (7.3. Benchmark)
- Modalities: Single modality (images). Evidence: "For 2D images, RoPE is typically applied *axially*: half of the hidden dimensions encode horizontal positions and the other half vertical ones, enabling independent offsets along each axis" (3.1. Preliminaries).
- Domain generalization / cross-domain transfer: Not claimed; the paper states it addresses "in-domain semantic harmonization" (Related Work).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Crop-and-paste semantic harmonization | Not specified (described as training-free on a single base model) | Not stated (described as "training-free") | Not specified | "a training-free image editing algorithm" (Figure 1) and "We base our method on the black-forest-labs/FLUX.1-Kontext-dev image editing diffusion model" (7.1. LooseRoPE) |

## 6. Input and Representation Constraints

- Input requires a pasted crop and binary mask: "an input image  $I_{\rm in}$  composed of a base image with an additional region crudely pasted on top, along with a binary mask M indicating the pasted area." (3.2. LooseRoPE)
- 2D image assumption: "For 2D images, RoPE is typically applied *axially*: half of the hidden dimensions encode horizontal positions and the other half vertical ones, enabling independent offsets along each axis" (3.1. Preliminaries).
- Tokenized latent representation: "the input image is encoded into the model's latent space, tokenized, and the resulting tokens are concatenated with those of the denoised image." (3.1. FLUX Kontext)
- Resizing to latent resolution: "The features are rescaled to fit the latent image resolution of  $64 \times 64$" (7.1. Saliency Estimation) and "bilinearly upsample it to the input resolution" (3.2. Saliency Estimation).
- Masked pixels set to zero: "we set all pixels outside of the crop mask M in the input image to [0,0,0]" (7.1. Saliency Estimation).
- Fixed input resolution, fixed patch size, fixed number of tokens: Not specified.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified.
- Attention type: Not explicitly categorized; attention range is modulated via RoPE. Evidence: "controlling how locally or globally each query attends to surrounding tokens during inference." (3.1. Preliminaries)
- Attention structure details: "we modulate the attention weights computed between the queries within the region of the pasted crop in the *output* image and the corresponding keys derived from the *input* image" (3.2. LooseRoPE) and "Our algorithm operates on each of FLUX Kontext's 58 attention layers over the first 22 of 28 diffusion timesteps." (7.1. LooseRoPE)
- Computational cost management: "this process can become very computationally inefficient. To overcome this... we quantize it to N = 5 possible values" (7.1. Content-Aware Attention Manipulation).

## 8. Positional Encoding (Critical Section)

- Mechanism: Rotary Positional Embedding (RoPE). Evidence: "**Rotary Positional Embeddings (RoPE).**" and "RoPE represents a position coordinate m as a series of 2D rotations at different frequencies." (3.1. Preliminaries)
- Where it is applied: "we first adjust the RoPE mechanism applied when computing attention between  $Q_{\text{out}}[M]$  and  $K_{\text{in}}$" (3.2. LooseRoPE) and "Our algorithm operates on each of FLUX Kontext's 58 attention layers" (7.1. LooseRoPE).
- Vision-specific application: "For 2D images, RoPE is typically applied *axially*: half of the hidden dimensions encode horizontal positions and the other half vertical ones, enabling independent offsets along each axis" (3.1. Preliminaries).
- Modified vs. fixed: "we augment the RoPE mechanism by introducing an additional *inverse range factor*  $r \in [0, 1]$  that scales the positional coordinate m" (3.1. Preliminaries) and "we introduce LooseRoPE, a saliency-guided modulation of rotational positional encoding (RoPE)" (Abstract).
- Ablated/compared: "the saliency-guided RoPE modulation ("w/o RoPE scaling")" (4.4. Ablations).

## 9. Positional Encoding as a Variable

- Core research variable or fixed assumption: Core variable. Evidence: "we introduce LooseRoPE, a saliency-guided modulation of rotational positional encoding (RoPE)" (Abstract) and "the saliency-guided RoPE modulation ("w/o RoPE scaling")" (4.4. Ablations).
- Multiple positional encodings compared: Not specified (only RoPE with/without scaling is discussed).
- PE choice claimed "not critical" or secondary: Not claimed.

## 10. Evidence of Constraint Masking

- Model sizes: "The model used Qwen3-VL-4B-Instruct (available on Hugging-Face), a 4-billion parameter vision-language model." (7.1. VLM Based Parameter Steering) Base model size not specified.
- Dataset sizes: "Our benchmark consists of 150 examples in total" (7.3. Benchmark).
- Performance gains attributed to components/architecture: "The results indicate that all components are necessary to achieve an optimal balance between image quality and identity preservation" (4.4. Ablations) and "LooseRoPE achieves this balance by modulating positional encoding according to saliency" (5. Conclusion).
- Scaling model size or data: Not claimed.

## 11. Architectural Workarounds

- Saliency-guided RoPE modulation to control attention range: "a saliency-guided modulation of rotational positional encoding (RoPE) that loosens the positional constraints to continuously control the attention field of view." (Abstract)
- Crop attention scaling within the mask: "we introduce a *crop attention factor*  $k(S(q))$ ... that scales the attention weights corresponding to keys within the crop mask." (3.2. LooseRoPE)
- VLM-based parameter steering during inference: "we leverage a vision-language model (VLM) to automatically steer these parameters during inference." (3.2. VLM Based Parameter Steering)
- Quantization to reduce computational cost: "this process can become very computationally inefficient. To overcome this... we quantize it to N = 5 possible values" (7.1. Content-Aware Attention Manipulation).
- Gradual relaxation across diffusion timesteps: "Over time, we gradually relax inverse range and attention scaling factors towards their equivalent value in the default FLUX Kontext model" (7.1. Content-Aware Attention Manipulation).

## 12. Explicit Limitations and Non-Claims

- Limited stylization flexibility: "our strong emphasis on identity preservation in salient regions often results in limited stylization flexibility." (6.4. Limitations)
- Occlusion handling: "our method struggles with occlusions introduced by the pasted object." (6.4. Limitations)
- Pose changes: "our method has limited ability to accommodate significant pose changes." (6.4. Limitations)
- Inherited base-model artifacts: "we inherit some of its characteristic limitations. These include slight enlargement of preserved regions and increased contrast" (6.4. Limitations)
- Future work (not a capability claim): "Future exploration may extend this framework to videos" (5. Conclusion).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple image domains (synthesized vs. web images) within a single modality; no cross-domain generalization claims.
> - Task structure: Single crop-and-paste semantic harmonization task, including relocation and iterative (compound) edits.
> - Representation rigidity: Requires input image plus binary crop mask; saliency map resized to a $64 \times 64$ latent grid; 2D image assumption.
> - Model sharing vs specialization: Training-free method on a single base model; no task-specific fine-tuning described.
> - Role of positional encoding: Central research variable, with saliency-guided RoPE modulation and ablations.

### 14. Final Classification

**Single-task, single-domain.** The evaluation centers on a single image-editing task: "explicit, prompt-free editing, where the user directly specifies the modification by cropping and pasting an object or sub-object into a chosen location within an image" (Abstract) with a goal to "produce a harmonized image" (3.2. LooseRoPE). Although the benchmark mixes synthesized and web images, it remains within one image modality and the paper emphasizes "in-domain semantic harmonization" rather than cross-domain transfer (7.3. Benchmark; Related Work).
