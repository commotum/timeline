# StoryMem: Multi-shot Long Video Storytelling with Memory (2025)
Source: StoryMem- Multi-shot Long Video Storytelling with Memory.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multi-shot narrative video generation | Per-shot text descriptions; memory keyframes from previous shots; optional reference images | 1D (t); 2D (x, y) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed | Multi-shot narrative video clips | 3D (x, y, t) (inferred) | Open (inferred) |

## Summary
The paper’s modeled task is generation of coherent multi-shot narrative videos from per-shot text prompts, conditioned on an explicit visual memory bank and optionally initialized with reference images. The justified input modalities are sequential text and visual keyframes/reference images, while outputs are spatiotemporal video clips. The framework uses dynamically updated memory and runtime context retrieval, so the attention behavior is dynamic and the state is constructed. The sequence-level interface is treated as open because generation is autoregressive over variable shot count \(N\) rather than a fixed-length joint output.

## Evidence
### Task: Multi-shot narrative video generation
- "Given a story script consisting of a sequence of textual descriptions for each shot  \(\mathcal{T} = \{t_i\}_{i=1}^N\) , our goal is to generate a coherent multi-shot narrative video  \(\mathcal{V} = \{v_i\}_{i=1}^N\) ." (Section 3.2)
- "This formulation enables *memory-based multi-shot generation*, where each shot is conditioned on both its textual description and an evolving memory that summarizes characters, scenes, and stylistic information from previous shots, thereby ensuring cross-shot consistency and narrative coherence throughout the entire video." (Section 3.2)
- "Another application is to personalize the initialization of the memory state  \(m_0\) . For instance, users can provide character or background reference images as the initial memory, enabling customized multi-shot video generation." (Section 3.5)
- "The diffusion process operates on the video latents  \(z_0 = \mathcal{E}(v) \in \mathbb{R}^{c \times f \times h \times w}\)  by encoding RGB video \(v\) with 3D VAE [19] encoder  \(\mathcal{E}\) ." (Section 3.1)
- "During generation, the memory is dynamically extracted, updated, and injected into the model to guide each new shot." (Section 1 Introduction)
- Inference: In Dimension is mapped to 1D (t) for the "sequence of textual descriptions" and 2D (x, y) for memory/reference keyframes; Out Dimension is mapped to 3D (x, y, t) from the video latent shape with frame axis \(f\). In Dynamics/Out Dynamics are marked Open from the autoregressive formulation \(\prod_{i=1}^{N}\) over variable story length and iterative cross-shot generation. Attention Dynamic is marked Dynamic because the paper explicitly states memory is "dynamically extracted, updated, and injected" per shot.
