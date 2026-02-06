# DynamicCity: Large-Scale 4D Occupancy Generation from Dynamic Scenes (Not specified in the paper.)
Source: DynamicCity- Large-Scale 4D Occupancy Generation from Dynamic Scenes.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D occupancy reconstruction | 4D occupancy scene/sequence (dynamic 3D occupancy sequence Q) | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | reconstructed 4D occupancy scenes with semantics (Q') | 4D (x, y, z, t) | Fixed (inferred) |
| 4D occupancy generation (unconditional) | Gaussian noise (diffusion input) | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Constructed (inferred) | generated 4D occupancy scenes (novel 4D scenes) | 4D (x, y, z, t) | Fixed (inferred) |
| HexPlane-conditional generation (autoregressive) | HexPlane condition from previous frames | 2D (x, y); 2D (x, z); 2D (y, z); 2D (t, x); 2D (t, y); 2D (t, z) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | extended 4D occupancy sequences (next sequence) | 4D (x, y, z, t) | Open (inferred) |
| Layout-conditioned generation | bird's-eye view layout sketch (T x H x W) | 3D (x, y, t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | 4D occupancy scenes with controlled vehicle placement/dynamics | 4D (x, y, z, t) | Fixed (inferred) |
| Command-conditioned generation | command label (STATIC, FORWARD, TURN LEFT, TURN RIGHT) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | 4D occupancy scenes with ego-motion control | 4D (x, y, z, t) | Fixed (inferred) |
| Trajectory-conditioned generation | trajectory XY coordinates over time (traj in R^{T x 2}) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | 4D occupancy scenes following trajectory | 4D (x, y, z, t) | Fixed (inferred) |
| 4D scene inpainting | HexPlane + 2D mask on XY plane | 2D (x, y); 2D (x, z); 2D (y, z); 2D (t, x); 2D (t, y); 2D (t, z) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | inpainted 4D occupancy scenes | 4D (x, y, z, t) | Fixed (inferred) |
| 4D scene outpainting | occupancy sequence (given scene) | 4D (x, y, z, t) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | larger 4D occupancy scenes with extended spatial dimensions | 4D (x, y, z, t) | Fixed (inferred) |
| Single-frame occupancy conditional generation | first-frame occupancy encoded as HexPlane | 3D (x, y, z) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | generated 4D occupancy sequences (conditional) | 4D (x, y, z, t) | Fixed (inferred) |
| Occupancy forecasting | HexPlane with context length 2 seconds (T=4) | 2D (x, y); 2D (x, z); 2D (y, z); 2D (t, x); 2D (t, y); 2D (t, z) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | next 2 seconds 4D occupancy sequence | 4D (x, y, z, t) | Fixed (inferred) |

## Summary
DynamicCity covers 4D occupancy reconstruction and unconditional 4D scene generation, plus multiple conditional generation tasks (HexPlane autoregressive extension, layout-, command-, and trajectory-guided generation, inpainting, outpainting, single-frame conditioning) and occupancy forecasting. Inputs span 4D occupancy sequences and HexPlane latent planes, as well as 3D bird's-eye layouts, 1D trajectories, and 0D command labels, while outputs are consistently 4D occupancy scenes/sequences. Most tasks operate on fixed-resolution grids/latents, and the HexPlane autoregressive mode is described as enabling arbitrarily long sequences; attention/state dynamics are inferred as static/constructed from the fixed tokenization and HexPlane representation.

## Evidence
### Task: 4D occupancy reconstruction
- "Given a 4D scene, *i.e.*, a dynamic 3D occupancy sequence  $\mathbf{Q} \in \mathbb{R}^{T \times X \times Y \times Z \times C}$" (Sec. 4 Our Approach)
- "generating dense semantic predictions  $\mathbf{Q}'$" (Sec. 4.1 Decoding HexPlane)
- Inference: Marked In/Out Dynamics as Fixed and Attention/State as Static/Constructed because scenes use fixed resolutions and the model encodes HexPlane representations (Sec. 5.1; Sec. 4).

### Task: 4D occupancy generation (unconditional)
- "a novel 4D occupancy generation framework capable of generating large-scale, high-quality dynamic 4D scenes" (Front matter)
- "After obtaining HexPlane embeddings, DynamicCity leverages a DiT-based framework for 4D occupancy generation." (Sec. 4 Our Approach)
- "convert Gaussian noise into data samples through denoising steps." (Sec. 3 Preliminaries)
- Inference: Marked Out Dynamics as Fixed and Attention/State as Static/Constructed based on fixed scene resolutions and HexPlane encoding/tokenization (Sec. 5.1; Sec. 4; Sec. 4.2).

### Task: HexPlane-conditional generation (autoregressive)
- "HexPlane: By autoregressively generating the HexPlane, we extend scene duration beyond temporal constraints." (Sec. 4.3 Downstream Applications)
- "By conditioning each new 4D sequence on the previous one, we sequentially extend the temporal dimension." (Sec. 7.5 Downstream Applications)
- "can model sequence of arbitrary length" (Sec. 7.5 Downstream Applications)
- Inference: Marked In Dynamics as Fixed because conditional generation is constrained by latent space dimensions, and Out Dynamics as Open based on the arbitrary-length claim (Sec. 7.5). Marked Attention/State as Static/Constructed due to HexPlane encoding/tokenization (Sec. 4; Sec. 4.2).

### Task: Layout-conditioned generation
- "Layout: We control vehicle placement and dynamics in 4D scenes using conditions learned from bird's-eye view sketches." (Sec. 4.3 Downstream Applications)
- "Pooling this binary image provides layout information as a  $T \times H \times W$  tensor" (Sec. 7.5 Downstream Applications)
- Inference: Marked In/Out Dynamics as Fixed and Attention/State as Static/Constructed because layout tensors are padded to fixed HexPlane size and scenes use fixed resolutions (Sec. 7.5; Sec. 5.1).

### Task: Command-conditioned generation
- "Command: Controls general ego vehicle motion via instructions." (Sec. 4.3 Downstream Applications)
- "we define four commands: STATIC, FORWARD, TURN LEFT, and TURN RIGHT" (Sec. 7.5 Downstream Applications)
- Inference: Marked Input Dimension as 0D and In Dynamics as Fixed because commands are discrete class labels (Sec. 7.5). Marked Attention/State as Static/Constructed and Out Dynamics as Fixed based on fixed scene resolutions (Sec. 5.1).

### Task: Trajectory-conditioned generation
- "Trajectory: Enables fine-grained control through specific trajectory inputs." (Sec. 4.3 Downstream Applications)
- "trajectory traj  $\in \mathbb{R}^{T \times 2}$" (Sec. 7.5 Downstream Applications)
- Inference: Marked Input Dimension as 1D (t) and In Dynamics as Fixed from the time-indexed trajectory definition (Sec. 7.5). Marked Attention/State as Static/Constructed and Out Dynamics as Fixed based on fixed scene resolutions (Sec. 5.1).

### Task: 4D scene inpainting
- "Inpaint: Edit 4D scenes by masking HexPlane regions and guiding sampling with the masked areas." (Sec. 4.3 Downstream Applications)
- "we define a 2D mask  $m \in \mathbb{R}^{X \times Y}$  on the XY plane" (Sec. 7.5 Downstream Applications)
- Inference: Marked In/Out Dynamics as Fixed and Attention/State as Static/Constructed because the mask is defined on a fixed XY grid and scenes use fixed resolutions (Sec. 7.5; Sec. 5.1).

### Task: 4D scene outpainting
- "Outpainting extends the spatial dimensions of a given occupancy sequence." (Sec. 7.5 Downstream Applications)
- "mask half of the scene, shift the latent representation, and apply the inpainting process." (Sec. 7.5 Downstream Applications)
- Inference: Marked In/Out Dynamics as Fixed and Attention/State as Static/Constructed because outpainting masks half the scene and uses the same inpainting procedure on fixed-resolution occupancy sequences (Sec. 7.5; Sec. 5.1).

### Task: Single-frame occupancy conditional generation
- "Single frame occupancy. We apply the same procedure for single-frame occupancy conditional generation" (Sec. 7.5 Downstream Applications)
- "encode the first frame of each training sequence as a HexPlane" (Sec. 7.5 Downstream Applications)
- Inference: Marked Input Dimension as 3D (x, y, z) and In Dynamics as Fixed because the condition is the first frame of a sequence (Sec. 7.5). Marked Attention/State as Static/Constructed and Out Dynamics as Fixed based on fixed scene resolutions (Sec. 5.1).

### Task: Occupancy forecasting
- "We train our HexPlane conditional generation pipeline on Occ3D-nuScenes [40] as an occupancy forecasting model." (Sec. 8.2 Occupancy Forecasting Results)
- "receives a HexPlane with a context length of 2 seconds" (Sec. 8.2 Occupancy Forecasting Results)
- "generates the next 2 seconds for evaluation." (Sec. 8.2 Occupancy Forecasting Results)
- Inference: Marked In/Out Dynamics as Fixed because the model uses a fixed context length (T=4) and predicts the next 2 seconds (Sec. 8.2). Marked Attention/State as Static/Constructed due to HexPlane encoding/tokenization (Sec. 4; Sec. 4.2).
