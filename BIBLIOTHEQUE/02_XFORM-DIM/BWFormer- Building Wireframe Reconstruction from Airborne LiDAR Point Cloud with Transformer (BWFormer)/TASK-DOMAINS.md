# BWFormer: Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer (Not specified in the paper)
Source: BWFormer- Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer (BWFormer).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3D building wireframe reconstruction | 2D height map projected from airborne LiDAR point cloud | 2D (x, y) | Fixed | Dynamic (inferred) | Direct (inferred) | 3D building wireframe (corners + edges) | 3D (x, y, z) | Capped (inferred) |
| LiDAR scanning simulation / sampling location generation (data augmentation) | Building footprint condition | 2D (x, y) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Synthetic sampling locations (binary sampling image) | 2D (x, y) | Not specified in the paper. |

## Summary
BWFormer primarily addresses 3D building wireframe reconstruction by projecting airborne LiDAR point clouds into fixed 256 x 256 2D height maps and predicting 3D corners/edges with capped query limits. The model uses deformable attention (dynamic attention inferred) and is described as an end-to-end mapping without explicit persistent state (direct state inferred). The paper also introduces a separate conditional latent diffusion model to simulate LiDAR sampling locations for data augmentation in a 2D footprint/image domain, but its interface dynamics are not explicitly specified.

## Evidence
### Task: 3D building wireframe reconstruction
- "In this paper, we present BWFormer, a novel Transformerbased model for building wireframe reconstruction from airborne LiDAR point cloud." (Abstract)
- "the proposed BWFormer reconstructs 3D building wireframes from them in an end-to-end manner." (Section 3)
- "We then project it onto the xy-plane and compute a  $256 \times 256$  height map" (Section 4.2)
- "the top N pixels are selected as the 2D corners, where N is the maximum 2D corner number." (Section 3.2)
- "H indicates maximum number of corners that share the same 2D coordinate" (Section 3.2)
- "With valid edges, the building wireframe is reconstructed." (Section 3.3)
- "a deformable self-attention layer" (Section 3.1)
- "a deformable crossattention / edge attention layer" (Section 3.1)
- Inference: Attention Dynamic marked Dynamic (inferred) because the model uses "a deformable self-attention layer" and "a deformable crossattention / edge attention layer" (Section 3.1); State Dynamic marked Direct (inferred) because BWFormer "reconstructs 3D building wireframes from them in an end-to-end manner" (Section 3); Out Dynamics marked Capped (inferred) because "N is the maximum 2D corner number" and "H indicates maximum number of corners" (Section 3.2).

### Task: LiDAR scanning simulation / sampling location generation (data augmentation)
- "a conditional latent diffusion model for LiDAR scanning simulation is utilized for data augmentation." (Abstract)
- "a conditional LDM is utilized to simulate the sampling locations with a given building footprint." (Section 3.5)
- "Given a sampling image I which is a binary image in pixel space" (Section 3.5)
- "With the LDM, the synthetic sampling locations are generated." (Section 3.5)
- "A latent space is constructed by training an autoencoder with the real LiDAR sampling images as input." (Section 3.5)
- "With the attention mechanism [25], the learning objective is represented as:" (Section 3.5)
- Inference: Attention Dynamic marked Dynamic (inferred) based on the use of an "attention mechanism" (Section 3.5); State Dynamic marked Constructed (inferred) because "A latent space is constructed" for the diffusion model (Section 3.5).
