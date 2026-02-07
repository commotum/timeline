# Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding (Not specified)
Source: Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding (PPTr).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4D semantic segmentation | point cloud videos | 4D (x, y, z, t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | semantic labels | 4D (x, y, z, t) (inferred) | Capped (inferred) |
| 3D action recognition | human body point cloud videos | 4D (x, y, z, t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | actions | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates PPTr on two tasks: 4D semantic segmentation and 3D action recognition, both operating on point cloud video/sequence inputs. Segmentation produces per-point semantic labels (4D outputs), while action recognition produces video-level action labels (0D outputs). The inputs are described as short clips and sequences indexed by time, so the input/output dynamics are capped and attention appears static; the segmentation setup additionally uses a memory pool, implying constructed state.

## Evidence
### Task: 4D semantic segmentation
- "4D semantic segmentation on Synthia4D [34] and HOI4D [29]." (Section 1 Introduction)
- "The input to network is a short video clip." (Fig. 4 caption)
- "For semantic segmentation, primitive features are concatenated to corresponding point features then classified into semantic labels." (Fig. 4 caption)
- "We represent a point cloud sequence as  $\Psi = \{(P_t, V_t) | t = 1, ..., L\}$" (Section 4.1)
- "a memory pool storing pre-computed primitive features from a long video." (Section 4 Method)
- Inference: Mapped the point-cloud sequence indexed by t and the short clip to 4D inputs/outputs and capped dynamics; treated attention as static and state as constructed due to fixed self-attention over the clip and a memory pool. (Section 4.1; Fig. 4 caption; Section 4 Method)

### Task: 3D action recognition
- "3D action recognition on MSR-Action [25]" (Section 1 Introduction)
- "we first conduct experiments on the 3D Action Recognition task." (Section 5.2)
- "the MAR-Action3D dataset which consists of 567 human body point cloud videos" (Section 5.2)
- "For action recognition, primitive features are merged by maxpooling to a global feature then classified into actions." (Fig. 4 caption)
- "we can avoid maintaining the long-term memory pool in this case." (Section 5.2)
- Inference: Treated point cloud videos as 4D (x, y, z, t) inputs with capped clip lengths, mapped the output to a fixed 0D action label, and marked attention as static; marked state as direct because this setup avoids the long-term memory pool. (Section 5.2; Fig. 4 caption)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
4D semantic segmentation,point cloud videos,"4D (x, y, z, t) (inferred)",Capped (inferred),Static (inferred),Constructed (inferred),semantic labels,"4D (x, y, z, t) (inferred)",Capped (inferred)
3D action recognition,human body point cloud videos,"4D (x, y, z, t) (inferred)",Capped (inferred),Static (inferred),Direct (inferred),actions,0D (inferred),Fixed (inferred)
