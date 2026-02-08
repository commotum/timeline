# ViewFormer: NeRF-free Neural Rendering from Few Images Using Transformers (Year not specified in the paper)
Source: ViewFormer- NeRF-free Neural Rendering from Few Images Using Transformers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Novel view synthesis (generation) | Context views (images and camera poses) + query camera pose | 2D (x, y); 0D (inferred) | Capped | Static | Constructed (inferred) | Novel-view image | 2D (x, y) | Fixed |
| Camera pose estimation (prediction) | Context views (images and camera poses) + query image | 2D (x, y); 0D (inferred) | Capped | Static | Constructed (inferred) | Camera pose (position + orientation) | 0D (inferred) | Fixed |

## Summary
The paper explicitly presents a single model that handles two tasks: novel view synthesis and camera pose estimation. Both tasks consume multi-view image context with pose information, which maps to primarily 2D image domains plus non-indexed pose objects. The supported input size is variable but bounded by configured context sizes (Capped), while outputs are fixed-form objects per query (an image for synthesis, a single pose for localization). The branching-attention setup uses a predefined masking policy (Static), and the model operates over learned latent/token representations rather than raw inputs alone (Constructed, inferred).

## Evidence
### Task: Novel view synthesis (generation)
- "In this work, we tackle the problem of image-based novel view synthesis – given a set of context views, the algorithm has to generate the image it would most likely observe from a query camera pose." (Section 3 Method)
- "Image-based novel view synthesis, i.e., rendering a 3D scene from a novel view-point given a set of context views (images and camera poses), is a long-standing problem in computer graphics with applications ranging from robotics (e.g. planning to grasp objects) to augmented and virtual reality (e.g. interactive virtual meetings)." (Section 1 Introduction)
- "Each training batch consists of a set of n views." (Section 3 Method)
- "With this formulation, we generate the next image in the sequence given all the previous views, effectively optimizing all different context sizes at once." (Section 3 Method)
- "In this case, we allow the model to attend to all previous images and all other vectors from the same image." (Section 3 Method, Branching attention)
- Inference: State Dynamic is marked as Constructed because "The codebook is used to map images to a smaller discrete latent space (code space), and back to the image space. In the code space, each image is represented by a sequence of tokens." (Section 3 Method), showing learned internal abstractions beyond direct raw-input mapping.

### Task: Camera pose estimation (prediction)
- "Besides rendering novel views, our model can also perform camera pose estimation, i.e., the \"inverse\" of the view synthesis task: given a set of context views and a query image, the model outputs the camera pose from which the image was taken." (Section 3 Method)
- "For the camera pose estimation task, the transformer is given the set of context views and the query image in the code space, and it generates the camera pose using a regression head attached to the output of the transformer corresponding to the query image tokens." (Section 3 Method)
- "For localization, we train the model to output the camera pose c_i given s_{\leq i} and c_{\leq i}." (Section 3 Method)
- "The context size was 19, but we did not optimize the first four views." (Section H Training details)
- Inference: Output Dimension is marked as 0D (inferred) because the model outputs one camera pose object per query image; State Dynamic is marked Constructed because the pose is regressed from latent code tokens and transformer hidden representations (Section 3 Method).
