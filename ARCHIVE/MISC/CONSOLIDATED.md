- **Self-Attention with Relative Position Representations** (2018)
  **Driver:** Absolute position added to inputs is an awkward fit for attention's permutation-invariance; poor inductive bias for distance.
  **Outcome:** Inject relative distance embeddings directly into attention (key/value side). ([arXiv][1-1])
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (2019)
  **Driver:** Absolute PE breaks "state reuse" across segments and causes temporal confusion in recurrence/memory.
  **Outcome:** A relative positional encoding formulation compatible with segment-level recurrence for long-context LM. ([arXiv][1-2])
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** (2020)
  **Driver:** Standard position injection entangles content/position too early and too tightly.
  **Outcome:** Disentangled attention that separately models content and (relative) position in attention scoring. ([arXiv][1-3])
- **Rethinking Positional Encoding in Language Pre-training (TUPE)** (2020)
  **Driver:** Mixing token and position correlations (and treating [CLS] position like ordinary tokens) is noisy/suboptimal.
  **Outcome:** Untie positional and token projections; separate positional correlation path (TUPE variants). ([OpenReview][1-4])
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** (2021)
  **Driver:** Absolute PE is inflexible; many relative schemes don't mesh cleanly with some attention variants.
  **Outcome:** Rotary position embedding that makes relative offsets emerge naturally in dot-products. ([arXiv][1-5])
- **Train Short, Test Long: Attention with Linear Biases (ALiBi)** (2021)
  **Driver:** Common PEs extrapolate poorly to lengths longer than trained.
  **Outcome:** Remove token-additive PE; add a distance-proportional linear bias directly to attention logits. ([arXiv][1-6])
- **Rethinking and Improving Relative Position Encoding for Vision Transformer (iRPE)** (2021)
  **Driver:** RPE works well in NLP but is unclear/controversial in vision; existing methods have tradeoffs when flattened to 2D images.
  **Outcome:** 2D image-aware RPE variants (directional distance modeling + attention interaction design). ([arXiv][1-7])
- **A Simple and Effective Positional Encoding for Transformers** (2021)
  **Driver:** Absolute PE doesn't directly express relative relations; prior RPE designs have practical issues.
  **Outcome:** A simplified, effective PE/RPE scheme for language. ([ACL Anthology][1-8])
- **Swin Transformer** (2021)
  **Driver:** Absolute position embeddings in ViTs are less suited to shifted-window attention and spatial generalization.
  **Outcome:** Learnable relative position bias within windows. Note: borderline (PE isn't the main thesis, but it is a deliberate PE choice). ([arXiv][1-9])
- **KERPLE: Kernelized Relative Positional Embedding for Length Extrapolation** (2022)
  **Driver:** Existing RPE variants extrapolate inconsistently and lack a unifying principle.
  **Outcome:** Kernelize positional differences via CPD/PD kernel framework to derive extrapolatable RPEs. ([arXiv][1-10])
- **XPos / Length-Extrapolatable Transformer** (2022)
  **Driver:** RoPE can become unstable or degrade for long extrapolation.
  **Outcome:** Add an exponential decay to RoPE rotation (XPos) to stabilize long-range behavior. ([arXiv][1-11])
- **Extending Context Window of Large Language Models via Positional Interpolation (PI)** (2023)
  **Driver:** Direct extrapolation past trained context can blow up attention scores and fail catastrophically.
  **Outcome:** Down-scale/interpolate position indices so inference stays in-range for RoPE; minimal finetune. ([arXiv][1-12])
- **YaRN: Efficient Context Window Extension of Large Language Models** (2023)
  **Driver:** RoPE models fail to generalize beyond training length; existing extension methods are compute-hungry.
  **Outcome:** Frequency-aware RoPE interpolation plus related scaling tricks for efficient long-context finetuning. ([arXiv][1-13])
- **A Length-Extrapolatable Transformer** (2023)
  **Driver:** Length extrapolation is tied to attention's positional "resolution."
  **Outcome:** Design changes including relative position embedding targeted at improving extrapolation indicators. ([ACL Anthology][1-14])
- **Resonance RoPE: Improving Context Length Generalization of Large Language Models** (2024)
  **Driver:** RoPE interpolation/remapping still leaves an OOD gap for positions.
  **Outcome:** Refine RoPE feature interpolation for OOD positions ("resonance" shaping). ([arXiv][1-15])
- **Rotary Position Embedding for Vision Transformer** (2024)
  **Driver:** Common axial 2D RoPE misses diagonal structure important in vision.
  **Outcome:** Mixed-axis frequency 2D RoPE variant for images. ([ECVA][1-16])
- **LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate** (2024)
  **Driver:** Standard patch position encodings cause a distribution shift when changing patch counts or resolution.
  **Outcome:** Drop-in replacement position handling using directed attention heads plus 2D masks/distance-penalized bias to improve extrapolation. ([arXiv][1-17])
- **RoPE scaling analysis + new scaling** (2024)
  **Driver:** Many RoPE scaling methods are empirical and poorly grounded in RoPE's internal distribution.
  **Outcome:** A more principled scaling approach within the RoPE-scaling family. ([ACL Anthology][1-18])
- **Exploring Context Window of LLMs** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. Note: borderline (more analysis-heavy, but includes PE scaling focus). ([NeurIPS Proceedings][1-19])
- **Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings (PoPE)** (2025)
  **Driver:** RoPE entangles content ("what") and position ("where"), harming tasks needing independent matches.
  **Outcome:** PoPE polar-coordinate positional embedding removing the confound; improves zero-shot length extrapolation. ([arXiv][1-20])
- **Context-aware Biases for Length Extrapolation (CABLE)** (2025)
  **Driver:** Fixed-form biases/PEs can be too rigid for long-context generalization.
  **Outcome:** Learns context-aware additive relative positional biases. ([ACL Anthology][1-21])

[1-1]: https://arxiv.org/abs/1803.02155
[1-2]: https://arxiv.org/abs/1901.02860
[1-3]: https://arxiv.org/abs/2006.03654
[1-4]: https://openreview.net/pdf?id=09-528y2Fgf
[1-5]: https://arxiv.org/abs/2104.09864
[1-6]: https://arxiv.org/abs/2108.12409
[1-7]: https://arxiv.org/abs/2107.14222
[1-8]: https://aclanthology.org/2021.emnlp-main.236.pdf
[1-9]: https://arxiv.org/pdf/2103.14030
[1-10]: https://arxiv.org/abs/2205.09921
[1-11]: https://arxiv.org/pdf/2212.10554
[1-12]: https://arxiv.org/abs/2306.15595
[1-13]: https://arxiv.org/abs/2309.00071
[1-14]: https://aclanthology.org/2023.acl-long.816.pdf
[1-15]: https://arxiv.org/abs/2403.00071
[1-16]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf
[1-17]: https://arxiv.org/abs/2405.13985
[1-18]: https://aclanthology.org/2024.emnlp-main.414.pdf
[1-19]: https://proceedings.neurips.cc/paper_files/paper/2024/file/1403ab1a427050538ec59c7f570aec8b-Paper-Conference.pdf
[1-20]: https://arxiv.org/abs/2509.10534
[1-21]: https://aclanthology.org/2025.emnlp-main.1545.pdf

- **OneFormer3D: One Transformer for Unified Point Cloud Segmentation** (2024)
  **Driver:** task-specific 3D segmentation models (semantic vs instance vs panoptic)
  **Outcome:** a single query-driven transformer that unifies 3D segmentation tasks in one architecture (set-style decoding over point clouds). ([CVF Open Access][10-1])
- **Spherical Mask: Coarse-to-Fine 3D Point Cloud Instance Segmentation with Spherical Representation** (2024)
  **Driver:** coarse-to-fine pipelines whose 3D proposals (e.g., AABB) propagate errors; also compared against stronger transformer-era instance segmentation baselines
  **Outcome:** a spherical (radial) 3D instance representation plus pipeline that lifts coarse detection into improved 3D mask assembly (representation/decoding is the centerpiece). ([CVF Open Access][10-2])
- **Open3DIS: Open-Vocabulary 3D Instance Segmentation with 2D Mask Guidance** (2024)
  **Driver:** closed-vocabulary 3D instance segmenters; limited ability to use 2D foundation signals in 3D
  **Outcome:** fuse multi-frame 2D mask guidance into 3D and perform open-vocabulary 3D instance segmentation (the 2D-to-3D mapping plus 3D transformer reasoning is central). ([CVF Open Access][10-3])
- **ASGFormer: Point cloud semantic segmentation with adaptive spatial graph transformer** (2024)
  **Driver:** point segmentation methods that struggle on complex scenes / adherent objects
  **Outcome:** explicit graph + transformer hybrid for point clouds, using graph structure to support global correlation modeling. ([ScienceDirect][10-4])
- **PointRegion: Transformer based 3D tooth segmentation via point cloud processing** (2024)
  **Driver:** transformer point methods that only aggregate locally and cannot efficiently model global context due to memory cost
  **Outcome:** a transformer segmentation design to process entire dental point clouds at lower cost (3D medical point-domain adaptation). ([PubMed][10-5])
- **SegPoint: Segment Any Point Cloud via Large Language** (2024)
  **Driver:** basic 3D segmentation transformers that do not naturally handle language-driven 3D segmentation tasks
  **Outcome:** unify multiple 3D segmentation modes (semantic / referring / instruction / open-vocab) by coupling a point encoder with an LLM-style reasoning + segmentation token path (a 3D+language adaptation). ([Source Missing][10-6])
- **Point Transformer V3: Simpler, Faster, Stronger** (2024)
  **Driver:** earlier point transformers with costly neighbor search / scaling friction
  **Outcome:** serialize/organize point clouds for efficient attention-driven processing, aiming to make point-transformer backbones truly scalable for large 3D scenes. ([CVF Open Access][10-7])
- **ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention** (2024)
  **Driver:** window-based voxel transformers that require heavy sorting/padding for variable-length voxel groups
  **Outcome:** treat voxels across windows as one sequence via scattered linear attention plus cross-window interaction, an efficiency-first transformer backbone for large-scale LiDAR detection. ([Source Missing][10-8])
- **DeLiVoTr: Deep and Light-weight Voxel Transformer for 3D Object Detection** (2024)
  **Driver:** voxel pipelines that downsample too aggressively and miss small objects in large-scale driving scenes
  **Outcome:** a voxel transformer backbone designed to keep feature-map scale while expanding effective receptive field for LiDAR detection. ([ScienceDirect][10-9])
- **Voxel self-attention and center-point for 3D object detector** (2024)
  **Driver:** LiDAR detectors that do not model scene context deeply in voxel space
  **Outcome:** apply self-attention on voxel features in an anchor-free LiDAR detection design (voxel-attention as the primary adaptation). ([Cell][10-10])
- **Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation** (2025)
  **Driver:** transformer instance segmenters that mainly model "scene <-> query" but underuse richer relations
  **Outcome:** strengthen relation modeling within transformer-style instance segmentation pipelines for 3D scenes. ([CVF Open Access][10-11])
- **CamPoint: Boosting Point Cloud Segmentation with Virtual Camera** (2025)
  **Driver:** point segmentation that struggles to find semantically relevant neighbors and to inject high-level global information per point
  **Outcome:** introduce virtual-camera visibility tokens/features and global interaction machinery to restructure how point neighborhoods and global context are formed (representation/interface change is central). ([CVPR][10-12])
- **PMFormer: Point mask transformer for outdoor point cloud semantic segmentation** (2025)
  **Driver:** outdoor point segmentation pipelines that do not translate cleanly into mask/query-style transformer formulations
  **Outcome:** cast outdoor point semantic segmentation into a mask transformer style pipeline (query/mask prediction framing in 3D). ([SciOpen][10-13])
- **BWFormer: Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer** (2025)
  **Driver:** non-transformer pipelines for airborne LiDAR building wireframe reconstruction
  **Outcome:** a transformer pipeline that projects LiDAR to a 2.5D height map for 2D corner detection, then lifts to 3D with transformer-based corner/edge reasoning (explicit 2D-to-3D restructuring). ([CVPR][10-14])
- **SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection** (2025)
  **Driver:** BEV-centric extraction and dense voxel inefficiency
  **Outcome:** operate directly on sparse 3D voxel features as transformer inputs for detection (voxel-domain transformerization is central). ([arXiv][10-15])
- **RetentiveBEV: BEV transformer for visual 3D object detection** (2025)
  **Driver:** BEV transformer approaches needing better spatiotemporal feature retention/learning
  **Outcome:** a transformer that learns spatiotemporal BEV features for 3D detection (camera-to-BEV-to-3D reasoning as the adaptation). ([SAGE Journals][10-16])
- **PV-DT3D: Point-voxel dual transformer for LiDAR 3D object detection** (2025)
  **Driver:** single-representation pipelines that miss complementary point vs voxel cues
  **Outcome:** a dual point-wise + channel-wise transformer encoder-decoder that fuses point and voxel abstractions for proposal refinement. ([Springer][10-17])
- **DCT: Dynamic Clustering Transformer for LiDAR-Based 3D** (2025)
  **Driver:** feature extraction that does not exploit LiDAR's "non-overlapping object" structure effectively
  **Outcome:** clustering-driven point cloud backbone that incorporates transformer attention around dynamic clusters. ([ScienceDirect][10-18])
- **LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias** (2025)
  **Driver:** NeRF / 3DGS-style methods requiring explicit 3D structure or strong geometric inductive bias
  **Outcome:** a transformer-based feed-forward NVS model that replaces explicit 3D structures with learned token-based scene representations (encoder-decoder and decoder-only variants). ([ICLR Proceedings][10-19])
- **Scaling Transformer-Based Novel View Synthesis with Models Token Disentanglement and Synthetic Data** (2025)
  **Driver:** transformer NVS constrained by limited real scene diversity and entangled token features
  **Outcome:** scale transformer NVS with synthetic data plus token disentanglement inside the transformer to improve reconstruction quality/generalization. ([CVF Open Access][10-20])
- **PanSt3R: Multi-view Consistent Panoptic Segmentation** (2025)
  **Driver:** 2D panoptic in multi-view/3D settings and test-time-optimization-heavy pipelines
  **Outcome:** a unified transformer-style method that jointly predicts 3D geometry plus multi-view panoptic segmentation in a single forward pass (multi-view to 3D adaptation is the core). ([CVF Open Access][10-21])
- **DT-NVS: Diffusion Transformers for Novel View Synthesis** (2025)
  **Driver:** diffusion NVS approaches limited to narrow scene assumptions
  **Outcome:** transformer backbone predicts a radiance field for NVS from minimal inputs. Note: borderline (diffusion-first, but transformer is the scene-to-radiance-field engine). ([arXiv][10-22])

[10-1]: https://openaccess.thecvf.com/content/CVPR2024/papers/Kolodiazhnyi_OneFormer3D_One_Transformer_for_Unified_Point_Cloud_Segmentation_CVPR_2024_paper.pdf
[10-2]: https://openaccess.thecvf.com/content/CVPR2024/papers/Shin_Spherical_Mask_Coarse-to-Fine_3D_Point_Cloud_Instance_Segmentation_with_Spherical_CVPR_2024_paper.pdf
[10-3]: https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_Open3DIS_Open-Vocabulary_3D_Instance_Segmentation_with_2D_Mask_Guidance_CVPR_2024_paper.pdf
[10-4]: https://www.sciencedirect.com/science/article/pii/S156984322400459X
[10-5]: https://pubmed.ncbi.nlm.nih.gov/39557955/
[10-6]: MISSING
[10-7]: https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf
[10-8]: MISSING
[10-9]: https://www.sciencedirect.com/science/article/pii/S2667305324000371
[10-10]: https://www.cell.com/iscience/fulltext/S2589-0042%2824%2901984-9
[10-11]: https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.html
[10-12]: https://cvpr.thecvf.com/virtual/2025/poster/34611
[10-13]: https://www.sciopen.com/article/10.26599/CVM.2025.9450388
[10-14]: https://cvpr.thecvf.com/virtual/2025/poster/32868
[10-15]: https://arxiv.org/html/2503.08092v1
[10-16]: https://journals.sagepub.com/doi/10.1177/01423312241308367
[10-17]: https://link.springer.com/article/10.1007/s11801-025-3134-9
[10-18]: https://www.sciencedirect.com/science/article/abs/pii/S0031320325011069
[10-19]: https://proceedings.iclr.cc/paper_files/paper/2025/hash/9676c5283df26cabca412ca66b164a7d-Abstract-Conference.html
[10-20]: https://openaccess.thecvf.com/content/ICCV2025/papers/Nair_Scaling_Transformer-Based_Novel_View_Synthesis_with_Models_Token_Disentanglement_and_ICCV_2025_paper.pdf
[10-21]: https://openaccess.thecvf.com/content/ICCV2025/papers/Zust_PanSt3R_Multi-view_Consistent_Panoptic_Segmentation_ICCV_2025_paper.pdf
[10-22]: https://arxiv.org/html/2511.08823v1

- **Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos (P4Transformer)** (2021)
  **Driver:** 1D Transformers (language) and single-frame 3D point cloud models
  **Outcome:** treat point cloud sequences as 4D (x,y,z,t) and model them directly with a spatiotemporal point transformer, avoiding explicit point tracking across frames. ([CVF Open Access][11-1])
- **Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding** (2022)
  **Driver:** short-range 4D point transformers that struggle with long-term context
  **Outcome:** a 4D backbone that builds and attends over spatiotemporal point primitives to extend modeling range in 4D sequences. ([ECVA][11-2])
- **LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection** (2022)
  **Driver:** 1D transformer success and 3D detection fusion pipelines that do not model sensor x time interactions well
  **Outcome:** a 4D spatiotemporal fusion transformer that explicitly fuses across sensors and time (4D = space + time, multi-modal fusion as central transformer adaptation). ([CVF Open Access][11-3])
- **Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion** (2022)
  **Driver:** BEV-style approaches and naive temporal fusion that do not scale well
  **Outcome:** "4D sampling" over (view, scale, timestamp, keypoints) to sparsely gather and fuse spatiotemporal evidence for detection (a representation + fusion adaptation for 4D perception). ([arXiv][11-4])
- **LeaF: Learning Frames for 4D Point Cloud Sequence Understanding** (2023)
  **Driver:** generic 4D tools that do not exploit the strong prior that 4D sequences come from structured frames/motion
  **Outcome:** a 4D point-sequence representation learning framework targeting geometry + motion descriptors for point cloud videos (4D understanding as the core). ([CVF Open Access][11-5])
- **4D-Former: Multimodal 4D Panoptic Segmentation** (2023)
  **Driver:** single-scan 3D panoptic segmentation and non-learned temporal association pipelines
  **Outcome:** transformer-style panoptic decoding plus learned tracklet association over LiDAR sequences (4D) with LiDAR+RGB fusion (4D panoptic + tracking as the centerpiece). ([Proceedings of Machine Learning Research][11-6])
- **Mask4Former: Mask Transformer for 4D Panoptic Segmentation** (2023)
  **Driver:** hand-crafted temporal association (tracking-by-heuristics) and pipelines that treat 3D segmentation/tracking separately
  **Outcome:** a mask-transformer that unifies semantic + instance segmentation + tracking on sequences of LiDAR point clouds via spatio-temporal instance queries. ([arXiv][11-7])
- **DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving** (2024)
  **Driver:** 2D-pretraining-centric perception and non-world-model temporal pipelines
  **Outcome:** 4D pretraining that models dynamic scenes over time for downstream 3D tasks (world-model framing: learn spatiotemporal dynamics as a core representation). ([CVF Open Access][11-8])
- **OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving** (2024)
  **Driver:** static / single-frame occupancy and limited temporal modeling
  **Outcome:** a world-model framework that explicitly models 4D occupancy evolution (3D space over time) as the central representation for planning/forecasting tasks. ([ECVA][11-9])
- **Interactive4D: Interactive 4D LiDAR Segmentation** (2024)
  **Driver:** interactive segmentation framed per-scan (ignores the full space-time volume)
  **Outcome:** interactive segmentation that operates over multiple LiDAR scans jointly (space-time volume), enabling multi-object segmentation across 4D LiDAR in one iteration (4D interaction paradigm is central). ([arXiv][11-10])
- **Enhanced Scene Understanding on 4D Point Cloud** (2024)
  **Driver:** 3D-only perception that fails to leverage temporal cues for dynamic scene understanding
  **Outcome:** transformer-based fusion of RGB sequences to 4D point cloud understanding, emphasizing temporal relationship modeling as the key mechanism. ([AAAI Conference Proceedings][11-11])
- **LLFormer4D: LiDAR-based lane detection method** (2024)
  **Driver:** frame-wise lane detection and limited temporal aggregation
  **Outcome:** explicit spatio-temporal transformer lane detection built around 4D LiDAR signal structure (space + time). ([IET Research Journals][11-12])
- **Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D)** (2025)
  **Driver:** closed-vocabulary 4D panoptic segmentation and training-data bottlenecks
  **Outcome:** a pipeline with a Transformer-based instance decoder operating in 4D LiDAR space, enabling zero-shot recognition by lifting/aligning external visual-language signals into 4D segmentation + tracking. ([CVF Open Access][11-13])
- **OccSora** (2024)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source. ([GitHub][11-14])
- **DynamicCity** (2025)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source. ([arXiv][11-15])

[11-1]: https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf
[11-2]: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890018.pdf
[11-3]: https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_LIFT_Learning_4D_LiDAR_Image_Fusion_Transformer_for_3D_Object_CVPR_2022_paper.pdf
[11-4]: https://arxiv.org/abs/2211.10581
[11-5]: https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_LeaF_Learning_Frames_for_4D_Point_Cloud_Sequence_Understanding_ICCV_2023_paper.pdf
[11-6]: https://proceedings.mlr.press/v229/athar23a/athar23a.pdf
[11-7]: https://arxiv.org/abs/2309.16133
[11-8]: https://openaccess.thecvf.com/content/CVPR2024/papers/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_CVPR_2024_paper.pdf
[11-9]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02024.pdf
[11-10]: https://arxiv.org/abs/2410.08206
[11-11]: https://ojs.aaai.org/index.php/AAAI/article/view/28045
[11-12]: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12338
[11-13]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Zero-Shot_4D_Lidar_Panoptic_Segmentation_CVPR_2025_paper.pdf
[11-14]: https://github.com/wzzheng/OccSora
[11-15]: https://arxiv.org/html/2410.18084v3

- **Point 4D Transformer Networks for Spatio-Temporal Modeling in Point Cloud Videos (P4Transformer)** (2021)
  **Driver:** 1D Transformers (language) and single-frame 3D point models
  **Outcome:** Treat point cloud sequences as **4D (x,y,z,t)** and model them directly with **spatiotemporal point attention** (explicitly avoiding point tracking). ([CVF Open Access][12-1])
- **LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection** (2022)
  **Driver:** 1D Transformer success + 3D perception fusion that underuses **cross-sensor temporal context**
  **Outcome:** A **LiDAR-camera fusion Transformer** that explicitly models **cross-sensor interactions over time** (4D sequential fusion as the core design). ([CVF Open Access][12-2])
- **Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion** (2022)
  **Driver:** BEV-style 3D detection and heavy dense view transforms / global attention for multi-view sequences
  **Outcome:** "Sparse **4D sampling**": assign **4D keypoints** per 3D anchor and fuse features across **view/scale/timestamp** via sparse sampling + hierarchical fusion (spatiotemporal 4D representation is central). ([arXiv][12-3])
- **Mask4D: End-to-End Mask-Based 4D Panoptic Segmentation for LiDAR Sequences** (2023)
  **Driver:** pipelines that do 3D panoptic per-frame then use **hand-crafted temporal association**
  **Outcome:** An end-to-end **mask + query** formulation that reuses/updates queries over time to keep consistent IDs (4D = segmentation + tracking across frames). ([IPB Uni Bonn][12-4])
- **4D-Former: Multimodal 4D Panoptic Segmentation** (2023)
  **Driver:** LiDAR-only 4D panoptic systems that lack strong appearance cues and struggle with temporal consistency
  **Outcome:** A **query-based Transformer** that fuses **LiDAR sequences + RGB** and iteratively refines semantic + temporally consistent instance masks (explicit 4D panoptic transformerization). ([arXiv][12-5])
- **Mask4Former: Mask Transformer for 4D Panoptic Segmentation** (2023)
  **Driver:** non-learned association strategies and "segmentation then tracking" decomposition
  **Outcome:** Unify **semantic + instance segmentation + tracking** with **spatio-temporal instance queries** in a single transformer-style model for LiDAR sequences. ([arXiv][12-6])
- **DriveWorld: 4D Pre-trained Scene Understanding via World Models for Autonomous Driving** (2024)
  **Driver:** 2D-pretraining-centric perception and weaker temporal scene representations for driving
  **Outcome:** A **world-model 4D representation learning** framework that pretrains on multi-camera sequences in a spatiotemporal (4D) fashion for downstream 3D tasks (4D representation is the centerpiece). ([CVF Open Access][12-7])
- **OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving** (2024)
  **Driver:** modeling dynamics via tracked boxes (limited scene-level fidelity)
  **Outcome:** Models **4D occupancy evolution** (3D space over time) as a world model for forecasting/planning; strongly "4D representation" driven, but the paper is sometimes framed as world-modeling rather than "Transformer-first." Note: borderline (Transformer involvement depends on implementation details). ([ECVA][12-8])
- **Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D)** (2025)
  **Driver:** supervised 4D panoptic segmentation bottlenecked by labels; limited transfer from 2D/video foundation models to LiDAR sequences
  **Outcome:** A pipeline that lifts multimodal foundation signals (e.g., video object segmentation + VLM features) into **4D LiDAR** and trains a model for **4D panoptic segmentation** without labeled 4D data (4D adaptation is central). ([CVF Open Access][12-9])
- **Streaming 4D Panoptic Segmentation via Dual Threads** (2025)
  **Driver:** offline 4D panoptic methods that aren't designed for real-time sequential deployment
  **Outcome:** Explicitly targets **streaming 4D** inference over point cloud sequences under strict time budgets (the domain shift is '4D + streaming'). Note: borderline (streaming setup is central; confirm transformer core if you want this kept "strict"). ([arXiv][12-10])

[12-1]: https://openaccess.thecvf.com/content/CVPR2021/papers/Fan_Point_4D_Transformer_Networks_for_Spatio-Temporal_Modeling_in_Point_Cloud_CVPR_2021_paper.pdf
[12-2]: https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_LIFT_Learning_4D_LiDAR_Image_Fusion_Transformer_for_3D_Object_CVPR_2022_paper.pdf
[12-3]: https://arxiv.org/abs/2211.10581
[12-4]: https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral-meem.pdf
[12-5]: https://arxiv.org/abs/2311.01520
[12-6]: https://arxiv.org/abs/2309.16133
[12-7]: https://openaccess.thecvf.com/content/CVPR2024/papers/Min_DriveWorld_4D_Pre-trained_Scene_Understanding_via_World_Models_for_Autonomous_CVPR_2024_paper.pdf
[12-8]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02024.pdf
[12-9]: https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Zero-Shot_4D_Lidar_Panoptic_Segmentation_CVPR_2025_paper.pdf
[12-10]: https://arxiv.org/html/2510.17664v1

- **Neural Turing Machines (NTM)** (2014)
  **Driver:** persistent, addressable memory + algorithmic state beyond a fixed hidden vector
  **Outcome:** differentiable **read/write heads** over an external memory matrix (end-to-end trainable "computer-like" inference). ([arXiv][13-1])
- **End-To-End Memory Networks (MemN2N)** (2015)
  **Driver:** multi-step retrieval / multi-hop reasoning rather than single-pass inference
  **Outcome:** differentiable **external memory** + repeated **attention "hops"** before producing an answer. ([arXiv][13-2])
- **Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets** (2015)
  **Driver:** counting + hierarchical/algorithmic behavior hard for standard RNNs
  **Outcome:** attach a differentiable **stack data structure** to a recurrent controller. ([arXiv][13-3])
- **Learning to Transduce with Unbounded Memory** (2015)
  **Driver:** unbounded structured memory for transduction-like tasks
  **Outcome:** differentiable **stack/queue/deque** operations integrated into the model's computation loop. ([arXiv][13-4])
- **Neural Programmer-Interpreters (NPI)** (2015)
  **Driver:** program induction + compositional execution with reusable subroutines
  **Outcome:** a recurrent core + persistent **program memory** that **calls subprograms** (learned program execution). ([arXiv][13-5])
- **Neural Programmer: Inducing Latent Programs with Gradient Descent** (2015)
  **Driver:** arithmetic/logic operations that pure sequence models struggle to learn
  **Outcome:** differentiable selection/composition of **built-in operators** across multiple steps (latent program execution). ([arXiv][13-6])
- **Neural GPUs Learn Algorithms** (2015)
  **Driver:** learning algorithms that generalize to much longer inputs than training
  **Outcome:** a highly-parallel **convolutional recurrent** computation grid that performs iterative algorithmic updates. ([arXiv][13-7])
- **Adaptive Computation Time (ACT)** (2016)
  **Driver:** variable depth / "think longer on hard cases"
  **Outcome:** differentiable **halting** that learns how many internal steps to run per input/time step. ([arXiv][13-8])
- **Differentiable Neural Computer (DNC)** (2016)
  **Driver:** learning and manipulating data structures (graphs, lists) with long-term memory
  **Outcome:** NTM-like external memory + differentiable **allocation and temporal linkage** for structured reads/writes. ([Nature][13-9])
- **Sparse Differentiable Neural Computer (SDNC)** (2016)
  **Driver:** scaling memory-augmented computation to larger memories
  **Outcome:** sparse/differentiable approximations enabling large external memory access efficiently. ([arXiv][13-10])
- **Neural Module Networks (NMN)** (2016)
  **Driver:** compositional reasoning pipelines (dynamic per-question computation graphs)
  **Outcome:** dynamically assemble a network from reusable **modules** conditioned on input structure. ([CVF Open Access][13-11])
- **End-to-End Differentiable Proving** (2017)
  **Driver:** symbolic-style multi-hop logical inference (backward chaining) in neural form
  **Outcome:** recursively construct computation inspired by **Prolog-style backward chaining** with differentiable unification. ([arXiv][13-12])
- **Lie-Access Neural Turing Machines (LANTM)** (2017)
  **Driver:** flexible memory addressing beyond standard content-based lookup
  **Outcome:** alternative differentiable **memory access geometry** (Lie group actions) for algorithmic tasks. ([OpenReview][13-13])
- **Recurrent Relational Networks (RRN)** (2018)
  **Driver:** iterative constraint satisfaction / multi-step relational inference
  **Outcome:** recurrent **message passing** over a relational graph for many inference steps. ([NeurIPS Papers][13-14])
- **Universal Transformers (UT)** (2018)
  **Driver:** iterative refinement per token position (beyond a fixed-depth feedforward stack)
  **Outcome:** apply a **recurrent** transformer block repeatedly (often with ACT-style halting) to revise representations. ([arXiv][13-15])
- **Neural Ordinary Differential Equations (Neural ODEs)** (2018)
  **Driver:** continuous-depth / adaptive computation tied to solver precision
  **Outcome:** replace discrete layers with an ODE-defined hidden-state dynamics solved by a **black-box ODE solver**. ([arXiv][13-16])
- **SATNet: Differentiable Satisfiability Solver** (2019)
  **Driver:** constraint solving (e.g., Sudoku) that standard nets struggle with
  **Outcome:** integrate a differentiable (smoothed) **MAXSAT solver** into end-to-end learning. ([arXiv][13-17])
- **Deep Equilibrium Models (DEQ)** (2019)
  **Driver:** effectively infinite-depth computation / equilibrium dynamics with constant memory
  **Outcome:** define the network by a **fixed point** and compute it via root-finding; backprop via implicit differentiation. ([arXiv][13-18])
- **Adaptive Attention Span in Transformers** (2019)
  **Driver:** dynamic allocation of compute/memory over long contexts
  **Outcome:** each head learns its **effective attention span** (soft mask) to trade accuracy vs compute. ([arXiv][13-19])
- **REALM: Retrieval-Augmented Language Model Pre-Training** (2020)
  **Driver:** updatable knowledge without baking everything into parameters
  **Outcome:** learned **retriever + reader**; retrieval is in the computation loop during pretrain and inference. ([arXiv][13-20])
- **Depth-Adaptive Transformers** (2020)
  **Driver:** variable compute at inference time for easy vs hard inputs
  **Outcome:** dynamically decide how many transformer layers to execute per input (ACT-inspired). ([Jiatao Gu][13-21])
- **PonderNet: Learning to Ponder** (2021)
  **Driver:** stable, learnable variable compute / iterative refinement
  **Outcome:** probabilistic halting over repeated computation steps with an explicit tradeoff between accuracy and compute. ([arXiv][13-22])
- **Memorizing Transformers** (2022)
  **Driver:** acquire new facts at inference time without weight updates
  **Outcome:** approximate **kNN retrieval** over stored key/value representations inside a transformer layer. ([arXiv][13-23])
- **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)
  **Driver:** grounded multi-step problem solving that can query the world / reduce hallucinations
  **Outcome:** interleave **reasoning traces** with **actions** (tool/API calls) as part of the inference procedure. ([arXiv][13-24])
- **Toolformer: Language Models Can Teach Themselves to Use Tools** (2023)
  **Driver:** reliable arithmetic/lookup/planning subskills via external tools
  **Outcome:** self-supervised training that inserts/filters **API calls** to improve future token prediction. ([arXiv][13-25])
- **Tree of Thoughts (ToT)** (2023)
  **Driver:** exploration, lookahead, backtracking - beyond left-to-right greedy decoding
  **Outcome:** explicit **tree search** over intermediate "thought" states, with self-evaluation and backtracking. ([arXiv][13-26])
- **RETRO-style retrieval-augmented pretraining and variants** (2024)
  **Driver:** parameter-efficient scaling via external memory and cross-attention to retrieved chunks
  **Outcome:** retrieval + chunked cross-attention in the forward pass. Note: RETRO originally 2021; many 2024 follow-ons compare/extend the compute pattern; included for continuity. ([arXiv][13-27])
- **Adaptive Thinking Using Dynamic Computation** (2025)
  **Driver:** allocate "thinking steps" dynamically (variable compute)
  **Outcome:** dynamic computation policies building on halting/dynamic-depth ideas (ACT/PonderNet lineage). ([ICLR Proceedings][13-28])

[13-1]: https://arxiv.org/abs/1410.5401
[13-2]: https://arxiv.org/abs/1503.08895
[13-3]: https://arxiv.org/abs/1503.01007
[13-4]: https://arxiv.org/abs/1506.02516
[13-5]: https://arxiv.org/abs/1511.06279
[13-6]: https://arxiv.org/abs/1511.04834
[13-7]: https://arxiv.org/abs/1511.08228
[13-8]: https://arxiv.org/abs/1603.08983
[13-9]: https://www.nature.com/articles/nature20101
[13-10]: https://arxiv.org/pdf/1610.09027
[13-11]: https://openaccess.thecvf.com/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html
[13-12]: https://arxiv.org/abs/1705.11040
[13-13]: https://openreview.net/pdf?id=Byiy-Pqlx
[13-14]: https://papers.neurips.cc/paper/7597-recurrent-relational-networks.pdf
[13-15]: https://arxiv.org/abs/1807.03819
[13-16]: https://arxiv.org/abs/1806.07366
[13-17]: https://arxiv.org/abs/1905.12149
[13-18]: https://arxiv.org/abs/1909.01377
[13-19]: https://arxiv.org/abs/1905.07799
[13-20]: https://arxiv.org/abs/2002.08909
[13-21]: https://jiataogu.me/papers/elbayad2020depth.pdf
[13-22]: https://arxiv.org/abs/2107.05407
[13-23]: https://arxiv.org/abs/2203.08913
[13-24]: https://arxiv.org/abs/2210.03629
[13-25]: https://arxiv.org/abs/2302.04761
[13-26]: https://arxiv.org/abs/2305.10601
[13-27]: https://arxiv.org/abs/2112.04426
[13-28]: https://proceedings.iclr.cc/paper_files/paper/2025/file/955499a8e2860ed746717c1374224c43-Paper-Conference.pdf

- **Deep Equilibrium Models (DEQ)** (2019)
  **Driver:** effectively infinite-depth / iterative refinement with constant memory (vs fixed feedforward stacks).
  **Outcome:** define the network by a fixed point and compute it via root-finding; train via implicit differentiation. ([Implicit Layers Tutorial][14-1])
- **Differentiable Convex Optimization Layers** (2019)
  **Driver:** exact constraint satisfaction / optimization inside neural pipelines.
  **Outcome:** embed a disciplined convex program solver as a differentiable layer (forward solves; backward differentiates through the solve). ([arXiv][14-2])
- **Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE Solvers** (2020)
  **Driver:** correcting / steering iterative solvers with learned components rather than replacing them.
  **Outcome:** learn correction functions that plug into and improve iterative PDE solvers ("solver-in-the-loop" compute). ([NeurIPS Proceedings][14-3])
- **Stabilizing Equilibrium Models by Jacobian Regularization** (2021)
  **Driver:** stable, reliable fixed-point inference/training (DEQs can be brittle/unstable).
  **Outcome:** Jacobian regularization of the fixed-point update to stabilize convergence in forward and backward passes. ([arXiv][14-4])
- **Self-Consistency Improves Chain-of-Thought Reasoning** (2022)
  **Driver:** robustness vs a single greedy reasoning path.
  **Outcome:** sample many reasoning traces and choose the most consistent answer (inference-time ensembling over thoughts). ([arXiv][14-5])
- **STaR: Self-Taught Reasoner (Bootstrapping Reasoning With Reasoning)** (2022)
  **Driver:** reliable reasoning skill without massive curated rationales.
  **Outcome:** iterative loop: generate rationales -> repair/regen on failures -> finetune -> repeat (train/test compute loop that explicitly iterates). ([arXiv][14-6])
- **Jacobian-Free Backpropagation for Implicit Networks (JFB)** (2022)
  **Driver:** efficient training of implicit/fixed-point models (implicit differentiation can be heavy).
  **Outcome:** Jacobian-free methods for backprop through implicit layers, reducing memory/compute pain. ([Emory Mathematics][14-7])
- **Training Iterative Refinement Algorithms with Implicit Differentiation** (2022)
  **Driver:** training models whose inference is itself an iterative refinement procedure (fixed-point style).
  **Outcome:** cast iterative refinement as a fixed-point procedure and train using implicit differentiation. ([NeurIPS Proceedings][14-8])
- **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)
  **Driver:** pure text-only reasoning cannot gather info / ground itself; acting and reasoning studied separately.
  **Outcome:** interleave reasoning traces and actions (tool/environment calls) as the inference procedure. ([arXiv][14-9])
- **Tree of Thoughts (ToT): Deliberate Problem Solving with Large Language Models** (2023)
  **Driver:** exploration, lookahead, backtracking beyond left-to-right decoding.
  **Outcome:** do tree search over thought chunks with self-evaluation and backtracking. ([arXiv][14-10])
- **Reasoning with Language Model is Planning with World Model (RAP)** (2023)
  **Driver:** deliberate planning (simulate future states/outcomes; explore alternatives).
  **Outcome:** repurpose the LLM as agent + world model and run MCTS-style planning over reasoning trajectories. ([arXiv][14-11])
- **Self-Refine: Iterative Refinement with Self-Feedback** (2023)
  **Driver:** one-shot generations are often suboptimal (no revision loop).
  **Outcome:** explicit feedback -> refine -> feedback loop using the model itself (no extra training required). ([arXiv][14-12])
- **Reflexion: Language Agents with Verbal Reinforcement Learning** (2023)
  **Driver:** agents do not learn from trial-and-error without costly weight updates.
  **Outcome:** store and reuse episodic reflections as memory to improve future attempts (test-time learning loop without gradient updates). ([arXiv][14-13])
- **Plan-and-Solve Prompting** (2023)
  **Driver:** zero-shot CoT makes missing-step / semantic errors; no explicit plan phase.
  **Outcome:** two-stage inference: plan (decompose) then solve (execute). ([arXiv][14-14])
- **Graph of Thoughts (GoT)** (2023)
  **Driver:** linear/tree-only reasoning structures limit reuse/merging of partial results.
  **Outcome:** model intermediate thoughts as an arbitrary graph with dependency edges; run graph-structured inference. ([arXiv][14-15])
- **One-Step Diffusion Distillation via Deep Equilibrium Models (Generative Equilibrium Transformer, GET)** (2023)
  **Driver:** many-step diffusion inference is expensive; want solve-to-equilibrium behavior.
  **Outcome:** introduce an equilibrium (fixed-point) transformer layer inside the generative pipeline. ([NeurIPS Papers][14-16])
- **Language Agent Tree Search (LATS)** (2023)
  **Driver:** act-only agents are myopic; need principled planning + exploration.
  **Outcome:** integrate MCTS with LM reasoning/acting plus LM-powered value/reflection signals. ([arXiv][14-17])
- **Test-Time Training on Nearest Neighbors for Large Language Models** (2024)
  **Driver:** static inference cannot adapt to distribution shift / specialized domains without retraining.
  **Outcome:** retrieve nearest-neighbor text for the test input and do small gradient updates at test time (dynamic evaluation style). ([arXiv][14-18])
- **Fixed Point Diffusion Models (FPDM)** (2024)
  **Driver:** fixed-depth denoisers give fixed compute per step; want variable compute tied to solution accuracy.
  **Outcome:** integrate an implicit fixed-point solving layer into diffusion denoising to allow variable computation. ([arXiv][14-19])
- **Implicit Factorized Transformer (IFactFormer)** (2024)
  **Driver:** stable very-deep transformer computation for long-horizon dynamics (e.g., PDE/turbulence).
  **Outcome:** implicit iteration over factorized attention (compute is solving an implicit system, not running N layers). ([ScienceDirect][14-20])
- **Revisiting the Test-Time Scaling of o1-like Models** (2025)
  **Driver:** one-pass inference caps reasoning quality; need systematic test-time compute scaling.
  **Outcome:** evaluate and organize parallel vs sequential test-time compute scaling strategies (sampling/reranking/longer reasoning). ([ACL Anthology][14-21])
- **Efficiently Allocating Test-Time Compute for LLM Agents** (2025)
  **Driver:** always planning is expensive; never planning fails on long-horizon tasks.
  **Outcome:** a framework for dynamic planning decisions: when to plan and spend compute where it matters. ([arXiv][14-22])
- **Test-Time Learning for Large Language Models (TLM / TTL)** (2025)
  **Driver:** LLMs struggle under domain shift; static inference cannot adapt.
  **Outcome:** test-time learning using unlabeled test data (explicitly making adaptation part of inference). ([OpenReview][14-23])
- **Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT)** (2025)
  **Driver:** open-loop next-token prediction can accumulate inconsistencies; needs iterative self-consistency before committing.
  **Outcome:** per token, solve a latent energy minimization via iterative refinement until reaching an equilibrium. ([arXiv][14-24])
- **A Fully First-Order Layer for Differentiable Optimization** (2025)
  **Driver:** implicit differentiation for optimization layers can be compute/memory intensive.
  **Outcome:** compute gradients for optimization layers using first-order information to reduce overhead. ([arXiv][14-25])

[14-1]: https://implicit-layers-tutorial.org/deep_equilibrium_models/
[14-2]: https://arxiv.org/abs/1910.12430
[14-3]: https://proceedings.neurips.cc/paper_files/paper/2020/hash/43e4e6a6f341e00671e123714de019a8-Abstract.html
[14-4]: https://arxiv.org/abs/2106.14342
[14-5]: https://arxiv.org/abs/2203.11171
[14-6]: https://arxiv.org/abs/2203.14465
[14-7]: https://www.math.emory.edu/site/cmds-reuret/projects/2022-implicit/JFB.pdf
[14-8]: https://proceedings.neurips.cc/paper_files/paper/2022/file/d301e2878a7ebadf1a95029e904fc7d0-Paper-Conference.pdf
[14-9]: https://arxiv.org/abs/2210.03629
[14-10]: https://arxiv.org/abs/2305.10601
[14-11]: https://arxiv.org/abs/2305.14992
[14-12]: https://arxiv.org/abs/2303.17651
[14-13]: https://arxiv.org/abs/2303.11366
[14-14]: https://arxiv.org/abs/2305.04091
[14-15]: https://arxiv.org/abs/2308.09687
[14-16]: https://papers.neurips.cc/paper_files/paper/2023/file/82f05a105c928c10706213952bf0c8b7-Paper-Conference.pdf
[14-17]: https://arxiv.org/abs/2310.04406
[14-18]: https://arxiv.org/abs/2305.18466
[14-19]: https://arxiv.org/html/2401.08741v1
[14-20]: https://www.sciencedirect.com/science/article/pii/S2095034924000382
[14-21]: https://aclanthology.org/2025.acl-long.232.pdf
[14-22]: https://arxiv.org/html/2509.03581v1
[14-23]: https://openreview.net/forum?id=iCYbIaGKSR&noteId=ScPdA3KZCL
[14-24]: https://arxiv.org/html/2511.21882v1
[14-25]: https://arxiv.org/pdf/2512.02494

- **Neural Turing Machines (NTM)** (2014)
  **Driver:** persistent, addressable memory + algorithmic state beyond fixed activations
  **Outcome:** differentiable **read/write** heads over external memory ([arXiv][15-1])
- **End-to-End Memory Networks (MemN2N)** (2015)
  **Driver:** multi-hop retrieval / iterative evidence aggregation
  **Outcome:** repeated **attention hops** over an external memory before answering ([Source Missing][15-2])
- **Neural Programmer-Interpreters (NPI)** (2015)
  **Driver:** program induction with reusable subroutines
  **Outcome:** a controller that **calls** learned/latent **subprograms** ([Source Missing][15-3])
- **Stack/Queue/Deque-augmented RNNs** (2015)
  **Driver:** algorithmic counting / hierarchical control
  **Outcome:** differentiable **data structures** (stack/queue/deque) as part of computation ([Source Missing][15-4])
- **Adaptive Computation Time (ACT)** (2016)
  **Driver:** adaptive depth ("think longer on hard inputs")
  **Outcome:** differentiable **halting** to choose computation steps per input ([arXiv][15-5])
- **Differentiable Neural Computer (DNC)** (2016)
  **Driver:** scalable algorithmic memory manipulation
  **Outcome:** external memory with differentiable **allocation + temporal linkage** ([Source Missing][15-6])
- **Neural Theorem Provers / Differentiable Proving** (2017)
  **Driver:** symbolic-style multi-hop logical inference
  **Outcome:** differentiable **backward chaining / unification** ([Source Missing][15-7])
- **Recurrent Relational Networks (RRN)** (2018)
  **Driver:** iterative constraint satisfaction / multi-step relational inference
  **Outcome:** recurrent **message passing** over relational graphs for many steps ([arXiv][15-8])
- **Universal Transformer (UT)** (2018)
  **Driver:** iterative refinement beyond fixed-depth feedforward stacks
  **Outcome:** recurrently apply a transformer block (often paired with adaptive halting) ([Source Missing][15-9])
- **Deep Equilibrium Models (DEQ)** (2019)
  **Driver:** infinite-depth-style computation with constant memory
  **Outcome:** solve for a **fixed point** via root-finding; train via **implicit differentiation** ([arXiv][15-10])
- **Differentiable Convex Optimization Layers** (2019)
  **Driver:** exact constraint/optimization inside neural pipelines
  **Outcome:** embed a **convex solver** as a differentiable layer (solve forward; differentiate backward) ([OpenReview][15-11])
- **REALM: Retrieval-Augmented Language Model Pre-Training** (2020)
  **Driver:** updatable factual knowledge without re-training weights
  **Outcome:** learned **retriever + reader**; retrieval is in-loop computation ([Source Missing][15-12])
- **Depth-adaptive / early-exit Transformers (dynamic depth)** (2020)
  **Driver:** variable compute per input
  **Outcome:** learn policies to **skip layers / exit early** (ACT lineage) ([Source Missing][15-13])
- **PonderNet: Learning to Ponder** (2021)
  **Driver:** stable adaptive compute / variable number of reasoning steps
  **Outcome:** probabilistic **halting distribution** over repeated computation steps ([arXiv][15-14])
- **Jacobian regularization for equilibrium models** (2021)
  **Driver:** stable DEQ convergence/training
  **Outcome:** regularize Jacobians to stabilize fixed-point dynamics (DEQ engineering line) ([Source Missing][15-15])
- **Self-Consistency for Chain-of-Thought** (2022)
  **Driver:** single greedy reasoning path is brittle
  **Outcome:** sample multiple reasoning traces and **aggregate/select** the consistent answer ([Source Missing][15-16])
- **ReAct: Synergizing Reasoning and Acting in Language Models** (2022)
  **Driver:** pure feedforward reasoning can't gather missing info / ground itself
  **Outcome:** interleave **reasoning + actions** (tool/env calls) as the inference loop ([arXiv][15-17])
- **Memorizing Transformers / kNN-augmented attention (inference-time memory)** (2022)
  **Driver:** store and use new facts at inference without weight updates
  **Outcome:** fast **kNN retrieval** over cached activations inside the model ([Source Missing][15-18])
- **Tree of Thoughts (ToT)** (2023)
  **Driver:** exploration, lookahead, backtracking beyond left-to-right decoding
  **Outcome:** explicit **tree search** over intermediate "thought" states with self-evaluation ([arXiv][15-19])
- **Toolformer: Language Models Can Teach Themselves to Use Tools** (2023)
  **Driver:** reliable tool use (calc/search/KB) as part of computation, not just prompting
  **Outcome:** self-supervised training to insert/execute **API calls** during inference ([arXiv][15-20])
- **Reflexion / Self-Refine family (test-time improvement loops)** (2023)
  **Driver:** learning from failures without gradient updates
  **Outcome:** store structured self-feedback ("reflections") and iteratively refine outputs at test time ([Source Missing][15-21])
- **Process Reward Models (PRM) introduced/standardized for reasoning traces** (2023)
  **Driver:** outcome-only rewards provide weak credit assignment for multi-step reasoning
  **Outcome:** step-level **process supervision/rewarding** used for search/RL at inference ([Source Missing][15-22])
- **Generative Verifiers / GenRM: Reward Modeling as Next-Token Prediction** (2024)
  **Driver:** better verification signals to guide best-of-N / search at test time
  **Outcome:** train verifiers as **generative** models; use them to steer **Best-of-N** and related TTS ([NeurIPS][15-23])
- **Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning (PAV / "progress rewards")** (2024)
  **Driver:** scalable, informative step-level rewards without dense human labeling
  **Outcome:** define process reward as **progress** (step-level advantage) under a prover policy; improves test-time search + online RL ([arXiv][15-24])
- **Advancing Process Verification for LLM Reasoning** (2024)
  **Driver:** reliable step verification beyond naive self-checking
  **Outcome:** improved **process verification** schemes that directly target better test-time selection/search ([ACL Anthology][15-25])
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025)
  **Driver:** strong multi-step reasoning *emerging from training* (not just prompting)
  **Outcome:** large-scale **RL fine-tuning** to elicit reasoning behaviors; includes an **R1-Zero** pure-RL variant ([arXiv][15-26])
- **Hierarchical Reasoning Model (HRM)** (2025)
  **Driver:** deep, stable latent reasoning with variable compute without emitting long CoT token traces
  **Outcome:** hierarchical recurrent computation with **fast/slow modules**, **adaptive halting (ACT)**, and **one-step equilibrium-style training** ([arXiv][15-27])
- **Scaling up Test-Time Compute with Latent Reasoning** (2025)
  **Driver:** test-time scaling that doesn't require generating ever-more tokens
  **Outcome:** iterate a recurrent block to "reason in latent space," unrolling to arbitrary depth at inference ([NeurIPS][15-28])
- **The Lessons of Developing Process Reward Models...** (2025)
  **Driver:** practical, effective PRMs that generalize (data/labeling is hard)
  **Outcome:** engineering + methodological guidance for building PRMs that actually work (enabling better search/RL at test time) ([arXiv][15-29])
- **R-PRM: Reasoning-Driven Process Reward Modeling** (2025)
  **Driver:** PRMs that are both data-efficient and accurate at step evaluation
  **Outcome:** redesign PRM training/objectives to improve step-level supervision quality ([ACL Anthology][15-30])
- **The Art of Scaling Test-Time Compute for LLMs** (2025)
  **Driver:** principled guidance on which test-time scaling strategies win under fixed budgets
  **Outcome:** systematic framework/analysis of TTS strategy choices ([arXiv][15-31])

[15-1]: https://arxiv.org/abs/1410.5401
[15-2]: MISSING
[15-3]: MISSING
[15-4]: MISSING
[15-5]: https://arxiv.org/abs/1603.08983
[15-6]: MISSING
[15-7]: MISSING
[15-8]: https://arxiv.org/abs/1711.08028
[15-9]: MISSING
[15-10]: https://arxiv.org/abs/1909.01377
[15-11]: https://openreview.net/forum?id=1EuxRTe0WN
[15-12]: MISSING
[15-13]: MISSING
[15-14]: https://arxiv.org/abs/2107.05407
[15-15]: MISSING
[15-16]: MISSING
[15-17]: https://arxiv.org/abs/2210.03629
[15-18]: MISSING
[15-19]: https://arxiv.org/abs/2305.10601
[15-20]: https://arxiv.org/abs/2302.04761
[15-21]: MISSING
[15-22]: MISSING
[15-23]: https://neurips.cc/virtual/2024/104300
[15-24]: https://arxiv.org/abs/2410.08146
[15-25]: https://aclanthology.org/2024.emnlp-main.125.pdf
[15-26]: https://arxiv.org/abs/2501.12948
[15-27]: https://arxiv.org/html/2506.21734v1
[15-28]: https://neurips.cc/virtual/2025/poster/117966
[15-29]: https://arxiv.org/pdf/2501.07301
[15-30]: https://aclanthology.org/2025.emnlp-main.679.pdf
[15-31]: https://arxiv.org/html/2512.02008v1

- **The Abstraction and Reasoning Corpus (ARC)** (2019)
  **Driver:** fluid abstraction + fast skill acquisition from few examples
  **Outcome:** benchmark framing, not a solver, establishes the why behind many later compute-mechanism proposals ([arXiv][16-1])
- **A Neurosymbolic Approach to Abstraction and Reasoning** (2021)
  **Driver:** compositional abstraction + search over symbolic hypotheses
  **Outcome:** neurosymbolic ARC solver framing ARC as program synthesis with learned components ([DSpace][16-2])
- **Tackling the Abstraction and Reasoning Corpus with Vision Transformers: the ViTARC Architecture** (2022)
  **Driver:** visual abstraction with limited examples (ARC's regime)
  **Outcome:** a ViT-style solver + ARC-specific training/inference components (attention-centric visual reasoning). Note: Closer to domain solver design than a single clean compute primitive, but clearly ARC-motivated ([OpenReview][16-3])
- **Generalized Planning for the Abstraction and Reasoning Corpus (GPAR)** (2024)
  **Driver:** explicit planning/program-like generalization instead of pattern fitting
  **Outcome:** cast ARC as generalized planning using PDDL + planning programs, with ARC-specific constraints to make planning scalable ([arXiv][16-4])
- **Combining Induction and Transduction for Abstract Reasoning** (2024)
  **Driver:** one-shot transduction misses precise compositional operations; pure induction misses fuzzy perceptual concepts
  **Outcome:** paired framework: inductive program synthesis model + transductive predictor, combined to cover complementary failure modes on ARC ([arXiv][16-5])
- **The Surprising Effectiveness of Test-Time Training for Few-Shot Learning** (2024)
  **Driver:** static inference underuses per-task compute; fails to adapt to a new puzzle distribution at inference
  **Outcome:** test-time training on the in-context examples/augmentations as the inference procedure (gradient-based refinement loop), reported with strong ARC gains ([arXiv][16-6])
- **Towards Efficient Neurally-Guided Program Induction for ARC-AGI** (2024)
  **Driver:** brute-force synthesis/search is inefficient; standard neural mapping struggles with ARC's OOD generalization
  **Outcome:** neurally-guided program enumeration/search across multiple spaces (grid/program/transform), explicitly evaluated on ARC-AGI ([arXiv][16-7])
- **Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models** (2024)
  **Driver:** bigger models aren't the answer; need iteration/refinement and per-task adaptation in small nets
  **Outcome:** small Transformer family + test-time training/refinement strategies as the core compute lever for ARC-like generalization ([Paul Fletcher-Hill][16-8])
- **Omni-ARC** (2024)
  **Driver:** a single "predict output" objective is too narrow for ARC; systems need richer internal task representations
  **Outcome:** multi-task training over ARC-related subtasks + adaptation/refinement pipeline for solving ARC tasks ([ARC Prize][16-9])
- **ARC Prize 2024: Technical Report** (2024)
  **Driver:** "static" models vs "refinement-loop" systems; need test-time adaptation + algorithmic search hybrids
  **Outcome:** report, not a single solver, but codifies the compute-mechanism trend (TTT, refinement loops, hybrid search) that subsequent solver papers operationalize ([arXiv][16-10])
- **A 2D nGPT Model for ARC Prize** (2024)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source. Note: Often centers on 2D-aware modeling and spatial handling (can drift into domain lifting/2D architecture); ARC Prize-relevant but not always cleanly a computation mechanism paper ([arXiv][16-11])
- **Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective** (2025)
  **Driver:** single-view prompting/decoding is brittle; lacks robust search over invariances/perspectives
  **Outcome:** inference as multi-augmentation product-of-experts, typically paired with test-time fine-tuning per task, as a concrete compute-at-test-time algorithm for ARC ([arXiv][16-12])
- **ARC-AGI Without Pretraining (CompressARC)** (2025)
  **Driver:** reliance on large pretraining; need per-task learning-as-inference
  **Outcome:** per-puzzle test-time training of a tiny model from scratch guided by an MDL/description-length objective (a neural code-golf refinement loop) ([Isaac Liao][16-13])
- **Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR)** (2025)
  **Driver:** hand-designed DSLs + static solvers don't scale; pure LLM prompting lacks reliable search/verification loops
  **Outcome:** evolutionary program synthesis + a loop that fine-tunes the LLM on its own search traces (self-improving refinement) ([OpenReview][16-14])
- **ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus** (2025)
  **Driver:** one-shot inference struggles with emergence of structured transformations
  **Outcome:** solve ARC via Neural Cellular Automata - iterative, local update dynamics producing emergent structure (developmental computation) ([arXiv][16-15])
- **Less is More: Recursive Reasoning with Tiny Networks (TRM)** (2025)
  **Driver:** autoregressive one-pass answers are fragile; need repeated self-correction with tiny compute units
  **Outcome:** a compact recursive refinement loop (latent + answer updated over multiple steps), evaluated on ARC-AGI-1/2 ([ar5iv][16-16])
- **ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory** (2025)
  **Driver:** no persistent cross-task learning of reusable abstractions; starts from scratch each puzzle
  **Outcome:** lifelong memory that stores/retrieves reusable reasoning components for ARC-like tasks (memory as a compute primitive) ([ar5iv][16-17])
- **Productive "refinement loop" systems from ARC Prize writeups (2025 winners list)** (2025)
  **Driver:** need explore->verify->refine loops instead of single-shot inference
  **Outcome:** ARC Prize 2025 highlights evolutionary test-time compute and evolutionary program synthesis as dominant mechanisms. Note: Many are released as competition papers; the ARC Prize post is the cleanest index ([ARC Prize][16-18])
- **Boosting Performance on ARC via Perspective / Augmentations** (2025)
  **Driver:** invariance handling + robust selection among candidates
  **Outcome:** multi-view inference aggregation as an explicit compute algorithm ([Proceedings of Machine Learning Research][16-19])
- **ARC Is a Vision Problem! (Vision ARC / VARC)** (2025)
  **Driver:** language-oriented solver framing misses strong visual priors; needs per-task adaptation
  **Outcome:** ARC as image-to-image translation with ViT trained on ARC + test-time training for adaptation; explicitly reports ARC-1 performance ([arXiv][16-20])
- **Vector Symbolic Algebras for the Abstraction and Reasoning Corpus** (2025)
  **Driver:** pure neural pattern learning lacks explicit structured manipulation and sample-efficient abstraction
  **Outcome:** a System-1/System-2 neurosymbolic solver using Vector Symbolic Algebras to represent objects and guide program synthesis/search on ARC-AGI ([arXiv][16-21])
- **Reflection System for the Abstraction and Reasoning Corpus** (2025)
  **Driver:** LLMs alone are unreliable; need iterative critique/verification loops
  **Outcome:** LLM-as-judge/referee + program synthesis solver in a reflection loop (search + verification) ([OpenReview][16-22])
- **ARC Is a Vision Problem! + test-time training theme (indexed by ARC Prize 2025)** (2025)
  **Driver:** static inference; lack of per-task adaptation
  **Outcome:** multiple 2025 ARC Prize papers emphasize test-time refinement loops as the central algorithm ([ARC Prize][16-23])

[16-1]: https://arxiv.org/abs/2412.04604
[16-2]: https://dspace.mit.edu/bitstream/handle/1721.1/139305/Alford-salford-meng-eecs-2021-thesis.pdf
[16-3]: https://openreview.net/forum?id=0gOQeSHNX1
[16-4]: https://arxiv.org/abs/2401.07426
[16-5]: https://arxiv.org/abs/2411.02272
[16-6]: https://arxiv.org/abs/2411.07279
[16-7]: https://arxiv.org/abs/2411.17708
[16-8]: https://www.paulfletcherhill.com/mini-arc.pdf
[16-9]: https://arcprize.org/blog/arc-prize-2024-winners-technical-report
[16-10]: https://arxiv.org/abs/2412.04604
[16-11]: https://arxiv.org/html/2412.04604v2
[16-12]: https://arxiv.org/abs/2505.07859
[16-13]: https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/ARC_AGI_Without_Pretraining.pdf
[16-14]: https://openreview.net/pdf?id=z4IG090qt2
[16-15]: https://arxiv.org/html/2505.08778v1
[16-16]: https://ar5iv.org/abs/2510.04871
[16-17]: https://ar5iv.org/abs/2509.04439
[16-18]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[16-19]: https://proceedings.mlr.press/v267/franzen25a.html
[16-20]: https://arxiv.org/abs/2511.14761
[16-21]: https://arxiv.org/abs/2511.08747
[16-22]: https://openreview.net/forum?id=kRFwzuv0ze
[16-23]: https://arcprize.org/blog/arc-prize-2025-results-analysis

- **On the Measure of Intelligence / ARC (Abstraction and Reasoning Corpus)** (2019)
  **Driver:** Rapid abstraction + recombination on novel tasks (few demonstrations, no training-set prep).
  **Outcome:** Benchmark framing (not a solver) that establishes ARC-AGI as a forcing function for computation beyond feedforward pattern matching. ([Source Missing][17-1])
- **Combining Induction and Transduction for Abstract Reasoning** (2024)
  **Driver:** Pure transduction is brittle; pure synthesis misses perceptual fuzziness.
  **Outcome:** A hybrid inference system combining program induction (synthesis) with direct transduction, covering complementary ARC failure modes. ([ARC Prize][17-2])
- **The Surprising Effectiveness of Test-Time Training for Abstract Reasoning** (2024)
  **Driver:** Static inference cannot adapt to a brand-new puzzle at test time.
  **Outcome:** Test-time training (TTT) treats gradient-based adaptation as the inference procedure per puzzle/task. ([ARC Prize][17-3])
- **Searching Latent Program Spaces** (2024)
  **Driver:** Discrete search alone is inefficient; direct prediction isn't reliable.
  **Outcome:** Latent-space search/optimization inside a model to discover task-solving programs, positioned as neither pure finetuning nor pure discrete search. ([ARC Prize][17-4])
- **The LLM ARChitect: Solving ARC-AGI Is a Matter of Perspective** (2024)
  **Driver:** Single-shot decoding is fragile under ARC's invariances.
  **Outcome:** Multi-view/perspective ensembling plus candidate selection as the core inference algorithm. ([ARC Prize][17-5])
- **Omni-ARC** (2024)
  **Driver:** Single mechanism doesn't cover ARC's diversity.
  **Outcome:** A system-of-systems approach (explicit mixture of methods; search/learn hybrids) aimed at ARC-AGI-1 generalization. ([ARC Prize][17-6])
- **Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models** (2024)
  **Driver:** "Just scale params" isn't the right axis; need compute procedures that work in small models.
  **Outcome:** Small-model ARC solvers emphasizing refinement/adaptation loops (mechanism-centric, not size-centric). ([ARC Prize][17-7])
- **Towards Efficient Neurally-Guided Program Induction for ARC-AGI** (2024)
  **Driver:** Naive enumeration/synthesis is too expensive.
  **Outcome:** Neurally-guided program search/induction (search shaped by learned priors). ([ARC Prize][17-8])
- **A 2D nGPT Model For ARC Prize** (2024)
  **Driver:** Robust reasoning over grid structure.
  **Outcome:** ARC solver built around a 2D-aware modeling stack; can be more "representation/architecture" than a clean compute primitive. Note: borderline (often drifts toward "2D domain lifting"). ([ARC Prize][17-9])
- **ARC Prize 2024: Technical Report** (2024)
  **Driver:** Standard deep learning "no adaptation at test time" fails on novelty.
  **Outcome:** Highlights three dominant compute-mechanism families for ARC-AGI-1: DL-guided program synthesis, TTT, and hybrids that combine synthesis + transduction; also calls out latent-space search as a distinct adaptation mechanism. ([Source Missing][17-10])
- **Less is More: Recursive Reasoning with Tiny Networks (TRM)** (2025)
  **Driver:** Stable multi-step reasoning with small models; one-pass inference is brittle.
  **Outcome:** A recursive refinement network with separate latent + answer states trained for multi-step improvement; reported on ARC-AGI-1 and ARC-AGI-2. ([ARC Prize][17-11])
- **Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR)** (2025)
  **Driver:** Reliable program induction requires search + learning, not just prompting.
  **Outcome:** Evolutionary program synthesis plus a loop that fine-tunes an LLM on its own search traces (self-improving refinement). ([ARC Prize][17-12])
- **ARC-AGI Without Pretraining (CompressARC)** (2025)
  **Driver:** Dependence on large pretraining; need per-puzzle learning-as-inference.
  **Outcome:** Per-task MDL/"neural code golf" with single-puzzle training (test-time optimization as the solver). ([ARC Prize][17-13])
- **Vector Symbolic Algebras for the Abstraction and Reasoning Corpus** (2025)
  **Driver:** Robust compositional manipulation/structured variable binding.
  **Outcome:** Vector-symbolic algebra machinery used as a compute substrate for ARC-style composition. ([ARC Prize][17-14])
- **From Parrots to Von Neumanns: How Evolutionary Test-Time Compute Achieved SOTA on ARC-AGI** (2025)
  **Driver:** One-shot inference doesn't do the explore->verify->refine loop needed for ARC-AGI-2.
  **Outcome:** Evolutionary Test-Time Compute with explicit program evolution driven by verification feedback during inference. ([ARC Prize][17-15])
- **Efficient Evolutionary Program Synthesis** (2025)
  **Driver:** Brittle single attempts; need scalable synthesis under tight budgets.
  **Outcome:** Per-task evolutionary synthesis + verification, with library/abstraction building to steer search. ([ARC Prize][17-16])
- **ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus** (2025)
  **Driver:** One-pass inference struggles to produce emergent structured transformations.
  **Outcome:** Neural cellular automata with iterative local update dynamics as the computation. ([ARC Prize][17-17])
- **ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory** (2025)
  **Driver:** No persistent accumulation of reusable abstractions across puzzles.
  **Outcome:** Lifelong external memory for storing/retrieving compositional reasoning components. ([ARC Prize][17-18])
- **ARC-AGI is a Vision Problem!** (2025)
  **Driver:** Mismatch between modality/inductive bias and ARC grid structure; needs per-task adaptation.
  **Outcome:** Solver framing emphasizing vision-first modeling + adaptation/refinement rather than pure text/program routes. ([ARC Prize][17-19])
- **Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective** (2025)
  **Driver:** Single-view decoding is unstable; lacks robust invariance handling.
  **Outcome:** Product-of-experts/multi-augmentation aggregation as the inference algorithm. ([ARC Prize][17-20])
- **Exploring the combination of search and learn for the ARC25 challenge** (2025)
  **Driver:** Neither pure search nor pure learning generalizes reliably.
  **Outcome:** Explicit search-learn hybrid loops for ARC-AGI-2. ([ARC Prize][17-21])
- **Beyond Brute Force: A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2** (2025)
  **Driver:** Brute force and direct mapping both fail on compositional novelty.
  **Outcome:** An explicit neuro-symbolic compute pipeline (structured program-like reasoning with learned components). ([ARC Prize][17-22])
- **Test-time Adaptation of Tiny Recursive Models** (2025)
  **Driver:** Fixed recursive models still need task-specific compute under ARC-AGI-2 constraints.
  **Outcome:** Test-time adaptation loop specialized to TRM-style recursive solvers. ([ARC Prize][17-23])
- **Rethinking Visual Intelligence: Insights from Video Pretraining** (2025)
  **Driver:** Better perceptual priors to support reasoning on ARC-like tasks.
  **Outcome:** Uses pretraining regime as the lever; sometimes less "compute mechanism" than the others, but included as an ARC Prize 2025 honorable mention. Note: borderline (can drift toward representation/pretraining). ([ARC Prize][17-24])
- **Don't throw the baby out with the bathwater: How and why deep learning for ARC** (2025)
  **Driver:** Dismissing deep learning misses the key: how it's used at inference.
  **Outcome:** Argues for deep learning paired with adaptation/refinement procedures (TTT/TTC style) as the core algorithmic shift. ([ARC Prize][17-25])
- **NVARC solution to ARC-AGI-2 2025** (2025)
  **Driver:** Need solver pipelines that explicitly optimize per-task under constraints.
  **Outcome:** Documented as the ARC Prize 2025 top-score solution paper (system-level refinement + components). ([ARC Prize][17-26])
- **NVARC** (2025)
  **Driver:** Single-pass inference; insufficient per-task adaptation under ARC-AGI-2 novelty.
  **Outcome:** A contest-constrained ensemble combining an Architects-style test-time-trained model with TRM-based components (explicit refinement pipeline). ([ARC Prize][17-27])
- **the ARChitects** (2025)
  **Driver:** Robust reasoning requires iterative improvement + perspective scoring.
  **Outcome:** A "2D-aware" masked/diffusion LLM with recursive self-refinement and perspective-based scoring (system-level refinement loop). ([ARC Prize][17-28])
- **MindsAI** (2025)
  **Driver:** Needs strong test-time compute orchestration under hard constraints.
  **Outcome:** Engineered TTT/TTFT pipeline with augmentation ensembles and other test-time controls (compute allocation as the solver). ([ARC Prize][17-29])

[17-1]: MISSING
[17-2]: https://arcprize.org/competitions/2024/
[17-3]: https://arcprize.org/competitions/2024/
[17-4]: https://arcprize.org/competitions/2024/
[17-5]: https://arcprize.org/competitions/2024/
[17-6]: https://arcprize.org/competitions/2024/
[17-7]: https://arcprize.org/competitions/2024/
[17-8]: https://arcprize.org/competitions/2024/
[17-9]: https://arcprize.org/competitions/2024/
[17-10]: MISSING
[17-11]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-12]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-13]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-14]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-15]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-16]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-17]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-18]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-19]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-20]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-21]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-22]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-23]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-24]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-25]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-26]: https://arcprize.org/blog/arc-prize-2025-results-analysis
[17-27]: https://arcprize.org/competitions/2025/
[17-28]: https://arcprize.org/competitions/2025/
[17-29]: https://arcprize.org/competitions/2025/

- **ImageNet: A Large-Scale Hierarchical Image Database** (2009)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([image-net.org][18-1])
- **Microsoft COCO: Common Objects in Context** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-2])
- **SQuAD: 100,000+ Questions for Machine Comprehension of Text** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-3])
- **Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-4])
- **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-5])
- **GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-6])
- **On the Measure of Intelligence** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-7])
- **DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-8])
- **HellaSwag: Can a Machine Really Finish Your Sentence?** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-9])
- **SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-10])
- **Measuring Massive Multitask Language Understanding (MMLU)** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-11])
- **RealToxicityPrompts: Evaluating Neural Toxic Degeneration in Language Models** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-12])
- **Measuring Mathematical Problem Solving With the MATH Dataset** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-13])
- **Training Verifiers to Solve Math Word Problems (GSM8K)** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-14])
- **TruthfulQA: Measuring How Models Mimic Human Falsehoods** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-15])
- **Evaluating Large Language Models Trained on Code (HumanEval)** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-16])
- **LAION-5B: An open large-scale dataset for training next generation image-text models** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([LAION][18-17])
- **Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (ScienceQA)** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-18])
- **Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models (BIG-bench)** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-19])
- **Holistic Evaluation of Language Models (HELM)** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-20])
- **LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-21])
- **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-22])
- **Instruction-Following Evaluation for Large Language Models (IFEval)** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-23])
- **Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-24])
- **MMBench: Is Your Multi-modal Model an All-around Player?** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-25])
- **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-26])
- **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-27])
- **GPQA: A Graduate-Level Google-Proof Q&A Benchmark** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-28])
- **RULER: What's the Real Context Size of Your Long-Context Language Models?** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-29])
- **Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-30])
- **Length-Controlled AlpacaEval** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-31])
- **SWE-bench** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ICLR Proceedings][18-32])
- **A Survey on Evaluation of Large Language Models** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACM Digital Library][18-33])
- **MT-Bench-101** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][18-34])
- **LTD-Bench: Evaluating Large Language Models by Letting Them Draw** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][18-35])
- **TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Evolution** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-36])
- **ONERULER: Benchmarking multilingual long-context language models** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][18-37])

[18-1]: https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf
[18-2]: https://arxiv.org/abs/1405.0312
[18-3]: https://arxiv.org/abs/1606.05250
[18-4]: https://arxiv.org/abs/1803.05457
[18-5]: https://arxiv.org/abs/1809.09600
[18-6]: https://arxiv.org/abs/1804.07461
[18-7]: https://arxiv.org/abs/1911.01547
[18-8]: https://arxiv.org/abs/1903.00161
[18-9]: https://arxiv.org/abs/1905.07830
[18-10]: https://arxiv.org/abs/1905.00537
[18-11]: https://arxiv.org/abs/2009.03300
[18-12]: https://arxiv.org/abs/2009.11462
[18-13]: https://arxiv.org/abs/2103.03874
[18-14]: https://arxiv.org/abs/2110.14168
[18-15]: https://arxiv.org/abs/2109.07958
[18-16]: https://arxiv.org/abs/2107.03374
[18-17]: https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/
[18-18]: https://arxiv.org/abs/2209.09513
[18-19]: https://arxiv.org/abs/2206.04615
[18-20]: https://arxiv.org/abs/2211.09110
[18-21]: https://arxiv.org/abs/2308.14508
[18-22]: https://arxiv.org/abs/2310.06770
[18-23]: https://arxiv.org/abs/2311.07911
[18-24]: https://arxiv.org/abs/2306.05685
[18-25]: https://arxiv.org/abs/2307.06281
[18-26]: https://arxiv.org/abs/2310.02255
[18-27]: https://arxiv.org/abs/2311.16502
[18-28]: https://arxiv.org/abs/2311.12022
[18-29]: https://arxiv.org/abs/2404.06654
[18-30]: https://arxiv.org/pdf/2403.04132
[18-31]: https://arxiv.org/abs/2404.04475
[18-32]: https://proceedings.iclr.cc/paper_files/paper/2024/file/edac78c3e300629acfe6cbe9ca88fb84-Paper-Conference.pdf
[18-33]: https://dl.acm.org/doi/full/10.1145/3641289
[18-34]: https://aclanthology.org/2024.acl-long.401.pdf
[18-35]: https://openreview.net/forum?id=TG5rvKyEbu
[18-36]: https://arxiv.org/html/2506.18421v1
[18-37]: https://arxiv.org/abs/2503.01996

- **VQA: Visual Question Answering** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-1])
- **VQA v2.0 ("Making the V in VQA Matter")** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-2])
- **CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CVF Open Access][19-3])
- **VizWiz Grand Challenge: Answering Visual Questions from Blind People** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-4])
- **VCR: Visual Commonsense Reasoning ("From Recognition to Cognition...")** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-5])
- **NLVR2: A Corpus for Reasoning about Natural Language Grounded in Photographs** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([lil.nlp.cornell.edu][19-6])
- **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-7])
- **OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CVF Open Access][19-8])
- **TextVQA: Towards VQA Models That Can Read** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-9])
- **DocVQA: A Dataset for VQA on Document Images** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-10])
- **ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-11])
- **ScienceQA ("Learn to Explain...")** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-12])
- **A-OKVQA: A Benchmark for VQA Using World Knowledge** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-13])
- **MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-14])
- **MMBench: Is Your Multi-modal Model an All-around Player?** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-15])
- **MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-16])
- **MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-17])
- **POPE (Polling-based Object Probing Evaluation) / "Evaluating Object Hallucination in Large Vision-Language Models"** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-18])
- **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-19])
- **SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CVF Open Access][19-20])
- **HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-21])
- **MHaluBench (from "Unified Hallucination Detection...")** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][19-22])
- **MM-Vet v2** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-23])
- **MVBench: A Comprehensive Multi-modal Video Understanding Benchmark** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CVF Open Access][19-24])
- **Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-25])
- **MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding and Reasoning Benchmark** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][19-26])
- **VisioMath: Benchmarking Figure-based Mathematical Reasoning in LMMs** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-27])
- **ChartQAPro: A More Diverse and Challenging Benchmark for Real-World Chart QA** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][19-28])

[19-1]: https://arxiv.org/abs/1505.00468
[19-2]: https://arxiv.org/abs/1612.00837
[19-3]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Johnson_CLEVR_A_Diagnostic_CVPR_2017_paper.pdf
[19-4]: https://arxiv.org/abs/1802.08218
[19-5]: https://arxiv.org/abs/1811.10830
[19-6]: https://lil.nlp.cornell.edu/nlvr/
[19-7]: https://arxiv.org/abs/1902.09506
[19-8]: https://openaccess.thecvf.com/content_CVPR_2019/papers/Marino_OK-VQA_A_Visual_Question_Answering_Benchmark_Requiring_External_Knowledge_CVPR_2019_paper.pdf
[19-9]: https://arxiv.org/abs/1904.08920
[19-10]: https://arxiv.org/abs/2007.00398
[19-11]: https://arxiv.org/abs/2203.10244
[19-12]: https://arxiv.org/abs/2209.09513
[19-13]: https://arxiv.org/abs/2206.01718
[19-14]: https://arxiv.org/abs/2306.13394
[19-15]: https://arxiv.org/abs/2307.06281
[19-16]: https://arxiv.org/abs/2308.02490
[19-17]: https://arxiv.org/abs/2310.02255
[19-18]: https://arxiv.org/abs/2305.10355
[19-19]: https://arxiv.org/abs/2311.16502
[19-20]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_SEED-Bench_Benchmarking_Multimodal_Large_Language_Models_CVPR_2024_paper.pdf
[19-21]: https://arxiv.org/abs/2310.14566
[19-22]: https://aclanthology.org/2024.acl-long.178.pdf
[19-23]: https://arxiv.org/abs/2408.00765
[19-24]: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_MVBench_A_Comprehensive_Multi-modal_Video_Understanding_Benchmark_CVPR_2024_paper.pdf
[19-25]: https://arxiv.org/abs/2405.21075
[19-26]: https://aclanthology.org/2025.acl-long.736.pdf
[19-27]: https://arxiv.org/abs/2506.06727
[19-28]: https://arxiv.org/html/2504.05506v1

- **LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens** (2024)
  **Driver:** naive RoPE scaling/long-context extension causes non-uniformity and mismatch as length grows.
  **Outcome:** a RoPE-extension scheme with non-uniform/optimized adjustments as a PE-side fix for very long contexts. ([arXiv][2-1])
- **HiRoPE: Length Extrapolation for Code Models Using Hierarchical Rotary Position Embedding** (2024)
  **Driver:** flat RoPE does not respect code's hierarchical structure; long-code completion stresses standard position handling.
  **Outcome:** hierarchical RoPE (structure-aware rotary positions) for length extrapolation in code LMs. ([ACL Anthology][2-2])
- **Resonance RoPE: Improving Context Length Generalization of Large Language Models** (2024)
  **Driver:** RoPE interpolation/remapping approaches still show distribution shift and generalization loss at longer lengths.
  **Outcome:** a refined RoPE modification ("resonance" shaping) aimed at stronger length generalization. ([ACL Anthology][2-3])
- **Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs** (2024)
  **Driver:** NoPE and standard PE fail beyond an effective range, but PE can be extended with the right design.
  **Outcome:** weave PE + Stair PE and the Mesa-Extrapolation method (chunk/triangular attention + weave PE) for extrapolation. ([arXiv][2-4])
- **Rotary Position Embedding for Vision Transformer** (2024)
  **Driver:** RoPE is underexplored in vision; 2D usage is non-trivial and common implementations have gaps.
  **Outcome:** practical 2D RoPE implementations/variants with analysis and design recommendations for ViTs. ([arXiv][2-5])
- **Length Generalization of Causal Transformers without Position Encoding** (2024)
  **Driver:** explicit PEs are not the only way; NoPE can generalize but has failure modes tied to attention distribution.
  **Outcome:** head temperature tuning to expand NoPE's usable context. Note: borderline (NoPE-centric, but still 'position mechanism'). ([arXiv][2-6])
- **HARPE: Head-Adaptive Rotary Position Encoding** (2025)
  **Driver:** multi-stage long-context training plus manual RoPE base tuning is brittle; single-stage training with one big base can be suboptimal.
  **Outcome:** per-head RoPE base frequencies trained toward target context length. ([ACL Anthology][2-7])
- **ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices** (2025)
  **Driver:** RoPE's fixed, hand-defined rotation matrices limit the transformation space and flexibility/robustness.
  **Outcome:** trainable commuting angle matrices that generalize RoPE while preserving offset-consistent behavior. ([arXiv][2-8])
- **CABLE: Context-aware Biases for Length Extrapolation** (2025)
  **Driver:** static/additive RPE biases or fixed PE schemes are too rigid; long-context behavior benefits from context-conditioning.
  **Outcome:** context-conditioned positional bias scores added to attention logits. ([ACL Anthology][2-9])
- **Wavelet-based Positional Representation for Long Context** (2025)
  **Driver:** ALiBi behaves like windowed attention and struggles with deep dependencies due to receptive-field limits.
  **Outcome:** a wavelet-transform-based positional representation to capture multiple scales without restricting attention's field. ([OpenReview][2-10])
- **Understanding the RoPE Extensions of Long-Context LLMs** (2025)
  **Driver:** many RoPE extensions are used in practice without a clear attention-perspective explanation.
  **Outcome:** analysis that maps the space of RoPE extensions rather than proposing a new PE. Note: borderline / analysis. ([ACL Anthology][2-11])
- **CoPE: A Lightweight Complex Positional Encoding** (2025)
  **Driver:** traditional PE methods have limitations for long sequences, with long-term decay or incompatibilities.
  **Outcome:** complex-valued encoding (real=content, imag=position) with phase-aware attention in early layers. ([arXiv][2-12])
- **TAPA: Positional Encoding via Token-Aware Phase Attention** (2025)
  **Driver:** RoPE introduces an intrinsic distance-dependent bias limiting long-context modeling; many RoPE extensions are post-hoc retuning.
  **Outcome:** a learnable phase function in attention (token-aware phase attention) as a new PE mechanism for extrapolation. ([arXiv][2-13])

[2-1]: https://arxiv.org/pdf/2402.13753
[2-2]: https://aclanthology.org/2024.acl-long.735/
[2-3]: https://aclanthology.org/2024.findings-acl.32.pdf
[2-4]: https://arxiv.org/abs/2410.15859
[2-5]: https://arxiv.org/abs/2403.13298
[2-6]: https://arxiv.org/abs/2404.12224
[2-7]: https://aclanthology.org/2025.coling-main.326/
[2-8]: https://arxiv.org/abs/2506.03737
[2-9]: https://aclanthology.org/2025.emnlp-main.1545.pdf
[2-10]: https://openreview.net/forum?id=OhauMUNW8T
[2-11]: https://aclanthology.org/2025.coling-main.600.pdf
[2-12]: https://arxiv.org/abs/2508.18308
[2-13]: https://arxiv.org/abs/2509.12635

- **Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks (bAbI)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-1])
- **Constructing Datasets for Multi-hop Reading Comprehension (WikiHop / MedHop; QAngaroo)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-2])
- **HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-3])
- **Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-4])
- **OpenBookQA: Can a Suit of Armor Conduct Electricity?** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-5])
- **CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-6])
- **DROP: Discrete Reasoning Over Paragraphs** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-7])
- **CLUTRR: A Diagnostic Benchmark for Inductive Reasoning from Text** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-8])
- **2WikiMultiHopQA: Comprehensive Evaluation of Reasoning Steps** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-9])
- **ReClor: A Reading Comprehension Dataset Requiring Logical Reasoning** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-10])
- **LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([IJCAI][20-11])
- **StrategyQA: A Benchmark with Implicit Reasoning Strategies** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-12])
- **GSM8K: Training Verifiers to Solve Math Word Problems** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-13])
- **MATH: Measuring Mathematical Problem Solving** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-14])
- **SVAMP: Simple Variations on Arithmetic Math Word Problems** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-15])
- **EntailmentBank: Explaining Answers with Entailment Trees** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-16])
- **ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-17])
- **miniF2F: a cross-system benchmark for formal Olympiad-level mathematics** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-18])
- **MuSiQue: Multihop Questions via Single-hop Question Composition** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-19])
- **FOLIO: Natural Language Reasoning with First-Order Logic** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-20])
- **BIG-Bench Hard (BBH): Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-21])
- **LogiQA 2.0 - An Improved Dataset for Logical Reasoning in NLU** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([FRC Chang][20-22])
- **AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-23])
- **GAIA: a benchmark for General AI Assistants** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-24])
- **LiveBench: A Challenging, Contamination-Free LLM Benchmark** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-25])
- **MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][20-26])
- **BIG-Bench Extra Hard (BBEH)** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][20-27])

[20-1]: https://arxiv.org/abs/1502.05698
[20-2]: https://arxiv.org/abs/1710.06481
[20-3]: https://aclanthology.org/D18-1259/
[20-4]: https://arxiv.org/abs/1803.05457
[20-5]: https://arxiv.org/abs/1809.02789
[20-6]: https://aclanthology.org/N19-1421/
[20-7]: https://aclanthology.org/N19-1246/
[20-8]: https://aclanthology.org/D19-1458.pdf
[20-9]: https://aclanthology.org/2020.coling-main.580/
[20-10]: https://arxiv.org/abs/2002.04326
[20-11]: https://www.ijcai.org/proceedings/2020/501
[20-12]: https://aclanthology.org/2021.tacl-1.21/
[20-13]: https://arxiv.org/abs/2110.14168
[20-14]: https://arxiv.org/abs/2103.03874
[20-15]: https://aclanthology.org/2021.naacl-main.168/
[20-16]: https://aclanthology.org/2021.emnlp-main.585.pdf
[20-17]: https://aclanthology.org/2021.findings-acl.317.pdf
[20-18]: https://arxiv.org/abs/2109.00110
[20-19]: https://arxiv.org/abs/2108.00573
[20-20]: https://arxiv.org/abs/2209.00840
[20-21]: https://arxiv.org/abs/2210.09261
[20-22]: https://frcchang.github.io/pub/An%20Improved%20Dataset%20for%20Logical%20Reasoning%20in%20Natural%20Language%20Understanding.pdf
[20-23]: https://arxiv.org/pdf/2304.06364
[20-24]: https://arxiv.org/abs/2311.12983
[20-25]: https://arxiv.org/abs/2406.19314
[20-26]: https://arxiv.org/abs/2406.01574
[20-27]: https://aclanthology.org/2025.acl-long.1285/

- **ListOps: A Diagnostic Dataset for Latent Tree Learning** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-1])
- **Analysing Mathematical Reasoning Abilities of Neural Models** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-2])
- **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-3])
- **Grokking Modular Arithmetic** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-4])
- **Machine Learning for Modular Multiplication** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-5])
- **Grokking Modular Polynomials** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-6])
- **Teaching Transformers Modular Arithmetic at Scale** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-7])
- **Transformers Can Do Arithmetic with the Right Embeddings** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. Note: not modular-only, but very similar "algorithmic arithmetic" infrastructure. ([NeurIPS Proceedings][21-8])
- **Repeated Examples Help Learn Arithmetic** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][21-9])
- **Learning Modular Exponentiation with Transformers** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][21-10])

[21-1]: https://arxiv.org/abs/1804.06028
[21-2]: https://arxiv.org/abs/1904.01557
[21-3]: https://arxiv.org/pdf/2201.02177
[21-4]: https://arxiv.org/abs/2301.02679
[21-5]: https://arxiv.org/html/2402.19254v1
[21-6]: https://arxiv.org/abs/2406.03495
[21-7]: https://www.arxiv.org/abs/2410.03569v1
[21-8]: https://proceedings.neurips.cc/paper_files/paper/2024/file/c35986bc1ee29b31c1011481b77fe540-Paper-Conference.pdf
[21-9]: https://openreview.net/pdf?id=qoUHqnE6A0
[21-10]: https://arxiv.org/html/2506.23679v1

- **Learning to Execute** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][22-1])
- **ListOps: A Diagnostic Dataset for Latent Tree Learning** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACL Anthology][22-2])
- **Analysing Mathematical Reasoning Abilities of Neural Models** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][22-3])
- **Long Range Arena (LRA): A Benchmark for Efficient Transformers** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][22-4])
- **How Can Self-Attention Networks Recognize Dyck-n Languages?** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. Note: math-adjacent (stack/counter structure). ([ACL Anthology][22-5])
- **Transformers Can Do Arithmetic with the Right Embeddings** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Proceedings][22-6])
- **A Benchmark to Evaluate Fundamental Numerical Abilities** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][22-7])
- **Arithmetic-Bench: Evaluating Multi-Step Reasoning in LLMs through Basic Arithmetic Operations** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][22-8])
- **MathClean: A Benchmark for Synthetic Mathematical Data** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. Note: meta-infrastructure (synthetic-math data quality). ([arXiv][22-9])

[22-1]: https://arxiv.org/abs/1410.4615
[22-2]: https://aclanthology.org/N18-4013/
[22-3]: https://arxiv.org/pdf/1904.01557
[22-4]: https://arxiv.org/abs/2011.04006
[22-5]: https://aclanthology.org/2020.findings-emnlp.384.pdf
[22-6]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/c35986bc1ee29b31c1011481b77fe540-Abstract-Conference.html
[22-7]: https://arxiv.org/pdf/2502.11075
[22-8]: https://openreview.net/forum?id=ae6bKeffGZ
[22-9]: https://arxiv.org/html/2502.19058v1

- **Automating String Processing in Spreadsheets using Input-Output Examples (FlashFill)** (2011)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Microsoft][23-1])
- **Learning Semantic String Transformations from Examples** (2012)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([VLDB][23-2])
- **Syntax-Guided Synthesis (SyGuS)** (2013)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CIS Penn][23-3])
- **General Program Synthesis Benchmark Suite (PSB / "PSB1")** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Hamilton College][23-4])
- **BlinkFill: Semi-supervised Programming By Example for Syntactic String Transformations** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([VLDB][23-5])
- **DeepCoder: Learning to Write Programs** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-6])
- **RobustFill: Neural Program Learning under Noisy I/O** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-7])
- **Neural Program Meta-Induction** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Papers][23-8])
- **Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-9])
- **On the Measure of Intelligence (ARC)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-10])
- **PSB2: The Second Program Synthesis Benchmark Suite** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-11])
- **SRBench (Symbolic Regression Benchmarks)** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Cava Lab][23-12])
- **SRBench++: principled benchmarking of symbolic regression** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([PMC][23-13])
- **H-ARC (Human-ARC): A Comprehensive Behavioral Dataset for the Abstraction and Reasoning Corpus** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][23-14])
- **SRBench update ("next generation" SRBench)** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][23-15])

[23-1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/synasc12.pdf
[23-2]: https://vldb.org/pvldb/vol5/p740_rishabhsingh_vldb2012.pdf
[23-3]: https://www.cis.upenn.edu/~alur/SyGuS13.pdf
[23-4]: https://www.cs.hamilton.edu/~thelmuth/Pubs/2015-GECCO-benchmark-suite.pdf
[23-5]: https://www.vldb.org/pvldb/vol9/p816-singh.pdf
[23-6]: https://arxiv.org/abs/1611.01989
[23-7]: https://arxiv.org/abs/1703.07469
[23-8]: https://papers.nips.cc/paper/6803-neural-program-meta-induction
[23-9]: https://arxiv.org/abs/1805.04276
[23-10]: https://arxiv.org/abs/1911.01547
[23-11]: https://arxiv.org/pdf/2106.06086
[23-12]: https://cavalab.github.io/symbolic-regression/
[23-13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/
[23-14]: https://www.nature.com/articles/s41597-025-05687-1
[23-15]: https://arxiv.org/html/2505.03977v1

- **Automating String Processing in Spreadsheets Using Input-Output Examples (FlashFill)** (2011)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Microsoft][24-1])
- **Learning Semantic String Transformations from Examples** (2012)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][24-2])
- **Syntax-Guided Synthesis (SyGuS) + SyGuS-Comp 2014** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([SyGuS][24-3])
- **General Program Synthesis Benchmark Suite (PSB / PSB1)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Hamilton College][24-4])
- **BlinkFill: Semi-supervised Programming By Example for Syntactic String Transformations** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([VLDB][24-5])
- **SyGuS-Comp 2016 Results/Benchmarks (PBE track)** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][24-6])
- **DeepCoder: Learning to Write Programs** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][24-7])
- **RobustFill: Neural Program Learning under Noisy I/O** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([GitHub][24-8])
- **Dataset for Learning Karel Programs (MSR Karel Dataset)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([msr-redmond.github.io][24-9])
- **Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][24-10])
- **On the Measure of Intelligence (ARC)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Cava Lab][24-11])
- **SyGuS-Comp "PBE tracks" mature (e.g., PBE-Strings, PBE-BV)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([sygus-org.github.io][24-12])
- **AI Feynman** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Science][24-13])
- **DreamCoder** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. Note: borderline as "benchmark," but very relevant. ([Courses at UW][24-14])
- **PROGRES: A large-scale benchmark for few-shot program induction and synthesis** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Proceedings of Machine Learning Research][24-15])
- **PSB2: The Second Program Synthesis Benchmark Suite** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][24-16])
- **SRBench** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([GitHub][24-17])
- **SRBench++: principled benchmarking of symbolic regression** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([PubMed][24-18])
- **Towards the Next Generation of Symbolic Regression Benchmarks** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][24-19])
- **PSB / PSB2** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Hamilton College][24-20])
- **DeepCoder DSL tasks** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][24-21])
- **SyGuS PBE-BV / PBE-Strings** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([sygus-org.github.io][24-22])
- **Karel dataset** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([msr-redmond.github.io][24-23])
- **ARC** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][24-24])
- **Symbolic regression benchmarks (AI Feynman, SRBench)** (Year unknown)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Science][24-25])

[24-1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/popl11-synthesis.pdf
[24-2]: MISSING
[24-3]: https://sygus.org/comp/2014/
[24-4]: https://www.cs.hamilton.edu/~thelmuth/Pubs/2015-GECCO-benchmark-suite.pdf
[24-5]: https://www.vldb.org/pvldb/vol9/p816-singh.pdf
[24-6]: https://arxiv.org/pdf/1611.07627
[24-7]: https://openreview.net/pdf?id=ByldLrqlx
[24-8]: https://github.com/thelmuth/program-synthesis-benchmark-datasets
[24-9]: https://msr-redmond.github.io/karel-dataset/
[24-10]: https://arxiv.org/abs/1805.04276
[24-11]: https://cavalab.org/srbench/
[24-12]: https://sygus-org.github.io/comp/2019/
[24-13]: https://www.science.org/doi/10.1126/sciadv.aay2631
[24-14]: https://courses.cs.washington.edu/courses/cse599j1/22sp/papers/dreamcoder.pdf
[24-15]: https://proceedings.mlr.press/v139/alet21a.html
[24-16]: https://arxiv.org/pdf/2106.06086
[24-17]: https://github.com/cavalab/srbench
[24-18]: https://pubmed.ncbi.nlm.nih.gov/40761553/
[24-19]: https://arxiv.org/html/2505.03977v1
[24-20]: https://www.cs.hamilton.edu/~thelmuth/Pubs/2015-GECCO-benchmark-suite.pdf
[24-21]: https://openreview.net/pdf?id=ByldLrqlx
[24-22]: https://sygus-org.github.io/comp/2019/
[24-23]: https://msr-redmond.github.io/karel-dataset/
[24-24]: MISSING
[24-25]: https://www.science.org/doi/10.1126/sciadv.aay2631

- **Component-based Synthesis Applied to Bitvector Programs** (2010)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Microsoft][25-1])
- **Syntax-Guided Synthesis (SyGuS)** (2013)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CIS UPenn][25-2])
- **ICFP Programming Competition** (2013)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([DSpace][25-3])
- **SyGuS-Comp'15 Results/Analysis** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Rishabh Singh][25-4])
- **SyGuS-Comp 2016: Results and Analysis** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([DSpace][25-5])
- **SyGuS-Comp track definitions (PBE-BV)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([sygus.org][25-6])
- **SyGuS-Comp 2017: Results and Analysis** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][25-7])
- **Accelerating Search-Based Program Synthesis using Learned Probabilistic Models (Euphony)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CIS UPenn][25-8])
- **Search-based Program Synthesis** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CIS UPenn][25-9])
- **NAPS: Natural Program Synthesis Dataset** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][25-10])
- **Solving Programming Tasks from Description and Examples** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][25-11])
- **Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE)** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([CSE UCSD][25-12])
- **Latent Execution for Neural Program Synthesis (LaSynth)** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Proceedings][25-13])
- **Enhanced Enumeration Techniques for Syntax-Guided Synthesis** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACM Digital Library][25-14])

[25-1]: https://www.microsoft.com/en-us/research/wp-content/uploads/2010/02/bv.pdf
[25-2]: https://www.cis.upenn.edu/~alur/SyGuS13.pdf
[25-3]: https://dspace.mit.edu/bitstream/handle/1721.1/137904/1611.07627.pdf?isAllowed=y&sequence=2
[25-4]: https://rishabhmit.bitbucket.io/papers/synt15.pdf
[25-5]: https://dspace.mit.edu/bitstream/handle/1721.1/137904/1611.07627.pdf?isAllowed=y&sequence=2
[25-6]: https://sygus.org/comp/2017/
[25-7]: https://arxiv.org/pdf/1711.11438
[25-8]: https://www.cis.upenn.edu/~alur/PLDI18.pdf
[25-9]: https://www.cis.upenn.edu/~alur/CACM18.pdf
[25-10]: https://arxiv.org/pdf/1807.03168
[25-11]: https://arxiv.org/pdf/1802.04335
[25-12]: https://cseweb.ucsd.edu/~hpeleg/probe-oopsla20.pdf
[25-13]: https://proceedings.neurips.cc/paper/2021/file/ba3c95c2962d3aab2f6e667932daa3c5-Paper.pdf
[25-14]: https://dl.acm.org/doi/10.1145/3632913

- **The Perceptron** (1957)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-1])
- **A Theory of the Learnable (PAC Learning)** (1984)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Wikipedia][26-2])
- **Learning Representations by Back-Propagating Errors** (1986)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][26-3])
- **Support-Vector Networks** (1995)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Springer][26-4])
- **Long Short-Term Memory (LSTM)** (1997)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Institute of Bioinformatics][26-5])
- **A Fast Learning Algorithm for Deep Belief Nets** (2006)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-6])
- **ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)** (2012)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-7])
- **word2vec (Skip-gram / CBOW)** (2013)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-8])
- **Dropout: A Simple Way to Prevent Neural Networks from Overfitting** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Journal of Machine Learning Research][26-9])
- **Adam: A Method for Stochastic Optimization** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-10])
- **Generative Adversarial Networks (GANs)** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-11])
- **Auto-Encoding Variational Bayes (VAE)** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-12])
- **Batch Normalization** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-13])
- **Deep Residual Learning (ResNet)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-14])
- **Layer Normalization** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-15])
- **Attention Is All You Need** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-16])
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-17])
- **Megatron-LM / large-scale transformer training** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-18])
- **The Lottery Ticket Hypothesis** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-19])
- **Scaling Laws for Neural Language Models** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-20])
- **Language Models are Few-Shot Learners (GPT-3)** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-21])
- **Sharpness-Aware Minimization (SAM)** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-22])
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-23])
- **LoRA: Low-Rank Adaptation of Large Language Models** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-24])
- **Training Compute-Optimal Large Language Models (Chinchilla)** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-25])
- **Training language models to follow instructions with human feedback (InstructGPT / RLHF pipeline)** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-26])
- **Constitutional AI** (2022)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-27])
- **Direct Preference Optimization (DPO)** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][26-28])
- **QLoRA** (2023)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-29])
- **FlashAttention-2 / efficient attention kernels** (2024)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Source Missing][26-30])

[26-1]: MISSING
[26-2]: https://en.wikipedia.org/wiki/Probably_approximately_correct_learning
[26-3]: https://www.nature.com/articles/323533a0
[26-4]: https://link.springer.com/article/10.1007/BF00994018
[26-5]: https://www.bioinf.jku.at/publications/older/2604.pdf
[26-6]: MISSING
[26-7]: MISSING
[26-8]: MISSING
[26-9]: https://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
[26-10]: https://arxiv.org/abs/1412.6980
[26-11]: MISSING
[26-12]: MISSING
[26-13]: https://arxiv.org/abs/1502.03167
[26-14]: https://arxiv.org/abs/1512.03385
[26-15]: https://arxiv.org/abs/1607.06450
[26-16]: https://arxiv.org/abs/1706.03762
[26-17]: MISSING
[26-18]: MISSING
[26-19]: MISSING
[26-20]: https://arxiv.org/abs/2001.08361
[26-21]: MISSING
[26-22]: MISSING
[26-23]: https://arxiv.org/abs/1910.02054
[26-24]: https://arxiv.org/abs/2106.09685
[26-25]: https://arxiv.org/abs/2203.15556
[26-26]: https://arxiv.org/abs/2203.02155
[26-27]: MISSING
[26-28]: https://arxiv.org/abs/2305.18290
[26-29]: MISSING
[26-30]: MISSING

- **A Markovian Decision Process** (1957)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([JSTOR][27-1])
- **Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems** (1983)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([incompleteideas.net][27-2])
- **Learning to Predict by the Methods of Temporal Differences** (1988)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([incompleteideas.net][27-3])
- **Dyna, an Integrated Architecture for Learning, Planning, and Reacting** (1991)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ACM Digital Library][27-4])
- **Technical Note: Q-learning** (1992)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Springer][27-5])
- **Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)** (1992)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Springer][27-6])
- **TD-Gammon, a Self-Teaching Backgammon Program, Achieves Master-Level Play** (1992)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([AAAI][27-7])
- **On-line Q-learning Using Connectionist Systems** (1994)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ResearchGate][27-8])
- **A Framework for Temporal Abstraction in Reinforcement Learning** (1999)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ScienceDirect][27-9])
- **Actor-Critic Algorithms** (1999)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Papers][27-10])
- **Human-level Control through Deep Reinforcement Learning (DQN)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Stanford University][27-11])
- **Trust Region Policy Optimization (TRPO)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-12])
- **Prioritized Experience Replay** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-13])
- **Continuous Control with Deep Reinforcement Learning (DDPG)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-14])
- **Asynchronous Methods for Deep Reinforcement Learning (A3C)** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-15])
- **Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo)** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][27-16])
- **Proximal Policy Optimization (PPO)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-17])
- **A Distributional Perspective on Reinforcement Learning** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-18])
- **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-19])
- **IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-20])
- **Soft Actor-Critic (SAC)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-21])
- **Rainbow: Combining Improvements in Deep Reinforcement Learning** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-22])
- **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][27-23])
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][27-24])

[27-1]: https://www.jstor.org/stable/24900506
[27-2]: https://incompleteideas.net/papers/barto-sutton-anderson-83.pdf
[27-3]: https://incompleteideas.net/papers/sutton-88-with-erratum.pdf
[27-4]: https://dl.acm.org/doi/abs/10.1145/122344.122377
[27-5]: https://link.springer.com/article/10.1023/A%3A1022676722315
[27-6]: https://link.springer.com/article/10.1007/BF00992696
[27-7]: https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-003.pdf
[27-8]: https://www.researchgate.net/profile/Mahesan-Niranjan/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems/links/5438d5db0cf204cab1d6db0f/On-Line-Q-Learning-Using-Connectionist-Systems.pdf
[27-9]: https://www.sciencedirect.com/science/article/pii/S0004370299000521
[27-10]: https://papers.nips.cc/paper/1786-actor-critic-algorithms
[27-11]: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
[27-12]: https://arxiv.org/abs/1502.05477
[27-13]: https://arxiv.org/abs/1511.05952
[27-14]: https://arxiv.org/abs/1509.02971
[27-15]: https://arxiv.org/abs/1602.01783
[27-16]: https://www.nature.com/articles/nature16961
[27-17]: https://arxiv.org/abs/1707.06347
[27-18]: https://arxiv.org/abs/1707.06887
[27-19]: https://arxiv.org/abs/1712.01815
[27-20]: https://arxiv.org/abs/1802.01561
[27-21]: https://arxiv.org/abs/1812.05905
[27-22]: https://arxiv.org/abs/1710.02298
[27-23]: https://www.nature.com/articles/s41586-020-03051-4
[27-24]: https://arxiv.org/abs/2501.12948

- **Policy Gradient Methods for Reinforcement Learning with Function Approximation** (2000)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Papers][28-1])
- **A Natural Policy Gradient** (2001)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([NeurIPS Papers][28-2])
- **Approximately Optimal Approximate Reinforcement Learning** (2002)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([People at EECS][28-3])
- **Least-Squares Policy Iteration (LSPI)** (2003)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Journal of Machine Learning Research][28-4])
- **Tree-Based Batch Mode Reinforcement Learning** (2005)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Journal of Machine Learning Research][28-5])
- **Neural Fitted Q Iteration (NFQ)** (2005)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Springer][28-6])
- **Natural Actor-Critic** (2008)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([ScienceDirect][28-7])
- **Relative Entropy Policy Search (REPS)** (2010)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([AAAI Open Access][28-8])
- **Deterministic Policy Gradient Algorithms** (2014)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Proceedings of Machine Learning Research][28-9])
- **High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-10])
- **Trust Region Policy Optimization (TRPO)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-11])
- **Prioritized Experience Replay** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-12])
- **Human-level Control through Deep Reinforcement Learning (DQN)** (2015)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Stanford University][28-13])
- **Deep Reinforcement Learning with Double Q-learning (Double DQN)** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-14])
- **Safe and Efficient Off-Policy Reinforcement Learning (Retrace(lambda))** (2016)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-15])
- **Proximal Policy Optimization (PPO)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-16])
- **A Distributional Perspective on Reinforcement Learning** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-17])
- **Combining Improvements in Deep Reinforcement Learning (Rainbow)** (2017)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-18])
- **IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-19])
- **Maximum a Posteriori Policy Optimisation (MPO)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-20])
- **Distributed Prioritized Experience Replay (Ape-X)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-21])
- **Soft Actor-Critic (SAC)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-22])
- **Twin Delayed DDPG (TD3)** (2018)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-23])
- **Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][28-24])
- **Dream to Control: Learning Behaviors by Latent Imagination (Dreamer)** (2019)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([OpenReview][28-25])
- **Agent57: Outperforming the Atari Human Benchmark** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Proceedings of Machine Learning Research][28-26])
- **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)** (2020)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][28-27])
- **Muesli: Combining Improvements in Policy Optimization** (2021)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Proceedings of Machine Learning Research][28-28])
- **Mastering diverse control tasks through world models (DreamerV3)** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([Nature][28-29])
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (2025)
  **Driver:** Not stated in source.
  **Outcome:** Not stated in source. ([arXiv][28-30])

[28-1]: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation
[28-2]: https://papers.neurips.cc/paper/2073-a-natural-policy-gradient.pdf
[28-3]: https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf
[28-4]: https://www.jmlr.org/papers/v4/lagoudakis03a.html
[28-5]: https://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf
[28-6]: https://link.springer.com/chapter/10.1007/11564096_32
[28-7]: https://www.sciencedirect.com/science/article/pii/S0925231208000532
[28-8]: https://ojs.aaai.org/index.php/AAAI/article/view/7727
[28-9]: https://proceedings.mlr.press/v32/silver14.html
[28-10]: https://arxiv.org/abs/1506.02438
[28-11]: https://arxiv.org/abs/1502.05477
[28-12]: https://arxiv.org/abs/1511.05952
[28-13]: https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
[28-14]: https://arxiv.org/abs/1509.06461
[28-15]: https://arxiv.org/abs/1606.02647
[28-16]: https://arxiv.org/abs/1707.06347
[28-17]: https://arxiv.org/abs/1707.06887
[28-18]: https://arxiv.org/abs/1710.02298
[28-19]: https://arxiv.org/abs/1802.01561
[28-20]: https://arxiv.org/abs/1806.06920
[28-21]: https://arxiv.org/abs/1803.00933
[28-22]: https://arxiv.org/abs/1801.01290
[28-23]: https://arxiv.org/pdf/1802.09477
[28-24]: https://openreview.net/forum?id=r1lyTjAqYX
[28-25]: https://openreview.net/forum?id=S1lOTC4tDS
[28-26]: https://proceedings.mlr.press/v119/badia20a.html
[28-27]: https://www.nature.com/articles/s41586-020-03051-4
[28-28]: https://proceedings.mlr.press/v139/hessel21a/hessel21a.pdf
[28-29]: https://www.nature.com/articles/s41586-025-08744-2
[28-30]: https://arxiv.org/abs/2501.12948

- **Logic Tensor Networks: Deep Learning and Logical Reasoning from Data and Knowledge** (2016)
  **Driver:** neural nets don't naturally enforce first-order logic constraints during learning/inference.
  **Outcome:** introduce Real Logic (truth values in [0,1]) and optimize neural parameters under differentiable logic satisfiability-style losses (constraints as soft differentiable objectives). ([arXiv][29-1])
- **End-to-End Differentiable Proving** (2017)
  **Driver:** dense embedding models struggle with explicit multi-hop symbolic proof structure (unification, rule application).
  **Outcome:** Neural Theorem Provers (NTPs): continuous relaxation of Prolog-style backward chaining, replacing unification with differentiable similarity kernels over embeddings. ([arXiv][29-2])
- **Learning a SAT Solver from Single-Bit Supervision (NeuroSAT)** (2018)
  **Driver:** classical SAT solvers have strong heuristics but aren't learnable end-to-end; neural nets lack structured clause-variable reasoning.
  **Outcome:** a message-passing network over the SAT factor graph that can run more iterations at test time to solve harder instances (iterative constraint propagation learned from data). ([arXiv][29-3])
- **Learning Explanatory Rules from Noisy Data / Differentiable ILP (dILP)** (2018)
  **Driver:** symbolic ILP is brittle to noise; neural models don't induce explicit logic programs.
  **Outcome:** make ILP end-to-end differentiable by relaxing rule selection/unification into soft, trainable weights, learning logic programs via gradient descent. ([Jair][29-4])
- **DeepProbLog: Neural Probabilistic Logic Programming** (2018)
  **Driver:** integrate neural perception with probabilistic logical inference in one training loop.
  **Outcome:** extend ProbLog with neural predicates, enabling joint learning where symbolic reasoning (probabilistic logic) and neural modules co-train. ([arXiv][29-5])
- **SATNet: Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver** (2019)
  **Driver:** standard nets can't reliably satisfy global discrete constraints; discrete SAT is non-differentiable.
  **Outcome:** a differentiable (smoothed) MAXSAT solver layer (SDP/coordinate-descent flavored) that can be embedded into neural systems and trained end-to-end. ([arXiv][29-6])
- **OptNet: Differentiable Optimization as a Layer in Neural Networks** (2017)
  **Driver:** feedforward layers struggle to enforce hard global constraints that are naturally expressed as optimization problems.
  **Outcome:** embed a constrained QP solver as a differentiable layer via implicit differentiation (demonstrated on constraint-heavy tasks like mini-Sudoku). ([arXiv][29-7])
- **Differentiable Convex Optimization Layers (CVXPYLayers / DPP)** (2019)
  **Driver:** constraint satisfaction / optimization priors are hard to integrate broadly; custom differentiable solvers are brittle.
  **Outcome:** general method + tooling for differentiating through disciplined convex programs, turning a wide class of convex CSP-like modules into plug-in layers. ([arXiv][29-8])
- **Neural Logic Machines (NLM)** (2019)
  **Driver:** robust lifted logical reasoning (quantifiers, connectives) and generalization to larger object sets.
  **Outcome:** neural-symbolic blocks that implement differentiable approximations to logical operations over relations, enabling rule-like, size-generalizing reasoning. ([arXiv][29-9])
- **Guiding SAT Solvers with Unsat-Core Predictions (NeuroCore)** (2019)
  **Driver:** CDCL SAT solvers rely on hand-designed branching heuristics that can miss structure in hard industrial instances.
  **Outcome:** a neural module predicts unsat cores to bias variable branching inside a CDCL solver, tightly integrating learning into symbolic search. ([arXiv][29-10])
- **Goal-Aware Neural SAT Solver (QuerySAT / goal-aware guidance)** (2021)
  **Driver:** purely learned solvers can be inefficient; classical solvers lack learned guidance tuned to goal/structure.
  **Outcome:** goal-aware neural guidance integrated into SAT solving workflow (positioned as a learned improvement to SAT search/heuristics). ([arXiv][29-11])
- **diff-SAT - Sampling and Probabilistic Reasoning for SAT and Answer Set Programming** (2021)
  **Driver:** standard SAT/ASP doesn't natively optimize against probabilistic constraints or gradient-like objectives.
  **Outcome:** a SAT/ASP solver variant supporting differentiable / probabilistic constraints to steer solution sampling/optimization. ([arXiv][29-12])
- **DiffSAT: Differential MaxSAT Layer for SAT Solving** (2024)
  **Driver:** end-to-end pipelines still struggle with discrete satisfaction; existing neural SAT solvers often learn predictors rather than a differentiable solving mechanism.
  **Outcome:** a differential MaxSAT layer (with structured initialization) that enables progressive search toward satisfying assignments in a differentiable framework. ([CUHK Computer Science and Engineering][29-13])
- **Enhancing Modern SAT Solver With Machine Learning** (2025)
  **Driver:** practical recipe for reliably injecting ML into CDCL pipelines.
  **Outcome:** consolidates/extends the NeuroCore-style idea of learning-guided heuristics in modern SAT workflows. Note: borderline (more "integration & evaluation" than a new core solver). ([ACM Digital Library][29-14])

[29-1]: https://arxiv.org/abs/1606.04422
[29-2]: https://arxiv.org/abs/1705.11040
[29-3]: https://arxiv.org/abs/1802.03685
[29-4]: https://jair.org/index.php/jair/article/view/11172/26376
[29-5]: https://arxiv.org/abs/1805.10872
[29-6]: https://arxiv.org/pdf/1905.12149
[29-7]: https://arxiv.org/abs/1703.00443
[29-8]: https://arxiv.org/abs/1910.12430
[29-9]: https://arxiv.org/abs/1904.11694
[29-10]: https://arxiv.org/pdf/1903.04671
[29-11]: https://arxiv.org/pdf/2106.07162
[29-12]: https://arxiv.org/abs/2101.00589
[29-13]: https://www.cse.cuhk.edu.hk/~byu/papers/C237-ICCAD2024-DiffSAT.pdf
[29-14]: https://dl.acm.org/doi/full/10.1145/3716368.3735251

- **Convolutional Sequence to Sequence Learning** (2017)
  **Driver:** early seq models needed an explicit mechanism to inject order.
  **Outcome:** uses trainable absolute positional embeddings as a learned table. ([arXiv PDF][3-1])
- **Attention Is All You Need** (2017)
  **Driver:** self-attention is permutation-invariant; without position, token order is lost.
  **Outcome:** proposes sinusoidal absolute positional encoding with a closed-form rule for extrapolation. ([arXiv PDF][3-2])
- **Self-Attention with Relative Position Representations** (2018)
  **Driver:** absolute PE added to inputs is not the only way; attention can be position-aware directly.
  **Outcome:** injects relative position representations into attention via key/value-side modifications. ([arXiv PDF][3-3])
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)
  **Driver:** transformers still require positional encodings; the simplest form is learned absolute tables.
  **Outcome:** uses trainable absolute positional embeddings. ([arXiv PDF][3-4])
- **Improving Language Understanding by Generative Pre-Training** (2018)
  **Driver:** learned absolute PE is straightforward but not extrapolatable by default.
  **Outcome:** uses trainable absolute positional embeddings (learned lookup). ([OpenAI PDF][3-5])
- **Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context** (2019)
  **Driver:** absolute PE clashes with segment-level recurrence and memory, so positions do not stay consistent across segments.
  **Outcome:** adds relative-position terms to attention scores with trainable vectors (u, v) and a sinusoidal-based relative signal (XLNet style). ([ACL Anthology][3-6])
- **XLNet: Generalized Autoregressive Pretraining for Language Understanding** (2019)
  **Driver:** Transformer-XL relative-position attention needed adoption in a strong pretraining regime.
  **Outcome:** adopts Transformer-XL relative positional attention formulation. ([arXiv PDF][3-7])
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** (2019)
  **Driver:** decouple input content and position interactions to simplify positional handling.
  **Outcome:** learned relative position bias added to attention logits with relative-position bucketing (T5 style). ([arXiv PDF][3-8])
- **Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER)** (2020)
  **Driver:** fixed-form absolute PE is restrictive; a learnable generator with extrapolation-friendly behavior is needed.
  **Outcome:** models positional encoding via a continuous dynamical system / neural ODE. ([arXiv PDF][3-9])
- **Rethinking Positional Encoding in Language Pre-training (TUPE)** (2020)
  **Driver:** adding word and position embeddings mixes heterogeneous correlations; [CLS] should not be treated like ordinary positions.
  **Outcome:** untied positional encoding with separate parameterizations for word-context and positional correlations, then combined. ([arXiv PDF][3-10])
- **DeBERTa: Decoding-enhanced BERT with Disentangled Attention** (2020)
  **Driver:** position and content interactions should be structured rather than entangled.
  **Outcome:** disentangled attention separating content and position vectors and their interactions (DeBERTa style). ([arXiv PDF][3-11])
- **How Much Position Information Do Convolutional Neural Networks Encode?** (2020)
  **Driver:** CNNs appear position-aware without explicit PE, so the source of position information is unclear.
  **Outcome:** argues absolute position leaks via padding and boundaries (zero-padding leakage). ([OpenReview PDF][3-12])
- **Encoding Word Order in Complex Embeddings** (2019)
  **Driver:** standard position embeddings capture absolute positions but not richer order relations; real-valued PE is not the only route.
  **Outcome:** complex-number encoding with a complex-valued network pipeline to exploit phase and order structure. ([arXiv PDF][3-13])

[3-1]: https://arxiv.org/pdf/1705.03122
[3-2]: https://arxiv.org/pdf/1706.03762
[3-3]: https://arxiv.org/pdf/1803.02155
[3-4]: https://arxiv.org/pdf/1810.04805
[3-5]: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
[3-6]: https://aclanthology.org/P19-1285.pdf
[3-7]: https://arxiv.org/pdf/1906.08237
[3-8]: https://arxiv.org/pdf/1910.10683
[3-9]: https://arxiv.org/pdf/2003.09229
[3-10]: https://arxiv.org/pdf/2006.15595
[3-11]: https://arxiv.org/pdf/2006.03654
[3-12]: https://openreview.net/pdf/2267055f8221e283014aba7ef46092ba93ff450f.pdf
[3-13]: https://arxiv.org/pdf/1912.12333

- **Methode generale pour la resolution des systemes d'equations simultanees** (1847)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Lemarechal (historical PDF)][30-1])
- **Early gradient/variation method note often attributed to Hadamard's "methode" discussions** (1908)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Schmidhuber (historical pointer)][30-2])
- **An Example of Statistical Investigation of the Text "Eugene Onegin" Concerning the Connection of Samples in Chains** (1913)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PDF][30-3])
- **A Mathematical Theory of Communication** (1948)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PDF][30-4])
- **Translation (Weaver Memorandum)** (1949)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PDF][30-5])
- **Algoritmin kumulatiivinen pyoristysvirhe yksittaisten pyoristysvirheiden Taylor-kehitelmana (The representation of the cumulative rounding error ...)** (1970)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PDF][30-6])
- **A Neural Probabilistic Language Model** (2003)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([JMLR PDF][30-7])
- **Efficient Estimation of Word Representations in Vector Space** (2013)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([arXiv PDF][30-8])
- **Adam: A Method for Stochastic Optimization** (2014)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([arXiv PDF][30-9])
- **Attention Is All You Need** (2017)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([NeurIPS PDF][30-10])

[30-1]: https://ems.press/content/book-chapter-files/27368?nt=1
[30-2]: https://people.idsia.ch/~juergen/who-invented-backpropagation-2014.html
[30-3]: https://nessie.ilab.sztaki.hu/~kornai/2021/KalmanCL/markov_1913.pdf
[30-4]: https://ia803209.us.archive.org/27/items/bstj27-3-379/bstj27-3-379_text.pdf
[30-5]: https://www.mt-archive.net/50/Weaver-1949.pdf
[30-6]: https://www.idsia.ch/~juergen/linnainmaa1970thesis.pdf
[30-7]: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
[30-8]: https://arxiv.org/pdf/1301.3781
[30-9]: https://arxiv.org/pdf/1412.6980
[30-10]: https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf
- **Certain Factors Affecting Telegraph Speed** (1924)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Monoskop][31-1])
- **Transmission of Information** (1928)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Monoskop][31-2])
- **Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?** (1935)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([APS Link][31-3])
- **A Mathematical Theory of Communication** (1948)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Harvard Math People][31-4])
- **Cognitive Maps in Rats and Men** (1948)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-5])
- **The Organization of Behavior** (1949)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Deng Fanxin][31-6])
- **The Magical Number Seven, Plus or Minus Two** (1956)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-7])
- **Receptive fields of single neurones in the cat's striate cortex** (1959)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PMC][31-8])
- **Receptive Fields, Binocular Interaction, and Functional Architecture in the Cat's Visual Cortex** (1962)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Gatsby][31-9])
- **Information Theory and Statistical Mechanics** (1957)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Bayes WUSTL][31-10])
- **The hippocampus as a spatial map** (1971)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-11])
- **Broadcast Channels** (1972)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Information Systems Laboratory][31-12])
- **Working Memory** (1974)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([ScienceDirect][31-13])
- **Vision: A Computational Investigation into the Human Representation and Processing of Visual Information** (1982)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([MIT Press Direct][31-14])
- **Neural networks and physical systems with emergent collective computational abilities** (1982)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PNAS][31-15])
- **Predictive coding in the visual cortex: a functional interpretation** (1999)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-16])
- **An Integrative Theory of Prefrontal Cortex Function** (2001)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-17])
- **Microstructure of a spatial map in the entorhinal cortex** (2005)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Nature][31-18])
- **A free energy principle for the brain** (2006)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-19])
- **The free-energy principle: a unified brain theory?** (2010)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([University of Alabama at Birmingham][31-20])
- **Working Memory: Theories, Models, and Controversies** (2012)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][31-21])
- **A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex** (2019)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Frontiers][31-22])

[31-1]: https://monoskop.org/images/9/9f/Nyquist_Harry_1924_Certain_Factors_Affecting_Telegraph_Speed.pdf
[31-2]: https://monoskop.org/images/a/a6/Hartley_Ralph_VL_1928_Transmission_of_Information.pdf
[31-3]: https://link.aps.org/doi/10.1103/PhysRev.47.777
[31-4]: https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf
[31-5]: https://pubmed.ncbi.nlm.nih.gov/18870876/
[31-6]: https://www.dengfanxin.cn/wp-content/uploads/2016/03/1949Hebb.pdf
[31-7]: https://pubmed.ncbi.nlm.nih.gov/13310704/
[31-8]: https://pmc.ncbi.nlm.nih.gov/articles/PMC1363130/
[31-9]: https://www.gatsby.ucl.ac.uk/~lmatthey/teaching/tn1/additional/systems/JPhysiol-1962-Hubel-106-54.pdf
[31-10]: https://bayes.wustl.edu/etj/articles/theory.1.pdf
[31-11]: https://pubmed.ncbi.nlm.nih.gov/5124915/
[31-12]: https://isl.stanford.edu/~cover/papers/transIT/0002cove.pdf
[31-13]: https://www.sciencedirect.com/science/article/pii/S0079742108604521
[31-14]: https://direct.mit.edu/books/monograph/3299/VisionA-Computational-Investigation-into-the-Human
[31-15]: https://www.pnas.org/doi/10.1073/pnas.79.8.2554
[31-16]: https://pubmed.ncbi.nlm.nih.gov/10195184/
[31-17]: https://pubmed.ncbi.nlm.nih.gov/11283309/
[31-18]: https://www.nature.com/articles/nature03721
[31-19]: https://pubmed.ncbi.nlm.nih.gov/17097864/
[31-20]: https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf
[31-21]: https://pubmed.ncbi.nlm.nih.gov/21961947/
[31-22]: https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full

- **The Theory of Dynamic Programming** (1954)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([RAND Corporation][32-1])
- **Dynamic Programming** (1957)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Gwern][32-2])
- **A New Approach to Linear Filtering and Prediction Problems** (1960)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([ASME Digital Collection][32-3])
- **New Results in Linear Filtering and Prediction Theory** (1961)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([ASME Digital Collection][32-4])
- **Deterministic Nonperiodic Flow** (1963)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([American Meteorological Society Journals][32-5])
- **Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons** (1972)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-6])
- **Adaptive pattern classification and universal recoding: II. Feedback, expectation, olfaction, illusions** (1976)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-7])
- **Synergetics: An Introduction** (1977)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Google Books][32-8])
- **A Feature-Integration Theory of Attention** (1980)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-9])
- **Two cortical visual systems** (1982)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([CNS NYU][32-10])
- **Cellular and circuit basis of working memory in prefrontal cortex of nonhuman primates** (1990)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-11])
- **Separate visual pathways for perception and action** (1992)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-12])
- **Cellular basis of working memory** (1995)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-13])
- **Predictive coding in the visual cortex** (1999)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PMC][32-14])
- **Optimal feedback control as a theory of motor coordination** (2002)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-15])
- **Optimality principles in sensorimotor control** (2004)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-16])
- **Dynamic dopamine modulation in the basal ganglia** (2005)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-17])
- **Making Working Memory Work: A Computational Model of Learning in the Prefrontal Cortex and Basal Ganglia** (2006)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([PubMed][32-18])
- **Toward an executive without a homunculus: computational models of the prefrontal cortex/basal ganglia system** (2007)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Royal Society Publishing][32-19])
- **Unified Theories of Cognition** (1990)
  **Driver:** Not stated in source
  **Outcome:** Not stated in source ([Harvard University Press][32-20])

[32-1]: https://www.rand.org/content/dam/rand/pubs/papers/2008/P550.pdf
[32-2]: https://gwern.net/doc/statistics/decision/1957-bellman-dynamicprogramming.pdf
[32-3]: https://asmedigitalcollection.asme.org/fluidsengineering/article/82/1/35/397706/A-New-Approach-to-Linear-Filtering-and-Prediction
[32-4]: https://asmedigitalcollection.asme.org/fluidsengineering/article/83/1/95/426820/New-Results-in-Linear-Filtering-and-Prediction
[32-5]: https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml
[32-6]: https://pubmed.ncbi.nlm.nih.gov/4332108/
[32-7]: https://pubmed.ncbi.nlm.nih.gov/963125/
[32-8]: https://books.google.com/books/about/Synergetics.html?id=KHn1CAAAQBAJ
[32-9]: https://pubmed.ncbi.nlm.nih.gov/7351125/
[32-10]: https://www.cns.nyu.edu/~tony/vns/readings/ungerleider-mishkin-1982.pdf
[32-11]: https://pubmed.ncbi.nlm.nih.gov/2094903/
[32-12]: https://pubmed.ncbi.nlm.nih.gov/1374953/
[32-13]: https://pubmed.ncbi.nlm.nih.gov/7695894/
[32-14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4311762/
[32-15]: https://pubmed.ncbi.nlm.nih.gov/12404008/
[32-16]: https://pubmed.ncbi.nlm.nih.gov/15332089/
[32-17]: https://pubmed.ncbi.nlm.nih.gov/15701239/
[32-18]: https://pubmed.ncbi.nlm.nih.gov/16378516/
[32-19]: https://royalsocietypublishing.org/doi/abs/10.1098/rstb.2007.2055
[32-20]: https://www.hup.harvard.edu/books/9780674921016

- **Attention Is All You Need** (2017)
  **Driver:** self-attention is permutation-invariant; order must be injected.
  **Outcome:** sinusoidal absolute positional encoding (closed-form; enables relative structure algebraically). ([PDF][4-1])
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)
  **Driver:** (as framed by Su) absolute PE is simple but typically not extrapolatable without tricks.
  **Outcome:** trainable absolute positional embeddings (learned lookup). ([PDF][4-2])
- **NEZHA: Neural Contextualized Representation for Chinese Language Understanding** (2019)
  **Driver:** (as used in Su's table) absolute PE is not ideal; relative PE can be a better inductive bias.
  **Outcome:** introduces/uses Functional Relative Positional Encoding in a BERT-like Chinese PLM. ([PDF][4-3])
- **CAIL2019-SCM: A Dataset of Similar Case Matching in Legal Domain** (2019)
  **Driver:** (implicit) long-text semantics need benchmarks where length matters.
  **Outcome:** provides a legal-domain similar-case matching dataset used as Su's long-text evaluation target. ([PDF][4-4])
- **Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention** (2021)
  **Driver:** vanilla attention is quadratic; need efficient approximations for long sequences.
  **Outcome:** approximates attention via Nystrom landmark-based low-rank reconstruction (linear-ish attention variant). ([PDF][4-5])
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021)
  **Driver:** sinusoidal absolute PE "almost" yields relative structure, but not cleanly; many relative schemes can't be used in linear attention because they act on the attention matrix.
  **Outcome:** RoPE: rotate Q/K by position-dependent orthogonal transforms so dot-products depend on relative offsets; enables a RoFormer model that extrapolates better to longer contexts and is compatible with linear-attention variants. ([PDF][4-6])

[4-1]: https://arxiv.org/pdf/1706.03762
[4-2]: https://arxiv.org/pdf/1810.04805
[4-3]: https://arxiv.org/pdf/1909.00204
[4-4]: https://arxiv.org/pdf/1911.08962
[4-5]: https://arxiv.org/pdf/2102.03902
[4-6]: https://arxiv.org/pdf/2104.09864

- **Attention Is All You Need** (2017)
  **Driver:** self-attention is permutation-invariant; order must be injected.
  **Outcome:** introduces the Transformer with sinusoidal absolute positional embeddings as the default solution. ([arXiv][5-1])
- **Self-Attention with Relative Position Representations** (2018)
  **Driver:** adding absolute position to inputs is not the only way; attention scoring can use relative distances.
  **Outcome:** integrates relative position representations into attention via key/value modifications. ([arXiv][5-2])
- **Music Transformer** (2018)
  **Driver:** long-range music structure stresses naive absolute position schemes.
  **Outcome:** uses a relative-position approach tailored for music sequence modeling. ([arXiv][5-3])
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** (2019)
  **Driver:** positional effects can be simplified; learn position influence as a bias term.
  **Outcome:** adds relative positional bias to attention logits with bucketing. ([arXiv][5-4])
- **Language Models are Few-Shot Learners (GPT-3)** (2020)
  **Driver:** scaling exposes practical issues in positional handling and long-context training choices.
  **Outcome:** large-scale autoregressive LM baseline using learned absolute positional embeddings in practice. ([arXiv][5-5])
- **Rethinking Attention with Performers** (2020)
  **Driver:** full softmax attention is quadratic; efficient variants cannot build the full NxN attention matrix.
  **Outcome:** Performer (FAVOR+) kernelized attention to approximate softmax efficiently. ([arXiv][5-6])
- **Rethinking Positional Encoding in Language Pre-training (TUPE)** (2020)
  **Driver:** standard token + position addition entangles correlations in suboptimal ways.
  **Outcome:** untied/decoupled treatment of positional effects in attention. ([arXiv][5-7])
- **Shortformer: Better Language Modeling using Shorter Inputs** (2020)
  **Driver:** absolute position injection can interact poorly with training setups and long contexts.
  **Outcome:** modifies where/how position information is introduced for better LM behavior under constrained contexts. ([arXiv][5-8])
- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision** (2020)
  **Driver:** standard LM supervision can be limited; preprocessing/training practices around contexts motivate alternatives.
  **Outcome:** adds visual grounding supervision ("vokens") to language learning. ([arXiv][5-9])
- **The Pile: An 800GB Dataset of Diverse Text for Language Modeling** (2021)
  **Driver:** positional encoding evaluations should be stress-tested on large, diverse corpora.
  **Outcome:** provides a large-scale dataset used in the post's billion-parameter experiments. ([arXiv][5-10])
- **Do Transformer Modifications Transfer Across Implementations and Applications?** (2021)
  **Driver:** many Transformer improvements do not transfer robustly across codebases and tasks.
  **Outcome:** systematic transferability study motivating robust architectural changes. ([arXiv][5-11])
- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (2021)
  **Driver:** large-scale training pipelines reshape how sequences/contexts are packed and consumed.
  **Outcome:** contrastive vision-language pretraining at scale. ([arXiv][5-12])
- **Efficient Large-Scale Language Model Training on GPU Clusters** (2021)
  **Driver:** scaling LMs introduces system constraints that influence sequence handling and positional schemes in practice.
  **Outcome:** large-scale distributed training methods (pipeline/tensor/data parallelism) supporting very large Transformers. ([arXiv][5-13])
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021)
  **Driver:** many relative position methods require the full attention matrix; sinusoidal absolute positional encoding only "sort of" behaves like relative.
  **Outcome:** RoPE rotates Q/K so dot-products depend on relative offsets; works in vanilla attention and adapts to efficient/linear-like settings. ([arXiv][5-14])

[5-1]: https://arxiv.org/pdf/1706.03762
[5-2]: https://arxiv.org/pdf/1803.02155
[5-3]: https://arxiv.org/pdf/1809.04281
[5-4]: https://arxiv.org/pdf/1910.10683
[5-5]: https://arxiv.org/pdf/2005.14165
[5-6]: https://arxiv.org/pdf/2009.14794
[5-7]: https://arxiv.org/pdf/2006.15595
[5-8]: https://arxiv.org/pdf/2012.15832
[5-9]: https://arxiv.org/pdf/2010.06775
[5-10]: https://arxiv.org/pdf/2101.00027
[5-11]: https://arxiv.org/pdf/2102.11972
[5-12]: https://arxiv.org/pdf/2103.00020
[5-13]: https://arxiv.org/pdf/2104.04473
[5-14]: https://arxiv.org/pdf/2104.09864

- **Attention Is All You Need** (2017)
  **Driver:** self-attention is permutation-invariant; order must be injected.
  **Outcome:** introduces the Transformer plus sinusoidal absolute positional encoding. ([PDF][6-1])
- **Self-Attention with Relative Position Representations** (2018)
  **Driver:** absolute positional encoding is not the only route; attention scoring can incorporate relative distance directly.
  **Outcome:** integrates relative position representations into attention. ([PDF][6-2])
- **Music Transformer** (2018)
  **Driver:** music's long-range structure stresses naive positional handling.
  **Outcome:** applies relative-position ideas for music sequence modeling, cited as a motivating multi-dimensional setting. ([PDF][6-3])
- **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)** (2019)
  **Driver:** content and position interactions can be simplified; position can act as a bias term.
  **Outcome:** adds relative positional bias directly to attention logits (with bucketing in practice). ([PDF][6-4])
- **Language Models are Few-Shot Learners (GPT-3)** (2020)
  **Driver:** scaling highlights practical long-context issues; learned absolute position embeddings can be brittle under context packing and truncation.
  **Outcome:** provides a large autoregressive LM baseline that uses learned absolute positional embeddings. ([PDF][6-5])
- **Rethinking Attention with Performers** (2020)
  **Driver:** full softmax attention is quadratic; many positional methods requiring an explicit N x N attention matrix do not fit efficient attention.
  **Outcome:** Performer (FAVOR+) kernelized attention approximates softmax efficiently. ([PDF][6-6])
- **Rethinking Positional Encoding in Language Pre-training (TUPE)** (2020)
  **Driver:** naive token + position addition can entangle correlations in a suboptimal way.
  **Outcome:** uses untied and decoupled positional treatment in attention. ([PDF][6-7])
- **Shortformer: Better Language Modeling using Shorter Inputs** (2020)
  **Driver:** training and inference setups make some positional choices inefficient or poorly behaved, motivating a robust alternative.
  **Outcome:** injects position by adding absolute position embeddings to queries and keys rather than token embeddings. ([PDF][6-8])
- **Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision** (2020)
  **Driver:** pure text-only self-supervision is limited; training practices and supervision shape what position mechanisms mean in practice.
  **Outcome:** adds visually grounded supervision ("vokens") mapped to language tokens. ([PDF][6-9])
- **The Pile: An 800GB Dataset of Diverse Text for Language Modeling** (2021)
  **Driver:** evaluation of positional encoding methods needs broad, diverse corpora at scale.
  **Outcome:** provides a large multi-source dataset used in large-model experiments. ([PDF][6-10])
- **Do Transformer Modifications Transfer Across Implementations and Applications?** (2021)
  **Driver:** many Transformer improvements do not transfer robustly across codebases and tasks.
  **Outcome:** systematic transferability study motivating robust architectural changes. ([PDF][6-11])
- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (2021)
  **Driver:** large-scale practice changes how representations are used and transferred.
  **Outcome:** large-scale contrastive vision-language pretraining (CLIP). ([PDF][6-12])
- **Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM** (2021)
  **Driver:** systems constraints drive practical sequence and context decisions; efficient training is part of the long-context story.
  **Outcome:** scalable distributed training with pipeline, tensor, and data parallelism plus schedules. ([PDF][6-13])
- **RoFormer: Enhanced Transformer with Rotary Position Embedding** (2021)
  **Driver:** many relative methods require explicit attention matrices; sinusoidal absolute positional encoding only partially yields relative structure.
  **Outcome:** RoPE rotates Q/K so dot-products depend on relative offsets and stays compatible with efficient attention variants. ([PDF][6-14])

[6-1]: https://arxiv.org/pdf/1706.03762
[6-2]: https://arxiv.org/pdf/1803.02155
[6-3]: https://arxiv.org/pdf/1809.04281
[6-4]: https://arxiv.org/pdf/1910.10683
[6-5]: https://arxiv.org/pdf/2005.14165
[6-6]: https://arxiv.org/pdf/2009.14794
[6-7]: https://arxiv.org/pdf/2006.15595
[6-8]: https://arxiv.org/pdf/2012.15832
[6-9]: https://arxiv.org/pdf/2010.06775
[6-10]: https://arxiv.org/pdf/2101.00027
[6-11]: https://arxiv.org/pdf/2102.11972
[6-12]: https://arxiv.org/pdf/2103.00020
[6-13]: https://arxiv.org/pdf/2104.04473
[6-14]: https://arxiv.org/pdf/2104.09864

- **Image Transformer** (2018)
  **Driver:** 1D Transformer sequence modeling for text
  **Outcome:** treats images as sequences for autoregressive image generation; introduces local 2D-aware attention restrictions to make attention practical on images. ([arXiv][7-1])
- **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** (2018)
  **Driver:** 1D ordered sequences -> set-structured data
  **Outcome:** transformer-style attention redesigned to be permutation-invariant (set encoder/aggregator) for set-like and unordered domains (incl. point sets as a motivating application). ([arXiv][7-2])
- **Generating Long Sequences with Sparse Transformers** (2019)
  **Driver:** quadratic-cost full attention that limits long sequences
  **Outcome:** sparse attention factorization enabling very long sequences, demonstrated on images/audio/text from raw bytes. ([arXiv][7-3])
- **Generative Pretraining from Pixels (iGPT)** (2020)
  **Driver:** 1D autoregressive Transformers for text
  **Outcome:** serialize 2D images into 1D pixel sequences and train a GPT-style model on pixels. ([OpenAI][7-4])
- **An Image Is Worth 16x16 Words (ViT)** (2020)
  **Driver:** Transformer encoders for 1D sequence classification
  **Outcome:** tokenize images into fixed-size patches (patch embedding sequence) and run a standard transformer encoder over patch tokens. ([arXiv][7-5])
- **End-to-End Object Detection with Transformers (DETR)** (2020)
  **Driver:** CNN pipeline + hand-crafted detection components
  **Outcome:** use transformer encoder-decoder on 2D image features plus learned object queries for set prediction (detection as set output). ([arXiv][7-6])
- **Stand-Alone Axial-Attention for Panoptic Segmentation (Axial-DeepLab)** (2020)
  **Driver:** expensive full 2D self-attention and CNN-dominant segmentation
  **Outcome:** factorize 2D attention into two 1D attentions along axes (row/column), enabling global-ish context at manageable cost. ([Department of Computer Science][7-7])
- **SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks** (2020)
  **Driver:** transformers that ignore 3D symmetry structure
  **Outcome:** redesign attention to be SE(3)-equivariant for 3D point clouds / 3D graphs (geometric domain). ([NeurIPS Proceedings][7-8])
- **SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers** (2020)
  **Driver:** FCN encoder-decoder segmentation paradigm
  **Outcome:** treat segmentation explicitly as seq2seq: image -> patch-token sequence -> transformer encoder + decoder for dense masks. ([arXiv][7-9])
- **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (2021)
  **Driver:** ViT-style "single-scale global attention" that's costly for high-res images
  **Outcome:** hierarchical backbone + windowed attention with shifting to scale to high-resolution 2D vision tasks (detection/segmentation). ([CVF Open Access][7-10])
- **Pyramid Vision Transformer (PVT)** (2021)
  **Driver:** "flat token grids" that do not match multi-scale needs of dense prediction
  **Outcome:** a pyramid / multiscale transformer backbone (spatial reduction attention) for dense prediction in 2D vision. ([arXiv][7-11])
- **SegFormer** (2021)
  **Driver:** heavyweight decoders and brittle scaling for segmentation
  **Outcome:** segmentation framework with hierarchical transformer encoder + lightweight MLP decoder (a "dense prediction ready" ViT variant). ([arXiv][7-12])
- **MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation** (2021)
  **Driver:** per-pixel classification framing for semantic segmentation
  **Outcome:** reframes segmentation as set prediction of masks (DETR-like set outputs) using transformer-style components. ([arXiv][7-13])
- **TimeSformer: Is Space-Time Attention All You Need for Video Understanding?** (2021)
  **Driver:** ViT for static images
  **Outcome:** extend patch tokens across time; propose factorized space-time attention variants. ([arXiv][7-14])
- **ViViT: A Video Vision Transformer** (2021)
  **Driver:** image transformers / 2D patch tokenization
  **Outcome:** spatiotemporal tokenization and efficient transformer variants by factorizing spatial vs temporal processing. ([CVF Open Access][7-15])
- **Multiscale Vision Transformers (MViT)** (2021)
  **Driver:** single-scale ViT video models that are expensive
  **Outcome:** multiscale hierarchy for video (token resolution downsampling + channel expansion). ([arXiv][7-16])
- **Video Swin Transformer** (2021)
  **Driver:** globally-attending video transformers
  **Outcome:** extend Swin's locality + hierarchy to spatiotemporal inputs (windowed attention in space-time). ([arXiv][7-17])
- **Graphormer: Do Transformers Really Perform Bad for Graph Representation?** (2021)
  **Driver:** vanilla transformers lacking graph structural inductive bias
  **Outcome:** transformer architecture augmented with graph structural encodings to operate on graph domains. ([arXiv][7-18])
- **Point Transformer** (2020)
  **Driver:** applying 1D attention to irregular 3D data
  **Outcome:** self-attention layers designed for point clouds (local neighborhoods + point features/coords). ([CVF Open Access][7-19])
- **PCT: Point Cloud Transformer** (2020)
  **Driver:** order sensitivity / irregular-domain difficulty in point clouds
  **Outcome:** transformer framework tailored to unordered point sets (point-cloud learning). ([arXiv][7-20])
- **AST: Audio Spectrogram Transformer** (2021)
  **Driver:** CNN-heavy audio pipelines
  **Outcome:** represent audio as spectrogram patches and apply a ViT-style transformer to the 2D time-frequency plane. ([arXiv][7-21])
- **Perceiver / Perceiver IO** (2021)
  **Driver:** transformers that scale poorly with very large inputs (e.g., huge 2D/3D arrays)
  **Outcome:** a general transformer-style architecture handling arbitrary structured inputs/outputs via latent bottlenecks and cross-attention. ([arXiv][7-22])
- **Flamingo: a Visual Language Model for Few-Shot Learning** (2022)
  **Driver:** pure-language transformers (1D text) and rigid multimodal fusion
  **Outcome:** introduces a Perceiver Resampler to compress variable-size image/video features into a small set of visual tokens, then interleave with language generation via cross-attention blocks. ([arXiv][7-23])
- **Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling** (2021)
  **Driver:** BERT-style pretraining in 1D text
  **Outcome:** port masked modeling + tokenization ideas to 3D point cloud patches with a point-cloud tokenizer and transformer backbone. ([arXiv][7-24])
- **Transformer-based 3D point cloud generation networks** (2023)
  **Driver:** earlier generative point cloud pipelines
  **Outcome:** transformer architectures for 3D point cloud generation in irregular domains. Note: borderline (domain adaptation is there, but often incremental vs earlier 3D transformer backbones). ([ACM Digital Library][7-25])
- **Graph Perceiver IO** (2025)
  **Driver:** Perceiver IO's generic recipe for structured inputs
  **Outcome:** instantiate Perceiver IO specifically for graph-structured inputs/outputs. Note: borderline (adapts a general nD transformer to graphs explicitly). ([ScienceDirect][7-26])
- **POS-BERT: Point cloud one-stage BERT pre-training** (2024)
  **Driver:** Point-BERT's two-stage tokenizer dependency
  **Outcome:** simplifies transformer pretraining for 3D point clouds (still "bring BERT to 3D"). Note: borderline (pretraining adaptation rather than core backbone invention). ([ScienceDirect][7-27])

[7-1]: https://arxiv.org/abs/1802.05751
[7-2]: https://arxiv.org/abs/1810.00825
[7-3]: https://arxiv.org/abs/1904.10509
[7-4]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf
[7-5]: https://arxiv.org/abs/2010.11929
[7-6]: https://arxiv.org/abs/2005.12872
[7-7]: https://www.cs.jhu.edu/~alanlab/Pubs20/wang2020axial.pdf
[7-8]: https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html
[7-9]: https://arxiv.org/abs/2012.15840
[7-10]: https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf
[7-11]: https://arxiv.org/abs/2102.12122
[7-12]: https://arxiv.org/abs/2105.15203
[7-13]: https://arxiv.org/abs/2107.06278
[7-14]: https://arxiv.org/abs/2102.05095
[7-15]: https://openaccess.thecvf.com/content/ICCV2021/papers/Arnab_ViViT_A_Video_Vision_Transformer_ICCV_2021_paper.pdf
[7-16]: https://arxiv.org/abs/2104.11227
[7-17]: https://arxiv.org/abs/2106.13230
[7-18]: https://arxiv.org/abs/2106.05234
[7-19]: https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf
[7-20]: https://arxiv.org/abs/2012.09688
[7-21]: https://arxiv.org/abs/2104.01778
[7-22]: https://arxiv.org/abs/2103.03206
[7-23]: https://arxiv.org/abs/2204.14198
[7-24]: https://arxiv.org/abs/2111.14819
[7-25]: https://dl.acm.org/doi/10.1145/3581783.3612226
[7-26]: https://www.sciencedirect.com/science/article/abs/pii/S0031320325005497
[7-27]: https://www.sciencedirect.com/science/article/abs/pii/S0957417423030658

- **Image Transformer** (2018)
  **Driver:** 1D self-attention sequence modeling (text)
  **Outcome:** recast images as sequences for likelihood-based generation; make 2D feasible via restricted/local self-attention neighborhoods rather than full global attention. ([arXiv][8-1])
- **Generating Long Sequences with Sparse Transformers** (2019)
  **Driver:** full attention's quadratic scaling (limits long sequences, including image bytes)
  **Outcome:** sparse/factorized attention patterns enabling extremely long sequences (used for modalities including images/bytes). ([arXiv][8-2])
- **Generative Pretraining from Pixels (iGPT)** (2020)
  **Driver:** 1D autoregressive Transformers for language
  **Outcome:** serialize 2D images into 1D pixel sequences and train GPT-style causal attention for image understanding/generation. ([arXiv][8-3])
- **An Image Is Worth 16x16 Words (ViT)** (2020)
  **Driver:** Transformers as primarily 1D NLP models; vision relying on CNN scaffolding
  **Outcome:** tokenize images into fixed-size patches and process a patch-token sequence with a standard transformer encoder. ([arXiv][8-4])
- **End-to-End Object Detection with Transformers (DETR)** (2020)
  **Driver:** CNN detection pipelines with hand-designed components (anchors, NMS, etc.)
  **Outcome:** transformer encoder-decoder with learned object queries; detection reframed as set prediction via attention. ([arXiv][8-5])
- **Stand-Alone Axial-Attention (Axial-DeepLab)** (2020)
  **Driver:** full 2D attention cost and CNN-centric segmentation
  **Outcome:** factorize 2D attention into 1D axial attentions (rows/cols) to scale global context for dense prediction. ([arXiv][8-6])
- **Swin Transformer** (2021)
  **Driver:** flat/global ViT that scales poorly to high-res and dense tasks
  **Outcome:** hierarchical vision transformer with shifted window attention (local attention + cross-window connectivity) designed for 2D scaling. ([arXiv][8-7])
- **Pyramid Vision Transformer (PVT)** (2021)
  **Driver:** difficulty porting Transformer to dense prediction with flat tokens
  **Outcome:** progressive pyramid + spatial-reduction attention to make transformer features usable for dense 2D prediction. ([arXiv][8-8])
- **CvT: Introducing Convolutions to Vision Transformers** (2021)
  **Driver:** pure ViT patch embedding limitations on local inductive bias and efficiency
  **Outcome:** add convolutional token embedding + convolutional projections inside transformer blocks (hybrid 2D inductive bias while keeping attention core). ([arXiv][8-9])
- **Tokens-to-Token ViT (T2T-ViT)** (2021)
  **Driver:** naive patch tokenization that discards local structure
  **Outcome:** progressive tokens-to-token re-structurization step that builds better 2D token sequences before attention. ([arXiv][8-10])
- **CoAtNet: Marrying Convolution and Attention** (2021)
  **Driver:** conv-only vs attention-only tradeoffs across data/scale
  **Outcome:** systematic stacking of conv + attention stages to scale image modeling while keeping attention as a core component. ([NeurIPS Proceedings][8-11])
- **DeiT: Data-efficient Image Transformers** (2021)
  **Driver:** ViT's dependence on massive pretraining for vision
  **Outcome:** distillation/token strategy enabling ViTs to work well on ImageNet-only training (still squarely in make transformers work for 2D images). Note: borderline (more training adaptation than domain adaptation). ([Proceedings of Machine Learning Research][8-12])
- **SETR: Rethinking Semantic Segmentation as Seq2Seq with Transformers** (2021)
  **Driver:** FCN encoder-decoder segmentation pipelines
  **Outcome:** treat segmentation as sequence-to-sequence: encode an image as a patch sequence using a pure transformer then decode to dense masks. ([arXiv][8-13])
- **SegFormer** (2021)
  **Driver:** heavy decoders and resolution sensitivity for transformer segmentation
  **Outcome:** hierarchical transformer encoder producing multiscale features + lightweight MLP decoder for dense 2D outputs. ([arXiv][8-14])
- **MaskFormer** (2021)
  **Driver:** per-pixel classification framing for segmentation
  **Outcome:** reframes segmentation as set prediction of masks using transformer-style decoding (DETR-like idea generalized to masks). ([CVF Open Access][8-15])
- **Deformable DETR** (2021)
  **Driver:** DETR's slow convergence and resolution limits from global attention on feature maps
  **Outcome:** replace dense attention with sparse sampling-based (deformable) attention for 2D feature maps. ([arXiv][8-16])
- **Taming Transformers for High-Resolution Image Synthesis (VQGAN + Transformer)** (2021)
  **Driver:** direct pixel-sequence transformers being inefficient for high-res
  **Outcome:** learn a discrete 2D codebook (VQGAN) then model token composition with an autoregressive transformer. ([arXiv][8-17])
- **Zero-Shot Text-to-Image Generation (DALL-E)** (2021)
  **Driver:** text-only autoregressive transformers
  **Outcome:** a transformer that autoregressively models a single stream of text tokens + image tokens (multimodal serialization of 2D images). ([arXiv][8-18])
- **ViTDet: Exploring Plain ViT Backbones for Object Detection** (2022)
  **Driver:** need to redesign hierarchical backbones for detection
  **Outcome:** shows plain ViT can be adapted to detection with minimal changes (simple pyramid + window attention blocks). ([arXiv][8-19])
- **Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation** (2022)
  **Driver:** separate architectures for semantic/instance/panoptic segmentation
  **Outcome:** a unified transformer mask architecture using masked attention to localize cross-attention within predicted mask regions. ([arXiv][8-20])
- **MaskGIT: Masked Generative Image Transformer** (2022)
  **Driver:** raster-scan autoregressive image token generation being slow/inefficient
  **Outcome:** treat images as token grids but generate via bidirectional transformer with iterative refinement (non-sequential decoding over 2D tokens). ([arXiv][8-21])
- **Masked Autoencoders Are Scalable Vision Learners (MAE)** (2022)
  **Driver:** transferring masked modeling from 1D language to 2D patch grids
  **Outcome:** asymmetric encoder-decoder over visible patch subsets; reconstruct masked patches for scalable 2D representation learning. Note: borderline (pretraining adaptation). ([arXiv][8-22])
- **BEiT: BERT Pre-Training of Image Transformers** (2022)
  **Driver:** bringing BERT-style masked modeling to image patch tokens
  **Outcome:** tokenize images into discrete visual tokens and run masked prediction with a transformer encoder over 2D patches. Note: borderline (pretraining adaptation). ([arXiv][8-23])
- **Scalable Diffusion Models with Transformers (DiT)** (2022)
  **Driver:** U-Net dominated diffusion backbones for images
  **Outcome:** replace U-Net with a ViT-like transformer operating on latent patches for diffusion image generation. ([arXiv][8-24])
- **Segment Anything (SAM)** (2023)
  **Driver:** task-specific segmentation pipelines
  **Outcome:** a transformer-centric system (ViT image encoder + prompt encoder + transformer mask decoder) enabling promptable segmentation over 2D images. ([CVF Open Access][8-25])
- **DiffusionDet: Diffusion Model for Object Detection** (2023)
  **Driver:** standard detection decoding paradigms
  **Outcome:** detection as a diffusion denoising process over boxes; uses transformer-style components to predict/refine sets. Note: borderline (detection via diffusion, but still attention-heavy systems). ([CVF Open Access][8-26])

[8-1]: https://arxiv.org/abs/1802.05751
[8-2]: https://arxiv.org/abs/2203.16527
[8-3]: https://arxiv.org/abs/2005.12872
[8-4]: https://arxiv.org/abs/2010.11929
[8-5]: https://arxiv.org/abs/2005.12872
[8-6]: https://arxiv.org/abs/2105.15203
[8-7]: https://arxiv.org/abs/2103.14030
[8-8]: https://arxiv.org/abs/2102.12122
[8-9]: https://arxiv.org/abs/2103.15808
[8-10]: https://arxiv.org/abs/2101.11986
[8-11]: https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html
[8-12]: https://proceedings.mlr.press/v139/touvron21a.html
[8-13]: https://arxiv.org/abs/2012.15840
[8-14]: https://arxiv.org/abs/2105.15203
[8-15]: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf
[8-16]: https://arxiv.org/abs/2010.04159
[8-17]: https://arxiv.org/abs/2012.09841
[8-18]: https://arxiv.org/abs/2102.12092
[8-19]: https://arxiv.org/abs/2203.16527
[8-20]: https://arxiv.org/abs/2112.01527
[8-21]: https://arxiv.org/abs/2202.04200
[8-22]: https://arxiv.org/abs/2111.06377
[8-23]: https://arxiv.org/abs/2106.08254
[8-24]: https://arxiv.org/abs/2212.09748
[8-25]: https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf
[8-26]: https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf

- **Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** (2018)
  **Driver:** 1D ordered-sequence Transformers for language
  **Outcome:** redesign attention blocks to be permutation-invariant (set encoders/aggregators), which naturally fits point sets (a common 3D representation). ([Source Missing][9-1])
- **SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks** (2020)
  **Driver:** generic attention that ignores 3D symmetry structure
  **Outcome:** make self-attention SE(3)-equivariant for 3D point clouds/3D graphs, so the model respects rotations and translations by construction. ([NeurIPS Proceedings][9-2])
- **PCT: Point Cloud Transformer** (2021)
  **Driver:** applying 1D attention to irregular, unordered point sets
  **Outcome:** a transformer-style framework for unstructured point clouds with offset-attention and point-set processing built around attention as the main operator. ([Springer][9-3])
- **Point Transformer** (2021)
  **Driver:** standard Transformers assuming token order/grid structure
  **Outcome:** define neighborhood-based self-attention layers for point clouds and build full 3D backbones for segmentation/part tasks. ([CVF Open Access][9-4])
- **Voxel Transformer for 3D Object Detection (VoTr)** (2021)
  **Driver:** voxel CNN backbones with limited receptive field for 3D detection
  **Outcome:** transform voxel features with sparse voxel self-attention to capture long-range 3D context efficiently (attention as the core backbone operator). ([arXiv][9-5])
- **3DETR: An End-to-End Transformer Model for 3D Object Detection** (2021)
  **Driver:** complex 3D detection pipelines with many hand-designed 3D operators
  **Outcome:** DETR-style set prediction for 3D point clouds: transformer encoder over points plus decoder producing 3D boxes from nonparametric queries. ([arXiv][9-6])
- **DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries** (2021)
  **Driver:** 2D-only detection from images/depth-heavy 3D pipelines
  **Outcome:** query objects in 3D space and use camera geometry to index multi-view 2D features (DETR-like transformer adapted to 3D reasoning). ([arXiv][9-7])
- **Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling** (2022)
  **Driver:** BERT-style masked modeling success in 1D language
  **Outcome:** tokenize point clouds into local 3D patches and pretrain a transformer with masked point modeling (BERT adapted to 3D sets/patches). ([arXiv][9-8])
- **Masked Autoencoders for Point Cloud Self-supervised Learning (Point-MAE)** (2022)
  **Driver:** MAE's 2D success and generic SSL not handling point-cloud properties well
  **Outcome:** redesign masked autoencoding around point cloud patching/masking challenges; transformer encoder/decoder tailored to irregular 3D data. ([arXiv][9-9])
- **BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers** (2022)
  **Driver:** 1D sequence Transformers and 2D perception pipelines that do not unify 3D space + time well
  **Outcome:** build a unified 3D perception representation (BEV grid) using spatiotemporal transformer attention; BEV queries attend across camera views and time. ([arXiv][9-10])
- **TransFusion: Robust LiDAR-Camera Fusion for 3D Object Detection with Transformers** (2022)
  **Driver:** brittle multi-sensor fusion for 3D detection
  **Outcome:** transformer-based fine-grained LiDAR-camera fusion for 3D detection (attention as the fusion mechanism). ([CVF Open Access][9-11])
- **VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion** (2023)
  **Driver:** BEV-only 3D perception and dense volumetric completion methods
  **Outcome:** predict complete 3D voxel semantics from images using sparse voxel queries and transformer attention over voxels/features. ([arXiv][9-12])
- **ViewFormer: NeRF-free Neural Rendering from Few Images Using Transformers** (2022)
  **Driver:** NeRF pipelines requiring many 3D samples + long optimization
  **Outcome:** a transformer-centric pipeline that maps multi-view context and query pose to novel views without explicit NeRF-style per-ray optimization (a transformerization of 3D view synthesis). ([arXiv][9-13])
- **TransNeRF: Generalizable Neural Radiance Fields for Novel View Synthesis with Transformer** (2022)
  **Driver:** MLP NeRFs that struggle to condition on arbitrary numbers of views/capture view relationships
  **Outcome:** a transformer-based NeRF that fuses information across observed views via attention (transformer as the conditioning/fusion engine for a 3D field). ([arXiv][9-14])
- **Point Transformer V3: Simpler, Faster, Stronger** (2024)
  **Driver:** earlier point Transformers trading speed/scale for accuracy via complex mechanisms
  **Outcome:** a re-architected point cloud transformer backbone optimized for scalability and efficiency while keeping attention central for large 3D scenes. ([CVF Open Access][9-15])
- **SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection** (2025)
  **Driver:** BEV-only extraction and inefficiencies in dense voxel processing
  **Outcome:** operate directly on sparse 3D voxel features with a transformer detector (voxel-space attention as the core). ([arXiv][9-16])

[9-1]: MISSING
[9-2]: https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html
[9-3]: https://link.springer.com/article/10.1007/s41095-021-0229-5
[9-4]: https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf
[9-5]: https://arxiv.org/abs/2109.02497
[9-6]: https://arxiv.org/abs/2109.08141
[9-7]: https://arxiv.org/abs/2110.06922
[9-8]: https://arxiv.org/abs/2111.14819
[9-9]: https://arxiv.org/abs/2203.06604
[9-10]: https://arxiv.org/abs/2203.17270
[9-11]: https://openaccess.thecvf.com/content/CVPR2022/papers/Bai_TransFusion_Robust_LiDAR-Camera_Fusion_for_3D_Object_Detection_With_Transformers_CVPR_2022_paper.pdf
[9-12]: https://arxiv.org/abs/2302.12251
[9-13]: https://arxiv.org/abs/2203.10157
[9-14]: https://arxiv.org/abs/2206.05375
[9-15]: https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf
[9-16]: https://arxiv.org/html/2503.08092v1

- **Deformable DETR: Deformable Transformers for End-to-End Object Detection** (2020)
  **Improves On:** DETR’s global attention for dense 2D detection.
  **Adaptation:** sparse, multi-scale *deformable attention* for efficient set-based object detection. ([arXiv][33-1])
- **Training data-efficient image transformers & distillation through attention** (2020)
  **Contribution Type:** training methodology for Vision Transformers.
  **Reason:** make ViTs work well with less data via a *distillation token* + teacher supervision. ([arXiv][33-2])
- **Taming Transformers for High-Resolution Image Synthesis** (2020)
  **Improves On:** pixel-space autoregressive image Transformers.
  **Adaptation:** learn a discrete image codebook (VQ-style) then model code sequences with a Transformer for high-res synthesis. ([arXiv][33-3])
- **Generative Pretraining from Pixels** (2020)
  **Improves On:** 1D autoregressive Transformers for text.
  **Adaptation:** serialize 2D images into long 1D pixel sequences for causal Transformer pretraining. ([OpenAI][33-4])
- **Zero-Shot Text-to-Image Generation** (2021)
  **Improves On:** separate text-only or image-only sequence modeling.
  **Adaptation:** jointly model text + discrete image tokens to generate images from text prompts (zero-shot). ([arXiv][33-5])
- **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions** (2021)
  **Improves On:** single-scale ViT backbones for dense prediction.
  **Adaptation:** multi-stage pyramid token hierarchies to support detection/segmentation-style features. ([arXiv][33-6])
- **CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification** (2021)
  **Improves On:** single-patch-scale ViT classification.
  **Adaptation:** dual (multi) patch scales with cross-attention fusion between token streams. ([arXiv][33-7])
- **CvT: Introducing Convolutions to Vision Transformers** (2021)
  **Improves On:** pure ViT patch embedding + attention blocks.
  **Adaptation:** convolutional token embedding/projection to inject locality while keeping attention central. ([arXiv][33-8])
- **Transformer in Transformer** (2021)
  **Improves On:** standard ViT tokenization and global attention.
  **Adaptation:** inner Transformers within local regions plus an outer Transformer across regions (hierarchical). ([arXiv][33-9])
- **Twins: Revisiting the Design of Spatial Attention in Vision Transformers** (2021)
  **Improves On:** full global attention cost at high resolution.
  **Adaptation:** mix local window attention with global (subsampled) attention for scalable 2D backbones. ([arXiv][33-10])
- **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** (2021)
  **Improves On:** heavyweight segmentation decoders and non-unified pipelines.
  **Adaptation:** Transformer encoder with a lightweight MLP decode head for efficient dense prediction. ([arXiv][33-11])
- **BEiT: BERT Pre-Training of Image Transformers** (2021)
  **Contribution Type:** self-supervised pretraining objective.
  **Reason:** adapt BERT-style masked prediction to image tokens to improve ViT representations. ([arXiv][33-12])
- **CoAtNet: Marrying Convolution and Attention for All Data Sizes** (2021)
  **Improves On:** purely conv or purely attention vision stacks.
  **Adaptation:** stage-wise hybrid backbone combining conv layers with attention layers for strong scaling. ([arXiv][33-13])
- **CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows** (2021)
  **Improves On:** window attention with limited long-range mixing.
  **Adaptation:** cross-shaped attention windows to expand receptive field efficiently on 2D grids. ([arXiv][33-14])
- **Swin Transformer V2: Scaling Up Capacity and Resolution** (2021)
  **Improves On:** Swin-style windowed Transformers at larger scales and resolutions.
  **Adaptation:** architectural/training refinements to stabilize and scale window-based attention for high-res vision. ([arXiv][33-15])
- **MaxViT: Multi-Axis Vision Transformer** (2022)
  **Improves On:** either local-window-only or global-only attention patterns.
  **Adaptation:** combine local window attention with grid/axis-style global mixing for scalable vision backbones. ([arXiv][33-16])
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks** (2022)
  **Contribution Type:** unified pretraining recipe.
  **Reason:** extend BEiT-style masked modeling to support transfer across vision and vision-language tasks. ([arXiv][33-17])
- **Segment Anything** (2023)
  **Target Domain:** promptable, general-purpose image segmentation.
  **Resource:** large-scale segmentation data + a prompt-conditioned model enabling broad, zero-shot segmentation behavior. ([arXiv][33-18])

[33-1]: https://arxiv.org/pdf/2010.04159.pdf "Deformable DETR (2020) — arXiv"
[33-2]: https://arxiv.org/pdf/2012.12877.pdf "DeiT: Data-efficient Image Transformers (2020) — arXiv"
[33-3]: https://arxiv.org/pdf/2012.09841.pdf "Taming Transformers (2020) — arXiv"
[33-4]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf "Generative Pretraining from Pixels (2020) — OpenAI"
[33-5]: https://arxiv.org/pdf/2102.12092.pdf "Zero-Shot Text-to-Image Generation (2021) — arXiv"
[33-6]: https://arxiv.org/pdf/2102.12122.pdf "Pyramid Vision Transformer (2021) — arXiv"
[33-7]: https://arxiv.org/pdf/2103.14899.pdf "CrossViT (2021) — arXiv"
[33-8]: https://arxiv.org/pdf/2103.15808.pdf "CvT (2021) — arXiv"
[33-9]: https://arxiv.org/pdf/2103.00112.pdf "Transformer in Transformer (2021) — arXiv"
[33-10]: https://arxiv.org/pdf/2104.13840.pdf "Twins (2021) — arXiv"
[33-11]: https://arxiv.org/pdf/2105.15203.pdf "SegFormer (2021) — arXiv"
[33-12]: https://arxiv.org/pdf/2106.08254.pdf "BEiT (2021) — arXiv"
[33-13]: https://arxiv.org/pdf/2106.04803.pdf "CoAtNet (2021) — arXiv"
[33-14]: https://arxiv.org/pdf/2107.00652.pdf "CSWin Transformer (2021) — arXiv"
[33-15]: https://arxiv.org/pdf/2111.09883.pdf "Swin Transformer V2 (2021) — arXiv"
[33-16]: https://arxiv.org/pdf/2204.01697.pdf "MaxViT (2022) — arXiv"
[33-17]: https://arxiv.org/pdf/2208.10442.pdf "Image as a Foreign Language / BEiT pretraining (2022) — arXiv"
[33-18]: https://arxiv.org/pdf/2304.02643.pdf "Segment Anything (2023) — arXiv"

- **Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs** (2024)
  **Critique:** weak position extrapolation beyond trained context windows.
  **Improvement:** “weave” positional encoding to improve long-context extrapolation behavior. ([NeurIPS Proceedings][34-1])
- **Adaptive Patch Selection for ViTs via Reinforcement Learning** (2025)
  **Missing capability:** adaptive compute—processing all patches is wasteful for many images.
  **Mechanism:** RL-driven patch selection to dynamically choose informative tokens for ViTs. ([Springer][34-2])
- **Olympiad-level formal mathematical reasoning with large language models (AlphaProof)** (2025)
  **Missing capability:** reliable multi-step formal proof search and verification.
  **Mechanism:** LLM-guided formal reasoning integrated with proof-system checking to reach olympiad-level results. ([Nature][34-3])
- **The Rotary Position Embedding May Cause Dimension Inefficiency** (2025)
  **Critique:** RoPE can waste representational capacity across embedding dimensions.
  **Improvement:** analysis + guidance/variants aimed at more dimension-efficient rotary position usage. ([arXiv][34-4])
- **VRoPE: Rotary Position Embedding for Video Large Language Models** (2025)
  **Improves On:** text-only RoPE that does not natively encode space-time structure.
  **Adaptation:** extend rotary positional encoding to video (spatiotemporal) token streams for Video LLMs. ([arXiv][34-5])
- **Maximizing the Position Embedding for Vision Transformers (MPVG)** (2025)
  **Critique:** standard ViT positional embeddings underutilize/limit positional capacity for vision.
  **Improvement:** redesigned vision position embedding scheme to better exploit positional signal in ViTs. ([Local Copy][34-6])
- **SmolVLM: Redefining small and efficient multimodal models** (2025)
  **Contribution Type:** efficiency-focused multimodal model design.
  **Reason:** achieve strong vision-language performance under tight size/compute constraints. ([arXiv][34-7])
- **LOOPE: Learnable Optimal Patch Order in Vision Transformers** (2025)
  **Critique:** fixed raster scan patch order is an arbitrary inductive bias for images.
  **Improvement:** learn an optimal patch ordering to improve positional treatment in ViTs. ([arXiv][34-8])
- **Gated Attention for LLMs: Non-linearity, Sparsity, Sink-Free** (2025)
  **Contribution Type:** attention-layer redesign.
  **Reason:** introduce gating/nonlinearity to encourage sparsity and avoid attention pathologies (e.g., sink behavior). ([arXiv][34-9])
- **Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models** (2025)
  **Critique:** standard RoPE can entangle content and geometry for vision-language alignment.
  **Improvement:** decoupled “cone-like” rotary scheme to better separate and encode spatial structure. ([arXiv][34-10])
- **Rotary Masked Autoencoders Are Versatile Learners** (2025)
  **Contribution Type:** vision self-supervised learning framework refinement.
  **Reason:** combine MAE-style pretraining with rotary-based position handling for broader transfer. ([arXiv][34-11])
- **LLaVA-4D: Embedding Spatiotemporal Prompt into LMMs** (2025)
  **Improves On:** image-centric VLMs that lack explicit spatiotemporal (4D) prompting.
  **Adaptation:** embed spatiotemporal prompts/tokens so multimodal LMs can operate over space-time inputs. ([arXiv][34-12])
- **ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices** (2025)
  **Critique:** fixed-form rotary angles can be brittle across scales/domains.
  **Improvement:** trainable commuting angle matrices to parameterize RoPE more flexibly and robustly. ([arXiv][34-13])
- **Hierarchical Reasoning Model (HRM)** (2025)
  **Missing capability:** deep, stable latent reasoning with variable compute.
  **Mechanism:** hierarchical recurrent computation with fast–slow modules and adaptive halting. ([arXiv][34-14])
- **EVA02-AT: Egocentric Video-Language with Spatial-Temporal RoPE** (2025)
  **Improves On:** VLM encodings that do not cleanly represent space-time for egocentric video.
  **Adaptation:** spatial-temporal RoPE for video-language modeling in egocentric settings. ([arXiv][34-15])
- **TransXSSM: Hybrid Transformer–SSM with Unified RoPE** (2025)
  **Missing capability:** long-context sequence modeling with better scaling than pure attention.
  **Mechanism:** hybrid Transformer–state-space model architecture with a unified RoPE scheme. ([arXiv][34-16])
- **Context-aware Rotary Position Embedding (CARoPE)** (2025)
  **Critique:** context-agnostic rotary encodings may be suboptimal across varying contexts/lengths.
  **Improvement:** make rotary position encoding adapt based on context to improve robustness/extrapolation. ([arXiv][34-17])
- **Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings (PoPE)** (2025)
  **Critique:** entangled content and position signals in common encodings.
  **Improvement:** polar-coordinate positional embedding to separate “what” from “where.” ([arXiv][34-18])
- **Less is More: Recursive Reasoning with Tiny Networks** (2025)
  **Missing capability:** strong multi-step reasoning in small models without simply scaling parameters.
  **Mechanism:** recursive/iterative reasoning procedure enabling tiny networks to refine answers over steps. ([arXiv][34-19])
- **Head-Wise Adaptive Rotary Positional Encoding (HARoPE)** (2025)
  **Critique:** a single shared RoPE schedule can be too rigid across attention heads and modalities.
  **Improvement:** head-wise adaptive rotary encoding so different heads learn different positional behaviors. ([arXiv][34-20])
- **Nested Learning: The Illusion of Deep Learning Architecture** (2025)
  **Contribution Type:** theory/analysis of learning dynamics and depth.
  **Reason:** argues observed “depth benefits” can arise from nested learning effects rather than architecture alone. ([Local Copy][34-21])
- **DoPE: Denoising Rotary Position Embedding** (2025)
  **Critique:** rotary position signals can accumulate noise or become unstable under long-context use.
  **Improvement:** denoising strategy for RoPE to improve stability and downstream performance. ([arXiv][34-22])
- **WALRUS: A Cross-Domain Foundation Model for Continuum Dynamics** (2025)
  **Improves On:** narrow, domain-specific models for PDE/continuum dynamics.
  **Adaptation:** foundation-model approach that tokenizes/represents continuum dynamics to transfer across domains. ([arXiv][34-23])
- **Selective Rotary Position Embedding** (2025)
  **Critique:** applying rotary position uniformly can be inefficient or harmful in some layers/heads/tokens.
  **Improvement:** selectively apply rotary position to the most beneficial components for better efficiency/quality. ([arXiv][34-24])

[34-1]: https://proceedings.neurips.cc/paper_files/paper/2024/file/9446c291a8744a125a0bda5b18f4d5a1-Paper-Conference.pdf "Mesa-Extrapolation (2024) — NeurIPS Proceedings"
[34-2]: https://doi.org/10.1007/s10489-025-06516-z "Adaptive Patch Selection for ViTs via RL (2025) — Springer"
[34-3]: https://www.nature.com/articles/s41586-025-09833-y.pdf "AlphaProof (2025) — Nature"
[34-4]: https://arxiv.org/pdf/2502.11276.pdf "The Rotary Position Embedding May Cause Dimension Inefficiency (2025) — arXiv"
[34-5]: https://arxiv.org/pdf/2502.11664.pdf "VRoPE (2025) — arXiv"
[34-6]: # "MPVG (2025) — Local Copy (no public PDF link provided)"
[34-7]: https://arxiv.org/pdf/2504.05299.pdf "SmolVLM (2025) — arXiv"
[34-8]: https://arxiv.org/pdf/2504.14386.pdf "LOOPE (2025) — arXiv"
[34-9]: https://arxiv.org/pdf/2505.06708.pdf "Gated Attention for LLMs (2025) — arXiv"
[34-10]: https://arxiv.org/pdf/2505.16416.pdf "Circle-RoPE (2025) — arXiv"
[34-11]: https://arxiv.org/pdf/2505.20535.pdf "Rotary Masked Autoencoders (2025) — arXiv"
[34-12]: https://arxiv.org/pdf/2505.12253.pdf "LLaVA-4D (2025) — arXiv"
[34-13]: https://arxiv.org/pdf/2506.03737.pdf "ComRoPE (2025) — arXiv"
[34-14]: https://arxiv.org/pdf/2506.21734.pdf "Hierarchical Reasoning Model (2025) — arXiv"
[34-15]: https://arxiv.org/pdf/2506.14356.pdf "EVA02-AT (2025) — arXiv"
[34-16]: https://arxiv.org/pdf/2506.09507.pdf "TransXSSM (2025) — arXiv"
[34-17]: https://arxiv.org/pdf/2507.23083.pdf "CARoPE (2025) — arXiv"
[34-18]: https://arxiv.org/pdf/2509.10534.pdf "PoPE (2025) — arXiv"
[34-19]: https://arxiv.org/pdf/2510.04871.pdf "Less is More: Recursive Reasoning with Tiny Networks (2025) — arXiv"
[34-20]: https://arxiv.org/pdf/2510.10489.pdf "HARoPE (2025) — arXiv"
[34-21]: # "Nested Learning (2025) — Local Copy (no public PDF link provided)"
[34-22]: https://arxiv.org/pdf/2511.09146.pdf "DoPE (2025) — arXiv"
[34-23]: https://arxiv.org/pdf/2511.15684.pdf "WALRUS (2025) — arXiv"
[34-24]: https://arxiv.org/pdf/2511.17388.pdf "Selective Rotary Position Embedding (2025) — arXiv"

- **Uni3DL: Unified Model for 3D and Language Understanding** (2023)
  **Improves On:** vision-language models that primarily operate on 2D images.
  **Adaptation:** unify 3D representations with language understanding for joint 3D + text reasoning. ([arXiv][35-1])
- **Evolutionary Test-Time Compute (write-up)** (2024)
  **Missing capability:** variable test-time compute and search to solve hard reasoning tasks.
  **Mechanism:** evolutionary-style test-time adaptation/selection over candidate solutions to boost performance. ([Substack][35-2])
- **Fixed Point Diffusion Models** (2024)
  **Contribution Type:** generative modeling framework.
  **Reason:** reframes diffusion sampling/training via fixed-point structure to stabilize or improve generation. ([CVF][35-3])
- **Solving olympiad geometry without human demonstrations (AlphaGeometry)** (2024)
  **Missing capability:** reliable multi-step geometry proof search without curated human solution traces.
  **Mechanism:** neuro-symbolic pipeline for generating and verifying formal geometry proofs at olympiad level. ([Nature][35-4])
- **Gaussian Adaptive Attention Is All You Need** (2024)
  **Contribution Type:** attention mechanism variant.
  **Reason:** replace/augment standard attention with a Gaussian adaptive form to improve efficiency and/or inductive bias. ([arXiv][35-5])
- **Beyond A*: Planning with Transformers** (2024)
  **Missing capability:** scalable planning/search that goes beyond classical A* heuristics.
  **Mechanism:** Transformer-based planning that learns to guide or replace components of traditional search. ([arXiv][35-6])
- **Rotary Position Embedding for Vision Transformer (RoPE-Mixed / 2D RoPE study)** (2024)
  **Critique:** common 2D positional schemes (or naive RoPE transfers) can be suboptimal for images.
  **Improvement:** study + variants for applying rotary positional encoding in ViTs (e.g., mixed/2D formulations). ([arXiv][35-7])
- **VG4D: Vision-Language Model Goes 4D Video Recognition** (2024)
  **Improves On:** image-centric VLMs that don’t model space-time well.
  **Adaptation:** extend VLMs to 4D (spatiotemporal) video recognition with space-time token modeling. ([arXiv][35-8])
- **DAPE: Data-Adaptive Positional Encoding for Length Extrapolation** (2024)
  **Critique:** fixed positional encodings can extrapolate poorly to longer contexts.
  **Improvement:** data-adaptive positional encoding designed to improve length extrapolation. ([NeurIPS Proceedings][35-9])
- **What matters when building vision-language models? (Idefics2)** (2024)
  **Contribution Type:** empirical study / model-building guidance.
  **Reason:** analyze design and training choices that most affect VLM performance and efficiency. ([arXiv][35-10])
- **Base of RoPE Bounds Context Length** (2024)
  **Critique:** RoPE’s base/frequency parameterization imposes constraints on usable context length.
  **Improvement:** theory/analysis (and implications) linking RoPE base to context-length limits. ([arXiv][35-11])
- **RoTHP: Rotary Position Embedding-based Transformer Hawkes Process** (2024)
  **Improves On:** standard Transformer sequence models not tailored to continuous-time event data.
  **Adaptation:** combine Hawkes process modeling with RoPE-parameterized Transformer representations for temporal point processes. ([arXiv][35-12])
- **LieRE: Lie Rotational Positional Encodings** (2024)
  **Critique:** conventional positional encodings can lack principled geometric structure.
  **Improvement:** positional encoding built from Lie-group rotational structure for more principled position handling. ([arXiv][35-13])
- **Learning Iterative Reasoning through Energy Diffusion** (2024)
  **Missing capability:** iterative refinement dynamics for multi-step reasoning beyond single-pass inference.
  **Mechanism:** energy/diffusion-inspired iterative reasoning procedure to refine solutions over steps. ([arXiv][35-14])
- **Simultaneous Instance Pooling & Bag Selection for MIL using ViTs** (2024)
  **Contribution Type:** architecture/training method for multiple instance learning (MIL).
  **Reason:** integrate instance pooling and bag selection within ViT-based MIL to improve performance and efficiency. ([Springer][35-15])
- **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** (2024)
  **Missing capability:** principled allocation of extra compute at inference time.
  **Mechanism:** strategy/analysis for optimally scaling test-time compute to outperform parameter scaling in some regimes. ([arXiv][35-16])
- **Length Extrapolation of Causal Transformers without Position Encoding (NoPE)** (2024)
  **Critique:** explicit positional encodings can be a bottleneck for length extrapolation in causal Transformers.
  **Improvement:** remove positional encoding (and analyze/enable extrapolation behavior without it). ([ACL Anthology][35-17])
- **H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark** (2024)
  **Target Domain:** human baseline measurement on ARC-style abstract reasoning tasks.
  **Resource:** a robust estimate/protocol for human performance on ARC, for fair model comparison. ([arXiv][35-18])
- **Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding** (2024)
  **Target Domain:** length extrapolation methods and failure modes in Transformers.
  **Resource:** taxonomy/survey of approaches framed through positional encoding choices and alternatives. ([ACL Anthology][35-19])
- **Beyond Position: The Emergence of Wavelet-like Properties in Transformers** (2024)
  **Contribution Type:** theoretical/interpretability analysis.
  **Reason:** characterize emergent wavelet-like behavior in Transformers beyond explicit positional mechanisms. ([arXiv][35-20])
- **ARC-Heavy / ARC-Potpourri (dataset description embedded in Cornell report)** (2024)
  **Target Domain:** harder and more varied ARC-style abstract reasoning evaluation.
  **Resource:** dataset variants (ARC-Heavy / ARC-Potpourri) described as infrastructure for improved measurement. ([Cornell][35-21])
- **Combining Induction and Transduction for Abstract Reasoning** (2024)
  **Contribution Type:** modeling/training approach for abstract reasoning.
  **Reason:** combine inductive generalization with transductive/test-time adaptation to improve ARC-like performance. ([arXiv][35-22])
- **Searching Latent Program Spaces** (2024)
  **Missing capability:** structured search over program-like solution spaces that is hard for pure next-token prediction.
  **Mechanism:** search procedure over latent program representations to solve reasoning tasks. ([arXiv][35-23])
- **The Surprising Effectiveness of Test-Time Training for Few-Shot Learning** (2024)
  **Missing capability:** on-the-fly adaptation when few labeled examples are available.
  **Mechanism:** test-time training procedure that updates/optimizes at inference to improve few-shot performance. ([Project Page][35-24])
- **Towards Efficient Neurally-Guided Program Induction for ARC-AGI** (2024)
  **Missing capability:** efficient program induction/search guided by neural priors for ARC-AGI tasks.
  **Mechanism:** neurally-guided program induction pipeline to search programs more efficiently for ARC solutions. ([arXiv][35-25])

[35-1]: https://arxiv.org/pdf/2312.03026.pdf "Uni3DL (2023) — arXiv"
[35-2]: https://jeremyberman.substack.com/p/how-i-got-a-record-536-on-arc-agi "Evolutionary Test-Time Compute (2024) — Substack"
[35-3]: https://openaccess.thecvf.com/content/CVPR2024/papers/Bai_Fixed_Point_Diffusion_Models_CVPR_2024_paper.pdf "Fixed Point Diffusion Models (2024) — CVF / CVPR"
[35-4]: https://www.nature.com/articles/s41586-023-06747-5.pdf "AlphaGeometry (2024) — Nature"
[35-5]: https://arxiv.org/pdf/2401.11143.pdf "Gaussian Adaptive Attention (2024) — arXiv"
[35-6]: https://arxiv.org/pdf/2402.14083.pdf "Beyond A*: Planning with Transformers (2024) — arXiv"
[35-7]: https://arxiv.org/pdf/2403.13298.pdf "RoPE for ViT / RoPE-Mixed 2D study (2024) — arXiv"
[35-8]: https://arxiv.org/pdf/2404.11605.pdf "VG4D (2024) — arXiv"
[35-9]: https://proceedings.neurips.cc/paper_files/paper/2024/file/2f050fa9f0d898e3f265d515f50ae8f9-Paper-Conference.pdf "DAPE (2024) — NeurIPS Proceedings"
[35-10]: https://arxiv.org/pdf/2405.02246.pdf "Idefics2 study (2024) — arXiv"
[35-11]: https://arxiv.org/pdf/2405.14591.pdf "Base of RoPE Bounds Context Length (2024) — arXiv"
[35-12]: https://arxiv.org/pdf/2405.06985.pdf "RoTHP (2024) — arXiv"
[35-13]: https://arxiv.org/pdf/2406.10322.pdf "LieRE (2024) — arXiv"
[35-14]: https://arxiv.org/pdf/2406.11179.pdf "Energy Diffusion Iterative Reasoning (2024) — arXiv"
[35-15]: https://doi.org/10.1007/s00521-024-09417-3 "MIL with ViTs (2024) — Springer"
[35-16]: https://arxiv.org/pdf/2408.03314.pdf "Scaling LLM Test-Time Compute Optimally (2024) — arXiv"
[35-17]: https://aclanthology.org/2024.findings-acl.834.pdf "NoPE (2024) — ACL Anthology"
[35-18]: https://arxiv.org/pdf/2409.01374.pdf "H-ARC (2024) — arXiv"
[35-19]: https://aclanthology.org/2024.findings-emnlp.582.pdf "Length Extrapolation Survey (2024) — ACL Anthology"
[35-20]: https://arxiv.org/pdf/2410.18067.pdf "Wavelet-like Properties in Transformers (2024) — arXiv"
[35-21]: https://www.cs.cornell.edu/~ellisk/documents/arc_induction_vs_transduction.pdf "ARC-Heavy / ARC-Potpourri (2024) — Cornell"
[35-22]: https://arxiv.org/pdf/2411.02272.pdf "Combining Induction and Transduction (2024) — arXiv"
[35-23]: https://arxiv.org/pdf/2411.08706.pdf "Searching Latent Program Spaces (2024) — arXiv"
[35-24]: https://ekinakyurek.github.io/papers/ttt.pdf "Test-Time Training for Few-Shot Learning (2024) — Project Page"
[35-25]: https://arxiv.org/pdf/2411.17708.pdf "Neurally-Guided Program Induction for ARC-AGI (2024) — arXiv"

- **Swin Transformer V2: Scaling Up Capacity and Resolution** (2021)
  **Improves On:** Swin-style windowed Transformers at larger scales and resolutions.
  **Adaptation:** refinements to stabilize and scale window-based attention for high-resolution vision backbones. ([arXiv][36-1])
- **BLIP: Bootstrapping Language-Image Pre-training** (2022)
  **Contribution Type:** vision-language pretraining method.
  **Reason:** bootstrap captioning and understanding with a unified VLP framework for stronger VLM training. ([arXiv][36-2])
- **Transformer Language Models without Positional Encodings Still Learn Positional Information** (2022)
  **Contribution Type:** theory/analysis of positional information.
  **Reason:** show how transformers can recover positional signals even when explicit positional encodings are removed. ([arXiv][36-3])
- **CoCa: Contrastive Captioners** (2022)
  **Contribution Type:** vision-language training objective/method.
  **Reason:** unify contrastive learning with captioning to improve image-text representation and generation. ([arXiv][36-4])
- **Flamingo: a Visual Language Model for Few-Shot Learning** (2022)
  **Contribution Type:** multimodal model architecture.
  **Reason:** enable strong few-shot visual-language learning via cross-attention over visual inputs with a large LM. ([arXiv][36-5])
- **MaxViT: Multi-Axis Vision Transformer** (2022)
  **Improves On:** either local-window-only or global-only attention patterns.
  **Adaptation:** combine local window attention with multi-axis global mixing for scalable vision backbones. ([arXiv][36-6])
- **Winoground: Probing Vision-Language Models for Compositionality** (2022)
  **Target Domain:** compositional vision-language understanding.
  **Resource:** diagnostic benchmark to measure compositional reasoning failures in VLMs. ([arXiv][36-7])
- **Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks** (2022)
  **Contribution Type:** unified pretraining recipe.
  **Reason:** extend BEiT-style masked modeling to support transfer across vision and vision-language tasks. ([arXiv][36-8])
- **LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models** (2022)
  **Target Domain:** large-scale vision–language pretraining.
  **Resource:** LAION-5B, a massive web-scale image–text dataset for training VLMs and text-to-image models. ([arXiv][36-9])
- **ScienceQA: Benchmark for Multimodal Reasoning** (2022)
  **Target Domain:** multimodal science question answering and reasoning.
  **Resource:** ScienceQA benchmark with multimodal context and explanation-focused evaluation. ([arXiv][36-10])
- **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models** (2023)
  **Contribution Type:** parameter-efficient multimodal training method.
  **Reason:** connect frozen vision and language models via a lightweight bridge for strong VLM performance. ([arXiv][36-11])
- **GPT-4 Technical Report** (2023)
  **Contribution Type:** system/reporting and evaluation.
  **Reason:** describe capabilities, evaluations, and safety considerations for GPT-4 as a large-scale model system. ([arXiv][36-12])
- **Kosmos-1: Language Is Not All You Need** (2023)
  **Contribution Type:** multimodal foundation model.
  **Reason:** train a generalist model over text and other modalities to extend beyond language-only competence. ([arXiv][36-13])
- **PaLM-E: An Embodied Multimodal Language Model** (2023)
  **Contribution Type:** embodied multimodal model.
  **Reason:** integrate vision and robotics/embodiment signals with a large LM for embodied reasoning and control. ([arXiv][36-14])
- **LLaVA: Large Language-and-Vision Assistant** (2023)
  **Contribution Type:** instruction-tuned vision-language assistant.
  **Reason:** align an LLM with visual features to enable conversational, instruction-following VQA and grounding. ([arXiv][36-15])
- **MiniGPT-4** (2023)
  **Contribution Type:** lightweight vision-language alignment method.
  **Reason:** align a frozen vision encoder to an LLM to obtain strong VLM/chat behavior with modest training. ([arXiv][36-16])
- **Segment Anything** (2023)
  **Target Domain:** promptable, general-purpose image segmentation.
  **Resource:** large-scale segmentation dataset + prompt-conditioned model enabling broad zero-shot segmentation. ([arXiv][36-17])
- **ConceptARC** (2023)
  **Target Domain:** concept-based abstract reasoning on ARC-style tasks.
  **Resource:** benchmark/dataset variant to probe concept generalization and systematicity in ARC-like settings. ([arXiv][36-18])
- **MMBench: Evaluating Multimodal LLMs** (2023)
  **Target Domain:** evaluation of multimodal LLM capabilities.
  **Resource:** benchmark suite for standardized assessment of multimodal LLM performance. ([arXiv][36-19])
- **A Length-Extrapolatable Transformer (XPOS / LeX)** (2023)
  **Critique:** standard positional encodings generalize poorly to longer-than-trained contexts.
  **Improvement:** length-extrapolatable positional strategy (e.g., XPOS) to improve long-context generalization. ([ACL Anthology][36-20])
- **Average-Hard Attention Transformers Are Threshold Circuits** (2023)
  **Contribution Type:** theoretical characterization.
  **Reason:** analyze computational class/limits of specific attention mechanisms via circuit complexity framing. ([arXiv][36-21])
- **YaRN: Efficient Context Window Extension of Large Language Models** (2023)
  **Critique:** RoPE-based models extrapolate poorly to longer contexts without adjustment.
  **Improvement:** RoPE interpolation + scaling tricks to extend context windows efficiently. ([arXiv][36-22])
- **Spherical Position Encoding for Transformers** (2023)
  **Critique:** common positional encodings can be geometry-mismatched for directional/angle-like structure.
  **Improvement:** spherical positional encoding to represent position/direction with spherical geometry. ([arXiv][36-23])
- **MMMU: A Massive Multidiscipline Multimodal Benchmark** (2023)
  **Target Domain:** broad multidisciplinary multimodal understanding.
  **Resource:** benchmark spanning many subjects to evaluate multimodal reasoning and knowledge. ([arXiv][36-24])
- **Gemini: A Family of Highly Capable Multimodal Models** (2023)
  **Contribution Type:** multimodal foundation model family.
  **Reason:** describe a suite of highly capable multimodal models and their evaluation across tasks. ([arXiv][36-25])

[36-1]: https://arxiv.org/pdf/2111.09883.pdf "Swin Transformer V2 (2021) — arXiv"
[36-2]: https://arxiv.org/pdf/2201.12086.pdf "BLIP (2022) — arXiv"
[36-3]: https://arxiv.org/pdf/2203.16634.pdf "Transformers without PEs still learn position (2022) — arXiv"
[36-4]: https://arxiv.org/pdf/2205.01917.pdf "CoCa (2022) — arXiv"
[36-5]: https://arxiv.org/pdf/2204.14198.pdf "Flamingo (2022) — arXiv"
[36-6]: https://arxiv.org/pdf/2204.01697.pdf "MaxViT (2022) — arXiv"
[36-7]: https://arxiv.org/pdf/2204.03162.pdf "Winoground (2022) — arXiv"
[36-8]: https://arxiv.org/pdf/2208.10442.pdf "Image as a Foreign Language (2022) — arXiv"
[36-9]: https://arxiv.org/pdf/2210.08402.pdf "LAION-5B (2022) — arXiv"
[36-10]: https://arxiv.org/pdf/2209.09513.pdf "ScienceQA (2022) — arXiv"
[36-11]: https://arxiv.org/pdf/2301.12597.pdf "BLIP-2 (2023) — arXiv"
[36-12]: https://arxiv.org/pdf/2303.08774.pdf "GPT-4 Technical Report (2023) — arXiv"
[36-13]: https://arxiv.org/pdf/2302.14045.pdf "Kosmos-1 (2023) — arXiv"
[36-14]: https://arxiv.org/pdf/2303.03378.pdf "PaLM-E (2023) — arXiv"
[36-15]: https://arxiv.org/pdf/2304.08485.pdf "LLaVA (2023) — arXiv"
[36-16]: https://arxiv.org/pdf/2304.10592.pdf "MiniGPT-4 (2023) — arXiv"
[36-17]: https://arxiv.org/pdf/2304.02643.pdf "Segment Anything (2023) — arXiv"
[36-18]: https://arxiv.org/pdf/2305.07141.pdf "ConceptARC (2023) — arXiv"
[36-19]: https://arxiv.org/pdf/2307.06281.pdf "MMBench (2023) — arXiv"
[36-20]: https://aclanthology.org/2023.acl-long.816.pdf "XPOS / LeX (2023) — ACL Anthology"
[36-21]: https://arxiv.org/pdf/2308.03212.pdf "Average-Hard Attention as Threshold Circuits (2023) — arXiv"
[36-22]: https://arxiv.org/pdf/2309.00071.pdf "YaRN (2023) — arXiv"
[36-23]: https://arxiv.org/pdf/2310.04454.pdf "Spherical Position Encoding (2023) — arXiv"
[36-24]: https://arxiv.org/pdf/2311.16502.pdf "MMMU (2023) — arXiv"
[36-25]: https://arxiv.org/pdf/2312.11805.pdf "Gemini (2023) — arXiv"

- **Taming Transformers for High-Resolution Image Synthesis** (2020)
  **Improves On:** pixel-space autoregressive image Transformers.
  **Adaptation:** learn a discrete image codebook (VQ-style) then model code sequences with a Transformer for high-res synthesis. ([arXiv][37-1])
- **Generative Pretraining from Pixels** (2020)
  **Improves On:** 1D autoregressive Transformers for text.
  **Adaptation:** serialize 2D images into long 1D pixel sequences for causal Transformer pretraining. ([OpenAI][37-2])
- **ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases** (2021)
  **Improves On:** pure ViT attention with weak locality priors.
  **Adaptation:** add soft convolutional inductive biases inside attention to inject locality while keeping Transformer flexibility. ([arXiv][37-3])
- **Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)** (2021)
  **Improves On:** Vision Transformers for static images.
  **Adaptation:** extend patch tokens across time with factorized spatial–temporal attention for video understanding. ([arXiv][37-4])
- **Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN)** (2021)
  **Contribution Type:** vision-language pretraining recipe.
  **Reason:** scale image–text contrastive learning using large noisy web supervision to learn strong aligned representations. ([arXiv][37-5])
- **Learning Transferable Visual Models From Natural Language Supervision (CLIP)** (2021)
  **Contribution Type:** vision-language pretraining objective.
  **Reason:** contrastive pretraining on image–text pairs to produce transferable zero-shot representations. ([arXiv][37-6])
- **Zero-Shot Text-to-Image Generation** (2021)
  **Improves On:** separate text-only or image-only sequence modeling.
  **Adaptation:** jointly model text + discrete image tokens to generate images from text prompts (zero-shot). ([arXiv][37-7])
- **Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions** (2021)
  **Improves On:** single-scale ViT backbones for dense prediction.
  **Adaptation:** multi-stage pyramid token hierarchies to support detection/segmentation-style features. ([arXiv][37-8])
- **Swin Transformer** (2021)
  **Improves On:** global-attention ViTs that scale poorly at high resolution.
  **Adaptation:** shifted window attention for efficient hierarchical vision backbones. ([arXiv][37-9])
- **CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification** (2021)
  **Improves On:** single-patch-scale ViT classification.
  **Adaptation:** dual (multi) patch scales with cross-attention fusion between token streams. ([arXiv][37-10])
- **CvT: Introducing Convolutions to Vision Transformers** (2021)
  **Improves On:** pure ViT patch embedding + attention blocks.
  **Adaptation:** convolutional token embedding/projection to inject locality while keeping attention central. ([arXiv][37-11])
- **Transformer in Transformer** (2021)
  **Improves On:** standard ViT tokenization and global attention.
  **Adaptation:** inner Transformers within local regions plus an outer Transformer across regions (hierarchical). ([arXiv][37-12])
- **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** (2021)
  **Critique:** absolute positional embeddings and their weak relative-position behavior.
  **Improvement:** rotary position embedding to encode relative position through rotation in attention. ([arXiv][37-13])
- **Twins: Revisiting the Design of Spatial Attention in Vision Transformers** (2021)
  **Improves On:** full global attention cost at high resolution.
  **Adaptation:** mix local window attention with global (subsampled) attention for scalable 2D backbones. ([arXiv][37-14])
- **SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers** (2021)
  **Improves On:** heavyweight segmentation decoders and non-unified pipelines.
  **Adaptation:** Transformer encoder with a lightweight MLP decode head for efficient dense prediction. ([arXiv][37-15])
- **VATT: Transformers for Multimodal Self-Supervised Learning** (2021)
  **Contribution Type:** multimodal self-supervised pretraining framework.
  **Reason:** learn general audio-visual-text representations with Transformer-based multimodal SSL at scale. ([arXiv][37-16])
- **BEiT: BERT Pre-Training of Image Transformers** (2021)
  **Contribution Type:** self-supervised pretraining objective.
  **Reason:** adapt BERT-style masked prediction to image tokens to improve ViT representations. ([arXiv][37-17])
- **CoAtNet: Marrying Convolution and Attention for All Data Sizes** (2021)
  **Improves On:** purely conv or purely attention vision stacks.
  **Adaptation:** stage-wise hybrid backbone combining conv layers with attention layers for strong scaling. ([arXiv][37-18])
- **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision** (2021)
  **Contribution Type:** simplified VLM architecture.
  **Reason:** operate directly on visual patches + text tokens with a single Transformer, avoiding region detectors and heavy visual backbones. ([arXiv][37-19])
- **VideoCLIP: Contrastive Pretraining for Zero-Shot Video-Text Understanding** (2021)
  **Contribution Type:** video-language pretraining method.
  **Reason:** contrastive pretraining on video–text pairs to enable zero-shot video-text retrieval and understanding. ([arXiv][37-20])
- **ALBEF: Align Before Fuse** (2021)
  **Contribution Type:** vision-language pretraining method.
  **Reason:** align image-text representations before fusion to improve cross-modal grounding and downstream VQA/retrieval. ([arXiv][37-21])
- **CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows** (2021)
  **Improves On:** window attention with limited long-range mixing.
  **Adaptation:** cross-shaped attention windows to expand receptive field efficiently on 2D grids. ([arXiv][37-22])
- **Train Short, Test Long: Attention with Linear Biases (ALiBi)** (2021)
  **Critique:** positional encodings often extrapolate poorly to longer contexts.
  **Improvement:** linear attention bias that enables better length extrapolation without learned position embeddings. ([arXiv][37-23])
- **LAION-400M: Open Dataset for CLIP Training** (2021)
  **Target Domain:** large-scale vision–language pretraining for CLIP-style models.
  **Resource:** LAION-400M dataset of image–text pairs to support open CLIP training and evaluation. ([arXiv][37-24])
- **Masked Autoencoders Are Scalable Vision Learners (MAE)** (2021)
  **Contribution Type:** self-supervised vision pretraining method.
  **Reason:** masked image modeling with an encoder–decoder to learn scalable, transferable ViT representations efficiently. ([arXiv][37-25])

[37-1]: https://arxiv.org/pdf/2012.09841.pdf "Taming Transformers (2020) — arXiv"
[37-2]: https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf "Generative Pretraining from Pixels (2020) — OpenAI"
[37-3]: https://arxiv.org/pdf/2103.10697.pdf "ConViT (2021) — arXiv"
[37-4]: https://arxiv.org/pdf/2102.05095.pdf "TimeSformer (2021) — arXiv"
[37-5]: https://arxiv.org/pdf/2102.05918.pdf "ALIGN (2021) — arXiv"
[37-6]: https://arxiv.org/pdf/2103.00020.pdf "CLIP (2021) — arXiv"
[37-7]: https://arxiv.org/pdf/2102.12092.pdf "Zero-Shot Text-to-Image Generation (2021) — arXiv"
[37-8]: https://arxiv.org/pdf/2102.12122.pdf "PVT (2021) — arXiv"
[37-9]: https://arxiv.org/pdf/2103.14030.pdf "Swin Transformer (2021) — arXiv"
[37-10]: https://arxiv.org/pdf/2103.14899.pdf "CrossViT (2021) — arXiv"
[37-11]: https://arxiv.org/pdf/2103.15808.pdf "CvT (2021) — arXiv"
[37-12]: https://arxiv.org/pdf/2103.00112.pdf "Transformer in Transformer (2021) — arXiv"
[37-13]: https://arxiv.org/pdf/2104.09864.pdf "RoFormer (2021) — arXiv"
[37-14]: https://arxiv.org/pdf/2104.13840.pdf "Twins (2021) — arXiv"
[37-15]: https://arxiv.org/pdf/2105.15203.pdf "SegFormer (2021) — arXiv"
[37-16]: https://arxiv.org/pdf/2104.11178.pdf "VATT (2021) — arXiv"
[37-17]: https://arxiv.org/pdf/2106.08254.pdf "BEiT (2021) — arXiv"
[37-18]: https://arxiv.org/pdf/2106.04803.pdf "CoAtNet (2021) — arXiv"
[37-19]: https://arxiv.org/pdf/2102.03334.pdf "ViLT (2021) — arXiv"
[37-20]: https://arxiv.org/pdf/2109.14084.pdf "VideoCLIP (2021) — arXiv"
[37-21]: https://arxiv.org/pdf/2107.07651.pdf "ALBEF (2021) — arXiv"
[37-22]: https://arxiv.org/pdf/2107.00652.pdf "CSWin Transformer (2021) — arXiv"
[37-23]: https://arxiv.org/pdf/2108.12409.pdf "ALiBi (2021) — arXiv"
[37-24]: https://arxiv.org/pdf/2111.02114.pdf "LAION-400M (2021) — arXiv"
[37-25]: https://arxiv.org/pdf/2111.06377.pdf "MAE (2021) — arXiv"

- **Feedforward and Recurrent Processing in Vision** (2000)
  **Contribution Type:** neuroscience perspective on visual computation.
  **Reason:** analyze complementary roles of feedforward and recurrent processing in visual perception. ([PubMed][38-1])
- **The Hidden Logic of Sudoku (Second Edition)** (2007)
  **Contribution Type:** reasoning and constraint-satisfaction exposition.
  **Reason:** formalize logical structures and strategies underlying Sudoku as a model reasoning system. ([Book][38-2])
- **Core Knowledge** (2007)
  **Contribution Type:** cognitive science foundations.
  **Reason:** propose innate core knowledge systems shaping human cognition and learning. ([Harvard LDS][38-3])
- **Canonical Microcircuits for Predictive Coding** (2012)
  **Contribution Type:** neuroscience theory.
  **Reason:** describe cortical microcircuits implementing predictive coding principles. ([PubMed][38-4])
- **Hierarchy of Intrinsic Timescales in Cortex** (2014)
  **Contribution Type:** neuroscience systems analysis.
  **Reason:** identify hierarchical temporal processing scales across cortical regions. ([NYU][38-5])
- **Neural Turing Machines** (2014)
  **Missing capability:** persistent, addressable memory in neural networks.
  **Mechanism:** differentiable read/write operations over an external memory matrix. ([arXiv][38-6])
- **Adaptive Computation Time** (2016)
  **Missing capability:** variable-depth computation per input.
  **Mechanism:** learned halting mechanism allowing networks to adapt computation steps dynamically. ([arXiv][38-7])
- **Attention Is All You Need** (2017)
  **Contribution Type:** architectural unification.
  **Reason:** introduce the Transformer, replacing recurrence and convolution with attention mechanisms. ([arXiv][38-8])
- **Recurrent Relational Networks** (2018)
  **Missing capability:** iterative relational reasoning.
  **Mechanism:** recurrent message passing over relational structures for constraint satisfaction. ([NeurIPS Proceedings][38-9])
- **Universal Transformers** (2018)
  **Missing capability:** iterative refinement with shared parameters.
  **Mechanism:** recurrent application of a Transformer block with adaptive depth. ([arXiv][38-10])
- **Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset** (2018)
  **Target Domain:** vision–language pretraining.
  **Resource:** large-scale cleaned and hypernymed image–text dataset for VLM training. ([arXiv][38-11])
- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding** (2018)
  **Contribution Type:** pretraining methodology.
  **Reason:** introduce masked language modeling and bidirectional Transformer pretraining. ([arXiv][38-12])
- **LXMERT: Learning Cross-Modality Encoder Representations** (2019)
  **Contribution Type:** multimodal Transformer architecture.
  **Reason:** dual-stream cross-attention model for vision–language reasoning tasks. ([arXiv][38-13])
- **ViLBERT: Pretraining Task-Agnostic Vision-and-Language Representations** (2019)
  **Contribution Type:** multimodal pretraining framework.
  **Reason:** two-stream Transformer with co-attention for transferable V–L representations. ([arXiv][38-14])
- **VisualBERT: A Simple and Performant Baseline for Vision and Language** (2019)
  **Contribution Type:** unified multimodal baseline.
  **Reason:** single-stream Transformer over image regions and text tokens for V–L tasks. ([arXiv][38-15])
- **Deep Equilibrium Models** (2019)
  **Missing capability:** infinite-depth/implicit computation.
  **Mechanism:** define models by fixed points of implicit layers trained via equilibrium solvers. ([arXiv][38-16])
- **VideoBERT: A Joint Model for Video and Language Representation Learning** (2019)
  **Contribution Type:** video–language pretraining.
  **Reason:** learn joint representations by tokenizing video and text sequences. ([arXiv][38-17])
- **HowTo100M: Learning a Text-Video Embedding by Watching Narrated Videos** (2019)
  **Target Domain:** large-scale video–language learning.
  **Resource:** 100M narrated video clips for weakly supervised V–L embedding learning. ([arXiv][38-18])
- **On the Measure of Intelligence** (2019)
  **Target Domain:** abstract and spatial reasoning.
  **Resource:** Abstraction and Reasoning Corpus (ARC) as a benchmark for general intelligence. ([arXiv][38-19])
- **UNITER: Universal Image-Text Representation Learning** (2020)
  **Contribution Type:** unified VLM pretraining.
  **Reason:** single-stream Transformer with multiple pretraining objectives for strong V–L transfer. ([arXiv][38-20])
- **OSCAR: Object-Semantics Aligned Pre-training for Vision-Language Tasks** (2020)
  **Contribution Type:** VLM pretraining with object tags.
  **Reason:** align object-level semantics with text to improve cross-modal grounding. ([arXiv][38-21])
- **End-to-End Object Detection with Transformers (DETR)** (2020)
  **Improves On:** CNN-based object detection pipelines.
  **Adaptation:** global attention with set prediction and bipartite matching loss for detection. ([arXiv][38-22])
- **An Image is Worth 16×16 Words: Vision Transformer (ViT)** (2020)
  **Improves On:** CNN-based image classification.
  **Adaptation:** tokenize images into patches and process with a standard Transformer encoder. ([arXiv][38-23])
- **Deformable DETR: Deformable Transformers for End-to-End Object Detection** (2020)
  **Improves On:** DETR’s global attention inefficiency.
  **Adaptation:** sparse, multi-scale deformable attention for efficient detection. ([arXiv][38-24])
- **Training data-efficient image transformers & distillation through attention** (2020)
  **Contribution Type:** training methodology for ViTs.
  **Reason:** use distillation tokens and teacher supervision to improve data efficiency. ([arXiv][38-25])

[38-1]: https://pubmed.ncbi.nlm.nih.gov/10925037/ "Feedforward and Recurrent Processing in Vision (2000) — PubMed"
[38-2]: # "The Hidden Logic of Sudoku (2007) — Book (local copy)"
[38-3]: https://www.harvardlds.org/wp-content/uploads/2017/01/SpelkeKinzler07-1.pdf "Core Knowledge (2007) — Harvard LDS"
[38-4]: https://pubmed.ncbi.nlm.nih.gov/23238495/ "Canonical Microcircuits for Predictive Coding (2012) — PubMed"
[38-5]: https://www.cns.nyu.edu/wanglab/publications/pdf/murray.nn2014.pdf "Hierarchy of Intrinsic Timescales in Cortex (2014) — NYU"
[38-6]: https://arxiv.org/pdf/1410.5401.pdf "Neural Turing Machines (2014) — arXiv"
[38-7]: https://arxiv.org/pdf/1603.08983.pdf "Adaptive Computation Time (2016) — arXiv"
[38-8]: https://arxiv.org/pdf/1706.03762.pdf "Attention Is All You Need (2017) — arXiv"
[38-9]: https://proceedings.neurips.cc/paper_files/paper/2018/file/b9f94c77652c9a76fc8a442748cd54bd-Paper.pdf "Recurrent Relational Networks (2018) — NeurIPS"
[38-10]: https://arxiv.org/pdf/1807.03819.pdf "Universal Transformers (2018) — arXiv"
[38-11]: https://arxiv.org/pdf/1803.10137.pdf "Conceptual Captions (2018) — arXiv"
[38-12]: https://arxiv.org/pdf/1810.04805.pdf "BERT (2018) — arXiv"
[38-13]: https://arxiv.org/pdf/1908.07490.pdf "LXMERT (2019) — arXiv"
[38-14]: https://arxiv.org/pdf/1908.02265.pdf "ViLBERT (2019) — arXiv"
[38-15]: https://arxiv.org/pdf/1908.03557.pdf "VisualBERT (2019) — arXiv"
[38-16]: https://arxiv.org/pdf/1909.01377.pdf "Deep Equilibrium Models (2019) — arXiv"
[38-17]: https://arxiv.org/pdf/1904.01766.pdf "VideoBERT (2019) — arXiv"
[38-18]: https://arxiv.org/pdf/1906.03327.pdf "HowTo100M (2019) — arXiv"
[38-19]: https://arxiv.org/pdf/1911.01547.pdf "On the Measure of Intelligence (2019) — arXiv"
[38-20]: https://arxiv.org/pdf/1909.11740.pdf "UNITER (2020) — arXiv"
[38-21]: https://arxiv.org/pdf/2004.06165.pdf "OSCAR (2020) — arXiv"
[38-22]: https://arxiv.org/pdf/2005.12872.pdf "DETR (2020) — arXiv"
[38-23]: https://arxiv.org/pdf/2010.11929.pdf "Vision Transformer (ViT) (2020) — arXiv"
[38-24]: https://arxiv.org/pdf/2010.04159.pdf "Deformable DETR (2020) — arXiv"
[38-25]: https://arxiv.org/pdf/2012.12877.pdf "DeiT (2020) — arXiv"
