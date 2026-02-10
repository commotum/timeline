1. **Number of distinct tasks evaluated:** 1

- “Experiments demonstrate that our method achieves state-ofthe-art performance for action recognition on both the NTU RGB+D 60 dataset and the NTU RGB+D 120 dataset.” (Abstract)
- “NTU RGB+D [56] is a large-scale benchmark dataset for action recognition...” (Section IV, **Dataset**)

2. **Number of trained model instances required to cover all tasks:** 1

- “As illustrated in Fig. 2, our proposed VG4D framework consists of 3 networks: 4D point cloud encoder  $E_P$ , video encoder  $E_V$  and text encoder  $E_T$  from VLM.” (Section III.A, **Overview of VG4D**)
- “Our VG4D also includes two classification heads to classify the 4D features and RGB video features extracted by im-PSTNet and Video encoder, respectively.” (Section III.B, **Cross-Modal Learning**)
- “In the testing phase, we ensemble the im-PSTNet with the VLM. Specifically, we fuse four 4D-text, RGB-text, 4D, and RGB scores as the final classification result.” (Section III.B, **Cross-Modal Learning**)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
