1. Number of distinct tasks evaluated: 6. "We use Kinetics-400 [59] (K400) (~240k training videos in 400 classes) and Kinetics-600 [11]." (Section 4. Experiments: Video Recognition) "We further assess transfer learning performance for on Something-Something-v2 [38], Charades [86], and AVA [39]." (Section 4. Experiments: Video Recognition) "We apply our video models on static image recognition by using them with single frame, T=1, on ImageNet-1K [22]." (Section 5. Experiments: Image Recognition)
2. Number of trained model instances required to cover all tasks: 6. "For Kinetics, we train for 200 epochs with 2 repeated augmentation [50] repetitions." (Section 4. Experiments: Video Recognition) "For **Kinetics-600** all hyper-parameters are identical to K400." (Section D.1. Details: Kinetics Action Classification) "We fine-tune our MViT models from the Kinetics models." (Section D.3. Details: Charades Action Classification) "We fine-tune the pre-trained Kinetics models." (Section D.4. Details: Something-Something V2 (SSv2)) "We initialize the network weights from the Kinetics models and adopt synchronized SGD training on 64 GPUs." (Section D.2. Details: AVA Action Detection) "We train models on the train set and report top-1 and top-5 classification accuracy (%) on the val set." (Section D.5. Details: ImageNet)
3. Task–Model Ratio:
$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
