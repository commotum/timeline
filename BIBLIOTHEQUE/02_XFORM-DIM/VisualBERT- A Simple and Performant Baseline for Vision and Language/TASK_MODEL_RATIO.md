1. Number of distinct tasks evaluated: 5 (VQA, VCR Q  $\rightarrow$  A, VCR QA  $\rightarrow$  R, NLVR $^2$ , Flickr $^3$ 0K). Evidence: "We evaluate VisualBERT on four different types of vision-and-language applications: (1) Visual Question Answering (VQA 2.0) (Goyal et al., 2017), (2) Visual Commonsense Reasoning (VCR) (Zellers et al., 2019), (3) Natural Language for Visual Reasoning (NLVR $^2$ ) (Suhr et al., 2019), and (4) Region-to-Phrase Grounding (Flickr $^3$ 0K) (Plummer et al., 2015), each described in more details in the following sections and the appendix." (Section "## 4 EXPERIMENT"); "The task is decomposed into two multi-choice sub-tasks wherein we train individual models: question answering (Q  $\rightarrow$  A) and answer justification (QA  $\rightarrow$  R)." (Section "#### 4.2 VCR")
2. Number of trained model instances required to cover all tasks: 5 (one fine-tuned model per task, with VCR requiring two individual models for its two sub-tasks). Evidence: "It is pre-trained with a masked language modeling (Objective 1), and sentence-image prediction task (Objective 2), on caption data and then fine-tuned for different tasks." (Figure 2 caption); "The task is decomposed into two multi-choice sub-tasks wherein we train individual models: question answering (Q  $\rightarrow$  A) and answer justification (QA  $\rightarrow$  R)." (Section "#### 4.2 VCR")
3. Task-Model Ratio:

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
