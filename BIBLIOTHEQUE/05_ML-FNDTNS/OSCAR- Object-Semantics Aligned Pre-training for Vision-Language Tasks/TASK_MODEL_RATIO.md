1. **Number of distinct tasks evaluated:** 7

   "We adapt the pre-trained models to seven downstream V+L tasks, including five understanding tasks and two generation tasks." (Section 4: Adapting to V+L Tasks)

   "Image-Text Retrieval heavily relies on the joint representations. There are two sub-tasks: image retrieval and text retrieval, depending on which modality is used as the retrieved target." (Section 4: Adapting to V+L Tasks)

   "Image Captioning requires the model to generate a natural language description of the content of an image." (Section 4: Adapting to V+L Tasks)

   "Novel Object Captioning (NoCaps) [1] extends the image captioning task" (Section 4: Adapting to V+L Tasks)

   "VQA [9] requires the model to answer natural language questions based on an image." (Section 4: Adapting to V+L Tasks)

   "GQA [13] is similar to VQA, except that GQA tests the reasoning capability of the model to answer a question." (Section 4: Adapting to V+L Tasks)

   "Natural Language Visual Reasoning for Real (NLVR2) [36] takes a pair of images and a natural language, and the goal is to determine whether the natural language statement is true about the image pair." (Section 4: Adapting to V+L Tasks)

2. **Number of trained model instances required to cover all tasks:** 6

   "Given that our method is based on single task fine-tuning" (Section 5.1: Performance Comparison with SoTA)

   "During training, we formulate it as a binary classification problem." (Section 4: Adapting to V+L Tasks, Image-Text Retrieval)

   "When fine-tuning on the VQA task, we construct one input sequence, which contains the concatenation of a given question, object tags and region features, and then the [CLS] output from OSCAR is fed to a task-specific linear classifier for answer prediction." (Section 4: Adapting to V+L Tasks)

   "We develop two fine-tuned models using  $Oscar_B$ ." (Section 4: Adapting to V+L Tasks, GQA)

   "When fine-tuning on the NLVR2 task, we first construct two input sequences, each containing the concatenation of the given sentence (the natural language description) and one image, and then two [CLS] outputs from OSCAR are concatenated as the joint input for a binary classifier, implemented by an MLP<sup>5</sup>." (Section 4: Adapting to V+L Tasks)

   "To enable sentence generation, we fine-tune OSCAR using the seq2seq objective." (Section 4: Adapting to V+L Tasks)

   "Following the restriction guideline of NoCaps, we use the predicted Visual Genome and Open Images labels to form tag sequences, and train OSCAR on COCO without the initialization of pre-training." (Section 4: Adapting to V+L Tasks)

3. **Task-Model Ratio = (1) / (2)**

$$
\boxed{
\frac{7\ \text{tasks}}{6\ \text{models}} = 1.17
}
$$
