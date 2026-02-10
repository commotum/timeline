1. **Number of distinct tasks evaluated:** 7

"### 5.1. Image-Text Retrieval" (Section 5.1)
"#### **5.2.** Image Captioning" (Section 5.2)
"#### 5.3. Visual Question Answering (VQA)" (Section 5.3)
"#### **5.4.** Natural Language Visual Reasoning (NLVR<sup>2</sup>)" (Section 5.4)
"## 5.5. Visual Dialog (VisDial)" (Section 5.5)
"In Table 10 and Table 11, we perform zero-shot transfer to *text-to-video retrieval* and *video question answering*" (Section 5.6)

2. **Number of trained model instances required to cover all tasks:** 5

"we briefly introduce each task and finetuning strategy." (Section 5)
"We finetune the pre-trained model using ITC and ITM losses." (Section 5.1)
"both evaluated using the model finetuned on COCO with the LM loss." (Section 5.2)
"The VQA model is finetuned with the LM loss using ground-truth answers as targets." (Section 5.3)
"An MLP classifier is applied on the output embedding of the [Encode] token." (Section 5.4)
"The dialog encoder is trained with the ITM loss to discriminate whether the answer is true or false for a question, given the entire dialog history and the image-caption embeddings." (Section 5.5)
"where we directly evaluate the models trained on COCO-retrieval and VQA, respectively." (Section 5.6)

3. **Task–Model Ratio**

$$
\boxed{
\frac{7\ \text{tasks}}{5\ \text{models}} = 1.4
}
$$
