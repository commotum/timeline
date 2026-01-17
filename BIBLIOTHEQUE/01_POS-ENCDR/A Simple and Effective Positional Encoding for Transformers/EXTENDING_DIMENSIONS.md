## 1. Basic Metadata

- Title: "A Simple and Effective Positional Encoding for Transformers" (Document header)
- Authors: "Pu-Chin Chen*, Henry Tsai*, Srinadh Bhojanapalli*, Hyung Won Chung, Yin-Wen Chang, Chun-Sung Ferng" (Document header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper introduces DIET as "a simple yet effective mechanism to encode position and segment information into the Transformer models," motivated by the finding that "the gain actually comes from moving positional information to attention layer from the input" (Abstract).

---

## 3. Tasks Evaluated

- Task name: MNLI (GLUE). Task type: Classification. Dataset(s) used: "MNLI<br>393k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: QQP (GLUE). Task type: Classification. Dataset(s) used: "<b>QQP</b><br>364k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: QNLI (GLUE). Task type: Classification. Dataset(s) used: "QNLI<br>105k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: SST2 (GLUE). Task type: Classification. Dataset(s) used: "<b>SST2</b> 67k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: CoLA (GLUE). Task type: Classification. Dataset(s) used: "CoLA<br>8.5k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: STS-B (GLUE). Task type: Other (semantic textual similarity / regression). Dataset(s) used: "STS-B<br>7k" (Table 2: GLUE). Domain: natural language text (English) - "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). Quote: "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results)
- Task name: XNLI (XTREME). Task type: Classification. Dataset(s) used: "We conduct 5 trials of fine-tuning for each model on the MultiNLI (Williams et al., 2018) training data, then perform zero-shot predictions on XNLI (Conneau et al., 2018), choosing median accuracy to report." (Section 4.2 Cross-lingual Model Results). Domain: multilingual natural language text - "we pre-train the models on Wikipedia corpus in 100 languages similar to (Lample and Conneau, 2019)" (Section 4.2 Cross-lingual Model Results).
- Task name: XQuAD (XTREME). Task type: Other (extractive question answering). Dataset(s) used: "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages), choosing median F1 / EM scores to report." (Section 4.2 Cross-lingual Model Results). Domain: multilingual natural language text - "we pre-train the models on Wikipedia corpus in 100 languages similar to (Lample and Conneau, 2019)" (Section 4.2 Cross-lingual Model Results).
- Task name: MLQA (XTREME). Task type: Other (extractive question answering). Dataset(s) used: "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages), choosing median F1 / EM scores to report." (Section 4.2 Cross-lingual Model Results). Domain: multilingual natural language text - "we pre-train the models on Wikipedia corpus in 100 languages similar to (Lample and Conneau, 2019)" (Section 4.2 Cross-lingual Model Results).
- Task name: TyDiQA-GoldP (XTREME). Task type: Other (extractive question answering). Dataset(s) used: "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages), choosing median F1 / EM scores to report." (Section 4.2 Cross-lingual Model Results). Domain: multilingual natural language text - "we pre-train the models on Wikipedia corpus in 100 languages similar to (Lample and Conneau, 2019)" (Section 4.2 Cross-lingual Model Results).
- Task name: WMT18 en-de machine translation. Task type: Generation. Dataset(s) used: "We train using WMT18 ((Europarl v7, Common Crawl corpus and News Commentary v13) en-de, de-en, en-cs and cs-en datasets." (Appendix A Experimental setup). Domain: parallel natural language text - "Translate" (Appendix A Experimental setup).
- Task name: WMT18 de-en machine translation. Task type: Generation. Dataset(s) used: "We train using WMT18 ((Europarl v7, Common Crawl corpus and News Commentary v13) en-de, de-en, en-cs and cs-en datasets." (Appendix A Experimental setup). Domain: parallel natural language text - "Translate" (Appendix A Experimental setup).
- Task name: WMT18 en-cs machine translation. Task type: Generation. Dataset(s) used: "We train using WMT18 ((Europarl v7, Common Crawl corpus and News Commentary v13) en-de, de-en, en-cs and cs-en datasets." (Appendix A Experimental setup). Domain: parallel natural language text - "Translate" (Appendix A Experimental setup).
- Task name: WMT18 cs-en machine translation. Task type: Generation. Dataset(s) used: "We train using WMT18 ((Europarl v7, Common Crawl corpus and News Commentary v13) en-de, de-en, en-cs and cs-en datasets." (Appendix A Experimental setup). Domain: parallel natural language text - "Translate" (Appendix A Experimental setup).

---

## 4. Domain and Modality Scope

- Single domain? Multiple domains within the same modality (text): "We conduct experiments in three different settings to cover a wide range of use cases. First, we examine the results of a popular transfer learning approach from masked-LM pretraining to the end tasks in GLUE (Devlin et al., 2018). Second, we study zero-shot cross-lingual transferability of the multilingual pretrained models (Hu et al., 2020) to classification and question answering tasks in the XTREME benchmark (Hu et al., 2020). Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments)
- Multiple domains within the same modality? Yes; see the same three-setting evaluation across GLUE, XTREME, and machine translation (Section 4 Experiments).
- Multiple modalities? Not indicated; the evaluation settings are NLP text benchmarks (Section 4 Experiments).
- Domain generalization or cross-domain transfer? "we study zero-shot cross-lingual transferability of the multilingual pretrained models (Hu et al., 2020) to classification and question answering tasks in the XTREME benchmark (Hu et al., 2020)." (Section 4 Experiments)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MNLI (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| QQP (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| QNLI (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| SST2 (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| CoLA (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| STS-B (GLUE) | Not specified | Yes (fine-tuned) | Not specified | "For Finetuning tasks we use the datasets from the GLUE benchmark (Wang et al., 2019)." (Section 4.1 English Transfer Learning Results) |
| XNLI (XTREME) | Yes (same fine-tuned model used for zero-shot XNLI) | Yes | Not specified | "We conduct 5 trials of fine-tuning for each model on the MultiNLI (Williams et al., 2018) training data, then perform zero-shot predictions on XNLI (Conneau et al., 2018)" (Section 4.2 Cross-lingual Model Results) |
| XQuAD (XTREME) | Yes (shared across XQuAD/MLQA/TyDiQA after SQuAD fine-tune) | Yes | Not specified | "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages)" (Section 4.2 Cross-lingual Model Results) |
| MLQA (XTREME) | Yes (shared across XQuAD/MLQA/TyDiQA after SQuAD fine-tune) | Yes | Not specified | "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages)" (Section 4.2 Cross-lingual Model Results) |
| TyDiQA-GoldP (XTREME) | Yes (shared across XQuAD/MLQA/TyDiQA after SQuAD fine-tune) | Yes | Not specified | "We conduct 5 trials of finetuning for each model on SQuAD V1.1 dataset, following by zero-shot predictions on XQuAD (11 languages), MLQA (7 languages) and TyDiQA-GoldP (9 languages)" (Section 4.2 Cross-lingual Model Results) |
| WMT18 en-de | Not specified | No (trained from scratch) | Not specified | "Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments) |
| WMT18 de-en | Not specified | No (trained from scratch) | Not specified | "Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments) |
| WMT18 en-cs | Not specified | No (trained from scratch) | Not specified | "Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments) |
| WMT18 cs-en | Not specified | No (trained from scratch) | Not specified | "Lastly, we consider training Transformer models from scratch for machine translation." (Section 4 Experiments) |

---

## 6. Input and Representation Constraints

- Fixed or variable input resolution/length: "One drawback of absolute position encoding is that it requires fixed length of input sequence and does not directly capture relative positions to each word." (Section 2.2.2 Relative Position Encodings)
- Fixed number of tokens / max sequence length: "Each input is constructed with full sentences from documents, and packed up to the maximum sequence length." (Appendix A Experimental setup). Also: "Sequence Length     | 128      | 128                      | 512      | 512" (Table 7: Hyperparameters for all models).
- Tokenization constraints: "We apply sub-word tokenization on raw text data using WordPiece (Wu et al., 2016) with a 30,000 token vocabulary." (Section 4.1 English Transfer Learning Results). "We use language-independent tokenizer, Sentence Piece (Kudo and Richardson, 2018) model, with 120,000 token vocabulary to encode input text." (Section 4.2 Cross-lingual Model Results)
- Segment/multi-sentence inputs: "For multi-segment tasks, additional segment embeddings can be added just like the position embeddings (Devlin et al., 2018)." (Section 1 Introduction)
- Fixed dimensionality: "We use the same architecture as BERT<sub>BASE</sub> (Devlin et al., 2018) (L = 12, H = 768, A = 12) for our experiments." (Appendix A Experimental setup)
- Fixed patch size: Not specified.
- Padding/resizing requirements: Not specified.

---

## 7. Context Window and Attention Structure

- Maximum sequence length: "Sequence Length     | 128      | 128                      | 512      | 512" (Table 7: Hyperparameters for all models).
- Fixed or variable sequence length: "One drawback of absolute position encoding is that it requires fixed length of input sequence and does not directly capture relative positions to each word." (Section 2.2.2 Relative Position Encodings), and inputs are "packed up to the maximum sequence length." (Appendix A Experimental setup)
- Attention type: Global self-attention implied by full dot-product attention over the sequence: "$$\mathbf{A}^i = (\mathbf{X}\mathbf{W}_Q^i)(\mathbf{X}\mathbf{W}_K^i)^\top$$" (Section 2.1 Transformer)
- Mechanisms for computational cost: "For long sequence inputs, Transformers suffer from quadratic dependence of computational complexity with respect to the sequence length. A class of methods reduce this complexity by using a low rank projection of the input sequence for attention computation" (Section 3.4 Application to Long-range Transformers). "Linformer (Wang et al., 2020), which projects the attention key and value matrices to a lower dimension k during attention computation." (Section 3.4 Application to Long-range Transformers)

---

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: "Absolute position encodings are computed in the input layer and are summed with the input token embeddings." (Section 2.2.1 Absolute Position Encodings). "Shaw et al. (2018) proposed using relative position encoding instead of absolute position encoding, and add position embeddings to the key and optionally value projections instead of the input." (Section 2.2.2 Relative Position Encodings). "Raffel et al. (2020) use scalars to encode relative position between query and key indices and add directly to the attention scores matrix." (Section 2.2.2 Relative Position Encodings). "We propose the following simple absolute position encoding method that adds position information to the token attention matrix directly in each attention head." (Section 3.1 Decoupled Absolute Positional Attention). "$$\mathbf{A}_{i,j}^{\text{REL}} = (\mathbf{X}_{i:} \mathbf{W}_{Q}) (\mathbf{X}_{j:} \mathbf{W}_{K})^{\top} / \sqrt{d} + \mathbf{R}_{i-j} + E_{S}(S(i), S(j)).$$" (Section 3.2 Decoupled Relative Positional Attention)
- Where it is applied: "We denote the methods that add position/segment information directly to input token embeddings with *input*, and methods that add position/segment information directly in attention layer with *per-head*." (Section 4 Experiments)
- Fixed across experiments vs modified/ablated: "We compare the following positional encoding approaches - absolute positional embedding (Devlin et al., 2018), relative positional embedding (Shaw et al., 2018), combined absolute and relative positional encoding (Ke et al., 2020), relative scalar approach (Raffel et al., 2020), our proposed DIET-ABS and DIET-REL per-head positional encoding approaches." (Section 4 Experiments). "Earlier we present an ablation study on XTREME in Table 5 for decoupled positional attention variants." (Section D Additional Ablation Study on GLUE)

---

## 9. Positional Encoding as a Variable

- Core research variable? Yes: "In this paper we undertake a systematic study to understand different position encoding methods." (Section 1 Introduction)
- Multiple positional encodings compared? Yes: "We compare the following positional encoding approaches - absolute positional embedding (Devlin et al., 2018), relative positional embedding (Shaw et al., 2018), combined absolute and relative positional encoding (Ke et al., 2020), relative scalar approach (Raffel et al., 2020), our proposed DIET-ABS and DIET-REL per-head positional encoding approaches." (Section 4 Experiments)
- PE choice claimed to be not critical or secondary? Not claimed. The paper emphasizes PE placement: "Our analysis shows that the gain actually comes from moving positional information to attention layer from the input." (Abstract)

---

## 10. Evidence of Constraint Masking

- Model size(s): "We consider two different models -  $BERT_{BASE}$  model and a smaller model,  $BERT_{SMALL}$ , that has hidden size 512, 4 layers and 8 attention heads." (Section 3.3 Training and Inference Costs). "We use the same architecture as BERT<sub>BASE</sub> (Devlin et al., 2018) (L = 12, H = 768, A = 12) for our experiments." (Appendix A Experimental setup). "| Devlin et al. (2018)          | 110.1M     | -         | 84.8 | 178.9M       | -         | 55.3   |  |" (Table 6: Model Parameters)
- Dataset size(s): "MNLI<br>393k" (Table 2: GLUE), "<b>QQP</b><br>364k" (Table 2: GLUE), "QNLI<br>105k" (Table 2: GLUE), "<b>SST2</b> 67k" (Table 2: GLUE), "CoLA<br>8.5k" (Table 2: GLUE), "STS-B<br>7k" (Table 2: GLUE); "Classification XNLI 393k" (Table 3: XTREME), "MLQA<br>8k" (Table 3: XTREME), "TyDiQA 3.7k" (Table 3: XTREME)
- Attribution of gains: "Our analysis shows that the gain actually comes from moving positional information to attention layer from the input." (Abstract). Scaling model size or data as the primary cause of gains is not stated.

---

## 11. Architectural Workarounds

- Per-head positional attention: "We propose the following simple absolute position encoding method that adds position information to the token attention matrix directly in each attention head." (Section 3.1 Decoupled Absolute Positional Attention)
- Per-head segment attention: "We further also add segment information to the token attention instead of the input embeddings." (Section 3.1 Decoupled Absolute Positional Attention)
- Low-rank projection for long sequences: "A class of methods reduce this complexity by using a low rank projection of the input sequence for attention computation" (Section 3.4 Application to Long-range Transformers). "Linformer (Wang et al., 2020), which projects the attention key and value matrices to a lower dimension k during attention computation." (Section 3.4 Application to Long-range Transformers)
- Rank-k positional attention: "With the decoupled positional embedding, we can increase  $d_p$  to any width k to break the low-rank bottleneck shown in Theorem 1." (Section 3.1 Decoupled Absolute Positional Attention)
- Parameter sharing to manage parameters: "Previous works (Raffel et al., 2020; Ke et al., 2020; Shaw et al., 2018) used different sharing methods for the positional encodings to reduce the model parameters." (Section 4.4 Ablation Study)

---

## 12. Explicit Limitations and Non-Claims

Not specified.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: NLP text only, evaluated across GLUE, XTREME, and WMT (Section 4 Experiments).
> - Task structure: multiple NLP tasks (classification, QA, translation) with pretraining and fine-tuning plus some zero-shot cross-lingual transfer (Sections 4 and 4.2).
> - Representation rigidity: fixed max sequence length (128/512) and subword tokenization (Table 7; Section 4.1; Section 4.2).
> - Model sharing vs specialization: GLUE tasks are fine-tuned per task; XTREME QA shares a SQuAD-fine-tuned model across XQuAD/MLQA/TyDiQA; classification uses MultiNLI fine-tuning then XNLI zero-shot (Section 4.2).
> - Role of positional encoding: central research variable with multiple PE variants compared and ablated (Section 4 Experiments; Section D Additional Ablation Study on GLUE).

---

## 14. Final Classification

**Multi-task, single-domain.** The evaluation spans multiple NLP task types: "classification and question answering tasks in the XTREME benchmark" and "machine translation" alongside GLUE (Section 4 Experiments). All evaluations are on natural language text (English and multilingual) with no other modalities described.
