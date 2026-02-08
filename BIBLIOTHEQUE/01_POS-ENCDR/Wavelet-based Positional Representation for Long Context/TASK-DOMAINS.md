# WAVELET-BASED POSITIONAL REPRESENTATION FOR LONG CONTEXT (Not specified in the paper.)
Source: Wavelet-based Positional Representation for Long Context.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (next-token prediction/perplexity) | tokens (text/code sequences) | 1D (t) | Capped | Static (inferred) | Direct (inferred) | tokens (next-token probabilities) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Multi-document question answering | tokens (question + multiple documents) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (answers) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Single-document question answering | tokens (question + single document) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (answers) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| Summarization | tokens (documents/dialogues) (inferred) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | tokens (summaries) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates one autoregressive language-modeling task and LongBench downstream long-context tasks, explicitly naming multi-document QA and single-document QA, with additional summarization evidence from dataset/metric entries. All supported tasks are text-token based and map to 1D (t) inputs and outputs. Sequence handling is Capped by explicit maximum lengths (for example, \(L_{\rm train}=512, 1024, 4096\)), while runtime attention control is Static and state usage is Direct based on the described transformer self-attention setup (inferred). Overall, the modality coverage is text-only with long-context extrapolation emphasis.

## Evidence
### Task: Language modeling (next-token prediction/perplexity)
- "We performed a comparative evaluation using a Transformer-based language model (Baevski & Auli, 2019)." (Section 6.1 Experimental Settings)
- "**Evaluation Metric** We use perplexity as our evaluation metric." (Section 6.1 Experimental Settings)
- "We used CodeParrot <sup>10</sup> for evaluation, which is good for long-distance testing because it requires an understanding of patterns and contextualization of information over long distances. <sup>11</sup>" (Section 7.1 EXPERIMENTAL SETTINGS)
- "The maximum allowable lengths of sequences were set to  $L_{\rm train} = 512$  and  $L_{\rm train} = 1024$ ." (Section 6.1 Experimental Settings)
- Inference: Output is treated as token prediction (hence 1D token output), and attention/state are labeled Static/Direct from the transformer self-attention description rather than an explicit taxonomy statement. Supporting text: "In a transformer model (Vaswani et al., 2017), the self-attention mechanism operates by projecting the input sequence into three distinct representations—queries (Q), keys (K), and values (V)—using learnable weight matrices." (Section 5.1 METHODOLOGY)

### Task: Multi-document question answering
- "Furthermore, the multi-document QA task and single-document QA task were evaluated on all datasets." (A.15 EVALUATION ON LONGBENCH)
- "The models pre-trained in Section 7 were evaluated on LongBench (Bai et al., 2024)." (A.15 EVALUATION ON LONGBENCH)
- "| HotpotQA        | 9,151   | F1      | 200     |" (Table 5)
- Inference: Input/output are represented as token sequences (question+documents to answer tokens), with 1D (t), Capped, Static, and Direct assigned from the language-model architecture and max-length setup. Supporting text: "The maximum allowable length of sequences in pre-training was set to  $L_{\rm train}=4096$ ." (Section 7.1 EXPERIMENTAL SETTINGS)

### Task: Single-document question answering
- "Furthermore, the multi-document QA task and single-document QA task were evaluated on all datasets." (A.15 EVALUATION ON LONGBENCH)
- "The models pre-trained in Section 7 were evaluated on LongBench (Bai et al., 2024)." (A.15 EVALUATION ON LONGBENCH)
- "| NarrativeQA     | 18,409  | F1      | 200     |" (Table 5)
- Inference: The OCR explicitly names the task intent (single-document QA), but does not explicitly restate representation labels; 1D (t), Capped, Static, Direct, and token-answer output are inferred from the same transformer+context-window setup used in Section 7.

### Task: Summarization
- "Table 5: Overview of the dataset statistics in LongBench (Bai et al., 2024). Avg len (average length) is computed using the number of words in the English." (Table 5)
- "| SAMSum          | 6258    | Rouge-L | 200     |" (Table 5)
- "| QMSum           | 10614   | Rouge-L | 200     |" (Table 5)
- Inference: Summarization is inferred from the SAMSum/QMSum rows and Rouge-L metric in Table 5. The A.15 prose explicitly names QA tasks, so summarization labeling is inference from table evidence rather than directly named task text.
