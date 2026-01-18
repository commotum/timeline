## 1. Basic Metadata
- Title: "WAVELET-BASED POSITIONAL REPRESENTATION FOR LONG CONTEXT" (Title line)
- Authors: "Yui Oka, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper proposes "a new position representation method that captures multiple scales (i.e., window sizes) by leveraging wavelet transforms without limiting the model's attention field" to improve extrapolation for long contexts (Abstract).

## 3. Tasks Evaluated
- Task name: Language modeling perplexity / extrapolation on WikiText-103
  - Task type: Other (language modeling / perplexity)
  - Dataset(s) used: WikiText-103
  - Domain: natural language (English Wikipedia text)
  - Quotes:
    - "We used the WikiText-103 dataset (Merity et al., 2017), which consists of over 103 million tokens of English Wikipedia articles." (Section 6.1 Experimental Settings)
    - "**Evaluation Metric** We use perplexity as our evaluation metric." (Section 6.1 Experimental Settings)
    - "Table 1: Perplexity of validation set in extrapolation experiments using Wikitext-103." (Table 1)
- Task name: Long-context evaluation on CodeParrot (perplexity)
  - Task type: Other (language modeling / long-range dependency evaluation)
  - Dataset(s) used: CodeParrot
  - Domain: text sequences (domain not explicitly described beyond dataset name)
  - Quotes:
    - "We used CodeParrot <sup>10</sup> for evaluation, which is good for long-distance testing because it requires an understanding of patterns and contextualization of information over long distances. <sup>11</sup>" (Section 7.1 Experimental Settings)
    - "Table 2: Perplexity in Non-overlapping Inference with  $L_{\text{train}} = 4096$ ." (Table 2)
- Task name: LongBench evaluation (multi-document QA and single-document QA)
  - Task type: Other (question answering)
  - Dataset(s) used: NarrativeQA, Qasper, MultiFieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, TriviaQA, SAMSum, QMSum
  - Domain: natural language (English)
  - Quotes:
    - "The models pre-trained in Section 7 were evaluated on LongBench (Bai et al., 2024)." (A.15 Evaluation on LongBench)
    - "Furthermore, the multi-document QA task and single-document QA task were evaluated on all datasets." (A.15 Evaluation on LongBench)
    - "Since pre-training was conducted using an English dataset, evaluation was conducted using only the English dataset." (A.15 Evaluation on LongBench)
    - "Table 5: Overview of the dataset statistics in LongBench (Bai et al., 2024). Avg len (average length) is computed using the number of words in the English." (Table 5)
    - "| Dataset         | Avg len | Metric  | Samples |" (Table 5)
    - "| NarrativeQA     | 18,409  | F1      | 200     |" (Table 5)
    - "| Qasper          | 3,619   | F1      | 200     |" (Table 5)
    - "| MultiFieldQA-en | 4,559   | F1      | 150     |" (Table 5)
    - "| HotpotQA        | 9,151   | F1      | 200     |" (Table 5)
    - "| 2WikiMQA        | 4,887   | F1      | 200     |" (Table 5)
    - "| MuSiQue         | 11,214  | F1      | 200     |" (Table 5)
    - "| TriviaQA        | 8,209   | F1      | 200     |" (Table 5)
    - "| SAMSum          | 6258    | Rouge-L | 200     |" (Table 5)
    - "| QMSum           | 10614   | Rouge-L | 200     |" (Table 5)

## 4. Domain and Modality Scope
- Evaluation performed on: Multiple domains within the same modality (text). Evidence: "We used the WikiText-103 dataset (Merity et al., 2017), which consists of over 103 million tokens of English Wikipedia articles." (Section 6.1 Experimental Settings); "We used CodeParrot <sup>10</sup> for evaluation, which is good for long-distance testing because it requires an understanding of patterns and contextualization of information over long distances. <sup>11</sup>" (Section 7.1 Experimental Settings); "Since pre-training was conducted using an English dataset, evaluation was conducted using only the English dataset." (A.15 Evaluation on LongBench)
- Multiple modalities: Not stated; only text datasets are described.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| WikiText-103 language modeling | Not specified (short-context experiment) | Not specified | Not specified | "First, we conducted a small-scale experiment to compare our approach with various position encodings." (Section 6.1 Experimental Settings) |
| CodeParrot evaluation | Yes (Section 7 pre-trained model) | Not specified | Not specified | "We pre-trained the Llama-2-7B<sup>9</sup> model from scratch." and "We used CodeParrot <sup>10</sup> for evaluation, which is good for long-distance testing because it requires an understanding of patterns and contextualization of information over long distances. <sup>11</sup>" (Section 7.1 Experimental Settings) |
| LongBench QA | Yes (Section 7 pre-trained model) | Not specified | Not specified | "The models pre-trained in Section 7 were evaluated on LongBench (Bai et al., 2024)." (A.15 Evaluation on LongBench) |

## 6. Input and Representation Constraints
- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens / max length: "The maximum allowable lengths of sequences were set to  $L_{\rm train} = 512$  and  $L_{\rm train} = 1024$ ." (Section 6.1 Experimental Settings); "The maximum allowable length of sequences in pre-training was set to  $L_{\rm train}=4096$ ." (Section 7.1 Experimental Settings); "The maximum sequence length is  $L_{max}=512$ , and the sequence length at inference is L=1012." (A.11)
- Fixed dimensionality: "The dimensionality of the word embedding  $d_{model}$  is 1024, the number of heads N is 8, the dimensionality of the heads d is 128, and the number of layers is 16." (Section 6.1 Experimental Settings); "The dimensionality of the word embedding  $d_{model}$  is 4096, the number of heads N is 32, the dimensionality of the heads d is 128, and the number of layers is 32." (A.8)
- Positional representation range: "In our method, there is no clipping, and the distance of the position expression is fixed regardless of the length of the sentence." (Section 5.1)
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length / context window: "The maximum allowable lengths of sequences were set to  $L_{\rm train} = 512$  and  $L_{\rm train} = 1024$ ." (Section 6.1 Experimental Settings); "The maximum allowable length of sequences in pre-training was set to  $L_{\rm train}=4096$ ." (Section 7.1 Experimental Settings); "The maximum sequence length is  $L_{max}=512$ , and the sequence length at inference is L=1012." (A.11); "Table 2: Perplexity in Non-overlapping Inference with  $L_{\text{train}} = 4096$ ." and "|                           | Sequence Length |      |      |      |  |" with "|                           | 4 k             | 8 k  | 16 k | 32 k |  |" (Table 2)
- Fixed or variable length: "To evaluate sequences longer than  $L_{\rm train}$  tokens, it is common to divide the sequence into  $L_{\rm train}$ -length sub-sequences, evaluate each independently, and report the average score." (Section 6.1 Experimental Settings)
- Attention type: Unrestricted/global attention for the proposed method is implied by "our method allows extrapolation of position information without limiting the model's attention field." (Abstract) and "the proposed method has demonstrated its superiority at capturing long dependencies without restricting the receptive field of attention." (Section 6.3.2); windowed attention is described for ALiBi: "ALiBi has a restricted receptive field and behaves in the manner of windowed attention." (Section 4)
- Computational cost management: "we report not only the perplexity of non-overlapping inference but also the normal perplexity when the sequence is not divided into partial sequences." (Section 6.1 Experimental Settings); "we further reduce the memory usage to (d, length) by using torch.scatter to scatter the wavelet position representation to the attention mask." (A.6)

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Relative position representation with wavelet functions. Evidence: "we adopt relative position representation using ALiBi because it is more suitable than absolute position representation." (Section 5.1); "RPE(Shaw et al., 2018) expresses position by calculating the inner product of the query and the relative position embedding. We incorporate the wavelet function into RPE as follows." (Section 5.1)
- Where it is applied: In attention score computation: "e_{m,n} = \frac{q_m k_n^T + q_m (p_{m,n})^T}{\sqrt{d}}," (Section 5.1)
- Fixed vs modified per task / comparisons: "First, we conducted a small-scale experiment to compare our approach with various position encodings." and "In addition to ALiBi and RoPE, the following position representations were also compared: NoPE (Kazemnejad et al., 2023), in which position information is given, and TransXL (Dai et al., 2019), which is a relative positional representation that uses sine waves." (Section 6.1 Experimental Settings); "We also conducted experiments to see whether the same effect could be obtained with other wavelets." (Section 6.3.1)
- Positional encoding specifics: "Instead of using learnable embeddings to represent  $p_{m,n}$ , we use d-pattern wavelet functions with multiple scales to calculate the position." (Section 5.1)

## 9. Positional Encoding as a Variable
- Core research variable? Yes: "First, we conducted a small-scale experiment to compare our approach with various position encodings." (Section 6.1 Experimental Settings)
- Multiple positional encodings compared? Yes: "In addition to ALiBi and RoPE, the following position representations were also compared: NoPE (Kazemnejad et al., 2023), in which position information is given, and TransXL (Dai et al., 2019), which is a relative positional representation that uses sine waves." (Section 6.1 Experimental Settings)
- Ablations or alternatives within wavelet PE? Yes: "We also conducted experiments to see whether the same effect could be obtained with other wavelets." (Section 6.3.1); "ablation study focusing on the shift and scale parameters of the Ricker and Gaussian wavelets." (A.13)
- Claim that PE choice is not critical or secondary? Not stated.

## 10. Evidence of Constraint Masking
- Model size(s): "The dimensionality of the word embedding  $d_{model}$  is 1024, the number of heads N is 8, the dimensionality of the heads d is 128, and the number of layers is 16." (Section 6.1 Experimental Settings); "We pre-trained the Llama-2-7B<sup>9</sup> model from scratch." (Section 7.1 Experimental Settings); "The dimensionality of the word embedding  $d_{model}$  is 4096, the number of heads N is 32, the dimensionality of the heads d is 128, and the number of layers is 32." (A.8)
- Dataset size(s): "We used the WikiText-103 dataset (Merity et al., 2017), which consists of over 103 million tokens of English Wikipedia articles." (Section 6.1 Experimental Settings); "For pre-training, we used the RedPajama dataset (Computer, 2023), which selects a 1B-token sample of all samples." (Section 7.1 Experimental Settings)
- Attribution of gains: The paper emphasizes architectural scale/shift parameters rather than model/data scaling: "This demonstrates the importance of having multiple scales, or in this case, window sizes." (Section 6.3.1) and "These findings underscore the significance of the scale parameters in extrapolation." (A.13)
- Capacity-related constraints: "In this experiment, due to the large model size and long sequence length, we report perplexity only for non-overlapping inference using  $L_{\rm train}$ , since the memory capacity is exceeded." (A.8)

## 11. Architectural Workarounds
- Sequence splitting for long inputs: "To evaluate sequences longer than  $L_{\rm train}$  tokens, it is common to divide the sequence into  $L_{\rm train}$ -length sub-sequences, evaluate each independently, and report the average score." (Section 6.1 Experimental Settings)
- Non-overlapping inference regime: "we report not only the perplexity of non-overlapping inference but also the normal perplexity when the sequence is not divided into partial sequences." (Section 6.1 Experimental Settings)
- Memory reduction for long contexts: "we further reduce the memory usage to (d, length) by using torch.scatter to scatter the wavelet position representation to the attention mask." (A.6)
- Decoder-only relative positions: "In the relative position representation in the decoder, only the position information of the token before the current token is required, for example, 0, -1, -2, etc." (A.6)
- Windowed attention baseline: "ALiBi has a restricted receptive field and behaves in the manner of windowed attention." (Section 4)

## 12. Explicit Limitations and Non-Claims
- Training cost limitation: "Unfortunately, we had to halt the learning process because it took over five times longer than anticipated." (A.6)
- Computational overhead: "However, the computational overhead of calculating relative positions may still impose a bottleneck, and thus reducing it is an important direction for future work." (Conclusion)
- Language limitation: "Since pre-training was conducted using an English dataset, evaluation was conducted using only the English dataset." (A.15 Evaluation on LongBench)
- Data scale limitation / future work: "Note that this is an evaluation of a model that was pre-trained on a small dataset (redpajama-1B). As future work, it will be necessary to pre-train the model with a larger dataset and conduct evaluations with other models that are effective for long sentences, such as LongRangeArena (Tay et al., 2021)." (A.15 Evaluation on LongBench)
- Explicit non-claims about open-world learning or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only evaluations (English Wikipedia, CodeParrot, LongBench English datasets).
> - Task structure: Language-modeling perplexity plus LongBench QA evaluations; tasks are evaluated separately.
> - Representation rigidity: Fixed maximum sequence lengths (L_train), fixed head dimensions, and fixed relative position distance.
> - Model sharing vs specialization: Separate short-context experiment; Section 7 pre-trained model reused for CodeParrot and LongBench; no fine-tuning noted.
> - Role of positional encoding: Central experimental variable with comparisons across PE types and wavelet-scale/shift ablations.

### 14. Final Classification

Multi-task, multi-domain (constrained). The paper evaluates multiple tasks, including language-modeling perplexity on WikiText-103 ("We used the WikiText-103 dataset (Merity et al., 2017), which consists of over 103 million tokens of English Wikipedia articles."; "**Evaluation Metric** We use perplexity as our evaluation metric.") and QA tasks on LongBench ("the multi-document QA task and single-document QA task were evaluated on all datasets"). It also evaluates on CodeParrot ("We used CodeParrot <sup>10</sup> for evaluation, which is good for long-distance testing because it requires an understanding of patterns and contextualization of information over long distances. <sup>11</sup>"), indicating multiple text domains within the same modality, and it restricts evaluation to English ("evaluation was conducted using only the English dataset"), keeping the setup constrained.
