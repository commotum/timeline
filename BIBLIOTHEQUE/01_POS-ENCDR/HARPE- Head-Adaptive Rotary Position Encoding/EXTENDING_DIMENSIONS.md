## 1. Basic Metadata
- Title: Breaking the Stage Barrier: A Novel Single-Stage Approach to Long Context Extension for Large Language Models
- Authors: Haoran Lian; Junmin Chen; Wei Huang; Yizhe Xiong; Wenping Hu; Guiguang Ding; Hui Chen; Jianwei Niu; Zijia Lin; Fuzheng Zhang; Di Zhang
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
The paper introduces Head-Adaptive Rotary Position Encoding (HARPE), a single-stage continual pretraining method that assigns different RoPE base frequencies across attention heads to enable long-context modeling without multi-stage training.

## 3. Tasks Evaluated
- Task name: Perplexity evaluation on Proof-pile; Task type: Other (perplexity / language modeling evaluation); Dataset(s): Proof-pile; Domain: documents; Evidence: "Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets." (Section 4.3 Evaluation Metric) and "Table 3: Sliding window perplexity (S = 256) for **Proof**pile and GovReport documents." (Table 3 caption)
- Task name: Perplexity evaluation on GovReport; Task type: Other (perplexity / language modeling evaluation); Dataset(s): GovReport; Domain: documents; Evidence: "Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets." (Section 4.3 Evaluation Metric) and "Table 3: Sliding window perplexity (S = 256) for **Proof**pile and GovReport documents." (Table 3 caption)
- Task name: Needle-in-a-Haystack (including multi-key, multi-value, multi-query variants); Task type: Other (needle retrieval/recitation in long document); Dataset(s): Not specified; Domain: lengthy document; Evidence: 'Needle-in-a-Haystack is a task that assesses a model's ability to accurately locate and recite a specific sentence, referred to as the "needle", within a lengthy document, known as the "haystack".' (Section 4.3 Evaluation Metric) and 'we extend this method, inspired by RULER (Hsieh et al., 2024), to include multi-key, multi-value and multi-query scenarios, as well as diverse types of needles and background documents in each scenario.' (Section 4.3 Evaluation Metric)
- Task name: RULER benchmark (13 long-context tasks); Task type: Other (long-context evaluation benchmark; includes question answering); Dataset(s): RULER benchmark; Domain: long-context tasks (not otherwise specified); Evidence: 'RULER is a comprehensive and widely recognized standard for long-context evaluation, comprising 13 tasks that include "needle in a haystack" as well as additional tasks such as Variable Tracing, Aggregation Ability, and Question Answering.' (Section 5.3 Comparative Results on RULER Evaluation) and "In this section, we evaluate HARPE against various open-source pre-trained models on a range of long-context tasks using the RULER benchmark." (Section 5.3 Comparative Results on RULER Evaluation)
- Task name: MMLU (5-shot); Task type: Other (short-context evaluation dataset; task type not specified in text); Dataset(s): MMLU; Domain: Not specified (short-context evaluation dataset); Evidence: "We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017)." (Section 4.3 Evaluation Metric)
- Task name: Hellaswag (10-shot); Task type: Other (short-context evaluation dataset; task type not specified in text); Dataset(s): Hellaswag; Domain: Not specified (short-context evaluation dataset); Evidence: "We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017)." (Section 4.3 Evaluation Metric)
- Task name: ARC-Challenge (25-shot); Task type: Other (short-context evaluation dataset; task type not specified in text); Dataset(s): ARC-Challenge; Domain: Not specified (short-context evaluation dataset); Evidence: "We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017)." (Section 4.3 Evaluation Metric)
- Task name: PIQA (0-shot); Task type: Other (short-context evaluation dataset; task type not specified in text); Dataset(s): PIQA; Domain: Not specified (short-context evaluation dataset); Evidence: "We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017)." (Section 4.3 Evaluation Metric)
- Task name: TriviaQA (5-shot); Task type: Other (short-context evaluation dataset; task type not specified in text); Dataset(s): TriviaQA; Domain: Not specified (short-context evaluation dataset); Evidence: "We include five widely used short-context evaluation datasets: 5-shot MMLU (Hendrycks et al., 2020), 10-shot Hellaswag (Zellers et al., 2019), 25-shot ARC-Challenge (Clark et al., 2018), 0-shot PiQA (Bisk et al., 2019), and 5-shot TriviaQA (Joshi et al., 2017)." (Section 4.3 Evaluation Metric)

## 4. Domain and Modality Scope
- Evaluation is within a single modality (text) across multiple datasets in NLP; Evidence: "Recently, Large language models (LLMs) have revolutionized Natural Language Processing (NLP)." (Abstract) and "Perplexity (PPL) is evaluated on the Proof-pile (Zhangir Azerbayev, 2022) and GovReport (Huang et al., 2021) datasets." (Section 4.3 Evaluation Metric)
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
The paper does not explicitly state per-task fine-tuning or separate heads; it describes a single continual-pretraining setup.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Perplexity (Proof-pile) | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| Perplexity (GovReport) | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| Needle-in-a-Haystack | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| RULER benchmark | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| MMLU | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| Hellaswag | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| ARC-Challenge | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| PIQA | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |
| TriviaQA | Not specified | Not specified | Not specified | "We employ the Llama2-7B-Base model as the pre-trained backbone" (Section 4.4 Training Configuration); "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration) |

## 6. Input and Representation Constraints
- Base model context length and RoPE base: "We select LLama2-7B-Base (Touvron et al., 2023b) as our base model, which is configured with a RoPE base frequency of 10k and a context length of 4k." (Experimental Setup)
- Long-context input length: "Concurrently, the input sequence length is increased to 128k." (Section 4.1 Baseline Systems)
- Multi-stage context lengths in baselines: "we divide the process into three stages: (1)b = 1m; l = 32k, (2)b = 2m; l = 64k, and(3)b = 5m; l = 128k." (Section 4.1 Baseline Systems)
- Proof-pile evaluation length constraints: "Following the setup in Yarn, for the Proof-pile dataset, we selected samples with a minimum of 128k tokens and measured perplexity for token lengths ranging from 2k to 128k in increments of 2k, averaging the scores for each length." (Section 4.3 Evaluation Metric)
- GovReport evaluation context window: "For the GovReport dataset, we reported the average PPL scores for samples with a context window of 32k tokens." (Section 4.3 Evaluation Metric)
- Sliding window evaluation size: "Evaluations are conducted using the sliding window method proposed by Press (Press et al., 2021), with a window size of 256 tokens." (Section 4.3 Evaluation Metric)
- Self-Extend evaluation parameters: "We apply it with  $window\_size = 1024$ and  $group\_size = 32$ ." (Section 4.1 Baseline Systems)
- Fixed patch size / image resolution / 2D grid assumptions: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: "Concurrently, the input sequence length is increased to 128k." (Section 4.1 Baseline Systems) and "ranging up to 128k tokens" (Figure 2 caption)
- Sequence length fixed or variable: "token lengths ranging from 2k to 128k in increments of 2k" (Section 4.3 Evaluation Metric) indicates variable evaluation lengths.
- Attention type (global/windowed/hierarchical/sparse): Not specified for HARPE; baseline method described as "Self-Extend (Jin et al., 2024) constructs a two-layer attention mechanism, consisting of group attention and neighbor attention, to successfully expand the context window without additional training." (Section 2 Related Works)
- Mechanisms to manage computational cost: Not specified for HARPE; evaluation uses a sliding window ("window size of 256 tokens") and Self-Extend uses group/neighbor attention (Section 4.3 Evaluation Metric; Section 2 Related Works).

## 8. Positional Encoding (Critical Section)
- Mechanism: RoPE with head-specific base values in HARPE; Evidence: "Our HARPE leverages different Rotary Position Encoding (RoPE) base frequency values across different attention heads" (Abstract) and "HARPE assigns a unique base  $b_h$  from B to the RoPE in each attention head h." (Section 3.2)
- Where applied: Positional encoding is applied to query/key embeddings; Evidence: "f represents the positional encoding function applied to the query embeddings  $q_m$  at position m and key embeddings  $k_n$  at position n." (Section 3.1 Preliminaries)
- Fixed vs. modified/ablated: Modified and compared across settings; Evidence: "we evaluate the performance of two base selection methods for the head-specific RoPE bases in HARPE" and "we test five variations with different base strides (10k, 20k, 30k, 40k, 50k)" (Section 5.2 Study of Various Base Schemes), plus "We compare HARPE with 4 continual pretraining methods and one training-free method." (Section 4.1 Baseline Systems)

## 9. Positional Encoding as a Variable
- Core research variable: Yes; Evidence: "In this paper, we introduce a novel single-stage continual pretraining method, Head-Adaptive Rotary Position Encoding (HARPE)" (Abstract) and "we propose a novel training strategy that distributes the training of different stages across multiple attention heads." (Section 1 Introduction)
- Multiple positional encodings compared: Yes; Evidence: "We compare HARPE with 4 continual pretraining methods and one training-free method." (Section 4.1 Baseline Systems)
- Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking
- Model sizes reported: "We select LLama2-7B-Base (Touvron et al., 2023b) as our base model" (Experimental Setup) and "our comparison in 10 base models primarily involves 7B models" (Section 5.3 Comparative Results on RULER Evaluation)
- Dataset size / training scale: "All models were continually pre-trained with 6B tokens using these consistent settings." (Section 4.4 Training Configuration)
- Long-context scale in evaluation/training: "Concurrently, the input sequence length is increased to 128k." (Section 4.1 Baseline Systems)
- Performance gains attributed to architectural/training strategy rather than scaling data or model size: "HARPE achieves a significant improvement of 5.46% over the multi-stage Adjused Base Frequency (ABF) (Xiong et al., 2023) approach" and "we propose a novel training strategy that distributes the training of different stages across multiple attention heads." (Section 1 Introduction)
- Sensitivity to training pipeline: "a carefully scheduled three-stage pipeline outperforms a naive approach by 13.5% on the NiaH benchmark" (Section 1 Introduction)

## 11. Architectural Workarounds
- Head-adaptive RoPE across attention heads to simulate staged training in one pass: "we propose to distribute the training of different stages across multiple attention heads concurrently" and "HARPE assigns a unique base  $b_h$  from B to the RoPE in each attention head h." (Section 1 Introduction; Section 3.2)
- Multi-stage training pipelines in prior work: "existing works commonly employ a multi-stage approach, progressively increasing the context length through a series of continued pretraining steps." (Section 1 Introduction)
- Self-Extend baseline uses grouped/neighbor attention to expand context without extra training: "Self-Extend (Jin et al., 2024) constructs a two-layer attention mechanism, consisting of group attention and neighbor attention, to successfully expand the context window without additional training." (Section 2 Related Works)

## 12. Explicit Limitations and Non-Claims
- Limitation/future work: "Our research is primarily concentrated on the continual pretraining stage, leaving its applicability to other stages, such as supervised fine-tuning, unexplored." (Section 7 Limitations)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or cross-domain transfer: Not stated.

### 13. Constraint Profile (Synthesis)
> **Constraint Profile:**
> - Domain scope: NLP text only, evaluated on multiple datasets/benchmarks within the same modality.
> - Task structure: multiple benchmark tasks (perplexity, Needle-in-a-Haystack, RULER tasks, short-context datasets) rather than a single task.
> - Representation rigidity: explicit token context lengths (4k base; up to 128k in training/evaluation) and fixed evaluation windows.
> - Model sharing vs specialization: single continual-pretraining setup described; no per-task fine-tuning or task-specific heads stated.
> - Role of positional encoding: central experimental variable (head-specific RoPE bases; multiple base-selection schemes compared).

### 14. Final Classification
Classification: **Multi-task, single-domain.** The evaluation spans multiple tasks/benchmarks including "Perplexity (PPL)" on "Proof-pile" and "GovReport" datasets, RULER's "13 tasks that include \"needle in a haystack\" as well as additional tasks such as Variable Tracing, Aggregation Ability, and Question Answering," and short-context datasets like "5-shot MMLU" and "10-shot Hellaswag" (Sections 4.3, 5.3). The work is framed within NLP ("Recently, Large language models (LLMs) have revolutionized Natural Language Processing (NLP).") and does not claim cross-domain or multi-modal transfer.
