# DECOUPLING THE "WHAT" AND "WHERE" WITH POLAR COORDINATE POSITIONAL EMBEDDING (Not specified in the paper.)
Source: Decoupling the -What- and -Where- With Polar Coordinate Positional Embeddings (PoPE).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target character prediction (Indirect Indexing) | character-level tokens: source string; source character; shift | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | target character token | 0D (inferred) | Fixed (inferred) |
| autoregressive sequence modeling (symbolic music) | MIDI-based token sequence (JSB/MAESTRO) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next-token prediction over music tokens (inferred) | 1D (t) (inferred) | Capped (inferred) |
| autoregressive sequence modeling (human genome) | genomic sequence tokens (Human Reference Genome) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next-token prediction (genome tokens) | 1D (t) (inferred) | Capped (inferred) |
| language modeling (OpenWebText; PG-19 length extrapolation) | text tokens (OpenWebText; GPT-2 tokenizer) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | next-token prediction (text tokens) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| zero-shot last-word prediction (LAMBADA) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | last word (can be multiple tokens) | Not specified in the paper. | Not specified in the paper. |
| zero-shot evaluation (BLiMP) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| zero-shot evaluation (Children's Book Test (CBT)) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| zero-shot evaluation (HellaSwag) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| zero-shot evaluation (PIQA) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| zero-shot evaluation (ARC-E) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates a diagnostic character-indexing task and autoregressive sequence modeling in symbolic music, genomics, and natural language, plus zero-shot downstream benchmarks (LAMBADA, BLiMP, CBT, HellaSwag, PIQA, ARC-E). For the modeling tasks, inputs are 1D token sequences with capped maximum lengths (e.g., 40, 1024, 2048, 1000) and outputs are token predictions (single target token for Indirect Indexing; token sequences for modeling). The downstream benchmarks are named but their inputs/outputs and interface dynamics are largely unspecified, aside from LAMBADA's last-word prediction.

## Evidence
### Task: target character prediction (Indirect Indexing)
- "Indirect Indexing. This diagnostic task (Indirect Idx.) requires the model to locate a target character in a variable-length source string that is at a certain relative distance (left or right) from a source character." (Section A.1 DATASETS)
- "We generate source strings of length between 20 and 40 characters from the set of uppercase [A-Z] and lowercase [a-z] letters by uniform sampling without replacement." (Section A.1 DATASETS)
- "The format of each examples is: <source string>, <source character>, <shift>, <target character> and ',' as a delimiter and the model is given the entire sequence except the target character." (Section A.1 DATASETS)
- "We compare RoPE (Su et al., 2024) against PoPE by training two Transformer models with cross-entropy loss applied only on the final (target) token and evaluated on the accuracy of final token." (Section 4 RESULTS)
- Inference: Assigned `In Dimension = 1D (t)` and `In Dynamics = Capped` because inputs are a "variable-length source string" with "length between 20 and 40 characters"; assigned `Attention Dynamic = Static` and `State Dynamic = Direct` based on "decoder-only Transformer architecture ... with causal masking for autoregressive sequence modeling"; assigned `Out Dimension = 0D` and `Out Dynamics = Fixed` because prediction is the "final (target) token." (Sections A.1 DATASETS, 4 RESULTS)

### Task: autoregressive sequence modeling (symbolic music)
- "Sequence modeling of symbolic music. We train Transformer models using cross-entropy loss on MIDI-based inputs with a maximum length of 2048 from two popular music datasets, Bach-Chorales (JSB) (Boulanger-Lewandowski et al., 2012) and MAESTRO (Hawthorne et al., 2019)." (Section 4 RESULTS)
- "Bach-Chorales. This dataset (JSB) consists of 4-part scored choral music, which are represented as a matrix with rows corresponding to voices and columns to time discretized to 16th notes." (Section A.1 DATASETS)
- "We serialize this matrix in raster-scan fashion by first going down the rows and then moving right through the columns as in prior work (Huang et al., 2019)." (Section A.1 DATASETS)
- "We use a maximum sequence length of 2048 for training with 229/76/77 sequences present in the train/validation/test sets." (Section A.1 DATASETS)
- Inference: Treated the task as 1D sequence modeling because the music matrix is serialized into a sequence; marked dynamics as capped based on the maximum length of 2048; marked attention/state as static/direct based on "decoder-only Transformer architecture ... with causal masking for autoregressive sequence modeling"; and treated the output as next-token prediction from "cross-entropy loss on MIDI-based inputs." (Sections A.1 DATASETS, 4 RESULTS)

### Task: autoregressive sequence modeling (human genome)
- "Sequence modeling of human genome. We train a Transformer on sequences from the Human Reference Genome dataset (Dalla-Torre et al., 2025) using the standard next-token prediction loss." (Section 4 RESULTS)
- "| Sequence length     | 40            | 1024        | 2048 | 2048    | 1000    |" (Section A.3 TRAINING DETAILS)
- Inference: Assigned 1D input and capped dynamics because the task is on genome "sequences" with a listed sequence length of 1000; marked attention/state as static/direct based on "decoder-only Transformer architecture ... with causal masking for autoregressive sequence modeling"; and set output dimensionality/dynamics to a 1D capped token sequence because the loss is "next-token prediction." (Sections 4 RESULTS, A.3 TRAINING DETAILS)

### Task: language modeling (OpenWebText; PG-19 length extrapolation)
- "Language modeling on OpenWebText. We test PoPE's efficacy on language modeling by training Transformers of three sizes on the OpenWebText dataset (Gokaslan & Cohen, 2019)." (Section 4 RESULTS)
- "The training and validation splits roughly contain 9B and 4M tokens respectively and maximum sequence length of 1024 for pretraining." (Section A.1 DATASETS)
- "We examine models pretrained on OpenWebText using a sequence length (context window) of 1024 tokens and assess zero-shot perplexity on much longer sequences (up to 10240 tokens) from the test split of the PG-19 dataset (Rae et al., 2020)." (Section 4 RESULTS)
- Inference: Labeled inputs/outputs as 1D token sequences with capped dynamics based on the stated context window and sequence lengths; marked attention/state as static/direct based on "decoder-only Transformer architecture ... with causal masking for autoregressive sequence modeling"; and treated the output as next-token prediction because it is "language modeling" with "autoregressive sequence modeling." (Sections 4 RESULTS, A.1 DATASETS)

### Task: zero-shot last-word prediction (LAMBADA)
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)
- "Following Gao et al. (2024), we use the detokenized version from OpenAI for LAMBADA and evaluate the top-one accuracy on the last word (which can be multiple tokens; we use greedy decoding)." (Section 4 RESULTS)

### Task: zero-shot evaluation (BLiMP)
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

### Task: zero-shot evaluation (Children's Book Test (CBT))
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

### Task: zero-shot evaluation (HellaSwag)
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

### Task: zero-shot evaluation (PIQA)
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

### Task: zero-shot evaluation (ARC-E)
- "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

## CSV Output (required)
Write a CSV file to "/home/jake/Developer/timeline/BIBLIOTHEQUE/01_POS-ENCDR/Decoupling the -What- and -Where- With Polar Coordinate Positional Embeddings (PoPE)/.TASK-DOMAINS.csv.tmp.d73a7ccf815d4f63a1fde5cb6a9db977" with the same rows and columns as the Task
Table. Use the exact header:

task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic

Do not add extra columns, commentary, or blank lines.
