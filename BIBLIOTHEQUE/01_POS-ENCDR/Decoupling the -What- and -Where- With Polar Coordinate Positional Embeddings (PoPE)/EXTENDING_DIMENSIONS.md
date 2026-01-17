## 1. Basic Metadata

Title: "DECOUPLING THE \"WHAT\" AND \"WHERE\" WITH POLAR COORDINATE POSITIONAL EMBEDDING" (Title/header)

Authors: "Anand Gopalakrishnan $^{1*}$  Robert Csordás $^{2}$  † Jürgen Schmidhuber $^{1,3}$  Michael C. Mozer $^4$" (Title/header)

Year: Year not specified.

Venue: Venue not specified.

---

## 2. One-Sentence Contribution Summary

The paper states: "We propose an improvement to RoPE, which we call *Polar Coordinate Position* Embedding or PoPE, that eliminates the what-where confound" and reports that "Transformers using PoPE as the positional encoding scheme outperform baselines using RoPE" (ABSTRACT).

---

## 3. Tasks Evaluated

Task name: Indirect Indexing (Indirect Idx.)
Task type: Reasoning / relational; Classification (final-token accuracy)
Dataset(s) used: Procedurally generated Indirect Indexing dataset
Domain: Synthetic character strings
Quotes: "Indirect Indexing. We introduce a task that requires identifying a target character within a variable-length source string." (Section 4 RESULTS) "The dataset for this task is constructed by procedurally generating examples of source strings, source character and relative shifts." (Section A.1 DATASETS) "evaluated on the accuracy of final token." (Section 4 RESULTS)

Task name: Sequence modeling of symbolic music
Task type: Generation (autoregressive sequence modeling)
Dataset(s) used: Bach-Chorales (JSB); MAESTRO
Domain: Symbolic music (MIDI)
Quotes: "Sequence modeling of symbolic music. We train Transformer models using cross-entropy loss on MIDI-based inputs with a maximum length of 2048 from two popular music datasets, Bach-Chorales (JSB) (Boulanger-Lewandowski et al., 2012) and MAESTRO (Hawthorne et al., 2019)." (Section 4 RESULTS)

Task name: Sequence modeling of human genome
Task type: Generation (next-token prediction)
Dataset(s) used: Human Reference Genome dataset
Domain: Genomic sequences
Quotes: "Sequence modeling of human genome. We train a Transformer on sequences from the Human Reference Genome dataset (Dalla-Torre et al., 2025) using the standard next-token prediction loss." (Section 4 RESULTS)

Task name: Language modeling on OpenWebText
Task type: Generation (language modeling)
Dataset(s) used: OpenWebText
Domain: Natural language (web text)
Quotes: "Language modeling on OpenWebText. We test PoPE's efficacy on language modeling by training Transformers of three sizes on the OpenWebText dataset (Gokaslan & Cohen, 2019)." (Section 4 RESULTS)

Task name: LAMBADA (zero-shot downstream evaluation)
Task type: Generation (last-word prediction)
Dataset(s) used: LAMBADA
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS) "we use the detokenized version from OpenAI for LAMBADA and evaluate the top-one accuracy on the last word" (Section 4 RESULTS)

Task name: BLiMP (zero-shot downstream evaluation)
Task type: Other (zero-shot accuracy evaluation; task type not specified)
Dataset(s) used: BLiMP
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS) "For CBT and BLiMP, we measure the accuracy for each task and report the average accuracy over all tasks." (Section 4 RESULTS)

Task name: Children's Book Test (CBT) (zero-shot downstream evaluation)
Task type: Other (zero-shot accuracy evaluation; task type not specified)
Dataset(s) used: Children's Book Test (CBT)
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS) "For CBT and BLiMP, we measure the accuracy for each task and report the average accuracy over all tasks." (Section 4 RESULTS)

Task name: HellaSwag (zero-shot downstream evaluation)
Task type: Other (zero-shot accuracy evaluation; task type not specified)
Dataset(s) used: HellaSwag
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

Task name: PIQA (zero-shot downstream evaluation)
Task type: Other (zero-shot accuracy evaluation; task type not specified)
Dataset(s) used: PIQA
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

Task name: ARC-E (zero-shot downstream evaluation)
Task type: Other (zero-shot accuracy evaluation; task type not specified)
Dataset(s) used: ARC-E
Domain: Natural language
Quotes: "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks, namely: LAMBADA Paperno et al. (2016), BLiMP (Warstadt et al., 2020), Children's Book Test (CBT) (Hill et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), and ARC-E (Clark et al., 2018)." (Section 4 RESULTS)

Task name: Test-time length extrapolation on PG-19
Task type: Generation (language modeling; perplexity)
Dataset(s) used: PG-19 (test split)
Domain: Natural language (books)
Quotes: "Test-time length extrapolation. We measure the ability of PoPE to generalize to test-time sequences that are longer than those presented during training." (Section 4 RESULTS) "assess zero-shot perplexity on much longer sequences (up to 10240 tokens) from the test split of the PG-19 dataset (Rae et al., 2020)." (Section 4 RESULTS)

---

## 4. Domain and Modality Scope

Single domain?: No. Evidence: "On autoregressive sequence modeling in music, genomic, and natural language domains" (ABSTRACT).

Multiple domains within the same modality?: Yes. Evidence: "Next, we test our method on sequence modeling in the domains of music and genomic data." (Section 4 RESULTS) and "On autoregressive sequence modeling in music, genomic, and natural language domains" (ABSTRACT).

Multiple modalities?: Not stated.

Domain generalization or cross-domain transfer?: Not claimed.

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Indirect Indexing | Not specified (trained for this task) | No | Not specified | "We compare RoPE (Su et al., 2024) against PoPE by training two Transformer models with cross-entropy loss applied only on the final (target) token" (Section 4 RESULTS) |
| Sequence modeling of symbolic music (JSB/MAESTRO) | Not specified | No | Not specified | "We train Transformer models using cross-entropy loss on MIDI-based inputs with a maximum length of 2048 from two popular music datasets, Bach-Chorales (JSB) (Boulanger-Lewandowski et al., 2012) and MAESTRO (Hawthorne et al., 2019)." (Section 4 RESULTS) |
| Sequence modeling of human genome | Not specified | No | Not specified | "We train a Transformer on sequences from the Human Reference Genome dataset (Dalla-Torre et al., 2025) using the standard next-token prediction loss." (Section 4 RESULTS) |
| Language modeling on OpenWebText | Not specified | No | Not specified | "We test PoPE's efficacy on language modeling by training Transformers of three sizes on the OpenWebText dataset" (Section 4 RESULTS) |
| LAMBADA | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| BLiMP | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| CBT | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| HellaSwag | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| PIQA | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| ARC-E | Yes (same OpenWebText-pretrained model) | No (zero-shot) | Not specified | "We evaluate the zero-shot performance of the Transformers pretrained on OpenWebText on six downstream tasks" (Section 4 RESULTS) |
| PG-19 length extrapolation | Yes (OpenWebText-pretrained model) | No (zero-shot); also reports PoPE+ft | Not specified | "We examine models pretrained on OpenWebText using a sequence length (context window) of 1024 tokens and assess zero-shot perplexity on much longer sequences" (Section 4 RESULTS) and "For PoPE, we simply finetune on longer sequences without interpolating the frequency components, we refer to this variant as PoPE+ft" (Section 4 RESULTS) |

---

## 6. Input and Representation Constraints

Indirect Indexing inputs are variable-length character strings: "We generate source strings of length between 20 and 40 characters" and use "character-level tokenization" (Section A.1 DATASETS). The input format is explicitly fixed: "The format of each examples is: <source string>, <source character>, <shift>, <target character> and ',' as a delimiter" (Section A.1 DATASETS).

Training sequence lengths are fixed per dataset: "Sequence length     | 40            | 1024        | 2048 | 2048    | 1000    |" (Table 7).

OpenWebText uses a fixed maximum length and tokenizer: "maximum sequence length of 1024 for pretraining" and "We use the GPT-2 tokenizer with a vocabulary size of 50257." (Section A.1 DATASETS).

JSB uses a 2D-to-1D serialization and padding tokens: it is "represented as a matrix with rows corresponding to voices and columns to time discretized to 16th notes" and they "serialize this matrix in raster-scan fashion" with "a maximum sequence length of 2048" and a "vocabulary size of 90 which includes the MIDI notes, silence and padding tokens." (Section A.1 DATASETS).

MAESTRO sequences are constrained to length and token set: they "divide it into sequences with a maximum length of 2048" and use "the REMI tokenizer (Huang & Yang, 2020) with EOS, BOS, MASK and PAD tokens leading to a total vocabulary size of 328." (Section A.1 DATASETS).

Human genome sequences are fixed-length: "obtain sequences with a maximum length of 1000 tokens and vocabulary size of 4107." (Section 4 RESULTS).

Length extrapolation uses fixed training context with longer test sequences: "sequence length (context window) of 1024 tokens" during pretraining and test sequences "up to 10240 tokens" (Section 4 RESULTS).

---

## 7. Context Window and Attention Structure

Attention structure is causal, decoder-only: "we use a decoder-only Transformer architecture (Vaswani et al., 2017; Radford et al., 2018) with causal masking for autoregressive sequence modeling." (Section 4 RESULTS) This implies global causal attention; no windowed or hierarchical attention is stated.

Maximum sequence lengths are fixed by dataset during training (Table 7) and extended at test time for extrapolation: "Sequence length     | 40            | 1024        | 2048 | 2048    | 1000    |" (Table 7) and "sequence length (context window) of 1024 tokens" with test sequences "up to 10240 tokens" (Section 4 RESULTS). Indirect Indexing inputs are variable in length within a range: "source strings of length between 20 and 40 characters" (Section A.1 DATASETS).

Compute efficiency mechanism: "We implemented PoPE using Triton, starting from the example code for Flash Attention 2" and "modify the kernel" (Section 3, Efficient Implementation).

---

## 8. Positional Encoding (Critical Section)

Positional encoding mechanism: PoPE is explicitly a relative positional encoding applied to attention via key/query phases: "We proposed a new relative positional encoding technique called PoPE whose query-key attention scores are based on a computation that decouples the match based on content and the match based on position." (Section 6 CONCLUSION) The position dependence is stated directly: "The phases are position dependent:" (Section 3 METHOD).

Where it is applied: PoPE is applied in the attention score via key/query rotations: "the attention score can elegantly be defined as:" (Section 3 METHOD).

Fixed or modified across experiments: positional encoding is the controlled experimental variable, with RoPE vs PoPE and PoPE variants: "In all experiments, we compare our method, PoPE, to the popular RoPE (Su et al., 2024) scheme using two Transformers with identical model and training hyperparameters, the only difference being their positional encoding schemes." (Section 4 RESULTS) and "We run ablation experiments by training the 124M model with PoPE variants that do not use either the softplus activation, $\sigma()$, or the learnable bias vector $\delta$" (Section 4 RESULTS).

---

## 9. Positional Encoding as a Variable

Core research variable?: Yes. Evidence: "the only difference being their positional encoding schemes" in PoPE vs RoPE comparisons (Section 4 RESULTS).

Multiple positional encodings compared?: Yes. Evidence: "we compare our method, PoPE, to the popular RoPE" (Section 4 RESULTS) and ablations of PoPE components (Section 4 RESULTS).

Claim that PE choice is "not critical" or secondary?: Not stated.

---

## 10. Evidence of Constraint Masking

Model sizes are varied explicitly: "On language modeling, these gains persist across model scale, from 124M to 774M parameters." (ABSTRACT) and "training Transformers of three sizes" (Section 4 RESULTS).

Dataset sizes are specified for several tasks: Indirect Indexing has "train/validation/test splits of size 1M/10k/10k" (Section A.1 DATASETS); OpenWebText has "training and validation splits roughly contain 9B and 4M tokens respectively" (Section A.1 DATASETS); Human Reference Genome has "a total of 3.2 billion nucleotides" (Section A.1 DATASETS); PG-19 test split contains "100 books or roughly 7M tokens" (Section A.1 DATASETS).

Performance gains are attributed to positional encoding changes rather than scaling: "the only difference being their positional encoding schemes" in comparisons (Section 4 RESULTS), and the performance gap "holds steady or possibly increases with model size" (Section 4 RESULTS), but no claim is made that scaling model size or data is the primary driver.

Training tricks noted for length extrapolation include fine-tuning for some baselines/variants: "YaRN ... applies an interpolation scheme ... and finetunes the model on longer sequences" and "For PoPE, we simply finetune on longer sequences" (Section 4 RESULTS).

---

## 11. Architectural Workarounds

Causal masking in a decoder-only Transformer is the core architectural constraint: "decoder-only Transformer architecture ... with causal masking for autoregressive sequence modeling." (Section 4 RESULTS).

Efficiency workaround for attention: "We implemented PoPE using Triton, starting from the example code for Flash Attention 2 ... We modify the kernel" (Section 3, Efficient Implementation).

PoPE introduces a learnable phase bias to tune offsets: "each attention head might benefit by introducing a learnable but fixed bias term" (Section 3 METHOD) to adjust relative offsets.

Windowed attention, hierarchical stages, or token pooling are not stated.

---

## 12. Explicit Limitations and Non-Claims

Limitation/uncertainty about language applicability: "Although we find a benefit of incorporating PoPE into Transformers on diverse domains like music and genomic sequences, we chose these domains specifically because they appear to require the separation of position and content as well as precise positional information. It is much less clear that these properties hold true for human language." (Section 4 RESULTS)

No explicit statements about open-world learning, unrestrained multi-task learning, or meta-learning are stated.

---

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Multiple sequence domains are tested (music, genomics, natural language, synthetic strings) rather than a single domain.
> - Task structure: Primarily autoregressive sequence modeling plus a diagnostic indexing task and zero-shot downstream evaluations.
> - Representation rigidity: Fixed sequence lengths per dataset (40/1024/2048/1000) with defined tokenizers and padding; some variable-length inputs (20-40 characters) for Indirect Indexing.
> - Model sharing vs specialization: Separate per-dataset training is described, with shared OpenWebText-pretrained weights only for zero-shot downstream/PG-19 evaluations.
> - Role of positional encoding: Central experimental variable (PoPE vs RoPE and ablations).

---

### 14. Final Classification

Multi-task, multi-domain (constrained). The paper evaluates multiple tasks across "music, genomic, and natural language domains" plus a synthetic diagnostic task, but does not describe joint multi-task training; instead, it "train[s]" models on each dataset and uses zero-shot evaluation on downstream tasks from an OpenWebText-pretrained model. This supports a constrained multi-domain evaluation rather than unrestrained multi-task learning.
