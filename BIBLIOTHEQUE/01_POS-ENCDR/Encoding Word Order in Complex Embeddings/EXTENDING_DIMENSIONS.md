## 1. Basic Metadata
- Title: "ENCODING WORD ORDER IN COMPLEX EMBEDDINGS" (Title)
- Authors: Benyou Wang; Donghao Zhao; Christina Lioma; Qiuchi Li; Peng Zhang; Jakob Grue Simonsen. Evidence: "Benyou Wang \* University of Padua wang@dei.unipd.it **Donghao Zhao** \*" (Title block); "Christina Lioma" (Title block); "Qiuchi Li University of Padua qiuchili@dei.unipd.it Peng Zhang †" (Title block); "Jakob Grue Simonsen University of Copenhagen simonsen@di.ku.dk" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary
It proposes complex-valued word embeddings to fix that "position embeddings capture the position of individual words, but not the ordered relationship (e.g., adjacency or precedence) between individual word positions" by providing "a novel and principled solution for modeling both the global absolute positions of words and their order relationships" (Abstract).

## 3. Tasks Evaluated
- Task name: Text classification
  - Task type: Classification
  - Dataset(s) used: CR; MPQA; SUBJ; MR; SST; TREC.
  - Domain: Natural language text (reviews/opinion/question classification).
  - Quotes: "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation); "We use six popular text classification datasets: CR, MPQA, SUBJ, MR, SST, and TREC (see Tab. 1)." (Section 3.1 Text Classification); "| CR (Hu & Liu, 2014)       | 4K    | CV   | 6K     | product reviews  | 2       |" (Table 1: Dataset Statistics); "| TREC (Li & Roth, 2002)    | 5.4k  | 0.5k | 10k    | Question         | 6       |" (Table 1: Dataset Statistics)
- Task name: Machine translation
  - Task type: Generation
  - Dataset(s) used: WMT 2016 English-German.
  - Domain: Natural language text (parallel sentence pairs).
  - Quotes: "We use the standard WMT 2016 English-German dataset (Sennrich et al., 2016), whose training set consists of 29,000 sentence pairs." (Section 3.2 MACHINE TRANSLATION); "We evaluate MT performance with the Bilingual Evaluation Understudy (BLEU) measure." (Section 3.2 MACHINE TRANSLATION)
- Task name: Language modeling
  - Task type: Generation
  - Dataset(s) used: text8.
  - Domain: Natural language text (character-level English Wikipedia).
  - Quotes: "We use the text8 (Mahoney, 2011) dataset, consisting of English Wikipedia articles." (Section 3.3 Language Modeling); "The text is lowercased from a to z, and space." (Section 3.3 Language Modeling); "The dataset contains 100M characters (90M for training, 5M for dev, and 5M for testing, as per Mikolov et al. (2012))." (Section 3.3 Language Modeling)

## 4. Domain and Modality Scope
- Is evaluation performed on a single domain? No; multiple domains within text are evaluated (classification, translation, language modeling). Evidence: "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation)
- Is evaluation performed on multiple domains within the same modality? Yes (text-only tasks). Evidence: "We use the standard WMT 2016 English-German dataset (Sennrich et al., 2016), whose training set consists of 29,000 sentence pairs." (Section 3.2 MACHINE TRANSLATION); "We use the text8 (Mahoney, 2011) dataset, consisting of English Wikipedia articles." (Section 3.3 Language Modeling)
- Is evaluation performed on multiple modalities? No; only text is described. Evidence: "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation)
- Does the paper claim domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text classification | Not specified. | Not specified. | Not specified. | "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation) |
| Machine translation | Not specified. | Not specified. | Not specified. | "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation) |
| Language modeling | Not specified. | Not specified. | Not specified. | "We evaluate our embeddings in text classification, machine translation and language modeling." (Section 3 Experimental Evaluation) |

## 6. Input and Representation Constraints
- Fixed dimensionality for embeddings: "A Word Embedding (WE) generally defines a map  $f_{we}: \mathbb{N} \to \mathbb{R}^D$  from a discrete **word** index to a D-dimensional real-valued vector" (Section 2 MODELLING WORD ORDER IN EMBEDDING SPACE).
- Discrete position indices for PE: "Similarly, a Position Embedding (PE) (Gehring et al., 2017; Vaswani et al., 2017) defines another map  $f_{pe}: \mathbb{N} \to \mathbb{R}^D$  from a discrete **position** index to a vector." (Section 2 MODELLING WORD ORDER IN EMBEDDING SPACE)
- Variable position indices (no fixed max length stated): "Additionally, a *boundedness* property is necessary to ensure that the position embedding can deal with text of any length (*pos* could be large in a long document)." (Section 2.2 Properties for the Functions to Capture word order)
- Pretrained embedding size (non-Transformer models): "We use pre-trained 300-dimensional vectors from word2vec (Mikolov et al., 2013a) in all models except for Transformers." (Section 3.1 Text Classification)
- Transformer representation sizes: "dimension of word and inner hidden are 256 and 512 respectively, and head number is 8." (Section 3.1 Text Classification)
- Character set constraint for text8: "The text is lowercased from a to z, and space." (Section 3.3 Language Modeling)
- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens: Not specified.
- Padding or resizing requirements: Not specified.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified.
- Sequence length fixed or variable: Not specified; the only explicit indication is that text length can be large: "the position embedding can deal with text of any length (*pos* could be large in a long document)." (Section 2.2 Properties for the Functions to Capture word order)
- Attention type: Self-attention is described for Transformer models. Evidence: "The main components in the Transformer are self-attention sublayers and position-wise feed-forward (FFN) sublayers." (Appendix B)
- Relative position handling in Transformer XL: "To extend another variant of Transformer called Transformer XL, we keep its original relative position embedding and additionally replace its word embedding with our proposed embedding." (Appendix B)
- Any mechanisms to manage computational cost (windowing/pooling/token pruning): Not specified.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Complex-valued embedding with position in the phase, defined as "our **general complex-valued embedding** is defined as  $f(j, pos) = g_j(pos) = r_j e^{i(\omega_j pos + \theta_j)}$ ." (Section 2.3 Complex-valued word Embedding)
- Absolute vs relative: The core embedding uses absolute position index pos (as above); for Transformer XL, relative position embeddings are retained: "To extend another variant of Transformer called Transformer XL, we keep its original relative position embedding and additionally replace its word embedding with our proposed embedding." (Appendix B)
- Where it is applied: At the embedding layer. Evidence: "Words functions are implemented in neural networks by storing the function parameters  $\{r,\omega,\theta\}$  and then construct the values based on the arguments." and "the implementation of the proposed embedding can easily be implemented with only modifying the embedding layer." (Appendix C)
- Input only vs every layer vs attention bias: Input embedding layer only (no statement of per-layer application beyond embeddings). Evidence: "The embedding layer does not use initial phases, i.e., following  $f(j, pos) = r_j e^{i(\omega_j pos)}$ ." (Section 3.2 MACHINE TRANSLATION)
- Fixed vs modified/ablated: Multiple positional embeddings are compared and ablated: "We use each of them: (1) without positional information; (2) with **Vanilla Position Embeddings (PE)** (randomly initialized and updated during training using the sum between word and position vectors (Gehring et al., 2017); (3) with **Trigonometric Position Embeddings (TPE)** (defining position embeddings as trigonometric functions as per Eq. 7); (4) with **Complex-vanilla** word embeddings (where the amplitude embedding is initialized by the pre-trained word vectors, and the phrase embedding is randomly initialized in a range from  $-\pi$  to  $\pi$  without considering word order (Wang et al., 2019b)); and (5) with our order-aware complex-valued word embeddings, **Complex-order** (which encode position in the phase parts, train the periods, and where the amplitude embedding is also initialized by pretrained word vectors)." (Section 3.1 Text Classification); "We perform an ablation test (Tab. 4) on Transformer because it is the most common NN to be used with position embeddings." (Section 3.1 Text Classification)

## 9. Positional Encoding as a Variable
- Does the paper treat positional encoding as a core research variable or fixed assumption? Core research variable; multiple positional encodings are explicitly compared. Evidence: "We use each of them: (1) without positional information; (2) with **Vanilla Position Embeddings (PE)** ...; (3) with **Trigonometric Position Embeddings (TPE)** ...; (4) with **Complex-vanilla** ...; and (5) with our order-aware complex-valued word embeddings, **Complex-order** ..." (Section 3.1 Text Classification)
- Are multiple positional encodings compared? Yes. Evidence: "We perform an ablation test (Tab. 4) on Transformer because it is the most common NN to be used with position embeddings." (Section 3.1 Text Classification)
- Does the paper claim PE choice is "not critical" or secondary? Not stated.

## 10. Evidence of Constraint Masking
- Model size(s): "Our embedding generally has  $3 \times D \times |\mathbb{W}|$  parameters with D-dimensional word vectors and  $|\mathbb{W}|$  words, while previous work (Mikolov et al., 2013b; Pennington et al., 2014) usually employs only  $D \times |\mathbb{W}|$  parameters for embedding lookup tables." (Section 3.1 Text Classification); "| Transformer-complex-order                  | $r_{j,d}e^{i(\omega_{j,d}\mathrm{pos})}$     | ×                    | 8.33M  | 0.813    | _      |" (Table 4: Ablation test for Transformer); "| vanilla Transformer (Vaswani et al., 2017) | $WE_{j,d} + PE_d$                            | -                    | 4.1M   | 0.761    | -0.052 |" (Table 4: Ablation test for Transformer)
- Dataset size(s): "| CR (Hu & Liu, 2014)       | 4K    | CV   | 6K     | product reviews  | 2       |" (Table 1: Dataset Statistics); "| MPQA (Wiebe et al., 2005) | 11k   | CV   | 6K     | opinion polarity | 2       |" (Table 1: Dataset Statistics); "| SUBJ (Pang & Lee, 2005)   | 10k   | CV   | 21k    | subjectivity     | 2       |" (Table 1: Dataset Statistics); "| MR (Pang & Lee, 2005)     | 11.9k | CV   | 20k    | movie reviews    | 2       |" (Table 1: Dataset Statistics); "| SST (Socher et al., 2013) | 67k   | 2.2k | 18k    | movie reviews    | 2       |" (Table 1: Dataset Statistics); "| TREC (Li & Roth, 2002)    | 5.4k  | 0.5k | 10k    | Question         | 6       |" (Table 1: Dataset Statistics); "We use the standard WMT 2016 English-German dataset (Sennrich et al., 2016), whose training set consists of 29,000 sentence pairs." (Section 3.2 MACHINE TRANSLATION); "The dataset contains 100M characters (90M for training, 5M for dev, and 5M for testing, as per Mikolov et al. (2012))." (Section 3.3 Language Modeling)
- Attribution of gains to scaling vs architecture/training tricks: Not claimed; improvements are discussed in terms of embedding variants (e.g., "Our complex-order embeddings outperform all other variations at all times.") (Section 3.1 Results)

## 11. Architectural Workarounds
- Shared initial phases for efficiency: "To increase efficiency and facilitate fair comparison with previous work we set initial phases  $\boldsymbol{\theta}_j = [\theta_{j,1},...,\theta_{j,D}]$  to a shared constant value (such as zero)." (Section 3.1 Text Classification)
- Parameter-sharing schemes to reduce embedding parameters: "To decrease the number of parameters, one can either use a word-sharing scheme (i.e.,  $\omega_{j,d} = \omega_{.,d}$ ), or a dimension-sharing scheme ( $\omega_{j,d} = \omega_{j,\cdot}$ ), leading to  $|\mathbb{W}| * D + |\mathbb{W}|$  and  $|\mathbb{W}| * D + D$  parameters in total for the embedding layer." (Section 3.1 Text Classification)
- CNN pooling: "We adopt narrow convolution and max pooling in CNN, with number of filters in  $\{64,128\}$ , and size of filters in  $\{3,4,5\}$ ." (Section 3.1 Text Classification)
- Encoder-only Transformer usage: "In all Transformer models, we only use the encoder layer to extract feature information, where the layer is 1, dimension of word and inner hidden are 256 and 512 respectively, and head number is 8." (Section 3.1 Text Classification)

## 12. Explicit Limitations and Non-Claims
- "We do not compare against them because they encode positional information inherently as part of the model, which makes redundant any additional encoding of positional information at the embedding level." (Section 3.1 Text Classification)
- "We choose 6 layers due to limitations in computing resources." (Section 3.3 Language Modeling)

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Text-only evaluations across classification, translation, and language modeling datasets.
> - Task structure: Multiple distinct NLP tasks evaluated separately; no joint multi-task training is described.
> - Representation rigidity: Fixed-dimensional embeddings with discrete word/position indices; text8 is constrained to lowercase a-z and space.
> - Model sharing vs specialization: Separate task setups/architectures per task; shared weights across tasks are not specified.
> - Role of positional encoding: Central experimental variable with multiple PE variants and ablations.

### 14. Final Classification

**Multi-task, single-domain**

The paper evaluates "text classification, machine translation and language modeling" (Section 3 Experimental Evaluation), all within text data such as an "English-German dataset" and "English Wikipedia articles" (Sections 3.2 and 3.3). It does not claim multi-modal or cross-domain transfer beyond text, so the scope is multi-task within a single domain.
