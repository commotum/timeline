## 1. Basic Metadata

- Title: "MUSIC TRANSFORMER: GENERATING MUSIC WITH LONG-TERM STRUCTURE" (Title)
- Authors: "Cheng-Zhi Anna Huang* Ashish Vaswani Jakob Uszkoreit Noam Shazeer Ian Simon Curtis Hawthorne Andrew M. Dai Matthew D. Hoffman Monica Dinculescu Douglas Eck" (Title block)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

It proposes a memory-efficient relative positional attention algorithm that "reduces their intermediate memory requirement to linear in the sequence length" and enables a Transformer to "generate minutelong compositions (thousands of steps, four times the length modeled in Oore et al. (2018))" (Abstract).

## 3. Tasks Evaluated

- Task name: Unconditioned symbolic music generation (language modeling) on chorales; Task type: Generation; Dataset(s): JSB Chorales; Domain: symbolic music (four-part chorales); Evidence: "We take a language-modeling approach to training generative models for symbolic music." (Section 3.1 Data representation) and "J.S. Bach chorales is a canonical dataset used for evaluating generative models for music" (Section 4.1 J.S. Bach Chorales).
- Task name: Unconditioned symbolic music generation on piano performances; Task type: Generation; Dataset(s): Piano-e-Competition; Domain: symbolic music (MIDI piano performances); Evidence: "The Piano-e-Competition dataset consists of MIDI recorded from performances of competition participants, bearing expressive dynamics and timing on the granularity of < 10 miliseconds." (Section 1 Introduction) and "Table 3: Validation NLL for Piano-e-Competition dataset, with event-based representation with lengths L=2048." (Table 3).
- Task name: Primed continuation (qualitative priming) for piano performance generation; Task type: Generation; Dataset(s): Piano-e-Competition; Domain: symbolic piano performance; Evidence: "Transformer with relative attention elaborates the motif and creates phrases with clear contour which are repeated and varied." (Section 4.2.1 Qualitative Priming Experiments) and "Note that the generated samples are twice as long as the training sequences." (Section 4.2.1 Qualitative Priming Experiments).
- Task name: Harmonization/accompaniment conditioned on melody (seq2seq); Task type: Generation; Dataset(s): Piano-e-Competition; Domain: symbolic music (melody plus accompaniment); Evidence: "conditioned generation task where the encoder takes in a given melody and the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 Harmonization: Conditioning on Melody) and "Table 4: Validation conditional NLL given groundtruth melody from Piano-e-Competition." (Table 4).

## 4. Domain and Modality Scope

- Evaluation spans two datasets within symbolic music: "We evaluate the Transformer with our relative attention mechanism on two datasets, JSB Chorales and Piano-e-Competition, and obtain state-of-the-art results on the latter." (Abstract)
- The datasets are both symbolic music but distinct subdomains: "The JSB Chorale dataset consists of four-part scored choral music" (Section 3.1 Data representation) and "The Piano-e-Competition dataset consists of MIDI recorded from performances of competition participants" (Section 1 Introduction).
- Domain generalization or cross-domain transfer: Not claimed; the closest claim is length generalization: "suggests that the Transformers with relative attention could generalize beyond the lengths it was trained on" (Section 1 Introduction).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Unconditioned chorale generation (JSB Chorales) | Not specified. | Not specified. | Not specified. | "J.S. Bach chorales is a canonical dataset used for evaluating generative models for music" (Section 4.1 J.S. Bach Chorales). |
| Unconditioned piano performance generation (Piano-e-Competition) | Not specified. | Not specified. | Not specified. | "We trained on random crops of 2000-token sequences" (Section 4.2 Piano-e-Competition). |
| Primed continuation (Piano-e-Competition) | Not specified. | Not specified. | Not specified. | "Transformer with relative attention elaborates the motif and creates phrases with clear contour which are repeated and varied." (Section 4.2.1 Qualitative Priming Experiments) |
| Melody-conditioned harmonization (Piano-e-Competition) | Not specified. | Not specified. | Not specified. | "conditioned generation task where the encoder takes in a given melody and the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 Harmonization: Conditioning on Melody) |

## 6. Input and Representation Constraints

- Sequence/token representation: "We take a language-modeling approach to training generative models for symbolic music. Hence we represent music as a sequence of discrete tokens, with the vocabulary determined by the dataset." (Section 3.1 Data representation)
- JSB fixed grid and sequence length: "We first discretize the scores onto a 16th-note grid, and then serialize it by iterating through all the voices within a time step and then advancing time" (Section 4.1 J.S. Bach Chorales) and "After serialization, the most common sequence length is 1024. Each token is represented as onehot in pitch." (Section A.1 Serialized Instrument/Time Grid)
- Piano-e event-based timing resolution and token set: "For the Piano-e-Competition we therefore use the performance encoding proposed by Oore et al. (2018) which consists of a vocabulary of 128 NOTE_ON events, 128 NOTE_OFFs, 100 TIME_SHIFTs allowing for expressive timing at 10ms and 32 VELOCITY bins for expressive dynamics" (Section 3.1 Data representation) and "allowing a minute of music with 10 milisecond resolution to be represented at lengths around 2K" (Section 1 Introduction).
- Fixed-length training/eval windows: "We trained on random crops of 2000-token sequences" (Section 4.2 Piano-e-Competition) and "Table 3: Validation NLL for Piano-e-Competition dataset, with event-based representation with lengths L=2048." (Table 3).
- Melody input quantization for harmonization: "The melody is encoded as a sequence of tokens as in Waite (2016), quantized to a 100ms grid" (Section 4.2.2 Harmonization: Conditioning on Melody).
- Fixed patch size or input padding/resizing requirements: Not specified.

## 7. Context Window and Attention Structure

- Sequence length constraints in data/training: "After serialization, the most common sequence length is 1024." (Section A.1 Serialized Instrument/Time Grid), "We trained on random crops of 2000-token sequences" (Section 4.2 Piano-e-Competition), and "Table 3: Validation NLL for Piano-e-Competition dataset, with event-based representation with lengths L=2048." (Table 3).
- Output length relative to training: "generate minutelong compositions (thousands of steps, four times the length modeled in Oore et al. (2018))" (Abstract) and "Note that the generated samples are twice as long as the training sequences." (Section 4.2.1 Qualitative Priming Experiments)
- Global causal self-attention: "Self-attention over its own previous outputs allows an autoregressive model to access any part of the previously generated output at every step of generation." (Section 1 Introduction) and "A upper triangular mask ensures that queries cannot attend to keys later in the sequence." (Section 3.2 Background: Self-Attention in Transformer)
- Windowed/local attention: "Local attention has been used for example in Wikipedia and image generation (Liu et al., 2018; Parmar et al., 2018) by chunking the input sequence into non-overlapping blocks. Each block then attends to itself and the one before" (Section 3.5 Relative Local Attention).
- Local attention parameters and relative distance limits: "We use block size (bs) 512 for local attention. We set the maximum relative distance to consider to half the training sequence length for relative global attention, and to the full memory length (which is two blocks) for relative local attention." (Section 4.2 Piano-e-Competition)
- Computational cost mechanisms: "We improve the implementation of relative attention by reducing its intermediate memory requirement from  $O(L^2D)$  to O(LD)" (Section 3.4 Memory efficient implementation of relative position-based attention).

## 8. Positional Encoding (Critical Section)

- Absolute positional encoding baseline: "In its original formulation, the Transformer relies on absolute position representations, using either positional sinusoids or learned position embeddings that are added to the per-position input representations." (Section 1 Introduction)
- Relative positional encoding in attention logits: "As the Transformer model relies solely on positional sinusoids to represent timing information, Shaw et al. (2018) introduced relative position representations to allow attention to be informed by how far two positions are apart in a sequence." (Section 3.3 Relative Positional Self-Attention) and "the relative embeddings interact with queries and give rise to a  $S^{rel}$ , an  $L \times L$  dimensional logits matrix which modulates the attention probabilities for each head" (Section 3.3 Relative Positional Self-Attention).
- Relative timing and pitch embeddings: "We learn separate relative embeddings for timing  $E_t$  and also pitch  $E^p$ ." (Section 4.1.1 Generalizing relative attention to capture relational information)
- Positional encoding applied at input via concatenation: "we also explored enhancing absolute timing through concatenating instead of adding the sinusoids to the input embeddings." (Section 4.1 J.S. Bach Chorales)
- Layer placement of extra relational signals: "It was sufficient to add the extra timing signals to the first layer" (Section 4.1.1 Generalizing relative attention to capture relational information).

## 9. Positional Encoding as a Variable

- Positional encoding is a core research variable: "Our relative attention mechanism is essential to the model's quality." (Section 1.1 Contributions)
- Multiple positional encoding variants are compared: "In addition to relative attention, we also explored enhancing absolute timing through concatenating instead of adding the sinusoids to the input embeddings." (Section 4.1 J.S. Bach Chorales) and "Relative attention, more timing and relational information improve performance." (Table 2)
- Relative vs regular attention is explicitly compared: "Table 3 show that relative attention (global or local) outperforms regular self-attention (global or local)." (Section 4.2 Piano-e-Competition)
- Claim that positional encoding is not critical or secondary: Not stated.

## 10. Evidence of Constraint Masking

- Dataset size (Piano-e-Competition): "resulting in about 1100 pieces, split 80/10/10." (Section 4.2 Piano-e-Competition)
- Model size examples: "Transformer (TF) baseline (Vaswani et al., 2017) (5L, 256hs, 256att, 1024ff, 8h)" (Table 2) and "Transformer (TF) baseline (6L, 256hs, 512att, 2048fs, 1024r, 8h)" (Table 3)
- Performance gains attributed to positional/architectural changes: "Relative attention, more timing and relational information improve performance." (Table 2) and "Our relative attention mechanism is essential to the model's quality." (Section 1.1 Contributions)
- Scaling via efficiency rather than data scale: "reducing its intermediate memory requirement from  $O(L^2D)$  to O(LD)" (Section 3.4 Memory efficient implementation of relative position-based attention).

## 11. Architectural Workarounds

- Memory-efficient relative attention: "reducing its intermediate memory requirement from  $O(L^2D)$  to O(LD)" (Section 3.4 Memory efficient implementation of relative position-based attention).
- Windowed attention for long sequences: "chunking the input sequence into non-overlapping blocks. Each block then attends to itself and the one before" (Section 3.5 Relative Local Attention).
- Local attention configuration limits: "We use block size (bs) 512 for local attention. We set the maximum relative distance to consider to half the training sequence length for relative global attention, and to the full memory length (which is two blocks) for relative local attention." (Section 4.2 Piano-e-Competition)
- Causal masking: "A upper triangular mask ensures that queries cannot attend to keys later in the sequence." (Section 3.2 Background: Self-Attention in Transformer)
- Extra relational signals limited to early layers: "It was sufficient to add the extra timing signals to the first layer" (Section 4.1.1 Generalizing relative attention to capture relational information).

## 12. Explicit Limitations and Non-Claims

- Limited scalability of pitch/time relative embeddings: "However this approach is not directly scalable beyond J.S. Bach Chorales because it involves explicitly gathering relative embeddings for  $R^t$  and  $R^p$ , resulting in a memory complexity of  $O(L^2D)$  as in Shaw et al. (2018)." (Section 4.1.1 Generalizing relative attention to capture relational information)
- Local attention limitation: "Each though local attention does not see all the history at once" (Section 4.2 Piano-e-Competition).
- Uncertainty about event-based representation: "As position in sequence no longer corresponds to time, a priori it is not obvious that relative attention should work as well with such a representation." (Section 1 Introduction)
- Explicit non-claims about open-world or unrestrained multi-task learning: Not stated.

### 13. Constraint Profile (Synthesis)

> **Constraint Profile:**
> - Domain scope: Two datasets within symbolic music (chorales and MIDI piano performance), not multiple modalities.
> - Task structure: Autoregressive generation plus conditioned continuation and melody-conditioned accompaniment.
> - Representation rigidity: Fixed 16th-note grid for chorales, event-based MIDI tokens with 10ms time-shifts, fixed-length training crops (2000 tokens) and typical lengths like 1024 or 2048.
> - Model sharing vs specialization: No explicit shared-weight multi-task training described; tasks are reported per dataset/task setup.
> - Role of positional encoding: Central experimental variable with relative attention, concatenated sinusoids, and relative pitch/time variants.

### 14. Final Classification

Classification: Multi-task, single-domain. The paper evaluates multiple generation tasks, including unconditional modeling on "two datasets, JSB Chorales and Piano-e-Competition" (Abstract) and a "conditioned generation task where the encoder takes in a given melody and the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 Harmonization: Conditioning on Melody). Both datasets are symbolic music ("The JSB Chorale dataset consists of four-part scored choral music" and "The Piano-e-Competition dataset consists of MIDI recorded from performances of competition participants"), so the evaluation stays within a single modality/domain despite multiple tasks (Section 3.1 Data representation; Section 1 Introduction).
