# MUSIC TRANSFORMER: GENERATING MUSIC WITH LONG-TERM STRUCTURE (Not specified in the paper)
Source: Music Transformer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (unconditioned chorale modeling) | Symbolic music tokens (chorale sequence) | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Symbolic music tokens (chorale sequence) | 1D (t) | Capped (inferred) |
| Generation (unconditioned piano performance modeling) | Symbolic performance event tokens (MIDI-like sequence) | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Symbolic performance event tokens (MIDI-like sequence) | 1D (t) | Capped (inferred) |
| Generation (motif-conditioned continuation) | Priming motif tokens (piano performance sequence) | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Continuation tokens (piano performance sequence) | 1D (t) | Capped (inferred) |
| Generation (melody-conditioned accompaniment/harmonization) | Melody tokens (100ms grid) | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Performance tokens (melody plus accompaniment) | 1D (t) | Capped (inferred) |

## Summary
The paper covers symbolic music generation tasks: unconditioned modeling for chorale scores and piano performances, plus conditioned continuation and melody-conditioned accompaniment. All tasks operate on token sequences (1D time) representing symbolic music. Dynamics, attention, and state are inferred as capped, static, and direct based on fixed-length training/evaluation windows and causal autoregressive decoding.

## Evidence
### Task: Generation (unconditioned chorale modeling)
- "We take a language-modeling approach to training generative models for symbolic music." (Section 3.1 Data representation)
- "The JSB Chorale dataset consists of four-part scored choral music" (Section 3.1 Data representation)
- "Hence we represent music as a sequence of discrete tokens, with the vocabulary determined by the dataset." (Section 3.1 Data representation)
- Inference: Marked dynamics as Capped, attention as Static, and state as Direct based on fixed-length sequences and causal autoregressive decoding ("After serialization, the most common sequence length is 1024."; "A upper triangular mask ensures that queries cannot attend to keys later in the sequence."; "The Transformer decoder is a autoregressive generative model"). (Sections A.1, 3.2)

### Task: Generation (unconditioned piano performance modeling)
- "We take a language-modeling approach to training generative models for symbolic music." (Section 3.1 Data representation)
- "The Piano-e-Competition dataset consists of MIDI recorded from performances of competition participants" (Section 1 Introduction)
- "consists of a vocabulary of 128 NOTE_ON events, 128 NOTE_OFFs, 100 TIME_SHIFTs" (Section 3.1 Data representation)
- Inference: Marked dynamics as Capped, attention as Static, and state as Direct based on fixed-length training and causal autoregressive decoding ("We trained on random crops of 2000-token sequences"; "A upper triangular mask ensures that queries cannot attend to keys later in the sequence."; "The Transformer decoder is a autoregressive generative model"). (Sections 4.2, 3.2)

### Task: Generation (motif-conditioned continuation)
- "generate continuations that coherently elaborate on a given motif" (Abstract)
- "When primed with an initial motif (Chopin's Étude Op. 10, No. 5)" (Section 4.2.1 Qualitative Priming Experiments)
- Inference: Marked dynamics as Capped, attention as Static, and state as Direct based on fixed-length training and causal autoregressive decoding ("We trained on random crops of 2000-token sequences"; "A upper triangular mask ensures that queries cannot attend to keys later in the sequence."; "The Transformer decoder is a autoregressive generative model"). (Sections 4.2, 3.2)

### Task: Generation (melody-conditioned accompaniment/harmonization)
- "conditioned generation task where the encoder takes in a given melody" (Section 4.2.2 Harmonization: Conditioning on Melody)
- "the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 Harmonization: Conditioning on Melody)
- "The melody is encoded as a sequence of tokens" (Section 4.2.2 Harmonization: Conditioning on Melody)
- Inference: Marked dynamics as Capped, attention as Static, and state as Direct based on fixed-length training and causal autoregressive decoding ("We trained on random crops of 2000-token sequences"; "A upper triangular mask ensures that queries cannot attend to keys later in the sequence."; "The Transformer decoder is a autoregressive generative model"). (Sections 4.2, 3.2)
