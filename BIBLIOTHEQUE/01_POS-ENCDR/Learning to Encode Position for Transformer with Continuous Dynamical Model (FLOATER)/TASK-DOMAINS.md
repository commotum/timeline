# Learning to Encode Position for Transformer with Continuous Dynamical Model (Not specified in the paper)
Source: Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Machine translation | Text sequence (source language tokens) (inferred) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Text sequence (target language tokens) (inferred) | 1D (t) (inferred) | Open (inferred) |
| Language understanding (GLUE benchmark) | Text sequence(s) (sentences) (inferred) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Language understanding (RACE benchmark) | Text context (inferred) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Question answering (SQuAD span prediction) | Paragraph and questions (text) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | Character range (answer span) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates FLOATER on NLP tasks spanning machine translation, language understanding (GLUE and RACE), and question answering (SQuAD). The model is described as a non-recurrent, self-attentive Transformer operating on variable-length sequences, which supports 1D sequential inputs and open-ended input dynamics. Output structure is only explicitly specified for SQuAD (character-span answers), while GLUE and RACE outputs are not detailed.

## Evidence
### Task: Machine translation
- "Neural Machine Translation (NMT) is the first application that demonstrates the superiority of a sequence-to-sequence Transformer model." (Section 4.1 Neural Machine Translation)
- "we present the BLEU scores on WMT14 En-De and En-Fr datasets" (Section 4.1 Neural Machine Translation)
- Inference: Input/output are 1D text sequences with Open dynamics, and attention/state are Static/Direct, based on "model sequence data of variable lengths," "non-recurrent but self-attentive neural architecture," and "FLOATER can handle sequences of any length" (Section 1 Introduction).

### Task: Language understanding (GLUE benchmark)
- "we focus on three language understanding benchmark sets, GLUE [16], RACE [15] and SQuAD [17]." (Section 4.2 Language Understanding and Question Answering)
- "GLUE Benchmark. This benchmark is commonly used to evaluate the language understanding skills of NLP models." (Section 4.2 Language Understanding and Question Answering)
- Inference: Input is treated as 1D text sequences with Open dynamics and Static/Direct processing, based on "language understanding skills of NLP models," "model sequence data of variable lengths," and "non-recurrent but self-attentive neural architecture" (Section 1 Introduction; Section 4.2).

### Task: Language understanding (RACE benchmark)
- "RACE benchmark is another widely used test suit for language understanding." (Section 4.2 Language Understanding and Question Answering)
- "each item in RACE contains a significantly longer context" (Section 4.2 Language Understanding and Question Answering)
- Inference: Input is a 1D text context with Open dynamics and Static/Direct processing, based on "model sequence data of variable lengths," "non-recurrent but self-attentive neural architecture," and the longer text "context" description (Section 1 Introduction; Section 4.2).

### Task: Question answering (SQuAD span prediction)
- "SQuAD benchmark [17, 18] is another challenging task to evaluate the question answering skills of NLP models." (Section 4.2 Language Understanding and Question Answering)
- "each item contains a lengthy paragraph containing facts and several questions related to the paragraph." (Section 4.2 Language Understanding and Question Answering)
- "The model needs to predict the range of characters that answer the questions." (Section 4.2 Language Understanding and Question Answering)
- Inference: In/Out are 1D (t); input dynamics are Open due to variable-length sequences and "FLOATER can handle sequences of any length"; output dynamics are Capped because the answer is a character range within the paragraph; attention/state are Static/Direct due to the "non-recurrent but self-attentive" Transformer description (Section 1 Introduction; Section 4.2).

---

## CSV Output (required)
