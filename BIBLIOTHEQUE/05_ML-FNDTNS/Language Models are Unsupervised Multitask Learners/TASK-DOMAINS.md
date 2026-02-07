# Language Models are Unsupervised Multitask Learners (2019)
Source: Language Models are Unsupervised Multitask Learners.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling | Text tokens (context sequence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Text tokens (generated sequence) | 1D (t) (inferred) | Open (inferred) |
| Cloze word prediction (multiple-choice) | Text with omitted word + candidate choices | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Choice for omitted word (one of 10) | 0D (inferred) | Fixed (inferred) |
| Final-word prediction (cloze) | Text context requiring long-range context | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Final word | 0D (inferred) | Fixed (inferred) |
| Commonsense ambiguity resolution (Winograd) | Ambiguous text (schema sentence) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Resolution choice | 0D (inferred) | Fixed (inferred) |
| Reading comprehension (conversational QA) | Document + conversation history + question | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer text | 1D (t) (inferred) | Open (inferred) |
| Summarization | Article text | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Summary text | 1D (t) (inferred) | Capped (inferred) |
| Machine translation | Source-language sentence (with example pairs) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Translated sentence | 1D (t) (inferred) | Open (inferred) |
| Factoid question answering | Question (with example QA pairs) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Short answer text | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates GPT-2 on language modeling and a range of zero-shot NLP tasks: cloze word prediction (CBT, LAMBADA), commonsense ambiguity resolution (Winograd), reading comprehension (CoQA), summarization, machine translation, and factoid question answering. Across tasks, inputs and outputs are textual token sequences, with outputs either single-choice/word decisions (0D) or generated text (1D). Based on the model's fixed context window and autoregressive setup, inputs are treated as capped and attention/state as static/direct, while output dynamics are fixed for single-choice tasks and open or capped for generation tasks like summarization.

## Evidence
### Task: Language modeling
- "language modeling" (Section 3.1. Language Modeling)
- Inference: Classified inputs/outputs as 1D token sequences and dynamics/attention/state as capped/static/direct based on the paper's description of variable-length symbol sequences and a fixed 1024-token context window (Sections 2. Approach, 2.3. Model); marked output dynamics open because the LM is described as able to generate any string (Section 2.2. Input Representation).

### Task: Cloze word prediction (multiple-choice)
- "cloze test" (Section 3.2. Children's Book Test)
- Inference: Treated the text-plus-choices format as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 0D fixed because the task requires selecting one of 10 choices (Section 3.2. Children's Book Test).

### Task: Final-word prediction (cloze)
- "final word" (Section 3.3. LAMBADA)
- Inference: Treated the context sentence as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 0D fixed because the task is to predict a single final word (Section 3.3. LAMBADA).

### Task: Commonsense ambiguity resolution (Winograd)
- "commonsense reasoning" (Section 3.4. Winograd Schema Challenge)
- Inference: Treated the schema text as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 0D fixed because the task is to choose a resolution of the ambiguity (Section 3.4. Winograd Schema Challenge).

### Task: Reading comprehension (conversational QA)
- "reading comprehension" (Section 3.5. Reading Comprehension)
- Inference: Treated the document+dialog+question input as 1D token sequence with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 1D open because answers are generated text without an explicit length limit (Section 3.5. Reading Comprehension).

### Task: Summarization
- "perform summarization" (Section 3.6. Summarization)
- Inference: Treated the article as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 1D capped because the summary is produced by generating 100 tokens and selecting the first 3 sentences (Section 3.6. Summarization).

### Task: Machine translation
- "translate" (Section 3.7. Translation)
- Inference: Treated the sentence-pair prompt as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 1D open because a translated sentence is generated without an explicit token limit (Section 3.7. Translation).

### Task: Factoid question answering
- "factoid-style questions" (Section 3.8. Question Answering)
- Inference: Treated the question prompt (seeded with example QA pairs) as 1D token input with capped window and static/direct processing (Sections 2. Approach, 2.3. Model); labeled output as 1D open because answers are generated in a brief answer format without an explicit length cap (Section 3.8. Question Answering).
