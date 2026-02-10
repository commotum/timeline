# Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes (Not specified in the paper.)
Source: Scaling Memory-Augmented Neural Networks with Sparse Reads and Writes.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence copying | random input sequence | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | copied sequence | 1D (t) (inferred) | Capped (inferred) |
| associative recall | sequence of (key, value) pairs and a cue key | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | associated value | 0D (inferred) | Fixed (inferred) |
| priority sorting | keys with priority values | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | top keys sorted by priority | 1D (t) (inferred) | Fixed (inferred) |
| question answering (bAbI reasoning tasks) | context text and question words (1-hot encoded) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | answer word (1-hot encoded) | 0D (inferred) | Fixed (inferred) |
| one-shot character classification (Omniglot) | character images presented over episodes, with previous-step label input | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | character class labels across episode steps | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates SAM/SDNC on five task domains: three synthetic algorithmic sequence tasks (Copy, Associative Recall, Priority Sort), bAbI language question answering, and Omniglot one-shot character classification. The confirmed modalities are mostly 1D (t) sequences, with Omniglot adding 2D (x, y) character images presented in temporal episodes. Most tasks are best supported as Capped because the paper specifies sequence-length/task-difficulty bounds and curriculum maxima, while bAbI input-size bounds are not explicitly specified. Across all tasks, the architecture evidence supports Dynamic attention and Constructed state because the controller emits runtime read/write queries and updates external memory.

## Evidence
### Task: sequence copying
- "1. Copy: copy a random input sequence of length 1–20" (Section 4.2 Learning with sparse memory access)
- "For example, level specifies the input sequence length for the copy task." (Section 4.3 Scaling with a curriculum)
- Inference: In Dimension and Out Dimension are inferred as 1D (t), and In/Out Dynamics as Capped, from explicit variable sequence length and curriculum maxima (Section 4.2, Section 4.3). Attention Dynamic and State Dynamic are inferred from runtime memory control: "The LSTM then produces a vector,  p_t = (q_t, a_t, alpha_t, gamma_t), of read and write parameters for memory access" and "the controller writes either to previously read locations ... or the least recently accessed location" (Sections 3.3 and 3.2).

### Task: associative recall
- "2. Associative Recall: given 3-6 random (key, value) pairs, and subsequently a cue key, return the associated value." (Section 4.2 Learning with sparse memory access)
- "Namely we trained SAM on the associative recall task up to sequences of length 10,000, and found it was then able to generalize to sequences of length 200,000" (Section 4.3 Scaling with a curriculum)
- Inference: In Dimension is inferred as 1D (t) from sequential presentation; In Dynamics as Capped from explicit pair counts and curriculum sequence limits. Out Dimension is inferred as 0D and Out Dynamics as Fixed from "return the associated value" (single retrieved value). Attention Dynamic and State Dynamic are inferred from controller-driven read/write memory access (Sections 3.2-3.3).

### Task: priority sorting
- "3. Priority Sort: Given 20 random keys and priority values, return the top 16 keys in descending order of priority." (Section 4.2 Learning with sparse memory access)
- "We parametrized three of the tasks described in Section 4.2: associative recall, copy, and priority sort, with a progressively increasing difficulty level which characterises the length of the sequence and number of entries to store in memory." (Section 4.3 Scaling with a curriculum)
- Inference: In Dimension and Out Dimension are inferred as 1D (t) because keys are processed and returned in ordered sequences. In Dynamics is inferred as Capped from explicit key counts and curriculum-level bounds; Out Dynamics as Fixed from "top 16 keys." Attention Dynamic and State Dynamic are inferred from runtime query/write memory operations (Sections 3.1-3.3).

### Task: question answering (bAbI reasoning tasks)
- "#### 4.4 Question answering on the Babi tasks" and "They are synthetically generated language tasks with a vocab of about 150 words that test various aspects of simple reasoning such as deduction, induction and coreferencing." (Section 4.4)
- "The task was encoded using straightforward 1-hot word encodings for both the input and output." (Supplementary Section G Babi results)
- Inference: In Dimension is inferred as 1D (t) for language-token sequences; Out Dimension as 0D and Out Dynamics as Fixed are inferred from single-answer QA framing ("determine an answer" in Section 2.2 background + Section 4.4 task framing). In Dynamics is not explicitly bounded in the paper and is therefore marked Not specified in the paper. Attention Dynamic and State Dynamic are inferred from repeated content-based querying and memory updates by the controller (Sections 2.2, 3.2, 3.3).

### Task: one-shot character classification (Omniglot)
- "Omniglot [12] is a dataset of 1623 characters taken from 50 different alphabets, with 20 examples of each character." (Section 4.5 Learning on real world data)
- "At each time step an example of one of the characters is presented, along with the correct label of the proceeding character." and "the model must learn to rapidly associate a novel character with the correct label, such that it can correctly classify subsequent examples of the same character class." (Section 4.5)
- Inference: In Dimension is inferred as 2D (x, y); 1D (t) because character images are presented over episode time steps. In/Out Dynamics are inferred as Capped from episode-level curriculum limits ("validation task with 500 characters"; "sequence lengths of approx 5000," Section 4.5 and Figure 4 caption). Out Dimension is inferred as 1D (t) for per-step label outputs across episodes. Attention Dynamic and State Dynamic are inferred from the same controller-driven external-memory read/write mechanism used throughout (Sections 3.2-3.3).
