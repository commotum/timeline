# Neural Turing Machines (Not specified in the paper.)
Source: Neural Turing Machines (NTM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sequence copying | sequence of random binary vectors + delimiter flag | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | copied sequence of binary vectors (without delimiter) | 1D (t) | Capped |
| repeat sequence copying | sequence of random binary vectors + repeat-count scalar | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | copied sequence repeated N times + end-of-sequence marker | 1D (t) | Capped |
| associative recall (next item retrieval) | sequence of items (binary vector sequences) with delimiters + query item | 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | subsequent item (sequence of binary vectors) | 1D (t) | Fixed |
| next-bit prediction (dynamic n-grams) | binary sequence (bits) | 1D (t) | Fixed | Dynamic (inferred) | Constructed (inferred) | next-bit predictions (binary) | 1D (t) | Fixed |
| priority sorting | sequence of binary vectors with scalar priorities | 1D (t) | Fixed | Dynamic (inferred) | Constructed (inferred) | sorted subset of input vectors (highest priorities) | 1D (t) | Fixed |

## Summary
The paper evaluates five algorithmic sequence tasks: copying, repeat copying, associative recall, dynamic n-gram next-bit prediction, and priority sorting. All tasks operate on 1D (t) sequences of binary vectors or bits, with capped sequence variability for copy/repeat/associative recall and fixed-length sequences for dynamic n-grams and priority sort. Outputs are also 1D sequences, including repeated or sorted vector sequences and next-bit predictions. Dynamic attention and constructed state are inferred from the NTM's attentional read/write interaction with external memory.

## Evidence
### Task: sequence copying
- "The network is presented with an input sequence of random binary vectors followed by a delimiter flag." (Section 4.1)
- "The target sequence was simply a copy of the input sequence (without the delimiter flag)." (Section 4.1)
- "sequence lengths were randomised between 1 and 20." (Section 4.1)
- Inference: Attention Dynamic = Dynamic (inferred) and State Dynamic = Constructed (inferred) because the model is "coupling them to external memory resources, which they can interact with by attentional processes." and "it also interacts with a memory matrix using selective read and write operations." (Abstract; Section 3)

### Task: repeat sequence copying
- "output the copied sequence a specified number of times" (Section 4.2)
- "emit an end-of-sequence marker." (Section 4.2)
- "The network receives random-length sequences of random binary vectors, followed by a scalar value indicating the desired number of copies" (Section 4.2)
- "both the sequence length and the number of repetitions were chosen randomly from one to ten." (Section 4.2)
- Inference: Attention Dynamic = Dynamic (inferred) and State Dynamic = Constructed (inferred) because the model is "coupling them to external memory resources, which they can interact with by attentional processes." and "it also interacts with a memory matrix using selective read and write operations." (Abstract; Section 3)

### Task: associative recall (next item retrieval)
- "we define an item as a sequence of binary vectors that is bounded on the left and right by delimiter symbols." (Section 4.3)
- "we query by showing a random item, and we ask the network to produce the next item." (Section 4.3)
- "each item consisted of three six-bit binary vectors" (Section 4.3)
- "we used a minimum of 2 items and a maximum of 6 items" (Section 4.3)
- Inference: Attention Dynamic = Dynamic (inferred) and State Dynamic = Constructed (inferred) because the model is "coupling them to external memory resources, which they can interact with by attentional processes." and "it also interacts with a memory matrix using selective read and write operations." (Abstract; Section 3)

### Task: next-bit prediction (dynamic n-grams)
- "drawing 200 successive bits" (Section 4.4)
- "The network observes the sequence one bit at a time and is then asked to predict the next bit." (Section 4.4)
- Inference: Attention Dynamic = Dynamic (inferred) and State Dynamic = Constructed (inferred) because the model is "coupling them to external memory resources, which they can interact with by attentional processes." and "it also interacts with a memory matrix using selective read and write operations." (Abstract; Section 3)

### Task: priority sorting
- "A sequence of random binary vectors is input to the network along with a scalar priority rating for each vector." (Section 4.5)
- "The target sequence contains the binary vectors sorted according to their priorities." (Section 4.5)
- "Each input sequence contained 20 binary vectors with corresponding priorities, and each target sequence was the 16 highest-priority vectors in the input." (Section 4.5)
- Inference: Attention Dynamic = Dynamic (inferred) and State Dynamic = Constructed (inferred) because the model is "coupling them to external memory resources, which they can interact with by attentional processes." and "it also interacts with a memory matrix using selective read and write operations." (Abstract; Section 3)
