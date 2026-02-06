# Adding Gradient Noise Improves Learning for Very Deep Networks (Not specified in the paper)
Source: Adding Gradient Noise Improves Learning for Very Deep Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification (handwritten digit) | MNIST handwritten digits | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | digit class label (inferred) | 0D (inferred) | Not specified in the paper. |
| question answering | context and question | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | answer | 0D (inferred) | Not specified in the paper. |
| question answering (table) | question and table (or database) | 1D (t); 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | correct answer | 0D (inferred) | Not specified in the paper. |
| retrieval (k-th element in linked list) | pointer to the head of the linked list | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | value of the k-th element | 0D (inferred) | Not specified in the paper. |
| multiplication (binary) | two concatenated sequences of binary digits separated by an operator token | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | product (binary digits) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates tasks spanning handwritten digit classification, question answering over text context and tables, and algorithmic problems like linked-list k-th element retrieval and binary multiplication. Inputs include 2D image-like data and 1D sequences/tables, while outputs are mostly scalar answers or 1D digit sequences; most dynamics are not specified except capped sequence lengths in the binary multiplication setup. Attention is explicitly used in the QA and NRAM tasks, and constructed internal state is supported by the described memory and working-memory mechanisms.

## Evidence
### Task: classification (handwritten digit)
- "training a very deep fully-connected network on the MNIST handwritten digit classification dataset" (Section 4.1 DEEP FULLY-CONNECTED NETWORKS)
- "successfully train a 20-layer rectifier network on MNIST with standard gradient descent." (Section 1 Introduction)
- Inference: Treated MNIST handwritten digits as 2D image inputs and the output as a 0D digit class label based on the "MNIST handwritten digit classification dataset" description. (Section 4.1 DEEP FULLY-CONNECTED NETWORKS)

### Task: question answering
- "We test added gradient noise for training End-To-End Memory Networks (Sukhbaatar et al., 2015), a new approach for Q&A using deep networks." (Section 4.2 END-TO-END MEMORY NETWORKS)
- "the model has access to a context, a question, and is asked to predict an answer." (Section 4.2 END-TO-END MEMORY NETWORKS)
- "Internally, the model has an attention mechanism which focuses on the right clue to answer the question." (Section 4.2 END-TO-END MEMORY NETWORKS)
- Inference: Mapped the context/question to 1D (t), the answer to 0D, and attention/state to Dynamic/Constructed based on the described attention mechanism over contexts. (Section 4.2 END-TO-END MEMORY NETWORKS)

### Task: question answering (table)
- "It is proposed for the task of question answering from tables (Neelakantan et al., 2015)." (Section 4.3 NEURAL PROGRAMMER)
- "Neural Programmer takes a question and a table (or database) as input and the goal is to predict the correct answer." (Section 4.3 NEURAL PROGRAMMER)
- "Key to Neural Programmer is the use of \"soft selection\" to assign a probability distribution over the list of operations." (Section 4.3 NEURAL PROGRAMMER)
- "at each step selects a data segment and an operation to apply to the selected data segment." (Section 4.3 NEURAL PROGRAMMER)
- Inference: Treated the question as 1D (t) and the table as 2D (x, y), and marked attention/state as Dynamic/Constructed due to soft selection and multi-step operation selection; the answer was treated as 0D. (Section 4.3 NEURAL PROGRAMMER)

### Task: retrieval (k-th element in linked list)
- "NRAM is a model for algorithm learning that can store data, and explicitly manipulate and dereference pointers." (Section 4.4 NEURAL RANDOM ACCESS MACHINES)
- "we consider a problem of searching k-th element's value on a linked list." (Section 4.4 NEURAL RANDOM ACCESS MACHINES)
- "The network is given a pointer to the head of the linked list, and has to find the value of the k-th element." (Section 4.4 NEURAL RANDOM ACCESS MACHINES)
- "At every step, the model selects both the operation to be executed and its inputs." (Section 4.4 NEURAL RANDOM ACCESS MACHINES)
- "These selections are made using soft attention (Bahdanau et al., 2014) making the model end-to-end differentiable." (Section 4.4 NEURAL RANDOM ACCESS MACHINES)
- Inference: Interpreted the linked list traversal as 1D (t) input, the output value as 0D, and attention/state as Dynamic/Constructed based on the soft-attention operation selection and explicit memory/registers. (Section 4.4 NEURAL RANDOM ACCESS MACHINES)

### Task: multiplication (binary)
- "In our experiments, we use Neural GPUs for the task of binary multiplication." (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
- "The input consists two concatenated sequences of binary digits separated by an operator token, and the goal is to multiply the given numbers." (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
- "During training, the model is trained on 20-digit binary numbers while at test time, the task is to multiply 200-digit numbers." (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
- "The additional dimension of the tensor serves as a working memory while the repeated operations are applied at each layer." (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
- "The output at the final layer is the predicted answer." (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
- Inference: Treated the input/output as 1D (t) sequences of binary digits and marked dynamics as Capped based on the specified 20-digit training and 200-digit test lengths; marked state as Constructed due to the working-memory tensor. (Section 4.5 CONVOLUTIONAL GATED RECURRENT NETWORKS (NEURAL GPUS))
