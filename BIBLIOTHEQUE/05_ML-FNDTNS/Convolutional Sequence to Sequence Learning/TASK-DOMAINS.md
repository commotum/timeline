# Convolutional Sequence to Sequence Learning (Not specified in the paper.)
Source: Convolutional Sequence to Sequence Learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| machine translation | input sequence (source sentence) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | output sequence (target sentence) | 1D (t) (inferred) | Capped (inferred) |
| abstractive summarization | long sentence | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | shortened version (summary sentence) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates a fully convolutional sequence-to-sequence model on machine translation and abstractive summarization, both framed as text sequence transduction. The tasks operate over 1D sequences with variable lengths that are capped by the model's position-embedding length limit, and they use attention that selects among input positions at runtime. The model constructs internal encoder/decoder state representations to produce output sequences.

## Evidence
### Task: machine translation
- "We consider three major WMT translation tasks" (Section 4.1 Datasets)
- "The encoder RNN processes an input sequence  $\mathbf{x}=(x_1,\ldots,x_m)$" (Section 2. Recurrent Sequence to Sequence Learning)
- "generates the output sequence  $\mathbf{y}=(y_1,\ldots,y_n)$" (Section 2. Recurrent Sequence to Sequence Learning)
- Inference: In/Out Dimension set to 1D (t) because the paper defines an "input sequence" $\mathbf{x}=(x_1,\ldots,x_m)$ and an "output sequence" $\mathbf{y}=(y_1,\ldots,y_n)$; In/Out Dynamics set to Capped because position embeddings "impose a restriction on the maximum sentence length." (Section 5.4 Position Embeddings); Attention Dynamic set to Dynamic because attention scores "allow the network to focus on different parts of the input sequence" (Section 2. Recurrent Sequence to Sequence Learning); State Dynamic set to Constructed because the model defines internal block outputs, e.g., "We denote the output of the l-th block as  $\mathbf{h}^l = (h_1^l, \\dots, h_n^l)$" and "$\\mathbf{z}^l = (z_1^l, \\dots, z_m^l)$  for the encoder network" (Section 3.2 Convolutional Block Structure).

### Task: abstractive summarization
- "We consider three major WMT translation tasks as well as a text summarization task." (Section 4.1 Datasets)
- "abstractive sentence summarization which takes a long sentence as input and outputs a shortened version." (Section 5.7 Summarization)
- Inference: In/Out Dimension set to 1D (t) because the task is described as taking a "long sentence" and producing a "shortened version" (Section 5.7 Summarization); In/Out Dynamics set to Capped because position embeddings "impose a restriction on the maximum sentence length." (Section 5.4 Position Embeddings); Attention Dynamic set to Dynamic because attention scores "allow the network to focus on different parts of the input sequence" (Section 2. Recurrent Sequence to Sequence Learning); State Dynamic set to Constructed because the model defines internal block outputs, e.g., "We denote the output of the l-th block as  $\mathbf{h}^l = (h_1^l, \\dots, h_n^l)$" and "$\\mathbf{z}^l = (z_1^l, \\dots, z_m^l)$  for the encoder network" (Section 3.2 Convolutional Block Structure).
