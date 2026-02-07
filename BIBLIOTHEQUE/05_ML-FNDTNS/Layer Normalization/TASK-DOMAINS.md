# Layer Normalization (Not specified in the paper.)
Source: Layer Normalization.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ranking (image-sentence retrieval) | images; sentences | 2D (x, y) (inferred); 1D (t) (inferred) | Fixed (inferred); Open (inferred) | Not specified in the paper. | Constructed (inferred) | similarity score (inferred) | 0D (inferred) | Fixed (inferred) |
| question-answering | passage; query description | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer token (inferred) | 0D (inferred) | Fixed (inferred) |
| contextual language modelling | sentence | 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | surrounding sentences | 1D (t) (inferred) | Open (inferred) |
| generative modelling (image generation) | MNIST images (inferred) | 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | images | 2D (x, y) (inferred) | Fixed (inferred) |
| handwriting sequence generation | input character string | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | sequence of x and y pen co-ordinates | 1D (t) (inferred) | Open (inferred) |
| classification (MNIST) | MNIST 784-length vector (inferred) | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Direct (inferred) | digit label (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
Across experiments, the paper applies layer normalization to six tasks: image-sentence ranking, question-answering, contextual language modelling, generative modelling on MNIST, handwriting sequence generation, and MNIST classification. Inputs span images and text/character sequences, implying 2D image domains and 1D sequence domains, with outputs including generated images and pen-coordinate sequences. Attention is explicitly mentioned for the attentive reader and DRAW experiments, while other tasks do not specify attention; recurrent models imply constructed state in the sequence tasks.

## Evidence
### Task: ranking (image-sentence retrieval)
- "Images and sentences from the Microsoft COCO dataset are embedded into a common vector space" (Section 6.1)
- "a GRU [Cho et al., 2014] is used to encode sentences" (Section 6.1)
- "the outputs of a pre-trained VGG ConvNet [Simonyan and Zisserman, 2015] (10-crop) are used to encode images" (Section 6.1)
- "replaces the cosine similarity scoring function used in Kiros et al. [2014] with an asymmetric one." (Section 6.1)
- "It is common among the NLP tasks to have different sentence lengths for different training cases." (Section 3.1)
- Inference: Marked 2D/1D dimensions from images/sentences; Fixed image dynamics from VGG ConvNet encoding; Open sentence dynamics from varying sentence lengths; Constructed state from GRU; similarity score/0D output from the scoring function.

### Task: question-answering
- "We train an unidirectional attentive reader model on the CNN corpus" (Section 6.2)
- "This is a question-answering task where a query description about a passage must be answered by filling in a blank." (Section 6.2)
- "each passage is limited to 4 sentences." (Section 6.2)
- "we only apply layer normalization within the LSTM." (Section 6.2)
- Inference: Marked 1D (t) input and Capped dynamics from passages limited to 4 sentences; Dynamic attention from the attentive reader model; Constructed state from LSTM; output token/0D fixed output from filling in a blank.

### Task: contextual language modelling
- "Skip-thoughts [Kiros et al., 2015] is a generalization of the skip-gram model [Mikolov et al., 2013] for learning unsupervised distributed sentence representations." (Section 6.3)
- "encoded with a encoder RNN and decoder RNNs are used to predict the surrounding sentences." (Section 6.3)
- "It is common among the NLP tasks to have different sentence lengths for different training cases." (Section 3.1)
- Inference: Marked 1D (t) dimensions and Open dynamics from sentence data and varying sentence lengths; Constructed state from encoder/decoder RNNs.

### Task: generative modelling (image generation)
- "We also experimented with the generative modeling on the MNIST dataset." (Section 6.4)
- "The model uses a differential attention mechanism and a recurrent neural network" (Section 6.4)
- "to sequentially generate pieces of an image." (Section 6.4)
- "using 64 glimpses and 256 LSTM hidden units." (Section 6.4)
- Inference: Marked 2D image dimensions from image generation; Fixed dynamics from the fixed 64-glimpse setup; Dynamic attention from the differential attention mechanism; Constructed state from the recurrent neural network.

### Task: handwriting sequence generation
- "When given the input character string, the goal is to predict a sequence of x and y pen co-ordinates" (Section 6.5)
- "The input string is typically more than 25 characters" (Section 6.5)
- "the average handwriting line has a length around 700." (Section 6.5)
- "three hidden layers of 400 LSTM cells" (Section 6.5)
- "A mixture of 10 Gaussian functions was used for the window parameters" (Section 6.5)
- Inference: Marked 1D (t) input/output and Open dynamics from variable string/sequence lengths; Dynamic attention from window parameters; Constructed state from LSTM.

### Task: classification (MNIST)
- "permutation invariant MNIST classification problem." (Section 6.6)
- "Permutation invariant MNIST 784-1000-1000-10 model" (Figure 6 caption)
- "we investigated layer normalization in feed-forward networks." (Section 6.6)
- "excludes the last softmax layer." (Section 6.6)
- Inference: Marked 1D input and Fixed dynamics from the 784-1000-1000-10 model; output label/0D from classification and softmax; Direct state from feed-forward networks.

## CSV Output (required)
```csv
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
"ranking (image-sentence retrieval)","images; sentences","2D (x, y) (inferred); 1D (t) (inferred)","Fixed (inferred); Open (inferred)","Not specified in the paper.","Constructed (inferred)","similarity score (inferred)","0D (inferred)","Fixed (inferred)"
"question-answering","passage; query description","1D (t) (inferred)","Capped (inferred)","Dynamic (inferred)","Constructed (inferred)","answer token (inferred)","0D (inferred)","Fixed (inferred)"
"contextual language modelling","sentence","1D (t) (inferred)","Open (inferred)","Not specified in the paper.","Constructed (inferred)","surrounding sentences","1D (t) (inferred)","Open (inferred)"
"generative modelling (image generation)","MNIST images (inferred)","2D (x, y) (inferred)","Fixed (inferred)","Dynamic (inferred)","Constructed (inferred)","images","2D (x, y) (inferred)","Fixed (inferred)"
"handwriting sequence generation","input character string","1D (t) (inferred)","Open (inferred)","Dynamic (inferred)","Constructed (inferred)","sequence of x and y pen co-ordinates","1D (t) (inferred)","Open (inferred)"
"classification (MNIST)","MNIST 784-length vector (inferred)","1D (t) (inferred)","Fixed (inferred)","Not specified in the paper.","Direct (inferred)","digit label (inferred)","0D (inferred)","Fixed (inferred)"
```
