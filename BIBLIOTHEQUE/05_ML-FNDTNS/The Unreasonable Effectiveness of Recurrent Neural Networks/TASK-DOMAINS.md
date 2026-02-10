# The Unreasonable Effectiveness of Recurrent Neural Networks (2015)
Source: The Unreasonable Effectiveness of Recurrent Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image captioning | images | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | sentence of words | 1D (t) (inferred) | Open (inferred) |
| Sentiment analysis | sentence tokens | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | positive/negative sentiment label | 0D (inferred) | Fixed (inferred) |
| Machine translation | English sentence tokens | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | French sentence tokens | 1D (t) (inferred) | Open (inferred) |
| Frame-level video classification | video frames | 3D (x, y, t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | per-frame labels | 1D (t) (inferred) | Open (inferred) |
| Character-level language modeling / text generation | character sequences from text files | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Direct (inferred) | next-character predictions / generated character sequences | 1D (t) (inferred) | Open (inferred) |

## Summary
The OCR text describes RNN coverage across both sequence transduction and sequence generation tasks, including image-to-text, text-to-label, text-to-text, video-to-label-sequence, and character-level language modeling. The supported domains span 1D (t), 2D (x, y), and 3D (x, y, t) task spaces, with outputs in both 0D and 1D (t). Sequence sides are justified as Open where the text states there are no pre-specified sequence-length constraints, while image inputs are treated as Fixed in the image-captioning example. Attention and state are inferred as Static and Direct for these examples based on the paper’s fixed recurrent update framing and next-step prediction setup.

## Evidence
### Task: Image captioning
- "Sequence output (e.g. image captioning takes an image and outputs a sentence of words)." (Section "What are RNNs anyway?")
- "Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like." (Section "What are RNNs anyway?")
- Inference: In Dimension = 2D (x, y), In Dynamics = Fixed, Attention Dynamic = Static, State Dynamic = Direct, Out Dimension = 1D (t), and Out Dynamics = Open are inferred from the explicit image-to-sentence mapping and the stated recurrent mechanism with unconstrained sequence length. Supporting text: "RNNs combine the input vector with their state vector with a fixed (but learned) function to produce a new state vector." (Section "What are RNNs anyway?")

### Task: Sentiment analysis
- "Sequence input (e.g. sentiment analysis where a given sentence is classified as expressing positive or negative sentiment)." (Section "What are RNNs anyway?")
- "Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like." (Section "What are RNNs anyway?")
- Inference: In Dimension = 1D (t), In Dynamics = Open, Attention Dynamic = Static, State Dynamic = Direct, Out Dimension = 0D, and Out Dynamics = Fixed are inferred from sentence-sequence input with single-label classification output in the given example and the fixed recurrent update description.

### Task: Machine translation
- "Sequence input and sequence output (e.g. Machine Translation: an RNN reads a sentence in English and then outputs a sentence in French)." (Section "What are RNNs anyway?")
- "Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like." (Section "What are RNNs anyway?")
- Inference: In/Out Dimension = 1D (t), In/Out Dynamics = Open, Attention Dynamic = Static, and State Dynamic = Direct are inferred from sentence-to-sentence sequential transduction with the stated unconstrained sequence-length behavior and fixed recurrent state update.

### Task: Frame-level video classification
- "Synced sequence input and output (e.g. video classification where we wish to label each frame of the video)." (Section "What are RNNs anyway?")
- "Notice that in every case are no pre-specified constraints on the lengths sequences because the recurrent transformation (green) is fixed and can be applied as many times as we like." (Section "What are RNNs anyway?")
- Inference: In Dimension = 3D (x, y, t), In Dynamics = Open, Attention Dynamic = Static, State Dynamic = Direct, Out Dimension = 1D (t), and Out Dynamics = Open are inferred from frame-indexed video input and per-frame label output under the stated unconstrained sequence-length regime.

### Task: Character-level language modeling / text generation
- "We'll train RNN character-level language models." (Section "Character-Level Language Models")
- "The input in each case is a single file with some text, and we're training an RNN to predict the next character in the sequence." (Section "Character-Level Language Models")
- "At **test time**, we feed a character into the RNN and get a distribution over what characters are likely to come next. We sample from this distribution, and feed it right back in to get the next letter." (Section "Character-Level Language Models")
- Inference: In/Out Dimension = 1D (t), In/Out Dynamics = Open, Attention Dynamic = Static, and State Dynamic = Direct are inferred from sequential next-character prediction and iterative sampling over arbitrary-length character streams.
