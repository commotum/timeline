# Deep Speech 2: End-to-End Speech Recognition in English and Mandarin (Not specified in the paper.)
Source: Deep Speech 2- End-to-End Speech Recognition in English and Mandarin.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Speech-to-text transcription (ASR) | speech spectrograms (audio features) | 2D (x, y) | Open (inferred) | Static (inferred) | Direct (inferred) | text transcriptions (graphemes/characters) | 1D (t) | Open (inferred) |

## Summary
The paper describes an end-to-end speech recognition system that maps speech spectrograms to text transcriptions for English and Mandarin. Inputs are time-frequency spectrograms (2D), and outputs are sequences of graphemes/characters (1D). The model operates over variable-length utterances and transcriptions (Open, inferred) and processes the full input sequence without any described dynamic attention or external state mechanism (Static attention and Direct state inferred).

## Evidence
### Task: Speech-to-text transcription (ASR)
- "Figure 1 shows the architecture of the DS2 system which at its core is similar to the previous DS1 system [26]: a recurrent neural network (RNN) trained to ingest speech spectrograms and generate text transcriptions." (Section 3.1 Preliminaries)
- "Each utterance,  $x^{(i)}$ , is a time-series of length  $T^{(i)}$  where every time-slice is a vector of audio features,  $x_t^{(i)},t=0,\ldots,T^{(i)}-1$ . We use a spectrogram of power normalized audio clips as the features to the system, so  $x_{t,p}^{(i)}$  denotes the power of the p'th frequency bin in the audio frame at time t." (Section 3.1 Preliminaries)
- "The outputs of the network are the graphemes of each language. At each output time-step t, the RNN makes a prediction over characters,  $p(\ell_t|x)$ , where  $\ell_t$  is either a character in the alphabet or the blank symbol." (Section 3.1 Preliminaries)
- "Two methods are currently used to map variable length audio sequences directly to variable length transcriptions." (Section 2 Related Work)
- Inference: In/Out Dynamics marked Open (inferred) because the paper describes "variable length audio sequences" and utterances as a "time-series of length  $T^{(i)}$" without a fixed maximum length (Section 2 Related Work; Section 3.1 Preliminaries). Attention Dynamic marked Static (inferred) and State Dynamic marked Direct (inferred) because the described DS2 architecture is an RNN trained to ingest spectrograms and generate transcriptions with no described runtime selection mechanism or external memory (Section 3.1 Preliminaries).
