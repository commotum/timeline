# Learning Online Alignments with Continuous Rewards Policy Gradient (Not specified in the paper)
Source: Learning Online Alignments with Continuous Rewards Policy Gradient.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phoneme recognition (speech-to-phoneme transcription) | audio utterances | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | phoneme sequences | 1D (t) (inferred) | Open (inferred) |
| Speech recognition (audio-to-character transcription) | audio signals | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | character sequences | 1D (t) (inferred) | Open (inferred) |
| Machine translation (online/instantaneous) | input sequences | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | output sequences | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates an online sequence-to-sequence model on two speech recognition tasks (TIMIT phoneme recognition and WSJ character-level transcription) and frames the approach as applicable to online machine translation. Inputs and outputs are sequences (audio-to-symbols for the speech tasks, and generic input/output sequences for translation), which are treated as 1D temporal streams. Based on the described online emission mechanism and LSTM recurrence, the attention policy and internal state are inferred as dynamic and constructed, and the sequence lengths are treated as open-ended.

## Evidence
### Task: Phoneme recognition (speech-to-phoneme transcription)
- "The TIMIT data set is a phoneme recognition task in which phoneme sequences have to be inferred from input audio utterances." (Section 3.1 TIMIT)
- "input sequence is given by  $(x_1, \ldots, x_{T_1})$  and let the desired sequence be  $(y_1,\ldots,y_{T_2})$" (Section 2 Methods)
- "Our model uses hard binary stochastic decisions to select the timesteps at which outputs will be produced." (Abstract)
- "h_i = LSTM(h_{i-1}, concat(x_i, \tilde{b}_{i-1}, \tilde{y}_{i-1}))" (Section 2 Methods)
- Inference: The input/output are 1D temporal sequences with Open dynamics because the paper defines variable-length sequences $(x_1,\ldots,x_{T_1})$ and $(y_1,\ldots,y_{T_2})$; attention is Dynamic because the model selects emission timesteps; state is Constructed because an LSTM hidden state $h_i$ is maintained. (Section 2 Methods; Abstract)

### Task: Speech recognition (audio-to-character transcription)
- "Wall Street Journal corpus to assess if the method worked on a large vocabulary speech recognition task" (Section 3 Experiments and Results)
- "This dataset consists of more than thirty seven thousand utterances, corresponding to around 81 hours of audio signals." (Section 3.2 Wall Street Journal)
- "We trained our model to predict the character sequences directly, without the use of pronounciation dictionaries, or language models, from the audio signal." (Section 3.2 Wall Street Journal)
- "input sequence is given by  $(x_1, \ldots, x_{T_1})$  and let the desired sequence be  $(y_1,\ldots,y_{T_2})$" (Section 2 Methods)
- "Our model uses hard binary stochastic decisions to select the timesteps at which outputs will be produced." (Abstract)
- "h_i = LSTM(h_{i-1}, concat(x_i, \tilde{b}_{i-1}, \tilde{y}_{i-1}))" (Section 2 Methods)
- Inference: The input/output are 1D temporal sequences with Open dynamics because the paper defines variable-length sequences $(x_1,\ldots,x_{T_1})$ and $(y_1,\ldots,y_{T_2})$; attention is Dynamic because the model selects emission timesteps; state is Constructed because an LSTM hidden state $h_i$ is maintained. (Section 2 Methods; Abstract)

### Task: Machine translation (online/instantaneous)
- "These tasks include both speech recognition and machine translation" (Section 1 Introduction)
- "solving supervised learning problems where both the inputs and the outputs are sequences." (Section 1 Introduction)
- "input sequence is given by  $(x_1, \ldots, x_{T_1})$  and let the desired sequence be  $(y_1,\ldots,y_{T_2})$" (Section 2 Methods)
- "Our model uses hard binary stochastic decisions to select the timesteps at which outputs will be produced." (Abstract)
- "h_i = LSTM(h_{i-1}, concat(x_i, \tilde{b}_{i-1}, \tilde{y}_{i-1}))" (Section 2 Methods)
- Inference: The input/output are treated as 1D temporal sequences with Open dynamics because the paper defines variable-length sequences $(x_1,\ldots,x_{T_1})$ and $(y_1,\ldots,y_{T_2})$; attention is Dynamic because the model selects emission timesteps; state is Constructed because an LSTM hidden state $h_i$ is maintained. (Section 2 Methods; Abstract)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
Phoneme recognition (speech-to-phoneme transcription),audio utterances,1D (t) (inferred),Open (inferred),Dynamic (inferred),Constructed (inferred),phoneme sequences,1D (t) (inferred),Open (inferred)
Speech recognition (audio-to-character transcription),audio signals,1D (t) (inferred),Open (inferred),Dynamic (inferred),Constructed (inferred),character sequences,1D (t) (inferred),Open (inferred)
Machine translation (online/instantaneous),input sequences,1D (t) (inferred),Open (inferred),Dynamic (inferred),Constructed (inferred),output sequences,1D (t) (inferred),Open (inferred)
