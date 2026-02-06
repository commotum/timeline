# An online sequence-to-sequence model for noisy speech recognition (Not specified in the paper)
Source: An online sequence-to-sequence model for noisy speech recognition.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phoneme recognition (speech transcription) | audio utterances (acoustic sequence) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | phoneme sequences | 1D (t) (inferred) | Open (inferred) |
| mixed-speech recognition (primary speaker transcription) | mixed speech audio (two speakers) | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | speaker 1 transcript (phoneme sequence) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates an online sequence-to-sequence model on speech recognition tasks, including phoneme recognition on TIMIT and mixed-speech recognition where one speaker's transcript is produced. Inputs and outputs are temporal sequences (1D (t)) with variable length (inferred), and the model processes streaming inputs causally with static attention and a constructed recurrent state (inferred).

## Evidence
### Task: phoneme recognition (speech transcription)
- "phoneme recognition task" (Section III-A, TIMIT)
- "phoneme sequences" (Section III-A, TIMIT)
- Inference: In/out dimensions, dynamics, attention, and state are inferred from Section II (Methods), which defines input/output sequences `(x_1,...,x_{T_1})` and `(y_1,...,y_{T_2})` processed step-by-step by an LSTM that decides when to emit outputs, implying 1D temporal sequences with variable length, static attention to streaming inputs, and a constructed recurrent state.

### Task: mixed-speech recognition (primary speaker transcription)
- "mixed speech from two speakers" (Section V, Conclusions)
- "transcript of the speaker 1" (Section III-B, Multi-TIMIT)
- "target phonemes" (Section III-B, Multi-TIMIT)
- Inference: In/out dimensions, dynamics, attention, and state are inferred from Section II (Methods), which defines input/output sequences `(x_1,...,x_{T_1})` and `(y_1,...,y_{T_2})` processed step-by-step by an LSTM that decides when to emit outputs, implying 1D temporal sequences with variable length, static attention to streaming inputs, and a constructed recurrent state.
