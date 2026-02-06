# A Neural Transducer (Not specified in the paper.)
Source: A Neural Transducer.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addition | digit tokens (two three-digit decimal numbers; second reversed) | 1D (t) | Fixed (inferred) | Not specified in the paper. | Constructed | digit tokens of sum (reversed) | 1D (t) | Capped (inferred) |
| phoneme recognition | acoustic feature frames (log Mel filterbanks) | 1D (t) | Open (inferred) | Dynamic (inferred) | Constructed | phoneme tokens (60 phones) | 1D (t) | Open (inferred) |

## Summary
The paper evaluates the Neural Transducer on two sequence-to-sequence domains: a toy task of adding two three-digit numbers (with reversed input/output order) and phoneme recognition on TIMIT using log-Mel filterbank inputs with phone targets. Both tasks operate on 1D temporal sequences and produce token sequences. The model maintains recurrent state across input blocks and, for TIMIT, uses an LSTM attention mechanism to compute context.

## Evidence
### Task: addition
- "toy task of adding two three-digit decimal numbers." (Section 4.1 Addition Toy Task)
- "The second number is presented in the reverse order, and so is the target output." (Section 4.1 Addition Toy Task)
- "transfers the hidden state across blocks." (Figure 1 caption)
- Inference: In Dynamics set to Fixed (inferred) and Out Dynamics set to Capped (inferred) because the task is "adding two three-digit decimal numbers," implying fixed-length inputs and bounded output length. (Section 4.1 Addition Toy Task)

### Task: phoneme recognition
- "TIMIT, a standard benchmark for speech recognition." (Section 4.2 TIMIT)
- "Log Mel filterbanks were computed every 10ms as inputs to the system." (Section 4.2 TIMIT)
- "The targets were the 60 phones defined for the TIMIT dataset" (Section 4.2 TIMIT)
- "This model used the LSTM attention mechanism." (Section 4.2 TIMIT)
- "maintains its state across the blocks." (Figure 2 caption)
- Inference: In/Out Dynamics set to Open (inferred) because the paper describes an "input acoustic sequence" with phone targets and does not state a fixed maximum length. Attention Dynamic set to Dynamic (inferred) because the model used an LSTM attention mechanism. (Figure 2 caption; Section 4.2 TIMIT)
