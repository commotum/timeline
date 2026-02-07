# LEARNING TO EXECUTE (Not specified in the paper.)
Source: Learning to Execute.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| program evaluation | program characters (Python-like code) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | output integer characters (digits; minus; dot) | 1D (t) (inferred) | Capped (inferred) |
| addition | two numbers with "+" as character sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | sum digits (characters) | 1D (t) (inferred) | Capped (inferred) |
| sequence memorization (copying) | digit sequence characters | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | copied digit sequence characters | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates sequence-to-sequence LSTMs on three character-level tasks: program evaluation, addition, and digit-sequence memorization/copying. All tasks operate on 1D sequences with bounded lengths, and outputs are produced one character at a time. Based on the sequential read/write setup and the LSTM memory description, attention is static and state is constructed (both inferred).

## Evidence
### Task: program evaluation
- "training them to evaluate short computer programs" (Abstract)
- "The LSTM reads the program character-by-character and computes the program's output." (Section 1 Introduction)
- "Every program ends with a single \"print\" statement whose output is an integer." (Section 3 Program Subclass)
- Inference: Inferred 1D (t) and static attention because it "reads the entire input one character at a time" and "produces the output one character at a time." (Section 3 Program Subclass). Inferred capped dynamics because programs are "parametrized by their length and nesting" and the LSTM is "unrolled for 50 steps." (Section 3 Program Subclass; Section 6 Experiments). Inferred constructed state from "memory cells." (Section 5 LSTM)

### Task: addition
- "We consider the addition of only two numbers of the same length" (Section 3.1 Addition Task)
- "Input: print (398345+425098)" (Figure 3)
- "Target: 823443" (Figure 3)
- Inference: Inferred 1D (t) and static attention because the model "reads the entire input one character at a time." (Section 3 Program Subclass). Inferred capped dynamics from "two numbers of the same length" and the LSTM being "unrolled for 50 steps." (Section 3.1 Addition Task; Section 6 Experiments). Inferred constructed state from "memory cells." (Section 5 LSTM)

### Task: sequence memorization (copying)
- "task of memorizing a random sequence of numbers." (Section 3.2 Memorization Task)
- "Given an example input 123456789, the LSTM reads it one character at a time" (Section 3.2 Memorization Task)
- "then outputs 123456789 one character at a time." (Section 3.2 Memorization Task)
- "input length ranges from 5 to 65 digits." (Section 6.3 Results on the Memorization Task)
- Inference: Inferred 1D (t) and static attention from the "one character at a time" read/write description. Inferred constructed state because it "stores it in memory." (Section 3.2 Memorization Task). Inferred capped dynamics from "input length ranges from 5 to 65 digits." (Section 6.3 Results on the Memorization Task)
