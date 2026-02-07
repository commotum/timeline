# Long Short-Term Memory (1997)
Source: Long Short-Term Memory.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-symbol prediction (embedded Reber grammar) | symbol sequences (embedded Reber grammar strings) | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | next-symbol predictions (per time step) | 1D (t) | Open (inferred) |
| Next-symbol prediction with long time lags (noise-free sequences) | symbol sequences over {a_1..a_{p-1}, x, y} | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | next-symbol predictions (per time step) | 1D (t) | Fixed |
| Next-symbol prediction with no local regularities (random subsequences) | symbol sequences with random subsequences over {a_1..a_{p-1}, x, y} | 1D (t) | Fixed | Static (inferred) | Constructed (inferred) | next-symbol predictions (per time step) | 1D (t) | Fixed |
| Final-symbol prediction with trigger and distractors | symbol sequences with distractors and trigger symbol e | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | final symbol prediction (x or y) at sequence end | 0D | Fixed |
| Binary sequence classification (2-sequence problem) | real-valued time series (single input line; first N elements convey class) | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | binary class label at sequence end (1.0 vs 0.0) | 0D | Fixed |
| Binary sequence classification with noisy informative elements | real-valued time series with Gaussian noise on informative elements | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | binary class label at sequence end | 0D | Fixed |
| Sequence regression of conditional expectations (noisy targets) | real-valued time series (single input line; first N elements convey class) | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | real-valued target at sequence end (0.2 vs 0.8) | 0D | Fixed |
| Adding problem (sum of marked values) | sequence of real-valued pairs with marker component | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | sum of marked values at sequence end (scaled) | 0D | Fixed |
| Multiplication problem (product of marked values) | sequence of real-valued pairs with marker component | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | product of marked values at sequence end | 0D | Fixed |
| Temporal order sequence classification (two symbols) | symbol sequences with two relevant X/Y positions | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | 4-class label at sequence end (Q/R/S/U) | 0D | Fixed |
| Temporal order sequence classification (three symbols) | symbol sequences with three relevant X/Y positions | 1D (t) | Capped | Static (inferred) | Constructed (inferred) | 8-class label at sequence end (Q/R/S/U/V/A/B/C) | 0D | Fixed |

## Summary
The paper evaluates LSTM on synthetic sequence tasks including next-symbol prediction, binary sequence classification under noise, and regression on continuous-valued sequences (adding, multiplication, and conditional expectation). Inputs are 1D temporal sequences with fixed, capped, and open lengths, while outputs are either 1D per-step predictions or 0D sequence-end labels/values. Across tasks, the setups imply static attention over the full input stream and constructed internal state to retain long-range information.

## Evidence
### Task: Next-symbol prediction (embedded Reber grammar)
- "The net's task is to read strings, one symbol at a time, and to permanently predict the next symbol." (Section 5.1)
- "To correctly predict the symbol before last, the net has to remember the second symbol." (Section 5.1)
- Inference: In/Out Dynamics marked Open (inferred) because strings are generated sequentially until termination with no maximum length stated; Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because the task requires remembering earlier symbols. (Section 5.1)

### Task: Next-symbol prediction with long time lags (noise-free sequences)
- "sequentially observes input symbol sequences, one at a time, permanently trying to predict the next symbol" (Section 5.2)
- "To predict the final element, the net has to learn to store a representation of the first element for p time steps." (Section 5.2)
- Inference: Attention Dynamic marked Static (inferred) because the task specifies sequential observation and prediction of each step; State Dynamic marked Constructed (inferred) because the net must store the first element across p time steps. (Section 5.2)

### Task: Next-symbol prediction with no local regularities (random subsequences)
- "we remove compressibility by replacing the deterministic subsequence  $(a_1,a_2,\ldots,a_{p-1})$  by a random subsequence (of length p-1)" (Section 5.2)
- "Again, every next sequence element has to be predicted." (Section 5.2)
- Inference: Attention Dynamic marked Static (inferred) because prediction is required for each sequence element; State Dynamic marked Constructed (inferred) because long-range dependencies remain despite random subsequences. (Section 5.2)

### Task: Final-symbol prediction with trigger and distractors
- "The goal is to predict the last symbol, which always occurs after the "trigger symbol" e." (Section 5.2)
- "To predict the final element, the net has to learn to store a representation of the second element for at least q+1 time steps" (Section 5.2)
- "For a given k, this leads to a uniform distribution on the possible sequences with length q+k+4." (Section 5.2)
- Inference: In Dynamics marked Open (inferred) because the sequence length is defined via q+k+4 with no maximum stated; Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because the task requires storing the second element until the trigger. (Section 5.2)

### Task: Binary sequence classification (2-sequence problem)
- "Task 3a ("2-sequence problem"). The task is to observe and then classify input sequences." (Section 5.3)
- "Only the first N real-valued sequence elements convey relevant information about the class." (Section 5.3)
- "The target at the sequence end is 1.0 for class 1 and 0.0 for class 2." (Section 5.3)
- Inference: Attention Dynamic marked Static (inferred) because the task is defined as sequential observation of the full sequence; State Dynamic marked Constructed (inferred) because relevant information appears early while the target is at the sequence end. (Section 5.3)

### Task: Binary sequence classification with noisy informative elements
- "Task 3b. Architecture, parameters, etc. like in Task 3a, but now with Gaussian noise (mean 0 and variance 0.2) added to the information-conveying elements" (Section 5.3)
- "The task is to observe and then classify input sequences." (Section 5.3)
- Inference: Attention Dynamic marked Static (inferred) because classification is defined over the full sequence; State Dynamic marked Constructed (inferred) because early informative elements must be retained until the sequence end. (Section 5.3)

### Task: Sequence regression of conditional expectations (noisy targets)
- "the targets are 0.2 and 0.8 for class 1 and class 2, respectively" (Section 5.3)
- "To minimize mean squared error, the system has to learn the conditional expectations of the targets given the inputs." (Section 5.3)
- Inference: Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because Task 3c inherits the 2-sequence setup where early elements determine a sequence-end target. (Section 5.3)

### Task: Adding problem (sum of marked values)
- "Each element of each input sequence is a pair of components." (Section 5.4)
- "the task is to output the sum of the first components of those pairs that are marked by second components equal to 1.0." (Section 5.4)
- "An error signal is generated only at the sequence end" (Section 5.4)
- Inference: Attention Dynamic marked Static (inferred) because the task processes the full sequence; State Dynamic marked Constructed (inferred) because the summed values must be retained until sequence end. (Section 5.4)

### Task: Multiplication problem (product of marked values)
- "Like the task in Section 5.4, except that the first component of each pair is a real value randomly chosen from the interval [0,1]." (Section 5.5)
- "The target at sequence end is the product  $X_1 \\times X_2$ ." (Section 5.5)
- Inference: Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because the task retains marked inputs from earlier in the sequence to compute the final product. (Sections 5.4-5.5)

### Task: Temporal order sequence classification (two symbols)
- "Task 6a: two relevant, widely separated symbols. The goal is to classify sequences." (Section 5.6)
- "There are 4 sequence classes Q, R, S, U which depend on the temporal order of X and Y." (Section 5.6)
- "With both tasks, error signals occur only at the end of a sequence." (Section 5.6)
- Inference: Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because classification depends on temporal order of widely separated inputs. (Section 5.6)

### Task: Temporal order sequence classification (three symbols)
- "Task 6b: three relevant, widely separated symbols. Again, the goal is to classify sequences." (Section 5.6)
- "There are 8 sequence classes Q, R, S, U, V, A, B, C which depend on the temporal order of the Xs and Ys." (Section 5.6)
- "With both tasks, error signals occur only at the end of a sequence." (Section 5.6)
- Inference: Attention Dynamic marked Static (inferred) and State Dynamic marked Constructed (inferred) because classification depends on temporal order of widely separated inputs. (Section 5.6)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
Next-symbol prediction (embedded Reber grammar),symbol sequences (embedded Reber grammar strings),1D (t),Open (inferred),Static (inferred),Constructed (inferred),next-symbol predictions (per time step),1D (t),Open (inferred)
"Next-symbol prediction with long time lags (noise-free sequences)","symbol sequences over {a_1..a_{p-1}, x, y}",1D (t),Fixed,Static (inferred),Constructed (inferred),next-symbol predictions (per time step),1D (t),Fixed
"Next-symbol prediction with no local regularities (random subsequences)","symbol sequences with random subsequences over {a_1..a_{p-1}, x, y}",1D (t),Fixed,Static (inferred),Constructed (inferred),next-symbol predictions (per time step),1D (t),Fixed
Final-symbol prediction with trigger and distractors,symbol sequences with distractors and trigger symbol e,1D (t),Open (inferred),Static (inferred),Constructed (inferred),final symbol prediction (x or y) at sequence end,0D,Fixed
Binary sequence classification (2-sequence problem),"real-valued time series (single input line; first N elements convey class)",1D (t),Capped,Static (inferred),Constructed (inferred),binary class label at sequence end (1.0 vs 0.0),0D,Fixed
Binary sequence classification with noisy informative elements,real-valued time series with Gaussian noise on informative elements,1D (t),Capped,Static (inferred),Constructed (inferred),binary class label at sequence end,0D,Fixed
Sequence regression of conditional expectations (noisy targets),"real-valued time series (single input line; first N elements convey class)",1D (t),Capped,Static (inferred),Constructed (inferred),real-valued target at sequence end (0.2 vs 0.8),0D,Fixed
Adding problem (sum of marked values),sequence of real-valued pairs with marker component,1D (t),Capped,Static (inferred),Constructed (inferred),sum of marked values at sequence end (scaled),0D,Fixed
Multiplication problem (product of marked values),sequence of real-valued pairs with marker component,1D (t),Capped,Static (inferred),Constructed (inferred),product of marked values at sequence end,0D,Fixed
Temporal order sequence classification (two symbols),symbol sequences with two relevant X/Y positions,1D (t),Capped,Static (inferred),Constructed (inferred),4-class label at sequence end (Q/R/S/U),0D,Fixed
Temporal order sequence classification (three symbols),symbol sequences with three relevant X/Y positions,1D (t),Capped,Static (inferred),Constructed (inferred),8-class label at sequence end (Q/R/S/U/V/A/B/C),0D,Fixed
