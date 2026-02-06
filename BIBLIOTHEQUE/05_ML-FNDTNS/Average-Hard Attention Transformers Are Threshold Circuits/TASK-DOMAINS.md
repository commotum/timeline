# Average-Hard Attention Transformers are Constant-Depth Uniform Threshold Circuits (n.d.)
Source: Average-Hard Attention Transformers Are Threshold Circuits.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language recognition | Symbol string (alphabet tokens) | 1D (t) (inferred) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Accept/reject label (binary decision) (inferred) | 0D (inferred) | Fixed (inferred) |

## Summary
The paper analyzes average-hard attention transformers as devices that recognize formal languages from symbol strings. The only explicit task is language recognition over variable-length sequences, which implies a 1D input domain and a single boolean decision output. Attention behavior is input-dependent via average-hard selection, and transformer layers build new representations; these dynamics are inferred from the model definitions rather than stated as task labels.

## Evidence
### Task: Language recognition
- "Merrill et al. (2022) prove that average-hard attention transformers recognize languages that fall within the complexity class TC<sup>0</sup>" (Abstract)
- "Every language that can be decided by a transformer with average-hard attention is in uniform  $TC^0$ ." (Theorem 2, Section 3 Main result)
- "Let  $\\Sigma = a_1, \\ldots, a_m$  be our alphabet, and let  $\\omega = a_{i_1}, \\ldots, a_{i_n}$  be our input string" (Theorem 2 proof, Section 3 Main result)
- "we need a model that can handle inputs of arbitrarily long strings as input." (Section 2.2 Circuit computations)
- "average-hard attention distributes the entire probability mass evenly among the indices whose values  $s_i$  are maximal." (Definition 3, Section 2.4 Attention)
- "and them being combined with a feed forward network f to produce the output  $X_{l+1}$ ." (Figure 2 caption, Section 2.3 Transformers)
- "Thus, a Boolean circuit defines a function mapping inputs from  $\\{0,1\\}^k$  to outputs in  $\\{0,1\\}$ ." (Section 2.2 Circuit computations)
- Inference: The input is a sequence (string) of symbols of arbitrary length, so the input dimension is 1D (t) and dynamics are Open; average-hard attention selects indices based on maximal scores, so attention is Dynamic; layer outputs \(X_{l+1}\) indicate constructed internal representations, so State is Constructed; language decision implies a single boolean accept/reject output, so the output is 0D and Fixed.
