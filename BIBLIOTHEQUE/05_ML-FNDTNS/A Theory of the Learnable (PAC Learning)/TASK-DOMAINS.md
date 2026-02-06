# A Theory of the Learnable (1984)
Source: A Theory of the Learnable (PAC Learning).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Concept learning (deducing recognition program for Boolean concepts) | Vectors of Boolean-variable assignments (0/1/*) from EXAMPLES; ORACLE responses on queried vectors | 1D (t) (inferred) | Fixed (inferred) | Not specified in the paper. | Constructed (inferred) | Boolean expression/program recognizing the concept | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper focuses on learning algorithms that deduce recognition programs for Boolean concepts from example vectors and oracle feedback. Inputs are vectors assigning values to a fixed set of t Boolean variables; outputs are Boolean expressions/programs that approximate the target concept. The learning procedures iteratively build hypothesis expressions (constructed state), while attention dynamics and output dimensionality/dynamics are not explicitly specified.

## Evidence
### Task: Concept learning (deducing recognition program for Boolean concepts)
- "We shall restrict ourselves to skills that consist of recognizing whether a concept (or predicate) is true or not for given data." (Section 1. INTRODUCTION)
- "We shall say that a concept Q has been learned if a program for recognizing it has been deduced." (Section 1. INTRODUCTION)
- "A vector is an assignment to each of the t variables of a value from {0, 1, *}." (Section 2. A LEARNING PROTOCOL FOR BOOLEAN FUNCTIONS)
- "EXAMPLES: This has no input. It gives as output a vector v such that F(v) = 1." (Section 2. A LEARNING PROTOCOL FOR BOOLEAN FUNCTIONS)
- "ORACLE(): On vector v, as input it outputs 1 or 0 according to whether F(v) = 1 or 0." (Section 2. A LEARNING PROTOCOL FOR BOOLEAN FUNCTIONS)
- "The deduction procedure will in each case output an expression that with high likelihood closely approximates the expression to be learned." (Section 1. INTRODUCTION)
- Inference: In Dimension = 1D (t) and In Dynamics = Fixed because the input is defined as a vector assigning values to each of the t variables; State Dynamic = Constructed because the algorithm maintains and updates a hypothesis formula g (e.g., "a new monomial m is added to g."). (Sections 2 and 6)
