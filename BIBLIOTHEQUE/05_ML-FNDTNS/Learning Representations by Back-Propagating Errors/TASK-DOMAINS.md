# Learning representations by back-propagating errors (1986)
Source: Learning Representations by Back-Propagating Errors.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Symmetry detection (classification) | Binary activity levels of a one-dimensional input vector | 1D (t) | Fixed (inferred) | Static (inferred) | Constructed | Symmetry indicator (output unit on/off) | 0D | Fixed (inferred) |
| Family-tree triple completion (relation inference) | Person 1 symbol + relationship symbol (two input units) | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Constructed | Person 2 symbol(s) | 0D (inferred) | Fixed (inferred) |
| Iterative search | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Sequential structure learning | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper demonstrates back-propagation on fixed-size feed-forward networks for symmetry detection in 1D binary input vectors and for completing family-tree relation triples. Both tasks use static, fixed-size inputs/outputs and rely on constructed internal representations via hidden units. It also states that iterative nets can learn iterative searches and sequential structures, but provides no task I/O details for those.

## Evidence
### Task: Symmetry detection (classification)
- "A network that has learned to detect mirror symmetry in the input vector." (Fig. 1)
- "binary activity levels of a one-dimensional array of input units" (Learning representations by back-propagating errors section)
- "the output unit, having a positive bias, will be on." (Fig. 1)
- "internal 'hidden' units which are not part of the input or output come to represent important features of the task domain" (Learning representations by back-propagating errors section)
- Inference: In/Out Dynamics and Attention are marked as fixed/static because the task uses a fixed set of input vectors and a fixed input-to-output mapping with no runtime selection described.

### Task: Family-tree triple completion (relation inference)
- "produce the third term of each triple when given the first two." (Fig. 2)
- "The first two terms are encoded by activating two of the input units" (Fig. 2)
- "activating the output unit that represents the third term." (Fig. 2)
- "24 input units on the left for representing (person 1) and 12 input units on the right for representing the relationship." (Fig. 3)
- "learn distributed representations of people." (Fig. 4)
- Inference: Input/Output dimension and dynamics and Attention are inferred from the fixed groups of input/output units and the fixed mapping described for the triple-completion network.

### Task: Iterative search
- "learn to perform iterative searches" (Fig. 5)

### Task: Sequential structure learning
- "learn sequential structures" (Fig. 5)
