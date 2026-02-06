# DeepProbLog: Neural Probabilistic Logic Programming (Not specified in the paper)
Source: DeepProbLog- Neural Probabilistic Logic Programming.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| addition (single-digit MNIST sum) | pairs of MNIST digit images | 2D (x, y) | Fixed | Not specified in the paper. | Constructed (inferred) | sum of two digits (integer label) | 0D | Fixed |
| addition (multi-digit MNIST sum) | two lists of digit images | 1D (t); 2D (x, y) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | sum number (integer) | 0D | Fixed |
| addition (Forth with carry) | two lists of digits and a carry | 1D (t); 0D | Open (inferred) | Not specified in the paper. | Constructed (inferred) | sum of the numbers and new carry | 1D (t); 0D | Open (inferred) |
| sorting | list of numbers | 1D (t) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | sorted list of numbers | 1D (t) | Open (inferred) |
| word algebra problem solving (WAP) | natural language sentence (math word problem) | 1D (t) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | solution to the question (number) | 0D | Fixed |
| game outcome prediction (coin-ball) | coin image and two RGB pairs | 2D (x, y); 1D (t) | Fixed | Not specified in the paper. | Constructed (inferred) | game outcome (win/loss) | 0D | Fixed |

## Summary
DeepProbLog is evaluated on image-based digit addition (single- and multi-digit), list-based algorithmic tasks (addition with carry, sorting), text-based word algebra, and a probabilistic coin/urn game combining images with RGB triples. Inputs span 2D images and 1D sequences (lists or text), with outputs mostly 0D labels or 1D lists. Dynamics are fixed for single-example inputs (T1, T6) and inferred as open for variable-length list/text tasks; attention dynamics are not specified. State is inferred as constructed because the tasks are implemented via multi-step logic programs with intermediate predicates and carries.

## Evidence
### Task: addition (single-digit MNIST sum)
- "T1: addition(3, 5,8): Instead of using labeled single digits, we train on pairs of images, labeled with the sum of the individual labels." (Section 6, Logical reasoning and deep learning)
- "The DeepProbLog program consists of the clause addition(X,Y,Z):-digit(X,X2), digit(Y,Y2), Z is X2+Y2." (Section 6, Logical reasoning and deep learning)
- "The addition/3 predicate's first two arguments are MNIST digits, and the last is the sum." (Appendix A, Listing 1)
- Inference: State Dynamic is Constructed (inferred) because the program computes intermediate digit variables before producing the sum (see the addition/3 clause with digit/2 and X2/Y2).

### Task: addition (multi-digit MNIST sum)
- "T2: addition([3], [7], [2], [63): the input consists of two lists of images, each element being a digit." (Section 6, Logical reasoning and deep learning)
- "The CNN does not generalize to this variable-length problem setting." (Section 6, Logical reasoning and deep learning)
- "The number/3 predicate's first argument is a list of MNIST images." (Appendix A, Listing 2)
- "It uses the digit/2 neural predicate on each image in the list, summing and multiplying by ten." (Appendix A, Listing 2)
- Inference: In Dynamics is Open (inferred) because the task is described as a "variable-length problem setting" and the inputs are lists without a stated maximum length. State Dynamic is Constructed (inferred) because the program composes digit/2 over list elements to compute a number.

### Task: addition (Forth with carry)
- "T3: forth_addition/4: where the input consists of two numbers and a carry, with the output being the sum of the numbers and the new carry." (Section 6, Program Induction)
- "we go from right to left over all digits, calculating the sum of two digits and taking the carry over to the next pair." (Section 6, Program Induction)
- "The add/5 predicate's arguments are: the two list of input digits, the input carry, the resulting carry and the resulting sum." (Appendix A, Listing 3)
- "It recursively calls itself to loop over both lists, calling the slot/5 predicate on each position." (Appendix A, Listing 3)
- Inference: In/Out Dynamics are Open (inferred) because the addition procedure recurses over lists of digits with no explicit maximum length stated. State Dynamic is Constructed (inferred) because the algorithm carries intermediate state (carry) across steps.

### Task: sorting
- "T4: sort/2: The input consists of a list of numbers, and the output is the sorted list." (Section 6, Program Induction)
- "The program implements bubble sort, but leaves open what to do on each step in a bubble (i.e. whether to swap or not, swap/2)." (Section 6, Program Induction)
- "The bubblesort/3 predicate uses the bubble/3 predicate, and recursively calls itself on the remaining list." (Appendix A, Listing 4)
- "training length of 2 and 3, but performs poorly on a training length of 4; DeepProbLog generalizes well to larger lengths." (Section 6, Program Induction)
- Inference: In/Out Dynamics are Open (inferred) because sorting is defined over lists and the paper discusses generalizing to larger lengths. State Dynamic is Constructed (inferred) because bubble sort proceeds via intermediate list states and swap decisions.

### Task: word algebra problem solving (WAP)
- "The input to the WAPs consists of a natural language sentence describing a simple mathematical problem." (Section 6, Program Induction)
- "the output is the solution to this question." (Section 6, Program Induction)
- "These WAPs always contain three numbers and are solved by chaining 4 steps: permuting the three numbers" (Section 6, Program Induction)
- Inference: In Dynamics is Open (inferred) because the input is a natural language sentence with no maximum length stated. State Dynamic is Constructed (inferred) because the program solves WAPs by chaining multiple intermediate steps.

### Task: game outcome prediction (coin-ball)
- "The input consists of an image, two RGB pairs and the output is the outcome of the game." (Section 6, Probabilistic programming and deep learning)
- "game (Coin, Urn1, Urn2, Result) :- coin (Coin, Side), urn(1, Urn1, C1)." (Appendix A, Listing 6)
- Inference: State Dynamic is Constructed (inferred) because the outcome is computed through intermediate predicates (coin/2, urn/3) before producing Result.
