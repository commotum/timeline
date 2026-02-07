# Learning Explanatory Rules from Noisy Data (Not specified in the paper.)
Source: Learning Explanatory Rules from Noisy Data.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| predicate learning (predecessor relation) | symbolic arithmetic facts (`zero/1`; `succ/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | predecessor relation (target predicate) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (even predicate; even/odd) | symbolic arithmetic facts (`zero/1`; `succ/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | even predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (even predicate; even/succ2) | symbolic arithmetic facts (`zero/1`; `succ/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | even predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (less-than relation) | symbolic arithmetic facts (`zero/1`; `succ/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | less-than relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (Fizz predicate) | symbolic arithmetic facts (`zero/1`; `succ/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Fizz predicate (divisible by 3) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (Buzz predicate) | symbolic arithmetic facts (`zero/1`; `succ/2`; `pred1/2`; `pred2/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Buzz predicate (divisible by 5) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (member relation) | symbolic list facts (`cons/2`; `value/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | member relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (length relation) | symbolic list facts (`cons/2`; `value/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | length relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (son-of relation) | symbolic family facts (`father/2`; `brother/2`; `sister/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | son-of relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (grandparent relation) | symbolic family facts (`father/2`; `mother/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | grandparent relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (husband-of relation) | symbolic family facts (`father/2`; `daughter/2`; `brother/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | husband-of relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (uncle-of relation) | symbolic family facts (`father/2`; `mother/2`; `brother/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | uncle-of relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (relatedness relation) | symbolic family facts (`parent/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | related relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (father-of relation) | symbolic family facts (`husband/2`; `mother/2`; `brother/2`; `aunt/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | father-of relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (undirected-edge relation) | symbolic graph facts (`edge/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | undirected-edge relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (adjacent-to-red predicate) | symbolic graph facts (`edge/2`; `colour/2`; `red/1`; `green/1`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | adjacent-to-red predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (has-two-children predicate) | symbolic graph facts (`edge/2`; `neq/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | has-at-least-two-children predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (is-bad-node predicate) | symbolic graph facts (`edge/2`; `colour/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | is-bad-node predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (connectedness relation) | symbolic graph facts (`edge/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | connected(X,Y) relation (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (is-cyclic predicate) | symbolic graph facts (`edge/2`) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | is-cyclic predicate (target) | Not specified in the paper. | Not specified in the paper. |
| predicate learning (even from images) | 28 × 28 MNIST images (sequence) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | binary label: even (1) vs odd (0) | 0D (inferred) | Fixed (inferred) |
| predicate learning (left image two-less-than right) | pair of images (left and right) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | binary label: left image is two less than right | 0D (inferred) | Fixed (inferred) |
| predicate learning (right image equals 1) | pair of images (left and right) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | binary label: right image equals 1 | 0D (inferred) | Fixed (inferred) |
| predicate learning (less-than from images) | pair of images (left and right) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | binary label: left image less than right | 0D (inferred) | Fixed (inferred) |
| predicate learning (zero from image) | image of an integer (zero task) | 2D (x, y) (inferred) | Fixed (inferred) | Not specified in the paper. | Not specified in the paper. | binary label: image represents zero | 0D (inferred) | Fixed (inferred) |
| predicate learning (identity relation from images) | classes of images of integers | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | identity relation | Not specified in the paper. | Not specified in the paper. |
| predicate learning (predecessor relation from images) | images of integers | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | predecessor relation on integers | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper evaluates 20 symbolic ILP tasks across arithmetic, list processing, family relations, and graph predicates, each framed as learning a target predicate from relational facts. It also reports ambiguous-data tasks using MNIST images, including even/odd classification, relational comparisons between paired images, and additional image-based zero/identity/predecessor experiments; it notes a total of 10 ambiguous-data tasks but does not specify all of them. Formal dimension/dynamics labels are not specified for symbolic tasks; the image-based tasks explicitly use 28 × 28 inputs with single binary labels, which support inferred 2D inputs and 0D outputs with fixed dynamics.

## Evidence
### Task: predicate learning (predecessor relation)
- "The aim of this task is to learn the *predecessor* relation from examples." (Appendix G.1)

### Task: predicate learning (even predicate; even/odd)
- "The aim of this task is to learn the *even* predicate from examples." (Appendix G.2)

### Task: predicate learning (even predicate; even/succ2)
- "The task is to learn the *even* predicate on natural numbers." (Section 5.3.1)

### Task: predicate learning (less-than relation)
- "The aim of this task is to learn the *less than* relation." (Appendix G.4)

### Task: predicate learning (Fizz predicate)
- "if the number is divisible by 3, they should say \"Fizz\"." (Section 5.3.3)

### Task: predicate learning (Buzz predicate)
- "if the number is divisible by 5, they should say \"Buzz\"." (Section 5.3.3)

### Task: predicate learning (member relation)
- "The task is to learn the *member* relation on lists" (Appendix G.7)

### Task: predicate learning (length relation)
- "The task is to learn the *length* relation, where length(X,Y) is true if the length of list X is Y." (Appendix G.8)

### Task: predicate learning (son-of relation)
- "The task here is to learn the son-of relation" (Appendix G.9)

### Task: predicate learning (grandparent relation)
- "The task here is to learn the *grandparent* relation from various facts involving the father-of and mother-of relations." (Appendix G.10)

### Task: predicate learning (husband-of relation)
- "The task here is to learn the *husband-of* relation from various facts about family relations" (Appendix G.11)

### Task: predicate learning (uncle-of relation)
- "The task here is to learn the *uncle-of* relation from various facts about family relations" (Appendix G.12)

### Task: predicate learning (relatedness relation)
- "The task here is to learn the *related* relation from facts about family relations involving the *parent-of* relation." (Appendix G.13)

### Task: predicate learning (father-of relation)
- "The task is to learn the *father-of* relation given this background data and just two positive examples" (Appendix G.14)

### Task: predicate learning (undirected-edge relation)
- "The task is to learn the unconnected-edge relation" (Appendix G.15)

### Task: predicate learning (adjacent-to-red predicate)
- "The task is to learn the predicate is adjacent to a red node." (Appendix G.16)

### Task: predicate learning (has-two-children predicate)
- "The task is to learn the predicate has at least two children." (Appendix G.17)

### Task: predicate learning (is-bad-node predicate)
- "The task here is to learn the *is-bad-node* predicate" (Appendix G.18)

### Task: predicate learning (connectedness relation)
- "the problem is to learn the connected(X, Y) relation" (Appendix G.19)

### Task: predicate learning (is-cyclic predicate)
- "The task is to learn the *is-cyclic* predicate." (Section 5.3.2)

### Task: predicate learning (even from images)
- "the system is given a sequence of 28 × 28 MNIST images." (Section 5.5.1)
- "The training signal is a single binary value, indicating whether the MNIST image represents a number that is even (1) or odd (0)." (Section 5.5.1)
- Inference: Assigned 2D (x, y) input and 0D output with Fixed dynamics from the 28 × 28 image size and single binary label description. (Section 5.5.1)

### Task: predicate learning (left image two-less-than right)
- "the system is given a pair of images (left and right) each training step." (Section 5.5.2)
- "the number represented by the left image is exactly two less than the number represented by the right image." (Section 5.5.2)
- Inference: Assigned 2D (x, y) input and 0D output with Fixed dynamics from the image-pair input and binary label description. (Section 5.5.2)

### Task: predicate learning (right image equals 1)
- "The training label is a 1 if the number represented by the right image is exactly 1." (Section 5.5.3)
- "The system must learn to ignore the label of the left image." (Section 5.5.3)
- Inference: Assigned 2D (x, y) input and 0D output with Fixed dynamics from the image-pair setup and binary label description. (Section 5.5.3)

### Task: predicate learning (less-than from images)
- "The training label is a 1 if the number represented by the left image is less than the number represented by the right image." (Section 5.5.4)
- Inference: Assigned 2D (x, y) input and 0D output with Fixed dynamics from the image-pair setup and binary label description. (Section 5.5.4)

### Task: predicate learning (zero from image)
- "the zero task (whether the image I am currently seeing represents the integer zero)" (Section 5.5.5)
- Inference: Assigned 2D (x, y) input and 0D output with Fixed dynamics from the image-based zero classification description. (Section 5.5.5)

### Task: predicate learning (identity relation from images)
- "when learning the identity relation, it creates some bijective mapping from classes of images to integers" (Section 5.5.5)

### Task: predicate learning (predecessor relation from images)
- "when learning the predecessor relation on integers, it learned to map the image of n to 9-n" (Section 5.5.5)
