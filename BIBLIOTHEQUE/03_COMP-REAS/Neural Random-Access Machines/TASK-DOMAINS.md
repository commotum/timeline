# Neural Random-Access Machines (Not specified in the paper)
Source: Neural Random-Access Machines.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Array element access | array A; index k | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | value A[k] | 0D (inferred) | Fixed (inferred) |
| Array increment | array A | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | array A with each element +1 | 1D (t) (inferred) | Capped (inferred) |
| Array copy | array A; destination pointer p | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | copied array at destination | 1D (t) (inferred) | Capped (inferred) |
| Array reverse | array A; destination pointer p | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | reversed array at destination | 1D (t) (inferred) | Capped (inferred) |
| Array element swap | array A; pointers p and q | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | array A with A[p] and A[q] swapped | 1D (t) (inferred) | Capped (inferred) |
| Array permutation | arrays P and A; pointer a to A | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | permuted array (A reordered by P) | 1D (t) (inferred) | Capped (inferred) |
| Linked-list k-th element lookup | linked list (head pointer); index k; output slot | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | value of k-th element | 0D (inferred) | Fixed (inferred) |
| Linked-list search | linked list (head pointer); value v | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | pointer to first node with value v | 0D (inferred) | Fixed (inferred) |
| Merge sorted arrays | two sorted arrays A and B; pointers a, b, o | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | merged sorted array | 1D (t) (inferred) | Capped (inferred) |
| Binary search tree path lookup | binary search tree (root pointer); path sequence; output slot | 1D (t) (inferred); 0D (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | value at end of path | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates an NRAM on 10 algorithmic tasks over arrays, linked lists, and binary search trees stored in memory. Inputs are primarily 1D sequences with scalar pointers/indices, and outputs are either scalar values/pointers or 1D arrays with sizes bounded by memory. Based on the pointer-based access and external memory tape described, attention is dynamic and state is constructed (inferred).

## Evidence
### Task: Array element access
- "Access Given a value k and an array A, return A[k]." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Array increment
- "**Increment** Given an array, increment all its elements by 1." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Array copy
- "**Copy** Given an array and a pointer to the destination, copy all elements from the array to the given location." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Array reverse
- "**Reverse** Given an array and a pointer to the destination, copy all elements from the array in reversed order." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Array element swap
- "**Swap** Given two pointers p, q and an array A, swap elements A[p] and A[q]." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Array permutation
- "**Permutation** Given two arrays of n elements: P (contains a permutation of numbers 1, ..., n )" (Sec. 4.2 Tasks)
- "and A (contains random elements), permutate A according to P." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Linked-list k-th element lookup
- "**ListK** Given a pointer to the head of a linked list and a number k, find the value of the k-th element on the list." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from lists stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Linked-list search
- "**ListSearch** Given a pointer to the head of a linked list and a value v to find" (Sec. 4.2 Tasks)
- "return a pointer to the first node on the list with the value v." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from lists stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Merge sorted arrays
- "Merge Given pointers to 2 sorted arrays A and B, merge them." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from arrays stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)

### Task: Binary search tree path lookup
- "**WalkBST** Given a pointer to the root of a Binary Search Tree, and a path to be traversed" (Sec. 4.2 Tasks)
- "return the element at the end of the path." (Sec. 4.2 Tasks)
- Inference: In/Out Dimension and Dynamics inferred from trees stored on a "memory tape, which consists of M memory cells"; Attention/State inferred from pointer-based access via the "READ module: this module takes as the input a pointer" and use of an "external variable-size random-access memory." (Sec. 3.2; Abstract)
