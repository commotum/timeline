# Supervising strong learners by amplifying weak experts (Not specified in the paper.)
Source: Supervising strong learners by amplifying weak experts.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Permutation powering query answering | A permutation sigma of N elements and a query asking for sigma^k(x) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | The queried element value sigma^k(x) | 0D (inferred) | Fixed (inferred) |
| Sequential assignments expression evaluation | A function f:{1..8}x{1..8}->{1..8}, a sequence of N assignments, and a variable query | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | The value of the queried variable | 0D (inferred) | Fixed (inferred) |
| Wildcard search summation query answering | A sparse function f:{0,1}^6->{-1,0,1} and a wildcard pattern query | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | The sum of f(x) over all matched x | 0D (inferred) | Fixed (inferred) |
| Shortest path query answering | A directed graph context and a node-pair query (x, y) | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Distance-to-target or first-vertex answer | 0D (inferred) | Fixed (inferred) |
| Union find / rooted-forest query answering | A forest context with component structure and a vertex query | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Root/component-label/path-vertex/distance answer | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates Iterated Amplification on five toy algorithmic question-answering tasks over large combinatorial contexts: permutation powering, sequential assignments, wildcard search, shortest path, and union-find/rooted-forest queries. Task descriptions support 1D (t) symbolic-sequence inputs for permutation/assignment/wildcard tasks and 2D (x, y) relational inputs for shortest-path/union-find style graph tasks (inferred from pairwise fact/edge representations). Input dynamics are capped by explicit bounds (difficulty N from 8 to 64 and 8 to 128 input sentences), while outputs are single queried answers (0D, fixed). Attention and state are inferred as Static and Direct from the described context-conditioned QA architecture.

## Evidence
### Task: Permutation powering query answering
- "Given a permutation  $\sigma: \{1, \ldots, 64\} \to \{1, \ldots, 64\}$ , compute  $\sigma^k(x)$  for k up to 64." (Section 4.1 Tasks)
- "Questions                                                                                    | What is $\sigma^k(x)$ ? (for $2 \le k < 64$ )" (Appendix C, Table 3)
- Inference: `1D (t)` is inferred because the paper states that facts are represented as token sequences and that permutation exponent k is represented symbolically ("For permutation powering, we represent the exponent k in binary." and "In each domain, we can unambiguously represent facts as a sequence of elements from the domain." in Appendix C.1 Representations). `Capped` input dynamics are inferred from explicit bounds ("Each task has a size parameter N that ranges from 8 to 64." in Section C.2 Curriculum, and "All of our inputs are sets of 8 to 128 sentences..." in Section D). `Static` attention and `Direct` state are inferred from fixed context-conditioned answering ("we answer a batch of questions about that context by using N=3 layers which attend over the context embedding" in Section D). `0D`/`Fixed` output are inferred from single-answer query form and architecture output ("This architecture outputs a single  $d_{\rm model}$  dimensional vector for each answer." in Section D).

### Task: Sequential assignments expression evaluation
- "Given a function  $f:\{1,\ldots,8\}^2 \to \{1,\ldots,8\}$  and a sequence of 64 assignments of the form x:=3 or x:=f(y,z), evaluate a particular variable." (Section 4.1 Tasks)
- "Questions                                                                                    | What is the value of $x$ ?" (Appendix C, Table 3)
- Inference: `1D (t)` is inferred from the explicit sequential structure of assignments and token-sequence fact representation (Section 4.1 Tasks; Appendix C.1 Representations). `Capped` input dynamics are inferred from the N bounds and bounded input-set size (Section C.2 Curriculum; Section D). `Static` attention and `Direct` state are inferred from fixed context-to-answer processing in the QA architecture (Section D). `0D`/`Fixed` output are inferred because each query asks for one variable value and each answer is emitted from a single answer vector (Appendix C, Table 3; Section D).

### Task: Wildcard search summation query answering
- "Given a function  $f:\{0,1\}^6 \to \{-1,0,1\}$ , answer questions of the form \"What is the sum of f(x) over all x matching the wildcard expression 0\*\*1\*\*?\"" (Section 4.1 Tasks)
- "Questions                                                                                    | A function $f: \{0,1\}^* \to \{-1,0,1\}$ with N non-zero values.<br>What is $\sum f(x)$ over x matching a wildcard expression (e.g. $0 **1*0$ )?" (Appendix C, Table 3)
- Inference: `1D (t)` is inferred because wildcard expressions and facts are represented as token sequences over bit-symbol strings (Appendix C.1 Representations). `Capped` input dynamics are inferred from explicit N-range curriculum and bounded input sentence counts (Section C.2; Section D). `Static` attention and `Direct` state are inferred from fixed context-conditioned decoding in X (Section D). `0D`/`Fixed` output are inferred from single summed-answer queries and single answer-vector decoding (Section 4.1; Section D).

### Task: Shortest path query answering
- "Given a directed graph with 64 vertices and 128 edges, find the distance from s to t." (Section 4.1 Tasks)
- "Context                                                                                      | A directed graph with $2N$ edges and $N$ vertices." (Appendix C, Table 3)
- "Questions                                                                                    | What is the distance from $x$ to $y$ ? What is the first vertex on the path from $x$ to $y$ ?" (Appendix C, Table 3)
- Inference: `2D (x, y)` is inferred from pairwise relational graph encoding ("we represent a graph as a list of pairs of vertices" in Section D, and "edges (x, y) as the pair xy" in Appendix C.1). `Capped` input dynamics are inferred from N bounds and bounded input set sizes (Section C.2; Section D). `Static` attention and `Direct` state are inferred from fixed context-attending QA computation (Section D). `0D`/`Fixed` output are inferred because each shortest-path query returns one requested value (distance or first vertex) per question (Appendix C, Table 3).

### Task: Union find / rooted-forest query answering
- "Given a rooted forest on 64 vertices, find the root of the tree containing a vertex x." (Section 4.1 Tasks)
- "Context                                                                                      | An undirected forest with $N$ vertices, $\sqrt{N}$ connected components, and one vertex assigned a label in $\{1, \ldots, 8\}$ in each component." (Appendix C, Table 3)
- "Questions                                                                                    | What is the unique label in the component containing $x$ ? What is a vertex on the path from $x$ to a labeled vertex? How far is $x$ from a labeled vertex?" (Appendix C, Table 3)
- Inference: `2D (x, y)` is inferred from pairwise neighborhood/connectivity relations in the forest task (e.g., "Are $x$ and $y$ connected?" in Appendix C, Table 3 primitive questions). `Capped` input dynamics are inferred from the N-range curriculum and bounded input size (Section C.2; Section D). `Static` attention and `Direct` state are inferred from the same fixed context-conditioned QA architecture used across tasks (Section D). `0D`/`Fixed` output are inferred because each question asks for one returned value (root/label/vertex/distance) per query (Section 4.1; Appendix C, Table 3).
