# Pointer Networks (Not specified in the paper.)
Source: Pointer Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Convex hull computation | Planar point set (2D coordinates) | 2D (x, y) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Sequence of input point indices representing convex hull | 1D (t) | Open (inferred) |
| Delaunay triangulation | Planar point set (2D coordinates) | 2D (x, y) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Sequence of triangle index triples representing triangulation | 1D (t) | Open (inferred) |
| Planar symmetric TSP tour | Planar city coordinates (2D points) | 2D (x, y) | Open (inferred) | Dynamic (inferred) | Constructed (inferred) | Permutation of city indices representing tour | 1D (t) | Open (inferred) |

## Summary
The paper applies Pointer Networks to three geometric/combinatorial optimization tasks: convex hulls, Delaunay triangulations, and planar symmetric TSP tours over 2D point sets. Inputs are planar coordinates, while outputs are sequences or permutations of input indices, giving 1D (t) outputs. The model operates over variable-length inputs/outputs and uses attention to point to inputs with encoder/decoder RNN state, which supports Dynamic attention and Constructed state (inferred).

## Evidence
### Task: Convex hull computation
- "Finding the convex hull of a finite number of points is a well understood task in computational geometry" (Section 3.1 Convex Hull)
- "The vectors  $\mathcal{P}_j$  are uniformly sampled from  $[0,1] \times [0,1]$ ." (Section 3.1 Convex Hull)
- "The elements  $C_i$  are indices between 1 and n corresponding to positions in the sequence  $\mathcal{P}$" (Section 3.1 Convex Hull)
- Inference: Marked In/Out Dynamics as Open and Attention/State as Dynamic/Constructed because the paper says the input length is "length of the input, which is variable." and it "uses attention as a pointer to select a member of the input sequence as the output." and "We use two separate RNNs (one to encode the sequence of vectors  $P_j$ , and another one to produce or decode the output symbols  $C_i$ )." (Abstract; Section 2.1 Sequence-to-Sequence Model).

### Task: Delaunay triangulation
- "A Delaunay triangulation for a set  $\mathcal{P}$  of points in a plane is a triangulation" (Section 3.2 Delaunay Triangulation)
- "Each  $C_i$  is a triple of integers from 1 to n corresponding to the position of triangle vertices in  $\mathcal{P}$" (Section 3.2 Delaunay Triangulation)
- Inference: Marked In/Out Dynamics as Open and Attention/State as Dynamic/Constructed because the paper says the input length is "length of the input, which is variable." and it "uses attention as a pointer to select a member of the input sequence as the output." and "We use two separate RNNs (one to encode the sequence of vectors  $P_j$ , and another one to produce or decode the output symbols  $C_i$ )." (Abstract; Section 2.1 Sequence-to-Sequence Model).

### Task: Planar symmetric TSP tour
- "given a list of cities, we wish to find the shortest possible route that visits each city exactly once" (Section 3.3 Travelling Salesman Problem)
- " $\mathcal{P}$  will be the cartesian coordinates representing the cities, which are chosen randomly in the  $[0,1]\times[0,1]$  square." (Section 3.3 Travelling Salesman Problem)
- " $\mathcal{C}^{\mathcal{P}}=\{C_1,\ldots,C_n\}$  will be a permutation of integers from 1 to n representing the optimal path (or tour)." (Section 3.3 Travelling Salesman Problem)
- Inference: Marked In/Out Dynamics as Open and Attention/State as Dynamic/Constructed because the paper says the input length is "length of the input, which is variable." and it "uses attention as a pointer to select a member of the input sequence as the output." and "We use two separate RNNs (one to encode the sequence of vectors  $P_j$ , and another one to produce or decode the output symbols  $C_i$ )." (Abstract; Section 2.1 Sequence-to-Sequence Model).
