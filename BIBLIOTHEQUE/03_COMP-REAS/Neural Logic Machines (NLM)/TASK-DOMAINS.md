# NEURAL LOGIC MACHINES (Not specified in the paper.)
Source: Neural Logic Machines (NLM).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Relational reasoning (family tree) | Family tree relations/properties (predicates over members) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Member property/relation predicates (e.g., HasFather, IsUncle) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| Relational reasoning (general graphs) | Graph relations and node properties (HasEdge adjacency, node color) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Node properties/relations (AdjacentToRed, k-Connectivity, k-OutDegree) | 1D (t); 2D (x, y) (inferred) | Capped (inferred) |
| Decision making (blocks world) | Object properties and pairwise comparison relations | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Action selection Move(i,j) | 2D (x, y) (inferred) | Capped (inferred) |
| Decision making (sorting) | Index relations and numeral relations between array elements | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Swap action pair (i,j) | 2D (x, y) (inferred) | Capped (inferred) |
| Decision making (path finding) | Graph relations with start/target node properties | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Next-node action choice | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates NLMs on symbolic relational reasoning over family trees and general graphs, plus decision-making algorithmic tasks in blocks world, sorting, and path finding. Inputs are object-centric predicates and pairwise relations; outputs are predicted predicates for properties/relations or action choices over objects/pairs. The supported address spaces are primarily 2D (x, y) relational grids with some unary 1D predicates, and task sizes vary within stated bounds (all inferred). Attention is static over full predicate tensors and state is constructed via multi-layer rule deductions (inferred).

## Evidence
### Task: Relational reasoning (family tree)
- "The family tree is a benchmark for inductive logic programming, where the machine is given a family tree containing m members." (Section 3.2)
- "The goal of the task is to reason out other properties of family members or relations between them." (Section 3.2)
- "for property prediction, we use tensor  $O_D^{(1)}$  to represent unary predicates" (Section 3.2)
- "for relation prediction we use tensor  $O_D^{(2)}$  to represent binary predicates" (Section 3.2)
- "All models are trained on instances of size 20 and tested on instances of size 20 and 100" (Section 3.2)
- Inference: In/Out Dimension and Dynamics inferred from unary/binary predicates and variable sizes; Attention Static and State Constructed inferred from full predicate tensors and layered abstractions ("NLMs take input tensors of predicates (premises), perform layer-by-layer computations, and output tensors as conclusions."; "As the number of layers increases, higher levels of abstraction can be formed.") (Section 2.2)

### Task: Relational reasoning (general graphs)
- "We treat each node in the graph as an object (symbol)." (Section 3.3)
- "The (undirected) graph is fed into the model in the form of a \"HasEdge\" relation between nodes (which is an adjacent matrix)." (Section 3.3)
- "A node has the property of AdjacentToRed if it is adjacent to a red node by an outgoing edge." (Section 3.3)
- "k-Connectivity is a relation between two nodes in the graph" (Section 3.3)
- "A node has property k-OutDegree if its out-degree is exactly k." (Section 3.3)
- "All models are trained on instances of size 10 and tested on instances of size 10 and 50" (Section 3.3)
- Inference: In/Out Dimension and Dynamics inferred from node properties plus pairwise relations and variable sizes; Attention Static and State Constructed inferred from full predicate tensors and layered abstractions ("NLMs take input tensors of predicates (premises), perform layer-by-layer computations, and output tensors as conclusions."; "As the number of layers increases, higher levels of abstraction can be formed.") (Section 2.2)

### Task: Decision making (blocks world)
- "The task is to take actions in the operating world and make its configuration the same as the target world." (Section 3.4)
- "Each object (blocks or ground) can be represented by four properties: world_id, object_id, coordinate_x, coordinate_y." (Section 3.4)
- "The input is the result of the numeral comparison among all pairs of objects" (Section 3.4)
- "The only operation is  $\mathtt{Move}(i,j)$" (Section 3.4)
- "The action space is  $(m+1)\times m$  where m is the number of blocks in the world" (Section 3.4)
- Inference: In/Out Dimension and Dynamics inferred from pairwise relations and bounded action space; Attention Static and State Constructed inferred from full predicate tensors and layered abstractions ("NLMs take input tensors of predicates (premises), perform layer-by-layer computations, and output tensors as conclusions."; "As the number of layers increases, higher levels of abstraction can be formed.") (Section 2.2)

### Task: Decision making (sorting)
- "Given a length-m array a of integers, the algorithm needs to iterative swap elements to sort the array in ascending order." (Section 3.5)
- "We treat each slot in the array as an object, and input their index relations (whether i < j)" (Section 3.5)
- "and numeral relations (whether a[i] < a[j])" (Section 3.5)
- "The action space is  $m \times (m-1)$  indicating the pair of integers to be swapped." (Section 3.5)
- Inference: In/Out Dimension and Dynamics inferred from pairwise relations and bounded action space; Attention Static and State Constructed inferred from full predicate tensors and layered abstractions ("NLMs take input tensors of predicates (premises), perform layer-by-layer computations, and output tensors as conclusions."; "As the number of layers increases, higher levels of abstraction can be formed.") (Section 2.2)

### Task: Decision making (path finding)
- "Given an undirected graph represented by its adjacency matrix as relations, the algorithm needs to find a path from a start node s" (Section 3.5)
- "to the target node t (with property  ${\tt IsTarget}(t) = {\tt True}$ )." (Section 3.5)
- "Specifically, the agent iteratively chooses the next node next along the path." (Section 3.5)
- "we set the maximum distance between s and t to be 5 during the training" (Section 3.5)
- "and set the distance between s and t to be 4 during the testing" (Section 3.5)
- Inference: In/Out Dimension and Dynamics inferred from node properties/relations and capped path length; Attention Static and State Constructed inferred from full predicate tensors and layered abstractions ("NLMs take input tensors of predicates (premises), perform layer-by-layer computations, and output tensors as conclusions."; "As the number of layers increases, higher levels of abstraction can be formed.") (Section 2.2)

## CSV Output (required)
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
"Relational reasoning (family tree)","Family tree relations/properties (predicates over members)","1D (t); 2D (x, y) (inferred)","Capped (inferred)","Static (inferred)","Constructed (inferred)","Member property/relation predicates (e.g., HasFather, IsUncle)","1D (t); 2D (x, y) (inferred)","Capped (inferred)"
"Relational reasoning (general graphs)","Graph relations and node properties (HasEdge adjacency, node color)","1D (t); 2D (x, y) (inferred)","Capped (inferred)","Static (inferred)","Constructed (inferred)","Node properties/relations (AdjacentToRed, k-Connectivity, k-OutDegree)","1D (t); 2D (x, y) (inferred)","Capped (inferred)"
"Decision making (blocks world)","Object properties and pairwise comparison relations","1D (t); 2D (x, y) (inferred)","Capped (inferred)","Static (inferred)","Constructed (inferred)","Action selection Move(i,j)","2D (x, y) (inferred)","Capped (inferred)"
"Decision making (sorting)","Index relations and numeral relations between array elements","2D (x, y) (inferred)","Capped (inferred)","Static (inferred)","Constructed (inferred)","Swap action pair (i,j)","2D (x, y) (inferred)","Capped (inferred)"
"Decision making (path finding)","Graph relations with start/target node properties","1D (t); 2D (x, y) (inferred)","Capped (inferred)","Static (inferred)","Constructed (inferred)","Next-node action choice","1D (t) (inferred)","Capped (inferred)"
