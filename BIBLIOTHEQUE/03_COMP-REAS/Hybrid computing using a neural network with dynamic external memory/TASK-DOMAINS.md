# Hybrid computing using a neural network with dynamic external memory (2016)
Source: Hybrid computing using a neural network with dynamic external memory.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question answering (synthetic bAbI) | word tokens (stories + questions) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | answer word tokens | 1D (t) | Not specified in the paper. |
| graph traversal | graph triples (source, edge, destination) + query triples with missing elements | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | sequence of completed triples along the path | 1D (t) | Not specified in the paper. |
| shortest path (graph) | graph triples + query specifying start and end nodes | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | sequence of triples for a minimum-length path | 1D (t) | Capped |
| graph relation inference | graph triples + query (start node, relation label, missing destination) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | completed triple (start, relation, end) | 0D | Fixed |
| block puzzle control (Mini-SHRDLU) | grid board state + goal constraint sequences + goal label | 2D (x, y); 1D (t) | Capped | Dynamic (inferred) | Constructed (inferred) | block-move actions | 1D (t) | Not specified in the paper. |

## Summary
The paper evaluates DNCs on synthetic natural-language question answering, symbolic graph reasoning (traversal, shortest path, relation inference), and a grid-based block puzzle control task. Inputs are mainly 1D sequences of tokens or graph triples, with Mini-SHRDLU adding a 2D grid board plus sequential goal constraints; outputs are answer tokens, triple sequences, or action sequences. Dynamics are largely unspecified, with explicit caps on shortest-path length and on the number of Mini-SHRDLU goals/constraints, and attention/state are dynamic/constructed by the external-memory architecture.

## Evidence
### Task: question answering (synthetic bAbI)
- "The dataset consists of short 'story' snippets followed by questions with answers that can be inferred from the stories." (Synthetic question answering experiments)
- "presented it to the network in the form of word vectors, one word at a time." (bAbI task descriptions)
- "the most probable word in the network's output distribution was selected as its answer." (bAbI task descriptions)
- Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC "uses differentiable attention mechanisms" and "the memory can be selectively written to as well as read." (System overview)

### Task: graph traversal
- "Each vector encoded a triple consisting of a source label, an edge label and a destination label." (Graph task descriptions)
- "A path on the graph was defined on the basis of a random walk from a random start node." (Traversal)
- "the first input to the network was an incomplete triple with the destination unspecified" (Traversal)
- "the target output was the sequence of complete triples along the path." (Traversal)
- Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC "uses differentiable attention mechanisms" and "the memory can be selectively written to as well as read." (System overview)

### Task: shortest path (graph)
- "a random start and end node were given as the query" (Graph experiments)
- "return a sequence of triples corresponding to a minimum-length path between them." (Graph experiments)
- "Because we considered paths of up to length five" (Graph experiments)
- Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC "uses differentiable attention mechanisms" and "the memory can be selectively written to as well as read." (System overview)

### Task: graph relation inference
- "A query consisted of an incomplete triple specifying a start node and a relation label" (Graph experiments)
- "The single target vector during the answer phase was the completed triple from the query" (Inference)
- Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC "uses differentiable attention mechanisms" and "the memory can be selectively written to as well as read." (System overview)

### Task: block puzzle control (Mini-SHRDLU)
- "Our environment, which we term Mini-SHRDLU, contains a set of numbered blocks on a grid board." (Block puzzle experiments)
- "An agent, given a view of the board as input, can move the top block from a column" (Block puzzle experiments)
- "Each goal, identified by a single-letter label, was composed of several individual constraints on adjacent block pairs" (Block puzzle experiments)
- "transmitted one constraint per time-step" (Block puzzle experiments)
- "Up to 10 goals with 6 constraints each can be sent to the network before action begins." (Mini-SHRDLU)
- "The policy's outputs define the probability of selecting each one of these actions" (Mini-SHRDLU)
- Inference: Attention Dynamic and State Dynamic are marked Dynamic/Constructed because a DNC "uses differentiable attention mechanisms" and "the memory can be selectively written to as well as read." (System overview)
