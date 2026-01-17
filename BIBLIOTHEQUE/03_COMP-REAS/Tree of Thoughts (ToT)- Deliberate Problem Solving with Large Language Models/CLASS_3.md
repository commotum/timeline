# Tree of Thoughts: Deliberate Problem Solving with Large Language Models (Not specified in the paper.)
Source: Tree of Thoughts (ToT)- Deliberate Problem Solving with Large Language Models.md

## Core reasons
- Proposes a new inference framework that changes computation from left-to-right token decoding to deliberate search over multiple reasoning paths.
- Introduces a tree-search mechanism (generation, evaluation, and search algorithms like BFS/DFS) to add planning, lookahead, and backtracking to LM reasoning.

## Evidence extracts
- "To surmount these challenges, we introduce a new framework for language model inference, \"Tree of Thoughts\" (ToT), which generalizes over the popular \"Chain of Thought\" approach to prompting language models, and enables exploration over coherent units of text (\"thoughts\") that serve as intermediate steps toward problem solving." (Abstract)
- "ToT frames any problem as a search over a tree, where each node is a **state**  $s = [x, z_{1\cdots i}]$  representing a partial solution with the input and the sequence of thoughts so far." (Section 3 Tree of Thoughts: Deliberate Problem Solving with LM)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
