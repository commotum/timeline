# Advancing Process Verification for Large Language Models via Tree-Based Preference Learning (2024)
Source: 9dd3d0-2024.pdf

## Core reasons
- Proposes Tree-based Preference Learning Verifier (Tree-PLV), which restructures verification as a best-first reasoning tree search that ranks paths via step-level preference training rather than binary labels, i.e., a new computation/reasoning mechanism.
- Constructs paired training data by tracing the tree from root to leaves and comparing sibling nodes at each decision point, letting the verifier learn a ranking-based objective that more closely matches the intended ranking evaluation of reasoning paths.

## Evidence extracts
- "Tree-based Preference Learning Verifier (Tree-PLV), a novel approach that constructs reasoning trees via a best-first search algorithm and collects step-level paired data for preference training." (p. 2086)
- "A reasoning tree illustrates all potential reasoning paths, starting from the root and branching out to various leaf nodes. Our objective is to create a dataset D consisting of pairs that express preferences of reasoning paths, and we conduct pairwise comparisons between sibling nodes at each decision point along the tree." (p. 2088)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
