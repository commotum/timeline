# Do Transformers Really Perform Bad for Graph Representation? (Not specified in the paper.)
Source: Graphormer- Do Transformers Really Perform Bad for Graph Representation-.md

## Core reasons
- The paper adapts the standard Transformer architecture specifically for graph representation learning, positioning the contribution as enabling Transformers to work on graph-structured data.
- The core contribution is a set of graph-structural encodings (centrality, spatial, edge) to inject graph structure into Transformer attention, which is an adaptation to the graph domain rather than a generic positional-encoding-only tweak.

## Evidence extracts
- "In this paper, we solve this mystery by presenting Graphormer, which is built upon the standard Transformer architecture, and could attain excellent results on a broad range of graph representation learning tasks" (Abstract)
- "The Transformer is originally designed for sequence modeling. To utilize its power in graphs, we believe the key is to properly incorporate structural information of graphs into the model." (Section 1 Introduction)

## Classification
Class name: Increasing Transformer's Dimensions
Class code: 2

$$
\boxed{2}
$$
