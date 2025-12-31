# Enhanced Enumeration Techniques for Syntax-Guided Synthesis of Bit-Vector Manipulations (2024)
Source: a3a00f-2024.pdf

## Core reasons
- The paper reframes SyGuS enumeration as a reasoning mechanism by ordering candidates via compact term graphs, filtering with examples, and supplementing the grammar with LLM-suggested subexpressions to focus search on valuable shared structures.
- It layers bottom-up deduction on top of enumeration so that partial expressions coordinate via closed position sets and deduce solutions without enumerating every large term, altering how the computation unfolds.

## Evidence extracts
- "In this paper, we introduce a novel synthesis approach that incorporates a distinct enumeration strategy, wherein the ordering of enumeration is influenced not only by expression size but also by factors such as the recurrence of subexpressions, example-based specification, and the guidance from large language models." (p. 3)
- "Now we are ready to establish a new enumeration order over all expressions in JG. Intuitively, the enumeration order prioritizes expressions with smaller compact term graph than those with larger compact term graph." (p. 9)
- "In Section 4, we have presented our enumerative synthesis algorithm expedited with various techniques, including term-graph-based enumeration order, example-guided filtration, and LLM-enhanced grammars. In this section, we show how the pure enumeration algorithm can be further improved through the incorporation of bottom-up deduction techniques." (p. 14)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
