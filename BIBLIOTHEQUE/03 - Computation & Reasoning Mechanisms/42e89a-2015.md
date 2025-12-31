# Pointer Networks (2017)
Source: 42e89a-2015.pdf

## Core reasons
- Proposes a new neural architecture (Ptr-Net) that changes how sequence outputs are computed by using attention as pointers to input elements.
- Addresses the missing capability of variable-size output dictionaries for combinatorial/sequence problems by modeling outputs over input positions via attention.

## Evidence extracts
- "that, instead of using attention to blend hidden units of an encoder to a context vector at each decoder step, it uses attention as a pointer to select a member of the input sequence as the output. We call this architecture a Pointer Net (Ptr-Net)." (p. 1)
- "where the size of the output dictionary is equal to the length of the input sequence. To solve this problem we model p(C |C ,...,C       , P) using the attention mechanism of Equation 3 as follows:" (p. 4)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
