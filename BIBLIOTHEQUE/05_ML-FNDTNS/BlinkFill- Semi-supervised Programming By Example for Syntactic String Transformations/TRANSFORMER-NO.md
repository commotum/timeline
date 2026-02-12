# BlinkFill: Semi-supervised Programming By Example for Syntactic String Transformations (2016)
Source: BlinkFill- Semi-supervised Programming By Example for Syntactic String Transformations.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The paper describes a program-synthesis/DSL system (BlinkFill) based on logical patterns and InputDataGraph, not a neural architecture with self-attention blocks.
- Auxiliary analyses characterize the method as inductive synthesis over string-transformation expressions; no central Transformer-style model is indicated, and the extending-dimensions file was unavailable (`MISSING`).

## Evidence
- "We present a semi-supervised learning technique to significantly reduce this ambiguity by using the logical information present in the input data to guide the synthesis algorithm. We develop a data structure InputDataGraph ... and use this graph to efficiently learn substring expressions in a new PBE system BLINKFILL." (Abstract, BlinkFill- Semi-supervised Programming By Example for Syntactic String Transformations.md)
- "The top-level string expression e is a concatenation of a finite list of substring expressions" (TASK-DOMAINS.md, Evidence citing Section 6.1 String Transformation Language)
- "We have implemented the inductive synthesis algorithm for the string transformation language of BlinkFill in C# as an add-in for Microsoft Excel as well as a Web app" (TASK_MODEL_RATIO.md, quote from Section 8. EXPERIMENTS)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient for a high-confidence NO decision; Extending-dimensions analysis was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 evidence was sufficient.
