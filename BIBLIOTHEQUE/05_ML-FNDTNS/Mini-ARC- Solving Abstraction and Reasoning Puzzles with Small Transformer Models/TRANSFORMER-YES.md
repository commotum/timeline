# Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models (2024)
Source: Mini-ARC- Solving Abstraction and Reasoning Puzzles with Small Transformer Models.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: abstract-aux-only

## Why
- The abstract explicitly states the method uses small Transformer models as the core solver for ARC, not just as a baseline.
- Auxiliary analyses consistently describe a Transformer encoder with self-attention as the central architecture used for results.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), but the abstract plus available auxiliary files already provide sufficient direct architecture evidence.

## Evidence
- "In this paper, I explain a novel approach to solving ARC puzzles that uses (1) small (67M param) Transformer models trained exclusively on ARC puzzles, (2) test-time training (TTT), and (3) refinement." (Mini-ARC- Solving Abstraction and Reasoning Puzzles with Small Transformer Models.md, Abstract, line 9)
- "The full embedded sequence is passed through 16 Transformer encoder layers with self-attention mechanisms." (TASK-DOMAINS.md, Evidence section, line 22)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - High-confidence YES from direct Transformer/self-attention statements in abstract and auxiliary files; one auxiliary input (extending-dimensions) was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient for a high-confidence binary decision.
