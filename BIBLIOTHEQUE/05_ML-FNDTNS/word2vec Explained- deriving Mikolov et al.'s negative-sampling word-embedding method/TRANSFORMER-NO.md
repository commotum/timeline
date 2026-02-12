# word2vec Explained: Deriving Mikolov et al.'s Negative-Sampling Word-Embedding Method (2014)
Source: word2vec Explained- deriving Mikolov et al.'s negative-sampling word-embedding method.md

## Binary decision
Decision: TRANSFORMER-NO
Confidence: high
Basis: abstract-aux-only

## Why
- The central method is skip-gram with negative sampling for word-context pairs, not a self-attention architecture.
- The auxiliary analyses describe fixed window/sampling behavior and explicitly characterize attention as static rather than Transformer-style.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract and available auxiliary files were sufficient for a high-confidence decision.

## Evidence
- "This note is an attempt to explain equation (4) (negative sampling) in \"Distributed Representations of Words and Phrases and their Compositionality\" by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado and Jeffrey Dean [2]." (word2vec Explained- deriving Mikolov et al.'s negative-sampling word-embedding method.md, intro before Section 1)
- "Attention Dynamic is marked Static because context consideration is predefined by the window rule rather than runtime selection (Section 3)." (TASK-DOMAINS.md, Evidence -> Task: Context prediction (skip-gram))

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - sufficient evidence for high-confidence TRANSFORMER-NO from abstract + TASK-DOMAINS.md + TASK-DOMAINS.csv + TASK_MODEL_RATIO.md; extending-dimensions analysis markdown was unavailable (`MISSING`).
Pass 2 (targeted source scan): skipped - Pass 1 was sufficient to finalize.
