# On neural scaling and the quanta hypothesis (2026)
Source: On neural scaling and the quanta hypothesis.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper explicitly analyzes GPT/Pythia large language models and discusses transformer-specific mechanisms (induction heads and self-attention layers) as part of the main scaling argument.
- Although a ReLU MLP is also used for sparse-parity experiments, Transformer-based LLM scaling is a central results section, so self-attention is materially involved in the paper's core analysis.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract/auxiliary files plus targeted source cues were sufficient for a high-confidence decision.

## Evidence
- "We study this with the Pythia suite of language models trained by Eleuther AI on The Pile corpus." (On neural scaling and the quanta hypothesis.md, Section: Large language model scaling, line 519)
- "Anthropic described the circuit that implements this operation (it requires two self-attention layers), and found that it forms early in training, producing a sharp transition" (On neural scaling and the quanta hypothesis.md, induction-heads discussion, lines 115-117)
- "Learning curve for a 5-layer GPT-2 style transformer trained to output whether the sequence seen so far is a palindrome." (On neural scaling and the quanta hypothesis.md, line 175)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - suggested a Transformer-involving paper (Pythia/GPT evidence in auxiliary analyses) but left mild ambiguity because the OCR abstract structure is noisy and one auxiliary file was unavailable.
Pass 2 (targeted source scan): performed - explicit Transformer/self-attention cues in core discussion confirmed TRANSFORMER-YES.
