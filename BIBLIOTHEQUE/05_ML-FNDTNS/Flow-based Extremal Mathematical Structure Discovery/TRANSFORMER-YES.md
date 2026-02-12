# Flow-Based Extremal Mathematical Structure Discovery (2026)
Source: Flow-based Extremal Mathematical Structure Discovery.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The core FLowBoost model explicitly uses a permutation-equivariant Transformer / set-transformer as the velocity-field architecture used for the main results.
- The extending-dimensions analysis file was unavailable (`MISSING`), but the abstract + available auxiliary files plus targeted model-section scan provided sufficient direct architecture evidence.

## Evidence
- "The velocity field  $v_{\theta}(x,t)$  is parameterized by a permutation-equivariant Transformer [24, 69]" (Flow-based Extremal Mathematical Structure Discovery.md, Section 3.2, Model architecture)
- "The flow model uses a set transformer of width 256, depth 6, and 8 heads" (Flow-based Extremal Mathematical Structure Discovery.md, Section 3.3.3, Experimental settings)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract/aux files established FLowBoost as the central method, but auxiliary files did not explicitly resolve Transformer backbone usage.
Pass 2 (targeted source scan): performed - model sections explicitly confirmed Transformer/set-transformer architecture as central.
