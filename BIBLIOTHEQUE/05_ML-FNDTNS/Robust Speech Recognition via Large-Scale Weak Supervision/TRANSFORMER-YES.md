# Robust Speech Recognition via Large-Scale Weak Supervision (2022)
Source: Robust Speech Recognition via Large-Scale Weak Supervision.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The model architecture is explicitly an encoder-decoder Transformer used as the core model for the paper’s main results.
- The paper describes one multitask sequence-to-sequence Transformer handling transcription, translation, language ID, and VAD in the main approach.
- The extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision used the abstract, available auxiliary files, and targeted model-section scan.

## Evidence
- "We chose an encoder-decoder Transformer (Vaswani et al., 2017) as this architecture has been well validated to scale reliably." (Robust Speech Recognition via Large-Scale Weak Supervision.md, Section 2.2 Model, line 61)
- "A sequence-to-sequence Transformer model is trained on many different speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection." (Robust Speech Recognition via Large-Scale Weak Supervision.md, Figure 1 caption, line 75)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract and auxiliary files were read in full; evidence suggested seq2seq multitask setup, but explicit self-attention/Transformer centrality was not fully explicit; extending-dimensions file was unavailable (`MISSING`).
Pass 2 (targeted source scan): performed - model section and nearby architecture lines explicitly confirmed encoder-decoder Transformer as the central architecture.
