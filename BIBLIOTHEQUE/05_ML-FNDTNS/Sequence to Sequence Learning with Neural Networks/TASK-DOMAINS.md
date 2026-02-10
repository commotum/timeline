# Sequence to Sequence Learning with Neural Networks (2014)
Source: Sequence to Sequence Learning with Neural Networks.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Sequence-to-sequence machine translation (direct generation) | Source-language sentence tokens | 1D (t) (inferred) | Open (inferred) | Static (inferred) | Constructed (inferred) | Target-language sentence tokens | 1D (t) (inferred) | Open (inferred) |
| Machine translation hypothesis rescoring (n-best reranking) | Source-language sentence tokens + candidate target-language sentence tokens (1000-best list) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Log-probability scores for candidate translations | 0D (inferred) | Capped (inferred) |

## Summary
The paper applies the model to English-to-French machine translation in two task modes: direct translation generation and rescoring of SMT n-best hypotheses. Both modes operate on token sequences, and the encoder-decoder architecture maps variable-length sentences into an internal fixed-dimensional representation before decoding or scoring. The sequence generation setup supports variable output length via `<EOS>` and is classified as open sequence output. The rescoring setup is tied to 1000-best hypothesis lists and is classified as capped, with scalar scoring outputs.

## Evidence
### Task: Sequence-to-sequence machine translation (direct generation)
- "Our main result is that on an English to French translation task from the WMT'14 dataset, the translations produced by the LSTM achieve a BLEU score of 34.8 on the entire test set, where the LSTM's BLEU score was penalized on out-of-vocabulary words." (Section **Abstract**)
- "The goal of the LSTM is to estimate the conditional probability  $p(y_1,\ldots,y_{T'}|x_1,\ldots,x_T)$  where  $(x_1,\ldots,x_T)$  is an input sequence and  $y_1,\ldots,y_{T'}$  is its corresponding output sequence whose length T' may differ from T." (Section 2 **The model**)
- "Note that we require that each sentence ends with a special end-of-sentence symbol \"<EOS>\", which enables the model to define a distribution over sequences of all possible lengths." (Section 2 **The model**)
- Inference: `In Dimension` and `Out Dimension` are `1D (t)` because both sides are word sequences indexed by order/time ("input sequence" and "output sequence"). `In Dynamics`/`Out Dynamics` are `Open` because the model defines sequences with "all possible lengths" via `<EOS>`. `Attention Dynamic` is `Static` because the described model encodes the full source into fixed vector `v` and decodes from it (Section 2), while attention is discussed as a mechanism used by other work (Section 1). `State Dynamic` is `Constructed` because the model computes outputs "by first obtaining the fixed-dimensional representation v of the input sequence" (Section 2).

### Task: Machine translation hypothesis rescoring (n-best reranking)
- "We applied our method to the WMT'14 English to French MT task in two ways. We used it to directly translate the input sentence without using a reference SMT system and we it to rescore the n-best lists of an SMT baseline." (Section 3 **Experiments**)
- "We also used the LSTM to rescore the 1000-best lists produced by the baseline system [29]. To rescore an n-best list, we computed the log probability of every hypothesis with our LSTM and took an even average with their score and the LSTM's score." (Section 3.2 **Decoding and Rescoring**)
- Inference: `In Dimension` is `1D (t)` because the rescoring objects are sentence hypotheses (token sequences). `In Dynamics` and `Out Dynamics` are `Capped` because the evaluated interface is explicitly "1000-best lists." `Output` is scalar hypothesis score, so `Out Dimension` is `0D` from "computed the log probability of every hypothesis". `Attention Dynamic` is `Static` and `State Dynamic` is `Constructed` by the same encoder-decoder representation mechanism described in Section 2.
