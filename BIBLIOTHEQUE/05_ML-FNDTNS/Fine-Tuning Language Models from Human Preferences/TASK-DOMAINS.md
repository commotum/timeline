# Fine-Tuning Language Models from Human Preferences (Not specified in the paper.)
Source: Fine-Tuning Language Models from Human Preferences.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Positive-sentiment text continuation | BookCorpus excerpt (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Text continuation (positive sentiment) | 1D (t) (inferred) | Capped (inferred) |
| Vividly descriptive text continuation | BookCorpus excerpt (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Text continuation (vividly descriptive) | 1D (t) (inferred) | Capped (inferred) |
| TL;DR summarization | Reddit posts (TL;DR dataset) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Summary text (TL;DR) | 1D (t) (inferred) | Capped (inferred) |
| CNN/Daily Mail summarization | Articles (CNN/Daily Mail dataset) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | Summary text (CNN/Daily Mail) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper applies RL fine-tuning to four text-generation tasks: positive-sentiment continuation, vividly descriptive continuation, TL;DR summarization, and CNN/Daily Mail summarization. All tasks operate on token sequences and produce token sequences; input and output lengths are explicitly bounded, so the interface is 1D (t) with capped dynamics (inferred). Attention and state dynamics are not explicitly described.

## Evidence
### Task: Positive-sentiment text continuation
- "**Sentiment:** Humans are asked to reward \"positive and happy\" continuations." (Section 3.1.2)
- "the policy is presented with an excerpt from the Book-Corpus dataset (Zhu et al., 2015) and generates a continuation of the text." (Section 3.1)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Capped because inputs/outputs are bounded token sequences ("We sample excerpts with lengths of 32 to 64 tokens, and the policy generates 24 additional tokens." Section 3.1).

### Task: Vividly descriptive text continuation
- "**Descriptiveness:** Humans are asked to reward \"vividly descriptive\" continuations." (Section 3.1.2)
- "the policy is presented with an excerpt from the Book-Corpus dataset (Zhu et al., 2015) and generates a continuation of the text." (Section 3.1)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Capped because inputs/outputs are bounded token sequences ("We sample excerpts with lengths of 32 to 64 tokens, and the policy generates 24 additional tokens." Section 3.1).

### Task: TL;DR summarization
- "the TL;DR dataset of Völske et al. (2017)." (Section 3.2)
- "We sample articles or Reddit posts, truncate to 500 tokens" (Section 3.2)
- "let the policy respond with up to 75 tokens." (Section 3.2)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Capped because inputs/outputs are bounded token sequences ("We sample articles or Reddit posts, truncate to 500 tokens" and "let the policy respond with up to 75 tokens." Section 3.2).

### Task: CNN/Daily Mail summarization
- "the CNN/Daily Mail dataset of Hermann et al. (2015)" (Section 3.2)
- "We sample articles or Reddit posts, truncate to 500 tokens" (Section 3.2)
- "let the policy respond with up to 75 tokens." (Section 3.2)
- Inference: In/Out Dimension set to 1D (t) and Dynamics set to Capped because inputs/outputs are bounded token sequences ("We sample articles or Reddit posts, truncate to 500 tokens" and "let the policy respond with up to 75 tokens." Section 3.2).
