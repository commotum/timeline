# Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Not specified in the paper.)
Source: Direct Preference Optimization- Your Language Model is Secretly a Reward Model.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| controlled sentiment generation | movie review prefix (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | text completion with positive sentiment | 1D (t) (inferred) | Not specified in the paper. |
| summarization | forum post (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | summary of the main points (text) | 1D (t) (inferred) | Not specified in the paper. |
| single-turn dialogue | human query (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | engaging and helpful response (text) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper evaluates DPO on three open-ended text generation tasks: controlled sentiment generation from movie-review prefixes, summarization of Reddit forum posts, and single-turn dialogue response generation. The inputs and outputs are textual sequences, so the task domains are 1D (t) in and out (inferred from prompts, summaries, and responses). Only the IMDb sentiment setup specifies prompt lengths (2-8 tokens), indicating capped input size there, while other dynamics and attention/state properties are not specified.

## Evidence
### Task: controlled sentiment generation
- "In controlled sentiment generation, x is a prefix of a movie review from the IMDb dataset [24]," (Section 6 Experiments, Tasks)
- "the policy must generate y with positive sentiment." (Section 6 Experiments, Tasks)
- "The prompts are prefixes from the IMDB dataset of length 2-8 tokens." (Appendix C.1)
- Inference: Movie-review prefixes and generated responses are text sequences, so In/Out Dimension are 1D (t) (inferred). The stated 2-8 token prompt length implies a capped input size, so In Dynamics is Capped (inferred).

### Task: summarization
- "In summarization, x is a forum post from Reddit; the policy must generate a summary y of the main points in the post." (Section 6 Experiments, Tasks)
- Inference: Forum posts and summaries are text sequences, so In/Out Dimension are 1D (t) (inferred).

### Task: single-turn dialogue
- "Finally, in single-turn dialogue, x is a human query, which may be anything from a question about astrophysics to a request for relationship advice." (Section 6 Experiments, Tasks)
- "A policy must produce an engaging and helpful response y to a user's query;" (Section 6 Experiments, Tasks)
- Inference: Queries and responses are text sequences, so In/Out Dimension are 1D (t) (inferred).

---

## CSV Output (required)
