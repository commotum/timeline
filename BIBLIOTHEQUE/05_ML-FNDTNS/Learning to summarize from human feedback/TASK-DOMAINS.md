# Learning to summarize from human feedback (Not specified in the paper)
Source: Learning to summarize from human feedback.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| summarization (Reddit TL;DR) | Reddit posts (text) | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | summaries / TL;DRs (tokens) | 1D (t) (inferred) | Capped |
| summarization (CNN/DM news) | news articles (CNN/DM) | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | summaries (tokens) | 1D (t) (inferred) | Not specified in the paper. |
| preference prediction (summary ranking) | post + candidate summary (tokens) | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | preference score / log-odds (scalar) | 0D | Fixed |

## Summary
The paper centers on abstractive English text summarization, training policies to summarize Reddit TL;DR posts and demonstrating transfer to CNN/DM news articles. Inputs are token sequences formatted as fixed-size BPE strings, while outputs are token sequences (summaries), with the TL;DR task explicitly capped to fewer than 48 tokens; CNN/DM output length is not explicitly bounded in the paper. In addition, a reward model performs preference prediction over summaries, producing a scalar score. Attention and state dynamics are inferred as static/direct because only fixed-size Transformer decoders without external memory or retrieval are described.

## Evidence
### Task: summarization (Reddit TL;DR)
- "To make short-term progress towards this goal, we focus on abstractive English text summarization, as it has a long history in the NLP community [16, 8, 54, 59, 50], and is a subjective task where we believe it is difficult to quantify summary quality without human judgments." (Section 1 Introduction)
- "We use the TL;DR summarization dataset [63], which contains ~3 million posts from reddit.com across a variety of topics (subreddits), as well summaries of the posts written by the original poster (TL;DRs)." (Section 3.2 Datasets)
- "We define our ground-truth task as producing a model that generates summaries fewer than 48 tokens long that are as good as possible, according to our judgments." (Section 3.2 Task)
- "Our model always receives a byte-pair encoded string of a fixed size." (Appendix B.2 Input format)
- Inference: Inputs/outputs are token sequences (1D), and attention/state are treated as static/direct because the paper describes fixed-size BPE strings and Transformer decoders without any external memory or retrieval mechanism. (Supported by: "Our model always receives a byte-pair encoded string of a fixed size." (Appendix B.2 Input format) and "All of our models are Transformer decoders [62] in the style of GPT-3 [47, 4]." (Section 3.4 Models))

### Task: summarization (CNN/DM news)
- "Our Reddit-trained human feedback models also generate high-quality summaries of news articles on the CNN/DailyMail (CNN/DM) dataset without any news-specific fine-tuning, almost matching the quality of the dataset's reference summaries." (Section 1 Introduction)
- "Our human feedback models can also generate excellent summaries of CNN/DM news articles without any further training (Figure 4)." (Section 4.2 Transfer to summarizing news articles)
- "Our model always receives a byte-pair encoded string of a fixed size. When the input is too small, we pad from the beginning of the input with a padding token, and if the input is too long we truncate the post/article field at newlines to stay under the limit." (Appendix B.2 Input format)
- Inference: Inputs/outputs are token sequences (1D), and attention/state are treated as static/direct because the paper uses fixed-size BPE strings and Transformer decoders without any external memory or retrieval mechanism. (Supported by: "Our model always receives a byte-pair encoded string of a fixed size..." (Appendix B.2 Input format) and "All of our models are Transformer decoders [62] in the style of GPT-3 [47, 4]." (Section 3.4 Models))

### Task: preference prediction (summary ranking)
- "Given a post and a candidate summary, we train a reward model to predict the log odds that this summary is the better one, as judged by our labelers." (Section 3.1 High-level methodology)
- "We train this model to predict which summary  $y \in \{y_0, y_1\}$  is better as judged by a human, given a post x." (Section 3.4 Reward models)
- "To train our reward models, we start from a supervised baseline, as described above, then add a randomly initialized linear head that outputs a scalar value." (Section 3.4 Reward models)
- "Our model always receives a byte-pair encoded string of a fixed size." (Appendix B.2 Input format)
- Inference: The input is treated as a token sequence (1D) and attention/state are treated as static/direct because the model is a fixed-size Transformer decoder with no described external memory or retrieval. (Supported by: "Our model always receives a byte-pair encoded string of a fixed size." (Appendix B.2 Input format) and "All of our models are Transformer decoders [62] in the style of GPT-3 [47, 4]." (Section 3.4 Models))
