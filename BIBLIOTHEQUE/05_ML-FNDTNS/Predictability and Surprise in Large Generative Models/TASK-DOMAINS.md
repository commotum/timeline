# Predictability and Surprise in Large Generative Models (2022)
Source: Predictability and Surprise in Large Generative Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| language modeling / text generation | text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| machine translation | source-language text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | translated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| speech recognition | speech/audio (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text transcription (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| image generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | images | 2D (x, y) (inferred) | Not specified in the paper. |
| video generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | video | 3D (x, y, t) (inferred) | Not specified in the paper. |
| math generation (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | math expressions/solutions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| audio generation (audition) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | audio (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| code generation / program synthesis | text prompts (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | computer programs | 1D (t) (inferred) | Not specified in the paper. |
| arithmetic addition | 3-digit numbers (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | sum (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| language understanding (MMLU) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| recommendation / rating prediction | user demographics + movie rating history + target movie (text prompt) | 1D (t) (inferred) | Capped | Not specified in the paper. | Not specified in the paper. | movie rating (1-5) | 0D (inferred) | Fixed (inferred) |
| recidivism prediction | defendant attribute prompt (sex, age, charges, priors, juvenile counts, race optional) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Yes/No recidivism prediction | 0D (inferred) | Fixed (inferred) |
| dialogue / role-play text generation | user prompts (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | dialogue/role-play text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| open-ended text generation | text prompts/queries (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| poetry generation | prompt including modern and contemporary poems | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | imitation poems | 1D (t) (inferred) | Not specified in the paper. |
| toxic comment generation | text prompts (RealToxicityPrompts) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | short comments / model responses | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The paper covers a wide range of language-centric tasks (language modeling, translation, speech recognition, dialogue, open-ended text generation, recommendation, and recidivism prediction) and also cites generative work in other modalities such as images, video, math, audio, and code. The explicitly described interfaces are mostly text prompts and responses (1D (t)), with scalar outputs for ratings and Yes/No predictions; a capped input length is stated for the recommendation experiment due to context-window limits. Attention and state dynamics are not specified, and several modality-specific tasks are mentioned without explicit interface constraints.

## Evidence
### Task: language modeling / text generation
- "test loss performance on language modeling tasks scales as a predictable function of model size" (Section 2.1)
- Inference: Treated language modeling as text-in/text-out sequences (1D (t)) based on the paper's reference to "language modeling tasks."

### Task: machine translation
- "capabilities such as machine translation and speech recognition increased in a smooth, predictable manner" (Section 2.1)
- Inference: Interpreted machine translation as text-to-text (1D (t)) based on the task name in the quote.

### Task: speech recognition
- "capabilities such as machine translation and speech recognition increased in a smooth, predictable manner" (Section 2.1)
- Inference: Interpreted speech recognition as audio-to-text sequences (1D (t)) based on the task name in the quote.

### Task: image generation
- "generative models for other modalities (e.g., images, video, math, etc.)" (Section 2.1)
- Inference: Treated outputs as images with 2D (x, y) structure based on the "images" modality in the quote.

### Task: video generation
- "generative models for other modalities (e.g., images, video, math, etc.)" (Section 2.1)
- Inference: Treated outputs as video with 3D (x, y, t) structure based on the "video" modality in the quote.

### Task: math generation (inferred)
- "generative models for other modalities (e.g., images, video, math, etc.)" (Section 2.1)
- Inference: Interpreted "math" in the list of generative modalities as math generation with 1D (t) outputs.

### Task: audio generation (audition) (inferred)
- "audition [23], transfer from text to programming [38]" (Section 2.1)
- Inference: Interpreted "audition" as audio generation with 1D (t) outputs.

### Task: code generation / program synthesis
- "transfer from text to programming [38]" (Section 2.1)
- "program synthesis models from Google display dramatic improvements in their ability to create computer programs" (Section 2.2)
- Inference: Treated inputs/outputs as text/code sequences (1D (t)) based on "text to programming" and "computer programs."

### Task: arithmetic addition
- "three digit addition is performed accurately less than 1% of the time" (Section 2.2)
- Inference: Treated the task as numeric input to sum output (1D (t)) based on "three digit addition."

### Task: language understanding (MMLU)
- "the MMLU language understanding benchmark [34]" (Section 2.2)

### Task: recommendation / rating prediction
- "GPT-3-like language models [3] to be used as recommendation systems with zero-shot learning." (A.3 Recommendation System Experiment)
- "it contains demographic information about users (age, occupation, gender, zip code)" (A.3 Recommendation System Experiment)
- "The goal of a recommendation system is to predict these missing values" (A.3 Recommendation System Experiment)
- "our models have a fundamental limit on how large input text sequences can be, as determined by the context window length" (A.3 Recommendation System Experiment)
- Inference: Treated the prompt as 1D (t) text and the rating output as 0D Fixed based on the prompt-and-rating description above.

### Task: recidivism prediction
- "recidivism prediction." (Section 2.3)
- "The defendant is a {sex} aged {age}." (A.4 COMPAS Experiment)
- "Do you think this person will commit another crime within 2 years?" (A.4 COMPAS Experiment)
- "we compute the probability that the next token in the prompt is a Yes and a No." (A.4 COMPAS Experiment)
- Inference: Treated the prompt as 1D (t) text and the Yes/No output as 0D Fixed based on the prompt and Yes/No computation.

### Task: dialogue / role-play text generation
- "AI Dungeon video game fine-tuned GPT-3 for fantasy role-playing" (Section 2.3)
- "we ask an AI assistant [3] to tell us something offensive" (Section 2.4)
- Inference: Treated role-play and assistant interactions as text prompts and responses (1D (t)).

### Task: open-ended text generation
- "Many applications for language models, including chat bots, search engines, text summarization systems, question answer systems, machine translation systems, etc., rely on open-ended text generation." (Section 2.4)
- Inference: Treated these applications as text-in/text-out sequences (1D (t)) based on the stated reliance on text generation.

### Task: poetry generation
- "sample of over three thousand imitation poems generated randomly from a large language model" (A.5 Open Ended Outputs and Creative Expression)
- "samples generated from a prompt including several modern and contemporary poems" (A.5 Open Ended Outputs and Creative Expression)
- Inference: Treated poems and prompts as text sequences (1D (t)).

### Task: toxic comment generation
- "RealToxicityPrompts [29] dataset to elicit short comments in response to real world samples of text (prompts)" (A.6 Toxicity Experiment Details)
- Inference: Treated prompts and comments as text sequences (1D (t)).
