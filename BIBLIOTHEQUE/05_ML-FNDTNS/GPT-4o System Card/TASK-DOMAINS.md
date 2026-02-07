# GPT-40 System Card (2024)
Source: GPT-4o System Card.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Generation (multimodal response) | text; audio; image; video | 1D (t) (inferred); 2D (x, y) (inferred); 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text; audio; image | 1D (t) (inferred); 2D (x, y) (inferred) | Not specified in the paper. |
| Generation (voice/voice cloning) | audio clip (voice sample) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | audio (synthetic voice) | 1D (t) (inferred) | Not specified in the paper. |
| Identification (speaker identification) | audio (voice) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | speaker identity | 0D (inferred) | Not specified in the paper. |
| Classification (sensitive trait attribution) | audio (voice) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | trait label (e.g., accent, nationality) | 0D (inferred) | Not specified in the paper. |
| Question answering (knowledge) | text questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Text continuation (commonsense) | text prompts | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text continuations | 1D (t) (inferred) | Not specified in the paper. |
| Question answering (clinical knowledge) | clinical questions (text) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Question answering (biological threat creation) | text questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Reading comprehension / QA (underrepresented languages) | text passages and questions (Amharic, Hausa, Yoruba, etc.) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Theory of mind / self-knowledge question answering | text scenarios/questions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text answers | 1D (t) (inferred) | Not specified in the paper. |
| Cybersecurity exploitation (CTF solving) | CTF challenge text; vulnerable systems (web apps, binaries, cryptography systems) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | textual flags | 1D (t) (inferred) | Not specified in the paper. |
| Software engineering / code generation | codebase and instructions | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | code (e.g., Python/CUDA) | 1D (t) (inferred) | Not specified in the paper. |
| Machine learning engineering (audio classification) | audio recordings dataset | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | class labels (JSON list) | 1D (t) (inferred) | Not specified in the paper. |
| Persuasive content generation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text articles/chatbots; audio clips/conversations | 1D (t) (inferred) | Not specified in the paper. |
| Scientific reasoning and figure interpretation | scientific problems; figures/images | 1D (t) (inferred); 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text explanations/identifications | 1D (t) (inferred) | Not specified in the paper. |
| Autonomous action execution (agentic tasks) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | autonomous actions | Not specified in the paper. | Not specified in the paper. |

## Summary
GPT-40 is described as an omni model that takes text, audio, image, and video inputs and can generate text, audio, and image outputs, with emphasis on speech-to-speech use. Evaluations span knowledge and clinical question answering, commonsense text continuation, biological threat QA, cybersecurity CTF solving, persuasion, scientific reasoning and figure interpretation, software/ML engineering tasks, and agentic autonomy. Input dimensions therefore include 1D (t) text/audio, 2D (x, y) images, and 3D (x, y, t) video (inferred from modalities), while dynamics, attention, and state are largely not specified.

## Evidence
### Task: Generation (multimodal response)
- "accepts as input any combination of text, audio, image, and video and generates any combination of text, audio, and image outputs." (Section 1 Introduction)
- Inference: Mapped text/audio to 1D (t), images to 2D (x, y), and video to 3D (x, y, t); outputs text/audio to 1D and images to 2D, based on the modalities listed in the quote.

### Task: Generation (voice/voice cloning)
- "Voice generation is the capability to create audio with a human-sounding synthetic voice, and includes generating voices based on a short input clip." (Section 3.3.1 Unauthorized voice generation)
- Inference: Treated the audio clip and generated voice as 1D (t) signals based on the audio modality described.

### Task: Identification (speaker identification)
- "Speaker identification is the ability to identify a speaker based on input audio." (Section 3.3.2 Speaker identification)
- Inference: Treated input audio as 1D (t) and speaker identity as a 0D label.

### Task: Classification (sensitive trait attribution)
- "Sensitive trait attribution (STA): Making inferences about a speaker that could plausibly be determined solely from audio content." (Section 3.3.4 Ungrounded inference / Sensitive trait attribution)
- "This includes inferences about things such as a speaker's accent or nationality." (Section 3.3.4 Ungrounded inference / Sensitive trait attribution)
- Inference: Treated audio as 1D (t) input and trait labels as 0D outputs.

### Task: Question answering (knowledge)
- "Capabilities: We evaluate on four tasks: TriviaQA, a subset of MMLU, HellaSwag and Lambada." (Section 3.3.3 Disparate performance on voice inputs)
- "TriviaQA and MMLU are knowledge-centric tasks" (Section 3.3.3 Disparate performance on voice inputs)
- Inference: Treated questions and answers as 1D (t) text.

### Task: Text continuation (commonsense)
- "HellaSwag and Lambada are common sense-centric or text-continuation tasks." (Section 3.3.3 Disparate performance on voice inputs)
- Inference: Treated prompts and continuations as 1D (t) text.

### Task: Question answering (clinical knowledge)
- "To better characterize the clinical knowledge of GPT-40, we ran 22 text-based evaluations based on 11 datasets" (Section 5.2 Health)
- Inference: Treated clinical questions and answers as 1D (t) text.

### Task: Question answering (biological threat creation)
- "answering questions relevant to creating a biological threat." (Section 3.6 Biological threats)
- "Tasks assessed covered all the main stages in the biological threat creation process (ideation, acquisition, magnification, formulation, and release)." (Section 3.6 Biological threats)
- Inference: Treated questions and answers as 1D (t) text.

### Task: Reading comprehension / QA (underrepresented languages)
- "creating small novel language-specific reading comprehension evaluation for Amharic, Hausa and Yoruba." (Section 5.4 Underrepresented Languages)
- "ARC-Easy: This subset of the AI2 Reasoning Challenge [59] benchmark focuses on evaluating a model's ability to answer common sense grade-school science questions" (Section 5.4 Underrepresented Languages)
- Inference: Treated reading comprehension inputs and answers as 1D (t) text.

### Task: Theory of mind / self-knowledge question answering
- "They tested whether GPT-4o can model itself (self-awareness) and others (theory of mind) in 14 agent and question-answering tasks." (Section 4.2 Apollo Research assessment)
- Inference: Treated scenarios/questions and answers as 1D (t) text.

### Task: Cybersecurity exploitation (CTF solving)
- "We evaluated GPT-40 on tasks from competitive hacking competitions called Capture the Flag (CTF) challenges." (Section 3.5 Cybersecurity)
- "These CTFs are offensive cybersecurity exercises where humans attempt to find textual flags hidden in purposely vulnerable systems" (Section 3.5 Cybersecurity)
- Inference: Treated challenges and flags as text/code artifacts (1D (t)) because the tasks are described as textual flags and challenges.

### Task: Software engineering / code generation
- "Simple software engineering in service of fraud (building an authenticated proxy for the OpenAI API)." (Section 3.8 Model autonomy)
- "Replace the original functions with CUDA kernels to improve performance, while preserving all the functionality of the original." (Appendix B Sample tasks from METR Evaluations)
- Inference: Treated code/instructions and generated code as 1D (t) text.

### Task: Machine learning engineering (audio classification)
- "Your task is to use this data to create a monkey sound classifier." (Appendix B Example Prompt Excerpts)
- "Answer with a JSON list of the predicted class labels when running inference on the test set." (Appendix B Example Prompt Excerpts)
- Inference: Treated audio recordings as 1D (t) inputs and label lists as 1D (t) outputs.

### Task: Persuasive content generation
- "we evaluated the persuasiveness of GPT-4o-generated articles and chatbots" (Section 3.7 Persuasion)
- "We evaluated the persuasiveness of GPT-40 voiced audio clips and interactive (multi-turn) conversations" (Section 3.7 Persuasion)
- Inference: Treated text and audio outputs as 1D (t) sequences based on the modalities named.

### Task: Scientific reasoning and figure interpretation
- "GPT-40 showed promise on tasks involving specialized scientific reasoning." (Section 5.3 Scientific capabilities)
- "GPT-40 was sometimes capable of interpreting these figures, as well as images of other scientific representations" (Section 5.3 Scientific capabilities)
- Inference: Treated scientific text as 1D (t) input and figures/images as 2D (x, y); outputs as 1D (t) text.

### Task: Autonomous action execution (agentic tasks)
- "We evaluated GPT-40 on an agentic task assessment to evaluate its ability to take autonomous actions required for self-exfiltration, self-improvement, and resource acquisition." (Section 3.8 Model autonomy)
- "Given API access to an Azure account, loading an open source language model for inference via an HTTP API." (Section 3.8 Model autonomy)
