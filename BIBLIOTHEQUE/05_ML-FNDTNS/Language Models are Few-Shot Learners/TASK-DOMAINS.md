# Language Models are Few-Shot Learners (Year not specified in the paper)
Source: Language Models are Few-Shot Learners.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Language modeling (PTB) | text sequence (PTB) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | next-token text | 1D (t) (inferred) | Capped (inferred) |
| Cloze word prediction (LAMBADA) | paragraph with blank (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | missing word (text) | 1D (t) (inferred) | Capped (inferred) |
| Ending selection (HellaSwag) | story/instruction context + candidate endings (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | best ending choice (label) | 0D (inferred) | Fixed (inferred) |
| Ending selection (StoryCloze) | story context + candidate endings (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct ending choice (label) | 0D (inferred) | Fixed (inferred) |
| Closed-book QA (Natural Questions) | question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Capped (inferred) |
| Closed-book QA (WebQuestions) | question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Capped (inferred) |
| Closed-book QA (TriviaQA) | question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Capped (inferred) |
| Machine translation (Fr/De/Ro <-> En) | source-language text | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | translated text | 1D (t) (inferred) | Capped (inferred) |
| Coreference resolution (Winograd WSC273) | sentence with ambiguous pronoun + candidate referents (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct referent choice (label) | 0D (inferred) | Fixed (inferred) |
| Coreference resolution (Winogrande XL) | sentence with ambiguous pronoun + candidate referents (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct referent choice (label) | 0D (inferred) | Fixed (inferred) |
| Commonsense reasoning (PIQA) | physical commonsense question + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option (label) | 0D (inferred) | Fixed (inferred) |
| Science QA (ARC Easy) | science exam question + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option (label) | 0D (inferred) | Fixed (inferred) |
| Science QA (ARC Challenge) | science exam question + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option (label) | 0D (inferred) | Fixed (inferred) |
| Science QA (OpenBookQA) | science question + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option (label) | 0D (inferred) | Fixed (inferred) |
| Reading comprehension (CoQA) | passage + conversational question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Capped (inferred) |
| Reading comprehension (DROP) | passage + question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text/number | 1D (t) (inferred) | Capped (inferred) |
| Reading comprehension (QuAC) | passage + dialog question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Capped (inferred) |
| Reading comprehension (SQuADv2) | passage + question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer text (or no-answer) | 1D (t) (inferred) | Capped (inferred) |
| Reading comprehension (RACE) | passage + question + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option (label) | 0D (inferred) | Fixed (inferred) |
| Yes/no QA (BoolQ) | passage + yes/no question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | truth-value label | 0D (inferred) | Fixed (inferred) |
| Textual entailment (CB) | premise + hypothesis (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | entailment label | 0D (inferred) | Fixed (inferred) |
| Causal reasoning choice (COPA) | premise + alternatives (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct alternative label | 0D (inferred) | Fixed (inferred) |
| Textual entailment (RTE) | sentence pair (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | entailment label | 0D (inferred) | Fixed (inferred) |
| Word sense disambiguation (WiC) | two sentences with target word (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | same-meaning label | 0D (inferred) | Fixed (inferred) |
| Coreference classification (WSC, SuperGLUE) | passage + pronoun question (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | coreference label | 0D (inferred) | Fixed (inferred) |
| Multi-answer RC (MultiRC) | passage + questions + answer candidates (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | answer correctness labels | 0D (inferred) | Fixed (inferred) |
| Reading comprehension (ReCoRD) | passage + query + candidate entities (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct entity choice (label) | 0D (inferred) | Fixed (inferred) |
| Natural language inference (ANLI) | sentence pair (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | entailment/contradiction/neutral label | 0D (inferred) | Fixed (inferred) |
| Arithmetic problem solving | arithmetic question in natural language | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | numeric answer | 0D (inferred) | Fixed (inferred) |
| Word scrambling/manipulation | scrambled word (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | original word (text) | 1D (t) (inferred) | Capped (inferred) |
| SAT analogies | analogy prompt + options (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | correct option label | 0D (inferred) | Fixed (inferred) |
| News article generation | title/subtitle + few-shot news articles (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | news article text | 1D (t) (inferred) | Capped (inferred) |
| Novel word usage in sentence | definition of new word + examples (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | sentence using new word (text) | 1D (t) (inferred) | Capped (inferred) |
| Grammar correction | ungrammatical sentence (text) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | corrected sentence (text) | 1D (t) (inferred) | Capped (inferred) |

## Summary
GPT-3 is evaluated on a wide range of text-only tasks spanning language modeling, cloze/completion, closed-book QA, translation, commonsense reasoning, reading comprehension, NLI, and synthetic reasoning/generation tasks (e.g., arithmetic, word manipulation, news generation). Inputs are natural-language sequences, and outputs range from generated text (translations, answers, articles) to discrete labels/choices. Where the paper specifies a 2048-token context window, we infer capped sequence dynamics; attention and state dynamics are not explicitly specified.

## Evidence

### Task: Language modeling (PTB)
- "We calculate zero-shot perplexity on the Penn Tree Bank (PTB) [MKM+94] dataset." (Section 3.1.1)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.1.1; Section 2.1)

### Task: Cloze word prediction (LAMBADA)
- "the model is asked to predict the last word of sentences" (Section 3.1.2)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.1.2; Section 2.1)

### Task: Ending selection (HellaSwag)
- "involves picking the best ending to a story or set of instructions." (Section 3.1.3)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.1.3; Section 2.1)

### Task: Ending selection (StoryCloze)
- "involves selecting the correct ending sentence for five-sentence long stories." (Section 3.1.4)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.1.4; Section 2.1)

### Task: Closed-book QA (Natural Questions)
- "we measure GPT-3's ability to answer questions about broad factual knowledge." (Section 3.2)
- "Natural Questions [KPR+19], WebQuestions [BCFL13], and TriviaQA [JCWZ17]" (Section 3.2)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.2; Section 2.1)

### Task: Closed-book QA (WebQuestions)
- "we measure GPT-3's ability to answer questions about broad factual knowledge." (Section 3.2)
- "Natural Questions [KPR+19], WebQuestions [BCFL13], and TriviaQA [JCWZ17]" (Section 3.2)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.2; Section 2.1)

### Task: Closed-book QA (TriviaQA)
- "we measure GPT-3's ability to answer questions about broad factual knowledge." (Section 3.2)
- "Natural Questions [KPR+19], WebQuestions [BCFL13], and TriviaQA [JCWZ17]" (Section 3.2)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.2; Section 2.1)

### Task: Machine translation (Fr/De/Ro <-> En)
- "we evaluate the model's ability to translate between languages" (Section 3)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3; Section 2.1)

### Task: Coreference resolution (Winograd WSC273)
- "involves determining which word a pronoun refers to" (Section 3.4)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.4; Section 2.1)

### Task: Coreference resolution (Winogrande XL)
- "We test GPT-3's performance on both Winogrand and Winogrande" (Section 3.4)
- "involves determining which word a pronoun refers to" (Section 3.4)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.4; Section 2.1)

### Task: Commonsense reasoning (PIQA)
- "asks common sense questions about how the physical world works" (Section 3.5)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.5; Section 2.1)

### Task: Science QA (ARC Easy)
- "ARC [CCE+18] is a dataset of multiple-choice questions collected from 3rd to 9th grade science exams." (Section 3.5)
- "On the "Easy" version of the dataset" (Section 3.5)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.5; Section 2.1)

### Task: Science QA (ARC Challenge)
- "ARC [CCE+18] is a dataset of multiple-choice questions collected from 3rd to 9th grade science exams." (Section 3.5)
- "On the "Challenge" version of the dataset" (Section 3.5)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.5; Section 2.1)

### Task: Science QA (OpenBookQA)
- "Organisms require energy in order to do what?" (Appendix G (Figure G.8))
- "mature and develop." (Appendix G (Figure G.8))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.8); Section 2.1)

### Task: Reading comprehension (CoQA)
- "CoQA [RCM19] a free-form conversational dataset" (Section 3.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.6; Section 2.1)

### Task: Reading comprehension (DROP)
- "DROP [DWD+19], a dataset testing discrete reasoning and numeracy in the context of reading comprehension" (Section 3.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.6; Section 2.1)

### Task: Reading comprehension (QuAC)
- "QuAC [CHI+18] a dataset which requires modeling structured dialog acts and answer span selections of teacher-student interactions." (Section 3.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.6; Section 2.1)

### Task: Reading comprehension (SQuADv2)
- "On SQuAD 2.0 [RJL18], GPT-3 demonstrates its few-shot learning capabilities" (Section 3.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.6; Section 2.1)

### Task: Reading comprehension (RACE)
- "RACE [LXL+17], a multiple choice dataset of middle school and high school english examinations" (Section 3.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.6; Section 2.1)

### Task: Yes/no QA (BoolQ)
- "question: Manhattan comes cheap. true, false, or neither?" (Appendix G (Figure G.29))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.29); Section 2.1)

### Task: Textual entailment (CB)
- "question: The Top Quark is the last of six flavors of quarks predicted by the standard model theory of particle physics. True or False?" (Appendix G (Figure G.30))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.30); Section 2.1)

### Task: Causal reasoning choice (COPA)
- "The son of a former Israeli Prime Minister who was assassinated wrote an op ed about the consequence of violent political rhetoric." (Appendix G (Figure G.5))
- "Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Donald Trump's aggressive rhetoric." (Appendix G (Figure G.5))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.5); Section 2.1)

### Task: Textual entailment (RTE)
- "SuperGLUE includes an NLI dataset, RTE, which evaluates the binary version of the task." (Section 3.8)
- "classifies whether the second sentence logically follows from the first, contradicts the first sentence, or is possibly true (neutral)." (Section 3.8)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.8; Section 2.1)

### Task: Word sense disambiguation (WiC)
- "WiC (which involves determining if a word is being used with the same meaning in two sentences)" (Section 3.7)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.7; Section 2.1)

### Task: Coreference classification (WSC, SuperGLUE)
- "WSC task in the SuperGLUE benchmark, which is presented as binary classification" (Section 3.4)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.4; Section 2.1)

### Task: Multi-answer RC (MultiRC)
- "There are three levels within MultiRC: (1) the passage, (2) the questions, and (3) the answers." (Appendix G (Figure G.15))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.15); Section 2.1)

### Task: Reading comprehension (ReCoRD)
- "(CNN) Yuval Rabin, whose father, Yitzhak Rabin, was assassinated while serving as Prime Minister of Israel" (Appendix G (Figure G.6))
- "Referencing his father, who was shot and killed by an extremist amid political tension in Israel in 1995, Rabin condemned Donald Trump's aggressive rhetoric." (Appendix G (Figure G.6))
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Appendix G (Figure G.6); Section 2.1)

### Task: Natural language inference (ANLI)
- "classifies whether the second sentence logically follows from the first, contradicts the first sentence, or is possibly true (neutral)." (Section 3.8)
- "ANLI is a difficult dataset employing a series of adversarially mined natural language inference questions in three rounds (R1, R2, and R3)." (Section 3.8)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.8; Section 2.1)

### Task: Arithmetic problem solving
- "asking GPT-3 a simple arithmetic problem in natural language" (Section 3.9.1)
- "Q: What is 48 plus 76? A: 124." (Section 3.9.1)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.9.1; Section 2.1)

### Task: Word scrambling/manipulation
- "Each task involves giving the model a word distorted by some combination of scrambling, addition, or deletion of characters, and asking it to recover the original word." (Section 3.9.2)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.9.2; Section 2.1)

### Task: SAT analogies
- "Analogies are a style of multiple choice question" (Section 3.9.3)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text inputs and discrete answer choices/labels for outputs (0D); the 2048-token context window implies capped input dynamics and fixed output size. (Section 3.9.3; Section 2.1)

### Task: News article generation
- "With the title and subtitle of a proposed next article, the model is able to reliably generate short articles in the "news" genre." (Section 3.9.4)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.9.4; Section 2.1)

### Task: Novel word usage in sentence
- "we give GPT-3 the definition of a nonexistent word, such as "Gigamuru", and then ask it to use it in a sentence." (Section 3.9.5)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.9.5; Section 2.1)

### Task: Grammar correction
- "prompts of the form "Poor English Input: <sentence>\n Good English Output: <sentence>"." (Section 3.9.6)
- "All models use a context window of  $n_{\rm ctx} = 2048$  tokens." (Section 2.1)
- Inference: Task descriptions indicate text sequences for input/output, so we mark 1D (t) for both; the 2048-token context window implies capped dynamics. (Section 3.9.6; Section 2.1)