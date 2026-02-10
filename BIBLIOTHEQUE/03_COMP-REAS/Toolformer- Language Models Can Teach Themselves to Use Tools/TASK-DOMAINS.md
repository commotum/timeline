# Toolformer: Language Models Can Teach Themselves to Use Tools (Not specified in the paper)
Source: Toolformer- Language Models Can Teach Themselves to Use Tools.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Tool-augmented language modeling / next-token generation | Plain text token sequences with interleaved API-call text and API responses | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Next-token text continuations (including inserted API results) | 1D (t) (inferred) | Open (inferred) |
| Factual cloze completion (LAMA) | Short text statements with a missing fact | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Missing fact value as a short text span | 0D (inferred) | Capped (inferred) |
| Mathematical reasoning / numeric answer prediction | Context and question text | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Numeric answer | 0D (inferred) | Capped (inferred) |
| Open-domain question answering | Natural-language question text (with optional retrieved text snippets) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Answer text span | 1D (t) (inferred) | Capped (inferred) |
| Multilingual question answering | English context paragraph plus non-English question text | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | English answer text | 1D (t) (inferred) | Capped (inferred) |
| Temporal question answering / temporal fact completion (TEMPLAMA, DATESET) | Temporal cloze queries and date/duration questions in text | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Time-dependent fact/date answer | 0D (inferred) | Capped (inferred) |

## Summary
The paper covers tool-augmented text generation plus five downstream text tasks: factual cloze completion, math numeric answering, open-domain QA, multilingual QA, and temporal QA/completion. All task inputs are naturally text sequences, so the supported input domain is predominantly 1D (t), while outputs include both text spans (1D (t)) and scalar-like answer values (0D) where the task asks for a single fact or number. Runtime behavior is consistently Dynamic attention because Toolformer decides when and which API to call, and state is Constructed because API calls/responses are inserted into the working sequence. Dynamics are mostly Capped in evaluation settings, with the core language-model generation behavior treated as Open.

## Evidence
### Task: Tool-augmented language modeling / next-token generation
- "Our aim is to equip a language model M with the ability to use different tools by means of API calls. We require that inputs and outputs for each API can be represented as text sequences." (Section 2 Approach)
- "We use this new dataset to finetune M, using a standard language modeling objective." (Section 2 Approach)
- Inference: `1D (t)` is inferred from "text sequences"; `Capped` input dynamics is inferred from "Max sequence length 1,024" (Section B Toolformer Training); `Dynamic` attention is inferred from "enables the language model to decide when and how to use which tool" (Section 2); `Constructed` state is inferred from building `x*` by interleaving API calls/results; `Open` output dynamics is inferred from "perform regular decoding" and then "continue the decoding process" after API responses (Section 2 Inference).

### Task: Factual cloze completion (LAMA)
- "For each of these subsets, the task is to complete a short statement with a missing fact (e.g., a date or a place)." (Section 4.2.1 LAMA)
- "we use a slightly more lenient evaluation criterion than exact match and simply check whether the correct word is within the first five words predicted by the model." (Section 4.2.1 LAMA)
- Inference: `1D (t)` input and `Capped` dynamics are inferred from prompted text input plus finite decoding/evaluation window; `0D` output is inferred because the target is a single missing fact value; `Dynamic` attention and `Constructed` state are inferred from the shared tool-use mechanism where Toolformer decides API calls and incorporates returned text (Sections 2 and 4.2).

### Task: Mathematical reasoning / numeric answer prediction
- "We test mathematical reasoning abilities on ASDiv (Miao et al., 2020), SVAMP (Patel et al., 2021) and the MAWPS benchmark (Koncel-Kedziorski et al., 2016)." (Section 4.2.2 Math Datasets)
- "As the required output is always a number, we simply check for the first number predicted by the model." (Section 4.2.2 Math Datasets)
- Inference: `1D (t)` input is inferred from text context/question prompts; `0D` output is inferred from scalar numeric answers; `Capped` output dynamics is inferred from first-number evaluation; `Dynamic` attention and `Constructed` state are inferred from calculator/tool selection and inserted API responses (Sections 2, 4.2.2).

### Task: Open-domain question answering
- "We look at Web Questions (Berant et al., 2013), Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), the three question answering datasets considered by Brown et al. (2020)." (Section 4.2.3 Question Answering)
- "For evaluation, we check whether the first 20 words predicted by a model contain the correct answer instead of requiring an exact match." (Section 4.2.3 Question Answering)
- Inference: `1D (t)` input/output is inferred from question/answer text; `Capped` output dynamics is inferred from the 20-word evaluation window; `Dynamic` attention and `Constructed` state are inferred from optional Wikipedia-search tool use during decoding (Sections 2, 4.2.3).

### Task: Multilingual question answering
- "A context paragraph for each question is provided in English, while the question can be in Arabic, German, Spanish, Hindi, Vietnamese, or Simplified Chinese." (Section 4.2.4 Multilingual Question Answering)
- "Our evaluation metric is the percentage of times the model's generation, capped at 10 words, contains the correct answer." (Section 4.2.4 Multilingual Question Answering)
- Inference: `1D (t)` input/output is inferred from multilingual question and textual answer generation; `Capped` output dynamics is directly supported by "capped at 10 words" and mapped to the glossary label; `Dynamic` attention and `Constructed` state are inferred from learned machine-translation tool usage during inference (Section 4.2.4 and Section 2).

### Task: Temporal question answering / temporal fact completion (TEMPLAMA, DATESET)
- "TEMPLAMA is a dataset built from Wikidata that contains cloze queries about facts that change with time (e.g., \"Cristiano Ronaldo plays for ____\") as well as the correct answer for the years between 2010 and 2020." (Section 4.2.5 Temporal Datasets)
- "DATESET, described in Appendix D, is also generated through a series of templates, but populated using a combination of random dates/durations (e.g., \"What day of the week was it 30 days ago?\"). Critically, knowing the current date is required to answer these questions." (Section 4.2.5 Temporal Datasets)
- Inference: `1D (t)` input and `Capped` dynamics are inferred from text-query prompting and finite answer evaluation; `0D` output is inferred because each query seeks one time-dependent value/fact; `Dynamic` attention and `Constructed` state are inferred from calendar/other tool calls being chosen at runtime and inserted into the sequence (Sections 2, 4.2.5).
