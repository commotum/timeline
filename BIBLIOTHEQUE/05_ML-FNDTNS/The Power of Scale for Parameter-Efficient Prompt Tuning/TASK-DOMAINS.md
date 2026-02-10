# The Power of Scale for Parameter-Efficient Prompt Tuning (2021)
Source: The Power of Scale for Parameter-Efficient Prompt Tuning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Classification | text tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | class labels as token sequences | 0D; 1D (t) | Fixed (inferred) |
| Question answering | question and context tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | extractive answer tokens | 1D (t) | Capped (inferred) |
| Paraphrase detection | sentence-pair tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | paraphrase label | 0D | Fixed (inferred) |
| Entity prediction (ReCoRD) | reading-comprehension tokens with masked entity and candidate entities | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | predicted entity text | 0D; 1D (t) | Capped (inferred) |
| Coreference referent generation (WSC) | sentence tokens with highlighted span | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | referent text tokens | 1D (t) | Capped (inferred) |
| Span-corruption reconstruction | text tokens with sentinel-masked spans | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | reconstructed masked-span text with sentinels | 1D (t) | Capped (inferred) |
| Language modeling continuation | natural text prefix tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | natural text continuation tokens | 1D (t) | Capped (inferred) |

## Summary
The paper covers multiple NLP task domains using frozen T5 models with prompt tuning: classification, question answering, paraphrase detection, ReCoRD entity prediction, WSC referent generation, and two generation objectives (span-corruption reconstruction and LM continuation). The task interfaces are text-based and predominantly 1D (t), with some outputs treated as 0D labels (or 0D; 1D (t) when labels/entities are generated as text). Based on architecture and interface descriptions, the tasks use Capped dynamics with Static attention and Direct state in this classification.

## Evidence
### Task: Classification
- "Following the \"text-to-text\" approach of T5 (Raffel et al., 2020), we cast all tasks as text generation. Instead of modeling classification as the probability of an output class given some input,  $\Pr(y|X)$ , where X is a series of tokens and y is a single class label, we now model it as conditional generation, where Y is a sequence of tokens that represent a class label." (Section 2 Prompt Tuning)
- "For classification tasks, a third option is to initialize the prompt with embeddings that enumerate the output classes, similar to the \"verbalizers\" of Schick and Schütze (2021). Since we want the model to produce these tokens in the output, initializing the prompt with the embeddings of the valid target tokens should prime the model to restrict its output to the legal output classes." (Section 2.1 Design Decisions)
- Inference: In Dynamics is Capped (inferred) from "the effectiveness of a prompt is limited by how much conditioning text can fit into the model's input" (Section 1 Introduction). Attention is Static (inferred) and State is Direct (inferred) from the fixed prepending formulation "Prompting is done by prepending a series of tokens, P, to the input X" while keeping base parameters fixed (Section 2 Prompt Tuning). Out Dynamics is Fixed (inferred) because outputs are constrained to "legal output classes" (Section 2.1 Design Decisions).

### Task: Question answering
- "We investigate zero-shot domain transfer on two tasks: question answering (QA) and paraphrase detection. For question answering, we use the MRQA 2019 shared task on generalization (Fisch et al., 2019). This task collects extractive QA datasets in a unified format..." (Section 5 Resilience to Domain Shift)
- "The question answering datasets are extractive datasets with a variety of answers, so there isn't a label distribution to report." (Section A.3 Datasets)
- Inference: In/Out Dynamics are Capped (inferred) from bounded sequence interfaces ("fit into the model's input" in Section 1 Introduction; "T5's shorter sequence length" in Section 3.1 Closing the Gap). Attention is Static (inferred) and State is Direct (inferred) from the fixed prompt-prepended text-to-text setup in Section 2 Prompt Tuning.

### Task: Paraphrase detection
- "Table 2: Mean and stddev of zero-shot domain transfer between two paraphrase detection tasks." (Table 2 caption, Section 5 Resilience to Domain Shift)
- "The first task is QQP (Iyer et al., 2017), which asks if two questions from the community Q&A site Quora are \"duplicates\". The second task is MRPC (Dolan and Brockett, 2005), which asks if two sentences drawn from news articles are paraphrases." (Section 5 Resilience to Domain Shift)
- Inference: In Dynamics is Capped (inferred) from the paper's bounded input-sequence statements (Section 1 Introduction; Section 3.1 Closing the Gap). Attention is Static (inferred) and State is Direct (inferred) from the fixed prompt-prepended formulation in Section 2 Prompt Tuning. Out Dynamics is Fixed (inferred) because the task is binary label prediction (duplicates/paraphrases vs not).

### Task: Entity prediction (ReCoRD)
- "Similarly, the ReCoRD dataset is a multiple choice dataset where the model must predict the masked out entity from a list of possible entities." (Section A.3 Datasets)
- "T5's handling of the ReCoRD and WSC tasks requires the model to generate short, free-form text." (Section 3.2 Ablation Study)
- Inference: Out Dimension is listed as 0D; 1D (t) (inferred) because the model predicts one entity choice and emits it as short generated text. In/Out Dynamics are Capped (inferred), and Attention/State are Static/Direct (inferred), based on the fixed prompt-prepended text-to-text interface and bounded sequence context (Section 2 Prompt Tuning; Section 1 Introduction).

### Task: Coreference referent generation (WSC)
- "By following the T5 preprocessing and text-to-text format, we recast the WSC dataset as a text generation task. Instead of predicting whether a supplied referent is correct for a highlighted span, our model predicts the correct referent directly." (Section A.3 Datasets)
- "Following T5, we cast the WSC dataset to a free-form text generation task where the model generates the referent to the highlighted span instead predicting if the supplied entity is the correct referent of the highlighted span." (Table 13 caption, Section A.3 Datasets)
- Inference: In/Out Dynamics are Capped (inferred), and Attention/State are Static/Direct (inferred), from the same fixed prompt-prepended text-to-text setup and bounded sequence interface described in Sections 1 and 2.

### Task: Span-corruption reconstruction
- "Specifically, T5 is tasked with \"reconstructing\" masked spans in the input text, which are marked with unique sentinel tokens." (Section 2.2 Unlearning Span Corruption)
- "The target output text consists of all the masked content, separated by sentinels, plus a final sentinel." (Section 2.2 Unlearning Span Corruption)
- Inference: In/Out Dynamics are Capped (inferred), and Attention/State are Static/Direct (inferred), from the seq2seq text interface and fixed prompt conditioning mechanism (Sections 1 and 2).

### Task: Language modeling continuation
- "(3) \"LM Adaptation\": ... given a natural text prefix as input, the model must produce the natural text continuation as output." (Section 2.2 Unlearning Span Corruption)
- "Through LM adaptation, we hope to \"quickly\" transform T5 into a model more similar to GPT-3, which always outputs realistic text..." (Section 2.2 Unlearning Span Corruption)
- Inference: In/Out Dynamics are Capped (inferred), and Attention/State are Static/Direct (inferred), based on bounded sequence length and fixed prompt-prepended conditioning in Sections 1 and 2.
