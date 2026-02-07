# In-context Learning and Induction Heads (2022)
Source: In-context Learning and Induction Heads.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| next-token prediction (language modeling) | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (next-token predictions) | 1D (t) (inferred) | Capped (inferred) |
| sequence copying / pattern completion | tokens (repeated sequences) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (copied continuation) | 1D (t) (inferred) | Capped (inferred) |
| translation | tokens (multilingual text) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (translated text) | 1D (t) (inferred) | Capped (inferred) |
| pattern classification (template labeling) | tokens (templated lines) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | label token (0-3) | 0D (inferred) | Fixed (inferred) |
| question answering | tokens (questions) (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (answers) (inferred) | 1D (t) (inferred) | Capped (inferred) |
| arithmetic (few-digit addition) | tokens (numbers) (inferred) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | tokens (numeric answers) (inferred) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper studies transformer language models that perform next-token prediction over token sequences and measures in-context learning within a length-512 context. It provides concrete in-context task behaviors for induction heads (literal sequence copying, multilingual translation, and template-based pattern labeling) and notes prompted capabilities like question answering and arithmetic. Across these tasks, inputs and outputs are token sequences (plus scalar label tokens), with 1D structure and capped context windows, while attention and internal processing are described as dynamically attending to prior tokens and copying information (dimensions/dynamics/attention/state inferred from those descriptions).

## Evidence
### Task: next-token prediction (language modeling)
- "tokens later in the context are easier to predict than tokens earlier in the context." (Section Key Concepts - In-context Learning)
- "use earlier elements in the sequence to predict later ones" (Section Key Concepts - In-context Learning)
- Inference: Marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads).

### Task: sequence copying / pattern completion
- "complete token sequences like  $[A][B] ... [A] \rightarrow [B]$ ." (Abstract)
- "the sequence ...[A][B]...[A] to be more likely to be completed with [B]." (Section Induction Heads)
- Inference: Marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads).

### Task: translation
- "showcasing translation between English, French, and German." (Argument 4: Behavior 2: Translation)
- Inference: Marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads).

### Task: pattern classification (template labeling)
- "Each line follows one of four templates, followed by a label for which template it is drawn from." (Argument 4: Behavior 3: Pattern matching)
- Inference: Marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads); marked 0D/Fixed output because the output is a "label" for the template (Argument 4: Behavior 3: Pattern matching).

### Task: question answering
- "such as translation, question-answering, arithmetic, and many other tasks." (Section Key Concepts - In-context Learning)
- Inference: Marked token I/O because tasks are "framed in a next-token-prediction format" (Section Key Concepts - In-context Learning); marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads).

### Task: arithmetic (few-digit addition)
- "few-digit addition" (Section Key Concepts - In-context Learning)
- Inference: Marked token I/O because tasks are "framed in a next-token-prediction format" (Section Key Concepts - In-context Learning); marked 1D (t)/Capped because of "length-512 context" (Section Key Concepts - In-context Learning); marked Dynamic/Constructed because "The first attention head copies information from the previous token into each token" and it can "attend to tokens based on what happened before them" (Section Induction Heads).
