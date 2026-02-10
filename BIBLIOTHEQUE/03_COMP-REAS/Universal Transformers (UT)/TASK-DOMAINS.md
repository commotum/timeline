# Universal Transformers (Not specified in the paper.)
Source: Universal Transformers (UT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bAbI question answering | English sentence sequences plus question | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer token | 0D (inferred) | Fixed (inferred) |
| Subject-verb agreement prediction | English sentence token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Verb-form prediction/ranking | 0D (inferred) | Fixed (inferred) |
| LAMBADA language modeling (target-word prediction) | Narrative passage token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Target word token | 0D (inferred) | Fixed (inferred) |
| LAMBADA reading comprehension (target-word selection) | Context tokens plus target-sentence query tokens | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Selected target word token | 0D (inferred) | Fixed (inferred) |
| Copy (algorithmic) | Decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Copied decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) |
| Reverse (algorithmic) | Decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Reversed decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) |
| Integer addition (algorithmic) | Decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Sum as decimal-symbol string sequence | 1D (t) (inferred) | Capped (inferred) |
| Program evaluation (LTE: program) | Program token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Program evaluation result sequence | 1D (t) (inferred) | Capped (inferred) |
| Program evaluation (LTE: control) | Program token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Program evaluation result sequence | 1D (t) (inferred) | Capped (inferred) |
| Program evaluation (LTE: addition) | Program token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Program evaluation result sequence | 1D (t) (inferred) | Capped (inferred) |
| Copy (LTE memorization) | Sequence of tokens/symbols | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Copied sequence | 1D (t) (inferred) | Capped (inferred) |
| Double (LTE memorization) | Sequence of tokens/symbols | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Doubled sequence | 1D (t) (inferred) | Capped (inferred) |
| Reverse (LTE memorization) | Sequence of tokens/symbols | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Reversed sequence | 1D (t) (inferred) | Capped (inferred) |
| Machine translation (WMT14 En-De) | English source-language token sequence | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | German target-language token sequence | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper evaluates Universal Transformers on a broad set of sequence tasks spanning language understanding, language modeling, reading comprehension, algorithmic sequence transformation, program execution, and machine translation. Based on the OCR text, all task inputs are sequence-structured and are best classified as 1D (t), while outputs are either single-token decisions (0D) or token sequences (1D (t)). Dynamics are mostly Capped from explicit maximum lengths or bounded sequence setups, with single-word outputs treated as Fixed. Attention is classified as Static (inferred) and state as Constructed (inferred) from the described self-attentive recurrent updates over internal representations.

## Evidence
### Task: bAbI question answering
- "The bAbi question answering dataset (Weston et al., 2015) consists of 20 different tasks, where the goal is to answer a question given a number of English sentences that encode potentially multiple supporting facts." (Section 3.1)
- "To encode the input, similar to Henaff et al. (2016), we first encode each fact in the story by applying a learned multiplicative positional mask to each word's embedding, and summing up all embeddings. We embed the question in the same way, and then feed the (Universal) Transformer with these embeddings of the facts and questions." (Section 3.1)
- Inference: In Dimension and In Dynamics are marked as `1D (t)` and `Capped` because the task is over sequences of sentences/facts/questions; Attention Dynamic is `Static` from fixed-sequence self-attention ("using a self-attention mechanism to exchange information across all positions in the sequence," Section 2.1); State Dynamic is `Constructed` from iterative state updates ("the model updates its states (memory) in each step," Section 3.1); Out Dimension and Out Dynamics are `0D`/`Fixed` from single-answer behavior shown in examples ("Model's output: bathroom," Appendix F).

### Task: Subject-verb agreement prediction
- "Next, we consider the task of predicting number-agreement between subjects and verbs in English sentences (Linzen et al., 2016)." (Section 3.2)
- "We use the dataset provided by (Linzen et al., 2016) and follow their experimental protocol of solving the task using a language modeling training setup, i.e. a next word prediction objective, followed by calculating the ranking accuracy of the target verb at test time." (Section 3.2)
- Inference: The sentence input is treated as `1D (t)` with `Capped` dynamics; `Static` attention and `Constructed` state follow the shared UT architecture text in Sections 2.1 and 3.1; the output is treated as a single decision/ranking target (`0D`, `Fixed`) from the explicit "rank are higher than is" formulation.

### Task: LAMBADA language modeling (target-word prediction)
- "The LAMBADA task (Paperno et al., 2016) is a language modeling task consisting of predicting a missing target word given a broader context of 4-5 preceding sentences." (Section 3.3)
- "In the former (more challenging) case, a model is simply trained for next-word prediction on the training data, and evaluated on the target words at test time (i.e. the model is trained to predict all words, not specifically challenging target words)." (Section 3.3)
- Inference: The passage is modeled as `1D (t)` with `Capped` dynamics; `Static` attention and `Constructed` state are inferred from Sections 2.1 and 3.1; predicting one target word is treated as `0D` output with `Fixed` output dynamics.

### Task: LAMBADA reading comprehension (target-word selection)
- "The task is evaluated in two settings: as *language modeling* (the standard setup) and as *reading comprehension*." (Section 3.3)
- "In the latter setting, introduced by Chu et al. (2017), the target sentence (minus the last word) is used as query for selecting the target word from the context sentences." (Section 3.3)
- Inference: Input remains sequence-based (`1D (t)`, `Capped`); `Static` attention and `Constructed` state follow Sections 2.1 and 3.1; selected target-word output is treated as a single-token result (`0D`, `Fixed`).

### Task: Copy (algorithmic)
- "We trained UTs on three algorithmic tasks, namely Copy, Reverse, and (integer) Addition, all on strings composed of decimal symbols ('0'-'9')." (Section 3.4)
- "In all the experiments, we train the models on sequences of length 40 and evaluated on sequences of length 400 (Kaiser & Sutskever, 2016)." (Section 3.4)
- Inference: Input and output are sequence strings (`1D (t)`); dynamics are `Capped` from explicit sequence length bounds; `Static` attention and `Constructed` state are inferred from Sections 2.1 and 3.1.

### Task: Reverse (algorithmic)
- "We trained UTs on three algorithmic tasks, namely Copy, Reverse, and (integer) Addition, all on strings composed of decimal symbols ('0'-'9')." (Section 3.4)
- "Table 4: Accuracy (higher better) on the algorithmic tasks." (Section 3.4)
- Inference: Reverse is treated as sequence-to-sequence transformation (`1D (t)` to `1D (t)`), with `Capped` dynamics from the explicit length setup and with `Static` attention / `Constructed` state inferred from Sections 2.1 and 3.1.

### Task: Integer addition (algorithmic)
- "We trained UTs on three algorithmic tasks, namely Copy, Reverse, and (integer) Addition, all on strings composed of decimal symbols ('0'-'9')." (Section 3.4)
- "Table 4: Accuracy (higher better) on the algorithmic tasks." (Section 3.4)
- Inference: Integer addition is treated as digit-string to digit-string transformation (`1D (t)` in/out), with `Capped` dynamics from the described length regime and `Static` attention / `Constructed` state from Sections 2.1 and 3.1.

### Task: Program evaluation (LTE: program)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "Table 6: Character-level (*char-acc*) and sequence-level accuracy (*seq-acc*) results on the Program Evaluation LTE tasks with maximum nesting of 2 and length of 5." (Section 3.4)
- Inference: Program-evaluation inputs are treated as program token sequences and outputs as evaluated result sequences (`1D (t)` in/out); `Capped` dynamics are inferred from explicit maximum nesting/length; `Static` attention and `Constructed` state follow Sections 2.1 and 3.1.

### Task: Program evaluation (LTE: control)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "LTE is a set of tasks indicating the ability of a model to learn to execute computer programs and was proposed by Zaremba & Sutskever (2015)." (Appendix D.4)
- Inference: Control-task inputs/outputs are inferred as sequence-structured program execution pairs (`1D (t)` in/out) with `Capped` dynamics from LTE limits; `Static` attention and `Constructed` state are inferred from Sections 2.1 and 3.1.

### Task: Program evaluation (LTE: addition)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "These tasks include two subsets: 1) program evaluation tasks (program, control, and addition) that are designed to assess the ability of models for understanding numerical operations, if-statements, variable assignments, the compositionality of operations, and more, as well as 2) memorization tasks (copy, double, and reverse)." (Appendix D.4)
- Inference: Program-addition is treated as sequence-structured program execution (`1D (t)` in/out), with `Capped` dynamics (maximum nesting/length) and `Static` attention / `Constructed` state inferred from Sections 2.1 and 3.1.

### Task: Copy (LTE memorization)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "Table 5: Character-level (*char-acc*) and sequence-level accuracy (*seq-acc*) results on the Memorization LTE tasks, with maximum length of 55." (Section 3.4)
- Inference: Memorization copy is treated as sequence-to-sequence mapping (`1D (t)` in/out) with `Capped` dynamics from maximum length 55; `Static` attention and `Constructed` state are inferred from Sections 2.1 and 3.1.

### Task: Double (LTE memorization)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "Table 5: Character-level (*char-acc*) and sequence-level accuracy (*seq-acc*) results on the Memorization LTE tasks, with maximum length of 55." (Section 3.4)
- Inference: Double is treated as sequence-to-sequence mapping (`1D (t)` in/out), with `Capped` dynamics from explicit maximum length and `Static` attention / `Constructed` state inferred from Sections 2.1 and 3.1.

### Task: Reverse (LTE memorization)
- "These tasks include program evaluation tasks (program, control, and addition), and memorization tasks (copy, double, and reverse)." (Section 3.5)
- "Table 5: Character-level (*char-acc*) and sequence-level accuracy (*seq-acc*) results on the Memorization LTE tasks, with maximum length of 55." (Section 3.4)
- Inference: Memorization reverse is treated as sequence-to-sequence mapping (`1D (t)` in/out), with `Capped` dynamics and `Static` attention / `Constructed` state inferred from Sections 2.1 and 3.1.

### Task: Machine translation (WMT14 En-De)
- "We trained a UT on the WMT 2014 English-German translation task using the same setup as reported in (Vaswani et al., 2017) in order to evaluate its performance on a large-scale sequence-to-sequence task." (Section 3.6)
- "Table 7: Machine translation results on the WMT14 En-De translation task trained on 8xP100 GPUs in comparable training setups. All *base* results have the same number of parameters." (Section 3.6)
- Inference: Input and output are treated as source and target token sequences (`1D (t)` in/out) with `Capped` dynamics; `Static` attention and `Constructed` state are inferred from the UT encoder/decoder description (Section 2.1), including "it produces its output one symbol at a time" for sequence generation.
