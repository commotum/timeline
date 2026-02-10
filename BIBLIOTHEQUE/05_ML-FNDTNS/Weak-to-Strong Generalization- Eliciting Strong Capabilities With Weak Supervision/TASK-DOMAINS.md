# Weak-to-Strong Generalization: Eliciting Strong Capabilities With Weak Supervision (2023)
Source: Weak-to-Strong Generalization- Eliciting Strong Capabilities With Weak Supervision.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Binary text classification (NLP benchmarks) | Text examples (questions, contexts, answers, statements) as token sequences | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Binary class label (0/1) or class probability | 0D | Fixed |
| Chess move prediction (first-move generation) | Chess positions with move-history prompts | 2D (x, y); 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | First/best chess move completion | 1D (t) (inferred) | Capped (inferred) |
| Pairwise preference classification (reward modeling) | Dialog prefix plus two candidate completions | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Preferred completion indicator / preference probability | 0D | Fixed |
| Generative language modeling on RM comparison data | Prefix-completion text pairs from ChatGPT comparison data | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Next-token predictions over completion text | 1D (t) | Capped (inferred) |
| Image classification (ImageNet) | Images from ImageNet validation data | 2D (x, y) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | Image class prediction (Top-1 label) | 0D | Fixed |

## Summary
The paper primarily studies token-based tasks: NLP binary classification, chess move generation from chess prompts, reward-model preference prediction, and an auxiliary language-modeling objective on chat comparison text. It also includes an appendix setting in computer vision (ImageNet image classification). The justified dimensions therefore span mostly 1D (t), with 2D (x, y) for image/chess board structure, and outputs are mostly 0D for classification except generative text outputs in chess/LM settings. Across rows, interface-level attention/state details are not explicitly labeled in the paper, but are inferable as Static attention and Constructed state from the described pretrained-model finetuning/linear-probe setups.

## Evidence
### Task: Binary text classification (NLP benchmarks)
- "We consider 22 popular NLP classification datasets covering ethics, commonsense reasoning, natural language inference, sentiment analysis, and other domains." (Section 4.1 Tasks)
- "We convert all datasets to binary classification tasks and approximately balance the classes." (Section 4.1 Tasks)
- "For multiple-choice datasets, suppose each datapoint has a question Q and multiple candidate answers  $A_1, \ldots, A_k$ ." (Section A.1 NLP Tasks)
- Inference: `In Dynamics = Capped`, `Attention Dynamic = Static`, and `State Dynamic = Constructed` are inferred from the described language-model interface and classifier adaptation ("we use pretrained base models from the GPT-4 family"; "we replace the unembedding layer of the model with a linear classification head with two outputs"). (Section A; Section A.1)

### Task: Chess move prediction (first-move generation)
- "Each puzzle consists of a chess position, and a sequence of optimal moves to play to solve the puzzle." (Section 4.1 Tasks)
- "For our evaluation, we predict the first move played, which is the best move in the given chess position." (Section 4.1 Tasks)
- "Note that unlike the other binary classification tasks we study in this paper, this is a generative task." (Section 4.1 Tasks)
- Inference: `In Dimension = 2D (x, y); 1D (t)`, `In Dynamics = Capped`, `Attention Dynamic = Static`, `State Dynamic = Constructed`, `Out Dimension = 1D (t)`, and `Out Dynamics = Capped` are inferred because the conceptual object is a chess board position while the implementation is serialized as move text ("We follow the pretraining format, and convert each puzzle to a list of moves leading up to the puzzle position, as illustrated in Figure 14.") and decoded as text move completions. (Section A.2 Chess Puzzles)

### Task: Pairwise preference classification (reward modeling)
- "A critical step of RLHF is to train a reward model (RM) to predict human preferences between model responses." (Section 4.1 Tasks)
- "Then, a reward model is trained to predict the results of pairwise comparisons between completions." (Section 4.1 Tasks)
- "the datapoints can be viewed as  $(d, c_1, c_2, y)$ , where the label y is 1 if the labeler preferred completion  $c_2$  and 0 otherwise." (Section A.3 ChatGPT Reward Modeling)
- Inference: `In Dynamics = Capped`, `Attention Dynamic = Static`, and `State Dynamic = Constructed` are inferred from the described pretrained language-model setup with a reward head and bounded prompt/completion processing ("we replace the unembedding layer of the model with a linear head with a single output"). (Section A.3 ChatGPT Reward Modeling)

### Task: Generative language modeling on RM comparison data
- "Comparisons are comprised of a prefix (a single request or conversation between the user and assistant) and at least two candidate completions." (Section 5.2.2 Generative supervision improves RM weak-to-strong generalization)
- "We finetune the base models with a language modeling loss on all prefix-completion pairs, ignoring the human preferences between those completions." (Section 5.2.2 Generative supervision improves RM weak-to-strong generalization)
- "these pairs include completions ranked worst by human raters" (Section 5.2.2 Generative supervision improves RM weak-to-strong generalization)
- Inference: `In Dynamics = Capped`, `Attention Dynamic = Static`, `State Dynamic = Constructed`, and `Out Dynamics = Capped` are inferred from autoregressive language-model finetuning over text sequences with model-context limits. (Section 5.2.2)

### Task: Image classification (ImageNet)
- "We additionally demonstrate weak-to-strong generalization in a simple image classification experiment." (Section D.1 Self-Supervised Vision Models)
- "We use a pretrained AlexNet model (Krizhevsky et al., 2012) as a weak supervisor, and use it to generate weak labels on the ImageNet (Russakovsky et al., 2015) validation set." (Section D.1 Self-Supervised Vision Models)
- "As a strong student, we use linear probing on frozen representations extracted by DINO models" (Section D.1 Self-Supervised Vision Models)
- Inference: `Attention Dynamic = Static` and `State Dynamic = Constructed` are inferred from fixed linear probing over frozen learned representations; `In Dynamics` is not explicitly specified for image sizing/preprocessing in the OCR text. (Section D.1 Self-Supervised Vision Models)
