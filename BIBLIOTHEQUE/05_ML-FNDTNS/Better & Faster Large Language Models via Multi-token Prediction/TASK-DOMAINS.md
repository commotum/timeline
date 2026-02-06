# Better & Faster Large Language Models via Multi-token Prediction (Not specified in the paper)
Source: Better & Faster Large Language Models via Multi-token Prediction.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Next-token prediction (language modeling) | history of past tokens | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | next future token | 0D (inferred) | Fixed (inferred) |
| Multi-token prediction (language modeling) | observed context tokens | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | n future tokens | 1D (t) (inferred) | Fixed (inferred) |
| Code generation (coding benchmarks) | coding problems/prompts (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | code solutions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Abstractive text summarization | documents to summarize (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | summaries | 1D (t) (inferred) | Not specified in the paper. |
| Natural language mathematics (GSM8K) answer generation | GSM8K math word problems with few-shot examples (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | final answer | 1D (t) (inferred) | Not specified in the paper. |
| Multiple-choice benchmark evaluation (choice tasks) | multiple-choice benchmark items (inferred) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | selected answer choice (inferred) | 0D (inferred) | Fixed (inferred) |
| Induction (pattern completion) prediction | sentences/stories with prior token pairs (e.g., AB then A) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | continuation token (e.g., B) | 0D (inferred) | Fixed (inferred) |
| Polynomial arithmetic (algorithmic reasoning) | polynomial expressions in F7[X]/(X^5) with operations | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | coefficients of the resulting polynomial | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper centers on sequence language modeling tasks (next-token and multi-token prediction) and evaluates models on code generation benchmarks, abstractive summarization, GSM8K math answering, and synthetic reasoning tasks (induction and polynomial arithmetic). It also reports results on multiple-choice NLP benchmarks. Inputs and outputs are primarily 1D token sequences with some single-token or single-choice outputs (0D), and the only explicit cap on input complexity is the polynomial arithmetic task's bounded operation count. Attention and state dynamics are not explicitly stated, so the table marks them as inferred static/direct for standard autoregressive sequence prediction.

## Evidence
### Task: Next-token prediction (language modeling)
- "Standard language modeling learns about a large text corpus  $x_1, \ldots x_T$  by implementing a next-token prediction task." (Section 2. Method)
- "maximize the probability of  $x_{t+1}$  as the next future token, given the history of past tokens  $x_{t:1} = x_t, \dots, x_1$ ." (Section 2. Method)
- Inference: In Dimension 1D (t), Attention Dynamic Static, State Dynamic Direct, Out Dimension 0D, and Out Dynamics Fixed are inferred because the task predicts a single next token from a token sequence context (Section 2. Method).

### Task: Multi-token prediction (language modeling)
- "(Top) During training, the model predicts 4 future tokens at once, by means of a shared trunk and 4 dedicated output heads." (Figure 1 caption)
- "multi-token prediction instructs the LLM to predict the *n* future tokens from each position in the training corpora, all at once and in parallel (Qi et al., 2020)." (Introduction)
- Inference: In Dimension 1D (t), Attention Dynamic Static, State Dynamic Direct, Out Dimension 1D (t), and Out Dynamics Fixed are inferred because the model predicts a fixed-length sequence of n future tokens from a token sequence context.

### Task: Code generation (coding benchmarks)
- "Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points." (Abstract)
- "Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models." (Abstract)
- "We compare models with 7B parameters trained from scratch on 200B and on 314B bytes of code on the MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021) and APPS (Hendrycks et al., 2021) benchmarks." (Table 1 caption)
- "We evaluate this by finetuning 7B parameter models from Section 3.3 on the CodeContests dataset (Li et al., 2022)." (Section 3.6)
- Inference: Input as coding problems/prompts and Output as code solutions are inferred from the paper's description of coding benchmarks and problem-solving evaluations; 1D (t) dimensions and Static/Direct dynamics are inferred because the models generate text/code sequences from textual prompts.

### Task: Abstractive text summarization
- "we conduct evaluations on summarization and natural language mathematics benchmarks" (Section 3.7)
- "For summarization, we use eight benchmarks where ROUGE metrics (Lin, 2004) with respect to a ground-truth summary allow automatic evaluation of generated texts." (Section 3.7)
- Inference: Input as documents to summarize, 1D (t) dimensions, and Static/Direct dynamics are inferred from the summarization setup and use of generated texts and ground-truth summaries.

### Task: Natural language mathematics (GSM8K) answer generation
- "For natural language mathematics, we evaluate the pretrained models in 8-shot mode on the GSM8K benchmark (Cobbe et al., 2021) and measure accuracy of the final answer produced after a chain-of-thought elicited by the fewshot examples." (Section 3.7)
- Inference: Input as GSM8K math word problems with few-shot examples, 1D (t) dimensions, and Static/Direct dynamics are inferred from the described evaluation protocol and generated final answers.

### Task: Multiple-choice benchmark evaluation (choice tasks)
- "However, we do not believe that multiple-choice and likelihood-based benchmarks are suited to effectively discern *generative capabilities* of language models." (Section 3.7)
- "Figure 5: Multi-token training with 7B models doesn't improve performance on choice tasks." (Figure 5 caption)
- "We evaluate the models from Section 3.7 on standard natural language processing benchmarks: ARC Challenge (Yadav et al., 2019), COPA (Roemmele et al., 2011), Hellaswag (Zellers et al., 2019), Natural Questions (Kwiatkowski et al., 2019), PIQA (Bisk et al., 2019), SIQA (Sap et al., 2019) and TriviaQA (Joshi et al., 2017)." (Appendix G)
- Inference: Input as multiple-choice benchmark items and Output as a selected answer choice are inferred from the paper's characterization of these as "choice tasks"; 1D input dimension, 0D output dimension, Fixed output dynamics, and Static/Direct dynamics are inferred accordingly.

### Task: Induction (pattern completion) prediction
- "Induction describes a simple pattern of reasoning that completes partial patterns by their most recent continuation (Olsson et al., 2022). In other words, if a sentence contains "AB" and later mentions "A", induction is the prediction that the continuation is "B"." (Section 4.1)
- "predicting the second token of each name's occurrence after it has been mentioned at least once can be seen as a pure induction task." (Section 4.1)
- Inference: In Dimension 1D (t), Attention Dynamic Static, State Dynamic Direct, Out Dimension 0D, and Out Dynamics Fixed are inferred because the task predicts a single continuation token from a token sequence context.

### Task: Polynomial arithmetic (algorithmic reasoning)
- "We train and evaluate models on a task on polynomial arithmetic in the ring  $\mathbb{F}_7[X]/(X^5)$  with unary negation, addition, multiplication and composition of polynomials as operations." (Section 4.2)
- "The task is to return the coefficients of the polynomials corresponding to the resulting expressions." (Section 4.2)
- "The number m of operations contained in the expressions is selected uniformly from the range from 1 to 5 at training time," (Section 4.2)
- Inference: In Dimension 1D (t), In Dynamics Capped, Attention Dynamic Static, State Dynamic Direct, Out Dimension 1D (t), and Out Dynamics Fixed are inferred because expressions are token sequences with a bounded number of operations and the ring F7[X]/(X^5) implies a fixed-length coefficient output.
