# The Capacity for Moral Self-Correction in Large Language Models (2023)
Source: The Capacity for Moral Self-Correction in Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Multiple-choice question answering for stereotype bias (BBQ) | Tokens (QA prompt with answer options and optional instructions) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Answer choice label (three-option multiple choice) | 0D (inferred) | Fixed |
| Pronoun prediction for occupation coreference (Winogender) | Tokens (sentence template with a pronoun blank and occupation context) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Pronoun choice/probability over {his, her, their} | 0D (inferred) | Fixed |
| Binary admission recommendation decision (law school benchmark) | Tokens (applicant description with race, sex, LSAT score, GPA, and instructions) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Admission decision label ("yes" or "no") | 0D (inferred) | Fixed |

## Summary
The paper evaluates three benchmark tasks on RLHF dialogue language models: multiple-choice stereotype QA (BBQ), pronoun prediction in Winogender, and binary law-school admission recommendation. All tasks are text-input tasks, so input domain is 1D (t) (inferred), and the prompt interface is best treated as Capped (inferred) in this setup. Outputs are categorical choices over fixed sets (MCQ option, pronoun set, yes/no), giving 0D (inferred) outputs with Fixed output dynamics. Based on the decoder-only dialogue setup, attention is Static (inferred) and state is Direct (inferred).

## Evidence
### Task: Multiple-choice question answering for stereotype bias (BBQ)
- "The benchmark tests for models' propensity to rely on stereotypes (in an American English-speaking context) when answering questions." (Section 3.2.2)
- "Each problem in the dataset is a multiple choice question with three possible answers." (Section 3.2.2)
- Inference: `1D (t)` input, `Capped` input dynamics, `Static` attention, `Direct` state, and `0D` output were inferred from the model/protocol description: "We study decoder-only transformer models fine-tuned with Reinforcement Learning from Human Feedback (RLHF) [13, 57] to function as helpful dialogue models." and "We then sample the Assistant response (for up to 256 tokens) conditioned on everything above." (Sections 3.1 and 3.2.2).

### Task: Pronoun prediction for occupation coreference (Winogender)
- "Overview The Winogender dataset consists of 120 sentence templates designed to test whether coreference systems are more or less likely to recognize a gender pronoun as coreferent with a particular occupation [49]." (Section 3.2.3)
- "The task is to evaluate the probability of a model filling in the blank with either "his", "her", or "their" given the prompt." (Section 3.2.3)
- Inference: `1D (t)` input, `Capped` input dynamics, `Static` attention, `Direct` state, and `0D` output were inferred from the Section 3.1 decoder-only dialogue model description and the Section 3.2.3 statement that the Q+IF and Q+IF+CoT protocol is shared with BBQ.

### Task: Binary admission recommendation decision (law school benchmark)
- "Here, we transform the data into a decision-relevant prompt for a language model. In particular, we consider a scenario in which a law professor consults a dialogue model to help them make a decision about whether or not they should accept a student into their class based on a short description of the applicant, which includes demographic information." (Section 3.2.4)
- "Ultimately, we ask the Assistant to answer with a "yes" or a "no" in response to whether the law school professor should admit a student into their class." (Section 3.2.4)
- Inference: `1D (t)` input, `Capped` input dynamics, `Static` attention, `Direct` state, and `0D` output were inferred from the decoder-only dialogue model setup in Section 3.1 and the prompting protocol in Section 3.2.4.
