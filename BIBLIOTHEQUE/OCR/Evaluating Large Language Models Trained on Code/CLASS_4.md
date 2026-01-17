# Evaluating Large Language Models Trained on Code (Not specified in the paper)
Source: Evaluating Large Language Models Trained on Code.md

## Core reasons
- The paper centers on an evaluation framework for code generation, including defining the pass@k metric and describing a benchmark dataset (HumanEval).
- It introduces and details a hand-written dataset with unit tests to measure functional correctness, positioning evaluation as the main contribution.

## Evidence extracts
- "In this section, we discuss the details of our evaluation framework. We begin by defining the pass@k metric, and explain its advantages over standard match-based metrics. Next, we describe the dataset of hand-written problems, called \"HumanEval,\" which we created in order to benchmark our models. Finally, we discuss the sandbox environment we used to safely execute model-generated code." (Section 2. Evaluation Framework)
- "We evaluate functional correctness on a set of 164 hand-written programming problems, which we call the HumanEval dataset. Each problem includes a function signature, docstring, body, and several unit tests, with an average of 7.7 tests per problem." (Section 2.2. HumanEval: Hand-Written Evaluation Set)

## Classification
Class name: Data, Benchmarks & Measurement
Class code: 4

$$
\boxed{4}
$$
