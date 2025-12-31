# Lost in the Middle: How Language Models Use Long Contexts (2023)
Source: 94014e-2023.pdf

## Core reasons
- The paper is squarely focused on understanding and measuring how long-context language models access their inputs, introducing new evaluation protocols that stress positional robustness rather than proposing a new architectural component.
- Its experimental analysis identifies practical trade-offs in giving models more retrieved context, showing that reader performance saturates long before retriever recall and that information in the middle of the context is hardest to use.

## Evidence extracts
- "Our analysis provides a better understanding of how language models use their input context and provides new evaluation protocols for future long-context language models." (Abstract)
- "Figure 11: Retriever recall and model performance as a function of the number of retrieved documents. Model performance saturates long before retriever recall, indicating that the models have difficulty making use of the extra retrieved documents." (Section 5)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
