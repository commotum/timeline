# Dense Passage Retrieval for Open-Domain Question Answering (2020)
Source: 6fe14a-2020.pdf

## Core reasons
- The paper’s main contribution is a dense passage retriever (dual-encoder) trained on dense representations to improve open-domain QA retrieval, i.e., a modeling/training method contribution.
- It focuses on encoder training scheme and retrieval effectiveness rather than introducing new benchmarks or positional encoding changes.

## Evidence extracts
- "                         retrieval can be practically implemented us-                versely, the dense, latent semantic encoding is com-
                         ing dense representations alone, where em-                  plementarytosparserepresentationsbydesign. For
                         beddings are learned from a small number                    example, synonyms or paraphrases that consist of
                         of questions and passages by a simple dual-                 completely different tokens may still be mapped to
                         encoder framework.       When evaluated on a                vectors close to each other. Consider the question" (p. 1)
- "                 pairs of questions and passages (or answers), with-                               s     s+1        e
                 out additional pretraining? By leveraging the now       one of the passages pi that can answer the question.
                 standard BERT pretrained model (Devlin et al.,          Notice that to cover a wide variety of domains, the
                 2019) and a dual-encoder architecture (Bromley          corpus size can easily range from millions of docu-
                 et al., 1994), we focus on developing the right         ments (e.g., Wikipedia) to billions (e.g., the Web).
                 training scheme using a relatively small number         Asaresult, any open-domain QA system needs to" (p. 2)

## Classification
Class name: ML Foundations & Principles
Class code: 5

$$
\boxed{5}
$$
