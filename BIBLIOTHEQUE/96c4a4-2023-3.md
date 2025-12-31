# Precise Zero-Shot Dense Retrieval without Relevance Labels (2023)
Source: 96c4a4-2023.pdf

## Core reasons
- The paper reframes zero-shot dense retrieval as a two-step computation where a generative instruction-following model crafts hypothetical documents and an unsupervised encoder locates similar corpus passages, avoiding any relevance labels.
- HyDE deliberately changes how retrieval computation happens by searching over document-only embeddings anchored by generated hypotheses, which is a new mechanism that pairs an NLG model with contrastive similarity search.

## Evidence extracts
- "Instead, we propose to pivot through Hypothetical Document Embeddings (HyDE). Given a query, HyDE first zero-shot instructs an instruction-following language model (e.g. InstructGPT) to generate a hypothetical document. The document captures relevance patterns but is unreal and may contain false details. Then, an unsupervised contrastively learned encoder (e.g. Contriever) encodes the document into an embedding vector. This vector identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity." (Section 1)
- "HyDE circumvents the aforementioned learning problem by performing search in document-only embedding space that captures document-document similarity. This can be easily learned using unsupervised contrastive learning (Izacard et al., 2021; Gao et al., 2021; Gao and Callan, 2022). We set document encoder enc directly as a contrastive encoder enccon. This function is also denoted as f for simplicity. This unsupervised contrastive encoder will be shared by all incoming document corpus." (Section 3.2)

## Classification
Class name: Computation & Reasoning Mechanism Proposal
Class code: 3

$$
\boxed{3}
$$
