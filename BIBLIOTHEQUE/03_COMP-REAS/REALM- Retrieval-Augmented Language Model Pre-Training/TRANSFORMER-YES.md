# REALM: Retrieval-Augmented Language Model Pre-Training (Year not specified)
Source: REALM- Retrieval-Augmented Language Model Pre-Training.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The core REALM architecture explicitly uses Transformer blocks in both retriever and encoder components, not just as a baseline mention.
- The method’s main training/inference path relies on attending over retrieved documents with a knowledge-augmented Transformer encoder for MLM and Open-QA.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision used the abstract, all available auxiliary files, and a targeted architecture scan.

## Evidence
- "We implement the embedding functions using BERT-style Transformers (Devlin et al., 2018)." (REALM- Retrieval-Augmented Language Model Pre-Training.md, Section 3.2 Model architecture, line 75)
- "We join x and z into a single sequence that we feed into a Transformer (distinct from the one used in the retriever). This allows us to perform rich crossattention between x and z before predicting y." (REALM- Retrieval-Augmented Language Model Pre-Training.md, Section 3.2 Model architecture, line 85)
- "which allows the model to retrieve and attend over documents from a large corpus such as Wikipedia, used during pre-training, fine-tuning and inference." (REALM- Retrieval-Augmented Language Model Pre-Training.md, Abstract, line 7)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - indicated dynamic attention/retrieval and BERT cues, but explicit Transformer-central architecture confirmation was finalized in Pass 2.
Pass 2 (targeted source scan): performed - model/method lines explicitly confirm BERT-style Transformer components are central to REALM.
