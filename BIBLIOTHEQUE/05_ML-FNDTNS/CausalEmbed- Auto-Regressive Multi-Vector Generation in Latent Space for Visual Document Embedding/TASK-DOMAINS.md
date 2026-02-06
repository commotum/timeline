# CAUSALEMBED: Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding (2026)
Source: CausalEmbed- Auto-Regressive Multi-Vector Generation in Latent Space for Visual Document Embedding.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visual document retrieval | visual document pages (images); text queries (tokens) | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | relevance score | 0D (inferred) | Fixed (inferred) |
| multi-vector embedding generation | visual document pages (images); text queries (tokens) | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | multi-vector embeddings (latent vector sequence) | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper targets visual document retrieval and the generation of multi-vector embeddings for visual document pages and text queries. Inputs are visual pages (2D) and tokenized text queries (1D), while outputs include scalar similarity scores and 1D embedding sequences with a token-budgeted length. Attention and state are inferred as static and constructed due to the auto-regressive latent generation process; input dynamics are not specified.

## Evidence
### Task: visual document retrieval
- "Our method enables efficient VDR tasks using only dozens of visual tokens" (Abstract)
- "let  $I \in \mathcal{I}$  denote a visually rendered document page, and let  $T \in \mathcal{T}$  denote a tokenized text query." (Section 3, Preliminary)
- "The similarity function  $\mathcal{S}$  specifies the scoring mechanism,  $\mathcal{S}: \mathcal{T} \times \mathcal{I} \to \mathbb{R}$ ." (Section 3, Preliminary)
- "the model  $\Psi$  conditions on the original input and the history of previously generated latent states to produce the next embedding" (Section 4, CAUSALEMBED)
- Inference: Interpreted the page as 2D (x, y) and the tokenized query as 1D (t); treated the score as a 0D Fixed output from $\mathcal{S}: \mathcal{T} \times \mathcal{I} \to \mathbb{R}$; attention Static and state Constructed from auto-regressive latent generation.

### Task: multi-vector embedding generation
- "maps a visual document page I and a textual query T into a shared latent space via a sequential generation process." (Section 3, Preliminary)
- "We aim to generate a sequence of latent vectors" (Section 4, CAUSALEMBED)
- "where  $L\in\{N_d,N_q\}$  denotes the target embedding budget." (Section 4, CAUSALEMBED)
- "the model  $\Psi$  conditions on the original input and the history of previously generated latent states to produce the next embedding" (Section 4, CAUSALEMBED)
- Inference: Interpreted visual pages as 2D (x, y) and tokenized queries as 1D (t); treated the latent vector sequence as 1D (t) with Capped dynamics from the "target embedding budget"; attention Static and state Constructed due to auto-regressive latent generation.
