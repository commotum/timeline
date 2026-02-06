# BROS: A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents (Not specified in the paper.)
Source: BROS- A Pre-trained Language Model Focusing on Text and Layout for Better Key Information Extraction from Documents.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Entity extraction | OCR text blocks with layout | 2D (x, y) | Capped | Static (inferred) | Not specified in the paper. | sequences of text blocks (key entities) | 1D (t) | Capped |
| Entity linking | OCR text blocks with layout | 2D (x, y) | Capped | Static (inferred) | Not specified in the paper. | relations between key entities (text blocks) | 2D (x, y) (inferred) | Capped |

## Summary
The paper applies BROS to document key information extraction on OCR text blocks with 2D layout, focusing on entity extraction and entity linking. Inputs are text blocks in two-dimensional space; EE outputs sequences of text blocks, while EL outputs relations between entities in the layout. The interface is capped by a maximum token count, uses full-pair attention over text blocks (Static, inferred), and does not specify state dynamics.

## Evidence
### Task: Entity extraction
- "Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space." (Abstract)
- "OCR detects the texts in the image and recognizes the content to generate a set of text blocks." (Introduction)
- "The EE task identifies sequences of text blocks that represent desired target texts." (Key Information Extraction Tasks)
- "N is the maximum number of tokens." (SPADE Decoder)
- Inference: Attention Dynamic marked Static because "BROS considers relative positions for all text block pairs." (Compare the Inference Speed of the Models)

### Task: Entity linking
- "Key information extraction (KIE) from document images requires understanding the contextual and spatial semantics of texts in two-dimensional (2D) space." (Abstract)
- "OCR detects the texts in the image and recognizes the content to generate a set of text blocks." (Introduction)
- "The EL task connects key entities through their hierarchical or semantic relations." (Key Information Extraction Tasks)
- "N is the maximum number of tokens." (SPADE Decoder)
- Inference: Attention Dynamic marked Static because "BROS considers relative positions for all text block pairs." (Compare the Inference Speed of the Models). Out Dimension marked 2D (x, y) because EL links entities in "texts in two-dimensional (2D) space." (Abstract)
