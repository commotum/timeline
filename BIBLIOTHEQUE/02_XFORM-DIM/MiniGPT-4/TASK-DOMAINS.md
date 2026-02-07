# MINIGPT-4: ENHANCING VISION-LANGUAGE UNDERSTANDING WITH ADVANCED LARGE LANGUAGE MODELS (Year not specified in the paper.)
Source: MiniGPT-4.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image description / captioning (detailed) | images; prompt text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | detailed image descriptions / captions (text) | 1D (t) (inferred) | Not specified in the paper. |
| Meme interpretation (humor explanation) | meme images; question text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | humor explanations (text) | 1D (t) (inferred) | Not specified in the paper. |
| Recipe generation from food photos | food photos; question text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | cooking recipes (text) | 1D (t) (inferred) | Not specified in the paper. |
| Advertisement creation from product images | product images; prompt text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | advertisements (text) | 1D (t) (inferred) | Not specified in the paper. |
| Poem composition from images | images; prompt text | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | poems (text) | 1D (t) (inferred) | Not specified in the paper. |
| Website creation from hand-written drafts | hand-written draft images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | website content / code (text) | 1D (t) (inferred) | Not specified in the paper. |
| Factual retrieval from images (people/movies/art) | images (e.g., movie photographs) | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | factual information (text) | 1D (t) (inferred) | Not specified in the paper. |
| Plant disease diagnosis and treatment suggestion | plant images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | diagnosis and treatment plans (text) | 1D (t) (inferred) | Not specified in the paper. |
| Visual question answering (VQA) | images; questions (VQA) | 2D (x, y) (inferred); 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | answers (text) | 1D (t) (inferred) | Not specified in the paper. |
| Story writing from images | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | stories (text) | 1D (t) (inferred) | Not specified in the paper. |
| Explanation of unusual visual phenomena | images | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | explanations (text) | 1D (t) (inferred) | Not specified in the paper. |
| Problem identification and solution generation from photos | photos | 2D (x, y) (inferred) | Not specified in the paper. | Static (inferred) | Direct (inferred) | problems and solutions (text) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
MiniGPT-4 is described as a vision-to-text system that spans detailed image description/captioning, creative writing (poems, stories), instructional generation (recipes, advertisements, websites), and reasoning-style tasks (meme humor, unusual phenomena explanations, problem/solution, factual retrieval), plus VQA. Inputs are primarily images, sometimes paired with explicit text prompts, and outputs are text; thus the inferred dimensionality is 2D (x, y) for images and 1D (t) for text. The paper does not specify fixed/capped/open interface constraints, and attention/state behavior is only implied by the fixed prompt-based generation interface.

## Evidence
### Task: Image description / captioning (detailed)
- "detailed image description generation" (Abstract)
- "generating detailed image descriptions" (Section 4 Experiments)
- "task of image captioning" (Section 4 Experiments)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Meme interpretation (humor explanation)
- "identifying amusing aspects within memes" (Section 4 Experiments)
- "interpret the humorous aspects of a given meme." (Section 4.1)
- "meme interpretation with the question \"Explain why this meme is funny.\"" (Section 4.2)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Recipe generation from food photos
- "providing food recipes from photos" (Section 4 Experiments)
- "generating a food recipe from a food image" (Section 4.1)
- "recipe generation with the question \"How should I make something like this?\"" (Section 4.2)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Advertisement creation from product images
- "creating advertising promotions based on a given image" (Section 4.1)
- "write advertisements for products in images" (Section 1 Introduction)
- "advertisement creation with the prompt \"Help me draft a professional advertisement for this.\"" (Section 4.2)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Poem composition from images
- "writing poems for images" (Section 4 Experiments)
- "writing poems inspired by an image" (Section 4.1)
- "poem composition with \"Can you craft a beautiful poem about this image?\"" (Section 4.2)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Website creation from hand-written drafts
- "website creation from hand-drawn drafts" (Abstract)
- "creating a website from a hand-written draft" (Section 4.1)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Factual retrieval from images (people/movies/art)
- "retrieve rich facts about people, movies, or art directly from images" (Section 1 Introduction)
- "retrieving factual information from a movie photograph" (Section 4.1)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Plant disease diagnosis and treatment suggestion
- "diagnosing plant diseases and suggesting treatment plans" (Section 4.1)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Visual question answering (VQA)
- "quantitative analysis of the VQA datasets A-OKVQA (multi-choice) and GQA" (Appendix A.2)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Story writing from images
- "writing stories and poems inspired by given images" (Abstract)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Explanation of unusual visual phenomena
- "explain unusual visual phenomena" (Section 1 Introduction)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)

### Task: Problem identification and solution generation from photos
- "identify problems shown in photos and provide corresponding solutions" (Section 1 Introduction)
- Inference: In/Out Dimensions and Attention/State inferred from image/text modalities and the fixed prompt template "###Human: <Img><ImageFeature></Img><Instruction>###Assistant:" used to generate text. (Section 3.3)
