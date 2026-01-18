## 1. Basic Metadata

- Title: "PaLM-E: An Embodied Multimodal Language Model" (Header)
- Authors: "Danny Driess <sup>12</sup> Fei Xia <sup>1</sup> Mehdi S. M. Sajjadi <sup>3</sup> Corey Lynch <sup>1</sup> Aakanksha Chowdhery <sup>3</sup> Brian Ichter <sup>1</sup> Ayzaan Wahid <sup>1</sup> Jonathan Tompson <sup>1</sup> Quan Vuong <sup>1</sup> Tianhe Yu <sup>1</sup> Wenlong Huang <sup>1</sup> Yevgen Chebotar <sup>1</sup> Pierre Sermanet <sup>1</sup> Daniel Duckworth <sup>3</sup> Sergey Levine <sup>1</sup> Vincent Vanhoucke <sup>1</sup> Karol Hausman <sup>1</sup> Marc Toussaint <sup>2</sup> Klaus Greff <sup>3</sup> Andy Zeng <sup>1</sup> Igor Mordatch <sup>3</sup> Pete Florence <sup>1</sup>" (Header)
- Year: Year not specified.
- Venue (conference/journal/arXiv): Venue not specified.

## 2. One-Sentence Contribution Summary

The paper proposes embodied language models that "directly incorporate real-world continuous sensor modalities into language models" to ground inference for embodied tasks (Abstract).

## 3. Tasks Evaluated

- Task name: TAMP VQA q1 (object color); Task type: Classification; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "the VQA task  $q_1$  is about the color of an object." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1); "Task and Motion Planning (TAMP) domain where a robot has to manipulate (grasp and stack) objects" (6.1).
- Task name: TAMP VQA q2 (object-table relation); Task type: Classification; Reasoning / relational; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "- $q_2$ : object-table relation. Example prompt: Given <img>. Q: Is the red object left, right, or center of the table? Target: A: The red object is in the center of the table." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1).
- Task name: TAMP VQA q3 (object-object relations); Task type: Classification; Reasoning / relational; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "- $q_3$ : object-object relations. Example prompt: Given <img>. Q: Is the yellow object below the blue object?. Target: A: No, the yellow object is not below the blue object." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1).
- Task name: TAMP VQA q4 (plan feasibility); Task type: Classification; Reasoning / relational; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "- $q_4$ : plan feasibility. Example prompt: Given <img>. Q: Is it possible to first grasp the blue object, then place it on the yellow object, and then grasp the yellow object? Target: A: No, this is not possible." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1).
- Task name: TAMP planning p1 (grasping); Task type: Generation; Reasoning / relational; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "- $\bullet$  p<sub>1</sub>: grasping. Example prompt: Given <img>. Q: How to grasp the green object?. Target: A: First grasp the orange object and place it on the table, then grasp the green object." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1).
- Task name: TAMP planning p2 (stacking); Task type: Generation; Reasoning / relational; Dataset(s) used: TAMP environment dataset with 3-5 cube-shaped objects; Domain: robotic manipulation (TAMP); Quotes: "- $p_2$ : stacking. Example prompt: Given <img>. Q: How to stack the white object on top of the red object?. Target: A: First grasp the green object and place it on the table, then grasp the white object and place it on the red object." (B.1); "The training scenes for the TAMP environment contain 3-5 cube-shaped objects of different sizes, colors and sampled initial poses." (B.1).
- Task name: Language-Table Task 1 (push closest block to same-color block); Task type: Generation; Reasoning / relational; Dataset(s) used: Language-Table dataset (Lynch et al., 2022); Domain: table-top pushing; Quotes: "| <b>Task 1.</b> Q: There is a block that is closest to |  |  |  |  |  |  |  |" (Table 3); "| {i.e., top right corner}. Push that block to          |  |  |  |  |  |  |  |" (Table 3); "| the other block of the same color.                    |  |  |  |  |  |  |  |" (Table 3); "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (6.1).
- Task name: Language-Table Task 2 (sort blocks by colors into corners); Task type: Generation; Reasoning / relational; Dataset(s) used: Language-Table dataset (Lynch et al., 2022); Domain: table-top pushing; Quotes: "| Task 2. Q: How to sort the blocks by colors | S |" (Table 3); "| into corners?                               |   |" (Table 3); "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (6.1).
- Task name: Language-Table Task 3 (push blocks on left/right together without mixing sides); Task type: Generation; Reasoning / relational; Dataset(s) used: Language-Table dataset (Lynch et al., 2022); Domain: table-top pushing; Quotes: "Task 3. Q: How to push all the blocks that are on the {left/right} side together, without bringing over any of the blocks that are on the {right/left} side?" (Table 3); "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (6.1).
- Task name: Mobile manipulation affordance prediction; Task type: Classification; Dataset(s) used: mobile manipulation runs from Ahn et al. (2022) (2912 sequences); Domain: kitchen mobile manipulation; Quotes: "Affordance prediction. We investigate PaLM-E's performance at affordance prediction, i.e. whether a skill of the low-level policy can be executed in the current environment. This can be formulated as the VQA problem Given <img>. Q: Is it possible to <skill> here?." (6.4); "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (6.4); "mobile manipulation domain similar to SayCan (Ahn et al., 2022), where a robot has to solve a variety of tasks in a kitchen environment, including finding objects in drawers, picking them, and bringing them to a human." (6.1).
- Task name: Mobile manipulation failure detection; Task type: Classification; Dataset(s) used: mobile manipulation runs from Ahn et al. (2022) (2912 sequences); Domain: kitchen mobile manipulation; Quotes: "Failure detection. For a robot to do closed-loop planning, it is also important to detect failures, as is shown in (Huang et al., 2022c). The multi-modal prompt is <code>Given <img>.</code> Q: <code>Was <skill> successful?</code>." (6.4); "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (6.4).
- Task name: Mobile manipulation long-horizon planning; Task type: Generation; Reasoning / relational; Dataset(s) used: mobile manipulation runs from Ahn et al. (2022) (2912 sequences); Domain: kitchen mobile manipulation; Quotes: "Real robot results: Long-horizon planning. Finally, we use PaLM-E to perform *embodied planning* end-to-end for mobile manipulation tasks. The prompt structure for this task is Human: <instruction> Robot: <step history>. I see <img>. PaLM-E is trained to generate the next step of the plan, conditioned on the history of taken steps and the current image observation of the scene." (6.4); "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (6.4).
- Task name: OK-VQA; Task type: Generation; Reasoning / relational; Dataset(s) used: OK-VQA; Domain: general vision-language tasks; Quotes: "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (6.5).
- Task name: VQA v2; Task type: Generation; Reasoning / relational; Dataset(s) used: VQA v2; Domain: general vision-language tasks; Quotes: "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (6.5).
- Task name: COCO captioning; Task type: Generation; Dataset(s) used: COCO; Domain: general vision-language tasks; Quotes: "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (6.5).
- Task name: TriviaQA (wiki) (EM); Task type: Generation; Dataset(s) used: TriviaQA (wiki); Domain: language benchmarks (NLU/NLG); Quotes: "TriviaQA (wiki) (EM)" (Table 8).
- Task name: Natural Questions (EM); Task type: Generation; Dataset(s) used: Natural Questions; Domain: language benchmarks (NLU/NLG); Quotes: "Natural Questions (EM)" (Table 8).
- Task name: WebQuestions (EM); Task type: Generation; Dataset(s) used: WebQuestions; Domain: language benchmarks (NLU/NLG); Quotes: "WebQuestions (EM)" (Table 8).
- Task name: Lambada; Task type: Generation; Dataset(s) used: Lambada; Domain: language benchmarks (NLU/NLG); Quotes: "Lambada" (Table 8).
- Task name: HellaSwag; Task type: Classification; Dataset(s) used: HellaSwag; Domain: language benchmarks (NLU/NLG); Quotes: "HellaSwag" (Table 8).
- Task name: StoryCloze; Task type: Classification; Dataset(s) used: StoryCloze; Domain: language benchmarks (NLU/NLG); Quotes: "StoryCloze" (Table 8).
- Task name: Winograd; Task type: Classification; Dataset(s) used: Winograd; Domain: language benchmarks (NLU/NLG); Quotes: "Winograd" (Table 8).
- Task name: Winogrande; Task type: Classification; Dataset(s) used: Winogrande; Domain: language benchmarks (NLU/NLG); Quotes: "Winogrande" (Table 8).
- Task name: RACE-M; Task type: Classification; Dataset(s) used: RACE-M; Domain: language benchmarks (NLU/NLG); Quotes: "RACE-M" (Table 8).
- Task name: RACE-H; Task type: Classification; Dataset(s) used: RACE-H; Domain: language benchmarks (NLU/NLG); Quotes: "RACE-H" (Table 8).
- Task name: PIQA; Task type: Classification; Dataset(s) used: PIQA; Domain: language benchmarks (NLU/NLG); Quotes: "PIQA" (Table 8).
- Task name: ARC-e; Task type: Classification; Dataset(s) used: ARC-e; Domain: language benchmarks (NLU/NLG); Quotes: "ARC-e" (Table 8).
- Task name: ARC-c; Task type: Classification; Dataset(s) used: ARC-c; Domain: language benchmarks (NLU/NLG); Quotes: "ARC-c" (Table 8).
- Task name: OpenBookQA; Task type: Classification; Dataset(s) used: OpenBookQA; Domain: language benchmarks (NLU/NLG); Quotes: "OpenBookQA" (Table 8).
- Task name: BoolQ; Task type: Classification; Dataset(s) used: BoolQ; Domain: language benchmarks (NLU/NLG); Quotes: "BoolQ" (Table 8).
- Task name: Copa; Task type: Classification; Dataset(s) used: Copa; Domain: language benchmarks (NLU/NLG); Quotes: "Copa" (Table 8).
- Task name: RTE; Task type: Classification; Dataset(s) used: RTE; Domain: language benchmarks (NLU/NLG); Quotes: "RTE" (Table 8).
- Task name: Wic; Task type: Classification; Dataset(s) used: Wic; Domain: language benchmarks (NLU/NLG); Quotes: "Wic" (Table 8).
- Task name: WSC; Task type: Classification; Dataset(s) used: WSC; Domain: language benchmarks (NLU/NLG); Quotes: "WSC" (Table 8).
- Task name: ReCoRD; Task type: Classification; Dataset(s) used: ReCoRD; Domain: language benchmarks (NLU/NLG); Quotes: "ReCoRD" (Table 8).
- Task name: CB; Task type: Classification; Dataset(s) used: CB; Domain: language benchmarks (NLU/NLG); Quotes: "CB" (Table 8).

## 4. Domain and Modality Scope

- Evaluation scope: Multiple modalities and multiple domains; Quotes: "Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings." (Abstract); "we evaluate PaLM-E also on general vision-language tasks such as visual-question-answering (VQA), image captioning, and established language modeling tasks." (6).
- Single domain vs multiple domains: Multiple domains within and across modalities (robotics environments, vision-language benchmarks, and language benchmarks) are evaluated, not a single domain; Quotes: "Our experiments consider diverse robotic (mobile) manipulation tasks across three different robot embodiments" (6); "we evaluate PaLM-E also on general vision-language tasks such as visual-question-answering (VQA), image captioning, and established language modeling tasks." (6).
- Domain generalization or cross-domain transfer: Claimed; Quote: "exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains." (Abstract).

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| TAMP VQA q1 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| TAMP VQA q2 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| TAMP VQA q3 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| TAMP VQA q4 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| TAMP planning p1 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| TAMP planning p2 | Varies (generalist mixture and TAMP-only training reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment, i.e. no other data is part of the mixture." (6.2). |
| Language-Table Task 1 | Yes (generalist mixture reported) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (B.2). |
| Language-Table Task 2 | Yes (generalist mixture reported) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (B.2). |
| Language-Table Task 3 | Yes (generalist mixture reported) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (B.2). |
| Mobile manipulation affordance prediction | Yes (generalist mixture reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6). |
| Mobile manipulation failure detection | Yes (generalist mixture reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6). |
| Mobile manipulation long-horizon planning | Yes (generalist mixture reported) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (6). |
| OK-VQA | Yes (generalist checkpoint) and task-specific finetuned models reported | Yes (task-specific finetuned models reported) | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use different-finetuned models for the different tasks." (Table 5). |
| VQA v2 | Yes (generalist checkpoint) and task-specific finetuned models reported | Yes (task-specific finetuned models reported) | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use different-finetuned models for the different tasks." (Table 5). |
| COCO captioning | Yes (generalist checkpoint) and task-specific finetuned models reported | Yes (task-specific finetuned models reported) | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use different-finetuned models for the different tasks." (Table 5). |
| TriviaQA (wiki) (EM) | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Natural Questions (EM) | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| WebQuestions (EM) | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Lambada | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| HellaSwag | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| StoryCloze | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Winograd | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Winogrande | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| RACE-M | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| RACE-H | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| PIQA | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| ARC-e | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| ARC-c | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| OpenBookQA | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| BoolQ | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Copa | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| RTE | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| Wic | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| WSC | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| ReCoRD | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |
| CB | Not specified | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (6.6). |

## 6. Input and Representation Constraints

- Fixed or variable input resolution: Not specified.
- Fixed patch size: Not specified.
- Fixed number of tokens per observation is implied by the encoders: "we train an encoder  $\phi: \mathcal{O} \to \mathcal{X}^q$  that maps a (continuous) observation space  $\mathcal{O}$  ... into a *sequence* of q-many vectors in  $\mathcal{X}$." (Section 3); "ViT  $\phi_{\rm ViT}$  ... mapping an image I into a number of token embeddings  $\tilde{x}_{1:m}$" (Section 4); "Note that individual objects are always tokenized into *multiple* embeddings each, i.e.  $\psi: \mathbb{R}^{\bar{k}} \to \mathbb{R}^{m \times k}$  for OSRT maps into m-many embeddings." (Section 4).
- Fixed dimensionality for multimodal tokens: "encoding the continuous observations into a sequence of vectors with the same dimension as the embedding space of the language tokens." (Section 3); "token embedding space  $\mathcal{X} \subset \mathbb{R}^k$" (Section 3).
- 2D vs 3D representations are explicitly named: "Vision Transformers (ViTs) ... for 2D image features" (Section 4); "OSRT ... learns 3D-centric neural scene representations" (Section 4).
- Padding or resizing requirements: Not specified.
- Token placement constraints: "the observation embeddings are not inserted at fixed positions, but instead placed dynamically within the surrounding text." (Section 3).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified.
- Fixed or variable sequence length: Not specified; inputs are described as "multi-modal sentences" with text and "(multiple) continuous observations" interleaved (Section 3).
- Attention type: Not specified; the model uses "self-attention layers of a Transformer-based LLM" (Introduction).
- Mechanisms for computational cost (windowing/pooling/pruning): Not specified.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism used: Not specified; the model "reuses its existing positional encodings." (Section 3).
- Where applied: Not specified.
- Fixed across experiments vs modified per task: Not specified; only reuse of existing encodings is stated (Section 3).

## 9. Positional Encoding as a Variable

- Positional encoding treated as a fixed architectural assumption: Yes; "reuses its existing positional encodings." (Section 3).
- Multiple positional encodings compared: Not specified.
- PE choice described as not critical or secondary: Not specified.

## 10. Evidence of Constraint Masking

- Model scale evidence: "Our largest model, PaLM-E-562B with 562B parameters" (Abstract); "We scale PaLM-E up to 562B parameters, integrating the 540B PaLM ... and the 22B Vision Transformer (ViT)" (Introduction).
- Dataset size/mixture evidence: "input representations are trained on a dataset containing 96,000 training scenes of solely the TAMP environment" (6.2); "only 8.9% of the full mixture is embodied data" (Section 5); "only 10 demos per task" (6.3); "which contains 2912 sequences" (6.4); "between 10 and 80 for Language Table or 320 for TAMP" (7).
- Performance gains attributed to scaling model size: "Scaling the 12B model to the 84B model leads to improvements on 2 of 3 tasks." (6.3).
- Performance gains attributed to scaling data/co-training: "co-training on the \"full mixture\" achieves more than double the performance." (7); "exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains." (Abstract).
- Architectural representation as compensating structure: "novel architectural idea of ingesting neural scene representations (i.e., OSRT) into the model is particularly effective, even without large-scale data." (Conclusion).

## 11. Architectural Workarounds

- Multimodal token injection into the LLM embedding space: "inject continuous, embodied observations such as images, state estimates, or other sensor modalities into the language embedding space of a pre-trained language model" (Section 3).
- Structured object-centric encoders: "structured encoders that aim to separate visual inputs into distinct objects before injecting them into the LLM." (Section 4).
- 3D-aware scene representation slots (OSRT): "OSRT ... learns 3D-centric neural scene representations" and uses "object slots" (Section 4).
- Entity referral tokens for object grounding: "we label the multi-modal tokens corresponding to an object in the input prompt as follows: Object 1 is  $obj_1$ ... Object j is  $obj_2$ ." (Section 4).
- TokenLearner variant for ViT: "We further investigate the ViT token learner architecture (ViT + TL)" (Section 4).
- Frozen-LLM training path: "we investigate whether it is possible to *freeze* the LLM and to just train the input encoders" (Section 5).
- Control-loop decomposition with low-level skills: "policies that can perform low-level skills from some (small) vocabulary" and PaLM-E is "integrated into a control-loop" (Section 3).

## 12. Explicit Limitations and Non-Claims

- "Although not the focus of our work, we evaluate PaLM-E also on general vision-language tasks such as visual-question-answering (VQA), image captioning, and established language modeling tasks." (6).
- "Although not the focus of our experimentation, we also find (Fig. 2) that PaLM-E-562B exhibits a wide array of capabilities" (Introduction).
- "Although it is not the focus of our work, we report in Tab. 5 results on general vision-language tasks" (6.5).
- "A promising opportunity for future work is to combine this with a method benefitting from large-scale visual data." (7).
- "although this approach occasionally struggled for robotics tasks (Tab. 2)." (7).
- "Although these policies are language conditioned, they are not capable of solving long-horizon tasks or taking in complex instructions." (Section 3).

### 13. Constraint Profile (Synthesis)

**Constraint Profile:**
- Domain scope: Multi-modal and multi-domain evaluations (robotics, vision-language, language) but only on defined datasets and environments.
- Task structure: Explicit VQA and planning prompts for TAMP/Language-Table/mobile tasks plus fixed benchmark suites.
- Representation rigidity: Observations are encoded into fixed-length token sequences (q/m) with fixed embedding dimension and object-centric slots.
- Model sharing vs specialization: Generalist shared checkpoint across tasks with optional domain/task finetuning.
- Role of positional encoding: Reuses existing LLM positional encodings without experimentation.

### 14. Final Classification

Multi-task, multi-domain (constrained). The paper evaluates across robotics domains and standard vision-language and language benchmarks ("Our experiments consider diverse robotic (mobile) manipulation tasks across three different robot embodiments" (6); "we evaluate PaLM-E also on general vision-language tasks such as visual-question-answering (VQA), image captioning, and established language modeling tasks." (6)), indicating multi-domain, multi-task scope. The evaluations are tied to specific datasets/environments and benchmark suites (TAMP, Language-Table, OK-VQA, VQA v2, COCO, and the listed NLU/NLG tasks), so the multi-domain coverage is constrained rather than unrestrained.
