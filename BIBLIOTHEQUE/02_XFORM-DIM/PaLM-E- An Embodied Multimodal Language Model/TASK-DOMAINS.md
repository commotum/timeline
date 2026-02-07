# PaLM-E: An Embodied Multimodal Language Model (Not specified in the paper.)
Source: PaLM-E- An Embodied Multimodal Language Model.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TAMP VQA q1 (object color) | image; question text (inferred) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| TAMP VQA q2 (object-table relation) | image; question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Not specified in the paper. |
| TAMP VQA q3 (object-object relation) | image; question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Not specified in the paper. |
| TAMP VQA q4 (plan feasibility) | image; question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text | 1D (t) (inferred) | Not specified in the paper. |
| TAMP planning p1 (grasping) | image; question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text plan steps | 1D (t) (inferred) | Not specified in the paper. |
| TAMP planning p2 (stacking) | image; question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text plan steps | 1D (t) (inferred) | Not specified in the paper. |
| Language-Table Task 1 (push closest block to same-color block) | image; high-level task text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text instruction | 1D (t) (inferred) | Not specified in the paper. |
| Language-Table Task 2 (sort blocks by colors into corners) | image; high-level task text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text instruction | 1D (t) (inferred) | Not specified in the paper. |
| Language-Table Task 3 (push blocks on one side together) | image; high-level task text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | text instruction | 1D (t) (inferred) | Not specified in the paper. |
| Mobile manipulation affordance prediction | image; skill question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Mobile manipulation failure detection | image; skill question text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Mobile manipulation long-horizon planning | image; instruction text; step history text | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | next-step text | 1D (t) (inferred) | Not specified in the paper. |
| OK-VQA | image; question text (inferred) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| VQA v2 | image; question text (inferred) | 2D (x, y); 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | answer text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| COCO captioning | image (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | caption text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| TriviaQA (wiki) (EM) (NLG) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Natural Questions (EM) (NLG) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| WebQuestions (EM) (NLG) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Lambada (NLG) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | generated text (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| HellaSwag (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| StoryCloze (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| Winograd (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| Winogrande (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| RACE-M (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| RACE-H (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| PIQA (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| ARC-e (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| ARC-c (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| OpenBookQA (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| BoolQ (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| Copa (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| RTE (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| Wic (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| WSC (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| ReCoRD (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |
| CB (NLU) | text (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label answer (inferred) | 0D (inferred) | Not specified in the paper. |

## Summary
PaLM-E is evaluated on embodied robotics tasks (TAMP, Language-Table, mobile manipulation) that include VQA-style perception and text-based planning/instruction generation, alongside general vision-language benchmarks (OK-VQA, VQA v2, COCO captioning) and a large suite of language benchmarks (NLU/NLG). The paper describes multimodal inputs combining images and text for embodied and vision-language tasks, and text-only inputs for language benchmarks, with outputs expressed as text answers/instructions or label-style decisions. From the task prompts and descriptions, the supported dimensions span 2D image inputs and 1D text inputs/outputs (with 0D label outputs for NLU), while input/output dynamics, attention dynamics, and state dynamics are not specified.

## Evidence
### Task: TAMP VQA q1 (object color)
- "the VQA task  $q_1$  is about the color of an object." (Section B.1)
- "Example prompt: Given <img>. Q: Is the red object left, right, or center of the table? Target: A: The red object is in the center of the table." (Section B.1)
- Inference: Treated q1 as a VQA-style image+question input with a text answer output, and assigned 2D (image) and 1D (text) dimensions based on the VQA prompt format shown in Section B.1.

### Task: TAMP VQA q2 (object-table relation)
- "- $q_2$ : object-table relation. Example prompt: Given <img>. Q: Is the red object left, right, or center of the table? Target: A: The red object is in the center of the table." (Section B.1)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and textual answer.

### Task: TAMP VQA q3 (object-object relation)
- "- $q_3$ : object-object relations. Example prompt: Given <img>. Q: Is the yellow object below the blue object?. Target: A: No, the yellow object is not below the blue object." (Section B.1)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and textual answer.

### Task: TAMP VQA q4 (plan feasibility)
- "- $q_4$ : plan feasibility. Example prompt: Given <img>. Q: Is it possible to first grasp the blue object, then place it on the yellow object, and then grasp the yellow object? Target: A: No, this is not possible." (Section B.1)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and textual answer.

### Task: TAMP planning p1 (grasping)
- "- $\bullet$  p<sub>1</sub>: grasping. Example prompt: Given <img>. Q: How to grasp the green object?. Target: A: First grasp the orange object and place it on the table, then grasp the green object." (Section B.1)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and textual plan.

### Task: TAMP planning p2 (stacking)
- "- $p_2$ : stacking. Example prompt: Given <img>. Q: How to stack the white object on top of the red object?. Target: A: First grasp the green object and place it on the table, then grasp the white object and place it on the red object." (Section B.1)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and textual plan.

### Task: Language-Table Task 1 (push closest block to same-color block)
- "| <b>Task 1.</b> Q: There is a block that is closest to |  |  |  |  |  |  |  |" (Table 3)
- "| {i.e., top right corner}. Push that block to          |  |  |  |  |  |  |  |" (Table 3)
- "| the other block of the same color.                    |  |  |  |  |  |  |  |" (Table 3)
- "Given the current image and high level task, PaLM-E issues a text instruction which a trained low-level policy executes for 4 seconds before PaLM-E issues a new text instruction." (Section B.2)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension based on the image+task input and text instruction output described for Language-Table.

### Task: Language-Table Task 2 (sort blocks by colors into corners)
- "| Task 2. Q: How to sort the blocks by colors | S |" (Table 3)
- "| into corners?                               |   |" (Table 3)
- "Given the current image and high level task, PaLM-E issues a text instruction which a trained low-level policy executes for 4 seconds before PaLM-E issues a new text instruction." (Section B.2)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension based on the image+task input and text instruction output described for Language-Table.

### Task: Language-Table Task 3 (push blocks on one side together)
- "Task 3. Q: How to push all the blocks that are on the {left/right} side together, without bringing over any of the blocks that are on the {right/left} side?" (Table 3)
- "Given the current image and high level task, PaLM-E issues a text instruction which a trained low-level policy executes for 4 seconds before PaLM-E issues a new text instruction." (Section B.2)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension based on the image+task input and text instruction output described for Language-Table.

### Task: Mobile manipulation affordance prediction
- "Affordance prediction. We investigate PaLM-E's performance at affordance prediction, i.e. whether a skill of the low-level policy can be executed in the current environment. This can be formulated as the VQA problem Given <img>. Q: Is it possible to <skill> here?." (Section 6.4)
- Inference: Treated the output as a text answer (yes/no) and assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension based on the VQA-style prompt.

### Task: Mobile manipulation failure detection
- "Failure detection. For a robot to do closed-loop planning, it is also important to detect failures, as is shown in (Huang et al., 2022c). The multi-modal prompt is <code>Given <img>.</code> Q: <code>Was <skill> successful?</code>." (Section 6.4)
- Inference: Treated the output as a text answer (yes/no) and assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension based on the VQA-style prompt.

### Task: Mobile manipulation long-horizon planning
- "Real robot results: Long-horizon planning. Finally, we use PaLM-E to perform *embodied planning* end-to-end for mobile manipulation tasks. The prompt structure for this task is Human: <instruction> Robot: <step history>. I see <img>. PaLM-E is trained to generate the next step of the plan, conditioned on the history of taken steps and the current image observation of the scene." (Section 6.4)
- Inference: Assigned 2D (image) and 1D (text) input dimensions and 1D (text) output dimension from the image+text prompt and text next-step output.

### Task: OK-VQA
- "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)
- "The inputs to PaLM-E consist of text and (multiple) continuous observations." (Section 3)
- "The output of PaLM-E is text generated auto-regressively by the model, which could be an answer to a question, or a sequence of decisions produced by PaLM-E in textual form that should be executed by a robot." (Section 3)
- Inference: Treated OK-VQA as image+question input with a text answer output and assigned 2D (image) and 1D (text) input dimensions plus 1D (text) output dimension based on the vision-language task listing and the model input/output description.

### Task: VQA v2
- "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)
- "The inputs to PaLM-E consist of text and (multiple) continuous observations." (Section 3)
- "The output of PaLM-E is text generated auto-regressively by the model, which could be an answer to a question, or a sequence of decisions produced by PaLM-E in textual form that should be executed by a robot." (Section 3)
- Inference: Treated VQA v2 as image+question input with a text answer output and assigned 2D (image) and 1D (text) input dimensions plus 1D (text) output dimension based on the vision-language task listing and the model input/output description.

### Task: COCO captioning
- "we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)
- "If the task can be accomplished by outputting text only as, e.g., in embodied question answering or scene description tasks, then the output of the model is directly considered to be the solution for the task." (Section 3)
- Inference: Treated COCO captioning as image input with text caption output and assigned 2D (image) input dimension and 1D (text) output dimension based on the captioning task listing and the model's text-output description.

### Task: TriviaQA (wiki) (EM) (NLG)
- "TriviaQA (wiki) (EM)    | 48.5    | 10.1                     | 72.7     | 31.8                     | 81.4      | 74.6                      | NLG" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLG benchmark as text input with generated text output and assigned 1D (text) input/output dimensions based on the NLG category and language-benchmark description.

### Task: Natural Questions (EM) (NLG)
- "Natural Questions (EM)  | 10.6    | 1.6                      | 23.1     | 7.6                      | 29.3      | 27.2                      | NLG" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLG benchmark as text input with generated text output and assigned 1D (text) input/output dimensions based on the NLG category and language-benchmark description.

### Task: WebQuestions (EM) (NLG)
- "WebQuestions (EM)       | 12.6    | 3.4                      | 19.8     | 7.9                      | 22.6      | 21.8                      | NLG" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLG benchmark as text input with generated text output and assigned 1D (text) input/output dimensions based on the NLG category and language-benchmark description.

### Task: Lambada (NLG)
- "Lambada                 | 57.8    | 1.4                      | 75.5     | 26.1                     | 81.8      | 83.3                      | NLG" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLG benchmark as text input with generated text output and assigned 1D (text) input/output dimensions based on the NLG category and language-benchmark description.

### Task: HellaSwag (NLU)
- "HellaSwag               | 68.2    | 48.4                     | 79.7     | 75.3                     | 83.6      | 83.5                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: StoryCloze (NLU)
- "StoryCloze              | 78.7    | 68.7                     | 83.8     | 83.9                     | 86.1      | 86.3                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: Winograd (NLU)
- "Winograd                | 82.4    | 71.8                     | 85.3     | 86.4                     | 87.5      | 89.0                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: Winogrande (NLU)
- "Winogrande              | 68.3    | 55.3                     | 76.8     | 72.5                     | 83.7      | 83.0                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: RACE-M (NLU)
- "RACE-M                  | 57.7    | 43.2                     | 64.1     | 57.4                     | 69.3      | 70.3                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: RACE-H (NLU)
- "RACE-H                  | 41.6    | 33.2                     | 48.7     | 42.3                     | 52.1      | 52.8                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: PIQA (NLU)
- "PIQA                    | 76.1    | 68.1                     | 80.9     | 78.2                     | 83.9      | 84.9                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: ARC-e (NLU)
- "ARC-e                   | 71.3    | 53.4                     | 78.9     | 71.4                     | 85.0      | 86.3                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: ARC-c (NLU)
- "ARC-c                   | 42.3    | 30.9                     | 51.8     | 46.7                     | 60.1      | 62.6                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: OpenBookQA (NLU)
- "OpenBookQA              | 47.4    | 41.4                     | 51.2     | 51.6                     | 53.6      | 55.8                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: BoolQ (NLU)
- "BoolQ                   | 64.7    | 61.6                     | 83.1     | 81.6                     | 88.7      | 89.4                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: Copa (NLU)
- "Copa                    | 82.0    | 77.0                     | 93.0     | 91.0                     | 91.0      | 93.0                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: RTE (NLU)
- "RTE                     | 57.8    | 54.9                     | 71.5     | 59.6                     | 78.7      | 75.1                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: Wic (NLU)
- "Wic                     | 50.6    | 50.0                     | 48.6     | 50.2                     | 63.2      | 64.1                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: WSC (NLU)
- "WSC                     | 81.4    | 68.4                     | 84.9     | 75.8                     | 86.3      | 85.6                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: ReCoRD (NLU)
- "ReCoRD                  | 87.8    | 71.2                     | 91.0     | 78.5                     | 92.8      | 92.5                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.

### Task: CB (NLU)
- "CB                      | 41.1    | 37.5                     | 55.4     | 73.2                     | 83.9      | 80.3                      | NLU" (Table 8)
- "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
- Inference: Treated this NLU benchmark as text input with a label-style output and assigned 1D (text) input dimension and 0D output dimension based on the NLU category and language-benchmark description.
