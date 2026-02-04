1. Number of distinct tasks evaluated: 36.

> "In the global version, we consider the following three VQA tasks:" (Section B.1. Task and Motion Planning (TAMP))
> "- $q_2$ : object-table relation." (Section B.1. Task and Motion Planning (TAMP))
> "- $q_3$ : object-object relations." (Section B.1. Task and Motion Planning (TAMP))
> "- $q_4$ : plan feasibility." (Section B.1. Task and Motion Planning (TAMP))
> "#### as well as the two planning tasks" (Section B.1. Task and Motion Planning (TAMP))
> "- $\bullet$  p<sub>1</sub>: grasping." (Section B.1. Task and Motion Planning (TAMP))
> "- $p_2$ : stacking." (Section B.1. Task and Motion Planning (TAMP))
> "the VQA task  $q_1$  is about the color of an object." (Section B.1. Task and Motion Planning (TAMP))
>
> "| <b>Task 1.</b> Q: There is a block that is closest to |" (Table 3, Section 6.3. Language-Table Environment)
> "| {i.e., top right corner}. Push that block to          |" (Table 3, Section 6.3. Language-Table Environment)
> "| the other block of the same color.                    |" (Table 3, Section 6.3. Language-Table Environment)
> "| Task 2. Q: How to sort the blocks by colors | S |" (Table 3, Section 6.3. Language-Table Environment)
> "| into corners?                               |   |" (Table 3, Section 6.3. Language-Table Environment)
> "Task 3. Q: How to push all the blocks that are on the {left/right} side together, without bringing over any of the blocks that are on the {right/left} side?" (Table 3, Section 6.3. Language-Table Environment)
>
> "we develop 3 use cases to test the embodied reasoning abilities of PaLM-E: affordance prediction, failure detection, and long-horizon planning." (Section 6.4. Mobile Manipulation Environment)
>
> "Although it is not the focus of our work, we report in Tab. 5 results on general vision-language tasks, including OK-VQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5. Performance on General Visual-Language Tasks)
>
> "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6. Performance on General Language Tasks)

2. Number of trained model instances required to cover all tasks: 1.

> "Here we show that a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6. Experiments)
> "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use different-finetuned models for the different tasks." (Table 5, Section 6.5. Performance on General Visual-Language Tasks)

3. Task–Model Ratio = (1) / (2):

$$
\boxed{
\frac{36\ \text{tasks}}{1\ \text{model}} = 36
}
$$
