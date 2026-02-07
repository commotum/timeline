# Graph of Thoughts: Solving Elaborate Problems with Large Language Models (Not specified in the paper.)
Source: Graph of Thoughts (GoT).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sorting | list of numbers | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | sorted list of numbers | 1D (t) (inferred) | Capped (inferred) |
| set intersection | two sets of numbers (lists) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | intersection set of numbers (list) | 1D (t) (inferred) | Capped (inferred) |
| keyword counting | input text with keywords (countries) | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | dictionary of keyword frequencies (JSON) | 2D (x, y) (inferred) | Capped (inferred) |
| document merging | four NDA documents | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | merged NDA document | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper's applied use cases cover numeric list manipulation (sorting and set intersection) and text tasks (keyword counting in text and NDA document merging). Inputs are 1D sequences (lists or text), and outputs are 1D sequences or a key-count dictionary that we classify as 2D tabular structure. Across tasks, sizes are capped by explicit list sizes or fixed numbers of documents/passages, and GoT's controller/GRS imply dynamic attention and constructed state (inferred).

## Evidence
### Task: sorting
- "Sort the following list of numbers in ascending order. Output only the sorted list of numbers, no additional text." (Table 3, Sorting prompts)
- "We present the prompts only for the sorting of 32-element lists, as those for 64-element and 128-element lists are identical" (Example Prompts - Sorting)
- Inference: Classified input/output as 1D (t) and dynamics as Capped based on "sorting of 32-element lists, as those for 64-element and 128-element lists are identical" (Example Prompts - Sorting). Classified attention/state as Dynamic/Constructed because "The Controller implements a specific strategy for selecting thoughts from its GRS structure" and "an instance of the GRS maintains the continually updated information about the LLM reasoning process." (Sections 4.4-4.5)

### Task: set intersection
- "we also consider set operations, focusing on set intersection." (Section 5.2 Set Operations)
- "Find the intersection of two sets of numbers. Output only the set of numbers that are present in both sets, no additional text." (Table 10)
- Inference: Classified input/output as 1D (t) and dynamics as Capped based on "different set sizes of 32, 64 and 128 elements." (Section 5.2 Set Operations). Classified attention/state as Dynamic/Constructed because "The Controller implements a specific strategy for selecting thoughts from its GRS structure" and "an instance of the GRS maintains the continually updated information about the LLM reasoning process." (Sections 4.4-4.5)

### Task: keyword counting
- "Keyword counting finds the frequency of keywords in a given category (countries in our example implementation) within the input text." (Section 5.3 Keyword Counting)
- "final output should only contain the frequency of each country that appears at least once in the following json format" (Table 16)
- Inference: Classified input as 1D (t) and dynamics as Capped based on "Split the following input text into 4 paragraphs of approximately same length." (Table 16). Classified the output as 2D (x, y) because it is a key-value dictionary of country counts. Classified attention/state as Dynamic/Constructed because "The Controller implements a specific strategy for selecting thoughts from its GRS structure" and "an instance of the GRS maintains the continually updated information about the LLM reasoning process." (Sections 4.4-4.5)

### Task: document merging
- "the goal is to generate a new Non-Disclosure Agreement (NDA) document based on several input ones" (Section 5.4 Document Merging)
- "Merge the following 4 NDA documents <Doc1> - <Doc4> into a single NDA" (Table 29)
- Inference: Classified input/output as 1D (t) and dynamics as Capped based on "Merge the following 4 NDA documents <Doc1> - <Doc4> into a single NDA" (Table 29). Classified attention/state as Dynamic/Constructed because "The Controller implements a specific strategy for selecting thoughts from its GRS structure" and "an instance of the GRS maintains the continually updated information about the LLM reasoning process." (Sections 4.4-4.5)
