# Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models (2024)
Source: Language Agent Tree Search (LATS).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Question answering (multi-hop) | Natural language question; retrieved Wikipedia passages/observations | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Answer text and action commands (Search/Lookup/Finish) | 1D (t) | Not specified in the paper. |
| Program synthesis (Python code generation) | Natural language docstring/description; function signature; test-suite/ compiler feedback observations | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Python function/program implementation (code) | 1D (t) | Not specified in the paper. |
| Web navigation / shopping control | Natural language instruction; webpage observations (structured text) | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Search and click action commands | 1D (t) | Not specified in the paper. |
| Math equation generation (Game of 24) | Four numbers with basic arithmetic operations | 1D (t) | Not specified in the paper. | Dynamic (inferred) | Constructed (inferred) | Equation that equals 24 using each number once | 1D (t) | Not specified in the paper. |

## Summary
The paper evaluates LATS on four task domains: multi-hop question answering with retrieval, program synthesis from natural language, web navigation/shopping with search-and-click actions, and the Game of 24 math puzzle. Inputs and outputs are described as language sequences (tokens), so the covered modalities are primarily 1D text. The paper does not specify fixed/capped/open interface dynamics, while the search-tree selection and external memory imply dynamic attention and constructed state (inferred) across tasks.

## Evidence
### Task: Question answering (multi-hop)
- "HotPotQA (Yang et al., 2018), a multi-hop question-answering benchmark that requires retrieval over two or more Wikipedia passages." (Sec. 5.1 HotPotQA)
- "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists." (Sec. E.1 Base Acting Prompt)
- "(3) Finish[answer], which returns the answer and finishes the task." (Sec. E.1 Base Acting Prompt)
- "Both x and y are language sequences, which are comprised of a list of tokens" (Sec. 3.1 Problem Setting and Prompting)
- Inference: Attention Dynamic = Dynamic (inferred) because "a child node is selected at each tree level"; State Dynamic = Constructed (inferred) because "This tree is stored in an external long-term memory structure." (Sec. 4.2 LATS)

### Task: Program synthesis (Python code generation)
- "Both datasets measure the correctness of synthesized programs in Python from natural language docstrings." (Sec. 5.2 Programming)
- "Each problem includes a function signature, docstring description, reference implementation, and multiple unit tests" (Sec. D.2 Programming)
- "We use individual solutions as the action space and test suite and compiler feedback as the external observation." (Sec. 5.2 Programming)
- "Both x and y are language sequences, which are comprised of a list of tokens" (Sec. 3.1 Problem Setting and Prompting)
- Inference: Attention Dynamic = Dynamic (inferred) because "a child node is selected at each tree level"; State Dynamic = Constructed (inferred) because "This tree is stored in an external long-term memory structure." (Sec. 4.2 LATS)

### Task: Web navigation / shopping control
- "Agents must navigate a website through a variety of commands to purchase an item matching a user specification." (Sec. 5.3 WebShop)
- "The action space consists of query searches and button clicks." (Sec. D.3 WebShop)
- "Instructions are crowdsourced natural language specifying product attributes and options" (Sec. D.3 WebShop)
- "simple mode converts the raw HTML into a structured text observation" (Sec. D.3 WebShop)
- "Both x and y are language sequences, which are comprised of a list of tokens" (Sec. 3.1 Problem Setting and Prompting)
- Inference: Attention Dynamic = Dynamic (inferred) because "a child node is selected at each tree level"; State Dynamic = Constructed (inferred) because "This tree is stored in an external long-term memory structure." (Sec. 4.2 LATS)

### Task: Math equation generation (Game of 24)
- "Game of 24 is a mathematical reasoning challenge where the goal is to use basic arithmetic operations to construct 24 out of 4 numbers." (Sec. D.4 Game of 24)
- "correct equation that equals 24 and uses each input number only once." (Sec. D.4 Game of 24)
- "Both x and y are language sequences, which are comprised of a list of tokens" (Sec. 3.1 Problem Setting and Prompting)
- Inference: Attention Dynamic = Dynamic (inferred) because "a child node is selected at each tree level"; State Dynamic = Constructed (inferred) because "This tree is stored in an external long-term memory structure." (Sec. 4.2 LATS)
