# Tree of Thoughts: Deliberate Problem Solving with Large Language Models (2023)
Source: Tree of Thoughts (ToT)- Deliberate Problem Solving with Large Language Models.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Game of 24 puzzle solving (equation generation) | 4 numbers | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | An equation to reach 24 | 1D (t) (inferred) | Capped (inferred) |
| Creative writing (constrained passage generation) | 4 random sentences | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | A coherent passage of 4 paragraphs ending in the 4 input sentences | 1D (t) (inferred) | Capped (inferred) |
| Mini crosswords solving (board completion) | 10 clues (5 horizontal, 5 vertical) | 1D (t) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | A 5x5 letter board | 2D (x, y) (inferred) | Fixed (inferred) |
| GSM8K question answering (numerical answer prediction) | Question text | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Numeric answer ("the answer is n") | 0D (inferred) | Fixed (inferred) |
| StrategyQA question answering (binary classification) | Question text | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | Binary answer ("the answer is yes"/"the answer is no") | 0D (inferred) | Fixed (inferred) |

## Summary
The OCR supports five tasks: three main tasks in Section 4 (Game of 24, Creative Writing, Mini Crosswords) plus two appendix tasks in Section B.1 (GSM8K and StrategyQA). Inputs are language/numeric sequences (1D (t) inferred), while outputs span sequence generation (1D), grid completion (2D), and scalar/binary answers (0D) depending on task. The paper’s ToT procedure repeatedly samples, evaluates, votes, and searches over candidate thoughts, which supports Dynamic attention and Constructed state for all listed tasks (inferred from Sections 3, 4, and B.1).

## Evidence
### Task: Game of 24 puzzle solving (equation generation)
- "Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords." (Abstract)
- "Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic arithmetic operations (+-\*/) to obtain 24." (Section 4.1)
- "| Input | 4 numbers (4 9 10 13) |" and "| Output | An equation to reach 24 (13-9)*(10-4)=24 |" (Table 1, Section 4)
- Inference: Input/Output dimensions and dynamics are inferred from "4 numbers" and "equation"; Attention/State are inferred from ToT search behavior, e.g., "ToT frames any problem as a search over a tree, where each node is a state s = [x, z_{1\cdots i}]" and BFS keeps/evaluates alternatives (Section 3, Section 4.1).

### Task: Creative writing (constrained passage generation)
- "Next, we invent a creative writing task where the input is 4 random sentences and the output should be a coherent passage with 4 paragraphs that end in the 4 input sentences respectively." (Section 4.2)
- "| Output     | An equation to reach 24 (13-9)*(10-4)=24                                          | A passage of 4 paragraphs ending in the 4 sentences      | 5x5 letters: SHOWN;<br>WIRRA; AVAIL;                |" (Table 1, Section 4)
- "We build a ToT with depth 2 (and only 1 intermediate thought step) — the LM first generates k=5 plans and votes for the best one (Figure 4), then similarly generate k=5 passages based on the best plan then vote for the best one." (Section 4.2)
- Inference: 1D sequence structure is inferred from sentence/passage text; input dynamics are inferred as Fixed from the explicit "4 random sentences" constraint, and output dynamics are inferred as Capped because passage length can vary under LM generation limits; Dynamic attention and Constructed state are inferred from sample-and-vote tree search over intermediate plans (Section 3 and Section 4.2).

### Task: Mini crosswords solving (board completion)
- "Here we explore  $5\times 5$  mini crosswords as a harder search problem involving natural language." (Section 4.3)
- "For each task, the input describes the 5 horizontal clues and 5 vertical clues, and the output should be a board of 5 × 5 = 25 letters to solve the crosswords." (Section 4.3)
- "We leverage a depth-first search (Algorithm 2) that keeps exploring the most promising subsequent word clue until the state is no longer promising, then backtrack to the parent state to explore alternative thoughts." (Section 4.3)
- Inference: Output dimension is 2D (x, y) from the explicit 5x5 board; input dimension is treated as 1D clue sequence; input/output dynamics are inferred as Fixed from the explicit "5 horizontal clues and 5 vertical clues" and fixed "5 × 5 = 25 letters" board; Dynamic attention and Constructed state are inferred from DFS with proposal aggregation, pruning, and backtracking (Sections 3 and 4.3).

### Task: GSM8K question answering (numerical answer prediction)
- "## B.1 Extension to new tasks (GSM8k, StrategyQA) with zero-shot ToT" (Section B.1)
- "we implemented a simple and generic zero-shot ToT-BFS similar to creative writing (sample 5 problem solving strategies then vote for the best one; then sample 5 solutions based on the best strategy then vote for the best one) for GSM8K and StrategyQA with few extra lines of code:" (Section B.1)
- "gsm8k_format = '\"the answer is n\" where n is a number'" (Section B.1)
- Inference: 1D input is inferred from question text and input dynamics as Capped under LM sequence interface limits; task type, 0D output, and Fixed output dynamics are inferred from the numeric answer format ("the answer is n"); Dynamic attention and Constructed state are inferred from the explicit ToT-BFS sample-and-vote process in B.1 ("sample 5 problem solving strategies then vote for the best one; then sample 5 solutions based on the best strategy then vote for the best one").

### Task: StrategyQA question answering (binary classification)
- "## B.1 Extension to new tasks (GSM8k, StrategyQA) with zero-shot ToT" (Section B.1)
- "we implemented a simple and generic zero-shot ToT-BFS similar to creative writing (sample 5 problem solving strategies then vote for the best one; then sample 5 solutions based on the best strategy then vote for the best one) for GSM8K and StrategyQA with few extra lines of code:" (Section B.1)
- "strategyqa_format = 'either \"the answer is yes\" or \"the answer is no\"'" (Section B.1)
- Inference: 1D input is inferred from question text and input dynamics as Capped under LM sequence interface limits; binary answer format supports 0D output classification with Fixed output dynamics; Dynamic attention and Constructed state are inferred from ToT-BFS sample-and-vote strategy/solution selection in Section B.1.
