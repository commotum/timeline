1. **Number of distinct tasks evaluated:** 12

> "We evaluate models on ARC-AGI and ConceptARC, where the challenge is to solve diverse tasks from only 2–5 demonstrations." (Section 3.1, **SETUP AND COMPARISON PROTOCOL**)
>
> "As part of our broader evaluation, we examine performance on a diverse set of five visual games that span both puzzle-solving and board play. These tasks provide an additional perspective on how the models handle structured visual inputs and varying interaction styles. The puzzle-based tasks, *Hitori 5x5* and two versions of *Sudoku* (standard one and *Mini*), focus on solving constraint-based problems in structured grids, where success depends on extracting spatial patterns and enforcing global consistency from local information. The board games, *Connect 4* and *Chess Mate-in-1*, shift attention to game scenarios where the goal is to identify the winning move in a given configuration." (Section 4.2.1, **VISUAL GAMES**)
>
> "We evaluate route planning in 2D grid environments through two tasks: *Maze* and *Shortest Path*." (Section 4.2.2, **ROUTE PLANNING**)
>
> "Our study spans one-dimensional Elementary Cellular Automata (ECA) Wolfram (1984), a foundational class of binary-state systems, as well as two-dimensional Life-like Cellular Automata, including Conway's Game of Life Gardner (1970), defined by various birth and survival (B/S) rules. Additionally, we consider Langton's ant Langton (1986), a deterministic agent-based system, where the task is to predict the complete grid state after n steps of evolution." (Section 4.2.3, **CELLULAR AUTOMATA**)

2. **Number of trained model instances required to cover all tasks:** 12

> "Let  $\mathcal{T}$  denote a task with dataset  $\mathcal{D}_{\mathcal{T}} = \{(x_i, y_i)\}_{i=1}^n$ , where each  $x_i$  and  $y_i$  is an input-output pair." (Section 3.1, **SETUP AND COMPARISON PROTOCOL**)
>
> "Here we systematically vary n, the number of training examples per task, to trace curves and quantify the rate of skill acquisition rather than focusing solely on endpoint accuracy." (Section 3.1, **SETUP AND COMPARISON PROTOCOL**)
>
> "Figure 6: Qualitative examples for *Base Maze* and *Shortest Path* tasks, after fine-tuning with n=300 samples." (Section 4.2.2, **ROUTE PLANNING**)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{12\ \text{tasks}}{12\ \text{models}} = 1
}
$$
