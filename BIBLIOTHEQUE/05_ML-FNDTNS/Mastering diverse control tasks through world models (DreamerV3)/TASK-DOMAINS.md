# Mastering diverse control tasks through world models (2025)
Source: Mastering diverse control tasks through world models (DreamerV3).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control (Atari 2600 games) | images (pixels) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (ProcGen games) | images (pixels) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (DMLab tasks) | images (pixels) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (Atari100k games) | images (pixels) (inferred) | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (Proprio Control Suite robot tasks) | proprioceptive vector inputs | 0D (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (Visual Control Suite robot tasks) | high-dimensional images | 2D (x, y) (inferred) | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (BSuite environments) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |
| control (Minecraft Diamond task) | pixels | 2D (x, y) (inferred) | Capped (inferred) | Not specified in the paper. | Constructed (inferred) | actions (categorical) | 0D (inferred) | Fixed (inferred) |
| control (Crafter) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Constructed (inferred) | actions | 0D (inferred) | Not specified in the paper. |

## Summary
Dreamer is evaluated across eight benchmark domains with over 150 tasks, and also in scaling experiments on Crafter, covering diverse control settings from games to simulated robotics. The benchmarks explicitly include visual and low-dimensional inputs with continuous and discrete actions, including proprioceptive vector inputs and high-dimensional images for the control suites. The Minecraft diamond task is an open-world setting with episodes capped at 36,000 steps, and the use of a learned world model that imagines futures implies constructed state across tasks.

## Evidence
### Task: control (Atari 2600 games)
- "This established benchmark contains 57 Atari 2600 games" (Section: Benchmarks)
- "Diverse visual domains used in the experiments." (Section: Fig. 2 caption)
- "Atari games, procedurally generated ProcGen levels, and DMLab tasks," (Section: Fig. 2 caption)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: Atari is listed among the paper's visual domains, so the input is treated as images and mapped to 2D (x, y). Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (ProcGen games)
- "This benchmark of 16 games features randomized levels and visual distractions" (Section: Benchmarks)
- "Diverse visual domains used in the experiments." (Section: Fig. 2 caption)
- "Atari games, procedurally generated ProcGen levels, and DMLab tasks," (Section: Fig. 2 caption)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: ProcGen is listed among the paper's visual domains, so the input is treated as images and mapped to 2D (x, y). Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (DMLab tasks)
- "This suite of 30 tasks features three-dimensional environments that test spatial and temporal reasoning" (Section: Benchmarks)
- "Diverse visual domains used in the experiments." (Section: Fig. 2 caption)
- "Atari games, procedurally generated ProcGen levels, and DMLab tasks," (Section: Fig. 2 caption)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: DMLab is listed among the paper's visual domains, so the input is treated as images and mapped to 2D (x, y). Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (Atari100k games)
- "This data-efficiency benchmark contains 26 Atari games and a budget of only 400,000 frames" (Section: Benchmarks)
- "Diverse visual domains used in the experiments." (Section: Fig. 2 caption)
- "Atari games, procedurally generated ProcGen levels, and DMLab tasks," (Section: Fig. 2 caption)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: Atari100k is described as Atari games; Atari is listed among the paper's visual domains, so the input is treated as images and mapped to 2D (x, y). Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (Proprio Control Suite robot tasks)
- "contains 20 simulated robot tasks with continuous actions, proprioceptive vector inputs" (Section: Benchmarks)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- Inference: Proprioceptive vector inputs are treated as 0D. Continuous actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (Visual Control Suite robot tasks)
- "the agent receives only high-dimensional images as input" (Section: Benchmarks)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: Image inputs are mapped to 2D (x, y). Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (BSuite environments)
- "includes 23 environments with a total of 468 configurations" (Section: Benchmarks)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state. Other input/dynamics details are not specified.

### Task: control (Minecraft Diamond task)
- "Every episode in this game is set in a unique randomly generated and infinite three-dimensional world." (Section: Minecraft)
- "requires exploring farsighted strategies from pixels and sparse rewards in an open world" (Section: Introduction)
- "Episodes last until the player dies or up to 36,000 steps" (Section: Minecraft)
- "We form a categorical action space of the actions provided by the MineRL competition" (Section: Minecraft)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- Inference: Pixel inputs are mapped to 2D (x, y). The 36,000-step episode limit implies Capped input dynamics. A categorical action space implies Fixed output dynamics, with single-step outputs (0D). The world model/imagination implies Constructed state.

### Task: control (Crafter)
- "we train 6 model sizes ranging from 12 million to 400 million parameters, as well as different replay ratios on Crafter" (Section: Scaling properties)
- "Dreamer learns a model of the environment and improves its behaviour by imagining future scenarios." (Section: Introduction)
- "an actor neural network chooses actions to reach the best outcomes." (Section: Introduction)
- Inference: Actions are treated as single-step outputs (0D). The world model/imagination implies Constructed state. Other input/dynamics details are not specified.

## CSV Output (required)
```csv
task,input,in_dimension,in_dynamic,attention_dynamic,state_dynamic,output,out_dimension,out_dynamic
"control (Atari 2600 games)","images (pixels) (inferred)","2D (x, y) (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (ProcGen games)","images (pixels) (inferred)","2D (x, y) (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (DMLab tasks)","images (pixels) (inferred)","2D (x, y) (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (Atari100k games)","images (pixels) (inferred)","2D (x, y) (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (Proprio Control Suite robot tasks)","proprioceptive vector inputs","0D (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (Visual Control Suite robot tasks)","high-dimensional images","2D (x, y) (inferred)","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (BSuite environments)","Not specified in the paper.","Not specified in the paper.","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
"control (Minecraft Diamond task)","pixels","2D (x, y) (inferred)","Capped (inferred)","Not specified in the paper.","Constructed (inferred)","actions (categorical)","0D (inferred)","Fixed (inferred)"
"control (Crafter)","Not specified in the paper.","Not specified in the paper.","Not specified in the paper.","Not specified in the paper.","Constructed (inferred)","actions","0D (inferred)","Not specified in the paper."
```
