# AI safety via debate (Not specified in the paper.)
Source: AI safety via debate.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| question answering (debate) | questions (sentences plus context) | 1D (t) (inferred) | Not specified in the paper. | Dynamic (inferred) | Not specified in the paper. | answers / debate statements (sentences) | 1D (t) (inferred) | Capped |
| debate judging (winner selection) | debate transcript (q, a, s) | 1D (t) (inferred) | Capped (inferred) | Not specified in the paper. | Not specified in the paper. | winner decision (which agent wins) | 0D (inferred) | Fixed (inferred) |
| image classification | images (MNIST digits; cat/dog) | 2D (x, y) (inferred) | Not specified in the paper. | Dynamic (inferred) | Not specified in the paper. | class label (digit or cat/dog) | 0D (inferred) | Fixed (inferred) |
| control (environment interaction) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | sequence of actions | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper frames debate as a question-answering system with text questions and answer statements, plus a judging task that selects a winner from the debate. It also applies debate to image classification (MNIST digits and cat-vs-dog) where agents reveal selected pixels, and discusses environment interaction where outputs are action sequences. Across these tasks, text is treated as 1D sequences and images as 2D grids (both inferred from the described modalities), outputs range from capped text statements to fixed labels and potentially open-ended action sequences, and dynamic attention is implied where agents choose which parts of inputs to reveal.

## Evidence
### Task: question answering (debate)
- "We will initially consider a question-answering setting, though Section 2.3 covers other settings including environment interaction. We have a set of questions Q, answers A, and debate statements S." (Section 2 The debate game)
- "For question-answering  $a \in A$  and  $s \in S$  could be any moderate length sentence, and  $q \in Q$  a sentence plus additional context." (Section 2 The debate game)
- "Given a question or proposed action, two agents take turns making short statements up to a limit, then a human judges which of the agents gave the most true, useful information." (Abstract)
- "To support large context, we let the agents reveal small parts of q in their statements." (Section 2.3 Removing oversimplifications)
- Inference: Marked 1D (t) for input/output because questions, answers, and statements are described as sentences; marked Dynamic attention because agents can choose which parts of q to reveal.

### Task: debate judging (winner selection)
- "The judge sees the debate (q, a, s) and decides which agent wins." (Section 2 The debate game)
- "The two agents take turns making statements  $s_0, s_1, \ldots, s_{n-1} \in S$ ." (Section 2 The debate game)
- Inference: Marked 1D (t) input because the debate is composed of textual statements; marked Capped input because the debate has a finite number of statements; marked 0D/Fixed output because the judge makes a single winner decision.

### Task: image classification
- "A random MNIST image is shown to the two debating agents but not the judge. The debaters state their claimed label up front, then reveal one nonzero pixel per turn to the judge up to a total of 4 or 6." (Figure 2 caption, Section 3)
- "Choose a random image of either a cat or a dog, and show the image to both human agents but not the human judge." (Section 3.2 Human experiment: cat vs. dog)
- "each agent is allowed to reveal a single pixel of the image to the judge." (Section 3.2 Human experiment: cat vs. dog)
- Inference: Marked 2D (x, y) input because the tasks operate over images/pixels; marked Dynamic attention because agents choose which pixels to reveal; marked 0D/Fixed output because the task is to select a single class label.

### Task: control (environment interaction)
- "Environment interaction: If we want a system to take actions that affect the environment such as operating a robot, the desired output is a sequence of actions  $a_0, a_1, \ldots$  where each action can only be computed once the previous action is taken." (Section 2.3 Removing oversimplifications)
- Inference: Marked 1D (t) output because actions are an ordered sequence; marked Open dynamics because the sequence is presented as ongoing (a0, a1, ...).
