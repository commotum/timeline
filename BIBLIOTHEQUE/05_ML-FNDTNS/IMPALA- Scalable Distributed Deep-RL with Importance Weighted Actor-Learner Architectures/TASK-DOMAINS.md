# IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (Not specified in the paper.)
Source: IMPALA- Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Control (reinforcement learning) in DeepMind Lab single-task suite: planning; maze navigation; laser tag; fruit collection | Visual observations (96x72 images) | 3D (x, y, t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | Discrete DeepMind Lab action set (e.g., forward/backward/strafe/look/fire) | 1D (t) (inferred) | Open (inferred) |
| Control (reinforcement learning) in DMLab-30 multi-task suite | Visual observations (96x72 images) and language instructions (text channel for some tasks) | 3D (x, y, t) (inferred); 1D (t) (inferred) | Open (inferred) | Not specified in the paper. | Constructed (inferred) | Discrete DeepMind Lab action set (e.g., forward/backward/strafe/look/fire) | 1D (t) (inferred) | Open (inferred) |
| Control (reinforcement learning) in Atari-57 (ALE) games | Visual observations (84x84 grayscale frames) with 4-frame stacking | 3D (x, y, t) (inferred) | Fixed (inferred) | Not specified in the paper. | Direct (inferred) | Discrete Atari action set (18 actions) | 1D (t) (inferred) | Open (inferred) |

## Summary
The paper evaluates a single RL agent on DeepMind Lab tasks (both a five-task single-task suite and the DMLab-30 multi-task suite) and on the Atari-57 ALE game suite, all of which are control tasks with visual observations and discrete actions. DMLab-30 also includes instruction-based tasks with grounded language, implying a mixed visual-plus-language input space, while Atari uses fixed-size grayscale frame stacks. Attention dynamics are not specified, while state is constructed for DMLab via LSTM-based models and direct for Atari via feed-forward models.

## Evidence
### Task: Control (reinforcement learning) in DeepMind Lab single-task suite: planning; maze navigation; laser tag; fruit collection
- "single-task scenario where we train agents individually on 5 different DeepMind Lab tasks." (Section 5.2)
- "planning task, two maze navigation tasks, a laser tag task with scripted bots and a simple fruit collection task." (Section 5.2)
- "| Image Width                          | 96                  |" (Table D.3)
- "| Image Height                         | 72                  |" (Table D.3)
- "Table D.2. Action set used in all tasks from the DeepMind Lab environment, including the DMLab-30 experiments." (Table D.2)
- Inference: In/Out Dimension and Open dynamics inferred from temporal trajectories ("trajectory of states, actions and rewards") and "discounted infinite-horizon RL in Markov Decision Processes"; constructed state inferred from "with an LSTM before the policy and value". (Sections 3, 4, 5)

### Task: Control (reinforcement learning) in DMLab-30 multi-task suite
- "DMLab-30, a set of 30 diverse tasks built on DeepMind Lab." (Section 5.3.1)
- "instruction-based tasks with grounded language" (Section 5.3.1)
- "For tasks with a language channel we used an LSTM with text embeddings as input." (Section 5)
- "| Image Width                          | 96                  |" (Table D.3)
- "Table D.2. Action set used in all tasks from the DeepMind Lab environment, including the DMLab-30 experiments." (Table D.2)
- Inference: Spatiotemporal + language dimensions and Open dynamics inferred from trajectories ("trajectory of states, actions and rewards") and "discounted infinite-horizon RL in Markov Decision Processes"; constructed state inferred from LSTM usage ("For tasks with a language channel we used an LSTM with text embeddings as input."). (Sections 3, 4, 5)

### Task: Control (reinforcement learning) in Atari-57 (ALE) games
- "Its 57 tasks pose challenging reinforcement learning problems including exploration, planning, reactive play and complex visual input." (Section 5.3.2)
- "stacking the 4 most recent observations at each step." (Section 5.3.2)
- "| Image Width                                       | 84                                                |" (Table G.1)
- "| Image Height                                      | 84                                                |" (Table G.1)
- "full Atari action set consisting of 18 actions." (Appendix G)
- "All agents trained on Atari are equipped only with a feed forward network" (Appendix G)
- Inference: 3D (x, y, t) input and Fixed input dynamics inferred from frame stacking ("stacking the 4 most recent observations at each step.") and fixed image size (image width/height); Direct state inferred from feed-forward-only models; Open action dynamics inferred from temporal trajectories and "discounted infinite-horizon RL in Markov Decision Processes." (Sections 3, 4, Appendix G)

---

## CSV Output (required)
CSV file written to: /home/jake/Developer/timeline/BIBLIOTHEQUE/05_ML-FNDTNS/IMPALA- Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures/.TASK-DOMAINS.csv.tmp.ba077cb0e77b47199c7af3edff1adc46
