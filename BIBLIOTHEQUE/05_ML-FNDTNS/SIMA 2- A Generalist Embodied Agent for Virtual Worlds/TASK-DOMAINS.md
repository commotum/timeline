# SIMA 2: A Generalist Embodied Agent for Virtual Worlds (2025)
Source: SIMA 2- A Generalist Embodied Agent for Virtual Worlds.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Embodied instruction-following control | Stream of RGB video frames plus natural-language instruction/history | 1D (t); 3D (x, y, t) | Open (inferred) | Dynamic (inferred) | Constructed | Keyboard-and-mouse action chunks parsed from structured text | 1D (t); 2D (x, y) | Open (inferred) |
| Embodied dialogue | RGB video frames plus user prompts and dialogue history | 1D (t); 3D (x, y, t) | Open (inferred) | Dynamic (inferred) | Constructed | Natural-language dialogue responses | 1D (t) | Open (inferred) |
| Embodied question-answering | User question/instruction plus visual observations and on-screen cues | 1D (t); 3D (x, y, t) | Open (inferred) | Dynamic (inferred) | Constructed | Grounded natural-language answer after embodied actions | 1D (t) | Open (inferred) |
| Multi-modal instruction following | Text plus image/sketch/diagram prompts and visual context | 1D (t); 2D (x, y); 3D (x, y, t) | Open (inferred) | Dynamic (inferred) | Constructed | Action sequences and progress dialogue | 1D (t); 2D (x, y) | Open (inferred) |
| Code generation | Natural-language coding prompt | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Program text/code | 1D (t) | Capped (inferred) |
| Mathematical reasoning/problem solving | Math problem text | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Textual math answer | 1D (t) | Capped (inferred) |
| Scientific question-answering | STEM question text | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Textual answer | 1D (t) | Capped (inferred) |

## Summary
The paper covers embodied control tasks in interactive 3D worlds, including instruction-following, dialogue, embodied question-answering, and multi-modal instruction-following with text plus images/sketches/diagrams. It also reports retained text-centric capabilities on code generation, mathematical reasoning, and scientific question-answering benchmarks. Supported task domains therefore span 1D language and 3D spatiotemporal visual streams (plus 2D image prompts and mouse-space outputs), with outputs ranging from natural-language responses to keyboard-mouse action sequences. The embodied rows support Dynamic attention and Constructed state, while text-only benchmark rows are marked as capped/static/direct by inference from benchmark framing plus the paper’s explicit limited context-window note.

## Evidence
### Task: Embodied instruction-following control
- "Our quantitative analysis focuses on embodied tasks, in which the agent is given a text-based instruction and executes a series of keyboard-and-mouse actions in the environment to achieve a goal." (Section 3.4)
- "The input to the agent consists of a stream of RGB video frames at a resolution of 720p." (Section 3.2)
- "The environmental action space emulates a standard human-computer interface, encompassing 96 standard keyboard keys, mouse clicks, and discretized mouse movements representing relative (x, y) position changes." (Section 3.2)
- Inference: In/Out Dynamics and Attention are marked Open/Dynamic from the interactive loop framing (ongoing frame stream plus action chunks each step), including "with the agent specifying which modalities to produce at any given step." (Figure 3; Section 3.2)

### Task: Embodied dialogue
- "Embodied Dialogue SIMA 2 is, at its core, a Gemini model. Thus, just like Gemini, it can engage in dialogue with a user, making use of Gemini's general world knowledge and visual question-answering capabilities." (Section 4.1)
- "However, because SIMA 2 is situated in a 3D world, it can also take actions in response to user prompts, enabling a new capability for *embodied dialogue*." (Section 4.1)
- "This covers a wide variety of interactions, including confirmations of users' requests and proactively responding when tasks have been completed. SIMA 2 can even ask clarifying questions when a user's request is ambiguous." (Section 4.1)
- Inference: Dynamics are marked Open and Attention Dynamic because the paper describes interactive, turn-based dialogue coupled with embodied action rather than one fixed, single-shot input.

### Task: Embodied question-answering
- "One particularly unique form of interaction is embodied question-answering, in which a user asks or instructs the agent to find some piece of information, to which SIMA 2 must take embodied actions to determine the answer and respond in natural language." (Section 4.1)
- "Figure 4 | **Embodied Dialogue & Basic Reasoning**. SIMA 2 contains a variety of new capabilities, including embodied dialogue and basic reasoning. Above, SIMA 2 answers a user's question through embodied interaction." (Figure 4; Section 4.1)
- Inference: Dynamic attention and open dynamics are inferred from the explicit information-seeking behavior where the agent must act first to gather evidence before producing the answer.

### Task: Multi-modal instruction following
- "Figure 5 | **Complex Instructions & Multi-modal Prompting**. By inheriting Gemini's language understanding capabilities, SIMA 2 can handle a variety of novel, complex instructions, including breaking down instructions to successfully navigate to a specific room. SIMA 2 can also be prompted with images, including sketches, to specify locations, paths, or objects." (Figure 5; Section 4.1)
- "**Multi-modal Prompting** Gemini is natively multi-modal, processing images, audio, and video in addition to text. SIMA 2 thus inherits multi-modal prompting capabilities, allowing us to instruct the agent in novel ways." (Section 4.1)
- Inference: Open dynamics and Dynamic attention are inferred because the agent parses visual prompts, decomposes multi-step goals, and interacts iteratively while tracking progress (Section 4.4, "Complex Multi-modal Instruction Following").

### Task: Code generation
- "To quantitatively assess whether this is the case for SIMA 2, we evaluate the agent's general capabilities on three diverse benchmarks." (Section 4.3)
- "For coding, we use LiveCodeBench (LCB) (Jain et al., 2024), specifically the code generation subset, to assess the model's ability to synthesize functional programs from natural language." (Section 4.3)
- Inference: Capped/Static/Direct are inferred from text-in/text-out benchmark usage plus "it must use a limited context window to achieve low-latency interaction." (Section 5)

### Task: Mathematical reasoning/problem solving
- "For advanced mathematical reasoning, we employ the American Invitational Mathematics Examination (AIME) dataset (Hendrycks et al., 2021), representing a high bar for multi-step problem solving." (Section 4.3)
- "| AIME (Math)                | -25.5% | -15.4%   |" (Table 1; Section 4.3)
- Inference: Capped/Static/Direct are inferred from benchmark framing and the explicit context-window limitation in Section 5.

### Task: Scientific question-answering
- "Finally, we evaluate scientific reasoning using the Diamond subset of GPQA (Rein et al., 2023), a difficult question-answering benchmark designed to be robust against search-engine retrieval." (Section 4.3)
- "| <b>GPQA Diamond (STEM)</b> | -16.3% | -19.5%   |" (Table 1; Section 4.3)
- Inference: Capped/Static/Direct are inferred from benchmark framing and the model’s explicit limited context-window statement in Section 5.
