# The Perceptron: A Perceiving and Recognizing Automaton (1957)
Source: The perceptron- A perceiving and recognizing automaton.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pattern recognition / discrimination (visual forms) | Optical stimulus patterns as illuminated TV-raster points | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Discrete response label (signal light or printed symbol) | 0D (inferred) | Fixed (inferred) |
| Detection-to-control for camera fixation | Presence/absence and location of peripheral visual stimuli | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | Camera-aiming control response | 0D (inferred) | Fixed (inferred) |

## Summary
The paper primarily describes a momentary visual pattern recognition/discrimination task that maps transformed optical forms to discrete responses. It also describes a separate control task that detects peripheral stimulus location and outputs camera-aiming actions for fixation. From the TV-raster and response-unit interface descriptions, the justified domain is 2D (x, y) input to 0D outputs with Fixed dynamics and Static attention (inferred). The association-value memory mechanism indicates Constructed state (inferred).

## Evidence
### Task: Pattern recognition / discrimination (visual forms)
- "We might consider the perceptron as a black box, with a TV camera for input, and an alphabetic printer or a set of signal lights as output. Its performance can then be described as a process of learning to give the same output signal (or print the same word) for all optical stimuli which belong to some arbitrarily constituted class." (Section II. GENERAL DESCRIPTION OF A PHOTOPERCEPTRON)
- "It is possible to teach the system to discriminate two such generalized forms, or \"percepts\", by presenting for each form a random sample from the set of its possible transformations, while simultaneously \"forcing\" the system to respond with Output 1 for Form 1, and Output 2 for Form 2." (Section II. GENERAL DESCRIPTION OF A PHOTOPERCEPTRON)
- "The A-Units are characterized by the fixed parameter Q, the threshold value which corresponds to the algebraic sum of input pulses necessary to evoke an output, and the stochastic variable v, the \"output value\", which may be any physically measurable characteristic of the output pulse, such as amplitude, frequency, or delay-period. The value of an A-unit's output will vary with its history, and acts as a counter, or register for the memory-function of the system." (Section II. GENERAL DESCRIPTION OF A PHOTOPERCEPTRON)
- Inference: `2D (x, y)` is inferred from "illuminated points in the TV raster"; `Fixed` input/output dynamics are inferred from fixed raster points and predefined response-unit sets; `Static` attention is inferred because no runtime input-selection mechanism is described; `Constructed` state is inferred from A-unit values that "vary with [their] history" and function as memory; `0D` output is inferred from mutually exclusive response labels/signals.

### Task: Detection-to-control for camera fixation
- "By including an independent R-set with feedback to a set of camera-aiming servos, the system can readily be made to train the camera on any forms occuring in peripheral locations in the field, without actually discriminating the particular forms." (Section III. PRINCIPLES OF STIMULUS DISCRIMINATION)
- "For this purpose, it would be necessary only to learn to distinguish the presence or absence of stimuli in different locations in the field, associating the presence of a pattern in the lower left, for example, with a control response moving the camera in this direction." (Section III. PRINCIPLES OF STIMULUS DISCRIMINATION)
- "The system would then be able to \"fixate\" any pattern which might prove significant, in much the same manner as the human eye, limiting its \"recognition learning\" to a relatively limited central field, analogous to the fovea in human vision." (Section III. PRINCIPLES OF STIMULUS DISCRIMINATION)
- Inference: `2D (x, y)` is inferred from location-based detection in the visual field; `Fixed` dynamics are inferred from fixed camera-field and predefined control responses; `Static` attention is inferred because the control mapping is trained over predefined field locations; `Constructed` state is inferred from the same learned association-memory mechanism used by the perceptron; `0D` output is inferred because the control signal is a discrete response choice.
