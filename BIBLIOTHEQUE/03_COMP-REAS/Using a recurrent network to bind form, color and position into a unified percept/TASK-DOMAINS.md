# Using a recurrent network to bind form, color and position into a unified percept (2001)
Source: Using a recurrent network to bind form, color and position into a unified percept.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Form/object classification | Visual form patterns in V1 orientation layers across object locations | 2D (x, y) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Object/form identity at AIT | 0D (inferred) | Fixed (inferred) |
| Position selection from object identity (form-to-location binding) | Selected form node in AIT plus feedforward visual activity across locations | 0D; 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Selected object location in the visual field | 2D (x, y) (inferred) | Fixed (inferred) |
| Color retrieval for a queried form (form-color binding) | Multi-object visual scene with form and color, plus selected form node in AIT | 0D; 2D (x, y) (inferred) | Fixed (inferred) | Dynamic (inferred) | Constructed (inferred) | Color identity for the queried object at AIT | 0D (inferred) | Fixed (inferred) |

## Summary
The paper describes a visual processing model that performs form/object identification, selects object position from form identity using feedback, and uses that selection to retrieve a bound color attribute. The modeled data domain is primarily a 2D retinotopic visual field, with categorical form/color decisions at AIT as 0D outputs. Based on the described architecture (fixed layers, fixed numbers of locations/feature channels), the interface dynamics are fixed. Attention behavior ranges from static feedforward recognition to dynamic object-based selection with feedback; state is direct for simple recognition and constructed for feedback-based binding operations.

## Evidence
### Task: Form/object classification
- "Objects were presented at the 'V1' layer, object recognition occurred at the 'AIT' layer." (Section 3)
- "Four different objects: 'square', 'diamond', 'horizontal cross' and 'diagonal cross' could be presented at four locations in the 'V1' layer, constituting a total of 16 input patterns. After training, the network correctly identified the objects." (Section 3)
- Inference: Dimension is labeled 2D (x, y) from "four locations in the 'V1' layer"; dynamics are Fixed from the fixed "four locations" and finite "16 input patterns". Attention is labeled Static because this step is described as feedforward object recognition without runtime selection. State is labeled Direct because this task is a direct mapping from presented pattern to identity in the described feedforward pass. (Section 3)

### Task: Position selection from object identity (form-to-location binding)
- "It is a clear indication that the monkey was able to find the location of the target object in the visual field based on the target object's identity (form) alone." (Section 3)
- "In AIT the node corresponding to one of the input patterns is selected and activity is fed into the feedback network (Fig. 1C)." (Section 3)
- "As can be seen in Fig. 1D one group of activities clearly stands out, revealing the location of the object of interest." (Section 3)
- Inference: Input dimension is 0D; 2D (x, y) because a categorical AIT node is selected (identity cue) and activity is evaluated over spatial locations in lower visual areas. Attention is Dynamic because selection depends on which AIT node is chosen at runtime and the resulting local consistency result. State is Constructed because the effective decision state is formed by combining feedforward and feedback activity to produce a selected location. Dynamics are Fixed because the location space is the fixed model visual field. (Section 3)

### Task: Color retrieval for a queried form (form-color binding)
- "The network is trained to identify color and form of the presented objects at the level of AIT." (Section 4)
- "The answer is produced by first selecting the form node for 'cross' at the level of AIT. Then, feedback activity, corresponding to 'cross' is sent to lower areas of the visual cortex." (Section 4)
- "This would then result in the selection of the color node for 'red' at the level of AIT, which represents the color of the cross." (Section 4)
- Inference: Input dimension is 0D; 2D (x, y) because the process uses a selected form category plus spatially distributed scene activity; output dimension is 0D because the result is a selected color category node. Attention is Dynamic and State is Constructed due the multi-step feedback selection and reprocessing pipeline. Dynamics are Fixed because the architecture is described with fixed orientation/color layers and fixed visual-field locations. (Section 4)
