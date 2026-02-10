# A History of Meta-gradient: Gradient Methods for Meta-learning (2022)
Source: A History of Meta-gradient- Gradient Methods for Meta-learning.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Supervised learning (classification/regression) | training examples (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | labels or continuous targets (inferred) | 0D (inferred) | Not specified in the paper. |
| Independent component analysis | signals (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | independent components (inferred) | Not specified in the paper. | Not specified in the paper. |
| Visual hand and body tracking | visual observations (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | tracked hand/body trajectories (inferred) | 3D (x, y, t) (inferred) | Not specified in the paper. |
| Turbulent flow modeling | turbulent-flow observations (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | flow predictions/models (inferred) | Not specified in the paper. | Not specified in the paper. |
| Brain-computer interfacing | neural interface signals (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | interface commands or predictions (inferred) | Not specified in the paper. | Not specified in the paper. |
| Natural language processing | tokens (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Reinforcement learning prediction | interaction trajectories (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | value predictions (inferred) | 1D (t) (inferred) | Not specified in the paper. |
| Reinforcement learning control | interaction trajectories (inferred) | 1D (t) (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | actions/policies (inferred) | 1D (t) (inferred) | Not specified in the paper. |

## Summary
The OCR text describes meta-gradient methods applied across supervised learning and multiple downstream domains, including independent component analysis, visual tracking, turbulent-flow modeling, brain-computer interfacing, natural language processing, and reinforcement learning. Most modality/interface details are not explicitly specified in this paper-level historical overview, so many fields remain "Not specified in the paper." The strongest supported structure is temporal/sequential usage in reinforcement-learning and language-processing mentions (1D (t), inferred), plus 0D outputs for classification/regression (inferred). Attention and state dynamics are not explicitly characterized in the OCR text.

## Evidence
### Task: Supervised learning (classification/regression)
- "The meta-parameter learned in all of the earliest meta-gradient methods was the step size or \"learning rate\" of supervised learning systems." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "A limitation of Delta-Bar-Delta is that it can be applied only to batch training, not incrementally to individual training examples, as in *stochastic* gradient descent." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "Koop (2008) developed a form of IDBD for logistic rather than linear functions such that it is suited to classification rather than regression." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- Inference: "training examples" input and "labels or continuous targets" with 0D output are inferred from explicit supervised-learning, classification, and regression wording.

### Task: Independent component analysis
- "Schraudolph and colleagues applied his method, called Stochastic Meta Descent, or SMD, to good effect in many problem areas, including independent component analysis" (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "All of the above meta-gradient methods treated *step-size* meta-parameters, but it was always clear that the ideas were more general." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- Inference: the input/output conceptual objects (signals/components) are inferred from the task name "independent component analysis"; dimensions/dynamics are not explicitly stated.

### Task: Visual hand and body tracking
- "Schraudolph and colleagues applied his method, called Stochastic Meta Descent, or SMD, to good effect in many problem areas, including independent component analysis (Schraudolph & Giannakopoulos 2000), visual hand and body tracking (Bray et al. 2005, 2007a, 2007b; Kehl & Van Gool 2006), conditional random fields (Vishwanathan et al. 2006a), and support vector machines in reproducing kernel Hilbert spaces (Karatzoglou et al. 2005; Vishwanathan et al. 2006b; Günter et al. 2007; see also He 2009)." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "Kehl, R., Van Gool, L. (2006). Markerless tracking of complex human motions from multiple views." (Section: References)
- Inference: visual observations and spatiotemporal trajectory structure (3D (x, y, t)) are inferred from explicit tracking language.

### Task: Turbulent flow modeling
- "Others independently applied SMD successfully in modeling turbulent flow" (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "Milano, M., Koumoutsakos, P. (2002). Neural network modeling for near wall turbulent flow." (Section: References)
- Inference: the task is treated as a modeling/prediction domain; concrete input/output dimensions and dynamics are not explicitly provided.

### Task: Brain-computer interfacing
- "Others independently applied SMD successfully in modeling turbulent flow (Milano & Koumoutsakos 2002), in brain computer interfaces (Buttfield, Ferrez & Millán 2006; Millán et al. 2007), in learning in recurrent neural networks better than real-time recurrent learning (Liu & Elhanany 2007, 2008; Liu 2007), and in natural language processing (Arun et al. 2009)." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "Millán, J. D. R., Buttfield, A., Vidaurre, C., Cabeza, R., Krauledat, M., Schlögl, A., Shenoy, P., Blankertz, B., Rao, R. P. N., Cabeza, R., Pfurtscheller, G., Müller, K.-R. (2007). Adaptation in brain-computer interfaces." (Section: References)
- Inference: neural interface signals and corresponding outputs are inferred from explicit brain-computer interface wording; interface details are not specified in the paper text.

### Task: Natural language processing
- "Others independently applied SMD successfully in modeling turbulent flow (Milano & Koumoutsakos 2002), in brain computer interfaces (Buttfield, Ferrez & Millán 2006; Millán et al. 2007), in learning in recurrent neural networks better than real-time recurrent learning (Liu & Elhanany 2007, 2008; Liu 2007), and in natural language processing (Arun et al. 2009)." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "Arun, A., Dyer, C., Haddow, B., Blunsom, P., Lopez, A., Koehn, P. (2009). Monte Carlo inference and maximization for phrase-based translation." (Section: References)
- Inference: token-based 1D (t) input is inferred from natural-language-processing framing; the specific output type is not stated in the OCR main text.

### Task: Reinforcement learning prediction
- "Particularly ambitious is recent work by Veeriah et al. (2019) that attempts to use meta-gradients to specify the \"question parameters\" of general value functions used as auxiliary tasks within deep reinforcement learning." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "TIDBD was applied to a variety of robotic prediction and control problems by Günther and others (2019a, 2019b)." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- Inference: interaction trajectories and 1D (t) temporal structure, with value-prediction outputs, are inferred from temporal-difference/value-function wording.

### Task: Reinforcement learning control
- "Meta-gradient methods have also been used in reinforcement learning. The first may have been that by Yu, Aberdeen, and Schraudolph (2006), who used SMD with policy-gradient reinforcement learning methods." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- "TIDBD was applied to a variety of robotic prediction and control problems by Günther and others (2019a, 2019b)." (Section: A History of Meta-gradient: Gradient Methods for Meta-learning)
- Inference: control outputs as actions/policies and 1D (t) temporal interaction inputs are inferred from policy-gradient and robotic-control language.
