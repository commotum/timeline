# A Tutorial Introduction to the Minimum Description Length Principle (2004)
Source: A Tutorial Introduction to the Minimum Description Length Principle.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model selection / hypothesis selection | data set D / data sequence x^n | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | selected model or hypothesis | Not specified in the paper. | Not specified in the paper. |
| Data compression / coding | data sequence D | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | compressed description / code for D | Not specified in the paper. | Not specified in the paper. |
| Prediction (sequential/probabilistic) | observed data sequence (past outcomes) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | predictions of future data / predictive distribution | Not specified in the paper. | Not specified in the paper. |
| Density estimation (non-parametric inference) | samples x^n | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | estimated density f (distribution) | Not specified in the paper. | Not specified in the paper. |
| Parameter estimation | data set x^n under model M^(k) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | parameter estimate theta | Not specified in the paper. | Not specified in the paper. |
| Regression | regressor values x_1..x_n and responses y_1..y_n | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | predictor function h: X->Y | Not specified in the paper. | Not specified in the paper. |
| Classification | features X | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | label or class | Not specified in the paper. | Not specified in the paper. |
| Transduction | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Clustering | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Similarity detection | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## Summary
The paper frames MDL as model/hypothesis selection grounded in data compression and gives a predictive (sequential) interpretation. It also discusses MDL applications to parameter estimation, non-parametric density estimation, regression, and classification, and mentions possible extensions to transduction, clustering, and similarity detection. The paper does not specify concrete input/output dimensionality, dynamics, attention, or state mechanisms for these tasks.

## Evidence
### Task: Model selection / hypothesis selection
- "How does one decide among competing explanations of data given limited observations? This is the problem of model selection." (Section 1.1 Introduction and Overview)
- "The Minimum Description Length (MDL) Principle is a relatively recent method for inductive inference that provides a generic solution to the model selection problem." (Section 1.1 Introduction and Overview)

### Task: Data compression / coding
- "any regularity in the data can be used to compress the data" (Section 1.1 Introduction and Overview)
- "*viewing learning as data compression*" (Section 1.3 MDL: The Basic Idea)

### Task: Prediction (sequential/probabilistic)
- "data compression is formally equivalent to a form of probabilistic prediction" (Section 1.1 Introduction and Overview)
- "selecting the model with the best predictive performance when sequentially predicting *unseen* test data" (Section 2.6.4 Prequential interpretation)

### Task: Density estimation (non-parametric inference)
- "We may still try to learn a distribution from  $\mathcal{M}$  in various ways, for example by histogram density estimation" (Section 2.8 Beyond Parametric Model Selection)
- "select a density f from a class  $\mathcal{M}^{(n)} \subset \mathcal{M}$" (Section 2.8 Beyond Parametric Model Selection)

### Task: Parameter estimation
- "The 'crude' MDL method (Section 2.4) was a means of doing model selection and parameter estimation at the same time." (Section 2.8 Beyond Parametric Model Selection)
- "parameter estimates are needed, they may be obtained in three different ways." (Section 2.8 Beyond Parametric Model Selection)

### Task: Regression
- "learning how the values  $y_1, \ldots, y_n$  of a regression variable Y depend on the values  $x_1, \ldots, x_n$" (Section 2.8 Beyond Parametric Model Selection)
- "we want to learn such an h from data." (Section 2.8 Beyond Parametric Model Selection)

### Task: Classification
- "Examples are applications of MDL in *classification* and regression." (Section 2.8 Beyond Parametric Model Selection)
- "the goal is to match each feature X (for example, a bit map of a handwritten digit) with its corresponding label or class" (Section 2.8 Beyond Parametric Model Selection)

### Task: Transduction
- "we mention *prediction*, *transduction* (as defined in [Vapnik 1998]), *clustering*" (Section 2.8 Beyond Parametric Model Selection)

### Task: Clustering
- "we mention *prediction*, *transduction* (as defined in [Vapnik 1998]), *clustering*" (Section 2.8 Beyond Parametric Model Selection)

### Task: Similarity detection
- "and *similarity detection*" (Section 2.8 Beyond Parametric Model Selection)
