# LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate (Not specified in the paper.)
Source: LookHere- Vision Transformers with Directed Attention Generalize and Extrapolate.md

## Core reasons
- The paper explicitly critiques current patch position encoding as causing distribution shift when extrapolating to more patches.
- The main contribution is a new position encoding method (LookHere) that constrains attention heads with directional 2D masks to improve extrapolation.

## Evidence extracts
- "We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating." (Abstract)
- "We introduce a novel position encoding method for plain ViTs that restricts attention heads to fixed fields of view (FOV) and points them in different directions via 2D masks." (1 Introduction)

## Classification
Class name: Positional Encoding Improvement Proposal
Class code: 1

$$
\boxed{1}
$$
