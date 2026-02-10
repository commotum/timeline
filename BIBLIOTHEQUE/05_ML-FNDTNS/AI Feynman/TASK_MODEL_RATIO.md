1. **Number of distinct tasks evaluated**: **N** mystery-level symbolic regression tasks.

   Verbatim evidence:
   - "its task is to predict *f* for each mystery taking the data table (and optionally the unit table) as input." (Section: **The Feynman Symbolic Regression Database**)
   - "the database is generated using 100 equations" (Section: **The Feynman Symbolic Regression Database**)
   - "a set of 20 more challenging \"bonus\" equations" (Section: **The Feynman Symbolic Regression Database**)
   - "we tested the performance of our algorithm on the mystery functions presented in (41)" (Section: **Results**, after Table 6)

   Exact total number of mysteries including the (41) set: **Not specified in the paper.**

2. **Number of trained model instances required to cover all tasks**: **N** models (one trained neural-network instance per mystery task when NN-based steps are used).

   Verbatim evidence:
   - "To obtain such an interpolating function for a given mystery, we train a neural network" (Section: **Neural network training**)
   - "For each mystery, we generated 100,000 data points... training for 100 epochs" (Section: **Neural network training**)

3. **Task–Model Ratio = (1) / (2)**:

$$
\boxed{
\frac{N\ \text{tasks}}{N\ \text{models}} = 1
}
$$
