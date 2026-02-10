1. **Number of distinct tasks evaluated:** 6

   Verbatim evidence:
   - "To show that DeepProbLog supports both logical reasoning and deep learning, we extend the classic learning task on the MNIST dataset (Lecun et al. [1998]) to two more complex problems that require reasoning:" followed by "- T1:" and "- T2:" (Section 6, **Logical reasoning and deep learning**).
   - "As in their work, we consider three tasks: addition, sorting [Reed and de Freitas, 2016] and word algebra problems (WAPs) [Roy and Roth, 2015]." followed by "- T3:", "- **T4:**", and "- T5:" (Section 6, **Program Induction**).
   - "Task T6 is thus to learn the game/4 predicate" (Section 6, **Probabilistic programming and deep learning**).

2. **Number of trained model instances required to cover all tasks:** 6

   Verbatim evidence:
   - "Listing 1: Single-digit MNIST addition (**T1**)"; "Listing 2: Multi-digit MNIST addition (**T2**)"; "Listing 3: Forth addition sketch (T3)"; "Listing 4: Forth sorting sketch (**T4**)"; "Listing 5: Forth WAP sketch (**T5**)"; "Listing 6: The coin-ball problem (**T6**)" (Appendix A, **DeepProbLog Programs**).
   - "where holes in given programs need to be filled by neural networks trained on input-output examples for the entire program" (Section 6, **Program Induction**).
   - "We simultaneously train one neural network to classify an image of the coin as being heads or tails (coin/2), and a neural network to classify the colour of the ball as being either red, blue or green (colour/4)." (Section 6, **Probabilistic programming and deep learning**).

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$
