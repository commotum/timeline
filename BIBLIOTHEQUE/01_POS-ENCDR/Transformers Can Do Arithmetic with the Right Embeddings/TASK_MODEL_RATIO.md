1. Number of distinct tasks evaluated: 5.
Evidence: "We train decoder-only causal language models to solve addition problems." (Section 3 Achieving Length Generalization for Addition). "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1 Addition and Subtraction). "We now study a harder task, multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2 Integer Multiplication). "we now analyze the task of sorting arrays of multiple variable length numbers" (Section 4.3 Array Sorting). "To do this we analyze the bitwise OR task, where the model has to output left aligned position wise OR of two binary vectors." (Section A.3 Bitwise OR on Binary Vectors).
2. Number of trained model instances required to cover all tasks: 4.
Evidence: "We train models on a dataset made up of an even mix of addition and subtraction samples." (Section 4.1 Addition and Subtraction). "We see that these small transformer models can simultaneously learn to extrapolate for both the symmetric operation of addition and the anti-symmetric operation of subtraction using Abacus Embeddings." (Section 4.1 Addition and Subtraction). "We now study a harder task, multiplication of natural numbers, where the length of the output may be the sum of the lengths of the operands." (Section 4.2 Integer Multiplication). "We train with arrays of up to 10 numbers each having up to 10 digits and then evaluate with arrays of up to 30 numbers each having up to 30 digits." (Section 4.3 Array Sorting). "We train standard transformer, standard transformer with input injection and looped transformer models on the position wise or task, on a dataset where the maximum length of either input vector is twenty." (Section A.3 Bitwise OR on Binary Vectors).
3. Task–Model Ratio:
$$
\boxed{
\frac{5\ \text{tasks}}{4\ \text{models}} = 1.25
}
$$
