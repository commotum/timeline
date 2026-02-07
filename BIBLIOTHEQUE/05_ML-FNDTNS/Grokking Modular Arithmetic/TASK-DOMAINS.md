# Grokking modular arithmetic (Not specified in the paper.)
Source: Grokking Modular Arithmetic.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Modular addition classification | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |
| Modular separable-sum function classification (`f1(n)+f2(m)`) | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |
| Modular transformed-sum function classification (`F(f1(n)+f2(m))`) | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |
| Modular multiplicative function classification (`g1(n)*g2(m)`) | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |
| Modular additive+multiplicative function classification (`f(n,m)+g(n,m)`) | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |
| Modular polynomial function classification (`n^3+nm^2+m`) | Two integers in Z_p encoded as one-hot vectors | 1D (t) (inferred) | Fixed (inferred) | Static (inferred) | Direct (inferred) | Integer in Z_p encoded as a one-hot vector | 1D (t) (inferred) | Fixed (inferred) |

## Summary
The paper studies bivariate modular arithmetic classification tasks over Z_p, with inputs and outputs represented as one-hot vectors of fixed size. Covered tasks include modular addition, separable-sum and transformed-sum functions, multiplicative functions, and mixed polynomial combinations (with some tasks harder or failing to generalize). Based on the fixed-size one-hot interface and two-layer MLP, the tasks use 1D fixed vectors with static attention and direct state (inferred).

## Evidence
### Task: Modular addition classification
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "f(n,m) = n + m \bmod p." (Section 3.1 Modular addition)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)

### Task: Modular separable-sum function classification (`f1(n)+f2(m)`)
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "for any function of the form  f(n,m) = f_1(n) + f_2(m) \mod p" (Section 2 Set up and overview of results)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)

### Task: Modular transformed-sum function classification (`F(f1(n)+f2(m))`)
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "a more general modular task  \tilde{f}(n,m) = F(f_1(n) + f_2(m)) \mod p" (Section 3.2 General modular functions and complexity)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)

### Task: Modular multiplicative function classification (`g1(n)*g2(m)`)
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "Functions of the form  g(n,m) = g_1(n) \cdot g_2(m) \mod p  can also be grokked" (Section 2 Set up and overview of results)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)

### Task: Modular additive+multiplicative function classification (`f(n,m)+g(n,m)`)
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "Functions of the form  f(n,m) + g(n,m) \mod p  are more difficult to grok" (Section 2 Set up and overview of results)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)

### Task: Modular polynomial function classification (`n^3+nm^2+m`)
- "Given this architecture, we then set up modular arithmetic tasks as classification problems." (Section 2 Set up and overview of results)
- "f(n,m) = n^3 + nm^2 + m" (Appendix C Some other modular functions)
- Inference: In/Out treated as fixed 1D vectors and attention/state as Static/Direct because "Each input integer is encoded as a one-hot vector.", "the input dimension is 2p, the output dimension is p", and "We consider a two-layer MLP network without biases". (Section 2 Set up and overview of results)
