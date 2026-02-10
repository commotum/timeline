1. **Number of distinct tasks evaluated:** 3

- **Section 4.1. Cart-Pole Balancina:** “Cart-Pole Balancing is a well-known benchmark for reinforcement learning.”
- **Section 4.2. Motor primitive learning for baseball:** “In Fig. 4, we show a comparison with GPOMDP for simple, single DOF task with a reward of”
- **Section 4.2. Motor primitive learning for baseball:** “We also evaluated the same setup in a challenging robot task, i.e., the planning of these motor primitives for a seven DOF robot task. The task of the robot is to hit the ball properly so that it flies as far as possible.”

2. **Number of trained model instances required to cover all tasks:** 3

- **Section 4.1. Cart-Pole Balancina:** “The policy is specified as  $\pi(\mathbf{u}|\mathbf{x}) = \mathcal{N}(\mathbf{K}\mathbf{x}, \sigma^2)$ .”
- **Section 4.1. Cart-Pole Balancina:** “Thus, the policy parameter vector becomes  $\boldsymbol{\theta} = [\mathbf{K}^T, \eta]^T$  and has the analytically computable optimal solution  $\mathbf{K} \approx [5.71, 11.3, -82.1, -21.6]^T$ , and  $\sigma = 0.1$ , corresponding to  $\eta \to \infty$ .”
- **Section 4.2. Motor primitive learning for baseball:** “where  $(q_{d,k}, \dot{q}_{d,k})$  denote the desired position and velocity of a joint,  $z_k$  the internal state of the dynamic system,  $g_k$  the goal (or point attractor) state of each DOF,  $\tau$  the movement duration shared by all DOFs, and  $\theta_k$  the open parameters of the function h.”
- **Section 4.2. Motor primitive learning for baseball:** “We also evaluated the same setup in a challenging robot task, i.e., the planning of these motor primitives for a seven DOF robot task.”
- A single jointly trained model instance that performs all tasks is **Not specified in the paper.**

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
