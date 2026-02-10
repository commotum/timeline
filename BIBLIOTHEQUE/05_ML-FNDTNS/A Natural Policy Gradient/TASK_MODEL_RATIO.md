1. **Number of distinct tasks evaluated:** 3

- "We simulated the natural policy gradient in a simple 1-dimensional linear quadratic regulator with dynamics  $x(t+1) = .7x(t) + u(t) + \epsilon(t)$  and noise distribution  $\epsilon \sim G(0,1)$ ." (Section 5 Experiments)
- "The effects of the weighting by  $\rho(s)$  are particularly clear in a simple 2-state MDP (Figure 1B), which has self- and cross-transition actions and rewards as shown." (Section 5 Experiments)
- "The game of Tetris provides a challenging high dimensional problem." (Section 5 Experiments)

2. **Number of trained model instances required to cover all tasks:** 3

- "The parameterized policy used was  $\pi(u;x,\theta) \propto \exp(\theta_1 x^2 + \theta_2 x)$ ." (Section 5 Experiments)
- "C top) The average reward vs. time (on a  $10^7$  scale) of a policy under standard gradient descent using the sigmoidal policy parameterization ( $\pi(1; s, \theta_i) \propto \exp(\theta_i)/(1 + \exp(\theta_i))$ ), with the initial conditions  $\pi(i,1) = .8$  and  $\pi(j,1) = .1$ ." (Figure 1 caption, Section 5 Experiments)
- "We consider a policy compatible with the linear function approximator used in [3] (ie  $\pi(a;s,\theta) \propto \exp(\theta^T \phi_{sa})$  where  $\phi_{sa}$  are the same feature vectors)." (Section 5 Experiments)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
