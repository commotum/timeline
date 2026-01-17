# Solving the Rubik's Cube Without Human Knowledge

# Stephen McAleer\*

Department of Statistics University of California, Irvine smcaleer@uci.edu

#### Alexander Shmakov\*

Department of Computer Science University of California, Irvine ashmakov@uci.edu

# Forest Agostinelli\*

Department of Computer Science University of California, Irvine fagostin@uci.edu

#### Pierre Baldi

Department of Computer Science University of California, Irvine pfbaldi@ics.uci.edu

# **Abstract**

A generally intelligent agent must be able to teach itself how to solve problems in complex domains with minimal human supervision. Recently, deep reinforcement learning algorithms combined with self-play have achieved superhuman proficiency in Go, Chess, and Shogi without human data or domain knowledge. In these environments, a reward is always received at the end of the game; however, for many combinatorial optimization environments, rewards are sparse and episodes are not guaranteed to terminate. We introduce Autodidactic Iteration: a novel reinforcement learning algorithm that is able to teach itself how to solve the Rubik's Cube with no human assistance. Our algorithm is able to solve 100% of randomly scrambled cubes while achieving a median solve length of 30 moves—less than or equal to solvers that employ human domain knowledge.

# 1 Introduction

Reinforcement learning seeks to create intelligent agents that adapt to an environment by analyzing their own experiences. Reinforcement learning agents have achieved superhuman capability in a number of competitive games [35, 20, 31, 29]. This recent success of reinforcement learning has been a product of the combination of classic reinforcement learning [3, 33, 38, 34], deep learning [17, 10], and Monte Carlo Tree Search (MCTS) [8, 14, 9]. The fusion of reinforcement learning and deep learning is known as deep reinforcement learning (DRL). Though DRL has been successful in many different domains, it relies heavily on the condition that an informatory reward can be obtained from an initially random policy. Current DRL algorithms struggle in environments with a high number of states and a small number of reward states such as *Sokoban* and *Montezuma's Revenge*. Other environments, such as short answer exam problems, prime number factorization, and combination puzzles like the Rubik's Cube have a large state space and only a single reward state.

The 3x3x3 Rubik's cube is a classic 3-Dimensional combination puzzle. There are 6 faces, or 3x3x1 planes, which can be rotated  $90^{\circ}$  in either direction. The goal state is reached when all stickers on each face of the cube are the same color, as shown in Figures 2a and 2b. The Rubik's cube has a large state space, with approximately  $4.3 \times 10^{19}$  different possible configurations. However, out of all of these configurations, only one state has a reward signal: the goal state. Therefore, starting from

<sup>\*</sup>Equal contribution

![](_page_1_Figure_0.jpeg)

Figure 1: An illustration of DeepCube. The training and solving process is split up into ADI and MCTS. First, we iteratively train a DNN by estimating the true value of the input states using breadth-first search. Then, using the DNN to guide exploration, we solve cubes using Monte Carlo Tree Search. See methods section for more details.

random states and applying DRL algorithms, such as asynchronous advantage actor-critic (A3C) [19], could theoretically result in the agent never solving the cube and never receiving a learning signal.

To overcome a sparse reward in a model-based environment with a large state space, we introduce Autodidactic Iteration (ADI): an algorithm inspired by policy iteration [3, 34] for training a joint value and policy network. ADI trains the value function through an iterative supervised learning process. In each iteration, the inputs to the neural network are created by starting from the goal state and randomly taking actions. The targets seek to estimate the optimal value function by performing a breadth-first search from each input state and using the current network to estimate the value of each of the leaves in the tree. Updated value estimates for the root nodes are obtained by recursively backing up the values for each node using a max operator. The policy network is similarly trained by constructing targets from the move that maximizes the value. After the network is trained, it is combined with MCTS to efficiently solve the Rubik's Cube. We call the resulting solver DeepCube.

#### 2 Related work

Erno Rubik created the Rubik's Cube in 1974. After a month of effort, he came up with the first algorithm to solve the cube. Since then, the Rubik's Cube has gained worldwide popularity and many human-oriented algorithms for solving it have been discovered [1]. These algorithms are simple to memorize and teach humans how to solve the cube in a structured, step-by-step manner.

Human-oriented algorithms to solve the Rubik's Cube, while easy to memorize, find long suboptimal solutions. Since 1981, there has been theoretical work on finding the upper bound for the number of moves necessary to solve any valid cube configuration [36, 23, 22, 16, 24]. Finally, in 2014, it was shown that any valid cube can be optimally solved with at most 26 moves in the quarter-turn metric, or 20 moves in the half-turn metric [26, 25]. The quarter-turn metric treats 180 degree rotations as two moves, whereas the half-turn metric treats 180 degree rotations as one move. For the remainder of this paper we will be using the quarter-turn metric. This upper bound on the number of moves required to solve the Rubik's cube is colloquially known as God's Number.

Algorithms used by machines to solve the Rubik's Cube rely on hand-engineered features and group theory to systematically find solutions. One popular solver for the Rubik's Cube is the Kociemba two-stage solver [13]. This algorithm uses the Rubik's Cube's group properties to first maneuver the cube to a smaller sub-group, after which finding the solution is trivial. Heuristic based search algorithms have also been employed to find optimal solutions. Korf first used a variation of the A\* heuristic search, along with a pattern database heuristic, to find the shortest possible solutions [15]. More recently, there has been an attempt to train a DNN – using supervised learning with hand-engineered features – to act as an alternative heuristic [7]. These search algorithms, however, take an extraordinarily long time to run and usually fail to find a solution to randomly scrambled cubes within reasonable time constraints. Besides hand-crafted algorithms, attempts have been made to solve the Rubik's Cube through evolutionary algorithms [32, 18]. However, these learned solvers can only reliably solve cubes that are up to 5 moves away from the solution.

![](_page_2_Picture_0.jpeg)

Figure 2: Visualizations of the Rubik's Cube. (a) and (b) show the solved cube as it appears in the environment. (c) and (d) show the cube reduced in dimensionality for input into the DNN. Stickers that are used by the DNN are white, whereas ignored stickers are dark.

We solve the Rubik's Cube using pure reinforcement learning without human knowledge. In order to solve the Rubik's Cube using reinforcement learning, the algorithm will learn a *policy*. The policy determines which move to take in any given state. Much work has already been done on finding optimal policy functions in classic reinforcement learning [33, 38, 39, 21]. Autodidactic Iteration can be seen as a type of policy iteration algorithm [34]. Since our implementation of ADI uses a depth-1 greedy policy when training the network, it can also be thought of as value iteration. Policy iteration is an iterative reinforcement learning algorithm which alternates between a policy evaluation step and a policy improvement step. The policy evaluation step estimates the state-value function given the current policy, while the policy improvement step uses the estimated state-value function to improve the policy. Though many policy iteration algorithms require multiple sweeps through the state space for each step, value iteration effectively combines policy iteration and policy improvement into a single sweep. Other works have used neural networks to augment MCTS [11, 28, 2]. Our approach is most similar to AlphaGo Zero [30] and Expert Iteration [2], which also do not use human knowledge.

# 3 The Rubik's Cube

The Rubik's Cube consists of 26 smaller cubes called cubelets. These are classified by their sticker count: center, edge, and corner cubelets have 1, 2, and 3 stickers attached respectively. There are 54 stickers in total with each sticker uniquely identifiable based on the type of cubelet the sticker is on and the other sticker(s) on the cubelet. Therefore, we may use a one-hot encoding for each of the 54 stickers to represent their location on the cube.

However, because the position of one sticker on a cubelet determines the position of the remaining stickers on that cubelet, we may actually reduce the dimensionality of our representation by focusing on the position of only one sticker per cubelet. We ignore the redundant center cubelets and only store the 24 possible locations for the edge and corner cubelets. This results in a 20x24 state representation which is demonstrated in Figures 2c and 2d.

Moves are represented using face notation originally developed by David Singmaster: a move is a letter stating which face to rotate. F, B, L, R, U, and D correspond to turning the front, back, left, right, up, and down faces, respectively. A clockwise rotation is represented with a single letter, whereas a letter followed by an apostrophe represents a counter-clockwise rotation. For example: R and R' would mean to rotate the right face  $90^{\circ}$  clockwise and counter-clockwise, respectively.

Formally, the Rubik's Cube environment consists of a set of  $4.3 \times 10^{19}$  states  $\mathcal S$  which contains one special state,  $s_{solved}$ , representing the goal state. At each timestep, t, the agent observes a state  $s_t \in \mathcal S$  and takes an action  $a_t \in \mathcal A$  with  $\mathcal A := \{F, F', \ldots, D, D'\}$ . After selecting an action, the agent observes a new state  $s_{t+1} = A(s_t, a_t)$  and receives a scalar reward,  $R(s_{t+1})$ , which is 1 if  $s_{t+1}$  is the goal state and -1 otherwise.

![](_page_3_Figure_0.jpeg)

Figure 3: Visualization of training set generation in ADI. (a) Generate a sequence of training inputs starting from the solved state. (b) For each training input, generate its children and evaluate the value network on all of them. (c) Set the value and policy targets based on the maximal estimated value of the children.

#### 4 Methods

We develop a novel algorithm called Autodidactic Iteration which is used to train a joint value and policy network. Once the network is trained, it is combined with MCTS to solve randomly scrambled cubes. The resulting solver is called DeepCube.

# 4.1 Autodidactic Iteration

ADI is an iterative supervised learning procedure which trains a deep neural network  $f_{\theta}(s)$  with parameters  $\theta$  which takes an input state s and outputs a value and policy pair (v, p). The policy output p is a vector containing the move probabilities for each of the 12 possible moves from that state. Once the network is trained, the policy is used to reduce breadth and the value is used to reduce depth in the MCTS. In each iteration of ADI, training samples for  $f_{\theta}$  are generated by starting from the solved cube. This ensures that some training inputs will be close enough to have a positive reward when performing a shallow search. Targets are then created by performing a depth-1 breadth-first search (BFS) from each training sample. The current value network is used to estimate each child's value. The value target for each sample is the maximum value and reward of each of its children, and the policy target is the action which led to this maximal value. Figure 3 displays a visual overview of ADI.

Formally, we generate training samples by starting with  $s_{solved}$  and scrambling the cube k times to generate a sequence of k cubes. We do this l times to generate N=k\*l training samples  $X=[x_i]_{i=1}^N$ . Each sample in the series has the number of scrambles it took to generate it,  $D(x_i)$ , associated with it. Then, for each sample  $x_i \in X$ , we generate a training target  $Y_i=(y_{v_i}, \boldsymbol{y}_{p_i})$ . To do this, we perform a depth-1 BFS to get the set of all children states of  $x_i$ . We then evaluate the current neural network at each of the children states to receive estimates of each child's optimal value and policy:  $\forall a \in \mathcal{A}, \ (v_{x_i}(a), \boldsymbol{p}_{x_i}(a)) = f_{\theta}(A(x_i, a))$ . We set the value target to be the maximal value from each of the children  $y_{v_i} \leftarrow \max_a(R(A(x_i, a)) + v_{x_i}(a))$ , and we set the policy target to be the move that results in the maximal estimated value  $\boldsymbol{y}_{p_i} \leftarrow \operatorname{argmax}_a(R(A(x_i, a)) + v_{x_i}(a))$ . We then train  $f_{\theta}$  on these training samples and targets  $[x_i, y_{v_i}]_{i=1}^N$  and  $[x_i, \boldsymbol{y}_{p_i}]_{i=1}^N$  to receive new neural network parameters  $\theta'$ . For training, we used the RMSProp optimizer [12] with a mean squared error

#### Algorithm 1: Autodidactic Iteration

**Initialization:**  $\theta$  initialized using Glorot initialization repeat

```
X \leftarrow \text{N scrambled cubes}
\text{for } x_i \in X \text{ do}
\text{for } a \in \mathcal{A} \text{ do}
```

loss for the value and softmax cross entropy loss for the policy. Although we use a depth-1 BFS for training, this process may be trivially generalized to perform deeper searches at each  $x_i$ .

#### Weighted samples

During testing, we found that the learning algorithm sometimes either converged to a degenerate solution or diverged completely. To counteract this, we assigned a higher training weight to samples that are closer to the solved cube compared to samples that are further away from solution. We assign a loss weight to each sample  $x_i$ ,  $W(x_i) = \frac{1}{D(x_i)}$ . We didn't see divergent behavior after this addition.

#### 4.2 Solver

We employ an asynchronous Monte Carlo Tree Search augmented with our trained neural network  $f_{\theta}$  to solve the cube from a given starting state  $s_0$ . We build a search tree iteratively by beginning with a tree consisting only of our starting state,  $T = \{s_0\}$ . We then perform simulated traversals until reaching a leaf node of T. Each state,  $s \in T$ , has a memory attached to it storing:  $N_s(a)$ , the number of times an action a has been taken from state s,  $W_s(a)$ , the maximal value of action a from state s,  $L_s(a)$ , the current virtual loss for action a from state s, and  $P_s(a)$ , the prior probability of action a from state s.

Every simulation starts from the root node and iteratively selects actions by following a tree policy until an unexpanded leaf node,  $s_{\tau}$ , is reached. The tree policy proceeds as follows: for each timestep t, an action is selected by choosing,  $A_t = \operatorname{argmax}_a U_{s_t}(a) + Q_{s_t}(a)$  where  $U_{s_t}(a) = c P_{s_t}(a) \sqrt{\sum_{a'} N_{s_t}(a')}/(1 + N_{s_t}(a))$ , and  $Q_{s_t}(a) = W_{s_t}(a) - L_{s_t}(a)$  with an exploration hyperparameter c. The virtual loss is also updated  $L_{s_t}(A_t) \leftarrow L_{s_t}(A_t) + \nu$  using a virtual loss hyperparameter  $\nu$ . The virtual loss prevents the tree search from visiting the same state more than once and discourages the asynchronous workers from all following the same path [27].

Once a leaf node,  $s_{\tau}$ , is reached, the state is expanded by adding the children of  $s_{\tau}$ ,  $\{A(s_{\tau}, a), \forall a \in \mathcal{A}\}$ , to the tree T. Then, for each child s', the memories are initialized:  $W_{s'}(\cdot) = 0$ ,  $N_{s'}(\cdot) = 0$ ,  $P_{s'}(\cdot) = p_{s'}$ , and  $L_{s'}(\cdot) = 0$ , where  $p_{s'}$  is the policy output of the network  $f_{\theta}(s')$ . Next, the value and policy are computed:  $(v_{s_{\tau}}, p_{s_{\tau}}) = f_{\theta}(s_{\tau})$  and the value is backed up on all visited states in the simulated path. For  $0 \le t \le \tau$ , the memories are updated:  $W_{s_t}(A_t) \leftarrow \max{(W_{s_t}(A_t), v_{s_{\tau}})}, N_{s_t}(A_t) \leftarrow N_{s_t}(A_t) + 1, L_{s_t}(A_t) \leftarrow L_{s_t}(A_t) - \nu$ . Note that, unlike other implementations of MCTS, only the maximal value encountered along the tree is stored, and not the total value. This is because the Rubik's Cube is deterministic and not adversarial, so we do not need to average our reward when deciding a move.

The simulation is performed until either  $s_{\tau}$  is the solved state or the simulation exceeds a fixed maximum computation time. If  $s_{\tau}$  is the solved state, then the tree T of the simulation is extracted and converted into an undirected graph with unit weights. A full breath-first search is then applied on T to find the shortest predicted path from the starting state to solution. Alternatively, the last sequence  $Path = \{A_t | 0 \le t \le \tau\}$  may be used, but this produces longer solutions.

![](_page_5_Figure_0.jpeg)

(a) Comparison of solve rates among different solvers. (b) DeepCube's Distribution of solve times for the 640 Each solver was given 30 minutes to compute a solu-fully scrambled cubes. All cubes were solved within tion for 50 cubes.

the 60 minute limit.

Figure 5

#### 5 Results

We used a feed forward network as the architecture for  $f_{\theta}$  as shown in Figure 4. The outputs of the network are a 1 dimensional scalar v, representing the value, and a 12 dimensional vector p, representing the probability of selecting each of the possible moves. The network was then trained using ADI for 2,000,000 iterations. The network witnessed approximately 8 billion cubes, including repeats, and it trained for a period of 44 hours. Our training machine was a 32-core Intel Xeon E5-2620 server with three NVIDIA Titan XP GPUs.

As a baseline, we compare DeepCube against two other solvers. The first baseline is the **Kociemba** two-stage solver [13, 37]. This algorithm relies on human domain knowledge of the group theory of the Rubik's Cube. Kociemba will always solve any cube given to it, and it runs very quickly. However, because of its general-purpose nature, it often finds a longer solution compared to the other solvers. The other baseline is the **Korf** Iterative Deepening A\* (IDA\*) with a pattern database heuristic [15, 6]. Korf's algorithm will always find the optimal solution from any given starting state; but, since it is a

![](_page_5_Figure_7.jpeg)

Figure 4: Architecture for  $f_{\theta}$ . Each layer is fully connected. We use elu activation on all layers except for the outputs. A combined value and policy network results in more efficient training compared to separate networks. [30].

heuristic tree search, it will often have to explore many different states and it will take a long time to compute a solution. We also compare the full DeepCube solver against two variants of itself. First, we do not calculate the shortest path of our search tree and instead extract the initial path from the MCTS: this will be named **Naive DeepCube**. We also use our trained value network as a heuristic in a greedy best-first search for a simple evaluation of the value network: this will be named **Greedy**. An overview of each of the solver's capacity to solve cubes is presented in Figure 5a.

We compare our results to Kociemba using 640 randomly scrambled cubes. Starting from the solved cube, each cube was randomly scrambled 1000 times. Both DeepCube and Kociemba solved all 640 cubes within one hour. Kociemba solved each cube in under a second, while DeepCube had a median solve time of 10 minutes. The systematic approach of Kociemba explains its low spread of solution lengths with an interquartile range of only 3. Although DeepCube has a much higher variance in solution length, it was able to match or beat Kociemba in 55% of cases. We also compare DeepCube against Naive DeepCube to determine the benefit of performing the BFS on our MCTS tree. We find that the BFS has a slight, but consistent, performance gain over the MCTS path (-3 median solution)length). This is primarily because the BFS is able to remove all cycles from the solution path. A comparison of solution length distributions for these three solvers is presented in the left graph of Figure 6.

We could not include Korf in the previous comparison because its runtime is prohibitively slow: solving just one of the 640 cubes took over 6 days. We instead evaluate the optimality of solutions

![](_page_6_Figure_0.jpeg)

Figure 6: Distribution of solution lengths for DeepCube, Kociemba, and Korf. The left graph features naive DeepCube to evaluate the effect of our shortest path search. The right graph features the Korf optimal solver to evaluate how well DeepCube can find short solutions. The red lines represent the 26 and 15 move upper bound on the left and right respectively.

found by DeepCube by comparing it to Korf on cubes closer to solution. We generated a new set of 100 cubes that were only scrambled 15 times. At this distance, all solvers could reliably solve all 100 cubes within an hour. We compare the length of the solutions found by the different solvers in the right graph of Figure 6. Noticeably, DeepCube performs much more consistently for close cubes compared to Kociemba, and it almost matches the performance of Korf. The median solve length for both DeepCube and Korf is 13 moves, and DeepCube matches the optimal path found by Korf in 74% of cases. However, DeepCube seems to have trouble with a small selection of the cubes that results in several solutions being longer than 15 moves. Note that Korf has one outlier that is above 15 moves. This is because Korf is based on the half-turn metric while we are using the quarter-turn metric.

Furthermore, our network also explores far fewer tree nodes when compared to heuristic-based searches. The Korf optimal solver requires an average expansion of 122 billion different nodes for fully scrambled cubes before finding a solution [15]. Our MCTS algorithm expands an average of only 1,136 nodes with a maximum of 3,425 expanded nodes on the longest running sample. This is why DeepCube is able to solve fully scrambled cubes much faster than Korf.

#### 5.1 Knowledge Learned

DeepCube discovered a notable amount of Rubik's Cube knowledge during its training process, including the knowledge of how to use complex permutation groups and strategies similar to the best human "speed-cubers". For example, DeepCube heavily uses one particular pattern that commonly appears when examining normal subgroups of the cube:  $aba^{-1}$ . That is, the sequences of moves that perform some action a, performs a different action b, and then reverses the first action with  $a^{-1}$ . An intelligent agent should use these conjugations often because it is necessary for manipulating specific cubelets while not affecting the position of other cubelets.

We examine all of the solutions paths that DeepCube generated for the 640 fully scrambled cubes by moving a sliding window across the solutions strings to gather all triplets. We then compute the frequency of each triplet and separate them into two categories: matching the conjugation pattern  $aba^{-1}$  and not matching it. We find that the top 14 most used triplets were, in fact, the  $aba^{-1}$  conjugation. We also compare the distribution of frequencies for the two types of triplets.

![](_page_6_Figure_7.jpeg)

Figure 7: An example of Deep-Cube's strategy. On move 17 of 30, DeepCube has created the 2x2x2 corner while grouping adjacent edges and corners together.

![](_page_7_Figure_0.jpeg)

Figure 8: Comparison of the distribution of frequency of the two types of triplets. We split the triplets into conjugations ( $aba^{-1}$ ) and non-conjugations. We then calculate the frequency of each triplet and plot the two distributions. The two vertical lines are the means of their respective distributions.

In Figure 8, we plot the distribution of frequencies for each of the categories. We notice that conjugations appear consistently more often than the other types of triplets.

We also examine the strategies that DeepCube learned. Often, the solver first prioritizes completing a 2x2x2 corner of the cube. This will occur approximately at the half way point in the solution. Then, it uses these conjugations to match adjacent edge and corner cubelets in the correct orientation, and it returns to either the same 2x2x2 corner or to an adjacent one. Once each pair of corner-edge pieces is complete, the solver then places them into their final positions and completes the cube. An example of this strategy is presented in Figure 7. This mirrors a strategy that advanced human "speed-cubers" employ when solving the cube, where they prioritize matching together corner and edge cubelets before placing them in their correct locations.

#### 6 Discussion

We are currently further improving DeepCube by extending it to harder cubes. Autodidactic Iteration can be used to train a network to solve a 4x4x4 cube and other puzzles such as n-dimensional sequential move puzzles and combination puzzles involving other polyhedra.

Besides further work with the Rubik's Cube, we are working on extending this method to find approximate solutions to other combinatorial optimization problems such as prediction of protein tertiary structure. Many combinatorial optimization problems can be thought of as sequential decision making problems, in which case we can use reinforcement learning. Bello et. al. train an RNN through policy gradients to solve simple traveling salesman and knapsack problems [4]. We believe that harnessing search will lead to better reinforcement learning approaches for combinatorial optimization. For example, in protein folding, we can think of sequentially placing each amino acid in a 3D lattice at each timestep. If we have a model of the environment, ADI can be used to train a value function which looks at a partially completed state and predicts the future reward when finished. This value function can then be combined with MCTS to find approximately optimal conformations.

Léon Bottou defines reasoning as "algebraically manipulating previously acquired knowledge in order to answer a new question"[5]. Many machine learning algorithms do not reason about problems but instead use pattern recognition to perform tasks that are intuitive to humans, such as object recognition. By combining neural networks with symbolic AI, we are able to create algorithms which are able to distill complex environments into knowledge and then reason about that knowledge to solve a problem. DeepCube is able to teach itself how to reason in order to solve a complex environment with only one reward state using pure reinforcement learning.

In summary, combining MCTS with neural networks is a powerful technique that helps to bridge the gap between symbolic AI and connectionism. Furthermore, it has the potential to provide approximate solutions to a broad class of combinatorial optimization problems.

# Acknowledgments

We thank Harm van Seijen and Yutian Chen for helpful discussions.

#### References

- [1] How to solve the rubik's cube beginners method. https://ruwix.com/the-rubiks-cube/how-to-solve-the-rubiks-cube-beginners-method/.
- [2] Thomas Anthony, Zheng Tian, and David Barber. Thinking fast and slow with deep learning and tree search, 2017.
- [3] Richard Bellman. Dynamic Programming. Princeton University Press, 1957.
- [4] Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi, and Samy Bengio. Neural combinatorial optimization with reinforcement learning, 2016.
- [5] Leon Bottou. From machine learning to machine reasoning, 2011.
- [6] Andrew Brown. Rubik's cube solver. https://github.com/brownan/Rubiks-Cube-Solver, 2017.
- [7] Robert Brunetto and Otakar Trunda. Deep heuristic-learning in the rubik's cube domain: an experimental evaluation. 2017.
- [8] Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In *International conference on computers and games*, pages 72–83. Springer, 2006.
- [9] Rémi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In H. Jaap van den Herik, Paolo Ciancarini, and H. H. L. M. (Jeroen) Donkers, editors, *Computers and Games*, pages 72–83, Berlin, Heidelberg, 2007. Springer Berlin Heidelberg.
- [10] Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. *Deep learning*, volume 1. MIT press Cambridge, 2016.
- [11] Xiaoxiao Guo, Satinder Singh, Honglak Lee, Richard L Lewis, and Xiaoshi Wang. Deep learning for real-time atari game play using offline monte-carlo tree search planning. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, *Advances in Neural Information Processing Systems* 27, pages 3338–3346. Curran Associates, Inc., 2014.
- [12] Geoffrey Hinton, Nitsh Srivastava, and Kevin Swersky. Overview of mini-batch gradient descent, 2014.
- [13] Herbert Kociemba. Two-phase algorithm details. http://kociemba.org/math/imptwophase.htm.
- [14] Levente Kocsis and Csaba Szepesvári. Bandit based monte-carlo planning. In *European conference on machine learning*, pages 282–293. Springer, 2006.
- [15] Richard E. Korf. Finding optimal solutions to rubik's cube using pattern databases. In Proceedings of the Fourteenth National Conference on Artificial Intelligence and Ninth Conference on Innovative Applications of Artificial Intelligence, AAAI'97/IAAI'97, pages 700–705. AAAI Press, 1997.
- [16] Daniel Kunkle and Gene Cooperman. Twenty-six moves suffice for rubik's cube. In *Proceedings* of the 2007 International Symposium on Symbolic and Algebraic Computation, ISSAC '07, pages 235–242, New York, NY, USA, 2007. ACM.
- [17] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. *nature*, 521(7553):436, 2015.
- [18] Peter Lichodzijewski and Malcolm Heywood. The rubik cube and gp temporal sequence learning: an initial study. In *Genetic Programming Theory and Practice VIII*, pages 35–54. Springer, 2011.

- [19] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In *International Conference on Machine Learning (ICML)*, 2016.
- [20] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. *Nature*, 518(7540):529–533, 2015.
- [21] David E Moriarty, Alan C Schultz, and John J Grefenstette. Evolutionary algorithms for reinforcement learning. *Journal of Artificial Intelligence Research*, 11:241–276, 1999.
- [22] Silviu Radu. Rubik's cube can be solved in 34 quarter turns. http://cubezzz.dyndns.org/drupal/?q=node/view/92, Jul 2007.
- [23] Michael Reid. Superflip requires 20 face turns. http://www.math.rwth-aachen.de/~Martin.Schoenert/Cube-Lovers/michael\_reid\_superflip\_requires\_20\_face\_turns.html, Jan 1995.
- [24] Tomas Rokicki. Twenty-two moves suffice for rubik's cube®. *The Mathematical Intelligencer*, 32(1):33–40, 2010.
- [25] Tomas Rokicki. God's number is 26 in the quarter-turn metric. http://www.cube20.org/qtm/, Aug 2014.
- [26] Tomas Rokicki, Herbert Kociemba, Morley Davidson, and John Dethridge. The diameter of the rubik's cube group is twenty. *SIAM Review*, 56(4):645–670, 2014.
- [27] Richard B. Segal. On the scalability of parallel uct. In H. Jaap van den Herik, Hiroyuki Iida, and Aske Plaat, editors, *Computers and Games*, pages 36–47, Berlin, Heidelberg, 2011. Springer Berlin Heidelberg.
- [28] David Silver, Aja Huang, Christopher J. Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search. *Nature*, 529:484–503, 2016.
- [29] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. *arXiv* preprint *arXiv*:1712.01815, 2017.
- [30] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, Yutian Chen, Timothy Lillicrap, Fan Hui, Laurent Sifre, George van den Driessche, Thore Graepel, and Demis Hassabis. Mastering the game of go without human knowledge. *Nature*, 550(7676):354–359, 10 2017.
- [31] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. *Nature*, 550(7676):354, 2017.
- [32] Robert J Smith, Stephen Kelly, and Malcolm I Heywood. Discovering rubik's cube subgroups using coevolutionary gp: A five twist experiment. In *Proceedings of the Genetic and Evolutionary Computation Conference* 2016, pages 789–796. ACM, 2016.
- [33] Richard S. Sutton. Learning to predict by the methods of temporal differences. *Machine Learning*, 3(1):9–44, Aug 1988.
- [34] Richard S Sutton and Andrew G Barto. *Reinforcement learning: An introduction*, volume 1. MIT press Cambridge, 1998.
- [35] Gerald Tesauro. Temporal difference learning and td-gammon. *Communications of the ACM*, 38(3):58–68, 1995.
- [36] Morwen Thistlethwaite. Thistlethwaite's 52-move algorithm. https://www.jaapsch.net/puzzles/thistle.htm, Jul 1981.
- [37] Maxim Tsoy. Kociemba. https://github.com/muodov/kociemba, 2018.
- [38] Christopher JCH Watkins and Peter Dayan. Q-learning. Machine learning, 8(3-4):279–292, 1992.

| [39] Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. In <i>Reinforcement Learning</i> , pages 5–32. Springer, 1992. |  |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--|
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |
|                                                                                                                                                                                   |  |