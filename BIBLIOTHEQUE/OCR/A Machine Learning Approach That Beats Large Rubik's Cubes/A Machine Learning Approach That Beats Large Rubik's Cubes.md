## A Machine Learning Approach That Beats Large Rubik's Cubes The CayleyPy Project

A. Chervovo<sup>1,2,3†</sup>, K. Khoruzhiio<sup>4†</sup>, N. Bukhal<sup>5</sup>, J. Naghiyev<sup>1</sup>, V. Zamkovoy<sup>5</sup>, I. Koltsovo<sup>5</sup>, L. Cheldieva<sup>5</sup>, A. Sychevo<sup>5</sup>, A. Lenino<sup>5</sup>, M. Obozov<sup>6</sup>, E. Urvanov<sup>5</sup>, A. Romanovo<sup>5\*</sup>

<sup>1</sup> Institut Curie, Université PSL, Paris, F-75005, France;

<sup>2</sup> INSERM U900, Paris, F-75005, France;

<sup>3</sup> CBIO, Mines ParisTech, Université PSL, Paris, F-75005, France;

<sup>4</sup> Technical University of Munich, Garching, 85748, Germany;

<sup>5</sup> Institute of Artificial Intelligence, RTU MIREA, Moscow, 119454, Russia;

<sup>6</sup> Yandex Research, Moscow, 119021, Russia

\*Corresponding author: romanov@mirea.ru †These authors contributed equally to this work.

The paper proposes a novel machine learning-based approach to the pathfinding problem on extremely large graphs. This method leverages diffusion distance estimation via a neural network and uses beam search for pathfinding. We demonstrate its efficiency by finding solutions for  $4\times4\times4$  and  $5\times5\times5$  Rubik's cubes with unprecedentedly short solution lengths, outperforming all available solvers and introducing the first machine learning solver beyond the  $3\times3\times3$  case. In particular, it surpasses every single case of the combined best results in the Kaggle Santa 2023 challenge, which involved over 1,000 teams. For the  $3\times3\times3$  Rubik's cube, our approach achieves an optimality rate exceeding 98%, matching the performance of task-specific solvers and significantly outperforming prior solutions such as DeepCubeA (60.3%) and EfficientCube (69.6%). Additionally, our solution is more than 26 times faster in solving  $3\times3\times3$  Rubik's cubes while requiring up to 18.5 times less model training time than the most efficient state-of-the-art competitor.

#### I. INTRODUCTION

Rubik's cube is one of the most famous puzzles, which is believed to be played by more than a billion people in the world [1]. According to [2], it was included in the 100 most influential inventions of the 20th century. Even decades after its first introduction, it is still used as a benchmark and model task in various fields: artificial intelligence [3], robotics [4], graphs algorithms [5, 6], cryptography [7], image encryption [8], statistical physics [9, 10], group theory [11, 12], for human cognitive abilities [13].

From a more general perspective, solving the Rubik's Cube is a particular case of a planning problem—one needs to plan a sequence of actions to transit between two given states. Planning robot moves and games like chess or Go represent similar problems—for example, a game's goal is to plan moves to transit from the initial position to the winning position(s). The mathematical framework for such problems is pathfinding on graphs (state transition graphs): all possible states are represented as nodes, and edges correspond to transitions between states based on actions (moves). The planning task thus reduces to finding a path from a given initial node to one or more desired nodes. A specific class of graphs represents the Rubik's Cube and similar puzzles—Cayley-type graphs of the puzzle's symmetry group. These are highly symmetric state transition graphs where the symmetry group can transform any node into another. Cayley graphs are of fundamental importance in modern mathematics [14, 15] and have numerous applications: in bioinformatics for estimating evolutionary distances [16–19]; in processor interconnection networks [20–22]; in coding theory for the construction of expander graphs and related codes [23]; in cryptography for constructing specific hash functions [7, 24]; in machine learning (ML) [18]; and in quantum computing [25–29].

Finding the shortest paths on generic finite Cayley graphs is an NP-hard problem [30], as it is for many particular groups: the Rubik's Cube group [31] and some others [18, 32]. Brute force breadth-first search, Dijkstra's, and related methods can find the shortest paths on graphs with billions of nodes, the bidirectional trick squares feasible sizes, but these methods require extremely large computational resources and are not practical for much larger sizes, which are of our interest. Moreover, no effective tools are currently available to find any (not just the shortest) paths on Cayley graphs of large finite groups. For example, modern computer algebra systems like GAP [33] fail on any sufficiently large group, such as the  $4\times4\times4$  Rubik's Cube.

**Results.** To address these issues, we develop machine learning-based methods to find paths on a broad class of graphs (specified below) of unprecedented sizes and the ability to produce unprecedentedly short paths. In the present paper, we provide code applicable to Cayley graphs (or, more generally, Schreier graphs) of any finite permutation group and focus on demonstrating its efficiency for Rubik's groups. The presented approach is the first machine learning-based method to successfully solve the  $4\times4\times4$  and  $5\times5\times5$  Rubik's Cubes, with  $7.4\times10^{45}$  and  $1.2\times10^{74}$  elements, respectively. The obtained solution lengths are shorter than those produced by any available method, including the combination of top re-

sults from the Kaggle Santa 2023 Challenge [34], where more than a thousand participants applied and developed various methods. Moreover, for the  $4\times4\times4$  cube, the average solution length is 46.51, which is below the conjectural diameter 48 [35], thus providing further evidence for the quality of our solutions. For the  $3\times3\times3$  Rubik's Cube, we achieve 98.4% optimality solving scrambles from the DeepCubeA dataset, surpassing previous machine learning approaches: DeepCubeA at 60. 3% [3] and EfficientCube at 69.6% [36].

To conclude, this research aims to make a significant step in advancing machine learning applications to graph pathfinding and to demonstrate its efficiency in the case of Rubik's cubes of different sizes, providing more optimal solutions than any available approach for large groups. The main contributions are the following:

- 1. We propose a novel multi-agent, machine learning-based approach to find paths on Cayley graphs of finite groups. It is the first machine learning approach capable of handling groups as large as 10<sup>74</sup>. It achieves over 98% optimality on the DeepCubeA dataset of 3×3×3 cubes, reaching the level of task-oriented solvers based on pattern databases. It produces better results (shorter solution paths) than any known competitor for 4×4×4 and 5×5×5 Rubik's Cubes, including the aggregated best results from the 2023 Kaggle Santa Challenge, representing the current state of the art.
- 2. We demonstrate that increasing the size of the set used to train multilayer perceptrons with residual blocks has a limited impact on the pathfinder's performance. At the same time, increasing the beam width and number of agents robustly improves the average solution length and optimality. This surprising finding helped choose the size of the train data for each agent and achieve best-in-class performance without wasting computational resources on additional training.
- 3. The training time and computational resources required for our approach are significantly smaller than those for state-of-the-art approaches. Our solution, tested on the same hardware and beam width as EfficientCube (the previous leading ML solution), performs pathfinding slightly better than EfficientCube, solving the task  $\approx 26 \mathrm{x}$  faster and requiring up to 18.5x less model training time than the competitor.

In recent years, machine learning has been emerging as "a tool in theoretical science" [37], leading to several noteworthy applications to mathematical problems [38–45]. This research is part of the larger project, which aims to create an open-source machine learning Python framework for analyzing Cayley graphs and contribute to the fascinating, emerging area of machine learning applications in theoretical sciences.

#### II. RESULTS

#### A. Proposed Machine Learning Approach

This paper presents a unified approach for finding paths on a large class of graphs, focusing on demonstrating its efficiency for Rubik's cube graphs. It does not rely on any prior knowledge or human expertise about the graphs. The approach has two main components: a neural network model and a graph search algorithm – similar to previous works such as AlphaGo/AlphaZero [46, 47], DeepCube [3, 48], and EfficientCube [36], among others. The model is trained to guide what moves should be done to get closer to the destination node ("solved state" for puzzles). The graph search algorithm starts from a given node and moves to nodes closer to the destination, based on the neural network's predictions, until the destination node is found.

The basic assumption on a graph is that there is a vector associated with each node (feature vector). These vectors serve as an input for the neural network [49]. For puzzles or permutation groups—it is just the vector describing the permutation p of l-symbols, i.e. vector of numbers  $(p(0), p(1), \ldots, p(l-2), p(l-1))$ . Additionally, we assume that a specific node on the graph, such as the 'solved state' for puzzles, is selected. The task is to find a path from any given node to this selected node. Since the graph sizes may exceed  $10^{70}$ , standard pathfinding methods are not applicable.

The key steps of the proposed method are illustrated in the figure 1a and described below:

Generating Training Data via Random Walks and Diffusion Distance. Generate N random walk trajectories starting from a selected node. (The generation of a random walk is a simple process: select a random neighbor of the current node and repeat this process iteratively for multiple steps.) Each random walk trajectory consists of up to  $K_{\text{max}}$  steps, where N and  $K_{\text{max}}$ are integer parameters of the method. For some nodes encountered during the random walks, we store a set of pairs (v, k), where v represents the vector corresponding to the node and k is the number of steps required to reach it via the random walk. This set will serve as the training data. For the Rubik's Cube, random walks correspond to random scrambling: starting from the "solved state," we perform a series of random scrambles and record the resulting positions and the number of scrambles performed. Conceptually, in the limit as  $N \to \infty$ , the average value of k measures the "diffusion distance"—roughly speaking, the length of the random path or an estimate of how quickly diffusion reaches a given node. In contrast to the DAVI approach used in [3], random walk generation is very computationally cheap, making it possible to generate them directly during the training procedure.

**Training the neural network.** The generated set of pairs (v, k) serves as the training set for the neural network. Specifically, v serves as the 'feature vector' (the input for the neural network), and k represents the 'target'

![](_page_2_Figure_1.jpeg)

FIG. 1. Proposed ML solution for Rubik's cube solving: (a) proposed multi-agent solver's process flow; (c) ResMLP neural network architecture; (b) an example of beam search pathfinding on  $3\times3\times3$  cube's graph using W=40.

(the output the network needs to predict). Thus, the neural network's predictions for a given node v estimate the diffusion distance from v to the selected destination node (solved state of the puzzle). We utilize a multilayer perceptron (MLP) architecture with several residual blocks and batch normalization, as shown in Figure 1b, which will be further called ResMLP. It is a general form of the MLPs used in [3, 36]. All the models are trained in advance before the solving phase.

Pathfinding with neural network heuristics and Beam Search. This step finds a path from a given node to the destination node. The neural network provides heuristics on where to make the next steps, while the graph pathfinding technique compensates for any possible incorrectness in the neural network predictions. The beam search pathfinding method is quite simple but has proven to be the most effective for us and works as follows. Fix a positive integer W — a parameter known as the "beam width" (or "beam size"). Starting from a given node, we take all its neighboring nodes and compute the neural network predictions for all of them. We then select the W nodes closest to the destination according to the neural network (i.e., the predictions have smaller values). We take these selected W nodes' neighbors, drop duplicates, and again compute the neural network predictions, choosing the top W nodes with the best (i.e., minimal) predictions. The search iterations are repeated until the destination node is found (or the limit of steps is exceeded). The whole process is illustrated in Figure 1c.

Multi-agency. The method described in the steps above relies on random walks for train set creation, and thus, due to that randomness, each new launch will create a new train set, and thus, each new neural network approximates the distance differently. This diversity is large enough to yield a new solution path for each launch typically. And hence, typically, several repetitions allow for the discovery of a shorter path than a single run. We call each trained neural network an agent. To solve any given state - we solve it with all the agents and then choose the best result (the shortest solution path among all the agents) – illustrated in Figure 1a.

# B. Optimality vs the Proposed Approach Parameters

The proposed solver has the following main parameters A – the number of agents, W – beam width used by each agent during pathfinding and ResMLP model general pa-

![](_page_3_Figure_1.jpeg)

FIG. 2. Influence of model parameters on solution length for  $3\times3\times3$  and  $4\times4\times4$  cubes (jitter plot): (a) influence of the model size, trainset sizes, and model depth on average solution length; (b) influence of the beam width on average solution length.

rameters:  $N_1$  – the size of the first layer,  $N_2$  – the size of the second layer and residual blocks' layers,  $N_r$  – the number of residual blocks and T – the trainset size. For easier comparison,  $N_1$ ,  $N_2$ , and  $N_r$  are also summarized by model size P – the total number of ResMLP parameters (weights and biases). In this section, we analyze the influence of these parameters on the solver's average solution length and optimality.

Train set size. First, in the example of  $3\times3\times3$  and  $4\times4\times4$  Rubik's cubes, we analyzed how the model and trainset sizes, as well as model depth, influence the average solution length using a single agent with fixed beam width  $(W=2^{18})$ . The experiment details are provided in Section IV E, while the results are presented in Figure 2a. It is seen from Figure 2a that from a certain point, the raise of T does not lead to any significant reduction of average solution length, especially considering the fact that the trainset size is demonstrated in logarithmic scale. Even more surprising, the T value corresponding to this point is very similar for  $3\times3\times3$  and  $4\times4\times4$  cubes and neural networks of different sizes and depths. Thus, the experiments above reveal a rather unexpected effect - performance stagnation with respect to the train size.

MLP layers and sizes. As expected, larger and deeper networks trained on train sets of the same size generally provide shorter solutions than smaller models. What is less expected is that the higher number of layers (higher  $N_1$ ,  $N_2$ , and  $N_r$ ) is more significant than a larger number of parameters P. More surprising is that even small models with 1M of parameters can reach the average solution length comparable to DeepCubeA and EfficientCube using neural networks with  $\approx 14.7 \mathrm{M}$  parameters. Based on these observations for further consideration, we used a deep neural network having the same number of layers (ten) as [3] and [36], but with the smaller model size of 4M parameters.

**Beam width.** It is the most important parameter. We performed multiple tests on a single agent equipped with this model, changing W from  $2^{12}$  to  $2^{24}$ . The results of these tests are presented in Figure 2b (the exact model parameters and details of the tests are provided in

Section IV E). From Figure 2b, it is clear that increasing W effectively reduces the average solution length. Moreover, the solution length decreases approximately linearly with the logarithm of the beam width W. On the  $3\times3\times3$  cube, increasing W up to  $2^{24}$  allows us to get close to the optimal solution, while for  $4\times4\times4$ , the same beam width results in a better average solution length than the best ones submitted to the 2023 Santa Challenge.

Agents number. In the third part of the experimental studies, we investigated the influence of the number of agents A on the solver's efficiency. These experiments were performed on  $3\times 3\times 3$ ,  $4\times 4\times 4$ , and  $5\times 5\times 5$  Rubik's Cube. We used 10-layer ResMLP models with 4M parameters trained on 8B states in all the cases. The beam width was chosen  $W=2^{24}$  so each agent could fit into the memory of a single GPU regardless of the solved cube size. The details of the performed experiments are available in Section IV E, while their results are provided in Figure 3. For ease of analysis, Figures 3a,3b,3c demonstrate lengths only for those agents whose solutions at least once were used as the solver's output.

Figure 3 clearly shows that the average solution rate of a multi-agent is always higher than the one achieved by the best single agent (up to 8 moves for the  $5\times5\times5$  cube). Solid lines on Figures 3a, 3b, 3c show how the size of the ensemble influences the average solution length for the random set of the agents. As seen in all three cases of  $3\times3\times3$ ,  $4\times4\times4$ , and  $5\times5\times5$  Rubik's cubes, the larger number of agents robustly provided more optimal pathfinding. The dashed line demonstrates the same dependency but for the set of agents jointly providing the best overall solution. Figures 3c, 3d, 3e demonstrate in color code how each agent from this set participates in the final solution for every scramble from the dataset. The scrambles which were not solved in less than 200 moves are marked with crosses.

If it is seen from Figures 3c,3d,3e that the worst agents in the ensemble not only provide much longer results than the final solution but also include multiple scrambles that were unsolved. In the case of using the single-model approach, these agents would be considered unsatisfyingly trained. Nevertheless, they are included in the best en-

![](_page_4_Figure_1.jpeg)

FIG. 3. Average solution length of the proposed multi-agent approach depending on the number of agents composing its output for (c)  $3\times3\times3$ , (e)  $4\times4\times4$ , and (f)  $5\times5\times5$  Rubik's cubes. Solid line – random set of the agents, dashed line – best set. Distribution of solution lengths for (c)  $3\times3\times3$ , (d)  $4\times4\times4$ , and (e)  $5\times5\times5$  Rubik's cubes for the best ensemble.

semble because they provided the shortest solution on one or two scrambles. Moreover, our approach reached efficiency unreachable for other ML solutions only due to such specialized agents.

Even though the results presented in Figure 3 on  $3\times3\times3$ ,  $4\times4\times4$ , and  $5\times5\times5$  cubes can be achieved using 5, 10, and 10 agents, respectively, the probability of training all these agents in a row is very low. For example, to beat all the  $5\times5\times5$  scrambles from the 2023 Santa Challenge dataset, we trained 69 different agents, while further analysis showed that only 10 of them composed all the output results. At the same time, the first agent trained to solve  $4\times4\times4$  cubes beat all the respective scrambles from the mentioned dataset but did not even get in the final ensemble because multiple other agents jointly surpassed it. Thus, achieving a high level of optimality requires many agents, as seen from the logarithmic nature of the plots demonstrated in Figure 3. Nevertheless, due to the high scalability of the proposed approach and the ability to run on distributed hardware using dozens of independent agents, it is not an issue using modern computational hardware.

## C. Results Summary and Comparison with Prior Art

Table I[51] summarizes the main results achieved by the proposed solver, highlighting its Superiority over the prior state of the art. Notably, it surpasses the 2023 Kaggle Santa Challenge results, where over a thousand teams competed in virtual puzzle solutions, representing the best available methods and results. It should be mentioned that we were limited in computation resources during our research. Thus, our results can be improved even more by using more advanced hardware, which will allow for an increase in beam width and the number of agents.

A single agent with single-layer MLP can solve all the DeepCubeA dataset with 90.4% of optimality, significantly enhancing results of the most advanced state-of-the-art ML solutions: DeepCubeA and EfficientCube. 26 agents equipped with 10-layer ResMLP models managed to solve all 1000 scrambles from DeepCubeA dataset with 97.6% optimality, which is the best result ever achieved by any ML solution (significantly surpassing 60.3%, 69.8% results from DeepCubeA and EfficientCube). A

TABLE I. The most notable results achieved by the proposed solution and comparison with competitors.

| No. | Solver                                        | Metric <sup>a</sup> , Size,<br>Dataset | Solver parameters |          |                  |                  | Average solution  | Optimality/<br>Superiority |
|-----|-----------------------------------------------|----------------------------------------|-------------------|----------|------------------|------------------|-------------------|----------------------------|
|     |                                               |                                        | $\overline{A}$    | W        | $\boldsymbol{P}$ | $\boldsymbol{T}$ | length            | - ,                        |
|     |                                               | 2x2x2 Rubi                             | k's cul           | е        |                  |                  |                   |                            |
| 1   | Genetic [50]                                  | HTM, 1, [50]                           | n/a               | n/a      | n/a              | n/a              | 30 <sup>b</sup>   | n/a <sup>b</sup>           |
| 2   | Breadth First Search                          | QTM, 100, Ours                         | n/a               | n/a      | n/a              | n/a              | 10.669            | Opt. 100%                  |
| 3   | Ours, 1-layer MLP                             | QTM, 100, Ours                         | 1                 | $2^{18}$ | 0.15M            | 8B               | 10.669            | Opt. $100\%$               |
| 4   | Ours, 10-layer ResMLP                         | QTM, 100, Ours                         | 1                 | $2^{18}$ | 0.92M            | 8B               | 10.669            | Opt. 100%                  |
|     |                                               | $3 \times 3 \times 3$ Rubi             | k's cu            | be       |                  |                  |                   |                            |
| 5   | Genetic [50]                                  | HTM, 1, [50]                           | n/a               | n/a      | n/a              | n/a              | $238^{\rm b}$     | $n/a^b$                    |
| 6   | Optimal PDB+ <sup>c</sup> solver [3]          | QTM, 1000, [3]                         | n/a               | n/a      | n/a              | n/a              | 20.637            | Opt. $100\%$               |
| 7   | DeepCube [3]                                  | QTM, 1000, [3]                         | 1                 | n/a      | 14.7M            | 10B              | 21.50             | Opt. $60.3\%$              |
| 8   | EfficientCube [36]                            | QTM, 1000, [3]                         | 1                 | $2^{18}$ | 14.7M            | 52B              | 21.26             | Opt. 69.6%                 |
| 9   | EfficientCube [36] (reproduced)               | QTM, 1000, [3]                         | 1                 | $2^{18}$ | 14.7M            | 52B              | 21.255            | Opt. $69.8\%$              |
| 10  | Ours, 10-layer ResMLP                         | QTM, 1000, [3]                         | 1                 | $2^{18}$ | 4M               | 8B               | 21.137            | Opt. <b>75.4</b> %         |
| 11  | Ours, 1-layer MLP                             | QTM, 1000, [3]                         | 1                 | $2^{24}$ | 0.34M            | 8B               | 20.829            | Opt. $90.4\%$              |
| 12  | Ours, 10-layer ResMLP                         | QTM, 1000, [3]                         | 1                 | $2^{24}$ | 4M               | 8B               | 20.691            | Opt. <b>97.3</b> %         |
| 13  | Ours, multi-agent 10-layer ResMLP             |                                        | 26                | $2^{24}$ | 4M               | 8B               | 20.669            | Opt. <b>98.4</b> %         |
| 14  | Santa Challenge [34]                          | UQTM, 82, [34]                         | n/a               | n/a      | n/a              | n/a              | 21.829            | $n/a^d$                    |
| 15  | Ours, 10-layer ResMLP                         | UQTM, 82, [34]                         | 1                 | $2^{24}$ | 4M               | 8B               | 19.512            | Sup. <b>100</b> %          |
|     |                                               | $4 \times 4 \times 4$ Rubi             | k's cu            | be       |                  |                  |                   |                            |
| 16  | Genetic [50]                                  | HTM, 1, [50]                           | n/a               | n/a      | n/a              | n/a              | $737^{\rm b}$     | n/a <sup>b</sup>           |
| 17  | Santa Challenge [34]                          | UQTM, 43, [34]                         | n/a               | n/a      | n/a              | n/a              | 53.49             | $n/a^d$                    |
| 18  | Ours, 10-layer ResMLP                         | UQTM, 43, [34]                         | 1                 | $2^{24}$ | 4M               | 8B               | 48.98             | Sup. <b>100</b> %          |
| 19  | $\mathbf{Ours}$ , multi-agent 10-layer ResMLP | UQTM, 43, [34]                         | 29                | $2^{24}$ | 4M               | 8B               | 46.51             | Sup. $100\%$               |
|     |                                               | $5{	imes}5{	imes}5$ Rubi               | k's cu            | be       |                  |                  |                   |                            |
| 20  | Genetic [50]                                  | HTM, 1, [50]                           | n/a               | n/a      | n/a              | n/a              | 1761 <sup>b</sup> | n/a <sup>b</sup>           |
| 21  | Santa Challenge [34]                          | UQTM, 19, [34]                         | n/a               | n/a      | n/a              | n/a              | 96.58             | $n/a^d$                    |
| 22  | Ours, multi-agent 10-layer ResMLP             | UQTM, 19, [34]                         | 69                | $2^{24}$ | 4M               | 8B               | 92.16             | Sup. <b>100</b> %          |

<sup>&</sup>lt;sup>a</sup> HTM – half-turn metric, QTM – quater-turn metric. The 2023 Kaggle Santa Challange dataset uses modified QTM with unfixed corners and centers of the cube, which is marked UQTM.

single-agent solution implemented using our approach and 10-layer ResMLP managed to beat each best result corresponding to  $3\times3\times3$  and  $4\times4\times4$  Rubik's cubes submitted on the 2023 Kaggle Santa Challenge (averages: 48.98 vs 53.49). At the same time, 29 agents managed to solve all the  $4\times4\times4$  cube's scrambles from the 2023 Kaggle Santa Challenge dataset with an average solution length of 46.51 - which is below 48 (a conjectured  $4\times4\times4$  Rubik's cube diameter [52]). Finally, an ensemble of 69 agents beat each best solutions for the  $5\times5\times5$ Rubik's cube submitted to the 2023 Kaggle Santa Challenge, shortening the average solution rate among all the datasets on more than 4.4 units in QTM metrics (ours: 92.16, Santa: 96.58). It is worth emphasizing that the solutions that were obtained outperformed the Santa results on average and in every single case.

The efficiency of our approach is driven not only by

the large number of agents but also by the efficiency of each single node. We performed an additional test to prove this statement and compared it with EfficentCube in terms of average computation time while running on the same hardware. The training procedure for EfficentCube took 86 hours 25 minutes, while the model for our solution was trained in 4 hours 40 minutes. Then, both solutions were used to solve all the scrambles from the DeepCubeA dataset (see results No.9 and 10 in Table I). Our solution provided slightly better results using the same beam width of  $2^{18}$ . At the same time, EfficientCube required 287.78 s on average to solve a single scramble, while our solution required 10.91 s, which is  $\approx 26$  times faster.

b The results presented in [50] evaluated using single cube scrambled with 100 random moves. For these results, column "Avg. solution len" contains the minimal length achieved by the most suitable genetic algorithm configuration. The optimality/superiority was not evaluated for these results.

 $<sup>^{\</sup>rm c}$  https://github.com/rokicki/cube20src

<sup>&</sup>lt;sup>d</sup> Superiority is not applicable for 2023 Kaggle Santa Challenge best results as [34] dataset is built of these results.

#### III. DISCUSSION

The paper proposes a machine learning-based approach to the pathfinding problem on large graphs. Experimental studies demonstrate that it is more efficient than state-of-the-art solutions in terms of average solution length, optimality, and computational performance.

The key parts of the approach are multi-agency, neural networks predicting diffusion distance and beam search. Deeper neural networks better approximate the graph of the large Rubik's cubes, though, for the  $3\times3\times3$  case, even a single-layer network provides excellent results. At the same time, the effect of enlarging the training set is limited: the trainset above 8196M examples for the tested models has no practical reason, which allowed us to avoid additional time spent during the training. Conversely, raising the beam width effectively lowers the solution length and increases optimality.

The complete set of the proposed solutions allowed the creation of the multi-agent pathfinder, which managed to beat all the ML-based competitors: an agent equipped with single-layer MLP solved all the DeepCubeA dataset with 90.4% of optimality significantly enhancing results of the most advanced state of the art solutions: Deep-CubeA and EfficientCube. 26 agents equipped with 10layer ResMLP models managed to solve all scrambles from DeepCubeA dataset with 97.6% optimality, which is the best result ever achieved by any ML solution. Singleagent solutions implemented using our approach and 10layer ResMLP beat all the best results corresponding to  $3\times3\times3$  and  $4\times4\times4$  Rubik's cubes submitted on the 2023 Santa Challenge. At the same time, six agents managed to solve all the  $4\times4\times4$  cube's scrambles from the 2023 Santa Challenge dataset with an average solution length below 48 (a  $4\times4\times4$  Rubik's cube diameter predicted in [52]). Finally, a composition of 69 agents beat all the best solutions for the  $5\times5\times5$  Rubik's cube submitted to the 2023 Santa Challenge, shortening the average solution rate among all the datasets on more than 4.4 units in QTM metrics.

The method's scope is quite wide and can be applied to various planning tasks in their graph pathfinding reformulation. In future works, we plan to explore its applications to mathematical, bioinformatic, and programming tasks.

#### IV. METHODS

#### A. Cayley graphs and Rubik's cubes

Moves of Rubik's cube can be described by permutations (e.g., Chapter 5 [53], or Kaggle notebook "Visualize allowed moves" [54]). Taking all the positions as nodes and connecting them by edges, which differ by single moves, one obtains a Cayley-type (Schreier) graph for Rubik's cube. Solving the puzzle is equivalent to finding

a path on the graph between nodes representing the Rubik's cube's scramble initial and solved state.

#### B. Random walks and train set generation

The training set is generated by scrambling (i.e., applying random moves) the selected solved state and creating a set of pairs (v, k), where k is a number of scrambles, and v is a vector describing the node obtained after k steps. In other words, we consider random walks on the graph. The main parameters are  $K_{\text{max}}$  and K, where  $K_{\text{max}}$  is a maximal number of scrambles (length of random walk trajectory), while  $K \cdot K_{\text{max}}$  is a number of nodes to generate.

In the current research, we used so-called non-backtracking random walks [55], that forbid scrambling to the state of the previous step. A PyTorch-optimized implementation of train set generation can be found in *trainer.py* in the code attached to this paper.

Current research does not investigate the influence of  $K_{\rm max}$  on the solver's performance. We used  $K_{\rm max}=26$  for solvers targeted on  $3\times3\times3$  cubes,  $K_{\rm max}=45$  – for  $4\times4\times4$  cubes, and  $K_{\rm max}=65$  for  $5\times5\times5$  cubes.

#### C. Neural Network and Training procedure

In this study, we used ResMLP, a generalized form of multilayer perceptrons as described in [3, 36]. Details of the architecture can be found in Figure 1b. The PyTorch implementation of ResMLP is available in *model.py* in the code attached to this paper.

The training procedure was performed using the Adam optimizer with a fixed learning rate of 0.001 and mean squared error as the loss function. A new dataset of 1M examples was generated before each training epoch. All models were pre-trained and remained unchanged during puzzle-solving. Training was conducted using 32-bit floating point precision, while inference used 16-bit floating point numbers to enhance computational efficiency. The PyTorch implementation of the training procedure is available in trainer.py in the code attached to this paper.

### D. Beam-search

Beam search is a simple but effective search procedure used for various optimization tasks [56–58] as well as to improve outputs of the modern transformer-based language models [59–61]. It has been used in EfficientCube [36] and by many participants of the Kaggle Challenge [34]. We implemented a modified version of traditional beam search, which uses hash functions to remove duplicates, reducing the computation complexity of the pathfinder. Finally, in all the experiments, the scramble was considered unsolved if the path to the solved state

TABLE II. The parameters of neural networks used in current research

| No. | Cube                  | Metric                    | Layers | $N_1$ | $N_2$ | $oldsymbol{N}_r$ | P     | Result No. |
|-----|-----------------------|---------------------------|--------|-------|-------|------------------|-------|------------|
| 1   | $3 \times 3 \times 3$ | QTM                       | 1      | 3050  | 0     | 0                | 1M    | _          |
| 2   | $3\times3\times3$     | $\overline{\mathrm{QTM}}$ | 2      | 850   | 850   | 0                | 1M    | -          |
| 3   | $3\times3\times3$     | $\overline{\mathrm{QTM}}$ | 6      | 800   | 340   | 2                | 1M    | -          |
| 4   | $3\times3\times3$     | $_{ m QTM}$               | 10     | 430   | 300   | 4                | 1M    | _          |
| 5   | $3\times3\times3$     | QTM                       | 1      | 12196 | 0     | 0                | 4M    | -          |
| 6   | $3\times3\times3$     | $_{ m QTM}$               | 2      | 1841  | 1841  | 0                | 4M    | _          |
| 7   | $3\times3\times3$     | $_{ m QTM}$               | 6      | 2000  | 697   | 2                | 4M    | _          |
| 8   | $3\times3\times3$     | $_{ m QTM}$               | 10     | 700   | 643   | 4                | 4M    | 10, 12, 13 |
| 9   | $4 \times 4 \times 4$ | UQTM                      | 2      | 750   | 750   | 0                | 1M    | -          |
| 10  | $4 \times 4 \times 4$ | UQTM                      | 4      | 530   | 470   | 1                | 1M    | =          |
| 11  | $4 \times 4 \times 4$ | UQTM                      | 6      | 720   | 300   | 2                | 1M    | =          |
| 12  | $4 \times 4 \times 4$ | UQTM                      | 10     | 500   | 266   | 4                | 1M    | _          |
| 13  | $4 \times 4 \times 4$ | UQTM                      | 2      | 1730  | 1730  | 0                | 4M    | _          |
| 14  | $4 \times 4 \times 4$ | UQTM                      | 6      | 1180  | 1024  | 1                | 4M    | _          |
| 15  | $4 \times 4 \times 4$ | UQTM                      | 6      | 2000  | 628   | 2                | 4M    | =          |
| 16  | $4 \times 4 \times 4$ | UQTM                      | 10     | 1010  | 592   | 4                | 4M    | 18, 19     |
| 17  | $4 \times 4 \times 4$ | UQTM                      | 6      | 2000  | 1126  | 2                | 8M    | _          |
| 18  | $4 \times 4 \times 4$ | UQTM                      | 10     | 1540  | 850   | 4                | 8M    | -          |
| 19  | $4 \times 4 \times 4$ | UQTM                      | 6      | 5000  | 1369  | 2                | 16M   | _          |
| 20  | $4 \times 4 \times 4$ | UQTM                      | 10     | 5000  | 1062  | 4                | 16M   | -          |
| 21  | 2x2x2                 | QTM                       | 1      | 1024  | 0     | 0                | 0.15M | 3          |
| 22  | 2x2x2                 | $_{ m QTM}$               | 10     | 430   | 300   | 4                | 0.92M | 4          |
| 23  | $3\times3\times3$     | $\overline{\mathrm{QTM}}$ | 1      | 1024  | 0     | 0                | 0.34M | 11         |
| 24  | $3\times3\times3$     | UQTM                      | 10     | 700   | 643   | 4                | 4M    | 15         |
| 25  | $5 \times 5 \times 5$ | UQTM                      | 10     | 1008  | 560   | 4                | 4M    | 22         |

was not found in 200 beam search steps. Additionally, the algorithm stops if the beam vector contains only already visited graph nodes. A PyTorch-optimized implementation of the beam search can be found in *searcher.py* in the code attached to this paper.

## E. Experiments design

All the experiments were conducted using software attached to this paper. The experiments targeting analysis of trainset size's influence on the solver's performance included solving 20 scrambles of both  $3\times3\times3$  and  $4\times4\times4$ Rubik's cubes using different models as beam search heuristics. For this experiment, we prepared 20 models, whose parameters are demonstrated in the first 20 rows of Table II. Each model was trained during 16384 epochs. The snapshots of the model parameters were saved after 16, 64, 256, 1024, 4096, and 16384 epochs. Then, each model snapshot was integrated as a heuristic into beam search with  $W=2^{18}$ , which was used to solve the first 20 scrambles from the dataset. DeepCubeA dataset [3] was used for  $3\times3\times3$  Rubik's cube, and the 2023 Kagle Santa Challenge [34] dataset was used for  $4\times4\times4$  puzzle. The results achieved by each solver configuration on the corresponding dataset were averaged and analyzed as experimental results. Unsolved scrambles were excluded

from consideration in this experiment.

The first experiment's results are demonstrated in Figure 2a. Single layer MLPs for  $4\times4\times4$  Rubik's cube are not presented in Table II as during preliminary research solvers equipped with this type of model did not manage to solve any scramble before reaching the 200 steps limit.

During the second experiment, we used only 10 layer models with size 4M (models No.8 and 16 from Table II). The first experiment's results did not show a significant effect of increasing T (train size) from 4B to 16B. Thus, the second experiment used finer granularity with 4B, 8B, and 16B train sizes to select the appropriate size more precisely. Each snapshot of the models trained with the mentioned above train sets was integrated into solvers with W of  $2^{12}$ ,  $2^{14}$ ,  $2^{16}$ ,  $2^{18}$ ,  $2^{20}$ ,  $2^{22}$ , and  $2^{24}$ . Then, these solvers were used to unscramble the first 20 puzzles from the same dataset used in the first experiment. The results achieved by each solver configuration on the corresponding dataset were averaged. Unsolved scrambles were excluded from consideration in this experiment.

The results of the second experiment are demonstrated in Figure 2b. A deeper analysis of the experimental results shows that if we consider only W values that give a puzzle winning probability above 50%, the agent with the model trained on 8B states has a slightly better average solution length than the competitors (Table III). Thus, for the rest of the experiments, we used the 8B

| TABLE III. Average solution length depending from trainset size $T$ and beam width $W$ for tested with | n puzzle win probability |
|--------------------------------------------------------------------------------------------------------|--------------------------|
| above 0.5.                                                                                             |                          |

| T                      | 4M        |                  | 81                 | <i>I</i>          | 16M       |           |
|------------------------|-----------|------------------|--------------------|-------------------|-----------|-----------|
| W                      | Win prob. | Avg. sol.        | Win prob.          | Avg. sol.         | Win prob. | Avg. sol. |
|                        | 3>        | 3×3 Rubik's solv | er results with wi | n probability abo | ove 0.5   |           |
| $2^{12}$               | 1.00      | 22.15            | 1.00               | 22.05             | 1.00      | 22.3      |
| $2^{14}$               | 1.00      | 21.55            | 1.00               | 21.55             | 1.00      | 21.6      |
| $2^{16}$               | 1.00      | 21.25            | 1.00               | 21.15             | 1.00      | 21.2      |
| $2^{18}$               | 1.00      | 21.15            | 1.00               | 20.95             | 1.00      | 21        |
| $2^{20}$               | 1.00      | 20.75            | 1.00               | 20.85             | 1.00      | 20.9      |
| $2^{22}$               | 1.00      | 20.65            | 1.00               | 20.65             | 1.00      | 20.7      |
| $2^{24}$               | 1.00      | 20.65            | 1.00               | 20.65             | 1.00      | 20.65     |
| Average solution 21.16 |           | 21.16            |                    | 21.12             |           | 21.19     |
|                        | 4>        | 4×4 Rubik's solv | er results with wi | n probability abo | ove 0.5   |           |
| $2^{16}$               | 0.80      | 61.88            | 0.55               | 60.18             | 0.85      | 64.12     |
| $2^{18}$               | 1.00      | 58.1             | 0.85               | 56.7              | 0.95      | 58.47     |
| $2^{20}$               | 1.00      | 54.7             | 1.00               | 55.3              | 1.00      | 54.7      |
| 22                     | 1.00      | 51.9             | 1.00               | 52.8              | 1.00      | 52.3      |
| $2^{24}$               | 1.00      | 50.4             | 1.00               | 50.6              | 1.00      | 49.6      |
| Average solution 55.4  |           | 55.4             |                    | 55.11             |           | 55.84     |

train set as a compromise between solver performance and training time.

The third experiment analyzed the influence of the number of agents A on the solver's efficiency. We used models No.8, 16, and 25 for this experiment from Table II. We trained each of these models multiple times during 8192 epochs. Then, each model was integrated into a dedicated agent. Due to computation limitations, we run only two agents in parallel at the same time, assuming that with more available GPU instances, it will be possible to compute all of them simultaneously. Finally, the total number of pretrained agents for  $3\times3\times3$ was 26,  $4\times4\times4-29$ , and  $5\times5\times5-69$ . As in previous experiments, the agents aimed to solve  $3\times3\times3$  cubes were tested on the scrambles DeepCubeA dataset, while the rest were verified using the 2023 Kagle Santa Challenge dataset. Due to the large size of the DeepCubeA dataset, in the third experiment, we used a subset of 69  $3\times3\times3$ scrambles, which were considered most difficult during preliminary research. The results of the experiment are shown in Figure 3.

The authors of [3], along with the well-known Deep-CubeA dataset, were using DeepCubeA<sub>h</sub> set containing the scrambles that are furthest away from the goal state, assuming these scrambles are more challenging to solve. At the same time, original DeepCubeA solutions were robustly and optimally solving them. During the current research, we found another subset of the DeepCubeA dataset containing 16 scrambles, which were not solved optimally during the experimental studies. We believe that a significantly rising number of agents will lead to finding solutions to all of them. However, the first element of this subset (scramble No.17 from orig-

inal DeepCubeA) was never optimally solved in any of our attempts, even during preliminary research and experiments not covered by this paper. We believe that analyzing the scrambles in this subset will help us understand why they are so hard to solve compared to the rest of the DeepCubeA data. Finally, this understanding will lead to the development of new, more efficient ML methods. [62] Thus, we decided to publish these 16 scrambles as a self-contained dataset accompanied by this paper.

The experimental results listed in Table I were achieved by solving all the scrambles from the corresponding dataset defined in the third column of Table I using the proposed solver. The key solver parameters are listed in the fourth column. The last column of Table II demonstrates for which results from Table I each model was used.

The last experiment conducted in the current research was aimed to compare computational efficiency with the EfficientCube [36] (a state-of-the-art solution claimed by its author to have better efficiency than DeepCubeA). For the fairness of comparison, we set up a virtual machine with the following resources: AMD EPYC 7513 32core Processor running at 2.6GHz; 240 GB RAM; 250 GB NFS file storage; a single dedicated GPU NVIDIA A100-SXM 80GB. The virtual machine ran Red Hat Enterprise Linux version 8.7 and CUDA version 11.8. The latest version (March 10th, 2024) of the EfficientCube was downloaded from the official GitHub repository [63] and configured according to the author's instructions to reproduce the results from [36]. Our solution was installed on the same virtual machine and configured with the same beam width of  $2^{18}$ . First, we sequentially trained a model for each solution and measured the time required for these procedures. Then, we tested both solvers on all the scrambles from the DeepCubeA dataset. We recorded the solving time for each scramble and then averaged it among the whole dataset. Finally, we compared training time and average solving time between EfficientCube and our solution. During this experiment, only one solution was running on the virtual machine at the same time.

#### V. ACKNOWLEDGMENTS

We express our gratitude to Graviton for providing servers for research in the field of AI.

A.C. is deeply grateful to M. Douglas for his interest in this work, engaging discussions, and invitation to present preliminary results at the Harvard CMSA program on "Mathematics and Machine Learning" in Fall 2024, to M. Gromov, S. Nechaev, and V. Rubtsov for their invitation to make a talk at "Representations, Probability, and Beyond: A Journey into Anatoly Vershik's World" at IHES, as well as for stimulating discussions. A.C. is grateful to J. Mitchel for involving into the Kaggle Santa 2023 challenge, from which this project originated and to M. Kontsevich, Y. Soibelman, A. Soibelman, S. Gukov, A. Hayat, T. Smirnova-Nagnibeda, D. Osipov, V. Kleptsyn, G. Olshanskii, A. Ershler, J. Ellenberg, G. Williamson, A. Sutherland, Y. Fregier, P.A. Melies, I. Vlassopoulos, F. Khafizov, A. Zinovyev, H. Isambert for the discussions, interest and comments.

K.K. is grateful to T. Ruzaikin, A. Fokin, and M. Goloshchapov for fruitful discussions.

We are deeply grateful to the many colleagues who have contributed to the CayleyPy project at various stages of its development, including: S.Fironov, A.Lukyanenko, A. Abramov, A. Ogurtsov, A. Trepetsky, A. Dolgorukova, S. Lytkin, S. Ermilov, L. Grunvald, A. Eliseev, G. Annikov, M. Evseev, F. Petrov, N. Narynbaev, S. Nikolenko, S. Krymskii, R. Turtayev, S. Kovalev, N. Rokotyan, G. Verbyi, L. Shishina, A. Korolkova, D. Mamaeva, M. Urakov, A. Kuchin, V. Nelin, B. Bulatov, F. Faizullin, A. Aparnev, O. Nikitina, A. Titarenko, U. Kniaziuk, D. Naumov, A. Krasnyi, S. Botman, R. Vinogradov, D. Gorodkov, I. Gaiur, I. Kiselev, A. Rozanov, K. Yakovlev, V. Shitov, E. Durymanov, A. Kostin, R. Magdiev, M. Krinitskiy, P. Snopov.

#### VI. DATA AVAILABILITY

The weights and datasets used in the experimental studies are openly available on Zenodo [64].

The dataset containing a subset of the 16 most difficult scrambles from the DeepCubeA dataset that were not solved optimally with our approach is available at [65].

#### VII. CODE AVAILABILITY

The source code used to perform experimental studies in current research is available on GitHub [66].

Notebooks related to the CayleyPy project development are available on Kaggle [67], [68].

- E. Rubik, Cubed: The Puzzle of Us All (Orion Publishing Co, Hachette, UK, 2020).
- [2] S. Van Dulken, Inventing the 20th Century: 100 Inventions that Shaped the World from the Airplane to the Zipper (NYU Press, New York, USA, 2002).
- [3] F. Agostinelli, S. McAleer, A. Shmakov, and P. Baldi, Solving the Rubik's cube with deep reinforcement learning and search, Nature Machine Intelligence 1, 356 (2019).
- [4] OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang, Solving rubik's cube with a robot hand, arXiv preprint arXiv:1910.07113 10.48550/arXiv.1910.07113 (2019).
- [5] R. E. Korf, Linear-time disk-based implicit graph search, Journal of the ACM (JACM) 55, 1 (2008).
- [6] N. R. Sturtevant and M. J. Rutherford, Minimizing writes in parallel external memory search, Proceedings of the Twenty-Third International Joint Conference on Artificial Intelligence IJCAI '13, 666–673 (2013).
- [7] C. Petit and J.-J. Quisquater, Rubik's for cryptographers, Notices of the American Mathematical Society 60,

- 733 (2013).
- [8] K. Loukhaoukha, J.-Y. Chouinard, and A. Berdai, A secure image encryption algorithm based on rubik's cube principle, Journal of Electrical and Computer Engineering 2012, 173931 (2012).
- [9] Y.-R. Chen and C.-L. Lee, Rubik's cube: An energy perspective, Physical Review E 89, 012815 (2014).
- [10] A. Gower, O. Hart, and C. Castelnovo, Saddlesto-minima topological crossover and glassiness in the rubik's cube, arXiv preprint arXiv:2410.14552 10.48550/arXiv.2410.14552 (2024).
- [11] D. Joyner, Adventures in group theory: Rubik's cube, merlin's machine, and other mathematical toys (2008).
- [12] C. Cornock, Teaching group theory using rubik's cubes, International Journal of Mathematical Education in Science and Technology 46, 957 (2015).
- [13] E. J. Meinz, D. Z. Hambrick, J. J. Leach, and P. J. Boschulte, Ability and Nonability Predictors of Real-World Skill Acquisition: The Case of Rubik's Cube Solving, Journal of Intelligence 11, 18 (2023).
- [14] M. Gromov, Geometric group theory: Asymptotic invariants of infinite groups (1993).
- [15] T. Tao, Expansion in finite simple groups of lie type (2015).

- [16] S. Hannenhalli and P. A. Pevzner, Transforming men into mice (polynomial algorithm for genomic distance problem), Proceedings of IEEE 36th annual foundations of computer science, 581 (1995).
- [17] S. Hannenhalli and P. A. Pevzner, Transforming cabbage into turnip: polynomial algorithm for sorting signed permutations by reversals, Journal of the ACM (JACM) 46, 1 (1999).
- [18] J. Wilson, M. Bechler-Speicher, and P. Veličković, Cayley graph propagation, arXiv preprint arXiv:2410.03424 10.48550/arXiv:2410.03424 (2024).
- [19] L. Bulteau and M. Weller, Parameterized algorithms in bioinformatics: an overview, Algorithms 12, 256 (2019).
- [20] S. B. Akers and B. Krishnamurthy, A group-theoretic model for symmetric interconnection networks, IEEE transactions on Computers 38, 555 (1989).
- [21] G. Cooperman, L. Finkelstein, and N. Sarawagi, Applications of cayley graphs, Applied Algebra, Algebraic Algorithms and Error-Correcting Codes: 8th International Conference, AAECC-8 Tokyo, Japan, August 20–24, 1990 Proceedings 8, 367 (1991).
- [22] M.-C. Heydemann, Cayley graphs and interconnection networks, Graph symmetry: algebraic methods and applications, 167 (1997).
- [23] S. Hoory, N. Linial, and A. Wigderson, Expander graphs and their applications, Bulletin of the American Mathematical Society 43, 439 (2006).
- [24] G. Zémor, Hash functions and cayley graphs, Designs, Codes and Cryptography 4, 381 (1994).
- [25] F. J. Ruiz, T. Laakkonen, J. Bausch, M. Balog, M. Barekatain, F. J. Heras, A. Novikov, N. Fitzpatrick, B. Romera-Paredes, J. van de Wetering, et al., Quantum circuit optimization with alphatensor, arXiv preprint arXiv:2402.14396 10.48550/arXiv.2402.14396 (2024).
- [26] R. S. Sarkar and B. Adhikari, Quantum circuit model for discrete-time three-state quantum walks on cayley graphs, Physical Review A 110, 012617 (2024).
- [27] I. Dinur, M.-H. Hsieh, T.-C. Lin, and T. Vidick, Good quantum ldpc codes with linear time decoders, Proceedings of the 55th annual ACM symposium on theory of computing, 905 (2023).
- [28] O. L. Acevedo, J. Roland, and N. J. Cerf, Exploring scalar quantum walks on cayley graphs, arXiv preprint quant-ph/0609234 10.26421/QIC8.1-2-5 (2006).
- [29] D. Gromada, Some examples of quantum graphs, Letters in Mathematical Physics 112, 122 (2022).
- [30] S. Even and O. Goldreich, The minimum-length generator sequence problem is np-hard, Journal of Algorithms 2, 311 (1981).
- [31] E. D. Demaine, S. Eisenstat, and M. Rudoy, Solving the rubik's cube optimally is np-complete, arXiv preprint arXiv:1706.06708 10.48550/arXiv.1706.06708 (2017).
- [32] L. Bulteau, G. Fertin, and I. Rusu, Pancake flipping is hard, Journal of Computer and System Sciences 81, 1556 (2015).
- [33] S. Linton, Gap: groups, algorithms, programming, ACM Communications in Computer Algebra 41, 108 (2007).
- [34] R. Holbrook, W. Reade, and A. Howard, Santa 2023 the polytope permutation puzzle, https://kaggle.com/competitions/santa-2023 (2023), kaggle.
- [35] S. Hirata, Graph-theoretical estimates of the diameters of the Rubik's Cube groups (2024), arXiv preprint arXiv:2407.12961.
- [36] K. Takano, Self-Supervision is All You Need for Solv-

- ing Rubik's Cube, Transactions on Machine Learning Research (2023).
- [37] M. R. Douglas, Machine learning as a tool in theoretical science, Nature Reviews Physics 4, 145 (2022).
- [38] G. Lample and F. Charton, Deep learning for symbolic mathematics, arXiv preprint arXiv:1912.01412 10.48550/arXiv.1912.01412 (2019).
- [39] A. Davies, P. Veličković, L. Buesing, S. Blackwell, D. Zheng, N. Tomašev, R. Tanburn, P. Battaglia, C. Blundell, A. Juhász, et al., Advancing mathematics by guiding human intuition with ai, Nature 600, 70 (2021).
- [40] J. Bao, Y.-H. He, E. Hirst, J. Hofscheier, A. Kasprzyk, and S. Majumder, Polytopes and machine learning, arXiv preprint arXiv:2109.09602 10.48550/arXiv.2109.09602 (2021).
- [41] B. Romera-Paredes, M. Barekatain, A. Novikov, M. Balog, M. P. Kumar, E. Dupont, F. J. Ruiz, J. S. Ellenberg, P. Wang, O. Fawzi, et al., Mathematical discoveries from program search with large language models, Nature 625, 468 (2024).
- [42] T. Coates, A. Kasprzyk, and S. Veneziale, Machine learning detects terminal singularities, Advances in Neural Information Processing Systems 36 (2024).
- [43] A. Alfarano, F. Charton, and A. Hayat, Global lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers, arXiv preprint arXiv:2410.08304 10.48550/arXiv.2410.08304 (2024).
- [44] F. Charton, J. S. Ellenberg, A. Z. Wagner, and G. Williamson, Patternboost: Constructions in mathematics with a little help from ai, arXiv preprint arXiv:2411.00566 10.48550/arXiv.2411.00566 (2024).
- [45] A. Shehper, A. M. Medina-Mardones, B. Lewandowski, A. Gruen, P. Kucharski, and S. Gukov, What makes math problems hard for reinforcement learning: a case study, arXiv preprint arXiv:2408.15332 10.48550/arXiv.2408.15332 (2024).
- [46] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al., Mastering the game of go with deep neural networks and tree search, nature 529, 484 (2016).
- [47] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, et al., Mastering the game of Go without human knowledge, Nature 550, 354 (2017).
- [48] S. McAleer, F. Agostinelli, A. Shmakov, and P. Baldi, Solving the rubik's cube with approximate policy iteration, International Conference on Learning Representations (2019).
- [49] The precise quantification of requirements for feature vectors that would ensure the successful operation of the proposed method is challenging. We aim to demonstrate its efficiency in the context of Rubik's group cases. On one extreme, even random vectors suffice if the training data covers all nodes an idea employed in well-known approaches such as DeepWalk [69] and Node2vec [70]. However, our focus is different: only a small subset of nodes will be covered by the training data (random walks). The key point is the ability of the neural network to generalize from that small subset to the entire graph something that is impossible with random features. Worse, the feature vectors are related to the distance between nodes on a graph more training data is required, and more advanced parameters and resources should be

- used at all steps of the proposed method. The role of the neural network is to transform the initial feature vectors into a latent representation, where nodes that are closer on the graph are also closer in the latent space.
- [50] R. Świta and Z. Suszyński, Solving Full N× N× N Rubik's Supercube Using Genetic Algorithm, International Journal of Computer Games Technology 2023, 2445335 (2023).
- [51] All the solvers presented in the Table I managed to solve all the scrambles from the listed datasets.
- [52] S. Hirata, Probabilistic estimates of the diameters of the Rubik's Cube groups, arXiv preprint arXiv:2404.07337 10.48550/arXiv.2404.07337 (2024).
- [53] J. Mulholland, Permutation puzzles: a mathematical perspective, Department Of mathematics Simon fraser University (2016).
- [54] https://www.kaggle.com/code/marksix/ visualize-allowed-moves.
- [55] N. Alon, I. Benjamini, E. Lubetzky, and S. Sodin, Non-backtracking random walks mix faster, Communications in Contemporary Mathematics 9, 585 (2007).
- [56] J. Hale, C. Dyer, A. Kuncoro, and J. Brennan, Finding syntax in human encephalography with beam search, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2727 (2018).
- [57] L. Huang, H. Zhang, D. Deng, K. Zhao, K. Liu, D. A. Hendrix, and D. H. Mathews, Linearfold: linear-time approximate rna folding by 5'-to-3'dynamic programming and beam search, Bioinformatics 35, i295 (2019).
- [58] H. Scheidl, S. Fiel, and R. Sablatnig, Word beam search: A connectionist temporal classification decoding algorithm, 2018 16th International conference on frontiers in handwriting recognition (ICFHR), 253 (2018).
- [59] M. Freitag and Y. Al-Onaizan, Beam search strategies for neural machine translation, Proceedings of the First Workshop on Neural Machine Translation, 56 (2017).

- [60] R. Pryzant, D. Iter, J. Li, Y. T. Lee, C. Zhu, and M. Zeng, Automatic prompt optimization with" gradient descent" and beam search, arXiv preprint arXiv:2305.03495 10.48550/arXiv.2305.03495 (2023).
- [61] M. Musolesi, Creative beam search: Llm-as-a-judge for improving response generation, Proc. of the 15th International Conference on Computational Creativity (ICCC'24) (2024).
- [62] A possible explanation is that the number of optimal solution paths for such cubes is lower than average or equal to one, making these paths more difficult to find.
- [63] https://github.com/kyo-takano/efficientcube.
- [64] A. Chervov, K. Khoruzhii, N. Bukhal, J. Naghiyev, Z. Vladislav, I. Koltsov, L. Cheldieva, A. Sychev, A. Lenin, M. Obozov, E. Urvanov, and A. Romanov, A machine learning approach that beats large rubik's cubes. weights and datasets, 10.5281/zenodo.14886876 (2025).
- [65] A. Chervov, K. Khoruzhii, and A. Romanov, A machine learning approach that beats large rubik's cubes. a subset of deepcubea's dataset hardest scrambles., 10.5281/zenodo.14887459 (2025).
- [66] https://github.com/k1242/cayleypy-cube.
- [67] https://www.kaggle.com/datasets/alexandervc/ growth-in-finite-groups.
- [68] https://www.kaggle.com/competitions/ lrx-oeis-a-186783-brainstorm-math-conjecture/ code
- [69] B. Perozzi, R. Al-Rfou, and S. Skiena, Deepwalk: Online learning of social representations, Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, 701 (2014).
- [70] A. Grover and J. Leskovec, node2vec: Scalable feature learning for networks, Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining, 855 (2016).