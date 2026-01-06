## Goal-Aware Neural SAT Solver

Emils Ozolins, Karlis Freivalds, Andis Draguns, Eliza Gaile, Ronalds Zakovskis, Sergejs Kozlovics Institute of Mathematics and Computer Science University of Latvia

ozolinsemils@gmail.com, karlis.freivalds@lumii.lv

Abstract—Modern neural networks obtain information about the problem and calculate the output solely from the input values. We argue that it is not always optimal, and the network's performance can be significantly improved by augmenting it with a query mechanism that allows the network at run time to make several solution trials and get feedback on the loss value on each trial. To demonstrate the capabilities of the query mechanism, we formulate an unsupervised (not depending on labels) loss function for Boolean Satisfiability Problem (SAT) and theoretically show that it allows the network to extract rich information about the problem. We then propose a neural SAT solver with a query mechanism called QuerySAT and show that it outperforms the neural baseline on a wide range of SAT tasks.

#### I. INTRODUCTION

Boolean Satisfiability Problem (SAT) is a significant NP-complete problem with numerous practical applications — product configuration, hardware verification, and software package management, to name a few. There is no single efficient algorithm that solves every SAT problem, but heuristics have been developed that are sufficient for solving many real-life tasks.

Neural networks have demonstrated superior performance on many complex tasks, e.g., image and language processing, protein folding, and playing board and computer games. So we may expect that, in the long run, neural networks could also replace handcrafted heuristics for SAT problems.

There are attempts for applying neural networks to solve SAT by integrating them in contemporary solvers or replacing them altogether. [1] proposes training NeuroSAT architecture with single-bit supervision, predicting whether the SAT instance is satisfiable. Yet, it does not directly produce variable assignments and requires labels for the training that may be troublesome to obtain. [2] introduces unsupervised loss for training a neural network to solve a related Circuit-SAT problem without requiring labels. They use differentiable constraint relaxation to evaluate the network output and penalise the network for violating constraints. That leads to successful training for Circuit-SAT problems.

Usually, neural networks are trained by minimizing the loss which is obtained at the final layer. We ask whether it is beneficial to let the network know the loss it is currently producing and adapting its output correspondingly. That is not possible with the traditional supervised losses since labels are not available at the inference time. Yet, such an approach is sound for an unsupervised loss and, indeed, provides an excellent basis for designing a neural SAT solver.

![](_page_0_Picture_10.jpeg)

Fig. 1: The proposed query mechanism works by producing a query, evaluating it using an unsupervised loss function, and passing the resulting value back to the neural network for interpretation. It allows the model to obtain the structure and meaning of the solvable instance and information about the expected model output. The same unsupervised loss can be used for evaluating the query and for training.

In this paper, we introduce a step-wise recurrent neural SAT solver that at each step comes up with a query of variable assignments, evaluates it with an unsupervised loss, and updates its state based on the evaluation results. At every step, it outputs a target solution that may contain information from all the queries performed so far.

We prove that the proposed query mechanism employing unsupervised loss function as query evaluator can provide the neural network with information about (a) satisfiability status of a solution if queried at integer points, (b) reveal the structure of the problem instance if queried at fractional (real) points. That provides rich information about the structure, meaning of the problem, and its solution.

The proposed architecture is evaluated on several SAT problems, and it outperforms the neural baseline on all of them. We also compare it with classical SAT solvers (Glucose 4 and GSAT) on 3-SAT and SHA-1 preimage attack tasks. Appendix of this paper is available on https://github.com/LUMII-Syslab/QuerySAT/blob/master/appendix.pdf.

### II. QUERY MECHANISM

We hypothesize that allowing a neural network to make several solution trials at the runtime and getting feedback about them can significantly improve network capabilities in some cases. It is easy to implement almost for any neural network, yet depending on the unsupervised loss function, some networks can be especially suitable for such augmentation.

The simplest such implementation (depicted in Fig. 1) consists of two different neural network layers. The first layer  $(NN_A)$  is given the current state  $(s_r)$  and it outputs a query (q) and some hidden state (h). We then use the unsupervised loss function (loss) to evaluate the query and obtain evaluation results (e). It is important to use a loss function that does not require labels for evaluation, as they may not be available at the test time. The evaluation results (e) together with the hidden state (h) are then passed as the input for the second layer  $(NN_B)$  that is responsible for interpreting the evaluated query and producing the next state  $(s_{r+1})$  and output logits (l). We can calculate the gradient of the evaluation results (e) with respect to query (q). The gradient can be optionally added as the third input for the second layer to show the network the direction for decreasing the loss [3]. The query mechanism is conceptually shown in Fig. 1 and can be defined as follows:

$$q, h = NN_A(s_r)$$

$$e = loss(q)$$

$$s_{r+1}, l = NN_B(h, e, \nabla_q e)$$

$$\mathcal{L} = loss(l).$$

Calculating output logits and loss  $\mathcal{L}$  at every step is not noteworthy per se but merely demonstrates that the same loss function can be used for both – query evaluation and model training. The training loss in principle could be evaluated only once – at the network's final layer and can be different from the loss used for the query evaluation. Nevertheless, it is essential not to merge the loss (and logits) used for query with those used for training. Although constructed similarly, their meaning and functioning are different (see Appendix C).

#### III. QUERY MECHANISM FOR SAT

We use the Boolean Satisfiability Problem (SAT) as the testbed for validating the benefits of the query mechanism. To this end, we propose an unsupervised loss function for SAT problems in conjunctive normal form and theoretically show that the query mechanism is indeed beneficial – it gives the model access to the problem structure and contributes to the performance of the model.

#### A. Boolean Satisfiability Problem

Boolean Satisfiability Problem (SAT) questions whether there exists an interpretation (True or False assignments to the variables) that satisfies the given Boolean formula. We undertake a related problem – finding a set of variable assignments that satisfies the given Boolean formula. Throughout the paper, we stick to common practice [4] and represent SAT formulas solely in a conjunctive normal form (CNF) – conjunction of one or more clauses where a clause is a disjunction of literals.

Any SAT formula can be naturally represented as a bipartite variables-clauses graph, likewise known as an SAT factor graph [4]. In such a graph, the edge between variables and clauses graph exists whenever the variable is present in the clause. Two types of edges are used to distinguish a variable from its negation. Variables-clauses graph of SAT formula with n

variables and m clauses can be represented as a sparse  $n \times m$  adjacency matrix. In order to perform batching, several SAT instances can be placed into a single factor graph yielding a single adjacency matrix for the whole batch.

#### B. Unsupervised SAT loss

A common way to train neural networks is by using a supervised loss, such as cross-entropy, which matches the network outputs with the correct labels. But such an approach does not work for training variable assignment for SAT due to several possible satisfying assignments for a single SAT instance. Also, obtaining labels involve SAT solving, which is time-consuming for large instances. Moreover, as we want to integrate the loss function into the neural network as a query mechanism, it has to be differentiable and cannot rely on labels as they are not available at the test time.

Therefore, we design a differentiable unsupervised loss function that directly optimizes towards finding a satisfiable variable assignment of the Boolean formula without knowing a correct solution. To this end, we relax the Boolean domain to continuous variables in the range [0,1] where 0 corresponds to False and 1 to True. The vector of all variable values in the assignment is denoted by x. The value  $V_c$  for a clause c is obtained by multiplying together negations of all literals in the clause and negating the result and the loss value  $\mathcal{L}_{\phi}$  for a formula  $\phi$  by multiplying all the clause values of  $\phi$  together:

$$V_c(x) = 1 - \prod_{i \in c^+} (1 - x_i) \prod_{i \in c^-} x_i,$$
  
$$\mathcal{L}_{\phi}(x) = \prod_{c \in \phi} V_c(x),$$

where  $x_i$  is the value of *i*-th variable and  $c^+$  gives the set of variables that occur in the clause c in the positive form and  $c^-$  in the negated form. The values  $V_c(x)$  and  $\mathcal{L}_{\phi}(x)$  are equal to 1 if and only if x is a satisfying assignment of clause c or formula  $\phi$ , respectively, and strictly smaller otherwise. So by maximizing  $\mathcal{L}_{\phi}(x)$  we can hope to find a satisfying assignment to the variables. But this function is not usable in practice since it often yields zero value in machine precision due to multiplying together many clause values, which are all less than one. Therefore, we use the negative logarithm of  $\mathcal{L}_{\phi}$  and minimize it:

$$\mathcal{L}_{\phi}^{\log}(x) = -\log(\mathcal{L}_{\phi}(x)) = -\sum_{c \in \phi} \log(V_c(x)).$$

Taking the logarithm of the loss does not change its minimum/maximum structure, so minimizing  $\mathcal{L}_{\phi}^{\log}$  is equivalent of maximizing  $\mathcal{L}_{\phi}$ .

A model trained by minimizing  $\mathcal{L}_{\phi}^{\log}$  produces "soft" variable assignments in the range [0,1]. We experimentally observe that when the network can find a solution, it is close to binary and rounding the values to the nearest integer produces excellent results.

Other works [5], [6], albeit in different contexts, have also proposed relaxing SAT formulas to continuous truth values similar to  $\mathcal{L}_{\phi}$  loss (one not in the log-space) but faces the same problems as  $\mathcal{L}_{\phi}$  if used for learning SAT solutions directly. Therefore, log-space formulation  $\mathcal{L}_{\phi}^{\log}$  of the loss trades capabilities of modelling general Boolean formulas for a loss function that gives non-zero loss for large CNF formulas and can be directly applied for learning their solutions.

#### C. Power of the query

The proposed loss function, when used in the query mechanism, gives the satisfiability status of a solution if queried at the binary points 0 or 1 and reveals the structure of the formula if queried at the intermediate (real) points. To maximize the amount of information gained by the query and to match the internal structure of Graph Neural Network (GNN), we employ queries that return the loss value  $V_c(x)$  for each clause at the query point x. The power of such query mechanism is formulated in the Theorem 1 and 2, respectively. The proofs of theorems are deferred to the Appendix A and B.

**Theorem 1.** For a binary query point x the losses  $\mathcal{L}_{\phi}(x)$  or  $V_c(x)$  are equal to 1 if the formula  $\phi$  or clause c is satisfied and 0 otherwise.

**Theorem 2.** A single query that returns the loss  $V_c$  for each clause c is sufficient to uniquely identify the SAT formula  $\phi$  to be solved.

The proof of theorem 1 follows immediately from the definitions of the losses. The proof of theorem 2 shows how to create a query point x in such a way that the literals making up the clauses can be uniquely decoded solely from the clause losses. The proof assumes that we know the variable and clause count of the formula, but this assumption can be relaxed by padding all the formulas to some fixed maximum size. A stronger result that a single query returning only the total loss  $\mathcal{L}_{\phi}$  (a single real number) might also be shown by using more intricate reasoning, but we did not pursue this direction since we use per-clause loss.

The construction used in theorem 2 assumes sufficiently high precision of the numbers, exceeding the limits of standard floating-point arithmetic. Regarding that, in our implementation, we issue several queries in parallel (up to 128) and organize the computation in multiple recurrent steps in which queries are performed repeatedly and one query can depend on the results of the previous one. Such design significantly enhances the power of queries and relieves the need for high precision.

## IV. QUERYSAT

We validate the theory in practice by building a neural SAT solver that we call QuerySAT and evaluate its performance on a wide range of SAT tasks - k-SAT, 3-SAT, 3-Clique, k-Coloring, and SHA-1 preimage attack. Experimental evaluation is performed on a single machine with 16GB RAM and a single Nvidia T4 GPU (16GB) using AdaBelief optimizer [7]. Code for reproducing experiments is implemented in TensorFlow and is available at https://github.com/LUMII-Syslab/QuerySAT.

#### A. Model

QuerySAT is based on a GNN employing the proposed query mechanism and unsupervised SAT loss function. It receives a CNF Boolean formula  $\phi$  in the input represented as two adjacency matrices of variables-clauses graphs -  $A_p \in \{0,1\}^{n \times m}$  and  $A_n \in \{0,1\}^{n \times m}$ , where n and m are the number of variables and clauses, respectively.  $A_p$  represents all positive variable mentions in the clauses, and  $A_n$  represents all negated variable mentions. The network outputs a vector  $out \in [0,1]^n \subseteq \mathbb{R}^n$  — a variable assignment. QuerySAT works in a step-wise manner with recurrent application of the following graph-based recurrent unit:

$$\begin{split} q_i &= \sigma(\text{MLP}_q(v_i,\ t)) \\ e_i &= \mathcal{V}_\phi(q_i) \\ c_{i+1} &= \text{PairNorm}(\text{MLP}_c(c_i,\ e_i)) \\ v_{i+1} &= \text{PairNorm}(\text{MLP}_v(v_i,\ A_pc_{i+1},\ A_nc_{i+1},\ \nabla_{q_i}e_i)) \\ out &= \sigma(\text{MLP}_q(v_{i+1})). \end{split}$$

At the beginning an empty state vector initialized with all ones is allocated for each variable and each clause and the unit is recurrently applied to them  $s_{train}$  steps at training and  $s_{test}$  steps at evaluation. At each step i starting from i=0, QuerySAT produces a query  $q_i \in \{0,1\}^{n\times d}$ , where d is the feature map count, by applying a 2-layer multi-layer perceptron MLP $_q$  to the variables state  $v_i \in \mathbb{R}^{n\times d}$ . A random noise vector  $t \in \mathbb{R}^{n\times r}$  is also given in the input to this MLP. The query is range-limited by the sigmoid function  $\sigma$  and evaluated by our unsupervised loss. Here we employ per-clause loss defined as  $\mathcal{V}_{\phi}(q_i) = \{V_c(q_i)) \mid c \in \phi\}$  which returns the vector of clause losses  $e_i$ .

We then obtain the new clauses state  $c_{i+1} \in \mathbb{R}^{m \times d}$  by applying a 2-layer perceptron MLP<sub>c</sub> and PairNorm [8] to the previous clauses state  $c_i$  and the query result  $e_i$ . Query mechanism thus replaces a variables-to-clauses message passing step that would be present in a classical message-passing architecture. Then, information aggregation from clauses-to-variables is performed by message-passing, which for some variable sums all the clause states in which the variable occurs. Positive and negative occurrences are treated separately and are implemented as a sparse matrix multiplication between the clause state  $c_{i+1}$  and occurrence matrices  $A_p$  and  $A_n$ .

The new variables state  $v_{i+1} \in \mathbb{R}^{n \times d}$  is then obtained by applying a 3-layer perceptron MLP<sub>v</sub> and PairNorm to the variables state  $v_i$ , aggregated clause messages and the query loss gradient with the respect to query  $q_i$ . The new variables state is then mapped to the output variable assignment  $out \in [0,1]^n$  using another 2-layer perceptron MLP<sub>o</sub> and sigmoid function  $\sigma$ . In all MLP layers we use LeakyReLU activation.

The model is trained using the loss function  $\mathcal{L}_{\phi}^{\log}$  proposed in section III-B. The loss is calculated from variable assignments out at each step, and the sum of all losses is minimized. Using the loss at each time-step has shown performance improvements [9], [10], [11] versus a single loss calculation at the end. Also, it enables using many more steps in evaluation than

![](_page_3_Figure_0.jpeg)

![](_page_3_Figure_1.jpeg)

![](_page_3_Figure_2.jpeg)

Fig. 3: Part of fully solved 3-SAT instances with 400 variables depending on the steps taken at test time. Models were trained with 32 recurrent steps on formulas with up to 100 variables.

![](_page_3_Figure_4.jpeg)

Fig. 4: Time vs fully solved instances trained with 32 recurrent steps on 3-SAT instances with 5-100 variables and then tested on instances with 5-405 variables.

in training. Without impacting the model's performance, we use a conditional exit at any step if the variable assignment satisfies the input formula.

Although giving an additional noise parameter t to  $\mathrm{MLP}_q$  seems like a minor detail, it serves several purposes. First, it avoids a blowup in the gradient during training, which may result from applying normalization to zero values. Secondly, it enables the model to learn a randomized algorithm by providing greater diversity between the queries, which is especially needed for obtaining good accuracy in testing with many steps. Thirdly, it allows the model to perform differently in several evaluations (we observed that in the experiments), mimicking restarts in the classical SAT solvers. We find that for our purposes  $t \in \mathcal{N}(0,1)^{n \times r}$ , where r=4, works well.

## B. Training tricks

In this section, we describe the model architecture and training tricks that are not per se important for the power of the proposed architecture yet slightly improves model performance.

**Gradient scaling.** As we calculate loss and minimize it at each step, large gradient values may accumulate in the backwards pass for the first-time steps. Such uneven gradient distribution slows down training, and to mitigate that, we introduce gradient scaling and apply it to the variables and clauses states. It works by downscaling the gradient in the backward pass at each time-step as follows:  $stop\_gradient(x)\alpha+x(1-\alpha)$ , where x is value and  $\alpha \in [0,1]$  is a hyperparameter. We find that for QuerySAT with 32 recurrent steps,  $\alpha=0.2$  works the best.

**Multi-assignment loss.** Instead of returning a single variable assignment, we let the model return several  $out \in [0,1]^{n \times u}$ , where  $u \in \mathbb{Z}^+$  is hyperparameter. The u=8 works best for us. For each assignment, we calculate loss value using the proposed loss function  $\mathcal{L}_{\phi}^{\log}$  and then obtain the final step loss as a weighted sum of assignment losses. Weighting is done as follows - we sort assignments losses in descending order and enumerate them from 1 to u. Let's call each such number the loss index. Then we calculate the final loss as the sum of each loss value multiplied with its squared index

and then dividing the sum with the sum of squared indices. From our observations, such multi-assignment loss only gives improvements when used with unsupervised loss as the model in the early stages of training is free to explore various outputs. The assignment with the lowest loss value (before weighting) is promoted as the final model assignment.

#### C. Evaluation

We evaluate the QuerySAT model on several SAT tasks, which are standard benchmarks for classical SAT solvers, and the majority of them are not an easy feat for neural solvers. Namely, we chose k-SAT, 3-SAT, and also 3-Clique, k-Coloring, and SHA-1 preimage attack problems represented as CNF Boolean formulas. All datasets consists only of satisfiable formulas. The QuerySAT architecture is compared to the derivative of NeuroSAT [1] that was used for predicting unsatisfiable cores by [12] but is generally applicable to any variables-wise predictions on SAT factor graph. Further on, we refer to this NeuroSAT derivative as NeuroCore. The results are also compared to the GSAT and Glucose 4 classical solvers. For all tasks, we generate a train set of 100k formulas and validation and test sets of 10k formulas each. The 3-SAT and SHA-1 tasks are an exception as their test sets consist of 40k and 5k formulas, respectively. For most tasks, we use larger formulas in the test set to evaluate the generalization capability of the model.

The k-SAT task is taken from [1] – each clause in the n variable formula is generated by sampling a small integer k (size of the clause), and then randomly without replacement taking k of the n variables. Each variable in the clause is then negated with 50% probability. The resulting clauses consist of roughly four variables on average. The train and validation sets consist of formulas with 3 to 100 variables, but the test set – with 3 to 200 variables.

Graph and 3-SAT tasks are generated using the CNFGen library [13], which allows encoding several popular problems as SAT instances. We generate hard 3-SAT instances at the satisfiability boundary where the relationship between the number of clauses (m) and variables (n) is  $m=4.258n+58.26n^{-\frac{2}{3}}$ 

TABLE I: Mean test accuracy (higher is better) as per cent of fully solved instances from the test set over 3 consecutive runs. Both models were trained with 32 recurrent steps for 500k training iterations and then tested with 32, 512, and 4096 recurrent steps. Value after  $\pm$  indicates the standard error.

| Task       | QuerySAT         |                  |                   | NeuroCore        |                  |                   |
|------------|------------------|------------------|-------------------|------------------|------------------|-------------------|
|            | $s_{test} = 32$  | $s_{test} = 512$ | $s_{test} = 4096$ | $s_{test} = 32$  | $s_{test} = 512$ | $s_{test} = 4096$ |
| k-SAT      | $72.12 \pm 0.19$ | $96.61 \pm 0.78$ | $99.05 \pm 0.38$  | $21.64 \pm 0.27$ | $46.85\pm5.02$   | $50.82 \pm 6.41$  |
| 3-SAT      | $61.89 \pm 5.19$ | $88.20 \pm 4.01$ | $93.32 \pm 3.21$  | $28.38 \pm 3.24$ | $53.49 \pm 3.94$ | $57.63 \pm 4.38$  |
| 3-Clique   | $82.00 \pm 4.73$ | $93.06 \pm 4.67$ | $94.74 \pm 4.62$  | $1.03 \pm 0.69$  | $1.03 \pm 0.66$  | $1.04 \pm 0.66$   |
| k-Coloring | $91.70 \pm 1.01$ | $97.76 \pm 0.98$ | $98.32 \pm 0.82$  | $0.0 \pm 0.0$    | $0.0 \pm 0.0$    | $0.0 \pm 0.0$     |
| SHA-1      | $33.25 \pm 4.17$ | $46.57 \pm 1.16$ | $46.45 \pm 1.10$  | $0.00 \pm 0.0$   | $0.27 \pm 0.09$  | $0.24 \pm 0.09$   |

[14]. The train and validation sets of 3-SAT tasks consist of formulas with 5 to 100 variables, but test set – with 5 to 405 variables.

For the graph-based tasks, we generate Erdős–Rényi graphs with edge probability p. For the 3-Clique task, where the main goal is to find all triangles in the graph, we use  $p=3^{\frac{1}{3}}/(v(2-3v+v^2))^{\frac{1}{3}}$ , where v is the vertex count in the graph. For the k-Coloring task we use  $p=\frac{(1+0.2)\ln v}{v}+0.05$  and the goal is to color the graph with at least k colours. Such generation produces mostly sparse connected graphs that are colourable using 3 to 5 colours. The graphs are then encoded as SAT instances using the CNFGen library. For both tasks, train and validation sets consist of graphs with 4 to 40 vertices, but the test set – with 4 to 100 vertices.

We also experiment with the SHA-1 preimage attack task from the SAT Race 2019 [15]. The goal of this task is to find the message value given the hash value. The original SAT competition task uses the SHA-1 algorithm with 17 rounds and asks the solver to find the first up to 160 message bits. Such configuration produces SAT instances at the threshold of satisfiability, and it is expected that only a single solution exists. That is a challenging task even for modern SAT solvers; hence, we use a smaller set-up to find the first 2 to 20 bits of the message. But even then, it is still a moderately hard task. We generate instances for this task using the CGen tool [16].

To make the QuerySAT and NeuroCore architectures comparable, we train NeuroCore with the same per-step loss and apply the same bag of tricks that we used for QuerySAT, as described in Sections IV-A and IV-B. NeuroCore is also given a chance to return results in any step if the correct solution has been found. The rest of the architecture is left intact. For both models, we use 128 feature maps that produce similarly sized models and train them with a batch size of 20000 nodes (max node count in the input factor graph), 32 recurrent steps and a learning rate  $2 \times 10^{-4}$ . Hyperparameters are selected by performing a grid search by hand. On the SHA-1 preimage attack, models are trained for 1M iterations, but on the rest of the tasks for 500k iterations. Afterwards, models are tested with 32, 512, and 4096 recurrent steps and the mean accuracy as a percentage of fully solved instances over 3 consecutive runs is depicted in Table I. We see that QuerySAT with 4096 steps performs consistently the best for all the tasks. It can solve

3-Clique, k-Coloring and SHA-1 tasks on which NeuroCore produces an accuracy of almost zero.

Detailed comparison of both architectures is conducted on the 3-SAT task, where we check step-wise and formula-wise generalization of both models. To evaluate both properties, we use models trained with 32 recurrent steps on formulas with 3 to 100 variables. Fig. 2 depicts generalization to harder formulas (up to 400 variables) with 32 and 4096 recurrent steps at the test time. Fig. 3 shows step-wise generalization by changing the number of steps  $s_{test}$  from 4 up to 131k when testing on the same formulas with 400 variables. QuerySAT outperforms NeuroCore on both generalization tasks by a wide margin and, with 32 steps, has a similar performance to the NeuroCore with 4096 steps. QuerySAT's performance increases with the step count at the test time, although it is trained only with 32 steps. NeuroCore, on the contrary, plateaus at approximately 16k steps and 40% accuracy. In Fig. 4, the cactus plot is shown comparing NeuroCore, and QuerySAT tested with 4096 recurrent steps on 3-SAT instances with 5 to 405 variables. Cactus plot shows cumulative time spend for solving instances (y-axis) versus cumulative solved instance count (x-axis). More solved instances in less time indicate better model performance. At the same time interval, QuerySAT solves  $\sim 36k$  formulas, while NeuroCore solves only  $\sim 18k$  from the total of 40k 3-SAT formulas in the test set (see Fig. 4).

To show current capabilities of QuerySAT, we also compare it to GSAT [17] and Glucose 4 [18], [19] classical solvers. GSAT is a widely known incomplete local-search solver, and Glucose is a contemporary conflict-driven clause learning solver. As the QuerySAT is also an incomplete solver, it's directly comparable to the GSAT algorithm. Comparison to Glucose is not on equal ground since Glucose can certify that some instance is UNSAT while QuerySAT runs indefinitely for such instance. That said, we present their comparison nonetheless to give a rough idea of their relative performance. Note that both - GSAT and Glucose 4 - may not accurately represent the performance of modern state-of-the-art SAT solvers. All solvers are evaluated on the same hardware but note that QuerySAT utilizes a single Nvidia T4 GPU while the others do not. We configure all three solvers to have approximately a 2-second time limit by giving QuerySAT 1024 recurrent steps, GSAT – 500k steps, and setting a 2-second timeout for Glucose solver.

![](_page_5_Figure_0.jpeg)

Fig. 5: Cactus plot representing the number of solved SHA-1 preimage attack instances vs. used time for QuerySAT, GSAT and Glucose 4 solvers.

![](_page_5_Figure_2.jpeg)

Fig. 7: Error as per cent of unsolved instances in the validation set depending on training iteration for the k-SAT task. The validation set consists of 10k formulas with 3-100 variables.  $s_{test} = 64$  is used for validation.

We compare these solvers on previously described test sets of 3-SAT and SHA-1 tasks. The QuerySAT was trained on the forenamed train sets. The results are depicted as cactus plots in the Fig. 5 and 6. Glucose utilizes the structure of the problem and therefore struggles on random 3-SAT instances yet solves the SHA-1 task in almost no time. Even though GSAT outperforms QuerySAT on 3-SAT instances, QuerySAT seems to utilize the SHA-1 structure and achieves better performance than GSAT. QuerySAT, similarly to other contemporary end-to-end neural solvers [1], [10], in the general case, is outperformed by classical solvers and requires breakthroughs to allow scaling them to large industrial instances.

#### V. EVALUATING THE QUERY MECHANISM

We evaluate the impact of the query mechanism by augmenting NeuroCore architecture with it and measuring the improvement. It is straightforward to do since NeuroCore is similar to QuerySAT. NeuroCore uses literals-to-clauses and clauses-to-literals message-passing to update the internal states for literals and clauses. Therefore, we add a query mechanism alongside the literals-to-clauses message-passing and also give the gradient of the evaluated query to the MLP that updates the literals state. We experiment with two variants of the query

![](_page_5_Figure_7.jpeg)

Fig. 6: Cactus plot representing the number of solved 3-SAT instances vs. used time for QuerySAT, GSAT and Glucose 4 solvers.

![](_page_5_Figure_9.jpeg)

Fig. 8: Error as per cent of unsolved instances in the validation set depending on training iteration for the 3-Clique task. The validation set consists of 10k graphs with 4-20 vertices.  $s_{test}=64$  is used for validation.

mechanism: with both query and gradient (+ Query + G) and only with query (+ Query). Both versions are compared with the standard NeuroCore architecture (NeuroCore).

We chose k-SAT and 3-Clique tasks (see section IV-C) for evaluation. We use the same dataset split as previously. For the k-SAT task, we use the same train and validation set as previously, but for the test set generate formulas with 100 to 200 variables. On the same note, train and validation sets for the 3-Clique task consists of graphs with 4 to 20 vertices but the test set of 20 to 40 vertices. The test set consists of harder formulas to see how various variants of query mechanism impacts generalization.

All three versions are trained for 100k iterations with the same training and network configuration as described in Section IV-C. The trained model is then tested with 32, 512, and 4096 recurrent steps. The mean results of 3 consecutive runs are depicted in Table II. We also include validation error in the train time for each version using 64 recurrent steps ( $s_{test}$ ), it is depicted in the Fig. 7 and 8.

In Table II, we can see that both versions with a query mechanism outperform the NeuroCore baseline. The version with a query mechanism and gradient trains faster and achieves better accuracy. Interestingly, adding a gradient significantly

TABLE II: Mean accuracy of NeuroCore and its augmentation with query and gradient and as per cent of fully solved instances from the test set over 3 consecutive runs. All models are trained with 32 recurrent steps and evaluated with 32, 512, and 4096 steps. Value after  $\pm$  indicates the standard error. Note that the training, validation and test instances are smaller than in Table I.

|               | k-SAT            |                  |                   | 3-Clique         |                  |                   |
|---------------|------------------|------------------|-------------------|------------------|------------------|-------------------|
|               | $s_{test} = 32$  | $s_{test} = 512$ | $s_{test} = 4096$ | $s_{test} = 32$  | $s_{test} = 512$ | $s_{test} = 4096$ |
| NeuroCore     | $38.86 \pm 1.04$ | $56.14 \pm 3.48$ | $60.97 \pm 6.19$  | $65.38 \pm 5.09$ | $67.12 \pm 3.25$ | $70.39 \pm 2.68$  |
| + Query       | $41.09 \pm 2.33$ | $61.21 \pm 9.89$ | $67.21 \pm 13.54$ | $62.90 \pm 2.55$ | $78.19 \pm 2.67$ | $84.55 \pm 2.93$  |
| + Query $+$ G | $47.92\pm1.66$   | $71.95 \pm 6.59$ | $75.50 \pm 7.15$  | $86.13 \pm 4.83$ | $93.96 \pm 2.59$ | $95.50 \pm 1.95$  |

improves the model's accuracy of the 3-Clique task. We reason that gradient is helpful for the tasks that represented as a SAT instance has a distinct structure. Such structure is very expressive for the 3-Clique task, yet, on the contrary, k-SAT doesn't have any distinct structure as it is sampled from a uniform distribution [20].

#### VI. RELATED WORK

Neural networks have been proposed as an effective alternative for automatically developing heuristic algorithms for NP-hard problems [21], [22]. Two main research directions are replacing handcrafted heuristics with a neural network in a classical solver and building end-to-end neural solvers.

[1] proposed a Graph Neural Network (GNN) architecture called NeuroSAT that is trained using single-bit supervision to predict whether the Boolean formula is satisfiable or not. The variable assignment is obtained from the last layer by performing 2-clustering on the output, but the network is never optimized for producing the assignment explicitly. Hence, it is not clear why such an approach should work. In a later work, [12] simplified NeuroSAT architecture (they call it NeuroCore) and used it for guiding high-performance solvers (e.g., MiniSAT, Glucose) by predicting how likely each variable is in the unsatisfiable core. [23] use Q-learning to train similar GNN for predicting branching heuristics in MiniSAT solver. Others [24], [23] similarly augment classical solvers with neural networks to solve 2-Quantified Boolean Formulas and logic locked circuits. On the same trend, [20] trained a GNN using reinforcement learning to learn a local-search heuristic that finds a solution similar to the GSAT / WalkSAT algorithm by flipping a single variable in each step. It is also theoretically shown that GNN can learn to mimic the WalkSAT algorithm [25].

Supervised and reinforcement learning are not the best options for solving SAT, as the variable assignment is time-consuming to obtain, many possible solutions exist, and training may be slow. [2] proposed a differentiable unsupervised approach for directly solving Circuit-SAT. For this purpose, they represent variables as real numbers in the range [0,1] and use GNN for producing variable predictions on the Circuit-SAT graph. The predicted variables are evaluated as Circuit-SAT, but the AND and OR functions are substituted with differentiable Softmax and Softmin functions. In a later work [10], they have applied the same method for solving SAT problems. To encourage exploration, they introduce a temperature parameter that is annealed in the training time,

making training cumbersome. Although the training method is closely related to ours, their loss function is more complex. [26] lately proposed to represent Boolean functions by multilinear polynomials and find a solution to a single formula by using gradient descent optimization. Even though their loss function is similar to ours, they solve the MAX-SAT problem and do not directly optimize towards finding a solution to the SAT problem.

#### VII. CONCLUSIONS

In this paper, we have proposed a query mechanism that allows the neural network to make several solution trials, obtain the loss of each trial and change its strategy accordingly. To evaluate the impact of the query mechanism, we propose an unsupervised SAT loss and integrate it with queries to form the QuerySAT architecture. We find that QuerySAT outperforms the message-passing neural baseline on all proposed tasks: k-SAT, 3-SAT, 3-Clique, k-Coloring, and SHA-1 preimage attack. Experiments show that query mechanism can significantly increase the performance of message-passing graph neural networks. To give a better insight into the current capabilities of QuerySAT, we also compare it with classical solvers on 3-SAT and SHA-1 preimage attack tasks. Although we have analyzed only SAT solving, we expect a similar benefit from the query mechanism on other neural architectures and tasks. Since QuerySAT employs unsupervised loss, not requiring to know the labels, it provides rich opportunities for integration with classical solvers.

#### ACKNOWLEDGMENT

We would like to thank the IMCS UL Scientific Cloud for the computing power and Leo Trukšāns for the technical support. This research is supported by Google Cloud and funded by the Latvian Council of Science, projects No. lzp-2018/1-0327, lzp-2021/1-0479.

## REFERENCES

- [1] D. Selsam, M. Lamm, B. Bünz, P. Liang, L. de Moura, and D. L. Dill, "Learning a SAT solver from single-bit supervision," *arXiv preprint arXiv:1802.03685*, 2018.
- [2] S. Amizadeh, S. Matusevych, and M. Weimer, "Learning To Solve Circuit-SAT: An Unsupervised Differentiable Approach," in *International Conference on Learning Representations*, 2019. [Online]. Available: https://openreview.net/forum?id=BJxgz2R9t7
- [3] M. Andrychowicz, M. Denil, S. Gomez, M. W. Hoffman, D. Pfau, T. Schaul, B. Shillingford, and N. De Freitas, "Learning to learn by gradient descent by gradient descent," arXiv preprint arXiv:1606.04474, 2016.

- [4] A. Biere, M. Heule, and H. van Maaren, *Handbook of satisfiability*. IOS press, 2009, vol. 185.
- [5] G. Ryan, J. Wong, J. Yao, R. Gu, and S. Jana, "Cln2inv: Learning loop invariants with continuous logic networks," in *International Conference on Learning Representations*, 2020. [Online]. Available: https://openreview.net/forum?id=HJlfuTEtvB
- [6] M. Fischer, M. Balunovic, D. Drachsler-Cohen, T. Gehr, C. Zhang, and M. Vechev, "DL2: Training and querying neural networks with logic," in Proceedings of the 36th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri and R. Salakhutdinov, Eds., vol. 97. PMLR, 09–15 Jun 2019, pp. 1931–1941. [Online]. Available: https://proceedings.mlr.press/v97/fischer19a.html
- [7] J. Zhuang, T. Tang, Y. Ding, S. Tatikonda, N. Dvornek, X. Papademetris, and J. S. Duncan, "Adabelief optimizer: Adapting stepsizes by the belief in observed gradients," arXiv preprint arXiv:2010.07468, 2020.
- [8] L. Zhao and L. Akoglu, "PairNorm: Tackling oversmoothing in GNNs," arXiv preprint arXiv:1909.12223, 2019.
- [9] R. B. Palm, U. Paquet, and O. Winther, "Recurrent relational networks," arXiv preprint arXiv:1711.08028, 2017.
- [10] S. Amizadeh, S. Matusevych, and M. Weimer, "PDP: A General Neural Framework for Learning Constraint Satisfaction Solvers," arXiv preprint arXiv:1903.01969, 2019.
- [11] E. Ozoliņš, K. Freivalds, and A. Šostaks, "Matrix Shuffle-Exchange Networks for Hard 2D Tasks," 2020.
- [12] D. Selsam and N. Bjørner, "Guiding High-Performance SAT Solvers with Unsat-Core Predictions," 2019.
- [13] M. Lauria, J. Elffers, J. Nordström, and M. Vinyals, "CNFgen: A generator of crafted benchmarks," in *International Conference on Theory* and Applications of Satisfiability Testing. Springer, 2017, pp. 464–473.
- [14] J. M. Crawford and L. D. Auton, "Experimental results on the crossover point in random 3-SAT," *Artificial intelligence*, vol. 81, no. 1-2, pp. 31–57, 1996.
- [15] V. Skladanivskyy, "Minimalistic Round-reduced SHA-1 Pre-image Attack," in SAT RACE 2019: Solver and Benchmark Descriptions, ser. B-2019-1, vol. 1. Department of Computer Science Series of Publications B, University of Helsinki, 2019, pp. 51–52.
- [16] ——, "Tailored compact cnf encoding for sha-1," J. Satisf. Boolean Model. Comput, 2020.
- [17] B. Selman, H. Levesque, and D. Mitchell, "A new method for solving hard satisfiability problems," in AAAI'92 Proceedings of the tenth national conference on Artificial intelligence, 1992, pp. 440–446.
- [18] G. Audemard and L. Simon, "On the glucose SAT solver," *International Journal on Artificial Intelligence Tools*, vol. 27, no. 01, p. 1840001, 2018.
- [19] N. Eén and N. Sörensson, "An extensible SAT-solver," in *International conference on theory and applications of satisfiability testing*. Springer, 2003, pp. 502–518.
- [20] E. Yolcu and B. Póczos, "Learning Local Search Heuristics for Boolean Satisfiability." in *NeurIPS*, 2019, pp. 7990–8001.
- [21] Y. Bengio, A. Lodi, and A. Prouvost, "Machine learning for combinatorial optimization: a methodological tour d'horizon," *European Journal of Operational Research*, 2020.
- [22] Q. Cappart, D. Chételat, E. Khalil, A. Lodi, C. Morris, and P. Veličković, "Combinatorial optimization and reasoning with graph neural networks," arXiv preprint arXiv:2102.09544, 2021.
- [23] V. Kurin, S. Godil, S. Whiteson, and B. Catanzaro, "Improving SAT solver heuristics with graph networks and reinforcement learning," arXiv preprint arXiv:1909.11830, 2019.
- [24] K. Z. Azar, H. M. Kamali, H. Homayoun, and A. Sasan, "NNgSAT: Neural network guided SAT attack on logic locked complex structures," in 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD). IEEE, 2020, pp. 1–9.
- [25] Z. Chen and Z. Yang, "Graph neural reasoning may fail in certifying boolean unsatisfiability," arXiv preprint arXiv:1909.11588, 2019.
- [26] A. Kyrillidis, A. Shrivastava, M. Vardi, and Z. Zhang, "FourierSAT: A Fourier Expansion-Based Algebraic Framework for Solving Hybrid Boolean Constraints," in *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 34, 2020, pp. 1552–1560.
- [27] D. Polymath, "Variants of the Selberg sieve, and bounded intervals containing many primes," *Research in the Mathematical sciences*, vol. 1, no. 1, pp. 1–83, 2014.

## APPENDIX A PROOF OF THEOREM 1

**Theorem 1.** For a binary query point x the losses  $\mathcal{L}_{\phi}(x)$  or  $V_c(x)$  are equal to 1 if the formula  $\phi$  or clause c is satisfied and 0 otherwise.

*Proof.* The idea is to give proof in two steps. First, we prove that when a binary query satisfies the formula, then formula value  $\mathcal{L}_{\phi}(x)$  and all clauses values  $V_c(x)$  should be 1. Secondly, when a binary query does not satisfy the formula, at least one clause value and formula value should be 0.

Firstly, assume that the query corresponds to a satisfying assignment, then all clauses are satisfied. That implies that in each clause, there is at least one positive literal with its query value being 1 or a negative literal with its query value being 0. Therefore in each clause loss  $V_c(x) = 1 - \prod_{i \in c^+} (1 - x_i) \prod_{i \in c^-} x_i$  at least one of the elements of the product is 0. That implies that all clause losses  $V_c(x)$  are equal to 1. Therefore the loss  $\mathcal{L}_\phi(x) = \prod_{c \in \phi} V_c(x)$  is 1.

Secondly, assume the query corresponds to an unsatisfying assignment, then there is at least one clause c that is unsatisfied. This implies that in the unsatisfied clause c the query value for all positive literals is 0 and for all negative literals it is 1. Therefore in the clause loss  $V_c(x) = 1 - \prod_{i \in c^+} (1 - x_i) \prod_{i \in c^-} x_i$  all elements of the product are 1. This implies that for the unsatisfied clause the corresponding clause loss  $V_c(x)$  is 0. Therefore the loss  $\mathcal{L}_\phi(x) = \prod_{c \in \phi} V_c(x)$  is 0, since one of the elements of the product is 0.

# APPENDIX B PROOF OF THEOREM 2

**Theorem 2.** A single query that returns the loss  $V_c$  for each clause c is sufficient to uniquely identify the SAT formula  $\phi$  to be solved.

*Proof.* The idea is to choose the query input values based on primes such that the clause loss contains an irreducible fraction with the numerator and denominator consisting of prime factors which point to the variables that appear in the clause. From the measured clause loss value, it is possible to infer the corresponding irreducible fraction, and from this fraction, we can infer the variables and the signs of their corresponding literals in the clause.

An element of the query  $x = (x_1, x_2, ..., x_n) \in \mathbb{R}^n$  corresponds to a value of a variable queried at an intermediate (real) point between 0 and 1, where n is the number of variables. Each query element is obtained by using a pair from the series of prime pairs where the gap between the two primes in the pair is a fixed constant H. For H = 2 the series would be pairs of twin primes: (3,5),(5,7),(11,13),... It has been proven that there is a constant H between 2 and 246 such that the corresponding prime pair series contains an infinite number of elements [27]. We fix a concrete value of H for constructing the query.

For constructing the query from the series, we take the first n pairs  $(a_1, b_1), (a_2, b_2), ..., (a_n, b_n)$  that fulfill the following

criteria:  $\forall_i \ a_i > H$  and  $\forall_{i,j} \ a_i \neq b_j$ . Note that  $b_i - H = a_i$ . We set the query x as  $(H/b_1, H/b_2, ..., H/b_n)$ .

Let us examine the clause loss  $V_c(x) = 1 - \prod_{i \in c^+} (1 - x_i) \prod_{i \in c^-} x_i$  corresponding to a clause c and the query x with i positive and k-i negative literals. To simplify the notation, we assume without a loss of generality that the first k variables appear in the clause and that the positive literals have lower indices than negative literals.

$$\begin{split} V_c(x) &= 1 - (1 - x_1)(1 - x_2) \dots (1 - x_i)x_{i+1}x_{i+2} \dots x_k = \\ &= 1 - \left(1 - \frac{H}{b_1}\right)\left(1 - \frac{H}{b_2}\right)\dots\left(1 - \frac{H}{b_i}\right)\frac{H}{b_{i+1}}\frac{H}{b_{i+2}}\dots\frac{H}{b_k} = \\ &= 1 - \frac{b_1 - H}{b_1}\frac{b_2 - H}{b_2}\dots\frac{b_i - H}{b_i}\frac{H}{b_{i+1}}\frac{H}{b_{i+2}}\dots\frac{H}{b_k} = \\ &= 1 - \frac{a_1}{b_1}\frac{a_2}{b_2}\dots\frac{a_i}{b_i}\frac{H}{b_{i+1}}\frac{H}{b_{i+2}}\dots\frac{H}{b_k} = 1 - \frac{a_1a_2\dots a_iH^{k-i}}{b_1b_2\dots b_k} \end{split}$$

The indices of the primes  $b_1b_2...b_k$  in the denominator of the fraction in the obtained expression correspond to the indices of variables appearing in the clause c. The indices of the primes  $a_1a_2...a_i$  in the numerator correspond to the indices of variables appearing as positive literals in the clause.

The primes  $a_1a_2 \dots a_i$  and  $b_1b_2 \dots b_k$  are non-overlapping and the prime factors of H are smaller than any prime  $a_i$  or  $b_i$  due to the pair selection criteria. This implies that the fraction in the representation of the clause loss that we obtained is irreducible.

We prove that given the constructed query x, the clause loss function is injective. To show that, we take a clause c' that is different from our arbitrarily chosen clause c. The clause c' being different from clause c implies that it will differ in at least one literal, and therefore the irreducible fraction in the corresponding clause losses  $V'_c(x)$  and  $V_c(x)$  will differ in  $a_1 a_2 \dots a_i$  or  $b_1 b_2 \dots b_k$ . Due to the fundamental theorem of arithmetic, there is only one way to write a number as a product of its prime factors (up to the order of the factors). The numerator or the denominator of the fraction in the losses will differ because their prime factors will differ. Therefore any two different clauses will produce clause losses that each contain a different irreducible fraction. Since each rational number can be written as an irreducible fraction in exactly one way, the two clause losses are different rational numbers, and hence the clause loss function is injective for the constructed query.

Therefore the clause can be identified from a query and its corresponding loss for the clause. Since the instance  $\phi$  to be solved is uniquely represented by its constituent clauses, it can be identified from a query and a set containing a clause loss for each of its clauses.

## APPENDIX C QUERY RESULTS

To analyze how the model uses queries and what information they contain, we train NeuroCore with the query mechanism on the 3-SAT task for 100k iterations as described in the section V. At each recurrent step, query and logit values are discretized by applying the sigmoid function and then rounding values to the closest integer (0 or 1). Fig. 10 reveals how many clauses the

![](_page_9_Figure_0.jpeg)

![](_page_9_Figure_1.jpeg)

![](_page_9_Figure_2.jpeg)

Fig. 9: Match between query and logits (higher value means more similar) of the same recurrent step depending on the training iteration on the 3-SAT task.

Fig. 10: Per cent of clauses satisfied by query depending on the training iteration on the 3-SAT task.

Fig. 11: Element-wise match (higher values indicate greater similarity) between two queries in consecutive steps depending on the training iteration on the 3-SAT task.

discretized query satisfies at each step in the training time. Fig. 9, on the other hand, depicts an element-wise match between discretized queries and logits as per cent of identical variables from the total variable count. In Fig. 9, we can see that the queries have different values from the logits of the same step and from the Fig. 10 we can see that they never satisfy given Boolean formula, strengthening our belief that queries are not used to output the true prediction but instead to obtain rich information about the problem.

We also validate that model produces different queries at each recurrent step, therefore obtaining rich information about the problem. Fig. 11 depicts the match as per cent of equal elements from the total element count between two discretized queries of two consecutive steps. As reasoned in section III-B, query mechanism can extract more information about the problem by issuing several different queries at each step and accumulating this knowledge. In Fig. 11, we can see the difference between the last and second last query increase with the training time. That indicates that the model learns to issue more different queries, increasing the obtained information about the problem.