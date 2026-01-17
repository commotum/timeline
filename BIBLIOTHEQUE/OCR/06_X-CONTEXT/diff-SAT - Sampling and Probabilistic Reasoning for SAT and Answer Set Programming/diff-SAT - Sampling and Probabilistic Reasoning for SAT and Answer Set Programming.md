# diff-SAT - A Software for Sampling and Probabilistic Reasoning for SAT and Answer Set Programming

## Matthias Nickles

School of Computer Science National University of Ireland, Galway matthias.nickles@nuigalway.ie

## Abstract

This paper describes diff-SAT, an Answer Set and SAT solver which combines regular solving with the capability to use probabilistic clauses, facts and rules, and to sample an optimal world-view (multiset of satisfying Boolean variable assignments or answer sets) subject to user-provided probabilistic constraints. The sampling process minimizes a user-defined differentiable objective function using a gradient descent based optimization method called Differentiable Satisfiability Solving ( $\partial$ SAT) respectively Differentiable Answer Set Programming ( $\partial$ ASP). Use cases are i.a. probabilistic logic programming (in form of Probabilistic Answer Set Programming), Probabilistic Boolean Satisfiability solving (PSAT), and distribution-aware sampling of model multisets (answer sets or Boolean interpretations).

#### Keywords:

Boolean Satisfiability, Answer Set Programming, Probabilistic Satisfiability, Probabilistic Programming, Gradient Descent

# 1. Introduction

Modern solvers for the Boolean Satisfiability Problem (SAT) are powerful, versatile and fast tools for logical inference. SAT solvers underlie a large number of other optimization and reasoning approaches as well as industry applications, for example many constraint programming and Satisfiability Modulo Theories (SMT) solvers, bounded model checking, package managers, hardware and software verification tools, crypto-analysis, and planning and scheduling approaches [13, 8, 9]. Answer Set Programming (ASP) [5] is a form of logic programming (with syntax similar to Prolog and Datalog) which is closely related to SAT solving and oriented primarily towards

NP-hard search and combinatorial problems.

Regular SAT and Answer Set solvers do not have the capability to handle or reason about probabilistic knowledge, or to sample multisets of models governed by specific probability distributions. diff-SAT (formerly named del-SAT) is an open source SAT and ASP solver which has those capabilities. diff-SAT is programmed in Scala and runs on any contemporary standard Java VM. Its solver core incorporates modern SAT solving techniques such as CDCL (Conflict-Driven Clause Learning)-style solving, Glucose-style restart policy [1], adaptive branching heuristics, rephasing, and parallel portfolio solving in multi-core environments.

Symbolic knowledge can be provided in various formats, including Boolean formulas in CNF and logic programs in Answer Set Programming (ASP) intermediate format (obtained from an ASP grounder). Soft constraints can be provided by attaching probabilities to clauses, facts or rules, or by specifying differentiable objective functions directly as mathematical expressions. In contrast to most existing logic programming approaches, probabilistic databases and approaches based on a reduction to Weighed Model Counting [3], diff-SAT does not require any independence assumptions about probabilistic variables. Compared to Markov Logic Networks [12] (the arguably most popular probabilistic logic), diff-SAT allows for probabilistic logic programming and has thus the ability to deal with inductive definitions and to use probabilities directly as weights despite being able to model complex interdependencies.

diff-SAT's output is a multiset of models (if the given problem has a solution) corresponding to a probability distribution over possible worlds, allowing for probabilistic inference in the usual way.

On the technical level, diff-SAT achieves its functionality by finding optimal distributions of entire multi-sets of satisfying models (related to the concept of "worldviews" in epistemic logic programming [6]). An incremental sampling process minimizes a user-defined objective function (loss function) over symbol statistics in model multisets using a gradient descent-based branching literal selection approach called Differentiable Satisfiability Solving respectively Differentiable Answer Set Programming [10, 11] ( $\partial$ SAT,  $\partial$ ASP).

Probabilistic annotations of logical items like clauses or rules can be automatically translated into such differentiable functions. Multi-models optimization could be stylized as a form of model optimization where the optimal model is a meta-model which comprises multiple models, but diff-SAT avoids

this stylization as it would require clunky techniques like model reification. Remark: Multi-models optimization should not be confused with other optimization types involving multiple models, such as the concurrent discovery of several optimal models.

In contrast to existing approaches to ASP- or SAT-based optimization such as MaxWalkSAT for Weighted MaxSAT solving [7], the differentiable objective function evaluates and optimizes statistics over an entire multiset of models instead of finding an optimal single model which is optimal wrt. the sum of the weights of satisfied soft constraints (soft clauses). Compared to other SAT sampling approaches, in particular [2], diff-SAT samples multiple models using a form of gradient descent until a given minimization threshold of the objective function has been reached. Clauses can be attached probabilities directly instead of non-normalized weights. Note that diff-SAT is not a Stochastic Local Search (SLS) solver but a complete solver (i.e., a solver which returns UNSAT if the input formula is not satisfiable) with probabilistic optimization features.

A more detailed description of diff-SAT's algorithm and comparison with further related approaches can be found in [11, 10].

## 2. Functionality and Input

Notwithstanding being based on a single, relatively simple technical principle ( $\partial SAT / \partial ASP$ ), diff-SAT goes beyond regular SAT/ASP solving by providing a wide range of additional use cases (besides plain SAT and ASP solving), including probabilistic SAT (PSAT), Probabilistic Answer Set Programming (PrASP), and distribution-dependent model sampling where a family of target distributions can be specified by differentiable objective functions of the form described in the previous section.

diff-SAT is open source. The source code, recent releases and documentation can be found at https://github.com/MatthiasNickles/diff-SAT.

## Features include:

- Parallel portfolio ASP and SAT solving based on Conflict-Driven Nogood Learning (CDNL), a variant of CDCL)
- Runs on the JVM (Java 1.8 or higher)
- Model sampling as randomized, gradient-steered search for optimal multisets of models (answer sets respectively Boolean assignments)

- Objective functions are arbitrary differentiable functions over atom frequencies in models, allowing for, e.g., probabilistic atoms (clauses, rules) and probabilistic inference, i.e., computation of the probabilities of logical query formulas.
- A range of alternative input formats, including for Probabilistic SAT and Probabilistic Answer Set Programming. See below for the complete list of supported formats.
- User API for working with probabilistic and non-probabilistic clause and rule sets
- Direct support for basic non-ground ASP rules (unrestricted non-ground answer set programs can also be used, but that requires a preceding grounding step using a grounder such as Gringo or Clingo [4], as usual with Answer Set solvers)

Input is accepted in various standard as well as non-standard forms (where no commonly accepted format exists yet, in particular in the area of probabilistic input):

- DIMACS CNF (for regular SAT solving)
- Probabilistic DIMACS CNF (PCNF) where each clause can optionally be annotated with a probability to turn it into a soft clause
- Enhanced DIMACS CNF with a list of parameter atoms and objective function terms appended to the DIMACS part
- ASP Intermediate Format (ASPIF), optionally extended with parameter atoms and objective function terms defined using special predicates
- Enhanced ASPIF format with a list of parameter atoms and objective function terms appended to the ASPIF part
- Probabilistic ASP Intermediate Format (PASPIF) which enhances AS-PIF standard format with optional rule probabilities

diff-SAT's output for satisfiable problems is a multiset of models (possible worlds) where the normalized count of a model represents its probability. The probabilities of individual clauses, facts or rules can be obtained using a simple subsequent filtering and counting step as in most other probabilistic logic programming approaches, but diff-SAT can optionally perform this step itself for user-specified query clauses, facts and rules.

## Code metadata

| Nr. | Code metadata description             |                                  |
|-----|---------------------------------------|----------------------------------|
| C1  | Current code version                  | 0.5.2                            |
| C2  | Link to code/repository used for this | https://github.com/              |
|     | code version                          | MatthiasNickles/diff-SAT         |
| C4  | Code License                          | MIT                              |
| C5  | Code versioning system used           | git                              |
| C6  | Software code languages, tools, and   | Scala, Java, JVM                 |
|     | services used                         |                                  |
| C7  | Compilation requirements, operat-     | JDK 1.8+ (e.g., OpenJDK), SBT or |
|     | ing environments & dependencies       | Maven                            |
| C8  | If available link to developer docu-  | https://github.com/              |
|     | mentation/manual                      | MatthiasNickles/diff-SAT         |
| С9  | Support email for questions           | (see author contact details)     |

Table 1: Code metadata

### References

- [1] Gilles Audemard and Laurent Simon. Predicting learnt clauses quality in modern sat solvers. *IJCAI International Joint Conference on Artificial Intelligence*, pages 399–404, 07 2009.
- [2] Supratik Chakraborty, Daniel J. Fremont, Kuldeep S. Meel, Sanjit A. Seshia, and Moshe Y. Vardi. Distribution-aware sampling and weighted model counting for SAT. In Carla E. Brodley and Peter Stone, editors, Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Québec City, Québec, Canada, pages 1722–1730. AAAI Press, 2014.
- [3] Mark Chavira and Adnan Darwiche. On probabilistic inference by weighted model counting. *Artificial Intelligence*, 172(6):772 799, 2008.
- [4] Martin Gebser, Roland Kaminski, Benjamin Kaufmann, and Torsten Schaub. Answer Set Solving in Practice. Synthesis Lectures on Artificial Intelligence and Machine Learning. Morgan & Claypool Publishers, 2012.
- [5] M. Gelfond and V. Lifschitz. The stable model semantics for logic programming. In *Proc. of the 5th Int'l Conference on Logic Programming*, volume 161, 1988.

- [6] Patrick Thor Kahl and Anthony P. Leclerc. Epistemic Logic Programs with World View Constraints. In Alessandro Dal Palu', Paul Tarau, Neda Saeedloei, and Paul Fodor, editors, Technical Communications of the 34th International Conference on Logic Programming (ICLP 2018), volume 64 of OpenAccess Series in Informatics (OASIcs), pages 1:1–1:17, Dagstuhl, Germany, 2018. Schloss Dagstuhl–Leibniz-Zentrum fuer Informatik.
- [7] Henry Kautz, Bart Selman, and Yueyen Jiang. A general stochastic approach to solving problems with hard and soft constraints. In *The Satisfiability Problem: Theory and Applications*, pages 573–586. American Mathematical Society, 1996.
- [8] Frédéric Lafitte, Jorge Nakahara, and D. Heule. Applications of sat solvers in cryptanalysis: Finding weak keys and preimages. *Journal on Satisfiability, Boolean Modeling and Computation*, 9:1–25, 08 2014.
- [9] Joao Marques-Silva. Practical applications of boolean satisfiability. In 9th International Workshop on Discrete Event Systems, pages 74 80, 06 2008.
- [10] Matthias Nickles. Differentiable SAT/ASP. In Elena Bellodi and Tom Schrijvers, editors, Proceedings of the 5th International Workshop on Probabilistic Logic Programming, PLP 2018, co-located with the 28th International Conference on Inductive Logic Programming (ILP 2018), Ferrara, Italy, September 1, 2018, volume 2219 of CEUR Workshop Proceedings, pages 62–74. CEUR-WS.org, 2018.
- [11] Matthias Nickles. Sampling-based SAT/ASP multi-model optimization as a framework for probabilistic inference. In Fabrizio Riguzzi, Elena Bellodi, and Riccardo Zese, editors, *Inductive Logic Programming 28th International Conference, ILP 2018, Ferrara, Italy, September 2-4, 2018, Proceedings*, volume 11105 of *Lecture Notes in Computer Science*, pages 88–104. Springer, 2018.
- [12] Matthew Richardson and Pedro Domingos. Markov logic networks. *Machine Learning*, 62(1-2), 2006.
- [13] Ashish Sabharwal. Modern sat solvers: Key advances and applications. 2011.