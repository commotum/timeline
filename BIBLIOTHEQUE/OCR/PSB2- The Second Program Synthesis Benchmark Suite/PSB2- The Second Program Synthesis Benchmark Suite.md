# **PSB2: The Second Program Synthesis Benchmark Suite**

Thomas Helmuth Hamilton College Clinton, New York, USA thelmuth@hamilton.edu

# **ABSTRACT**

For the past six years, researchers in genetic programming and other program synthesis disciplines have used the General Program Synthesis Benchmark Suite to benchmark many aspects of automatic program synthesis systems. These problems have been used to make notable progress toward the goal of general program synthesis: automatically creating the types of software that human programmers code. Many of the systems that have attempted the problems in the original benchmark suite have used it to demonstrate performance improvements granted through new techniques. Over time, the suite has gradually become outdated, hindering the accurate measurement of further improvements. The field needs a new set of more difficult benchmark problems to move beyond what was previously possible.

In this paper, we describe the 25 new general program synthesis benchmark problems that make up PSB2, a new benchmark suite. These problems are curated from a variety of sources, including programming katas and college courses. We selected these problems to be more difficult than those in the original suite, and give results using PushGP showing this increase in difficulty. These new problems give plenty of room for improvement, pointing the way for the next six or more years of general program synthesis research.

### **CCS CONCEPTS**

Software and its engineering → Automatic programming.

### **KEYWORDS**

automatic program synthesis, benchmarking, genetic programming

# **ACM Reference Format:**

Thomas Helmuth and Peter Kelly. 2021. PSB2: The Second Program Synthesis Benchmark Suite. In 2021 Genetic and Evolutionary Computation Conference (GECCO '21), July 10–14, 2021, Lille, France. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3449639.3459285

# 1 INTRODUCTION

Automatic general program synthesis, with the aim of automatically generating programs of the type humans write from scratch, has long been a goal of artificial intelligence and machine learning.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

GECCO '21, July 10-14, 2021, Lille, France

© 2021 Association for Computing Machinery. ACM ISBN 978-1-4503-8350-9/21/07...\$15.00 https://doi.org/10.1145/3449639.3459285 Peter Kelly Hamilton College Clinton, New York, USA pxkelly@hamilton.edu

Yet, for many years there were no common benchmark problems for evaluating general program synthesis<sup>1</sup> systems; existing problems were either easy toy problems or were situated in specific domains where solution programs were composed of a small set of domain-specific instructions. In 2015, the General Program Synthesis Benchmark Suite (PSB1) introduced 29 problems that could be used to benchmark program synthesis systems [17]. Since then, more than 80 research papers have benchmarked 10+ program synthesis systems using PSB1, producing numerous insights into program synthesis.

Of the systems that have adopted PSB1, most fall within the field of genetic programming (GP), including PushGP [17], grammarguided GP [6], grammatical evolution [22], and linear GP [29]. However, non-evolutionary program synthesis methods have also been applied to PSB1, including those based on delayed-acceptance hill-climbing [44] and Monte Carlo tree search [31]. We expand on the details of these methods and the results they have achieved using PSB1 in Section 2, but to summarize, many of these systems have improved performance and demonstrated new techniques.

When PSB1 was first introduced, the initial PushGP runs were able to solve 22 of the 29 problems, with an average success rate of 23 successful runs out of 100 [17]. The best-performing PushGP results have now solved 25 problems, with an average success rate of 42/100 [18]. Some of the most drastic improvements have come on some of the most informative problems in PSB1, such as Double Letters (6  $\rightarrow$  50 successes between [17] and [18]), Replace Space with Newline (51  $\rightarrow$  100), Syllables (18  $\rightarrow$  64), Vector Average (16  $\rightarrow$  97), and X-Word Lines (8  $\rightarrow$  91).

Thus, for PushGP and other synthesis systems, the problems of PSB1 have become less useful over time. In particular, the very high performance achieved on some PSB1 problems leaves little room for exhibiting improvement; a few other problems have never been solved and are likely too difficult to be solved any time soon. Additionally, peculiarities in some of the problems in PSB1 make them less ideal as benchmarks, either because of how synthesis systems move through their search space or how slow they are to run. Finally, some decisions about the specification of problems in PSB1 make them difficult to implement in hindsight, potentially preventing wider adoption.

With these drawbacks in mind, we have created a second Program Synthesis Benchmark suite, which we refer to as PSB2. PSB2 consists of 25 problems curated from programming challenges, programming katas, and college courses. In order to facilitate the uptake of PSB2, we provide a reference implementation of each problem, as well as datasets that can be sampled to more easily implement each problem in new synthesis systems. <sup>2</sup> Just like PSB1,

 $<sup>^1\</sup>mathrm{Also}$  known as automatic programming or software synthesis.

 $<sup>^2</sup>Reference$  implementation, datasets, and other resources can be found on this paper's companion website: https://cs.hamilton.edu/~thelmuth/PSB2/PSB2.html.

the problems in PSB2 require a wide range of programming techniques, data types, and control flow structures to solve. However, they are markedly harder to solve than problems in PSB1, with our initial results solving 13 of the 25 problems for an average success rate of 10/100. These more difficult problems will drive program synthesis research toward solving more realistic program synthesis tasks.

The purpose of benchmark problems is to allow us to empirically show what changes to a system produce improvements that may transfer to real-world problems. To achieve this goal, they must be sufficiently difficult, unlike toy problems that have been used as benchmarks in the past. They must also be representative of the types of tasks we want our system to perform. However, we also want benchmarks to be easier and faster to run than an actual real-world problem in order to aid reasonable testing of a system. Given that automatic program synthesis is still in its fledgling stages, we see the problems in PSB2 as a stepping stone toward solving more realistic problems.

PSB2 also addresses calls from the GP community to produce and adopt realistic benchmarks. GP community discussions calling for better benchmarks [34, 59, 60] inspired the creation of PSB1; these calls also highlighted the need to periodically update and replace benchmark problems in order to keep advancing the field without over-optimizing to a single set of problems. More recently, a call to refocus the efforts of GP on automatic programming stated, "We are in no doubt of the need for the further principled development of additional benchmarks that can be used in a targeted manner to push the boundaries along different dimensions such as scalability, generalisation, and adaptation, and to facilitate comparison across a range of very different approaches to automatic programming" [40]. The creation of PSB2 aims to push the boundaries of program synthesis research and give synthesis systems a fresh set of problems to explore. Of course, there is no need to entirely throw out the problems of PSB1; we could imagine some of the harder problems continuing to provide useful data, and newer systems may need to start on the easier problems as a jumping off point.

The remainder of this paper is structured as follows: in the next section, we discuss research that has used PSB1. In Section 3, we highlight lessons learned about program synthesis benchmarking from PSB1. Sections 4 and 5 describe the sources of PSB2's problems and describe the problems in detail. We then give general guidance on benchmarking with PSB2, and give details of the parameters we used in our experiments in Sections 6 and 7. Finally, Section 8 presents initial results using PushGP.

# 2 PAST RESEARCH USING PSB1

PSB1 has been used in a variety of research projects on automatic program synthesis, many of them using GP as the synthesis system. The paper that introduced PSB1 [17] used PushGP, a GP system based on the stack-based Push programming language; a variety of papers using PushGP have made use of PSB1 since [14–16, 18, 21, 46, 47]. Code-building GP is a stack-based GP system borrowing some inspirations from Push that constructs programs in a host language; it solved some of the PSB1 problems, producing solution programs in Python [42].

General program synthesis requires the manipulation of multiple data types; stack-based GP systems have handled this requirement well, but so have other GP systems that handle strong typing of programs. In particular, grammar-based approaches such as grammar guided GP (G3P) [6–9] and grammatical evolution (GE) [22, 27, 39, 50] have made good progress at solving the problems in PSB1. Many of these use the type-based grammar design patterns introduced to flexibly handle problems with different type requirements [6]. Another use of these grammars trains a sequence-to-sequence variational autoencoder to embed programs in a continuous space and then uses an evolutionary algorithm to optimize programs in this space [32]. Finally, a linear GP system with tagbased memory has also been explored using PSB1 [5, 23, 29].

As for non-GP systems, an approach using delayed-acceptance hillclimbing for inductive synthesis proved competitive with GP on PSB1, including producing the only known solutions to the Collatz Numbers problem [44]. A comparison was made between Flash Fill [11], MagicHaskeller [26], PushGP, and G3P, finding that the non-GP methods fared much worse but ran much faster than the GP methods [41]. Finally, Monte Carlo tree search was used to generate Java bytecode programs using a few of the problems in PSB1 [31].

#### 3 LESSONS LEARNED

While PSB1 has been successfully used in a variety of research, it was a first attempt at a general-purpose program synthesis benchmark suite. The research community has grown from using it, both in terms of improving program synthesis methods as well as lessons learned about how to best define program synthesis benchmarks. Here, we discuss some issues of the latter type and how they have influenced our creation of PSB2.

One major issue with PSB1 is that every system that uses it needs to implement all of the problems from scratch. This hurdle likely decreased wider adoption. Additionally, there may be inconsistencies between implementations in different systems, leading to less comparable results; one known such inconsistency is that some systems use new randomized data for each run, while others use the same dataset for every run. Four years after its initial release, the authors of PSB1 created large datasets of the inputs and correct outputs for each problem [12]. These datasets can be sampled for each program synthesis run, meaning there is no need for each system to implement each problem. We have copied this model, and provide datasets for each problem in PSB2 (see Section 6).

A handful of the problems in PSB1 require programs to produce Boolean outputs, as such functions are common in programming exercises. A trend noted across completely different program representations is that solutions to these Boolean-output problems often do not generalize to unseen data. A simple explanation for this phenomenon is that it is relatively easy for a solution program to produce the correct answers for the wrong reasons when there are only two possible answers, thus overfitting to the training data. It is much harder to perfectly answer training data for the wrong reasons when the output is an integer or string, for example. Because of this issue, we have selected fewer Boolean-output problems for PSB2, including only one representative problem.

PSB1 was designed to emulate the textbook problems it was curated from as closely as possible. For example, many problems from the original textbook required the program to print its answers. PSB1 suggested that synthesis systems develop methods for emulating an output buffer and common printing instructions in order to mimic these problems. However, this approach was infeasible for some synthesis systems, which instead simply returned string outputs. As PSB2 is less loosely coupled with its problem sources, we decided to have all programs return their outputs instead of "printing" them. Another wrinkle related to outputs is that some problems require a solution to return multiple outputs. While multiple outputs may prove difficult in some systems, it is generally feasible in all; we have included 4 multi-output problems in PSB2.

PSB1 recommended different training and test set sizes, as well as program evaluation budgets, for each problem. This led to difficulties and confusion in both implementation and reporting results. For PSB2, we recommend using a fixed setting for each parameter across problems, as well as for system-specific parameters such as maximum sizes for program size control.

Forstenlechner et al. [9] discussed understanding and refining the problems in PSB1, making some general recommendations about both synthesis systems and benchmark problems. One suggestion put forth is using larger and more targeted training sets, to better guide synthesis and increase generalization. We take this recommendation and use large training sets (200 examples) that have a variety of specific edge cases purposefully included. Most of their other suggestions relate to specific system settings, such as the length of evolution; these parameters are not prescribed by PSB2, and can be chosen by the researcher.

#### 4 PROBLEM SELECTION AND SOURCES

Below we describe the four sources we used as inspiration for the problems included in this suite. Each of these sources presents problems for humans to use to improve their programming skills, whether for experienced programmers or students in class. As such, these sources contain problems representative of the types of programming that we expect humans to perform.

**Code Wars (CW)** - A website full of user-created programming challenges, called *coding kata*. The aim of the site is for users to spend small amounts of time programming every day to hone their coding skills.

**Advent of Code (AoC)** - An Advent calendar of coding problems created every year in December. These problems can be used for any number of things, like training, interview prep, or coursework. Problems tend to become harder throughout the month.

Homework Problems (HW) - These problems come from programming homework given in our undergraduate programming courses. The problems come from two courses: an introductory programming course, and a program languages course. These problems do not have citations, since we created them for our courses.

**Project Euler (PE)** - A website containing hundreds of problems in an archive. Users are free to submit answers to validate their solutions. Most problems tend to be mathematically focused and often require efficient and elegant solutions.

These sources contain a large number and variety of problems; we considered over 75 problems from these sources and implemented and tested over 50 of them, filtering out problems that seemed too easy, too difficult, or inaccessible. While curating the suite, we did not include any problems on which PushGP produced a success rate over 60%, to ensure that problems are sufficiently difficult to allow for improvement. In order to be transparent about our curation process, we have created a table containing all of the problems we considered, including the reason for rejection for rejected problems, initial results if we implemented the problem, and a link to the source of the problem.<sup>3</sup>

We aimed to include problems that require a large variety of data types and control flow structures to solve, with a balance between data types across problems. Most, if not all, of the problems require some type of iteration and/or conditional execution. Required data types include integers, floats, Booleans, characters, strings, vectors of integers, and vectors of floats. In order to produce large datasets, we aimed to select problems that have at least 1 million possible unique inputs; the only exception being the Coin Sums problem, which has 10,000 possible integer inputs.

# 5 PROBLEM DESCRIPTIONS

Below is a list of the English language descriptions of the 25 benchmark problems in PSB2. Each problem (besides those from our courses) has a citation of its source with a link to the original problem. The types of the input(s) and output(s) for each problem are given in Table 1. For more precise details of each problem, see the reference implementation.<sup>4</sup>

- Basement (AoC) Given a vector of integers, return the first index such that the sum of all integers from the start of the vector to that index (inclusive) is negative. [55]
- 2. **Bouncing Balls (CW)** Given a starting height and a height after the first bounce of a dropped ball, calculate the *bounciness index* (height of first bounce / starting height). Then, given a number of bounces, use the bounciness index to calculate the total distance that the ball travels across those bounces. [10]
- Bowling (CW) Given a string representing the individual bowls in a 10-frame round of 10 pin bowling, return the score of that round. [1]
- 4. Camel Case (CW) Take a string in kebab-case and convert all of the words to camelCase. Each group of words to convert is delimited by "-", and each grouping is separated by a space. For example: "camel-case example-test-string" → "camelCase exampleTestString". [25]
- 5. **Coin Sums (PE)** Given a number of cents, find the fewest number of US coins (pennies, nickles, dimes, quarters) needed to make that amount, and return the number of each type of coin as a separate output. [3]

 $<sup>^3</sup>$ https://docs.google.com/spreadsheets/d/e/2PACX-1vQKO1D2sZA9KosXpOJNuiDW6yDQEZnMrwzNo $^4$ https://github.com/thelmuth/Clojush/releases/tag/psb2-v1.0

| Name                 | Inputs                                                                                                                                                                      | Outputs               |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| Basement             | vector of integers of length [1, 20] with each integer in [-100, 100]                                                                                                       | integer               |
| Bouncing Balls       | float in [1.0, 100.0], float in [1.0, 100.0], integer in [1, 20]                                                                                                            | float                 |
| Bowling              | string in form of completed bowling card, with one character per roll                                                                                                       | integer               |
| Camel Case           | string of length [1,20]                                                                                                                                                     | string                |
| Coin Sums            | integer in [1, 10000]                                                                                                                                                       | 4 integers            |
| Cut Vector           | vector of integers of length [1, 20] with each integer in [1, 10000]                                                                                                        | 2 vectors of integers |
| Dice Game            | 2 integers in [1, 1000]                                                                                                                                                     | float                 |
| Find Pair            | vector of integers of length $[2,20]$ with each integer in $[-10000,10000]$ , integer in $[-20000,20000]$                                                                   | 2 integers            |
| Fizz Buzz            | integer in [1, 1000000]                                                                                                                                                     | string                |
| Fuel Cost            | vector of integers of length [1, 20] with each integer in [6, 100000]                                                                                                       | integer               |
| GCD                  | 2 integers in [1, 1000000]                                                                                                                                                  | integer               |
| Indices of Substring | 2 strings of length [1, 20]                                                                                                                                                 | vector of integers    |
| Leaders              | vector of integers of length [0, 20] with each integer in [0, 1000]                                                                                                         | vector of integers    |
| Luhn                 | vector of integers of length 16 with each integer in [1,9]                                                                                                                  | integer               |
| Mastermind           | 2 strings of length 4 made of B, R, W, Y, O, G                                                                                                                              | 2 integers            |
| Middle Character     | string of length [1,100]                                                                                                                                                    | string                |
| Paired Digits        | string of digits of length [2, 20]                                                                                                                                          | integer               |
| Shopping List        | vector of floats of length [1, 20] with each float in [0.0, 50.0], vector of floats of length [1, 20] with each float in [0.0, 100.0]. Both vectors must be the same length | float                 |
| Snow Day             | integer in [0, 20], float in [0.0, 20.0], float in [0.0, 10.0], float in [0.0, 1.0]                                                                                         | float                 |
| Solve Boolean        | string of length [1,20] made of characters from {t, f,  , &}                                                                                                                | Boolean               |
| Spin Words           | string of length [0, 20]                                                                                                                                                    | string                |
| Square Digits        | integer in [0, 1000000]                                                                                                                                                     | string                |
| Substitution Cipher  | 3 strings of length [0, 26]                                                                                                                                                 | string                |
| Twitter              | string of length [0, 200]                                                                                                                                                   | string                |
| Vector Distance      | 2 vectors of floats of length [1, 20] with each float in [-100.0, 100.0]                                                                                                    | float                 |

Table 1: For each problem, the types of the inputs and outputs, and the limits imposed on the inputs.

- 6. **Cut Vector (CW)** Given a vector of positive integers, find the spot where, if you cut the vector, the numbers on both sides are either equal, or the difference is as small as possible. Return the two resulting subvectors as two outputs. [36]
- 7. **Dice Game (PE)** Peter has an *n* sided die and Colin has an *m* sided die. If they both roll their dice at the same time, return the probability that Peter rolls strictly higher than Colin. [4]
- 8. **Find Pair (AoC)** Given a vector of integers, return the two elements that sum to a target integer. [58]
- 9. **Fizz Buzz (CW)** Given an integer x, return "Fizz" if x is divisible by 3, "Buzz" if x is divisible by 5, "FizzBuzz" if x is divisible by 3 and 5, and a string version of x if none of the above hold. [54]
- 10. **Fuel Cost (AoC)** Given a vector of positive integers, divide each by 3, round the result down to the nearest integer, and subtract 2. Return the sum of all of the new integers in the vector. [57]
- 11. **GCD [Greatest Common Divisor] (CW)** Given two integers, return the largest integer that divides each of the integers evenly. [45]

- 12. **Indices of Substring (CW)** Given a text string and a target string, return a vector of integers of the indices at which the target appears in the text. If the target string overlaps itself in the text, all indices (including those overlapping) should be returned. [49]
- 13. **Leaders (CW)** Given a vector of positive integers, return a vector of the leaders in that vector. A leader is defined as a number that is greater than or equal to all the numbers to the right of it. The rightmost element is always a leader. [35]
- 14. **Luhn (CW)** Given a vector of 16 digits, implement Luhn's algorithm to verify a credit card number, such that it follows the following rules: double every other digit starting with the second digit. If any of the results are over 9, subtract 9 from them. Return the sum of all of the new digits. [33]
- 15. Mastermind (HW) Based on the board game Mastermind. Given a Mastermind code and a guess, each of which are 4-character strings consisting of 6 possible characters, return the number of white pegs (correct color, wrong place) and black pegs (correct color, correct place) the codemaster should give as a clue.

- 16. Middle Character (CW) Given a string, return the middle character as a string if it is odd length; return the two middle characters as a string if it is even length. [48]
- 17. **Paired Digits (AoC)** Given a string of digits, return the sum of the digits whose following digit is the same. [56]
- 18. **Shopping List (CW)** Given a vector of floats representing the prices of various shopping goods and another vector of floats representing the percent discount of each of those goods, return the total price of the shopping trip after applying the discount to each item. [43]
- 19. **Snow Day (HW)** Given an integer representing a number of hours and 3 floats representing how much snow is on the ground, the rate of snow fall, and the proportion of snow melting per hour, return the amount of snow on the ground after the amount of hours given. Each hour is considered a discrete event of adding snow and then melting, not a continuous process.
- 20. **Solve Boolean (CW)** Given a string representing a Boolean expression consisting of T, F, |, and &, evaluate it and return the resulting Boolean. [28]
- 21. **Spin Words (CW)** Given a string of one or more words (separated by spaces), reverse all of the words that are five or more letters long and return the resulting string. [61]
- 22. **Square Digits (CW)** Given a positive integer, square each digit and concatenate the squares into a returned string. [37]
- 23. **Substitution Cipher (CW)** This problem gives 3 strings. The first two represent a cipher, mapping each character in one string to the one at the same index in the other string. The program must apply this cipher to the third string and return the deciphered message. [24]
- 24. **Twitter (HW)** Given a string representing a tweet, validate whether the tweet meets Twitter's original character requirements. If the tweet has more than 140 characters, return the string "Too many characters". If the tweet is empty, return the string "You didn't type anything". Otherwise, return "Your tweet has X characters", where the X is the number of characters in the tweet.
- 25. **Vector Distance (CW)** Given two *n*-dimensional vectors of floats, return the Euclidean distance between the two vectors in *n*-dimensional space. [53]

#### 6 USING PSB2

While Section 5 provides English-language descriptions of the 25 benchmark problems, these are not sufficient to implement each problem in a new system. Here we discuss the system-agnostic details for using these problems.

For reasons discussed in Section 3, we have created datasets consisting of large numbers of inputs and correct outputs for every

problem [13].<sup>5</sup> The dataset for each problem consists of a small number of hand-chosen inputs, often addressing edge cases for the problem, and 1 million randomly-generated inputs falling within the constraints of the problem. We recommend each different program synthesis run use a different set of data, composed of every one of the hand-chosen inputs and a random sample of the randomly-generated inputs. The alternative method of using the same fixed set of inputs for every run could happen to use a particularly lucky (or unlucky) set of inputs; using randomized inputs avoids this issue. Our datasets will allow those implementing PSB2 to simply sample the provided data, greatly decreasing the barrier to using PSB2. The PSB2 datasets can be found permanently on Zenodo.<sup>6</sup> For more information about distributions of inputs in randomly-generated inputs, see the reference implementation, which was used to generate the datasets.<sup>7</sup>

When using our provided datasets, one could sample different sizes of training and unseen test sets to fit a given experiment. Our recommendation, which we use in the our experiments, is to use 200 example cases for the training set (including all hand-chosen inputs) and 2000 for the unseen test set. However, some synthesis methods may need smaller or larger training sets, and PSB2 can flexibly adapt to such systems. In order to produce fairer comparisons between systems, we recommend using a fixed program executions budget to limit the number of generated program executions in a single program synthesis run. We recommend a budget of 60 million program executions; we allocate these to 200 training cases used to evaluate a population of 1000 individuals for 300 generations in our experimental GP runs, but other allocations of the same executions would be reasonable.

Program synthesis methods that have been applied to PSB1 have used varying methods for constraining the instruction set and other program syntax. For example, some have used grammars [6, 7, 22, 27, 32, 39, 50] while others have used data-type categorized subsets of an instruction set [16, 17]. We do not want to constrain what a reasonable approach to selecting instructions may look like for any given program synthesis system. However, we also warn against cherry-picking a small subset of instructions suspected of being useful for a particular problem. Part of the difficulty of general program synthesis is that a system must manage a large set of potentially useful instructions, finding those relevant to a particular problem. We recommend employing a large set of general-purpose instructions when using PSB2 to benchmark program synthesis to best replicate the conditions of a real-world scenario.

When evaluating the performance of a synthesis system on PSB2, we recommend using *success rate* (the number of synthesis runs that produce a solution) as the primary measure of performance, as was recommended in PSB1 [17]. For the synthesis of software, generating programs that pass most training cases is not sufficient; for this reason success rate is a better measure of performance than other metrics such as mean best fitness or mean number of training cases passed. In particular, a solution must not only pass all cases in the training set, but also all of the cases in the test set, to ensure that it generalizes to unseen data. This avoids considering

<sup>&</sup>lt;sup>5</sup>Our datasets follow the model of other machine learning datasets such as Penn ML Benchmarks [30, 38] and the UCI ML Repository [2].

<sup>&</sup>lt;sup>6</sup>https://zenodo.org/record/4678739

 $<sup>^7</sup> https://github.com/thelmuth/Clojush/releases/tag/psb2-v1.0$ 

programs that overfit the training data, such as by memorizing the correct output to each input, as solutions.

# 7 EXPERIMENTAL METHODS AND SYSTEM PARAMETERS

In this section we will discuss the system-specific parameters and choices that must be decided in order to use PSB2. In contrast with the previous section on general considerations, the choices here may differ considerably for different program synthesis systems. For our experiments, we used PushGP; we will describe in general the decisions that must be made and give our specific choices.

PushGP evolves programs in the language Push, a stack-based programming language built specifically for use in genetic programming [51, 52]. Every data type has its own stack, and each Push instruction acts by pushing and popping various elements on and off the stacks. The output of each problem is typically the top element on a particular stack. The interpreter executes programs that are themselves placed on an exec stack, allowing exec instructions to manipulate control flow as well as the program itself as it runs. We provide a reference implementation in Clojure of the PushGP system used to produce our results, which includes each problem in PSB2. This reference implementation is the same implementation of PushGP used in recent research using PSB1, e.g. [14, 15, 18].

We discuss the general design of program synthesis instruction sets in Section 6. For our PushGP experiments, we use the general process recommended in PSB1, where, for each problem, we identify which data types (corresponding to stacks) are relevant and include all implemented instructions that use those stacks [17]. In Table 2, we present the data types we chose to include for each problem, and the total number of instructions in the instruction set. These large instruction sets contain a wide range of general-purpose Push instructions, including some new instructions implemented since PSB1, avoiding the cherry-picking of clearly useful instructions. See the reference implementation for a complete listing of instructions.

Research utilizing PSB1 in using transfer-learned instruction sets showed that the composition of the instruction set matters a great deal to problem-solving performance [16]. While we do not use fully transfer-learned instruction sets here, we do make use of one simple take-away: that including larger proportions of input instructions and constants/ERCs improves performance. An explanation of this result is that most Push instructions decrease stack sizes by consuming arguments and producing fewer return values, so increasing inputs and constants creates more data on which instructions can act. We boost the presence of input instructions and constants in the instruction set, making input instructions fill 15% of the instruction set and constants/ERCs fill 5% of the instruction set. The additional input instructions are evenly distributed between each input for problems with multiple inputs, and constants/ERCs are similarly evenly distributed for each listed in the last column of Table 2.

For problems with multiple outputs, different synthesis systems will need to make choices specific to the language of the synthesized programs. Initial experiments in PushGP show that it achieves

better results on multi-output problems when using one output instruction per output. These output instructions are included in the instruction set for such problems and will always appear in solution programs. An example of this is for Coin Sums, which has 4 outputs. We provide four corresponding output instructions, each of which takes the top integer from the integer stack and stores it in a write-only register for that output; further calls to an output instruction will overwrite this output register.

In order to define each problem for GP, we not only need the inputs and correct outputs for each problem, but also how to calculate the error function based on the correct output and a program's output. Here we describe the error functions we employed in our experiments, which we recommend for any GP system implementing PSB2; other non-GP program synthesis systems may require entirely different metrics. For each output data type, we use the following standard error functions for problems outputting that data type:

- Integer or float: absolute value of the difference between program output and correct output.
- Boolean: 0 for correct and 1 for incorrect output.
- String: Levenshtein string edit distance between the program output and correct output.
- Vector of integers: add the difference in length between the program's output vector and the correct vector times 1000 to the absolute difference between each integer and the corresponding integer in the correct vector.

The only exception is for the Indices of Substring problem, where we used Levenshtein distance to compare vectors of integers, since it makes more sense for that problem. In PushGP, some evolving programs will not return values of a program's output data type; we give a penalty error value specific to the problem when this occurs.

As has been shown to be effective at improving generalization, we use an automatic simplification procedure on every evolved Push program that passes all of the training cases before testing it on the test set [14].

Unlike for PSB1, we aimed to keep all system-specific parameters constant between problems, increasing ease of use for both implementation and reporting of results. These parameters were chosen based on prior experience and reasonable performance; we leave optimizing parameter settings as an open research question. Other systems may choose to use different system-specific parameters

Our PushGP system uses linear Plush genomes that are initialized by generating lists of random instructions from the instruction set [21]. We list the important parameters used in our experiments below:

- Maximum initial genome size: 250 genes
- Maximum genome size: 500 genes
- Population size: 1000
- Maximum generations per run: 300
- Maximum steps of the Push interpreter when executing one program: 2000
- Parent selection: lexicase selection [20]
- Genetic operator: Uniform Mutation with Additions and Deletions (UMAD), used to make 100% of children [15].

 $<sup>^8</sup> https://github.com/thelmuth/Clojush/releases/tag/psb2-v1.0$ 

Table 2: Instructions and data types used in our PushGP implementation of each problem. The column "# Instructions" reports the number of instructions, terminals, and ephemeral random constants (ERC) used for each problem. The middle columns show which data types were used for each problem. For example, the Basement problem used all instructions relevant to exec, integers, Booleans, and vectors of integers. The last column lists the constants and ERCs used for the problem. Here, char constants are represented in the Clojure style, starting with a backslash, and strings are surrounded by double quotation marks. The "Problems" row simply counts how many problems use each data type. The "Instructions" row shows the number of Push instructions that primarily use each data type; some use multiple types but are only counted once.

| Problem                  | # Instructions | ехес     | integer  | float   | Boolean  | char     | string   | vector of integers | vector of floats | Constants and ERCs (besides inputs)                                                          |
|--------------------------|----------------|----------|----------|---------|----------|----------|----------|--------------------|------------------|----------------------------------------------------------------------------------------------|
| Basement                 | 117            | х        | х        |         | х        |          |          | х                  |                  | [], -1, 0, 1, integer ERC                                                                    |
| Bouncing Balls           | 127            | X        | X        | x       | X        |          |          |                    |                  | 0.0, 1.0, 2.0                                                                                |
| Bowling                  | 161            | x        | x        |         | x        | x        | x        |                    |                  | \-, \X, \/, \1, \2, \3, \4, \5, \6, \7, \8, \9, \10, integer ERC                             |
| Camel Case               | 151            | X        | X        |         | X        | X        | X        |                    |                  | \-, \space, visible character ERC, string ERC                                                |
| Coin Sums                | 86             | x        | x        |         | x        |          |          |                    |                  | 0, 1, 5, 10, 25                                                                              |
| Cut Vector               | 116            | X        | X        |         | X        |          |          | X                  |                  | [], 0                                                                                        |
| Dice Game                | 125            | x        | x        | x       | x        |          |          |                    |                  | 0.0, 1.0                                                                                     |
| Find Pair                | 120            | X        | X        |         | X        |          |          | X                  |                  | -1, 0, 1, 2, integer ERC                                                                     |
| Fizz Buzz                | 118            | x        | x        |         | x        |          | x        |                    |                  | "Fizz", "Buzz", "FizzBuzz", 0, 3, 5                                                          |
| Fuel Cost                | 117            | X        | X        |         | X        |          |          | X                  |                  | 0, 1, 2, 3, integer ERC                                                                      |
| GCD                      | 79             | X        | x        |         | x        |          |          |                    |                  | integer ERC                                                                                  |
| Indices of Substring     | 184            | X        | X        |         | x        | X        | X        | X                  |                  | [], "", 0, 1                                                                                 |
| Leaders                  | 114            | X        | X        |         | x        |          |          | X                  |                  | [], vector ERC                                                                               |
| Luhn                     | 117            | X        | X        |         | X        |          |          | X                  |                  | 0, 2, 9, 10, integer ERC                                                                     |
| Mastermind               | 123            | X        | X        |         | x        | X        | X        |                    |                  | 0, 1, \B, \R, \W, \Y, \O, \G                                                                 |
| Middle Character         | 151            | X        | X        |         | x        | X        | X        |                    |                  | "", 0, 1, 2, integer ERC                                                                     |
| Paired Digits            | 149            | X        | X        |         | X        | X        | X        |                    |                  | 0, char digit ERC, integer ERC                                                               |
| Shopping List            | 161            | X        | X        | X       | X        |          |          |                    | X                | 0.0, 100.0, float ERC                                                                        |
| Snow Day                 | 131            | X        | X        | X       | x        |          |          |                    |                  | 0, 1, -1, 0.0, 1.0, -1.0                                                                     |
| Solve Boolean            | 153            | X        | X        |         | x        | X        | X        |                    |                  | true, false, \t, \f, \&, \                                                                   |
| Spin Words               | 152            | X        | X        |         | X        | X        | X        |                    |                  | 4, 5, \space, visible character ERC, string ERC                                              |
| Square Digits            | 151            | X        | X        |         | x        | X        | X        |                    |                  | "", 0, 1, 2, integer ERC                                                                     |
| Substitution Cipher      | 151            | X        | X        |         | x        | x        | X        |                    |                  | "", 0                                                                                        |
| Twitter                  | 153            | X        | X        |         | X        | X        | X        |                    |                  | 0, 140, "Too many characters", "You didn't type anything", "your tweet has " , " characters" |
| Vector Distance          | 160            | X        | x        | x       | x        |          |          |                    | x                | [], 0                                                                                        |
| Problems<br>Instructions |                | 25<br>29 | 25<br>33 | 5<br>45 | 25<br>21 | 11<br>21 | 12<br>47 | 7<br>34            | 2<br>34          |                                                                                              |

### • UMAD addition rate: 0.09

As described in Section 6, using the exact same population size and generations is not necessary for comparisons between systems; instead, we recommend using a maximum budget of 60 million program executions regardless of other settings.

### 8 EXPERIMENTAL RESULTS

In order to give a baseline performance of the 25 problems in PSB2, we conducted 100 PushGP runs on each problem using the experimental methods described in Section 7. We present success rates of

our runs in Table 3. Out of the 25 problems in PSB2, 13 were solved by PushGP. Of these 13, 3 of them had 50 or more successes (Fuel Cost, Middle Character, and Substitution Cipher) and 2 others had 25 or more successes (Fizz Buzz and Twitter). The remaining 8 had fewer than 10 solutions, showing that they are solvable by GP but leave a lot of room for improvement.

PushGP did not solve the remaining 12 problems. However, we note that in our initial exploratory runs of the Bouncing Balls and Leaders problems, PushGP produced 2 generalizing solutions to

Table 3: Results from 100 PushGP runs on each problem. "Succ." gives the number of runs that successfully find a program that pass every training case and perfectly pass a set of 2000 unseen test cases. "Gen." gives the proportion of solutions on the training data that generalize to unseen data. "Size" gives the size of the smallest automatically simplified solution that generalized to unseen data. Time is the average number of seconds taken per generation.

| Problem              | Succ. | Gen. | Size | Time |
|----------------------|-------|------|------|------|
| Basement             | 1     | 1.00 | 18   | 250  |
| Bouncing Balls       | 0     | 0.00 | -    | 311  |
| Bowling              | 0     | -    | -    | 206  |
| Camel Case           | 1     | 1.00 | 20   | 95   |
| Coin Sums            | 2     | 1.00 | 33   | 213  |
| Cut Vector           | 0     | -    | -    | 194  |
| Dice Game            | 0     | -    | -    | 287  |
| Find Pair            | 4     | 1.00 | 16   | 763  |
| Fizz Buzz            | 25    | 0.96 | 19   | 281  |
| Fuel Cost            | 50    | 1.00 | 9    | 305  |
| GCD                  | 8     | 0.67 | 19   | 198  |
| Indices of Substring | 0     | -    | -    | 241  |
| Leaders              | 0     | -    | -    | 302  |
| Luhn                 | 0     | -    | -    | 239  |
| Mastermind           | 0     | -    | -    | 126  |
| Middle Character     | 57    | 0.86 | 10   | 547  |
| Paired Digits        | 8     | 1.00 | 15   | 250  |
| Shopping List        | 0     | -    | -    | 714  |
| Snow Day             | 4     | 1.00 | 11   | 263  |
| Solve Boolean        | 5     | 1.00 | 18   | 373  |
| Spin Words           | 0     | -    | -    | 443  |
| Square Digits        | 0     | -    | -    | 435  |
| Substitution Cipher  | 60    | 0.98 | 9    | 395  |
| Twitter              | 31    | 0.74 | 22   | 527  |
| Vector Distance      | 0     | -    | -    | 667  |

each, but did not replicate these successes in our runs with finalized parameter settings. Additionally, in continued work using the same PushGP settings as this paper except using down-sampled lexicase selection [18], small numbers of generalizing solutions were found to the problems Bouncing Balls, Dice Game, Indices of Substring, and Square Digits problems [19]. Thus at least 18 of the 25 problems are solvable with the our PushGP implementation. While we have no guarantees that the other 7 problems can be solved by any program synthesis system, they provide useful targets for future research.

The second column in Table 3 gives the generalization rate of all evolved solutions for problems on which PushGP produced at least one program that solved every training case. The generalization rate is calculated as the number of solution programs that pass the unseen test set divided by the number of solution programs that pass the training set. For most problems with training set solutions, those solutions tended to generalize well with rates of 0.95 to 1.0. The three problems with lower generalization rates, GCD, Middle Character, and Twitter all had rates over 0.5. However, Bouncing

Balls found 2 solutions on the training data, but neither of them generalized to the test, which resulted in 0 successful runs and a 0.00 generalization rate.

Another way of approximating the difficulty of these problems is by looking at the size of the smallest solution program found for each problem. Smaller solutions are easier for a program synthesis system to generate, simply because they require assembling fewer instructions in the right order. Our results are particular to Push program solutions, but should correlate with the sizes of programs needed to solve these problems in other systems. In order to find each size, we took each solution program and automatically simplified it to produce a smaller equivalent program [14]. Of these simplified programs, in Table 3 we report the smallest simplified solution size out of all simplified solutions to each problem. We see that the smallest solution size is 9 instructions for two problems, and two others have sizes of 10 and 11; three of these problems also had the highest success rates in PSB2. Many others have larger smallest solution sizes, though we note that with the small sample sizes of solutions for some problems, smaller solutions may exist. In comparison, [17] reported that 8 of the problems in PSB1 had a smallest solution size less than 9, the minimum for PSB2. Along with success rates, these sizes of smallest solutions give evidence that the problems in PSB2 are more difficult than those in PSB1.

The last column of Table 3 gives the average number of seconds per generation over all of the PushGP runs for the problem. Note that these runs were conducted on two different computing clusters, each of which is composed of heterogeneous machines, so these measurements should only be considered as rough approximations of running time. To that end, we note that all problems have generational running times within one order of magnitude of each other, meaning there are not any exceptionally slow or fast problems.

#### 9 CONCLUSIONS

We have presented PSB2, the second generation of general program synthesis benchmark problems. We discussed the past research that has used PSB1, the lessons learned from years of its use, and why we need a new benchmark suite. We then provided the sources and problems that make up PSB2, giving details of how to implement and use it in new systems. Finally, we presented experimental results showing the increased difficulty of the problems of PSB2 compared to PSB1.

After the results we have presented using PushGP, we anticipate using other GP systems (such as those we mention in Section 2 that have used PSB1) to produce initial results on PSB2 will provide a useful comparison. Additionally, we encourage the application of non-evolutionary automatic program synthesis methods to these problems, to better gauge the strengths and weaknesses of these different methods.

The lessons learned from PSB1 will make it easier to implement PSB2 in new program synthesis systems, increasing adoption in the community and furthering the field. PSB2 will provide a new target for program synthesis systems, stretching their capabilities and moving the field toward the types of problems that may be encountered in real-world program synthesis applications.

#### ACKNOWLEDGMENTS

The authors would like to thank Lee Spector, Grace Woolson, and Amr Abdelhady for discussions that helped shape this work.

#### REFERENCES

- [1] dnolan. 2015. Code Wars: Ten-Pin Bowling. https://www.codewars.com/kata/5531abe4855bcc8d1f00004c/javascript Accessed: 2020-01-20.
- [2] Dheeru Dua and Casey Graff. 2017. UCI Machine Learning Repository. http://archive.ics.uci.edu/ml
- [3] Project Euler. 2002. Project Euler: Coin Sums https://projecteuler.net/problem=31 Accessed: 2020-01-20.
- [4] Project Euler. 2008. Project Euler: Dice Game. https://projecteuler.net/problem=205 Accessed: 2020-01-20.
- [5] Austin J. Ferguson, Jose Guadalupe Hernandez, Daniel Junghans, Alexander Lalejini, Emily Dolson, and Charles Ofria. 2019. Characterizing the effects of random subsampling and dilution on Lexicase selection. In Genetic Programming Theory and Practice XVII, Wolfgang Banzhaf, Erik Goodman, Leigh Sheneman, Leonardo Trujillo, and Bill Worzel (Eds.). East Lansing, MI, USA.
- [6] Stefan Forstenlechner, David Fagan, Miguel Nicolau, and Michael O'Neill. 2017. A Grammar Design Pattern for Arbitrary Program Synthesis Problems in Genetic Programming. In EuroGP 2017: Proceedings of the 20th European Conference on Genetic Programming (LNCS, Vol. 10196). Springer Verlag, Amsterdam, 262–277. https://doi.org/10.1007/978-3-319-55696-3\_17
- [7] Stefan Forstenlechner, David Fagan, Miguel Nicolau, and Michael O'Neill. 2018. Extending Program Synthesis Grammars for Grammar-Guided Genetic Programming. In 15th International Conference on Parallel Problem Solving from Nature (LNCS, Vol. 11101), Anne Auger, Carlos M. Fonseca, Nuno Lourenco, Penousal Machado, Luis Paquete, and Darrell Whitley (Eds.). Springer, Coimbra, Portugal, 197–208. https://doi.org/10.1007/978-3-319-99253-2\_16
- [8] Stefan Forstenlechner, David Fagan, Miguel Nicolau, and Michael O'Neill. 2018. Towards effective semantic operators for program synthesis in genetic programming. In GECCO '18: Proceedings of the Genetic and Evolutionary Computation Conference. ACM, Kyoto, Japan, 1119–1126. https://doi.org/10.1145/3205455.3205592
- [9] Stefan Forstenlechner, David Fagan, Miguel Nicolau, and Michael O'Neill. 2018. Towards Understanding and Refining the General Program Synthesis Benchmark Suite with Genetic Programming. In 2018 IEEE Congress on Evolutionary Computation (CEC), Marley Vellasco (Ed.). IEEE, Rio de Janeiro, Brazil. https://doi.org/doi:10.1109/CEC.2018.8477953
- [10] g964. 2015. Code Wars: Bouncing Balls. https://www.codewars.com/kata/5544c7a5cb454edb3c000047 Accessed: 2020-01-20.
- [11] Sumit Gulwani. 2011. Automating String Processing in Spreadsheets Using Input-output Examples. SIGPLAN Not. 46, 1 (Jan. 2011), 317–330. https://doi.org/10.1145/1925844.1926423
- [12] Thomas Helmuth and Peter Kelly. 2019. General Program Synthesis Benchmark Suite Datasets. https://github.com/thelmuth/program-synthesis-benchmark-datasets
- [13] Thomas Helmuth and Peter Kelly. 2021. PSB2: The Second Program Synthesis Benchmark Suite. https://doi.org/10.5281/zenodo.4678739
- [14] Thomas Helmuth, Nicholas Freitag McPhee, Edward Pantridge, and Lee Spector. 2017. Improving Generalization of Evolved Programs Through Automatic Simplification. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17). ACM, Berlin, Germany, 937–944. https://doi.org/10.1145/3071178.3071330
- [15] Thomas Helmuth, Nicholas Freitag McPhee, and Lee Spector. 2018. Program Synthesis using Uniform Mutation by Addition and Deletion. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '18). ACM, Kyoto, Japan, 1127–1134. https://doi.org/10.1145/3205455.3205603
- [16] Thomas Helmuth, Edward Pantridge, Grace Woolson, and Lee Spector. 2020. Genetic Source Sensitivity and Transfer Learning in Genetic Programming. In Artificial Life Conference Proceedings. MIT Press, 303–311. https://doi.org/10.1162/isal\_a\_00326
- [17] Thomas Helmuth and Lee Spector. 2015. General Program Synthesis Benchmark Suite. In GECCO '15: Proceedings of the 2015 conference on Genetic and Evolutionary Computation Conference. ACM, Madrid, Spain, 1039–1046. https://doi.org/doi:10.1145/2739480.2754769
- [18] Thomas Helmuth and Lee Spector. 2020. Explaining and Exploiting the Advantages of Down-sampled Lexicase Selection. In Artificial Life Conference Proceedings. MIT Press, 341–349. https://doi.org/10.1162/isal\_a\_00334
- [19] Thomas Helmuth and Lee Spector. 2021. Problem-solving benefits of down-sampled lexicase selection. Artificial Life (2021). In press.

- [20] Thomas Helmuth, Lee Spector, and James Matheson. 2015. Solving Uncompromising Problems with Lexicase Selection. IEEE Transactions on Evolutionary Computation 19, 5 (Oct. 2015), 630–643. https://doi.org/10.1109/TEVC.2014.2362729
- [21] Thomas Helmuth, Lee Spector, Nicholas Freitag McPhee, and Saul Shanabrook. 2016. Linear Genomes for Structured Programs. In Genetic Programming Theory and Practice XIV (Genetic and Evolutionary Computation). Springer, Ann Arbor, 115 A
- [22] Erik Hemberg, Jonathan Kelly, and Una-May O'Reilly. 2019. On domain knowledge and novelty to improve program synthesis performance with grammatical evolution. In GECCO '19: Proceedings of the Genetic and Evolutionary Computation Conference. ACM, Prague, Czech Republic, 1039–1046. https://doi.org/doi:10.1145/3321707.3321865
- [23] Jose Guadalupe Hernandez, Alexander Lalejini, Emily Dolson, and Charles Ofria. 2019. Random subsampling improves performance in lexicase selection. In GECCO '19: Proceedings of the Genetic and Evolutionary Computation Conference Companion. ACM, Prague, Czech Republic, 2028–2031. https://doi.org/doi:10.1145/3319619.3326900
- [24] jacobb. 2014. Code Wars: Simple Substitution Cipher Helper. https://www.codewars.com/kata/52eb114b2d55f0e69800078d Accessed: 2020-01-20.
- [25] jhoffner. 2013. Code Wars: Convert string to camel case. https://www.codewars.com/kata/517abf86da9663f1d2000003 Accessed: 2020-01-20.
- [26] Susumu Katayama. 2010. Recent Improvements of MagicHaskeller. In Approaches and Applications of Inductive Programming. Springer. https://doi.org/10.1007/978-3-642-11931-6\_9
- [27] Jonathan Kelly, Erik Hemberg, and Una-May O'Reilly. 2019. Improving Genetic Programming with Novel Exploration - Exploitation Control. In EuroGP 2019: Proceedings of the 22nd European Conference on Genetic Programming, Lukas Sekanina, Ting Hu, Nuno Lourenço, Hendrik Richter, and Pablo García-Sánchez (Eds.). Springer International Publishing, 64–80.
- [28] KenKamau. 2017. Code Wars: The boolean order. https://www.codewars.com/kata/59eb1e4a0863c7ff7e000008 Accessed: 2020-01-20
- [29] Alexander Lalejini and Charles Ofria. 2019. Tag-accessed memory for genetic programming. In GECCO '19: Proceedings of the Genetic and Evolutionary Computation Conference Companion. ACM, Prague, Czech Republic, 346–347. https://doi.org/doi:10.1145/3319619.3321892
- [30] Trang T Le, William La Cava, Joseph D Romano, John T Gregg, Daniel J Goldberg, Praneel Chakraborty, Natasha L Ray, Daniel Himmelstein, Weixuan Fu, and Jason H Moore. 2020. PMLB v1.0: an open source dataset collection for benchmarking machine learning methods. arXiv preprint arXiv:2012.00058 (2020).
- [31] Jinsuk Lim and Shin Yoo. 2016. Field report: Applying monte carlo tree search for program synthesis. In *International Symposium on Search Based Software En*gineering. Springer, 304–310.
- [32] David Lynch, James McDermott, and Michael O'Neill. 2020. Program Synthesis in a Continuous Space using Grammars and Variational Autoencoders. In 16th International Conference on Parallel Problem Solving from Nature, Part II (LNCS, Vol. 12270), Thomas Baeck, Mike Preuss, Andre Deutz, Hao Wang2, Carola Doerr, Michael Emmerich, and Heike Trautmann (Eds.). Springer, Leiden, Holland, 33– 47. https://doi.org/doi:10.1007/978-3-030-58115-2\_3
- [33] mcclaskc. 2014. Code Wars: Validate Credit Card Number. https://www.codewars.com/kata/5418a1dd6d8216e18a0012b2 Accessed: 2020-01-20.
- [34] James McDermott, David R. White, Sean Luke, Luca Manzoni, Mauro Castelli, Leonardo Vanneschi, Wojciech Jaskowski, Krzysztof Krawiec, Robin Harper, Kenneth De Jong, and Una-May O'Reilly. 2012. Genetic programming needs better benchmarks. In GECCO '12: Proceedings of the Genetic and evolutionary computation conference. ACM, Philadelphia, Pennsylvania, USA, 791–798. https://doi.org/doi:10.1145/2330163.2330273
- [35] MrZizoScream. 2018. Code Wars: Array Leaders. https://www.codewars.com/kata/5a651865fd56cb55760000e0 Accessed: 2020-01-20.
- [36] myjinxin2015. 2016. Code Wars: Fastest Code: Half it IV. https://www.codewars.com/kata/5719b28964a584476500057d Accessed: 2020-01-20.
- [37] MysteriousMagenta. 2014. Code Wars: Square Every Digit. https://www.codewars.com/kata/546e2562b03326a88e000020 Accessed: 2020-01-20.
- [38] Randal S. Olson, William La Cava, Patryk Orzechowski, Ryan J. Urbanowicz, and Jason H. Moore. 2017. PMLB: a large benchmark suite for machine learning evaluation and comparison. *BioData Mining* 10, 1 (11 Dec 2017), 36. https://doi.org/10.1186/s13040-017-0154-4
- [39] Michael O'Neill and Anthony Brabazon. 2019. Mutational Robustness and Structural Complexity in Grammatical Evolution. In 2019 IEEE Congress on Evolutionary Computation, CEC 2019, Carlos A. Coello Coello (Ed.). IEEE Computational Intelligence Society, IEEE Press, Wellington, New Zealand, 1338–1344.

- https://doi.org/doi:10.1109/CEC.2019.8790010
- [40] Michael O'Neill and Lee Spector. 2020. Automatic programming: The open issue? Genetic Programming and Evolvable Machines 21, 1-2 (June 2020), 251–262. https://doi.org/doi:10.1007/s10710-019-09364-2 Twentieth Anniversary Issue.
- [41] Edward Pantridge, Thomas Helmuth, Nicholas Freitag McPhee, and Lee Spector. 2017. On the Difficulty of Benchmarking Inductive Program Synthesis Methods. In Proceedings of the Genetic and Evolutionary Computation Conference Companion (GECCO '17). ACM, Berlin, Germany, 1589–1596. https://doi.org/doi:10.1145/3067695.3082533
- [42] Edward Pantridge and Lee Spector. 2020. Code Building Genetic Programming. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference (GECCO '20). Association for Computing Machinery, internet, 994–1002. https://doi.org/doi:10.1145/3377930.3390239
- [43] rb50. 2017. Code Wars: Shopping List. https://www.codewars.com/kata/596266482f9add2@fAj0@fephenyu.Accessed: 2020-01-20. https://www.codewars.com/kata/596266482f9add2@fAj0@fephenyu.
- [44] Christopher D. Rosin. 2019. Stepping Stones to Inductive Synthesis of Low-Level Looping Programs. In Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI '19, Vol. 33). AAAI Press, Palo Alto, California USA.
- [45] RVdeKoning. 2015. Code Wars: Greatest common diviso https://www.codewars.com/kata/5500d54c2ebe0a8e8a0003fd/python Accessed: 2020-01-20.
- [46] Anil Kumar Saini and Lee Spector. 2019. Using Modularity Metrics as Design Features to Guide Evolution in Genetic Programming. In Genetic Programming Theory and Practice XVII, Wolfgang Banzhaf, Erik Goodman, Leigh Sheneman, Leonardo Trujillo, and Bill Worzel (Eds.). Springer, East Lansing, MI, USA, 165– 180. https://doi.org/doi:10.1007/978-3-030-39958-0\_9
- [47] Anil Kumar Saini and Lee Spector. 2020. Why and When Are Loops Useful in Genetic Programming?. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference Companion (GECCO '20). Association for Computing Machinery, internet, 247–248. https://doi.org/doi:10.1145/3377929.3389919
- [48] Shivo. 2015. Code Wars: Get the Middle Character. https://www.codewars.com/kata/56747fd5cb988479af000028 Accessed: 2020-01-20.
- [49] smile67. 2016. Code Wars: Text Search. https://www.codewars.com/kata/56b78faebd06e61870001191 Accessed: 2020-01-20.
- [50] Dominik Sobania and Franz Rothlauf. 2020. Challenges of Program Synthesis with Grammatical Evolution. In EuroGP 2020: Proceedings of the 23rd European Conference on Genetic Programming (LNCS, Vol. 12101), Ting Hu, Nuno

- Lourenco, and Eric Medvet (Eds.). Springer Verlag, Seville, Spain, 211–227.  $https://doi.org/doi:10.1007/978-3-030-44094-7\_14$
- [51] Lee Spector, Jon Klein, and Maarten Keijzer. 2005. The Push3 execution stack and the evolution of control. In GECCO 2005: Proceedings of the 2005 conference on Genetic and evolutionary computation, Vol. 2. ACM Press, Washington DC, USA, 1689–1696. https://doi.org/doi:10.1145/1068009.1068292
- [52] Lee Spector and Alan Robinson. 2002. Genetic Programming and Autoconstructive Evolution with the Push Programming Language. Genetic Programming and Evolvable Machines 3, 1 (March 2002), 7–40. https://doi.org/doi:10.1023/A:1014538503543
- [53] StephenLastname2. 2017. Code Wars: Distance between two points. https://www.codewars.com/kata/5a0b72484bebaefe60001867 Accessed: 2020-01-20.
- **205A**()0816phenyu. 2014. *Code Wars: Fizz Buzz.* https://www.codewars.com/kata/5300901726d12b80e8000498 Accessed: 2020-01-20.
- [55] Eric Wastl. 2015. Advent of Code: Not Quite Lisp. https://adventofcode.com/2015/day/1 Accessed: 2020-01-20.
- [56] Eric Wastl. 2017. Advent of Code: Inverse Captcha. https://adventofcode.com/2017/day/1 Accessed: 2020-01-20.
- [57] Eric Wastl. 2019. Advent of Code: The Tyranny of the Rocket Empire. https://adventofcode.com/2019/day/1 Accessed: 2020-01-20.
- [58] Eric Wastl. 2020. Advent of Code: Report Repair. https://adventofcode.com/2020/day/1 Accessed: 2020-01-20.
- [59] David R. White, James Mcdermott, Mauro Castelli, Luca Manzoni, Brian W. Goldman, Gabriel Kronberger, Wojciech Jaškowski, Una-May O'Reilly, and Sean Luke. 2013. Better GP benchmarks: community survey results and proposals. Genetic Programming and Evolvable Machines 14, 1 (March 2013), 3–29. https://doi.org/10.1007/s10710-012-9177-2
- [60] John Woodward, Simon Martin, and Jerry Swan. 2014. Benchmarks that matter for genetic programming. In GECCO 2014 4th workshop on evolutionary computation for the automated design of algorithms. ACM, Vancouver, BC, Canada, 1397–1404. https://doi.org/doi:10.1145/2598394.2609875
- [61] xDranik. 2013. Code Wars: Stop gninnipS My sdroW! https://www.codewars.com/kata/5264d2b162488dc400000001 Accessed: 2020-01-20.