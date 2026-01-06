# H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark

Solim LeGris<sup>1</sup>, Wai Keen Vong<sup>2</sup>, Brenden M. Lake <sup>1,2</sup> and Todd M. Gureckis<sup>1</sup> Department of Psychology, <sup>2</sup>Center for Data Science

New York University

## Abstract

The Abstraction and Reasoning Corpus (ARC) is a visual program synthesis benchmark designed to test challenging out-of-distribution generalization in humans and machines. Since 2019, limited progress has been observed on the challenge using existing artificial intelligence methods. Comparing human and machine performance is important for the validity of the benchmark. While previous work explored how well humans can solve tasks from the ARC benchmark, they either did so using only a subset of tasks from the original dataset, or from variants of ARC, and therefore only provided a tentative estimate of human performance. In this work, we obtain a more robust estimate of human performance by evaluating 1729 humans on the full set of 400 training and 400 evaluation tasks from the original ARC problem set. We estimate that average human performance lies between 73.3% and 77.2% correct with a reported empirical average of 76.2% on the training set, and between 55.9% and 68.9% correct with a reported empirical average of 64.2% on the public evaluation set. However, we also find that 790 out of the 800 tasks were solvable by at least one person in three attempts, suggesting that the vast majority of the publicly available ARC tasks are in principle solvable by typical crowd-workers recruited over the internet. Notably, while these numbers are slightly lower than earlier estimates, human performance still greatly exceeds current state-of-the-art approaches for solving ARC. To facilitate research on ARC, we publicly release our dataset, called H-ARC (human-ARC), which includes all of the submissions and action traces from human participants.

## 1 Introduction

In the last several years, large language models (LLMs) have reached impressive performance on a wide variety of benchmarks, demonstrating competency in natural language understanding, coding and mathematics [1, 15]. With larger and more powerful LLMs, many benchmarks have had a limited shelf life, with performance rapidly increasing to human or superhuman levels [10]. In contrast, The Abstraction and Reasoning Corpus (ARC, [5]) has proven to be a persistent and formidable challenge for state-of-the-art AI systems.

The ARC benchmark [5] was designed to evaluate broad generalization, measuring how algorithms handle a broad category of novel tasks given just a few examples each. Each task requires inferring an underlying transformation rule or program from a series of training input-output pairs which consist of abstract visual grids (see Figure 1), and to use this rule to correctly generate an output grid given a novel test input. Although visually simple, the tasks are conceptually rich and challenging, requiring the identification of compositional rules involving objects and relations, geometry, counting, visual instructions, and logical operations.

Previous attempts at benchmarking human performance on ARC found human accuracy to be 83.8% correct, which was estimated using a semi-randomly selected subset of 40 tasks

<sup>1</sup>https://arc-visualizations.github.io

![](_page_1_Figure_0.jpeg)

Figure 1: ARC Demonstration Tasks. Shown above are examples of easy (nearly everyone solved them in two attempts or less) and hard (almost no one solved them in two attempts or less) tasks with corresponding training examples and test for each. Interestingly, the easy evaluation task shown here has not been solved by the top LLM solutions to ARC reported in this document. Below the tasks are state space graphs representing all visited states by participants, from starting state (blue nodes) to correct or incorrect submitted grid (green and red nodes respectively). From left to right: f76d97a5.json., e9ac8c9e.json, e3497940.json and dd2401ed.json.

from the training set [8]. However, since this result was obtained using only a subset of tasks from the training set, it was unclear how robust this estimate of human performance is, nor whether the same level of performance is achievable on the evaluation set, which is believed to be much harder. The current study aimed to close this gap by providing a robust estimate of human performance on the ARC benchmark, based on human attempts to solve all 400 training and evaluation tasks respectively. We also publicly release H-ARC as the resulting dataset which consists of a total of 15744 attempts on ARC tasks with step-by-step action traces to facilitate further progress for developing more intelligent and human-like systems. Additionally, from the perspective of cognitive science, we believe this an invaluable dataset for enriching our understanding of how people solve a range of novel problems. Although research using variants of ARC tasks [12, 11] and a modified experimental setup [2] have also reported estimates of human performance, to the best of our knowledge, there was no comprehensive estimate of human performance on both the training and evaluation sets prior to this work.

# 2 Methods

## 2.1 Design

We collected human data on each of the 400 training tasks and 400 evaluation tasks from ARC in two separate phases (extending the subset of 40 training tasks previously collected and described in [8]). Each task has 1–10 training input-output pairs, and 1–3 test input-output pairs. While only a small proportion of tasks have multiple test input-output pairs (16 and 19 pairs in the training and evaluation set respectively), we opted to evaluate humans using only the first test example for all tasks. On average, 11.8 participants completed each of the 400 training tasks while 10.3 participants completed each of the 400 evaluation tasks.

![](_page_2_Figure_0.jpeg)

Figure 2: **ARC Experiment Interface**. Participants were given instructions about the different controls and layout of the interface followed by a tutorial task. Shown here is the evaluation tutorial task e9afcf9a.json.

#### 2.2 Participants

We recruited 784 participants (60.2% male, 37.3% female, 2.5% other) on the training set tasks and 948 participants (51.1% male, 46.1% female, 2.8% other) on the evaluation set tasks from Amazon Mechanical Turk using the CloudResearch<sup>2</sup> platform to ensure high quality data [7]. Participants were between 18 and 77 years old (M=39.8, SD=10.4). They were compensated \$10 and were also given a bonus of \$1 if they succeeded at a randomly selected task and its written solution description was judged adequate by the experimenters.<sup>3</sup>

# 2.3 Experiment

We evaluated humans using the same evaluation procedure proposed in the original paper describing the ARC benchmark [5]. In particular, human participants were allowed three attempts per task to generate a correct solution, and were only given minimal feedback on whether each submission attempt was correct or not.

User Interface. Participants were first given instructions about the experiment and explanations about the different aspects of the ARC user interface. As in previous experiments [8], the user interface closely mirrored the original interface proposed by [5] (see Figure 2). The interface allowed participants to select different colors and either edit one cell at a time or multiple selected cells. More sophisticated tools allowed the participant to copy and paste a selection from the test input to the test output grid or use the flood fill tool to change the color of all neighbouring cells of the same color to a new color. Participants could resize the grid height and width as well as copy the full test input grid to the test output grid. A reset button allowed participants to revert the output grid back to the initial state, a  $3 \times 3$  black grid. Finally, unlike in previous iterations of the interface, we added an additional tool allowing participants to undo actions and revert the state of the output grid to the previous state before the last action was taken. At any point in time, the participant could click the help button to display the full set of instructions.

**Tutorial:** At the beginning of the experiment, participants were provided with animated instructions outlining the user interface with an example task, and then asked to solve the same task to familiarize themselves with the interface. A relatively simple task<sup>4</sup> was given to participants for the tutorial and they were required to generate the correct test output to proceed (see Figure 2 for an example). After the tutorial, participants were asked to answer several comprehension questions to make sure they understood the instructions. The

<sup>&</sup>lt;sup>2</sup>https://cloudresearch.com

<sup>&</sup>lt;sup>3</sup>Best judgement was used: if a description was at least one complete sentence and was relevant to the task, it was counted as adequate.

<sup>&</sup>lt;sup>4</sup>For the training set experiment, we chose task 21f83797.json from the evaluation set, whereas for the evaluation set experiment, we chose task e9afcf9a.json from the training set.

experiment started immediately after successful completion of the quiz. Participants were given unlimited attempts at the quiz.

![](_page_3_Figure_1.jpeg)

Figure 3: Mean 3-shot task success rate. Tasks are ordered from lowest success rate to highest, showing the distribution of empirically estimated task difficulty for both the 400 tasks in the training and evaluation sets. Dotted lines show average accuracy across all tasks in either the training (blue) or evaluation (orange) split.

**Procedure**: The experiment consisted of 5 ARC tasks which were randomly selected from either the set of 400 training tasks or from the 400 evaluation tasks.<sup>5</sup> On average, participants completed the experiment in 23 minutes and 1 second (SD=13m 24s) on the training set, and 28 minutes and 51 seconds (SD=16m 19s) on the evaluation set. There was no time limit for completing a task.<sup>6</sup>

For each task, participants were given three attempts. After each attempt, feedback was given on whether the submitted solution was correct or not. Participants were not allowed to resubmit a previously incorrect output grid, ensuring that each of their attempts would be unique. If the participant failed to generate the solution after three attempts, they automatically proceeded onto the next task.

We also collected natural language descriptions of the inferred solutions by asking participants to write down their solution in words. This was first done after submitting an initial attempt before any feedback was given. If the initial submission was incorrect, participants were asked to submit a second natural language description, either after a subsequent correct submission or on their last (but still incorrect) submission.

Incomplete participant data: Participant data collected online can be incomplete for many reasons: participants may find the task too hard, have technical difficulties, find the experiment uninteresting, not understand the instructions or even run out of time. In our experiments, a number of participants withdrew from the experiment after completing between 0 and 4 tasks, although most did not provide any particular reason for withdrawing from the experiment. We found that 90 out of 783 participants' data from the training set experiment is incomplete, while 242 out of 946 participants' data from the evaluation set experiment is incomplete. Our results indicate that 7.5% and 13.3% of the training and evaluation set task data respectively are missing for a total of 10.3% missing task data. We obtained these values by computing the proportion of expected task data (number of participants × number of tasks assigned) that was missing from our dataset. To account for these substantial rates of incomplete data, we report pessimistic and optimistic performance estimates to provide a range

<sup>&</sup>lt;sup>5</sup>To reduce the potential for attrition or dropouts, we reduced the amount of ARC tasks participants were required to solve from 10 to 5 tasks after collecting 241 out of 783 participants from the first phase of data collection on the training set.

<sup>&</sup>lt;sup>6</sup>Participants who exceeded the total time limit of 90 minutes were dealt with manually by email and were included in our dataset nonetheless.

<sup>&</sup>lt;sup>7</sup>We implemented this feature after collecting data from the first 340 participants in the training set experiment. Prior to that, we observed that approximately 8% of incorrect second and third submission attempts were the same as earlier submission attempts on the same task.

![](_page_4_Figure_0.jpeg)

Figure 4: Human behavior action traces on ARC problems. In the left column, we show the test input seen by participants and the true test output grid for three different problems from the ARC evaluation set. In the middle column, sampled action traces show successive states of the grid with the last state corresponding to a correct (green box) or incorrect (red box) submission. In the last column, we show the first natural language descriptions submitted by participants along with their solution. From top to bottom: 34b99a2b.json, 4364c1c4.json and a8610ef7.json.

| Model                                                     | Training Set (%)                                        |                      |                      | Evaluation Set (%)                        |                      |                      |
|-----------------------------------------------------------|---------------------------------------------------------|----------------------|----------------------|-------------------------------------------|----------------------|----------------------|
|                                                           | avg / best                                              | pess                 | opt                  | avg / best                                | pess                 | opt                  |
| Human 1-shot<br>Human 2-shot<br>Human 3-shot              | 59.9 / 96.8<br>72.6 / 98.5<br><b>76.2</b> / <b>98.8</b> | 57.6<br>69.7<br>73.3 | 61.5<br>73.7<br>77.2 | 47.8 / 95.8<br>60.2 / 97.8<br>64.2 / 98.8 | 41.6<br>52.4<br>55.9 | 54.6<br>65.4<br>68.9 |
| Claude-3.5- $N$ (1 / 2-shot)<br>GPT-4o- $NS$ (1 / 2-shot) | -<br>-                                                  |                      |                      | $19.3 \ / \ 20.7 \ 38.5 \ / \ 42.0$       |                      |                      |

Table 1: **Human ARC Performance Summary**: Human average (avg) represents empirically estimated human performance after one, two or three attempts, averaged across all tasks in the training and evaluation sets respectively. Human best refers to the overall proportion of tasks that any human participant successfully solved, across all tasks in either the training and evaluation sets. We also report pessimistic (pess) and optimistic (opt) estimates of human performance. For comparison, we also report performance from the current public leaderboard for the ARC challenge. Note that Claude-3.5-N was only evaluated on a subset of 150 evaluation tasks.

of values within which human performance on ARC is likely to lie. This was accomplished by simulating imputed scores to the missing data from participants who dropped out.

## 3 Results

#### 3.1 Performance

For both ARC datasets, we report a range of performance values that reflect different ways of thinking about our estimate of human performance on ARC tasks and its inherent uncertainty (see Figure 5). In general, we find that human performance is higher on the training set compared to the evaluation set of ARC, validating previous intuitions about the relative difficulty between the two splits. Yet, despite the drop in performance on the evaluation set, humans still greatly outperform current state-of-the-art approaches to ARC, and we report and compare performance with two recent LLM-based solutions from the ARC prize public leaderboard (see Table 1).

The first LLM-based solution [9] we report involves minimal prompting of Claude 3.5 Sonnet [3] using text-only representations of ARC problems. Input and output training examples as well as test inputs are labelled and represented as lists of lists. The model is instructed to generate a JSON formatted list of lists response representing the output grid after applying the

 $<sup>^8</sup>$ https://arcprize.org/leaderboard

| Metric                          | Training Set | Evaluation Set |
|---------------------------------|--------------|----------------|
| Number of tasks                 | 400          | 400            |
| Total number of participants    | 783          | 946            |
| Incomplete participants         | 90           | 242            |
| Average participants per task   | 11.8         | 10.3           |
| Average attempts to solution    | 1.3          | 1.4            |
| Total attempts                  | 7924         | 7820           |
| Unique number of visited states | 127146       | 208214         |
| Total action traces             | 241697       | 344569         |

Table 2: **Human ARC Descriptives**: Here we report numerical values summarizing our behavioral dataset. "Total attempts" represents the number of individual submissions across all tasks and "total action traces" represents the number of single actions recorded across all tasks and participants.

inferred pattern to the test input. This model was only evaluated on a subset of 150 tasks from the evaluation set. Throughout the rest of this report, we will simply refer to this approach as "Claude-3.5-N" for Claude-3.5-neural. The second LLM-based solution [6] we report involves elaborate few-shot prompting of GPT-4o [1] with both images and text representations of ARC problems as well as instructions about how to reason about the task. The model is instructed to generate code that parses the test input to produce an output grid and also attempts to make corrections to its own generated code. After extensive sampling of the model (approximately 8000 code completions), the top 3 outputs are submitted using a majority vote over programs. We will refer to this method as "GPT-4o-NS" for GPT-4o-neurosymbolic since it explicitly combines a neural and symbolic, code-based approach.

Training set. We computed accuracy by calculating the proportion of successful submissions on each task after three attempts or less and averaged across all tasks. We also estimate pessimistic and optimistic performance values. In the pessimistic case, for each participant that completed k < 5 tasks, we sample 5 - k random tasks and assume failure. Conversely, in the optimistic case, we repeat the same procedure but assume success on every sampled task. We run 1000 simulations for each case, imputing missing data using the sampled tasks and outcomes. We then take the resulting average mean task success rate to compute pessimistic and optimistic estimates which we report in brackets. Our results suggest an estimated average task accuracy of 76.2% (SD=21.5%, [73.3%, 77.2%]) on the training set of ARC. We also report average task accuracy based on participants' first and second attempts, resulting in average task accuracy of 59.9% (SD=24.8%, [57.6%, 61.5%]) and 72.6% (SD=22.9%, [69.7%, 73.7%]) respectively.

Participants solve ARC training tasks in 1.3 attempts on average, with the modal and median number of attempts being one. Of the 400 training tasks, we find 74 tasks (18.5% of the training set) for which all participant who attempted the task generated the correct solution after three submissions or less. Conversely, we also find 5 tasks (1.3% of the training set) which no participants were able to solve correctly after three attempts (see Figure 3). Finally, we find that 40.1% of participants solved all training set tasks they were presented and that 8.6% of participants solved none (see Figure 6).

**Evaluation set.** Independent samples t-tests suggest that evaluation tasks are significantly harder for people than training tasks, t(798) = 7.67, p < .001. We estimate that the average task accuracy after three attempts on the evaluation set is 64.2% (SD=22.8%, [55.9%, 68.9%]). In addition to this result, we report a first and second attempt average task accuracy of 47.8% (SD=23.2%, [41.6%, 54.6%] and 60.2% (SD=23.3%, [52.4%, 65.4%]) respectively.

<sup>&</sup>lt;sup>9</sup>We consider these pessimistic and optimistic estimates somewhat unrealistic. In the pessimistic case, there are many reasons a participant might have dropped out of the task, including running out of time, technical issues, or lack of interest. There is no reason to assume that *all* remaining tasks would have been failed. Similarly, in the optimistic case there is no reason to think that people would succeed on every subsequent task they did not attempt. However, these bounds help us bracket how much the drop out effect might have altered our estimates.

<sup>&</sup>lt;sup>10</sup>Note that since each problem is attempted by approximately 10 people, this result simply means that we did not find anyone in a set of 10 that could solve the problem, not that they are in-principle not solvable.

![](_page_6_Figure_0.jpeg)

Figure 5: Minimum number of correct submissions across tasks. In this figure, we report the proportion of tasks where a minimum of N (x-axis) participants submitted a correct solution in three attempts or less. Note that higher values of N are biased negatively since posterior inference on a binomial outcome with small sample size is skewed.

On average participants solve ARC evaluation tasks in 1.4 attempts, with the modal and median number of attempts being one. Of the 400 evaluation tasks, we find 22 tasks (5.5%) of the evaluation set) for which participants always found the correct solution. Conversely, we find 5 tasks (1.3%) of the evaluation set) which no participants were able to solve (see Figure 3). We also find that 33.8% of participants solved all evaluation set tasks they tried and that 16.7% solved none (see Figure 6). After comparing 2-shot success across evaluation set tasks for people and GPT-40-NS, we find 225 tasks that were only solved by humans, 7 tasks solved by neither and 2 tasks solved only by GPT-40-NS (see Figure 7).

#### 3.2 Cross-dataset comparisons

Although we don't find any conclusive indicators of why the evaluation set is harder than the training set, we report our attempt to understand what drives the increased difficulty. In particular, we examined two aspects of each dataset which we hypothesized to be contributors to difficulty: test output grid size and time spent on problems.

Output grid size. An independent samples t-test confirmed that output grid size (number of grid cells) is significantly larger on average in the evaluation set (M=235.1, SD=246.7) than in the training set (M=136.2, SD=164.9), t(798)=6.81, p<.001. Although we find that grid size is mildly correlated with difficulty on the training set, r(398)=-0.16, p=.001, we did not find a similar relationship between output grid size and performance in the evaluation set r(398)=-0.02, p=.649.

Time spent on problems. Reaction time or time spent thinking about a problem, has been shown to reflect value of computation in chess games [13], relates effort allocation and resulting scores in IQ tests [4] and has a strong history in psychology as a window into underlying cognitive processes. We performed a number of temporal analyses based on various timing-related aspects of the human behavioral data.

First, we computed the time taken from seeing a new task to submitting a final solution for each participant and task. We then normalized these values by the number of attempts taken by each participant on each task. An independent samples t-test showed that participants spend significantly less time solving tasks from the training set  $(M=4\text{m }19\text{s},\ SD=4\text{m }4\text{s})$  than tasks from the evaluation set  $(M=6\text{m }39\text{s},\ SD=5\text{m }59\text{s}),\ t(8874)=-21.63,\ p<.001.$  Surprisingly, average performance on the evaluation set was lower despite participants spending more time and effort on average per task compared to the training tasks.

Second, how much time do people spend thinking instead of acting? As a first approximation of how much time is spent *thinking* about ARC puzzles, for each attempt per participant, we computed the sequence of inter-action times: the time spent between each action (i.e., clicks) taken to modify the state of the output grid or otherwise interact with the user interface. We then filtered out any inter-action time interval smaller than 5 seconds, corrected for time spent

![](_page_7_Figure_0.jpeg)

Figure 6: **Proportion of tasks solved by participants.** We report the proportion of tasks solved instead of discrete number of tasks since there were a minority of participants that completed a different number of tasks (e.g. due to dropping out, or having completed an earlier version of the experiment with ten distinct tasks).

writing and only considered series of submissions that led to a correct solution. Finally, we sum the remaining inter-action time intervals and normalized by the number of attempts taken to solve each task. An independent samples t-test confirmed that participants spend significantly less time thinking about training set tasks (M=1m 6s, SD=1m 10s) than they do about evaluation set tasks (M=1m 37s, SD=1m 39s), t(1505)=-6.90, p<.001. This result provides evidence that more thinking is required to infer the underlying rule or program of evaluation tasks as opposed to training tasks, and suggests that the reasoning process underlying finding a solution for evaluation tasks is more computationally demanding than for training tasks.

#### 3.3 Errors

Another aspect of the behavioral data that is potentially revealing of the underlying cognitive processes involved in solving ARC problems is looking at the types and patterns of errors that participants make (see Figure 4). We explore a number of preliminary analyses looking at errors along a variety of dimensions: errors on the dimensions of the test output grid, edit distance from the true test output, error divergence and copying, and compare them to the machine outputs reported in this document (see Table 1).

Grid dimension errors. One of the most common first actions that people take on both training and evaluation set tasks (33.2%) is to resize the output grid to the intended size by selecting the height or width drop-down menu. Although people make a non-negligible amount of height and width errors, within the set of incorrect submissions, we find that 68.2% and 73.5% of submission attempts have both the correct height and width in training and evaluation set tasks respectively. Conversely, we find that both current top solutions to the public ARC leaderboard reported in this document (see Table 1) make fewer mistakes in selecting the correct height and width of the test output compared to humans, with Claude-3.5-N making grid dimension errors on 10.3% of incorrect submissions and GPT-40-NS making grid dimension errors 8.3% of the time.

Edit distance. As a proxy for how close to the true test output incorrect submissions tend to be for ARC tasks, we computed the distribution of edit distances for all errors on each task. <sup>11</sup> To facilitate comparison, we computed edit distance on grids with correct dimensions and normalized by grid size. We find that edit distance to ground truth distributions for GPT-40-NS (M=0.19, SD=0.18), Claude-3.5-N (M=0.19, SD=0.16) and humans (M=0.19, SD=0.15) are strikingly similar. Nonetheless, we find an average normalized pairwise distance between human errors and machine errors across tasks of 0.25 for both reported models ( $SD_{GPT-4o}$ =0.19,

<sup>&</sup>lt;sup>11</sup>Edit distance may be an imperfect proxy for errors, as large edit distances may still arise from inferring an almost correct rule or program to a given task.

 $SD_{Claude3.5}$ =0.20). Despite people and machines having similar edit distance to ground truth distributions, this result clearly suggests that they are making substantially different errors.

**Error divergence**. We computed error divergence as the number of unique incorrect grids over the total number of submitted grids for each problem. Task success rate and error divergence were found to be strongly negatively correlated on training tasks, r(378) = -0.81, p < .001, and on evaluation tasks r(395) = -0.88, p < .001. Since we do not have repeated samples data from models, we omit comparative analyses here.

Copy errors. We also find that 10.0% and 9.2% of errors on training and evaluation set tasks respectively were copy errors. Copy errors refer to any incorrect submission that was a copy of either the example inputs (0.6% and 0.3% for training and evaluation sets respectively), example outputs (2.4% and 3.6% for training and evaluation sets respectively) or the test input (7.0% and 5.4% for training and evaluation sets respectively). In contrast, only 3.5% and 1.6% of incorrect outputs from GPT-40-NS and Claude-3.5-N were copy errors, respectively.

![](_page_8_Figure_3.jpeg)

Figure 7: **Human-machine performance comparison.** In this figure, we compare human (2-shot) performance with GPT-4o-NS. We show the proportion of tasks solved by humans or machine only, both and neither for varying levels of human performance. We find 225 tasks that were only solved by humans, 7 tasks solved by neither and 2 tasks solved only by GPT-4o-NS.

## 4 Discussion

The present study aimed to provide a comprehensive evaluation of human performance on the ARC benchmark. To estimate performance on ARC, we collected data from 1729 Amazon Mechanical Turk workers who were each assigned 5 randomly selected tasks from either the training or evaluation set. We estimate that 3-shot human performance is between 73.3% and 77.2% on the training set with an observed empirical average of 76.2%, and between 55.9% and 68.9% on the evaluation set with an observed empirical average of 64.2%. We also report that 98.8% of both the training and evaluation sets are solved by at least one person. Overall, we find that the evaluation set is more difficult for people than the training set. Although it remains unclear why evaluation set ARC puzzles are harder, we find that people spend significantly more time thinking about evaluation set tasks than they do about training tasks. Finally, we analyzed error patterns related to grid dimension, edit distance, error divergence and copying for both people and machines. Although people solve more tasks than state-of-the-art approaches, we find that machines outperform people on most error metrics we analyzed. These results along with edit distance results suggest that the errors people and machines make on ARC tasks are of a different nature, further emphasizing that these approaches do not capture how people solve ARC problems. We discuss these issues further in the following sections.

# 4.1 Competence versus performance: contextualizing average performance

It is important to bear in mind that average performance on ARC tasks as reported here can be affected by many contextual factors and does not reflect some absolute measure of

human performance. For example, if participants were paid more for only correct responses, were put under more extreme time pressure, or if participants were sampled from a broader population we might find different values of average performance. The average scores simply reflect "how a paid Internet worker from Amazon Mechanical Turk will typically do at the given pay rate" rather than a universal estimate of human ability. For this reason the performance measures might have less durability than measures of human *competence* (i.e., can any human, in principle, solve the problems?). That said, the estimates we provide here still exceed those of state-of-the-art AI algorithms applied to the task, suggesting a substantial human-machine intelligence gap.

#### 4.2 Human competence

For almost every task (98.8%) in the combined ARC training and evaluation sets, there is at least one person that solved it and over 90% of tasks were solved by at least three randomly assigned online participants. This is interesting because effectively this means that if you contact 10 random people on the Internet (via CloudResearch), at least one will be able to solve any given ARC problem. This is in contrast to biased sampling from a selective group like AI researchers (including the inventor of the ARC problem set) or highly educated academics. And this estimate is clearly biased to be low because only 10 people completed each problem with three attempts (if we had 1000 people complete each problem, or gave each person unlimited attempts, we would expect the odds of a human solving every problem would go up). The point is we didn't have to do this exhaustive sampling to find almost universal solvability of the ARC problems. We believe this critically highlights that human intelligence is in principle capable of carrying out the required computations to solve any ARC task which is unlike any AI model reported so far. This result would appear to put aside doubts about the solvability of ARC tasks by humans in both the training and evaluation sets. Furthermore, similarly to how expertise at games like chess and Go had been pitted as the threshold for successful AI models, our human best performance score suggests a challenging goal for AI.

#### 4.3 Self-correction through minimal feedback

Although people make errors, our analyses as well as qualitative judgements suggest that people are better at learning from minimal feedback, and correcting for those errors than machines. In fact, most correct answers from either top solution reported here are obtained on a first attempt, with only +7% and +9.1% proportional increase in accuracy with a second attempt for Claude-3.5-N and GPT-4o-NS respectively. For humans, we see substantial improvements in accuracy after 2 attempts (+21.2% on the training set and +25.9% on the evaluation set) and still more proportional increase in accuracy when given 3 attempts (+5.0% on the training set and +10.0% on the evaluation set). People will often make initially wrong guesses but they are capable of self-correction and can flexibly consider alternative solutions. Understanding how people achieve this is likely to be useful for improving machine intelligence in ARC tasks, and more generally for problem-solving.

#### 4.4 Why is the evaluation set more difficult?

The evaluation set is found to be significantly more difficult than the training set for people. Although it is still unclear why that is the case, our results suggest that factors other than output grid size are contributing to difficulty on the evaluation set and that for equally sized output grids, evaluation tasks are still often more difficult than training tasks. We suspect that the primitive operations underlying the transformations for evaluation tasks are more difficult to infer and/or execute than in the training set. For instance, we have found that logic-based operations that require superimposing grid sections tend to be difficult for people (see Figure 4).

ARC problems present an interesting challenge: how can the outputs of a program given its inputs be decomposed into subroutines and primitive operations that recover the underlying program? Previous work has explored a hypothesis-generation model using an LLM to solve a subset of ARC tasks [14]. Results from this research demonstrated that an important bottleneck on LLM performance in this setting was generating accurate and useful natural language hypotheses about the underlying program or rule of each task. Another line of research has found that nameability or codability can modulate concept acquisition [16], making it harder to learn concepts that have features that are more difficult to name. Similarly, we believe that certain primitive operations underlying ARC tasks in the evaluation set might be more difficult to identify or retrieve for people, in turn making it more difficult to generate plausible

hypotheses about the underlying program. Another possible source of difficulty is that the complexity or description length of programs in the evaluation set may be longer on average than in the training set. We hope that the natural language descriptions and action traces from H-ARC will help elucidate these questions.

## 5 Conclusion

Although we provide a comprehensive estimate of human performance on ARC tasks and preliminary analyses, more in-depth analyses of essential elements of people's problem solving strategies such as state space trajectories, action traces and natural language descriptions are needed. H-ARC affords answering questions about these aspects of behavior which are likely to be informative for understanding the underlying mental representations that support abstract reasoning and problem-solving in people. As a result, we hope that the dataset we release along with this document will allow developments on two fronts: our understanding of how people solve novel abstract problems and how machines can be improved to reason about such problems in a more human-like and intelligent way.

#### Acknowledgements

We thank Mark Ho for comments and feedback during the revision of this document, Aysja Johnson for collecting the data from the first few hundred participants on the training set and Nicholas Emery Kirsch for funding the data collection and personnel efforts on the evaluation set. This work was also generously supported by NSF BCS grant 2121102 to TMG.

# References

- [1] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. (2023). GPT-4 technical report. <u>arXiv</u> preprint arXiv:2303.08774.
- [2] Acquaviva, S., Pu, Y., Kryven, M., Sechopoulos, T., Wong, C., Ecanow, G., Nye, M., Tessler, M., and Tenenbaum, J. (2022). Communicating natural programs to humans and machines. Advances in Neural Information Processing Systems, 35:3731–3743.
- [3] Anthropic (2024). Claude 3.5 sonnet model card addendum. preprint. Accessed on 8/15/24.
- [4] Cheyette, S. J. and Piantadosi, S. T. (2024). Response to Difficulty Drives Variation in IQ Test Performance. Open Mind, 8:265–277.
- [5] Chollet, F. (2019). On the measure of intelligence. arXiv preprint arXiv:1911.01547.
- [6] Greenblatt, R. (2024). Getting 50% (SoTA) on ARC-AGI with GPT-40. https://redwoodresearch.substack.com/p/getting-50-sota-on-arc-agi-with-gpt. Accessed on 8/15/24.
- [7] Hauser, D. J., Moss, A. J., Rosenzweig, C., Jaffe, S. N., Robinson, J., and Litman, L. (2023). Evaluating cloudresearch's approved group as a solution for problematic data quality on mturk. Behavior Research Methods, 55(8):3953–3964.
- [8] Johnson, A., Vong, W. K., Lake, B., and Gureckis, T. M. (2021). Fast and flexible: Human program induction in abstract reasoning tasks. In <u>Proceedings of the Annual Meeting of the Cognitive Science Society</u>, volume 43.
- [9] Kamradt, G. (2024). Using frontier models on arc agi via langchain. https://www.kaggle.com/code/gregkamradt/using-frontier-models-on-arc-agi-via-langchain? scriptVersionId=185410590. Kaggle Notebook, Version 185410590, Accessed on 8/15/24.
- [10] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. (2020). Scaling laws for neural language models.
- [11] Mitchell, M., Palmarini, A. B., and Moskvichev, A. (2023). Comparing humans, gpt-4, and gpt-4v on abstraction and reasoning tasks. <u>arXiv preprint arXiv:2311.09247</u>.
- [12] Moskvichev, A., Odouard, V. V., and Mitchell, M. (2023). The conceptarc benchmark: Evaluating understanding and generalization in the arc domain. <u>arXiv preprint</u> arXiv:2305.07141.
- [13] Russek, E., Acosta-Kane, D., van Opheusden, B., Mattar, M. G., and Griffiths, T. (2022). Time spent thinking in online chess reflects the value of computation.
- [14] Wang, R., Zelikman, E., Poesia, G., Pu, Y., Haber, N., and Goodman, N. D. (2023). Hypothesis search: Inductive reasoning with language models. arXiv preprint arXiv:2309.05660.

- [15] Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. (2022). Emergent abilities of large language models. <u>arXiv</u> preprint arXiv:2206.07682.
- [16] Zettersten, M. and Lupyan, G. (2020). Finding categories through words: More nameable features improve category learning. Cognition, 196(104135):104135.