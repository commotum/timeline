# ARCPrize 2024: Technical Report (2025)
Source: 9c3f73-2024.pdf

## Core reasons
- The report centers on the ARC-AGI benchmark and the ARC Prize competition as the vehicle for driving progress on the dataset and leaderboard, framing the contribution as infrastructure for measurement rather than a novel modeling innovation.
- The authors critique the limited private evaluation set, detail the risk of overfitting, and outline ARC-AGI-2 to improve sampling and scoring, reinforcing the paper’s focus on benchmark reliability and measurement.

## Evidence extracts
- "As of December 5, 2024, the ARC-AGI benchmark is five years old and remains unbeaten. We believe it is currently the most important unsolved AI benchmark in the world because it seeks to measure generalization on novel tasks – the essence of intelligence – as opposed to skill at tasks that can be prepared for in advance. This year, we launched ARC Prize, a global competition to inspire new ideas and drive open progress towards AGI by reaching a target benchmark score of 85%. As a result, the state-of-the-art score on the ARC-AGI private evaluation set increased from 33% to 55.5%, propelled by several frontier AGI reasoning techniques including deep learning-guided program synthesis and test-time training. In this paper, we survey top approaches, review new open-source implementations, discuss the limitations of the ARC-AGI-1 dataset, and share key insights gained from the competition." (p. 1)
- "The ARC-AGI-1 private evaluation set has been unchanged since 2019, and it is known to suffer from a number of flaws. First of all, the private evaluation set is limited to only 100 tasks. These 100 tasks have been used by all four ARC-AGI competitions for reporting intermediate leaderboard scores, and, as a result, on the order of 10,000 private evaluation set scores have been reported to participants so far. This presents a significant risk of overfitting, since each score has the potential to extract a tiny but non-zero amount of information about the content of the hidden tasks. The benchmark’s reliability can be improved by increasing the sample size and using two separate datasets: one for intermediate leaderboard scores (a larger semi-private evaluation set) and another for final scoring (a larger private evaluation set). This approach eliminates the risk of overfitting to the private evaluation set." (p. 9)

## Classification
Class name: Data, Benchmarks & Measurement
Class code: 4

$$
\boxed{4}
$$
