1. **Number of distinct tasks evaluated:** 24

- "which includes 20 types of synthetically generated questions designed to mimic aspects of textual reasoning." (Section "Synthetic question answering experiments")
- "defined three kinds of query: 'traversal', 'shortest path' and 'inference' (Fig. 2)." (Section "Graph experiments")
- "Our environment, which we term Mini-SHRDLU, contains a set of numbered blocks on a grid board." (Section "Block puzzle experiments")

2. **Number of trained model instances required to cover all tasks:** 6

- "We found that a single DNC, jointly trained on all 20 question types with 10,000 instances each, was able to achieve a mean test error rate of 3.8%" (Section "Synthetic question answering experiments")
- "defined three kinds of query: 'traversal', 'shortest path' and 'inference' (Fig. 2)." (Section "Graph experiments")
- "The hyper-parameters were selected from large grid searches, and are listed for each experiment in Extended Data Table 2." (Section "Optimization")
- "The architecture of the reinforcement learning agent presented here contains two DNC networks: a policy network that selects an action and a value network that estimates the expected future reward given the policy network and current state." (Section "Reinforcement learning")
- A single jointly trained model instance spanning bAbI, all graph queries, and Mini-SHRDLU: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{24\ \text{tasks}}{6\ \text{models}} = 4
}
$$
