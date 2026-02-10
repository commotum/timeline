1. **Number of distinct tasks evaluated:** 173

"We evaluate the generality of Dreamer across 8 domains—with over 150 tasks—under fixed hyperparameters." (Evaluation)

"Atari<sup>35</sup> uses 57 tasks with sticky actions<sup>55</sup>." (Methods, Benchmarks, Protocols)

"DMLab<sup>39</sup> uses 30 tasks<sup>52</sup> and we use the corrected action space<sup>33,56</sup>." (Methods, Benchmarks, Protocols)

"Atari100k18 includes 26 tasks with a budget of 400,000 environment steps, 100,000 after action repeat." (Methods, Benchmarks, Protocols)

"Visual control and proprioceptive control span the same 20 tasks<sup>22,42</sup> with a 1 million budget." (Methods, Benchmarks, Protocols)

"| Minecraft   | 1     | 100M         | 1             | 64               | 32              | 8.9         | 200M          |"
"| ProcGen     | 16    | 50M          | 1             | 16               | 32              | 8.3         | 200M          |"
"| BSuite      | 23    | _            | 1             | 1                | 1024            | 0.5         | 200M          |" (Extended Data Table 2 | Benchmark overview)

2. **Number of trained model instances required to cover all tasks:** 173 models

"All Dreamer agents are trained on a single Nvidia A100 graphics processing unit (GPU) each, making it reproducible for many research labs." (Evaluation)

"All Dreamer and PPO agents in this paper were trained on a single Nvidia A100 GPU each." (Methods, Computational choices)

"Dreamer paves the way for future research directions, including teaching agents world knowledge from internet videos and learning a single world model across domains to allow artificial agents to build up increasingly general knowledge and competency." (Conclusion)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{173\ \text{tasks}}{173\ \text{models}} = 1
}
$$
