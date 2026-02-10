1. **Number of distinct tasks evaluated:** 3  
   "We perform experiments on (1) decision-making tasks to test sequential action choices over long trajectories, (2) reasoning tasks to test knowledge-intensive, single-step generation improvement, and (3) programming tasks to teach the agent to effectively use external tools such as compilers and interpreters." (Section 1: Introduction)  
   "Across all three types of tasks, we observe Reflexion agents are better decision-makers, reasoners, and programmers." (Section 1: Introduction)

2. **Number of trained model instances required to cover all tasks:** 1  
   "We propose *Reflexion*, a novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback." (Abstract)  
   "Reflexion has several advantages compared to more traditional RL approaches like policy or value-based learning: 1) it is lightweight and doesn't require finetuning the LLM, 2) it allows for more nuanced forms of feedback (e.g. targeted changes in actions), compared to scalar or vector rewards that are challenging to perform accurate credit assignment with, 3) it allows for a more explicit and interpretable form of episodic memory over prior experiences, and 4) it provides more explicit hints for actions in future episodes." (Section 1: Introduction)

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
