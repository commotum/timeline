1. **Number of distinct tasks evaluated:** 4

   Verbatim evidence: "For arithmetic reasoning, we utilize GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021)." (§3.1 Experimental Setup, Tasks and Datasets)

   Verbatim evidence: "For commonsense reasoning, we employ CSQA (Talmor et al., 2018) and StrategyQA (Geva et al., 2021)." (§3.1 Experimental Setup, Tasks and Datasets)

2. **Number of trained model instances required to cover all tasks:** 3

   Verbatim evidence: "The Tree-PLV, initially trained on the GSM8K dataset, yields effective results on the more complicated MATH500 dataset, illustrating its strong generalization capabilities." (§3.2 Main Results, Arithmetic Reasoning)

   Verbatim evidence: "To construct the training dataset, we selected 6,000 problems from each of the GSM8K and CSQA training sets to generate paired data." (§3.1 Experimental Setup, Data Collection)

   Verbatim evidence: "For additional evaluation, we sampled 750 questions from the StrategyQA training set, which yielded 15k pairs." (§3.1 Experimental Setup, Data Collection)

   Verbatim evidence: "The verifier then trains for an epoch on the corresponding dataset based on task types." (§3.1 Experimental Setup, Data Collection)

3. **Task–Model Ratio**

$$
\boxed{
\frac{4\ \text{tasks}}{3\ \text{models}} = 1.33
}
$$
