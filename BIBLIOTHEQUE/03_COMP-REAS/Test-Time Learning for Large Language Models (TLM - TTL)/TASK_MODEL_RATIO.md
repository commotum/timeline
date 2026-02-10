1. **Number of distinct tasks evaluated:** 10

   - "AdaptEval consists of three categories of datasets. 1) **DomainBench** includes four vertical domain knowledge datasets: Geography, Agriculture, Medicine, and Finance... 2) InstructionBench contains three general-purpose instruction-following datasets: Alpaca-GPT4, Dolly, and Instruction-Wild... 3) ReasoningBench comprises three reasoning capability datasets: GSM8K, MetaMath, and Logiqa..." (Section 5.1. Experimental Settings)

2. **Number of trained model instances required to cover all tasks:** 10

   - "In the Offline setting, all test data is processed at once, and the model's parameters are updated using all available test samples before any testing is performed." (Section C.2. Implementation Details)
   - "We conduct experiments on different types of datasets, including DomainBench, InstructionBench, and ReasoningBench, as summarized in Table 2 and 3." (Section 5.2. Comparison Experiments)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{10\ \text{tasks}}{10\ \text{models}} = 1
}
$$
