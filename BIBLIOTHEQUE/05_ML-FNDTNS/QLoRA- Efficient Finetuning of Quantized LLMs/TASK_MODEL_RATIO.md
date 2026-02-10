1. **Number of distinct tasks evaluated:** 5

"Our evaluations include GLUE [58] with RoBERTa-large [38], Super-NaturalInstructions (TKInstruct) [61] with T5 [49], and 5-shot MMLU [24] after finetuning LLaMA on Flan v2 [39] and Alpaca [55]." (Section 4, "Experimental setup")

"We evaluate on two curated datasets of queries (questions): the Vicuna prompts [10] ... We term these two datasets the Vicuna and OA benchmarks." (Section 5.2, "Benchmark Data")

2. **Number of trained model instances required to cover all tasks:** 3

"We consider three architectures (encoder, encoder-decoder, and decoder only) ..." and the same sentence maps them to "GLUE ... RoBERTa-large," "Super-NaturalInstructions ... T5," and "5-shot MMLU ... finetuning LLaMA ..." (Section 4, "Experimental setup")

"To assess the performance of instruction finetuning these models, we evaluate on a challenging Natural Language Understanding benchmark (MMLU) and develop new methods for real-world chatbot performance evaluation." (Section 5)

"**Table 5:** MMLU 5-shot test results for different sizes of LLaMA finetuned on the corresponding datasets using QLoRA." (Table 5), together with Vicuna/OA benchmark evaluation in Section 5.2, indicates one LLaMA instruction-tuned instance can be evaluated on MMLU and chatbot benchmarks.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{5\ \text{tasks}}{3\ \text{models}} = 1.67
}
$$
