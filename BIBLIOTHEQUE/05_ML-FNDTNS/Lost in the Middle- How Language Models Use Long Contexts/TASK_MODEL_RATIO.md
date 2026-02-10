1. **Number of distinct tasks evaluated: 3**
- "We analyze the performance of language models on two tasks that require identifying relevant information in their input contexts: multi-document question answering and key-value retrieval." (Section: Abstract)
- "To better understand this trade-off in practice, we perform a case study with retriever-reader models on open-domain question answering (§5)." (Section: 1 Introduction)

2. **Number of trained model instances required to cover all tasks: 1 model**
- "These models perform downstream tasks primarily via prompting: all relevant task specification and data to process is formatted as a textual input context, and the model returns a generated text completion." (Section: 1 Introduction)
- "We use the same set of models as the multi-document question answering experiments, see §2.2 for more details." (Section: 3.2 Results and Discussion)
- "Using more than 20 retrieved documents only marginally improves reader performance ( $\sim 1.5\%$  for GPT-3.5-Turbo and  $\sim 1\%$  for Claude-1.3)" (Section: 5 Is More Context Is Always Better? A Case Study With Open-Domain QA)

3. **Task–Model Ratio**

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
