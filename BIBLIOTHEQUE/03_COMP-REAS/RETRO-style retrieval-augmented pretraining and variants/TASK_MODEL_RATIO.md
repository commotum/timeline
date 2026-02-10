1. Number of distinct tasks evaluated: 2.
"We first report results on language modelling benchmarks. Second, we show how to Retrofit pre-trained Transformer language models into retrieval models with few additional FLOPs. Next, we report Retro results on question answering." (Section 4. Results)

2. Number of trained model instances required to cover all tasks: 2 models.
"For C4, Wikitext103, the Pile, and our Wikipedia dataset we evaluate the language modelling performance on entire documents and measure the bits-per-byte (bpb)." (Section 4.1. Language modelling)
"We fine-tune all the weights of our 7.5B pre-trained Retro model for 25,000 steps using the top 20 retrieved passages." (Section 4.3. Question answering)
Not specified in the paper.

3. Task–Model Ratio = (1) / (2)

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
