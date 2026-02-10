"For our experiments, we use GPT models of appropriate sizes. In particular, for any given microbenchmark, the model needs to fit on the number of model-parallel GPUs used in the experiment. We use standard model architectures such as GPT-3 [11] when appropriate." (§5 EVALUATION)

"We consider the end-to-end performance of our system on GPT models ranging from a billion to a trillion parameters, using tensor, pipeline, and data parallelism (degrees picked using heuristics described in §3)." (§5.1 End-to-End Performance)

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
