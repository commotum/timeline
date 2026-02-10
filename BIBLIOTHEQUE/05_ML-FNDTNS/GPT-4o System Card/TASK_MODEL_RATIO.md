1. **Number of distinct tasks evaluated:** Not specified in the paper.
   - "In this System Card, we provide a detailed look at GPT-40's capabilities, limitations, and safety evaluations across multiple categories, with a focus on speech-to-speech (voice)<sup>1</sup> while also evaluating text and image capabilities, and the measures we've implemented to ensure the model is safe and aligned." (Section 1: Introduction)
   - "Capabilities: We evaluate<sup>6</sup> on four tasks: TriviaQA, a subset of MMLU<sup>7</sup>, HellaSwag and Lambada." (Section 3.3.3: Disparate performance on voice inputs)
   - "The 172 CTF tasks in our evaluation covered four categories: web application exploitation, reverse engineering, remote exploitation, and cryptography." (Section 3.5: Cybersecurity)
   - "The 86 tasks (across 31 task \"families\") are designed to capture activities with real-world impact, across the domains of software engineering, machine learning, and cybersecurity, as well as general research and computer use." (Section 4.1: METR assessment)
   - "To better characterize the clinical knowledge of GPT-40, we ran 22 text-based evaluations based on 11 datasets, shown in 7." (Section 5.2: Health)

2. **Number of trained model instances required to cover all tasks:** 1 model.
   - "GPT-40[1] is an autoregressive omni model, which accepts as input any combination of text, audio, image, and video and generates any combination of text, audio, and image outputs." (Section 1: Introduction)
   - "It's trained end-to-end across text, vision, and audio, meaning that all inputs and outputs are processed by the same neural network." (Section 1: Introduction)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{N\ \text{tasks}}{1\ \text{model}} = N
}
$$
