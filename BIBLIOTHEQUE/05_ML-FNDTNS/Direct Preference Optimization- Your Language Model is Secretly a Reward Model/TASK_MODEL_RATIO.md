1. **Number of distinct tasks evaluated:** 3

> "Tasks. Our experiments explore three different open-ended text generation tasks." (Section 6, "Tasks")

> "In **controlled sentiment generation**, x is a prefix of a movie review from the IMDb dataset [24], and the policy must generate y with positive sentiment." (Section 6, "Tasks")

> "In summarization, x is a forum post from Reddit; the policy must generate a summary y of the main points in the post." (Section 6, "Tasks")

> "Finally, in single-turn dialogue, x is a human query, which may be anything from a question about astrophysics to a request for relationship advice." (Section 6, "Tasks")

2. **Number of trained model instances required to cover all tasks:** 3 models

> "For SFT, we fine-tune GPT-2-large until convergence on reviews from the train split of the IMDB dataset (further details in App C.1)." (Section 6, "Tasks")

> "DPO, PPO and Preferred-FT all fine-tune the same GPT-J SFT model<sup>4</sup>." (Section 6.2)

> "As there is no standard SFT model for this task, we start with a pre-trained Pythia-2.8B, use Preferred-FT to train a reference model on the chosen completions such that completions are within distribution of the model, and then train using DPO." (Section 6.2)

A single jointly trained model covering all three tasks is not described: **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
