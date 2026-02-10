1. **Number of distinct tasks evaluated:** 3 tasks.

- Task 1 (raw-audio music generation): "We introduce Jukebox, a model that generates music with singing in the raw audio domain." (Abstract)
- Task 2 (lyrics-to-singing): "Lyrics-to-singing (LTS) task: The conditioning signal only includes the text of the lyrics, without timing or vocalisation information." (Section 4.2, "Lyrics Conditioning")
- Task 3 (music continuation/completion): "**Primed sampling**: Instead of sampling the entire token sequence from the model, we can also run a forward pass of the VQ-VAE to obtain the top, middle, and bottom level codes corresponding to a segment from an actual song, as shown in Figure 2c. We can use these as the initial tokens in our ancestral sampling process and continue sampling from these to produce novel completions of the song." (Section 4.4, "Sampling")

2. **Number of trained model instances required to cover all tasks:** 2 models.

- The paper distinguishes a non-lyrics top prior and a lyrics-conditional top prior: "by swapping the top prior with a conditional prior, we can condition on lyrics to tell the singer what to sing, or on midi to control the composition." (Section 1, "Introduction").
- "To reduce computation required to train the lyrics conditional model, we use a pretrained unconditional top-level prior as our decoder and introduce the lyrics encoder" (Section 4.3, "Decoder Pretraining").
- "For lyrics conditioning, we reuse the prior and add a small encoder, after which we train the model" (Section 5.2, "Training Details").
- Whether one lyrics-conditioned top prior alone was used to cover every non-lyrics evaluation setting is **Not specified in the paper.**

3. **Task–Model Ratio:**

$$
\boxed{
\frac{3\ \text{tasks}}{2\ \text{models}} = 1.5
}
$$
