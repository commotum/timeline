1. **Number of distinct tasks evaluated:** 2

   "We evaluate on several benchmarks and measure tokenized BLEU (Papineni et al., 2002):"

   "**IWSLT'14 German to English (De-En).**"

   "WMT'14 English to French (En-Fr)."

   (Section 4.1 EXPERIMENTAL SETUP)

2. **Number of trained model instances required to cover all tasks:** 2 models

   "We use the setup of Edunov et al. (2018) and train on 160K sentence pairs."

   "We also experiment on the much larger WMT'14 English-French task comprising 35.5m training sentence pairs."

   "We train for 50k updates on 128 GPUs with a batch size of 460k tokens for WMT'14 En-Fr and on 2 GPUs with 8k tokens per batch for IWSLT'14 De-En."

   (Section 4.1 EXPERIMENTAL SETUP)

3. **Task–Model Ratio = (1) / (2):** 1

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
