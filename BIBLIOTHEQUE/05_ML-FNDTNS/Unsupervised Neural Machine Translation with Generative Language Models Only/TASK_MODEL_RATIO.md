1. **Number of distinct tasks evaluated:** 2

   Evidence (Section 5.1, Table 1): "English-French (top) and French-English (bottom) test BLEU throughout the few-shot self-distillation bootstrap across multiple model scales."

2. **Number of trained model instances required to cover all tasks:** 1

   Evidence (Abstract): "During backtranslation, we repeatedly generate translations for a set of inputs and then fine-tune a single language model on both directions of the translation task at once, ensuring cycle-consistency by swapping the roles of gold monotext and generated translations when fine-tuning."

   Evidence (Section 3, "BACKTRANSLATION VIA LANGUAGE MODELING"): "In our present work, we cast machine translation as a language modeling task and jointly train and sample generations from a single language model for both source-to-target and target-to-source translation."

   Evidence (Section 3, "BACKTRANSLATION VIA LANGUAGE MODELING"): "we use a single language model for both forwards and backwards translation and train on both directions jointly at every iteration."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$
