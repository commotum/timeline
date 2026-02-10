1. **Number of distinct tasks evaluated:** 16

"**Tasks and datasets.** We evaluate self-consistency on the following reasoning benchmarks." (Section 3.1)

"- Arithmetic reasoning. For these tasks, we used the Math Word Problem Repository (Koncel-Kedziorski et al., 2016), including AddSub (Hosseini et al., 2014), MultiArith (Roy & Roth, 2015), and ASDiv (Miao et al., 2020). We also included AQUA-RAT (Ling et al., 2017), a recently published benchmark of grade-school-math problems (GSM8K; Cobbe et al., 2021), and a challenge dataset over math word problems (SVAMP; Patel et al., 2021)." (Section 3.1)

"- Commonsense reasoning. For these tasks, we used CommonsenseQA (Talmor et al., 2019), StrategyQA (Geva et al., 2021), and the AI2 Reasoning Challenge (ARC) (Clark et al., 2018)." (Section 3.1)

"- **Symbolic Reasoning**. We evaluate two symbolic reasoning tasks: last letter concatenation (e.g., the input is \"Elon Musk\" and the output should be \"nk\"), and Coinflip (e.g., a coin is heads-up, after a few flips is the coin still heads-up?) from Wei et al. (2022)." (Section 3.1)

"Here we perform a study using self-consistency to see if it can help fill in the gap, over a set of common NLP tasks, including (1) Closed-Book Question Answering: BoolQ (Clark et al., 2019), HotpotQA (Yang et al., 2018), and (2) Natural Language Inference: e-SNLI (Camburu et al., 2018), ANLI (Nie et al., 2020) and RTE (Dagan et al., 2005; Bar-Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009)." (Section 3.3)

2. **Number of trained model instances required to cover all tasks:** 1

"We perform all experiments in the few-shot setting, without training or fine-tuning the language models." (Section 3.1)

"Self-consistency also differs from a typical ensemble approach where multiple models are trained and the outputs from each model are aggregated, it acts more like a \"self-ensemble\" that works on top of a *single* language model." (Section 1)

3. **Task–Model Ratio:**

$$
\boxed{
\frac{16\ \text{tasks}}{1\ \text{model}} = 16
}
$$
