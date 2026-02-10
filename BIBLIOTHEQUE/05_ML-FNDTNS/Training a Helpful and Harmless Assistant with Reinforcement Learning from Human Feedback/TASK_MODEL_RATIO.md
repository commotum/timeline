1. **Number of distinct tasks evaluated:** 15

Quoted evidence:
- (Section **1.2 Summary of Evaluations and Metrics**) "NLP and Code Evaluations: We evaluate our models on MMLU [Hendrycks et al., 2021b], Lambada [Paperno et al., 2016], Hellaswag [Zellers et al., 2019], OpenBookQA [Mihaylov et al., 2018], ARC [Clark et al., 2018], and TriviaQA [Joshi et al., 2017]; see Figures 28 and 29 for full results and Figure 3 for the mean. In every case except for TriviaQA, 12B and 52B RLHF-trained models perform better than base LMs. Separately, we take Python coding models and finetune them with natural language RLHF, and then evaluate them on the codex HumanEval [Chen et al., 2021]; see Figure 21. We also experiment with mixing PM training for HH with summarization [Stiennon et al., 2020] as a specialized skill, and evaluate the resulting PM performance (Figure 20), finding that mixed training does not degrade PM accuracies."
- (Section **1.2 Summary of Evaluations and Metrics**) "Static Alignment Evaluations: We evaluate our PMs using our HHH Evaluations [Askell et al., 2021] from BIG-Bench<sup>6</sup> (Figure 5), on Bot Adversarial Dialogues [Xu et al., 2020], and for gender bias [Rae et al., 2021] (Figure 12). We evaluate our RLHF models on TruthfulQA [Lin et al., 2021] (Figure 5), BBQ-Lite [Parrish et al., 2021] from BIG-Bench, gender bias (Figure 40), and sentiment based on race and religion [Rae et al., 2021] (Figure 17). RLHF improves sentiment towards all groups, but does not remove bias."
- (Section **1.2 Summary of Evaluations and Metrics**) "Human Evaluations: We compute Elo scores based on the preferences of our crowdworkers, comparing context-distilled models, base RLHF trained models, and final online RLHF models (Figure 1)."

2. **Number of trained model instances required to cover all tasks:** 3

Quoted evidence:
- (Section **4.1 Training Setup**) "Prepare a dataset of comparisons, and train a PM to assign a higher score to the 'better' item in each comparison."
- (Section **4.1 Training Setup**) "Extract all the prompts from the preceding dataset, and train an RL policy to generate a response to each prompt autoregressively, with a reward signal provided by the PM score at the end of the response."
- (Section **5.2 Summarization as a Specialized Skill**) "As shown in Figure 20, large preference models trained on a mixture of HH and LtS datasets perform equally well on both."
- (Section **5.3 Natural Language RLHF on Code-Finetuned Models**) "Our 'base code models' were finetuned on Python code scraped from Github as described in [Askell et al., 2021]. Starting from these Python fine-tuned (Python FT) models, we then ran our standard natural language RLHF training using 'static' preference models and prompts."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{15\ \text{tasks}}{3\ \text{models}} = 5
}
$$
