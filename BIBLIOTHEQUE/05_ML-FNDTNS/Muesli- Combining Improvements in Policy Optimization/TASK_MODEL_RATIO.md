1. **Number of distinct tasks evaluated:** \((58 + N_{\text{MuJoCo}})\), where \(N_{\text{MuJoCo}}\) is **Not specified in the paper.**

Citations (verbatim):
- Section 1 (Introduction): "The majority of our experiments were performed on 57 classic Atari games from the Arcade Learning Environment (Bellemare et al., 2013; Machado et al., 2018), a popular benchmark for deep RL."
- Section 1 (Introduction): "Additionally, to evaluate how well our method generalises to different domains, we performed experiments on a suite of continuous control environments (based on MuJoCo and sourced from the OpenAI Gym (Brockman et al., 2016))."
- Section 1 (Introduction): "We also conducted experiments in 9x9 Go in self-play, to evaluate our policy update in a domain traditionally dominated by search methods."
- Section 5 (An empirical study): "Finally, we tested the same agents on MuJoCo environments in OpenAI Gym (Brockman et al., 2016), to test if Muesli can be effective on continuous domains and on smaller data budgets (2M frames)."

2. **Number of trained model instances required to cover all tasks:** \((58 + N_{\text{MuJoCo}})\), where \(N_{\text{MuJoCo}}\) is **Not specified in the paper.**

Citations (verbatim):
- Section 5 (An empirical study): "First, we use the 57 Atari games in the Arcade Learning Environment (Bellemare et al., 2013) to investigate the key design choices in Muesli, by comparing it to suitable baselines and ablations."
- Section 5 (An empirical study): "Next, we evaluated Muesli on learning 9x9 Go from self-play."
- Section 5 (An empirical study): "Finally, we tested the same agents on MuJoCo environments in OpenAI Gym (Brockman et al., 2016), to test if Muesli can be effective on continuous domains and on smaller data budgets (2M frames)."
- Section 5 (An empirical study): "For each update, we separately tuned hyperparameters on 10 of the 57 Atari games."

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{(58 + N_{\text{MuJoCo}})\ \text{tasks}}{(58 + N_{\text{MuJoCo}})\ \text{models}} = 1
}
$$

\(N_{\text{MuJoCo}}\): **Not specified in the paper.**
