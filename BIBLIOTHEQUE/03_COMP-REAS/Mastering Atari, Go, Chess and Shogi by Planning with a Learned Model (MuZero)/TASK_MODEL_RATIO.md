1. **Number of distinct tasks evaluated:** 60

"When evaluated on 57 different Atari games - the canonical video game environment for testing AI techniques, in which model-based planning approaches have historically struggled - our new algorithm achieved a new state of the art. When evaluated on Go, chess and shogi, without any knowledge of the game rules, *MuZero* matched the superhuman performance of the *AlphaZero* algorithm that was supplied with the game rules." (Abstract)

"We applied the *MuZero* algorithm to the classic board games Go, chess and shogi <sup>2</sup>, as benchmarks for challenging planning problems, and to all 57 games in the Atari Learning Environment [2], as benchmarks for visually complex RL domains." (Section 4 Results)

2. **Number of trained model instances required to cover all tasks:** 60

"For each board game, we used 16 TPUs for training and 1000 TPUs for selfplay. For each game in Atari, we used 8 TPUs for training and 32 TPUs for selfplay." (Appendix G Training)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{60\ \text{tasks}}{60\ \text{models}} = 1
}
$$
