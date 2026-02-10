1. **Number of distinct tasks evaluated:** 60

   - "We compare our algorithms' performance on 60 different Atari 2600 games in the Arcade Learning Environment (Bellemare et al., 2013) using Bellemare et al.'s inter-algorithm score distribution." (Section 5: Experimental Results).
   - "Our experiments comprise 60 Atari 2600 games in ALE (Bellemare et al., 2013), with \"life\" loss treated as episode termination." (Section F: Experimental Methods).

2. **Number of trained model instances required to cover all tasks:** 60

   - "As before, for each  $\lambda$  we compute the inter-algorithm scores on a per-game basis." (Section F.1: Algorithmic Performance in Function of $\lambda$).
   - "If  $g \in \{1,\ldots,60\}$  is a game and  $z_{g,a}$  the inter-algorithm score on g for algorithm g, then the score distribution function is  $f(x) := |\{g: z_{g,a} \ge x\}|/60$ ." (Section 5: Experimental Results).
   - "Reported performance averages over four trials with different random seeds for each experimental configuration." (Section F: Experimental Methods).

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{60\ \text{tasks}}{60\ \text{models}} = 1
}
$$
