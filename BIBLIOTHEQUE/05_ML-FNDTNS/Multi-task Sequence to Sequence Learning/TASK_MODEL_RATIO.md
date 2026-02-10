1. **Number of distinct tasks evaluated:** 7

> "We evaluate the multi-task learning setup on a wide variety of sequence-to-sequence tasks: constituency parsing, image caption generation, machine translation, and a number of unsupervised learning as summarized in Table 1." (Section "4 EXPERIMENTS")

> "| English→German Translation     |"
> "| German→English Translation     |"
> "| English unsupervised           |"
> "| German unsupervised            |"
> "| Penn Tree Bank Parsing         |"
> "| High-Confidence Corpus Parsing |"
> "| Image Captioning               |" (Section "4.1 DATA", Table 1)

2. **Number of trained model instances required to cover all tasks:** 4

> "As described in Section 3, for each multi-task experiment, we need to choose one task to be the *reference task* (which corresponds to  $\alpha_1=1$ )." (Section "4.2 Training Details")

> "| Translation + PTB Parsing (0.01x) |" (Section "4.3.1 LARGE TASKS WITH SMALL TASKS", Table 2)
> "| Translation + Captioning (0.05x) |" (Section "4.3.2 Large Tasks With Medium Tasks", Table 3)
> "| Translation + HC Parsing (0.1x)  |" (Section "4.3.3 Large Tasks with Large Tasks", Table 4)
> "| Translation + autoencoders (0.05x)               |" (Section "4.3.4 MULTI-TASKS AND UNSUPERVISED LEARNING", Table 6)

3. **Task–Model Ratio**

$$
\boxed{
\frac{7\ \text{tasks}}{4\ \text{models}} = 1.75
}
$$
