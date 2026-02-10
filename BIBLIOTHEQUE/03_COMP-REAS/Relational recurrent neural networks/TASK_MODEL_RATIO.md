1. **Number of distinct tasks evaluated:** 12

   Verbatim evidence: "$N^{th}$ Farthest The $N^{th}$ Farthest task is designed to stress a capacity for relational reasoning across time." (Section 4.1); "is broken down into three categories: *addition, control,* and *full program*." (Section 4.1, Program Evaluation); "we also evaluated on *memorization tasks*" (Section 4.1) with "Copy", "Reverse", and "Double" shown as memorization tasks (Figure 7, Appendix A.2); "**Mini Pacman with viewport**" and "another RL task called BoxWorld" (Section 4.2); "For all three language modeling tasks" (Section 5.4).

2. **Number of trained model instances required to cover all tasks:** 12

   Verbatim evidence: "For all models (RMC, LSTM, DNC) we used the Adam optimiser [44] with a batch size of 1600, learning rates tuned between  $1e^{-5}$  and  $1e^{-3}$ , and trained using a softmax cross entropy loss function." (Appendix A.1); "Each model was trained for  $200 \mathrm{K}$  iterations" (Appendix A.2); "The sequential model consists of an encoder and a decoder" (Appendix A.2); "We trained an Importance Weighted Actor-Learner Architectures agent augmented with the RMC on BoxWorld levels" (Appendix A.3.1); "We trained the Recurrent Memory Core" and "We used these same parameters for GigaWord and Project Gutenberg" (Appendix A.4). A single jointly trained model covering all tasks is **Not specified in the paper.**

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{12\ \text{tasks}}{12\ \text{models}} = 1
}
$$
