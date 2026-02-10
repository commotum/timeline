1. **Number of distinct tasks evaluated**: Not specified in the paper as a fixed numeric total. The paper explicitly specifies multitask sparse parity subtasks plus a language-modeling task, i.e., \(n_{\mathrm{tasks}} + 1\) tasks.
   - "For each subtask  $i \in \{1, \ldots, n_{\mathrm{tasks}}\}$ , we sample a random subset  $S_i$  of k bits from the task bits." (Section: The Quanta Hypothesis / Multitask sparse parity)
   - "We study this with the Pythia suite of language models trained by Eleuther AI on The Pile corpus." (Section: Large language model scaling)

2. **Number of trained model instances required to cover all tasks**: 2 models.
   - "We train ReLU MLPs with a single hidden layer using the Adam optimizer." (Section: The Quanta Hypothesis / Multitask sparse parity)
   - "We study this with the Pythia suite of language models trained by Eleuther AI on The Pile corpus." (Section: Large language model scaling)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{(n_{\mathrm{tasks}} + 1)\ \text{tasks}}{2\ \text{models}} = \frac{n_{\mathrm{tasks}} + 1}{2}
}
$$
