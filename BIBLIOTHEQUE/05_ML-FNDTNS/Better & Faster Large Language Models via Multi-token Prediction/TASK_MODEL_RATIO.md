1. **Number of distinct tasks evaluated: 22**

   Verbatim evidence:
   - "We compare models with 7B parameters trained from scratch on 200B and on 314B bytes of code on the MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021) and APPS (Hendrycks et al., 2021) benchmarks." (Section 3.2/Table 1)
   - "on the CodeContests dataset (Li et al., 2022)." (Section 3.6)
   - "We evaluate the models from Section 3.7 on standard natural language processing benchmarks: ARC Challenge (Yadav et al., 2019), COPA (Roemmele et al., 2011), Hellaswag (Zellers et al., 2019), Natural Questions (Kwiatkowski et al., 2019), PIQA (Bisk et al., 2019), SIQA (Sap et al., 2019) and TriviaQA (Joshi et al., 2017)." (Appendix G)
   - "For summarization, we use eight benchmarks where ROUGE metrics (Lin, 2004) with respect to a ground-truth summary allow automatic evaluation of generated texts." (Section 3.7)
   - "For natural language mathematics, we evaluate the pretrained models in 8-shot mode on the GSM8K benchmark (Cobbe et al., 2021) and measure accuracy of the final answer produced after a chain-of-thought elicited by the fewshot examples." (Section 3.7)
   - "Section 4.1 shows that for small model sizes, *induction capability*—as discussed by Olsson et al. (2022)—either only forms when using multi-token prediction as training loss, or it is vastly improved by it. Moreover, Section 4.2 shows that multi-token prediction improves generalization on an arithmetic task, even more so than tripling model size." (Section 4)

2. **Number of trained model instances required to cover all tasks: 13**

   Verbatim evidence and counting:
   - Code benchmarks can be covered by one code-trained model instance: "We compare models with 7B parameters trained from scratch on 200B and on 314B bytes of code on the MBPP (Austin et al., 2021), HumanEval (Chen et al., 2021) and APPS (Hendrycks et al., 2021) benchmarks." (Section 3.2/Table 1) -> 1 model for 3 tasks.
   - CodeContests requires task-specific finetuning: "Pretrained models with multi-token prediction loss also outperform next-token models for use in finetunings. We evaluate this by finetuning 7B parameter models from Section 3.3" and "on the CodeContests dataset (Li et al., 2022)." (Section 3.6) -> +1 model.
   - Natural-language choice benchmarks and GSM8K are evaluated from pretrained language models: "To evaluate multi-token prediction training on natural language, we train models of size 7B parameters on 200B tokens of natural language with a 4-token, 2-token and next-token prediction loss, respectively. In Figure 5, we evaluate the resulting checkpoints on 6 standard NLP benchmarks." and "For natural language mathematics, we evaluate the pretrained models in 8-shot mode on the GSM8K benchmark (Cobbe et al., 2021) and measure accuracy of the final answer produced after a chain-of-thought elicited by the fewshot examples." (Section 3.7) -> +1 model.
   - Summarization needs task-specific finetuning per benchmark: "For summarization, we use eight benchmarks where ROUGE metrics (Lin, 2004) with respect to a ground-truth summary allow automatic evaluation of generated texts. We finetune each pretrained model on each benchmark's training dataset for three epochs and select the checkpoint with the highest ROUGE-L  $F_1$  score on the validation dataset." (Section 3.7) -> +8 models.
   - Induction capability uses separately trained models: "Training small models of sizes 1M to 1B nonembedding parameters on a dataset of children stories, we measure induction capability by means of an adapted test set." (Section 4.1) -> +1 model.
   - Polynomial arithmetic uses separately trained models: "We train and evaluate models on a task on polynomial arithmetic in the ring  $\mathbb{F}_7[X]/(X^5)$  with unary negation, addition, multiplication and composition of polynomials as operations." (Section 4.2) -> +1 model.

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{22\ \text{tasks}}{13\ \text{models}} = 1.69
}
$$
