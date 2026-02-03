1. Number of distinct tasks evaluated: 12 (WMT14 En-De, WMT14 En-Fr, GLUE's eight datasets, SQuAD, RACE).
> "For neural machine translation problems (WMT14 En-De and En-Fr)" (Section B.1)
> "GLUE benchmark consists of eight datasets and each have different hyperparameter settings." (Section B.3)
> "SQuAD benchmark." (Section B.3)
> "RACE benchmark." (Section B.3)

2. Number of trained model instances required to cover all tasks: 12 models.
> "With the warm-initialized FLOWER checkpoint, retrain on the same dataset for 10 epochs (En-De) or 1 epoch (En-Fr)." (Section B.2)
> "GLUE benchmark consists of eight datasets and each have different hyperparameter settings." (Section B.3)
> "For this benchmark we wrote our own finetuning code because currently there is no official code available." (Section B.3)
> "Similar to other benchmarks, we then repeat the training process using exactly the same training hyperparameters to make a fair comparison. In this benchmark we freeze the weights  $w_h$  and only finetune the weights of RoBERTa." (Section B.3)

$$
oxed{
rac{12\ 	ext{tasks}}{12\ 	ext{models}} = 1
}
$$
