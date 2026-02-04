1. Number of distinct tasks evaluated: 3
"For the prediction of next event type and timestamp, we train two linear layers  $W^e, W^t$" (Section 3.1.2 Training)
"By definition,  $\mathcal{L}_{event}$  measures the accuracy of the event type prediction and  $\mathcal{L}_{time}$  measures the mean square loss the of time prediction." (Section 3.1.2 Training)
"The evaluation metrics employed were log-likelihood and accuracy." (Section 4.4 Result)
"**Accuracy and RMSE** We consider the accuracy and Root Mean Square Error (RMSE) estimation on the following datasets: Financial, Mimic-II, and SO." (Section 4.4 Result)

2. Number of trained model instances required to cover all tasks: 1
"Denote the log-likelihood of  $\mathcal{S}$  as  $\mathcal{L}$ , then the training loss can be defined by" (Section 3.1.2 Training)
"$$\mathcal{L}(\mathcal{S}) = -\mathcal{L} + \beta_1 \mathcal{L}_{event}(\mathcal{S}) + \beta_2 \mathcal{L}_{time}(\mathcal{S}), \tag{24}$$" (Section 3.1.2 Training)

3. Task–Model Ratio:
$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$
