> "We first conduct experiments on the recent OGB-LSC [21] quantum chemistry regression (i.e., PCQM4M-LSC) challenge, which is currently the biggest graph-level prediction dataset and contains more than 3.8M graphs in total. Then, we report the results on the other three popular tasks: ogbgmolhiv, ogbg-molpcba and ZINC, which come from the OGB [22] and benchmarking-GNN [14] leaderboards." (Section 4 Experiments)
>
> "Since pre-training is encouraged by OGB, we mainly explore the transferable capability of a Graphormer model pre-trained on OGB-LSC (i.e., PCQM4M-LSC)." (Section 4.2 Graph Representation)
>
> "**Fine-tuning.** Table 8 summarizes the hyper-parameters used for fine-tuning Graphormer on OGBG-MolPCBA." (Section B.2.2 OGBG-MolPCBA)
>
> "**Fine-tuning.** The hyper-parameters for fine-tuning Graphormer on OGBG-MolHIV are presented in Table 9." (Section B.2.3 OGBG-MolHIV)
>
> "For benchmarking-GNN, which does not encourage large pre-trained model, we train an additional Graphormer<sub>SLIM</sub> (L=12, d=80, total param.= 489K) from scratch on ZINC." (Section 4.2 Graph Representation)

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$
