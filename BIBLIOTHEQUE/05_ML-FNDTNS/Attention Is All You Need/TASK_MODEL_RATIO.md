1. **Number of distinct tasks evaluated:** 3. "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train." (Abstract) "On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models..." (Section 6.1 Machine Translation) "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0..." (Section 6.1 Machine Translation) "To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing." (Section 6.3 English Constituency Parsing)

2. **Number of trained model instances required to cover all tasks:** 3. "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs." (Section 5.1 Training Data and Batching) "For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences..." (Section 5.1 Training Data and Batching) "The Transformer (big) model trained for English-to-French used dropout rate  $P_{drop}=0.1$ , instead of 0.3." (Section 6.1 Machine Translation) "We trained a 4-layer transformer with  $d_{model}=1024$  on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences." (Section 6.3 English Constituency Parsing)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
