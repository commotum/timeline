1. **Number of distinct tasks evaluated:** 1

Verbatim evidence: "A goal of statistical language modeling is to learn the joint probability function of sequences of words in a language." (Abstract)

Verbatim evidence: "The objective is to learn a good model  $f(w_t, \cdots, w_{t-n+1}) = \hat{P}(w_t|w_1^{t-1})$ , in the sense that it gives high out-of-sample likelihood." (Section 2. A Neural Model)

Verbatim evidence: "Comparative experiments were performed on the Brown corpus" and "An experiment was also run on text from the Associated Press (AP) News from 1995 and 1996." (Section 4. Experimental Results)

2. **Number of trained model instances required to cover all tasks:** 1

Verbatim evidence: "The model learns simultaneously (1) a distributed representation for each word along with (2) the probability function for word sequences, expressed in terms of these representations." (Abstract)

Verbatim evidence: "Below are measures of test set perplexity (geometric average of  $1/\hat{P}(w_t|w_1^{t-1})$ ) for different models  $\hat{P}$ ." (Section 4.2 Results)

Verbatim evidence: "mix: whether the output probabilities of the neural network are mixed with the output of the trigram (with a weight of 0.5 on each)." (Table 1 caption, Section 4.2 Results)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{1\ \text{task}}{1\ \text{model}} = 1
}
$$
