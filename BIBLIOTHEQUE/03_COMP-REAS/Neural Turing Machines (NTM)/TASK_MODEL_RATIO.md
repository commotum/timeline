1. Number of distinct tasks evaluated: **5**.
   Verbatim task names:
   - "## **4.1** Copy" (Section 4.1)
   - "## 4.2 Repeat Copy" (Section 4.2)
   - "#### 4.3 Associative Recall" (Section 4.3)
   - "## 4.4 Dynamic N-Grams" (Section 4.4)
   - "## 4.5 Priority Sort" (Section 4.5)
   Supporting quote: "Tables 1 to 3 give details about the network configurations and learning rates used in the experiments." (Section 4.6)

2. Number of trained model instances required to cover all tasks: **5 models**.
   Verbatim evidence of task-specific training:
   - "The networks were trained to copy sequences of eight bit random vectors, where the sequence lengths were randomised between 1 and 20." (Section 4.1)
   - "The networks were trained to reproduce sequences of size eight random binary vectors, where both the sequence length and the number of repetitions were chosen randomly from one to ten." (Section 4.2)
   - "During training, we used a minimum of 2 items and a maximum of 6 items in a single episode." (Section 4.3)
   - "For each training example, we first generated random 6-Gram probabilities by independently drawing all 32 probabilities from the  $Beta(\frac{1}{2},\frac{1}{2})$  distribution." (Section 4.4)
   - "Each input sequence contained 20 binary vectors with corresponding priorities, and each target sequence was the 16 highest-priority vectors in the input.<sup>5</sup>" (Section 4.5)
   Jointly training one single model across all five tasks: **Not specified in the paper.**

3. Task–Model Ratio:

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
