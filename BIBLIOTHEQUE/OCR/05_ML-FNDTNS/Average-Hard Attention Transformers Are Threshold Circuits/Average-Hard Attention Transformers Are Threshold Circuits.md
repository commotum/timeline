# Average-Hard Attention Transformers are Constant-Depth Uniform Threshold Circuits

#### Lena Strobl

Department of Computing Science Umeå University, Sweden lena.strobl@umu.se

## **Abstract**

Transformers have emerged as a widely used neural network model for various natural language processing tasks. Previous research explored their relationship with constant-depth threshold circuits, making two assumptions: average-hard attention and logarithmic precision for internal computations relative to input length. Merrill et al. (2022) prove that average-hard attention transformers recognize languages that fall within the complexity class TC<sup>0</sup>, denoting the set of languages that can be recognized by constant-depth polynomial-size threshold circuits. Likewise, Merrill and Sabharwal (2023a) show that log-precision transformers recognize languages within the class of uniform TC<sup>0</sup>. This shows that both transformer models can be simulated by constantdepth threshold circuits, with the latter being more robust due to generating a uniform circuit family. This paper shows that the first result can be extended to yield uniform circuits as well.

# 1 Introduction

The dominance of recurrent neural network (RNN) architectures in the realm of natural language processing gradually waned with the advent of transformers, as initially introduced by Vaswani et al. (2017). Unlike RNNs, which heavily rely on autoregressive mechanisms, transformers revolutionized the field by leveraging parallelism to process sequential data.

While RNNs could be analyzed through the lens of automata theory (notably by Weiss et al. (2018); Peng et al. (2018)) thanks to their recurrence-based nature, the characterization of transformers necessitates a different approach. Considering the circuit-based perspective seems natural, given the absence of explicit recurrence in transformers. Notably, some studies have attempted to reintroduce recurrences into transformers, as exemplified by the work on shortcut connections by Liu et al. (2023).

However, for coherence and maintaining the focus of our discussion, we will refrain from delving deeper into these tangential directions.

Recent advances in the analysis of transformer models have shed light on their computational capabilities, particularly through the investigation of two distinct formal models: average-hard (= saturated) by Merrill et al. (2022) and softmax (= soft) transformers by Merrill and Sabharwal (2023a). Average-hard attention enables a connection to be established between these models and devices of formal language theory. Building upon this line of research, Merrill and Sabharwal (2023a) introduced a different model, demonstrating that transformer networks with logarithmic precision in relation to the input length can be simulated by constant-depth uniform threshold circuits. Consequently, the complexity class TC<sup>0</sup> serves as an upper bound for the formal languages recognized by these transformers.

Motivated by the inherent uniformity possessed by transformers, we want to investigate whether average-hard attention transformers only recognize languages in uniform TC<sup>0</sup>. Our primary contribution lies in our proof, showcasing that average-hard attention transformers can indeed be simulated by uniform TC<sup>0</sup> circuits, thereby solidifying their association with uniform TC<sup>0</sup>. Consequently, these transformers are inherently limited to solving problems within uniform TC<sup>0</sup>.

This result does not follow from the result presented by Merrill and Sabharwal (2023a), as both the underlying assumptions and the specific attention mechanisms differ between the two studies. Concretely, we consider the implications of the results from Merrill and Sabharwal (2023a) and Merrill et al. (2022): Merrill et al. (2022, Theorem 4) demonstrated that average-hard attention transformers are only capable of producing floating-point numbers of logarithmic size. Consequently, one might argue that average-hard attention trans-

formers can be considered log-precision transformers, and therefore the result Merrill and Sabharwal (2023a) establish should be applicable in this context. Merrill et al. (2022) Theorem 4 relies on the assumption of "size preserving" functions, while we adopt the fundamental definitions provided by Merrill and Sabharwal (2023a). This discrepancy in the underlying assumptions creates a distinction between the two frameworks. Furthermore, it should be emphasized that the attention mechanism Merrill and Sabharwal (2023a) employed is softmax, which is a difference in the formal definition of attention itself. As a result, even disregarding the question of precision, the direct applicability of Merrill and Sabharwal (2023a) result would be incorrect as it would disregard the difference in the attention mechanisms.

The findings of this paper open up new approaches for future research, specifically in exploring the distinction between average-hard and softmax attention mechanisms, with the potential to unveil a clear demarcation between the two.

#### Previous work:

| TICVIOUS WOIK.         |                                  |
|------------------------|----------------------------------|
| Transformer Model      | Simulated by                     |
| average-hard attention | TC <sup>0</sup> circuits         |
| log-precision          | uniform TC <sup>0</sup> circuits |

This paper:

| average-hard attention | uniform $TC^0$ circuits |
|------------------------|-------------------------|
|------------------------|-------------------------|

Table 1: Overview of previous results by Merrill et al. (2022); Merrill and Sabharwal (2023a) (top) and the contribution of this paper (bottom). Previous research demonstrated the simulation of average-hard attention transformers using  $\mathsf{TC}^0$  circuits with integer values. Another study showed the simulation of log-precision transformers with uniform  $\mathsf{TC}^0$  circuits. In this paper, we extend these results by demonstrating the simulation of average-hard attention transformers with uniform  $\mathsf{TC}^0$  circuits.

# 2 Preliminaries

In this section, we establish the foundational definitions and notation for circuit computations, drawing from the textbook of Arora and Barak (2009) in Chapters 6 and 14. This established framework forms the basis for our subsequent analysis.

Moreover, we revisit the average-hard attention transformer model proposed by Merrill and Sabhar-

wal (2023a). To do so, we provide definitions of average-hard attention and average-hard attention heads. These essential concepts serve as the cornerstone of the average-hard attention transformer model.

# 2.1 Basic Mathematical Notation and Definitions

We employ the notation and definitions commonly used in mathematics and formal language theory. Specifically, we denote the sets of natural numbers, including zero, and integers as  $\mathbb N$  and  $\mathbb Z$  respectively.

For any natural number n, the set containing the numbers from 1 to n (inclusive) is denoted as [n]. Notably, when n=0, we represent the set as  $\emptyset$ .

The set of all strings composed of elements from a given set  $\Sigma$  is represented as  $\Sigma^*$ . Here, we denote the empty string as  $\epsilon$ , and we define  $\Sigma^+ = \Sigma^* \setminus \{\epsilon\}$ .

The canonical extension of a function  $f \colon \Sigma \to \Delta$  to a function from  $\Sigma^*$  to  $\Delta^*$  is denoted by f as well. Thus,  $f(\sigma_1 \cdots \sigma_n) = f(\sigma_1) \cdots f(\sigma_n)$  for all  $\sigma_1, \ldots, \sigma_n \in \Sigma$ . This notation allows us to apply the function f to each individual element within the string.

Due to the inherent limitations of Boolean circuits, which can only process values of 1 and 0, representing floating point numbers used in neural networks becomes a challenge. To accommodate this discrepancy, these numerical values are transformed into bit strings, belonging to the set  $\{0,1\}^*$ . Furthermore, the operations performed on these bit strings must be simulated through Boolean operations, which are the fundamental building blocks available to the specific circuit type under consideration. Consequently, any manipulations or computations on these floating point representations necessitate a translation into operations that can be expressed using the available Boolean operations.

**Binary representation.** The binary representation of  $n \in \mathbb{Z}$  is the unique string

$$w = b_0 b_1 \cdots b_m \in \{0, 1\}^+$$

with  $b_1=1$  if m>0, and  $n=-1^{b_0}\sum_{i=1}^m b_i 2^{i-1}$ . We denote the length m of this representation by  $\|n\|$ , i.e.,

$$||n|| = \lceil \log_2(|n|+1) \rceil + 1.$$

**Precision.** Let  $p \in \mathbb{N}$  be called *precision*. Following Merrill and Sabharwal (2023a) work, we define the set  $\mathbb{F}_p$  to be the set of all rational numbers that can be written as  $m \cdot 2^z$  where  $m, z \in \mathbb{Z}$  are such that  $\|m\|, \|z\| \le p/2$ . (Thus, we may always assume that p is positive and even because  $\mathbb{F}_0 = \emptyset$  and for odd p,  $\mathbb{F}_p = \mathbb{F}_{p-1}$ .)

In other words, a number in  $\mathbb{F}_p$  can be specified by two bit strings of length p/2 denoting the mantissa m and the exponent z.

**Arithmetic on floats.** Float arithmetic involves performing operations on floating-point numbers by first carrying out computations in  $\mathbb{Q}$  and then managing potential overflow and excess bits.

To formalize this process, we introduce a value q, defined as  $2^{\lfloor p/2 \rfloor - 1} - 1$ , where p represents the precision. This value represents the largest natural number q such that  $||q|| \le p/2$ .

Given a rational number  $r \in \mathbb{Q}$ , and let  $[r]_p$  denote the truncation of r to a float in  $\mathbb{F}_p$ , assuming r>0. This truncation is defined as follows. To determine the exponent z, we select a value within the range -q to q such that multiplying r by  $2^z$  scales it as much as possible without exceeding q. Next, we truncate the mantissa, retaining up to  $\lfloor p/2 \rfloor -1$  bits (unless z=q, indicating that the exponent would result in an overflow).

$$[r]_p = \begin{cases} -\left[-r\right]_p & \text{if } r < 0, \\ q \cdot 2^q & \text{if } r > q \cdot 2^q, \text{ and } \\ \left\lfloor q \cdot 2^z \right\rfloor \cdot 2^{-z} & \text{if } 0 \leq r \leq q \cdot 2^q, \end{cases}$$

where z is the largest integer such that  $-q \le z \le q$  and  $r \cdot 2^z \le q$ .

Note, that the choice of z in the third case ensures that we retain the maximum number of bits in the mantissa during the truncation process.

For instance, we have p=6 and  $r=0.0101_2$ , then we select z=q=3. Consequently, the truncation of r to p bits, denoted as  $[r]_p$ , is given by  $|10.1_2| \cdot 2^{-3} = 0.01_2$ .

#### 2.2 Circuit computations

Formally,

**Circuits.** A *Boolean circuit*, denoted as C, is a directed acyclic procedural computational graph that encompasses binary input gates, represented as  $\mathsf{in}_1,\ldots,\mathsf{in}_n$ , which serve as the leaf nodes of the graph. These input gates correspond to n input values, each taking the value of either 1 or 0. Intermediate nodes within the circuit, referred to

as internal gates, are composed of basic Boolean functions such as logical OR  $(\lor)$ , logical AND  $(\land)$ , and logical NOT  $(\neg)$ .

The output of the circuit, denoted as C(x), is determined recursively by applying the logical operations from the input gates through the graph until reaching the root. Thus, a Boolean circuit defines a function mapping inputs from  $\{0,1\}^k$  to outputs in  $\{0,1\}$ . It is also possible to consider circuits with a multiple output gates, allowing the computation of functions from  $\{0,1\}^k$  to  $\{0,1\}^\ell$ , where  $\ell$  represents the number of output gates.

**Size and depth** The size of a circuit, denoted as |C|, is the number of nodes present within the graph. Additionally, we define the depth of the circuit as the longest directed path within it, capturing the length of the computational flow from the input gates to the output.

In Fig. 1, a circuit is depicted that performs the logical operation XOR by receiving two inputs and producing an output of 1 if exactly one of the inputs is 1; otherwise, it outputs 0.

![](_page_2_Figure_15.jpeg)

Figure 1: A circuit of size 7 and depth 3 performing the logical XOR operation on a 2-bit input.

**Circuit Families** In traditional circuit theory, circuits are limited to operating on a fixed input size. However, we need a model that can handle inputs of arbitrarily long strings as input. As is customary, we thus use a *circuit family*: a collection  $(C_n)_{n \in \mathbb{N}}$ , where each circuit  $C_n$  has n inputs gates. Consequently, the size and depth of circuits within this family become functions of n, allowing for flexibility in handling inputs of varying lengths.

We now recall the definitions of two fundamental classes of circuit families.

**Definition 1** (The Class  $AC^0$ ). A language L belongs to the class  $AC^0$  if it can be decided by a circuit family  $(C_n)_{n\in\mathbb{N}}$ , where each circuit  $C_n$  is

constructed only using gates from the set  $\{\land, \lor, \neg\}$ . Furthermore, the circuits in  $(C_n)_{n \in \mathbb{N}}$  are required to have polynomial size and constant depth in n.

**Definition 2** (The Class  $\mathsf{TC}^0$ ). Let majority be the function that takes a sequence of bits as input and returns 1 if the number of 1s in the sequence is greater than the number of 0s, and 0 otherwise. A language L is in  $\mathsf{TC}^0$  if it can be decided by a circuit family  $(C_n)_{n\in\mathbb{N}}$ , where each circuit  $C_n$  is constructed using gates from the set  $\{\land,\lor,\lnot,\mathsf{majority}\}$ . Similar to  $\mathsf{AC}^0$ , the circuits in  $(C_n)_{n\in\mathbb{N}}$  have polynomial size and constant depth in n.

**Uniform Circuit Families** A family of circuits  $(C_n)_{n\in\mathbb{N}}$  is called *logspace uniform*, or simply *uniform*, if there exists a Turing machine (TM) that can compute  $C_n$  from  $1^n$  (the number n in unary notation) using  $O(\log n)$  space. In particular, uniform  $\mathsf{TC}^0$  is the set of languages that can be decided by a uniform  $\mathsf{TC}^0$  circuit family.

In the context of this paper, the transformers under study operate on vectors over  $\mathbb{F}_p$ . In circuit representations, we typically assume that elements of  $\mathbb{F}_p$  are encoded as bit strings of length p, obtained by concatenating the mantissa and exponent in binary notation. To ensure consistency, these bit strings are padded with leading zeroes after the sign bit, making both components exactly p/2 in length.

A mapping  $f: (\mathbb{F}_p^k)^n \to (\mathbb{F}_p^k)^n$  is considered *uniformly*  $AC_0$  *computable* (or *uniformly*  $TC_0$  *computable*) if there exists a uniform  $AC_0$  (or  $TC_0$ ) circuit family that, given the bit string representation of  $x_1 \cdots x_n \in (\mathbb{F}_p^k)^n$  as input, computes  $f(x_1, \ldots, x_n)$ .

#### 2.3 Transformers

A transformer model is composed of a finite number of layers, where each layer comprises multiple so-called attention heads working in parallel, followed by a feed-forward network. Fig. 2 provides a visual representation of the layer's structure, the arrangement of an individual attention head within the layer as well as the internal configuration of an attention head can be observed in Fig. 3.

In this paper, particular emphasis is placed on the attention head component. The attention head implements the attention mechanism, which facilitates the mapping of a sequence of n vectors to a probability distribution over the set [n], ultimately yielding a weighted sum of these vectors.

![](_page_3_Figure_8.jpeg)

Figure 2: Schematic representation showcasing the structure of a transformer layer l+1 with the outputs of layer l as inputs, namely  $X_l=(x_1^l,\ldots,x_n^l)$ , attention heads  $\operatorname{Att}_{\sigma_1}^l,\ldots\operatorname{Att}_{\sigma_n}^l$  and them being combined with a feed forward network f to produce the output  $X_{l+1}$ .

The findings presented in this paper are based on the assumption that the vector components are floats with a precision of  $O(\log n)$ , where n denotes the length of the input. Consequently, the analysis is focused on transformers in which all internal computations occur within  $\mathbb{F}_p$ , where the value of p is determined by  $p = c_1 \log n + c_0$ , with  $c_0$  and  $c_1$  as constants greater than zero. Throughout the remainder of this paper, the symbol p represents this specific value, which depends on the input length.

In the subsequent sections of this paper, we will consider a fixed natural number, denoted as k, which corresponds to the number of dimensions of the vectors handled by the transformer under consideration.

### 2.4 Attention

Attention within a transformer model is computed using attention heads. In this study, our analysis focuses on average-hard attention, as introduced by Merrill et al. (2022). We proceed by formalizing the concepts of average-hard attention and average-hard attention heads.

**Definition 3** (Average-hard attention function). For  $s = s_1 \cdots s_n \in \mathbb{F}_p^+$ , let  $\mathcal{M}(s) = \{i \in [n] \mid s_i = \max_{j \in [n]} s_j\}$ . The average-hard attention function  $\xi$  maps s to the probability distribution  $\xi(s) \colon [n] \to [0,1]$  given by

$$\xi(s)_i = \begin{cases} 1/\mathcal{M}(s) & \text{if } i \in \mathcal{M}(s) \\ 0 & \text{otherwise.} \end{cases}$$
 (1)

Thus, average-hard attention distributes the entire probability mass evenly among the indices whose values  $s_i$  are maximal.

The attention head induced by s computes the sequence of n scores, denoted as  $\sigma_{x_i}(X) = \sigma_{x_i}(x_1) \cdots \sigma_{x_i}(x_n)$ , for each input  $x_i$ . Subsequently, this sequence is transformed into a probability distribution using the average-hard attention function  $\xi$ , and the resulting attention value at position i is the weighted sum of  $x_1, \ldots, x_n$  based on this distribution. The formal definition follows.

**Definition 4** (Average-hard attention head). Let  $\sigma: \mathbb{F}_p^k \times \mathbb{F}_p^k \to \mathbb{F}_p$ , be a linear space computable function called a *scoring function*. We usually write  $\sigma_x(x')$  for  $\sigma(x,x')$ , called the *score* of x' with respect to x.

The average-hard attention head induced by  $\sigma$  is the function  $Att_{\sigma} \colon (\mathbb{F}_p^k)^* \to \mathbb{F}_p^*$ , such that for all  $n \in \mathbb{N}$ ,  $X = x_1 \cdots x_n \in (\mathbb{F}_p^k)^*$ , and  $i \in [n]$ , we have

$$Att_{\sigma}(X)_{i} = \left[\xi(\sigma_{x_{i}}(X)) \cdot X^{\mathrm{T}}\right]_{p}, \qquad (2)$$

where  $X^{\mathrm{T}}$  is the transpose of X (viewed as an n-dimensional vector) and  $\cdot$  denotes matrix multiplication.

While the attention function in a typical transformer model is not average-hard, we specifically focus on the analysis of average-hard attention transformers in this paper. For clarity, let  $s_{ij}$  represent the score assigned to  $x_j$  with respect to  $x_i$ , denoted as  $s_{ij} = \sigma_{x_i}(x_j)$ . Applying Definition 3 to Eq. (2) yields

$$Att_{\sigma}(X)_{i} = \left[\sum_{j \in \mathcal{M}(s)} \frac{x_{j}}{|\mathcal{M}(s)|}\right]_{p}$$

$$= \left[\frac{1}{|\mathcal{M}(s)|} \cdot \sum_{j \in \mathcal{M}(s)} x_{j}\right]_{p} \tag{3}$$

for every  $i \in [n]$ .

#### 3 Main result

In this section, we present a construction for attention that, when integrated into the construction of a constant-depth uniform threshold circuit as described by Merrill and Sabharwal (2023a), enables the complete simulation of an average-hard attention transformer. Our approach relies on the

utilization of a fundamental lemma established by Merrill and Sabharwal (2023b) (note that this is an earlier version of the same paper).

**Lemma 1** (Merrill and Sabharwal (2023b, Lemma 3)). Let  $f: \{0,1\}^* \to \{0,1\}$  be a linear space computable boolean function and  $c \in \mathbb{R}^+$ . There exists a TM that, for all  $n \in \mathbb{N}$ , uses  $O(\log n)$  space to map input  $1^n$  to a circuit of size at most  $n^c + c \cdot \log n + 1$  and depth 3 that computes f on inputs of size  $c \cdot \log n$ .

In our paper, we will have functions that transform bit-sequences to bit-sequences and not just to  $\{0,1\}$ . Here, Lemma 1 still suffices. In principle, we could have the input length as an additional input to the circuit, e.g., using a one-hot vector which is 1 at the position of the bit we want to output. Then we take all these circuits for the entire input length, just copy them and iterate over the position (first one outputs bit 1, second outputs bit 2, ...). That does not change the size of the circuit significantly, since we have a size of at most  $n^c$ . The length is polynomial in  $\log n$ , so we can produce them all separately. Hence, Lemma 1 can be and is used for functions from now on which output a bit-sequence, e.g., addition.

Note that Lemma 1 does not mean that f is computable by uniform  $\mathsf{TC}^0$  circuits. The circuits work on input of size  $\log n$  and are thus, relative to this, exponentially large. However, used in a circuit that is applied to a sequence of n values of size  $\log n$  each (such as the bit string representation of elements of  $\mathbb{F}_p$  the size does indeed become polynomial (now in  $n \cdot \log n$ )). This lemma provides a significant implication: certain key operations performed by a transformer head can be effectively implemented using circuits.

**Lemma 2.** Let  $p = O(\log n)$ . The functions listed below can be computed by uniform families of  $TC^0$  circuits, of size polynomial in n:

- 1.  $\sigma: \mathbb{F}_p^k \times \mathbb{F}_p^k \to \mathbb{F}_p$ : Every scoring function  $\sigma$ ,
- 2. max:  $\mathbb{F}_p^n \to \mathbb{F}_p$ : computes the maximum of its arguments,
- 3. eq:  $\mathbb{F}_p \times \mathbb{F}_p \to \{0,1\}$ : such that eq(x,y) = 1 if and only if x = y,
- 4. sel:  $\mathbb{F}_p^k \times \{0,1\} \to \mathbb{F}_p^k$ : such that sel(x,y) = x if y = 1 and sel(x,y) = 0 otherwise, for all  $x \in \mathbb{F}_p$ ,

- 5. sum:  $(\mathbb{F}_p^k)^n \to \mathbb{F}_p^k$ : given as the function sum $(x_1,\ldots,x_n) = [\sum_{i=1}^n x_i]_p$  for all  $x_1,\ldots,x_n \in \mathbb{F}_p^k$ , and
- 6. div:  $\mathbb{F}_p^k \times [n] \to \mathbb{F}_p^k$ : given by div $(x,d) = [x/d]_p$  for all  $x \in \mathbb{F}_p^k$  and  $d \in [n]$ , where x/d is defined componentwise.

*Proof.* The computability of scoring functions in linear space follows directly from their definition. The functions max, eq, and sel are evidently computable by a TM operating within constant space, as their operations involve simple comparisons and selections.

Statement 5 corresponds to a reformulation of Lemma 5 by Merrill and Sabharwal (2023a). According to this lemma, the function sum can be computed by a uniform family of TC<sup>0</sup> circuits with polynomial size. Thus, the summation operation can be effectively executed within this computational framework.

We now look at Statement 6, first considering integer division and extending this to floating-points. It is well-established, that for computation of integer division, a TM operating in linear space can perform this operation. One approach involves left-shifting the second operand by the maximum number of bits, denoted as k, such that it does not exceed the value of the first operand. By subsequently adding  $2^k$  to the result and subtracting the bit-shifted second operand from the first operand, the division operation can be iteratively carried out. Extending this algorithm to floating-point numbers is straightforward, as it primarily involves subtracting the exponents. Finally, the component-wise extension to  $\mathbb{F}_p^k$  is simple.

In the following, we establish the existence of a TM that, given an input of  $1^n$ , can compute a circuit  $C_n$  in logarithmic space. This circuit, denoted as  $C_n$ , effectively simulates the operation of a strong average-hard head.

**Theorem 1.** The function computed by an averagehard attention head, as defined in Definition 4, can be effectively computed using a uniform family of  $TC^0$  circuits of polynomial size.

*Proof.* A schematic representation of the circuit structure for input size n, illustrating how it computes the attention vector for the i-th input position from n vectors  $x_1, \ldots, x_n \in \mathbb{F}_p^k$ , is presented in

Fig. 3. The circuit structure closely adheres to the specifications outlined in Definition 4.

![](_page_5_Figure_9.jpeg)

Figure 3: Schematic representation showcasing the structure of a transformer layer (as in Fig. 2) and one of its average-hard attention heads simulated by a circuit.

From Lemma 2, it is evident that all the constituent elements employed in constructing the circuit depicted in Fig. 3 are uniform families of  $TC^0$  circuits of polynomial size. By comparing the various circuit levels with the specifications outlined in Definition 4 and utilizing Eq. (3) for the topmost level, it becomes apparent that the circuit accurately computes  $Att_{\sigma}(x_1 \cdots x_n)_i$ . Moreover, due to the constant depth and polynomial size characteristics of each individual building block, the overall circuit also possesses these required properties.

To complete our argument, we need to establish that the circuit depicted in Fig. 3 can be constructed in logarithmic space by a TM that takes  $1^n$  as input. Given that each of the sub-circuits can be constructed in logarithmic space (as stated in Lemma 2), our main focus is to demonstrate that the interconnection of the individual sub-circuits, as depicted by the edges in Fig. 3, can also be computed within logarithmic space. Specifically, we aim to show that a fixed number of loops, utilizing loop variables that range between 1 and n, are sufficient to generate both the sub-circuits and the edges connecting them. Since the former is self-evident, we will now focus our attention on the latter aspect.

To construct the structure presented in Fig. 3 for

 $<sup>^{1}</sup>$ It is worth noting that each of the edges shown in Fig. 3 represents a bundle of p edges, thereby necessitating an additional internal loop variable to generate each of them.

each  $i \in [n]$ , it is necessary to maintain a variable that tracks the index i.

To generate the 'scores' level and its input edges, an additional loop variable (also ranging from 1 to n) is required to keep track of the index  $j \in [n]$  of the sub-circuit being added to the overall circuit, responsible for implementing  $\sigma$ . For each j, edges are added from both the j-th input gate and the i-th input gate. The same approach is employed for the 'max' and 'select' levels.

In the 'maximum' level, only one loop variable j is needed to establish edges from each of the n scoring sub-circuits to the single max circuit. A similar process is followed for the two summation sub-circuits at the 'summation' level. Lastly, there are only two edges that connect to the sub-circuit at the 'divide' level.

By incorporating the construction presented in the proof of Theorem 1 into the construction of a constant-depth uniform threshold circuit described by Merrill and Sabharwal (2023a), we achieve a complete simulation of an average-hard attention transformer. Due to the similarity with the proof provided by Merrill and Sabharwal (2023a), we outline the proof for brevity.

**Theorem 2.** Every language that can be decided by a transformer with average-hard attention is in uniform  $TC^0$ .

Proof sketch. Let  $\Sigma = a_1, \ldots, a_m$  be our alphabet, and let  $\omega = a_{i_1}, \ldots, a_{i_n}$  be our input string, where  $i_1, \ldots, i_n \in [m]$ . Layer 1 of the transformer receives a positional encoding  $\operatorname{enc}(\omega) = \operatorname{enc}(a_{i_1,1}), \ldots, \operatorname{enc}(a_{i_n},n) \in \mathbb{F}_p^k$  as input  $X_1$ . Two examples of positional encodings are binary encoding as  $\operatorname{enc}(a_i,j) = (i,j,0,\ldots,0)$  and one-hot encoding as  $\operatorname{enc}(a_i,j) = (i,2^j,0,\ldots,0)$ .

For each positional encoding enc (assuming it is log-precision), there exists a  $TC^0$  circuit family that takes the input w (in some binary representation) and produces the output enc(w). The existence of such circuits is straightforward for the examples given above, as a logspace-TM can create a circuit that copies the n input symbols to the output and appends the remaining components  $j, 0, \ldots, 0$  of the vector as constant outputs. A counter for  $j, 0, \ldots, 0$  is sufficient for this purpose.

The proof proceeds by induction on the number of layers. Since each layer transforms inputs in  $(\mathbb{F}_p^k)^n$  to outputs in  $(\mathbb{F}_p^k)^n$  by precondition, the in-

duction is trivial. The main point is to show that using Theorem 1, a single layer can be simulated by a log-space-uniform TC<sup>0</sup> circuit family.

It is crucial to note that all components of a log-precision layer of an average-hard attention transformer are identical to those by Merrill and Sabharwal (2023a), except for the ones related to average-hard attention. Merrill and Sabharwal (2023a) demonstrate that each of these components can be simulated by a uniform TC<sup>0</sup> circuit family and can be combined uniformly into one circuit for the entire layer.

By replacing the sub-circuit used for softmax attention in Merrill and Sabharwal (2023a) construction with the circuit from construction Theorem 1, we can obtain circuits for the average-hard attention layer.

#### **4** Conclusions and Future Directions

In conclusion, this paper has shown that logprecision transformers can simulate average-hard attention transformers. This has significant implications for both theoretical analysis and practical applications of transformer models.

Moving forward, there are several promising avenues for future research in this area. Firstly, an in-depth investigation comparing the expressive power of average-hard and softmax attention transformers would provide valuable insights into the underlying mechanisms of these models. Understanding whether they possess the same level of expressive capacity or if average-hard attention transformers are strictly less powerful (and to what extent) would shed light on the computational capabilities of transformers.

Furthermore, exploring the implications of these findings for practical applications is crucial. If average-hard attention transformers are found to be equivalent to log-precision transformers, it would provide a more efficient and simplified approach for implementing transformers. On the other hand, if there are fundamental differences between the two models, it would be important to understand the impact of these differences on the performance and generalization capabilities of transformer-based systems.

Addressing the challenge of establishing a comprehensive and concise definition of a transformer that can effectively accommodate various models

is crucial for future research in this field. In the specific context of the compared models in this study, the discrepancies in fundamental definitions posed significant challenges when comparing the models. This issue extends beyond the scope of this particular paper and is a prevalent obstacle when comparing transformer models in theoretical research. Therefore, it would be essential to establish a standardized definition that is accessible and convenient for researchers in the field of formal languages to utilize.

#### Acknowledgements

The author would like to acknowledge the valuable feedback provided by Frank Drewes throughout, helpful comments by Gail Weiss, as well as the early-stage discussions with William Merrill, which contributed to the development of this paper.

#### References

- Sanjeev Arora and Boaz Barak. 2009. *Computational Complexity: A Modern Approach*. Cambridge University Press.
- Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. 2023. Transformers learn shortcuts to automata. In *Proc. ICLR*.
- William Merrill and Ashish Sabharwal. 2023a. The parallelism tradeoff: Limitations of log-precision transformers. *Trans. ACL*, 11:531–545.
- William Merrill and Ashish Sabharwal. 2023b. The parallelism tradeoff: Limitations of log-precision transformers.
- William Merrill, Ashish Sabharwal, and Noah A. Smith. 2022. Saturated transformers are constant-depth threshold circuits. *Transactions of the Association for Computational Linguistics*, 10:843–856.
- Hao Peng, Roy Schwartz, Sam Thomson, and Noah A. Smith. 2018. Rational recurrences.
- Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems, NeurIPS.
- Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. On the practical computational power of finite precision RNNs for language recognition. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 740–745, Melbourne, Australia. Association for Computational Linguistics.