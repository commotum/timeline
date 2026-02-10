1. **Number of distinct tasks evaluated:** 3  
   “The three classes are (1) conjunctive normal form expressions with a bounded number of literals in each clause, (2) monotone disjunctive normal form expressions, and (3) arbitrary expressions in which each variable occurs just once.” (Section 1, “INTRODUCTION”)

2. **Number of trained model instances required to cover all tasks:** 3  
   “THEOREM A: For any positive integer k, the class of k-CNF expressions is learnable via an algorithm A that uses  $L = L(h, (2t)^{k+1})$  calls of EXAMPLES and no calls of ORA-CLE, where t is the number of variables.” (Section 5, “BOUNDED CNF EXPRESSIONS”)  
   “THEOREM B: The class of monotone DNF expressions is learnable via an algorithm B that uses L = L(h,d) calls of EXAMPLES and dt calls of ORACLE, where d is the degree of the DNF expression f to be learned and t the number of variables.” (Section 6, “DNF EXPRESSIONS”)  
   “THEOREM C: The class of  $\mu$ -expressions in learnable via a deduction procedure C that uses  $0(t^3)$  calls of N-ORACLE RP and RA altogether, where t is the number of variables (and no calls of EXAMPLES). The procedure always deduces exactly the correct expression.” (Section 7, “µ-EXPRESSIONS”)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
