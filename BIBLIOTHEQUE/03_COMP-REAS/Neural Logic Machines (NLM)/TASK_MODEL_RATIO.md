1. **Number of distinct tasks evaluated:** 13

   "Table 4 shows hyper-parameters used by NLM for different tasks." (Appendix B.2, Table 4)

   "HasFather", "HasSister", "IsGrandparent", "IsUncle", "IsMGUncle", "AdajacentToRed", "4-Connectivity", "6	ext{-Connectivity}", "1-OutDegree", "2-OutDegree", "Sorting", "Path", "BlocksWorld". (Appendix B.2, Table 4)

2. **Number of trained model instances required to cover all tasks:** 13

   "For  $\partial$ ILP, we take the grounded probability of the \"target\" predicate as the output; for an NLM with D layers, we take the corresponding group of output predicates at the last layer (for property prediction, we use tensor  $O_D^{(1)}$  to represent unary predicates, while for relation prediction we use tensor  $O_D^{(2)}$  to represent binary predicates) and classify the property or relation with a linear layer." (Section 3.2 Family tree reasoning)

   "Table 4 shows hyper-parameters used by NLM for different tasks." (Appendix B.2, Table 4)

   "In supervised learning tasks, a model is called \"graduated\" if its training loss is below a threshold depending on the task (usually 1e-6)." (Appendix B.2)

   Single jointly trained model covering all tasks: Not specified in the paper.

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{13\ \text{tasks}}{13\ \text{models}} = 1
}
$$
