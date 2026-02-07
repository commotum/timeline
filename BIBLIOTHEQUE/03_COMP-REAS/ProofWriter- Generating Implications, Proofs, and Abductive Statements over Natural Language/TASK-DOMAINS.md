# ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language (Not specified in the paper)
Source: ProofWriter- Generating Implications, Proofs, and Abductive Statements over Natural Language.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proof (inc. QA) | Theory C (facts + rules in English) and question Q (hypothesis fact) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Answer A (True/False/Unknown) and proof P (English facts/rules) | 0D (inferred); 1D (t) (inferred) | Capped (inferred) |
| Implication enumeration | Theory C (facts + rules in English) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | Implications I_i (English facts) | 1D (t) (inferred) | Capped (inferred) |
| Abduction (single fact) | Theory C (facts + rules in English) and unprovable fact Q | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | Missing fact(s) f_m (English) that make Q true or None | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper defines three reasoning tasks over natural language theories: proof/QA, implication enumeration, and single-fact abduction. Inputs are English facts and rules (plus a hypothesis question for proof/QA and abduction), and outputs are answer labels, proofs, implications, or missing facts expressed in English. The tasks operate over text sequences with capped sizes implied by the 512-token input limit and dataset-bounded proof depths or implication/missing-fact counts, with static attention and constructed state for iterative reasoning while abduction is described as a direct mapping.

## Evidence
### Task: Proof (inc. QA)
- "C be a theory, a set of English sentences C consisting of facts F and rules R" (Section 3.1 Definitions)
- "Q be a question, a hypothesis fact in English whose truth is to be determined based solely on the information in C." (Section 3.1 Definitions)
- "Given C and hypothesis fact Q, what is the truth A and proof P (if any) of Q?" (Section 3.1, Table 1)
- "A be an answer, where A in {True, False} ... or A in {True, False, Unknown}" (Section 3.1 Definitions)
- "Each node in P is either a fact f (a ground literal) or a rule r (a logical implication), expressed in English." (Section 3.3)
- "the input to the model is of the form: \"$question$ = question; $context$ = theory-sentences\"" (Section 3.5)
- "iteratively generating 1-hop inferences and their (simple) proofs, adding implications back into the context for deeper reasoning" (Introduction)
- "there are a few cases where the iterative model reaches the default 512 token limit of T5 when adding implications to the theory." (Appendix D)
- "Each dataset contains questions whose answers require reasoning up to depths D (D = 0, 1, 2, 3, 5)." (Section 4)
- Inference: Dimensions labeled 1D (t) and 0D because inputs/outputs are English sentences and answer labels; dynamics capped from the 512-token input limit and bounded proof depths; attention static from fixed question+context input format; state constructed from iterative addition of implications. (supporting quotes above)

### Task: Implication enumeration
- "enumeration: C -> I1, ..., In: Which Ii follow from C?" (Section 3.1, Table 1)
- "I be an implication, a fact that logically follows from C." (Section 3.1 Definitions)
- "Facts and rules are English statements, and implications are English statements that logically follow from those facts and rules." (Section 3.1)
- "Each train/test example is of then of the form: given C, predict all the I_i." (Section 3.7)
- "The number of implications can be as high as 21." (Appendix D.3)
- "there are a few cases where the iterative model reaches the default 512 token limit of T5 when adding implications to the theory." (Appendix D)
- "iteratively generating 1-hop inferences and their (simple) proofs, adding implications back into the context for deeper reasoning" (Introduction)
- Inference: 1D (t) dimension from English statements; capped dynamics from the 512-token limit and maximum implication counts; static attention from fixed-context input; constructed state from iterative implication generation. (supporting quotes above)

### Task: Abduction (single fact)
- "abduction (restricted form): CQ -> f_m : Which extra fact f_m will make Q true given C?" (Section 3.1, Table 1)
- "Given a theory C and a possible implication Q not provable from C, identify a new fact f_m such that C ∪ {f_m} implies Q." (Section 3.8)
- "Each abduction question can have zero or more missing facts as answer" (Appendix A.1)
- "D0-Ab                | 18011 | 85705 | 0/0.8/15     | 6" (Table 11)
- "Facts and rules are English statements, and implications are English statements that logically follow from those facts and rules." (Section 3.1)
- "there are a few cases where the iterative model reaches the default 512 token limit of T5 when adding implications to the theory." (Appendix D)
- Inference: 1D (t) dimension from English statements; capped dynamics from the 512-token limit and max missing-fact counts (Table 11); static attention from fixed-context input; state direct because the task is specified as a single mapping from C and Q to missing facts. (supporting quotes above)
