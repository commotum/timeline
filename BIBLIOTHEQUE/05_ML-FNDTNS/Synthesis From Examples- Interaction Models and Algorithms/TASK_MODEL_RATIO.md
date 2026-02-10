1. **Number of distinct tasks evaluated:** 9
   "We briefly discuss applications of inductive synthesizers to not only synthesis of a variety of traditional programs such as bitvector algorithms (§IV-A) and spreadsheet macros (§IV-B) but also to synthesis of more general structured concepts or *artifacts* such as geometric constructions (§IV-C), algebraic identities (§IV-D), sequences (§IV-E), and even drawings (§IV-F)." (§I. INTRODUCTION)
   "The DSL for *Syntactic string transformations* [26] includes substring and concatenate operators along with limited forms of regular expressions, conditionals, and loops. *Semantic string transformations* [27] combine syntactic transformations with lookup operations from other relational tables (containing required background knowledge). *Number transformations* [28] allow for formatting and rounding transformations on numbers. *Table transformations* [29] allow for layout transformations on tables." (§IV.B. Spreadsheet Macros)
   "The inductive synthesizers described in [26]–[29] can automate the tasks in (a), (b), (c)/(d), and (e) respectively." (Figure 2, §IV.B)

2. **Number of trained model instances required to cover all tasks:** 9 models
   "[14] describes a constraint solving based (§III-C) inductive synthesizer for such bitvector programs." (§IV.A. Bitvector Algorithms)
   "For each of these languages, we have developed a version-space algebra based inductive synthesizer (§III-A) that can generate scripts for automating repetitive tasks from input-output examples." (§IV.B. Spreadsheet Macros)
   "The underlying synthesis algorithm performs brute-force search (over an extended library of ruler/compass operators) using goal-directed heuristics (§III-B)." (§IV.C. Geometry Constructions)
   "Figure 4 shows some algebraic proof problems that have automatically synthesized starting from a given example problem [19]." (§IV.D. Algebraic Identities)
   "Our tool [34] can predict the bold parts in each of the above three texts from the remaining prefixes." (Figure 5, §IV.E. Mathematical Terms)
   "Predicting the repetitive objects in a drawing from few examples of initial objects can be phrased as a synthesis-from-example problem [36]." (§IV.F. Repetitive Drawings)
   "Not specified in the paper." (whether one jointly trained/shared-weight multi-task model covers all tasks)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{9\ \text{tasks}}{9\ \text{models}} = 1
}
$$
