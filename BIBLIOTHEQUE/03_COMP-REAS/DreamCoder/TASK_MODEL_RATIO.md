1. **Number of distinct tasks evaluated:** 8
   - "Fig. 1 shows examples of program induction tasks in eight different domains that DreamCoder is applied to (Fig. 1A), along with an in-depth illustration of one task in the classic list-processing domain: learning a program that sorts lists of numbers (Fig. 1B), given a handful of input-output examples." (Introduction)
   - "We describe applications to eight domains (Fig. 1A): classic program synthesis challenges, more creative visual drawing and building problems, and finally, library learning that captures the basic languages of recursive programming, vector algebra, and physics." (Introduction)

2. **Number of trained model instances required to cover all tasks:** 8
   - "DreamCoder addresses both of these bottlenecks by learning to compactly represent and efficiently induce programs in a given domain." (Introduction)
   - "The system learns to learn – to write better programs, and to search for them more efficiently – by jointly growing two distinct kinds of domain expertise: (1) explicit declarative knowledge, in the form of a learned domain-specific language, capturing conceptual abstractions common across tasks in a domain; and (2) implicit procedural knowledge, in the form of a neural network that guides how to use the learned language to solve new tasks, embodied by a learned domain-specific search strategy." (Introduction)
   - "Could this approach be extended to learn not just one domain at a time, but to simultaneously develop expertise across many different classes of problems, starting from only a single minimal basis?" (Discussion, "What to build in, and how to learn the rest")

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$
