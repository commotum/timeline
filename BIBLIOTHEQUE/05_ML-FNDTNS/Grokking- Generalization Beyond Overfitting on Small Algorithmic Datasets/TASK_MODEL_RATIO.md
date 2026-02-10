1. **Number of distinct tasks evaluated:** 11

> "The following are the binary operations that we have tried (for a prime number p = 97):" (Section A.1.1, "BINARY OPERATIONS")
>
> "x \circ y = x + y \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x - y \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x/y \pmod{p} \text{ for } 0 \leq x < p, 0 < y < p
> x \circ y = [x/y \pmod{p} \text{ if } y \text{ is odd, otherwise } x - y \pmod{p}] \text{ for } 0 \leq x, y < p
> x \circ y = x^2 + y^2 \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x^2 + xy + y^2 \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x^2 + xy + y^2 + x \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x^3 + xy \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x^3 + xy^2 + y \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x^3 + xy^2 + y \pmod{p} \text{ for } 0 \leq x, y < p
> x \circ y = x \cdot y \text{ for } x, y \in S_5
> x \circ y = x \cdot y \cdot x \text{ for } x, y \in S_5" (Section A.1.1, "BINARY OPERATIONS")

2. **Number of trained model instances required to cover all tasks:** 11

> "For each binary operation we constructed a dataset of equations..." (Section A.1.1, "BINARY OPERATIONS")
>
> "For each training run, we chose a fraction of all available equations at random and declared them to be the training set..." (Section A.1.1, "BINARY OPERATIONS")
>
> "We've measured the mean accuracy across three runs for training datasets consisting of different fractions of all available equations for a variety of binary operations listed in Appendix A.1.1." (Section 3.2, "Grokking on a variety of problems")

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
