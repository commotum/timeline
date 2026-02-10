1. **Number of distinct tasks evaluated:** 4

> "Consider the standard task of next-token prediction, which consists of two phases at test time:" (Section 2 Method)
>
> "Specifically, we evaluate all the 3B models fine-tuned at 128K context length, on the three NIAH tasks in RULER [42]." (Section 3.5 Needle in a Haystack)
>
> "|                      | S-NIAH-1             |      |      |      | S-NIAH-2             |      |      |      | S-NIAH-3           |      |      |      |      |      |      |"
>
> "|                      | (pass-key retrieval) |      |      |      | (number in haystack) |      |      |      | (UUID in haystack) |      |      |      |      |      |      |" (Table 2, Section 3.5 Needle in a Haystack)

2. **Number of trained model instances required to cover all tasks:** 1

> "The motivation for our method, as discussed in Section 1, was to use longer context to achieve better performance in language modeling without having to recall every detail."
>
> "Specifically, we evaluate all the 3B models fine-tuned at 128K context length, on the three NIAH tasks in RULER [42]." (Section 3.5 Needle in a Haystack)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$
