1. **Number of distinct tasks evaluated:** 11

Citation (Section 5.7, Table 11):

> "| 1    | t1               | [-0.2, 0.2] | 256           | training & test correctly pred. | see text  |"
>
> "| 2a   | $^{\mathrm{t1}}$ | [-0.2, 0.2] | no test set   | after 5 million exemplars       | ABS(0.25) |"
>
> "| 2b   | t2               | [-0.2, 0.2] | 10000         | after 5 million exemplars       | ABS(0.25) |"
>
> "| 2c   | t2               | [-0.2, 0.2] | 10000         | after 5 million exemplars       | ABS(0.2)  |"
>
> "| 3a   | t3               | [-0.1, 0.1] | 2560          | ST1 and ST2 (see text)          | ABS(0.2)  |"
>
> "| 3 b  | t3               | [-0.1, 0.1] | 2560          | ST1 and ST2 (see text)          | ABS(0.2)  |"
>
> "| 3c   | t3               | [-0.1, 0.1] | 2560          | ST1 and ST2 (see text)          | see text  |"
>
> "| 4    | t3               | [-0.1, 0.1] | 2560          | ST3(0.01)                       | ABS(0.04) |"
>
> "| 5    | t3               | [-0.1, 0.1] | 2560          | see text                        | ABS(0.04) |"
>
> "| 6a   | t3               | [-0.1, 0.1] | 2560          | ST3(0.1)                        | ABS(0.3)  |"
>
> "| 6b   | t3               | [-0.1, 0.1] | 2560          | ST3(0.1)                        | ABS(0.3)  |"

2. **Number of trained model instances required to cover all tasks:** 11

Explicit statement of one jointly trained model for all tasks: Not specified in the paper.

Citation (Section 5.7, Table 10):

> "Table 10: Summary of experimental conditions for LSTM, Part I. 1st column: task number."
>
> "8th column: number of weights w."

Citation (Section 5.2, Task 2c):

> "The net has p+4 input units and 2 output units."

Citation (Section 5.4, Architecture):

> "Architecture. We use a 3-layer net with 2 input units, 1 output unit, and 2 cell blocks of size 2."

Citation (Section 5.6, Architecture):

> "Architecture. We use a 3-layer net with 8 input units, 2 (3) cell blocks of size 2 and 4 (8) output units for Task 6a (6b)."

3. **Task-Model Ratio**

$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$
