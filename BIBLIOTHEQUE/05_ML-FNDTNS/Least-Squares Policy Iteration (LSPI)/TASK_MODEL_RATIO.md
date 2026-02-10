1. **Number of distinct tasks evaluated:** 3

   "LSPI was implemented<sup>9</sup> using a combination of MATLAB and C and was tested on the following problems: chain walk, inverted pendulum balancing, and bicycle balancing and riding." (Section 9, *Experimental Results*)

2. **Number of trained model instances required to cover all tasks:** 3

   "LSPI was applied on the same problem using the same basis functions repeated for each of the two actions so that each action gets its own parameters" (Section 9.1, *Chain Walk*).

   "We applied LSPI with a set of 10 basis functions for each of the 3 actions, thus a total of 30 basis functions, to approximate the value function." (Section 9.2, *Inverted Pendulum*)

   "This block of basis functions is repeated for each of the 5 actions, giving a total of 100 basis functions (and parameters)." (Section 9.3, *Bicycle Balancing and Riding*)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
