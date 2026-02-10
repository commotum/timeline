1. **Number of distinct tasks evaluated:** 2  
   - "We demonstrate the effectiveness of this approach by evaluating it on two datasets." (Section 1 Introduction)  
   - "As explained above, the model was trained to classify digits into ten classes." (Section 4.1 Jittered MNIST)  
   - "There were 40 possible output phoneme classes." (Section 4.2 Monophone Speech)

2. **Number of trained model instances required to cover all tasks:** 2  
   - "We trained and tested our model on MNIST with random uniform translations of  $\pm 4$  pixels, resulting in grayscale images of size  $36 \times 36$ ." (Section 4.1 Jittered MNIST)  
   - "In addition, we ran our model on a dataset of monophone speech samples." (Section 4.2 Monophone Speech)  
   - "We trained a model with 4 experts at the first layer and 16 at the second layer." (Section 4.2 Monophone Speech)

3. **Task–Model Ratio = (1) / (2):**

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$
