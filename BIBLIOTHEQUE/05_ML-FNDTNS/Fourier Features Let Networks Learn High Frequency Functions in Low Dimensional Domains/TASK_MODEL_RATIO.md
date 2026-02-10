1. **Number of distinct tasks evaluated:** 5

"2D image regression." (Section 6.2 Tasks)

"3D shape regression." (Section 6.2 Tasks)

"2D computed tomography (CT)." (Section 6.2 Tasks)

"3D magnetic resonance imaging (MRI)." (Section 6.2 Tasks)

"3D inverse rendering for view synthesis." (Section 6.2 Tasks)

2. **Number of trained model instances required to cover all tasks:** 5

"For each target signal, we train an MLP on a training subset of the signal and compute error over the remaining test subset." (Section 6.2 Tasks)

"All tasks (except 3D shape regression) use L2 loss and a ReLU MLP with 4 layers and 256 channels. The 3D shape regression task uses cross-entropy loss and a ReLU MLP with 8 layers and 256 channels." (Section 6.2 Tasks)

"For the Indirect supervision tasks, the network outputs are passed through a forward model before the loss is applied (integral projection for CT, the Fourier transform for MRI, and nonlinear volume rendering for NeRF)." (Table 1 caption, Section 6.2)

3. **Task–Model Ratio:**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
