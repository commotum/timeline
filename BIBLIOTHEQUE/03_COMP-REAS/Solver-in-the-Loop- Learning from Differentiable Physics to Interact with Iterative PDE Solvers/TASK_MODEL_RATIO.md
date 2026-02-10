1. **Number of distinct tasks evaluated:** 5

   - "For each of the five scenarios, we implement the non-interacting evaluation (NON) by pre-computing a large-scale data set..." (Section 3.1)
   - "Wake Flow", "Buoyancy", "Advdiff.", "*CG Solver", "3D Wake" (Table 1, Section 5)

2. **Number of trained model instances required to cover all tasks:** 5

   - "In total, we target four scenarios: pure non-linear advection-diffusion (Burger's equation), two-dimensional Navier-Stokes flow, Navier-Stokes coupled with a second advection-diffusion equation for a buoyancy-driven flow, and a 3D Navier-Stokes case. Also, we discuss CG solvers in the context of differentiable operators below." (Section 3.1)
   - "This scenario requires significantly larger models to learn a correction function..." (Section 4, Three-dimensional Fluid Flow)

3. **Task–Model Ratio = (1) / (2)**

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$
