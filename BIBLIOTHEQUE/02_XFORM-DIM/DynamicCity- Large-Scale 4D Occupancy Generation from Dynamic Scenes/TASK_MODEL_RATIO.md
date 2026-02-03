Number of distinct tasks evaluated: 10.
"Reconstruction. To evaluate the effectiveness of the proposed VAE in encoding the 4D occupancy sequence," (Section 5.2)
"Generation. To demonstrate the effectiveness of DynamicCity in 4D scene generation," (Section 5.2)
"HexPlane: By autoregressively generating the HexPlane, we extend scene duration beyond temporal constraints." (Section 4.3)
"Layout: We control vehicle placement and dynamics in 4D scenes using conditions learned from bird's-eye view sketches." (Section 4.3)
"Command: Controls general ego vehicle motion via instructions." (Section 4.3)
"Trajectory: Enables fine-grained control through specific trajectory inputs." (Section 4.3)
"Inpaint: Edit 4D scenes by masking HexPlane regions and guiding sampling with the masked areas." (Section 4.3)
"Outpainting extends the spatial dimensions of a given occupancy sequence." (Section 7.5)
"Single frame occupancy. We apply the same procedure for single-frame occupancy conditional generation as for HexPlane conditional generation." (Section 7.5)
"We train our HexPlane conditional generation pipeline on Occ3D-nuScenes [40] as an occupancy forecasting model." (Section 8.2)

Number of trained model instances required to cover all tasks: 4.
"mainly consists of a VAE for 4D occupancy encoding using HexPlane [7, 11] (Sec. 4.1), and a DiT for HexPlane generation (Sec. 4.2)." (Section 4)
"We demonstrate that our model can handle versatile applications by training a conditional DiT for the previous tasks." (Section 7.5)
"fine-tune our HexPlane generation model for single-frame conditional generation." (Section 7.5)
"We train our HexPlane conditional generation pipeline on Occ3D-nuScenes [40] as an occupancy forecasting model." (Section 8.2)

$$
\boxed{
\frac{10\ \text{tasks}}{4\ \text{models}} = 2.5
}
$$
