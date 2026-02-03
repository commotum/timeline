Number of distinct tasks evaluated: 3.
"This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation." (Section 4 EXPERIMENTS)

Number of trained model instances required to cover all tasks: 3.
"For image understanding, we train ViT-B from scratch with AdamW, learning rate  $5 \times 10^{-4}$  and a 5-epoch warmup from  $1 \times 10^{-6}$ , batch size 256, and 300 training epochs." (Section 4.1 EXPERIMENTAL SETUPS)
"For class-conditional image generation, we use DiT-B/2 with a constant learning rate  $1 \times 10^{-4}$ , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." (Section 4.1 EXPERIMENTAL SETUPS)
"For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate  $2 \times 10^{-5}$ , weight decay 0.01, and batch size 64." (Section 4.1 EXPERIMENTAL SETUPS)

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$
