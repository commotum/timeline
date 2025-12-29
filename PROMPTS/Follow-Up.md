### Clarifying “Multi-Task” vs. Architecture Reuse (Capability-Focused)

The term **“multi-task”** is often overloaded. In many papers, it refers to evaluating a single **architecture** across multiple tasks or datasets by training or fine-tuning **separate task-specific models**. In this case, the architecture is reused, but **task capability is fragmented across multiple trained model instances**.

This differs fundamentally from a stronger notion of multi-task learning, where **a single trained model instance** simultaneously supports **multiple distinct tasks** using shared weights, a unified representation, and a common inference interface (e.g., conditioning via prompts, task tokens, or a unified objective), **without requiring task-specific fine-tuning**.


### Capability Question (do not count ablations or variants):

**For the purposes of task capability (not ablations, architectural variants, backbones, or hyperparameter sweeps):**
If a practitioner wanted a system that can perform **all distinct tasks evaluated in this paper**, how many **separately trained model instances** would be required?

- Do **not** count ablations, architectural variants, backbones, or experimental configurations as separate models **unless** they are required to perform different tasks.
- If a task requires training or adding a **task-specific head, decoder, output module, or extension** (even when built on a shared or frozen backbone), it **counts as a separate trained model instance** for the purposes of this analysis.

### Please report (with citations):

1. **Number of distinct tasks evaluated**
2. **Number of trained model instances required to cover all tasks**
3. **Task–Model Ratio = (1) / (2)**


### Output format (required)

Report the **Task–Model Ratio** using the following **display-math box** format.

**Examples:**

**Pretrain once → fine-tune 8 times (8 tasks, 8 models):**

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$

**Jointly train one model on 8 tasks (single multi-task model):**

$$
\boxed{
\frac{8\ \text{tasks}}{1\ \text{model}} = 8
}
$$

**Task-agnostic model (GPT-2–like; task via conditioning):**

$$
\boxed{
\frac{N\ \text{tasks}}{1\ \text{model}} = N
}
$$

Use this exact boxed equation format for all reported results.
