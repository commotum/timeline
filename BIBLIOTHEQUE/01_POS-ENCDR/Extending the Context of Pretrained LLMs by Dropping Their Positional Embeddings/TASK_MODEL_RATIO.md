> "**Needle-in-a-haystack.** We evaluate long-context retrieval using the *needle-in-a-haystack* (NIAH) setup, which places a short "needle" inside a long distractor "haystack." Following prior work (Kamradt, 2023), our haystack is a random excerpt from Paul Graham's essays, and each needle is a seven-digit "magic number" paired with a short key/descriptor. We study three variants:
>
> - (Standard NIAH) We insert a single needle and prompt the model to retrieve it.
> - Multi-Query NIAH: We insert multiple (key, value) pairs and prompt the model to return as many values as possible for a given list of keys. For example: The special magic numbers for whispering-workhorse and elite-butterfly mentioned in the provided text are:
> - (Multi-Key NIAH) We insert multiple (key, value) pairs but query for a single key, e.g., The special magic number for elite-butterfly mentioned in the provided text is:
> - (Multi-Value NIAH) We associate multiple values with one key and ask for all of them without pointing to specific positions, e.g., What are all the special magic numbers for cloistered-colonization mentioned in the provided text?"
> (Section C.2. Evaluation)

> "Table 2 | DroPE outperforms RoPE-scaling methods on long context-tasks. We evaluate Smollm DroPE and the base Smollm model, extended with different RoPE scaling methods, on four long context language modeling tasks from Bai et al. (2023) and needle-in-a-haystack.
>
> | Method            | MultiFieldQA | MuSiQue | GovReport | LCC   | NIAH  | Avg.  |"
> (Section 5.1. Large-scale empirical evaluation)

> "Language modeling benchmarks. We evaluate SMOLLM and SMOLLM-DROPE on six standard multiple-choice benchmarks using the LIGHTEVAL harness (Habib et al., 2023): ARC-E/C: grade-school science QA split into Easy and Challenge sets, the latter defined by questions that defeat simple IR and co-occurrence baselines (Clark et al., 2018); HellaSwag: adversarially filtered commonsense sentence completion that is easy for humans but challenging for LMs (Zellers et al., 2019); Open-BookQA: combining a small "open book" of science facts with broad commonsense to answer 6K
>
> questions (Mihaylov et al., 2018); **PIQA:** two-choice physical commonsense reasoning (Bisk et al., 2020); and **WinoGrande:** a large-scale, adversarial Winograd-style coreference/commonsense benchmark (Sakaguchi et al., 2021). We follow the harness defaults for prompt formatting, decoding, and scoring, and do not perform any task-specific fine-tuning or data adaptation."
> (Section C.2. Evaluation)

> "Table 5 | **DroPE matches base model in-context performance.** Comparison of the pretrained SMOLLM-360M and SMOLLM-1.7B models with SMOLLM-360M-DROPE and SMOLLM-1.7B-DROPE respectively. Modes are evaluated on a variety of LM benchmarks across question answering and reasoning tasks.
>
> | Model                            | ARC-E                 | ARC-C               | HellaSwag             | OpenBookQA            | PIQA               | Winogrande            | Avg.                  |"
> (Section C.2. Evaluation)

Number of distinct tasks evaluated: 14 (Standard NIAH, Multi-Query NIAH, Multi-Key NIAH, Multi-Value NIAH, MultiFieldQA, MuSiQue, GovReport, LCC, ARC-E, ARC-C, HellaSwag, OpenBookQA, PIQA, WinoGrande). (Sections 5.1. Large-scale empirical evaluation; C.2. Evaluation)

Number of trained model instances required to cover all tasks: 1 (no task-specific fine-tuning or data adaptation is used for the benchmark tasks). (Section C.2. Evaluation)

$$
\boxed{
\frac{14\ \text{tasks}}{1\ \text{model}} = 14
}
$$
