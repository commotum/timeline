# Are We on the Right Way for Evaluating Large Vision-Language Models?

 $\begin{array}{cccccccccccccccccccccccccccccccccccc$ 

#### **Abstract**

Large vision-language models (LVLMs) have recently achieved rapid progress, sparking numerous studies to evaluate their multi-modal capabilities. However, we dig into current evaluation works and identify two primary issues: 1) Visual **content is unnecessary for many samples.** The answers can be directly inferred from the questions and options, or the world knowledge embedded in LLMs. This phenomenon is prevalent across current benchmarks. For instance, GeminiPro achieves 42.9% on the MMMU benchmark without any visual input, and outperforms the random choice baseline across six benchmarks over 24% on average. 2) Unintentional data leakage exists in LLM and LVLM training. LLM and LVLM could still answer some visual-necessary questions without visual content, indicating the memorizing of these samples within large-scale training data. For example, Sphinx-X-MoE gets 43.6% on MMMU without accessing images, surpassing its LLM backbone with 17.9%. Both problems lead to misjudgments of actual multi-modal gains and potentially misguide the study of LVLM. To this end, we present MMStar, an elite vision-indispensable multi-modal benchmark comprising 1,500 samples meticulously selected by humans. MMStar benchmarks 6 core capabilities and 18 detailed axes, aiming to evaluate LVLMs' multi-modal capacities with carefully balanced and purified samples. These samples are first roughly selected from current benchmarks with an automated pipeline, human review is then involved to ensure each curated sample exhibits visual dependency, minimal data leakage, and requires advanced multi-modal capabilities. Moreover, two metrics are developed to measure data leakage and actual performance gain in multi-modal training. We evaluate 16 leading LVLMs on MMStar to assess their multi-modal capabilities, and on 7 benchmarks with the proposed metrics to investigate their data leakage and actual multi-modal gain.

#### 1 Introduction

Encouraged by the rapid development of large language models (LLMs) [47, 4, 8, 9, 13, 1, 43], integrating visual modality into LLMs to enhance models' interactivity capabilities has witnessed ever-changing advances in recent days [54, 26, 24, 11, 52, 2, 48, 31, 5, 12]. These large vision-language models (LVLMs) showcase powerful visual perception and understanding capabilities, enabling them to accept image inputs from users and engage in dialogues, thereby offering a more enriched interactive experience. These achievements have further inspired the research community

<sup>\*</sup>Equal contribution. This work is done during internship in Shanghai AI Laboratory.

<sup>&</sup>lt;sup>†</sup>Correspoding author.

![](_page_1_Figure_0.jpeg)

Figure 1: We highlight cases in existing multi-modal benchmarks where evaluation samples either **lack visual dependency** or **have unintentionally leaked into the training data of LLMs and LVLMs**. (a) Some samples can be answered by LLMs using only text-based world knowledge; (b) For some instances, the question itself contains the answer, making images superfluous; (c) Some samples are leaked into LLMs' training corpora can be "recalled" with the textual questions and answers directly; (d) Some samples indiscernible to LLMs but solved by LVLMs without accessing images suggest leakage into LVLMs' multi-modal training data.

to develop a variety of multi-modal benchmarks [21, 14, 27, 38, 50, 51, 29, 20, 30], constructed to explore the powerful capabilities emerging from LVLMs and provide a comprehensive and objective platform for quantitatively comparing the continually evolving models. Despite the race among existing evaluation works to construct as many axes as possible to assess the capabilities of LVLMs, we have identified two primary issues upon delving into existing evaluation samples and processes.

First, **visual content is unnecessary for many samples**. A qualified multi-modal evaluation sample should compel LVLMs to understand and reason with the visual content for correct answers. Otherwise, the evaluation sample would degrade into assessing the textual capabilities of LLM bases. Unfortunately, we have identified numerous samples across multiple popular benchmarks [27, 21, 51, 30, 20] where answers can be correctly deduced without relying on visual content. As shown in Figure 1 (a) and (b), some samples have answers directly included within the questions (e.g., What is the shape of the round dirt circle?), while others can be effortlessly answered by leveraging the rich world knowledge embedded within the LLM bases (e.g., What is the capital of Nebraska?). With a comprehensive quantitative analysis of 25 LLMs on 6 benchmarks, we observe this phenomenon is prevalent and serious. For example, more than 50% questions of ScienceQA and 20% questions of MMMU can be solved by most LLMs directly. For the powerful close source LLM GeminiPro, it achieves 42.9% on the MMMU benchmark without any visual input, and outperforms the random choice baseline across six benchmarks by over 24% on average.

Taking aside the inappropriate samples in evaluation, we also observed strange results that LLM and LVLM could still answer some visual-necessary questions without visual content (Figure 1 (c) and (d)). A plausible explanation for this could be the inadvertent memorization of these samples during the large-scale training process, suggesting the presence of **unintentional data leakage in the training of LLM and LVLM**. Through a detailed study of 22 LVLMs on the 6 benchmarks, we find the unexpected leaking problem during the LVLM training is particularly serious. For example, we find Yi-VL-34B gets 15.0% higher performance than its LLM backbone on ScienceQA, Sphinx-X-MoE gets 43.6% on MMMU *without* accessing images, surpassing its LLM backbone with 17.9%, even surpassing many leading LVLMs with accessing images.

The existence of inappropriate questions and data leaking would lead to misjudgments of actual multi-modal performance gains and potentially misguide the study of LVLM. In pursuit of a more accurate and comprehensive evaluation, we introduce the MMStar Benchmark. MMStar is a premier, vision-critical multi-modal benchmark that includes 1,500 challenging samples, each rigorously validated by humans. It is structured to test 6 fundamental capabilities and 18 specific di-

mensions, aiming to evaluate the multi-modal capacities of LVLMs with a carefully balanced and purified selection of samples.

The MMStar is a new benchmark that "Stands on the shoulders of giants". Samples are first roughly selected from current benchmarks with an automated pipeline. In detail, we use eight powerful LLMs as candidates inspectors for visual dependency and LLM leakage, including two closed-source APIs (GPT4-Turbo [34], and GeminiPro [41]) and six leading open-source models (e.g., LLaMA-70B [43], Qwen-1.5-72B [1], and Mixtral-8x7B [19]). Samples that could be answered by more than 2 of the 8 LLMs are excluded as they may exist leaking or visual-unnecessary problems. Then we use 16 leading LVLMs (e.g., GPT4V [35], GeminiPro-Vision [41], LLaVA series [24, 26]) to gauge the difficulty of the samples and split them to four levels. Ultimately, based on the difficulty of the rough-filtered samples, **strict manual review and selection** are applied to curate 1,500 high-quality multimodal evaluation samples. As shown in Figure 5, these samples span 6 core multimodal capabilities with a purified and high-quality set of samples. Moreover, we design the multi-modal gain (MG) and multi-modal leakage (ML) metrics to probe LVLMs' actual performance gain and data leakage degrees derived from multi-modal training in a benchmark-specific manner.

We evaluate the accuracy, MG, and ML of 16 leading LVLMs on our MMStar benchmark, the high-resolution version of GPT-4V ranks first with 57.1% accuracy, showcasing its superb multi-modal capability. GPT-4V also gets the best MG and a small ML, indicating its effective multi-modal training strategy and has less data leaking.

In a nutshell, our contributions are threefold:

- We delve into existing evaluation benchmarks and processes and identify two key issues: (1) Visual content is unnecessary for many samples. (2) Unintentional data leakage exists in LLM and LVLM training. Both lead to misjudgment of LVLM capability and may misguide the following study.
- We curate MMStar, an elite vision-indispensable multi-modal benchmark comprising 1,500 challenge samples meticulously selected by humans. MMStar covers samples from diverse tasks and difficulties, aiming to evaluate the actual multi-modal capacities of LVLMs.
- Based on MMStar, we evaluate LVLMs with Accuracy and two newly proposed metrics: multi-modal gain and multi-modal leakage. The high-resolution version of GPT-4V outperforms the 16 leading LLMs and ranks first.

## 2 Related Work

Large Vision-Language Models. As large language models (LLMs) [8, 43, 43, 47, 42, 34, 36, 9] rapidly advance, a growing fraction of the research community is focusing on integrating visual content into LLMs to build a powerful intelligent assistant with more interactive ways. Central to these large vision-language models (LVLMs) are the seminal works in modality alignment within the vision-language learning area [37, 17]. The foundation work CLIP [37] exemplifies the alignment of vision and language modalities through contrastive learning on extensive image-text pairs. Built upon the CLIP image encoder which is somewhat aligned with the language modality, current LVLMs typically utilize vast image-text pairs to connect the vision encoder and LLM, enabling LLM to receive and understand visual content [54, 26, 24, 11, 52, 2, 48, 31, 5]. For example, MiniGPT4 [54] and LLaVA [26] directly connect the vision encoder and LLM with QFormer [22] and MLP [40], showing proficiency in multi-modal dialogues. Subsequent works have further enhanced LVLMs by improving the multi-modal instruction data [24, 48, 5, 44] and designing novel modules [2, 23, 45, 28, 15, 12] for more sufficient modality alignment.

**Evaluations of LVLMs.** To probe the true capabilities of the emerging LVLMs, the research community has developed many multi-modal benchmarks encompassing a wide range of evaluation axes [27, 14, 38, 51, 39, 21, 26, 50, 46]. Early single-task benchmarks, such as VQA [16], MS-COCO [39], and OK-VQA [38], fail to holistically assess LVLMs' general multi-modal perception and reasoning capabilities. To address this issue, comprehensive multi-modal benchmarks have been constructed [26, 21, 51, 14, 27, 7, 46]. For example, SEED [21] and MMBench [27] cover 12 and 20 evaluation dimensions respectively, while MMMU [51] spans 30 college-level subjects, providing some competitive arenas for a comprehensive comparison of cutting-edge LVLMs. However, existing evaluations of LVLMs overlook some critical issues. On the one hand, they do not guar-

Table 1: **Evaluation of various LLMs on six popular multi-modal benchmarks.** We employ a 0-shot inference strategy for evaluating all LLMs. We report the results of 2 closed-source LLMs and 20 open-source LLMs with varying sizes and architectures. The evaluated benchmarks include MMMU (MMMU-Val [51]), MMB (MMBench-EN-Dev [27]), ScienceQA (ScienceQA-Test [30]), AI2D (AI2D-Test [20]), SEED (SEED-Image [21]), and MathVista (MathVista-Mini [29]). The **best** results are highlighted in **bold and underlined.** 

| Model              | Strategy | MMMU        | MMB         | ScienceQA         | AI2D        | SEED        | MathVista   | Avg. |  |  |  |  |
|--------------------|----------|-------------|-------------|-------------------|-------------|-------------|-------------|------|--|--|--|--|
| Baselines          |          |             |             |                   |             |             |             |      |  |  |  |  |
| Random Choice      | -        | 22.1        | 0.0         | 24.2              | 23.8        | 24.3        | 17.9        | 18.7 |  |  |  |  |
| Closed-source LLMs |          |             |             |                   |             |             |             |      |  |  |  |  |
| GPT4-Turbo[34]     | 0-shot   | 41.2        | 12.2        | 64.3              | 59.7        | 10.1        | 24.2        | 35.3 |  |  |  |  |
| GeminiPro[41]      | 0-shot   | <u>42.9</u> | <u>18.4</u> | <u>68.9</u>       | 59.2        | <u>35.5</u> | 23.3        | 41.4 |  |  |  |  |
|                    |          | (           | Open-soui   | rce LLMs          |             |             |             |      |  |  |  |  |
| Qwen1.5-1.8B[1]    | 0-shot   | 29.0        | 10.0        | 54.3              | 37.9        | 28.9        | 20.4        | 30.1 |  |  |  |  |
| Phi2-2.7B[32]      | 0-shot   | 20.0        | 7.2         | 47.1              | 38.7        | 26.4        | 22.0        | 26.9 |  |  |  |  |
| Yi-6B[49]          | 0-shot   | 25.7        | 9.5         | 58.1              | 39.1        | 27.4        | 21.2        | 30.2 |  |  |  |  |
| LLaMA2-7B[43]      | 0-shot   | 23.6        | 11.5        | 56.8              | 43.5        | 31.7        | 24.1        | 31.9 |  |  |  |  |
| Qwen-7B[1]         | 0-shot   | 19.8        | 8.4         | 52.7              | 42.6        | 7.6         | 20.5        | 25.3 |  |  |  |  |
| Deepseek-7B[3]     | 0-shot   | 21.6        | 8.4         | 56.3              | 38.1        | 13.4        | 20.6        | 26.4 |  |  |  |  |
| InternLM2-7B[42]   | 0-shot   | 32.8        | 8.9         | 64.0              | 48.3        | 31.9        | 18.9        | 34.1 |  |  |  |  |
| Qwen1.5-7B[1]      | 0-shot   | 25.0        | 11.4        | 62.3              | 49.4        | 19.4        | 19.9        | 31.2 |  |  |  |  |
| Vicuna-v1.5-7B[8]  | 0-shot   | 29.9        | 10.3        | 58.9              | 42.5        | 32.6        | 22.0        | 32.7 |  |  |  |  |
| Baichuan2-7B[47]   | 0-shot   | 25.7        | 10.5        | 52.7              | 44.0        | 29.2        | 20.8        | 30.5 |  |  |  |  |
| Mistral-7B[18]     | 0-shot   | 30.0        | 13.2        | 63.4              | 48.5        | 34.3        | 22.6        | 35.3 |  |  |  |  |
| LLaMA2-13B[43]     | 0-shot   | 24.4        | 10.1        | 59.1              | 45.0        | 33.6        | 23.8        | 32.7 |  |  |  |  |
| Vicuna-v1.5-13B[8] | 0-shot   | 28.3        | 11.6        | 59.5              | 45.0        | 26.3        | 19.6        | 31.7 |  |  |  |  |
| Baichuan2-13B[47]  | 0-shot   | 22.1        | 4.7         | 51.1              | 32.8        | 25.4        | 20.3        | 26.1 |  |  |  |  |
| InternLM2-20B[42]  | 0-shot   | 32.2        | 15.9        | 63.8              | 55.7        | 26.0        | 21.3        | 35.8 |  |  |  |  |
| Yi-34B[49]         | 0-shot   | 37.1        | 10.5        | 53.6              | 57.3        | 37.3        | 21.7        | 36.3 |  |  |  |  |
| Mixtral-8x7B[19]   | 0-shot   | 25.7        | 8.6         | 57.2              | 48.7        | 13.5        | 23.4        | 29.5 |  |  |  |  |
| Deepseek-67B[3]    | 0-shot   | 30.9        | 14.8        | 64.3              | <u>57.5</u> | 17.1        | 23.2        | 34.6 |  |  |  |  |
| LLaMA2-70B[43]     | 0-shot   | 28.9        | 12.3        | $\overline{62.2}$ | 48.6        | 34.3        | <u>25.2</u> | 35.3 |  |  |  |  |
| Qwen1.5-72B[1]     | 0-shot   | 21.4        | 10.1        | 57.5              | 44.2        | 8.8         | 19.5        | 26.9 |  |  |  |  |

antee that all evaluation samples can not be correctly answered without the visual content. On the other hand, current evaluations consistently adhere to the process of inferring on given benchmarks and calculating scores for LVLMs, overlooking the possibility of data leakage during multi-modal training. This oversight can lead to unfair comparisons and misjudgments of the real gains in multi-modal capabilities brought by multi-modal training.

#### 3 Two Overlooked Issues for Evaluating LVLMs

In this section, we delve into two commonly overlooked issues in current LVLM evaluation works. Moreover, we present detailed experimental results to further substantiate our observations.

First issue: visual content is unnecessary for many evaluation samples. The key distinction between evaluating LLMs and LVLMs lies in the necessity for LVLM evaluations to strictly ensure that the correct answers can only be derived based on a thorough understanding of visual content. Without this, evaluating LVLMs' multi-modal capabilities degrades to merely assessing their LLM backbones' uni-modal abilities. However, upon examining samples from some popular LVLM benchmarks, we find many samples lack vital visual dependency and can yield correct answers even without the image inputs! Through analysis of these failure samples, we categorize them into two groups: (1) Answers can be directly obtained from the world knowledge embedded in LLMs, owing to the LLMs' extensive pertaining on the large corpus of data. For example, as illustrated in Figure 1(a), the question "What is the capital of Nebraska?" already provides the key information "Nebraska", eliminating the need for extracting relevant location information from visual content. A more appropriate question is "What is the capital of the highlighted area in the image?" to emphasize

Table 2: **Evaluation of various LLMs on six popular multi-modal benchmarks under 2-shot.** We employ a 2-shot inference strategy for evaluating all LLMs to reduce instances of refusal to answer and align the answer formats. We report the results of 2 closed-source LLMs and 20 open-source LLMs with varying sizes and architectures. The evaluated benchmarks include MMMU (MMMU-Val [51]), MMB (MMBench-EN-Dev [27]), ScienceQA (ScienceQA-Test [30]), AI2D (AI2D-Test [20]), SEED (SEED-Image [21]), and MathVista (MathVista-Mini [29]). The **best** results are highlighted in **bold and underlined.** 

| Model              | Strategy         | MMMU        | MMB               | ScienceQA   | AI2D | SEED        | MathVista | Avg.        |  |  |  |  |  |
|--------------------|------------------|-------------|-------------------|-------------|------|-------------|-----------|-------------|--|--|--|--|--|
| Baseline           |                  |             |                   |             |      |             |           |             |  |  |  |  |  |
| Random Choice      | -                | 22.1        | 2.1 0.0 24.2 23.8 |             |      |             | 17.9      | 18.7        |  |  |  |  |  |
| Closed-source LLMs |                  |             |                   |             |      |             |           |             |  |  |  |  |  |
| GPT4-Turbo[34]     | 2-shot           | 42.0        | 15.5              | 67.5        | 61.3 | 26.8        | 25.6      | 39.8        |  |  |  |  |  |
| GeminiPro[41]      | 2-shot           | <u>42.7</u> | <u>18.7</u>       | <u>69.3</u> | 60.1 | <u>38.1</u> | 25.5      | <u>42.4</u> |  |  |  |  |  |
|                    | Open-source LLMs |             |                   |             |      |             |           |             |  |  |  |  |  |
| Qwen1.5-1.8B[1]    | 2-shot           | 33.0        | 8.6               | 55.6        | 41.3 | 32.1        | 22.7      | 32.2        |  |  |  |  |  |
| Phi2-2.7B[32]      | 2-shot           | 19.9        | 4.3               | 50.8        | 41.7 | 6.9         | 18.4      | 23.7        |  |  |  |  |  |
| Yi-6B[49]          | 2-shot           | 32.9        | 16.0              | 64.6        | 51.5 | 36.7        | 24.5      | 37.7        |  |  |  |  |  |
| LLaMA2-7B[43]      | 2-shot           | 25.9        | 7.7               | 57.9        | 42.8 | 32.8        | 22.8      | 31.7        |  |  |  |  |  |
| Qwen-7B[1]         | 2-shot           | 30.6        | 15.0              | 63.0        | 50.0 | 32.6        | 21.0      | 35.4        |  |  |  |  |  |
| Deepseek-7B[3]     | 2-shot           | 28.7        | 11.6              | 61.9        | 46.0 | 34.1        | 21.7      | 34.0        |  |  |  |  |  |
| InternLM2-7B[42]   | 2-shot           | 33.6        | 11.4              | 63.6        | 52.1 | 34.4        | 20.4      | 35.9        |  |  |  |  |  |
| Qwen1.5-7B[1]      | 2-shot           | 33.3        | 13.1              | 65.1        | 52.1 | 32.1        | 22.8      | 36.4        |  |  |  |  |  |
| Vicuna-v1.5-7B[8]  | 2-shot           | 31.3        | 9.5               | 58.9        | 45.5 | 32.0        | 20.7      | 33.0        |  |  |  |  |  |
| Baichuan2-7B[47]   | 2-shot           | 28.2        | 13.7              | 58.1        | 44.1 | 32.3        | 21.7      | 33.0        |  |  |  |  |  |
| Mistral-7B[18]     | 2-shot           | 29.8        | 17.2              | 66.1        | 50.0 | 34.4        | 13.4      | 35.2        |  |  |  |  |  |
| LLaMA2-13B[43]     | 2-shot           | 32.9        | 10.1              | 58.9        | 43.8 | 32.1        | 24.8      | 33.8        |  |  |  |  |  |
| Vicuna-v1.5-13B[8] | 2-shot           | 31.3        | 12.8              | 63.0        | 46.8 | 33.6        | 20.8      | 34.7        |  |  |  |  |  |
| Baichuan2-13B[47]  | 2-shot           | 32.2        | 13.1              | 61.0        | 47.1 | 35.2        | 23.4      | 35.3        |  |  |  |  |  |
| InternLM2-20B[42]  | 2-shot           | 35.6        | 17.4              | 66.4        | 55.9 | 30.4        | 20.8      | 37.8        |  |  |  |  |  |
| Yi-34B[49]         | 2-shot           | 35.8        | 15.8              | 67.9        | 59.6 | 37.2        | 26.9      | 40.5        |  |  |  |  |  |
| Mixtral-8x7B[19]   | 2-shot           | 35.1        | 17.3              | 66.3        | 55.1 | 35.8        | 22.7      | 38.7        |  |  |  |  |  |
| Deepseek-67B[3]    | 2-shot           | 38.3        | 17.2              | 68.3        | 59.7 | 37.3        | 23.4      | 40.7        |  |  |  |  |  |
| LLaMA2-70B[43]     | 2-shot           | 30.4        | 17.2              | 63.4        | 49.3 | 34.9        | 24.2      | 36.6        |  |  |  |  |  |
| Qwen1.5-72B[1]     | 2-shot           | 42.4        | 21.1              | <b>70.1</b> | 60.9 | 40.7        | 26.3      | 43.6        |  |  |  |  |  |

the importance of visual understanding. (2) Answers are directly included in the textual questions. As shown in Figure 1(b), LLMs can derive the correct answer "circle" through simple reasoning based on the question "What is the shape of the round dirt circle?".

To quantitatively substantiate our findings, we further experiment to gauge the proportion of these two types of samples in existing benchmarks. Specifically, we evaluate several benchmarks with two closed-source LLMs (GPT4-Turbo [34], and GeminiPro [41]) and six opensource heavy LLMs (InternLM2-20B [42], Yi-34B [49], Mixtral-8x7B [19], Deepseek-67B [3], LLaMA2-70B [43], and Qwen1.5-72B [1]), recording the hit count for each question. Here, the 'hit' refers to the ability of an LLM to correctly answer the question without relying on vi-

![](_page_4_Figure_4.jpeg)

Figure 2: LLM hit rate across various benchmarks.

sual input. We then calculate the percentage of samples with a hit count of six or more (representing 80%) against the total number of samples to determine the abnormal hit rate for each benchmark. As depicted in Figure 2, every benchmark shows a certain degree of samples that visual contents are unnecessary, with ScienceQA [30] and AI2D [20] exhibiting amazing abnormal hit rates of 57.2% and 46.2%, respectively. Based on our observations, most multi-modal benchmarks have yet to fully assess the multi-modal capabilities of LVLMs.

Table 3: **Evaluation of various LVLMs on six popular multi-modal benchmarks.** For the "strategy" column, "LLM" refers to evaluating using the corresponding LLM base of the LVLM, while "LVLM-text" denotes evaluating LVLMs without accessing images. We employ the 0-shot inference strategy for LLMs to align the evaluation protocols of LVLMs. We only report the results of 2 closed-source LVLMs and 8 open-source LVLMs due to space limits. For the entire LVLMs' results, please refer to the appendix. The **highest** results of the LVLM-text setting across the models are highlighted in **bold and underlined.** 

| Model                                     | Param.                                          | Strategy                 | MMMU                 | MMB                        | ScienceQA                   | AI2D                        | SEED                        | MathVista                   | Avg.                 |  |  |  |  |  |
|-------------------------------------------|-------------------------------------------------|--------------------------|----------------------|----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|----------------------|--|--|--|--|--|
|                                           | Baseline                                        |                          |                      |                            |                             |                             |                             |                             |                      |  |  |  |  |  |
| Random Choice                             | -                                               | -                        | 22.1                 | 0.0                        | 24.2                        | 23.8                        | 24.3                        | 17.9                        | 18.7                 |  |  |  |  |  |
|                                           | Closed-source LVLMs and corresponding LLM bases |                          |                      |                            |                             |                             |                             |                             |                      |  |  |  |  |  |
| GPT4V[35]<br>(GPT4-Turbo[34])             | -<br>-<br>-                                     | LLM<br>LVLM-text<br>LVLM | 41.2<br>45.1<br>53.6 | 12.2<br>17.6<br>69.6       | 64.3<br>68.2<br>81.4        | 59.7<br><b>62.5</b><br>75.3 | 10.1<br><b>28.4</b><br>71.6 | 24.2<br><b>25.4</b><br>44.7 | 35.3<br>41.2<br>66.0 |  |  |  |  |  |
| GeminiPro-Vision[41]<br>(GeminiPro[41])   |                                                 | LLM<br>LVLM-text<br>LVLM | 42.9<br>39.4<br>44.4 | 18.4<br>16.7<br>68.1       | 68.9<br>66.3<br>80.6        | 59.2<br>54.5<br>68.0        | 35.5<br>27.9<br>64.3        | 23.3<br>24.5<br>36.0        | 41.4<br>38.2<br>60.2 |  |  |  |  |  |
|                                           |                                                 | Open-source              | LVLMs and            | correspo                   | nding LLM bas               | ies                         |                             |                             |                      |  |  |  |  |  |
| TinyLLaVA[53]<br>(Phi2-2.7B[32])          | 3В                                              | LLM<br>LVLM-text<br>LVLM | 20.0<br>30.0<br>36.0 | 7.2<br>21.0<br>66.9        | 47.1<br>62.3<br>69.1        | 38.7<br>51.9<br>62.4        | 26.4<br>37.2<br>70.1        | 22.0<br>23.5<br>28.9        | 26.9<br>37.7<br>55.6 |  |  |  |  |  |
| LLaVA-1.5[24]<br>(Vicuna-v1.5-7B[8])      | 7В                                              | LLM<br>LVLM-text<br>LVLM | 29.9<br>29.9<br>34.4 | 10.3<br>19.5<br>65.0       | 58.9<br>64.1<br>68.7        | 42.5<br>48.7<br>55.6        | 32.6<br>37.5<br>65.6        | 22.0<br>20.3<br>23.6        | 32.7<br>36.7<br>52.2 |  |  |  |  |  |
| InternLM2-XC2[12]<br>(InternLM2-7B[42])   | 7B                                              | LLM<br>LVLM-text<br>LVLM | 32.8<br>34.2<br>41.7 | 8.9<br><b>26.2</b><br>79.6 | 64.0<br><b>71.9</b><br>96.7 | 48.3<br>63.3<br>81.4        | 31.9<br>38.1<br>74.9        | 18.9<br><b>29.4</b><br>57.4 | 34.1<br>43.9<br>72.0 |  |  |  |  |  |
| Monkey-Chat[23]<br>(Qwen-7B[1])           | 10B                                             | LLM<br>LVLM-text<br>LVLM | 19.8<br>32.4<br>37.1 | 8.4<br>15.6<br>71.0        | 52.7<br>71.1<br>82.4        | 42.6<br>56.8<br>68.5        | 7.6<br>36.1<br>69.1         | 20.5<br>25.0<br>34.0        | 25.3<br>39.5<br>60.4 |  |  |  |  |  |
| CogVLM-Chat[45]<br>(Vicuna-v1.5-7B[8])    | 17B                                             | LLM<br>LVLM-text<br>LVLM | 29.9<br>30.1<br>34.2 | 10.3<br>15.5<br>63.4       | 58.9<br>54.6<br>66.3        | 42.5<br>52.5<br>63.3        | 32.6<br>36.7<br>68.7        | 22.0<br>25.0<br>34.7        | 32.7<br>35.7<br>55.1 |  |  |  |  |  |
| Yi-VL[49]<br>(Yi-34B[49])                 | 34B                                             | LLM<br>LVLM-text<br>LVLM | 37.1<br>37.3<br>43.2 | 10.5<br>23.2<br>71.5       | 53.6<br>68.6<br>75.3        | 57.3<br>59.9<br>65.9        | 37.3<br><b>41.0</b><br>68.1 | 21.7<br>22.7<br>25.6        | 36.3<br>42.1<br>58.3 |  |  |  |  |  |
| InternVL-Chat-v1.2[6]<br>(NH2-Yi-34B[33]) | 40B                                             | LLM<br>LVLM-text<br>LVLM | 37.6<br>41.7<br>49.1 | 20.1<br>23.9<br>82.4       | 69.4<br>70.3<br>82.5        | 60.2<br>65.0<br>78.5        | 35.0<br>40.5<br>75.4        | 17.9<br>24.0<br>47.7        | 40.0<br>44.2<br>69.3 |  |  |  |  |  |
| Sphinx-X-MoE[15]<br>(Mixtral-8x7B[19])    | 57B                                             | LLM<br>LVLM-text<br>LVLM | 25.7<br>43.6<br>44.8 | 8.6<br>20.5<br>69.2        | 57.2<br>68.4<br>72.2        | 48.7<br>61.1<br>65.0        | 13.5<br>39.9<br>71.1        | 23.4<br>28.4<br>38.1        | 29.5<br>43.7<br>60.1 |  |  |  |  |  |

Second issue: unintentional data leaking exists in LLM and LVLM training. Although the community has the trend towards developing new multi-modal benchmarks to assess LVLMs' capabilities from various dimensions, there is scant consideration for fairness and reliability during evaluation. Training LLMs and LVLMs requires vast and diverse data, inevitably leading to the leakage of evaluation samples. Such incidents are usually unintended, as it's impractical to predict which data will be used in future evaluation benchmarks during the preparation for training corpus.

Figure 1 (c) showcases an evaluation sample leaked by LLMs. Though the question requires an understanding of image content, 16 out of 22 tested LLMs astonishingly provide the correct response by "recalling" their training data. To quantitatively support our observations, we evaluate 22 leading LLMs across 6 popular benchmarks and report the 0-shot results in Table 1 and 2-shot results in Table 2. Specifically, we find the 2-shot evaluation strategy is more stable than the 0-shot to reduce refusal for answering and align answer formats. Under the impact of vision-independent samples and data leakage from LLMs, GeminiPro [41] and Qwen1.5-72B [1] achieve a remarkable average accuracy of 41.4% and 43.6% under the 2-shot setting, outperforming random choice by 20.4% and 22.6%, respectively. Furthermore, Qwen1.5-72B achieves a score of 42.4% on MMMU [51], even surpassing the performance of the majority of LVLMs with accessing images. This result serves as a reminder: if we only consider the final accuracy on benchmarks when evaluating LVLMs, potential data leakage from LLMs could lead to unfair comparisons.

![](_page_6_Figure_0.jpeg)

Figure 3: Illustration of data leakage during LVLMs' multi-modal training processes. We showcase samples that LLMs cannot answer correctly but LVLMs without accessing images (LVLM-text) can. Each LLM-LVLM $^{text}$  pair represents an LLM and its corresponding LVLM without accessing images, totaling 16 pairs. The chart in the center tallies the number of samples in existing benchmarks hit by more than half of the LLM-LVLM $^{text}$  pairs, underscoring the issue of data leakage during the multi-modal training process.

In Figure 1 (d) and Figure 3, we showcase some examples where original LLMs fail, but LVLMs without accessing images succeed. Despite these questions requiring image content for accurate answers, the LVLMs without accessing images are capable of correctly answering these questions which stump original LLMs. To further support our hypotheses of data leakage during LVLMs' multi-modal training, we conduct an intriguing experiment. We remove the image inputs for LVLMs and only utilize textual questions and options for evaluation, with the results reported in Table 3. We compare the performance gains of LVLMs set to receive only text inputs (LVLM-text) against their corresponding LLM bases (LLM) to quantitatively assess the degree of data leakage in LVLMs' multi-modal training. As shown in Table 3, most LVLMs exhibit varying degrees of data leakage during multi-modal training. For example, the LLMs of Sphinx-X-8x7B [15] and Monkey-Chat [23], show a respective average performance gain of 14.1% and 14.2% compared to their original LLMs.

Drawing from our observations, we posit that the issue of data leakage in multi-modal datasets is a significant concern that warrants attention. For research to be transparent and equitable, it is imperative to account for and mitigate such leakage. This will ensure that the performance of models is evaluated on their true ability to integrate and interpret multi-modal data, rather than their capacity to memorize specific samples. A proper benchmark would be a crucial step toward advancing the field of multi-modal language model research.

## 4 MMStar

With the aforementioned analysis, we present an elite vision-dependent multi-modal benchmark, dubbed as **MMStar**. In Section 4.1, we elaborate on the data curation process of MMStar. Section

![](_page_7_Figure_0.jpeg)

Figure 4: **Statics of the data sources during the data curation process.** After applying the coarse filter process and manual review, we narrow down from a total of 22,401 samples to 11,607 candidate samples and finally select 1,500 high-quality samples to construct our MMStar benchmark.

4.2 provides a detailed analysis of the constructed MMStar benchmark. In Section 4.3, we introduce two benchmark-specific metrics developed to evaluate the degree of data leakage as well as the actual performance gains in multimodal capabilities from the multi-modal training.

#### 4.1 Data Curation Process

Criteria for data curation. The evaluation samples for constructing the MMStar benchmark should meet three fundamental criteria: 1) Visual dependency. The collected samples can be correctly answered only based on understanding the visual content; 2) Minimal data leakage. The collected samples should minimize the risk of unintentional inclusion in LLMs' training corpus, or be effectively transformed from uni-modal to multi-modal formats to prevent LLMs from "recalling" the correct answers; 3) Requiring advanced multi-modal capabilities for resolution. In addition to ensuring fairness and reliability by adhering to the above criteria, we also aim for samples to cover various difficulty levels. We expect to comprehensively capture LVLMs' multi-modal capabilities with succinct high-quality samples.

**Data filter.** We first choose two benchmarks [27, 21] focused on natural images and four centered on scientific and technical knowledge [51, 30, 20, 29] for our sample collection. We then develop an automated pipeline to preliminarily filter out samples that do not meet the first two criteria. Specifically, we employ two closed-source LLMs [41, 34] and six open-source LLMs [1, 42, 49, 3, 19, 43] sizing 20B or larger to serve as inspectors. These open-source LLMs are applied with a 2-shot in-context inference strategy to minimize response refusals and ensure consistency in answer formatting. Following this, we evaluate the sample pool with these LLM inspectors, documenting the hit frequency for each evaluation sample. Finally, we only retain those samples with hit counts of two or fewer hits, indicating that around 75% of LLM inspectors fail to provide the correct answer. As illustrated in Figure 4, following this initial coarse filtering, our sample pool was reduced from 22,401 to 11,607.

**Manual review.** After the coarse filtering with LLM inspectors, we further employ three experts to conduct the manual review process to ensure: 1) each sample's answer should be based on the understanding of visual content; 2) selected samples should cover a comprehensive range of capability assessment dimensions; 3) most samples should require LVLMs to possess advanced multi-modal abilities for resolution. To expedite the manual selection of samples with varying difficulty levels for LVLMs, we tally the hit counts of all 16 LVLMs on the coarsely filtered samples and split them into four difficulty categories: easy (12-16), moderate (8-11), hard (4-7), and tough (0-3). Finally, after considering both the diversity of capability dimensions and difficulty levels, we manually curated **1,500** high-quality samples from the coarsely filtered set. Figure 4 showcases the detailed composition of data sources for our final selection of samples.

#### 4.2 Core Capabilities

We select and consolidate the dimensions used for assessing LVLMs' multi-modal capabilities in existing benchmarks and identify six core capability dimensions along with eighteen detailed axes.

![](_page_8_Figure_0.jpeg)

Figure 5: **Distribution of capability dimensions on the MMStar benchmark.** In MMStar, we display 6 core capabilities in the inner ring, with 18 detailed axes presented in the outer ring. The middle ring showcases the number of samples for each detailed dimension. Each core capability contains a meticulously balanced 250 samples. We further ensure a relatively even distribution across the 18 detailed axes.

In Figure 5, we provide statistics for each core capability and their detailed axes on the MMStar benchmark.

**Coarse Perception (CP).** This core dimension refers to the capability to understand and interpret the overarching characteristics and themes of an image without delving into the finer details. It encompasses a broad, holistic view of the visual content, enabling the identification of: 1) image style & quality; 2) image scene & topic; and 3) image emotion.

**Fine-grained Perception (FP).** This core dimension represents a sophisticated level of image understanding that focuses on the detailed and nuanced aspects of visual content. It involves a deep dive into the specifics of images: 1) attribute & celebrity recognition; 2) object location; and 3) object counting. This core dimension unveils the subtle intricacies that coarse perception might overlook.

**Instance Reasoning (IR).** It encapsulates a set of advanced cognitive capabilities focused on understanding and interpreting individual and collective object attributes and interrelations within an image. This process goes beyond mere recognition, delving into the analytical assessment of: 1) single-instance attribute reasoning; 2) cross-instance attribute comparison; and 3) cross-instance relation reasoning. It is a critical component for systems requiring a deep semantic understanding of visual content, enabling nuanced interaction with and response to complex visual content.

**Logical Reasoning (LR).** This core dimension encompasses a sophisticated framework of cognitive processes designed to interpret, deduce, and infer conclusions from visual content through a structured approach to logic and reasoning. This multi-faceted capability marries the intuitive understanding of visual content with the structured rigor of logical deduction, enabling: 1) diagram reasoning; 2) code & sequence reasoning; and 3) common reasoning.

**Science & Technology (ST).** It consists of a comprehensive framework for the application and integration of knowledge across a broad spectrum of science and technology. This domain combines the theoretical underpinnings and practical applications of various fields: 1) natural science; 2) engineering; and 3) geography & earth science.

**Mathematics (MA).** Math is a foundational pillar of logical and analytical reasoning and encompasses a broad spectrum of capabilities essential for understanding, applying, and interpreting quantitative and spatial information. We primarily consider three aspects for evaluating LVLMs' logical thinking provess: 1) numeric commonsense & calculation; 2) geometry; and 3) statistical analysis.

#### 4.3 Multi-modal Gain/Leakage

Given our observation of the potential for inadvertent leakage of some evaluation samples during the multi-modal training process, the vanilla evaluation approach struggles to reveal LVLMs' actual performance gains derived from multi-modal training and fails to enable fair comparison with other competitors. Therefore, we propose two novel metrics to separately assess the degree of data leakage and actual performance gain from the multi-modal training process.

To calculate the multi-modal gain (MG) metric for a given LVLM on a particular benchmark, we need to compute the scores of the same LVLM with and without visual inputs, separately denoted as  $S_v$  and  $S_{wv}$ . Then the MG metric can be derived from the following formulation:

$$MG = S_v - S_{wv}. (1)$$

To calculate the multi-modal leakage (ML) metric, we need to compute the extra score of the given LVLM's LLM base (without any multi-modal training), denoted as  $S_t$ . Then the ML metric is formulated as follows:

$$ML = \max(0, S_{wv} - S_t). \tag{2}$$

#### 5 Experiments

In this section, we conduct a systematic analysis of the proposed MMStar benchmark along with the MG/ML metrics. We detail the experimental setup in Section 5.1, study and analyze the performance of 16 leading LVLMs on MMStar in Section 5.2, and extensively investigate the MG/ML metrics of 16 LVLMs across 6 popular benchmarks and our MMStar in Section 5.3.

## 5.1 Experimental Setups

**Evaluation models.** 1) **Baselines**: We utilize random choice and frequent choice strategies to serve as the baselines. The former randomly selects an option as the answer, while the latter selects the most frequent option within each benchmark dimension. 2) **Large Language Models**: We prepare two closed-source LLMs, GPT4 [34] and GeminiPro [41], and 20 popular open-source LLMs sizing from 1.8B to 72B for text-only evaluation, such as Qwen series [1], LLaMA2 series [43], Phi2 [32], Vicuna series [8], Deepseek series [3], InternLM2 series [42], Baichuan2 series [47], Yi series [49], Mistral series [18, 19]. Additionally, all the open-source LLMs we used are their Chat versions. and 3) **Large Vision-Language Models**: We prepare two closed-source LVLMs, GPT4V [35] and GeminiPro-Vision [41], and 14 popular open-source LVLMs sizing from 3B to 60B, such as TinyLLaVA-3B [53], Yi-VL series [49], Qwen-VL-Chat [2], LLaVA-1.5 series [24], ShareGPT4V-7B [5], Monkey-Chat [23], LLaVA-Next [25], Deepseek-VL-7B [28], LLaVA-Next-34B [25], CogVLM-Chat-17B [45], InternVL-Chat-v1.2 [6], Sphinx-X-8x7B [15].

Implementation details. For evaluating LLMs on existing benchmarks, we employ both 0-shot and 2-shot strategies and will specify which is utilized when reporting results. For evaluating LLMs on MMStar, the 0-shot strategy yields poor scores, making comparisons difficult. Therefore, we exclusively utilize the 2-shot strategy to decrease the frequency of refusal to answer. Moreover, All LVLMs are evaluated utilizing the 0-shot strategy across all benchmarks to ensure a fair comparison. When evaluating LVLMs under the 'LVLM-text' setting (*i.e.* answer without the image), most LVLMs work well by simply removing the image tokens from their default input tokens.However, GeminiPro-Vision [41] and CogVLM-Chat [45] require the replacement of the original images with pure grey images to bypass image content input and operate correctly. Given that all questions are ensured to be converted into a multiple-choice format, we develop some heuristic matching rules to calculate accuracy, avoiding the cumbersome process of re-invoking GPT4 for answer extraction. Moreover, all experiments in this study are conducted within the same codebase modified from VLMEvalKit [10], and utilize NVIDIA A100 GPUs for non-API-based evaluation.

Table 4: LLMs failed to solve problems in MMStar and performed close to random guessing, visual content is necessary to solve MMStar. We evaluate various LLMs on MMStar with the 2-shot inference strategy. We report the results of 2 closed-source LLMs and 20 open-source LLMs with varying sizes and architectures. We report the detailed results of the CP (coarse perception), FP (fine-grained perception), IR(instance reasoning), LR (logical reasoning), ST (science & technology), and MA (mathematics) core capabilities. The <a href="mailto:best">best</a> results are highlighted in **bold and underlined.** 

| Model              | СР          | FP          | IR                | LR          | ST          | MA          | Avg.        |  |  |  |  |  |
|--------------------|-------------|-------------|-------------------|-------------|-------------|-------------|-------------|--|--|--|--|--|
| Baselines          |             |             |                   |             |             |             |             |  |  |  |  |  |
| Random Choice      | 23.7        | 24.5        | 25.3              | 24.3        | 24.8        | 25.1        | 24.6        |  |  |  |  |  |
| Closed-source LLMs |             |             |                   |             |             |             |             |  |  |  |  |  |
| GPT4-Turbo[34]     | 2.4         | 4.0         | 9.6               | 18.0        | 13.6        | 25.6        | 12.2        |  |  |  |  |  |
| Gemini-Pro[41]     | <u>16.8</u> | <u>13.6</u> | <u>20.4</u>       | <u>24.4</u> | <u>19.6</u> | <u>28.8</u> | <u>20.6</u> |  |  |  |  |  |
|                    |             | Open-s      | source LLN        | As          |             |             |             |  |  |  |  |  |
| Qwen1.5-1.8B[1]    | 28.4        | 28.4        | 25.6              | 23.2        | 23.2        | 29.6        | 26.4        |  |  |  |  |  |
| Phi2-2.7B[32]      | 11.2        | 11.2        | 15.2              | 10.8        | 11.6        | 12.0        | 12.0        |  |  |  |  |  |
| Yi-6B-Chat[49]     | 23.6        | 19.2        | 28.4              | 25.2        | 12.4        | 29.6        | 23.1        |  |  |  |  |  |
| LLaMA2-7B[43]      | 28.0        | <u>30.4</u> | 26.0              | 18.0        | 18.8        | 21.6        | 23.8        |  |  |  |  |  |
| Qwen-7B[1]         | 11.6        | 5.6         | 12.8              | 5.6         | 7.2         | 0.4         | 7.2         |  |  |  |  |  |
| Deepseek-7B[3]     | 26.8        | 16.0        | 28.4              | 21.6        | 23.2        | 25.6        | 23.6        |  |  |  |  |  |
| InternLM2-7B[42]   | 22.0        | 14.8        | 22.0              | 21.6        | 15.2        | 23.2        | 19.8        |  |  |  |  |  |
| Qwen1.5-7B[1]      | 15.6        | 8.0         | 9.2               | 9.2         | 15.2        | 9.2         | 11.1        |  |  |  |  |  |
| Vicuna-v1.5-7B[8]  | 22.0        | 27.6        | 29.6              | 26.4        | 18.0        | 24.4        | 24.7        |  |  |  |  |  |
| Baichuan2-7B[47]   | 20.8        | 18.4        | 27.6              | 18.8        | 18.8        | 21.2        | 20.9        |  |  |  |  |  |
| Mistral-7B[18]     | 20.0        | 23.6        | 24.4              | 23.6        | 20.0        | 27.2        | 23.1        |  |  |  |  |  |
| LLaMA2-13B[43]     | 23.6        | 23.6        | 28.0              | 21.2        | 16.4        | 10.4        | 20.5        |  |  |  |  |  |
| Vicuna-v1.5-13B[8] | 32.8        | 24.0        | 28.8              | 17.6        | 22.0        | 14.4        | 23.3        |  |  |  |  |  |
| Baichuan2-13B[47]  | 26.4        | 18.0        | $\overline{28.0}$ | 20.4        | 21.2        | 25.6        | 23.3        |  |  |  |  |  |
| InternLM2-20B[42]  | 18.2        | 17.8        | 22.6              | 23.8        | 17.8        | 13.4        | 18.9        |  |  |  |  |  |
| Yi-34B[49]         | 20.4        | 18.0        | 24.0              | 24.0        | 14.4        | 30.8        | 21.9        |  |  |  |  |  |
| Mixtral-8x7B[19]   | 24.4        | 17.6        | 19.2              | 28.0        | 16.0        | 33.6        | 23.1        |  |  |  |  |  |
| Deepseek-67B[3]    | 29.2        | 22.4        | 18.4              | 26.0        | 20.4        | 22.4        | 23.1        |  |  |  |  |  |
| LLaMA2-70B[43]     | 22.4        | 20.0        | 19.6              | 14.4        | 7.2         | 9.6         | 15.5        |  |  |  |  |  |
| Qwen1.5-72B[1]     | 21.6        | 16.0        | 21.2              | 14.0        | 17.2        | 27.2        | 19.5        |  |  |  |  |  |

## 5.2 Results Analysis of MMStar

In this section, we present a comprehensive comparison of various LLMs and LVLMs performed on our MMStar benchmark and summarize our key observations in the following parts.

**Observation from LLMs.** We comprehensively evaluate 2 closed-source LLMs and 20 open-source LLMs of varying sizes and architectures on the MMStar benchmark and report the results in Table 4. Encouragingly, the performance of these LLMs is almost indistinguishable from random choice, effectively validating that the evaluation samples of our MMStar exhibit significant visual dependency and minimal data leakage from LLMs. Notably, the smallest model, Qwen1.5-1.8B, achieves the best score. We conjecture this is due to it suffering the least stringent safety restrictions, thereby reducing instances of refusal to answer. Moreover, among the six core capabilities of MMStar, science & technology (ST) prove to be the most challenging dimension for LLMs. The best score on ST is only 23.2%, significantly lower than the best scores of around 30% in other dimensions. We speculate this may be that samples within the ST dimension have the least degree of data leakage from LLMs' training data.

**Observation from LVLMs.** We evaluate 2 closed-source and 14 open-source LVLMs on our MM-Star, with the results reported in Table 5. As shown in the table, GPT4V[35] with a high-resolution

Table 5: **Evaluation of various LVLMs on MMStar.** We report the results of 2 closed-source LLMs and 14 open-source LLMs with varying sizes and architectures. We report the detailed results of the CP (coarse perception), FP (fine-grained perception), IR(instance reasoning), LR (logical reasoning), ST (science & technology), and MA (mathematics) core capabilities. The **best** results are highlighted in **bold and underlined.** The **worst** results of multi-modal gain (MG) and multi-modal leakage (ML) metrics are in **italic red**.

| Model                 | LLM                | Param.  | СР          | FP          | IR          | LR          | ST          | MA   | Avg.        | MG↑         | ML↓              |  |  |
|-----------------------|--------------------|---------|-------------|-------------|-------------|-------------|-------------|------|-------------|-------------|------------------|--|--|
| Baselines             |                    |         |             |             |             |             |             |      |             |             |                  |  |  |
| Random Choice         | -                  | -       | 23.7        | 24.5        | 25.3        | 24.3        | 24.8        | 25.1 | 24.6        | -           | -                |  |  |
| Closed-source LVLMs   |                    |         |             |             |             |             |             |      |             |             |                  |  |  |
| GeminiPro-Vision[41]  | GeminiPro[41]      | -       | 51.6        | 28.8        | 50.8        | 46.0        | 28.4        | 50.0 | 42.6        | 27.4        | 0.0              |  |  |
| GPT4V (low)[35]       | GPT4-Turbo[34]     | -       | 62.0        | 32.8        | 55.2        | 48.0        | 33.6        | 44.8 | 46.1        | 32.6        | 1.3              |  |  |
| GPT4V (high)[35]      | GPT4-Turbo[34]     | -       | <u>76.6</u> | <u>51.4</u> | <u>66.6</u> | <u>55.8</u> | <u>42.6</u> | 49.8 | <u>57.1</u> | <u>43.6</u> | 1.3              |  |  |
|                       | C                  | pen-sou | rce LV      | LMs         |             |             |             |      |             |             |                  |  |  |
| TinyLLaVA[53]         | Phi2-2.7B[32]      | 3B      | 60.4        | 31.6        | 50.8        | 30.4        | 18.0        | 24.8 | 36.0        | 16.4        | 7.6              |  |  |
| Yi-VL[49]             | Yi-6B[49]          | 6B      | 58.0        | 33.6        | 46.4        | 34.8        | 20.4        | 34.0 | 37.9        | 15.6        | 0.0              |  |  |
| LLaVA-1.5[24]         | Vicuna-v1.5-7B[8]  | 7B      | 58.8        | 24.0        | 38.8        | 24.0        | 13.6        | 22.8 | 30.3        | <i>10.7</i> | $\overline{0.0}$ |  |  |
| ShareGPT4V[5]         | Vicuna-v1.5-7B[8]  | 7B      | 58.8        | 28.0        | 45.6        | 24.4        | 17.2        | 24.0 | 33.0        | 11.9        | 0.0              |  |  |
| InternLM-XC2[12]      | InternLM2-7B[42]   | 7B      | 70.8        | 48.8        | 65.2        | 56.4        | 42.0        | 49.2 | 55.4        | 28.1        | 7.5              |  |  |
| Qwen-VL-Chat[2]       | Qwen-7B[1]         | 8B      | 59.6        | 32.0        | 50.8        | 29.2        | 22.0        | 31.6 | 37.5        | 23.9        | 0.0              |  |  |
| Deepseek-VL[28]       | Deepseek-7B[3]     | 8B      | 64.0        | 30.8        | 49.2        | 36.4        | 21.6        | 20.4 | 37.1        | 15.7        | $\overline{0.0}$ |  |  |
| Monkey-Chat[23]       | Qwen-7B[1]         | 10B     | 57.6        | 36.4        | 51.6        | 33.2        | 26.4        | 24.4 | 38.3        | 13.5        | <i>17.6</i>      |  |  |
| LLaVA-1.5[24]         | Vicuna-v1.5-13B[8] | 13B     | 58.8        | 28.0        | 41.6        | 24.4        | 18.4        | 25.6 | 32.8        | 13.9        | 0.0              |  |  |
| CogVLM-Chat[45]       | Vicuna-v1.5-7B[8]  | 17B     | 66.8        | 36.8        | 49.2        | 31.2        | 23.6        | 11.6 | 36.5        | 14.9        | $\overline{0.0}$ |  |  |
| Yi-VL[49]             | Yi-34B[49]         | 34B     | 53.2        | 31.2        | 52.0        | 32.4        | 12.4        | 35.2 | 36.1        | 18.8        | 0.0              |  |  |
| LLaVA-Next[25]        | NH2-Yi-34B[33]     | 34B     | 66.4        | 52.0        | 62.4        | 46.0        | 32.4        | 53.6 | 52.1        | 29.4        | 2.4              |  |  |
| InternVL-Chat-V1.2[6] | NH2-Yi-34B[33]     | 40B     | 67.6        | 43.2        | 61.2        | 47.2        | 24.0        | 19.2 | 43.7        | 32.6        | 0.0              |  |  |
| Sphinx-X-MOE[15]      | Mixtral-8x7B[19]   | 57B     | 58.4        | 40.8        | 47.6        | 35.2        | 19.2        | 32.0 | 38.9        | 14.8        | 1.0              |  |  |

setting can achieve the best average score of 57.1% among all LVLMs. Increasing the resolution and number of image tokens can boost the average score from 46.1% to 57.1% for GPT4V, offering a positive signal to the research community. Among the open-source LVLMs, InternLM-Xcomposer2 [12] achieves an impressive score of 55.4%. LLaVA-Next [25] even surpasses GPT4V and GeminiPro-Vision [41] in the mathematics (MA) core capability. Notably, no LVLMs managed to reach a passing average score (*i.e.* 60%) in the core capabilities of fine-grained perception (FP), logical reasoning (LR), science & Technology (ST), and mathematics (MA), highlighting these dimensions as particularly challenging for existing LVLMs. Moreover, TinyLLaVA [53], despite its modest 3B scale, outperformed some competitors of 7B and even 13B surprisingly, underscoring the potential of smaller-scale LVLMs. Additionally, even with the same architecture, ShareGPT4V-7B [5] even outperforms LLaVA-1.5-13B with high-quality caption data. This result highlights the significance of high-quality caption data for LVLM performance to the community.

## 5.3 Results Analysis of MG/ML

In this section, we present the results of our proposed multi-modal gain (MG) and multi-modal leakage (ML) metrics of 16 LVLMs with varying sizes and architectures on 6 popular benchmarks and our MMStar benchmark. We then detail our observations and analyses from both the model and benchmark perspectives.

Analysis from the model perspective. In Table 6, we illustrate the MG/ML (Multi-modal Gain/Multi-modal Leakage) metrics for each LVLM across each benchmark and provide an average MG/ML metric across all benchmarks in the final column. For closed-source LVLMs, GPT4V demonstrates notable performance gains attributed to its multi-modal training, while GeminiPro-Vision shows lesser data leakage during multi-modal training. This suggests that GPT4V may have utilized a broader range of multi-modal training data compared to GeminiPro-Vision. Among the open-source LVLMs, InternLM-XComposer2 achieves the highest average multi-modal gain of 28.1 across all benchmarks, whereas LLaVA-1.5-7B records the lowest at 14.8. This outcome is reason-

Table 6: Evaluation of various LVLMs on 7 Benchmarks with multi-modal gain (MG) and multi-modal leakage (ML) metrics. We report the results of 2 closed-source LLMs and 14 open-source LLMs with varying sizes and architectures. The bottom row represents the average across models for the same benchmark, while the rightmost column shows the average across benchmarks for the same LVLM. The <u>best</u> results are highlighted in **bold and underlined.** The *worst* results of MG and ML metrics are in *italic red*.

| Model                 | Param.  | MM   | MU                | M           | ИΒ                | Scien       | ceQA              | AI          | 2D                | SE   | ED         | Math | Vista             | MM          | Star              | Av       | /g.         |
|-----------------------|---------|------|-------------------|-------------|-------------------|-------------|-------------------|-------------|-------------------|------|------------|------|-------------------|-------------|-------------------|----------|-------------|
| Wiodei                | araiii. | MG↑  | ML↓               | MG↑         | ML↓               | MG↑         | ML↓               | MG↑         | ML↓               | MG↑  | ML↓        | MG↑  | ML↓               | MG↑         | ML↓               | MG↑      | ML↓         |
| Closed-source LVLMs   |         |      |                   |             |                   |             |                   |             |                   |      |            |      |                   |             |                   |          |             |
| GPT4V[35]             | -       | 8.5  | 3.9               | 52.0        | 5.4               | 13.2        | 3.9               | 12.8        | 2.8               | 43.2 | 18.3       | 19.3 | 1.2               | 32.6        | 1.3               | 25.9     | 5.3         |
| GeminiPro-Vision[41]  | -       | 5.0  | 0.0               | 51.4        | $\underline{0.0}$ | <u>14.3</u> | $\underline{0.0}$ | <u>13.5</u> | $\underline{0.0}$ | 36.4 | 0.0        | 11.5 | 1.2               | 27.4        | $\underline{0.0}$ | 22.8     | 0.2         |
| Open-source LVLMs     |         |      |                   |             |                   |             |                   |             |                   |      |            |      |                   |             |                   |          |             |
| TinyLLaVA[53]         | 3B      | 6.0  | 10.0              | 45.9        | 13.8              | 6.8         | 15.2              | 10.5        | 13.2              | 32.9 | 10.8       | 5.4  | 1.5               | 16.4        | 7.6               | 17.7     | 10.3        |
| Yi-VL[49]             | 6B      | 5.3  | 7.4               | 45.6        | 14.1              | 5.1         | 9.4               | 3.9         | <i>16.6</i>       | 29.2 | 10.9       | 3.8  | 3.0               | 15.6        | 0.0               | 15.5     | 8.8         |
| LLaVA-1.5[24]         | 7B      | 4.5  | $\underline{0.0}$ | 45.5        | 9.2               | 4.6         | 5.2               | 6.9         | 6.2               | 28.1 | 4.9        | 3.3  | $\underline{0.0}$ | <i>10.7</i> | $\underline{0.0}$ | 14.8     | 3.6         |
| ShareGPT4V[5]         | 7B      | 3.5  | 1.8               | 49.1        | 10.1              | 4.2         | 6.3               | 8.5         | 6.9               | 31.7 | 5.1        | 3.0  | 0.7               | 11.9        | $\underline{0.0}$ | 16.0     | 4.4         |
| InternLM-XC2[12]      | 7B      | 7.5  | 1.4               | 53.4        | <i>17.3</i>       | 24.8        | 7.9               | <u>18.1</u> | 15.0              | 36.8 | 6.2        | 28.0 | <i>10.5</i>       | 28.1        | 7.5               | 28.1     | 9.4         |
| Qwen-VL-Chat[2]       | 8B      | 10.0 | 4.2               | 49.6        | 0.3               | 11.0        | 4.0               | 12.3        | 6.4               | 44.5 | 11.9       | 11.4 | 0.3               | 23.9        | 0.0               | 23.2     | 3.9         |
| Deepseek-VL[28]       | 8B      | 3.2  | 10.6              | 49.6        | 15.5              | 14.3        | 10.8              | 11.6        | 14.9              | 33.7 | 23.1       | 11.4 | 3.3               | 15.7        | 0.0               | 19.9     | 11.2        |
| Monkey-Chat[23]       | 10B     | 4.7  | 12.6              | 55.4        | 7.2               | 11.3        | 18.4              | 11.7        | 14.2              | 33.0 | 28.5       | 9.0  | 4.5               | 13.5        | 11.1              | 19.8     | <i>13.8</i> |
| LLaVA-1.5[24]         | 13B     | 9.6  | 0.0               | 47.2        | 9.8               | 5.7         | 7.0               | 8.6         | 7.2               | 31.1 | 10.7       | 5.3  | 1.5               | 13.9        | 0.0               | 17.3     | 5.2         |
| CogVLM-Chat[45]       | 17B     | 4.1  | 0.2               | 47.9        | 5.2               | 11.7        | 0.0               | 10.8        | 10.0              | 32.0 | 4.1        | 9.7  | 3.0               | 14.9        | 0.0               | 18.7     | 3.2         |
| Yi-VL[49]             | 34B     | 5.9  | 0.2               | 48.3        | 12.7              | 6.7         | 15.0              | 6.0         | 2.6               | 27.1 | <u>3.7</u> | 2.9  | 1.0               | 18.8        | 0.0               | 16.5     | 5.0         |
| LLaVA-Next[25]        | 34B     | 6.6  | 2.8               | 54.7        | 4.8               | 11.2        | 1.5               | 12.8        | 5.6               | 34.1 | 6.7        | 16.5 | 4.3               | 29.4        | 2.4               | 23.6     | 4.0         |
| InternVL-Chat-v1.2[6] | 40B     | 7.4  | 4.1               | <u>58.5</u> | 3.8               | 12.2        | 0.9               | 13.5        | 4.8               | 34.9 | 5.5        | 23.7 | 6.1               | 32.6        | 0.0               | 26.1     | 3.6         |
| Sphinx-X-MoE[15]      | 57B     | 1.2  | <i>17.9</i>       | 48.7        | 11.9              | 3.8         | 11.2              | 3.9         | 12.4              | 31.2 | 26.4       | 9.7  | 5.0               | 14.8        | 1.0               | 16.2     | 12.3        |
| Avg. across models    | -       | 5.8  | 4.9               | 50.1        | 8.9               | 10.0        | 7.4               | 10.3        | 8.7               | 33.7 | 11.1       | 10.8 | 3.0               | 20.0        | <u>1.9</u>        | <u> </u> |             |

able given that LLaVA-1.5-7B employed the least amount of multi-modal training data among these open-source LVLMs. Despite LLaVA-1.5-7B having the lowest average multi-modal gain, it exhibits minimal multi-modal leakage. Additionally, models like Monkey-Chat, Spinx-X-MoE, and Deepspeek-VL display higher degrees of multi-modal leakage, highlighting the need for the community to consider this factor for fair comparisons.

Analysis from the benchmark perspective. In the final row of Table 6, we list the average multimodal gain and multi-modal leakage for existing LVLMs across all benchmarks for analysis. MM-Bench registers the highest average multi-modal gain at 50.1, indicating a significant overlap between the domains covered by existing LVLMs' training data and MMBench. Conversely, MMMU exhibits the lowest average multi-modal gain at 5.8, suggesting a lesser degree of overlap between the domains of existing LVLMs' training corpora and those included in MMMU. Additionally, MM-Star, as expected, has the lowest degree of multi-modal leakage at 1.9. This provides a comprehensive and fair arena for comparing existing LVLMs. Moreover, we believe evaluating existing LVLMs to derive average ML metrics can also be helpful to the following works in examining newly developed multi-modal benchmarks.

#### 6 Conclusion

In this work, we dig into current evaluation works for large vision-language models (LVLMs) and identify two primary issues: 1) visual content is unnecessary for many samples, and 2) unintentional data leakage exists in LLM and LVLM training. To address these issues, we develop an elite vision-dependent multi-modal benchmark named MMStar and propose two metrics to measure the data leakage and actual performance gain in LVLMs' multi-modal training. MMStar undergoes the manual review of each sample, covering 6 core capabilities and 18 detailed axes for an in-depth evaluation of LVLMs' multimodal capabilities. In our evaluation of 16 diverse LVLMs on MMStar, even the best model scores under 60 on average. We also analyze the MG and ML metrics across 6 multimodal benchmarks and MMStar, providing valuable insights for the community on gathering multimodal training data and crafting new benchmarks. In the future, we plan to expand MMStar into a larger, online test set and explore dynamic evaluation methods to maintain sample visual dependency and reduce accidental data leakage into LLM's and LVLM's training corpora.

# A Cases of Lacking Visual Dependency

![](_page_13_Figure_1.jpeg)

Figure 6: We highlight cases in existing benchmarks where evaluation samples lack the visual necessary.

## B Cases of Data Leakage in LLMs' Training Data

![](_page_14_Figure_1.jpeg)

Figure 7: We highlight cases in existing benchmarks where evaluation samples are leaked into LLMs' training data

## C Cases of Data Leakage in LVLMs' Multi-Modal Training Data

![](_page_15_Figure_1.jpeg)

Figure 8: We highlight cases in existing benchmarks where evaluation samples are leaked into LVLMs' multimodal training data.

# D Detailed Evaluation Results of LVLMs on Six Multi-modal Benchmarks

Table 7: **Evaluation of various LVLMs on six popular multi-modal benchmarks.** For the "strategy" column, "LLM" refers to evaluating using the corresponding LLM base of the LVLM, while "LVLM-text" denotes evaluating LVLMs without accessing images. We employ the 0-shot inference strategy for LLMs to align the evaluation protocols of LVLMs. The highest results of the LVLM-text setting across the models are highlighted in **bold and underlined.** 

| Model                                           | Param. | Strategy          | MMMU         | MMB          | ScienceQA               | AI2D                | SEED         | MathVista    | Avg.         |  |  |  |  |
|-------------------------------------------------|--------|-------------------|--------------|--------------|-------------------------|---------------------|--------------|--------------|--------------|--|--|--|--|
| Baseline                                        |        |                   |              |              |                         |                     |              |              |              |  |  |  |  |
| Random Choice                                   | -      | -                 | 22.1         | 0.0          | 24.2                    | 23.8                | 24.3         | 17.9         | 18.7         |  |  |  |  |
| Closed-source LVLMs and corresponding LLM bases |        |                   |              |              |                         |                     |              |              |              |  |  |  |  |
| GPT4V[35]                                       |        | LLM               | 41.2         | 12.2         | 64.3                    | 59.7                | 10.1         | 24.2         | 35.3         |  |  |  |  |
| (GPT4-Turbo[34])                                | -      | LVLM-text         | 45.1         | <u>17.6</u>  | 68.2                    | $\frac{62.5}{75.3}$ | 28.4         | 25.4         | 41.2         |  |  |  |  |
|                                                 |        | LVLM<br>LLM       | 53.6<br>42.9 | 69.6<br>18.4 | 81.4<br>68.9            | 75.3<br>59.2        | 71.6<br>35.5 | 44.7<br>23.3 | 66.0<br>41.4 |  |  |  |  |
| GeminiPro-Vision[41]                            | _      | LVLM-text         | 39.4         | 16.7         | 66.3                    | 54.5                | 27.9         | 24.5         | 38.2         |  |  |  |  |
| (GeminiPro[41])                                 |        | LVLM              | 44.4         | 68.1         | 80.6                    | 68.0                | 64.3         | 36.0         | 60.2         |  |  |  |  |
| Open-source LVLMs and corresponding LLM bases   |        |                   |              |              |                         |                     |              |              |              |  |  |  |  |
| Tiny I aVA[52]                                  |        | LLM               | 20.0         | 7.2          | 47.1                    | 38.7                | 26.4         | 22.0         | 26.9         |  |  |  |  |
| TinyLLaVA[53]<br>(Phi2-2.7B[32])                | 3B     | LVLM-text         | 30.0         | 21.0         | 62.3                    | 51.9                | 37.2         | 23.5         | 37.7         |  |  |  |  |
| (T III 2 2.7 D[32])                             |        | LVLM              | 36.0         | 66.9         | 69.1                    | 62.4                | 70.1         | 28.9         | 55.6         |  |  |  |  |
| Yi-VL[49]                                       | (D     | LLM               | 25.7         | 9.5          | 58.1                    | 39.1                | 27.4         | 21.2         | 30.2         |  |  |  |  |
| (Yi-6B[49])                                     | 6B     | LVLM-text<br>LVLM | 33.1<br>38.4 | 23.6<br>69.2 | 67.5<br>72.6            | 55.7<br>59.6        | 38.3<br>67.5 | 24.2<br>28.0 | 40.4<br>55.9 |  |  |  |  |
|                                                 |        | LLM               | 29.9         | 10.3         | 58.9                    | 42.5                | 32.6         | 22.0         | 32.7         |  |  |  |  |
| LLaVA-1.5[24]                                   | 7B     | LVLM-text         | 29.9         | 19.5         | 64.1                    | 48.7                | 37.5         | 20.3         | 36.7         |  |  |  |  |
| (Vicuna-v1.5-7B[8])                             |        | LVLM              | 34.4         | 65.0         | 68.7                    | 55.6                | 65.6         | 23.6         | 52.2         |  |  |  |  |
| ShareGPT4V[5]                                   |        | LLM               | 29.9         | 10.3         | 58.9                    | 42.5                | 32.6         | 22.0         | 32.7         |  |  |  |  |
| (Vicuna-v1.5-7B[8])                             | 7B     | LVLM-text         | 31.7         | 20.4         | 65.2                    | 49.4                | 37.7         | 22.7         | 37.9         |  |  |  |  |
| (vicula vi.s /B[o])                             |        | LVLM              | 35.2         | 69.5         | 69.4                    | 57.9                | 69.4         | 25.7         | 54.5         |  |  |  |  |
| InternLM2-XC2[12]                               | an.    | LLM               | 32.8         | 8.9          | 64.0                    | 48.3                | 31.9         | 18.9         | 34.1         |  |  |  |  |
| (InternLM2-7B[42])                              | 7B     | LVLM-text<br>LVLM | 34.2<br>41.7 | 26.2<br>79.6 | <del>71.9</del><br>96.7 | 63.3<br>81.4        | 38.1<br>74.9 | 29.4<br>57.4 | 43.9<br>72.0 |  |  |  |  |
|                                                 |        | LLM               | 19.8         | 8.4          | 52.7                    | 42.6                | 7.6          | 20.5         | 25.3         |  |  |  |  |
| Qwen-VL-Chat[2]                                 | 8B     | LVLM-text         | 24.0         | 8.7          | 56.7                    | 49.0                | 19.5         | 20.8         | 29.8         |  |  |  |  |
| (Qwen-7B[1])                                    |        | LVLM              | 34.0         | 58.3         | 67.7                    | 61.3                | 64.0         | 32.2         | 52.9         |  |  |  |  |
| Deepseek-VL[28]                                 |        | LLM               | 21.6         | 8.4          | 56.3                    | 38.1                | 13.4         | 20.6         | 26.4         |  |  |  |  |
| (Deepseek-7B[3])                                | 8B     | LVLM-text         | 32.2         | 23.9         | 67.1                    | 53.0                | 36.5         | 23.9         | 39.4         |  |  |  |  |
| (Deepseer-/B[3])                                |        | LVLM              | 35.4         | 73.5         | 81.4                    | 64.6                | 70.2         | 35.3         | 60.1         |  |  |  |  |
| Monkey-Chat[23]                                 | 100    | LLM               | 19.8         | 8.4          | 52.7                    | 42.6                | 7.6          | 20.5         | 25.3         |  |  |  |  |
| (Qwen-7B[1])                                    | 10B    | LVLM-text<br>LVLM | 32.4<br>37.1 | 15.6<br>71.0 | 71.1<br>82.4            | 56.8<br>68.5        | 36.1<br>69.1 | 25.0<br>34.0 | 39.5<br>60.4 |  |  |  |  |
|                                                 |        | LLM               | 28.3         | 11.6         | 59.5                    | 45.0                | 26.3         | 19.6         | 31.7         |  |  |  |  |
| LLaVA-1.5[24]                                   | 13B    | LVLM-text         | 26.0         | 21.4         | 66.5                    | 52.2                | 37.0         | 21.1         | 37.4         |  |  |  |  |
| (Vicuna-v1.5-13B[8])                            |        | LVLM              | 35.6         | 68.6         | 72.2                    | 60.8                | 68.1         | 26.4         | 55.3         |  |  |  |  |
| CogVLM-Chat[45]                                 |        | LLM               | 29.9         | 10.3         | 58.9                    | 42.5                | 32.6         | 22.0         | 32.7         |  |  |  |  |
| (Vicuna-v1.5-7B[8])                             | 17B    | LVLM-text         | 30.1         | 15.5         | 54.6                    | 52.5                | 36.7         | 25.0         | 35.7         |  |  |  |  |
| (Teana The /E[0])                               |        | LVLM              | 34.2         | 63.4         | 66.3                    | 63.3                | 68.7         | 34.7         | 55.1         |  |  |  |  |
| Yi-VL[49]                                       | 24D    | LLM               | 37.1         | 10.5         | 53.6                    | 57.3                | 37.3         | 21.7         | 36.3         |  |  |  |  |
| (Yi-34B[49])                                    | 34B    | LVLM-text<br>LVLM | 37.3<br>43.2 | 23.2<br>71.5 | 68.6<br>75.3            | 59.9<br>65.9        | 41.0<br>68.1 | 22.7<br>25.6 | 42.1<br>58.3 |  |  |  |  |
|                                                 |        | LLM               | 37.6         | 20.1         | 69.4                    | 60.2                | 35.0         | 17.9         | 37.2         |  |  |  |  |
| LLaVA-Next[25]                                  | 34B    | LVLM-text         | 40.4         | 24.9         | 70.9                    | 65.8                | 41.7         | 22.2         | 44.3         |  |  |  |  |
| (NH2-Yi-34B[33])                                |        | LVLM              | 47.0         | 79.6         | 82.1                    | 78.6                | 75.8         | 38.7         | 67.0         |  |  |  |  |
| InternVL-Chat-v1.2[6]                           |        | LLM               | 37.6         | 20.1         | 69.4                    | 60.2                | 35.0         | 17.9         | 40.0         |  |  |  |  |
| (NH2-Yi-34B[33])                                | 40B    | LVLM-text         | 41.7         | 23.9         | 70.3                    | <u>65.0</u>         | 40.5         | 24.0         | 44.2         |  |  |  |  |
| ( = = = = = = = = = = = = = = = = = = =         |        | LVLM              | 49.1         | 82.4         | 82.5                    | 78.5                | 75.4         | 47.7         | 69.3         |  |  |  |  |
| Sphinx-X-MoE[15]                                | 57P    | LUI M toxt        | 25.7         | 8.6          | 57.2                    | 48.7                | 13.5         | 23.4         | 29.5         |  |  |  |  |
| (Mixtral-8x7B[19])                              | 57B    | LVLM-text<br>LVLM | 43.6<br>44.8 | 20.5<br>69.2 | 68.4<br>72.2            | 61.1<br>65.0        | 39.9<br>71.1 | 28.4<br>38.1 | 43.7<br>60.1 |  |  |  |  |
|                                                 | I      | TA PIAI           | ++.0         | 09.4         | 12.2                    | 05.0                | / 1.1        | 30.1         | 00.1         |  |  |  |  |

#### References

- [1] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al. Qwen technical report. *arXiv preprint arXiv:2309.16609*, 2023.
- [2] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. *arXiv preprint arXiv:2308.12966*, 2023.
- [3] X. Bi, D. Chen, G. Chen, S. Chen, D. Dai, C. Deng, H. Ding, K. Dong, Q. Du, Z. Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. *arXiv preprint arXiv:2401.02954*, 2024.
- [4] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.
- [5] L. Chen, J. Li, X. Dong, P. Zhang, C. He, J. Wang, F. Zhao, and D. Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.
- [6] Z. Chen, J. Wu, W. Wang, W. Su, G. Chen, S. Xing, Z. Muyan, Q. Zhang, X. Zhu, L. Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [7] S. Cheng, Z. Guo, J. Wu, K. Fang, P. Li, H. Liu, and Y. Liu. Can vision-language models think from a first-person perspective? *arXiv* preprint arXiv:2311.15596, 2023.
- [8] W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality. *See https://vicuna.lmsvs.org (accessed 14 April 2023)*, 2023.
- [9] A. Chowdhery, S. Narang, J. Devlin, M. Bosma, G. Mishra, A. Roberts, P. Barham, H. W. Chung, C. Sutton, S. Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.
- [10] O. Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.
- [11] W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.
- [12] X. Dong, P. Zhang, Y. Zang, Y. Cao, B. Wang, L. Ouyang, X. Wei, S. Zhang, H. Duan, M. Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. *arXiv preprint arXiv:2401.16420*, 2024.
- [13] Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang. Glm: General language model pretraining with autoregressive blank infilling. *arXiv* preprint arXiv:2103.10360, 2021.
- [14] C. Fu, P. Chen, Y. Shen, Y. Qin, M. Zhang, X. Lin, Z. Qiu, W. Lin, J. Yang, X. Zheng, K. Li, X. Sun, and R. Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. *arXiv* preprint arXiv:2306.13394, 2023.
- [15] P. Gao, R. Zhang, C. Liu, L. Qiu, S. Huang, W. Lin, S. Zhao, S. Geng, Z. Lin, P. Jin, et al. Sphinx-x: Scaling data and parameters for a family of multi-modal large language models. *arXiv preprint arXiv:2402.05935*, 2024.
- [16] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pages 6904–6913, 2017.
- [17] C. Jia, Y. Yang, Y. Xia, Y.-T. Chen, Z. Parekh, H. Pham, Q. Le, Y.-H. Sung, Z. Li, and T. Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In *International conference on machine learning*, pages 4904–4916. PMLR, 2021.
- [18] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.
- [19] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.
- [20] A. Kembhavi, M. Salvato, E. Kolve, M. Seo, H. Hajishirzi, and A. Farhadi. A diagram is worth a dozen images. ArXiv, abs/1603.07396, 2016.

- [21] B. Li, R. Wang, G. Wang, Y. Ge, Y. Ge, and Y. Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. *arXiv* preprint arXiv:2307.16125, 2023.
- [22] J. Li, D. Li, S. Savarese, and S. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. *arXiv preprint arXiv:2301.12597*, 2023.
- [23] Z. Li, B. Yang, Q. Liu, Z. Ma, S. Zhang, J. Yang, Y. Sun, Y. Liu, and X. Bai. Monkey: Image resolution and text label are important things for large multi-modal models. *arXiv* preprint arXiv:2311.06607, 2023.
- [24] H. Liu, C. Li, Y. Li, and Y. J. Lee. Improved baselines with visual instruction tuning. *arXiv preprint* arXiv:2310.03744, 2023.
- [25] H. Liu, C. Li, Y. Li, B. Li, Y. Zhang, S. Shen, and Y. J. Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [26] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023.
- [27] Y. Liu, H. Duan, Y. Zhang, B. Li, S. Zhang, W. Zhao, Y. Yuan, J. Wang, C. He, Z. Liu, et al. Mmbench: Is your multi-modal model an all-around player? *arXiv preprint arXiv:2307.06281*, 2023.
- [28] H. Lu, W. Liu, B. Zhang, B. Wang, K. Dong, B. Liu, J. Sun, T. Ren, Z. Li, Y. Sun, et al. Deepseek-vl: Towards real-world vision-language understanding. *arXiv preprint arXiv:2403.05525*, 2024.
- [29] P. Lu, H. Bansal, T. Xia, J. Liu, C. Li, H. Hajishirzi, H. Cheng, K.-W. Chang, M. Galley, and J. Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [30] P. Lu, S. Mishra, T. Xia, L. Qiu, K.-W. Chang, S.-C. Zhu, O. Tafjord, P. Clark, and A. Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. *Advances in Neural Information Processing Systems*, 35:2507–2521, 2022.
- [31] G. Luo, Y. Zhou, T. Ren, S. Chen, X. Sun, and R. Ji. Cheap and quick: Efficient vision-language instruction tuning for large language models. *arXiv* preprint arXiv:2305.15023, 2023.
- [32] Microsoft. Phi2: The surprising power of small language models. https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/, 2023.
- [33] NousResearch. Nous-hermes-2-yi-34b. https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B, 2023.
- [34] OpenAI. Chatgpt. https://chat.openai.com/, 2023.
- [35] OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV\_System\_Card.pdf, 2023.
- [36] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems*, 35:27730–27744, 2022.
- [37] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In *International conference on machine learning*, pages 8748–8763. PMLR, 2021.
- [38] D. Schwenk, A. Khandelwal, C. Clark, K. Marino, and R. Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In *European Conference on Computer Vision*, pages 146– 162. Springer, 2022.
- [39] P. Sharma, N. Ding, S. Goodman, and R. Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In *Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 2556–2565, 2018.
- [40] H. Taud and J.-F. Mas. Multilayer perceptron (mlp). Geomatic approaches for modeling land change scenarios, pages 451–455, 2018.
- [41] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al. Gemini: a family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805*, 2023.
- [42] I. Team. Internlm: A multilingual language model with progressively enhanced capabilities, 2023.

- [43] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.
- [44] J. Wang, L. Meng, Z. Weng, B. He, Z. Wu, and Y.-G. Jiang. To see is to believe: Prompting gpt-4v for better visual instruction tuning. *arXiv preprint arXiv:2311.07574*, 2023.
- [45] W. Wang, Q. Lv, W. Yu, W. Hong, J. Qi, Y. Wang, J. Ji, Z. Yang, L. Zhao, X. Song, et al. Cogvlm: Visual expert for pretrained language models. *arXiv* preprint arXiv:2311.03079, 2023.
- [46] H. Wu, Z. Zhang, E. Zhang, C. Chen, L. Liao, A. Wang, C. Li, W. Sun, Q. Yan, G. Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. *arXiv preprint arXiv:2309.14181*, 2023.
- [47] A. Yang, B. Xiao, B. Wang, B. Zhang, C. Yin, C. Lv, D. Pan, D. Wang, D. Yan, F. Yang, et al. Baichuan 2: Open large-scale language models. *arXiv preprint arXiv:2309.10305*, 2023.
- [48] Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. *arXiv preprint arXiv:2304.14178*, 2023.
- [49] A. Young, B. Chen, C. Li, C. Huang, G. Zhang, G. Zhang, H. Li, J. Zhu, J. Chen, J. Chang, et al. Yi: Open foundation models by 01. ai. *arXiv preprint arXiv:2403.04652*, 2024.
- [50] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [51] X. Yue, Y. Ni, K. Zhang, T. Zheng, R. Liu, G. Zhang, S. Stevens, D. Jiang, W. Ren, Y. Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. *arXiv* preprint arXiv:2311.16502, 2023.
- [52] P. Zhang, X. D. B. Wang, Y. Cao, C. Xu, L. Ouyang, Z. Zhao, S. Ding, S. Zhang, H. Duan, H. Yan, et al. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. *arXiv preprint arXiv:2309.15112*, 2023.
- [53] B. Zhou, Y. Hu, X. Weng, J. Jia, J. Luo, X. Liu, J. Wu, and L. Huang. Tinyllava: A framework of small-scale large multimodal models. *arXiv preprint arXiv:2402.14289*, 2024.
- [54] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.