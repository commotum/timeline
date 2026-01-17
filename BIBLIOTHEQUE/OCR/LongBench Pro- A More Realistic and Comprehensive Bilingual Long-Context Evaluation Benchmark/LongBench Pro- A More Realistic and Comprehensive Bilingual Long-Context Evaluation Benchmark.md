# LongBench Pro: A More Realistic and Comprehensive Bilingual **Long-Context Evaluation Benchmark**

Ziyang Chen 12 Xing Wu 1 Junlong Jia 3 Chaochen Gao 12 Qi Fu 4 Debing Zhang 4 Songlin Hu 12

![](_page_0_Figure_4.jpeg)

Figure 1. Performance of advanced long-context LLMs on LongBench Pro.

#### Abstract

The rapid expansion of context length in large language models (LLMs) has outpaced existing evaluation benchmarks. Current long-context benchmarks often trade off scalability and realism: synthetic tasks underrepresent real-world complexity, while fully manual annotation is costly to scale to extreme lengths and diverse scenarios. We present **LongBench Pro**, a more realistic and comprehensive bilingual benchmark of 1,500 naturally occurring long-context samples in English and Chinese spanning 11 primary tasks and 25 secondary tasks, with input lengths from 8k to 256k tokens. LongBench Pro supports finegrained analysis with task-specific metrics and a

Our datasets are available at https://huggingface.co/ datasets/caskcsq/LongBench-Pro.

multi-dimensional taxonomy of context requirement (full vs. partial dependency), length (six levels), and difficulty (four levels calibrated by model performance). To balance quality with scalability, we propose a Human-Model Collaborative Construction pipeline: frontier LLMs draft challenging questions and reference answers, along with design rationales and solution processes, to reduce the cost of expert verification. Experts then rigorously validate correctness and refine problematic cases. Evaluating 46 widely used long-context LLMs on LongBench Pro yields three findings: (1) long-context optimization contributes more to long-context comprehension than parameter scaling; (2) effective context length is typically shorter than the claimed context length, with pronounced cross-lingual misalignment; and (3) the "thinking" paradigm helps primarily models trained with native reasoning, while mixedthinking designs offer a promising Pareto tradeoff. In summary, LongBench Pro provides a robust testbed for advancing long-context understanding.

<sup>&</sup>lt;sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences <sup>2</sup>School of Cyber Security, University of Chinese Academy of Sciences <sup>3</sup>School of Artificial Intelligence, Beihang University <sup>4</sup>Xiaohongshu Inc. Correspondence to: Xing Wu, Songlin Hu <wuxing@iie.ac.cn, husonglin@iie.ac.cn>.

# 1. Introduction

Understanding and reasoning over long contexts has become a core capability of large language models (LLMs). With advances in architectures and computational resources, context length has continuously expanded (Comanici et al., 2025; OpenAI, 2025; Anthropic, 2025), making it possible for LLMs to tackle complex long-context tasks such as large-scale codebase analysis, legal document comprehension, and systematic literature review. This rapid progress demands that long-context benchmarks evolve swiftly to accurately assess models' true long-context capabilities and facilitate open-source models in closing the gap with closed-source models.

Several long-context evaluation benchmarks currently exist. Synthetic benchmarks (Hsieh et al., 2024; Yen et al., 2024) enable controlled evaluation at scale, while manually annotated benchmarks (Qiu et al., 2024; Bai et al., 2025) ensure authenticity through complex tasks and realistic scenarios. Although these benchmarks have effectively guided model development over the past few years, they are increasingly insufficient (in terms of task coverage, difficulty level, etc.) for evaluating next-generation models, as context lengths extend to millions of tokens and model capabilities continue to advance rapidly. This creates a pressing need for new benchmarks that satisfy more authentic and comprehensive evaluation of diverse long-context capabilities.

To address this urgent need, we introduce **LongBench Pro**, a realistic and comprehensive long-context evaluation benchmark containing 1,500 bilingual samples in English and Chinese. Built entirely on authentic, naturally occurring long documents, LongBench Pro encompasses 11 primary tasks and 25 secondary tasks, covering the full spectrum of long-context capabilities assessed by all existing benchmarks, as shown in Figure 2. It employs diverse evaluation metrics for fine-grained measurement and introduces a multidimensional taxonomy: (1) Context Requirement—Full context (global integration) versus Partial context (localized retrieval); (2) Length—six levels uniformly distributed from 8k to 256k tokens to analyze scaling behavior; and (3) Difficulty—four levels ranging from Easy to Extreme, calibrated based on model performance. Table 1 summarizes the key differences between LongBench Pro and existing benchmarks.

Creating such a benchmark requires moving beyond traditional construction approaches. Synthetic methods, despite their scalability, cannot fully capture the semantic complexity of realistic long context scenarios. Conversely, purely manual annotation becomes prohibitively expensive and cognitively demanding at extreme lengths, limiting practicality and scalability. To achieve both authenticity and scalability, we propose a novel *Human-Model Collaborative Construction* strategy that synergizes the strengths of advanced LLMs

and human expertise. In this strategy, models generate challenging questions and reference answers, along with design rationales and solution processes, from authentic, long documents corresponding to realistic, long-context scenarios. Meanwhile, human experts act as rigorous critics to verify correctness, filter flawed samples, and calibrate difficulty levels. This approach achieves high realism through the use of authentic documents, ensures quality through expert verification, and reduces costs through model involvement, enabling the efficient production of challenging, comprehensive, and high-quality evaluation samples.

We conduct extensive evaluations of 46 widely studied longcontext LLMs on LongBench Pro and derive three main findings:

- (1) Long-context optimization surpasses parameter scaling for improving long-context comprehension. Performance gains from extending effective context length significantly exceed those from scaling model size. This marks a shift from the traditional "scale-first" to a "context-optimization-first" paradigm.
- (2) Context length–capability gaps and language bias constrain performance consistency. Models' effective context lengths often fall short of their claimed context lengths, and cross-lingual performance remains misaligned. While stronger models reduce these gaps, they persist across most long-context models.
- (3) The thinking paradigm requires native training to overcome long-context bottlenecks. Thinking substantially improves long-context performance, but primarily in models trained with native reasoning. Conventional instruct models show limited or degraded gains when prompted to think. Mixed-thinking models combining fast response and deep reasoning achieve Pareto-optimal performance and may define the future paradigm.

Our contributions are threefold:

- We release LongBench Pro, a realistic and comprehensive bilingual benchmark with 1,500 samples and multi-dimensional categorization for rigorous long-context evaluation.
- We validate a Human-Model Collaborative Construction strategy that transcends the cost-quality trade-off, enabling high-quality sample generation at significantly lower cost than purely manual annotation.
- We provide comprehensive analysis of 46 longcontext LLMs, revealing insights into context length scaling, cross-lingual performance, and reasoning mechanisms.

| Benchmark                           | Text Type          | #Task Metric    | Language | Dimensional Categorization |            |              |             |  |
|-------------------------------------|--------------------|-----------------|----------|----------------------------|------------|--------------|-------------|--|
| Denemmark                           | Teste Type         | "Idaga" Ivictic |          | Lunguage                   | Ctx-Req    | Length       | Difficulty  |  |
| RULER (Hsieh et al., 2024)          | Fully Synthetic    | 4               | Single   | EN                         | 0          | •            | 0           |  |
| $\infty$ BENCH (Zhang et al., 2024) | Synthetic, Natural | 6               | Diverse  | EN, ZH                     | $\bigcirc$ | $\bigcirc$   | $\bigcirc$  |  |
| CLongEval (Qiu et al., 2024)        | Synthetic, Natural | 7               | Diverse  | ZH                         |            | lacktriangle | $\bigcirc$  |  |
| HELMET (Yen et al., 2024)           | Synthetic, Natural | 7               | Diverse  | EN                         | $\bigcirc$ |              | $\bigcirc$  |  |
| LongBench v2 (Bai et al., 2025)     | Fully Natural      | 6               | Single   | EN                         | $\bigcirc$ | $lackbox{}$  | $lackbox{}$ |  |
| LongBench Pro (Ours) Fully Natural  |                    | 11              | Diverse  | EN, ZH                     | •          | •            | •           |  |

Table 1. Comparison of long-context benchmarks. Ctx-Req denotes the Context Requirement dimension.  $\bigcirc$  indicates the absence of dimensional categorization; ● indicates the presence of detailed categorization in this dimension (Ctx-Req  $\ge 2$ ; Length > 3; Difficulty > 2); ● indicates the presence of categorization with a coarser granularity that does not meet the fine-grained criteria.

![](_page_2_Figure_2.jpeg)

Figure 2. Task mapping between LongBench Pro and existing benchmarks.

# 2. Task Framework of LongBench Pro

To systematically characterize long-context capabilities, we organize LongBench Pro into a two-level task taxonomy. We consolidate task formulations from prior long-context benchmarks into 11 primary categories. Figure 2 maps our taxonomy to existing benchmark tasks, illustrating that LongBench Pro covers all the core capability dimensions evaluated by existing benchmarks.

Beyond task type, we also introduce the *context requirement*, an orthogonal dimension capturing how globally a model must depend on the input document:

- Full: solving the task requires integrating evidence dispersed across multiple, distant spans of the document, emphasizing integration and reasoning;
- Partial: solving the task primarily relies on localized spans, emphasizing localization and retrieval.

Crossing the 11 primary categories with context requirements to design 25 secondary categories. Table 2 lists the full taxonomy, and Appendix A provides detailed task definitions.

# 3. Construction Process of LongBench Pro

*Note:* The prompts and guidelines involved in the construction process are detailed in Appendix B.

# 3.1. Document Collection

To ensure realism and coverage, we curate naturally occurring long documents from the public internet across diverse domains (e.g., news, medicine, science, literature, law, and education) and formats (e.g., reports, tables, code, dialogues, lists, and JSON). We balance the collection across single-document and multi-document settings, as well as English and Chinese, and six target length buckets (8k/16k/32k/64k/128k/256k tokens), where token length is measured using the Qwen tokenizer (Yang et al., 2025). Since raw documents rarely match target lengths exactly, we assign a document to a bucket if its length falls within  $\pm 20\%$  of the target. All documents undergo a compliance review by human annotators to exclude content that is privacy-sensitive, copyrighted, or otherwise non-compliant.

|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                                                 | Context     |                   |  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|-------------|-------------------|--|
| Task                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Description                                     | Requirement | Metric            |  |
| T1 Retrieval & Ranking                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Retrieve content and rank most relevant first   |             |                   |  |
| T1.1 Global Cohesive Retrieval                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Retrieve full text and reorganize               | Full        | NDCG@k            |  |
| T1.2 Key-Snippet Retrieval                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Locate target fragment in specified paragraph   | Partial     | NDCG@k            |  |
| T2 Sequencing & Structure Reconstruction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Restore timeline or logical order               |             |                   |  |
| T2.1 Global Timeline Reconstruction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Sort unordered events in the whole text         | Full        | Pairwise Accuracy |  |
| T2.2 Local Causal Chain Sorting                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Sort content in a specific paragraph            | Partial     | Pairwise Accuracy |  |
| <sup>♠</sup> T3 Evidence-Grounded QA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Answer fact/reasoning questions based on eviden | ce          |                   |  |
| T3.1 Multi-Doc Integration QA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Use multi-hop information to answer questions   | Full        | Accuracy          |  |
| T3.2 Single-Hop Fact QA                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Answer questions based on local paragraphs      | Partial     | Accuracy          |  |
| ₹ T4 Summarization & Synthesis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Generate abstract summary under given constrair | nts         |                   |  |
| T4.1 Global-Coverage Constrained Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Generate summary of full text                   | Full        | SemSim, ROUGE-L   |  |
| T4.2 Query-Focused Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Generate summary of specific subtopic           | Partial     | SemSim, ROUGE-L   |  |
| T5 Attribution & Citation Alignment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Bind correct sources to generated text          |             |                   |  |
| T5.1 Full-Sentence Citation Alignment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Citation alignment for all sentences            | Full        | F1                |  |
| T5.2 Key-Statement Citation Alignment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Citation alignment for specified sentences      | Partial     | F1                |  |
| T6 Aggregation & Clustering                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Cluster and output statistics/examples/sort     |             |                   |  |
| T6.1 Large-Scale Document Clustering                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Return all category proportions                 | Full        | SubEM             |  |
| T6.2 Targeted Subset Cluster Identification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Return query category instances                 | Partial     | F1                |  |
| T6.3 Global Frequency Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Count and sort global word frequency            | Full        | Pairwise Accuracy |  |
| T7 Consistency & Compliance Checking                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Detect and locate contradictions/violations     |             |                   |  |
| T7.1 Global Conflict & Inconsistency Localization                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Locate contradictory segments in the full text  | Full        | F1                |  |
| T7.2 Targeted Rule or Condition Violation Detection                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Locate content that violates specific rules     | Partial     | F1                |  |
| T7.3 Comprehensive Error & Anomaly Sweep                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Locate spelling errors in the full text         | Full        | F1                |  |
| ■ T8 Structured & Numeric Reasoning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Numerical calculations in structured text       |             |                   |  |
| T8.1 Structured Multi-Source Consistency Verification                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Numerical computation in multi-source           | Full        | SubEM             |  |
| T8.2 Single-Source Targeted Aggregation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Query computation in single-source              | Partial     | SubEM             |  |
| T8.3 Long-Context Procedural State Tracking                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Track entity state evolution                    | Full        | SubEM             |  |
| T9 Version & Code Diff Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Compare changes in different text/code versions |             |                   |  |
| T9.1 Dependency-Aware Multi-Version Impact Analysis                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | Track dependency changes across versions        | Full        | F1                |  |
| T9.2 Localized Interface Change Detection                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Detect local version differences                | Partial     | F1                |  |
| ₱ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induction & In-Context Learning  ■ T10 Rule Induct | Summarize rules and make decisions on new sam   | ples        |                   |  |
| T10.1 Large-Scale In-Context Rule Induction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Induce rules from the global context            | Full        | SubEM             |  |
| T10.2 Targeted Example-Based Rule Induction                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Induce rules from the targeted examples         | Partial     | SubEM             |  |
| T11 Dialogue Memory & Long-Horizon Tracking                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Track and respond to dialogue history           |             |                   |  |
| T11.1 Long-Range Entity & Commitment Tracking                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Track entity states across the global context   | Full        | Accuracy          |  |
| T11.2 Short-Range Reference Resolution & State Query                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Resolve references and states in local context  | Partial     | Accuracy          |  |

Table 2. Task definitions of LongBench Pro.

# 3.2. Human-Model Collaborative Sample Generation

To balance authenticity with annotation cost, we adopt a human-model collaborative construction strategy. Given a lengthy document, we prompt multiple frontier LLMs (Gemini-2.5-Pro (Comanici et al., 2025), GPT-5 (OpenAI, 2025), Claude-4-Sonnet (Anthropic, 2025), DeepSeek-V3.2 (Liu et al., 2025), and Qwen3-235B-A22B-Thinking-2507 (Yang et al., 2025)) to draft three candidate samples aligned with a target task definition and context requirement, including (i) questions, (ii) reference answers, and (iii) the corresponding design rationales and solution processes to support later verification.

Subsequently, human annotators critically evaluate the model-generated content, focusing primarily on the following aspects:

- (1) Verify task alignment and context requirement based on the provided design rationale;
- (2) Validate answer correctness using the accompanying solution processes;
- (3) Estimate difficulty using the responses of the five drafting models (a sample is considered challenging if at least one model answers incorrectly);
- (4) Select the best sample that meets the criteria or can be edited with minimal changes to meet the criteria; if none, move to the next document.

This workflow leverages models for scalable drafting and humans for rigorous verification, mitigating both human cognitive limitations at extreme lengths and model hallucinations. Each accepted sample is reviewed by a long-context

![](_page_4_Figure_0.jpeg)

Figure 3. The construction process of LongBench Pro includes document collection, human-model collaborative sample generation, question standardization, answer review, and difficulty classification.

expert; failed cases must be revised until they satisfy the criteria. Section 5.7 empirically evaluates the effectiveness of this construction strategy.

#### 3.3. Question Standardization

Chain-of-Thought prompting (CoT) (Wei et al., 2022) demonstrates that the thinking process enhances model performance. To systematically evaluate the upper bound of model capabilities, we construct two rigorously standardized prompt templates for each question: a non-thinking prompt and a thinking prompt. Each prompt includes a task description, output requirements, and an output example, uniformly instructing the model to present answer elements line by line, following the identifier "[Answer]" for automated extraction and evaluation. The only difference between the two types of prompts is that the non-thinking prompt requires the model to answer directly. In contrast, the thinking prompt requires it to perform explicit step-by-step thinking before producing the final answer.

#### 3.4. Answer Review

To ensure sample quality, we systematically review all samples. Annotation experience shows that human annotators are more reliable in judging the correctness of answer components, while models are better at generating diverse candidate components. Based on this complementarity, we first collect predictions from five advanced models for each

sample. Then, we instruct annotators to examine each component of the original answer to ensure precision, followed by reviewing model predictions to improve recall. Two annotators independently verify each sample. Samples without detected issues are directly included in the benchmark. If either annotator identifies a potential problem, an additional long-context expert evaluates the sample and decides whether it requires reconstruction.

# 3.5. Difficulty Classification

To improve the benchmark's utility in real-world applications, we assign each sample a difficulty label defined from a model-centric perspective rather than subjective human ratings (Bai et al., 2025), which aligns more closely with the practical needs of contemporary LLM evaluation and provides a more natural foundation for the co-evolution of benchmarks and model capabilities. Concretely, we rank models by overall performance and partition them into three tiers (high/mid/low). Within each tier, we select five representative models that perform the best while covering diverse architectures, which reduces sensitivity to outliers and avoids the bias introduced by relying on a single model. This tiered design provides multiple decision boundaries for fine-grained difficulty labeling. The selected models are:

• High-performing models: Gemini-2.5-Pro (Comanici et al., 2025), GPT-5 (OpenAI, 2025), Claude-4-Sonnet (Anthropic, 2025), DeepSeek-V3.2 (Liu et al.,

![](_page_5_Figure_0.jpeg)

Figure 4. Overview of LongBench Pro sample distributions.

2025), Qwen3-235B-A22B-Thinking-2507 (Yang et al., 2025)

- Mid-performing models: GLM-4.6 (, Z.ai), DeepSeek-V3-0324 (Liu et al., 2024), Kimi-K2-Instruct-0905 (Team et al., 2025b), Qwen3-30B-A3B-Instruct-2507 (Yang et al., 2025), MiniMax-M2 (Team, 2025)
- Low-performing models: Ministral-3-8B-Instruct-2512 (AI, 2025), Qwen3-8B (Yang et al., 2025), Qwen2.5-72B-Instruct (Qwen et al., 2025), Llama-3.1-405B-Instruct (AI, 2024), Gemma-3-27B-It (Team et al., 2025a)

On this basis, we divide the samples into four difficulty levels according to the answering performance of three groups of models, defined as follows:

- Extreme: samples that at most one high-performing model can answer correctly (for which a score greater than 0.65 on the summarization task is considered correct);
- Hard: after excluding Extreme samples, samples that at most one mid-performing model can answer correctly;
- Moderate: after further excluding Hard samples, samples that at most one low-performing model can answer correctly;
- Easy: the remaining samples are automatically assigned to this level.

This multi-tier progressive approach yields fine-grained difficulty labels aligned with model capabilities and provides a unified, scalable framework for analyzing cross-difficulty performance.

# 4. Data Statistics and Validation of LongBench Pro

We construct LongBench Pro with a balanced design: 5 samples for each combination of 25 secondary tasks, 2 languages, and 6 length buckets, resulting in 1,500 samples in total. Figure 4 reports the distributions of text type, language, context requirement, length, and difficulty, as well as the per-task composition. To validate sample quality, we uniformly select 300 samples across secondary tasks, languages, and lengths and audit (i) attribute correctness (whether language, length, secondary task, and context requirement are all correct) and (ii) answer correctness (whether the answer is fully correct). Among the 300 samples, the attribute correctness reaches 99.3%, and the answer correctness reaches 97.3%, with problematic samples exhibiting only minor deviations (impacting the overall score by only 0.96), demonstrating the high quality of the benchmark samples.

# 5. Evaluation

#### 5.1. Evaluation Settings

Evaluation Metrics: We use task-specific metrics summarized in Table 2. T1 (Retrieval & Ranking) is evaluated by NDCG@k. T2 and T6.3 (ordering-style tasks) use pairwise accuracy based on rank consistency. T3 and T11 are multiple-choice and use accuracy. For tasks with potentially multiple answer components extracted from the source text (T5, T6.2, T7, and T9), we use F1 to penalize spurious components. For tasks with a single canonical answer that is not directly copied from the source (T6.1, T8, and T10), we use SubEM. For T4 (Summary), we combine semantic similarity (SemSim) and ROUGE-L to balance semantic faithfulness and coverage. Each summarization sample includes three reference summaries. The metrics are first computed between the generated summary and each reference summary individually, and the maximum value for each metric is taken to reflect consistency with the bestmatching reference. The final weighted score is calculated

$$\begin{aligned} \text{Score}_{\text{summary}} &= 0.5 \cdot \max_{i} \text{SemSim}(S_{\text{gen}}, S_{\text{ref}_{i}}) \\ &+ 0.5 \cdot \max_{i} \text{ROUGE-L}(S_{\text{gen}}, S_{\text{ref}_{i}}) \end{aligned} \tag{1}$$

All metrics have a value range of [0,1]. We report the average score over all samples and multiply by 100.

Evaluated Models: We evaluate 46 long-context models that vary in transparency (closed-source, such as GPT-5 (OpenAI, 2025); open-source, such as GPT-OSS-120B (OpenAI et al., 2025)), thinking mode (thinking, such as DeepSeek-R1 (DeepSeek-AI et al., 2025); mixedthinking, such as DeepSeek-V3.2 (Liu et al., 2025); nonthinking, such as DeepSeek-V3-0324 (Liu et al., 2024)), size (3B, such as Ministral-3-3B-Instruct-2512 (AI, 2025); 1T, such as Kimi-K2-Instruct-0905 (Team et al., 2025b)), architecture (dense, such as Qwen3-32B (Yang et al., 2025); MoE, such as Qwen3-235B-A22B-Instruct-2507 (Yang et al., 2025)), and context length (128k, such as Gemma-3-27B-It (Team et al., 2025a); 1M, such as Gemini-2.5-Pro (Comanici et al., 2025)), with the goal of comprehensively assessing the long-context performance of current LLMs.

**Inference Settings:** We uniformly use each model's default inference parameters to run inference three times, and report both the general performance (the average score across multiple responses) and the upper-bound performance (Best-of-N, the highest score among the multiple responses; Pass@N, the probability that at least one response is completely correct). For models without default inference parameters, we set the temperature to 1.0. For thinking models, we use the non-thinking prompt and report their thinking scores. For mixed-thinking models, we use non-thinking prompts and report both the non-thinking and thinking scores under the disabled and enabled thinking states. For non-thinking (instruct) models, we use non-thinking prompts and thinking prompts separately to report the corresponding non-thinking and thinking scores. For the thinking score, models that support a 256k context length set the output length to 32k to enable more thorough reasoning, while for other models, we set the output length to 8k to reserve more budget for the input. For the non-thinking score, the output length is uniformly set to 1k. When the sample length exceeds the model's context length, we truncate the sample from the middle to an appropriate length for input, with the truncation length uniformly set to the model's context length minus the output length. Detailed inference parameter settings for different models are provided in Appendix D. Unless otherwise specified, we report the thinking scores by default.

#### 5.2. General Performance

Table 3 summarizes the general performance of 46 models on LongBench Pro (gray-shaded cells are thinking scores).

We observe a clear stratification in overall long-context performance. The top three models are Gemini-2.5-Pro (73.42), GPT-5 (72.61), and Claude-4-Sonnet (69.87). Among open-source models, DeepSeek-V3.2 (67.82) and Qwen3-235B-A22B-Thinking-2507 (66.97) are the strongest, narrowing the gap to the best closed-source model to within 6 points.

Based on an in-depth analysis of the evaluation results in Table 3, we draw the following insights:

- (1) Long-Context Optimization Outperforms Model Size Scaling. For long-context tasks, the Model Size Scaling Law still holds. For example, Qwen3 (32k natively and 128k with YaRN) improves in performance from 4B to 32B (40.82 → 51.12), but its marginal gains decrease. In contrast, the long-context optimized Qwen3-4B-Instruct-2507 (256k) achieves a score of 45.68, surpassing Qwen3-8B (44.34), while Qwen3-30B-A3B-Instruct-2507 (256k) attains a high score of 54.52, outperforming the larger Qwen3-32B (51.12). This demonstrates that extending the effective context length constitutes the primary approach for improving long-context performance, and its effectiveness far exceeds that of scaling up parameters by several times.
- (2) Discrepancy between Claimed Context Length and Effective Context Length. For some models, the claimed context length does not positively correlate with their actual performance. For example, although MiniMax-Text-01 claims to support a context length of up to 4M, its overall score is only 45.00, even falling behind most models that have a context length of merely 128k. This clear inconsistency indicates that the ability of a model to accept longer text inputs does not equate to its ability to effectively leverage this information for association, integration, and reasoning. In other words, the claimed context length reflects the model's input capacity, whereas the effective context length reflects its actual long-context understanding and processing capability, and a significant gap may exist between the two.
- (3) Uneven Distribution of Long-Context Capabilities Between Chinese and English. Different series of models exhibit clearly uneven performance in long-context tasks across languages. For instance, series such as GPT, Claude, Mistral, and Llama generally perform better in English long-context scenarios, whereas series like GLM, Kimi, and MiniMax demonstrate stronger capabilities in Chinese long-context tasks. This phenomenon indicates that the "language alignment" of current LLMs in long-context processing remains insufficient, and language differences significantly affect model robustness and generalization. However, it is noteworthy that as model scale increases and overall capabilities improve, the performance gap between languages gradually narrows. High-performing models (e.g., DeepSeek-V3.2 and Qwen3-235B-A22B-Thinking-2507) leverage stronger cross-lingual semantic representation and

|           | Model Context Company Language Difficulty                                                                                                                                                                                                            |                                                                                                                |                                                              |                                                                                    |                                                                                                                   |                                                                               |                                                                                                                   |                                                                              |                                                                                                                   |                                                                      |                                                                                                                   |                                                                               |                                                                                                                   |                                                                               |                                                                                                                   |                                                                               |                                                                                                                   |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
|           | Model                                                                                                                                                                                                                                                |                                                                                                                | Context<br>Length                                            | Ove                                                                                | erall                                                                                                             | Eng                                                                           | Lang<br>glish                                                                                                     |                                                                              | nese                                                                                                              | Extr                                                                 | eme                                                                                                               | Ha                                                                            | rd<br>ard                                                                                                         |                                                                               | erate                                                                                                             | Easy                                                                          |                                                                                                                   |
| <b>*</b>  | Gemini-2.5-Pro<br>Gemini-2.5-Flash<br>Gemma-3-27B-It<br>Gemma-3-12B-It<br>Gemma-3-4B-It                                                                                                                                                              | Thinking<br>Mixed<br>Instruct<br>Instruct<br>Instruct                                                          | 1M<br>1M<br>128k<br>128k<br>128k                             | 55.92<br>36.14<br>32.16<br>21.76                                                   | <b>73.42</b> 67.41 37.34 31.92 21.20                                                                              | 55.29<br>37.46<br>33.03<br>22.63                                              | <b>72.35</b> 67.22 40.89 34.43 23.28                                                                              | 56.54<br>34.81<br>31.28<br>20.89                                             | 74.49<br>67.59<br>33.78<br>29.41<br>19.12                                                                         | 44.26<br>30.22<br>26.44<br>19.31                                     | 50.77<br>47.39<br>27.78<br>25.74<br>18.72                                                                         | 57.87<br>33.20<br>30.43<br>20.70                                              | 81.03<br>72.19<br>30.56<br>28.02<br>19.87                                                                         | 53.99<br>25.04<br>23.39<br>15.82                                              | 81.98<br>72.39<br>24.53<br>22.61<br>13.85                                                                         | 66.55<br>49.96<br>43.66<br>28.18                                              | 84.40<br>79.82<br>57.81<br>45.48<br>28.66                                                                         |
| <b>\$</b> | GPT-5<br>GPT-40<br>GPT-OSS-120B<br>GPT-OSS-20B                                                                                                                                                                                                       | Thinking<br>Instruct<br>Thinking<br>Thinking                                                                   | 272k<br>128k<br>128k<br>128k                                 | 46.67<br>-                                                                         | <b>72.61</b> 49.44 52.61 44.66                                                                                    | -<br>47.67<br>-<br>-                                                          | <b>73.24</b> 52.61 54.67 47.83                                                                                    | -<br>45.66<br>-<br>-                                                         | 71.97<br>46.26<br>50.54<br>41.49                                                                                  | 36.30<br>-<br>-                                                      | 48.37<br>34.39<br>35.4<br>31.59                                                                                   | -<br>44.88<br>-<br>-                                                          | <b>78.74</b> 41.35 44.97 35.89                                                                                    | 43.03                                                                         | <b>82.31</b> 43.07 50.66 39.33                                                                                    | 59.38<br>-<br>-                                                               | 85.23<br>71.84<br>74.06<br>65.05                                                                                  |
| *         | Claude-4-Sonnet<br>Claude-3.7-Sonnet                                                                                                                                                                                                                 | Mixed<br>Mixed                                                                                                 | 1M<br>200k                                                   | <b>56.07</b> 51.45                                                                 | <b>69.87</b> 59.66                                                                                                | <b>57.14</b> 51.89                                                            | <b>71.09</b> 60.49                                                                                                | <b>54.99</b> 51.00                                                           | <b>68.65</b> 58.84                                                                                                | <b>42.92</b> 37.31                                                   | 47.05<br>40.07                                                                                                    | <b>57.57</b> 47.29                                                            | <b>74.72</b> 56.58                                                                                                | <b>53.96</b> 48.38                                                            | <b>76.58</b> 61.56                                                                                                | 68.42<br>68.69                                                                | 83.78<br>78.26                                                                                                    |
| <b>⊘</b>  | DeepSeek-V3.2<br>DeepSeek-V3.1<br>DeepSeek-R1-0528<br>DeepSeek-R1<br>DeepSeek-V3-0324                                                                                                                                                                | Mixed<br>Mixed<br>Thinking<br>Thinking<br>Instruct                                                             | 160k<br>128k<br>128k<br>128k<br>128k                         | 51.67<br>51.39<br>-<br>-<br>51.70                                                  | 67.82<br>66.22<br>61.89<br>60.07<br>56.71                                                                         | 50.61<br>50.35<br>-<br>-<br>51.62                                             | 67.89<br>66.17<br>59.90<br>62.00<br>58.14                                                                         | 52.73<br>52.42<br>-<br>51.78                                                 | 67.75<br>66.26<br>63.89<br>58.13<br>55.27                                                                         | 40.45<br>41.07<br>-<br>-<br>40.40                                    | 44.27<br>42.68<br>41.49<br>40.76<br>38.69                                                                         | 51.63<br>49.29<br>-<br>-<br>48.30                                             | 67.73<br>62.22<br>53.68<br>53.39<br>46.20                                                                         | 51.36<br>48.80<br>-<br>-<br>49.68                                             | 75.08<br>73.53<br>66.53<br>58.83<br>57.14                                                                         | 62.12<br>63.61<br>-<br>65.26                                                  | 85.02<br>85.72<br>82.67<br>82.44<br>79.20                                                                         |
| \$        | Qwen3-235B-A22B-Thinking-2507<br>Qwen3-Part-2507<br>Qwen3-Next-80B-A3B-Thinking<br>Qwen3-Next-80B-A3B-Thinking-2507<br>Qwen3-30B-A3B-Thinking-2507<br>Qwen3-30B-A3B-Instruct-2507<br>Qwen3-4B-Thinking-2507<br>Qwen3-4B-Instruct-2507<br>Qwen3-4B-10 | Thinking Instruct Thinking Instruct Thinking Instruct Thinking Instruct Mixed Mixed Mixed Mixed Mixed Instruct | 256k<br>256k<br>256k<br>256k<br>256k<br>256k<br>256k<br>256k | 52.51<br>51.54<br>43.84<br>-<br>36.78<br>40.28<br>37.11<br>33.41<br>31.26<br>39.64 | 66.97<br>63.77<br>63.95<br>60.76<br>59.68<br>54.52<br>50.10<br>45.68<br>51.12<br>47.14<br>44.34<br>40.82<br>44.09 | 52.22<br>50.39<br>43.17<br>35.70<br>39.78<br>36.61<br>33.04<br>31.60<br>39.48 | 66.83<br>63.88<br>62.91<br>60.30<br>60.14<br>55.55<br>49.81<br>46.27<br>52.24<br>50.53<br>44.60<br>41.94<br>44.99 | 52.80<br>52.69<br>44.5<br>37.85<br>40.77<br>37.61<br>33.79<br>30.92<br>39.79 | 67.12<br>63.65<br>64.99<br>61.22<br>59.22<br>53.49<br>50.39<br>45.10<br>50.01<br>43.75<br>44.08<br>39.70<br>43.18 | 39.39<br>35.44<br>30.29<br>32.56<br>31.13<br>29.99<br>27.20<br>32.36 | 43.39<br>43.24<br>42.47<br>40.47<br>40.47<br>37.05<br>35.31<br>31.09<br>36.45<br>33.66<br>33.50<br>30.69<br>31.71 | 52.34<br>49.93<br>41.32<br>33.94<br>38.61<br>35.34<br>31.09<br>30.10<br>35.90 | 67.10<br>58.60<br>61.46<br>54.74<br>52.76<br>44.04<br>40.99<br>36.96<br>42.24<br>38.41<br>37.10<br>34.07<br>36.45 | 53.14<br>48.77<br>39.04<br>30.21<br>34.30<br>29.07<br>25.20<br>23.33<br>31.36 | 75.12<br>68.15<br>69.23<br>64.16<br>62.55<br>56.47<br>47.66<br>39.69<br>46.18<br>39.03<br>30.16<br>31.27<br>31.03 | 61.76<br>65.25<br>55.89<br>48.33<br>51.90<br>48.44<br>42.86<br>40.43<br>53.48 | 83.55<br>82.98<br>81.90<br>80.84<br>79.64<br>75.59<br>70.53<br>67.82<br>72.80<br>69.55<br>67.08<br>59.85<br>67.80 |
| <b>Z</b>  | GLM-4.6<br>GLM-4.5                                                                                                                                                                                                                                   | Mixed<br>Mixed                                                                                                 | 198k<br>128k                                                 | 45.85<br>43.04                                                                     | 58.21<br>55.48                                                                                                    | 45.64<br>43.05                                                                | 56.50<br>53.57                                                                                                    | 46.07<br>43.02                                                               | 59.92<br>57.39                                                                                                    | 37.30<br>35.06                                                       | 38.88<br>37.94                                                                                                    | 43.07<br>40.21                                                                | 48.92<br>47.38                                                                                                    | 40.05<br>36.92                                                                | 60.95<br>55.13                                                                                                    | 58.82<br>55.68                                                                | 79.78<br>76.55                                                                                                    |
| <u>K</u>  | Kimi-K2-Instruct-0905                                                                                                                                                                                                                                | Instruct                                                                                                       | 256k                                                         | 50.09                                                                              | 55.53                                                                                                             | 49.90                                                                         | 56.96                                                                                                             | 50.29                                                                        | 54.10                                                                                                             | 39.61                                                                | 38.25                                                                                                             | 43.43                                                                         | 43.75                                                                                                             | 49.05                                                                         | 57.33                                                                                                             | 64.92                                                                         | 77.29                                                                                                             |
| A)][h     | MiniMax-M2<br>MiniMax-Text-01                                                                                                                                                                                                                        | Thinking<br>Instruct                                                                                           | 192k<br>4M                                                   | 41.14                                                                              | 53.21<br>45.00                                                                                                    | 40.21                                                                         | 52.87<br>44.17                                                                                                    | 42.06                                                                        | 53.55<br>45.82                                                                                                    | 33.57                                                                | 34.98<br>33.78                                                                                                    | 38.67                                                                         | 42.58<br>38.02                                                                                                    | 38.23                                                                         | 59.92<br>40.82                                                                                                    | 51.26                                                                         | 72.20<br>61.92                                                                                                    |
| м         | Ministral-3-14B-Instruct-2512<br>Ministral-3-8B-Instruct-2512<br>Ministral-3-3B-Instruct-2512<br>Magistral-Small-2509<br>Mistral-Small-3,2-24B-Instruct-2506<br>Mistral-Large-Instruct-2411<br>Ministral-8B-Instruct-2410                            | Instruct Instruct Instruct Thinking Instruct Instruct Instruct                                                 | 256k<br>256k<br>256k<br>128k<br>128k<br>128k<br>128k         | 40.14<br>37.80<br>30.18<br>-<br>37.32<br>31.69<br>17.56                            | 45.80<br>44.46<br>34.54<br>38.40<br>39.87<br>36.25<br>14.43                                                       | 39.71<br>36.61<br>27.75<br>-<br>39.23<br>33.10<br>18.65                       | 47.75<br>46.17<br>36.42<br>40.40<br>42.56<br>39.14<br>16.53                                                       | 40.56<br>39.00<br>32.61<br>-<br>35.41<br>30.28<br>16.47                      | 43.85<br>42.75<br>32.66<br>36.40<br>37.18<br>33.36<br>12.33                                                       | 33.85<br>31.88<br>25.97<br>-<br>31.79<br>27.39<br>17.83              | 31.66<br>31.86<br>26.70<br>30.52<br>29.77<br>28.65<br>15.06                                                       | 35.04<br>32.11<br>28.57<br>-<br>33.73<br>29.88<br>18.61                       | 37.48<br>34.99<br>30.23<br>32.92<br>31.76<br>29.42<br>13.98                                                       | 34.02<br>31.73<br>23.80<br>-<br>27.42<br>23.42<br>12.26                       | 39.35<br>35.26<br>25.60<br>29.44<br>27.74<br>25.62<br>9.89                                                        | 52.60<br>50.27<br>38.81<br>-<br>50.45<br>41.66<br>19.86                       | 67.56<br>67.14<br>49.65<br>54.25<br>61.22<br>53.65<br>16.84                                                       |
| <b>∞</b>  | Llama-3.1-405B-Instruct<br>Llama-3.3-70B-Instruct<br>Llama-3.1-70B-Instruct<br>Llama-3.1-8B-Instruct<br>Llama-3.2-3B-Instruct                                                                                                                        | Instruct Instruct Instruct Instruct Instruct Instruct                                                          | 128k<br>128k<br>128k<br>128k<br>128k<br>128k                 | 40.07<br>31.89<br>31.53<br>21.09<br>15.71                                          | 40.66<br>33.69<br>32.12<br>20.06<br>12.58                                                                         | 42.03<br>35.12<br>35.10<br>24.28<br>20.90                                     | 44.46<br>39.15<br>36.85<br>25.40<br>16.45                                                                         | 38.11<br>28.66<br>27.96<br>17.91<br>10.51                                    | 36.86<br>28.23<br>27.40<br>14.71<br>8.71                                                                          | 33.44<br>26.53<br>26.22<br>21.00<br>16.63                            | 29.81<br>24.32<br>23.93<br>19.68<br>15.57                                                                         | 35.51<br>29.07<br>28.86<br>21.22<br>15.01                                     | 34.09<br>28.59<br>28.04<br>17.99<br>10.48                                                                         | 29.07<br>22.61<br>21.46<br>13.82<br>10.37                                     | 29.22<br>22.61<br>21.44<br>12.32<br>7.17                                                                          | 55.45<br>44.04<br>44.02<br>25.47<br>18.49                                     | 61.36<br>51.94<br>48.46<br>26.28<br>14.35                                                                         |

Table 3. General performance on LongBench Pro. Gray-shaded cells represent thinking scores. The best three performance results are highlighted using red  $(1^{st})$ , green  $(2^{nd})$ , and blue  $(3^{rd})$  font colors, respectively.

deep reasoning abilities, which partially mitigate the impact of language differences, enabling more stable and balanced performance in multilingual long-context tasks. This trend also suggests that future LLMs are likely to further reduce language-induced performance gaps, achieving genuine cross-lingual consistency and universality.

(4) Extreme Difficulty Reveals the True Gap in Long-Context Capabilities. The performance gap between open-source and closed-source models is minimal on Easy samples. For example, GPT-5 achieves 85.23, while DeepSeek-V3.2 achieves 85.02. However, the gap widens dramatically on the Extreme samples. For instance, Gemini-2.5-Pro scores 50.77, GPT-5 scores 48.37, DeepSeek-V3.2 scores 44.27, and Qwen3-235B-A22B-Thinking-2507 scores 43.39. Furthermore, the performance gains brought by "thinking" exhibit significant diminishing returns across tasks of dif-

ferent difficulty levels: the gains on Easy samples are much larger than those on Extreme samples. For example, after enabling thinking, Claude-4-Sonnet's score rises from 68.42 to 83.78 on Easy samples (+15.36), but only from 42.92 to 47.05 on Extreme samples (+4.13). Similarly, Gemini-2.5-Flash improves from 66.55 to 79.82 on Easy samples (+13.27), but only from 44.26 to 47.39 on Extreme samples (+3.13). These results indicate that the Extreme samples in LongBench Pro not only test a single capability of the models but also evaluate their combined abilities in long-context memory, integration, and reasoning. Current models still have considerable room for improvement on tasks of extreme difficulty.

(5) The "Thinking" Paradigm Becomes a Key Breakthrough for Long-Context Performance. Almost all models benefit from "thinking". For example, Gemini-2.5-Flash

![](_page_8_Figure_0.jpeg)

Figure 5. Trends in Best-of-N metrics.

![](_page_8_Figure_2.jpeg)

Figure 6. Trends in Pass@N metrics.

improves its score from 55.92 to 67.41 when thinking is enabled; Qwen3-235B-A22B-Instruct-2507 increases its score from 52.51 to 63.77 after performing thinking. Notably, Qwen3-4B with thinking enabled (40.82) even surpasses the non-thinking performance of Qwen3-32B (40.28), bridging the gap between different model sizes. These results indicate that long-context tasks involve cross-paragraph connections, which simple non-thinking modes easily miss, thereby limiting performance. Enabling the model to perform thinking becomes key to improving information retrieval and aggregation in long-context scenarios.

**(6) The Gap Between "Native Thinking" and "Prompted Thinking".** Not all models benefit from "thinking." Thinking and mixed-thinking models internalize the thinking process, and the incorporation of thinking yields significant performance improvements for these models. For example, Claude-4-Sonnet achieves a gain of  $13.80 (56.07 \rightarrow 69.87)$ ; DeepSeek-V3.2 achieves a gain of  $16.15 (51.67 \rightarrow 67.82)$ ; and Qwen3-235B-A22B-Thinking-2507, compared with Qwen3-235B-A22B-Instruct-2507 with thinking enabled, still improves by  $3.20 (63.77 \rightarrow 66.97)$ . Compared with them, traditional instruct models obtain very limited gains even when they are forced to think, and some smaller models

even exhibit performance degradation caused by thinking. For instance, Llama-3.1-405B-Instruct yields only a 0.59 improvement (40.07  $\rightarrow$  40.66); Gemma-3-12B-It shows a 0.24 decrease (32.16  $\rightarrow$  31.92); and Llama-3.1-8B-Instruct suffers a 1.03 decrease (21.09  $\rightarrow$  20.06). These results demonstrate that models without thinking training may fail to effectively leverage test-time compute. "Thinking" is not merely a form of prompt engineering, but a fundamental post-training paradigm shift. Compared to prompted thinking, native thinking ability is the key to improving long-context performance.

(7) Mixed-Thinking Models Exhibit Pareto Optimality. Mixed-thinking models achieve Pareto-optimal performance between instruct models that cannot perform deep reasoning and thinking models that cannot respond quickly. They maintain efficient and robust baseline capability when thinking is disabled, and can approach or even surpass thinking models when thinking is enabled. For example, Gemini-2.5-Flash approaches the performance of Gemini-2.5-Pro in thinking mode, and DeepSeek-V3.2 significantly outperforms DeepSeek-R1 in thinking mode. This phenomenon indicates that mixed-thinking, which dynamically chooses between fast output and deep reasoning based on user needs, is highly likely to become the most competitive paradigm for future long-context models.

#### **5.3. Upper-Bound Performance**

To filter out metric deviations caused by generation instability, we report the trend of Best-of-N metrics for mainstream LLMs, as shown in Figure 5. All models show a clear monotonic increase. Gemini-2.5-Pro and GPT-5 exhibit strong stability, as their single-shot performance is already high and the marginal gains from additional inference converge quickly. In contrast, models such as Qwen3-235B-A22B-Thinking-2507 exhibit markedly high potential: increasing N significantly corrects the bias in their initial reasoning, enabling them to make a leap toward the top tier.

LongBench Pro adopts a fine-grained scoring mechanism, granting partial credit for partially correct answers. This mechanism enables a more nuanced differentiation of performance across models; however, it fails to fully reflect the intrinsic difficulty of LongBench Pro. Therefore, we further report the trend of Pass@N metrics for mainstream LLMs on LongBench Pro, as shown in Figure 6.

The above upper-bound evaluation substantiates the effectiveness and soundness of LongBench Pro. Under N = 3, LongBench Pro still maintains the following properties:

(1) Discriminability: There remains a clear performance gap across model tiers, indicating that LongBench Pro assesses deep long-context understanding rather than surface-level tricks that can be compensated for by probabilistic guessing.

(2) Difficulty: Even the strongest model, Gemini-2.5-Pro, achieves only a Pass@3 of 10.68 on the Extreme samples. After excluding factors related to model instability, the benchmark still exhibits a substantial headroom, sufficient to support the evaluation of more capable models.

### 5.4. Comparison Across Length Dimension

Figure 7 presents the performance levels of mainstream LLMs across different sample lengths. The results show that most models exhibit a declining trend in performance as sample length increases. For these models, sample length remains a significant factor affecting long-context performance. However, Gemini-2.5-Pro breaks this pattern, demonstrating remarkable length insensitivity: its score at 256k (71.77) is very close to its score at 8k (74.50). This phenomenon indicates that, within the range of 256k in length, for current state-of-the-art long-context models, merely increasing sample length to stress-test model performance has reached a point of saturation. The current bottleneck in long-context performance does not lie in the model's ability to "read" 256k tokens, but in its capacity to handle long-range dependencies and complex logical relationships. The focus of long-context evaluation shifts from "how much can it read" to "how deeply can it comprehend," making the enhancement of models' deep comprehension ability in long contexts a major ongoing challenge.

### 5.5. Comparison Across Task Dimension

Figure 8 shows the performance levels of mainstream LLMs across different tasks. We make the following observations:

- (1) There is a significant gap between retrieval ability and aggregation ability. Although most models demonstrate very high proficiency in basic information retrieval (T1) and sequence reconstruction (T2) (with average scores above 80), their performance drops sharply on semantic aggregation (T6, average score 57.72), which requires complex information integration. This contrast indicates that current models, although capable of precisely performing "needle-in-a-haystack" localizations, still face significant challenges when it comes to semantically aggregating and integrating dispersed information across long contexts.
- (2) There is an imbalance in forward and backward inference between evidence and outcomes. Most models perform relatively well on evidence retrieval (T5, average score 63.47), but their performance is comparatively lower on question answering (T3) and summarization (T4), tasks that require deriving results from detailed document information (average scores below 55). This indicates that current models are more robust in backward alignment from outcomes to evidence, but forward generation from evidence to outcomes is more susceptible to document complexity and long-context effects.

![](_page_10_Figure_0.jpeg)

Figure 7. Performance across different sample lengths.

![](_page_10_Figure_2.jpeg)

Figure 8. Performance across different tasks.

(3) Logical reasoning and consistency maintenance constitute current high-level bottlenecks. Most models perform moderately to poorly on logical reasoning (T8–T10, average scores around 60), whereas some models enhanced with reasoning training perform exceptionally well on these

tasks, highlighting the significant effect of targeted training on enhancing task-specific capabilities. In contrast, models generally score low on consistency maintenance (T7 and T11, average score below 49), exposing inherent limitations in sustaining global states over very long sequences.

![](_page_11_Figure_0.jpeg)

Figure 9. Performance across different context requirements.

![](_page_11_Figure_2.jpeg)

Figure 10. Comparison of sample construction strategies.

# 5.6. Comparison Across Context Requirement Dimension

Figure 9 presents the performance differences of mainstream LLMs under varying context requirements. We observe a clear and widespread performance stratification: all models perform substantially better on Partial tasks, which emphasize localization and retrieval, than on Full tasks, which require integration and reasoning. Specifically, when questions shift from relying on a single local segment to requiring the integration of multiple segments across the entire document, the models exhibit a performance drop of 7.32 to 10.84 points. This result indicates that although current models demonstrate relatively mature capabilities in accurately retrieving local information from long contexts, they

still show notable limitations in associating dispersed information across segments and performing holistic reasoning.

# 5.7. Comparison Across Construction Strategies

We uniformly sample 50 documents based on secondary task categories, lengths, and languages, and construct samples using three different strategies to evaluate the effectiveness of our strategy. The compared sample-construction strategies include human-only, model-only, and human-model collaboration (ours). Figure 10 (a) presents the time cost of constructing samples of different lengths under these strategies. The time required by the human-only strategy increases exponentially with sample length, while the model-only strategy remains consistently low. The human-model

collaborative strategy falls between the other two and increases slowly with sample length, highlighting the substantial efficiency gains enabled by model involvement in the sample-construction process.

We also evaluate the quality of samples constructed using three strategies. Specifically, we design a sample quality assessment framework comprising five dimensions. According to this framework, each sample is scored by three experts on three levels, with scores ranging from 0 to 1. The results show that samples constructed via the human-model collaborative strategy achieve the highest average quality score  $(0.9609 \pm 0.0415)$ , outperforming those constructed by the human-only strategy (0.9484  $\pm$  0.0450) and the model-only strategy (0.8964  $\pm$  0.0536). The Fleiss' Kappa among the three experts is 0.76, indicating high agreement. Figure 10 (b) shows the quality of samples constructed by different strategies across various dimensions. Thanks to our clear and explicit definitions of task and context requirements, as well as fully authentic natural text, all three sample construction strategies achieve consistently high scores in task alignment, context requirement alignment, and authenticity. On the difficulty dimension, the model-only strategy exhibits a relatively low level due to the absence of human sample filtering. In terms of answer correctness, the human-only strategy, without model assistance, tends to have omissions in answer components, resulting in the lowest correctness. The model-only strategy, without human error correction, tends to produce erroneous hallucinated answers, leading to slightly lower correctness. The human-model collaboration strategy effectively compensates for the weaknesses of both strategies, achieving the highest answer correctness.

# 6. Related Works

Long-context evaluation measures whether LLMs can reliably retrieve, integrate, and reason over evidence that is sparse and distributed across lengthy documents, and it must address confounders such as positional effects and the gap between advertised context length and effective reasoning length (Liu et al., 2023). Existing benchmarks span a spectrum from controlled probes to realistic, human-verified tasks: synthetic benchmarks such as RULER (Hsieh et al., 2024), MRCR (OpenAI, 2025), and GSM-∞ (Zhou et al., 2025) provide scalable diagnostics of usable context size, while long-document NLP suites (e.g., SCROLLS/ZeroSCROLLS (Shaham et al., 2022; 2023)) and standardized evaluation protocols (e.g., L-Eval and its length-adaptable extension Ada-L-Eval (An et al., 2023; Wang et al., 2024)) broaden task coverage and comparability. More realistic benchmark datasets further emphasize natural documents and deeper reasoning, including Long-Bench (Bai et al., 2023) and LongBench v2 (Bai et al., 2025), as well as mixed synthetic/natural stress tests such as

∞Bench (Zhang et al., 2024) and multilingual evaluation such as CLongEval (Qiu et al., 2024); methodology-focused work like HELMET (Yen et al., 2024) argues for systematic designs and analyses to avoid over-optimistic conclusions. In contrast to prior benchmarks that are primarily synthetic probes, protocol suites, or narrower in language/task coverage, LongBench Pro is built on fully natural long documents with bilingual (EN/ZH) coverage and diverse tasks/metrics, and it supports fine-grained analysis via multi-dimensional categorization (context requirement, length, difficulty) enabled by a scalable human-model collaborative construction pipeline.

#### 7. Conclusion and Future Work

In this work, we introduce LongBench Pro, a realistic and comprehensive bilingual benchmark for long-context evaluation. We evaluate 46 representative long-context large language models (LLMs) on LongBench Pro and provide analyses across task, length, context requirement, difficulty, and language settings.

However, as task length and complexity continue to grow, even human-model collaborative construction can face a tension between verification accuracy and production efficiency. We are exploring a recursive critique scheme ("Critique-of-Critique"), which shares some similar ideas with the metaverification design in DeepSeekMath-V2 (Shao et al., 2025), to recursively and progressively decompose verification into easier subproblems that are tractable for human annotators. We have achieved preliminary results in this direction and look forward to sharing more findings in the near future.

# References

AI, M. Introducing meta llama 3.1: Our most capable models to date. https://ai.meta.com/blog/meta-llama-3-1/, July 2024.

AI, M. Introducing mistral 3. https://mistral.ai/news/mistral-3, December 2025.

An, C., Gong, S., Zhong, M., Zhao, X., Li, M., Zhang, J., Kong, L., and Qiu, X. L-Eval: Instituting standardized evaluation for long context language models. arXiv preprint arXiv:2307.11088, 2023. URL https://arxiv.org/abs/2307.11088.

Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4, May 2025.

Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., et al. Longbench: A bilingual, multitask benchmark for long context understanding. *arXiv preprint* 

- *arXiv:2308.14508*, 2023. URL https://arxiv.org/abs/2308.14508.
- Bai, Y., Tu, S., Zhang, J., Peng, H., Wang, X., Lv, X., Cao, S., Xu, J., Hou, L., Dong, Y., et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 3639–3664, 2025.
- Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E., et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang, H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.
- Hsieh, C.-P., Sun, S., Kriman, S., Acharya, S., Rekesh, D.,

- Jia, F., Zhang, Y., and Ginsburg, B. Ruler: What's the real context size of your long-context language models? *arXiv preprint arXiv:2404.06654*, 2024.
- Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. *arXiv preprint arXiv:2412.19437*, 2024.
- Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu,
  B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3.
  2: Pushing the frontier of open large language models.
  arXiv preprint arXiv:2512.02556, 2025.
- Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. *arXiv* preprint arXiv:2307.03172, 2023. URL https://arxiv.org/abs/2307.03172.
- OpenAI. Introducing gpt-5, August 2025. URL https://openai.com/index/introducing-gpt-5/.
- OpenAI. Openai mrcr: Long context multiple needle in a haystack benchmark. https://huggingface.co/datasets/openai/mrcr, 2025. URL https://huggingface.co/datasets/openai/mrcr. Dataset, MIT License, accessed 2025-12-19.
- OpenAI, :, Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., Barak, B., Bennett, A., Bertao, T., Brett, N., Brevdo, E., Brockman, G., Bubeck, S., Chang, C., Chen, K., Chen, M., Cheung, E., Clark, A., Cook, D., Dukhan, M., Dvorak, C., Fives, K., Fomenko, V., Garipov, T., Georgiev, K., Glaese, M., Gogineni, T., Goucher, A., Gross, L., Guzman, K. G., Hallman, J., Hehir, J., Heidecke, J., Helyar, A., Hu, H., Huet, R., Huh, J., Jain, S., Johnson, Z., Koch, C., Kofman, I., Kundel, D., Kwon, J., Kyrylov, V., Le, E. Y., Leclerc, G., Lennon, J. P., Lessans, S., Lezcano-Casado, M., Li, Y., Li, Z., Lin, J., Liss, J., Lily, Liu, Liu, J., Lu, K., Lu, C., Martinovic, Z., McCallum, L., McGrath, J., McKinney, S., McLaughlin, A., Mei, S., Mostovoy, S., Mu, T., Myles, G., Neitz, A., Nichol, A., Pachocki, J., Paino, A., Palmie, D., Pantuliano, A., Parascandolo, G., Park, J., Pathak, L., Paz, C., Peran, L., Pimenov, D., Pokrass, M., Proehl, E., Qiu, H., Raila, G., Raso, F., Ren, H., Richardson, K., Robinson, D., Rotsted, B., Salman, H., Sanjeev, S., Schwarzer, M., Sculley, D., Sikchi, H., Simon, K., Singhal, K., Song, Y., Stuckey, D., Sun, Z., Tillet, P., Toizer, S., Tsimpourlas, F., Vyas, N., Wallace, E., Wang, X., Wang, M., Watkins, O., Weil, K., Wendling, A., Whinnery, K., Whitney, C., Wong, H., Yang, L., Yang, Y., Yasunaga, M., Ying, K., Zaremba, W., Zhan, W., Zhang, C., Zhang, B., Zhang, E., and Zhao, S. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925.

- Qiu, Z., Li, J., Huang, S., Jiao, X., Zhong, W., and King, I. Clongeval: A chinese benchmark for evaluating long-context large language models. *arXiv* preprint *arXiv*:2403.03514, 2024.
- Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.
- Shaham, U., Segal, E., Ivgi, M., Efrat, A., Yoran, O., Haviv, A., Gupta, A., Xiong, W., Geva, M., Berant, J., and Levy, O. Scrolls: Standardized comparison over long language sequences. *arXiv preprint arXiv:2201.03533*, 2022. URL https://arxiv.org/abs/2201.03533.
- Shaham, U., Ivgi, M., Efrat, A., Berant, J., and Levy, O. Zeroscrolls: A zero-shot benchmark for long text understanding. *arXiv preprint arXiv:2305.14196*, 2023. URL https://arxiv.org/abs/2305.14196.
- Shao, Z., Luo, Y., Lu, C., Ren, Z., Hu, J., Ye, T., Gou, Z., Ma, S., and Zhang, X. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. *arXiv* preprint *arXiv*:2511.22570, 2025.
- Team, G., Kamath, A., Ferret, J., Pathak, S., Vieillard, N., Merhej, R., Perrin, S., Matejovicova, T., Ramé, A., Rivière, M., et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025a.
- Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. *arXiv preprint arXiv:2507.20534*, 2025b.
- Team, M. Minimax m2 & agent: Ingenious in simplicity. https://minimaxi.com/news/minimax-m2, December 2025.
- Wang, C., Duan, H., Zhang, S., Lin, D., and Chen, K. Ada-L-Eval: Evaluating long-context LLMs with length-adaptable benchmarks. *arXiv preprint arXiv:2404.06480*, 2024. URL https://arxiv.org/abs/2404.06480.
- Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. *Advances in neural information processing systems*, 35:24824–24837, 2022.

- Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. *arXiv preprint arXiv:2505.09388*, 2025.
- Yen, H., Gao, T., Hou, M., Ding, K., Fleischer, D., Izsak, P., Wasserblat, M., and Chen, D. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694, 2024.
- (Z.ai), Z. A. Glm-4.6: Advanced agentic, reasoning and coding capabilities. https://z.ai/blog/glm-4.6, December 2025.
- Zhang, X., Chen, Y., Hu, S., Xu, Z., Chen, J., Hao, M., Han, X., Thai, Z., Wang, S., Liu, Z., et al. ∞ bench: Extending long context evaluation beyond 100k tokens. In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 15262–15277, 2024.
- Zhou, Y., Liu, H., Chen, Z., Tian, Y., and Chen, B. Gsm-infinite: How do your llms behave over infinitely increasing context length and reasoning complexity? *arXiv* preprint arXiv:2502.05252, 2025.

#### A. Task Definitions

### T1 Retrieval & Ranking

Retrieve content and rank most relevant first.

# T1.1 Global Cohesive Retrieval

Retrieve full text and reorganize. **Context Requirement:** Full

Metric: NDCG@k

**Example:** 

Retrieve all reviews that received 50 or more votes, and output the reviewerIDs in ascending order of vote value. Output the "[Answer]" identifier first, and then output reviewerIDs line by line, without any additional content.

# Output example:

[Answer]

A2AV7Q95QGPTO0

A3NM0RAYSL6PA8

A1C9C1OOOB94RT

A1B80MVU7ZODF9

# **T1.2 Key-Snippet Retrieval**

Locate target fragment in specified paragraph.

Context Requirement: Partial

Metric: NDCG@k

**Example:** 

From the subset of articles written by "Dr. Seuss" (for which an artificial ID field has been added): Retrieve articles in the category "Life" and sort them by popularity from lowest to highest. Output the "[Answer]" identifier first, and then output the sorted reviewerID line by line, without any additional content.

# Output example:

[Answer]

ID1

ID2

ID3 ID4

# **T2** Sequencing & Structure Reconstruction

Restore timeline or logical order.

#### **T2.1** Global Timeline Reconstruction

Sort unordered events in the whole text.

**Context Requirement:** Full **Metric:** Pairwise Accuracy

**Example:** 

This article is divided into nine parts, each part is preceded by "Part x", and the order is shuffled. Please sort these nine parts in the correct order of events. Output the "[Answer]" identifier first, and then output correct sequence of part numbers line by line, without any additional content.

# Output example:

[Answer]

Part 5

Part 1

Part 9

Part 3

Part 7

Part 2

Part 4 Part 8

Part 6

# **T2.2** Local Causal Chain Sorting

Sort content in a specific paragraph. **Context Requirement:** Partial Metric: Pairwise Accuracy

**Example:** 

The "SELECTED PARAGRAPH" of the article is out of order. Please reorder this paragraph according to the original text and options. Output the "[Answer]" identifier first, and then output the sorted option letters line by line, without any additional content.

# Output example:

[Answer]

В C

D

Е

# T3 Evidence-Grounded QA

Answer fact/reasoning questions based on evidence.

# T3.1 Multi-Doc Integration QA

Use multi-hop information to answer questions.

**Context Requirement:** Full

Metric: Accuracy

**Example:** 

On their way to Egloshayle, from the conversation of the passengers on the coach, it can be inferred that the characteristics of Mr. Alan Torrington do not include the following? Output the "[Answer]" identifier first, and then output the answer option letter (A/B/C/D), without any additional content.

A. He was rumored by some to be a reclusive person, with a habit of wandering near the rocks at night, and was even falsely labeled as a "mysterious wizard." B. His personality was in stark contrast to that of his brother Oscar Torrington, and he was considered to be a miser, with rumors that he hid his wealth in secret places. C. He was in poor health, and his trip to Switzerland with his brother was aimed at improving his health, as he usually required someone to take care of his daily life.

D. He had once placed a light in the tower window to mislead ships into running aground in order to acquire their cargo, becoming a figure associated with the local history of "wrecking."

#### Output example:

[Answer]

C

# T3.2 Single-Hop Fact QA

Answer questions based on local paragraphs.

**Context Requirement:** Partial

Metric: Accuracy

**Example:** 

What are all the topic categories listed in the paper "Delta Activities: A Representation for Finetuned Large Language Models" (arXiv: 2509.04442)? Output the "[Answer]" identifier first, and then output the answer option letter (A/B/C/D), without any additional content.

- A. Machine Learning (cs.LG); Computation and Language (cs.CL); Artificial Intelligence (cs.AI)
- B. Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Information Retrieval (cs.IR)
- C. Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Information Retrieval (cs.IR)
- D. Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Sound (cs.SD)

#### Output example:

[Answer]

Ā

# **T4 Summarization & Synthesis**

Generate abstract summary under given constraints.

# **T4.1 Global-Coverage Constrained Summary**

Generate summary of full text. **Context Requirement:** Full

**Metric:** 0.5\*max(SemSim)+0.5\*max(ROUGE-L)

**Example:** 

The above are multiple chapters of a novel, and based on the given text, summarize and generalize the entire content, with a requirement of no more than 100 words. Output the "[Answer]" identifier first, and then output the summary, without any additional content.

### Output example:

[Answer]

your summary

# **T4.2 Query-Focused Summary**

Generate summary of specific subtopic.

Context Requirement: Partial

**Metric:** 0.5\*max(SemSim)+0.5\*max(ROUGE-L)

#### **Example:**

Summarize the plot of the absurd victory celebration only in the first section ( $\leq 200$  words). Output the "[Answer]" identifier first, and then output the summary, without any additional content.

# Output example:

[Answer]

your summary

### **T5** Attribution & Citation Alignment

Bind correct sources to generated text.

# **T5.1 Full-Sentence Citation Alignment**

Citation alignment for all sentences. **Context Requirement:** Full

Metric: F1 Example:

You will be provided with a summary sentence. Your task is to identify the original Part number(s) from the provided text that fully support this sentence. Output the "[Answer]" identifier first, and then output the cited content (format as "Part xx") line by line, without any additional content.

### Generated Summary:

After taking a leave of absence post the 1984 Macintosh launch (to recharge and avoid supervisor Bob Belleville), Andy Hertzfeld clashed with Jobs first. Over a denied Mac team bonus (which Jobs initially blamed on Belleville. Before relenting, leaving Hertzfeld upset) and later when Jobs dismissed his. Concerns about the Mac software team's low morale and Burrell's frustration. As out of touch, insisting the team was doing well.

#### Output example:

[Answer]

Part 1

Part 12

Part 15

# **T5.2** Key-Statement Citation Alignment

Citation alignment for specified sentences.

**Context Requirement:** Partial

Metric: F1 Example:

You will see a generated 3-sentence summary. Identify and cite the original source location for Sentence 2 only, using the paragraph identifiers S1–S120 from the source text. Output the "[Answer]" identifier first, and then output the paragraph identifier(s) line by line, without any additional content.

#### Generated summary:

Sentence 1: During Danielle Mitterrand's factory

tour, she pressed on labor conditions—overtime and vacation—while Jobs, increasingly annoyed, emphasized automation and just-in-time processes.

Sentence 2: Jobs believed that Sculley did not understand business and was not suitable to manage the company, and designed a plan to remove him.

Sentence 3: Driving back toward Cupertino, Jobs was ticketed for speeding near 100 mph and, unfazed, accelerated again—evidence, Rossmann said, that Jobs believed normal rules didn't apply to him.

### Output example:

[Answer]

S1

S2.

#### **T6 Aggregation & Clustering**

Cluster and output statistics/examples/sort.

# **T6.1 Large-Scale Document Clustering**

Return all category proportions. Context Requirement: Full

Metric: SubEM Example:

Cluster the 8 documents (IDs: A–H) by research methodology into exactly three clusters: Randomized trial/protocol; Quantitative observational; Qualitative. Output the "[Answer]" identifier first, and then output the three clusters in the format "ClusterName Proportion% (rounded to two decimal places)" line by line, without any additional content.

#### Output example:

[Answer]

Randomized trial/protocol 25.00% Quantitative observational 25.00%

Qualitative 50.00%

# **T6.2 Targeted Subset Cluster Identification**

Return query category instances. **Context Requirement:** Partial

Metric: F1 Example:

Find the first three software reviews in the Software Reviews category. Output the "[Answer]" identifier first, and then output the answer reviewer IDs that meet the conditions line by line, without any additional content.

# Output example:

[Answer]

A22V1MD93T2FW9 ACJT8MUCOLRFO A38NELQT98S4H8

#### **T6.3 Global Frequency Analysis**

Count and sort global word frequency.

**Context Requirement:** Full **Metric:** Pairwise Accuracy

**Example:** 

Sort the given terms in descending order by the number of times they appear in the above text. Output the "[Answer]" identifier first, and then output the sorted terms line by line, without any additional content.

Given terms: would, this, that, went, have

# Output example:

[Answer] would this

that

went have

# **T7 Consistency & Compliance Checking**

Detect and locate contradictions/violations.

### **T7.1 Global Conflict & Inconsistency Localization**

Locate contradictory segments in the full text.

**Context Requirement:** Full

Metric: F1 Example:

Compare the inconsistent articles between clauses A and B in two documents. Output the "[Answer]" identifier first, and then output all inconsistent articles line by line in the format "A[articleID] B[articleID]", without any additional content

# Output example:

[Answer]

A1 B2

A2 B3

### **T7.2** Targeted Rule or Condition Violation Detection

Locate content that violates specific rules.

**Context Requirement:** Partial

Metric: F1 Example:

Read and list which terms of Enzuzo, Inc. Privacy Policy were violated in the first article. Output the "[Answer]" identifier first, and then output all the violated term IDs line by line, without any additional content.

# Output example:

[Answer]

5

6

# T7.3 Comprehensive Error & Anomaly Sweep

Locate spelling errors in the full text.

Context Requirement: Full

Metric: F1

#### **Example:**

Find and list all misspelled English words in a given complete text file. Output the "[Answer]" identifier first, and then output all misspelled English words line by line, without any additional content.

Output example:

[Answer] beleive freind tommorow

T8 Structured & Numeric Reasoning Numerical calculations in structured text.

**T8.1 Structured Multi-Source Consistency Verification** 

Numerical computation in multi-source.

**Context Requirement:** Full

Metric: SubEM Example:

Above are two tables. Please use the following formula to determine if there are any issues after merging the two tables (i.e., whether the sales amount calculated using the formula for Table 1 is consistent with the sales amount in Table 2). If there are no issues, return "No Error". If there are any issues, return the corresponding Invoice IDs. Output the "[Answer]" identifier first, and then output "No Error" or output corresponding Invoice IDs line by line, without any additional content.

Formula: Sales Amount = Unit Price × Quantity + 5% Tax

Output example:

[Answer]

No Error/226 31 3081

**T8.2** Single-Source Targeted Aggregation

Query computation in single-source. **Context Requirement:** Partial

Metric: SubEM Example:

According to the table "Share repurchases for the fourth quarter of fiscal year 2023", calculate the percentage increase in the "Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs" for June compared to April and May. Output the "[Answer]" identifier first, and then output percentages (with one decimal place retained) line by line, without any additional content.

Output example:

[Answer] 5.5%

-2.2%

**T8.3** Long-Context Procedural State Tracking

Track entity state evolution.

Context Requirement: Full

Metric: F1 Example:

Find and list all misspelled English words in a given complete text file. Output the "[Answer]" identifier first, and then output all misspelled English words line by line, without any additional content.

Output example:

[Answer] beleive freind tommorow

# T9 Version & Code Diff Analysis

Compare changes in different text/code versions.

**T9.1 Dependency-Aware Multi-Version Impact Analysis** 

Track dependency changes across versions.

**Context Requirement:** Full

Metric: F1 Example:

Based on the two provided API documents for hadoopcore (version 0.20.0 and 0.21.0), identify all methods within the "org.apache.hadoop.fs.FileSystem" class that were not deprecated in version 0.20.0 but became deprecated in version 0.21.0. Use the method signature as the identifier in the format "MethodName(ParameterTypes)". If a method has no parameters, use "MethodName()". For multiple parameters, use a comma-separated list of fully qualified types. Output the "[Answer]" identifier first, and then output the identifiers line by line, without any additional content.

Output example:

[Answer]

MethodA(java.lang.String)

MethodB()

MethodC(int, java.lang.String)

# **T9.2** Localized Interface Change Detection

Detect local version differences. **Context Requirement:** Partial

Metric: F1 Example:

Identify the fields within the "Lesson" resource model that were either renamed, removed due to architectural changes, or refactored into a more complex structure when evolving from V1.0 to V2.0. List these fields using the format "Lesson Model: [V2.0 Field Name]" for renamed/refactored fields and "Lesson Model: [V1.0 Field Name]" for fields that were entirely removed. Output the "[Answer]" identifier first, and then output

fields line by line, without any additional content.

Output example:

[Answer]

Lesson Model: key Lesson Model: seq\_length

### T10 Rule Induction & In-Context Learning

Summarize rules and make decisions on new samples.

# **T10.1 Large-Scale In-Context Rule Induction**

Induce rules from the global context.

Context Requirement: Full

Metric: SubEM Example:

You are a script formatter. Based on the full range of script formatting conventions demonstrated in the provided text, reformat the following raw script snippet. Pay close attention to scene headings, character names, dialogue parenthetical actions, and transitions. Output the "[Answer]" identifier first, and then output the fully formatted script snippet, without any additional content.

Raw Snippet:

interior, lab - day

Dr. Banner (looking at monitor): the readings are stable now. Tony Stark: That's what I like to hear.

(Tony smiles)

cut to:

exterior, city rooftop - night

A figure (hooded) stands looking at the skyline.

Output example:

[Answer]

your answer

# **T10.2 Targeted Example-Based Rule Induction**

Induce rules from the targeted examples.

Context Requirement: Partial

Metric: SubEM Example:

Answer the following questions based on the content of the case in Part 1 of the provided document: UKCo is a private limited company specializing in cross-border private equity investments. In 2022, UKCo signed a Joint Investment Agreement with three EU investment institutions (A, B, C), agreeing to jointly establish a European infrastructure cooperation plan to invest in EU new energy infrastructure projects. In 2024, the European Infrastructure Cooperation Program fell into a debt crisis due to project defaults, owing 8 million euros to supplier Company D (a German company). At the same time, A, B, and C found that UKCo had misappropriated funds from the European Infrastructure Cooperation Scheme in its management, so they applied to the English

Commercial Court: Should the English court support A, B, and C's application for compulsory liquidation of the European Infrastructure Cooperation Scheme as a compulsory liquidation? Output the "[Answer]" identifier first, and then output answer. The answer must be a fixed value ("support/don't support"), and without any additional content.

Output example:

[Answer]

support

T11 Dialogue Memory & Long-Horizon Tracking

Track and respond to dialogue history.

**T11.1** Long-Range Entity & Commitment Tracking

Track entity states across the global context.

**Context Requirement:** Full

Metric: Accuracy

**Example:** 

Based on the conversation between the user and the model above, which of the following options best fits the content in the text? Output the "[Answer]" identifier first, and then output A/B/C/D/E, without any additional content.

A. About the book "Ship of Theseus", Based on the conversation, here is a simple and insightful summary of the book "S." / "Ship of Theseus": "S." is a unique, single level reading experience that casts the reader as a detective. It is a book within a book, containing.

B. About the book "Piranesi", Based on the conversation, here is a summary of the discussion about Susanna Clarke's novel "Piranesi": A signature style of the author, Susanna Clarke, She uses authentic academic citations, which lends the story a remarkable sense of authenticity and depth while also concealing clues.

C. About the movie "Lobster", Based on the conversation, here is a summary of the analysis of the movie "The Lobster": The discussion analyzes Yorgos Lanthimos's 2015 film "The Lobster," identifying its core premise-a society where single people must find a partner within a month or become an animal-as a satirical critique of modern society's pressure to be in a relationship.

D. About "Dark City", The user and model discuss the 1999 Alex Proyas film "Dark City", agreeing it is an underrated sci-fi classic often overshadowed by "The Matrix". They identify its core premise: a city perpetually trapped in night where alien "Strangers" experiment on humans by altering their memories and physical reality to understand the human soul, which they themselves lack. E. "The Ship of Theseus", The Joy of Collaboration: A central theme is that the best way to enjoy these complex works is through collaboration. The model encourages the user to solve puzzles oneself, suggesting they can

divide tasks and that debating theories will lead to deeper insights. This is supported by real-world examples of fan communities and is also applied to finding hidden details (like changing tie patterns) in the film "Predestination" ("Former Destination").

#### Output example:

[Answer]

T11.2 Short-Range Reference Resolution & State Query Resolve references and states in local context.

Context Requirement: Partial

Metric: Accuracy Example:

In the scene in the elegant dining room where Cobb and Arthur are pitching their services to Saito (pages 2-4), what specific reason does Arthur give for why a person's thoughts become vulnerable to theft in the dream state? Output the "[Answer]" identifier first, and then output the answer option letter (A/B/C/D), without any additional content.

- A. Because a person can be trained to reveal their own secrets.
- B. Because killing someone in a dream will not wake them up.
- C. Because a person's conscious defenses are lowered.
- D. Because an idea, once it takes hold, is impossible to eradicate.

#### Output example:

[Answer]

# **B.** Annotation Guidelines

# **B.1. Sample-Generation Prompt**

# **Role Definition**

You are a professional question-design specialist who:

- possesses solid capabilities in textual deconstruction and deep reading;
- has extensive cross-domain experience in exam item construction:
- demonstrates accurate command of the following domains: legal and regulatory texts, literature and creative writing, academic and professional materials, consulting and media content, history and culture, as well as general-interest texts;
- is familiar with each domain's technical terminology, logical systems, textual features, and core assessment points.

# **Task Objective**

Given an input **long context** and its **task category** (**primary task + secondary task**), you create **three questions** that strictly fulfill the task requirements (including **answers**, **design rationale**, and detailed **solution process**).

#### **Task Procedure**

#### **Step 1: Language Identification**

Determine the language of the input long text (e.g., Chinese/English). All subsequent content (questions, answers, explanations) must **fully use the same language** as the long context.

#### **Step 2: Question Construction**

# 2.1 Number of Questions

Based on the long context and the secondary task type, extract deep assessment points and generate **three questions** that satisfy the secondary task specifications.

# 2.2 Construction Requirements

- All questions must strictly satisfy the evaluation goals of the primary task, the required I/O format, and the definition of the secondary task.
- Question phrasing must follow the secondary task's example patterns.
- Answer formats must match the task examples.
- The three questions must exhibit clear differentiation, avoiding repeated formats or duplicate assessment points.

#### 2.3 Sample Requirements

(1) Identifier Rules (ensuring answer uniqueness)

To avoid ambiguity caused by overlapping textual expressions, each question must apply identifiers:

- ① Natural identifiers (preferred). Use naturally occurring, unique elements in the document as identifiers, such as titles, section names, or explicit entities.
- ② Constructed identifiers. If natural identifiers are insufficient, create identifiers for document segments, e.g., DocumentX ParagraphX IDX D11. Requirements:
- Use only letters and numbers; do not include symbols (such as "- , \_ .").
- Explicitly explain the numbering rule in the question.
- Require respondents to answer using the identifiers.
- (2) Requirements for Multiple-Choice Questions (if applicable)
- Must include distractors based on the document.
- At least four options.
- The correct option must be verifiable in the text.
- Distractors must be reasonable and sufficiently plausible.
- (3) Reference Answer (for summarization tasks only)
- Three results are required for semantic similarity evaluation; must be accurate and cover key points.
- (4) Question-Design Rationale (required for every

question)

You must explain:

- how the question satisfies the task type;
- which document information the question draws upon;
- why the answer is unique and verifiable.
- (5) Solution and Evidence (required for every question) You must provide:
- the complete, correct answer;
- detailed solution steps;
- clear citation of all textual evidence, with one-to-one correspondence.

(6) Unified Output Format

• All questions must require line-by-line answers to support automated evaluation.

# 2.4 Task Types

(1) Context Requirement Dimension

- Full: The context appears in fully scrambled order. A complete reading of the entire text is necessary to retrieve all content required for the question; missing any segment leads to an inaccurate answer.
- Partial: Since the relevant segments are contiguous and ordered, one only needs to locate the target segment; the rest is not essential.

(2) Specific Task Type

{Primary\_Task\_Definition} {Secondary\_Task\_Definition, Context\_Requirement, I/O\_Specification, and Examples}

#### Step 3: Self-Check

After generating the three questions, you must conduct a rigorous self-check (no need to output the self-check content):

- Whether the questions fully comply with the task type and secondary task requirements;
- Whether the answers are accurate, verifiable, and free from speculation;
- Whether the questions contain no ambiguity, avoid repeated assessment points, and exhibit sufficient difficulty;
- Whether the domain-specific expression is correct (e.g., legal or classical-text domains);
- Whether the language of the questions and answers matches the long context.

If any criterion fails  $\rightarrow$  You must return to Step 2 and regenerate the questions.

# Input

```
[Long Context]: {Long_Context}
[Primary Task]: {Primary_Task}
[Secondary Task]: {Secondary_Task}
```

# **Output Constraints**

1. Output three questions, and the question types must

match the secondary task requirements.

- 2. The language of the questions, answers, and explanations must match the long text.
- 3. The output format must strictly contain: the questions, the answers, the question-design rationale, and the solution process. No additional content may be included.

# **B.2. Sample Verification Criteria**

#### Task Objective

This task systematically and formally verifies question—answer samples initially generated by large models. The goal is to ensure that each sample meets the requirements in task authenticity, reasoning correctness, and challenge level.

#### **Sample Generation Process**

To balance sample authenticity with human labor costs, we adopt a human-model collaborative approach to generate test samples:

1. Multi-model Generation

For each long document, annotators generate three questions and answers using the following state-of-theart models with specific prompts. Each model output includes: the question, the answer, the design rationale, and the solving process.

- o Gemini-2.5-Pro
- o GPT-5
- o Claude-Sonnet-4
- o DeepSeek-V3.2
- o Qwen3-235B-A22B-Thinking-2507
- 2. Human Critique

Annotators critically review the model outputs and decide whether to adopt, modify, or discard each sample.

3. Secondary Review

Expert perform a final audit of generated samples; any sample failing the review returns for further revision.

# **Verification Tasks Overview**

Annotators complete the following three assessments for each model-generated sample:

# Task A: Question Compliance with Task Type and Context Requirement

Based on the model's "design rationale," annotators evaluate whether the question:

- Aligns with the task definition and context requirement.
- Relies on document content only, without using external information.
- Avoids subjective judgment or unverifiable inference.
- Remains on-topic and does not include unsupported

facts.

#### **Task B: Answer Correctness**

Based on the model's "solving process," annotators evaluate whether:

- The reasoning steps are grounded in the document.
- The logic is internally consistent.
- The final answer is verifiable from the document.
- The answer does not contain hallucinations (fabricated information).

#### Task C: Question Challenge Level

Annotators input the question into the five models listed above and assess:

- At least one model answers incorrectly  $\rightarrow$  the sample is considered challenging.
- $\bullet$  If all five models answer correctly  $\rightarrow$  the sample is not challenging.

#### **Verification Procedure**

**Step 1**: Read the model output, including the question, answer, design rationale, and solving process. Annotators fully understand the intended assessment points of the question.

**Step 2**: Execute Task A - evaluate question compliance: o If the question does not match the task type or diverges from the document  $\rightarrow$  discard the sample.

o If minor edits can fix the issue  $\rightarrow$  modify and proceed to Step 3.

o If fully compliant  $\rightarrow$  proceed to Step 3.

**Step 3**: Execute Task B - evaluate answer correctness by checking against the original document and reasoning process:

o If the answer is incorrect or the reasoning contains hallucinations/jumps  $\rightarrow$  discard the sample.

o If minor edits can fix the issue  $\rightarrow$  modify and proceed to Step 4.

o If fully correct  $\rightarrow$  proceed to Step 4.

**Step 4**: Execute Task C - verify challenge level by testing the question with five models:

o If all models answer correctly  $\rightarrow$  the question is not challenging; discard unless the question is highly valuable.

o If at least one model answers incorrectly  $\rightarrow$  the question is challenging; proceed to Step 5.

*Step 5*: Select the best sample among all candidates:

o If multiple samples pass  $\rightarrow$  select the optimal one (most aligned with definitions and highest difficulty) and add it to the sample set.

o If no sample passes  $\rightarrow$  proceed to the next document. **Step 6**: Submit for expert review:

o Annotators submit verified samples to long-context expert.

o If experts reject  $\rightarrow$  revise according to expert feedback and resubmit until approval.

#### **Precautions**

#### **Prohibited Actions**

- Do not fabricate information not present in the document.
- Do not treat model hallucinations as facts.
- Do not use subjective questions.
- Do not generate unverifiable answers.
- Avoid overly simple questions.

### **Permitted/Encouraged Actions**

- Retain questions that require deep reasoning, crossparagraph inference, or implicit clues.
- Refine model-generated questions to improve quality.
- Use the model reasoning chain to assist verification, but final judgment must be human-led.

#### **Sample Quality Checklist**

| Item                                | Check |
|-------------------------------------|-------|
| Task type: Matches design           |       |
| Document reliance: Context only     |       |
| Clarity: Question is clear          |       |
| Answer verifiable: Checkable        |       |
| Reasoning: Grounded logic           |       |
| Challenge: At least one model fails |       |
| Readability: Clear format           |       |

#### **B.3. Sample Rewriting Criteria**

### **Task Objective**

Standardize the prompts of different question types to a uniform format to improve the accuracy and success rate of subsequent automatic evaluation. Each question requires two prompts:

- **Non-Thinking Prompt** (does not output the thinking process)
- **Thinking Prompt** (outputs the thinking process first, then the answer)

Both types of prompts share the same structure, with only minor differences in Output Requirement and Output Example.

# **Unified Structure (Mandatory)**

Each prompt follows the three sections below:

- ① *First Section*: Task Description + Output Requirement (Mandatory)
- Task Description: Explain what the question asks to do.
- Output Requirement: Write differently for the Non-Thinking or Thinking version (see below).
- ② **Second Section**: Supplementary Content (Optional) If the question contains additional materials (e.g., abstract, full text, options, label list), include them in this section. Omit this section entirely if no supplementary

content exists.

③ *Third Section*: Output Example (Mandatory)

The output example must:

- Fully comply with the required answer format.
- Differ from the real answer content.
- Contain the "[Answer]" identifier.

# Differences Between the Two Prompt Types (Core Part)

The following content is used to rewrite Output Requirement and Output Example. The rest (Task Description, Supplementary Content) remains unchanged.

# 1. Non-Thinking Prompt Specification

Output Requirement (Non-Thinking Version) — Fixed Template, Cannot Change

, , ,

Output the "[Answer]" identifier first, and then output {elements} line by line, without any additional content.

. . .

Here, elements are provided by the annotator according to the question, e.g., sorted numbers, option letters, titles, years, or custom items. The format of each element must be clear, e.g., "Number Paragraph", "A", "Item1 Item2".

Output Example (Non-Thinking Version) — Fixed Format

, , ,

Output example:

[Answer] {Specific element 1} {Specific element 2}

# 2. Thinking Prompt Specification

Output Requirement (Thinking Version) — Fixed Template, Cannot Change

Think step by step. After your thinking process, output the "[Answer]" identifier, and then output {elements} line by line.

**, , ,** 

The elements are also specified according to the question.

Output Example (Thinking Version) — Fixed Format

Output example:

¡Your thinking process¿ [Answer] {Specific element 1} {Specific element 2}

, , ,

Prompt Rewriting Procedure (Mandatory for

# Annotators)

The following steps are used to rewrite the original question into the standardized format.

Step 1: Break Down Original Question Content

Extract four parts from the original sample:

- Task Description
- Output Requirement (use fixed template for Non-Thinking / Thinking)
- Supplementary Content (if any)
- Output Example (write according to requirements)

Step 2: Construct Non-Thinking Prompt

**Template** 

, , ,

Task Description Output the "[Answer]" identifier first, and then output {elements} line by line, without any additional content.

Supplementary Content (optional)

Output example:

```
[Answer]
{Specific element 1}
{Specific element 2}
```

Step 3: Construct Thinking Prompt

#### **Template**

**, , ,** 

Task Description Think step by step. After your thinking process, output the "[Answer]" identifier, and then output {elements} line by line.

Supplementary Content (optional)

Output example:

```
¡Your thinking process¿
[Answer]
{Specific element 1}
{Specific element 2}
```

# Step 4: Final Review

Check each item:

- The three-section structure is complete.
- Both Non-Thinking and Thinking formats use the fixed templates.
- Example format fully matches the answer format and differs from the real answer.
- "[Answer]" identifier exists and is correct.
- No mixing of Chinese and English punctuation.

#### **B.4.** Answer Review Criteria

# Task Objective

To ensure the accuracy and reliability of all sample answers, we perform consistent, standardized, and high-quality verification of long-context test samples. The verification process follows these principles:

- Human annotators are skilled at judging whether each component of an answer is correct  $\rightarrow$  responsible for **precision**.
- Models can generate a diverse set of possible answer components → responsible for **recall**.

Therefore, during verification, we combine human judgment with model predictions to ensure data completeness.

#### **Verification Procedure**

**Step 1**: Generate model predictions For each sample, we pre-generate a list of predicted answers from five state-of-the-art models.

Step 2: Initial annotator verification (two annotators)

- *Step A*: Check the correctness of each component in the current answer (ensures precision).
- *Step B*: Check for any missing reasonable components, referring to model predictions if necessary (ensures recall).

# **Step 3**: Result determination

- ullet If both annotators identify no issues  $\to$  mark the sample as problem-free and directly include it in the benchmark.
- ullet If any annotator identifies a potential issue  $\to$  send the sample to a long-context expert for final judgment.
- The long-context expert determines, based on the issue type, whether the sample needs reconstruction.

# **Three Types of Potential Issues**

During verification, three types of issues require identification:

### **① Document Issues**

The provided document content is incomplete, lacking the necessary information to derive the current answer.

# **2 Question Issues**

The question description is unclear, incomplete, or ambiguous, preventing a reasonable derivation of the current answer.

# **3** Answer Issues

The document and question are fine, but the existing answer is incorrect or incomplete.

#### **Annotator Operational Guidelines**

#### 1. Independent judgment

Both annotators must complete verification independently to avoid human consistency bias.

### 2. Ensure precision

Evaluate the correctness of each "component" in the answer to guarantee fine-grained assessment.

#### 3. Ensure recall

Check for any missing components; the model prediction list serves as a reference.

- o If a reasonable component appears in model predictions but is not covered in the current answer  $\to$  mark as missing.
- o If model predictions are unreasonable  $\rightarrow$  do not consider it an issue.

# 4. Flag any doubts

Any uncertainty or potential error must be flagged for review by the long-context expert.

#### **Long-Context Experts Guidelines**

The long-context expert is responsible for:

- 1. Making final judgments on disputed samples.
- 2. Determining whether a sample requires reconstruction.
- 3. If reconstruction is needed, specifying the exact reason and corresponding step (modify document, rewrite question, or revise answer).

#### **B.5. Sample Quality Evaluation Criteria**

#### Task Objective

Each long-context question is scored along the following five evaluation dimensions. Each dimension allows scores of 0, 0.5, or 1.

# **Evaluation Dimensions and Criteria**

# 1. Task Alignment

**Definition:** Measures whether the question aligns with the predefined task objectives, i.e., whether it matches the intended task design (25 secondary tasks in total).

#### Scoring Criteria:

- 1: Fully aligns with the task objective; the question format matches the task type closely.
- 0.5: Largely aligns, but the task intention is unclear or slightly deviates.
- 0: Clearly deviates from the task objective or uses an incorrect task type.

# 2. Context Requirement Alignment

**Definition:** Evaluates whether the question's dependence on document information matches the intended context requirement level. For example, Full questions require the entire document, while Partial questions require only local paragraphs.

### Scoring Criteria:

• 1: The context dependency fully matches the intended level.

- 0.5: The dependency level is generally reasonable but slightly deviates.
- 0: The dependency level is incorrect (e.g., a Partial question requires the entire document).

#### 3. Difficulty

**Definition:** Measures the challenge and discriminative power of the question, estimated based on the accuracy of five models answering the question.

#### Scoring Criteria:

- 1: Accuracy is between 0%–20%; at most one model answers correctly.
- 0.5: Accuracy is between 40%–60%; two to three models answer correctly.
- 0: Accuracy is between 80%–100%; four to five models answer correctly.

# 4. Authenticity

**Definition:** Measures whether the question reflects real user information needs and inquiry style, i.e., whether the question is natural and meaningful in practical scenarios.

#### Scoring Criteria:

- 1: Natural tone, fluent expression, reasonable question with authentic motivation.
- 0.5: Generally natural but slightly templated or artificial.
- 0: Clearly synthetic or inconsistent with real-world question style.

#### 5. Answer Correctness

**Definition:** Measures whether the reference answer is accurate, consistent with the document, and reasonably verifiable.

# Scoring Criteria:

- 1: Answer fully matches the document; facts are correct
- 0.5: Answer is mostly correct with minor deviations or imprecise wording.
- 0: Answer is clearly incorrect or inconsistent with the document.

# **Scoring Procedure**

- 1. **Independent Scoring**: Each sample is independently scored by three experts along the five dimensions.
- 2. **Score Level**s: Each dimension allows only 0, 0.5, or 1.
- 3. **Result Aggregation**: The final score for each dimension is the average of the three experts' scores.

#### 4. Notes:

- o Scoring strictly follows the criteria without subjective assumptions.
- o For ambiguous cases, the context and task type may be considered, but consistency must be maintained.
- o All scores require reasonable explanations for subsequent review.

# C. Annotator Statistics and Compensation

The construction of LongBench Pro involves a total of 63 annotators, with the statistical distribution shown in Figure 11. The 63 annotators are divided into 51 general annotators and 12 long-context experts (all long-text experts have at least one year of annotation experience and receive a two-month specialized training in long-context annotation). The ages of the annotators mainly range from 23 to 32 years, the gender ratio is balanced, and their major backgrounds are diverse. In addition, more than half of the annotators have over one year of annotation experience. Most annotators hold a bachelor's degree, and approximately 25% possess a graduate degree. Annotators are compensated at a rate of 50 RMB per hour.

# **D. Inference Parameter Settings**

Table 5 presents the detailed inference parameter settings of different models, with the parameters taken from the open-source documentation of each model.

# E. Truncation Length Setting

The actual output length of some thinking models is often greater than the default 8k setting. For example, in MiniMax-M2, 538 out of 4,500 inferences have lengths far exceeding 8k. Setting the output length to 8k would result in a large number of evaluation failures. Therefore, to reserve sufficient thinking space, we set the truncation length of DeepSeek-V3.2, GLM-4.6, and MiniMax-M2 to 120k, ensuring that the output length can be set to 32k.

In addition, there is a significant discrepancy between the effective context length and the claimed context length for some models. Table 4 presents a comparison of the non-thinking scores of GLM-4.6 (which claims a context length of 198k) under truncation lengths of 190k and 120k. When the truncation length is set to 190k, for 256k samples (whose length approaches 190k), the model outputs become unstable, leading to a sharp drop in metrics. This indicates that GLM-4.6's effective context length significantly deviates from its claimed context length. This is also one of the reasons why we set its truncation length to 120k.

| Truncation | Sample Length |       |       |       |       |       |  |  |  |  |  |
|------------|---------------|-------|-------|-------|-------|-------|--|--|--|--|--|
| Length     | 8k            | 16k   | 32k   | 64k   | 128k  | 256k  |  |  |  |  |  |
| 190k       | 53.74         | 50.76 | 51.93 | 45.60 | 37.73 | 2.55  |  |  |  |  |  |
| 120k       | 53.98         | 49.88 | 52.26 | 46.18 | 38.68 | 34.14 |  |  |  |  |  |

*Table 4.* Non-thinking scores of GLM-4.6 on samples of different lengths under varying truncation lengths.

![](_page_26_Figure_0.jpeg)

Figure 11. Distribution of annotators' role, age, gender, major, experience, and education.

|          | Model                               |          | Context Length<br>Input Output | Truncation<br>Length | Output Lo | ength<br>Thk. | Temperature<br>Non-Thk. Thk. |      |  |
|----------|-------------------------------------|----------|--------------------------------|----------------------|-----------|---------------|------------------------------|------|--|
|          | Gemini-2.5-Pro                      | Thinking | 1M 64k                         | 1M                   | -         | 32k           | _                            | 1.0  |  |
|          | Gemini-2.5-Flash                    | Mixed    | 1M 64k                         | 1M                   | 1k        | 32k           | 1.0                          | 1.0  |  |
| <b>*</b> | Gemma-3-27B-It                      | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
| •        | Gemma-3-12B-It                      | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
|          | Gemma-3-4B-It                       | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
|          | GPT-5                               | Thinking | 272k 128k                      | 272k                 | -         | 32k           | -                            | 1.0  |  |
| (S)      | GPT-40                              | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
| <b>S</b> | GPT-OSS-120B                        | Thinking | 128k                           | 120k                 | -         | 8k            | -                            | 1.0  |  |
|          | GPT-OSS-20B                         | Thinking | 128k                           | 120k                 | -         | 8k            | -                            | 1.0  |  |
| *        | Claude-4-Sonnet                     | Mixed    | 1M 64k                         | 1M                   | 1k        | 32k           | 1.0                          | 1.0  |  |
|          | Claude-3.7-Sonnet                   | Mixed    | 200k 128k                      | 200k                 | 1k        | 32k           | 1.0                          | 1.0  |  |
|          | DeepSeek-V3.2 *                     | Mixed    | 160k                           | 120k                 | 1k        | 32k           | 1.0                          | 1.0  |  |
|          | DeepSeek-V3.1                       | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |
| <b>3</b> | DeepSeek-R1-0528                    | Thinking | 128k                           | 120k                 | -         | 8k            | -                            | 0.6  |  |
|          | DeepSeek-R1                         | Thinking | 128k                           | 120k                 | -         | 8k            | -                            | 0.6  |  |
|          | DeepSeek-V3-0324                    | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.3                          | 0.3  |  |
|          | Qwen3-235B-A22B-Thinking-2507       | Thinking | 256k                           | 224k                 | -         | 32k           | -                            | 0.6  |  |
|          | Qwen3-235B-A22B-Instruct-2507       | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.7                          | 0.7  |  |
|          | Qwen3-Next-80B-A3B-Thinking         | Thinking | 256k                           | 224k                 | -         | 32k           | -                            | 0.6  |  |
|          | Qwen3-Next-80B-A3B-Instruct         | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.7                          | 0.7  |  |
|          | Qwen3-30B-A3B-Thinking-2507         | Thinking | 256k                           | 224k                 | -         | 32k           | -                            | 0.6  |  |
| _        | Qwen3-30B-A3B-Instruct-2507         | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.7                          | 0.7  |  |
| <b>1</b> | Qwen3-4B-Thinking-2507              | Thinking | 256k                           | 224k                 | -         | 32k           | -                            | 0.6  |  |
|          | Qwen3-4B-Instruct-2507              | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.7                          | 0.7  |  |
|          | Qwen3-32B                           | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 0.7                          | 0.6  |  |
|          | Qwen3-14B                           | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 0.7                          | 0.6  |  |
|          | Qwen3-8B                            | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 0.7                          | 0.6  |  |
|          | Qwen3-4B                            | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 0.7                          | 0.6  |  |
|          | Qwen2.5-72B-Instruct                | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.7                          | 0.7  |  |
| <b>Z</b> | GLM-4.6 *                           | Mixed    | 198k                           | 120k                 | 1k        | 32k           | 1.0                          | 1.0  |  |
|          | GLM-4.5                             | Mixed    | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
| <u> </u> | Kimi-K2-Instruct-0905               | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.6                          | 0.6  |  |
| սիի      | MiniMax-M2 *                        | Thinking | 192k                           | 120k                 | -         | 32k           | -                            | 1.0  |  |
| -000.    | MiniMax-Text-01                     | Instruct | 4M                             | 1M                   | 1k        | 32k           | 1.0                          | 1.0  |  |
|          | Ministral-3-14B-Instruct-2512       | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.1                          | 0.1  |  |
|          | Ministral-3-8B-Instruct-2512        | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.1                          | 0.1  |  |
|          | Ministral-3-3B-Instruct-2512        | Instruct | 256k                           | 224k                 | 1k        | 32k           | 0.1                          | 0.1  |  |
| 264      | Magistral-Small-2509                | Thinking | 128k                           | 120k                 | -         | 8k            | -                            | 0.7  |  |
|          | Mistral-Small-3.2-24B-Instruct-2506 | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.15                         | 0.15 |  |
|          | Mistral-Large-Instruct-2411         | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
|          | Ministral-8B-Instruct-2410          | Instruct | 128k                           | 120k                 | 1k        | 8k            | 1.0                          | 1.0  |  |
|          | Llama-3.1-405B-Instruct             | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |
| ~~       | Llama-3.3-70B-Instruct              | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |
| $\infty$ | Llama-3.1-70B-Instruct              | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |
|          | Llama-3.1-8B-Instruct               | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |
|          | Llama-3.2-3B-Instruct               | Instruct | 128k                           | 120k                 | 1k        | 8k            | 0.6                          | 0.6  |  |

*Table 5.* Detailed inference parameter settings. **Non-Thk.** denotes Non-Thinking, and **Thk.** denotes Thinking. **Length** is uniformly measured by the number of tokens. \*: Although these models support longer context lengths, we set their truncation length uniformly to 120k and the thinking output length to 32k to enable more thorough thinking. Appendix E provides specific notes regarding this part.