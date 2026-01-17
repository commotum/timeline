# **FOLIO: Natural Language Reasoning with First-Order Logic**

Hailey Schoelkopf<sup>1</sup> Yilun Zhao<sup>1</sup> Zhenting Qi<sup>2</sup> Simeng Han<sup>1</sup> Martin Riddell<sup>1</sup> Wenfei Zhou<sup>3</sup> James Coady<sup>1</sup> David Peng<sup>1</sup> Lucy Sun<sup>1</sup> Alex Wardle-Solano<sup>1</sup> Yujie Qiao<sup>1</sup> Luke Benson<sup>1</sup> Hannah Szabo<sup>1</sup> Ekaterina Zubova<sup>1</sup> Matthew Burtell<sup>1</sup> Jonathan Fan<sup>4</sup> Ansong Ni<sup>1</sup> Yixin Liu<sup>1</sup> Brian Wong<sup>1</sup> Malcolm Sailor<sup>1</sup> Linyong Nan<sup>1</sup> Jungo Kasai<sup>5</sup> Tao Yu<sup>6</sup> Rui Zhang<sup>7</sup> Alexander R. Fabbri<sup>9</sup> Wojciech Kryściński<sup>9</sup> Semih Yavuz<sup>9</sup> Ye Liu<sup>9</sup> Xi Victoria Lin<sup>8</sup> Shafiq Joty<sup>9</sup> Yingbo Zhou<sup>9</sup> Caiming Xiong<sup>9</sup> Rex Ying<sup>1</sup> Arman Cohan<sup>1</sup> **Dragomir Radev**<sup>1,9</sup> <sup>1</sup>Yale University, <sup>2</sup>Harvard University, <sup>3</sup>NVIDIA, <sup>4</sup>Iowa City West High School <sup>5</sup>University of Washington, <sup>6</sup>University of Hong Kong <sup>7</sup>Penn State University, <sup>8</sup>Meta AI, <sup>9</sup>Salesforce Research

#### **Abstract**

Large language models (LLMs) have achieved remarkable performance on a variety of natural language understanding tasks. However, existing benchmarks are inadequate in measuring the complex logical reasoning capabilities of a model. We present FOLIO, a human-annotated, logically complex and diverse dataset for reasoning in natural language (NL), equipped with first-order logic (FOL) annotations. FOLIO consists of 1,430 examples (unique conclusions), each paired with one of 487 sets of premises used to deductively reason for the validity of each conclusion. The logical correctness of the premises and conclusions is ensured by their FOL annotations, which are automatically verified by an FOL inference engine. In addition to the main NL reasoning task, NL-FOL pairs in FOLIO constitute a new NL-FOL translation dataset. Our experiments on FOLIO systematically evaluate the FOL reasoning ability of supervised fine-tuning on medium-sized language models. For both NL reasoning and NL-FOL translation, we benchmark multiple state-of-the-art language models. Our results show that a subset of FOLIO presents a challenge for one of the most capable Large Language Model (LLM) publicly available, GPT-4.

## 1 Introduction

Large language models (LLMs) have achieved remarkable performance on a variety of natural language tasks (OpenAI et al., 2023; Touvron et al., 2023; Srivastava et al., 2023; Wang et al., 2019a). Logical reasoning is a central component for intelligent systems and should be sufficiently and independently evaluated (Russell and Norvig, 2010).

However, existing natural language tasks are inadequate in measuring the complex logical reasoning capability of a model (Srivastava et al., 2023; Saparov and He, 2023; Tian et al., 2021).

Several datasets related to logical reasoning have recently been proposed. However, existing benchmarks often exhibit limited complexity in reasoning or lack language naturalness. Some of these common benchmarks do not specifically evaluate logical reasoning independently of other forms of reasoning (Yu et al., 2020; Liu et al., 2021). Those specifically designed for measuring logical reasoning are insufficient in terms of logical reasoning complexity and natural language variety. As shown in Table 1, examples in RuleTaker (Clark et al., 2020) and LogicNLI (Tian et al., 2021) need at most five depths of reasoning. The entire corpus of RuleTaker or LogicNLI has fewer than 50 distinct abstract syntax trees. RuleTaker has only 101 words in its vocabulary and LogicNLI has 1077 words in the vocabulary. Moreover, none of them are written by humans with information drawn from real-world knowledge, making them less applicable to real-world reasoning scenarios. The logical deduction portion in BigBench (Srivastava et al., 2023) requires commonsense reasoning besides logical deduction. ProntoQA (Saparov and He, 2023) only contains logical reasoning questions that are answerable with repeated applications of the Modus Ponens inference rule.

We present a natural language reasoning dataset, *FOLIO*, with first-order logic reasoning problems which require the models to decide the correctness of conclusions given a *world* defined by the premises. In FOLIO, we aim to ensure high lan-

| Dataset            | Size  | Reasoning   | Text Source      | Real-World<br>Resources | # Reasoning<br>Depth | Vocab | # Distinct<br>AST |
|--------------------|-------|-------------|------------------|-------------------------|----------------------|-------|-------------------|
| CLUTTER (2019)     | 6k    | Inductive   | Synthetic        | ×                       | ×                    | -     | ×                 |
| RECLOR (2020)      | 6k    | Mixed forms | GMAT, LSAT exams | $\checkmark$            | ×                    | -     | ×                 |
| LogiQA (2021)      | 8.6k  | Mixed forms | NCSE exams       | $\checkmark$            | ×                    | -     | ×                 |
| RuleTaker (2020)   | 500k  | Deductive   | Synthetic        | ×                       | $0 \sim 5$           | 101   | 48                |
| ProofWriter (2021) | 500k  | Deductive   | Synthetic        | ×                       | $0 \sim 5$           | 101   | 48                |
| LogicNLI (2021)    | 20k   | FOL         | Synthetic        | ×                       | $1 \sim 5$           | 1077  | 30                |
| BigBench (2022)    | 1300  | Mixed forms | Human-Written    | Partially               | ×                    | -     | -                 |
| ProntoQA (2023)    | 200   | Deductive   | Synthetic        | ✓                       | 1, 3, 5              | -     | -                 |
| FOLIO (ours)       | 1,435 | FOL         | Expert-written   | ✓                       | $0\sim7$             | 4351  | 76                |

Table 1: Comparison of FOLIO with other datasets related to logical reasoning. #Distinct AST stands for the number of distinct abstract syntax trees, representing the number of distinct sentence-level logic structures in the corpus. FOLIO is the first expert-written dataset for FOL reasoning equipped with parallel FOL formulas. The examples are mostly aligned with real-world knowledge and use highly natural wordings. It also has a greater variety than the previous datasets in terms of reasoning depths with a larger number of distinct logic patterns and a large vocabulary.

| A FOLIO example based on the Wild Turkey Wikipedia page: https://en.wikipedia.org/wiki/Wild_turkey                                             | У                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| NL premises  1. There are six types of wild turkeys: Eastern wild turkey, Osceola wild turkey, Gould's wild turkey,                            | NL Conclusions -> Labels A. Tom is an Ocellated wild turkey> Tru |
| Merriam's wild turkey, Rio Grande wild turkey, and the Ocellated wild turkey.                                                                  | B. Tom is an Eastern wild turkey> False                          |
| 2. Tom is not an Eastern wild turkey.                                                                                                          | C. Joey is a wild turkey> Unknown                                |
| 3. Tom is not an Osceola wild turkey.                                                                                                          |                                                                  |
| 4. Tom is also not a Gould's wild turkey.                                                                                                      |                                                                  |
| 5. Tom is neither a Merriam's wild turkey, nor a Rio Grande wild turkey.                                                                       |                                                                  |
| 6. Tom is a wild turkey.                                                                                                                       |                                                                  |
| FOL Premises                                                                                                                                   | FOL conclusions -> Labels                                        |
| 1. $\forall x (\text{WildTurkey}(x) \rightarrow (\text{EasternWildTurkey}(x) \lor \text{OsceolaWildTurkey}(x) \lor \text{GouldsWildTurkey}(x)$ | A. OcellatedWildTurkey $(tom)$ -> True                           |
| $\vee$ MerriamsWildTurkey $(x) \vee$ RiograndeWildTurkey $(x) \vee$ OcellatedWildTurkey $(x)$ ))                                               | B. EasternWildTurkey $(tom)$ -> False                            |
| 2. ¬EasternWildTurkey(tom)                                                                                                                     | C. WildTurkey(joey) -> Unknown                                   |
| $3. \neg OsceolaWildTurkey(tom))$                                                                                                              |                                                                  |
| 4. $\neg GouldsWildTurkey(tom)$                                                                                                                |                                                                  |
| 5. $\neg$ MerriamsWildTurkey $(tom) \land \neg$ RiograndeWildTurkey $(tom)$                                                                    |                                                                  |
| 6. WildTurkey $(tom)$                                                                                                                          |                                                                  |

Table 2: An example story in FOLIO based on the knowledge from the Wikipedia page on wild turkeys. The story consists of five premises and three conclusions with their corresponding FOL formulas and labels for the conclusions. All five premises are needed to infer the conclusions. The model needs to reason under logic patterns with universal quantification  $(\forall)$ , negation  $(\neg)$ , conjunction  $(\land)$ , and disjunction  $(\lor)$ .

guage naturalness and complexity, an abundant vocabulary, and factuality while also maintaining high reasoning complexity. FOLIO is a high-quality and manually curated dataset, written by CS undergraduate and graduate students and researchers in academia and industry. To ensure the conclusions of our examples follow the premises logically, we annotated all reasoning examples with first-order logic (FOL) formulas. An example of FOLIO is shown in Table 2. Based on our annotations, we propose a new NL-FOL translation task where an NL reasoning example is translated into its FOL counterpart. Finally, we benchmark the performance of strong LMs in both fully supervised and few-shot settings to understand their capabilities in logical reasoning (i.e., deriving the truth value of a logical conclusion from NL premises). Under the few-shot setting, the most capable publicly available LLM so far achieves only 53.1% on the stories written in a hybrid manner, which is slightly better than random.

To sum up, the contributions of this paper are threefold. 1) We release a natural language reasoning dataset written by expert annotators, FOLIO, with first-order logical reasoning problems. 2) We use formal logic, i.e., FOL to ensure the logical validity of the examples written in NL and propose a new NL-FOL translation task. 3) We benchmark the performance of LMs by fine-tuning models and prompting LLMs with few-shot examples, on the FOLIO reasoning task. We hope that FOLIO, as a challenging logical reasoning dataset, will be used to facilitate measuring progress in the logical reasoning capabilities of language models.

## 2 Related Work

#### 2.1 Datasets for reasoning from text

Developing models that can reason in texts has been a core goal in NLP since the field's early days (Cooper et al., 1996). Since then, there has been massive progress in reasoning over text. Various benchmarks that focus on different aspects of reasoning over textual inputs are proposed, including natural language inference (NLI) (Bowman et al., 2015; Wang et al., 2019b), reasoning for commonsense knowledge (Talmor et al., 2019; He et al., 2021) and multi-hop reasoning (Yang et al., 2018; Chen et al., 2020). Among these reasoning abilities, logical reasoning has recently attracted an increasing amount of study. ReClor (Yu et al., 2020) and LogiQA (Liu et al., 2021) both collected multiplechoice questions from standardized graduate admission examinations, answering which requires various types of logical reasoning. However, these datasets cover mixed forms of reasoning and are not intended to test logical reasoning in isolation.

Meanwhile, testing logical reasoning in isolation without involving other forms of reasoning has also attracted researchers in recent years. CLUTRR (Sinha et al., 2019) covers inductive reasoning, which is beyond the scope of first-order logic. Synthetic corpuses of deductive reasoning are proposed to evaluate the deductive reasoning ability of pretrained LMs (Clark et al., 2021; Saeed et al., 2021; Tian et al., 2021). However, these datasets do not contain highly natural sentences and often cover limited forms of logic while FOL is much more expressive. Kazemi et al. (2023) created a dataset for reasoning with contradictory information. Kawabata and Sugawara (2023) crowdsourced rationales for over 3000 examples based on ReClor (Yu et al., 2020). ProntoQA (Saparov and He, 2023) is comprised solely of logical reasoning queries that can be resolved through applying the Modus Ponens inference rule while FOLIO questions require applications of multiple types of inference rules. As shown in Table 1, FOLIO is the first large-scale first-order logic (FOL) reasoning dataset with formal logic annotations in FOL. FO-LIO is logically diverse and complex with complex natural language sentences and a rich vocabulary.

## 2.2 Reasoning using large language models

Reasoning has been demonstrated as one of the emergent abilities of LLMs of sufficient scale recently (Talmor et al., 2020; Wei et al., 2022a;

Chowdhery et al., 2022). One such emergent behavior, Chain-of-Thought prompting (Wei et al., 2022b), consists of a series of intermediate reasoning steps output by an LLM. This improves the performance on arithmetic, commonsense, and symbolic reasoning benchmarks significantly. There has been a line of research continuing on from Chain-of-Thought (Kojima et al., 2022; Li et al., 2022; Yao et al., 2023) to elicit reasoning behavior from LLMs. Building on Chain-of-Thought prompting, many techniques used on top of LLMs to improve downstream performance have been formalized into control flows and programs. These are called language model cascades (Dohan et al., 2022), subsuming techniques such as Chain-of-Thought prompting, STaR (Zelikman et al., 2022), and Selection-Inference (Creswell et al., 2022) for reasoning. Dasgupta et al. (2022) studied the reasoning ability of LLMs but only used a small set of 48 syllogisms with only two premises each. Saparov and He (2023) created a synthetic dataset that and showed that LLMs are capable of making correct individual deduction steps.

With FOLIO, we aim to set a high standard, ensuring that achieving high performance through superficial strategies and shallow heuristics is prevented, allowing a robust evaluation of the first-order logic reasoning capabilities of LLMs. We show that many LLMs fall short on complex first-order logic reasoning, and that significant room for improvement in this area remains.

## **3 FOLIO Corpus Construction**

We collected FOLIO through a carefully designed manual annotation process to achieve high-quality examples that necessitate complex logical reasoning. Writing natural language reasoning stories with FOL requires sufficient knowledge in both semantic parsing and first-order logic, as well as strong analytical skills. Given the complexities of such annotations, we selected annotators based on a few important criteria to ensures that our dataset is annotated with the highest level of precision and expertise, reflecting the complexity and nuance required for first-order logical reasoning. 1). Our annotators are either college or graduate students who are native English speakers or possess nearnative proficiency in English.4 2). They possess formal education in first-order logic, having either completed relevant coursework or undertaken self-directed studies in first-order logic or semantic parsing. At the NL quality check stage, only annotators who are experts in natural language processing or computational linguistics are involved. For the FOL quality check, only annotators who are experts in first-order logic are involved. We also give the annotators several training sessions on how to write a story, by providing them with detailed annotation guidelines. All stories and FOL annotations in FOLIO are written and reviewed by expert annotators, including CS undergraduate and graduate students, and senior researchers, who met the aforementioned criteria.

We develop our dataset in six stages: WikiLogic collection, HybLogic collection, NL quality control, FOL quality control, NL-FOL alignment and FOL verification, spending 980 man-hours in total.

### 3.1 Example collection

We collected our dataset using two different methods in order to obtain examples that are both logically diverse and complex and have abundant abstract syntax tree (AST) variations. The annotators are free to write stories based on any topic they want while writing the stories.

WikiLogic: annotation from scratch using Wikipedia articles as seeds. At this annotation stage, the annotators are asked to select random Wikipedia pages by repeatedly using the Wikipedia Special Random link.<sup>1</sup> The Wikipedia articles are used to develop ideas for topics to write new stories. We ask the annotators to create new stories from scratch without using templates based on realworld knowledge, which should be plausible in general. Each of the stories is composed of several premises and conclusions with truth values of True, False, or Unknown (see Table 2 for an example). We also ask the annotators to write parallel FOL sentences for both the premises and conclusions. This results in a wide range of topics, abundant AST variations, and a wide vocabulary for FOLIO. Table 1 shows a comparison of FOLIO with other reasoning datasets that purely evaluate first-order logic or deductive reasoning.

**HybLogic:** hybrid annotation The task of generating logically sound stories from scratch for a set of facts is very time-consuming for human writers, where the main challenge is to create complex and varied logical patterns to arrive at a conclusion. To address the problems of solely using manual

annotation, we also consider a hybrid approach to facilitate the process. Our hybrid method is based on a common form of logical stories: *syllogisms*. A syllogism consists of two premises and a single conclusion, and the conclusion states some facts about the entities and categories in the premises.

In this approach, we first generate logically valid stories, which are templates containing abstract categories and entities, by combining multiple syllogisms into a single story template: the conclusion of one syllogism is used as a premise for the next syllogism. There are 256 logically distinct types of syllogisms and 24 of them are valid (Lehman, 1973). We use various combinations of 24 valid syllogisms. We also add in conjunction, disjunction, and implication. We show an example of the resulting templates in Appendix B. We then ask human annotators to assign nouns, phrases, or clauses to the abstract entities or categories that reflect real-life scenarios to each template and write logically-valid stories in natural language. The usage of the template is to ensure that we have a set of varied and complex logical stories with multiple conclusions. There are many ways of expressing the same logic template in natural language, and so the generated templates augment, rather than limit, the creativity of humans.

# 3.2 Quality control for NL sentences

To ensure the highest quality of the dataset, we dedicated considerable attention to the following key aspects of the natural language sentences during the quality control process.

**Factuality and bias** Our dataset prioritizes realism and factual accuracy, steering clear of biases and stereotypes linked to identity markers like race, ethnicity, gender, sexuality, nationality, class, and religion. Toward these objectives, we manually screened all stories and found that 39.2% of the stories suffer from at least one of these issues. We implemented a detailed protocol to rewrite these stories. The protocol is in Appendix C.

Language quality Apart from grammar, we make sure the sentences in our dataset are highly natural. All the sentences are first checked with a grammar checking tool, Grammarly. Our annotators who have graduated from or are senior students studying English Literature conducted a thorough round of review for grammatical correctness and language naturalness. We also eliminate natural language ambiguity when it is possible. We include

https://en.wikipedia.org/wiki/Special:Random

rules on eliminating ambiguity in Appendix D. Employing these rules effectively reduces the ambiguity of natural language in this reasoning dataset, but incurs the tradeoff of limiting variations in some usage of language. However, we note that there is still sufficient variation in terms of sentence structures and logical structures as shown in Table 1.

## 3.3 Quality control for FOL formulas

We adopt the FOL definitions and syntax most widely used in the AI community (Russell and Norvig, 2010). We include more details on the definition of FOL we consider and the FOL modelling convention in Appendix E In preliminary investigations, we found that the human-written FOL formulas suffer from FOL consistency issues, which necessitates an additional round of quality control for FOL formulas.

FOL consistency One NL sentence can be translated into FOL through multiple non-equivalent ways. For example, sometimes additional information inferred from a sentence can be represented in FOL, leading to multiple representations. We therefore design an annotation protocol for FOL translation in order to ensure that our FOL translations are as consistent as possible across all examples in our dataset. We highlight a few important strategies used in the annotation protocol in Appendix F.

## 3.4 NL-FOL alignment review

Apart from checking whether NL and FOL express equivalent meanings, we also add necessary commonsense knowledge in both the NL and FOL premises. Sometimes humans do not write certain commonsense knowledge in the premises that is required in the FOL reasoning process, which is based solely on the premises given. We add such knowledge as additional premises at this stage. In particular, intrinsic properties of some predicates are required in the FOL reasoning process. For example, "LocatedIn(x,y)" should be transitive and "BeFamily(x,y)" should be symmetric.

# 3.5 FOL verification

Recognizing that the FOL formula annotations can be error-prone, we verify the syntactic validity and label consistency of FOL formula annotations with an FOL inference engine. We include the details of the FOL inference engine in Appendix G.

![](_page_4_Figure_8.jpeg)

Figure 1: Distribution of reasoning depths

#### 3.6 Dataset statistics

We show basic statistics of FOLIO and demonstrate the abundant vocabulary and logical complexity of FOLIO: Tables 1, 3 and Figure 1.

**Basic statistics** Table 3 shows that examples based on Wikipedia make up the largest portion of FOLIO, with 304 stories, 1,353 NL and FOL premise pairs, and 753 NL and FOL conclusion pairs. Hybrid annotations consist of 183 stories with 1,054 NL and FOL premise pairs, and 682 NL and FOL conclusion pairs in total.

Natural language complexity We use the Dale-Chall Readability Formula (Dale and Chall, 1948, 1995) to show the text complexity of FOLIO following (Singh et al., 2023; Arps et al., 2022; Wei et al., 2021). We show the distribution of readability in Appendix H.

**Logical complexity and diversity statistics** As shown in Figure 1, the mode of reasoning depths is four in FOLIO. 28.7% of the examples need five or more depths of reasoning to infer the conclusions, while the previous datasets needed at most five reasoning depths as shown in Table 1. This illustrates the logical complexity of FOLIO. Table 1 shows that FOLIO also has a much larger number of distinct ASTs than the previous datasets, indicating that FOLIO is much more logically diverse. Figure 1 demonstrates the distribution of the number of examples in the WikiLogic and HybLogic sets versus the number of premises needed to arrive at a conclusion, showing that most of the conclusions from WikiLogic require one to five premises while those from HybLogic require five to eight premises.

Vocabulary and topics Table 3 shows that our dataset has a vocabulary of 4,351 words, and the examples based on Wikipedia account for 74% of the total vocabulary even though the WikiLogic stories take up only 63% of the total number of stories. The vocabulary of FOLIO is also significantly

| Source    | #Stories #Premises |            | #Conclusions | NL   |        |              | Logic  |     |
|-----------|--------------------|------------|--------------|------|--------|--------------|--------|-----|
| Source    | "Stories           | ma remises | Voca         |      | #Words | Complexity   | #Depth | AST |
| WikiLogic | 304                | 1353       | 753          | 3250 | 8.50   | 0 - 14 grade | 1 - 5  | 51  |
| HybLogic  | 183                | 1054       | 682          | 1902 | 11.52  | 0 - 14 grade | 5 - 8  | 25  |
| Total     | 487                | 2407       | 1435         | 4351 | 9.86   | 0 - 14 grade | 76     | 5-8 |

Table 3: Statistics based on different data collection methods of FOLIO. #Words is the average number of words per NL sentence.

larger than the previous synthetically constructed datasets for logical reasoning.

## 4 Task Definition

We define two new tasks based on FOLIO, natural language reasoning with first-order logic and NL-FOL translation.

# 4.1 Natural language reasoning with first-order logic

Each natural language (NL) story S in FOLIO consists of n premises:  $P = \{p_1, p_2, ..., p_n\}$  and m conclusions:  $H = \{h_1, h_2, ..., h_m\}$ . All NL stories are annotated with parallel FOL stories SF, which are sets of FOL formulas consisting of n premises  $PF = \{pf_1, pf_2, ..., pf_n\}$  and m conclusions  $HF = \{hf_1, hf_2, ..., hf_m\}$ .  $pf_i$  and  $hf_i$  are logically and semantically similar to  $p_i$  and  $h_i$ , respectively. Given P and H, the goal is to determine the truth values of the conclusions: "True", "False" or "Unknown", based on FOL reasoning.

#### 4.2 NL-FOL translation

We propose a new natural language to first-order logic translation (NL-FOL translation) task alongside our reasoning dataset. The goal of this task is to translate an NL story S to an FOL story FS. In particular, each of the NL sentence  $p_i$  or  $h_i$  and the parallel FOL formula  $pf_i$  or  $hf_i$  should be logically and semantically equivalent. Moreover, the truth values for the conclusions should be the same based on the NL story S and the parallel FOL story FS. In our dataset, the premises and conclusions are set up in such a way to ensure that the inference engine always returns an answer given enough resources such as time and memory. Unlike previous work (Singh et al., 2020) which translates problems with a single premise and a single hypothesis, our task is on translating examples of various lengths with a focus on stories with multiple premises. Thus, it also requires the models to

consider *discourse-level* consistencies as opposed to translation at the sentence level.

NL-FOL evaluation metrics Two metrics are adopted to evaluate NL-FOL translation to capture different aspects of the generation results: 1). Syntactic validity (SynV). The Syntactic Validity score measures whether the FOL formulas are syntactically valid. The score will be 1 if all FOL formulas of an example can pass the syntactic check and 0 otherwise 2). Inference Engine execution accuracy (ExcAcc). The group of translated FOL for premises and conclusions in one story is fed into our inference engine to output the truth value for each conclusion. We define the accuracy of the output labels as the execution accuracy. We leave for future work the design of a more reliable metric of NL-FOL translation.

## 5 Experiments

In this section, we describe our experiments and main results.

## 5.1 Experimental setup

**Tasks** We conduct experiments on the two tasks in §4: *NL reasoning with first-order logic (logical reasoning)* and *NL-FOL translation (NL-FOL)*.

**Dataset split** We split FOLIO by 70%/15%/15% split for the train/validation/test sets with 1,001/203/226 examples respectively. We split by story so that models are evaluated on unseen stories.

**Evaluation metrics** We use accuracy for evaluating logical reasoning performance. For NL-FOL translation, we use the metrics in Section 4.2.

## 5.2 Models

We test the logical reasoning capabilities of LMs using fully supervised fine-tuning and few-shot prompting. We also test NL-FOL translation with few-shot prompting.

Fully supervised fine-tuning As fine-tuning baselines, we experiment with BERT (Devlin et al., 2019), and RoBERTa (Liu et al., 2020). We fine-tune the base and large versions of both BERT and RoBERTa, with an additional two-layer classification layer to predict the truth values. For the second task, i.e., NL-FOL translation, we only report few-shot prompting methods.

**Few-shot prompting** We conduct zero-shot and few-shot prompting experiments on larger LMs with few-shot capabilities. For open-source models, we test LLaMA-13B and LLaMA-70B (Touvron et al., 2023), GPT-NeoX-20B (Black et al., 2022); for proprietary models we test GPT-3 (Brown et al., 2020), GPT-3.5-Turbo and GPT-4 (OpenAI et al., 2023) using prompts with 8 examples.<sup>2</sup>

**Prompting strategies** We experiment with incorporating recent prompting strategies into GPT-4 as they have shown improvements in the general reasoning performance of LLMs. The prompting strategies include chain-of-thought (CoT) prompting (Wei et al., 2022b), chain-of-thought prompting with self-consistency (Wang et al., 2023) and tree-of-thought prompting (Yao et al., 2023).

Logical reasoning methods We also test recent methods specifically designed for logical reasoning: Logic-LM (2023), LINC (Olausson et al., 2023) and DetermLR(Sun et al., 2023), using GPT-4 as the base model. For the second task (NL-FOL translation), we use the same examples as in the Few-Shot NL experiments except that all the conclusions are included in each example.

We run experiments on five randomly sampled sets of examples from the training set and report the average accuracy.

#### 5.3 Main results

**Logical reasoning** The majority baseline of our dataset is 38.5% since in our test set, there are 87, 78 and 61 examples with labels of true, false and unknown respectively. As shown in Table 4, BERT-base and RoBERTa-base have similar performance on FOLIO with 56.83% accuracy. BERT-large has a 2.2% improvement over BERT-base. RoBERTa-large improves 3.1% over BERT-large. Flan-T5-Large achieves the highest performance in the fine-tuning setting and the accuracy is 65.7%.

| Model                      | Size | Acc (%) |
|----------------------------|------|---------|
| majority baseline          | -    | 38.5%   |
| random probability         | -    | 33.3 %  |
| Fully supervised fine-tune |      |         |
| BERT-base                  | 110M | 56.8    |
| BERT-large                 | 340M | 59.0    |
| RoBERTa-base               | 110M | 56.8    |
| RoBERTa-large              | 340M | 62.1    |
| Flan-T5-Large              | 783M | 65.9    |
| 0-shot NL Prompt           |      |         |
| GPT-3.5-Turbo              | -    | 53.1    |
| GPT-4                      | -    | 61.3    |
| 8-shot NL Prompt           |      |         |
| LLama-13B                  | 13B  | 33.6    |
| LLama-70B                  | 70B  | 44.0    |
| LLama-70B - CoT            | 70B  | 47.8    |
| LLama-70B - ToT            | 70B  | 48.4    |
| text-davinci-002           | -    | 49.5    |
| GPT-3.5-Turbo              | -    | 58.3    |
| GPT-4                      | -    | 64.2    |
| GPT-4 - CoT (2022b)        | -    | 68.9    |
| GPT-4 - CoT with SC (2023) | -    | 69.5    |
| GPT-4 ToT (2023)           | -    | 70.0    |
| LR-specific Methods        |      |         |
| Logic-LM (2023)            | -    | 78.1    |
| LINC (2023)                | -    | 73.1    |
| DetermLR (2023)            | -    | 77.5    |

Table 4: Logical reasoning results of fully supervised fine-tuning and few-shot prompting on FOLIO test set. The model sizes of text-davinci-002, GPT-3.5-Turbo and GPT-4 are hidden from public<sup>3</sup>. CoT stands for chain-of-thought prompting (Wei et al., 2022b). SC stands for self-consistency (Wang et al., 2023). ToT stands for tree-of-thought prompting (Yao et al., 2023).

We show that zero-shot prompting GPT-3.5 achieves better results than few-shot prompting text-davinci-002. Under few-shot NL prompting setting, LLama-13B achieves 33.63%, which is only slightly better than chance (33%). LLama-70B achieves 43.97%, around 10% better than LLaMA-13B and obtains improvements of around 4% with Chain-of-thought prompting and Tree of Thought prompting. Text-davinci-002 achieves 49.53% and GPT-3.5 achieves 58.34%. GPT-4 achieves the best results among GPT series models.

Incorporating recent prompting strategies increases the performance of vanilla few-shot prompting. Chain-of-thought prompting achieves more than a 4% increase over GPT-4. Self-consistency (SC) improves chain-of-thought prompting by

<sup>&</sup>lt;sup>2</sup>In experimenting with different prompts, we found 8 shot examples to perform slightly better. It is also the maximum number of examples that fits in the text-davinci-002 context.

<sup>&</sup>lt;sup>3</sup>Hereafter, "GPT-3.5" refers to GPT-3.5-Turbo.

| Model                  | Zer          | o-Shot       | Few-Shot     |              |  |
|------------------------|--------------|--------------|--------------|--------------|--|
|                        | Synv         | ExcAcc       | Sync         | ExcAcc       |  |
| GPT-3.5-Turbo<br>GPT-4 | 68.4<br>86.1 | 50.4<br>51.7 | 93.3<br>93.9 | 56.0<br>63.8 |  |

Table 5: NL-FOL translation results on FOLIO. SynV measures syntactic validity and ExcAcc measures the inference engine execution accuracy.

0.6% percent. Tree-of-thought prompting achieves slightly better result than self-consistency with chain-of-thought prompting. For the results of recent methods developed for logical reasoning, LINC (Olausson et al., 2023) achieves around a 9% increase over few-shot prompting GPT-4. Both Logic-LM (GPT-4)(2023) and DetermLR (2023) achieves more than a 13% increase over few-shot prompting GPT-4, showing the superiority of the methods on logical reasoning.

NL-FOL translation Table 5 shows the results of NL-FOL translation. The syntactic validity scores are around 93% with both GPT-3.5-Turbo and GPT-4. This indicates that language models with sufficient scales are good at picking up the patterns for FOL formulas and generating syntactically valid FOL formulas. However, GPT-3.5-Turbo and GPT-4 are not yet good at translating an NL story to a logically or semantically similar FOL counterpart, as indicated by the low inference engine execution accuracy score.

## 6 Error Analysis

Below we provide analysis of our results and key findings, providing additional insights into our dataset FOLIO and the current capabilities of LLMs in logical reasoning.

Models have higher accuracy on examples with fewer reasoning depths than on those with higher number of reasoing depths. We show the accuracy categorized by reasoning depths in Figure 2. With few-shot prompting, GPT-3.5 and GPT-4 both perform much better on examples with a  $0 \sim 3$  reasoning depth, indicating that examples with a  $4 \sim 7$  reasoning depth pose a challenge to the SoTA LMs. With fine-tuning, RoBERTa has slightly higher performance on test examples with  $0 \sim 3$  reasoning depth than on those with  $4 \sim 7$  reasoning depth, but the difference is much smaller. This indicates that fine-tuning on longer and more difficult reasoning chains in the training set can improve model performance on equally-long test

![](_page_7_Figure_7.jpeg)

Figure 2: Accuracies of different models categorized into examples with different reasoning depths.

| Method        | Model         | Wiki  | Hyb   |
|---------------|---------------|-------|-------|
| Fine-tuning   | RoBERTa-large | 60.71 | 63.48 |
| NL Prompting  | GPT-3.5-Turbo | 68.88 | 47.70 |
|               | GPT-4         | 75.43 | 53.10 |
| NL-FOL ExcAcc | GPT-3.5-Turbo | 45.17 | 61.82 |
|               | GPT-4         | 59.12 | 67.93 |

Table 6: Performance differences on the WikiLogic and HybLogic subset of FOLIO. WikiLogic has more diverse logical structures while HybLogic stories have higher reasoning depths.

example chains. We note that the presence and prevalence of these difficult examples are unique to FOLIO. FOLIO's unique complexity reveals that current LMs are limited in their ability to extrapolate to longer and more complex reasoning chains, and suggests an avenue for further study.

Models have higher accuracy on WikiLogic than on HybLogic As shown in Table 6, in logical reasoning, GPT-3.5 and GPT-4 achieve substantially lower results on HybLogic than on WikiLogic and the result is slightly higher than chance. We hypothesize that this is because HybLogic has high logical complexity that the SoTA LLMs like GPT-4 cannot solve yet while WikiLogic examples require shorter reasoning chains which the model is already capable of solving. Moreover, since the examples in WikiLogic are created from scratch by humans, it is possible that LLMs have seen similar texts with similar logical patterns in the training data. However, fine-tuning RoBERTa-large achieves higher performance on HybLogic than on WikiLogic. This is likely because HybLogic is created from templates and some of the logical patterns can be learned during fine-tuning.

In NL-FOL translation, performs 10 points better on HybLogic than WikiLogic. This could be because WikiLogic has more distinct and diverse sentence-level logical and language patterns and FOL annotations. WikiLogic has 53 ASTs while HybLogic has 33. Despite being more logically complex on a story level, FOL translations for Hy-

bLogic stories have simpler logical structures on a statement level. We include case study for one WikiLogic example and one HybLogic example in Appendix I and further analysis on model performance in Appendix J.

| Faulty path                   | 65% |
|-------------------------------|-----|
| Wrong derivation              | 25% |
| Wrong syntactic comprehension | 5%  |
| Spurious shortcut             | 5%  |

Table 7: Human evaluation on GPT-4 model outputs with incorrect truth value predictions

#### **Example**

Premises 1. All children are human. 2. If someone is underage, then they are a child. 3. People are either underage or of age. 4. If someone is of age, then they can vote. 5. If someone is of age, they can legally get married. 6. If Jack is a child and a human, then Jack is neither able to vote nor able to get married. Conclusion -> Label: Jack is able to vote and get married. -> True.

#### **GPT-4 Output**

We know that all children are human (premise 1) and if someone is underage, they are a child (premise 2). People are either underage or of age (premise 3). If someone is of age, they can vote (premise 4) and get married (premise 5). If Jack is a child and a human, then Jack is neither able to vote nor get married (premise 6). We don't have any information about Jack's age, so we cannot determine if he is a child or of age. Therefore, we cannot determine if Jack is able to vote and get married.

Table 8: Case study for the scenario where a model is unable to form the correct reasoning chain.

**Human evaluation on model outputs** We conduct human evaluation on the GPT-4 model outputs with wrong truth value predictions. As shown in Table 7, approximately 65% of the time, the model struggles to construct accurate reasoning chains for complex problems with intricate steps, leading to faulty reasoning paths and indicating a limited ability to solve problems with long reasoning chains. In 25% of cases, erroneous derivations occur within certain reasoning steps, highlighting potential inaccuracies and flaws in logical deductions. 5% of conclusions in FOLIO have a complex syntactic structure, posing comprehension challenges for GPT-4. 5% of outputs show that GPT-4 leverage commonsense reasoning to employ spurious shortcuts that lead to the wrong truth value for the conclusion. We provide a case study for the "Faulty path" scenario in Table 8. In this instance, the model can perform simple derivations from the

premises, like "If someone is of age, they can vote and get married." However, because of the problem's complexity, the model struggles to identify the essential intermediate steps and cannot ascertain the truth value of conclusions, such as "Jack is not a child."

## 6.1 Human performance

We collected truth value annotations of logical reasoning for FOLIO test set from expert and non-expert annotators. Our expert annotators are computer science college students familiar with FOL. Non-expert annotators are community college or high school students who have not taken the SAT. Both expert and non-expert annotators are native English speakers. Expert annotations achieve an accuracy of 95.98% while non-expert annotations achieves 61.82%, with a gap of 34.16%. This shows that sufficient domain knowledge of FOL is necessary for good performance on FOLIO. The expert and GPT-4 gap is 31.82%, suggesting significant room for model improvement.

#### 7 Conclusion

We introduced FOLIO, an expert-written dataset for logical reasoning equipped with FOL formulas. The examples in FOLIO are created based on real-world knowledge with natural language. It exhibits a large number of distinct logic patterns and a large vocabulary. Experiments show that FOLIO presents a challenge for one of the most capable Large Language Model publicly available.

#### 8 Limitations

We focus on collecting a very high-quality dataset in evaluating logical reasoning rather than merely a large dataset. Optimizing for quality required us to adopt a rigorous annotation process with domain experts selected based on a few important criteria as mentioned in Appendix A: Annotator Selection. Significantly scaling up this process would have required resources beyond our current means and we are unable further expand our dataset for investigating how the size of training data affects the performance of fine-tuning experiments. We encourage the community to apply our annotation protocol to expand this realistic and complex FOL reasoning story set.

## References

- David Arps, Jan Kels, Florian Krämer, Yunus Renz, Regina Stodden, and Wiebke Petersen. 2022. HHU-plexity at text complexity DE challenge 2022. In *Proceedings of the GermEval 2022 Workshop on Text Complexity Assessment of German Text*, pages 27–32, Potsdam, Germany. Association for Computational Linguistics.
- Sid Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, USVSN Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. 2022. Gpt-neox-20b: An open-source autoregressive language model. *arXiv* preprint.
- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In *Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing*, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.
- Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.
- Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. 2020. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In *Findings of the Association for Computational Linguistics: EMNLP 2020*, pages 1026–1036, Online. Association for Computational Linguistics.
- Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. *arXiv preprint arXiv:2204.02311*.
- Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2020. Transformers as soft reasoners over language. *CoRR*, abs/2002.05867.
- Peter Clark, Oyvind Tafjord, and Kyle Richardson. 2021. Transformers as soft reasoners over language. In *Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence*, pages 3882–3890.

- Robin Cooper, Dick Crouch, Jan Van Eijck, Chris Fox, Johan Van Genabith, Jan Jaspars, Hans Kamp, David Milward, Manfred Pinkal, Massimo Poesio, et al. 1996. Using the framework. Technical report, Technical Report LRE 62-051 D-16, The FraCaS Consortium.
- Antonia Creswell, Murray Shanahan, and Irina Higgins. 2022. Selection-inference: Exploiting large language models for interpretable logical reasoning. *arXiv* preprint arXiv:2205.09712.
- Edgar Dale and Jeanne S. Chall. 1948. A formula for predicting readability. *Educational Research Bulletin*, 27(1):11–28.
- Edgar Dale and Jeanne S. Chall. 1995. *Readability Revisited: The New Dale-Chall Readability Formula*. Brookline Books.
- Ishita Dasgupta, Andrew K Lampinen, Stephanie CY Chan, Antonia Creswell, Dharshan Kumaran, James L McClelland, and Felix Hill. 2022. Language models show human-like content effects on reasoning. arXiv preprint arXiv:2207.07051.
- Donald Davidson. 2001. 105The Logical Form of Action Sentences. In *Essays on Actions and Events*. Oxford University Press.
- Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.
- David Dohan, Winnie Xu, Aitor Lewkowycz, Jacob Austin, David Bieber, Raphael Gontijo Lopes, Yuhuai Wu, Henryk Michalewski, Rif A. Saurous, Jascha Sohl-dickstein, Kevin Murphy, and Charles Sutton. 2022. Language model cascades. *arXiv preprint*.
- Weinan He, Canming Huang, Yongmei Liu, and Xiaodan Zhu. 2021. WinoLogic: A zero-shot logic-based diagnostic dataset for Winograd Schema Challenge. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 3779–3789, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Akira Kawabata and Saku Sugawara. 2023. Evaluating the rationale understanding of critical reasoning in logical reading comprehension. *Preprint*, arXiv:2311.18353.
- Mehran Kazemi, Quan Yuan, Deepti Bhatia, Najoung Kim, Xin Xu, Vaiva Imbrasaite, and Deepak Ramachandran. 2023. Boardgameqa: A dataset for natural language reasoning with contradictory information. *Preprint*, arXiv:2306.07934.

- Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. *arXiv preprint arXiv:2205.11916*.
- Anne Lehman. 1973. Two sets of perfect syllogisms. Notre Dame Journal of Formal Logic, 14(3):425 – 429.
- Sarah-Jane Leslie. 2008. Generics: Cognition and Acquisition. *The Philosophical Review*, 117(1):1–47.
- Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. 2022. On the advance of making language models better reasoners. *arXiv preprint arXiv:2206.02336*.
- Jian Liu, Leyang Cui, Hanmeng Liu, Dandan Huang, Yile Wang, and Yue Zhang. 2021. Logiqa: a challenge dataset for machine reading comprehension with logical reasoning. In *Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence*, pages 3622–3628.
- Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Ro{bert}a: A robustly optimized {bert} pretraining approach. arXiv preprint arXiv:1907.11692.
- W. McCune. 2005–2010. Prover9 and mace4. http://www.cs.unm.edu/~mccune/prover9/.
- Ansong Ni, Pengcheng Yin, Yilun Zhao, Martin Riddell, Troy Feng, Rui Shen, Stephen Yin, Ye Liu, Semih Yavuz, Caiming Xiong, Shafiq Joty, Yingbo Zhou, Dragomir Radev, and Arman Cohan. 2023. L2ceval: Evaluating language-to-code generation capabilities of large language models. *Preprint*, arXiv:2309.17446.
- Tobias Nipkow, Lawrence C. Paulson, and Markus Wenzel. 2002. *Isabelle/Hol a Proof Assistant for Higher-Order Logic*. Springer.
- Theo Olausson, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama, Joshua Tenenbaum, and Roger Levy. 2023. LINC: A neurosymbolic approach for logical reasoning by combining language models with first-order logic provers. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, pages 5153–5176, Singapore. Association for Computational Linguistics.
- OpenAI, Josh Achiam, and Others. 2023. Gpt-4 technical report. *Preprint*, arXiv:2303.08774.
- Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. 2023. Logic-LM: Empowering large language models with symbolic solvers for faithful logical reasoning. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 3806–3824, Singapore. Association for Computational Linguistics.

- Terence Parsons. 1990. Events in the Semantics of English. MIT Press, Cambridge, MA, USA.
- Stuart Russell and Peter Norvig. 2010. *Artificial Intelligence: A Modern Approach*, 3 edition. Prentice Hall.
- Mohammed Saeed, Naser Ahmadi, Preslav Nakov, and Paolo Papotti. 2021. RuleBERT: Teaching soft rules to pre-trained language models. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 1460–1476, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Abulhair Saparov and He He. 2023. Language models can (kind of) reason: A systematic formal analysis of chain-of-thought. In *International Conference on Learning Representations*.
- Hrituraj Singh, Milan Aggrawal, and Balaji Krishnamurthy. 2020. Exploring neural models for parsing natural language into first-order logic. *arXiv preprint arXiv:2002.06544*.
- Pranaydeep Singh, Luna De Bruyne, Orphée De Clercq, and Els Lefever. 2023. Misery loves complexity: Exploring linguistic complexity in the context of emotion detection. In *Findings of the Association for Computational Linguistics: EMNLP 2023*, pages 12871–12880, Singapore. Association for Computational Linguistics.
- Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. 2019. CLUTRR: A diagnostic benchmark for inductive reasoning from text. In *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)*, pages 4506–4515, Hong Kong, China. Association for Computational Linguistics.
- Aarohi Srivastava, Abhinav Rastogi, and +447 Authors. 2023. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *Preprint*, arXiv:2206.04615.
- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al. 2022. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. *arXiv preprint arXiv:2206.04615*.
- Hongda Sun, Weikai Xu, Wei Liu, Jian Luan, Bin Wang, Shuo Shang, Ji-Rong Wen, and Rui Yan. 2023. From indeterminacy to determinacy: Augmenting logical reasoning capabilities with large language models. *Preprint*, arXiv:2310.18659.
- G. Sutcliffe. 2017. The TPTP Problem Library and Associated Infrastructure. From CNF to TH0, TPTP v6.4.0. *Journal of Automated Reasoning*, 59(4):483–502.

- Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. 2021. ProofWriter: Generating implications, proofs, and abductive statements over natural language. In *Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021*, pages 3621–3634, Online. Association for Computational Linguistics.
- Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.
- Alon Talmor, Oyvind Tafjord, Peter Clark, Yoav Goldberg, and Jonathan Berant. 2020. Leap-of-thought: Teaching pre-trained models to systematically reason over implicit knowledge. *Advances in Neural Information Processing Systems*, 33:20227–20237.
- Jidong Tian, Yitian Li, Wenqing Chen, Liqiang Xiao, Hao He, and Yaohui Jin. 2021. Diagnosing the first-order logical reasoning ability through LogicNLI. In *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 3738–3747, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. *Preprint*, arXiv:2302.13971.
- Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019a. Superglue: A stickier benchmark for general-purpose language understanding systems. In *Advances in Neural Information Processing Systems*, volume 32. Curran Associates, Inc.
- Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. 2019b. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32.
- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In *The Eleventh International Conference on Learning Representations*.
- Jason Wei, Kelly Finn, Emma Templeton, Thalia Wheatley, and Soroush Vosoughi. 2021. Linguistic complexity loss in text-based therapy. In *Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics:*

- Human Language Technologies, pages 4450–4459, Online. Association for Computational Linguistics.
- Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. 2022a. Emergent abilities of large language models. *arXiv preprint arXiv:2206.07682*.
- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. 2022b. Chain of thought prompting elicits reasoning in large language models. *arXiv preprint arXiv:2201.11903*.
- Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.
- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik R Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In *Thirty-seventh Conference on Neural Information Processing Systems*.
- Weihao Yu, Zihang Jiang, Yanfei Dong, and Jiashi Feng. 2020. Reclor: A reading comprehension dataset requiring logical reasoning. In *International Conference on Learning Representations*.
- Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. 2022. Star: Bootstrapping reasoning with reasoning. *arXiv preprint*.

# **A** Annotator Selection

Given the complexities of our annotations, we selected annotators based on a few important criteria 1). Our annotators are either college or graduate students who are native English speakers or possess near-native proficiency in English.<sup>4</sup> 2). They possess formal education in first-order logic, having either completed relevant coursework or undertaken self-directed studies in first-order logic or semantic parsing. At the NL quality check stage, only annotators who are experts in natural language processing or computational linguistics are involved. For the FOL quality check, only annotators who are experts in first-order logic are involved. We also give the annotators several training sessions on how to write a story, by providing them with detailed annotation guidelines. All stories and FOL annotations in FOLIO are written and reviewed by

<sup>&</sup>lt;sup>4</sup>By "near-native" we mean with English speaking and understanding ability that closely mirrors that of a native English speakers.

expert annotators, including CS undergraduate and graduate students, and senior researchers, who met the aforementioned criteria.

# **B** HybLogic Template Example

An example the resulting template is as follows:

Premises:
All M are P. All S are M.
Either S or A. All A are B.
All D are B. No C are B.
a is either a C or a P.

Conclusions:
[Unknown] a is an S.
[True] If a is either a C or a D,
then a is not either an A or a B.

# C Factuality and Bias Elimination Protocol

We rewrote those that are not reflective of wellestablished scientific, historical, or legal facts. We took out stories that had strongly opinionated language and contained gender, racial, and classist biases. We accept certain classes of "psychologically fundamental generalizations" (Leslie, 2008), however, such as "Covid is transmitted through the air" or "Tigers eat other animals," that may not be factually invariant but add logical and semantic nuances to the stories. For stories that pertain to generalization, such as "All As are Bs," we have added specifiers like "all Dan knows" to give a degree of reasonable factuality. For example, "All science fiction that Dan knows comes from an imaginative process" has a more reasonable degree of factuality than "All science fiction comes from an imaginative process."

# **D** Language Quality Control

- We always use "either-or" to express exclusive disjunction. We use either "A or B" or "A or B, or both" to express inclusive disjunction. In English "or" itself can be interpreted as either inclusive disjunction or exclusive disjunction. Adding "or both" cancels the exclusive disjunction distinctly. However, it is less common in the wild than just using "or". we could add "or both" if it is important to emphasize the inclusive part semantically or contextually or for factuality; and do not add "or both" if it is not. We rely on the language model to figure out if it should be inclusive or exclusive, therefore not sacrificing naturalness.
- It is more natural to say "Some A is B" rather

than "there exists an A such that A is B." "All A are B" can be more natural than "If A then B".

• Writing NL sentences that express negation over exclusive-or ("either both or neither") can be cumbersome but we found one natural ways of expressing these situations: "Each morning, John either works out and stretches, or he does neither".

Other common issues in NL quality include singular/plural issues, especially in statements that deal with both categories and individual members of those categories; as well as ambiguities resulting from improper introduction of, or failure to introduce, proper nouns.

## **E** First-Order Logic

# E.1 First-Order Logic VS Natural Language

FOL enables deriving facts from other facts (Russell and Norvig, 2010). In the context of logical reasoning in modern NLP, FOL, as a logical form, is a more explicit logical representation than its NL counterpart and can be used as input to an FOL prover in order to obtain the exact truth values for the conclusions. FOL has no ambiguity while ambiguity can occur at various levels of NLP. FOL can thus be a good interface between how LMs are trained and how logical conclusions are reasoned.

#### E.2 FOL definition

We include the following operators: negation  $\neg$ , conjunction  $\land$ , disjunction  $\lor$ , implication  $\rightarrow$ , universal quantifier  $\forall$ , existential quantifier  $\exists$ , equal =. Following (Russell and Norvig, 2010), we consider temporal logic and modal logic as special-purpose logics. Consequently, they are beyond the scope of the definition of first-order logic used in our dataset.

## **E.3** FOL modeling conventions

We use n-place predicates when applicable for the expressivity of the FOL formulas. However, we do not use the Davidsonian (Davidson, 2001) or neo-Davidsonian semantics (Parsons, 1990) because translating the majority of the FOL formulas in our dataset only requires one-place and twoplace predicates. Therefore the Davidsonian or neo-Davidsonian semantics are not necessary for the expressivity of the FOL formulas.

For example, "Enjoy dressing up in old-fashioned clothing" is rendered as "Enjoy(x, dressingUp, oldFashionedClothing)".

#### F FOL Annotation Protocol

We therefore design an annotation protocol for first-order logic translation in order to ensure that our FOL translations are as consistent as possible across all examples in our dataset. We highlight a few important strategies used in the annotation protocol. a). First-order logic formulas need to preserve as much as possible the semantics of natural language sentences. b). First-order logic formulas should stay as faithful to the structure of the original NL sentence as possible. c). Semantic decomposition is not needed unless necessary for maintaining the NL expressivity. This means that "John is a bachelor" can be translated into FOL simply as "Bachelor(John)". d). In terms of abstraction, we neglect tense and remove all the plural forms of verbs.

## **G** FOL Inference Engine

Although there are many provers widely used in the community (McCune, 2005–2010; Sutcliffe, 2017; Nipkow et al., 2002), we adopt the inference engine provided in the Stanford CS221 course page<sup>5</sup>, which is a compact module designed specifically for procesing first-order logic statements. The inference engine does not support input in the FOL syntax adopted by standard education material (Russell and Norvig, 2010), which is used in our dataset. We therefore developed a FOL parser in order to convert the FOL formulas written by humans to the input format of the inference engine. The converter is a semantic parser tool written in Python. Although LLMs such as GPT-4 can be utilized to conduct the conversion, it is hard to ensure the GPT-4 outputs are always correct.

Proving a story requires three steps. First, the FOL statements of the premises and conclusions of a story annotated by humans are converted to Python code. Then, the code snippets are used as input to the theorem prover. Finally, the theorem prover outputs whether the conclusions are True / False / Unknown, based on the premises.

## **H** Distribution of Readability

We show the distribution of readability in Figure 3.

![](_page_13_Figure_7.jpeg)

Figure 3: Dale-Chall Readability Distribution.

**NL Conclusions** 

| 1. A moth is                  | not a butte   | A. Cerura vinula emerges |                             |  |  |
|-------------------------------|---------------|--------------------------|-----------------------------|--|--|
| 2. Butterflies                | have thin a   | from cocoons.            |                             |  |  |
| 3. Moths emerge from cocoons. |               |                          | B. Cerura vinula does not   |  |  |
| 4. Some moths are pests.      |               |                          | have thin antennae.         |  |  |
| 5. Cerura vin                 | ula is a mo   | oth.                     | C. Cerura vinula is a pest. |  |  |
|                               |               |                          |                             |  |  |
| Labels                        | GPT-4         | Fine-tune                |                             |  |  |
| Labels<br>A. True             | GPT-4<br>True | <b>Fine-tune</b> Unknown |                             |  |  |
|                               | True          |                          |                             |  |  |
| A. True                       | True<br>True  | Unknown<br>True          |                             |  |  |

Table 9: A WikiLogic story and model predictions.

Table 10: A HybLogic story and model predictions.

## I Case study

**NL Premises** 

**NL Premises** 

Table 9 shows a story from WikiLogic along with the GPT-4 and RoBERTa-Large predictions. Conclusion A is True given premises 5 and 3. From the premises, it cannot be determined if Cerura vinula has thin antennae or if it is a pest. Thus conclusions B and C are Unknown. GPT-4 predictions are correct for conclusions A and C while RoBERTa

<sup>5</sup>https://stanford-cs221.github.io/spring2022/
assignments/logic/index.html

![](_page_14_Figure_0.jpeg)

Figure 4: Confusion matrices for the results of finetuning RoBERTa-Large and few-shot prompting GPT-4.

predictions are wrong for all conclusions.

Table 10 shows a story from HybLogic with a more complex FOL reasoning process. Inferred from premises 4 and 5, James does not perform better than others. With premises 3, 2 and 1, we know that James is not good at time management. Therefore, conclusion B is False. It cannot be determined if James exercises every week, thus the first conclusion is Unknown. The truth value of  $p \rightarrow q$  is the same as  $\neg p \lor q$ . It is not true that James does not perform better than others. It is also false that James exercises every week and is good at time management. Thus conclusion C is False. For this example, GPT-4 predicted the correct truth value only for conclusion A and RoBERTa made correct predictions for conclusions A and B.

# J Model Performance Analysis

Models have more tendency to predict "True" compared with "False" or "Unknown" labels Confusion matrices in Figure 4 for the fine-tuning and 8-shot NL prompt results both show that LLMs are significantly better at making the correct predictions for conclusions with labels of True than the conclusions with labels of False or Unknown. The accuracy on examples with False or Unknown conclusions is 61.9% with fine-tuning and 54.0% with few-shot prompting. They also tend to make

more predictions of True than the other labels.

Model performance is not affected by the premise ordering. To test if the premise ordering in FOLIO has spurious correlations with the conclusion label which a model can exploit, we shuffle the input premises to evaluate models. We find that accuracy increases or decreases by roughly 1% in most settings compared to our unshuffled premises. This indicates that the ordering of premises in FOLIO examples does not yield significant information about the label, and thus models will not be able to use the premise ordering as a strong heuris-

| Model   | NL    | NL-FOL | FOL   | NL+FOL |
|---------|-------|--------|-------|--------|
| GPT-3.5 | 58.34 | 55.96  | 57.92 | 57.75  |
| GPT-4   | 64.16 | 63.82  | 64.01 | 65.21  |

Table 11: Comparison of the results across different input formats with few-shot prompting. NL, NL-FOL, FOL, NL + FOL stands for NL prompting, execution accuracy of NL-FOL translation, using only FOL in the prompt and using concatenated NL and FOL in the prompt respectively.

tic or statistical feature for its predictions.

Using both NL sentences and FOL formulas in the prompt performs better FOL formulas have a clearer and more straightforward logical structure than NL sentences. Therefore, we test GPT-3.5 and GPT-4 with another two settings for truth value prediction using few-shot prompting: 1) using only FOL formulas in the prompt; 2) using both NL sentences and FOL formulas by concatenating each NL sentence and its annotated FOL statement. As shown in Table 11, the performance slightly increases in the NL+FOL setting for GPT-4 while GPT-3.5 performs worse in both the NL+FOL and the FOL-only settings. In other words, FOL always serves as additional useful information for GPT-4, but not for GPT-3.5 regardless of whether FOL is concatenated with NL. This observation resonates with the finding that GPT-4 performs much better than GPT-3.5 on code-related tasks (Ni et al., 2023).