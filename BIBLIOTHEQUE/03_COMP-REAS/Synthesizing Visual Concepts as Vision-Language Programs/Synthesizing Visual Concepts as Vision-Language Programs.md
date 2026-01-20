# Synthesizing Visual Concepts as Vision-Language Programs

Antonia Wüst<sup>1</sup> Wolfgang Stammer<sup>2</sup> Hikaru Shindo<sup>1</sup> Lukas Helff<sup>1,3</sup>

Devendra Singh Dhami<sup>4</sup> Kristian Kersting<sup>1,3,5</sup>

<sup>1</sup>AIML Lab, TU Darmstadt <sup>2</sup>Max Planck Institute for Informatics, SIC <sup>3</sup>Hessian Center for AI (hessian.AI) <sup>4</sup>Uncertainty in AI Group, TU Eindhoven <sup>5</sup>German Research Center for AI (DFKI)

# **Abstract**

Vision-Language models (VLMs) achieve strong performance on multimodal tasks but often fail at systematic visual reasoning tasks, leading to inconsistent or illogical outputs. Neuro-symbolic methods promise to address this by inducing interpretable logical rules, though they exploit rigid, domain-specific perception modules. We propose Vision-Language Programs (VLP), which combine the perceptual flexibility of VLMs with systematic reasoning of program synthesis. Rather than embedding reasoning inside the VLM. VLP leverages the model to produce structured visual descriptions that are compiled into neurosymbolic programs. The resulting programs execute directly on images, remain consistent with task constraints, and provide human-interpretable explanations that enable easy shortcut mitigation. Experiments on synthetic and real-world datasets demonstrate that VLPs outperform direct and structured prompting, particularly on tasks requiring complex logical reasoning.

# 1. Introduction

Vision–language models (VLMs) have achieved impressive results across multimodal tasks, yet they continue to struggle with visual reasoning. Studies reveal frequent failures in both perception and reasoning, even on relatively simple tasks [11, 16, 25, 26, 42, 47, 50, 51]. *E.g.*, in inductive visual reasoning tasks where models must propose rules that distinguish between image sets (*cf.* Fig. 1), VLMs often fail by generating statements that violate the task constraints. In this example, the VLM proposes the rule "contains candle or candles", incorrectly satisfying one of the negative images. Such errors highlight the gap between pattern recog-

![](_page_0_Figure_10.jpeg)

Figure 1. VLMs cannot reliably perform inductive logic learning from images, failing to capture visual compositions like "candles and birthday cake". Vision-Language Programs (VLP) employ explicit symbolic reasoning to overcome such visual reasoning errors while maintaining perceptual flexibility.

nition and systematic reasoning in VLMs.

Recent work attempts to address this gap through testtime scaling, where models "think" longer via extended chain-of-thought generation [34, 43]. While effective in some cases, this approach is computationally expensive and prone to contradictions or repetitive loops [12, 31, 44, 49]. This raises the question of whether language-based inference alone is sufficient for robust reasoning.

Neuro-symbolic (NeSy) AI offers a promising alternative by integrating neural processing with structured symbolic inference [6, 17, 20, 27, 29, 30, 46]. This paradigm has demonstrated improved robustness, compositional gen-

<sup>&</sup>lt;sup>1</sup>Project page: ml-research.github.io/vision-language-programs

eralization, and interpretability, *e.g.*, over monolithic VLMs under domain shifts [15]. Hereby, program synthesis [9], which induces interpretable and logically consistent programs from examples, provides a particularly natural mechanism for implementing this integration in visual reasoning tasks. However, existing neuro-symbolic approaches for visual reasoning face critical limitations. Methods either require explicit queries to drive program generation, limiting their applicability to inductive reasoning tasks, or depend on domain-specific image encoders, preventing generalization across diverse visual domains.

We therefore propose combining VLMs with program synthesis in form of VISION-LANGUAGE PROGRAMS (VLP) to overcome these shortcomings. Importantly, instead of embedding reasoning inside the VLM, VLP produces structured visual descriptions that can be compiled into symbolic programs, decoupling perception from reasoning. VLP automatically induces symbolic rules from small sets of labeled image examples by first discovering candidate symbols through VLM-based analysis, then defining VLM functions that extract structured representations from images, which can be composed with symbolic reasoning functions in a domain-specific language. This allows the resulting programs to leverage neural perception at execution time while maintaining symbolic interpretability and logical consistency. This dualistic process allows VLP to execute directly on images.

Evaluations on both synthetic and real-world datasets demonstrate that even small VLMs, when embedded in our framework, surpass direct prompting, especially on tasks requiring complex logical reasoning. This hybrid approach thus leverages the perceptual priors of VLMs while enabling symbolic reasoning, marking a crucial step toward models that not only achieve strong performance but also provide transparent, structured decision processes.

To this end, we introduce: (i) VISION-LANGUAGE PROGRAMS, a framework combining VLMs with program synthesis to induce symbolic rules from labeled images without hand-crafted detectors or task-specific queries; (ii) a domain-specific language integrating compositional VLM perception functions with symbolic reasoning operators; (iii) a probabilistic synthesis procedure that discovers and ranks programs by accuracy and likelihood; (iv) comprehensive empirical evidence showing our approach outperforms direct prompting, particularly on logically complex tasks; and (v) analysis demonstrating how programmatic structure enables shortcut detection and mitigation through transparent decision processes.

# 2. Related Work

**Neuro-Symbolic Concept Induction.** Neuro-symbolic AI [6, 17, 27] seeks to integrate the strengths of neural representations with the structure of symbolic reasoning. A

common approach is to extract structured representations from raw perceptual inputs, then perform rule learning over the resulting symbols [29, 30, 35, 39, 46, 48]. These methods typically extract structured representations from raw inputs, then perform rule learning over the resulting symbols, for instance via inductive logic programming (ILP) [29, 30] or probabilistic logic frameworks [20, 32, 33]. While effective for complex synthetic scenes, their applicability to open-world settings is often limited due to reliance on predefined predicates and domain-specific object detectors for constructing intermediate symbolic representations.

Program-based Visual Reasoning. A prominent recent approach to complex visual reasoning leverages programs for systematic image analysis. Frameworks such as VisProg [10], ViperGPT [40], CodeVQA [38], and NeP-Tune [15] employ foundation models to generate executable programs conditioned on visual inputs and natural language instructions or questions. In contrast, our approach (VLP) focuses on inducing programs from labeled visual examples in the absence of task-specific queries, thereby uncovering programmatic representations that explain the conceptual differences between them. Program synthesis [8] has long been successful in rule induction, with early efforts focusing on domains such as list processing and text editing [4, 9]. A benefit of explicit program synthesis over LLM-prompted approaches is the guarantee of syntactically valid and executable programs, eliminating formatting errors that can hinder downstream reasoning. More recent work extends these ideas to abstract reasoning tasks, such as the Abstract Reasoning Corpus [3], by leveraging foundation models to discover higher-level concepts [1, 41]. However, the intersection of program synthesis and foundation models for visual rule induction remains largely unexplored. An initial step in this direction was made by Wüst et al. [46], who proposed a program synthesis approach for visual concept induction, inducing programs from positive and negative visual samples. Their method, however, depends on domainspecific object detectors to convert images into symbolic representations, which limits its generality. In contrast, VLP extends this idea by performing program synthesis directly on natural images, eliminating the need for domainspecific pretraining and enabling broader applicability.

# 3. Vision Language Programs

We introduce VISION-LANGUAGE PROGRAMS (VLP), a framework designed to induce symbolic programs that explain the underlying visual rule from a small set of labeled image examples. VLP combines the perceptual strengths of VLMs with structured reasoning in a Domain-Specific Language (DSL), allowing for explicit, interpretable, and systematic reasoning grounded in visual perception.

VLP operates in three consecutive stages. First, during *symbol grounding* (i), relevant object, property, and

![](_page_2_Figure_0.jpeg)

Figure 2. **Overview of VISION-LANGUAGE PROGRAMS synthesis**. Relevant variables are first discovered from the input examples (i) and used to construct a task-specific DSL, including VLM-based functions (ii). Program synthesis (iii) then searches this space to retrieve the most probable program that also achieves the highest accuracy on the input.

action symbols are grounded for the task at hand. Second, a *Probabilistic Context-Free Grammar* is formed from the Domain-Specific Language (DSL) and the previously ground symbols, including symbolic and neural, VLM-based functions. This forms a probabilistic interface for program search based on visual inputs. Finally, *program synthesis (iii)* searches over the solution space created by the grammar to synthesize a program that best distinguishes positive from negative examples. The resulting output program captures the semantic relations between the input images and can be executed on new test examples to infer new labels. We next detail each stage.

# 3.1. Problem Setup

We formulate inductive visual reasoning as the task of discovering a latent visual rule that explains a set of example images (denoted as few-shot examples in the remainder). Formally, let  $\mathcal{X} = \{(I_1, y_1), (I_2, y_2), \dots, (I_n, y_n)\}$  denote a task, where each  $I_i$  is an input image and  $y_i \in \{0, 1\}$  is the corresponding binary label. A label  $y_i = 1$  indicates that  $I_i$  satisfies the latent visual rule (positive example), whereas  $y_i = 0$  indicates that it does not (negative example). Each task is additionally associated with a set of held-out query samples for evaluation.

# 3.2. Symbol Grounding

The first stage of VLP establishes an interface between continuous visual inputs and discrete symbolic representations. This process, which we refer to as *symbol grounding*, maps perceptual information from images into structured, typeconstrained symbols that form the atomic units for subse-

quent reasoning. We define three fundamental symbol types

$$G = \{ \text{object}, \text{property}, \text{action} \}$$

These types constrain the semantic roles that individual symbols play in downstream program construction. Rather than relying on a fixed vocabulary, the vocabulary is dynamically adjusted based on the task at hand. This preserves generality and adaptation to novel domains and unseen visual compositions. In detail, given a reasoning task  $\mathcal{X}$ , VLP provides task-specific groundings of these abstract types by querying a pretrained VLM  $\mathcal{M}$ . For each symbol type  $G_i \in G$ , the model proposes a set of groundings  $E_i$ :

$$\mathcal{M}(G_i, \mathcal{X}) = E_i, \quad E_i = \{e_1, \dots, e_m\}, \tag{1}$$

where each  $e \in E_i$  denotes an individual grounding for symbol type  $G_i$ . E.g., in Fig. 2, cake and candles ground symbol type object, while colorful, blow, and burn ground property and action symbols, respectively. The process is guided by type-specific queries (cf. Sec. G.2).

The result is a structured pool of symbols  $\mathcal{E} := \{E_1, \ldots, E_{|G|}\}$  that captures the objects, properties, and actions relevant to the current task. This pool acts as the semantic substrate for forming visual concepts.

# 3.3. Vision-Language DSL

The second element of VLP is a Domain-Specific Language (DSL) [5] that formalizes an interface between perception and reasoning. Unlike task-specific DSLs, ours is VLP-specific: it defines a general symbolic interface that remains invariant across domains and tasks. While the grounded symbols capture task-specific semantics, the DSL defines the syntax and functional structures shared across

all tasks. It includes a small set of syntactic primitives such as bool and int, and a dedicated type image, representing the input domain. Beyond these primitives, the DSL defines three function types: *VLM functions, symbolic functions,* and *program operators.* VLM functions map perceptual features from images to symbolic representations, symbolic functions encode logical or arithmetic operations over these symbolic representations, and program operators compose these components into executable reasoning programs.

**VLM functions**  $\mathcal{V}$  equip VLP with a perceptual interface to extract symbolic information from visual inputs using VLMs. Each function  $(v \in \mathcal{V})$  takes as input an image (I) together with one or more sets of ground symbols  $(E_i)$  obtained during symbol grounding (Sec. 3.2), and outputs a nested symbolic representation s:

$$v(I, E_1, \dots, E_m) = s, \tag{2}$$

where  $E_i \in \mathtt{list}[G_i]$  with  $G_i \in \mathcal{G}$  for  $m \geq 1$ , and  $s \in \mathtt{list}[\mathtt{list}[\mathcal{G}]]$ . In essence, VLM functions translate raw visual inputs into structured symbolic representations. For example, the VLM function <code>get\_objects</code> (see Fig. 2) takes an image along with the symbol groundings for *object* and *property*, and extracts a structured *object-property* mapping for the given image. For the first input image (top left in Fig. 2), this mapping could be represented as:

```
[[birthday cake], [candles, colorful]].
```

The output provides a symbolic representation of an image's semantics for downstream reasoning. Since  $\mathcal E$  is constant across task images, we omit explicit symbol inputs and write functions like <code>get\_objects</code> and <code>get\_actions</code> as depending only on the image I in the following.

**Symbolic functions**  $\mathcal{F}$  form the core reasoning primitives of VLP. They operate directly on the symbolic representations s and capture the relationships, attributes, and interactions among them. Each function represents an interpretable reasoning step. For instance, the function <code>exists\_object(s, e)</code> in Fig. 2 evaluates whether an object e appears in the representation s, returning a boolean. Other symbolic functions count objects or properties, verify the presence of specific actions, or symbol combinations. Collectively, these functions form the reasoning vocabulary of VLP, enabling structured queries over visual elements.

**Program operators**  $\mathcal{O}$  specify how reasoning primitives in VLP can be composed. They encompass logical connectives (AND, OR, NOT) and comparison operators (=,<,>), enabling the construction of complex executable programs that represent abstract visual concepts.

# 3.4. DSL to Probabilistic Context-Free Grammar

With semantics defined by *symbol grounding* and syntax specified by the *DSL*, VLP employs a Probabilistic Context-Free Grammar (PCFG) [5] to formalize program synthesis. The PCFG serves as a prior over the space of syntactically valid programs, defining probabilistic production rules from the DSL that guide program search.

Formally, a PCFG is defined as a tuple G=(N,T,R,S,P), where N denotes nonterminal symbols (types), T denotes terminal symbols (grounded symbols), R is the set of production rules, S is the start symbol, and P assigns probabilities to each rule in R. The set of all possible symbol types is given by:

$$\mathcal{T} = G \cup \{\text{image, bool, int}\},\$$

Each type  $\tau \in \mathcal{T}$  corresponds to a nonterminal in N, representing all expressions that evaluate to that type.

**Grammar Construction** The core of the PCFG lies in the production rules R that govern how programs are composed from DSL primitives and grounded symbols. Each type corresponds to one or more productions rules that define which expressions from the DSL lead to this type. A general production rule  $r \in R$  takes the form:

$$r: \tau_i \to f(\tau_1, \tau_2, \dots, \tau_k),$$

where  $f \in \mathcal{V} \cup \mathcal{F} \cup \mathcal{O} \cup \mathcal{E}$  is a DSL element producing an output of type  $\tau_i$ , and each  $\tau_j$  denotes the expected argument type. Applying these rules recursively expands the start symbol S (of output type bool) until all nonterminals are resolved, yielding a complete, type-consistent program.

Enumerating all such rules under  $\mathcal{T}$  defines the complete space of syntactically valid, type-consistent programs under R. For example, several valid rules for constructing boolean expressions would be:

```
bool → and(bool, bool)
bool → or(bool, bool)
bool → exists_object(s, object).
```

Likewise, perceptual rules may link to visual grounding functions, such as:

```
\texttt{s} \ \rightarrow \ \texttt{get\_objects} \, (\texttt{IMG}) \\ \texttt{object} \ \rightarrow \ \texttt{birthday} \ \ \texttt{cake}
```

Repeatedly applying such rules expands bool into a complete, type-consistent program operating on visual inputs.

**Probabilistic Weighting.** Each production rule  $r \in R$  is associated with a probability P(r) that determines how likely specific functions, operators, or grounded symbols  $\mathcal{E}$ 

are to be selected during synthesis. Since VLP is training-free, most rules are assigned uniform probabilities. The only exception concerns the symbols  $\mathcal E$  discovered by the VLM  $\mathcal M$ . Here we leverage their relative occurrence frequencies in positive and negative examples for weighting their likelihood. This employs a simple yet effective form of data-driven inductive bias. We estimate these weights using an occurrence-based distribution, where a symbol  $e \in \mathcal E$  that occurs more frequently in positive examples is assigned higher probability:

$$P(e) \propto \frac{n_{\rm pos}}{N_{\rm pos}} \cdot \frac{n_{\rm pos}}{n_{\rm pos} + n_{\rm neg}},$$
 (3)

where  $(n_{\rm pos})$  and  $(n_{\rm neg})$  denote the number of positive and negative occurrences respectively, and  $(N_{\rm pos})$  is the total count of positive examples. When  $(n_{\rm pos}=0)$ , a constant  $(\epsilon=0.01)$  is used to avoid zero-probability assignments.

# 3.5. Program Synthesis

Given the PCFG, VLP performs program synthesis by searching for an executable program p that best explains the task  $\mathcal{X}$ . Each program transforms an input image  $I_i$  into a boolean prediction  $\hat{y}_i = p(I_i)$ , whose correctness is evaluated against the ground-truth label  $y_i$ .

**Program search.** The search explores the space of candidate programs defined by the grammar. For each candidate p, we compute its accuracy on  $\mathcal{X}$ ,

$$Acc(p) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1} [p(I_i) = y_i],$$

and its probability under the PCFG, P(p), given by the product of the probabilities of the rules used to construct p. To improve efficiency, outputs of all VLM functions  $\mathcal{V}$  are precomputed for every  $I_i \in \mathcal{X}$ .

Candidate programs are ranked by a two-level criterion: 1. *Primary:* accuracy Acc(p); 2. *Secondary:* probability P(p) to break ties. The top-ranked program  $p^*$  is selected as the final solution (cf. Fig. 2, bottom left).

# 4. Experiments

In the following, we evaluate VISION-LANGUAGE PRO-GRAMS on several datasets and tasks to assess neurosymbolic VLP across various visual reasoning settings. We hereby address the following research questions:

- Can VLP improve concept learning performance over VLMs across varying visual domains? (**RQ1**)
- Can VLPs with non-reasoning models improve over dedicated reasoning models? (RQ2)
- Do VLP models exhibit advantages over VLMs with an increased number of training samples? (RQ3)

• Do VLP facilitate knowledge incorporation, *e.g.*, for shortcut mitigation? (**RQ4**)

**Data.** For the evaluation of our method, we use a set of different datasets across multiple domains that are based on synthetic and real-world images. Specifically, we evaluate on the datasets Bongard-HOI [13], Bongard-OpenWorld [45] and Bongard-RWR[19], which are based on real-world images and incorporate a diverse set of visual concepts. For a real-world dataset that provides more complex logical rules, we utilize COCOLogic [37] and create 10 tasks from it, one for each class. For the synthetic dataset, we use CLEVR-Hans3 [35], where we leverage the three classes to construct three tasks with complex logical rules from it. The Bongard datasets provide 12 samples from which the target rule should be induced, 6 positives and 6 negatives. For the other two datasets, we take 20 balanced support samples. Additional details are provided in Suppl. A.

**Experimental Setup.** We incorporate and compare to a selection of open source VLMs with varying sizes. Namely, we utilize InternVL3-8B and InternVL3-14B [2], Kimi-VL-A3B-Instruct [18], as well as Qwen2.5-VL-72B [23] and Qwen3-VL-30B-A3B-Instruct [24]. We prompt a model three times using sampling-based generation and perform program synthesis using the Heap Search Algorithm from [5] with a 10-second budget. The maximum program depth is 4 for Bongard-OpenWorld and Bongard HOI, and 6 for COCOLogic and CLEVR-Hans3. The DSL used for the experiments is provided in Suppl. F. In the context of **RQ1** we compare the base models with and without VLP integration. In the context of RQ2 we compare instruction-tuned models with VLP to dedicated reasoning models Kimi-VL-A3B-Thinking [18], Qwen3-VL-30B-A3B-Thinking [24] and gpt-5 [21]. For **RQ3**, we evaluate all models from RQ1 using an increased number of support samples and average their results. For evaluations regarding knowledge incorporation (RQ4) we use CLEVR-Hans3 to investigate how DSL edits can be used to improve performance and mitigate shortcut learning. Across all evaluations, model performance is measured using balanced accuracy, reflecting each model's ability to correctly classify the query (test) images.

**VLPs boost VLMs across domains (RQ1).** In our first evaluation, we investigate the potential of our neurosymbolic VLP framework to leverage the power of VLMs in diverse visual concept learning tasks. For this, we compare the performance of five different base VLMs with and without VLP processing on five datasets.

We observe in Tab. 1 that VLPs obtain substantially higher results across all models, with improvements of up to 13.5% on average across all datasets. Interestingly, model size appears to have minimal effect on VLP performance; however, smaller models (e.g., InternVL3-8B and Qwen2.5-VL-7B) tend to benefit most from VLP-based

Table 1. **Comparison of base VLMs and VLMs with VLP (averaged over three runs).** Balanced accuracy (%) on 6-shot Bongard tasks and 10-shot logical reasoning benchmarks. Improvements (green / gray) denote changes relative to the baseline. Best results per model are shown in **bold**; overall best results per dataset column are <u>underlined</u>.

| Model                            | Avg.                       | Bongard Tasks (6-shot)     |                         |                         | Logical Reasoning (10-shot) |                             |
|----------------------------------|----------------------------|----------------------------|-------------------------|-------------------------|-----------------------------|-----------------------------|
|                                  |                            | Bongard-HOI                | Bongard-OW              | Bongard-RWR             | COCOLogic                   | CLEVR-Hans3                 |
| InternVL3-8B                     | 57.4                       | 60.5                       | 59.2                    | 47.2                    | 71.5                        | 48.3                        |
| w/ VLP                           | <b>70.9</b> (+13.5)        | <b>77.7</b> (+17.2)        | <b>67.5</b> (+8.3)      | <b>53.9</b> (+6.7)      | <b>81.0</b> (+9.5)          | <b>74.4</b> (+26.1)         |
| InternVL3-14B                    | 61.5                       | 66.9                       | 62.5                    | 51.4                    | <b>78.4</b> 76.7 (-1.7)     | 48.3                        |
| w/ VLP                           | <b>68.3</b> (+6.8)         | <b>74.3</b> (+7.4)         | <b>64.7</b> (+2.2)      | <b>52.8</b> (+1.4)      |                             | <b>73.3</b> (+25.0)         |
| Kimi-VL-A3B-Instruct             | 58.5                       | 59.8                       | 58.6                    | 46.4                    | <b>77.9</b> 70.1 (-7.8)     | 50.0                        |
| w/ VLP                           | <b>65.5</b> (+7.0)         | <b>69.4</b> (+9.6)         | <b>59.4</b> (+0.8)      | <b>52.5</b> (+6.1)      |                             | <b>76.1</b> (+26.1)         |
| Qwen2.5-VL-7B-Instruct<br>w/ VLP | 60.1<br><b>69.5</b> (+9.4) | 65.2<br><b>68.8</b> (+3.6) | <b>66.2</b> 62.9 (-3.3) | <b>49.7</b> 49.2 (-0.5) | 73.2<br><b>80.5</b> (+7.3)  | 46.1<br><b>86.1</b> (+40.0) |
| Qwen3-VL-30B-A3B-Instruct        | 63.4                       | 69.0                       | 68.5                    | 55.8                    | 73.9                        | 50.0                        |
| w/ VLP                           | <b>68.9</b> (+5.5)         | <b>74.5</b> (+5.5)         | 66.3 (-2.2)             | <b>58.3</b> (+2.5)      | <b>79.1</b> (+5.2)          | <b>66.1</b> (+16.1)         |

processing. The strongest improvements across all models occur on the compositionally complex CLEVR-Hans3 dataset, where the synthetic images are likely more out-of-distribution for pretrained VLMs, usually including 5 to 10 objects. This pattern suggests that structured reasoning offers greater advantages when perceptual uncertainty is higher. Notably, none of the model encoders were specifically finetuned on these datasets, demonstrating that VLP grants domain-independent flexibility.

Fig. 3 illustrates this effect of VLP's decoupling of perception from reasoning on a Bongard-RWR task. When prompted directly, the base Qwen3-VL model proposes a complex rule but fails to identify the simple underlying concept of "round vs. non-round objects," ultimately classifying both test images as positive. In contrast, the same base model with VLP successfully discovers the program

$$p^* = (exists\_property (get\_objects IMG) round)$$

by first identifying individual objects in each image, then inferring through program search that all positive images contain objects with the property round. The resulting Vision-Language Program  $p^*$  correctly classifies all test images.

The only dataset where VLP does not achieve the overall best performance is Bongard-OpenWorld. Upon examining the failure cases on this dataset, we identified numerous annotation errors (*cf.* Suppl. E). In such cases, standard VLM prompting tends to generate plausible but loosely defined rules that accommodate these inconsistencies. In contrast, VLP 's structured approach induces programs that remain logically consistent with the (incorrectly labeled) few-shot examples, but consequently fail the query samples.

Within the context of RQ1, we additionally investigate whether the observed improvements arise from the use of structured symbolic image representations

![](_page_5_Figure_8.jpeg)

Figure 3. **Qualitative comparison on Bongard-RWR.** Direct VLM (Qwen3) prompting produces an incorrect rule about "abundance", misclassifying a query image. Qwen3 w/ VLP discovers a correct program that identifies round objects and achieves perfect query classification accuracy.

(e.g., get\_objects, get\_actions) by comparing LLM-based reasoning on symbolic inputs with VLP's reasoning in Suppl. Sec. C.1. The results indicate that, in fact, VLP's performance gains are primarily driven by the symbolic search process rather than the representation format. Conclusively, the induced rules by VLP are more reliable than rules obtained by prompting the model directly, even with structured symbolic representations at hand.

In summary, across these evaluations, we conclude that our neuro-symbolic VLP framework consistently enhances the reasoning capabilities of VLMs by effectively combining visual perception with symbolic search.

Table 2. Comparison between *thinking* and *non-thinking* models with VLP. VLP performs comparably or better than thinking models while using significantly fewer tokens, and can even build upon thinking-based approaches for further improvements.

|                    | COCOLogic |         | CLEV | R-Hans3 |
|--------------------|-----------|---------|------|---------|
|                    | Acc       | Tokens  | Acc  | Tokens  |
| Kimi w/ Think      | 52.2      | 106,446 | 30.0 | 46,941  |
| Kimi w/ VLP        | 68.3      | 43,958  | 83.3 | 6,584   |
| Qwen3 w/ Think     | 81.8      | 108,346 | 46.7 | 106,922 |
| Qwen3 w/ VLP       | 78.8      | 22,341  | 68.3 | 5,052   |
| GPT-5 w/ Think     | 78.5      | 115,813 | 65.0 | 98,046  |
| GPT-5 w/ VLP       | 81.8      | 17,404  | 68.3 | 8,058   |
| GPT-5 w/ Think+VLP | 84.3      | 183,387 | 70.0 | 84,642  |

**Symbolic reasoning of VLPs outperforms VLM-based reasoning (RQ2).** In our next evaluation, we compare the performance of VLP with instruction-tuned base VLMs to dedicated "reasoning" models. We focus on COCOLogic and CLEVR-Hans3 for this comparison, as these datasets exhibit the highest compositional and logical complexity, making them ideal testbeds for distinguishing structured symbolic reasoning from pure language-based inference. We compare Kimi-VL-A3B and Qwen3-VL-30B-Instruct with VLP to their "Thinking" counterparts. Additionally, we evaluate gpt-5 (high reasoning effort) against gpt-5-chat (no reasoning) and gpt-5 (low reasoning effort), both with VLP. We conduct the experiments with one run per model.

The results in Tab. 2 show that integrating VLP with non-reasoning models yields substantial improvements over dedicated reasoning models in all cases except Qwen3 on COCOLogic. Moreover, VLP-based reasoning requires substantially fewer tokens than dedicated reasoning models, despite executing more prompts, thereby demonstrating greater computational efficiency. Interestingly, integrating gpt-5 with low reasoning effort into VLP achieves improvements over both gpt-5 with Thinking and gpt-5-chat with VLP, while increasing token count only moderately on COCOLogic and even reducing it on CLEVR-Hans3. This suggests that VLP reasoning provides an orthogonal enhancement to the thinking mode of VLMs, yielding complementary performance benefits.

Overall, our findings suggest that VLP achieves strong reasoning performance while maintaining computational efficiency. We thus answer RQ2 affirmatively.

VLPs reliably integrate larger numbers of samples (RQ3). In many cases, having more samples for a specific task can provide valuable additional evidence for a concept and help refine it by excluding alternative explanations or shortcuts. In our third set of evaluations, we therefore examine how VLP handles larger numbers of input images,

![](_page_6_Figure_6.jpeg)

Figure 4. VLP performance improves as more input images are provided, in contrast to baselines, which stagnate or decline. Results are aggregated over models from Table 1.

particularly compared to the base VLM's image processing capacity. As shown in Fig. 4, we increase the number of input images per task from 20 to 100 across the Bongard-HOI, COCOLogic, and CLEVR-Hans3 datasets. For a fair comparison, the number of test images remains fixed in all runs, and we only consider concepts with sufficient support samples (*cf.* Suppl. A). For each dataset, we report the average performance across all VLM models from Tab. 1, comparing the base models with and without VLP.

We observe that performance improves for models using VLPs, as the additional evidence helps identify more accurate programs. For example, for the concept "carrying surfboard" in Bongard-HOI (cf. Suppl. Fig. 6), InternVL3-14B with 20 task examples discovers the program ((exists\_action (get\_actions IMG) holding)), which does not specify the object being held but still covers 96% of the few-shot examples. With 100 examples, however, the model retrieves a more precise program, including the actions holding and walking in combination with the object surfboard (cf. Sec. B.1).

In contrast, the base VLMs do not benefit from an increased number of examples. Since they process all input images jointly, individual samples likely receive less attention, resulting in overly abstract learned representations. *E.g.*, for the previous *surfing* task with 20 images, InternVL3-14B produces the concept "*The activity involves walking with or actively surfing on a surfboard in a beach or ocean setting*," which is overly general and mistakenly includes negative examples (*surfing* instead of *carrying*).

Overall, these results highlight that VLP can effectively leverage larger sets of input images by synthesizing more specific and semantically grounded programs, while base VLMs struggle to capitalize on the additional data. We hence answer RQ3 affirmatively.

VLPs enable interaction with the Solution Space of the Program (RQ4). An important feature of VLPs is their inherent modularity, which allows users to inspect and in-

![](_page_7_Figure_0.jpeg)

Figure 5. VLP performance on CLEVR-Hans3 with DSL edits. For InternVL3 size-related VLM functions were added, for Qwen3 shortcut-related colors (red, gold) removed.

teract with individual processing steps. This enables targeted debugging of failure cases and refinement of solutions through step-specific feedback. To investigate how this facilitates intuitive interactions with the task's solution space, valuable for model debugging and shortcut mitigation [7, 28, 35, 36], we investigate the solutions of the models InternVL3-8B and Qwen3-VL on the dataset CLEVR-Hans3 from Tab. 1.

Here, we observed that, *e.g.*, InternVL3-8B seldom considers the size of the objects, which is a relevant property for the rules. Looking at the grounded properties and the object-property representations, we observed that even though the size properties, *small* and *large*, are discovered, they are not always used in the object-property representation obtained by get\_objects. An explanation for this could be that the concept *size* is very relative and depends on the context. Fortunately, thanks to the flexibility of VLP one can very simply add VLM functions to the DSL that directly ask for the size of a given object (*cf.* Sec. G.4) in *relation* to the other objects. As Fig. 5 (left) highlights, by incorporating these functions into the DSL the performance on the query samples improves substantially, *i.e.*, InternVL3-14B can now utilize this bias to achieve 96% accuracy.

In contrast, when inspecting the results of Qwen3-VL, we identified that the VLPs indeed make use of the VLM-proposed properties, however in some cases a VLP falsely considers the colors "red" and "gold" as relevant properties. Manual inspection of corresponding images (e.g. of task #2 in Suppl. Fig. 11), indeed identified these features as potential shortcuts for the task. Equipped with such discoveries a human user can easily interact with the DSL by removing "red" and "gold" from the available properties and by this targeted revision improve the model's performance by 13.3% (cf. Fig. 5 (right)).

These evaluations demonstrated that even with limited data, solution quality can be increased by incorporating additional task knowledge, *e.g.*, in a human-in-the-loop setting. We therefore conclude RQ4 positively.

# 5. Discussion

Across our evaluations, we have observed that VLP can be used to effectively improve visual inductive reasoning in terms of predictive performance improvements, but also failure analysis and model debugging. Below, we discuss additional considerations of our evaluations.

First, we investigated the impact of VLP's occurrencebased symbol weighting in Sec. C.2. The results confirm its advantage over default uniform weighting, particularly in compositional settings. Additionally, our comparative analysis in context of RQ1 revealed that InternVL3-8B w/ VLP achieved substantially stronger results than other configurations (Tab. 1), which we attribute to InternVL3-8B producing higher quality symbolic image representations, cf. Suppl. D). However, our failure analysis indicates that all models occasionally exhibited errors that limit VLP's overall effectiveness (cf. Suppl. D). These include formatting errors (e.g., Kimi produced malformed symbolic representations in approximately 13% of COCOLogic cases) and incomplete property descriptions, where VLMs omit taskrelevant properties despite being capable of perceiving them (e.g., omitting "size" for CLEVR-Hans3, cf. Sec. D.2). Such perception failures constrain VLP performance by limiting available symbolic knowledge for reasoning. Notably, however, unlike end-to-end VLMs, where misclassifications remain opaque, VLPs enable tracing errors back to individual images within a task, providing actionable insights for model debugging and improvement.

Looking forward to improving perceptual grounding quality, VLMs could be prompted for each symbol individually rather than all at once (e.g., get\_objects), though this would substantially increase prompt overhead. We view the current approach as an effective trade-off, with future work potentially pre-selecting promising symbols during search to reduce computational costs. Additionally, VLP currently lacks explicit object representations, limiting its effectiveness for object-centric concepts such as spatial relations. Extending VLP with structured object representations would enable applications in complex expert domains like healthcare or mechanical engineering, where interactive human feedback during synthesis could prove particularly valuable. Future work could also investigate VLM robustness to out-of-distribution data and explore multi-model program synthesis, e.g., combining different VLMs or integrating classical algorithms within a unified DSL.

# 6. Conclusion

In this work, we introduced VISION-LANGUAGE PRO-GRAMS (VLP), a framework that integrates the perceptual flexibility of VLMs with the systematic reasoning capabilities of program synthesis. By leveraging VLMs to generate structured symbolic descriptions that can be compiled into executable programs, VLP enable explicit composition, disambiguation, and, importantly, human-level interpretability beyond the natural language outputs of VLMs. Our empirical results across synthetic and real-world datasets show that VLP consistently improve generalization and predictive performance compared to direct prompting and structured prompting, even without the need for domain-specific encoders. Our work demonstrates that VLMs can be naturally integrated into a programmatic framework as callable functions rather than monolithic end-to-end predictors, bringing together the richness of neural vision-language representations with the reliability and controllability of symbolic program synthesis.

# Acknowledgments

This work was supported by the Priority Program (SPP) 2422 in the subproject "Optimization of active surface design of high-speed progressive tools using machine and deep learning algorithms" funded by the German Research Foundation (DFG). The Eindhoven University of Technology authors received support from their Department of Mathematics and Computer Science and the Eindhoven Artificial Intelligence Systems Institute. Furthermore, this work has benefited from early stages of the Cluster of Excellence "Reasonable AI" funded by the German Research Foundation (DFG) under Germany's Excellence Strategy (EXC-3057), funding will begin in 2026.

We gratefully acknowledge support from the hessian.AI Service Center (funded by the Federal Ministry of Research, Technology and Space, BMFTR, grant no. 16IS22091) and the hessian.AI Innovation Lab (funded by the Hessian Ministry for Digital Strategy and Innovation, grant no. S-DIW04/0013/003).

# References

- [1] Shraddha Barke, Emmanuel Anaya Gonzalez, Saketh Ram Kasibatla, Taylor Berg-Kirkpatrick, and Nadia Polikarpova. Hysynth: Context-free LLM approximation for guiding program synthesis. Advances in Neural Information Processing Systems, 37:15612–15645, 2024. 2
- [2] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, and Lewei Lu. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In *IEEE/CVF* Conference on Computer Vision and Pattern Recognition, 2024. 5
- [3] François Chollet. On the measure of intelligence. *arXiv* preprint arXiv:1911.01547, 2019. 2
- [4] Kevin Ellis, Lionel Wong, Maxwell Nye, Mathias Sable-Meyer, Luc Cary, Lore Anaya Pozo, Luke Hewitt, Armando Solar-Lezama, and Joshua B. Tenenbaum. Dreamcoder: growing generalizable, interpretable knowledge with wake-

- sleep Bayesian program learning. *Philosophical Transactions of the Royal Society A*, 381(2251):20220050, 2023. 2
- [5] Nathanaël Fijalkow, Guillaume Lagarde, Théo Matricon, Kevin Ellis, Pierre Ohlmann, and Akarsh Nayan Potta. Scaling neural program synthesis with distribution-based search. In AAAI Conference on Artificial Intelligence, 2022. 3, 4, 5
- [6] Artur d'Avila Garcez, Tarek R. Besold, Luc De Raedt, Peter Földiak, Pascal Hitzler, Thomas Icard, Kai-Uwe Kühnberger, Luis C. Lamb, Risto Miikkulainen, and Daniel L. Silver. Neural-symbolic learning and reasoning: contributions and challenges. In AAAI Spring Symposium Series, 2015. 1, 2
- [7] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard S. Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2020. 8
- [8] Sumit Gulwani. Automating string processing in spreadsheets using input-output examples. ACM SIGPLAN Notices, 46(1):317–330, 2011. 2
- [9] Sumit Gulwani, Oleksandr Polozov, and Rishabh Singh. Program synthesis. Foundations and Trends in Programming Languages, 4(1-2):1–119, 2017.
- [10] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. 2
- [11] Lukas Helff, Wolfgang Stammer, Hikaru Shindo, Devendra Singh Dhami, and Kristian Kersting. V-LoL: A diagnostic dataset for visual logical learning. *Journal of Data-centric Machine Learning Research*, 2024. 1
- [12] Lukas Helff, Ahmad Omar, Felix Friedrich, Antonia Wüst, Hikaru Shindo, Rupert Mitchell, Tim Woydt, Patrick Schramowski, Wolfgang Stammer, and Kristian Kersting. SLR: Automated synthesis for scalable logical reasoning. arXiv preprint arXiv:2506.15787, 2025.
- [13] Huaizu Jiang, Xiaojian Ma, Weili Nie, Zhiding Yu, Yuke Zhu, and Anima Anandkumar. Bongard-HOI: Benchmarking few-shot visual reasoning for human-object interactions. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2022. 5, 1
- [14] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pages 2901–2910, 2017. 1
- [15] Danial Kamali and Parisa Kordjamshidi. NePTune: A neuropythonic framework for tunable compositional reasoning on vision-language. arXiv preprint arXiv:2509.25757, 2025. 2
- [16] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What's "up" with vision-language models? Investigating their struggle with spatial reasoning. In *Conference on Empirical Methods in Natural Language Processing*, 2023. 1
- [17] Henry Kautz. The third AI summer: AAAI Robert S. Engelmore memorial lecture. AI Magazine, 2022. 1, 2
- [18] Kimi Team. Kimi-VL technical report, 2025. 5

- [19] Mikołaj Małkiński, Szymon Pawlonka, and Jacek Mańdziuk. Reasoning limitations of multimodal large language models. a case study of bongard problems. In Forty-second International Conference on Machine Learning, 2025. 5
- [20] Robin Manhaeve, Sebastijan Dumancic, Angelika Kimmig, Thomas Demeester, and Luc De Raedt. DeepProbLog: Neural probabilistic logic programming. In Advances in Neural Information Processing Systems, 2018. 1, 2
- [21] OpenAI. GPT-5. https://openai.com/index/ introducing-gpt-5/, 2025. Accessed: 2025-11-04.
- [22] Szymon Pawlonka, Mikolaj Malkinski, and Jacek Mandziuk. Bongard-RWR+: Real-world representations of fine-grained concepts in bongard problems. *CoRR*, abs/2508.12026, 2025. 1, 11
- [23] Owen Team. Owen2.5-VL, 2025. 5
- [24] Qwen Team. Qwen3 technical report, 2025. 5
- [25] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Asian Conference on Computer Vision, 2024.
- [26] Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, and Katerina Fragkiadaki. Grounded reinforcement learning for visual reasoning. arXiv preprint arXiv:2505.23678, 2025. 1
- [27] Md Kamruzzaman Sarker, Lu Zhou, Aaron Eberhart, and Pascal Hitzler. Neuro-symbolic artificial intelligence. AI Communications, 34(3):197–209, 2021. 1, 2
- [28] Patrick Schramowski, Wolfgang Stammer, Stefano Teso, Anna Brugger, Franziska Herbert, Xiaoting Shao, Hans-Georg Luigs, Anne-Katrin Mahlein, and Kristian Kersting. Making deep neural networks right for the right scientific reasons by interacting with their explanations. *Nature Machine Intelligence*, 2(8):476–486, 2020. 8
- [29] Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, and Kristian Kersting. αILP: thinking visual scenes as differentiable logic programs. *Machine Learning*, 2023. 1, 2
- [30] Hikaru Shindo, Viktor Pfanschilling, Devendra Singh Dhami, and Kristian Kersting. Learning differentiable logic programs for abstract visual reasoning. *Machine Learning*, 2024. 1, 2
- [31] Parshin Shojaee, Iman Mirzadeh, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity. arXiv preprint arXiv:2506.06941, 2025. 1
- [32] Arseny Skryagin, Wolfgang Stammer, Daniel Ochs, Devendra Singh Dhami, and Kristian Kersting. Neural-probabilistic answer set programming. In *International Conference on Principles of Knowledge Representation and Reasoning*, pages 463–473, 2022. 2
- [33] Arseny Skryagin, Daniel Ochs, Devendra Singh Dhami, and Kristian Kersting. Scalable neural-probabilistic answer set programming. *Journal of Artificial Intelligence Research*, 2023. 2

- [34] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024.
- [35] Wolfgang Stammer, Patrick Schramowski, and Kristian Kersting. Right for the right concept: Revising neurosymbolic concepts by interacting with their explanations. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 2021. 2, 5, 8, 1
- [36] David Steinmann, Felix Divo, Maurice Kraus, Antonia Wüst, Lukas Struppek, Felix Friedrich, and Kristian Kersting. Navigating shortcuts, spurious correlations, and confounders: From origins via detection to mitigation. *CoRR*, abs/2412.05152, 2024. 8
- [37] David Steinmann, Wolfgang Stammer, Antonia Wüst, and Kristian Kersting. Object centric concept bottlenecks. Advances in Neural Information Processing Systems, 2025. 5,
- [38] Sanjay Subramanian, Medhini Narasimhan, Kushal Khangaonkar, Kevin Yang, Arsha Nagrani, Cordelia Schmid, Andy Zeng, Trevor Darrell, and Dan Klein. Modular visual question answering via code generation. In Annual Meeting of the Association for Computational Linguistics, 2023. 2
- [39] Vishal Sunder, Ashwin Srinivasan, Lovekesh Vig, Gautam Shroff, and Rohit Rahul. One-shot information extraction from document images using neuro-deductive program synthesis. In *International Workshop on Neural-Symbolic Learning and Reasoning*, 2019. 2
- [40] Dídac Surís, Sachit Menon, and Carl Vondrick. ViperGPT: Visual inference via python execution for reasoning. In IEEE/CVF International Conference on Computer Vision, 2023. 2
- [41] Ruocheng Wang, Eric Zelikman, Gabriel Poesia, Yewen Pu, Nick Haber, and Noah D. Goodman. Hypothesis search: Inductive reasoning with language models. In *International Conference on Learning Representations*, 2024. 2
- [42] Yiqi Wang, Wentao Chen, Xiaotian Han, Xudong Lin, Haiteng Zhao, Yongfei Liu, Bohan Zhai, Jianbo Yuan, Quanzeng You, and Hongxia Yang. Exploring the reasoning abilities of multimodal large language models (MLLMs): A comprehensive survey on emerging trends in multimodal reasoning. arXiv preprint arXiv:2401.06805, 2024.
- [43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. *Advances in Neural Information Processing Systems*, 2022. 1
- [44] Guojun Wu. It's not that simple. An analysis of simple testtime scaling. arXiv preprint arXiv:2507.14419, 2025. 1
- [45] Rujie Wu, Xiaojian Ma, Zhenliang Zhang, Wei Wang, Qing Li, Song-Chun Zhu, and Yizhou Wang. Bongard-OpenWorld: Few-shot reasoning for free-form visual concepts in the real world. In *International Conference on Learning Representations*, 2024. 5, 1
- [46] Antonia Wüst, Wolfgang Stammer, Quentin Delfosse, Devendra Singh Dhami, and Kristian Kersting. Pix2Code:

- Learning to compose neural visual concepts as programs. In *Conference on Uncertainty in Artificial Intelligence*, 2024. 1, 2
- [47] Antonia Wüst, Tim Tobiasch, Lukas Helff, Inga Ibs, Wolfgang Stammer, Devendra Singh Dhami, Constantin A. Rothkopf, and Kristian Kersting. Bongard in wonderland: Visual puzzles that still make AI go mad? In *International Conference on Machine Learning*, 2025. 1, 11
- [48] Yuan Yang and Le Song. Learn to explain efficiently via neural logic inductive learning. In *International Conference* on Learning Representations, 2020. 2
- [49] Zhiyuan Zeng, Qinyuan Cheng, Zhangyue Yin, Yunhua Zhou, and Xipeng Qiu. Revisiting the test-time scaling of o1-like models: Do they truly possess test-time scaling capabilities? In Annual Meeting of the Association for Computational Linguistics, 2025. 1
- [50] Yizhe Zhang, He Bai, Ruixiang Zhang, Jiatao Gu, Shuangfei Zhai, Josh Susskind, and Navdeep Jaitly. How far are we from intelligent visual deductive reasoning? *ICLR Workshop: How Far Are We From AGI*, 2024. 1
- [51] Kankan Zhou, Eason Lai, Wei Bin Au Yeong, Kyriakos Mouratidis, and Jing Jiang. Rome: Evaluating pre-trained vision-language models on reasoning beyond visual common sense. Findings of the Association for Computational Linguistics (EMNLP), 2023.

# Synthesizing Visual Concepts as Vision-Language Programs Supplementary Material

The following appendix provides supplementary materials referenced in the main text. We begin with an overview of the datasets used in the evaluations (Suppl. A), followed by a discussion of qualitative examples (Suppl. B). Additional supporting experiments are presented in Suppl. C, and failure cases are analyzed in Suppl. D. We discuss limitations of the dataset Bongard-OpenWorld in Suppl. E. Finally, we describe the DSL used for VLP (Suppl. F) and list the prompts employed in the experiments (Suppl. G).

# A. Datasets

In the following, we present the datasets used in the evaluations in more detail, along with representative samples.

**Bongard-HOI.** The Bongard-HOI dataset [13] is one of the real-world image datasets used for evaluating our method, specifically focusing on logical rules related to Human-Object Interactions (HOI). This dataset requires the model to induce rules based on how *people are interacting with objects* in an image. Like all Bongard datasets used, each task provides 12 samples from which the target rule should be induced: 6 positive examples that conform to the latent rule (the rule set) and 6 negative examples that violate the rule (the anti-rule set). We evaluate on all 166 test concepts of the four different test splits. For Tab. 1, we take the first sample of each rule. For RQ3 we reduce the test concepts to 67, only keeping those concepts that have enough few-shot examples (50 or more for positive and negative set respectively). To achieve this, we collect all available images per rule and discard the problems that have fewer than 100 unique few-shot examples. The number of test samples is set to 4. We ensure that test examples are not present in the few-shot examples. This reduces the dataset from 166 to 67 problems. An example concept that is tested in RQ1 and RQ3 both is depicted in Fig. 6.

**Bongard-OpenWorld.** The Bongard-OpenWorld dataset [45] is based on real-world images and concepts, designed to extend the challenge of rule induction to broader, open-world concepts and complex relational patterns. It shares the fundamental Bongard problem structure, where the goal is to induce a hidden logical rule from a small, balanced set of visual examples. Each task consists of 12 samples in total: 6 positive examples conforming to the latent rule, and 6 negative examples that violate it. An example is depicted in Fig. 7. We evaluate our method on all 200 test samples of the dataset.

**Bongard-RWR** The Bongard-RWR dataset [22] is also based on real-world images, specifically focusing on abstract visual concepts derived from the original Bongard problems. It is utilized to test the induction of abstract and complex rules within natural, diverse imagery. Following the standard format, each task instance provides 12 samples (6 positives and 6 negatives) from which the underlying logical rule must be determined. We evaluate on all 60 concepts of the dataset and take the first sample per concept (same as Bongard-HOI). Examples of the dataset are provided in Fig. 8 and Fig. 9.

**COCOLogic.** The COCOLogic dataset [37] is selected for its inclusion of complex logical rules applied to real-world images. Built upon the widely used COCO dataset, it introduces logic-based classification tasks that require reasoning over object co-presence and object counts. We construct ten distinct tasks from COCOLogic, one for each dataset class. Positive samples are drawn from the target class to ensure all relevant objects appear at least once in the few-shot examples, while negative samples are taken from the remaining classes. Each few-shot task includes 20 balanced samples. For testing, we select up to ten query samples per task for both positive and negative classes, although some classes contain fewer available examples. For RQ3, similar to Bongard-HOI, we use a subset of 8 tasks to ensure a sufficient number of few-shot examples. Overall, COCOLogic provides a challenging benchmark for evaluating model performance on complex logical reasoning tasks within diverse, natural image contexts.

**CLEVR-Hans3** For the synthetic dataset component, we utilize CLEVR-Hans3 [35]. This dataset is a variant of the CLEVR [14] dataset, specifically constructed to enforce complex logical rules. We leverage the three primary classes present in CLEVR-Hans3 to construct three corresponding tasks, each requiring the induction of complex logical rules related to object color, shape, size, and material. For the support set, we take 20 balanced samples per task, same for the query samples. We utilize the original confounded validation set of CLEVR-Hans3 in our evaluations as a test set, thus turning the originally confounded classification task into a standard one. An example is provided in Fig. 11.

## Bongard HOI Task 9 GT: "carry surfboard"

![](_page_12_Figure_1.jpeg)

Figure 6. Example task from Bongard-HOI (Task 9, GT: "carry surfboard"). Bongard-HOI focuses on human-object interactions in natural images, requiring models to distinguish specific actions or relationships between people and objects. Positive examples show people carrying surfboards, while negative examples depict other surfboard-related activities such as surfing or standing near surfboards without carrying them.

# **B.** Qualitative Examples

In the following we show an example of results from VLP on a Bongard-HOI task (Sec. B.1).

# **B.1. Bongard-HOI Example**

VLP is not entirely free from shortcut learning. However, we consider this to be a limitation inherent to the nature of the underlying problems, such as the scarcity of hard negative examples and the presence of ambiguous concepts. We illustrate this with a qualitative example of shortcuts identified in VLP below.

For the concept "carry surfboard" in Bongard-HOI, which was analyzed in RQ1 and RQ3, we observed that with only 12 and 20 support samples, our method retrieved the following program in one run with InternVL-14B:

```
(exists_action(get_actions IMG) holding)
```

This program checks if the positive examples all include the action "holding" and correctly classifies all few-shot examples but fails to specify the object being held. InternVL3-8B, discovers a composition of two actions, holding or walking:

```
(or (exists_action (get_actions IMG) holding)
  (exists_action (get_actions IMG) walking))
```

## Bongard OW Task 39 GT: "lighthouse by sea"

![](_page_13_Figure_1.jpeg)

Figure 7. Example task from Bongard-OpenWorld (Task 39, GT: "lighthouse by sea"). Each task consists of 6 positive few-shot images (left) demonstrating the target concept, 6 negative few-shot images (right) showing contrasting examples, and two query images (bottom) to be classified. Models must induce the visual rule from the few-shot examples and apply it to novel queries.

In the RQ3 experiments, we increase the number of few-shot examples, thereby providing stronger evidence for the target concept. Consequently, the retrieved programs become more refined. For example, with 100 few-shot examples, InternVL3-14B retrieves:

```
(or (exists_action_with_object (get_actions var0) holding surfboard)
  (exists_action_with_object (get_actions var0) standing surfboard))
```

checking, if there is either the action *holding surfboard* or *standing* with *surfboard* present. InternVL3-8B retrieves the program:

```
(or (exists_action_with_object (get_actions IMG) walking surfboard)
  (exists_action_with_object (get_actions IMG) holding surfboard)).
```

which checks if there is the action walking with surfboard or holding surfboard.

# Rongard RWR Task 1 GT: "triangular objects" FS: Negative Imgs Image 8 FS: Negative Imgs Image 8 FS: Negative Imgs Image 8 FS: Negative Imgs Image 8 FRICTION Image 10 FRICTION Image 10 FRICTION Image 10 Query: Positive Img Query: Negative Img Query: Negative Img

Figure 8. Example task from Bongard-RWR (Task 1, GT: "triangular objects"). Similar to Bongard-OW, each task provides 6 positive and 6 negative few-shot examples, followed by two query images to classify. Bongard-RWR features real-world photographic images with more complex visual scenes and contextual variation compared to simplified geometric tasks.

![](_page_15_Figure_0.jpeg)

Figure 9. Bongard-RWR Task 15 (GT: "round objects"). Positive examples feature prominent circular or round objects (donuts, coins, clocks, wheels), while negative examples show rectangular or non-round items (money cases, croissants, notebooks). The task requires identifying roundness as the distinguishing visual property across diverse object categories and contexts.

# CoCoLogic Conflicted Companions (Leash vs License) GT: "dog XOR car"

![](_page_16_Figure_1.jpeg)

Figure 10. Example task from COCOLogic ("Conflicted Companions", GT: "dog XOR car"). Each task contains 10 positive and 10 negative few-shot examples (shown), plus 2 query images (shown at bottom). The dataset uses natural images from COCO to test logical reasoning with Boolean operators (AND, OR, XOR, NOT). This task requires identifying images containing either dogs or cars, but not both.

![](_page_17_Figure_0.jpeg)

Figure 11. Example task from CLEVR-Hans3 (Task 1, GT: "small metal cube and small metal sphere"). Each task contains 10 positive and 10 negative few-shot examples (shown), plus 10 positive and 10 negative query images (not shown for space). The dataset uses synthetically rendered 3D scenes with controlled object properties (size, material, shape, color), enabling systematic evaluation of compositional reasoning with multi-attribute conjunctive rules. A shortcut that is discovered by Qwen3-VL in these few-shot examples is the rule "red object XOR not gold metallic cylinder" (true for all images except for image 6).

# C. Additional Experiments

To complement the results presented in the main paper, this appendix provides a set of additional experiments designed to deepen our understanding of how VLP behaves under different design choices and ablations. These analyses address four key questions:

- How much of VLP's performance stems from its structured prompting strategy as opposed to direct end to end reasoning with VLMs? (cf. Sec. C.1)
- Does weighting symbols by their occurrence frequency lead to better programs than using a uniform distribution? (cf. Sec. C.2)
- Does stochastic generation during the VLM extraction stage matter, or would greedy decoding suffice? (cf. Sec. C.3) Together, these studies allow us to isolate the contribution of individual components, quantify their effect on performance, and verify the robustness of VLP across modeling and decoding settings. The following subsections report these results and discuss their implications.

# C.1. Ablation of Program Synthesis

We examine whether structured symbolic representations of images provide an advantage over reasoning directly with the images. To achieve this, we introduce a second baseline in which VLMs receive the structured representations generated by the VLM functions of VLP, rather than the raw images. This setup allows a more direct comparison of rule induction, since both approaches operate on the same underlying information. The results are presented in Tab. 3 as "w/ VLM functions" together with the original results from (RQ1). For Qwen3-VL we report one seed only, since the model got stuck in reasoning traces and was not able to formulate a final response in over 60% of the cases.

Interestingly, this alternative increases performance slightly in some cases, but also performs worse than the standard baseline in others. On average, it is not able to come close to the performance of VLP. Overall, the results indicate that rules induced by VLP are by far more reliable than those obtained by prompting the model directly, even when structured symbolic representations are provided.

Table 3. Comparison of base VLMs, base VLMs with VLM functions and VLMs with VLP (averaged over three runs). Balanced accuracy (%) on 6-shot Bongard tasks and 10-shot logical reasoning benchmarks. Best results per model are shown in **bold**; overall best results per dataset column are <u>underlined</u>. \*Qwen3-VL with one seed only, generation did not terminate.

| Model                     | Avg.                       | Bongard Tasks (6-shot) |                    |                    | Logical Reasoning (10-shot) |                            |
|---------------------------|----------------------------|------------------------|--------------------|--------------------|-----------------------------|----------------------------|
|                           |                            | Bongard-HOI            | Bongard-OW         | Bongard-RWR        | COCOLogic                   | CLEVR-Hans3                |
| InternVL3-8B              | 57.4                       | 60.5                   | 59.2               | 47.2               | 71.5                        | 48.3                       |
| w/ VLM functions          | 56.6                       | 60.5                   | 56.2               | 50.0               | 70.2                        | 46.1                       |
| w/ VLP                    | <u><b>70.9</b></u> (+13.5) | <u>77.7</u> (+17.2)    | <b>67.5</b> (+8.3) | <b>53.9</b> (+6.7) | <u><b>81.0</b></u> (+9.5)   | <b>74.4</b> (+26.1)        |
| InternVL3-14B             | 61.5                       | 66.9                   | 62.5               | 51.4               | 78.4                        | 48.3                       |
| w/ VLM functions          | 61.5                       | 70.8                   | 62.5               | 50.0               | 74.0                        | 50.0                       |
| w/ VLP                    | <b>68.3</b> (+6.8)         | <b>74.3</b> (+7.4)     | <b>64.7</b> (+2.2) | <b>52.8</b> (+1.4) | 76.7 (-1.7)                 | <b>73.3</b> (+25.0)        |
| Kimi-VL-A3B-Instruct      | 58.5                       | 59.8                   | 58.6               | 46.4               | 77.9                        | 50.0                       |
| w/ VLM functions          | 55.9                       | 57.3                   | 54.2               | 53.3               | 62.2                        | 52.2                       |
| w/ <b>VLP</b>             | <b>65.5</b> (+7.0)         | <b>69.4</b> (+9.6)     | <b>59.4</b> (+0.8) | <b>52.5</b> (+6.1) | 70.1 (-7.8)                 | <b>76.1</b> (+26.1)        |
| Qwen2.5-VL-7B-Instruct    | 60.1                       | 65.2                   | 66.2               | 49.7               | 73.2                        | 46.1                       |
| w/ VLM functions          | 56.9                       | 57.3                   | 55.5               | 53.9               | 61.9                        | 56.1                       |
| w/ VLP                    | <b>69.5</b> (+9.4)         | <b>68.8</b> (+3.6)     | 62.9 (-3.3)        | 49.2 (-0.5)        | <b>80.5</b> (+7.3)          | <u><b>86.1</b></u> (+40.0) |
| Qwen3-VL-30B-A3B-Instruct | 63.4                       | 69.0                   | 68.5               | 55.8               | 73.9                        | 50.0                       |
| w/ VLM functions*         | 56.0                       | 60.2                   | $\overline{57.5}$  | 50.8               | 61.5                        | 50.0                       |
| w/ <b>VLP</b>             | <b>68.9</b> (+5.5)         | <b>74.5</b> (+5.5)     | 66.3 (-2.2)        | <b>58.3</b> (+2.5) | <b>79.1</b> (+5.2)          | <b>66.1</b> (+16.1)        |

# C.2. Uniform vs. Occurrence-based weighted distribution

In this section, we investigate whether our proposed occurrence-based symbol weighting improves the performance of the VLP compared to a uniform symbol distribution. Tab. 4 reports the corresponding delta values. Overall, the weighted

distribution leads to performance gains in most settings, indicating that emphasizing frequently observed symbols in the positive examples helps the model induce more accurate rules.

Table 4. Difference of using occurence-based weighting instead of uniform for symbols.

|                           | Average | Bongard-HOI | Bongard-OW | Bongard-RWR | COCOLogic | CLEVR-Hans3 |
|---------------------------|---------|-------------|------------|-------------|-----------|-------------|
| InternVL3-8B              | +1.1    | +0.5        | +1.0       | 0.0         | +0.3      | +3.8        |
| InternVL3-14B             | +0.2    | +1.3        | +0.4       | -0.8        | -0.3      | +0.5        |
| Kimi-VL-A3B-Instruct      | +0.3    | +1.2        | +0.3       | +2.5        | +1.8      | -4.5        |
| Qwen2.5-VL-7B-Instruct    | -0.9    | -0.5        | +0.7       | -3.3        | -1.0      | -0.6        |
| Qwen3-VL-30B-A3B-Instruct | +0.6    | +0.1        | 0.0        | +0.2        | +2.1      | +0.5        |

# C.3. Comparing Deterministic and Sampling Based Decoding

In Tab. 5, we re-evaluate the setups from Tab. 1 using greedy decoding instead of sampling, both for the baseline model and for the VLP prompts. The relative performance pattern across models remains largely unchanged, indicating that the improvements from VLP do not depend on stochastic decoding but arise from the structure of the prompting itself.

Greedy decoding yields a small decline for the baseline with Intern-VL3 and Qwen2.5-VL models, while Kimi-VL and Qwen3-VL benefit slightly from it. Interestingly, while the results for VLP with InternVL3-8B slightly decrease, for the other models, the results improve. Kimi-VL increases from an average of 65.5 to 67.7 under greedy decoding, and Qwen3-VL even gains an improvement from 68.9 to 71.4. These gains suggest that VLP interacts well with deterministic generation, likely because the structured prompts restrict the model's search space and reduce the chance of drifting into suboptimal continuations.

However, the previous best score of InternVL3-8B under sampling at 70.9 reduces to 67.3 in this setting. This indicates that, for some models, the small amount of exploration introduced by sampling might still be beneficial and allows the decoder to escape overly conservative predictions, *e.g.*, during symbol grounding.

In general, greedy decoding produces more stable outputs when applying VLM functions, while sampling can provide useful diversity during symbol grounding by generating a broader set of candidate symbols. A balanced combination of both approaches may therefore offer the strongest performance, and exploring such hybrid strategies is an interesting direction for future work.

Table 5. Comparison of baseline and VLP prompting with greedy decoding. Accuracy (%) across Bongard benchmarks and logical reasoning tasks. Improvements (green / gray) denote changes relative to the baseline. Best results per model are shown in **bold**; overall best results per dataset column are <u>underlined</u>.

| Model                                      | Avg.                        | Bongard Tasks               |                             |                             | Logical Reasoning           |                             |
|--------------------------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|-----------------------------|
|                                            |                             | Bongard-HOI                 | Bongard-OP                  | Bongard-RWR                 | COCOLogic                   | CLEVR-Hans3                 |
| InternVL3-8B<br>w/ VLP                     | 56.7<br><b>67.3</b> (+10.6) | 58.4<br><b>79.2</b> (+20.8) | 56.2<br><b>67.8</b> (+11.6) | <b>47.5</b> 42.5 (-5.0)     | 74.6<br><b>78.6</b> (+4.0)  | 46.7 <b>68.3</b> (+21.6)    |
| InternVL3-14B<br>w/ VLP                    | 59.6<br><b>69.1</b> (+9.5)  | 67.2<br><b>75.0</b> (+7.8)  | 63.5<br><b>66.2</b> (+2.7)  | 48.3<br><b>51.7</b> (+3.4)  | 70.5<br><b>81.1</b> (+10.6) | 48.3<br><b>71.7</b> (+23.4) |
| Kimi-VL-A3B-Instruct<br>w/ VLP             | 59.2<br><b>67.2</b> (+8.0)  | 59.3<br><b>70.2</b> (+10.9) | <b>58.5</b> 58.0 (-0.5)     | 50.0<br><b>53.3</b> (+3.3)  | <b>78.4</b> 72.8 (-5.6)     | 50.0<br><b>81.7</b> (+31.7) |
| Qwen2.5-VL-7B-Instruct<br>w/ <b>VLP</b>    | 60.0<br><b>69.6</b> (+9.6)  | 63.3<br><b>67.2</b> (+3.9)  | <b>65.0</b> 63.5 (-1.5)     | <b>51.7</b> 48.3 (-3.4)     | 75.0<br><b>80.7</b> (+5.7)  | 45.0<br><b>88.3</b> (+43.3) |
| Qwen3-VL-30B-A3B-Instruct<br>w/ <b>VLP</b> | 63.7<br><b>71.4</b> (+7.7)  | 72.0<br><b>74.4</b> (+2.4)  | 69.5<br>68.5 (-1.0)         | 50.0<br><b>60.0</b> (+10.0) | 76.8<br><b>80.8</b> (+4.0)  | 50.0<br><b>73.3</b> (+23.3) |

# D. Failure Cases

In this section we discuss potential failure cases of our method. These failures occur primarily for two reasons: (1) the VLM produces incorrect symbolic representations or misinterprets visual content (Sec. D.1), or (2) the VLM fails to retrieve the relevant image elements during symbol grounding (Sec. D.2).

# **D.1. VLM Generation Quality**

We evaluate the syntactic validity of the symbolic representations produced by the VLMs for <code>get\_objects</code> and <code>get\_actions</code> in Tab. 6. Models such as InternVL3-14B reliably generate parsable outputs, whereas others, like Kimi, produce numerous non-parsable representations, particularly on COCOLogic. These failures typically arise when the model enters a repetitive loop, repeating list elements without closing the list. In such cases, we automatically repair the list, though the resulting content is often less informative and may contain hallucinations. This behavior also helps explain why Kimi is the model that performs worst when paired with VLP.

Table 6. Object and Action Parse Rates Across Models and Datasets

| Dataset       | Model                     | Object parse rate | Action parse rate |
|---------------|---------------------------|-------------------|-------------------|
|               | InternVL3-14B             | 1.00              | 1.00              |
| bongard-hoi   | InternVL3-8B              | 1.00              | 1.00              |
|               | Kimi-VL-A3B-Instruct      | 0.96              | 0.95              |
|               | Qwen2.5-VL-7B-Instruct    | 0.98              | 0.99              |
|               | Qwen3-VL-30B-A3B-Instruct | 1.00              | 1.00              |
|               | InternVL3-14B             | 1.00              | 1.00              |
| bongard-op    | InternVL3-8B              | 1.00              | 1.00              |
| boligaru-op   | Kimi-VL-A3B-Instruct      | 0.94              | 0.93              |
|               | Qwen2.5-VL-7B-Instruct    | 0.98              | 0.98              |
|               | Qwen3-VL-30B-A3B-Instruct | 0.99              | 1.00              |
|               | InternVL3-14B             | 1.00              | 1.00              |
| bongard-rwr   | InternVL3-8B              | 1.00              | 1.00              |
| boligaru-i wi | Kimi-VL-A3B-Instruct      | 0.97              | 0.73              |
|               | Qwen2.5-VL-7B-Instruct    | 0.98              | 1.00              |
|               | Qwen3-VL-30B-A3B-Instruct | 0.99              | 1.00              |
|               | InternVL3-14B             | 1.00              | 1.00              |
| cocologic     | InternVL3-8B              | 1.00              | 1.00              |
| cocologic     | Kimi-VL-A3B-Instruct      | 0.87              | 0.87              |
|               | Qwen2.5-VL-7B-Instruct    | 0.94              | 0.97              |
|               | Qwen3-VL-30B-A3B-Instruct | 0.97              | 1.00              |
|               | InternVL3-14B             | 1.00              | -                 |
| CLEVR-Hans3   | InternVL3-8B              | 0.99              | -                 |
|               | Kimi-VL-A3B-Instruct      | 0.99              | -                 |
|               | Qwen2.5-VL-7B-Instruct    | 1.00              | -                 |
|               | Qwen3-VL-30B-A3B-Instruct | 1.00              | -                 |

# **D.2. Symbol Discovery Quality**

VLP can only reason about symbols that have been retrieved during the *symbol grounding* stage. Therefore, one obvious failure case of VLP is that one or more relevant symbols for the visual concept are not retrieved in this initial step. To analyze how often this occurs, we conduct an experiment, checking the quality of the grounded symbols for each model and dataset. Hereby, we proceed as follows. If a dataset has objects, properties, or actions explicitly specified in the ground truth rule, such as Bongard-HOI, COCOLogic, and CLEVR-Hans, we leverage them to check the grounded symbols against the elements present in the rule. For the datasets Bongard-OpenWorld and Bongard-RWR, the visual concepts are more vague and not always bound to objects, *e.g.* "living room" or "aerial view". Here, we ask an LLM to extract objects, properties, and actions from the rule to serve as a comparison to the grounded symbols of VLP, as an approximation.

Since there might be multiple terms with the same or close semantic meaning, we do not perform exact equality checking on the retrieved symbols but rather ask an LLM to judge whether the ground truth symbol is present in the symbols retrieved by VLP. For both LLM usages described, we take the model "gpt-4o-2024-08-06".

The results of this analysis are displayed in Fig. 12 for objects, properties, and actions, respectively. Datasets for a symbol type are only considered if the symbol type is part of the ground truth concept.

![](_page_21_Figure_2.jpeg)

Figure 12. Hit ratios of the discovered object, property, and action symbols across all datasets. These values measure how often the symbols extracted by the VLM match the ground truth concepts. Datasets on which VLP performs well in rule induction also show higher symbol quality in this analysis.

We observe that datasets on which VLP achieves the strongest results also tend to exhibit higher quality ratios for the discovered symbols. This relationship is expected, as strong downstream performance implies that the model can reliably detect the visual aspects relevant to the ground-truth concept. Conversely, low symbol quality is likely a contributing factor to weak performance in the rule induction task. For example, Bongard-RWR shows the lowest quality ratios across all components, which aligns with its status as the most challenging dataset in our main evaluation.

While the ground truth symbols in this analysis are approximations generated by an LLM from the official Bongard solutions<sup>2</sup>, the overall trend remains plausible. The original Bongard concepts are well known to be difficult for modern VLMs, even when translated into real-world imagery [22, 47].

 $<sup>^2</sup>$ https://www.foundalis.com/res/bps/bongard\_problems\_solutions.htm

# E. Dataset Quality Issues in Bongard-OpenWorld

During our experiments, we identified systematic annotation errors in the Bongard-OpenWorld dataset that fundamentally compromise both learning and evaluation. These issues fall into three categories: (1) mislabeled few-shot examples that contradict the ground-truth rule, (2) query images that violate the stated concept, and (3) inconsistent application of conjunctive rules. We document representative examples below to illustrate how these errors prevent reliable performance assessment.

# E.1. Inconsistent Few-Shot Labeling

**Partial Rule Satisfaction in Conjunctive Concepts.** Task 23 (GT: "wooden floor living room") exemplifies how annotators sometimes apply only partial matching to multi-component rules. As shown in Figure 13, Images 4 and 5 in the positive set depict a dining area and a staircase respectively—neither of which are living rooms, despite having wooden floors. This creates fundamental ambiguity: should models learn that *both* conditions must be satisfied (wooden floor AND living room), or that *either* condition suffices? Such errors make it impossible to determine whether poor model performance reflects genuine conceptual limitations or simply confusion induced by contradictory training signals.

**Contradictory Labels Within the Same Task.** Task 47 (GT: "tomato dishes") contains multiple logical inconsistencies that render the rule unlearnable (Figure 14). Image 12, labeled as a negative example, clearly shows a tomato-based dish. Meanwhile, Image 4 in the positive set shows a pizza, yet the negative query image also depicts a pizza with visible tomatoes. These contradictions make it impossible for any model—or human—to extract a coherent rule from the provided examples.

# E.2. Query Image Mislabeling

**Attribute Violations in Query Sets.** Task 76 (GT: "gift box pink ribbon") demonstrates systematic query set errors (Figure 15). While the few-shot examples correctly distinguish pink ribbons (positive) from other colors (negative), both query images contain white ribbons. Since neither query satisfies the ground-truth rule, their assigned labels are arbitrary, making evaluation metrics meaningless for this task.

**Categorical Mismatches.** Task 92 (GT: "statue buddha intricate carvings") reveals more severe errors where query images belong to entirely different conceptual categories (Figure 16). The positive query depicts Hindu deity statues rather than Buddha statues—a fundamental categorical error, not merely a boundary case. Such mismatches indicate inadequate quality control in dataset construction and invalidate any assessment of model generalization.

# **E.3. Implications for Performance Evaluation**

These examples are not isolated incidents but represent systematic issues that affect multiple tasks in the dataset. When ground-truth labels are internally contradictory or violate the stated rules, performance drops cannot be meaningfully interpreted. A model that fails on Task 47, for instance, may actually be learning the correct concept but struggling with the dataset's logical inconsistencies. Conversely, high accuracy on Task 76 may reflect memorization of spurious correlations rather than genuine rule understanding. These quality issues must be considered when interpreting any quantitative results on Bongard-OpenWorld.

# FS: Positive Imgs FS: Negative Imgs Image 1 Image 3 Image 4 Image 9 Image 10 Image 10 Image 11 Query: Positive Img Query: Negative Img

**Bongard OW Task 23** 

Figure 13. Annotation errors in Bongard-OW Task 23 reveal inconsistent rule application (GT: "wooden floor living room"). Positive few-shot examples include a dining space (Image 4) and a staircase (Image 5, red boxes), which contain wooden floors but are not living rooms.

# Bongard OW Task 47 GT: "tomato dishes" FS: Negative Imgs Image 8 Image 9 Image 10 Image 11 Query: Positive Img Query: Negative Img Query: Negative Img

Figure 14. Annotation inconsistencies in Bongard-OW Task 47 (GT: "tomato dishes"). The dataset contains contradictory labels: Image 12 (negative few-shot) clearly shows a tomato-based dish, while the negative query image shows a pizza with tomatoes despite Image 4 (positive few-shot) also being a pizza. These inconsistencies (highlighted in red) make the underlying rule ambiguous and evaluation unreliable.

# GT: "gift box pink ribbon" FS: Positive Imgs FS: Negative Imgs lmage 1 Image 2 Image 8 lmage 4 Image 9 lmage 10 Image 3 Image 5 Image 6 Image 11 **Query: Positive Img** Query: Negative Img

Bongard OW Task 76

Figure 15. Query image mislabeling in Bongard-OW Task 76 (GT: "gift box pink ribbon"). While the few-shot examples correctly distinguish pink ribbons (positive) from non-pink ribbons (negative), both query images (highlighted in red) contain white ribbons. This violates the ground-truth rule and renders the queries unanswerable based on the provided examples.

# Bongard OW Task 92 GT: "statue buddha intricate carviings" FS: Positive Imgs Image 1 Image 3 Image 4 Image 9 Image 10 Query: Positive Img Query: Negative Img Query: Negative Img

Figure 16. Annotation error in Bongard-OW Task 92 query set (GT: "statue buddha intricate carvings"). While all positive few-shot images correctly show Buddha statues with intricate carvings, the positive query image (red box) contains Hindu deity statues instead. This categorical mismatch undermines the task's validity and prevents meaningful evaluation of rule generalization.

# F. DSL used for VLP

In Tab. 7, we present the DSL used in the experiments. While the functions are defined in a general form, they are adapted depending on the focus and characteristics of each dataset. For example, CLEVR-Hans contains synthetic objects that are not associated with actions, and therefore action related functions are unnecessary. Similarly, datasets with lower logical complexity do not require counting or numerical comparison operations, whereas Bongard-RWR and COCOLogic benefit from these additional functions due to their more demanding reasoning requirements.

Table 7. Overview of all primitive functions of the DSL available for each dataset, together with their corresponding type signatures. The table groups functions by dataset and specifies the input and output types using arrow based type notation.

| Dataset     | Primitive                                                                                                                                          | Туре                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Bongard-HOI | get_objects get_actions exists_object exists_object_with_property exists_property exists_action exists_action_with_object and, or not              | $\begin{split} & IMG \rightarrow List(List(STRING)) \\ & IMG \rightarrow List(List(STRING)) \\ & List(List(STRING)) \rightarrow OBJECT \rightarrow BOOL \\ & List(List(STRING)) \rightarrow OBJECT \rightarrow PROPERTY \rightarrow BOOL \\ & List(List(STRING)) \rightarrow PROPERTY \rightarrow BOOL \\ & List(List(STRING)) \rightarrow ACTION \rightarrow BOOL \\ & List(List(STRING)) \rightarrow ACTION \rightarrow OBJECT \rightarrow BOOL \\ & BOOL \rightarrow BOOL \rightarrow BOOL \\ & BOOL \rightarrow BOOL \\ & BOOL \rightarrow BOOL \end{split}$ |
| Bongard-OW  | Same as Bongard-HOI + exists_properties exists_object_with_properties                                                                              | $\begin{aligned} & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{PROPERTY} \rightarrow \text{PROPERTY} \rightarrow \text{BOOL} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{OBJECT} \rightarrow \text{PROPERTY} \rightarrow \text{PROPERTY} \rightarrow \text{BOOL} \end{aligned}$                                                                                                                                                                                                                                                 |
| Bongard-RWR | Same as Bongard-OW + count_object_in_img count_objects_with_property max_objects_of_same_type count_all_objects xor gt?, eq? 0, 1, 2, 3, 4, 5, 6   | $\begin{aligned} & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{OBJECT} \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{PROPERTY} \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{INT} \\ & \text{BOOL} \rightarrow \text{BOOL} \rightarrow \text{BOOL} \\ & \text{INT} \rightarrow \text{INT} \rightarrow \text{BOOL} \\ & \text{INT} \end{aligned}$                                               |
| COCOLogic   | Same as Bongard-HOI + count_objects_in_img count_objects_with_property max_objects_of_same_type count_all_objects xor gt?, eq? 0, 1, 2, 3, 4, 5, 6 | $\begin{aligned} & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{OBJECT} \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{PROPERTY} \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{INT} \\ & \text{List}(\text{List}(\text{STRING})) \rightarrow \text{INT} \\ & \text{BOOL} \rightarrow \text{BOOL} \rightarrow \text{BOOL} \\ & \text{INT} \rightarrow \text{INT} \rightarrow \text{BOOL} \\ & \text{INT} \end{aligned}$                                               |
| CLEVR-Hans3 | Similar to Bongard-OW, but without action                                                                                                          | predicates.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |

# **G. Prompts**

In the following we provide the used prompts for the baseline Sec. G.1. For VLP we provide the prompts for symbol grounding Sec. G.2 as well as the VLM functions of VLP Sec. G.3. Further we describe the VLM functions added in the course of RQ4 in Sec. G.4. And finally the prompts for the ablation on program synthesis are provided in Sec. G.5.

# **G.1. Baseline Prompts**

Prompts for the baseline used in the experiments. First, the VLM is prompted for a rule that separates the images of the problem, and then it is prompted to evaluate the rule on the test images.

# Baseline prompt for obtaining visual rule

You are given  $\{n\}$  images. Each image depicts a scene with specific objects, interactions, and environments.

Your task is to determine the underlying concept that distinguishes the positive examples from the negative examples, based on the objects, their properties, and the actions occurring in each scene.

- The first  $\{m\}$  images are positive examples.
- The remaining  $\{o\}$  images are negative examples.

## Task

Identify the rule that defines the positive examples:

- The rule must apply to all positive examples.
- The rule must not apply to any negative example.

## Step-by-Step Process

- 1. Image Analysis:
- Carefully describe each image, noting objects, their attributes, and conceptual features (such as relationships, actions, or settings).
- 2. Rule Derivation:
- From your analysis, infer the rule that uniquely characterizes the positive examples.
- Confirm that the rule does not hold for the negative examples.

## Final Answer Format

Provide your final answer in the following format:

```
"'python
answer = {
'rule': '[RULE]',
}
```

Ensure that the rule is clearly defined and concise.

# Baseline prompt for evaluating direct result

Given the rule '{response}', determine if the image follows the rule or not. Answer with 'Yes' or 'No', nothing else.

# **G.2. Symbol Grounding**

In the following the prompts used during symbol grounding are provided. The VLM receives the task images together with the prompt instructions to identify relevant *objects*, *properties* and *actions*.

For the amount of symbols requested, we use the parameters listed in Tab. 8 across all experiments.

Table 8. Variables used per dataset in program search. Number of objects, properties, and actions requested when generating programs for each dataset.

| Dataset     | #Objects | #Properties | #Actions |
|-------------|----------|-------------|----------|
| Bongard-HOI | 10       | 5           | 10       |
| Bongard-OW  | 10       | 10          | 3        |
| Bongard-RWR | 10       | 10          | 5        |
| COCOLogic   | 10       | 10          | 3        |
| CLEVR-Hans3 | 10       | 10          | 0        |

# **Objects**

You are analyzing images to identify notable objects. Focus on clearly identifiable, semantically meaningful objects that appear across the image set, especially those present in some images but not others. Objects include persons, animals, and things. Avoid minor details like shadows or textures. Use specific, descriptive names (e.g., "bicycle" not "vehicle"). Return exactly  $\{n\}$  objects in a Python list.

```
Answer format:
"'python
objects = [...]
```

No comments or explanations. If no objects found, return [].

# **Properties**

# # Property Discovery Task

You are analyzing a set of images to identify important properties that describe objects in the image set. ## Objective

Discover \*\*notable properties\*\* that characterize objects in the images. Focus on properties that meaningfully distinguish or describe objects (visual attributes, spatial relationships, geometric features, states).

# ## Instructions

- 1. \*\*Examine all images carefully\*\* Look for properties that apply to the relevant objects across the image set
- 2. \*\*Identify important properties\*\* Focus on significant, clearly observable properties that meaningfully describe objects
- 3. \*\*Consider property variation\*\* Properties that vary across images or objects may be particularly noteworthy
- 4. \*\*Prioritize meaningful properties\*\* Choose properties that help distinguish or characterize objects (e.g., color, size, position, orientation, state)
- 5. \*\*Return exactly n properties\*\* If fewer notable properties exist, return as many as available
- 6. \*\*Use descriptive names\*\* Name properties clearly and specifically (e.g., "red" rather than "colored", "horizontal" rather than "oriented")

# ## Relevant Objects

The objects to consider are: {objects}

# ## Property Categories

- \*\*Visual attributes\*\*: color, texture, pattern, brightness
- \*\*Spatial properties\*\*: position (left/right, top/bottom, center), proximity, distance
- \*\*Geometric attributes\*\*: size, shape, orientation, symmetry
- \*\*States\*\*: filled/outlined, open/closed, active/inactive

# ## Output Requirements

- Return a Python list assigned to variable 'properties'
- Include only the Python code, no explanations or comments
- If no notable properties are found, return an empty list '[]'
- Use clear, specific property names (e.g., "blue", "large", "leftmost", "vertical")
- Be general (e.g., if there's a yellow triangle, use "yellow" not "yellow triangle")

```
## Example Format
```

```
"python
```

properties = ["red", "large", "horizontal", "outlined", "centered"]

### Actions

# # Action Discovery Task

You are analyzing a set of images to identify important actions performed by objects in the image set.

# ## Objective

Discover \*\*notable actions\*\* that characterize objects in the images. Focus on actions that meaningfully distinguish or describe what objects are doing (movements, behaviors, states of activity).

## ## Instructions

- 1. \*\*Examine all images carefully\*\* Look for actions that apply to the relevant objects across the image set
- 2. \*\*Identify important actions\*\* Focus on significant, clearly observable actions that meaningfully describe what objects are doing
- 3. \*\*Consider action variation\*\* Actions that vary across images or objects may be particularly noteworthy (contrasting actions)
- 4. \*\*Prioritize meaningful actions\*\* Choose actions that help distinguish or characterize object behaviors (e.g., running, jumping, standing, flying)
- 5. \*\*Return exactly n actions\*\* If fewer notable actions exist, return as many as available
- 6. \*\*Use descriptive names\*\* Name actions clearly and specifically (e.g., "running" rather than "moving", "sitting" rather than "positioned")

# ## Relevant Objects

The objects to consider are: {objects}

# ## Action Categories

- \*\*Movement actions\*\*: walking, running, jumping, flying, rolling
- \*\*Positional actions\*\*: standing, sitting, lying, hanging
- \*\*Interactive actions\*\*: holding, pushing, pulling, touching
- \*\*State actions\*\*: opening, closing, rotating, tilting

# ## Output Requirements

- Return a Python list assigned to variable 'actions'
- Include only the Python code, no explanations or comments
- If no notable actions are found, return an empty list '[]'
- Use clear, specific action names (e.g., "jumping", "sitting", "rotating", "falling")

```
## Example Format "python
```

```
actions = ["running", "jumping", "standing", "flying", "sitting"]
```

# G.3. VLM Functions of the DSL

In the following the two VLM functions used for VLP are presented. These are executed on individual images to obtain a structured representation of the objects and actions in the image.

# get\_objects - VLM function for obtaining object-property representation

```
Identify objects and their properties from the image using only the provided lists.
**Objects:** {objects}
**Properties:** {properties}
## Rules
1. Only use objects/properties from the provided lists
2. Return empty list if no valid objects found
3. No explanations or additional text
## Output Format
"'python
objects = [
   ['object_name', 'property1', 'property2', ...],
['object_name', 'property1'],
**If no valid objects:** 'objects = [[]]'
## Examples
**Example 1**
- Objects: ["car", "person", "tree"]
- Properties: ["red", "tall", "small", "standing"]
- Image: Red car under tall tree with small standing person
"'python
objects = [
   ['car', 'red'],
['tree', 'tall'],
   ['person', 'standing', 'small']
**Example 2**
- Objects: ["dog", "ball", "book", "chair"]
- Properties: ["blue", "sitting", "round"]
- Image: Dog sitting by round ball and blue chair
"'python
   ['dog', 'sitting'],
['ball', 'round'],
['chair', 'blue']
**Example 3**
- Objects: ["bicycle", "lamp", "table", "cup"]
- Properties: ["green", "broken", "wooden", "white"]
- Image: Table with laptop and cup
"python
objects = [[]]
*Note: Even though 'table' and 'cup' are in the objects list and visible in the image, neither has properties from the provided list, so no valid object-property
combinations exist*
**Analyze the image now:**
```

```
Identify actions occurring in the image using only the provided lists.
**Actions:** actions
**Objects:** objects
## Rules
1. Only use actions/objects from the provided lists
2. Only detect actions that are actually happening in the image
3. Do not include actions from the list if they are not occurring in the image
4. If an action involves an object, include the object name
5. Return empty list if no valid actions found
6. No explanations or additional text
## Output Format
"python
actions = [
   ['action_name1'],
['action_name2', 'object_name2'],
   ['action_name2', 'object_name1', 'object_name2'],
**If no valid actions: ** 'actions = [[]]'
## Examples
**Example 1**
- Actions: ["running", "jumping", "sitting", "dancing"]
- Objects: ["chair", "ball", "person", "table"]
- Image: Person sitting on chair
"'python
actions = [
  ['sitting', 'person', 'chair']
*Note: 'running', 'jumping', and 'dancing' are in the actions list but not happening in the image, so they're excluded*
- Actions: ["throwing", "catching", "walking", "reading", "sleeping"]
- Objects: ["ball", "book", "dog", "frisbee"]
- Image: Person throwing a ball while dog is walking
"'python
actions = [
  ['throwing', 'person', 'ball'],
['walking', 'dog']
*Note: 'catching', 'reading', and 'sleeping' are in the actions list but not occurring in the image, so they're excluded*
**Example 3**
- Actions: ["swimming", "flying", "cooking"]
- Objects: ["pool", "bird", "kitchen"]
- Image: Person eating at a restaurant
"'python
actions = [[]]
*Note: Even though actions are happening in the image, none match the provided actions list, so no valid actions exist*
**Analyze the image now:**
```

# **G.4.** VLM Functions for property size

In the course of (RQ4) we add VLM functions to the DSL that explicitly ask the VLM for objects of small and large size. These are listed in the following.

# exists\_object\_small\_in\_img

Does the image contain any '{obj}' that is relatively small in size compared to the other objects? Answer with 'YES' or 'NO'.

# exists\_object\_large\_in\_img

Does the image contain any '{obj}' that is relatively large in size compared to the other objects? Answer with 'YES' or 'NO'.

# exists\_object\_with\_property\_small\_in\_img

Does the image contain any '{obj}' with the property '{prop}' that is relatively small in size compared to the other objects? Answer with 'YES' or 'NO'.

# exists\_object\_with\_property\_large\_in\_img

Does the image contain any '{obj}' with the property '{prop}' that is relatively large in size compared to the other objects? Answer with 'YES' or 'NO'.

# G.5. Structure-based baseline

For the ablation study on program synthesis in Sec. C.1, we implement a structure-based approach that uses the VLM functions defined in the DSL (see above) to extract symbolic representations of the input images. These representations are then passed back to the VLM, which reasons over them to induce a rule. Below we provide the prompt used for this procedure, along with the prompt applied during evaluation when test images are assessed based on their structure-based representations.

# Task: Concept Identification from Structured Image Representations

```
You will be given a sequence of structured image representations, where each image is labeled as either a **positive** or **negative** example of an underlying
## Structure of Each Image Representation
### Objects
A list of objects, where each object is described together with its properties.
**Format:**
[[object\_name, property1, property2, ...], [object\_name, property1, ...], ...]\\
**Example:**
[[dog, brown, large], [cat, small, cute], [ball, red, round]]
A list of actions occurring in the image, including the entities involved.
[[action, participant1, participant2, ...], [action, participant], ...]
**Example:**
[[playing, dog, cat], [shining, sun], [rolling, ball]]
## Your Objective
Identify the rule that defines the positive examples:
- The rule must apply to all positive examples.
- The rule must not apply to any negative example.
## Step-by-Step Process
1. Image Analysis:
- Analyze the objects, their properties, and the actions across all examples to identify the concept that distinguishes positive examples from negative ones.
2. Rule Derivation:
- From your analysis, infer the rule that uniquely characterizes the positive examples.
- Confirm that the rule does not hold for the negative examples.
## Structured Image Representations
Image: 1
Objects: {obj_repr}
Actions: {action_repr}
Label: 'Positive'
Image: {n}
Objects: {obj_repr}
Actions: {action_repr}
Label: 'Negative'
## Final Answer Format
Provide your final answer in the following format:
"'python
answer = {
'rule': '[RULE]',
```

Ensure that the rule is clearly defined and concise.

```
# Task: Classify Image based on Structured Image Representations
You are given a structured image representation.
## Structure of Each Image Representation
### Objects
A list of objects, where each object is described together with its properties.
**Format:**
[[object_name, property1, property2, ...], [object_name, property1, ...], ...]
**Example:**
[[dog, brown, large], [cat, small, cute], [ball, red, round]]
### Actions
A list of actions occurring in the image, including the entities involved.
**Format:**
[[action, participant1, participant2, ...], [action, participant], ...]
**Example:**
[[playing, dog, cat], [shining, sun], [rolling, ball]]
## Your Objective
Given the rule '{rule}', determine if the image follows the rule or not. Answer with 'Yes' or 'No', nothing else.
## Structured Image Representations
{representation}
```