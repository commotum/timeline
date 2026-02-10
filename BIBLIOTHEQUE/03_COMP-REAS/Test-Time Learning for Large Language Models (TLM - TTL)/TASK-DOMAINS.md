# Test-Time Learning for Large Language Models (2025)
Source: Test-Time Learning for Large Language Models (TLM - TTL).md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Named entity recognition | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | token-level entity labels | 1D (t) (inferred) | Capped (inferred) |
| Relation inference | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | relation labels | 0D (inferred) | Fixed (inferred) |
| Fact verification | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | verification labels | 0D (inferred) | Fixed (inferred) |
| Question answering | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | answer tokens | 1D (t) (inferred) | Capped (inferred) |
| Medical dialogue response generation | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | response/diagnosis tokens | 1D (t) (inferred) | Capped (inferred) |
| Sentiment analysis and opinion mining | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | sentiment labels; opinion spans | 0D (inferred); 1D (t) (inferred) | Capped (inferred) |
| Instruction following | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | instruction-conditioned response tokens | 1D (t) (inferred) | Capped (inferred) |
| Text summarization | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | summary tokens | 1D (t) (inferred) | Capped (inferred) |
| Text classification | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | class labels | 0D (inferred) | Fixed (inferred) |
| Mathematical problem solving and multi-step reasoning | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | reasoning traces and answers | 1D (t) (inferred) | Capped (inferred) |
| Logical reading comprehension and deductive reasoning | tokens | 1D (t) (inferred) | Capped (inferred) | Dynamic (inferred) | Constructed (inferred) | multiple-choice answers | 0D (inferred) | Fixed (inferred) |

## Summary
The paper evaluates TLM on a broad text-task suite spanning domain knowledge tasks, instruction-following tasks, and reasoning tasks. Explicitly named task intents include NER, relation inference, fact verification, QA, summarization, classification, sentiment/opinion analysis, medical dialogue, mathematical reasoning, and logical reading comprehension. Across rows, the evidence supports token-based interfaces, with 1D (t) inputs and a mix of 1D token outputs and 0D label outputs (all marked as inferred where needed). Attention and state are marked as Dynamic/Constructed (inferred) because the method performs runtime sample selection and test-time LoRA parameter updates.

## Evidence
### Task: Named entity recognition
- "It includes tasks such as Named Entity Recognition (NER), relation inference, fact verification, and question answering" (Section B.1. DomainBench)
- "such as named entity recognition, judgment, and question answering" (Section B. AdaptEval)
- Inference: Input/output dimensionality and dynamics are inferred from the token-sequence formulation in Section 4.1, runtime sample selection via Eqn. (6) in Section 4.2, and persistent LoRA updates ("we update only ΔΘ during the Test-Time Learning") in Section 4.3.

### Task: Relation inference
- "It includes tasks such as Named Entity Recognition (NER), relation inference, fact verification, and question answering" (Section B.1. DomainBench)
- "This category includes four vertical domain knowledge datasets: Geography, Agriculture, Medicine, and Finance. It evaluates the adaptability of LLMs to specialized fields by assessing their ability to handle tasks requiring domain-specific expertise, such as named entity recognition, judgment, and question answering." (Section B. AdaptEval)
- Inference: Relation inference is treated as token-input prediction of relation labels (0D, Fixed), with 1D token inputs and Capped/Dynamic/Constructed attributes inferred from Sections 4.1-4.3.

### Task: Fact verification
- "It includes tasks such as Named Entity Recognition (NER), relation inference, fact verification, and question answering" (Section B.1. DomainBench)
- "such as named entity recognition, judgment, and question answering" (Section B. AdaptEval)
- Inference: Fact verification is mapped to verification-label output (0D, Fixed) by task intent; token-input, Capped dynamics, Dynamic attention, and Constructed state are inferred from Sections 4.1-4.3.

### Task: Question answering
- "It includes tasks such as Named Entity Recognition (NER), relation inference, fact verification, and question answering" (Section B.1. DomainBench)
- "including, but not limited to, question answering (QA), text summarization, and classification" (Section B.2. InstructionBench)
- Inference: QA is represented as token-to-token generation (1D to 1D) using the sequence-token formulation in Section 4.1 and adaptation behavior in Sections 4.2-4.3.

### Task: Medical dialogue response generation
- "GenMedGPT-5k with a total of 5.45k samples is a medical dialogue dataset generated by ChatGPT, and is designed to emulate real-life conversations between patients and doctors." (Section B.1. DomainBench)
- "answer the medical questions based on the patient's description" (Table 7, Section B.1. DomainBench)
- Inference: Dialogue input/output are treated as token sequences (1D to 1D), with Capped/Dynamic/Constructed attributes inferred from Sections 4.1-4.3.

### Task: Sentiment analysis and opinion mining
- "This dataset is extensively used for sentiment analysis, opinion mining, and QA tasks in financial texts" (Section B.1. DomainBench)
- "It integrates general task data (Alpaca dataset), financial domain data (FiQA dataset), and custom task data generated using GPT-3.5." (Section B.1. DomainBench)
- Inference: Output is marked as mixed label/span form (0D; 1D) because sentiment analysis is label-like while opinion mining can be span/text oriented; other attributes are inferred from Sections 4.1-4.3.

### Task: Instruction following
- "InstructionBench aims to assess the adaptability and performance of models across a diverse range of general instruction tasks, including, but not limited to, question answering (QA), text summarization, and classification." (Section B.2. InstructionBench)
- "ability to comprehend, interpret, and execute a diverse range of user instructions" (Section B. AdaptEval)
- Inference: Instruction following is modeled as token-input to token-output generation (1D to 1D), with Capped/Dynamic/Constructed attributes inferred from Sections 4.1-4.3.

### Task: Text summarization
- "including, but not limited to, question answering (QA), text summarization, and classification" (Section B.2. InstructionBench)
- "The dataset encompasses common instruction fine-tuning tasks, including QA, summarization, and classification." (Section B.2. InstructionBench)
- Inference: Summarization is mapped to token-to-token generation (1D to 1D), with Capped/Dynamic/Constructed attributes inferred from Sections 4.1-4.3.

### Task: Text classification
- "including, but not limited to, question answering (QA), text summarization, and classification" (Section B.2. InstructionBench)
- "tasks, including QA, summarization, and classification" (Section B.2. InstructionBench)
- Inference: Classification is mapped to class-label output (0D, Fixed) by task intent; input and adaptation attributes are inferred from Sections 4.1-4.3.

### Task: Mathematical problem solving and multi-step reasoning
- "tasks such as mathematical problem solving, multi-step reasoning" (Section B.3. ReasoningBench)
- "The primary task type is multi-step mathematical reasoning" (Section B.3. ReasoningBench, GSM8K)
- Inference: Reasoning tasks are treated as token-sequence generation (often CoT plus answer), yielding 1D outputs with Capped/Dynamic/Constructed attributes inferred from Sections 4.1-4.3.

### Task: Logical reading comprehension and deductive reasoning
- "ReasoningBench is designed to evaluate models' logical reasoning and problem-solving abilities through tasks such as mathematical problem solving, multi-step reasoning, and logical reading comprehension." (Section B.3. ReasoningBench)
- "covering a variety of deductive reasoning tasks" (Section B.3. ReasoningBench, LogiQA)
- Inference: Logical reading comprehension (multiple-choice prompts) is mapped to option-label output (0D, Fixed); other attributes are inferred from Sections 4.1-4.3.
