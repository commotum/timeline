# Origins

Ilya Sutskever shared a private reading list of 30 papers with John Carmack and said, “If you really learn all of these, you’ll know 90% of what matters today.” This is that list.

Deep learning’s modern arc is a loop between **compression** (what generalizes), **inductive bias** (what’s easy to learn), and **scale** (what becomes possible). The list starts with ideas about complexity and description length, then moves through the key architectural eras (CNNs → RNNs → attention/Transformers), and ends in the LLM era where **systems, retrieval, long-context behavior, evaluation, and alignment** determine what works in practice.

## Table of contents (recommended reading order)

### 0. The big questions: intelligence, complexity, and what “progress” means

*Connecting thread:* Why learning looks like compression, and why “intelligence” is measurable (at least in principle).

* [Machine Super Intelligence](1_Machine_Super_Intelligence.pdf)
* [The First Law of Complexodynamics](2_The_First_Law_of_Complexodynamics.pdf)
* [Quantifying the Rise and Fall of Complexity in Closed Systems- The Coffee Automaton](3_Quantifying_the_Rise_and_Fall_of_Complexity_in_Closed_Systems_The_Coffee_Automaton.pdf)

### 1. Compression, MDL, and generalization

*Connecting thread:* A unifying lens: good models don’t just fit—they compress.

* [Kolmogorov Complexity and Algorithmic Randomness](4_Kolmogorov_Complexity_and_Algorithmic_Randomness.pdf)
* [A Tutorial Introduction to the Minimum Description Length Principle](5_A_Tutorial_Introduction_to_the_Minimum_Description_Length_Principle.pdf)
* [Keeping Neural Networks Simple by Minimizing the Description Length of the Weights](6_Keeping_Neural_Networks_Simple_by_Minimizing_the_Description_Length_of_the_Weights.pdf)
* [VARIATIONAL LOSSY AUTOENCODER](7_VARIATIONAL_LOSSY_AUTOENCODER.pdf)

### 2. Vision and the rise of deep learning in practice

*Connecting thread:* Inductive bias + optimization tricks + scale turns neural nets into reliable workhorses.

* [Stanford CS231n Convolutional Neural Networks for Visual Recognition](8_Stanford_CS231n_Convolutional_Neural_Networks_for_Visual_Recognition.pdf)
* [ImageNet Classification with Deep Convolutional Neural Networks](9_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks.pdf)
* [Deep Residual Learning for Image Recognition](10_Deep_Residual_Learning_for_Image_Recognition.pdf)
* [Identity Mappings in Deep Residual Networks](11_Identity_Mappings_in_Deep_Residual_Networks.pdf)
* [MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS](12_MULTI_SCALE_CONTEXT_AGGREGATION_BY_DILATED_CONVOLUTIONS.pdf)

### 3. Sequence modeling before Transformers

*Connecting thread:* How recurrence, gating, and regularization made sequence learning work (and where it struggled).

* [The Unreasonable Effectiveness of Recurrent Neural Networks](13_The_Unreasonable_Effectiveness_of_Recurrent_Neural_Networks.pdf)
* [Understanding LSTM Networks](14_Understanding_LSTM_Networks.pdf)
* [RECURRENT NEURAL NETWORK REGULARIZATION](15_RECURRENT_NEURAL_NETWORK_REGULARIZATION.pdf)
* [Deep Speech 2- End-to-End Speech Recognition in English and Mandarin](16_Deep_Speech_2_End_to_End_Speech_Recognition_in_English_and_Mandarin.pdf)

### 4. Attention becomes the architecture

*Connecting thread:* Attention starts as alignment, becomes a pointer, then becomes the whole model.

* [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](17_NEURAL_MACHINE_TRANSLATION_BY_JOINTLY_LEARNING_TO_ALIGN_AND_TRANSLATE.pdf)
* [Pointer Networks](18_Pointer_Networks.pdf)
* [ORDER MATTERS- SEQUENCE TO SEQUENCE FOR SETS](19_ORDER_MATTERS_SEQUENCE_TO_SEQUENCE_FOR_SETS.pdf)
* [Attention Is All You Need](20_Attention_Is_All_You_Need.pdf)

### 5. Memory, reasoning, and graphs

*Connecting thread:* Different routes to structured computation: external memory, relational modules, and message passing.

* [Neural Turing Machines](21_Neural_Turing_Machines.pdf)
* [A simple neural network module for relational reasoning](22_A_simple_neural_network_module_for_relational_reasoning.pdf)
* [Neural Message Passing for Quantum Chemistry](23_Neural_Message_Passing_for_Quantum_Chemistry.pdf)
* [Relational recurrent neural networks](24_Relational_recurrent_neural_networks.pdf)

### 6. Scaling and systems

*Connecting thread:* Once optimization is stable, the frontier shifts to compute, data, and infrastructure.

* [GPipe- Easy Scaling with Micro-Batch Pipeline Parallelism](25_GPipe_Easy_Scaling_with_Micro_Batch_Pipeline_Parallelism.pdf)
* [Scaling Laws for Neural Language Models](26_Scaling_Laws_for_Neural_Language_Models.pdf)
* [Better & Faster Large Language Models via Multi-token Prediction](27_Better_&_Faster_Large_Language_Models_via_Multi_token_Prediction.pdf)

### 7. Retrieval, long context, new knowledge, and verification

*Connecting thread:* Modern LLM stacks: augment memory with retrieval, diagnose context failure modes, evaluate reliability.

* [Dense Passage Retrieval for Open-Domain Question Answering](28_Dense_Passage_Retrieval_for_Open_Domain_Question_Answering.pdf)
* [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](29_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_Tasks.pdf)
* [Precise Zero-Shot Dense Retrieval without Relevance Labels](30_Precise_Zero_Shot_Dense_Retrieval_without_Relevance_Labels.pdf)
* [Lost in the Middle- How Language Models Use Long Contexts](31_Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf)
* [ALCUNA- Large Language Models Meet New Knowledge](32_ALCUNA_Large_Language_Models_Meet_New_Knowledge.pdf)
* [The Perils & Promises of Fact-checking with Large Language Models](33_The_Perils_&_Promises_of_Fact_checking_with_Large_Language_Models.pdf)

### 8. Alignment as a training recipe

*Connecting thread:* Turning raw capability into behavior you can deploy and steer.

* [ZEPHYR- DIRECT DISTILLATION OF LM ALIGNMENT](34_ZEPHYR_DIRECT_DISTILLATION_OF_LM_ALIGNMENT.pdf)
