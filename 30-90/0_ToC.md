# Origins

Ilya Sutskever shared a private reading list of 30 papers with John Carmack and said, “If you really learn all of these, you’ll know 90% of what matters today.” This is that list.

Deep learning’s modern arc is a loop between **compression** (what generalizes), **inductive bias** (what’s easy to learn), and **scale** (what becomes possible). The list starts with ideas about complexity and description length, then moves through the key architectural eras (CNNs → RNNs → attention/Transformers), and ends in the LLM era where **systems, retrieval, long-context behavior, evaluation, and alignment** determine what works in practice.

## Table of contents (recommended reading order)

### 0. The big questions: intelligence, complexity, and what “progress” means

*Connecting thread:* Why learning looks like compression, and why “intelligence” is measurable (at least in principle).

* [24 - Machine Super Intelligence.pdf](24_Machine_Super_Intelligence.pdf)
* [1 - The First Law of Complexodynamics.pdf](1_The_First_Law_of_Complexodynamics.pdf)
* [19 - Quantifying the Rise and Fall of Complexity in Closed Systems- The Coffee Automaton.pdf](19_Quantifying_the_Rise_and_Fall_of_Complexity_in_Closed_Systems_The_Coffee_Automaton.pdf)

### 1. Compression, MDL, and generalization

*Connecting thread:* A unifying lens: good models don’t just fit—they compress.

* [25 - Kolmogorov Complexity and Algorithmic Randomness.pdf](25_Kolmogorov_Complexity_and_Algorithmic_Randomness.pdf)
* [23 - A Tutorial Introduction to the Minimum Description Length Principle.pdf](23_A_Tutorial_Introduction_to_the_Minimum_Description_Length_Principle.pdf)
* [5 - Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.pdf](5_Keeping_Neural_Networks_Simple_by_Minimizing_the_Description_Length_of_the_Weights.pdf)
* [17 - VARIATIONAL LOSSY AUTOENCODER.pdf](17_VARIATIONAL_LOSSY_AUTOENCODER.pdf)

### 2. Vision and the rise of deep learning in practice

*Connecting thread:* Inductive bias + optimization tricks + scale turns neural nets into reliable workhorses.

* [26 - Stanford CS231n Convolutional Neural Networks for Visual Recognition.pdf](26_Stanford_CS231n_Convolutional_Neural_Networks_for_Visual_Recognition.pdf)
* [7 - ImageNet Classification with Deep Convolutional Neural Networks.pdf](7_ImageNet_Classification_with_Deep_Convolutional_Neural_Networks.pdf)
* [10 - Deep Residual Learning for Image Recognition.pdf](10_Deep_Residual_Learning_for_Image_Recognition.pdf)
* [15 - Identity Mappings in Deep Residual Networks.pdf](15_Identity_Mappings_in_Deep_Residual_Networks.pdf)
* [11 - MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS.pdf](11_MULTI_SCALE_CONTEXT_AGGREGATION_BY_DILATED_CONVOLUTIONS.pdf)

### 3. Sequence modeling before Transformers

*Connecting thread:* How recurrence, gating, and regularization made sequence learning work (and where it struggled).

* [2 - The Unreasonable Effectiveness of Recurrent Neural Networks.pdf](2_The_Unreasonable_Effectiveness_of_Recurrent_Neural_Networks.pdf)
* [3 - Understanding LSTM Networks.pdf](3_Understanding_LSTM_Networks.pdf)
* [4 - RECURRENT NEURAL NETWORK REGULARIZATION.pdf](4_RECURRENT_NEURAL_NETWORK_REGULARIZATION.pdf)
* [21 - Deep Speech 2- End-to-End Speech Recognition in English and Mandarin.pdf](21_Deep_Speech_2_End_to_End_Speech_Recognition_in_English_and_Mandarin.pdf)

### 4. Attention becomes the architecture

*Connecting thread:* Attention starts as alignment, becomes a pointer, then becomes the whole model.

* [14 - NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE.pdf](14_NEURAL_MACHINE_TRANSLATION_BY_JOINTLY_LEARNING_TO_ALIGN_AND_TRANSLATE.pdf)
* [6 - Pointer Networks.pdf](6_Pointer_Networks.pdf)
* [8 - ORDER MATTERS- SEQUENCE TO SEQUENCE FOR SETS.pdf](8_ORDER_MATTERS_SEQUENCE_TO_SEQUENCE_FOR_SETS.pdf)
* [13 - Attention Is All You Need.pdf](13_Attention_Is_All_You_Need.pdf)

### 5. Memory, reasoning, and graphs

*Connecting thread:* Different routes to structured computation: external memory, relational modules, and message passing.

* [20 - Neural Turing Machines.pdf](20_Neural_Turing_Machines.pdf)
* [16 - A simple neural network module for relational reasoning.pdf](16_A_simple_neural_network_module_for_relational_reasoning.pdf)
* [12 - Neural Message Passing for Quantum Chemistry.pdf](12_Neural_Message_Passing_for_Quantum_Chemistry.pdf)
* [18 - Relational recurrent neural networks.pdf](18_Relational_recurrent_neural_networks.pdf)

### 6. Scaling and systems

*Connecting thread:* Once optimization is stable, the frontier shifts to compute, data, and infrastructure.

* [9 - GPipe- Easy Scaling with Micro-Batch Pipeline Parallelism.pdf](9_GPipe_Easy_Scaling_with_Micro_Batch_Pipeline_Parallelism.pdf)
* [22 - Scaling Laws for Neural Language Models.pdf](22_Scaling_Laws_for_Neural_Language_Models.pdf)
* [27A - Better & Faster Large Language Models via Multi-token Prediction.pdf](27A_Better_&_Faster_Large_Language_Models_via_Multi_token_Prediction.pdf)

### 7. Retrieval, long context, new knowledge, and verification

*Connecting thread:* Modern LLM stacks: augment memory with retrieval, diagnose context failure modes, evaluate reliability.

* [27B - Dense Passage Retrieval for Open-Domain Question Answering.pdf](27B_Dense_Passage_Retrieval_for_Open_Domain_Question_Answering.pdf)
* [27C - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf](27C_Retrieval_Augmented_Generation_for_Knowledge_Intensive_NLP_Tasks.pdf)
* [30A - Precise Zero-Shot Dense Retrieval without Relevance Labels.pdf](30A_Precise_Zero_Shot_Dense_Retrieval_without_Relevance_Labels.pdf)
* [29 - Lost in the Middle- How Language Models Use Long Contexts.pdf](29_Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf)
* [30B - ALCUNA- Large Language Models Meet New Knowledge.pdf](30B_ALCUNA_Large_Language_Models_Meet_New_Knowledge.pdf)
* [30C - The Perils & Promises of Fact-checking with Large Language Models.pdf](30C_The_Perils_&_Promises_of_Fact_checking_with_Large_Language_Models.pdf)

### 8. Alignment as a training recipe

*Connecting thread:* Turning raw capability into behavior you can deploy and steer.

* [28 - ZEPHYR- DIRECT DISTILLATION OF LM ALIGNMENT.pdf](28_ZEPHYR_DIRECT_DISTILLATION_OF_LM_ALIGNMENT.pdf)
