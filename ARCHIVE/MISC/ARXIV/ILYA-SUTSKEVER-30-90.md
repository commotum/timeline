# Origins

Ilya Sutskever shared a private reading list of 30 papers with John Carmack and said, “If you really learn all of these, you’ll know 90% of what matters today.” This is that list.

Deep learning’s modern arc is a loop between **compression** (what generalizes), **inductive bias** (what’s easy to learn), and **scale** (what becomes possible). The list starts with ideas about complexity and description length, then moves through the key architectural eras (CNNs → RNNs → attention/Transformers), and ends in the LLM era where **systems, retrieval, long-context behavior, evaluation, and alignment** determine what works in practice.

## Table of contents (recommended reading order)

### 0. The big questions: intelligence, complexity, and what “progress” means

*Connecting thread:* Why learning looks like compression, and why “intelligence” is measurable (at least in principle).

* [24 - Machine Super Intelligence.pdf](sandbox:/mnt/data/24%20-%20Machine%20Super%20Intelligence.pdf)
* [1 - The First Law of Complexodynamics.pdf](sandbox:/mnt/data/1%20-%20The%20First%20Law%20of%20Complexodynamics.pdf)
* [19 - Quantifying the Rise and Fall of Complexity in Closed Systems- The Coffee Automaton.pdf](sandbox:/mnt/data/19%20-%20Quantifying%20the%20Rise%20and%20Fall%20of%20Complexity%20in%20Closed%20Systems-%20The%20Coffee%20Automaton.pdf)

### 1. Compression, MDL, and generalization

*Connecting thread:* A unifying lens: good models don’t just fit—they compress.

* [25 - Kolmogorov Complexity and Algorithmic Randomness.pdf](sandbox:/mnt/data/25%20-%20Kolmogorov%20Complexity%20and%20Algorithmic%20Randomness.pdf)
* [23 - A Tutorial Introduction to the Minimum Description Length Principle.pdf](sandbox:/mnt/data/23%20-%20A%20Tutorial%20Introduction%20to%20the%20Minimum%20Description%20Length%20Principle.pdf)
* [5 - Keeping Neural Networks Simple by Minimizing the Description Length of the Weights.pdf](sandbox:/mnt/data/5%20-%20Keeping%20Neural%20Networks%20Simple%20by%20Minimizing%20the%20Description%20Length%20of%20the%20Weights.pdf)
* [17 - VARIATIONAL LOSSY AUTOENCODER.pdf](sandbox:/mnt/data/17%20-%20VARIATIONAL%20LOSSY%20AUTOENCODER.pdf)

### 2. Vision and the rise of deep learning in practice

*Connecting thread:* Inductive bias + optimization tricks + scale turns neural nets into reliable workhorses.

* [26 - Stanford CS231n Convolutional Neural Networks for Visual Recognition.pdf](sandbox:/mnt/data/26%20-%20Stanford%20CS231n%20Convolutional%20Neural%20Networks%20for%20Visual%20Recognition.pdf)
* [7 - ImageNet Classification with Deep Convolutional Neural Networks.pdf](sandbox:/mnt/data/7%20-%20ImageNet%20Classification%20with%20Deep%20Convolutional%20Neural%20Networks.pdf)
* [10 - Deep Residual Learning for Image Recognition.pdf](sandbox:/mnt/data/10%20-%20Deep%20Residual%20Learning%20for%20Image%20Recognition.pdf)
* [15 - Identity Mappings in Deep Residual Networks.pdf](sandbox:/mnt/data/15%20-%20Identity%20Mappings%20in%20Deep%20Residual%20Networks.pdf)
* [11 - MULTI-SCALE CONTEXT AGGREGATION BY DILATED CONVOLUTIONS.pdf](sandbox:/mnt/data/11%20-%20MULTI-SCALE%20CONTEXT%20AGGREGATION%20BY%20DILATED%20CONVOLUTIONS.pdf)

### 3. Sequence modeling before Transformers

*Connecting thread:* How recurrence, gating, and regularization made sequence learning work (and where it struggled).

* [2 - The Unreasonable Effectiveness of Recurrent Neural Networks.pdf](sandbox:/mnt/data/2%20-%20The%20Unreasonable%20Effectiveness%20of%20Recurrent%20Neural%20Networks.pdf)
* [3 - Understanding LSTM Networks.pdf](sandbox:/mnt/data/3%20-%20Understanding%20LSTM%20Networks.pdf)
* [4 - RECURRENT NEURAL NETWORK REGULARIZATION.pdf](sandbox:/mnt/data/4%20-%20RECURRENT%20NEURAL%20NETWORK%20REGULARIZATION.pdf)
* [21 - Deep Speech 2- End-to-End Speech Recognition in English and Mandarin.pdf](sandbox:/mnt/data/21%20-%20Deep%20Speech%202-%20End-to-End%20Speech%20Recognition%20in%20English%20and%20Mandarin.pdf)

### 4. Attention becomes the architecture

*Connecting thread:* Attention starts as alignment, becomes a pointer, then becomes the whole model.

* [14 - NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE.pdf](sandbox:/mnt/data/14%20-%20NEURAL%20MACHINE%20TRANSLATION%20BY%20JOINTLY%20LEARNING%20TO%20ALIGN%20AND%20TRANSLATE.pdf)
* [6 - Pointer Networks.pdf](sandbox:/mnt/data/6%20-%20Pointer%20Networks.pdf)
* [8 - ORDER MATTERS- SEQUENCE TO SEQUENCE FOR SETS.pdf](sandbox:/mnt/data/8%20-%20ORDER%20MATTERS-%20SEQUENCE%20TO%20SEQUENCE%20FOR%20SETS.pdf)
* [13 - Attention Is All You Need.pdf](sandbox:/mnt/data/13%20-%20Attention%20Is%20All%20You%20Need.pdf)

### 5. Memory, reasoning, and graphs

*Connecting thread:* Different routes to structured computation: external memory, relational modules, and message passing.

* [20 - Neural Turing Machines.pdf](sandbox:/mnt/data/20%20-%20Neural%20Turing%20Machines.pdf)
* [16 - A simple neural network module for relational reasoning.pdf](sandbox:/mnt/data/16%20-%20A%20simple%20neural%20network%20module%20for%20relational%20reasoning.pdf)
* [12 - Neural Message Passing for Quantum Chemistry.pdf](sandbox:/mnt/data/12%20-%20Neural%20Message%20Passing%20for%20Quantum%20Chemistry.pdf)
* [18 - Relational recurrent neural networks.pdf](sandbox:/mnt/data/18%20-%20Relational%20recurrent%20neural%20networks.pdf)

### 6. Scaling and systems

*Connecting thread:* Once optimization is stable, the frontier shifts to compute, data, and infrastructure.

* [9 - GPipe- Easy Scaling with Micro-Batch Pipeline Parallelism.pdf](sandbox:/mnt/data/9%20-%20GPipe-%20Easy%20Scaling%20with%20Micro-Batch%20Pipeline%20Parallelism.pdf)
* [22 - Scaling Laws for Neural Language Models.pdf](sandbox:/mnt/data/22%20-%20Scaling%20Laws%20for%20Neural%20Language%20Models.pdf)
* [27A - Better & Faster Large Language Models via Multi-token Prediction.pdf](sandbox:/mnt/data/27A%20-%20Better%20%26%20Faster%20Large%20Language%20Models%20via%20Multi-token%20Prediction.pdf)

### 7. Retrieval, long context, new knowledge, and verification

*Connecting thread:* Modern LLM stacks: augment memory with retrieval, diagnose context failure modes, evaluate reliability.

* [27B - Dense Passage Retrieval for Open-Domain Question Answering.pdf](sandbox:/mnt/data/27B%20-%20Dense%20Passage%20Retrieval%20for%20Open-Domain%20Question%20Answering.pdf)
* [27C - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf](sandbox:/mnt/data/27C%20-%20Retrieval-Augmented%20Generation%20for%20Knowledge-Intensive%20NLP%20Tasks.pdf)
* [30A - Precise Zero-Shot Dense Retrieval without Relevance Labels.pdf](sandbox:/mnt/data/30A%20-%20Precise%20Zero-Shot%20Dense%20Retrieval%20without%20Relevance%20Labels.pdf)
* [29 - Lost in the Middle- How Language Models Use Long Contexts.pdf](sandbox:/mnt/data/29%20-%20Lost%20in%20the%20Middle-%20How%20Language%20Models%20Use%20Long%20Contexts.pdf)
* [30B - ALCUNA- Large Language Models Meet New Knowledge.pdf](sandbox:/mnt/data/30B%20-%20ALCUNA-%20Large%20Language%20Models%20Meet%20New%20Knowledge.pdf)
* [30C - The Perils & Promises of Fact-checking with Large Language Models.pdf](sandbox:/mnt/data/30C%20-%20The%20Perils%20%26%20Promises%20of%20Fact-checking%20with%20Large%20Language%20Models.pdf)

### 8. Alignment as a training recipe

*Connecting thread:* Turning raw capability into behavior you can deploy and steer.

* [28 - ZEPHYR- DIRECT DISTILLATION OF LM ALIGNMENT.pdf](sandbox:/mnt/data/28%20-%20ZEPHYR-%20DIRECT%20DISTILLATION%20OF%20LM%20ALIGNMENT.pdf)
