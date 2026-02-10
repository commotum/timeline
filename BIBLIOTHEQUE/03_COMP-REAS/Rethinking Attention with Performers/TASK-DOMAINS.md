# RETHINKING ATTENTION WITH PERFORMERS (2020)
Source: Rethinking Attention with Performers.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Text language modeling (next-token prediction) | text token sequences | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | next text tokens | 1D (t) | Capped (inferred) |
| Text masked language modeling | text token sequences with masked tokens | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | masked text tokens | 1D (t) | Capped (inferred) |
| Protein language modeling (next-token prediction) | protein sequence tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | next amino-acid tokens | 1D (t) | Capped |
| Protein masked language modeling | protein sequence tokens with masked tokens | 1D (t) | Capped | Static (inferred) | Direct (inferred) | masked amino-acid tokens | 1D (t) | Capped |
| Pixel prediction (ImageNet64, autoregressive) | image pixels | 1D (t); 2D (x, y) (inferred) | Fixed | Static (inferred) | Direct (inferred) | predicted pixel values | 1D (t); 2D (x, y) (inferred) | Fixed |
| Protein interaction prediction (concatenated TrEMBL benchmark) | concatenated protein sequence tokens | 1D (t) | Fixed | Static (inferred) | Direct (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| ListOps classification (Long Range Arena) | ListOps token sequences | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D | Fixed (inferred) |
| Byte-level text classification (Long Range Arena) | byte-level text sequences | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D | Fixed (inferred) |
| Byte-level document retrieval (Long Range Arena) | byte-level documents | 1D (t) | Capped (inferred) | Static (inferred) | Direct (inferred) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Image classification on sequences of pixels (Long Range Arena) | pixel sequences | 1D (t); 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D | Fixed (inferred) |
| Pathfinder task (long-range spatial dependency) | spatial layouts/images | 2D (x, y) (inferred) | Capped (inferred) | Static (inferred) | Direct (inferred) | class label (inferred) | 0D | Fixed (inferred) |

## Summary
The paper evaluates Performers on text, protein, and image tasks, and additionally reports Long Range Arena results spanning classification, retrieval, and spatial-reasoning style benchmarks. Most workloads are sequence-oriented (tokens/bytes/pixels as 1D streams), while image and spatial tasks also support a 2D interpretation. Reported sequence lengths (for example, L=1024, L=8192, L=12288) support Capped/Fixed dynamics for many tasks. Attention and state are mostly inferred as Static and Direct, consistent with standard Transformer-style sequence processing in the described setups.

## Evidence
### Task: Text language modeling (next-token prediction)
- "For unidirectional models, we measure the accuracy on next-token prediction, averaged across all sequence positions in the dataset." (Appendix A.1 Metrics)
- "The PG-19 dataset (Rae et al., 2020) is presented as a challenging long range text modeling task." (Appendix A.1.1 PG-19 Preprocessing)
- Inference: `Capped`, `Static`, and `Direct` are inferred from sequence-model setup and Transformer attention over bounded sequence windows discussed throughout Section 4 and Appendix A.

### Task: Text masked language modeling
- "For shorthand notation, we denote unidirectional/causal modelling as (U) and bidirectional/masked language modelling as (U)." (Section 4 Experiments)
- "For bidirectional models, we mask each token with 15% probability (same as (Devlin et al., 2018)) and measure accuracy across the masked positions." (Appendix A.1 Metrics)
- Inference: `Capped`, `Static`, and `Direct` are inferred from the same Transformer sequence interface and masked-token evaluation protocol.

### Task: Protein language modeling (next-token prediction)
- "We further benchmark the Performer on both (U) and (B) cases by training a 36-layer model using protein sequences from the Jan. 2019 release of TrEMBL (Consortium, 2019)." (Section 4.4 Multiple Layer Training for Proteins)
- "Table 2 contains the results on the single protein sequence modeling task (L=1024)." (Appendix C.3 Tabular Results)
- Inference: `Static` and `Direct` are inferred from standard Transformer-style attention/state use; token-level next-token behavior follows the unidirectional metric definition in Appendix A.1.

### Task: Protein masked language modeling
- "Table 2: Results on single protein sequence modeling (L=1024). We note that the empirical baseline results are applicable to both the unidirectional (UNI) and bidirectional (BID) models." (Appendix C.3 Tabular Results)
- "For bidirectional models, we mask each token with 15% probability (same as (Devlin et al., 2018)) and measure accuracy across the masked positions." (Appendix A.1 Metrics)
- Inference: `Static` and `Direct` are inferred from the same architecture-level reasoning as above.

### Task: Pixel prediction (ImageNet64, autoregressive)
- "We tested Performers on a rich set of tasks stretching from pixel-prediction through text models to protein sequence modeling." (Abstract)
- "On the standard (U) ImageNet64 benchmark from (Parmar et al., 2018) with L=12288 which is unfeasible for regular Transformers, we set all models to use the same  $(n_{heads}, d_{ff}, d)$  but varying  $n_{layers}$ ." (Section 4.5 Large length training - Common datasets)
- Inference: 2D image interpretation and `Static`/`Direct` are inferred; the paper explicitly reports sequence form and fixed length (`L=12288`).

### Task: Protein interaction prediction (concatenated TrEMBL benchmark)
- "we also create an initial protein benchmark for predicting interactions among groups of proteins by concatenating protein sequences to length L=8192 from TrEMBL" (Section 4.5 Large length training - Common datasets)
- "In the long sequence task, the training and validation sets are obtained by concatenating the sequences, separated by an end-of-sequence token, and grouping the resulting chain into non-overlapping sequences of length L=8192." (Appendix C.1 TrEMBL Dataset)
- Inference: `Static` and `Direct` are inferred from the same Transformer setup; output object/domain is not explicitly specified, so output fields are marked as not specified.

### Task: ListOps classification (Long Range Arena)
- "Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018)" (Section D.5 Long Range Arena)
- "Performers are compared against many additional (scalable and not scalable) methods not included in our paper: *Local Attention, Sparse Attention, Longformer, Sinkhorn Transformer, Synthesizer, Big Bird* and the aforementioned *Linear Transformer* on challenging long range context tasks in the Long Range Arena (Tay et al., 2021), with Fig. 19 displaying the original paper's results." (Section D.5 Long Range Arena)
- Inference: `Capped`, `Static`, `Direct`, and fixed-size class output are inferred from benchmark framing and Transformer usage.

### Task: Byte-level text classification (Long Range Arena)
- "Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018), (2) byte-level text classification using real-world data, (3) byte-level document retrieval, (4) image classification on sequences of pixels, and (5) Pathfinder task (long-range spatial dependency problem)." (Section D.5 Long Range Arena)
- "Performers are compared against many additional (scalable and not scalable) methods not included in our paper: *Local Attention, Sparse Attention, Longformer, Sinkhorn Transformer, Synthesizer, Big Bird* and the aforementioned *Linear Transformer* on challenging long range context tasks in the Long Range Arena (Tay et al., 2021), with Fig. 19 displaying the original paper's results." (Section D.5 Long Range Arena)
- Inference: `Capped`, `Static`, `Direct`, and fixed-size class output are inferred from benchmark framing and Transformer usage.

### Task: Byte-level document retrieval (Long Range Arena)
- "Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018), (2) byte-level text classification using real-world data, (3) byte-level document retrieval, (4) image classification on sequences of pixels, and (5) Pathfinder task (long-range spatial dependency problem)." (Section D.5 Long Range Arena)
- "Performers are compared against many additional (scalable and not scalable) methods not included in our paper: *Local Attention, Sparse Attention, Longformer, Sinkhorn Transformer, Synthesizer, Big Bird* and the aforementioned *Linear Transformer* on challenging long range context tasks in the Long Range Arena (Tay et al., 2021), with Fig. 19 displaying the original paper's results." (Section D.5 Long Range Arena)
- Inference: `Capped`, `Static`, and `Direct` are inferred from benchmark framing; output structure is not explicitly specified in this OCR text.

### Task: Image classification on sequences of pixels (Long Range Arena)
- "Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018), (2) byte-level text classification using real-world data, (3) byte-level document retrieval, (4) image classification on sequences of pixels, and (5) Pathfinder task (long-range spatial dependency problem)." (Section D.5 Long Range Arena)
- "Performers are compared against many additional (scalable and not scalable) methods not included in our paper: *Local Attention, Sparse Attention, Longformer, Sinkhorn Transformer, Synthesizer, Big Bird* and the aforementioned *Linear Transformer* on challenging long range context tasks in the Long Range Arena (Tay et al., 2021), with Fig. 19 displaying the original paper's results." (Section D.5 Long Range Arena)
- Inference: 2D image interpretation, `Capped`, `Static`, `Direct`, and fixed-size class output are inferred from task phrasing and Transformer benchmark setup.

### Task: Pathfinder task (long-range spatial dependency)
- "Tasks used for comparison include: (1) a longer variation of the standard ListOps task proposed in (Nangia & Bowman, 2018), (2) byte-level text classification using real-world data, (3) byte-level document retrieval, (4) image classification on sequences of pixels, and (5) Pathfinder task (long-range spatial dependency problem)." (Section D.5 Long Range Arena)
- "In the Long Range Arena paper, the authors found that all models do not learn anything on Path-X task (denoted by FAIL), contrary to the Pathfinder task, which shows that increasing the sequence length can cause seriously difficulties for model training." (Section D.5 Long Range Arena)
- Inference: spatial 2D input framing, `Capped`, `Static`, `Direct`, and fixed-size class output are inferred from the benchmark description.
