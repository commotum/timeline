## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column contains a non-URL sentinel like "MISSING" or similar text.
- Examples: 2006 "A Fast Learning Algorithm for Deep Belief Nets" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2017 "A Distributional Perspective on Reinforcement Learning" with url `https://arxiv.org/abs/1707.06887` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different URLs (e.g., arXiv vs proceedings, or multiple mirrors).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2023 "4D-Former: Multimodal 4D Panoptic Segmentation" has both `https://proceedings.mlr.press/v229/athar23a/athar23a.pdf` and `https://arxiv.org/abs/2311.01520`; 1948 "A Mathematical Theory of Communication" has `https://ia803209.us.archive.org/...` and `https://people.math.harvard.edu/...`; 2024 "A 2D nGPT Model for ARC Prize" appears with `https://arcprize.org/competitions/2024/` and `https://arxiv.org/html/2412.04604v2`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include aliases, parenthetical variants, or inconsistent casing that refer to the same work.
- Structural characteristics: near-identical titles differing only by case or added alias text.
- Examples: 2023 "A Length-Extrapolatable Transformer" vs 2023 "A Length-Extrapolatable Transformer (XPOS / LeX)"; 2024 "A 2D nGPT Model For ARC Prize" vs 2024 "A 2D nGPT Model for ARC Prize".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after removing parentheticals and normalizing case.

## URL Field Contains Extra Text
- Description: The url cell includes additional notes or labels beyond a single URL.
- Structural characteristics: url field contains whitespace and non-URL text after a URL; often quoted.
- Examples: 2023 "A Length-Extrapolatable Transformer (XPOS / LeX)" has a url field that includes `https://aclanthology.org/2023.acl-long.816.pdf` plus extra label text like `XPOS / LeX (2023)`.
- Why harmful: URL parsing fails, and automated linkers treat the value as invalid.
- Detection signals: url field fails strict URL regex or contains spaces/quoted annotations after a URL.

## Non-paper Context URL
- Description: The url points to a context or encyclopedia page rather than the paper itself.
- Structural characteristics: domains or paths indicate competitions, wiki pages, or general info pages.
- Examples: 2024 "A 2D nGPT Model For ARC Prize" uses `https://arcprize.org/competitions/2024/`; 1984 "A Theory of the Learnable (PAC Learning)" uses `https://en.wikipedia.org/wiki/Probably_approximately_correct_learning`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources (wikipedia.org, competitions, events, wiki).

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `/abs/`, `/doi/`, or publisher article pages.
- Examples: `https://aclanthology.org/2020.coling-main.580/` (ACL landing), `https://arxiv.org/abs/2109.08141` (arXiv abstract), `https://pubmed.ncbi.nlm.nih.gov/7351125/` (PubMed), `https://www.sciencedirect.com/science/article/pii/S0004370299000521` (ScienceDirect), `https://www.science.org/doi/10.1126/sciadv.aay2631` (Science DOI).
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF/DOI.
- Structural characteristics: URL contains `/html/` or `/full` endpoints.
- Examples: `https://arxiv.org/html/2412.04604v2`, `https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00121/full`, `https://dl.acm.org/doi/full/10.1145/3641289`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/` or `/full` (including `/doi/full/`).

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal host instead of an official publisher or DOI.
- Structural characteristics: domains like archive.org or personal university homepages.
- Examples: 1948 "A Mathematical Theory of Communication" uses `https://ia803209.us.archive.org/...` and `https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `archive.org` or `~` in paths.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied in the URL or DOI metadata.
- Structural characteristics: DOI or URL includes a different year token than the row year.
- Examples: 2019 "A Framework for Intelligence and Cortical Function Based on Grid Cells in the Neocortex" has DOI `10.3389/fncir.2018.00121` (suggesting 2018).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in DOI/URL and compare with the year column.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2024 "A 2D nGPT Model for ARC Prize" uses `https://arxiv.org/html/2412.04604v2`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms across the dataset.
- Structural characteristics: same base domain `arxiv.org` with varying path types.
- Examples: `https://arxiv.org/abs/2109.08141`, `https://arxiv.org/pdf/2502.11075`, `https://arxiv.org/html/2412.04604v2`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv URLs and flag mixed path types.

## Landing or Abstract Page URL
- Description: The url points to an abstract/landing page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `/abs/` or publisher article pages.
- Examples: 2023 "Grokking Modular Arithmetic" uses `https://arxiv.org/abs/2301.02679`; 2025 "HARPE: Head-Adaptive Rotary Position Encoding" uses `https://aclanthology.org/2025.coling-main.326/`; 2025 "H-ARC (Human-ARC): A Comprehensive Behavioral Dataset for the Abstraction and Reasoning Corpus" uses `https://www.nature.com/articles/s41597-025-05687-1`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended notes/labels instead of a single clean URL value.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or label fragments.
- Examples: 2024 "H-ARC: A Robust Estimate of Human Performance on the Abstraction and Reasoning Corpus Benchmark" has `https://arxiv.org/pdf/2409.01374.pdf` plus label text like `H-ARC (2024) - arXiv`; 2025 "Head-Wise Adaptive Rotary Positional Encoding (HARoPE)" has `https://arxiv.org/pdf/2510.10489.pdf` plus label text; 2019 "HowTo100M: Learning a Text-Video Embedding by Watching Narrated Videos" has `https://arxiv.org/pdf/1906.03327.pdf` plus label text.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF/DOI.
- Structural characteristics: URL contains `/html/` endpoints.
- Examples: 2025 "Hierarchical Reasoning Model (HRM)" uses `https://arxiv.org/html/2506.21734v1`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/`.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "Hierarchical Reasoning Model (HRM)" uses `https://arxiv.org/html/2506.21734v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms within the same dataset chunk.
- Structural characteristics: same base domain `arxiv.org` with varying path types across rows.
- Examples: 2023 "Grokking Modular Arithmetic" uses `https://arxiv.org/abs/2301.02679`; 2022 "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets" uses `https://arxiv.org/pdf/2201.02177`; 2025 "Hierarchical Reasoning Model (HRM)" uses `https://arxiv.org/html/2506.21734v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv URLs and flag mixed path types.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (arXiv vs proceedings or HTML vs PDF).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2018 "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" appears with `https://arxiv.org/abs/1809.09600` and `https://aclanthology.org/D18-1259/`; 2025 "Hierarchical Reasoning Model (HRM)" appears with `https://arxiv.org/html/2506.21734v1` and `https://arxiv.org/pdf/2506.21734.pdf` (plus label text); 2022 "Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks" appears twice with the same base URL but different appended labels.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2015 "Human-level Control through Deep Reinforcement Learning (DQN)" appears twice with `https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf`; 2018 "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures" appears twice with `https://arxiv.org/abs/1802.01561`; 2018 "Image Transformer" appears twice with `https://arxiv.org/abs/1802.05751`.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Non-canonical Mirror Source
- Description: The url points to a mirror or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains like university class pages, lab sites, or institutional repositories.
- Examples: 2015 "Human-level Control through Deep Reinforcement Learning (DQN)" uses `https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf`; 2014 "Hierarchy of Intrinsic Timescales in Cortex" uses `https://www.cns.nyu.edu/wanglab/publications/pdf/murray.nn2014.pdf`; 2013 "ICFP Programming Competition" uses `https://dspace.mit.edu/bitstream/handle/1721.1/137904/1611.07627.pdf?isAllowed=y&sequence=2`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `~`, `class`, `lab`, or `dspace` in paths.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL.
- Structural characteristics: URL includes a different year token than the row year.
- Examples: 2013 "ICFP Programming Competition" links to a PDF path containing `1611.07627` (suggesting 2016).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (e.g., arXiv-style IDs) and compare with the year column.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column is a sentinel like `MISSING` rather than an http(s) URL.
- Examples: 2012 "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)" has url `MISSING`; 2021 "Jacobian regularization for equilibrium models" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholders (MISSING, TBD, N/A).

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label or notes.
- Structural characteristics: quoted url field contains a URL followed by extra label text and spaces.
- Examples: 2021 "Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)" uses `https://arxiv.org/pdf/2102.05095.pdf` plus `TimeSformer (2021) - arXiv`; 2023 "Kosmos-1: Language Is Not All You Need" uses `https://arxiv.org/pdf/2302.14045.pdf` plus `Kosmos-1 (2023) - arXiv`; 2023 "LLaVA: Large Language-and-Vision Assistant" uses `https://arxiv.org/pdf/2304.08485.pdf` plus label text.
- Why harmful: URL parsing fails, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Landing or Abstract Page URL
- Description: The url points to a landing, abstract, or DOI page instead of a full-text PDF or canonical artifact.
- Structural characteristics: urls lack `.pdf` and use `/abs/`, `/doi/`, or forum/abstract landing patterns.
- Examples: 2015 "Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets" uses `https://arxiv.org/abs/1503.01007`; 2024 "Implicit Factorized Transformer (IFactFormer)" uses `https://www.sciencedirect.com/science/article/pii/S2095034924000382`; 2024 "LLFormer4D: LiDAR-based lane detection method" uses `https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12338`; 2024 "LTD-Bench: Evaluating Large Language Models by Letting Them Draw" uses `https://openreview.net/forum?id=TG5rvKyEbu`; 2025 "LVSM: A Large View Synthesis Model with Minimal 3D Inductive Bias" uses `https://proceedings.iclr.cc/paper_files/paper/2025/hash/9676c5283df26cabca412ca66b164a7d-Abstract-Conference.html`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` forms within the same dataset chunk.
- Structural characteristics: arxiv.org URLs use different path types for the same corpus.
- Examples: 2015 "Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets" uses `https://arxiv.org/abs/1503.01007` while 2021 "Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)" uses `https://arxiv.org/pdf/2102.05095.pdf`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-canonical Mirror Source
- Description: The url points to a mirror or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains indicate university hosts, lab sites, or personal homepages.
- Examples: 1957 "Information Theory and Statistical Mechanics" uses `https://bayes.wustl.edu/etj/articles/theory.1.pdf`; 2022 "Jacobian-Free Backpropagation for Implicit Networks (JFB)" uses `https://www.math.emory.edu/site/cmds-reuret/projects/2022-implicit/JFB.pdf`; 2020 "Just-in-Time Learning for Bottom-Up Enumerative Synthesis (PROBE)" uses `https://cseweb.ucsd.edu/~hpeleg/probe-oopsla20.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `~` or institutional path patterns.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2022 "LIFT: Learning 4D LiDAR Image Fusion Transformer for 3D Object Detection" appears twice with `https://openaccess.thecvf.com/content/CVPR2022/papers/Zeng_LIFT_Learning_4D_LiDAR_Image_Fusion_Transformer_for_3D_Object_CVPR_2022_paper.pdf`.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2022 "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models" appears with an arXiv PDF URL (plus label text) and also as 2022 "LAION-5B: An open large-scale dataset for training next generation image-text models" at `https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include case or wording variants that refer to the same work.
- Structural characteristics: near-identical titles differing only by case or minor phrasing.
- Examples: 2022 "LAION-5B: An Open Large-Scale Dataset for Training Next Generation Image-Text Models" vs 2022 "LAION-5B: An open large-scale dataset for training next generation image-text models".

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label or notes instead of a single clean URL value.
- Structural characteristics: quoted url field contains a URL followed by extra text, often separated by spaces and additional quotes.
- Examples: 2018 "Recurrent Relational Networks" has a url value starting with `https://proceedings.neurips.cc/paper_files/paper/2018/file/b9f94c77652c9a76fc8a442748cd54bd-Paper.pdf` followed by appended label text; 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)" has `https://arxiv.org/pdf/2104.09864.pdf` plus appended label text; 2024 "RoTHP: Rotary Position Embedding-based Transformer Hawkes Process" has `https://arxiv.org/pdf/2405.06985.pdf` plus appended label text.
- Why harmful: URL parsing fails, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted fragments after a URL.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2020 "Rethinking Attention with Performers" with `https://arxiv.org/pdf/2009.14794` appears twice; 2020 "Rethinking Positional Encoding in Language Pre-training (TUPE)" with `https://arxiv.org/pdf/2006.15595` appears three times; 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding" with `https://arxiv.org/pdf/2104.09864` appears three times.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (arXiv vs proceedings, PDF vs landing page, or code repo).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2018 "Recurrent Relational Networks" appears with a NeurIPS PDF (`https://proceedings.neurips.cc/...-Paper.pdf`), a papers.neurips PDF (`https://papers.neurips.cc/paper/7597-recurrent-relational-networks.pdf`), and an arXiv abstract (`https://arxiv.org/abs/1711.08028`); 2024 "Resonance RoPE: Improving Context Length Generalization of Large Language Models" appears with `https://arxiv.org/abs/2403.00071` and `https://aclanthology.org/2024.findings-acl.32.pdf`; 2020 "Rethinking Positional Encoding in Language Pre-training (TUPE)" appears with `https://openreview.net/pdf?id=09-528y2Fgf` and `https://arxiv.org/pdf/2006.15595`; 2017 "RobustFill: Neural Program Learning under Noisy I/O" appears with `https://arxiv.org/abs/1703.07469` and `https://github.com/thelmuth/program-synthesis-benchmark-datasets`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include aliases or parenthetical variants that refer to the same work.
- Structural characteristics: near-identical titles differing only by added acronyms or parenthetical labels.
- Examples: 2018 "Recurrent Relational Networks" vs 2018 "Recurrent Relational Networks (RRN)"; 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding" vs 2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after removing parentheticals and normalizing case.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column contains a non-URL sentinel like "MISSING".
- Examples: 2023 "Reflexion / Self-Refine family (test-time improvement loops)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Landing or Abstract Page URL
- Description: The url points to a landing, abstract, or forum page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `/abs/`, `/forum`, `/article/view`, or `/doi/`.
- Examples: 2018 "Recurrent Relational Networks (RRN)" uses `https://arxiv.org/abs/1711.08028`; 2023 "Reflexion: Language Agents with Verbal Reinforcement Learning" uses `https://arxiv.org/abs/2303.11366`; 2025 "Reflection System for the Abstraction and Reasoning Corpus" uses `https://openreview.net/forum?id=kRFwzuv0ze`; 2010 "Relative Entropy Policy Search (REPS)" uses `https://ojs.aaai.org/index.php/AAAI/article/view/7727`; 2025 "RetentiveBEV: BEV transformer for visual 3D object detection" uses `https://journals.sagepub.com/doi/10.1177/01423312241308367`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF/DOI.
- Structural characteristics: URL contains `/html/` paths and ends with `.html`.
- Examples: 2025 "Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation" uses `https://openaccess.thecvf.com/content/CVPR2025/html/Lu_Relation3D__Enhancing_Relation_Modeling_for_Point_Cloud_Instance_Segmentation_CVPR_2025_paper.html`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/` or ends with `.html`.

## Non-paper Context URL
- Description: The url points to a blog post, project page, or repository rather than the paper itself.
- Structural characteristics: domains or paths indicate blogs, competition posts, or code repositories.
- Examples: 2025 "Rethinking Visual Intelligence: Insights from Video Pretraining" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2017 "RobustFill: Neural Program Learning under Noisy I/O" uses `https://github.com/thelmuth/program-synthesis-benchmark-datasets`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources (blogs, competitions, github.com).

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` forms (and sometimes `.pdf` suffixes) within the same chunk.
- Structural characteristics: same base domain `arxiv.org` with varying path types across rows.
- Examples: `https://arxiv.org/abs/1711.08028` (2018 "Recurrent Relational Networks (RRN)"), `https://arxiv.org/pdf/2009.14794` (2020 "Rethinking Attention with Performers"), `https://arxiv.org/pdf/2104.09864.pdf` (2021 "RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)").
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Composite or Umbrella Entry
- Description: The title describes a family of methods or a conceptual grouping rather than a single paper.
- Structural characteristics: title uses slashes, "family", or explanatory parentheticals indicating multiple related works, often paired with missing or placeholder URLs.
- Examples: 2023 "Reflexion / Self-Refine family (test-time improvement loops)" with url `MISSING`.
- Why harmful: Cannot be linked to a single canonical artifact, leading to ambiguous indexing and broken citations.
- Detection signals: title contains terms like "family", multiple names separated by slashes, or descriptive parentheticals not tied to a specific paper.
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: normalize case and punctuation, then compare fuzzy title similarity.

## Non-paper Context URL
- Description: The url points to a project, dataset, or announcement page rather than a paper or canonical artifact.
- Structural characteristics: domains or paths indicate dataset home pages, project sites, or blog posts.
- Examples: "Year unknown" "Karel dataset" uses `https://msr-redmond.github.io/karel-dataset/`; 2022 "LAION-5B: An open large-scale dataset for training next generation image-text models" uses `https://laion.ai/laion-5b-a-new-era-of-open-large-scale-multi-modal-datasets/`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper when one exists.
- Detection signals: domain/path matches known project sites, blog/news slugs, or dataset homepage patterns.

## Non-numeric or Placeholder Year Value
- Description: The year column contains non-numeric placeholder text instead of a four-digit year.
- Structural characteristics: year value includes words like "Year unknown" or other non-year tokens.
- Examples: "Year unknown" "Karel dataset" uses year `Year unknown`.
- Why harmful: Breaks sorting, timeline analyses, and numeric parsing.
- Detection signals: year fails numeric regex `^[0-9]{4}$` or contains alphabetic characters.

## Placeholder URL Value
- Description: The url field uses a placeholder token instead of a real URL.
- Structural characteristics: url column equals `MISSING` or another sentinel in place of http(s).
- Examples: 2020 "Language Models are Few-Shot Learners (GPT-3)" has url `MISSING`; 2012 "Learning Semantic String Transformations from Examples" has url `MISSING`.
- Why harmful: Prevents automated linking and introduces gaps that can be mistaken for distinct entries.
- Detection signals: url fails URL regex and matches known placeholder tokens (MISSING, TBD, N/A).

## Exact Duplicate Row
- Description: Identical year, title, and url rows appear more than once.
- Structural characteristics: duplicate CSV lines with the same three fields.
- Examples: 2020 "Language Models are Few-Shot Learners (GPT-3)" with `https://arxiv.org/pdf/2005.14165` appears twice; 2021 "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" with `https://arxiv.org/pdf/2103.00020` appears twice.
- Why harmful: Inflates counts and biases trend analyses.
- Detection signals: row-level hash or (year,title,url) key repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: One paper is repeated across rows with different url values (including placeholders).
- Structural characteristics: same title/year normalized appears with multiple distinct urls.
- Examples: 2020 "Language Models are Few-Shot Learners (GPT-3)" has `MISSING` and `https://arxiv.org/pdf/2005.14165`; 2012 "Learning Semantic String Transformations from Examples" has `MISSING` and `https://vldb.org/pvldb/vol5/p740_rishabhsingh_vldb2012.pdf`; 2025 "Less is More: Recursive Reasoning with Tiny Networks" appears with an arXiv PDF (plus label text), `https://ar5iv.org/abs/2510.04871`, and an arcprize blog URL.
- Why harmful: Splits citations across rows and complicates canonicalization.
- Detection signals: normalize title/year, then flag multiple distinct url strings.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended labels or notes.
- Structural characteristics: quoted url field contains a URL followed by extra text or a second quoted fragment.
- Examples: 2024 "Learning Iterative Reasoning through Energy Diffusion" has `https://arxiv.org/pdf/2406.11179.pdf` plus label text like "Energy Diffusion Iterative Reasoning (2024) - arXiv"; 2021 "Learning Transferable Visual Models From Natural Language Supervision (CLIP)" has `https://arxiv.org/pdf/2103.00020.pdf` plus "CLIP (2021) - arXiv"; 2024 "Length Extrapolation of Causal Transformers without Position Encoding (NoPE)" has an ACL Anthology PDF plus appended label text.
- Why harmful: URL parsing fails or produces invalid links, and noisy annotations fragment deduplication.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/` or publisher article pages.
- Examples: 2016 "Layer Normalization" uses `https://arxiv.org/abs/1607.06450`; 2024 "Length Generalization of Causal Transformers without Position Encoding" uses `https://arxiv.org/abs/2404.12224`; 1986 "Learning Representations by Back-Propagating Errors" uses `https://www.nature.com/articles/323533a0`; 2003 "Least-Squares Policy Iteration (LSPI)" uses `https://www.jmlr.org/papers/v4/lagoudakis03a.html`.
- Why harmful: Automated retrieval may require scraping and can be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/html/` endpoints.
- Examples: 2025 "Learning Modular Exponentiation with Transformers" uses `https://arxiv.org/html/2506.23679v1`.
- Why harmful: HTML renders can change, rely on scripts, and are less stable than PDFs.
- Detection signals: url contains `/html/` or `/full`.

## Version-specific arXiv URL
- Description: The arXiv url includes a specific version suffix (v1, v2, etc.) instead of the canonical ID.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "Learning Modular Exponentiation with Transformers" uses `https://arxiv.org/html/2506.23679v1`.
- Why harmful: Creates duplicate entries across versions and may become stale.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` forms within the same chunk.
- Structural characteristics: same arxiv.org domain with different path types across rows.
- Examples: 2020 "Language Models are Few-Shot Learners (GPT-3)" uses `https://arxiv.org/pdf/2005.14165`; 2016 "Layer Normalization" uses `https://arxiv.org/abs/1607.06450`; 2025 "Learning Modular Exponentiation with Transformers" uses `https://arxiv.org/html/2506.23679v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal host rather than an official publisher or DOI.
- Structural characteristics: domains indicate personal sites or mirrors (e.g., ar5iv.org).
- Examples: 1988 "Learning to Predict by the Methods of Temporal Differences" uses `https://incompleteideas.net/papers/sutton-88-with-erratum.pdf`; 2025 "Less is More: Recursive Reasoning with Tiny Networks (TRM)" uses `https://ar5iv.org/abs/2510.04871`.
- Why harmful: Higher risk of link rot, unclear versioning, and duplicate sources.
- Detection signals: domain matches known mirrors (ar5iv.org) or personal-site patterns.

## Non-paper Context URL
- Description: The url points to a blog post or results writeup instead of the paper.
- Structural characteristics: domain/path indicates blog or announcement content rather than a paper artifact.
- Examples: 2025 "Less is More: Recursive Reasoning with Tiny Networks (TRM)" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents the source and blocks retrieval of the actual paper.
- Detection signals: url contains blog/news slugs or competition result pages.

## Title Variants or Aliases
- Description: The same work appears with and without parenthetical aliases or acronyms.
- Structural characteristics: titles differ only by appended parenthetical tags.
- Examples: 2025 "Less is More: Recursive Reasoning with Tiny Networks" vs 2025 "Less is More: Recursive Reasoning with Tiny Networks (TRM)".
- Why harmful: Dedupe by exact title fails and fragments the timeline.
- Detection signals: strip parentheticals and compare normalized titles for high similarity.

## Exact Duplicate Row
- Description: Identical year, title, and url rows are repeated within the same CSV chunk.
- Structural characteristics: duplicate lines with matching values across all three columns.
- Examples: 2018 "Leveraging Grammar and Reinforcement Learning for Neural Program Synthesis" with `https://arxiv.org/abs/1805.04276` appears twice; 2023 "MMBench: Is Your Multi-modal Model an All-around Player?" with `https://arxiv.org/abs/2307.06281` appears twice; 2023 "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI" with `https://arxiv.org/abs/2311.16502` appears twice.
- Why harmful: Inflates counts and breaks deduplication logic.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: The same work appears in multiple rows with different url values.
- Structural characteristics: identical or near-identical titles/years paired with different URLs (e.g., arXiv vs proceedings).
- Examples: 2018 "ListOps: A Diagnostic Dataset for Latent Tree Learning" appears with `https://arxiv.org/abs/1804.06028` and `https://aclanthology.org/N18-4013/`; 2023 "MMBench" entries use `https://arxiv.org/abs/2307.06281` and a PDF URL with appended label text; 2023 "MMMU" appears with `https://arxiv.org/abs/2311.16502` and `https://arxiv.org/pdf/2311.16502.pdf` (plus label text).
- Why harmful: Duplicates fragment citations and complicate canonicalization.
- Detection signals: normalize titles and years, then flag multiple distinct URLs for the same work.

## Landing or Abstract Page URL
- Description: The url points to an abstract or landing page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and match known landing patterns such as arXiv `/abs/` or proceedings index pages.
- Examples: 2024 "LiveBench: A Challenging, Contamination-Free LLM Benchmark" uses `https://arxiv.org/abs/2406.19314`; 2021 "LoRA: Low-Rank Adaptation of Large Language Models" uses `https://arxiv.org/abs/2106.09685`; 2020 "LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning" uses `https://www.ijcai.org/proceedings/2020/501`; 2018 "ListOps: A Diagnostic Dataset for Latent Tree Learning" uses `https://aclanthology.org/N18-4013/`.
- Why harmful: Automated retrieval may fail or require extra scraping steps.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract domains or path patterns.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended notes/labels instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by additional text or labels.
- Examples: 2024 "LieRE: Lie Rotational Positional Encodings" has `https://arxiv.org/pdf/2406.10322.pdf "LieRE (2024) — arXiv"`; 2023 "MMBench: Evaluating Multimodal LLMs" has `https://arxiv.org/pdf/2307.06281.pdf "MMBench (2023) — arXiv"`; 2023 "MMMU: A Massive Multidiscipline Multimodal Benchmark" has `https://arxiv.org/pdf/2311.16502.pdf "MMMU (2023) — arXiv"`.
- Why harmful: URL parsing fails and automated linkers treat the value as invalid.
- Detection signals: url field contains spaces or quoted labels after a URL.

## Non-canonical Mirror Source
- Description: The url points to a personal or institutional host rather than an official publisher or DOI.
- Structural characteristics: domains indicate personal pages or university hosts instead of publisher domains.
- Examples: 2023 "LogiQA 2.0 - An Improved Dataset for Logical Reasoning in NLU" uses `https://frcchang.github.io/pub/An%20Improved%20Dataset%20for%20Logical%20Reasoning%20in%20Natural%20Language%20Understanding.pdf`; 1997 "Long Short-Term Memory (LSTM)" uses `https://www.bioinf.jku.at/publications/older/2604.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: domain matches personal or institutional host patterns (e.g., github.io, university subdomains).

## Title Variants or Aliases
- Description: Titles differ in wording or hyphenation while referring to the same work.
- Structural characteristics: near-identical titles with variant phrasing, punctuation, or abbreviated forms.
- Examples: 2023 "MMBench: Evaluating Multimodal LLMs" vs 2023 "MMBench: Is Your Multi-modal Model an All-around Player?"; 2023 "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI" vs 2023 "MMMU: A Massive Multidiscipline Multimodal Benchmark".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after normalizing hyphenation and punctuation.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types for the same corpus.
- Examples: 2021 "LoRA: Low-Rank Adaptation of Large Language Models" uses `https://arxiv.org/abs/2106.09685` while 2024 "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens" uses `https://arxiv.org/pdf/2402.13753`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Title Contains Commentary or Provenance Note
- Description: The title field includes explanatory notes or provenance text that is not part of the official title.
- Structural characteristics: parenthetical phrases like "from ..." or embedded quoted fragments appended to the title.
- Examples: 2024 `MHaluBench (from "Unified Hallucination Detection...")`.
- Why harmful: Causes title-based matching to fail and obscures the canonical work name.
- Detection signals: title contains phrases like "from", embedded quotes, or ellipses that indicate commentary rather than a proper title.

## Landing or Abstract Page URL
- Description: The url points to an abstract or landing page instead of a full-text PDF or canonical artifact.
- Structural characteristics: url lacks `.pdf` and uses known landing patterns such as `arxiv.org/abs/`, `nature.com/articles/`, or `pubmed.ncbi.nlm.nih.gov`.
- Examples: 2022 "Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation" uses `https://arxiv.org/abs/2112.01527`; 2006 "Making Working Memory Work: A Computational Model of Learning in the Prefrontal Cortex and Basal Ganglia" uses `https://pubmed.ncbi.nlm.nih.gov/16378516/`; 2019 "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)" uses `https://www.nature.com/articles/s41586-020-03051-4`.
- Why harmful: Automated retrieval may fail or require scraping, and access can be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract domains or path patterns.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view instead of a canonical PDF or DOI.
- Structural characteristics: url contains `/html/` endpoints (often on arXiv).
- Examples: 2024 "Machine Learning for Modular Multiplication" uses `https://arxiv.org/html/2402.19254v1`; 2025 "MathClean: A Benchmark for Synthetic Mathematical Data" uses `https://arxiv.org/html/2502.19058v1`.
- Why harmful: HTML renders can change, require scripts, or be less stable than PDFs.
- Detection signals: url contains `/html/` or similar full-text render endpoints.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2024 "Machine Learning for Modular Multiplication" uses `https://arxiv.org/html/2402.19254v1`; 2025 "MathClean: A Benchmark for Synthetic Mathematical Data" uses `https://arxiv.org/html/2502.19058v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` forms within the same chunk.
- Structural characteristics: same arxiv.org domain with different path types across rows.
- Examples: 2022 "Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation" uses `https://arxiv.org/abs/2112.01527`; 2021 "Masked Autoencoders Are Scalable Vision Learners (MAE)" uses `https://arxiv.org/pdf/2111.06377.pdf` (plus label text); 2024 "Machine Learning for Modular Multiplication" uses `https://arxiv.org/html/2402.19254v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or DOI.
- Structural characteristics: URL includes a different year token than the row year (e.g., arXiv ID or publisher path year).
- Examples: 2022 "Mask2Former: Masked-attention Mask Transformer for Universal Image Segmentation" uses `https://arxiv.org/abs/2112.01527` (implies 2021); 2021 "MaskFormer" uses `.../ICCV2023/...` in the URL; 2019 "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)" uses `https://www.nature.com/articles/s41586-020-03051-4` (implies 2020); 2022 "Masked Autoencoders Are Scalable Vision Learners (MAE)" uses `https://arxiv.org/abs/2111.06377` (implies 2021).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs and compare with the year column.

## Exact Duplicate Row
- Description: Identical year, title, and url rows appear more than once.
- Structural characteristics: duplicate CSV lines with matching values across all three columns.
- Examples: 2023 "Mask4Former: Mask Transformer for 4D Panoptic Segmentation" with `https://arxiv.org/abs/2309.16133` appears twice; 2023 "MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts" with `https://arxiv.org/abs/2310.02255` appears twice; 2022 "MaxViT: Multi-Axis Vision Transformer" with `"https://arxiv.org/pdf/2204.01697.pdf ""MaxViT (2022) — arXiv"""` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text and spaces.
- Examples: 2021 "Masked Autoencoders Are Scalable Vision Learners (MAE)" has `"https://arxiv.org/pdf/2111.06377.pdf ""MAE (2021) — arXiv"""`; 2022 "MaxViT: Multi-Axis Vision Transformer" has `"https://arxiv.org/pdf/2204.01697.pdf ""MaxViT (2022) — arXiv"""`.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column equals a sentinel like `MISSING`.
- Examples: 2019 "Megatron-LM / large-scale transformer training" has url `MISSING`; 2022 "Memorizing Transformers / kNN-augmented attention (inference-time memory)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Non-canonical Mirror Source
- Description: The url points to a mirror or institutional host rather than an official publisher or DOI.
- Structural characteristics: domain indicates a university or lab host instead of a publisher.
- Examples: 2023 "Mask4D: End-to-End Mask-Based 4D Panoptic Segmentation for LiDAR Sequences" uses `https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/marcuzzi2023ral-meem.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain matches institutional host patterns or contains university subdomains.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (including missing URLs).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: "Masked Autoencoders Are Scalable Vision Learners (MAE)" appears with `https://arxiv.org/abs/2111.06377` and `https://arxiv.org/pdf/2111.06377.pdf` (plus label text); 2022 "Memorizing Transformers" appears with `https://arxiv.org/abs/2203.08913` and another row with url `MISSING`; 2021 "MaskFormer" appears with a CVF URL for a different paper and with `https://arxiv.org/abs/2107.06278`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include shortened or expanded variants that refer to the same work.
- Structural characteristics: near-identical titles differing by omitted subtitles or appended descriptors.
- Examples: 2021 "MaskFormer" vs 2021 "MaskFormer: Per-Pixel Classification is Not All You Need for Semantic Segmentation"; 2022 "Memorizing Transformers" vs 2022 "Memorizing Transformers / kNN-augmented attention (inference-time memory)".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping subtitles or alias phrases.

## Title-URL Mismatch
- Description: The url points to a different paper than the one named in the title.
- Structural characteristics: URL slug or filename clearly references another work unrelated to the title.
- Examples: 2021 "MaskFormer" links to `https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_DiffusionDet_Diffusion_Model_for_Object_Detection_ICCV_2023_paper.pdf`, which is a DiffusionDet paper, not MaskFormer.
- Why harmful: Mislinks entries, breaks citation integrity, and confuses automated retrieval.
- Detection signals: URL text contains a different paper name or keywords that do not appear in the title.

## Same Work Listed Under Multiple Years
- Description: The same paper is listed multiple times with different year values.
- Structural characteristics: identical or near-identical titles occur across rows with different years.
- Examples: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)" appears in 2019 and 2020 with the same URL; "Masked Autoencoders Are Scalable Vision Learners (MAE)" appears in 2021 and 2022.
- Why harmful: Skews timeline analyses and inflates counts across years.
- Detection signals: normalized title matches across rows with differing year values.

## URL Field Contains Comment or Local Copy Note (No URL)
- Description: The url field contains a comment or local-copy note but no actual URL.
- Structural characteristics: url begins with `#` or is quoted text without any `http`/`https` URL.
- Examples: 2025 "Maximizing the Position Embedding for Vision Transformers (MPVG)" has `"# ""MPVG (2025) — Local Copy (no public PDF link provided)"""`.
- Why harmful: Non-URL values break parsers and hide missing-source gaps behind commentary.
- Detection signals: url lacks `http`/`https` and contains comment markers or phrases like "Local Copy".

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash, indicating multiple labels in one row.
- Structural characteristics: title includes ` / ` to join alternative names or descriptions.
- Examples: 2019 "Megatron-LM / large-scale transformer training"; 2022 "Memorizing Transformers / kNN-augmented attention (inference-time memory)".
- Why harmful: Complicates title matching and can imply multiple works in a single entry.
- Detection signals: title contains slash-separated segments or multiple alias phrases.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label or note text instead of a single clean URL value.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or labels.
- Examples: 2024 "Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs" has a NeurIPS PDF URL plus appended label text; 2023 "MiniGPT-4" has `https://arxiv.org/pdf/2304.10592.pdf` plus extra label text.
- Why harmful: URL parsing fails, and deduplication by URL becomes unreliable.
- Detection signals: url field contains whitespace or additional quoted segments after a URL.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year repeated with different URLs.
- Examples: 2024 "Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs" appears with an arXiv abstract URL and a NeurIPS PDF URL; 2024 "Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models" appears with a personal-site PDF and an ARC Prize competition page; 2015 "Neural Programmer-Interpreters (NPI)" appears with an arXiv abstract URL and `MISSING`.
- Why harmful: Creates duplicate entries, fragments citations, and complicates canonicalization.
- Detection signals: normalized title+year appears multiple times with distinct URLs (including placeholders).

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2018 "Music Transformer" with `https://arxiv.org/pdf/1809.04281` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/` or publisher article pages.
- Examples: 2014 "Microsoft COCO: Common Objects in Context" uses `https://arxiv.org/abs/1405.0312`; 2005 "Microstructure of a spatial map in the entorhinal cortex" uses `https://www.nature.com/articles/nature03721`; 2008 "Natural Actor-Critic" uses `https://www.sciencedirect.com/science/article/pii/S0925231208000532`; 2017 "Neural Program Meta-Induction" uses `https://papers.nips.cc/paper/6803-neural-program-meta-induction`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns or domains.

## Non-paper Context URL
- Description: The url points to a competition, project, or blog context page instead of a paper.
- Structural characteristics: domain/path indicates competitions, project home pages, or results writeups.
- Examples: 2025 "MindsAI" and 2025 "NVARC" use `https://arcprize.org/competitions/2025/`; 2019 "NLVR2: A Corpus for Reasoning about Natural Language Grounded in Photographs" uses `https://lil.nlp.cornell.edu/nlvr/`; 2025 "NVARC solution to ARC-AGI-2 2025" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: url matches competition/blog/project site patterns and lacks a paper artifact.

## Non-canonical Mirror Source
- Description: The url points to a personal or non-publisher host instead of an official publisher or DOI.
- Structural characteristics: domain indicates a personal site hosting a PDF.
- Examples: 2024 "Mini-ARC: Solving Abstraction and Reasoning Puzzles with Small Transformer Models" uses `https://www.paulfletcherhill.com/mini-arc.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain is a personal site or non-publisher host.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column is a sentinel like `MISSING`.
- Examples: 2015 "Neural Programmer-Interpreters (NPI)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied in the URL or identifier.
- Structural characteristics: arXiv-style IDs encode a different year than the row year.
- Examples: 2022 "MuSiQue: Multihop Questions via Single-hop Question Composition" uses `https://arxiv.org/abs/2108.00573` (implies 2021).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URL/ID and compare with the year column.

## URL Field Contains Comment or Local Copy Note (No URL)
- Description: The url field contains a comment or local-copy note but no actual URL.
- Structural characteristics: url begins with `#` or is quoted text without any `http`/`https` URL.
- Examples: 2025 "Nested Learning: The Illusion of Deep Learning Architecture" has `# "Nested Learning (2025) -- Local Copy (no public PDF link provided)"`.
- Why harmful: Non-URL values break parsers and hide missing-source gaps behind commentary.
- Detection signals: url lacks `http`/`https` and contains comment markers or phrases like "Local Copy".

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/html/` and ends in `.html`.
- Examples: 2016 "Neural Module Networks (NMN)" uses `https://openaccess.thecvf.com/content_cvpr_2016/html/Andreas_Neural_Module_Networks_CVPR_2016_paper.html`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/` or ends with `.html` on paper-hosting domains.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types across rows.
- Examples: 2014 "Microsoft COCO: Common Objects in Context" uses `https://arxiv.org/abs/1405.0312` while 2018 "Music Transformer" uses `https://arxiv.org/pdf/1809.04281` and 2019 "NEZHA: Neural Contextualized Representation for Chinese Language Understanding" uses `https://arxiv.org/pdf/1909.00204`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column is a sentinel like `MISSING`.
- Examples: 2017 "Neural Theorem Provers / Differentiable Proving" has url `MISSING`; 2019 "On the Measure of Intelligence / ARC (Abstraction and Reasoning Corpus)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text and spaces.
- Examples: 2014 "Neural Turing Machines" has `"https://arxiv.org/pdf/1410.5401.pdf ""Neural Turing Machines (2014) - arXiv"""`; 2020 "OSCAR: Object-Semantics Aligned Pre-training for Vision-Language Tasks" has `"https://arxiv.org/pdf/2004.06165.pdf ""OSCAR (2020) - arXiv"""`; 2025 "Olympiad-level formal mathematical reasoning with large language models (AlphaProof)" has `"https://www.nature.com/articles/s41586-025-09833-y.pdf ""AlphaProof (2025) - Nature"""`; 2019 "On the Measure of Intelligence" has `"https://arxiv.org/pdf/1911.01547.pdf ""On the Measure of Intelligence (2019) - arXiv"""`.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains whitespace or extra quoted segments after a URL.

## Exact Duplicate Row
- Description: Identical year, title, and url rows appear more than once.
- Structural characteristics: duplicate CSV lines with matching values across all three columns.
- Examples: 2014 "Neural Turing Machines (NTM)" with `https://arxiv.org/abs/1410.5401` appears twice; 2024 "OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving" with `https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02024.pdf` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (including missing URLs).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2014 "Neural Turing Machines" appears with `https://arxiv.org/pdf/1410.5401.pdf` (plus label text) and `https://arxiv.org/abs/1410.5401` under the NTM title; 2024 "Omni-ARC" appears with `https://arcprize.org/blog/arc-prize-2024-winners-technical-report` and `https://arcprize.org/competitions/2024/`; 2019 "On the Measure of Intelligence" appears with `https://arxiv.org/abs/1911.01547`, `https://arxiv.org/pdf/1911.01547.pdf` (plus label text), and a MISSING variant in "On the Measure of Intelligence / ARC (Abstraction and Reasoning Corpus)".
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with distinct URLs (including placeholders).

## Title Variants or Aliases
- Description: Titles include aliases, parenthetical variants, or inconsistent wording that refer to the same work.
- Structural characteristics: near-identical titles differing only by parenthetical acronyms or alias phrases.
- Examples: 2014 "Neural Turing Machines" vs 2014 "Neural Turing Machines (NTM)"; 2019 "On the Measure of Intelligence" vs 2019 "On the Measure of Intelligence (ARC)".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after removing parentheticals and normalizing case.

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash, indicating multiple labels in one row.
- Structural characteristics: title includes ` / ` to join alternative names or descriptions.
- Examples: 2017 "Neural Theorem Provers / Differentiable Proving"; 2019 "On the Measure of Intelligence / ARC (Abstraction and Reasoning Corpus)".
- Why harmful: Complicates title matching and can imply multiple works or aliases in a single entry.
- Detection signals: title contains slash-separated segments or multiple alias phrases.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `arxiv.org/abs/`, publisher article pages, or PubMed records.
- Examples: 2014 "Neural Turing Machines (NTM)" uses `https://arxiv.org/abs/1410.5401`; 2019 "On the Measure of Intelligence" uses `https://arxiv.org/abs/1911.01547`; 1982 "Neural networks and physical systems with emergent collective computational abilities" uses `https://www.pnas.org/doi/10.1073/pnas.79.8.2554`; 1961 "New Results in Linear Filtering and Prediction Theory" uses `https://asmedigitalcollection.asme.org/fluidsengineering/article/83/1/95/426820/New-Results-in-Linear-Filtering-and-Prediction`; 2002 "Optimal feedback control as a theory of motor coordination" uses `https://pubmed.ncbi.nlm.nih.gov/12404008/`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract domains or path patterns.

## Non-paper Context URL
- Description: The url points to a context page (competition, blog, or repository) rather than the paper itself.
- Structural characteristics: domains or paths indicate competitions, blogs, or code repositories instead of paper artifacts.
- Examples: 2024 "OccSora" uses `https://github.com/wzzheng/OccSora`; 2024 "Omni-ARC" uses `https://arcprize.org/competitions/2024/` and `https://arcprize.org/blog/arc-prize-2024-winners-technical-report`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources (github.com, arcprize.org/competitions, /blog/).

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal host instead of an official publisher or DOI.
- Structural characteristics: domains indicate personal sites or third-party hosting (e.g., ResearchGate) rather than publisher domains.
- Examples: 1983 "Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems" uses `https://incompleteideas.net/papers/barto-sutton-anderson-83.pdf`; 1994 "On-line Q-learning Using Connectionist Systems" uses `https://www.researchgate.net/.../On-Line-Q-Learning-Using-Connectionist-Systems.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain matches personal or third-party host patterns (e.g., researchgate.net, author sites).

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types for the same corpus.
- Examples: 2014 "Neural Turing Machines" uses `https://arxiv.org/pdf/1410.5401.pdf` while 2014 "Neural Turing Machines (NTM)" uses `https://arxiv.org/abs/1410.5401`; 2025 "ONERULER: Benchmarking multilingual long-context language models" uses `https://arxiv.org/abs/2503.01996` while 2021 "Nystromformer: A Nystrom-Based Algorithm for Approximating Self-Attention" uses `https://arxiv.org/pdf/2102.03902`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Title-URL Mismatch
- Description: The url points to a different work or site than the one named in the title.
- Structural characteristics: URL slug or domain references an unrelated benchmark or project.
- Examples: 2019 "On the Measure of Intelligence (ARC)" links to `https://cavalab.org/srbench/`, which is unrelated to ARC.
- Why harmful: Mislinks entries, breaks citation integrity, and confuses automated retrieval.
- Detection signals: URL text/domain does not contain title keywords and clearly references a different work.

## Exact Duplicate Row
- Description: The same year, title, and url are repeated as identical rows in the chunk.
- Structural characteristics: duplicate CSV lines with identical values across all three columns.
- Examples: 2021 "PSB2: The Second Program Synthesis Benchmark Suite" with `https://arxiv.org/pdf/2106.06086` appears twice; 2021 "PonderNet: Learning to Ponder" with `https://arxiv.org/abs/2107.05407` appears twice; 2024 "Point Transformer V3: Simpler, Faster, Stronger" with `https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.pdf` appears twice.
- Why harmful: Inflates counts, distorts timelines, and forces downstream deduplication to do extra work.
- Detection signals: row-level hash or (year,title,url) key repeats within the file.

## Same Work Listed Under Multiple Years
- Description: The same work appears multiple times with different year values.
- Structural characteristics: identical or near-identical titles appear in multiple rows with differing years.
- Examples: 2020 and 2021 "Point Transformer" both link to `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf`; 2021 and 2022 "Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling" both link to `https://arxiv.org/abs/2111.14819`.
- Why harmful: Skews temporal analyses and can misplace papers in the timeline.
- Detection signals: normalized title matches across rows with different year values.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different urls.
- Structural characteristics: same or near-identical title refers to the same work but url differs across rows.
- Examples: 2020 "PCT: Point Cloud Transformer" uses `https://arxiv.org/abs/2012.09688` while 2021 "PCT: Point Cloud Transformer" uses `https://link.springer.com/article/10.1007/s41095-021-0229-5`; 1999 "Predictive coding in the visual cortex" uses `https://pmc.ncbi.nlm.nih.gov/articles/PMC4311762/` while 1999 "Predictive coding in the visual cortex: a functional interpretation" uses `https://pubmed.ncbi.nlm.nih.gov/10195184/`.
- Why harmful: Creates duplicate entries, fragments citations, and complicates canonicalization.
- Detection signals: normalized title+year pairs map to multiple distinct url values.

## Title Variants or Aliases
- Description: The same work is listed with shortened or expanded title variants.
- Structural characteristics: titles differ only by added subtitles or omitted phrases.
- Examples: 1999 "Predictive coding in the visual cortex" vs 1999 "Predictive coding in the visual cortex: a functional interpretation".
- Why harmful: Exact-title deduplication fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping subtitles and punctuation.

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash.
- Structural characteristics: title contains ` / ` or multiple alias phrases in one cell.
- Examples: 2023 "POPE (Polling-based Object Probing Evaluation) / Evaluating Object Hallucination in Large Vision-Language Models"; "Year unknown" "PSB / PSB2"; 2021 "Perceiver / Perceiver IO".
- Why harmful: Conflates aliases or multiple works into a single entry and complicates matching.
- Detection signals: title contains slash-separated segments or multiple alias phrases.

## Non-numeric or Placeholder Year Value
- Description: The year column contains placeholder text instead of a four-digit year.
- Structural characteristics: year value includes words like "Year unknown".
- Examples: "Year unknown" "PSB / PSB2" uses year `Year unknown`.
- Why harmful: Breaks sorting, filtering, and numeric validation.
- Detection signals: year fails regex `^[0-9]{4}$` or contains alphabetic characters.

## Year Placeholder Despite URL Year Signal
- Description: The year field is a placeholder even though the url encodes an explicit year that could be inferred.
- Structural characteristics: non-numeric year value alongside a url containing a 4-digit year token in its path or filename.
- Examples: "Year unknown" "PSB / PSB2" links to `https://www.cs.hamilton.edu/~thelmuth/Pubs/2015-GECCO-benchmark-suite.pdf`, which embeds `2015`.
- Why harmful: Leaves fixable gaps in the timeline and makes automated normalization harder.
- Detection signals: year fails numeric regex while url contains a clear 19xx/20xx token.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by additional quoted text or labels.
- Examples: 2023 "PaLM-E: An Embodied Multimodal Language Model" has `"https://arxiv.org/pdf/2303.03378.pdf ""PaLM-E (2023) — arXiv"""` in the url field.
- Why harmful: URL parsing fails and deduplication by URL becomes unreliable.
- Detection signals: url field contains whitespace or extra quoted segments after a URL.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `/abs/`, `/article/abs/`, or publisher article pages.
- Examples: 2020 "PCT: Point Cloud Transformer" uses `https://arxiv.org/abs/2012.09688`; 2024 "POS-BERT: Point cloud one-stage BERT pre-training" uses `https://www.sciencedirect.com/science/article/abs/pii/S0957417423030658`; 2025 "PV-DT3D: Point-voxel dual transformer for LiDAR 3D object detection" uses `https://link.springer.com/article/10.1007/s11801-025-3134-9`; 2024 "PointRegion: Transformer based 3D tooth segmentation via point cloud processing" uses `https://pubmed.ncbi.nlm.nih.gov/39557955/`.
- Why harmful: Automated retrieval may fail or require scraping and can be blocked by paywalls.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types for the same corpus.
- Examples: 2020 "PCT: Point Cloud Transformer" uses `https://arxiv.org/abs/2012.09688` while 2021 "PSB2: The Second Program Synthesis Benchmark Suite" uses `https://arxiv.org/pdf/2106.06086`; 2023 "PaLM-E: An Embodied Multimodal Language Model" uses `https://arxiv.org/pdf/2303.03378.pdf` while 2023 "POPE (Polling-based Object Probing Evaluation) / Evaluating Object Hallucination in Large Vision-Language Models" uses `https://arxiv.org/abs/2305.10355`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-canonical Mirror Source
- Description: The url points to a personal or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains indicate personal sites or university hosts (e.g., `~user` paths).
- Examples: "Year unknown" "PSB / PSB2" links to `https://www.cs.hamilton.edu/~thelmuth/Pubs/2015-GECCO-benchmark-suite.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: domain/path matches institutional or personal-host patterns such as `~` in the path.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the url.
- Structural characteristics: url includes a year token that does not match the row year.
- Examples: 2020 "Point Transformer" links to `https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf` (url suggests 2021); 2022 "Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling" uses `https://arxiv.org/abs/2111.14819` (arXiv id suggests 2021).
- Why harmful: Misorders entries in time-based analyses and confuses chronology.
- Detection signals: parse year-like tokens in URLs (conference year, DOI, arXiv IDs) and compare to the year column.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text page instead of a canonical PDF or DOI.
- Structural characteristics: URL points to a full-text HTML article page rather than a PDF.
- Examples: 1999 "Predictive coding in the visual cortex" uses `https://pmc.ncbi.nlm.nih.gov/articles/PMC4311762/`.
- Why harmful: HTML renders can change, may be less stable, and complicate automated download pipelines.
- Detection signals: url points to known HTML full-text hosts (e.g., `pmc.ncbi.nlm.nih.gov/articles/`).

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2015 "Prioritized Experience Replay" with `https://arxiv.org/abs/1511.05952` appears twice; 2017 "Proximal Policy Optimization (PPO)" with `https://arxiv.org/abs/1707.06347` appears twice; 2021 "Pyramid Vision Transformer (PVT)" with `https://arxiv.org/abs/2102.12122` appears twice; 2022 "ReAct: Synergizing Reasoning and Acting in Language Models" with `https://arxiv.org/abs/2210.03629` appears three times.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column contains a non-URL sentinel like `MISSING`.
- Examples: 2023 "Process Reward Models (PRM) introduced/standardized for reasoning traces" has url `MISSING`; 2023 "QLoRA" has url `MISSING`; 2020 "REALM: Retrieval-Augmented Language Model Pre-Training" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year repeated with different URLs (including placeholders or annotated URLs).
- Examples: 2025 "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective" appears with `https://arxiv.org/abs/2505.07859` and `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2021 "Pyramid Vision Transformer (PVT)" appears with `https://arxiv.org/abs/2102.12122` and the PDF URL with appended labels; 2020 "REALM: Retrieval-Augmented Language Model Pre-Training" appears with `https://arxiv.org/abs/2002.08909` and `MISSING`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with distinct URLs (including placeholders or annotated variants).

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text instead of a single clean URL value.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or labels.
- Examples: 2021 "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" has `"https://arxiv.org/pdf/2102.12122.pdf ""Pyramid Vision Transformer (2021) — arXiv"""`; another row has `"https://arxiv.org/pdf/2102.12122.pdf ""PVT (2021) — arXiv"""`.
- Why harmful: URL parsing fails, and deduplication by URL becomes unreliable.
- Detection signals: url field contains whitespace or additional quoted segments after a URL.

## Title Variants or Aliases
- Description: Titles include shortened or expanded variants that refer to the same work.
- Structural characteristics: near-identical titles differing by omitted subtitles or appended descriptors.
- Examples: 2021 "Pyramid Vision Transformer (PVT)" vs 2021 "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping subtitles or alias phrases.

## Title Contains Commentary or Provenance Note
- Description: The title field includes explanatory notes or provenance text that is not part of the official title.
- Structural characteristics: titles include phrases like "introduced/standardized" or "from ... writeups" that indicate commentary.
- Examples: 2023 "Process Reward Models (PRM) introduced/standardized for reasoning traces"; 2025 `Productive "refinement loop" systems from ARC Prize writeups (2025 winners list)`.
- Why harmful: Causes title-based matching to fail and obscures the canonical work name.
- Detection signals: title contains explanatory verbs or provenance phrases that are not part of a formal paper title.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types for the same corpus.
- Examples: 2015 "Prioritized Experience Replay" uses `https://arxiv.org/abs/1511.05952` while 2021 "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions" uses `https://arxiv.org/pdf/2102.12122.pdf` (with appended label text).
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/` or forum/landing patterns.
- Examples: 2017 "Proximal Policy Optimization (PPO)" uses `https://arxiv.org/abs/1707.06347`; 2022 "ReAct: Synergizing Reasoning and Acting in Language Models" uses `https://arxiv.org/abs/2210.03629`; 2019 "Recurrent Experience Replay in Distributed Reinforcement Learning (R2D2)" uses `https://openreview.net/forum?id=r1lyTjAqYX`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns or domains.

## Non-paper Context URL
- Description: The url points to a competition, project, or blog context page instead of a paper.
- Structural characteristics: domain/path indicates competitions, project home pages, or results writeups.
- Examples: 2025 "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2025 `Productive "refinement loop" systems from ARC Prize writeups (2025 winners list)` uses the same ARC Prize blog URL.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: url matches competition/blog/project site patterns and lacks a paper artifact.

## Non-canonical Mirror Source
- Description: The url points to a personal or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains indicate personal sites or university hosts (often with `~` in the path).
- Examples: 1962 "Receptive Fields, Binocular Interaction, and Functional Architecture in the Cat's Visual Cortex" uses `https://www.gatsby.ucl.ac.uk/~lmatthey/teaching/tn1/additional/systems/JPhysiol-1962-Hubel-106-54.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: domain/path matches institutional or personal-host patterns such as `~` in the path.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the url.
- Structural characteristics: arXiv-style IDs encode a different year than the row year.
- Examples: 2018 "Rainbow: Combining Improvements in Deep Reinforcement Learning" uses `https://arxiv.org/abs/1710.02298` (implies 2017); 2024 "RETRO-style retrieval-augmented pretraining and variants" uses `https://arxiv.org/abs/2112.04426` (implies 2021).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (conference year, DOI, arXiv IDs) and compare to the year column.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text page instead of a canonical PDF or DOI.
- Structural characteristics: URL points to a full-text HTML article page rather than a PDF.
- Examples: 1959 "Receptive fields of single neurones in the cat's striate cortex" uses `https://pmc.ncbi.nlm.nih.gov/articles/PMC1363130/`.
- Why harmful: HTML renders can change, may be less stable, and complicate automated download pipelines.
- Detection signals: url points to known HTML full-text hosts (e.g., `pmc.ncbi.nlm.nih.gov/articles/`).

## Aggregated or Conceptual Entry (Not a Single Work)
- Description: The title describes a concept, collection, or family of systems rather than a single paper, model, or benchmark.
- Structural characteristics: titles use plural nouns or phrases like "variants", "systems", "winners list", or "introduced/standardized" without naming a specific work.
- Examples: 2025 `Productive "refinement loop" systems from ARC Prize writeups (2025 winners list)`; 2024 "RETRO-style retrieval-augmented pretraining and variants"; 2023 "Process Reward Models (PRM) introduced/standardized for reasoning traces".
- Why harmful: Conflates multiple works or conceptual milestones into one row, making deduplication and citation tracking ambiguous.
- Detection signals: title contains pluralized nouns or "and variants"/"winners list"/"introduced" phrasing without a canonical paper title.

## URL Field Contains Extra Text
- Description: The url cell contains a URL plus appended labels or notes in the same field.
- Structural characteristics: quoted url field includes a URL followed by extra text or a second quoted fragment.
- Examples: 2021 "ALBEF: Align Before Fuse" has `https://arxiv.org/pdf/2107.07651.pdf` plus label text; 2016 "Adaptive Computation Time" has `https://arxiv.org/pdf/1603.08983.pdf` plus appended label text; 2025 "Adaptive Patch Selection for ViTs via Reinforcement Learning" has `https://doi.org/10.1007/s10489-025-06516-z` plus a trailing label.
- Why harmful: URL parsing fails or yields invalid links, and annotations fragment deduplication.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Placeholder URL Value
- Description: The url field uses a placeholder token instead of a real URL.
- Structural characteristics: url column equals `MISSING` or another sentinel.
- Examples: "Year unknown" "ARC" uses `MISSING`; 2024 "ARC Prize 2024: Technical Report" has `MISSING`.
- Why harmful: Prevents automated linking and hides missing data gaps.
- Detection signals: url fails URL regex and matches known placeholder tokens.

## Non-numeric or Placeholder Year Value
- Description: The year column contains non-numeric placeholder text instead of a four-digit year.
- Structural characteristics: year value includes words like "Year unknown".
- Examples: "Year unknown" "ARC" uses year `Year unknown`.
- Why harmful: Breaks sorting and numeric parsing.
- Detection signals: year fails `^[0-9]{4}$` or contains alphabetic characters.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF.
- Structural characteristics: URLs lack `.pdf` and use `/abs/` or publisher landing pages.
- Examples: 2025 "ARC Is a Vision Problem! (Vision ARC / VARC)" uses `https://arxiv.org/abs/2511.14761`; 2014 "Adam: A Method for Stochastic Optimization" uses `https://arxiv.org/abs/1412.6980`; 2024 "ASGFormer: Point cloud semantic segmentation with adaptive spatial graph transformer" uses `https://www.sciencedirect.com/science/article/pii/S156984322400459X`; 1976 "Adaptive pattern classification and universal recoding: II. Feedback, expectation, olfaction, illusions" uses `https://pubmed.ncbi.nlm.nih.gov/963125/`.
- Why harmful: Retrieval may require scraping or paywalled access.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns.

## Non-paper Context URL
- Description: The url points to a blog, competition, or context page instead of a paper.
- Structural characteristics: domain/path indicates blog results or competition writeups.
- Examples: 2025 "ARC Is a Vision Problem! + test-time training theme (indexed by ARC Prize 2025)" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2025 "ARC-AGI Without Pretraining (CompressARC)" appears with the same ARC Prize blog URL.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: url matches blog/competition patterns and lacks a paper artifact.

## Non-canonical Mirror Source
- Description: The url points to a personal or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains include university or personal sites, often with `~` in the path.
- Examples: 2018 "Accelerating Search-Based Program Synthesis using Learned Probabilistic Models (Euphony)" uses `https://www.cis.upenn.edu/~alur/PLDI18.pdf`; 2024 "ARC-Heavy / ARC-Potpourri (dataset description embedded in Cornell report)" uses `https://www.cs.cornell.edu/~ellisk/documents/arc_induction_vs_transduction.pdf`; 1913 "An Example of Statistical Investigation of the Text \"Eugene Onegin\" Concerning the Connection of Samples in Chains" uses `https://nessie.ilab.sztaki.hu/~kornai/2021/KalmanCL/markov_1913.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: domain/path matches institutional or personal-host patterns such as `~`.

## Exact Duplicate Row
- Description: Identical year, title, and url rows appear more than once.
- Structural characteristics: duplicate CSV lines with the same three fields.
- Examples: 2016 "Adaptive Computation Time (ACT)" with `https://arxiv.org/abs/1603.08983` appears twice.
- Why harmful: Inflates counts and biases analyses.
- Detection signals: row-level hash or (year,title,url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: One work appears across rows with different url values (including placeholders).
- Structural characteristics: same title/year normalized appears with multiple distinct urls.
- Examples: 2014 "Adam: A Method for Stochastic Optimization" appears with `https://arxiv.org/abs/1412.6980` and `https://arxiv.org/pdf/1412.6980`; 2024 "ARC Prize 2024: Technical Report" appears with `https://arxiv.org/abs/2412.04604` and `MISSING`; 2025 "ARC-AGI Without Pretraining (CompressARC)" appears with a GitHub Pages PDF and the ARC Prize blog URL.
- Why harmful: Splits citations and complicates canonicalization.
- Detection signals: normalize title/year and flag multiple distinct url strings.

## Title Variants or Aliases
- Description: Titles include shortened or expanded variants that refer to the same work.
- Structural characteristics: near-identical titles differing by added aliases or abbreviations.
- Examples: 2016 "Adaptive Computation Time" vs 2016 "Adaptive Computation Time (ACT)".
- Why harmful: Exact-title dedupe fails and search results fragment.
- Detection signals: fuzzy title matching after stripping parentheticals or abbreviations.

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash.
- Structural characteristics: title includes ` / ` joining alternative names.
- Examples: 2025 "ARC Is a Vision Problem! (Vision ARC / VARC)"; 2024 "ARC-Heavy / ARC-Potpourri (dataset description embedded in Cornell report)".
- Why harmful: Complicates title matching and can imply multiple works in one row.
- Detection signals: title contains slash-separated segments.

## Title Contains Commentary or Provenance Note
- Description: The title field includes explanatory or provenance text not part of the formal title.
- Structural characteristics: titles include notes like "indexed by" or "embedded in report".
- Examples: 2025 "ARC Is a Vision Problem! + test-time training theme (indexed by ARC Prize 2025)"; 2024 "ARC-Heavy / ARC-Potpourri (dataset description embedded in Cornell report)".
- Why harmful: Obscures canonical titles and breaks title-based matching.
- Detection signals: title contains explanatory phrases like "indexed by", "embedded in", or "report".

## Title-URL Mismatch
- Description: The url points to a different artifact or generic page that does not match the titled work.
- Structural characteristics: URL slug lacks title keywords and points to a generic results page.
- Examples: 2025 "ARC-AGI Without Pretraining (CompressARC)" links to `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2025 "ARC-AGI is a Vision Problem!" links to the same results analysis page.
- Why harmful: Mislinks entries and confuses automated retrieval.
- Detection signals: URL text does not contain title keywords and points to unrelated content.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/html/` and serves rendered HTML.
- Examples: 2025 "ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus" uses `https://arxiv.org/html/2505.08778v1`.
- Why harmful: HTML renders can change and complicate automated downloads.
- Detection signals: url contains `/html/` or ends with `.html`.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version suffix (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` plus digits.
- Examples: 2025 "ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus" uses `https://arxiv.org/html/2505.08778v1`.
- Why harmful: Creates duplicates across versions and becomes stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types across rows.
- Examples: 2021 "ALBEF: Align Before Fuse" uses `https://arxiv.org/pdf/2107.07651.pdf`; 2025 "ARC Is a Vision Problem! (Vision ARC / VARC)" uses `https://arxiv.org/abs/2511.14761`; 2025 "ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus" uses `https://arxiv.org/html/2505.08778v1`.
- Why harmful: URL-based deduplication fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types in the same chunk.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with year tokens embedded in the URL path.
- Structural characteristics: URL path includes a different year-like token.
- Examples: 1913 "An Example of Statistical Investigation of the Text \"Eugene Onegin\" Concerning the Connection of Samples in Chains" links to `https://nessie.ilab.sztaki.hu/~kornai/2021/KalmanCL/markov_1913.pdf` (contains `2021`).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URL paths and compare to the year column.

## Aggregated or Conceptual Entry (Not a Single Work)
- Description: The title describes a theme or concept rather than a single paper.
- Structural characteristics: titles mention "theme" or competition indexing without a specific paper title.
- Examples: 2025 "ARC Is a Vision Problem! + test-time training theme (indexed by ARC Prize 2025)".
- Why harmful: Conflates multiple works and makes canonicalization ambiguous.
- Detection signals: title uses conceptual phrasing like "theme" or "indexed by" without a formal paper title.

## URL Field Contains Extra Text
- Description: The url cell contains a URL plus appended label text rather than a single clean URL.
- Structural characteristics: quoted url field includes a URL followed by extra words, labels, or quoted fragments.
- Examples: 2025 "Rotary Masked Autoencoders Are Versatile Learners" has `https://arxiv.org/pdf/2505.20535.pdf` followed by appended label text; 2024 "Rotary Position Embedding for Vision Transformer (RoPE-Mixed / 2D RoPE study)" has `https://arxiv.org/pdf/2403.13298.pdf` plus a label; 2024 "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" has `https://arxiv.org/pdf/2408.03314.pdf` plus a label; 2021 "Scaling Up Vision-Language Learning With Noisy Text Supervision (ALIGN)" has `https://arxiv.org/pdf/2102.05918.pdf` plus a label.
- Why harmful: URL parsing fails or yields invalid links; duplicated entries proliferate due to annotation variants.
- Detection signals: url field contains whitespace or quoted label text after a URL.

## Same Work Listed Multiple Times (Different URLs)
- Description: The same work appears in multiple rows with different url values (conference PDF vs arXiv, or different host pages).
- Structural characteristics: same or near-identical title and year across rows, but url differs.
- Examples: 2024 "Rotary Position Embedding for Vision Transformer" appears with `https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01584.pdf` and `https://arxiv.org/abs/2403.13298`; 2019 "SATNet: Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver" appears with `https://arxiv.org/pdf/1905.12149` and `https://arxiv.org/abs/1905.12149`; 2024 "SRBench++: principled benchmarking of symbolic regression" appears with `https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/` and `https://pubmed.ncbi.nlm.nih.gov/40761553/`; 2023/2024 "SWE-bench" appears with `https://arxiv.org/abs/2310.06770` and `https://proceedings.iclr.cc/paper_files/paper/2024/file/edac78c3e300629acfe6cbe9ca88fb84-Paper-Conference.pdf`.
- Why harmful: Creates duplicates, fragments citations, and complicates canonicalization.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include aliases, parentheticals, or phrasing variants that refer to the same work.
- Structural characteristics: titles differ by added parenthetical text, shortened subtitles, or minor wording changes.
- Examples: 2024 "Rotary Position Embedding for Vision Transformer" vs 2024 "Rotary Position Embedding for Vision Transformer (RoPE-Mixed / 2D RoPE study)"; 2019 "SATNet: Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver" vs 2019 "SATNet: Differentiable Satisfiability Solver"; 2020 "SETR: Rethinking Semantic Segmentation from a Sequence-to-Sequence Perspective with Transformers" vs 2021 "SETR: Rethinking Semantic Segmentation as Seq2Seq with Transformers"; 2024 "SWE-bench" vs 2023 "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?".
- Why harmful: Exact-title deduplication fails and search results fragment.
- Detection signals: fuzzy-title matching after removing parentheticals or normalizing abbreviations.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: urls lack `.pdf` and use known landing patterns (arXiv `/abs/`, conference abstract pages, PubMed entries).
- Examples: 2020 "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" uses `https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html`; 2021 "SVAMP: Simple Variations on Arithmetic Math Word Problems" uses `https://aclanthology.org/2021.naacl-main.168/`; 2023 "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" uses `https://arxiv.org/abs/2310.06770`; 2024 "SRBench++: principled benchmarking of symbolic regression" uses `https://pubmed.ncbi.nlm.nih.gov/40761553/`; 2025 "Scaling up Test-Time Compute with Latent Reasoning" uses `https://neurips.cc/virtual/2025/poster/117966`.
- Why harmful: Automated retrieval requires scraping or fails; access can be unstable or paywalled.
- Detection signals: url does not end with `.pdf` and matches known landing-page domains or patterns.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view instead of a canonical PDF/DOI.
- Structural characteristics: URL contains `/html/` and serves rendered HTML.
- Examples: 2025 "SRBench update (\"next generation\" SRBench)" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: HTML renders can change and are less stable for automated retrieval.
- Detection signals: url contains `/html/`.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version suffix rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` plus digits.
- Examples: 2025 "SRBench update (\"next generation\" SRBench)" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: Creates duplicates across versions and becomes stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types across rows.
- Examples: 2024 "Rotary Position Embedding for Vision Transformer" uses `https://arxiv.org/abs/2403.13298`; 2019 "SATNet: Bridging Deep Learning and Logical Reasoning Using a Differentiable Satisfiability Solver" uses `https://arxiv.org/pdf/1905.12149`; 2025 "SRBench update (\"next generation\" SRBench)" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: URL-based deduplication fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types in the same chunk.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2020 "SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks" with `https://proceedings.neurips.cc/paper/2020/hash/15231a7ce4ba789d13b722cc5c955834-Abstract.html` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column contains a non-URL sentinel like `MISSING`.
- Examples: 2024 "ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (MISSING, TBD, N/A).

## Non-paper Context URL
- Description: The url points to a code repository or project page rather than the paper itself.
- Structural characteristics: domains indicate GitHub or project sites without paper identifiers.
- Examples: 2022 "SRBench" links to `https://github.com/cavalab/srbench`; 2021 "SRBench (Symbolic Regression Benchmarks)" links to `https://cavalab.github.io/symbolic-regression/`.
- Why harmful: Misrepresents the source and prevents retrieval of the actual paper.
- Detection signals: domain is a repo or project site; URL lacks PDF or DOI-like patterns.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with year tokens embedded in the URL.
- Structural characteristics: URL includes a different year-like token than the row year.
- Examples: 2021 "SETR: Rethinking Semantic Segmentation as Seq2Seq with Transformers" links to `https://arxiv.org/abs/2012.15840` (arXiv ID implies 2020).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (e.g., arXiv-style IDs) and compare with the year column.

## Same Work Listed Multiple Times (Different Years)
- Description: A single work appears in multiple rows with different year values.
- Structural characteristics: same or near-identical title or identical url appears across different years.
- Examples: 2020 and 2021 "SETR" entries share `https://arxiv.org/abs/2012.15840`; 2023 "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?" vs 2024 "SWE-bench" likely refer to the same work with different years; 2021 "SRBench (Symbolic Regression Benchmarks)" vs 2022 "SRBench" suggests the same benchmark listed under different years.
- Why harmful: Distorts timeline ordering and duplicates entries during aggregation.
- Detection signals: match normalized titles or identical urls across rows with different years.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column equals `MISSING` or another non-URL sentinel.
- Examples: 2024 "SegPoint: Segment Any Point Cloud via Large Language" has url `MISSING`; 2022 "Self-Consistency for Chain-of-Thought" has url `MISSING`; 2018 "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" has url `MISSING`; 2020 "Sharpness-Aware Minimization (SAM)" has url `MISSING`.
- Why harmful: Prevents automated linking and hides missing data gaps.
- Detection signals: url fails URL regex and matches known placeholder tokens.

## URL Field Contains Extra Text
- Description: The url field contains a URL plus appended label or notes instead of a single clean URL.
- Structural characteristics: quoted url field includes a URL followed by extra label text and spaces.
- Examples: 2022 "ScienceQA: Benchmark for Multimodal Reasoning" has `https://arxiv.org/pdf/2209.09513.pdf` plus label text; 2024 "Searching Latent Program Spaces" has `https://arxiv.org/pdf/2411.08706.pdf` plus label text; 2021 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" has `https://arxiv.org/pdf/2105.15203.pdf` plus label text; 2024 "Simultaneous Instance Pooling & Bag Selection for MIL using ViTs" has `https://doi.org/10.1007/s00521-024-09417-3` plus label text.
- Why harmful: URL parsing fails or yields invalid links; annotations fragment deduplication.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2021 "SegFormer" with `https://arxiv.org/abs/2105.15203` appears twice; 2021 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" with `https://arxiv.org/pdf/2105.15203.pdf` plus label text appears twice; 2023 "Segment Anything" with `https://arxiv.org/pdf/2304.02643.pdf` plus label text appears twice; 2020 "Shortformer: Better Language Modeling using Shorter Inputs" with `https://arxiv.org/pdf/2012.15832` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (arXiv vs conference, placeholder vs real).
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2024 "Searching Latent Program Spaces" appears with `https://arcprize.org/competitions/2024/` and `https://arxiv.org/pdf/2411.08706.pdf` (plus label text); 2023 "Segment Anything" appears with `https://arxiv.org/pdf/2304.02643.pdf` (plus label text) and `https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf`; 2018 "Self-Attention with Relative Position Representations" appears with `https://arxiv.org/abs/1803.02155` and `https://arxiv.org/pdf/1803.02155`; 2018 "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" appears with `https://arxiv.org/abs/1810.00825` and `MISSING`; 2025 "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR)" appears with `https://openreview.net/pdf?id=z4IG090qt2` and `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Creates duplicates, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url strings.

## Title Variants or Aliases
- Description: Titles include shortened, expanded, or parenthetical variants that refer to the same work.
- Structural characteristics: near-identical titles differing by added subtitles or acronyms.
- Examples: 2022 "ScienceQA (\"Learn to Explain...\")" vs 2022 "ScienceQA: Benchmark for Multimodal Reasoning"; 2021 "SegFormer" vs 2021 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"; 2023 "Segment Anything" vs 2023 "Segment Anything (SAM)"; 2022 "Self-Consistency Improves Chain-of-Thought Reasoning" vs 2022 "Self-Consistency for Chain-of-Thought".
- Why harmful: Exact-title dedupe fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping parentheticals/subtitles and normalizing case.

## Non-paper Context URL
- Description: The url points to a competition or blog context page instead of a paper.
- Structural characteristics: domain/path indicates competitions or results writeups without paper artifacts.
- Examples: 2024 "Searching Latent Program Spaces" uses `https://arcprize.org/competitions/2024/`; 2025 "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR)" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: URL matches competition/blog patterns and lacks a PDF/DOI artifact.

## Landing or Abstract Page URL
- Description: The url points to a landing/abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use patterns like `/abs/`, publisher article pages, or PubMed/DOI resolvers.
- Examples: 2022 "ScienceQA (\"Learn to Explain...\")" uses `https://arxiv.org/abs/2209.09513`; 2021 "SegFormer" uses `https://arxiv.org/abs/2105.15203`; 2023 "Self-Refine: Iterative Refinement with Self-Feedback" uses `https://arxiv.org/abs/2303.17651`; 1992 "Separate visual pathways for perception and action" uses `https://pubmed.ncbi.nlm.nih.gov/1374953/`; 1992 "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (REINFORCE)" uses `https://link.springer.com/article/10.1007/BF00992696`.
- Why harmful: Automated retrieval may fail or require scraping or paywalled access.
- Detection signals: url does not end with `.pdf` and matches known landing-page patterns or domains.

## Non-canonical Mirror Source
- Description: The url points to a personal or institutional host instead of an official publisher or DOI.
- Structural characteristics: domain/path indicates a university personal page (often with `~`).
- Examples: 2018 "Search-based Program Synthesis" uses `https://www.cis.upenn.edu/~alur/CACM18.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: URL host is a personal/university page with `~` or similar patterns.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: arxiv.org URLs use different path types across rows.
- Examples: 2021 "SegFormer" uses `https://arxiv.org/abs/2105.15203` while 2021 "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" uses `https://arxiv.org/pdf/2105.15203.pdf`; 2018 "Self-Attention with Relative Position Representations" appears with both `https://arxiv.org/abs/1803.02155` and `https://arxiv.org/pdf/1803.02155`.
- Why harmful: URL-based deduplication fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed `/abs/` vs `/pdf/` forms in the same chunk.

## Title-URL Mismatch
- Description: The url points to a generic page that does not correspond to the titled work.
- Structural characteristics: URL lacks title keywords and points to a competition/blog page.
- Examples: 2025 "Self-Improving Language Models for Evolutionary Program Synthesis: A Case Study on ARC-AGI (SOAR)" links to `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2024 "Searching Latent Program Spaces" links to `https://arcprize.org/competitions/2024/`.
- Why harmful: Mislinks entries and confuses automated retrieval.
- Detection signals: URL slug does not match title keywords and points to unrelated context pages.

## Truncated or Incomplete Title
- Description: The title appears cut off mid-phrase and lacks a complete name.
- Structural characteristics: title ends with a dangling modifier or missing noun (e.g., "via Large Language").
- Examples: 2024 "SegPoint: Segment Any Point Cloud via Large Language" (missing the final word after "Language").
- Why harmful: Breaks title-based matching and makes it unclear which work is referenced.
- Detection signals: title ends with prepositions/adjectives and lacks a complete noun phrase; compare against known paper titles to flag likely truncation.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or label fragments.
- Examples: 2025 "SmolVLM: Redefining small and efficient multimodal models" has `https://arxiv.org/pdf/2504.05299.pdf` plus label text like `SmolVLM (2025) - arXiv`; 2024 "Solving olympiad geometry without human demonstrations (AlphaGeometry)" has a Nature PDF plus label text; 2023 "Spherical Position Encoding for Transformers" has `https://arxiv.org/pdf/2310.04454.pdf` plus label text; 2021 "Swin Transformer" has `https://arxiv.org/pdf/2103.14030.pdf` plus label text; 2021 "Swin Transformer V2: Scaling Up Capacity and Resolution" has `https://arxiv.org/pdf/2111.09883.pdf` plus label text.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains whitespace or extra quoted segments after a URL.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use known landing patterns such as `/abs/`, publisher article pages, or `-Abstract.html`.
- Examples: 2020 "Solver-in-the-Loop: Learning from Differentiable Physics to Interact with Iterative PDE Solvers" uses `...-Abstract.html`; 2018 "Soft Actor-Critic (SAC)" uses `https://arxiv.org/abs/1812.05905`; 2021 "StrategyQA: A Benchmark with Implicit Reasoning Strategies" uses `https://aclanthology.org/2021.tacl-1.21/`; 1995 "Support-Vector Networks" uses `https://link.springer.com/article/10.1007/BF00994018`; 2019 "SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems" uses `https://arxiv.org/abs/1905.00537`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns or domains.

## Exact Duplicate Row
- Description: Identical year, title, and url rows appear more than once.
- Structural characteristics: duplicate CSV lines with matching values across all three columns.
- Examples: 2022 "Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion" appears twice with `https://arxiv.org/abs/2211.10581`; 2025 "SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection" appears twice with `https://arxiv.org/html/2503.08092v1`; 2021 "Swin Transformer V2: Scaling Up Capacity and Resolution" appears twice with the same annotated arXiv PDF URL.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view instead of a canonical PDF or DOI.
- Structural characteristics: url contains `/html/` endpoints (often on arXiv).
- Examples: 2025 "SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection" uses `https://arxiv.org/html/2503.08092v1`; 2025 "Streaming 4D Panoptic Segmentation via Dual Threads" uses `https://arxiv.org/html/2510.17664v1`.
- Why harmful: HTML renders can change, require scripts, or be less stable than PDFs.
- Detection signals: url contains `/html/` or similar full-text render endpoints.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection" uses `https://arxiv.org/html/2503.08092v1`; 2025 "Streaming 4D Panoptic Segmentation via Dual Threads" uses `https://arxiv.org/html/2510.17664v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` forms within the same chunk.
- Structural characteristics: same arxiv.org domain with different path types across rows (including `/pdf/` with and without `.pdf`).
- Examples: 2018 "Soft Actor-Critic (SAC)" uses `https://arxiv.org/abs/1812.05905`; 2025 "SmolVLM: Redefining small and efficient multimodal models" uses `https://arxiv.org/pdf/2504.05299.pdf`; 2025 "SparseVoxFormer: Sparse Voxel-based Transformer for Multi-modal 3D Object Detection" uses `https://arxiv.org/html/2503.08092v1`; 2021 "Swin Transformer" appears with both `https://arxiv.org/abs/2103.14030` and `https://arxiv.org/pdf/2103.14030`.
- Why harmful: URL-based deduplication fails and retrieval logic must handle multiple variants.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column equals a sentinel like `MISSING`.
- Examples: 2015 "Stack/Queue/Deque-augmented RNNs" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Non-canonical Mirror Source
- Description: The url points to a mirror or institutional host rather than an official publisher or DOI.
- Structural characteristics: domain indicates a university or lab host instead of a publisher.
- Examples: 2020 "Stand-Alone Axial-Attention for Panoptic Segmentation (Axial-DeepLab)" uses `https://www.cs.jhu.edu/~alanlab/Pubs20/wang2020axial.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain matches institutional host patterns or contains `~` or lab subpaths.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values (including arXiv vs proceedings or mirror links).
- Structural characteristics: same or near-identical title/year repeated with different URLs.
- Examples: 2018 "Soft Actor-Critic (SAC)" appears with `https://arxiv.org/abs/1812.05905` and `https://arxiv.org/abs/1801.01290`; 2020 "Stand-Alone Axial-Attention (Axial-DeepLab)" appears with `https://arxiv.org/abs/2105.15203` and a JHU-hosted PDF; 2021 "Swin Transformer" appears with arXiv abs/pdf URLs and a CVF ICCV PDF.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with distinct URLs.

## Title Variants or Aliases
- Description: Titles include shortened or expanded variants that refer to the same work.
- Structural characteristics: near-identical titles differing by omitted subtitles or appended descriptors.
- Examples: 2020 "Stand-Alone Axial-Attention (Axial-DeepLab)" vs 2020 "Stand-Alone Axial-Attention for Panoptic Segmentation (Axial-DeepLab)"; 2021 "Swin Transformer" vs 2021 "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping subtitles or alias phrases.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or DOI.
- Structural characteristics: URL includes a different year token than the row year (e.g., arXiv ID or publisher path year).
- Examples: 2020 "Stand-Alone Axial-Attention (Axial-DeepLab)" uses `https://arxiv.org/abs/2105.15203` (implies 2021); 2024 "Solving olympiad geometry without human demonstrations (AlphaGeometry)" uses `https://www.nature.com/articles/s41586-023-06747-5.pdf` (implies 2023).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs and compare with the year column.

## Non-numeric or Placeholder Year Value
- Description: The year field contains a placeholder or non-numeric token instead of a year.
- Structural characteristics: year value includes words like "Year unknown".
- Examples: "Year unknown" "SyGuS PBE-BV / PBE-Strings" uses year `Year unknown`.
- Why harmful: Breaks sorting and makes timeline analyses ambiguous.
- Detection signals: year value fails numeric parsing or matches known placeholders.

## Non-paper Context URL
- Description: The url points to a competition, project, or context page rather than a paper artifact.
- Structural characteristics: domain/path indicates competition pages or results sites without paper PDFs.
- Examples: "Year unknown" "SyGuS PBE-BV / PBE-Strings" uses `https://sygus-org.github.io/comp/2019/`; 2019 "SyGuS-Comp \"PBE tracks\" mature (e.g., PBE-Strings, PBE-BV)" uses the same competition page.
- Why harmful: Misrepresents sources and prevents retrieval of an actual paper.
- Detection signals: url matches competition or event site patterns and lacks a paper artifact.

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash.
- Structural characteristics: title includes ` / ` to join alternative names or descriptions.
- Examples: "Year unknown" "SyGuS PBE-BV / PBE-Strings".
- Why harmful: Complicates title matching and can imply multiple works in a single entry.
- Detection signals: title contains slash-separated segments or multiple alias phrases.

## Title Contains Commentary or Provenance Note
- Description: The title field includes explanatory notes or commentary that is not part of the official title.
- Structural characteristics: parenthetical phrases like "e.g." or quoted fragments appended to the title.
- Examples: 2019 "SyGuS-Comp \"PBE tracks\" mature (e.g., PBE-Strings, PBE-BV)".
- Why harmful: Causes title-based matching to fail and obscures the canonical work name.
- Detection signals: title contains commentary terms like "e.g." or embedded quotes that indicate notes rather than a proper title.

## Same Work Listed Under Multiple Years
- Description: The same work appears in multiple rows with different year values.
- Structural characteristics: identical or near-identical titles point to the same URL but with different years.
- Examples: "Year unknown" "SyGuS PBE-BV / PBE-Strings" and 2019 "SyGuS-Comp \"PBE tracks\" mature (e.g., PBE-Strings, PBE-BV)" both link to `https://sygus-org.github.io/comp/2019/`.
- Why harmful: Skews timeline analyses and inflates counts across years.
- Detection signals: normalized title or shared URL matches across rows with differing year values.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2013 "Syntax-Guided Synthesis (SyGuS)" with `https://www.cis.upenn.edu/~alur/SyGuS13.pdf` appears twice; 2020 "Taming Transformers for High-Resolution Image Synthesis" with the annotated arXiv PDF url appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year across rows, but url differs (often mirror vs canonical, or abs vs pdf).
- Examples: 2016 "SyGuS-Comp 2016 Results/Benchmarks (PBE track)" at `https://arxiv.org/pdf/1611.07627` and 2016 "SyGuS-Comp 2016: Results and Analysis" at `https://dspace.mit.edu/.../1611.07627.pdf`; "Taming Transformers for High-Resolution Image Synthesis" uses `https://arxiv.org/pdf/2012.09841.pdf` while the 2021 variant uses `https://arxiv.org/abs/2012.09841`.
- Why harmful: Creates duplicates, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year or shared arXiv ID appears multiple times with distinct url values.

## Same Work Listed Under Multiple Years
- Description: The same work appears in multiple rows with different year values.
- Structural characteristics: identical or near-identical titles point to the same URL or arXiv ID but with different years.
- Examples: 2020 "Taming Transformers for High-Resolution Image Synthesis" and 2021 "Taming Transformers for High-Resolution Image Synthesis (VQGAN + Transformer)" both point to arXiv `2012.09841`.
- Why harmful: Skews timeline ordering and inflates counts across years.
- Detection signals: match normalized titles or shared canonical IDs across rows with different year values.

## Title Variants or Aliases
- Description: Titles include shortened, expanded, or parenthetical variants that refer to the same work.
- Structural characteristics: near-identical titles differing only by added descriptors or acronyms.
- Examples: "Taming Transformers for High-Resolution Image Synthesis" vs "Taming Transformers for High-Resolution Image Synthesis (VQGAN + Transformer)".
- Why harmful: Exact-title dedupe fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping parentheticals/subtitles and normalizing case.

## Compound Title with Slash-Separated Aliases
- Description: The title field combines multiple names or aliases using a slash.
- Structural characteristics: title includes ` / ` or tokens like `Results/Analysis`.
- Examples: 2016 "SyGuS-Comp 2016 Results/Benchmarks (PBE track)"; 2015 "SyGuS-Comp'15 Results/Analysis"; 2025 "Test-Time Learning for Large Language Models (TLM / TTL)".
- Why harmful: Complicates title matching and can imply multiple works in a single entry.
- Detection signals: title contains slash-separated segments or multiple alias phrases.

## Multiple Works Combined in One Entry
- Description: A single row title merges distinct works, benchmarks, or events into one entry.
- Structural characteristics: title uses `+` or comma-separated lists to combine multiple entities.
- Examples: 2014 "Syntax-Guided Synthesis (SyGuS) + SyGuS-Comp 2014"; "Year unknown" "Symbolic regression benchmarks (AI Feynman, SRBench)".
- Why harmful: Ambiguous whether one row represents one work or multiple; breaks dedupe and linkage.
- Detection signals: title contains `+` or multiple item lists in parentheses (comma-separated proper names).

## Non-numeric or Placeholder Year Value
- Description: The year field contains a placeholder or non-numeric token instead of a year.
- Structural characteristics: year value includes words like "Year unknown".
- Examples: "Year unknown" "Symbolic regression benchmarks (AI Feynman, SRBench)".
- Why harmful: Breaks sorting and makes timeline analyses ambiguous.
- Detection signals: year value fails numeric parsing or matches known placeholders.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column equals a sentinel like `MISSING` rather than an http(s) URL.
- Examples: 2019 "The Lottery Ticket Hypothesis" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra label text and spaces.
- Examples: 2020 "Taming Transformers for High-Resolution Image Synthesis" has `https://arxiv.org/pdf/2012.09841.pdf` followed by label text like `Taming Transformers (2020) - arXiv`.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Non-URL Annotation or Local Reference in URL Field
- Description: The url field contains a comment or local-copy note instead of a URL.
- Structural characteristics: url value starts with comment markers (e.g., `#`) or plain text rather than http(s).
- Examples: 2007 "The Hidden Logic of Sudoku (Second Edition)" has url `# "The Hidden Logic of Sudoku (2007) - Book (local copy)"`.
- Why harmful: Breaks URL validation and prevents automated retrieval.
- Detection signals: url does not match URL regex and contains comment tokens or "local copy" text.

## Non-paper Context URL
- Description: The url points to a competition, blog, or context page rather than a paper artifact.
- Structural characteristics: domain/path indicates competitions or results writeups without paper PDFs.
- Examples: 2017 "SyGuS-Comp track definitions (PBE-BV)" uses `https://sygus.org/comp/2017/`; 2024 "The LLM ARChitect: Solving ARC-AGI Is a Matter of Perspective" uses `https://arcprize.org/competitions/2024/`; 2025 "Test-time Adaptation of Tiny Recursive Models" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: URL matches competition/blog patterns and lacks a PDF/DOI artifact.

## Landing or Abstract Page URL
- Description: The url points to a landing, abstract, or DOI page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/`, publisher article pages, or forum landing pages.
- Examples: 2025 "TAPA: Positional Encoding via Token-Aware Phase Attention" uses `https://arxiv.org/abs/2509.12635`; 2022 "Tackling the Abstraction and Reasoning Corpus with Vision Transformers: the ViTARC Architecture" uses `https://openreview.net/forum?id=0gOQeSHNX1`; 1992 "Technical Note: Q-learning" uses `https://link.springer.com/article/10.1023/A%3A1022676722315`; 1956 "The Magical Number Seven, Plus or Minus Two" uses `https://pubmed.ncbi.nlm.nih.gov/13310704/`; 1977 "Synergetics: An Introduction" uses `https://books.google.com/books/about/Synergetics.html?id=KHn1CAAAQBAJ`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns or domains.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view instead of a canonical PDF or DOI.
- Structural characteristics: url contains `/html/` endpoints.
- Examples: 2025 "TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Evolution" uses `https://arxiv.org/html/2506.18421v1`; 2025 "The Art of Scaling Test-Time Compute for LLMs" uses `https://arxiv.org/html/2512.02008v1`.
- Why harmful: HTML renders can change, require scripts, or be less stable than PDFs.
- Detection signals: url contains `/html/` or similar full-text render endpoints.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Evolution" uses `https://arxiv.org/html/2506.18421v1`; 2024 "Teaching Transformers Modular Arithmetic at Scale" uses `https://www.arxiv.org/abs/2410.03569v1`; 2025 "The Art of Scaling Test-Time Compute for LLMs" uses `https://arxiv.org/html/2512.02008v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` forms within the same chunk.
- Structural characteristics: same arxiv.org domain with different path types across rows.
- Examples: 2016 "SyGuS-Comp 2016 Results/Benchmarks (PBE track)" uses `https://arxiv.org/pdf/1611.07627`; 2025 "TAPA: Positional Encoding via Token-Aware Phase Attention" uses `https://arxiv.org/abs/2509.12635`; 2025 "TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Evolution" uses `https://arxiv.org/html/2506.18421v1`.
- Why harmful: URL-based deduplication fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-canonical Mirror Source
- Description: The url points to a mirror or institutional host instead of an official publisher or DOI.
- Structural characteristics: domains indicate personal or institutional hosts rather than publishers.
- Examples: 2015 "SyGuS-Comp'15 Results/Analysis" uses `https://rishabhmit.bitbucket.io/papers/synt15.pdf`; 2013 "Syntax-Guided Synthesis (SyGuS)" uses `https://www.cis.upenn.edu/~alur/SyGuS13.pdf`; 1949 "The Organization of Behavior" uses `https://www.dengfanxin.cn/wp-content/uploads/2016/03/1949Hebb.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `~`, personal pages, or non-publisher hosts.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or repository metadata.
- Structural characteristics: URL includes a different year token than the row year (e.g., arXiv ID or conference year in path).
- Examples: 2019 "The Abstraction and Reasoning Corpus (ARC)" uses `https://arxiv.org/abs/2412.04604` (implies 2024); 2024 "Test-Time Training on Nearest Neighbors for Large Language Models" uses `https://arxiv.org/abs/2305.18466` (implies 2023); 1992 "TD-Gammon, a Self-Teaching Backgammon Program, Achieves Master-Level Play" uses `https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-003.pdf` (implies 1993).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (arXiv IDs, conference-year paths) and compare with the year column.

## Title-URL Mismatch
- Description: The url points to a page that does not correspond to the titled work.
- Structural characteristics: URL slug does not match title keywords or points to a generic competition/blog page.
- Examples: 2019 "The Abstraction and Reasoning Corpus (ARC)" links to `https://arxiv.org/abs/2412.04604`; 2025 "Test-time Adaptation of Tiny Recursive Models" links to `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2024 "The LLM ARChitect: Solving ARC-AGI Is a Matter of Perspective" links to `https://arcprize.org/competitions/2024/`.
- Why harmful: Mislinks entries and confuses automated retrieval.
- Detection signals: URL slug does not align with title keywords or points to unrelated context pages.

## Truncated or Incomplete Title
- Description: The title appears cut off mid-phrase and lacks a complete name.
- Structural characteristics: title ends with ellipses or a dangling modifier.
- Examples: 2025 "The Lessons of Developing Process Reward Models..." ends with an ellipsis rather than a complete title.
- Why harmful: Breaks title-based matching and makes it unclear which work is referenced.
- Detection signals: title ends with `...` or a dangling phrase; compare against known full titles to flag likely truncation.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column is a sentinel like `MISSING` rather than an http(s) URL.
- Examples: 1957 "The Perceptron" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url fails a URL regex and matches known placeholders (MISSING, TBD, N/A).

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2021 "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" with url `https://arxiv.org/pdf/2101.00027` appears twice; 2018 "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge" with url `https://arxiv.org/abs/1803.05457` appears twice; 2023 "Toolformer: Language Models Can Teach Themselves to Use Tools" with url `https://arxiv.org/abs/2302.04761` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values or url variants.
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2024 "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning" uses `https://arxiv.org/abs/2411.07279` and `https://ekinakyurek.github.io/papers/ttt.pdf "Test-Time Training for Few-Shot Learning (2024) — Project Page"`; 2024 "Towards Efficient Neurally-Guided Program Induction for ARC-AGI" appears with `https://arxiv.org/abs/2411.17708`, `https://arcprize.org/competitions/2024/`, and `https://arxiv.org/pdf/2411.17708.pdf "Neurally-Guided Program Induction for ARC-AGI (2024) — arXiv"`; 2021 "Train Short, Test Long: Attention with Linear Biases (ALiBi)" appears with `https://arxiv.org/abs/2108.12409` and `https://arxiv.org/pdf/2108.12409.pdf "ALiBi (2021) — arXiv"`; 2020 "Training data-efficient image transformers & distillation through attention" appears twice with different label text after the same base PDF.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text or notes.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or label fragments.
- Examples: 2025 "The Rotary Position Embedding May Cause Dimension Inefficiency" has `https://arxiv.org/pdf/2502.11276.pdf "The Rotary Position Embedding May Cause Dimension Inefficiency (2025) — arXiv"`; 2024 "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning" has `https://ekinakyurek.github.io/papers/ttt.pdf "Test-Time Training for Few-Shot Learning (2024) — Project Page"`; 2021 "Train Short, Test Long: Attention with Linear Biases (ALiBi)" has `https://arxiv.org/pdf/2108.12409.pdf "ALiBi (2021) — arXiv"`; 2020 "Training data-efficient image transformers & distillation through attention" has `https://arxiv.org/pdf/2012.12877.pdf "DeiT (2020) — arXiv"`.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL; fails strict URL regex.

## Non-paper Context URL
- Description: The url points to a competition or context page rather than the paper itself.
- Structural characteristics: domains or paths indicate competitions, events, or general info pages.
- Examples: 2024 "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" uses `https://arcprize.org/competitions/2024/`; 2024 "Towards Efficient Neurally-Guided Program Induction for ARC-AGI" uses `https://arcprize.org/competitions/2024/`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources (competitions, events, wiki).

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use landing patterns such as `/abs/`, `/doi/abs/`, or database landing pages.
- Examples: 2024 "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning" uses `https://arxiv.org/abs/2411.07279`; 2021 "TimeSformer: Is Space-Time Attention All You Need for Video Understanding?" uses `https://arxiv.org/abs/2102.05095`; 1971 "The hippocampus as a spatial map" uses `https://pubmed.ncbi.nlm.nih.gov/5124915/`; 2007 "Toward an executive without a homunculus: computational models of the prefrontal cortex/basal ganglia system" uses `https://royalsocietypublishing.org/doi/abs/10.1098/rstb.2007.2055`; 2022 "Training Compute-Optimal Large Language Models (Chinchilla)" uses `https://arxiv.org/abs/2203.15556`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF/DOI.
- Structural characteristics: URL contains `/html/` endpoints.
- Examples: 2025 "Towards the Next Generation of Symbolic Regression Benchmarks" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/`.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "Towards the Next Generation of Symbolic Regression Benchmarks" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms within the same dataset chunk.
- Structural characteristics: same arxiv.org domain with varying path types across rows.
- Examples: 2021 "The Pile: An 800GB Dataset of Diverse Text for Language Modeling" uses `https://arxiv.org/pdf/2101.00027`; 2021 "TimeSformer: Is Space-Time Attention All You Need for Video Understanding?" uses `https://arxiv.org/abs/2102.05095`; 2025 "Towards the Next Generation of Symbolic Regression Benchmarks" uses `https://arxiv.org/html/2505.03977v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal host instead of an official publisher or DOI.
- Structural characteristics: domains like university hosts, lab sites, or personal pages.
- Examples: 2010 "The free-energy principle: a unified brain theory?" uses `https://www.uab.edu/medicine/cinl/images/KFriston_FreeEnergy_BrainTheory.pdf`; 2024 "The Surprising Effectiveness of Test-Time Training for Few-Shot Learning" uses `https://ekinakyurek.github.io/papers/ttt.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of personal or institutional host patterns.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or repository metadata.
- Structural characteristics: URL includes a different year token than the row year.
- Examples: 1954 "The Theory of Dynamic Programming" uses `https://www.rand.org/content/dam/rand/pubs/papers/2008/P550.pdf` (implies 2008).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs and compare with the year column.

## Title-URL Mismatch
- Description: The url points to a page that does not correspond to the titled work.
- Structural characteristics: URL slug does not match title keywords or points to a generic competition page.
- Examples: 2024 "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning" links to `https://arcprize.org/competitions/2024/`; 2024 "Towards Efficient Neurally-Guided Program Induction for ARC-AGI" links to `https://arcprize.org/competitions/2024/`.
- Why harmful: Mislinks entries and confuses automated retrieval.
- Detection signals: URL slug does not align with title keywords or points to unrelated context pages.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column is a sentinel like `MISSING` rather than an http(s) URL.
- Examples: 2018 "Universal Transformer (UT)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url fails URL regex and matches known placeholders (MISSING, TBD, N/A).

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label text or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra label text and spaces.
- Examples: 2021 "Transformer in Transformer" has `https://arxiv.org/pdf/2103.00112.pdf` followed by label text; 2020 "UNITER: Universal Image-Text Representation Learning" has `https://arxiv.org/pdf/1909.11740.pdf` followed by label text; 2023 "Uni3DL: Unified Model for 3D and Language Understanding" has `https://arxiv.org/pdf/2312.03026.pdf` followed by label text.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL; fails strict URL regex.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2021 "Transformer in Transformer" appears twice with the same url; 2023 "Tree of Thoughts (ToT)" appears twice with the same url; 2015 "Trust Region Policy Optimization (TRPO)" appears twice with the same url; 2021 "Twins: Revisiting the Design of Spatial Attention in Vision Transformers" appears twice with the same url.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Landing or Abstract Page URL
- Description: The url points to a landing, abstract, or DOI page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/`, `/doi/`, or publisher landing/abstract pages.
- Examples: 2022 "TransNeRF: Generalizable Neural Radiance Fields for Novel View Synthesis with Transformer" uses `https://arxiv.org/abs/2206.05375`; 2019 "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" uses `https://arxiv.org/abs/1901.02860`; 2023 "Transformer-based 3D point cloud generation networks" uses `https://dl.acm.org/doi/10.1145/3581783.3612226`; 2024 "Transformers Can Do Arithmetic with the Right Embeddings" has `...Abstract-Conference.html`; 1990 "Unified Theories of Cognition" uses `https://www.hup.harvard.edu/books/9780674921016`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing/abstract patterns or domains.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values or url variants.
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2019 "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context" appears with `https://arxiv.org/abs/1901.02860` and `https://aclanthology.org/P19-1285.pdf`; 2024 "Transformers Can Do Arithmetic with the Right Embeddings" appears with the NeurIPS PDF and the NeurIPS abstract HTML page; 2018 "Universal Transformers" appears with an arXiv PDF (with label text) and an arXiv abstract URL as "Universal Transformers (UT)".
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include aliases, parenthetical variants, or expanded subtitles that refer to the same work.
- Structural characteristics: near-identical titles differing only by parenthetical tags, pluralization, or subtitle expansions.
- Examples: 2023 "Tree of Thoughts (ToT)" vs 2023 "Tree of Thoughts (ToT): Deliberate Problem Solving with Large Language Models"; 2018 "Universal Transformer (UT)" vs 2018 "Universal Transformers" vs 2018 "Universal Transformers (UT)".
- Why harmful: Exact-title dedupe fails and search results fragment.
- Detection signals: fuzzy-title matching after stripping parentheticals/subtitles and normalizing case.

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal/institutional host instead of an official publisher or DOI.
- Structural characteristics: domains indicate archives, personal pages, or institutional hosts.
- Examples: 1949 "Translation (Weaver Memorandum)" uses `https://www.mt-archive.net/50/Weaver-1949.pdf`; 1928 "Transmission of Information" uses `https://monoskop.org/...`; 1982 "Two cortical visual systems" uses `https://www.cns.nyu.edu/~tony/...`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `~` or archive-style hosts.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or repository metadata.
- Structural characteristics: URL includes a different year token than the row year (e.g., arXiv ID).
- Examples: 2020 "UNITER: Universal Image-Text Representation Learning" uses `https://arxiv.org/pdf/1909.11740.pdf` (implies 2019).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (arXiv IDs, conference-year paths) and compare with the year column.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: same arxiv.org domain with different path types across rows.
- Examples: 2022 "TransNeRF: Generalizable Neural Radiance Fields for Novel View Synthesis with Transformer" uses `https://arxiv.org/abs/2206.05375` while 2025 "TransXSSM: Hybrid Transformer-SSM with Unified RoPE" uses `https://arxiv.org/pdf/2506.09507.pdf`; 2015 "Trust Region Policy Optimization (TRPO)" uses `https://arxiv.org/abs/1502.05477` while 2021 "Twins: Revisiting the Design of Spatial Attention in Vision Transformers" uses `https://arxiv.org/pdf/2104.13840.pdf`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended labels or notes instead of a single clean URL.
- Structural characteristics: quoted url field contains a PDF URL followed by extra label text or quoted fragments.
- Examples: 2021 "VATT: Transformers for Multimodal Self-Supervised Learning" has `https://arxiv.org/pdf/2104.11178.pdf` plus label text like `VATT (2021) - arXiv`; 2024 "VG4D: Vision-Language Model Goes 4D Video Recognition" has `https://arxiv.org/pdf/2404.11605.pdf` plus label text; 2019 "ViLBERT: Pretraining Task-Agnostic Vision-and-Language Representations" has `https://arxiv.org/pdf/1908.02265.pdf` plus label text.
- Why harmful: URL parsing fails, linkers treat the field as invalid, and dedupe breaks across annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL; regex match URL then trailing text.

## Landing or Abstract Page URL
- Description: The url points to landing or abstract pages rather than full-text PDFs or canonical artifacts.
- Structural characteristics: URLs lack `.pdf` and use arXiv `/abs/`, OpenReview forum pages, or publisher book landing pages.
- Examples: 2019 "VCR: Visual Commonsense Reasoning (From Recognition to Cognition...)" uses `https://arxiv.org/abs/1811.10830`; 2024 "Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis" uses `https://arxiv.org/abs/2405.21075`; 2025 "Wavelet-based Positional Representation for Long Context" uses `https://openreview.net/forum?id=OhauMUNW8T`; 1982 "Vision: A Computational Investigation into the Human Representation and Processing of Visual Information" uses `https://direct.mit.edu/books/monograph/3299/VisionA-Computational-Investigation-into-the-Human`.
- Why harmful: Retrieval requires scraping or extra navigation; access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches `/abs/` or known landing-page domains (openreview.net/forum, direct.mit.edu/books).

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: same arxiv.org domain appears with different path types across rows.
- Examples: 2019 "VCR: Visual Commonsense Reasoning (From Recognition to Cognition...)" uses `https://arxiv.org/abs/1811.10830` while 2021 "VATT: Transformers for Multimodal Self-Supervised Learning" uses `https://arxiv.org/pdf/2104.11178.pdf`; 2022 "ViTDet: Exploring Plain ViT Backbones for Object Detection" uses `https://arxiv.org/abs/2203.16527` while 2019 "VideoBERT: A Joint Model for Video and Language Representation Learning" uses `https://arxiv.org/pdf/1904.01766.pdf`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2020 "Vokenization: Improving Language Understanding with Contextualized, Visual-Grounded Supervision" appears twice with `https://arxiv.org/pdf/2010.06775`.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2025 "Vector Symbolic Algebras for the Abstraction and Reasoning Corpus" appears with `https://arxiv.org/abs/2511.08747` and `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Non-paper Context URL
- Description: The url points to a context or blog page rather than the paper itself.
- Structural characteristics: domains or paths indicate blog posts or results analysis rather than a paper artifact.
- Examples: 2025 "Vector Symbolic Algebras for the Abstraction and Reasoning Corpus" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources or contains `/blog/`, `/results`, or competition-style paths.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL or repository metadata.
- Structural characteristics: URL includes a different year token than the row year (e.g., arXiv ID).
- Examples: 2019 "VCR: Visual Commonsense Reasoning (From Recognition to Cognition...)" uses arXiv `1811.10830` (implies 2018); 2017 "VQA v2.0 (Making the V in VQA Matter)" uses `https://arxiv.org/abs/1612.00837` (implies 2016).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URLs (arXiv IDs, conference-year paths) and compare with the year column.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/fulltext/` or similar HTML render endpoints.
- Examples: 2024 "Voxel self-attention and center-point for 3D object detector" uses `https://www.cell.com/iscience/fulltext/S2589-0042%2824%2901984-9`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/`, `/full`, or `/fulltext/`.

## URL Field Contains Extra Text
- Description: The url cell includes a URL followed by appended label text rather than a single clean URL.
- Structural characteristics: quoted url field contains a URL then spaces or quoted label fragments.
- Examples: 2022 "Winoground: Probing Vision-Language Models for Compositionality" has `https://arxiv.org/pdf/2204.03162.pdf` plus label text; 2023 "YaRN: Efficient Context Window Extension of Large Language Models" has `https://arxiv.org/pdf/2309.00071.pdf` plus label text; 2021 "Zero-Shot Text-to-Image Generation" has `https://arxiv.org/pdf/2102.12092.pdf` plus label text.
- Why harmful: URL parsing and validation can fail, and annotation variants create duplicates.
- Detection signals: url field contains spaces or quoted text after a URL.

## Landing or Abstract Page URL
- Description: The url points to a landing or abstract page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/` or publisher landing pages (e.g., ScienceDirect, PubMed).
- Examples: 1974 "Working Memory" uses `https://www.sciencedirect.com/science/article/pii/S0079742108604521`; 2012 "Working Memory: Theories, Models, and Controversies" uses `https://pubmed.ncbi.nlm.nih.gov/21961947/`; 2023 "YaRN: Efficient Context Window Extension of Large Language Models" uses `https://arxiv.org/abs/2309.00071`.
- Why harmful: Automated retrieval may require scraping or fail, and access may be paywalled.
- Detection signals: url lacks `.pdf` and matches `/abs/`, `pubmed.ncbi.nlm.nih.gov`, or `sciencedirect.com/science/article`.

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2025 "Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D)" appears twice with `https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Zero-Shot_4D_Lidar_Panoptic_Segmentation_CVPR_2025_paper.pdf`; 2021 "Zero-Shot Text-to-Image Generation" appears twice with the same arXiv PDF plus label text.
- Why harmful: Inflates counts and breaks unique key assumptions.
- Detection signals: exact row-level hashing repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same title/year across rows but differing URL types (arXiv abs vs pdf).
- Examples: 2023 "YaRN: Efficient Context Window Extension of Large Language Models" appears with `https://arxiv.org/abs/2309.00071` and with `https://arxiv.org/pdf/2309.00071.pdf` (plus label text); 2021 "Zero-Shot Text-to-Image Generation" appears with `https://arxiv.org/pdf/2102.12092.pdf` (plus label text) and as "Zero-Shot Text-to-Image Generation (DALL-E)" with `https://arxiv.org/abs/2102.12092`.
- Why harmful: Creates duplicates and fragments citations.
- Detection signals: normalized title+year matches multiple URLs.

## Title Variants or Aliases
- Description: Titles include aliases, parenthetical notes, or merged names for the same work.
- Structural characteristics: titles include slashes or parenthetical aliases that change the surface form.
- Examples: 2021 "Zero-Shot Text-to-Image Generation" vs 2021 "Zero-Shot Text-to-Image Generation (DALL-E)"; 2022 "XPos / Length-Extrapolatable Transformer"; 2013 "word2vec (Skip-gram / CBOW)".
- Why harmful: Exact-title dedupe fails and canonical naming becomes ambiguous.
- Detection signals: titles with parentheses, slashes, or alias tokens.

## Placeholder URL Value
- Description: The url field uses a placeholder instead of a real URL.
- Structural characteristics: url is a sentinel like `MISSING`.
- Examples: 2013 "word2vec (Skip-gram / CBOW)" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents retrieval.
- Detection signals: url equals known placeholders (MISSING, TBD, N/A).

## Non-paper Context URL
- Description: The url points to a competition or context page rather than a paper.
- Structural characteristics: domain/path indicates competitions or event pages.
- Examples: 2025 "the ARChitects" uses `https://arcprize.org/competitions/2025/`.
- Why harmful: Misrepresents sources and blocks automated paper retrieval.
- Detection signals: domain/path matches competition-style URLs (e.g., `/competitions/`).

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL.
- Structural characteristics: arXiv ID or URL suggests a different year than the row year.
- Examples: 2020 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" uses `https://arxiv.org/abs/1910.02054` (implies 2019).
- Why harmful: Misorders timeline entries and complicates temporal analysis.
- Detection signals: parse year-like tokens from URLs (arXiv IDs) and compare with year.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` URL forms within the same chunk.
- Structural characteristics: same arxiv.org domain appears with different path types across rows.
- Examples: 2023 "YaRN: Efficient Context Window Extension of Large Language Models" uses `https://arxiv.org/abs/2309.00071` while 2022 "Winoground: Probing Vision-Language Models for Compositionality" uses `https://arxiv.org/pdf/2204.03162.pdf`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Non-standard arXiv PDF URL (Missing .pdf Extension)
- Description: arXiv PDF URLs omit the `.pdf` suffix, creating non-canonical links.
- Structural characteristics: arXiv URL uses `/pdf/<id>` without trailing `.pdf`.
- Examples: 2019 "XLNet: Generalized Autoregressive Pretraining for Language Understanding" uses `https://arxiv.org/pdf/1906.08237`.
- Why harmful: File-type detection and strict validators may fail; duplicates arise if `.pdf` and non-`.pdf` forms both appear.
- Detection signals: regex match `arxiv.org/pdf/\\d+\\.\\d+$` (no `.pdf` suffix).

## Exact Duplicate Row
- Description: The same year, title, and url appear multiple times as identical rows.
- Structural characteristics: duplicate lines with identical values across all three columns.
- Examples: 2020 "An Image Is Worth 16x16 Words (ViT)" with `https://arxiv.org/abs/2010.11929` appears twice; 2017 "Attention Is All You Need" with `https://arxiv.org/pdf/1706.03762` appears multiple times; 2018 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" with `https://arxiv.org/pdf/1810.04805` appears twice.
- Why harmful: Inflates counts, skews analytics, and causes deduplication failures downstream.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different url values.
- Structural characteristics: same or near-identical title/year across rows, but url differs.
- Examples: 2019 "Analysing Mathematical Reasoning Abilities of Neural Models" appears with `https://arxiv.org/abs/1904.01557` and `https://arxiv.org/pdf/1904.01557`; 2017 "Attention Is All You Need" appears with arXiv `/abs/1706.03762`, arXiv `/pdf/1706.03762`, and `https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf`; 2025 "ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory" appears with `https://ar5iv.org/abs/2509.04439` and `https://arcprize.org/blog/arc-prize-2025-results-analysis`; 2011 "Automating String Processing in Spreadsheets Using Input-Output Examples (FlashFill)" appears with `https://www.microsoft.com/.../popl11-synthesis.pdf` and `https://www.microsoft.com/.../synasc12.pdf`.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values.

## Title Variants or Aliases
- Description: Titles include casing, punctuation, or symbol variants that refer to the same work.
- Structural characteristics: near-identical titles differing only by case, punctuation, or symbol substitution.
- Examples: 2020 "An Image Is Worth 16x16 Words (ViT)" vs 2020 "An Image is Worth 16×16 Words: Vision Transformer (ViT)"; 2011 "Automating String Processing in Spreadsheets Using Input-Output Examples (FlashFill)" vs 2011 "Automating String Processing in Spreadsheets using Input-Output Examples (FlashFill)".
- Why harmful: Dedupe by exact title fails and search results fragment.
- Detection signals: fuzzy-title matching after normalizing case, punctuation, and Unicode symbols.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended notes or labels instead of a single clean URL value.
- Structural characteristics: quoted url field contains a URL followed by extra quoted text or label fragments.
- Examples: 2020 "An Image is Worth 16×16 Words: Vision Transformer (ViT)" has `https://arxiv.org/pdf/2010.11929.pdf` plus `Vision Transformer (ViT) (2020) — arXiv`; 2017 "Attention Is All You Need" has `https://arxiv.org/pdf/1706.03762.pdf` plus `Attention Is All You Need (2017) — arXiv`; 2018 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" has `https://arxiv.org/pdf/1810.04805.pdf` plus `BERT (2018) — arXiv`; 2023 "Average-Hard Attention Transformers Are Threshold Circuits" has `https://arxiv.org/pdf/2308.03212.pdf` plus `Average-Hard Attention as Threshold Circuits (2023) — arXiv`.
- Why harmful: URL parsing fails, automated linkers treat the value as invalid, and duplicates proliferate due to annotation variants.
- Detection signals: url field contains spaces or extra quoted segments after a URL.

## Placeholder URL Value
- Description: The url field contains a placeholder token instead of a real URL.
- Structural characteristics: url column contains a non-URL sentinel like `MISSING`.
- Examples: 2014 "Auto-Encoding Variational Bayes (VAE)" has url `MISSING`; 2018 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated retrieval or linking.
- Detection signals: url does not match URL pattern and matches known placeholder tokens (e.g., MISSING, TBD, N/A).

## Landing or Abstract Page URL
- Description: The url points to a landing, abstract, or forum page instead of a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/`, publisher landing pages, or review forums.
- Examples: 2020 "An Image Is Worth 16x16 Words (ViT)" uses `https://arxiv.org/abs/2010.11929`; 2016 "Asynchronous Methods for Deep Reinforcement Learning (A3C)" uses `https://arxiv.org/abs/1602.01783`; 2001 "An Integrative Theory of Prefrontal Cortex Function" uses `https://pubmed.ncbi.nlm.nih.gov/11283309/`; 2025 "Arithmetic-Bench: Evaluating Multi-Step Reasoning in LLMs through Basic Arithmetic Operations" uses `https://openreview.net/forum?id=ae6bKeffGZ`; 2022 "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers" uses `https://arxiv.org/abs/2203.17270`.
- Why harmful: Automated retrieval may fail or require scraping, and access may be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns or domains.

## Non-paper Context URL
- Description: The url points to a blog, results analysis, or context page rather than the paper itself.
- Structural characteristics: domain/path indicates blog or news content instead of a paper artifact.
- Examples: 2025 "ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches known non-paper sources (blog, news, competitions, results-analysis).

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal host instead of an official publisher or DOI.
- Structural characteristics: domains like personal university hosts or third-party mirrors.
- Examples: 2002 "Approximately Optimal Approximate Reinforcement Learning" uses `https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf`; 2025 "ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory" uses `https://ar5iv.org/abs/2509.04439`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: domain not in known publisher list; presence of `~` or known mirror domains like `ar5iv.org`.

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/html/` endpoints.
- Examples: 2025 "Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT)" uses `https://arxiv.org/html/2511.21882v1`.
- Why harmful: Rendered HTML views can change, require scripts, or be less stable than PDFs/DOIs.
- Detection signals: url contains `/html/` (including `/doi/full/`).

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) rather than the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT)" uses `https://arxiv.org/html/2511.21882v1`.
- Why harmful: Creates duplicates across versions and can become stale as versions update.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` URL forms within the same chunk.
- Structural characteristics: same base domain `arxiv.org` appears with different path types across rows.
- Examples: 2020 "An Image Is Worth 16x16 Words (ViT)" uses `https://arxiv.org/abs/2010.11929`; 2018 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" uses `https://arxiv.org/pdf/1810.04805`; 2025 "Autoregressive Modeling as Iterative Latent Equilibrium (Equilibrium Transformers, EqT)" uses `https://arxiv.org/html/2511.21882v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types within a chunk.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year implied by tokens in the URL.
- Structural characteristics: URL includes a different year token than the row year.
- Examples: 2011 "Automating String Processing in Spreadsheets using Input-Output Examples (FlashFill)" links to `https://www.microsoft.com/.../synasc12.pdf` (token suggests 2012 while row year is 2011).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse year-like tokens in URL filenames (e.g., `synasc12`) and compare with the year column.

## Non-standard arXiv PDF URL (Missing .pdf Extension)
- Description: arXiv PDF URLs omit the `.pdf` suffix, creating non-canonical links.
- Structural characteristics: arXiv URL uses `/pdf/<id>` without trailing `.pdf`.
- Examples: 2017 "Attention Is All You Need" uses `https://arxiv.org/pdf/1706.03762`; 2019 "Analysing Mathematical Reasoning Abilities of Neural Models" uses `https://arxiv.org/pdf/1904.01557`; 2018 "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" uses `https://arxiv.org/pdf/1810.04805`.
- Why harmful: File-type detection and strict validators may fail; duplicates arise if `.pdf` and non-`.pdf` forms both appear.
- Detection signals: regex match `arxiv.org/pdf/\\d+\\.\\d+$` (no `.pdf` suffix).

## URL Field Contains Extra Text
- Description: The url cell contains a URL plus appended label text instead of a single clean URL.
- Structural characteristics: quoted url field includes a URL followed by extra words and embedded quotes (often `""...""`).
- Examples: 2021 "BEiT: BERT Pre-Training of Image Transformers" has `https://arxiv.org/pdf/2106.08254.pdf` plus `BEiT (2021) — arXiv`; 2023 "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models" has `https://arxiv.org/pdf/2301.12597.pdf` plus `BLIP-2 (2023) — arXiv`; 2012 "Canonical Microcircuits for Predictive Coding" has `https://pubmed.ncbi.nlm.nih.gov/23238495/` plus `Canonical Microcircuits for Predictive Coding (2012) — PubMed`.
- Why harmful: URL parsers fail, automated linkers reject the value, and annotation variants cause duplicate records.
- Detection signals: url field contains spaces or quoted annotations after a URL; presence of `""` within a quoted url cell.

## Exact Duplicate Row
- Description: The same year, title, and url are repeated as identical rows.
- Structural characteristics: duplicate lines where all three columns match exactly.
- Examples: 2021 "BEiT: BERT Pre-Training of Image Transformers" with the arXiv PDF+label url appears twice; 2016 "BlinkFill: Semi-supervised Programming By Example for Syntactic String Transformations" appears twice with `https://www.vldb.org/pvldb/vol9/p816-singh.pdf`; 2021 "CSWin Transformer: A General Vision Transformer Backbone with Cross-Shaped Windows" appears twice with the same arXiv PDF+label url.
- Why harmful: Inflates counts, skews analytics, and complicates deduplication logic.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work is listed in multiple rows with different URL forms.
- Structural characteristics: same title (or near-identical) across rows while URL differs (e.g., arXiv `/abs/` vs `/pdf/` with labels).
- Examples: "BEiT: BERT Pre-Training of Image Transformers" appears as 2022 with `https://arxiv.org/abs/2106.08254` and as 2021 with `https://arxiv.org/pdf/2106.08254.pdf` plus label text.
- Why harmful: Creates duplicate entries, fragments citations, and obscures the canonical record.
- Detection signals: normalized title matches and URLs share the same DOI/arXiv ID but differ in path or formatting.

## Same Work Listed Multiple Times (Different Years)
- Description: The same work appears in multiple rows but with different year values.
- Structural characteristics: identical or near-identical titles paired with the same DOI/arXiv ID across different years.
- Examples: "BEiT: BERT Pre-Training of Image Transformers" appears as 2022 and 2021 while both reference arXiv ID `2106.08254`.
- Why harmful: Misorders timeline entries, breaks year-based analyses, and complicates deduplication.
- Detection signals: same DOI/arXiv ID or URL root appears under multiple years for the same title.

## Landing or Abstract Page URL
- Description: The url points to a landing/metadata page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use abstract/landing patterns or poster pages.
- Examples: 2025 "BIG-Bench Extra Hard (BBEH)" uses `https://aclanthology.org/2025.acl-long.1285/`; 2015 "Batch Normalization" uses `https://arxiv.org/abs/1502.03167`; 2025 "BWFormer: Building Wireframe Reconstruction from Airborne LiDAR Point Cloud with Transformer" uses `https://cvpr.thecvf.com/virtual/2025/poster/32868`; 2025 "Boosting Performance on ARC via Perspective / Augmentations" uses `https://proceedings.mlr.press/v267/franzen25a.html`; 1935 "Can Quantum-Mechanical Description of Physical Reality Be Considered Complete?" uses `https://link.aps.org/doi/10.1103/PhysRev.47.777`.
- Why harmful: Automated retrieval may fail or require scraping, and access can be paywalled or unstable.
- Detection signals: url does not end with `.pdf` and matches known landing patterns (`/abs/`, `/doi/`, `/virtual/.../poster/`, or HTML paper pages).

## Non-paper Context URL
- Description: The url points to a blog or analysis page rather than the paper itself.
- Structural characteristics: domain/path indicates blog or competition analysis rather than a paper artifact.
- Examples: 2025 "Beyond Brute Force: A Neuro-Symbolic Architecture for Compositional Reasoning in ARC-AGI-2" uses `https://arcprize.org/blog/arc-prize-2025-results-analysis`.
- Why harmful: Misrepresents the source and prevents retrieval of the cited work.
- Detection signals: url contains `/blog/` or domains known for announcements/analysis rather than publications.

## Non-canonical Mirror Source
- Description: The url points to a mirror or personal/institutional host instead of an official publisher page.
- Structural characteristics: domains include personal directories, `~` paths, or third-party archive sites.
- Examples: 1972 "Broadcast Channels" uses `https://isl.stanford.edu/~cover/papers/transIT/0002cove.pdf`; 1924 "Certain Factors Affecting Telegraph Speed" uses `https://monoskop.org/images/9/9f/Nyquist_Harry_1924_Certain_Factors_Affecting_Telegraph_Speed.pdf`.
- Why harmful: Higher link-rot risk, unclear versioning, and duplicate source proliferation.
- Detection signals: presence of `~` in paths or domains not associated with official publishers.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/` and `/pdf/` forms within the same chunk.
- Structural characteristics: multiple arxiv.org URL path types appear across rows.
- Examples: 2022 "BEiT: BERT Pre-Training of Image Transformers" uses `https://arxiv.org/abs/2106.08254` while 2023 "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models" uses `https://arxiv.org/pdf/2301.12597.pdf` and 2019 "CAIL2019-SCM: A Dataset of Similar Case Matching in Legal Domain" uses `https://arxiv.org/pdf/1911.08962`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types in the same file.

## Year-URL Metadata Mismatch
- Description: The year column conflicts with the year token implied by the URL.
- Structural characteristics: arXiv IDs encode a year/month that differs from the row year.
- Examples: 2022 "BEiT: BERT Pre-Training of Image Transformers" links to `https://arxiv.org/abs/2106.08254` (token suggests 2021).
- Why harmful: Misorders timeline entries and complicates temporal analyses.
- Detection signals: parse arXiv IDs or DOI tokens for year-like patterns and compare with the year column.

## Placeholder URL Value
- Description: The url field uses a placeholder token instead of a real URL.
- Structural characteristics: url column contains a sentinel such as `MISSING` rather than an http(s) URL.
- Examples: 2022 "Constitutional AI" has url `MISSING`.
- Why harmful: Breaks URL validation and prevents automated linking or retrieval.
- Detection signals: url equals known placeholders (MISSING, TBD, N/A).

## Exact Duplicate Row
- Description: Identical rows repeat with the same year, title, and url.
- Structural characteristics: duplicate lines with identical year+title+url values.
- Examples: 2021 "CoAtNet: Marrying Convolution and Attention for All Data Sizes" with url `https://arxiv.org/pdf/2106.04803.pdf ""CoAtNet (2021) - arXiv""` appears twice; 2021 "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification" with url `https://arxiv.org/pdf/2103.14899.pdf ""CrossViT (2021) - arXiv""` appears twice.
- Why harmful: Inflates counts and complicates deduplication and analytics.
- Detection signals: exact row-level hashing or unique key (year+title+url) repeats.

## URL Field Contains Extra Text
- Description: The url cell includes a URL plus appended label/notes text instead of a single clean URL.
- Structural characteristics: quoted url field contains a URL followed by extra words and embedded quotes (e.g., `""...""`).
- Examples: 2025 "Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Large Vision-Language Models" has `https://arxiv.org/pdf/2505.16416.pdf ""Circle-RoPE (2025) - arXiv""`; 2025 "ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices" has `https://arxiv.org/pdf/2506.03737.pdf ""ComRoPE (2025) - arXiv""`; 2024 "Combining Induction and Transduction for Abstract Reasoning" has `https://arxiv.org/pdf/2411.02272.pdf ""Combining Induction and Transduction (2024) - arXiv""`; 2021 "CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification" has `https://arxiv.org/pdf/2103.14899.pdf ""CrossViT (2021) - arXiv""`.
- Why harmful: URL parsing fails and annotation variants cause duplicate records.
- Detection signals: url field contains spaces or extra quoted segments after a URL; presence of `""` inside the quoted url cell.

## Landing or Abstract Page URL
- Description: The url points to an abstract/landing page rather than a full-text PDF or canonical artifact.
- Structural characteristics: URLs lack `.pdf` and use `/abs/`, publisher landing pages, or abstract HTML pages.
- Examples: 2022 "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning" uses `https://arxiv.org/abs/2203.10244`; 2021 "CoAtNet: Marrying Convolution and Attention" uses `https://proceedings.neurips.cc/paper/2021/hash/20568692db622456cc42a2e853ca21f8-Abstract.html`; 2019 "CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge" uses `https://aclanthology.org/N19-1421/`; 1948 "Cognitive Maps in Rats and Men" uses `https://pubmed.ncbi.nlm.nih.gov/18870876/`.
- Why harmful: Automated retrieval may require scraping or access to paywalled pages and does not point to a stable full-text artifact.
- Detection signals: url does not end with `.pdf` and matches known landing patterns (`/abs/`, `/Abstract.html`, ACL Anthology landing pages, PubMed).

## HTML Full-Text Render URL
- Description: The url targets an HTML-rendered full-text view rather than a canonical PDF or DOI.
- Structural characteristics: URL contains `/html/` endpoints.
- Examples: 2025 "ChartQAPro: A More Diverse and Challenging Benchmark for Real-World Chart QA" uses `https://arxiv.org/html/2504.05506v1`.
- Why harmful: HTML render pages can change and are less stable than PDFs/DOIs.
- Detection signals: url contains `/html/`.

## Version-specific arXiv URL
- Description: The url includes a specific arXiv version (v1, v2, etc.) instead of the canonical identifier.
- Structural characteristics: arXiv URL ends with `v` followed by digits.
- Examples: 2025 "ChartQAPro: A More Diverse and Challenging Benchmark for Real-World Chart QA" uses `https://arxiv.org/html/2504.05506v1`.
- Why harmful: Creates duplicates across versions and becomes stale as newer versions appear.
- Detection signals: regex match `arxiv.org/.*/\\d+\\.\\d+v\\d+`.

## Inconsistent arXiv URL Forms
- Description: arXiv entries mix `/abs/`, `/pdf/`, and `/html/` forms within the same chunk.
- Structural characteristics: same base domain `arxiv.org` appears with different path types across rows.
- Examples: 2022 "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning" uses `https://arxiv.org/abs/2203.10244`; 2024 "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" uses `https://arxiv.org/pdf/2403.04132`; 2025 "ChartQAPro: A More Diverse and Challenging Benchmark for Real-World Chart QA" uses `https://arxiv.org/html/2504.05506v1`.
- Why harmful: Deduplication by URL fails and retrieval logic must handle multiple formats.
- Detection signals: normalize arXiv IDs and flag mixed path types in the same file.

## Same Work Listed Multiple Times (Different URLs)
- Description: A single work appears in multiple rows with different URL values.
- Structural characteristics: same or near-identical title/year appears more than once with different urls.
- Examples: 2025 "ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices" appears with `https://arxiv.org/abs/2506.03737` and `https://arxiv.org/pdf/2506.03737.pdf ""ComRoPE (2025) - arXiv""`; 2024 "Combining Induction and Transduction for Abstract Reasoning" appears with `https://arxiv.org/abs/2411.02272`, `https://arcprize.org/competitions/2024/`, and `https://arxiv.org/pdf/2411.02272.pdf ""Combining Induction and Transduction (2024) - arXiv""`; 2021 "CoAtNet: Marrying Convolution and Attention" appears alongside "CoAtNet: Marrying Convolution and Attention for All Data Sizes" with NeurIPS abstract vs arXiv PDF URLs.
- Why harmful: Creates duplicate entries, complicates canonicalization, and fragments citation counts.
- Detection signals: normalized title+year appears multiple times with different url values (including arXiv abs/pdf or context pages).

## Non-paper Context URL
- Description: The url points to a competition or context page rather than the paper itself.
- Structural characteristics: domain/path indicates competitions or event pages.
- Examples: 2024 "Combining Induction and Transduction for Abstract Reasoning" uses `https://arcprize.org/competitions/2024/`.
- Why harmful: Misrepresents sources and prevents retrieval of the actual paper.
- Detection signals: domain/path matches competition-style URLs (e.g., `/competitions/`).

## Non-standard arXiv PDF URL (Missing .pdf Extension)
- Description: arXiv PDF URLs omit the `.pdf` suffix, producing non-canonical links.
- Structural characteristics: arXiv URL uses `/pdf/<id>` without trailing `.pdf`.
- Examples: 2024 "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" uses `https://arxiv.org/pdf/2403.04132`; 2017 "Convolutional Sequence to Sequence Learning" uses `https://arxiv.org/pdf/1705.03122`.
- Why harmful: File-type detection and strict validators may fail; duplicates arise if both `.pdf` and non-`.pdf` forms appear.
- Detection signals: regex match `arxiv.org/pdf/\\d+\\.\\d+$`.

## Title Variants or Aliases
- Description: Titles include aliases or truncated variants that refer to the same work.
- Structural characteristics: titles include parenthetical acronyms, slash-separated aliases, or shortened forms.
- Examples: 2021 "CoAtNet: Marrying Convolution and Attention" vs 2021 "CoAtNet: Marrying Convolution and Attention for All Data Sizes"; 2017 "Combining Improvements in Deep Reinforcement Learning (Rainbow)"; 2017 "Constructing Datasets for Multi-hop Reading Comprehension (WikiHop / MedHop; QAngaroo)"; 2015 "Continuous Control with Deep Reinforcement Learning (DDPG)".
- Why harmful: Exact-title dedupe fails and canonical naming becomes ambiguous.
- Detection signals: titles with parentheses, slashes, or alias tokens; fuzzy-title matching after removing parentheticals.

## Non-canonical Mirror Source
- Description: The url points to a lab or institutional host rather than an official publisher page.
- Structural characteristics: domains indicate lab sites or institutional file hosting.
- Examples: 2007 "Core Knowledge" uses `https://www.harvardlds.org/wp-content/uploads/2017/01/SpelkeKinzler07-1.pdf`.
- Why harmful: Higher link-rot risk and unclear versioning.
- Detection signals: domain not in known publisher list; lab site domains or institutional uploads.
