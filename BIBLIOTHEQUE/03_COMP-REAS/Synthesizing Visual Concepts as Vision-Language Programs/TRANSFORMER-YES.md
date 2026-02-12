# Synthesizing Visual Concepts as Vision-Language Programs (Year not specified)
Source: Synthesizing Visual Concepts as Vision-Language Programs.md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: medium
Basis: source-targeted-scan

## Why
- The core VLP pipeline materially depends on pretrained VLM components for symbol grounding and image-to-symbol extraction, so a central hybrid model includes modern VLM backbones.
- The evaluated base models are InternVL, Qwen-VL, Kimi-VL, and GPT-5 variants; these are Transformer-family vision-language systems (inference from model families named in the paper), and VLP is evaluated as "w/ VLP" on top of them.
- The Extending-dimensions analysis markdown was unavailable (`MISSING`), so the decision is based on the abstract, available auxiliary files, and a targeted source scan.

## Evidence
- "We propose Vision-Language Programs (VLP), which combine the perceptual flexibility of VLMs with systematic reasoning of program synthesis." (Abstract, `Synthesizing Visual Concepts as Vision-Language Programs.md`)
- "In detail, given a reasoning task  $\mathcal{X}$ , VLP provides task-specific groundings of these abstract types by querying a pretrained VLM  $\mathcal{M}$ ." (Section 3.2, `Synthesizing Visual Concepts as Vision-Language Programs.md`)
- "Namely, we utilize InternVL3-8B and InternVL3-14B [2], Kimi-VL-A3B-Instruct [18], as well as Qwen2.5-VL-72B [23] and Qwen3-VL-30B-A3B-Instruct [24]." (Section 4 Experimental Setup, `Synthesizing Visual Concepts as Vision-Language Programs.md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - VLP is clearly VLM-centered, but explicit architecture-family cues were insufficiently explicit for high-confidence Transformer attribution from auxiliary files alone.
Pass 2 (targeted source scan): performed - confirmed concrete VLM model families used as central backbones in the main experiments, supporting TRANSFORMER-YES.
