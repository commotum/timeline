# Zero-Shot 4D Lidar Panoptic Segmentation (Year not specified)
Source: Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D).md

## Binary decision
Decision: TRANSFORMER-YES
Confidence: high
Basis: source-targeted-scan

## Why
- The paper’s central SAL-4D model explicitly uses a Transformer decoder-based architecture and a Transformer-based object instance decoder.
- These Transformer components are described inside the main model definition (Section 3.3), indicating they are core to the method’s main results rather than peripheral baselines.
- The extending-dimensions analysis file was unavailable (`MISSING`), so the decision used the abstract plus available auxiliary files and a targeted scan of the source model section.

## Evidence
- "To operationalize this, we employ a Transformer decoder-based architecture [12]." (Section 3.3, `Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D).md`)
- "by a Transformer-based object instance decoder that localizes objects in the 4D Lidar space (*cf.*, [55, 105])." (Section 3.3, `Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D).md`)
- "Our segmentation decoder follows the design of [12, 14, 55]. Inputs to the decoder are a set of M learnable queries that interact with voxel features" (Section 3.3, `Zero-Shot 4D LiDAR Panoptic Segmentation (SAL-4D).md`)

## Pass accounting
Pass 1 (abstract + auxiliary files): performed - abstract and all available auxiliary files reviewed; no explicit decisive architecture cue in auxiliary summaries alone.
Pass 2 (targeted source scan): performed - model section confirms Transformer decoders are central to SAL-4D.
