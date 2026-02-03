# BIBLIOTHEQUE Structure

Quick reference for what lives here and how it is organized.

## Top level
- `BIBLIOTHEQUE.csv` is the master index of papers and their classification.
- `00 - Class.md` documents the class criteria and the code map used in the CSV.
- `NEW/Scripts/download_pdfs.sh` is a helper script for fetching PDFs (see the script for details).
- `01_POS-ENCDR/` Positional Encoding Improvement Proposal (class_id 1).
- `02_XFORM-DIM/` Increasing Transformer Dimensions (class_id 2).
- `03_COMP-REAS/` Computation and Reasoning Mechanism Proposal (class_id 3).
- `04_DATA-BNCH/` Data, Benchmarks and Measurement (class_id 4).
- `05_ML-FNDTNS/` ML Foundations and Principles (class_id 5).
- `06_X-CONTEXT/` External Foundations and Cross-Disciplinary Context (class_id 6).
- `07_MIS-CLASS/` Misclassifications (class_id 7).
- `STRUCTURE.md` is this reference file.

## BIBLIOTHEQUE.csv columns
- `year` publication year (integer).
- `title` paper title (original or preferred display name).
- `class` class code (matches the suffix of the class folder, like `POS-ENCDR`).
- `class_id` numeric class identifier (1-7, matches `CLASS_<id>.md` and folder prefix).
- `filename` base name used for the paper subdirectory and file names.
- `url` source URL for the PDF.

## Class directories
- Each class folder contains subdirectories named after the `filename` field in the CSV.
- Each paper subdirectory uses a consistent file layout:
  - `<Paper>.pdf` is the downloaded source PDF.
  - `<Paper>.md` is the OCR-extracted text of the PDF (same base name as the folder).
  - `<Paper>_meta.json` contains extraction metadata (table of contents and page layout info).
  - `CLASS_<class_id>.md` is the classification record written at initial classification.
  - `_page_<n>_Figure_<m>.jpeg` and `_page_<n>_Picture_<m>.jpeg` are images extracted from the PDF (figures, tables, or pictures). Some folders may have none.
