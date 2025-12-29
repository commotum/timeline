You are given a CSV file with columns:

year,title,url

Each row is intended to represent one academic paper, benchmark, or model.

Your task has four phases.

PHASE 1 — EXAMINE THE CSV
Examine each entry (row) in the CSV individually and in relation to other rows in the same file.

PHASE 2 — IDENTIFY DATA QUALITY ISSUES
For each entry, identify all data quality issues present, including but not limited to:
- Invalid, placeholder, or non-URL values in the url field
- URLs containing embedded notes, comments, or multiple logical values
- URLs that do not correspond to the paper named in the title
- Index or landing pages used in place of full papers
- HTML renders used in place of canonical identifiers or PDFs
- Duplicate or near-duplicate entries (exact or semantic)
- The same paper listed multiple times under different years
- arXiv, conference, and journal versions treated as separate works
- Year inconsistencies or ambiguities, including inferable but missing years
- Title mismatches, aliases, merged titles, or non-canonical naming
- Schema or formatting violations (broken quoting, field misuse, inconsistent structure)
- Use of unstable or non-canonical sources when better ones likely exist
- Any other anomalies, patterns, or structural problems that could cause ambiguity, duplication, or downstream processing failures, even if not previously defined

Do not correct, normalize, merge, or rewrite any data. Your role is analysis and classification only.

PHASE 3 — CONSULT DATA QUALITY GLOSSARY
Open and consult the data quality issues glossary located at:

/home/jake/Developer/timeline/PROMPTS/DATA-QUALITY-PROBLEMS.md

Use existing issue names and definitions where applicable. If an observed issue does not fit an existing category, treat it as a new data quality pattern that should be added to the glossary.

PHASE 4 — APPEND TO THE GLOSSARY
Append entries to DATA-QUALITY-PROBLEMS.md for each distinct data quality issue identified. Do not modify or delete existing content.

Each appended entry must include:
- Issue name (existing or newly proposed)
- Description of the issue
- Structural characteristics (how it appears in CSV form)
- Concrete examples from the given CSV (titles, years, URL patterns)
- Why the issue is harmful (e.g. deduplication failure, automation breakage)
- Notes on detection signals or heuristics for identifying the issue later

Append new entries in a consistent, readable format suitable for long-term accumulation.

**YOU MUST ACTIVELY USE YOUR OWN JUDGMENT TO IDENTIFY AND DOCUMENT ANY DATA QUALITY ISSUES YOU OBSERVE, EVEN IF THEY ARE NOT EXPLICITLY LISTED ABOVE, WITH THE GOAL OF EXHAUSTIVELY CAPTURING ALL DISTINCT PATTERNS OF STRUCTURAL OR SEMANTIC PROBLEMS IN THE DATA.**
