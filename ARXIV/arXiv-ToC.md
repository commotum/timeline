## arXiv CSV Pipeline

### Step 1 — Destination: `arXiv-1.csv`
- Source: `CHUNKED/CHUNK-*.csv`
- Transform: move rows where `url` matches `arxiv.org/pdf/<id>.pdf`
- Notes: duplicates retained

### Step 2 — Destination: `arXiv-2.csv`
- Source: `arXiv-1.csv`
- Transform: de-duplicate by full row (`year,title,url`); add `duplicate_count`; clean URL field to keep only the URL

### Step 3 — Destination: `arXiv-3.csv`
- Source: `arXiv-2.csv`
- Transform: de-duplicate by first three columns only; sum `duplicate_count`; sort by title

### Step 4 — Destination: `arXiv-4.csv`
- Source: `arXiv-3.csv`
- Transform: move rows where `url` is duplicated across multiple entries; sort by URL
- Notes: moved out of `arXiv-3.csv`

### Step 5 — Destination: `arXiv-5.csv`
- Notes: present but not processed yet
