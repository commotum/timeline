## arXiv CSV Pipeline

### Step 1
- Source: `CHUNKED/CHUNK-*.csv`
- Destination: `arXiv-1.csv`
- Transforms:
  - move rows where `url` matches `arxiv.org/pdf/<id>.pdf`
- Notes: duplicates retained

### Step 2
- Source: `arXiv-1.csv`
- Destination: `arXiv-2.csv`
- Transforms:
  - de-duplicate by full row (`year,title,url`)
  - add `duplicate_count`
  - clean URL field to keep only the URL

### Step 3
- Source: `arXiv-2.csv`
- Destination: `arXiv-3.csv`
- Transforms:
  - de-duplicate by first three columns only
  - sum `duplicate_count`
  - sort by title

### Step 4
- Source: `arXiv-3.csv`
- Destination: `arXiv-4.csv`
- Transforms:
  - move rows where `url` is duplicated across multiple entries
  - sort by URL
- Notes: moved out of `arXiv-3.csv`

### Step 5
- Source: `arXiv-4.csv`
- Destination: `arXiv-5.csv`
- Transforms:
  - move rows where the `(year, url)` pair is duplicated
- Notes: duplicates retained in `arXiv-5.csv`; remaining rows in `arXiv-4.csv` have unique `(year, url)` pairs

### Step 6
- Source: `arXiv-5.csv`
- Destination: `arXiv-5.csv`
- Transforms:
  - normalize titles in place to the most complete/accurate title for each duplicate `(year, url)` group
