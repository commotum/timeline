## arXiv CSV Contents

- arXiv-1.csv: All rows moved from `CHUNKED/CHUNK-*.csv` whose `url` matched `arxiv.org/pdf/<id>.pdf`; duplicates retained.
- arXiv-2.csv: De-duplicated `arXiv-1.csv` by full row (`year,title,url`), adds `duplicate_count`, and URL fields cleaned to keep only the URL.
- arXiv-3.csv: De-duplicated `arXiv-2.csv` by the first three columns only, with `duplicate_count` summed across duplicates; sorted by title.
- arXiv-4.csv: Rows from `arXiv-3.csv` where the `url` is duplicated across multiple entries; moved out of `arXiv-3.csv` and sorted by URL.
- arXiv-5.csv: Present but not processed yet (no filter/derivation recorded).
