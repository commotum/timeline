

1. Run /home/jake/Developer/timeline/NEW/Scripts/download_pdfs.sh on /home/jake/Developer/timeline/NEW/NEW.csv saving the files to /home/jake/Developer/timeline/NEW/Downloads
2. Run /home/jake/Developer/timeline/NEW/Scripts/extract.py to get the OCR for the pdfs and afterwards move them back to /home/jake/Developer/timeline/NEW/Markdown
3. Run /home/jake/Developer/timeline/RUNNERS/initial-classification/script-initial-classification.py on /home/jake/Developer/timeline/NEW/Markdown to get the classes
4. Run /home/jake/Developer/timeline/NEW/Scripts/transfer.py to move the subfolders from /home/jake/Developer/timeline/NEW/Markdown to their appropriate folders within /home/jake/Developer/timeline/BIBLIOTHEQUE and update the ledger in /home/jake/Developer/timeline/BIBLIOTHEQUE/BIBLIOTHEQUE.csv 



[2026-02-03 11:43:35] [INFO] Renamed local files (0):
[2026-02-03 11:43:35] [INFO]   - none
[2026-02-03 11:43:35] [INFO] Downloaded files (3):
[2026-02-03 11:43:35] [INFO]   - RL Grokking Recipe- How Does RL Unlock and Transfer New Algorithms in LLMs-
[2026-02-03 11:43:35] [INFO]   - Improving MoE Compute Efficiency by Composing Weight and Data Sparsity
[2026-02-03 11:43:35] [INFO]   - Language Models are Unsupervised Multitask Learners
[2026-02-03 11:43:35] [INFO] Failed entries (0)
[2026-02-03 11:43:35] [INFO] Skipped entries (0)
[pipeline] download_pdfs: done elapsed=18.8s