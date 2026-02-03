

1. Run /home/jake/Developer/timeline/NEW/Scripts/download_pdfs.sh on /home/jake/Developer/timeline/NEW/NEW.csv saving the files to /home/jake/Developer/timeline/NEW/Downloads
2. Run /home/jake/Developer/timeline/NEW/Scripts/extract.py to get the OCR for the pdfs and afterwards move them back to /home/jake/Developer/timeline/NEW/Markdown
3. Run /home/jake/Developer/timeline/RUNNERS/initial-classification/script-initial-classification.py on /home/jake/Developer/timeline/NEW/Markdown to get the classes
4. Run /home/jake/Developer/timeline/NEW/Scripts/transfer.py to move the subfolders from /home/jake/Developer/timeline/NEW/Markdown to their appropriate folders within /home/jake/Developer/timeline/BIBLIOTHEQUE and update the ledger in /home/jake/Developer/timeline/BIBLIOTHEQUE/BIBLIOTHEQUE.csv 