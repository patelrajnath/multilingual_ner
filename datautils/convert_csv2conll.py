import sys
import pandas as pd

csv_file = sys.argv[1]
out = sys.argv[2]

data_df = pd.read_csv(csv_file, delimiter='\t', encoding='utf8')
doc_size = 1
with open(out, 'w', encoding='utf-8') as outfile:
    counter = 0
    doc_marker = '-DOCSTART- -X- -X- O'
    for linet, linel in zip(data_df.text.tolist(), data_df.labels.tolist()):
        if counter % doc_size == 0:
            outfile.write(doc_marker)
            outfile.write('\n\n')

        wordst = linet.split()
        wordsl = linel.split()
        for wt, wl in zip(wordst, wordsl):
            wl = wl.replace('B_O', 'O')
            wl = wl.replace('_', '-')
            wl = wl.replace('[SEP]', 'O')

            line = ' '.join([wt, wl]) + '\n'
            outfile.write(line)
        counter += 1
        outfile.write('\n')
