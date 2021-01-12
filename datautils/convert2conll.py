import sys

prefix = sys.argv[1]
with open(f'outputs/{prefix}_text_bert.txt') as ftext, \
        open(f'outputs/{prefix}_label_bert.txt') as flabel, \
        open(f'outputs/{prefix}_predict_bert.txt') as fpredict, \
        open(f'outputs/{prefix}_conlleval.txt', 'w', encoding='utf-8') as outfile:
    for linet, linel, linep in zip(ftext, flabel, fpredict):
        wordst = linet.split()
        wordsl = linel.split()
        wordsp = linep.split()
        for wt, wl, wp in zip(wordst, wordsl, wordsp):
            wl = wl.replace('B_O', 'O')
            wp = wp.replace('B_O', 'O')
            wl = wl.replace('_', '-')
            wp = wp.replace('_', '-')
            line = ' '.join([wt, wl, wp]) + '\n'
            outfile.write(line)
        outfile.write('\n')
