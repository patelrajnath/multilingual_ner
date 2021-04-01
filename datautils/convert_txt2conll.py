import sys

text = sys.argv[1]
labels = sys.argv[2]
out = sys.argv[3]

with open(text) as ftext, \
        open(labels) as flabel, \
        open(out, 'w', encoding='utf-8') as outfile:

    for linet, linel in zip(ftext, flabel):
        wordst = linet.split()
        wordsl = linel.split()
        for wt, wl in zip(wordst, wordsl):
            wl = wl.replace('B_O', 'O')
            wl = wl.replace('_', '-')
            wl = wl.replace('[SEP]', 'O')

            line = ' '.join([wt, wl]) + '\n'
            outfile.write(line)
        outfile.write('\n')
