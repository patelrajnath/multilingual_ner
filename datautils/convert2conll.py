with open('../outputs/ubuntu_text_bert.txt') as ftext, \
        open('../outputs/ubuntu_label_bert.txt') as flabel, \
        open('../outputs/ubuntu_predict_bert.txt') as fpredict, \
        open('../outputs/ubuntu_conlleval.txt', 'w', encoding='utf-8') as outfile:
    for linet, linel, linep in zip(ftext, flabel, fpredict):
        wordst = linet.split()
        wordsl = linel.split()
        wordsp = linep.split()
        for wt, wl, wp in zip(wordst, wordsl, wordsp):
            wl = wl.replace('B_', '')
            wp = wp.replace('B_', '')
            line = ' '.join([wt, wl, wp]) + '\n'
            outfile.write(line)
        outfile.write('\n')
