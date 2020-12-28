import time
import numpy as np

from bert_serving.client import BertClient
bc = BertClient(ip='185.190.206.134')
start = time.time()
# with open('data/nlu/nlu_train_text.txt') as fin:
# with open('data/wallet/wallet_train_text.txt', encoding='utf8') as fin:
# with open('data/ubuntu/ubuntu_train_text.txt', encoding='utf8') as fin:
with open('data/alliance/alliance_train_text.txt', encoding='utf8') as fin:
    lines = fin.readlines()
    sorted_idx = np.argsort([len(s) for s in lines])
    lines = [lines[id] for id in sorted_idx]
    encoded = bc.encode(lines)
    print(time.time() - start)
    print(encoded.shape)
    # s = 0
    # batch_size = 32
    # e = s + batch_size
    # while e <= len(lines):
    #     batch = lines[s:e]
    #     s += batch_size
    #     e += batch_size
    #     encoded = bc.encode(batch, blocking=False)
    #     vecs = bc.fetch_all()
    #     print(time.time()-start)
    #     print(vecs)
