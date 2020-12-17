import ast
import os


def prepare(data_dir, train_text, train_label,
            test_text=None, test_label=None,
            dev_text=None, dev_label=None,
            vocab=None, tag_set=None):
    train_path_text = os.path.join(data_dir, train_text)
    train_path_label = os.path.join(data_dir, train_label)
    test_path_text = os.path.join(data_dir, test_text)
    test_path_label = os.path.join(data_dir, test_label)
    vocab_path = os.path.join(data_dir, vocab)
    tag_path = os.path.join(data_dir, tag_set)

    vocab = {'UNK': 0, 'PAD': 1}
    num_specials_tokens = len(vocab)
    with open(vocab_path, encoding='utf8') as f:
        words = ast.literal_eval(f.read()).keys()
        for i, l in enumerate(words):
            vocab[l] = i + num_specials_tokens

    idx_to_word = {vocab[key]: key for key in vocab}

    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_idx = {START_TAG: 0, STOP_TAG: 1}
    num_specials_tags = len(tag_to_idx)
    with open(tag_path, encoding='utf8') as f:
        words = ast.literal_eval(f.read()).keys()
        for i, l in enumerate(words):
            tag_to_idx[l] = i + num_specials_tags

    idx_to_tag = {tag_to_idx[key]: key for key in tag_to_idx}

    train_sentences = []
    train_labels = []

    with open(train_path_text, encoding='utf8') as f:
        for sentence in f:
            # replace each token by its index if it is in vocab else use index of UNK
            s = [vocab[token] if token in vocab else vocab['UNK'] for token in sentence.strip().split()]
            train_sentences.append(s)

    with open(train_path_label, encoding='utf8') as f:
        for sentence in f:
            # replace each label by its index
            l = [tag_to_idx.get(label, tag_to_idx.get('O')) for label in sentence.strip().split()]
            train_labels.append(l)

    # Sort the data according to the length
    # sorted_idx = np.argsort([len(s) for s in train_sentences])
    # train_sentences = [train_sentences[id] for id in sorted_idx]
    # train_labels = [train_labels[id] for id in sorted_idx]

    test_sentences = []
    test_labels = []
    with open(test_path_text, encoding='utf8') as f:
        for sentence in f:
            # replace each token by its index if it is in vocab else use index of UNK
            s = [vocab[token] if token in vocab else vocab['UNK']
                 for token in sentence.strip().split()]
            test_sentences.append(s)

    with open(test_path_label, encoding='utf8') as f:
        for sentence in f:
            # replace each label by its index
            l = [tag_to_idx.get(label, tag_to_idx.get('O')) for label in sentence.strip().split()]
            test_labels.append(l)

    count = 1
    train_sentences_fixed = []
    train_labels_fixed = []
    for t, l in zip(train_sentences, train_labels):
        count += 1
        if len(t) != len(l):
            print(f'Error:{len(t)}, {len(l)}, {count}')
        else:
            train_sentences_fixed.append(t)
            train_labels_fixed.append(l)

    train_sentences = train_sentences_fixed
    train_labels = train_labels_fixed

    test_sentences_fixed = []
    test_labels_fixed = []
    for t, l in zip(test_sentences, test_labels):
        count += 1
        if len(t) != len(l):
            print(f'Error:{len(t)}, {len(l)}, {count}')
        else:
            test_sentences_fixed.append(t)
            test_labels_fixed.append(l)
    test_sentences = test_sentences_fixed
    test_labels = test_labels_fixed

    return idx_to_word, idx_to_tag, train_sentences, train_labels, test_sentences, test_labels
