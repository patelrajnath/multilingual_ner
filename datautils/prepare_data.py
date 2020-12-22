import ast
import logging
import os

logger = logging.getLogger(__name__)


def prepare(options):
    train_path_text = os.path.join(options.data_dir, options.train_text)
    train_path_label = os.path.join(options.data_dir, options.train_label)
    test_path_text = os.path.join(options.data_dir, options.test_text)
    test_path_label = os.path.join(options.data_dir, options.test_label)
    vocab_path = os.path.join(options.data_dir, options.vocab)
    tag_path = os.path.join(options.data_dir, options.tag_set)

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
    if os.path.exists(train_path_text) and os.path.exists(train_path_label):
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

        count = 1
        train_sentences_fixed = []
        train_labels_fixed = []
        for t, l in zip(train_sentences, train_labels):
            count += 1
            if len(t) != len(l):
                logger.warning(f'WARNING: Token and tags length is different: {len(t)}, {len(l)}, {count}')
            else:
                train_sentences_fixed.append(t)
                train_labels_fixed.append(l)

        train_sentences = train_sentences_fixed
        train_labels = train_labels_fixed

        # Sort the data according to the length
        # sorted_idx = np.argsort([len(s) for s in train_sentences])
        # train_sentences = [train_sentences[id] for id in sorted_idx]
        # train_labels = [train_labels[id] for id in sorted_idx]

    test_sentences = []
    test_labels = []
    if os.path.exists(test_path_text) and os.path.exists(test_path_label):
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

        test_sentences_fixed = []
        test_labels_fixed = []
        count = 0
        for t, l in zip(test_sentences, test_labels):
            count += 1
            if len(t) != len(l):
                logger.warning(f'WARNING: Token and tags length is different: {len(t)}, {len(l)}, {count}')
            else:
                test_sentences_fixed.append(t)
                test_labels_fixed.append(l)
        test_sentences = test_sentences_fixed
        test_labels = test_labels_fixed

    return idx_to_word, idx_to_tag, train_sentences, train_labels, test_sentences, test_labels
