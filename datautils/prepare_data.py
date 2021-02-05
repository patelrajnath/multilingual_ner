import logging
import os

from flair.data import Sentence

logger = logging.getLogger(__name__)


def prepare(options, word_to_idx, tag_to_idx):
    train_path_text = None
    train_path_label = None
    test_path_text = None
    test_path_label = None
    if hasattr(options, 'train_text'):
        train_path_text = os.path.join(options.data_dir, options.train_text)
        train_path_label = os.path.join(options.data_dir, options.train_label)
    if hasattr(options, 'test_text'):
        test_path_text = os.path.join(options.data_dir, options.test_text)
        test_path_label = os.path.join(options.data_dir, options.test_label)

    train_sentences = []
    train_labels = []
    if train_path_text and os.path.exists(train_path_text) and \
            train_path_label and os.path.exists(train_path_label):
        with open(train_path_text, encoding='utf8') as f:
            for sentence in f:
                # replace each token by its index if it is in vocab else use index of UNK
                s = [word_to_idx[token] if token in word_to_idx else word_to_idx['UNK'] for token in
                     sentence.strip().split()]
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
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
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
    if test_path_text and os.path.exists(test_path_text) and \
            test_path_label and os.path.exists(test_path_label):
        with open(test_path_text, encoding='utf8') as f:
            for sentence in f:
                # replace each token by its index if it is in vocab else use index of UNK
                s = [word_to_idx[token] if token in word_to_idx else word_to_idx['UNK']
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
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
            else:
                test_sentences_fixed.append(t)
                test_labels_fixed.append(l)
        test_sentences = test_sentences_fixed
        test_labels = test_labels_fixed

    return train_sentences, train_labels, test_sentences, test_labels


def prepare_flair(options, tag_to_idx):
    train_path_text = None
    train_path_label = None
    test_path_text = None
    test_path_label = None
    if hasattr(options, 'train_text'):
        train_path_text = os.path.join(options.data_dir, options.train_text)
        train_path_label = os.path.join(options.data_dir, options.train_label)
    if hasattr(options, 'test_text'):
        test_path_text = os.path.join(options.data_dir, options.test_text)
        test_path_label = os.path.join(options.data_dir, options.test_label)

    train_sentences = []
    train_labels = []
    if train_path_text and os.path.exists(train_path_text) and \
            train_path_label and os.path.exists(train_path_label):
        with open(train_path_text, encoding='utf8') as f:
            for sentence in f:
                train_sentences.append(Sentence(sentence, use_tokenizer=False))

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
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
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
    if test_path_text and os.path.exists(test_path_text) and \
            test_path_label and os.path.exists(test_path_label):
        with open(test_path_text, encoding='utf8') as f:
            for sentence in f:
                test_sentences.append(Sentence(sentence, use_tokenizer=False))
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
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
            else:
                test_sentences_fixed.append(t)
                test_labels_fixed.append(l)
        test_sentences = test_sentences_fixed
        test_labels = test_labels_fixed

    return train_sentences, train_labels, test_sentences, test_labels


def prepare_text(options, tag_to_idx):
    train_path_text = None
    train_path_label = None
    test_path_text = None
    test_path_label = None
    if hasattr(options, 'train_text'):
        train_path_text = os.path.join(options.data_dir, options.train_text)
        train_path_label = os.path.join(options.data_dir, options.train_label)
    if hasattr(options, 'test_text'):
        test_path_text = os.path.join(options.data_dir, options.test_text)
        test_path_label = os.path.join(options.data_dir, options.test_label)

    train_sentences = []
    train_labels = []
    if train_path_text and os.path.exists(train_path_text) and \
            train_path_label and os.path.exists(train_path_label):
        with open(train_path_text, encoding='utf8') as f:
            for sentence in f:
                train_sentences.append(sentence.strip())

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
            if len(t.split()) != len(l):
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
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
    if test_path_text and os.path.exists(test_path_text) and \
            test_path_label and os.path.exists(test_path_label):
        with open(test_path_text, encoding='utf8') as f:
            for sentence in f:
                test_sentences.append(sentence.strip())
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
            if len(t.split()) != len(l):
                print(t)
                print(l)
                logger.warning(f'WARNING: Token counts:{len(t)} and Tag counts:{len(l)} '
                               f'are different at line no. {count}, will be ignored in training.')
            else:
                test_sentences_fixed.append(t)
                test_labels_fixed.append(l)
        test_sentences = test_sentences_fixed
        test_labels = test_labels_fixed

    return train_sentences, train_labels, test_sentences, test_labels
