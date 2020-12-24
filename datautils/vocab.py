import ast


def load_vocabs(vocab_path, tag_path = None):
    word_to_idx = {'UNK': 0, 'PAD': 1}
    num_specials_tokens = len(word_to_idx)
    with open(vocab_path, encoding='utf8') as f:
        words = ast.literal_eval(f.read()).keys()
        for i, l in enumerate(words):
            word_to_idx[l] = i + num_specials_tokens

    idx_to_word = {word_to_idx[key]: key for key in word_to_idx}
    idx_to_tag = None
    tag_to_idx = None
    if tag_path:
        START_TAG = "<START>"
        STOP_TAG = "<STOP>"
        tag_to_idx = {START_TAG: 0, STOP_TAG: 1}
        num_specials_tags = len(tag_to_idx)
        with open(tag_path, encoding='utf8') as f:
            words = ast.literal_eval(f.read()).keys()
            for i, l in enumerate(words):
                tag_to_idx[l] = i + num_specials_tags

        idx_to_tag = {tag_to_idx[key]: key for key in tag_to_idx}

    return word_to_idx, idx_to_word, tag_to_idx, idx_to_tag
