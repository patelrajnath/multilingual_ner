from datautils.document.token import Token


class Doc(object):
    def __init__(self, sentence: str):
        self.text = sentence
        self.tokens = self.get_tokens(sentence)

    @staticmethod
    def get_tokens(sentence: str):
        words = sentence.split()
        tokens = []
        start = 0
        for i, word in enumerate(words):
            tokens.append(Token(word, start, i))
            start += len(word) + 1 # +1 for white space
        return tokens

    def __iter__(self):
        return iter(self.tokens)

    def __str__(self):
        return self.text

    def get_token(self, idx):
        return self.tokens[idx]
