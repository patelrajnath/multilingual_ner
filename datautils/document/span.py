from datautils.document.doc import Doc


class Span(object):
    def __init__(self, doc: Doc, s, e, label):
        self.start_char = self.get_start_char(doc, s)
        self.end_char = self.get_end_char(doc, e)
        self.label_ = label

    @staticmethod
    def get_start_char(doc, start):
        return doc.get_token(start).idx

    @staticmethod
    def get_end_char(doc, end):
        token = doc.get_token(end)
        return token.idx + len(token.text)
