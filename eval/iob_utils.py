from typing import Iterable, Tuple, Union, List

from document.doc import Doc
from document.span import Span


def biluo_from_offset(doc: Doc, entities: Iterable[Tuple[int, int, Union[str, int]]], missing: str = 'O'):
    """
    :param doc:
    :param entities:
    :param missing:
    :return:
    """
    starts = {token.idx: token.i for token in doc}
    ends = {token.idx + len(token.text): token.i for token in doc}
    tokens_in_ents = {}
    biluo = ["-" for _ in doc]
    # Handle entity cases
    for start_char, end_char, label in entities:
        if not label:
            for s in starts:  # account for many-to-one
                if start_char <= s < end_char:
                    biluo[starts[s]] = "O"
        else:
            for token_index in range(start_char, end_char):
                if token_index in tokens_in_ents.keys():
                    span1 = (
                        tokens_in_ents[token_index][0],
                        tokens_in_ents[token_index][1],
                        tokens_in_ents[token_index][2],
                    )
                    span2 = (start_char, end_char, label)
                    raise ValueError(f"Trying to set conflicting doc.ents: "
                                     f"'{span1}' and '{span2}'. A token can "
                                     f"only be part of one entity, so make sure "
                                     f"the entities you're setting don't overlap.")

                tokens_in_ents[token_index] = (start_char, end_char, label)
            start_token = starts.get(start_char)
            end_token = ends.get(end_char)
            # Only interested if the tokenization is correct
            if start_token is not None and end_token is not None:
                if start_token == end_token:
                    biluo[start_token] = f"U-{label}"
                else:
                    biluo[start_token] = f"B-{label}"
                    for i in range(start_token + 1, end_token):
                        biluo[i] = f"I-{label}"
                    biluo[end_token] = f"L-{label}"
    # Now distinguish the O cases from ones where we miss the tokenization
    entity_chars = set()
    for start_char, end_char, label in entities:
        for i in range(start_char, end_char):
            entity_chars.add(i)
    for token in doc:
        for i in range(token.idx, token.idx + len(token.text)):
            if i in entity_chars:
                break
        else:
            biluo[token.i] = missing
    if "-" in biluo and missing != "-":
        ent_str = str(entities)
        warning_text = text = doc.text[:50] + "..." if len(doc.text) > 50 else doc.text
        warning_ent = ent_str[:50] + "..." if len(ent_str) > 50 else ent_str
        print(f"Some entities could not be aligned in the text \"{warning_text}\" with "
              f"entities \"{warning_ent}\". Misaligned entities ('-') will be "
              f"ignored during training.")
    return biluo


def biluo_to_iob(tags: Iterable[str]) -> List[str]:
    out = []
    for tag in tags:
        if tag is None:
            out.append(tag)
        else:
            tag = tag.replace("U-", "B-", 1).replace("L-", "I-", 1)
            out.append(tag)
    return out


def biluo_tags_to_spans(doc: Doc, tags: Iterable[str]) -> List[Span]:
    token_offsets = tags_to_entities(tags)
    spans = []
    for label, start_idx, end_idx in token_offsets:
        span = Span(doc, start_idx, end_idx, label=label)
        spans.append(span)
    return spans


def offset_from_biluo(doc: Doc, tags: Iterable[str]):
    spans = biluo_tags_to_spans(doc, tags)
    return [(span.start_char, span.end_char, span.label_) for span in spans]


def tags_to_entities(tags: Iterable[str]) -> List[Tuple[str, int, int]]:
    """Note that the end index returned by this function is inclusive.
    To use it for Span creation, increment the end by 1."""
    entities = []
    start = None
    for i, tag in enumerate(tags):
        if tag is None or tag.startswith("-"):
            # TODO: We shouldn't be getting these malformed inputs. Fix this.
            if start is not None:
                start = None
            else:
                entities.append(("", i, i))
        elif tag.startswith("O"):
            pass
        elif tag.startswith("I"):
            if start is None:
                start = "I"
                tags = tags[: i + 1]
                raise ValueError(f"Invalid BILUO tag sequence: Got a tag starting with {start} "
                                 f"without a preceding 'B' (beginning of an entity). "
                                 f"Tag sequence:\n{tags}")
        elif tag.startswith("U"):
            entities.append((tag[2:], i, i))
        elif tag.startswith("B"):
            start = i
        elif tag.startswith("L"):
            if start is None:
                tags = tags[: i + 1]
                start = "L"
                raise ValueError(f"Invalid BILUO tag sequence: Got a tag starting with {start} "
                                 f"without a preceding 'B' (beginning of an entity). "
                                 f"Tag sequence:\n{tags}")
            entities.append((tag[2:], start, i))
            start = None
        else:
            raise ValueError(f"Invalid BILUO tag: '{tag}'.")
    return entities


if __name__ == '__main__':
    # input_text = 'Unable to register an account using a mobile number or email address'
    # tags = "O O O O U-hms_service O O B-user_info L-user_info O B-user_info L-user_info".split()
    input_text = "Is UNK UNK update UNK my UNK ?"
    tags = "O O O O O O U-request O".split()
    docs = Doc(input_text)
    ent = offset_from_biluo(docs, tags)
    print(ent)
    biluo = biluo_from_offset(docs, ent)
    print(input_text)
    print(tags)
    print(biluo)
