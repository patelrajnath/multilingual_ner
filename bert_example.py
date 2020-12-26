import collections

from transformers import BertTokenizer, BertModel, WordpieceTokenizer
import torch

from tokenizer.tokenizer_utils import load_vocab
from tokenizer.word_piece import BasicTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

embeddings = model.get_input_embeddings()
print(embeddings.weight.data.size())
embeddings.weight.requires_grad = False
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.size())

inputs = tokenizer.tokenize("Hello, my quantization are cute")
inputs = ['[CLS]'] + inputs + ['[SEP]']
print(tokenizer.convert_tokens_to_ids(inputs))
print(inputs)
print(tokenizer.save_vocabulary('.', 'codes'))
# print(tokenizer.get_vocab())


text = "Hello, my quantization are cute"
vocab = load_vocab('tokenizer/codes-vocab.txt')
wordpiece_tokenizer = WordpieceTokenizer(vocab, unk_token='unk')
basic_tokenizer = BasicTokenizer()
do_basic_tokenize = True
split_tokens = []
if do_basic_tokenize:
    for token in basic_tokenizer.tokenize(text):

        # If the token is part of the never_split set
        if token in basic_tokenizer.never_split:
            split_tokens.append(token)
        else:
            split_tokens += wordpiece_tokenizer.tokenize(token)
else:
    split_tokens = wordpiece_tokenizer.tokenize(text)
print(split_tokens)

