# import collections
#
# from transformers import BertTokenizer, BertModel, WordpieceTokenizer
# import torch
#
# from tokenizer.tokenizer_utils import load_vocab
# from tokenizer.word_piece import BasicTokenizer
#
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
# embeddings = model.get_input_embeddings()
# print(embeddings.weight.data.size())
# embeddings.weight.requires_grad = False
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state
# print(last_hidden_states.size())

# inputs = tokenizer.tokenize("Hello, my quantization are cute")
# inputs = ['[CLS]'] + inputs + ['[SEP]']
# print(tokenizer.convert_tokens_to_ids(inputs))
# print(inputs)
# print(tokenizer.save_vocabulary('.', 'codes'))
# print(tokenizer.get_vocab())


# text = "Hello, my quantization are cute"
# vocab = load_vocab('tokenizer/codes-vocab.txt')
# wordpiece_tokenizer = WordpieceTokenizer(vocab, unk_token='unk')
# basic_tokenizer = BasicTokenizer()
# do_basic_tokenize = True
# split_tokens = []
# if do_basic_tokenize:
#     for token in basic_tokenizer.tokenize(text):

        # If the token is part of the never_split set
        # if token in basic_tokenizer.never_split:
        #     split_tokens.append(token)
        # else:
        #     split_tokens += wordpiece_tokenizer.tokenize(token)
# else:
#     split_tokens = wordpiece_tokenizer.tokenize(text)
# print(split_tokens)


# Load model
import numpy
from transformers import *
import torch
model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# First example
batch_id = [[101, 1996, 3035, 2038, 2741, 1037, 1056, 28394, 2102, 2000, 1996, 3035, 2012, 17836, 4186, 2000, 8439, 2014, 3938, 2705, 5798, 102]]
batch_id = torch.tensor(batch_id)
max_seq_len = max(len(a) for a in batch_id)
pad_len = max_seq_len - len(batch_id[0])
attn_mask = torch.tensor([[1] * len(batch_id[0]) + [0] * pad_len])
with torch.no_grad():
    last_hidden_states = model(batch_id, attn_mask)[0].cpu().numpy()
print(last_hidden_states[0])

# Second example
batch_id = [[101, 1996, 3035, 2038, 2741, 1037, 1056, 28394, 2102, 2000, 1996, 3035, 2012, 17836, 4186, 2000, 8439, 2014, 3938, 2705, 5798, 102, 0, 0]]
batch_id = torch.tensor(batch_id)
print(len(batch_id[0]))
pad_len = 2
attn_mask = torch.tensor([[1] * max_seq_len + [0] * pad_len])

with torch.no_grad():
    last_hidden_states = model(batch_id, attn_mask)[0].cpu().numpy()
print(last_hidden_states[0][:max_seq_len])
padded = last_hidden_states[0][:max_seq_len]
padded = numpy.pad(padded, (0, pad_len))
print(padded.shape)
