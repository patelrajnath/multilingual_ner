import logging
import re
from abc import abstractmethod
from typing import List

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, GPT2Tokenizer
from transformers.tokenization_auto import XLNetTokenizer, T5Tokenizer

from models import tqdm

log = logging.getLogger("ner")


class TransformerWordEmbeddings(object):
    def __init__(
            self,
            model: str = "bert-base-uncased",
            layers: str = "-1,-2,-3,-4",
            pooling_operation: str = "first",
            batch_size: int = 1,
            use_scalar_mix: bool = False,
            fine_tune: bool = False,
            device: str = 'cpu',
            allow_long_sentences: bool = True,
            **kwargs
    ):
        """
        Bidirectional transformer embeddings of words from various transformer architectures.
        :param model: name of transformer model (see https://huggingface.co/transformers/pretrained_models.html for
        options)
        :param layers: string indicating which layers to take for embedding (-1 is topmost layer)
        :param pooling_operation: how to get from token piece embeddings to token embedding. Either take the first
        subtoken ('first'), the last subtoken ('last'), both first and last ('first_last') or a mean over all ('mean')
        :param batch_size: How many sentence to push through transformer at once. Set to 1 by default since transformer
        models tend to be huge.
        :param use_scalar_mix: If True, uses a scalar mix of layers as embedding
        :param fine_tune: If True, allows transformers to be fine-tuned during training
        """
        super().__init__()
        # temporary fix to disable tokenizer parallelism warning
        # (see https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # load tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        config = AutoConfig.from_pretrained(model, output_hidden_states=True, **kwargs)
        self.model = AutoModel.from_pretrained(model, config=config, **kwargs)

        self.allow_long_sentences = allow_long_sentences

        if allow_long_sentences:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = self.tokenizer.model_max_length // 2
        else:
            self.max_subtokens_sequence_length = self.tokenizer.model_max_length
            self.stride = 0

        # model name
        self.name = 'transformer-word-' + str(model)

        # when initializing, embeddings are in eval mode by default
        self.model.eval()
        self.model.to(device)

        # embedding parameters
        if layers == 'all':
            # send mini-token through to check how many layers the model has
            hidden_states = self.model(torch.tensor([1], device=device).unsqueeze(0))[-1]
            self.layer_indexes = [int(x) for x in range(len(hidden_states))]
        else:
            self.layer_indexes = [int(x) for x in layers.split(",")]
        # self.mix = ScalarMix(mixture_size=len(self.layer_indexes), trainable=False)
        self.pooling_operation = pooling_operation
        self.use_scalar_mix = use_scalar_mix
        self.fine_tune = fine_tune
        self.training = False
        self.device = device
        self.static_embeddings = not self.fine_tune
        self.batch_size = batch_size

        self.special_tokens = []
        # check if special tokens exist to circumvent error message
        if self.tokenizer._bos_token:
            self.special_tokens.append(self.tokenizer.bos_token)
        if self.tokenizer._cls_token:
            self.special_tokens.append(self.tokenizer.cls_token)

        # most models have an intial BOS token, except for XLNet, T5 and GPT2
        self.begin_offset = 1
        if type(self.tokenizer) == XLNetTokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == T5Tokenizer:
            self.begin_offset = 0
        if type(self.tokenizer) == GPT2Tokenizer:
            self.begin_offset = 0

    def encode(self, sentences: List[str]) -> List[torch.tensor]:
        """Add embeddings to all words in a list of sentences."""

        # split into micro batches of size self.batch_size before pushing through transformer
        sentence_batches = [sentences[i * self.batch_size:(i + 1) * self.batch_size]
                            for i in range((len(sentences) + self.batch_size - 1) // self.batch_size)]

        # embed each micro-batch
        sentences_encoded = []
        for batch in tqdm(sentence_batches, total=len(sentence_batches), leave=False, desc=f"Encoding {self.name}"):
            sentences_encoded.extend(self._add_embeddings_to_sentences(batch))
        return sentences_encoded

    @staticmethod
    def _remove_special_markup(text: str):
        # remove special markup
        text = re.sub('^Ġ', '', text)  # RoBERTa models
        text = re.sub('^##', '', text)  # BERT models
        text = re.sub('^▁', '', text)  # XLNet models
        text = re.sub('</w>$', '', text)  # XLM models
        return text

    def _get_processed_token_text(self, token: str) -> str:
        pieces = self.tokenizer.tokenize(token)
        token_text = ''
        for piece in pieces:
            token_text += self._remove_special_markup(piece)
        token_text = token_text.lower()
        return token_text

    def _add_embeddings_to_sentences(self, sentences: List[str]):
        """Match subtokenization to Flair tokenization and extract embeddings from transformers for each token."""

        # first, subtokenize each sentence and find out into how many subtokens each token was divided
        subtokenized_sentences = []
        subtokenized_sentences_token_lengths = []

        sentence_parts_lengths = []

        # TODO: keep for backwards compatibility, but remove in future
        # some pretrained models do not have this property, applying default settings now.
        # can be set manually after loading the model.
        if not hasattr(self, 'max_subtokens_sequence_length'):
            self.max_subtokens_sequence_length = None
            self.allow_long_sentences = False
            self.stride = 0

        non_empty_sentences = []
        empty_sentences = []

        for sentence in sentences:
            tokenized_string = sentence

            # method 1: subtokenize sentence
            # subtokenized_sentence = self.tokenizer.encode(tokenized_string, add_special_tokens=True)

            # method 2:
            # transformer specific tokenization
            subtokenized_sentence = self.tokenizer.tokenize(tokenized_string)
            if len(subtokenized_sentence) == 0:
                empty_sentences.append(sentence)
                continue
            else:
                non_empty_sentences.append(sentence)

            token_subtoken_lengths = self.reconstruct_tokens_from_subtokens(sentence, subtokenized_sentence)
            subtokenized_sentences_token_lengths.append(token_subtoken_lengths)

            subtoken_ids_sentence = self.tokenizer.convert_tokens_to_ids(subtokenized_sentence)

            nr_sentence_parts = 0

            while subtoken_ids_sentence:
                nr_sentence_parts += 1
                encoded_inputs = self.tokenizer.encode_plus(subtoken_ids_sentence,
                                                            max_length=self.max_subtokens_sequence_length,
                                                            stride=self.stride,
                                                            return_overflowing_tokens=self.allow_long_sentences,
                                                            truncation=True,
                                                            )

                subtoken_ids_split_sentence = encoded_inputs['input_ids']
                subtokenized_sentences.append(torch.tensor(subtoken_ids_split_sentence, dtype=torch.long))

                if 'overflowing_tokens' in encoded_inputs:
                    subtoken_ids_sentence = encoded_inputs['overflowing_tokens']
                else:
                    subtoken_ids_sentence = None

            sentence_parts_lengths.append(nr_sentence_parts)

        # empty sentences get zero embeddings
        # for sentence in empty_sentences:
        #     for token in sentence:
        #         token.set_embedding(self.name, torch.zeros(self.embedding_length))

        # only embed non-empty sentences and if there is at least one
        sentences = non_empty_sentences
        if len(sentences) == 0: return

        # find longest sentence in batch
        longest_sequence_in_batch: int = len(max(subtokenized_sentences, key=len))

        total_sentence_parts = sum(sentence_parts_lengths)
        # initialize batch tensors and mask
        input_ids = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros(
            [total_sentence_parts, longest_sequence_in_batch],
            dtype=torch.long,
            device=self.device,
        )
        for s_id, sentence in enumerate(subtokenized_sentences):
            sequence_length = len(sentence)
            input_ids[s_id][:sequence_length] = sentence
            mask[s_id][:sequence_length] = torch.ones(sequence_length)

        # put encoded batch through transformer model to get all hidden states of all encoder layers
        hidden_states = self.model(input_ids, attention_mask=mask)[-1]
        # make the tuple a tensor; makes working with it easier.
        hidden_states = torch.stack(hidden_states)

        sentence_idx_offset = 0

        # gradients are enabled if fine-tuning is enabled
        gradient_context = torch.enable_grad() if (self.fine_tune and self.training) else torch.no_grad()

        with gradient_context:

            # iterate over all subtokenized sentences
            batch_embeddings = []
            for sentence_idx, (sentence, subtoken_lengths, nr_sentence_parts) in enumerate(
                    zip(sentences, subtokenized_sentences_token_lengths, sentence_parts_lengths)):

                sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]

                for i in range(1, nr_sentence_parts):
                    sentence_idx_offset += 1
                    remainder_sentence_hidden_state = hidden_states[:, sentence_idx + sentence_idx_offset, ...]
                    # remove stride_size//2 at end of sentence_hidden_state, and half at beginning of remainder,
                    # in order to get some context into the embeddings of these words.
                    # also don't include the embedding of the extra [CLS] and [SEP] tokens.
                    sentence_hidden_state = torch.cat((sentence_hidden_state[:, :-1 - self.stride // 2, :],
                                                       remainder_sentence_hidden_state[:, 1 + self.stride // 2:, :]), 1)

                subword_start_idx = self.begin_offset

                # for each token, get embedding
                sentence_embeddings = []
                for token_idx, (token, number_of_subtokens) in enumerate(zip(sentence, subtoken_lengths)):
                    subtoken_embeddings: List[torch.FloatTensor] = []

                    # some tokens have no subtokens at all (if omitted by BERT tokenizer) so return zero vector
                    if number_of_subtokens == 0:
                        subtoken_embeddings.append(torch.zeros(self.embedding_length))
                        continue

                    subword_end_idx = subword_start_idx + number_of_subtokens

                    # get states from all selected layers, aggregate with pooling operation
                    for layer in self.layer_indexes:
                        current_embeddings = sentence_hidden_state[layer][subword_start_idx:subword_end_idx]

                        if self.pooling_operation == "first":
                            final_embedding: torch.FloatTensor = current_embeddings[0]

                        if self.pooling_operation == "last":
                            final_embedding: torch.FloatTensor = current_embeddings[-1]

                        if self.pooling_operation == "first_last":
                            final_embedding: torch.Tensor = torch.cat([current_embeddings[0], current_embeddings[-1]])

                        if self.pooling_operation == "mean":
                            all_embeddings: List[torch.FloatTensor] = [
                                embedding.unsqueeze(0) for embedding in current_embeddings
                            ]
                            final_embedding: torch.Tensor = torch.mean(torch.cat(all_embeddings, dim=0), dim=0)

                        subtoken_embeddings.append(final_embedding)

                    # use scalar mix of embeddings if so selected
                    if self.use_scalar_mix:
                        sm_embeddings = torch.mean(torch.stack(subtoken_embeddings, dim=1), dim=1)
                        # sm_embeddings = self.mix(subtoken_embeddings)

                        subtoken_embeddings = [sm_embeddings]

                    # set the extracted embedding for the token
                    # token.set_embedding(self.name, torch.cat(subtoken_embeddings))
                    sentence_embeddings.append(torch.cat(subtoken_embeddings))
                    subword_start_idx += number_of_subtokens
                embedding = torch.stack(sentence_embeddings)
                batch_embeddings.append(embedding)
            return batch_embeddings

    def reconstruct_tokens_from_subtokens(self, sentence, subtokens):
        tokens = sentence.split()
        word_iterator = iter(tokens)
        token = next(word_iterator)
        token_text = self._get_processed_token_text(token)
        token_subtoken_lengths = []
        reconstructed_token = ''
        subtoken_count = 0
        # iterate over subtokens and reconstruct tokens
        for subtoken_id, subtoken in enumerate(subtokens):

            # remove special markup
            subtoken = self._remove_special_markup(subtoken)

            # TODO check if this is necessary is this method is called before prepare_for_model
            # check if reconstructed token is special begin token ([CLS] or similar)
            if subtoken in self.special_tokens and subtoken_id == 0:
                continue

            # some BERT tokenizers somehow omit words - in such cases skip to next token
            if subtoken_count == 0 and not token_text.startswith(subtoken.lower()):

                while True:
                    token_subtoken_lengths.append(0)
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                    if token_text.startswith(subtoken.lower()): break

            subtoken_count += 1

            # append subtoken to reconstruct token
            reconstructed_token = reconstructed_token + subtoken

            # check if reconstructed token is the same as current token
            if reconstructed_token.lower() == token_text:

                # if so, add subtoken count
                token_subtoken_lengths.append(subtoken_count)

                # reset subtoken count and reconstructed token
                reconstructed_token = ''
                subtoken_count = 0

                # break from loop if all tokens are accounted for
                if len(token_subtoken_lengths) < len(tokens):
                    token = next(word_iterator)
                    token_text = self._get_processed_token_text(token)
                else:
                    break

        # if tokens are unaccounted for
        while len(token_subtoken_lengths) < len(tokens) and len(token) == 1:
            token_subtoken_lengths.append(0)
            if len(token_subtoken_lengths) == len(sentence): break
            token = next(word_iterator)

        # check if all tokens were matched to subtokens
        if token != tokens[-1]:
            log.error(f"Tokenization MISMATCH in sentence '{sentence}'")
            log.error(f"Last matched: '{token}'")
            log.error(f"Last sentence: '{sentence[-1]}'")
            log.error(f"subtokenized: '{subtokens}'")
        return token_subtoken_lengths

    def train(self, mode=True):
        # if fine-tuning is not enabled (i.e. a "feature-based approach" used), this
        # module should never be in training mode
        if not self.fine_tune:
            pass
        else:
            super().train(mode)

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""

        if not self.use_scalar_mix:
            length = len(self.layer_indexes) * self.model.config.hidden_size
        else:
            length = self.model.config.hidden_size

        if self.pooling_operation == 'first_last': length *= 2

        return length

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tokenizer"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        # reload tokenizer to get around serialization issues
        model_name = self.name.split('transformer-word-')[-1]
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            pass


class StackTransformerEmbeddings(object):
    def __init__(self, encoders: List[TransformerWordEmbeddings]):
        self.encoders = encoders
        self.embedding_length = 0

    def encode(self, segments):
        segments_encoded = [encoder.encode(segments) for encoder in self.encoders]
        segments_enc = []
        discarded = 0
        i = 0
        for emb in zip(*segments_encoded):
            i += 1
            try:
                emb_cat = torch.cat(emb, dim=-1)
                segments_enc.append(emb_cat)
            except:
                discarded += 1
                print("The index is:", i)
                exit()
        print(f'Number of samples discarded: {discarded}')

        # Get the first segments and its embedding dim
        self.embedding_length = segments_enc[0].shape[1]
        return segments_enc
