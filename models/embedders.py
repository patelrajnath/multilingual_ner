import logging
import os
import pickle
from transformers import BertModel
import torch
import atexit

from models.constants import MODEL_CLASSES

glog = logging.getLogger(__name__)


class BERTEmbedder(torch.nn.Module):
    def __init__(self, model, config,
                 cache_dir='./',
                 encoder_id='bert_multilingual_embeddings',
                 caching=True):
        super(BERTEmbedder, self).__init__()
        self.caching = caching
        self.config = config
        self.model = model
        if self.config["mode"] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

        self._encodings_dict_path = os.path.join(
            cache_dir, encoder_id)
        try:
            # Create cache dir if doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
        except:
            pass
        if self.caching:
            self._encodings_dict = self._load_or_create_encodings_dict()

    def init_weights(self):
        if self.config["mode"] == "weighted":
            torch.nn.init.xavier_normal_(self.bert_gamma)
            torch.nn.init.xavier_normal_(self.bert_weights)

    @classmethod
    def create(
            cls, model_name='bert-base-multilingual-cased',
            device="cuda", mode="weighted",
            is_freeze=True):
        config = {
            "model_name": model_name,
            "device": device,
            "mode": mode,
            "is_freeze": is_freeze
        }
        model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        model.to(device)
        model.decode()
        self = cls(model, config)
        if is_freeze:
            self.freeze()
        return self

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def _load_or_create_encodings_dict(self):
        if os.path.exists(self._encodings_dict_path):
            with open(self._encodings_dict_path, "rb") as f:
                encodings_dict = pickle.load(f)
        else:
            encodings_dict = {}
        return encodings_dict

    def _save_encodings_dict(self):
        with open(self._encodings_dict_path, "wb") as f:
            pickle.dump(self._encodings_dict, f)

    def forward(self, batch):
        """
        batch has the following structure:
            data[0]: list, tokens ids
            data[1]: list, tokens mask
            data[2]: list, tokens type ids (for bert)
            data[3]: list, bert labels ids
        """
        if self.caching:
            sentences = batch[0]
            missing_sentences = []
            for sentence in sentences:
                sentence_key = " ".join([str(item) for item in sentence.tolist()])
                if sentence_key not in self._encodings_dict:
                    missing_sentences.append(sentence)
            if len(sentences) != len(missing_sentences):
                glog.info(f"{len(sentences) - len(missing_sentences)} cached "
                          f"sentences will not be encoded")
            if missing_sentences:
                print('Missing segements...')
                encoded_layers = self.model(
                    input_ids=batch[0],
                    token_type_ids=batch[2],
                    attention_mask=batch[1])
                encoded_layers = encoded_layers[-1]
                if self.config["mode"] == "weighted":
                    encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                    encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
                for sentence, encoding in zip(missing_sentences,
                                              encoded_layers):
                    sentence_key = " ".join([str(item) for item in sentence.tolist()])
                    self._encodings_dict[sentence_key] = encoding
                self._save_encodings_dict()

            encoded_layers = torch.stack([self._encodings_dict[" ".join([str(item) for item in sentence.tolist()])]
                                          for sentence in sentences])
            return encoded_layers
        else:
            encoded_layers = self.model(
                input_ids=batch[0],
                token_type_ids=batch[2],
                attention_mask=batch[1])
            encoded_layers = encoded_layers[-1]
            if self.config["mode"] == "weighted":
                encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
            return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False


class PretrainedEmbedder(torch.nn.Module):
    def __init__(self, args, device, cache_dir='./cache', encoder_id='cache_encodings'
                 ):
        super(PretrainedEmbedder, self).__init__()
        self.args = args

        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.model_type]
        self.config = config_class.from_pretrained(self.args.model_name)
        self.tokenizer = tokenizer_class.from_pretrained(self.args.model_name)
        self.model = model_class(self.config, self.args, self.tokenizer, device)

        self.mode = self.args.mode
        self.model.to(device)
        self.model.eval()
        self.device = device

        self._encodings_dict_path = os.path.join(cache_dir, encoder_id)
        try:
            # Create cache dir if doesn't exist
            os.makedirs(cache_dir, exist_ok=True)
        except:
            pass
        if self.args.cache_features:
            if self.args.in_memory_cache or self.args.reset_cache:
                self._encodings_dict = {}
            else:
                self._encodings_dict = self._load_or_create_encodings_dict()

        # Save the cached features
        if self.args.save_cache_features and not self.args.in_memory_cache:
            # At exit save cache dictionary
            atexit.register(self._save_encodings_dict)

        if self.args.freeze_bert_weights:
            self.freeze()

        if self.mode == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

    def init_weights(self):
        if self.mode == "weighted":
            torch.nn.init.xavier_normal_(self.bert_gamma)
            torch.nn.init.xavier_normal_(self.bert_weights)

    @classmethod
    def from_config(cls, config):
        return cls.create(**config)

    def _load_or_create_encodings_dict(self):
        if os.path.exists(self._encodings_dict_path):
            with open(self._encodings_dict_path, "rb") as f:
                encodings_dict = pickle.load(f)
        else:
            encodings_dict = {}
        return encodings_dict

    def _save_encodings_dict(self):
        with open(self._encodings_dict_path, "wb") as f:
            pickle.dump(self._encodings_dict, f)

    def forward(self, input_):
        """
        batch has the following structure:
            data[0]: list, tokens ids
            data[1]: list, tokens mask
            data[2]: list, tokens type ids (for bert)
            data[3]: list, bert labels ids
        """
        if self.device.type == 'cpu' and not self.args.only_embedding and self.args.cache_features:
            sentences = input_["input_ids"]
            attn_masks = input_["attention_mask"]
            token_type_ids = None
            if "token_type_ids" in input_:
                token_type_ids = input_["token_type_ids"]
            missing_sentences = []
            missing_attn_masks = []
            missing_token_type_ids = []

            if "token_type_ids" in input_:
                for sentence, attn_mask, token_type_id in  zip(sentences, attn_masks, token_type_ids):
                    sentence_len = int(torch.sum(attn_mask).item())
                    sentence_key = " ".join([str(item) for item in sentence.tolist()[:sentence_len]])
                    if sentence_key not in self._encodings_dict:
                        missing_sentences.append(sentence)
                        missing_attn_masks.append(attn_mask)
                        missing_token_type_ids.append(token_type_id)
            else:
                for sentence, attn_mask in zip(sentences, attn_masks):
                    sentence_len = int(torch.sum(attn_mask).item())
                    sentence_key = " ".join([str(item) for item in sentence.tolist()[:sentence_len]])
                    if sentence_key not in self._encodings_dict:
                        missing_sentences.append(sentence)
                        missing_attn_masks.append(attn_mask)

            if len(sentences) != len(missing_sentences):
                glog.info(f"{len(sentences) - len(missing_sentences)} cached "
                          f"sentences will not be encoded")
            if missing_sentences:
                if missing_token_type_ids:
                    missing_input_ = {"input_ids": torch.stack(missing_sentences),
                                      "attention_mask": torch.stack(missing_attn_masks),
                                      "token_type_ids": torch.stack(missing_token_type_ids)
                                      }
                else:
                    missing_input_ = {"input_ids": torch.stack(missing_sentences),
                                      "attention_mask": torch.stack(missing_attn_masks)}

                encoded_layers = self.model(**missing_input_)
                encoded_layers = torch.stack(encoded_layers[-1])
                encoded_layers = torch.sum(encoded_layers, dim=0)
                for sentence, attn_mask, encoding in zip(missing_sentences, missing_attn_masks, encoded_layers):
                    sentence_len = int(torch.sum(attn_mask).item())
                    sentence_key = " ".join([str(item) for item in sentence.tolist()[:sentence_len]])
                    self._encodings_dict[sentence_key] = encoding[:sentence_len]

            sentence_encoding = []
            bs, seq_len = sentences.size()
            for sentence, attn_mask in zip(sentences, attn_masks):
                sentence_len = int(torch.sum(attn_mask).item())
                encoding = self._encodings_dict[" ".join([str(item) for item in sentence.tolist()[:sentence_len]])]
                _, emb = encoding.size()
                target = torch.zeros(seq_len, emb)
                target[:sentence_len] = encoding
                sentence_encoding.append(target)
            encoded_layers = torch.stack(sentence_encoding)
            return encoded_layers
        else:
            encoded_layers = self.model(**input_)
            if self.args.only_embedding:
                return encoded_layers

            encoded_layers = encoded_layers[-1]
            if self.mode == "weighted":
                encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
            return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
