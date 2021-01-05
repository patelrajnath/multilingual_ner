import logging
import os
import pickle

from transformers import BertModel
import torch

from models.constants import MODEL_CLASSES

glog = logging.getLogger(__name__)


class BERTEmbedder(torch.nn.Module):
    def __init__(self, model, config,
                 cache_dir='./',
                 encoder_id='bert_multilingual_embeddings',
                 caching=False):
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


class TokenEmbedder(torch.nn.Module):
    def __init__(self, model_class, model_name,
                 only_embedding=True):
        super(TokenEmbedder, self).__init__()
        self.only_embedding = only_embedding
        if self.only_embedding:
            bert_model = model_class.from_pretrained(model_name)
            self.model = bert_model.get_input_embeddings()
            self.model.weight.requires_grad = False
            bert_model = None
        else:
            self.model = model_class.from_pretrained(model_name, output_hidden_states=True)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Use only the embedding layer
        if self.only_embedding:
            return self.model(input_ids)

        # Use the the model as encoder
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class PretrainedEmbedder(torch.nn.Module):
    def __init__(self, model_type, model_name,
                 device="cuda",
                 mode="weighted",
                 is_freeze=True,
                 only_embedding=True,
                 cache_dir='./',
                 encoder_id='bert_multilingual_embeddings',
                 caching=True):
        super(PretrainedEmbedder, self).__init__()

        config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.only_embedding = only_embedding
        self.config = config_class.from_pretrained(model_name)

        self.model = TokenEmbedder(model_class, model_name, only_embedding)
        self.mode = mode
        self.model.to(device)
        self.model.eval()

        self.caching = caching
        self._encodings_dict_path = os.path.join(cache_dir, encoder_id)
        if self.caching:
            self._encodings_dict = self._load_or_create_encodings_dict()

        if is_freeze:
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
        if self.caching:
            sentences = input_["input_ids"]
            missing_sentences = []
            for sentence in sentences:
                sentence_key = " ".join([str(item) for item in sentence.tolist()])
                if sentence_key not in self._encodings_dict:
                    missing_sentences.append(sentence)
            if len(sentences) != len(missing_sentences):
                glog.info(f"{len(sentences) - len(missing_sentences)} cached "
                          f"sentences will not be encoded")
            if missing_sentences:
                encoded_layers = self.model(**input_)
                encoded_layers = encoded_layers[-1]
                if self.mode == "weighted":
                    encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                    encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
                for sentence, encoding in zip(missing_sentences,
                                              encoded_layers):
                    sentence_key = " ".join([str(item) for item in sentence.tolist()])
                    self._encodings_dict[sentence_key] = encoding
                # self._save_encodings_dict()

            encoded_layers = torch.stack([self._encodings_dict[" ".join([str(item) for item in sentence.tolist()])]
                                          for sentence in sentences])
            return encoded_layers
        else:
            encoded_layers = self.model(**input_)
            if self.only_embedding:
                return encoded_layers

            encoded_layers = encoded_layers[-1]
            if self.mode == "weighted":
                encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
            return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
