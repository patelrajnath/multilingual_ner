import logging
import os
import pickle

import numpy
from pytorch_pretrained_bert import BertModel
import torch
from torch import from_numpy
glog = logging.getLogger(__name__)


class BERTEmbedder(torch.nn.Module):
    def __init__(self, model, config, cache_dir='./', encoder_id='bert_multilingual_embeddings'):
        super(BERTEmbedder, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.config = config
        self.model = model
        if self.config["mode"] == "weighted":
            self.bert_weights = torch.nn.Parameter(torch.FloatTensor(12, 1))
            self.bert_gamma = torch.nn.Parameter(torch.FloatTensor(1, 1))
        self.init_weights()

        self._encodings_dict_path = os.path.join(
            cache_dir, encoder_id)
        if not self.use_cuda:
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
        model = BertModel.from_pretrained(model_name)
        model.to(device)
        model.train()
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
        if not self.use_cuda:
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
                encoded_layers, _ = self.model(
                    input_ids=batch[0],
                    token_type_ids=batch[2],
                    attention_mask=batch[1],
                    output_all_encoded_layers=self.config["mode"] == "weighted")
                if self.config["mode"] == "weighted":
                    encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                    encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
                for sentence, encoding in zip(missing_sentences,
                                              encoded_layers):
                    sentence_key = " ".join([str(item) for item in sentence.tolist()])
                    self._encodings_dict[sentence_key] = encoding.cpu()
                self._save_encodings_dict()

            encoded_layers = torch.stack([self._encodings_dict[" ".join([str(item) for item in sentence.tolist()])]
                                          for sentence in sentences]).to(self.device)
            return encoded_layers
        else:
            encoded_layers, _ = self.model(
                input_ids=batch[0],
                token_type_ids=batch[2],
                attention_mask=batch[1],
                output_all_encoded_layers=self.config["mode"] == "weighted")
            if self.config["mode"] == "weighted":
                encoded_layers = torch.stack([a * b for a, b in zip(encoded_layers, self.bert_weights)])
                encoded_layers = self.bert_gamma * torch.sum(encoded_layers, dim=0)
            return encoded_layers

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
