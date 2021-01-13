from torch import nn

from models.constants import MODEL_CLASSES


class TokenEncoders(nn.Module):
    def __init__(self, args, device):
        super(TokenEncoders, self).__init__()
        self.args = args
        self._device = device
        self.model_types = self.args.model_type
        self.model_names = self.args.model_name
        self.encoders = nn.ModuleList()
        for model_name, model_type in zip(self.model_types, self.model_names):
            config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
            config = config_class.from_pretrained(model_name)
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class(config, args, tokenizer, self._device)
            self.encoders.append(model)

    def forward(self, input_):
        for enc in self.encoders:
            tensor = enc(**input_)
            print(tensor)
            exit()
