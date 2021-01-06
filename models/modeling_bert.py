from transformers import BertPreTrainedModel, BertModel


class BertTokenEmbedder(BertPreTrainedModel):
    def __init__(self, config, model_name, only_embedding=True, output_hidden_states=True):
        super(BertTokenEmbedder, self).__init__(config)

        self.only_embedding = only_embedding
        self.model_name = model_name
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=output_hidden_states)
        if self.only_embedding:
            self.model = self.model.get_input_embeddings()
            self.model.weight.requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
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
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
