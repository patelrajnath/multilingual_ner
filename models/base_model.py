from torch import nn


class BaseModel(nn.Module):
    """Base class for models."""
    def __init__(self):
        super(BaseModel, self).__init__()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def _get_inputs_dict(self, batch):
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
        }
        # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
        if self.args.model_type in ["bert", "xlnet", "albert", "layoutlm"]:
            inputs["token_type_ids"] = batch[2]

        if self.args.model_type == "layoutlm":
            inputs["bbox"] = batch[4]

        return inputs
