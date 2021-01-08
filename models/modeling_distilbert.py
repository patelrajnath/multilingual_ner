import os
import tempfile
from pathlib import Path

import torch
import numpy as np
from onnxruntime.quantization import quantize_dynamic
from transformers import DistilBertModel, DistilBertPreTrainedModel
from transformers.convert_graph_to_onnx import convert, generate_identified_filename
from onnxruntime import InferenceSession, SessionOptions, ExecutionMode


class DistilBertTokenEmbedder(DistilBertPreTrainedModel):
    def __init__(self, config, options, tokenizer, device, output_hidden_states=True):
        super(DistilBertTokenEmbedder, self).__init__(config)

        self.config = config
        self.options = options
        self.tokenizer = tokenizer
        self._device = device
        self.only_embedding = self.options.only_embedding
        self.model = DistilBertModel.from_pretrained(self.options.model_name, output_hidden_states=output_hidden_states)

        if self.options.onnx:
            self.onnx_model = self.convert_to_onnx()

        if self.only_embedding:
            self.model = self.model.get_input_embeddings()
            self.model.weight.requires_grad = False

    def save_model(self, output_dir=None, model=None):
        if not output_dir:
            output_dir = self.options.output_dir
        os.makedirs(output_dir, exist_ok=True)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(output_dir)

    def convert_to_onnx(self, onnx_output_dir=None, set_onnx_arg=True):
        """Convert the model to ONNX format and save to output_dir
        Args:
            onnx_output_dir (str, optional): If specified, ONNX model will be saved to output_dir (else args.output_dir will be used). Defaults to None.
            set_onnx_arg (bool, optional): Updates the model args to set onnx=True. Defaults to True.
        """  # noqa
        if not onnx_output_dir:
            onnx_output_dir = os.path.join(self.options.output_dir, self.options.model_type,
                                           self.options.model_name, "onnx")
        os.makedirs(onnx_output_dir, exist_ok=True)

        if not os.listdir(onnx_output_dir):
            onnx_model_name = os.path.join(onnx_output_dir, "onnx_model.onnx")
            with tempfile.TemporaryDirectory() as temp_dir:
                basedir = os.path.basename(temp_dir)
                temp_dir = os.path.join(self.options.output_dir, basedir)
                self.save_model(output_dir=temp_dir, model=self.model)

                convert(
                    framework="pt",
                    model=temp_dir,
                    tokenizer=self.tokenizer,
                    output=Path(onnx_model_name),
                    pipeline_name="ner",
                    opset=11,
                )
            self.tokenizer.save_pretrained(onnx_output_dir)
            self.config.save_pretrained(onnx_output_dir)

        onnx_options = SessionOptions()
        use_cuda = True if torch.cuda.is_available() and self._device != 'cpu' else False
        onnx_execution_provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
        onnx_options.intra_op_num_threads = 1
        onnx_options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        onnx_model_path = os.path.join(onnx_output_dir, "onnx_model.onnx")
        if self.options.dynamic_quantize:
            # Append "-quantized" at the end of the model's name
            quantized_model_path = generate_identified_filename(Path(onnx_model_path), "-quantized")
            quantize_dynamic(Path(onnx_model_path), quantized_model_path)
            onnx_model_path = quantized_model_path.as_posix()

        return InferenceSession(onnx_model_path, onnx_options,
                                providers=[onnx_execution_provider])

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

        # Use the the onnx model as encoder
        if self.options.onnx:
            inputs_onnx = {"input_ids": input_ids, "attention_mask": attention_mask}
            tokens = {name: np.atleast_2d(value.cpu()) for name, value in inputs_onnx.items()}
            out = self.onnx_model.run(None, tokens)
            out = out[1:] # the first vector is CLS output
            return [[torch.from_numpy(a).to(self._device) for a in out]]

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
