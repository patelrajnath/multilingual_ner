import json
import os
import random
import traceback
from types import SimpleNamespace

import torch
from torch.serialization import default_restore_location
import logging
import numpy as np

from models import build_model


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, model, criterion, optimizer,
               num_updates, optim_history=None, extra_state=None, args=None):
    if optim_history is None:
        optim_history = []
    if extra_state is None:
        extra_state = {}
    print("Saving checkpoint at-", filename)
    state_dict = {
        'model': model.state_dict(),
        'num_updates': num_updates,
        'optimizer_history': optim_history + [
            {
                'criterion_name': criterion.__class__.__name__,
                'optimizer_name': optimizer.__class__.__name__,
            }
        ],
        'extra_state': extra_state,
    }
    if args:
        basedir = os.path.dirname(filename)
        with open(os.path.join(basedir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, data_parallel=False):
    if not os.path.exists(filename):
        print("Starting training from scratch.")
        return 0

    def dict_to_sns(d):
        return SimpleNamespace(**d)

    basedir = os.path.dirname(filename)
    with open(os.path.join(basedir, 'config.json')) as f:
        args_dict = json.load(f, object_hook=dict_to_sns)

    model = build_model(args_dict)

    print("Loading model from checkpoints", filename)
    state = torch.load(filename, map_location=lambda s, l: default_restore_location(s, 'cpu'))

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    # create new OrderedDict that does not contain `module.`
    if data_parallel:
        for k, v in state['model'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state['model']
    # load model parameters
    try:
        model.load_state_dict(new_state_dict)
    except Exception:
        raise Exception('Cannot load model parameters from checkpoint, '
                        'please ensure that the architectures match')
    return model, args_dict


def set_seed(seed_value=1234):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)


def loss_fn(outputs, labels, mask):
    # the number of tokens is the sum of elements in mask
    num_labels = int(torch.sum(mask).item())

    # pick the values corresponding to labels and multiply by mask
    outputs = outputs[range(outputs.shape[0]), labels]*mask

    # cross entropy loss for all non 'PAD' tokens
    return -torch.sum(outputs)/num_labels


def get_attn_pad_mask(seq_q, seq_k, pad_id):
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    b_size, len_q = seq_q.size()
    b_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(pad_id).unsqueeze(1)  # b_size x 1 x len_k
    return pad_attn_mask.expand(b_size, len_q, len_k)  # b_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    assert seq.dim() == 2
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    if seq.is_cuda:
        subsequent_mask = subsequent_mask.cuda()

    return subsequent_mask