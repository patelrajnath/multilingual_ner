import os
import random
import traceback
import torch
from torch.serialization import default_restore_location
import logging
import numpy as np


def torch_persistent_save(*args, **kwargs):
    for i in range(3):
        try:
            return torch.save(*args, **kwargs)
        except Exception:
            if i == 2:
                logging.error(traceback.format_exc())


def save_state(filename, model, criterion, optimizer,
               num_updates, optim_history=None, extra_state=None):
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
    torch_persistent_save(state_dict, filename)


def load_model_state(filename, model, data_parallel=False):
    if not os.path.exists(filename):
        print("Starting training from scratch.")
        return 0

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
    return state['num_updates']


def set_seed(seed_value=1234):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)