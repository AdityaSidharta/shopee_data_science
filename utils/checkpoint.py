import datetime
import os

import torch

from utils.envs import model_cp_path
from callbacks.callback import CallBacks


def save_checkpoint(model, optimizer, fname=None):
    if fname is None:
        fname = str(datetime.datetime.now())
    model_filename = "{}_model.pth".format(fname)
    optim_filename = "{}_optim.pth".format(fname)
    model_filepath = os.path.join(model_cp_path, model_filename)
    optim_filepath = os.path.join(model_cp_path, optim_filename)
    torch.save(model.state_dict(), model_filepath)
    torch.save(optimizer.state_dict(), optim_filepath)


def load_cp_model(model, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)


def load_cp_optim(optimizer, checkpoint_path):
    state_dict = torch.load(checkpoint_path)
    optimizer.load_state_dict(state_dict)


class CheckpointSaver(CallBacks):
    def __init__(self, model_fn, is_epoch_cp=True):
        self.model_fn = model_fn
        self.is_epoch_cp = is_epoch_cp

    def on_epoch_end(self, epoch_idx, model, optimizer, val_loss=None, val_metric=None):
        model_filename = "{}_{}".format(self.model_fn, epoch_idx)
        save_checkpoint(model, optimizer, model_filename)

    def on_train_end(self, model, optimizer):
        model_filename = "{}_final", format(self.model_fn)
        save_checkpoint(model, optimizer, model_filename)
