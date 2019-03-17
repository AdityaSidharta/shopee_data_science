import copy

import seaborn as sns
from torch import optim as optim

from utils.pytorch.callbacks.optim import LR_Finder
from utils.pytorch.callbacks import LossRecorder
from model.image.pytorch.train import fit_model_full
from utils.pytorch.utils.common import get_batch_info
from utils.logger import logger


def lr_find(model, dataloader, criterion, loss_fn, metric_fn, min_lr=1e-8, max_lr=10.0):
    clone_model = copy.deepcopy(model)
    optimizer = optim.SGD(clone_model.parameters(), lr=min_lr)
    n_epoch = 1
    n_obs, batch_size, batch_per_epoch = get_batch_info(dataloader)
    lr_finder = LR_Finder(n_epoch, batch_per_epoch, min_lr, max_lr)
    loss_recorder = LossRecorder(n_epoch, batch_per_epoch, is_val=False)
    model, callbacks = fit_model_full(
        model=clone_model,
        n_epoch=n_epoch,
        dev_dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        loss_fn=loss_fn,
        metric_fn=metric_fn,
        callbacks=[lr_finder, loss_recorder],
        val_dataloader=None,
    )
    train_loss = loss_recorder.smooth_batch_list
    while train_loss[-1] > (train_loss[-2] * 2.0):
        logger.info("removing last train_loss...")
        train_loss.pop()
    sns.lineplot(x=lr_finder.lr_schedule, y=loss_recorder.smooth_batch_list)
