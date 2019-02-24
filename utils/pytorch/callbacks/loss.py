import numpy as np
import seaborn as sns

from utils.logger import logger
from utils.pytorch.callbacks import CallBacks


class LossRecorder(CallBacks):
    def __init__(self, n_epoch, n_batch_per_epoch, loss_momentum=0.8, is_val=True):
        self.batch_list = []
        self.smooth_batch_list = []
        self.epoch_train_list = []
        self.epoch_val_list = []

        self.m_batch_list = []
        self.m_smooth_batch_list = []
        self.m_epoch_train_list = []
        self.m_epoch_val_list = []

        self.is_val = is_val
        self.smooth_loss = 0
        self.smooth_metric = 0

        self.n_epoch = n_epoch
        self.n_batch_per_epoch = n_batch_per_epoch
        self.momentum = loss_momentum

    def calc_loss(self, new_loss):
        n_loss = len(self.batch_list)
        mom_loss = self.momentum * self.smooth_loss + (1 - self.momentum) * new_loss
        smooth_loss = mom_loss / float(1 - self.momentum ** n_loss)
        return smooth_loss

    def calc_metric(self, new_metric):
        n_metric = len(self.m_batch_list)
        mom_metric = (
            self.momentum * self.smooth_metric + (1 - self.momentum) * new_metric
        )
        smooth_metric = mom_metric / float(1 - self.momentum ** n_metric)
        return smooth_metric

    def record_train_loss(self, new_loss):
        self.smooth_loss = self.calc_loss(new_loss)
        self.batch_list.append(new_loss)
        self.smooth_batch_list.append(self.smooth_loss)

    def record_train_metric(self, new_metric):
        self.smooth_metric = self.calc_metric(new_metric)
        self.m_batch_list.append(new_metric)
        self.smooth_batch_list.append(self.smooth_metric)

    def plot_batch_loss(self, smooth=False):
        n_batches = len(self.batch_list)
        if smooth:
            sns.lineplot(x=range(n_batches), y=self.smooth_batch_list)
        else:
            sns.lineplot(x=range(n_batches), y=self.batch_list)

    def plot_batch_metric(self, smooth=False):
        n_batches = len(self.m_batch_list)
        if smooth:
            sns.lineplot(x=range(n_batches), y=self.m_smooth_batch_list)
        else:
            sns.lineplot(x=range(n_batches), y=self.m_batch_list)

    def plot_epoch_loss(self, val=False):
        n_epoch = len(self.epoch_train_list)
        if val:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_list)
            sns.lineplot(x=range(n_epoch), y=self.epoch_val_list)
        else:
            sns.lineplot(x=range(n_epoch), y=self.epoch_train_list)

    def plot_epoch_metric(self, val=False):
        n_epoch = len(self.m_epoch_train_list)
        if val:
            sns.lineplot(x=range(n_epoch), y=self.m_epoch_train_list)
            sns.lineplot(x=range(n_epoch), y=self.m_epoch_val_list)
        else:
            sns.lineplot(x=range(n_epoch), y=self.m_epoch_train_list)

    def get_epoch_train_mean(self):
        epoch_loss = np.mean(self.batch_list[-self.n_batch_per_epoch :])
        return epoch_loss

    def get_m_epoch_train_mean(self):
        epoch_metric = np.mean(self.m_batch_list[-self.n_batch_per_epoch :])
        return epoch_metric

    def on_train_end(self, model, optimizer):
        logger.info("Plotting Batch Loss")
        self.plot_batch_loss(smooth=True)
        logger.info("Plotting Batch Metric")
        self.plot_batch_metric(smooth=True)
        logger.info("Plotting Epoch Loss")
        self.plot_epoch_loss(val=self.is_val)
        logger.info("Plotting Epoch Metric")
        self.plot_epoch_metric(val=self.is_val)

    def on_epoch_end(self, epoch_idx, model, optimizer, val_loss=None, val_metric=None):
        epoch_loss = self.get_epoch_train_mean()
        epoch_metric = self.get_m_epoch_train_mean()
        self.epoch_train_list.append(epoch_loss)
        self.m_epoch_train_list.append(epoch_metric)

        if self.is_val:
            assert (val_loss is not None) and (val_metric is not None)
            self.epoch_val_list.append(val_loss)
            self.m_epoch_val_list.append(val_metric)

    def on_batch_end(self, batch_idx, model, optimizer, batch_loss, batch_metric):
        self.record_train_loss(batch_loss)
        self.record_train_metric(batch_metric)
