import numpy as np
import seaborn as sns

from utils.pytorch.callbacks import CallBacks


class OptimCallBacks(CallBacks):
    @staticmethod
    def set_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def set_mom(optimizer, momentum):
        if "betas" in optimizer.param_groups[0]:
            for param_group in optimizer.param_groups:
                param_group["betas"] = (momentum, param_group["betas"][1])
        else:
            for param_group in optimizer.param_groups:
                param_group["momentum"] = momentum

    def get_lr(self):
        pass


class LR_Finder(OptimCallBacks):
    def __init__(self, n_epoch, n_batch_per_epoch, min_lr, max_lr, linear=False):
        self.n_epoch = n_epoch
        self.n_batch_per_epoch = n_batch_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.linear = linear

        self.total_batch = self.n_epoch * self.n_batch_per_epoch
        self.lr_schedule = self.get_lr()

    def get_lr(self):
        if self.linear:
            return np.linspace(self.min_lr, self.max_lr, self.total_batch).tolist()
        else:
            return np.logspace(
                np.log10(self.min_lr), np.log10(self.max_lr), self.total_batch
            ).tolist()

    def on_batch_begin(self, batch_idx, model, optimizer):
        lr = self.lr_schedule[batch_idx]
        self.set_lr(self.optimizer, lr)


class CLR(OptimCallBacks):
    def __init__(
        self,
        n_epoch,
        n_batch_per_epoch,
        min_lr,
        max_lr,
        min_mom,
        max_mom,
        epoch_per_cycle,
        up_ratio=0.5,
        scale_ratio=0.8,
        leftover_ratio=0.5,
    ):
        self.n_epoch = n_epoch
        self.n_batch_per_epoch = n_batch_per_epoch
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.epoch_per_cycle = epoch_per_cycle
        self.up_ratio = up_ratio
        self.scale_ratio = scale_ratio
        self.leftover_ratio = leftover_ratio
        self.min_mom = min_mom
        self.max_mom = max_mom

        self.total_batch = self.n_epoch * self.n_batch_per_epoch
        self.total_batch_per_cycle = self.epoch_per_cycle * self.n_batch_per_epoch
        self.batch_per_up = int(self.up_ratio * self.total_batch_per_cycle)
        self.batch_per_down = self.total_batch_per_cycle - self.batch_per_up

        self.clr_cycle = self.n_epoch // self.epoch_per_cycle
        self.leftover_cycle = self.n_epoch % self.epoch_per_cycle
        self.leftover_batch = self.leftover_cycle * self.n_batch_per_epoch
        self.lr_schedule = self.get_lr()
        self.mom_schedule = self.get_mom()

    def get_mom(self):
        list_mom = []
        for cycle_idx in range(self.clr_cycle):
            difference = (self.max_mom - self.min_mom) * (self.scale_ratio ** cycle_idx)
            max_mom = self.max_mom
            min_mom = self.max_mom - difference
            down_mom = np.linspace(max_mom, min_mom, self.batch_per_up).tolist()
            up_mom = np.linspace(min_mom, max_mom, self.batch_per_down).tolist()
            list_mom.extend(down_mom)
            list_mom.extend(up_mom)
        leftover_mom = [self.max_mom] * self.leftover_batch
        list_mom.extend(leftover_mom)
        return list_mom

    def get_lr(self):
        list_lr = []
        for cycle_idx in range(self.clr_cycle):
            difference = (self.max_lr - self.min_lr) * (self.scale_ratio ** cycle_idx)
            min_lr = self.min_lr
            max_lr = self.min_lr + difference
            up_lr = np.linspace(min_lr, max_lr, self.batch_per_up).tolist()
            down_lr = np.linspace(max_lr, min_lr, self.batch_per_down).tolist()
            list_lr.extend(up_lr)
            list_lr.extend(down_lr)

        leftover_lr = np.linspace(
            self.min_lr, self.min_lr * self.leftover_ratio, self.leftover_batch
        ).tolist()
        list_lr.extend(leftover_lr)
        return list_lr

    def plot_lr(self):
        sns.lineplot(x=np.arange(self.total_batch), y=np.array(self.lr_schedule))

    def plot_mom(self):
        sns.lineplot(x=np.arange(self.total_batch), y=np.array(self.mom_schedule))

    def on_batch_begin(self, batch_idx, model, optimizer):
        lr = self.lr_schedule[batch_idx]
        mom = self.mom_schedule[batch_idx]
        self.set_lr(optimizer, lr)
        self.set_mom(optimizer, mom)
