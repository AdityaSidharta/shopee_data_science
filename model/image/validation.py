import numpy as np
from tqdm import tqdm_notebook as tqdm
import torch

from utils.pytorch.utils.common import get_batch_info


def validate_model(model, criterion, loss_fn, metric_fn, val_dataloader):
    n_val_obs, val_batch_size, val_batch_per_epoch = get_batch_info(val_dataloader)
    total_loss = np.zeros(val_batch_per_epoch)
    total_metric = np.zeros(val_batch_per_epoch)
    model = model.eval()
    t = tqdm(enumerate(val_dataloader), total=val_batch_per_epoch)
    with torch.no_grad():
        for idx, data in t:
            loss = loss_fn(model, criterion, data)
            metric = metric_fn(model, data)
            total_loss[idx] = loss
            total_metric[idx] = metric
    return total_loss.mean(), total_metric.mean()
