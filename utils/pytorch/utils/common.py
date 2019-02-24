import math
import torch


def get_batch_info(dataloader):
    n_obs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    n_batch_per_epoch = math.ceil(n_obs / float(batch_size))
    return n_obs, batch_size, n_batch_per_epoch


def img2tensor(img_array, device):
    img_array = img_array.transpose((2, 0, 1))
    return torch.from_numpy(img_array).float().to(device)
