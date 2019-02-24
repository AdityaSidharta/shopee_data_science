import torch
from tqdm import tqdm_notebook as tqdm

from utils.pytorch.utils.common import get_batch_info


def predict_model(model, test_dataloader, pred_fn):
    n_obs, batch_size, batch_size_per_epoch = get_batch_info(test_dataloader)
    prediction_list = []
    model = model.eval()
    t = tqdm(enumerate(test_dataloader), total=batch_size_per_epoch)
    with torch.no_grad():
        for idx, data in t:
            prediction = pred_fn(model, data)
            prediction_list.extend(prediction)
    return prediction_list
