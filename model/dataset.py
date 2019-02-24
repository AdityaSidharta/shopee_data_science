import os

import torch
from skimage import io
from skimage.color import gray2rgb
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, \
    RandomVerticalFlip, RandomAffine, Normalize, ToTensor, ToPILImage, Grayscale
from categorical_encoder import OneHotEncoder

train_transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = Compose([
    ToPILImage(),
    Resize((224, 224)),
    ToTensor(),
    Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class TrainDataset(Dataset):
    def __init__(self, df, attribute, train_tsfm, device):
        self.df = df
        self.attribute = attribute
        self.train_tsfm = train_tsfm
        self.device = device

        self.img_array = self.create_img_array(self.df)
        self.label = self.create_label(self.df, attribute)
        self.encoder_model =

    def create_img_array(self, df):
        return df['image_path'].values

    def create_label(self, df, attribute):
        label_array = df[attribute].values


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_path, self.img_array[idx])
        img_array = io.imread(img_path)
        img_tensor = self.train_tsfm(img_array)
        img_tensor = img_tensor.type(torch.float).to(self.device)
        label = self.label[idx, :]
        label_tensor = torch.Tensor(label)
        label_tensor = label_tensor.type(torch.float).to(self.device)
        return img_tensor, label_tensor


class TestDataset(Dataset):
    def __init__(self, df, attribute, train_tsfm, device):
        self.df = df
        self.attribute = attribute
        self.train_tsfm = train_tsfm
        self.device = device

        self.img_array = self.create_img_array(self.df)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.train_path, self.img_array[idx])
        img_array = io.imread(img_path)
        img_tensor = self.train_tsfm(img_array)
        img_tensor = img_tensor.type(torch.float).to(self.device)
        return img_tensor
