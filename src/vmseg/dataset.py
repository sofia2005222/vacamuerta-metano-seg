"""
Dataset de PyTorch que lee recortes con el formato de docs/FORMATO_RECORTE.md.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T2

from vmseg.config import DatasetConfig
from vmseg.recorte import leer_indice, leer_recorte


class RecorteDataset(Dataset):
    def __init__(self, carpeta, split, bandas, mean=None, std=None, transform=None):
        indice = leer_indice(carpeta)
        self.ids = list(indice[indice["split"] == split]["id"])
        self.carpeta = carpeta
        self.bandas = list(bandas)
        self.mean = mean
        self.std = std
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img, mask = leer_recorte(self.carpeta, self.ids[idx])

        img = torch.from_numpy(img[self.bandas])
        mask = torch.from_numpy(mask.astype(np.int64))

        # normalizo con la media y el desvío de train
        if self.mean is not None:
            img = (img - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)

        # los píxeles sin dato (NaN) quedan en 0, que después de normalizar es la media
        img = torch.nan_to_num(img, nan=0.0)

        # data augmentation: la misma transformación geométrica a imagen y máscara
        if self.transform is not None:
            img, mask = self.transform(tv_tensors.Image(img), tv_tensors.Mask(mask))

        return img.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor)


def calcular_media_std(dataset):
    """Media y desvío por banda sobre todos los píxeles del dataset (como en el tutorial 09)."""
    n_bandas = len(dataset.bandas)
    sum_channels = torch.zeros(n_bandas, dtype=torch.float64)
    sum_sq_channels = torch.zeros(n_bandas, dtype=torch.float64)
    N = torch.zeros(n_bandas, dtype=torch.float64)

    for img, _ in dataset:
        img = img.double()
        validos = ~torch.isnan(img)
        img = torch.nan_to_num(img, nan=0.0)
        sum_channels += img.sum(dim=[1, 2])
        sum_sq_channels += (img ** 2).sum(dim=[1, 2])
        N += validos.sum(dim=[1, 2])

    mean = sum_channels / N
    std = (sum_sq_channels / N - mean ** 2).sqrt()
    return mean.float(), std.float()


def get_data(config: DatasetConfig):
    # transformaciones geométricas: giran y espejan imagen y máscara juntas
    # No usamos ColorJitter: cambiaría la reflectancia, que es lo que mide el satélite
    augment = T2.Compose([
        T2.RandomHorizontalFlip(),
        T2.RandomVerticalFlip(),
        T2.RandomChoice([T2.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
    ])

    # la media y el desvío se calculan solo sobre train
    train_crudo = RecorteDataset(config.data_root, "train", config.bandas)
    mean, std = calcular_media_std(train_crudo)

    preprocess_train = augment if config.augment else None

    train_dataset = RecorteDataset(config.data_root, "train", config.bandas, mean, std, preprocess_train)
    valid_dataset = RecorteDataset(config.data_root, "val", config.bandas, mean, std)

    train_loader = DataLoader(
                        train_dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers
                    )

    valid_loader = DataLoader(
                        valid_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.num_workers
                    )

    return train_loader, valid_loader, mean, std
