"""
Dataset de PyTorch que lee pares con el formato de docs/FORMATO_RECORTE.md (ADR-0011).

Cada ejemplo devuelve T1 y T2 apiladas en un solo tensor: primero los canales de T1 y
después los de T2. La fusión temprana usa ese tensor tal cual; la siamesa lo parte al
medio. Así el bucle de entrenamiento de entrenar.py sirve para los dos modelos.
"""
import albumentations as A
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from vmseg.config import DatasetConfig
from vmseg.par import leer_indice_pares, leer_par
from vmseg.recorte import BANDAS, LOCACION, PILETA

# posición de cada banda en el recorte
B2, B4, B8, B11 = (BANDAS.index(b) for b in ("B2", "B4", "B8", "B11"))
INDICES = ["NDVI", "NDBI", "BSI"]
K_FOLDS = 5

def calcular_indices(img):
    """
    Calcula NDVI, NDBI y BSI a partir de un recorte de 6 bandas.

    img: array (6, H, W) con reflectancia. Devuelve un array (3, H, W).
    """
    eps = 1e-6   # para no dividir por cero
    azul, rojo, nir, swir = img[B2], img[B4], img[B8], img[B11]
    ndvi = (nir - rojo) / (nir + rojo + eps)
    ndbi = (swir - nir) / (swir + nir + eps)
    bsi = ((swir + rojo) - (nir + azul)) / ((swir + rojo) + (nir + azul) + eps)
    return np.stack([ndvi, ndbi, bsi]).astype(np.float32)


def preparar_fecha(img, bandas, indices):
    """Elige las bandas de un recorte y, si se pide, le agrega los índices al final."""
    salida = img[bandas]
    if indices:
        salida = np.concatenate([salida, calcular_indices(img)])
    return salida

class ParesDataset(Dataset):
    """
    Pares de un split, con la estructura del Dataset de segmentación del tutorial 11
    (is_train, build_transforms y apply_transforms).
    """
    def __init__(self, config: DatasetConfig, split=None, is_train=False, mean=None, std=None, ids=None):
        self.config = config
        self.split = split
        self.is_train = is_train
        self.mean = mean
        self.std = std
        # con ids se usan esos pares (un fold del K-fold); si no, los validados del split
        self.ids = list(ids) if ids is not None else self.find_files()
        self.transforms = self.build_transforms()

    def find_files(self):
        """ids de los pares del split, según indice.csv."""
        indice = leer_indice_pares(self.config.data_root)
        return list(indice[(indice["split"] == self.split) & indice["validada"]]["id"])


    def canales_por_fecha(self):
        return len(self.config.bandas) + (len(INDICES) if self.config.indices else 0)

    def __len__(self):
        return len(self.ids)

    def build_transforms(self):
        transforms = []
        if self.is_train and self.config.augment:
            # augmentation solo en train; en validación o test es un error metodológico
            # Solo giros y espejados: no cambian la reflectancia ni
            # interpolan los píxeles, y giran las dos fechas y la máscara juntas.
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ])
        return A.Compose(transforms)

    def apply_transforms(self, image, mask):
        """image: array (H, W, canales); mask: array (H, W)."""
        transformed = self.transforms(image=image, mask=mask)
        return transformed["image"], transformed["mask"]

    def leer_fechas(self, idx):
        """T1 y T2 sin normalizar, cada una (canales, H, W), y la máscara."""
        img_t1, img_t2, mask = leer_par(self.config.data_root, self.ids[idx])
        t1 = preparar_fecha(img_t1, self.config.bandas, self.config.indices)
        t2 = preparar_fecha(img_t2, self.config.bandas, self.config.indices)
        if self.config.num_classes == 2:
            # sin clase pileta: el hueco de la pileta es parte de la locación
            mask = np.where(mask == PILETA, LOCACION, mask)

        return t1, t2, mask

    def __getitem__(self, idx):
        t1, t2, mask = self.leer_fechas(idx)

        # normalizo las dos fechas con la misma media y desvío (calculados sobre train)
        if self.mean is not None:
            mean = self.mean.numpy()[:, None, None]
            std = self.std.numpy()[:, None, None]
            t1 = (t1 - mean) / std
            t2 = (t2 - mean) / std

        # apilo: primero los canales de T1, después los de T2
        x = np.concatenate([t1, t2])

        # los píxeles sin dato (NaN) quedan en 0, que después de normalizar es la media
        x = np.nan_to_num(x, nan=0.0)

        # albumentations trabaja con (H, W, canales): la misma transformación a todo
        x, mask = self.apply_transforms(x.transpose(1, 2, 0), mask)

        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask)).long()
        return x, mask

def calcular_media_std_pares(dataset):
    """Media y desvío por canal sobre los píxeles de T1 y T2 de todo el dataset."""
    canales = dataset.canales_por_fecha()
    suma = torch.zeros(canales, dtype=torch.float64)
    suma_cuad = torch.zeros(canales, dtype=torch.float64)
    N = torch.zeros(canales, dtype=torch.float64)

    for idx in range(len(dataset)):
        t1, t2, _ = dataset.leer_fechas(idx)
        for img in (t1, t2):
            img = torch.from_numpy(img).double()
            validos = ~torch.isnan(img)
            img = torch.nan_to_num(img, nan=0.0)
            suma += img.sum(dim=[1, 2])
            suma_cuad += (img ** 2).sum(dim=[1, 2])
            N += validos.sum(dim=[1, 2])

    mean = suma / N
    std = (suma_cuad / N - mean ** 2).sqrt()
    return mean.float(), std.float()

def asignar_folds(indice, k=K_FOLDS):
    """
    Fold (0 a k-1) de cada par validado de train y val, para la validación cruzada por bloques.

    Todos los pares de un bloque van al mismo fold, así el modelo nunca se valida sobre un
    terreno vecino al que vio al entrenar. Los bloques se reparten de mayor a menor, cada uno
    al fold que tiene menos pares en ese momento, para que los folds queden parejos.
    Devuelve una Series indexada por id.
    """
    pares = indice[indice["split"].isin(["train", "val"]) & indice["validada"]]
    tamanos = pares.groupby(["zona", "bloque"]).size().sort_values(ascending=False, kind="stable")

    pares_por_fold = [0] * k
    fold_de_bloque = {}
    for bloque, n in tamanos.items():
        fold = pares_por_fold.index(min(pares_por_fold))
        fold_de_bloque[bloque] = fold
        pares_por_fold[fold] += n

    folds = [fold_de_bloque[(z, b)] for z, b in zip(pares["zona"], pares["bloque"])]
    return pd.Series(folds, index=pares["id"].values, name="fold")

def get_data_pares(config: DatasetConfig):
    # la media y el desvío se calculan solo sobre train, sin aumento de datos
    mean, std = calcular_media_std_pares(ParesDataset(config, "train"))

    train_dataset = ParesDataset(config, "train", is_train=True, mean=mean, std=std)
    valid_dataset = ParesDataset(config, "val", mean=mean, std=std)

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

def get_data_pares_fold(config: DatasetConfig, fold, k=K_FOLDS):
    """Loaders del K-fold por bloques: valida con el fold `fold` y entrena con los otros k-1."""
    folds = asignar_folds(leer_indice_pares(config.data_root), k)
    ids_train = folds.index[folds != fold]
    ids_val = folds.index[folds == fold]

    # la media y el desvío se calculan solo sobre los pares de entrenamiento de este fold
    mean, std = calcular_media_std_pares(ParesDataset(config, ids=ids_train))

    train_dataset = ParesDataset(config, is_train=True, mean=mean, std=std, ids=ids_train)
    valid_dataset = ParesDataset(config, mean=mean, std=std, ids=ids_val)

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

def get_data_pares_final(config: DatasetConfig):
    """Loader del modelo final: entrena con todos los pares del K-fold (train + val), sin validación."""
    ids = asignar_folds(leer_indice_pares(config.data_root)).index

    # la media y el desvío se calculan sobre todos los pares de entrenamiento
    mean, std = calcular_media_std_pares(ParesDataset(config, ids=ids))

    train_dataset = ParesDataset(config, is_train=True, mean=mean, std=std, ids=ids)
    train_loader = DataLoader(
                        train_dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers
                    )
    return train_loader, mean, std