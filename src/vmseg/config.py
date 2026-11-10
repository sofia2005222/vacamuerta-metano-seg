"""
Configuración de datos y de entrenamiento.
"""
import os
import random

import numpy as np
import torch


class DatasetConfig:
    def __init__(
        self,
        data_root="data/recortes",
        bandas=(0, 1, 2, 3, 4, 5),   # índices sobre BANDAS: (2, 1, 0) es RGB
        num_classes=3,
        batch_size=8,
        num_workers=2,
        augment=False
    ):
        self.data_root = data_root
        self.bandas = list(bandas)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment


class TrainingConfig:
    def __init__(
        self,
        epochs=20,
        learning_rate=1e-3,
        weight_decay=1e-4,
        checkpoints_dir="runs/prueba",
        device="cpu"
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.checkpoints_dir = checkpoints_dir
        self.device = device


def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def create_checkpoint_dir(checkpoint_dir):
    # Si no existe crea el directorio
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print(f"Checkpoint directory: {checkpoint_dir}")
    return checkpoint_dir
