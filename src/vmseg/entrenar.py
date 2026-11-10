"""
Pipeline de entrenamiento: train, validate y main.
"""
import copy
import os
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from vmseg.config import TrainingConfig
from vmseg.recorte import SIN_ETIQUETA

# negrita para los números en la terminal
bold = f"\033[1m"
reset = f"\033[0m"


def acumular_confusion(confusion, pred, target, num_classes):
    """Suma a la matriz de confusión (num_classes x num_classes) los píxeles con etiqueta."""
    validos = target != SIN_ETIQUETA
    t = target[validos]
    p = pred[validos]
    confusion += torch.bincount(t * num_classes + p, minlength=num_classes ** 2).reshape(num_classes, num_classes).cpu()
    return confusion


def iou_por_clase(confusion):
    """IoU de cada clase: intersección / unión. Si una clase no aparece, da NaN."""
    interseccion = confusion.diag().float()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag()
    return interseccion / union.float()


def miou_objetos(confusion):
    """Promedio del IoU de locación y pileta (sin el fondo, que lo infla)."""
    return float(torch.nanmean(iou_por_clase(confusion)[1:]))


def train(
    train_config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int,
    num_classes: int = 3,
) -> Tuple[float, float, list]:

    # pongo el modelo en modo entrenamiento
    model.train()

    count_sample = 0
    step_loss = 0
    iter_losses = []
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    status = f"Train:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    prog_bar = tqdm(train_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    prog_bar.set_description(status)

    for data, target in prog_bar:

        # mando los datos y las etiquetas al dispositivo (cpu o cuda)
        data, target = data.to(train_config.device), target.to(train_config.device)

        # pongo en cero los gradientes de la iteración anterior
        optimizer.zero_grad()

        # paso hacia adelante: la salida es (B, num_classes, H, W)
        output = model(data)

        # cross entropy por píxel, ignorando los píxeles sin etiqueta
        loss = F.cross_entropy(output, target, ignore_index=SIN_ETIQUETA)

        # calculo los gradientes (backpropagation)
        loss.backward()

        # actualizo los pesos con los gradientes
        optimizer.step()

        # clase predicha en cada píxel
        pred_idx = output.detach().argmax(dim=1)
        confusion = acumular_confusion(confusion, pred_idx, target, num_classes)

        # loss de la iteración (para la curva) y acumulado de la época
        iter_losses.append(loss.item())
        step_loss += loss.item() * data.shape[0]
        count_sample += data.shape[0]

        # actualizo el texto de la barra de progreso
        step_status = status + f"\tLoss: {float(step_loss/count_sample):.4f}, "
        step_status += f"mIoU: {miou_objetos(confusion):.4f}"
        prog_bar.set_description(step_status)

    epoch_loss = float(step_loss / count_sample)
    epoch_miou = miou_objetos(confusion)

    prog_bar.close()

    return epoch_loss, epoch_miou, iter_losses


def validate(
    train_config: TrainingConfig,
    model: nn.Module,
    valid_loader: torch.utils.data.DataLoader,
    epoch_idx: int,
    total_epochs: int,
    num_classes: int = 3,
) -> Tuple[float, float]:

    # pongo el modelo en modo evaluación
    model.eval()

    count_sample = 0
    step_loss = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    status = f"Valid:\t{bold}Epoch: {epoch_idx}/{total_epochs}{reset}"
    prog_bar = tqdm(valid_loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    prog_bar.set_description(status)

    for data, target in prog_bar:

        # mando los datos y las etiquetas al dispositivo (cpu o cuda)
        data, target = data.to(train_config.device), target.to(train_config.device)

        # en validación no se calculan gradientes
        with torch.no_grad():
            output = model(data)

        # cross entropy por píxel, ignorando los píxeles sin etiqueta
        valid_loss = F.cross_entropy(output, target, ignore_index=SIN_ETIQUETA).item()

        # clase predicha en cada píxel
        pred_idx = output.argmax(dim=1)
        confusion = acumular_confusion(confusion, pred_idx, target, num_classes)

        step_loss += valid_loss * data.shape[0]
        count_sample += data.shape[0]

        # actualizo el texto de la barra de progreso
        step_status = status + f"\tLoss: {float(step_loss/count_sample):.4f}, "
        step_status += f"mIoU: {miou_objetos(confusion):.4f}"
        prog_bar.set_description(step_status)

    valid_loss = float(step_loss / count_sample)
    valid_miou = miou_objetos(confusion)

    prog_bar.close()

    return valid_loss, valid_miou


def main(
    model: nn.Module,
    optimizer,
    train_loader,
    valid_loader,
    train_config: TrainingConfig,
    num_classes: int = 3,
    scheduler=None
) -> dict:

    # carpeta donde se guarda el mejor modelo
    ckpt_dir = train_config.checkpoints_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    best_score = -1
    best_weights = copy.deepcopy(model.state_dict())

    # historia de loss y mIoU para las curvas
    epoch_train_loss, epoch_valid_loss = [], []
    epoch_train_miou, epoch_valid_miou = [], []
    iter_train_loss = []

    # mido el tiempo de entrenamiento
    t_begin = time.time()
    for epoch in range(train_config.epochs):

        train_loss, train_miou, iter_losses = train(train_config, model, optimizer, train_loader,
                                                    epoch + 1, train_config.epochs, num_classes)

        val_loss, val_miou = validate(train_config, model, valid_loader,
                                      epoch + 1, train_config.epochs, num_classes)

        print(f"\n{bold}Train Loss: {train_loss:.4f}{reset}   {bold}Train mIoU: {train_miou:.4f}{reset}")
        print(f"{bold}Val Loss:   {val_loss:.4f}{reset}   {bold}Val mIoU:   {val_miou:.4f}{reset}")

        iter_train_loss += iter_losses
        epoch_train_loss.append(train_loss)
        epoch_train_miou.append(train_miou)
        epoch_valid_loss.append(val_loss)
        epoch_valid_miou.append(val_miou)

        # si hay scheduler, baja el learning rate cuando la loss de validación se estanca
        if scheduler is not None:
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"lr {old_lr:.6f} -> {new_lr:.6f}")

        # guardo el mejor modelo según el mIoU de validación
        score = val_miou
        if score > best_score:
            best_score = score
            print(f"\nModel Improved... Saving Model ... ", end="")
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))
            print("Done.\n")

        print(f"{'='*72}\n")

    print(f"Total time: {(time.time() - t_begin):.2f}s, Best Score: {best_score:.3f}")

    # cargo los pesos del mejor modelo
    model.load_state_dict(best_weights)

    history = dict(
        model=model,
        iter_train_loss=iter_train_loss,
        train_loss=epoch_train_loss,
        train_miou=epoch_train_miou,
        valid_loss=epoch_valid_loss,
        valid_miou=epoch_valid_miou,
        train_config=train_config,
        best_score=best_score,
    )

    return history
