"""
Validación cruzada por bloques (#20): entrena un modelo en cada fold y resume los resultados.

Cada fold usa los mismos pasos que el entrenamiento con val fijo (tutorial 09: deterministic,
modelo, optimizador y `main`), pero con los loaders de `get_data_pares_fold`.
"""
import copy
import os

import numpy as np
import torch

from vmseg.config import deterministic
from vmseg.dataset_pares import K_FOLDS, get_data_pares_fold
from vmseg.entrenar import guardar_curvas, main
from vmseg.siamesa import crear_modelo, nombres_canales


def resumir(history):
    """Mejor IoU de val, su época (contando desde 1) y el promedio de las últimas 10 épocas."""
    iou = history["valid_miou"]
    return dict(
        iou_val=round(float(max(iou)), 4),
        mejor_epoca=int(np.argmax(iou)) + 1,
        iou_val_ult10=round(float(np.mean(iou[-10:])), 4),
    )


def entrenar_fold(dataset_config, train_config, arquitectura, modo, fold, k=K_FOLDS, semilla=42):
    """Entrena un modelo validando con el fold `fold`. Devuelve la historia de `main`."""
    deterministic(semilla)
    train_loader, valid_loader, mean, std = get_data_pares_fold(dataset_config, fold, k)

    canales = nombres_canales(dataset_config)
    model = crear_modelo(arquitectura, canales, num_classes=dataset_config.num_classes,
                         modo=modo).to(train_config.device)

    # con el encoder congelado, solo se optimizan los parámetros que entrenan
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=train_config.learning_rate, weight_decay=train_config.weight_decay)
    # baja el learning rate a la mitad si la loss de val no mejora en 5 épocas
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    history = main(model, optimizer, train_loader, valid_loader, train_config,
                   num_classes=dataset_config.num_classes, scheduler=scheduler)
    history["pares_train"] = len(train_loader.dataset)
    history["pares_val"] = len(valid_loader.dataset)
    return history


def entrenar_kfold(dataset_config, train_config, arquitectura, modo, k=K_FOLDS, folds=None, al_terminar_fold=None):
    """
    Entrena un modelo por fold. En `train_config.checkpoints_dir` deja una carpeta `fold<i>`
    con `best.pt` y `curvas.png`. Devuelve una lista con el resumen de cada fold.

    folds: qué folds correr (por defecto todos); sirve para retomar si Colab se corta.
    al_terminar_fold: función que se llama con (fold, resumen) después de cada fold,
                      por ejemplo para subir los resultados al bucket.
    """
    resultados = []
    for fold in (range(k) if folds is None else folds):
        print(f"\n######## fold {fold + 1} de {k} ########")
        config_fold = copy.copy(train_config)
        config_fold.checkpoints_dir = os.path.join(train_config.checkpoints_dir, f"fold{fold}")

        history = entrenar_fold(dataset_config, config_fold, arquitectura, modo, fold, k)
        guardar_curvas(history, os.path.join(config_fold.checkpoints_dir, "curvas.png"))

        resumen = dict(fold=fold, pares_train=history["pares_train"], pares_val=history["pares_val"],
                       **resumir(history))
        print(resumen)
        resultados.append(resumen)
        if al_terminar_fold is not None:
            al_terminar_fold(fold, resumen)
    return resultados


def media_desvio(resultados, clave):
    """Media y desvío (muestral) de una métrica entre los folds."""
    valores = np.array([r[clave] for r in resultados], dtype=float)
    desvio = valores.std(ddof=1) if len(valores) > 1 else 0.0
    return round(float(valores.mean()), 4), round(float(desvio), 4)