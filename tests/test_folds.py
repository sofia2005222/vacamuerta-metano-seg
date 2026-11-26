import pandas as pd

from vmseg.config import DatasetConfig
from vmseg.dataset_pares import asignar_folds, get_data_pares_fold
from vmseg.par import generar_pares_falsos


def _indice():
    """11 bloques de train y val con tamaños distintos, más pares de test y uno sin validar."""
    filas = []
    for bloque, n in enumerate([5, 12, 15, 7, 18, 27, 13, 4, 17, 17, 13]):
        for i in range(n):
            filas.append(dict(id=f"anelo_b{bloque}_{i}", zona="anelo", bloque=bloque,
                              split="val" if bloque in (1, 9) else "train", validada=True))
    filas.append(dict(id="anelo_b20_0", zona="anelo", bloque=20, split="test_anelo", validada=True))
    filas.append(dict(id="anelo_b3_sin", zona="anelo", bloque=3, split="train", validada=False))
    return pd.DataFrame(filas)


def test_folds_por_bloque_parejos():
    indice = _indice()
    folds = asignar_folds(indice, k=5)

    # entran solo los validados de train y val
    assert len(folds) == 148
    assert "anelo_b20_0" not in folds.index and "anelo_b3_sin" not in folds.index

    # cada bloque entero en un solo fold
    bloques = indice.set_index("id").loc[folds.index, "bloque"]
    assert (folds.groupby(bloques).nunique() == 1).all()

    # folds parejos
    tamanos = folds.value_counts()
    assert len(tamanos) == 5 and tamanos.max() - tamanos.min() <= 4


def test_loaders_de_un_fold_no_se_mezclan(tmp_path):
    generar_pares_falsos(tmp_path, n=12, tamano=32)
    config = DatasetConfig(data_root=str(tmp_path), batch_size=2, num_workers=0)
    folds = asignar_folds(pd.read_csv(tmp_path / "indice.csv"), k=2)

    train_loader, valid_loader, mean, std = get_data_pares_fold(config, fold=0, k=2)
    ids_train, ids_val = set(train_loader.dataset.ids), set(valid_loader.dataset.ids)
    assert ids_train.isdisjoint(ids_val)
    assert ids_train | ids_val == set(folds.index)
    assert len(mean) == len(config.bandas)