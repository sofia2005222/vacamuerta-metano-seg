from vmseg.config import DatasetConfig, TrainingConfig
from vmseg.kfold import entrenar_final, entrenar_kfold, media_desvio, resumir
from vmseg.par import generar_pares_falsos
from vmseg.dataset_pares import asignar_folds
from vmseg.par import generar_pares_falsos, leer_indice_pares

def test_resumir_toma_el_mejor_y_las_ultimas_10():
    history = {"valid_miou": [0.1] * 10 + [0.5, 0.9] + [0.3] * 10}
    r = resumir(history)
    assert r["iou_val"] == 0.9 and r["mejor_epoca"] == 12 and r["iou_val_ult10"] == 0.3


def test_media_desvio():
    assert media_desvio([{"x": 0.6}, {"x": 0.8}], "x") == (0.7, 0.1414)


def test_kfold_entrena_cada_fold(tmp_path):
    generar_pares_falsos(tmp_path / "pares", n=12, tamano=64)
    dataset_config = DatasetConfig(data_root=str(tmp_path / "pares"), num_classes=2,
                                   batch_size=2, num_workers=0)
    train_config = TrainingConfig(epochs=1, checkpoints_dir=str(tmp_path / "runs"), perdida="ce_dice")

    vistos = []
    resultados = entrenar_kfold(dataset_config, train_config, "siamesa", "cero", k=2,
                                al_terminar_fold=lambda fold, r: vistos.append(fold))

    assert [r["fold"] for r in resultados] == [0, 1] and vistos == [0, 1]
    for fold in (0, 1):
        assert (tmp_path / "runs" / f"fold{fold}" / "best.pt").exists()
        assert (tmp_path / "runs" / f"fold{fold}" / "curvas.png").exists()
    # train_config no se modifica
    assert train_config.checkpoints_dir == str(tmp_path / "runs")

def test_entrenar_final_usa_todos_los_pares_y_guarda_final(tmp_path):
    generar_pares_falsos(tmp_path / "pares", n=12, tamano=64)
    dataset_config = DatasetConfig(data_root=str(tmp_path / "pares"), num_classes=2,
                                   batch_size=2, num_workers=0)
    train_config = TrainingConfig(epochs=1, checkpoints_dir=str(tmp_path / "final"), perdida="ce_dice")

    history, mean, std = entrenar_final(dataset_config, train_config, "siamesa", "cero")

    # entrena con todos los pares de train y val, sin validar
    assert history["pares_train"] == len(asignar_folds(leer_indice_pares(str(tmp_path / "pares"))))
    assert history["valid_miou"] == []
    assert (tmp_path / "final" / "final.pt").exists()
    assert (tmp_path / "final" / "curvas.png").exists()
    assert not (tmp_path / "final" / "best.pt").exists()
    assert len(mean) == len(std) == 6