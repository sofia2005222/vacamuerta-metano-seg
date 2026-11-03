import numpy as np
import pytest

from vmseg.metrics import add_confusions, confusion, object_counts, object_metrics, pixel_metrics
from vmseg.tiling import assign_splits, block_key, iter_patches, window_starts


def test_window_starts_cover_the_edge():
    assert window_starts(512, 256, 256) == [0, 256]
    assert window_starts(600, 256, 256) == [0, 256, 344]  # el último pega al borde
    with pytest.raises(ValueError):
        window_starts(100, 256, 256)


def test_splits_are_deterministic_and_block_disjoint():
    keys = [("s", r, c) for r in range(6) for c in range(6)]
    a = assign_splits(keys, seed=1)
    b = assign_splits(keys, seed=1)
    assert a == b
    assert set(a.values()) == {"train", "val", "test"}
    # un mismo bloque nunca está en dos splits (es una función de la clave)
    assert len(a) == len(set(keys))


def test_patches_with_too_many_nans_are_dropped():
    img = np.ones((4, 128, 128), dtype=np.float32)
    img[:, :64, :64] = np.nan  # el cuadrante superior izquierdo no tiene datos
    weak = np.zeros((128, 128), dtype=np.uint8)
    patches = list(iter_patches(img, weak, None, size=64, stride=64, min_valid=0.9))
    assert len(patches) == 3
    assert all(np.isfinite(p["image"]).all() for p in patches)


def test_block_key_groups_neighbors():
    assert block_key("a", 0, 0, 512) == block_key("a", 255, 255, 512)
    assert block_key("a", 0, 0, 512) != block_key("a", 512, 0, 512)


def test_pixel_metrics_known_case():
    gt = np.zeros((10, 10), dtype=bool); gt[:5, :5] = True       # 25 px
    pred = np.zeros((10, 10), dtype=bool); pred[:5, :10] = True  # 50 px, contiene al gt
    m = pixel_metrics(confusion(pred, gt))
    assert m["recall"] == pytest.approx(1.0)
    assert m["precision"] == pytest.approx(0.5)
    assert m["iou"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(2 / 3)


def test_confusions_accumulate():
    a = confusion(np.array([1, 0]), np.array([1, 1]))
    b = confusion(np.array([0, 0]), np.array([0, 1]))
    assert add_confusions(a, b) == {"tp": 1, "fp": 0, "fn": 2, "tn": 1}


def test_object_metrics():
    gt = np.zeros((20, 20), dtype=bool)
    gt[2:6, 2:6] = True      # locación 1 (detectada)
    gt[12:16, 12:16] = True  # locación 2 (no detectada)
    pred = np.zeros_like(gt)
    pred[2:6, 2:6] = True
    pred[0:2, 15:19] = True  # falso positivo
    c = object_counts(pred, gt)
    assert c == {"n_gt": 2, "n_pred": 2, "gt_detected": 1, "pred_correct": 1}
    m = object_metrics(c)
    assert m["object_recall"] == pytest.approx(0.5)
    assert m["object_precision"] == pytest.approx(0.5)


def test_object_counts_empty():
    z = np.zeros((8, 8), dtype=bool)
    assert object_counts(z, z) == {"n_gt": 0, "n_pred": 0, "gt_detected": 0, "pred_correct": 0}
