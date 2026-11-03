import json

import numpy as np
import pandas as pd
import pytest

from vmseg.build import build_patches, read_scene
from vmseg.geo import load_aoi, utm_epsg
from vmseg.infer import polygonize, predict_raster
from vmseg.labels import points_to_weak_mask, weak_label_quality
from vmseg.synthetic import write_scene


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    return write_scene(out, name="synthetic", seed=3), out


def test_utm_zone_for_neuquen():
    assert utm_epsg(-68.8, -38.7) == 32719


def test_aoi_bbox():
    aoi = load_aoi("sierra_barrosa", "configs/aois.yaml")
    w, s, e, n = aoi["bbox"]
    assert w < -68.83698 < e and s < -38.76047 < n


def test_weak_mask_overlaps_true_mask(synth):
    import geopandas as gpd

    info, _ = synth
    _, transform, crs = read_scene(info["scene"])
    wells = gpd.read_file(info["wells"])
    weak = points_to_weak_mask(wells, transform, info["true_mask"].shape, crs, radius_m=60.0)
    q = weak_label_quality(weak, info["true_mask"])
    assert weak.sum() > 0
    assert q["weak_covers_manual"] > 0.4  # el disco cubre buena parte de la locación...
    assert q["iou"] < 0.95                # ...pero NO es perfecto: es una etiqueta ruidosa


def test_buffer_needs_projected_crs(synth):
    import geopandas as gpd

    from vmseg.geo import rasterize_geoms

    info, _ = synth
    wells = gpd.read_file(info["wells"])
    with pytest.raises(ValueError):
        rasterize_geoms(wells, info["transform"], (10, 10), "EPSG:4326", buffer_m=50)


def test_build_patches_end_to_end(synth, tmp_path):
    info, _ = synth
    summary = build_patches(
        [info["scene"]], info["wells"], tmp_path, size=64, radius_m=60.0,
        manual_path=info["manual"], labeled_path=info["labeled"], block_px=192, seed=0,
    )
    df = pd.read_csv(tmp_path / "index.csv")
    assert summary["n_patches"] == len(df) == 144
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df["has_manual"] == 1).all()
    assert summary["weak_vs_manual"] is not None
    assert (tmp_path / "to_label.gpkg").exists()
    # sin fuga: cada bloque espacial pertenece a un único split
    df["block"] = list(zip(df["y"] // 192, df["x"] // 192))
    assert df.groupby("block")["split"].nunique().max() == 1
    # los parches se pueden leer y tienen la forma esperada
    d = np.load(tmp_path / df.loc[0, "file"])
    assert d["image"].shape == (4, 64, 64) and d["weak"].shape == d["manual"].shape == (64, 64)
    assert json.loads((tmp_path / "summary.json").read_text())["radius_m"] == 60.0


def test_build_requires_labeled_with_manual(synth, tmp_path):
    info, _ = synth
    with pytest.raises(ValueError):
        build_patches([info["scene"]], info["wells"], tmp_path, size=64, manual_path=info["manual"])


def test_predict_raster_reconstructs_identity():
    rng = np.random.default_rng(0)
    img = rng.random((4, 300, 340)).astype(np.float32)
    out = predict_raster(img, lambda b: b[:, 0], size=64, stride=32, batch_size=5)
    assert out.shape == (300, 340)
    np.testing.assert_allclose(out, img[0], atol=1e-6)


def test_polygonize_drops_tiny_blobs(synth):
    info, _ = synth
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[10:30, 10:30] = 0.9   # 20x20 px de 10 m = 40.000 m²
    prob[60:61, 60:61] = 0.9   # 1 px = 100 m²
    gdf = polygonize(prob, info["transform"], f"EPSG:{info['epsg']}", 0.5, min_area_m2=2000)
    assert len(gdf) == 1
    assert gdf["area_m2"].iloc[0] == pytest.approx(40000.0)
