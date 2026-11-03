"""Prueba el armado del compuesto Sentinel-2 con archivos LOCALES (sin Internet).

No prueba `search_items` (consulta al catálogo real): eso hay que verificarlo en Colab.
"""
import datetime as dt

import numpy as np
import pytest

pytest.importorskip("stackstac")
pystac = pytest.importorskip("pystac")

import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import box, mapping  # noqa: E402

from vmseg.stac import composite_from_items  # noqa: E402

EPSG = 32719
H = W = 400
TRANSFORM = from_origin(380000.0, 5700000.0, 10.0, 10.0)
BANDS = ("blue", "green", "red", "nir")


def _write_band(path, arr, dtype):
    with rasterio.open(path, "w", driver="GTiff", height=H, width=W, count=1, dtype=dtype, crs=f"EPSG:{EPSG}", transform=TRANSFORM) as dst:
        dst.write(arr.astype(dtype), 1)


def _make_item(tmp_path, name, day, cloudy):
    rng = np.random.default_rng(0)
    base = {b: (0.10 + 0.02 * i + 0.001 * rng.random((H, W))).astype(np.float32) for i, b in enumerate(BANDS)}
    scl = np.full((H, W), 4, dtype=np.uint8)  # 4 = vegetación (válido)
    if cloudy:
        for b in BANDS:
            base[b][100:300, 100:300] = 0.9   # nube muy brillante
        scl[100:300, 100:300] = 9              # SCL: nube de probabilidad alta
    extra = {"proj:shape": [H, W], "proj:transform": [10, 0, 380000, 0, -10, 5700000, 0, 0, 1], "proj:epsg": EPSG}
    west, south, east, north = 380000.0, 5700000.0 - H * 10, 380000.0 + W * 10, 5700000.0
    tr = Transformer.from_crs(f"EPSG:{EPSG}", "EPSG:4326", always_xy=True)
    (lon_w, lat_s), (lon_e, lat_n) = tr.transform(west, south), tr.transform(east, north)
    item = pystac.Item(
        id=name, geometry=mapping(box(lon_w, lat_s, lon_e, lat_n)), bbox=[lon_w, lat_s, lon_e, lat_n],
        datetime=dt.datetime(2026, 1, day, tzinfo=dt.timezone.utc), properties={"eo:cloud_cover": 5.0, "proj:epsg": EPSG},
    )
    for b in BANDS:
        p = tmp_path / f"{name}_{b}.tif"
        _write_band(p, base[b], "float32")
        item.add_asset(b, pystac.Asset(href=str(p), media_type="image/tiff; application=geotiff", roles=["data"], extra_fields=extra))
    p = tmp_path / f"{name}_scl.tif"
    _write_band(p, scl, "uint8")
    item.add_asset("scl", pystac.Asset(href=str(p), media_type="image/tiff; application=geotiff", roles=["data"], extra_fields=extra))
    return item, (lon_w, lat_s, lon_e, lat_n), base


def test_composite_masks_clouds(tmp_path):
    cloudy, bbox, _ = _make_item(tmp_path, "cloudy", day=1, cloudy=True)
    clean, _, base = _make_item(tmp_path, "clean", day=11, cloudy=False)
    out = composite_from_items([cloudy, clean], bbox, tmp_path / "comp.tif", epsg=EPSG)

    with rasterio.open(out) as src:
        comp = src.read()
        assert src.count == 4 and src.crs.to_epsg() == EPSG
        assert abs(src.res[0] - 10.0) < 1e-6
    # la caja en lat/lon, convertida de vuelta a UTM, es un poco más grande que la escena original
    assert H - 3 <= comp.shape[1] <= H + 25 and W - 3 <= comp.shape[2] <= W + 25

    cy, cx = comp.shape[1] // 2, comp.shape[2] // 2
    centro = comp[:, cy - 30 : cy + 30, cx - 30 : cx + 30]  # dentro de la zona con nube en una de las dos fechas
    for i, b in enumerate(BANDS):
        esperado = 0.10 + 0.02 * i
        assert np.nanmax(centro[i]) < 0.2, f"la nube se coló en la banda {b}"
        assert abs(np.nanmean(centro[i]) - esperado) < 0.01
