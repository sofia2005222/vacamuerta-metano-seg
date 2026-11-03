"""Descarga de un compuesto de Sentinel-2 L2A desde un catálogo STAC público (Earth Search de AWS).

Se divide en dos pasos:
  1. `search_items`: consulta el catálogo (necesita Internet; NO se pudo probar contra la red real
     durante la escritura del repo, porque ese entorno no tenía salida).
  2. `composite_from_items`: arma la mediana libre de nubes a partir de los ítems. Este paso SÍ está
     probado con archivos locales (tests/test_stac_local.py).

Los nombres de assets ('blue', 'green', 'red', 'nir', 'scl') corresponden a la colección sentinel-2-l2a
de Earth Search v1. Si el catálogo cambió, ajustar `ASSETS`/`SCL_ASSET` o pasar otro `stac_url`
(p. ej. Microsoft Planetary Computer usa 'B02', 'B03', 'B04', 'B08', 'SCL'). Probarlo primero en
Colab con un AOI chico.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .geo import utm_epsg

EARTH_SEARCH = "https://earth-search.aws.element84.com/v1"
ASSETS = ("blue", "green", "red", "nir")  # orden de bandas del compuesto: azul, verde, rojo, NIR
SCL_ASSET = "scl"
SCL_BAD = (0, 1, 3, 8, 9, 10)  # sin dato, saturado, sombra de nube, nube (media/alta) y cirros


def search_items(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    max_cloud: float = 30.0,
    stac_url: str = EARTH_SEARCH,
):
    """Escenas Sentinel-2 L2A que cubren `bbox` (oeste, sur, este, norte) entre `start` y `end` (AAAA-MM-DD)."""
    from pystac_client import Client

    items = Client.open(stac_url).search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    ).item_collection()
    if len(items) == 0:
        raise RuntimeError("El catálogo no devolvió escenas. Probá ampliar las fechas o subir --max-cloud.")
    print(f"[stac] {len(items)} escenas encontradas")
    return items


def composite_from_items(
    items,
    bbox: tuple[float, float, float, float],
    out_path: str | Path,
    assets: tuple[str, ...] = ASSETS,
    scl_asset: str = SCL_ASSET,
    resolution: int = 10,
    epsg: int | None = None,
) -> Path:
    """Mediana temporal, enmascarando nubes y sombras con la banda SCL, escrita como GeoTIFF multibanda."""
    import rasterio
    import stackstac

    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    epsg = epsg or utm_epsg(lon, lat)
    # dtype float64: con float32 stackstac se niega a aplicar escala/offset de reflectancia (rescale=True).
    # El resultado se pasa a float32 más abajo; el cómputo es perezoso (dask), así que no carga todo junto.
    stack = stackstac.stack(
        items, assets=list(assets) + [scl_asset], resolution=resolution, bounds_latlon=bbox,
        epsg=epsg, dtype="float64", fill_value=np.nan,
    )
    transform = stack.attrs["transform"]
    bad = stack.sel(band=scl_asset).isin(list(SCL_BAD))
    bands = stack.sel(band=list(assets)).where(~bad)
    composite = bands.median(dim="time", skipna=True).compute().values.astype(np.float32)

    if np.nanmax(composite) > 2.0:  # si el catálogo no aplicó escala/offset, vienen en 0-10000
        composite = composite / 10000.0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=composite.shape[1], width=composite.shape[2], count=composite.shape[0],
        dtype="float32", crs=f"EPSG:{epsg}", transform=transform, nodata=np.nan, compress="deflate",
    ) as dst:
        dst.write(composite)
        for i, name in enumerate(assets, start=1):
            dst.set_band_description(i, name)
    print(f"[stac] escrito {out_path} ({composite.shape[2]}x{composite.shape[1]} px, EPSG:{epsg})")
    return out_path


def fetch_s2_composite(
    bbox: tuple[float, float, float, float],
    start: str,
    end: str,
    out_path: str | Path,
    max_cloud: float = 30.0,
    stac_url: str = EARTH_SEARCH,
) -> Path:
    """Busca y compone en un solo paso (es lo que llama el comando `fetch-s2`)."""
    items = search_items(bbox, start, end, max_cloud, stac_url)
    return composite_from_items(items, bbox, out_path)
