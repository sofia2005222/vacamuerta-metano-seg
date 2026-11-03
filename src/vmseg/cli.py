"""Línea de comandos del proyecto.  Uso:  python -m vmseg.cli <comando> --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .labels import DEFAULT_RADIUS_M


def _cmd_synth(a):
    from .synthetic import write_scene

    out = write_scene(a.out, name=a.name, seed=a.seed, h=a.size, w=a.size, n_pads=a.n_pads)
    print(json.dumps({k: str(v) for k, v in out.items() if k in ("scene", "wells", "manual", "labeled")}, indent=2))


def _cmd_fetch_s2(a):
    from .geo import load_aoi
    from .stac import fetch_s2_composite

    aoi = load_aoi(a.aoi, a.aois)
    out = a.out or f"data/raw/{a.aoi}_{a.start}_{a.end}.tif"
    fetch_s2_composite(aoi["bbox"], a.start, a.end, out, max_cloud=a.max_cloud)


def _cmd_prepare_wells(a):
    import geopandas as gpd

    from .geo import load_aoi

    gdf = gpd.read_file(a.input).to_crs("EPSG:4326")
    print(f"[wells] {len(gdf)} registros; columnas: {list(gdf.columns)}")
    if a.aoi or a.bbox:
        west, south, east, north = load_aoi(a.aoi, a.aois)["bbox"] if a.aoi else a.bbox
        gdf = gdf.cx[west:east, south:north]
        print(f"[wells] {len(gdf)} dentro de la caja ({west:.3f}, {south:.3f}, {east:.3f}, {north:.3f})")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(a.out, driver="GPKG")
    print(f"[wells] escrito {a.out}")


def _cmd_build_patches(a):
    from .build import build_patches

    summary = build_patches(
        a.scene, a.wells, a.out, size=a.size, stride=a.stride, radius_m=a.radius_m,
        manual_path=a.manual, labeled_path=a.labeled, block_px=a.block_px, seed=a.seed,
    )
    print(json.dumps(summary, indent=2))


def _cmd_train(a):
    from .train import load_config, run_training

    run_training(load_config(a.config, a.set))


def _cmd_evaluate(a):
    from .evaluate import run_evaluation

    res = run_evaluation(a.run, a.split, a.threshold)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=2))


def _cmd_predict(a):
    from .infer import run_prediction

    print(json.dumps(run_prediction(a.run, a.scene, a.out, a.threshold, a.min_area_m2), indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vmseg", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="genera una escena sintética para probar el pipeline sin datos reales")
    s.add_argument("--out", default="data/synthetic"); s.add_argument("--name", default="synthetic"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--size", type=int, default=768, help="lado de la escena en píxeles"); s.add_argument("--n-pads", type=int, default=12)
    s.set_defaults(fn=_cmd_synth)

    s = sub.add_parser("fetch-s2", help="descarga un compuesto Sentinel-2 de un AOI (ver configs/aois.yaml)")
    s.add_argument("--aoi", required=True); s.add_argument("--aois", default="configs/aois.yaml")
    s.add_argument("--start", required=True); s.add_argument("--end", required=True)
    s.add_argument("--max-cloud", type=float, default=30.0); s.add_argument("--out")
    s.set_defaults(fn=_cmd_fetch_s2)

    s = sub.add_parser("prepare-wells", help="recorta el shapefile oficial de pozos a un AOI")
    s.add_argument("--input", required=True, help="ruta al .shp / .zip / .gpkg de pozos")
    s.add_argument("--aoi"); s.add_argument("--aois", default="configs/aois.yaml")
    s.add_argument("--bbox", type=float, nargs=4, metavar=("OESTE", "SUR", "ESTE", "NORTE"))
    s.add_argument("--out", default="data/interim/wells.gpkg")
    s.set_defaults(fn=_cmd_prepare_wells)

    s = sub.add_parser("build-patches", help="arma parches + etiquetas débiles (+ manuales) + splits espaciales")
    s.add_argument("--scene", nargs="+", required=True, help="uno o más GeoTIFF de Sentinel-2")
    s.add_argument("--wells", required=True)
    s.add_argument("--out", default="data/processed")
    s.add_argument("--size", type=int, default=256); s.add_argument("--stride", type=int)
    s.add_argument("--radius-m", type=float, default=DEFAULT_RADIUS_M)
    s.add_argument("--manual", help="polígonos de locaciones dibujados a mano")
    s.add_argument("--labeled", help="zonas etiquetadas exhaustivamente (obligatorio si hay --manual)")
    s.add_argument("--block-px", type=int, default=512); s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=_cmd_build_patches)

    s = sub.add_parser("train", help="entrena la U-Net")
    s.add_argument("--config", default="configs/default.yaml")
    s.add_argument("--set", action="append", default=[], metavar="clave=valor", help="pisa un valor de la config")
    s.set_defaults(fn=_cmd_train)

    s = sub.add_parser("evaluate", help="métricas por píxel y por objeto en val/test")
    s.add_argument("--run", required=True); s.add_argument("--split", default="test")
    s.add_argument("--threshold", type=float)
    s.set_defaults(fn=_cmd_evaluate)

    s = sub.add_parser("predict", help="segmenta una escena completa y exporta polígonos")
    s.add_argument("--run", required=True); s.add_argument("--scene", required=True)
    s.add_argument("--out", default="runs/predictions.gpkg")
    s.add_argument("--threshold", type=float, default=0.5); s.add_argument("--min-area-m2", type=float, default=2000.0)
    s.set_defaults(fn=_cmd_predict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
