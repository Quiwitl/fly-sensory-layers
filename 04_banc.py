# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow"]
# ///
"""Fase 3: modelo de recorrido en el BANC v888 (cerebro y cordón), para H2 y H3 del PLAN.

Uso:  uv run 04_banc.py --revisar                 # solo cuenta neuronas, conexiones, semillas y blancos
      uv run 04_banc.py --h2 [--corridas 1000]   # cerebro sin cordón, tope 15 (H2)
      uv run 04_banc.py --h3 [--corridas 1000]   # sistema nervioso completo, tope 30 y 15 (H3)
      uv run 04_banc.py --h3 --azar 1 --corridas 100   # control: grafo al azar que conserva grados
      <python del entorno fly-gpu> 04_banc.py --h3 --azar 2 --gpu --lote 500   # lo mismo en la GPU
      opciones: --solo olfativas johnston · --procesos N · --lote 100 · --etiqueta _prueba
                --region-h2 cerebro|central   (cerebro = central_brain + optic_lobe, lo que pide el PLAN)
Lee:  datos/banc/banc_888_edgelist_simple_v2.feather  (pre, post, count; sin umbral)
      datos/banc/banc_888_meta.feather                 (se une por banc_888_id)
Deja: resultados/04_<modo><etiqueta>_capas.parquet     capa media por neurona y modalidad (NaN = nunca alcanzada)
      resultados/04_<modo><etiqueta>_alcance.parquet   fracción de corridas que alcanzó cada neurona
      resultados/04_h3<etiqueta>_capas_p15.parquet y _alcance_p15.parquet  lo mismo con tope 15 (H3)
      resultados/04_<modo><etiqueta>_blancos.npz       capa de cada motoneurona en cada corrida (0 = no alcanzada)
      resultados/04_<modo><etiqueta>_resumen.json      conteos, corridas, convergencia y tiempos

El modelo es el de 02_recorrido.py (regla por arista con reintento, como navis), sin umbral
de sinapsis: la fracción de entrada se calcula dentro del subgrafo de cada modo.
Los id del BANC vienen como texto: se convierten a int64 antes de cruzar tablas (ver la Fase 2).
Con tope 30, la corrida hasta el paso 15 es idéntica a una con tope 15, así que la versión
con tope 15 de H3 sale de las mismas corridas (capas > 15 cuentan como no alcanzadas).
"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor

AQUI = Path(__file__).parent
BANC = AQUI / "datos" / "banc"
RESULTADOS = AQUI / "resultados"

spec = importlib.util.spec_from_file_location("recorrido", AQUI / "02_recorrido.py")
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

NO_NEURONAS = ["glia", "not_a_neuron", "trachea"]
SENSORIALES = ["sensory", "sensory_ascending", "sensory_descending"]
# Mecanosensoriales de pata: las clases del BANC que son mecanorreceptores.
# Fuera: gustativas, multidendríticas (mezclan nocicepción), huérfanas y sin clase.
MECANO = ["bristle_neuron", "campaniform_sensillum_neuron", "chordotonal_organ_neuron",
          "hair_plate_neuron", "taste_bristle_tactile_neuron"]
GUSTATIVAS = ["taste_bristle_gustatory_neuron", "taste_peg_gustatory_neuron",
              "internal_taste_sensillum_gustatory_neuron"]
PATAS = {"T1": "front_leg", "T2": "middle_leg", "T3": "hind_leg"}
BLANCOS = {"MN_T1": "front_leg", "MN_T2": "middle_leg", "MN_T3": "hind_leg", "MN_ala": "wing"}
REGIONES_H2 = {"cerebro": ["central_brain", "optic_lobe"], "central": ["central_brain"]}
COLUMNAS = ["banc_888_id", "region", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
            "fafb_cell_type", "fafb_match", "body_part_sensory", "body_part_effector"]


# ----------------------------------------------------------------- datos

def a_int64(col):
    """Texto -> int64 con pyarrow (los id del BANC tienen 18 dígitos; float64 los confunde)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    return pc.cast(col, pa.int64()).to_numpy(zero_copy_only=False)


def cargar(modo, region_h2="cerebro"):
    import pyarrow.feather as feather
    for f in ("banc_888_edgelist_simple_v2.feather", "banc_888_meta.feather"):
        if not (BANC / f).exists():
            sys.exit(f"Falta datos/banc/{f}. Corre primero: uv run 00_descargar.py")
    meta = feather.read_table(BANC / "banc_888_meta.feather", columns=COLUMNAS)
    ids = a_int64(meta["banc_888_id"])
    meta = meta.to_pandas()
    meta["banc_888_id"] = ids
    cuentas = {"meta_filas": int(len(meta))}
    meta = meta.drop_duplicates("banc_888_id")
    quitar = meta["super_class"].isna() | meta["super_class"].isin(NO_NEURONAS)
    cuentas["quitadas_no_neurona_o_sin_clase"] = int(quitar.sum())
    meta = meta[~quitar]
    if modo == "h2":
        meta = meta[meta["region"].isin(REGIONES_H2[region_h2])]
    meta = meta.reset_index(drop=True)

    t = feather.read_table(BANC / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post", "count"])
    pre, post = a_int64(t["pre"]), a_int64(t["post"])
    cuenta = t["count"].to_numpy().astype(np.float64)
    cuentas["pares_archivo"] = int(len(pre))
    indice = pd.Index(meta["banc_888_id"].to_numpy())
    i, j = indice.get_indexer(pre), indice.get_indexer(post)
    ok = (i >= 0) & (j >= 0)
    cuentas["pares_dentro"] = int(ok.sum())
    return meta, i[ok], j[ok], cuenta[ok], cuentas


def matriz(i, j, w, n):
    return sparse.csr_matrix((w, (i, j)), shape=(n, n))   # suma pares repetidos


def aleatorizar(i, j, n, rondas, rng):
    """Intercambio de aristas que conserva el grado de entrada y de salida de cada neurona:
    (a->b, c->d) -> (a->d, c->b); el peso se queda con su neurona de salida.
    Vectorizado: en cada ronda se emparejan todas las aristas al azar y se aceptan los
    intercambios que no crean autapsis ni aristas repetidas. Las autapsis originales no se tocan."""
    j = j.copy()
    movibles = np.flatnonzero(i != j)
    aceptados = 0
    for _ in range(rondas):
        claves = np.sort(i.astype(np.int64) * n + j)
        p = rng.permutation(movibles)
        h = len(p) // 2
        A, B = p[:h], p[h:2 * h]
        jA, jB = j[B], j[A]
        kA = i[A].astype(np.int64) * n + jA
        kB = i[B].astype(np.int64) * n + jB
        ok = (i[A] != jA) & (i[B] != jB) & (jA != j[A])
        ok &= ~np.isin(kA, claves, assume_unique=False) & ~np.isin(kB, claves, assume_unique=False)
        # que las aristas nuevas no se repitan entre sí
        todas = np.concatenate([kA[ok], kB[ok]])
        u, inv, cnt = np.unique(todas, return_inverse=True, return_counts=True)
        rep = cnt[inv] > 1
        malos = np.zeros(h, bool)
        idx = np.flatnonzero(ok)
        malos[idx[rep[:len(idx)]]] = True
        malos[idx[rep[len(idx):]]] = True
        ok &= ~malos
        j[A[ok]] = jA[ok]
        j[B[ok]] = jB[ok]
        aceptados += int(ok.sum())
    return j, aceptados


def semillas_y_blancos(meta, modo):
    c = meta["cell_class"].fillna("")
    s = meta["cell_sub_class"].fillna("")
    sup = meta["super_class"].fillna("")
    reg = meta["region"].fillna("")
    sens_parte = meta["body_part_sensory"].fillna("")
    efec = meta["body_part_effector"].fillna("")
    grupos = {
        "olfativas": c == "olfactory_receptor_neuron",
        "gustativas": c.isin(GUSTATIVAS) & (reg == "central_brain"),   # las de la cabeza, no las de las patas
        "johnston": s.str.startswith("johnstons_organ"),
        "proyeccion_visual": sup == "visual_projection",
    }
    if modo == "h3":
        for seg, parte in PATAS.items():
            grupos[f"mecano_{seg}"] = sup.isin(SENSORIALES) & (sens_parte == parte) & c.isin(MECANO)
    blancos = {k: (sup == "motor") & (efec == v) for k, v in BLANCOS.items()}
    a_idx = lambda d: {k: np.flatnonzero(v.to_numpy()) for k, v in d.items()}
    return a_idx(grupos), a_idx(blancos)


# ----------------------------------------------------------------- recorrido

_M = None


def _iniciar(M):
    global _M
    _M = M


def _trabajo(args):
    """Una modalidad en un proceso aparte. Siempre `corridas` corridas (el PLAN pide al menos 1,000)."""
    nombre, sem, blancos_idx, semilla, azar, lote, corridas, max_pasos, tope_corto = args
    t0 = time.time()
    rng = np.random.default_rng([semilla, sum(map(ord, nombre)), azar])
    n = _M["candT"].shape[0]
    suma, cuenta = np.zeros(n), np.zeros(n)
    suma_c, cuenta_c = np.zeros(n), np.zeros(n)
    capas_blancos = np.zeros((len(blancos_idx), corridas), dtype=np.int16)
    media_ant, cambio, hechas = None, np.inf, 0
    while hechas < corridas:
        k = min(lote, corridas - hechas)
        c = rec.recorrer_lote(_M, sem, k, rng, max_pasos)
        suma += c.sum(axis=1)
        cuenta += (c > 0).sum(axis=1)
        if tope_corto and tope_corto < max_pasos:
            cc = np.where(c <= tope_corto, c, 0)
            suma_c += cc.sum(axis=1)
            cuenta_c += (cc > 0).sum(axis=1)
        capas_blancos[:, hechas:hechas + k] = c[blancos_idx]
        hechas += k
        media = np.divide(suma, cuenta, out=np.full(n, np.nan), where=cuenta > 0)
        if media_ant is not None:
            ok = ~np.isnan(media) & ~np.isnan(media_ant)
            cambio = float(np.quantile(np.abs(media[ok] - media_ant[ok]), 0.99)) if ok.any() else 0.0
        media_ant = media
    out = {"media": media, "frac": cuenta / hechas}
    if tope_corto and tope_corto < max_pasos:
        out["media_c"] = np.divide(suma_c, cuenta_c, out=np.full(n, np.nan), where=cuenta_c > 0)
        out["frac_c"] = cuenta_c / hechas
    return nombre, out, capas_blancos, hechas, cambio, time.time() - t0


class _Nada:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--h2", action="store_true", help="cerebro sin cordón, tope 15")
    g.add_argument("--h3", action="store_true", help="sistema nervioso completo, tope 30 (y 15)")
    g.add_argument("--revisar", action="store_true", help="solo contar")
    ap.add_argument("--region-h2", choices=list(REGIONES_H2), default="cerebro")
    ap.add_argument("--corridas", type=int, default=1000)
    ap.add_argument("--lote", type=int, default=100)
    ap.add_argument("--solo", nargs="*")
    ap.add_argument("--semilla", type=int, default=888)
    ap.add_argument("--azar", type=int, default=0, help="K > 0: grafo al azar número K (control)")
    ap.add_argument("--rondas", type=int, default=10, help="rondas de intercambio de aristas")
    ap.add_argument("--procesos", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--gpu", action="store_true", help="correr el recorrido en la GPU (CuPy; ver recorrido_gpu.py)")
    args = ap.parse_args()

    t0 = time.time()
    if args.revisar:
        for modo in ("h2", "h3"):
            meta, i, j, w, cuentas = cargar(modo, args.region_h2)
            grupos, blancos = semillas_y_blancos(meta, modo)
            print(f"[{modo}] {len(meta):,} neuronas · {cuentas['pares_dentro']:,} de {cuentas['pares_archivo']:,} pares "
                  f"· quitadas {cuentas['quitadas_no_neurona_o_sin_clase']:,}")
            print("   semillas:", {k: len(v) for k, v in grupos.items()})
            print("   blancos: ", {k: len(v) for k, v in blancos.items()})
            print("   regiones:", meta["region"].value_counts(dropna=False).to_dict())
        print(f"{time.time() - t0:.0f} s")
        return

    modo = "h2" if args.h2 else "h3"
    max_pasos, tope_corto = (15, None) if modo == "h2" else (30, 15)
    meta, i, j, w, cuentas = cargar(modo, args.region_h2)
    n = len(meta)
    sufijo = f"04_{modo}" + (f"_azar{args.azar}" if args.azar else "") + args.etiqueta
    if args.azar:
        rng = np.random.default_rng([args.semilla, args.azar])
        j, aceptados = aleatorizar(i, j, n, args.rondas, rng)
        cuentas["intercambios_aceptados"] = aceptados
        print(f"Grafo al azar {args.azar}: {aceptados:,} intercambios ({time.time() - t0:.0f} s)", flush=True)
    W = matriz(i, j, w, n)
    M = rec.preparar(rec.normalizar(W))
    del W
    grupos, blancos = semillas_y_blancos(meta, modo)
    if args.solo:
        grupos = {k: v for k, v in grupos.items() if k in args.solo}
    todos_blancos = np.concatenate(list(blancos.values())) if blancos else np.zeros(0, int)
    print(f"BANC {modo}: {n:,} neuronas, {M['candT'].nnz:,} conexiones, tope {max_pasos} "
          f"({time.time() - t0:.0f} s)", flush=True)
    print("semillas:", {k: len(v) for k, v in grupos.items()}, "· blancos:", {k: len(v) for k, v in blancos.items()},
          flush=True)

    resumen = {"modo": modo, "region_h2": args.region_h2 if modo == "h2" else None, "max_pasos": max_pasos,
               "tope_corto": tope_corto, "neuronas": n, "conexiones": int(M["candT"].nnz), "azar": args.azar,
               "procesos": args.procesos, "lote": args.lote, **cuentas,
               "blancos": {k: int(len(v)) for k, v in blancos.items()}, "modalidades": {}}
    tareas = [(nombre, sem, todos_blancos, args.semilla, args.azar, args.lote, args.corridas, max_pasos, tope_corto)
              for nombre, sem in grupos.items()]
    res = {}
    if args.gpu:
        sys.path.insert(0, str(AQUI))
        import recorrido_gpu
        G = recorrido_gpu.preparar_gpu(M)
        del M["logqT"], M["segurasT"]
        resultados_iter = (recorrido_gpu.trabajo_gpu(G, *t) for t in tareas)
        resumen["gpu"] = True
    else:
        ex = ProcessPoolExecutor(min(args.procesos, len(tareas)), initializer=_iniciar, initargs=(M,))
        resultados_iter = ex.map(_trabajo, tareas)
    with (ex if not args.gpu else _Nada()):
        for nombre, out, cb, hechas, cambio, seg in resultados_iter:
            res[nombre] = (out, cb)
            resumen["modalidades"][nombre] = {"semillas": int(len(grupos[nombre])), "corridas": int(hechas),
                                              "cambio_ultimo_lote_p99": round(cambio, 4),
                                              "alcanzadas_siempre": int((out["frac"] == 1).sum()),
                                              "segundos": round(seg, 1)}
            print(f"{nombre:>18}: {len(grupos[nombre]):6,} semillas · {hechas:5,} corridas · "
                  f"cambio p99 {cambio:.4f} · {seg:5.0f} s", flush=True)

    RESULTADOS.mkdir(exist_ok=True)
    ids = meta[["banc_888_id", "region", "super_class", "cell_class", "cell_type", "fafb_cell_type", "fafb_match",
                "body_part_effector"]].reset_index(drop=True)
    pd.concat([ids, pd.DataFrame({k: v[0]["media"] for k, v in res.items()})], axis=1) \
        .to_parquet(RESULTADOS / f"{sufijo}_capas.parquet", index=False)
    pd.concat([ids[["banc_888_id"]], pd.DataFrame({k: v[0]["frac"] for k, v in res.items()})], axis=1) \
        .to_parquet(RESULTADOS / f"{sufijo}_alcance.parquet", index=False)
    if tope_corto:
        pd.concat([ids, pd.DataFrame({k: v[0]["media_c"] for k, v in res.items()})], axis=1) \
            .to_parquet(RESULTADOS / f"{sufijo}_capas_p{tope_corto}.parquet", index=False)
        pd.concat([ids[["banc_888_id"]], pd.DataFrame({k: v[0]["frac_c"] for k, v in res.items()})], axis=1) \
            .to_parquet(RESULTADOS / f"{sufijo}_alcance_p{tope_corto}.parquet", index=False)
    np.savez_compressed(RESULTADOS / f"{sufijo}_blancos.npz",
                        banc_888_id=meta["banc_888_id"].to_numpy()[todos_blancos],
                        grupo=np.concatenate([[k] * len(v) for k, v in blancos.items()]) if blancos else np.array([]),
                        **{f"capas_{k}": v[1] for k, v in res.items()})
    resumen["segundos_total"] = round(time.time() - t0, 1)
    (RESULTADOS / f"{sufijo}_resumen.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False),
                                                        encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s -> resultados/{sufijo}_*", flush=True)


if __name__ == "__main__":
    main()
