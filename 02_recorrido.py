# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow"]
# ///
"""Fase 1: modelo de recorrido probabilístico (Schlegel et al. 2021) en el cerebro central de FlyWire v783.

Uso:  uv run 02_recorrido.py [--max-corridas 10000] [--lote 100] [--regla arista|neurona] [--solo olfativas] [--procesos N]
      [--alcance central|completo] [--max-pasos 15]
Lee:  datos/proofread_connections_783.feather  y  datos/neuron_annotations.tsv
Deja: resultados/02_capas.parquet      capa media por neurona y modalidad
      resultados/02_percentiles.parquet percentil del rango por modalidad (0 = la más cercana)
      resultados/02_alcance.parquet    fracción de corridas en que se alcanzó cada neurona
      resultados/02_resumen.json       corridas, convergencia y tiempos por modalidad

El modelo
---------
Se parte de las neuronas semilla (capa 1). En cada paso, cada arista i -> j que sale
de una neurona ya recorrida i hacia una neurona j no recorrida se prueba por separado
con probabilidad p_ij = min(1, A_ij / 0.3), donde A_ij es la fracción de las entradas de
j que viene de i. j se recorre si al menos una de sus aristas tiene éxito, y su capa es
el paso en que pasa eso. Una neurona que falla se vuelve a sortear en el paso siguiente.
Se para cuando ya no quedan aristas candidatas o al llegar a max_pasos.

Así lo implementa navis.models.TraversalModel, la herramienta de los autores del método.
Para la neurona j, que todas sus aristas fallen tiene probabilidad prod_i (1 - p_ij), así
que basta un sorteo por neurona con P_j = 1 - exp(sum_i log(1 - p_ij)): es la misma
distribución que sortear arista por arista, pero en una sola multiplicación dispersa.

La regla «por neurona» (p_j = min(1, sum_i A_ij / 0.3)) queda como opción para medir
cuánto cambia el resultado (--regla neurona).
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse
from concurrent.futures import ProcessPoolExecutor

AQUI = Path(__file__).parent
DATOS = AQUI / "datos"
RESULTADOS = AQUI / "resultados"
UMBRAL_SINAPSIS = 5
UMBRAL_P = 0.3
MAX_PASOS = 30


# ----------------------------------------------------------------- modelo

def normalizar(W):
    """A[i, j] = W[i, j] / entradas totales de j. Columnas suman 1 (o 0)."""
    W = sparse.csc_matrix(W, dtype=np.float64)
    entradas = np.asarray(W.sum(axis=0)).ravel()
    inv = np.divide(1.0, entradas, out=np.zeros_like(entradas), where=entradas > 0)
    return sparse.csr_matrix(W @ sparse.diags(inv))


def preparar(A, umbral=UMBRAL_P, regla="arista"):
    """Matrices transpuestas (fila = neurona destino) que usa cada paso."""
    A = sparse.csr_matrix(A, dtype=np.float64)
    if regla == "neurona":
        return {"regla": regla, "AT": sparse.csr_matrix(A.T, dtype=np.float32), "umbral": umbral}
    p = A.copy()
    p.data = np.minimum(1.0, p.data / umbral)
    seguras = p.copy()
    seguras.data = (seguras.data >= 1.0).astype(np.float32)
    seguras.eliminate_zeros()
    logq = p.copy()
    logq.data = np.where(logq.data < 1.0, np.log1p(-np.minimum(logq.data, 1 - 1e-12)), 0.0)
    logq.eliminate_zeros()
    # candidatas: hay al menos una arista con p > 0
    cand = A.copy()
    cand.data = (cand.data > 0).astype(np.float32)
    return {"regla": regla,
            "logqT": sparse.csr_matrix(logq.T, dtype=np.float32),
            "segurasT": sparse.csr_matrix(seguras.T),
            "candT": sparse.csr_matrix(cand.T)}


def recorrer_lote(M, semillas, corridas, rng, max_pasos=MAX_PASOS):
    """Capas de `corridas` recorridos independientes. Devuelve int16 (n, corridas); 0 = no alcanzada."""
    clave = "AT" if M["regla"] == "neurona" else "candT"
    n = M[clave].shape[0]
    capas = np.zeros((n, corridas), dtype=np.int16)
    capas[semillas, :] = 1
    X = np.zeros((n, corridas), dtype=np.float32)
    X[semillas, :] = 1.0
    for t in range(2, max_pasos + 1):
        libres = capas == 0
        if M["regla"] == "neurona":
            P = np.minimum(1.0, (M["AT"] @ X) / M["umbral"])
            pendientes = libres & (P > 0)
        else:
            pendientes = libres & ((M["candT"] @ X) > 0)
            if pendientes.any():
                P = -np.expm1(M["logqT"] @ X)          # 1 - prod(1 - p) de las aristas no seguras
                P[(M["segurasT"] @ X) > 0] = 1.0        # alguna arista con p = 1
        if not pendientes.any():
            break                                       # ya no hay aristas candidatas
        nuevas = pendientes & (rng.random((n, corridas), dtype=np.float32) < P)
        capas[nuevas] = t
        X[nuevas] = 1.0
    return capas


def recorrer(A, semillas, rng, umbral=UMBRAL_P, max_pasos=50, regla="arista"):
    """Una corrida (la interfaz de pruebas_recorrido.py)."""
    M = preparar(A, umbral, regla)
    return recorrer_lote(M, np.asarray(semillas), 1, rng, max_pasos)[:, 0].astype(np.int64)


def correr_modalidad(M, semillas, rng, lote=100, max_corridas=10_000, tol=0.01, cuantil=0.99, max_pasos=MAX_PASOS):
    """Lotes de corridas hasta que la capa media deje de cambiar (cambio < tol en el 99 % de las neuronas)."""
    n = M["candT" if M["regla"] == "arista" else "AT"].shape[0]
    suma = np.zeros(n)
    cuenta = np.zeros(n)
    media_ant = None
    hechas = 0
    cambio = np.inf
    while hechas < max_corridas:
        c = recorrer_lote(M, semillas, lote, rng, max_pasos)
        suma += c.sum(axis=1)
        cuenta += (c > 0).sum(axis=1)
        hechas += lote
        media = np.divide(suma, cuenta, out=np.full(n, np.nan), where=cuenta > 0)
        if media_ant is not None:
            ok = ~np.isnan(media) & ~np.isnan(media_ant)
            cambio = float(np.quantile(np.abs(media[ok] - media_ant[ok]), cuantil)) if ok.any() else 0.0
            if hechas >= 5 * lote and cambio < tol:
                break
        media_ant = media
    return media, cuenta / hechas, hechas, cambio


# ----------------------------------------------------------------- datos

def cargar(alcance="central", umbral_sinapsis=UMBRAL_SINAPSIS):
    import pyarrow.feather as feather
    for f in ("proofread_connections_783.feather", "neuron_annotations.tsv"):
        if not (DATOS / f).exists():
            sys.exit(f"Falta datos/{f}. Corre primero: uv run 00_descargar.py")
    t = feather.read_table(DATOS / "proofread_connections_783.feather",
                           columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"]).to_pandas()
    pares = t.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False, sort=False)["syn_count"].sum()
    pares = pares[pares["syn_count"] >= umbral_sinapsis]
    anot = pd.read_csv(DATOS / "neuron_annotations.tsv", sep="\t",
                       usecols=["root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type"])
    # "central": sin lóbulos ópticos (como dicen los Métodos de Dorkenwald et al. 2024).
    # "completo": todo el cerebro; así están calculados los rangos publicados en Zenodo 12588557.
    central = anot if alcance == "completo" else anot[anot["super_class"] != "optic"]
    central = central.reset_index(drop=True)
    indice = pd.Series(np.arange(len(central)), index=central["root_id"].to_numpy())
    pares = pares[pares["pre_pt_root_id"].isin(indice.index) & pares["post_pt_root_id"].isin(indice.index)]
    W = sparse.csr_matrix((pares["syn_count"].to_numpy(np.float64),
                           (indice[pares["pre_pt_root_id"]].to_numpy(), indice[pares["post_pt_root_id"]].to_numpy())),
                          shape=(len(central), len(central)))
    return W, central


def semillas_por_modalidad(anot):
    """Las 12 modalidades de entrada al cerebro central (Dorkenwald et al. 2024; Lin, Yang et al. 2024)."""
    c, s, sup = anot["cell_class"], anot["cell_sub_class"], anot["super_class"]
    mecano = c == "mechanosensory"
    grupos = {
        "olfativas": c == "olfactory",
        "gustativas": c == "gustatory",
        "johnston": mecano & s.isin(["auditory", "wind_gravity"]),
        "cerdas_cabeza": mecano & (s == "head bristle"),
        "cerdas_ojo": mecano & (s == "eye bristle"),
        "taste_peg": mecano & (s == "taste peg"),
        "termosensoriales": c == "thermosensory",
        "higrosensoriales": c == "hygrosensory",
        "proyeccion_visual": sup == "visual_projection",
        "fotorreceptores": (c == "visual") & (s != "ocellar"),
        "ocelos": (c == "visual") & (s == "ocellar"),
        "ascendentes": c == "AN",
    }
    grupos["todas"] = (anot["flow"] == "afferent") | (sup == "visual_projection")
    return {k: np.flatnonzero(v.to_numpy()) for k, v in grupos.items()}


def _trabajo(args):
    """Una modalidad en un proceso aparte (la semilla aleatoria depende solo del nombre)."""
    M, nombre, sem, semilla, lote, max_corridas, max_pasos = args
    t = time.time()
    rng = np.random.default_rng([semilla, sum(map(ord, nombre))])
    media, frac, hechas, cambio = correr_modalidad(M, sem, rng, lote, max_corridas, max_pasos=max_pasos)
    return nombre, media, frac, hechas, cambio, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-corridas", type=int, default=10_000)
    ap.add_argument("--lote", type=int, default=100)
    ap.add_argument("--regla", choices=["arista", "neurona"], default="arista")
    ap.add_argument("--solo", nargs="*", help="modalidades a correr (por defecto, todas)")
    ap.add_argument("--semilla", type=int, default=783)
    ap.add_argument("--alcance", choices=["central", "completo"], default="central")
    ap.add_argument("--umbral-sinapsis", type=int, default=UMBRAL_SINAPSIS, help="sinapsis mínimas por par")
    ap.add_argument("--max-pasos", type=int, default=MAX_PASOS, help="navis usa 15 por defecto")
    ap.add_argument("--procesos", type=int, default=max(1, min(13, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args()

    t0 = time.time()
    W, anot = cargar(args.alcance, args.umbral_sinapsis)
    A = normalizar(W)
    M = preparar(A, regla=args.regla)
    grupos = semillas_por_modalidad(anot)
    if args.solo:
        grupos = {k: v for k, v in grupos.items() if k in args.solo}
    print(f"Cerebro {args.alcance}: {W.shape[0]:,} neuronas, {W.nnz:,} conexiones ({time.time() - t0:.0f} s)", flush=True)

    capas, alcance, resumen = {}, {}, {"regla": args.regla, "umbral_p": UMBRAL_P, "umbral_sinapsis": args.umbral_sinapsis,
                                       "neuronas": int(W.shape[0]), "conexiones": int(W.nnz),
                                       "procesos": args.procesos, "alcance": args.alcance,
                                       "max_pasos": args.max_pasos, "modalidades": {}}
    tareas = [(M, nombre, sem, args.semilla, args.lote, args.max_corridas, args.max_pasos)
              for nombre, sem in grupos.items()]
    with ProcessPoolExecutor(args.procesos) as ex:
        for nombre, media, frac, hechas, cambio, seg in ex.map(_trabajo, tareas):
            capas[nombre], alcance[nombre] = media, frac
            sem = grupos[nombre]
            resumen["modalidades"][nombre] = {"semillas": int(len(sem)), "corridas": int(hechas),
                                              "cambio_final_p99": round(cambio, 4),
                                              "alcanzadas_siempre": int((frac == 1).sum()),
                                              "segundos": round(seg, 1)}
            print(f"{nombre:>18}: {len(sem):6,} semillas · {hechas:6,} corridas · cambio p99 {cambio:.4f} · "
                  f"{seg:5.0f} s", flush=True)

    sufijo = ("" if args.regla == "arista" else "_neurona") + ("" if args.alcance == "central" else "_completo")
    if args.max_pasos != MAX_PASOS:
        sufijo += f"_p{args.max_pasos}"
    if args.umbral_sinapsis != UMBRAL_SINAPSIS:
        sufijo += f"_u{args.umbral_sinapsis}"
    ids = anot[["root_id", "super_class", "cell_class", "cell_type"]]
    capas = pd.concat([ids, pd.DataFrame(capas)], axis=1)
    alcance = pd.concat([ids[["root_id"]], pd.DataFrame(alcance)], axis=1)
    # Percentil del rango (0 = la más cercana). Las semillas quedan fuera del orden, como en el artículo.
    perc = capas[["root_id"]].copy()
    for nombre, sem in grupos.items():
        v = capas[nombre].copy()
        v.iloc[sem] = np.nan
        perc[nombre] = v.rank(pct=True, method="average")
    RESULTADOS.mkdir(exist_ok=True)
    capas.to_parquet(RESULTADOS / f"02_capas{sufijo}.parquet", index=False)
    perc.to_parquet(RESULTADOS / f"02_percentiles{sufijo}.parquet", index=False)
    alcance.to_parquet(RESULTADOS / f"02_alcance{sufijo}.parquet", index=False)
    resumen["segundos_total"] = round(time.time() - t0, 1)
    (RESULTADOS / f"02_resumen{sufijo}.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False),
                                                         encoding="utf-8")
    print(f"Listo en {time.time() - t0:.0f} s → resultados/02_*{sufijo}")


if __name__ == "__main__":
    main()
