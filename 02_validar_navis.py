# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow", "navis"]
# ///
"""Fase 1: comparar 02_recorrido.py con navis.models.TraversalModel en un subgrafo real.

Uso:  uv run 02_validar_navis.py
Subgrafo: las semillas gustativas y las neuronas a 1–2 conexiones de ellas (hasta ~1500,
quedándose con las de más sinapsis desde el grupo anterior). Los dos modelos reciben la
misma matriz de fracciones de entrada, el mismo umbral (0.3) y el mismo tope de 15 pasos.
Deja: resultados/02_validacion_navis.json y .csv

Resultado (20 nov 2026, navis 1.12.0, 2000 corridas): en las neuronas que no son semilla la
diferencia media es ~0.001 capas y la máxima ~0.1 (ruido de muestreo); Spearman 0.987 contando
todo. Las únicas diferencias grandes son semillas sin aristas de salida dentro del subgrafo:
navis las descarta como semillas y luego las alcanza como neuronas normales (capa ~3).
"""
from pathlib import Path
import importlib.util
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

AQUI = Path(__file__).parent
spec = importlib.util.spec_from_file_location("recorrido", AQUI / "02_recorrido.py")
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

CORRIDAS = 2000
MAX_PASOS = 15  # el valor por defecto de navis


def subgrafo(W, semillas, por_capa=600):
    W = sparse.csr_matrix(W)
    dentro = list(semillas)
    frontera = np.asarray(semillas)
    for _ in range(2):
        fuerza = np.asarray(W[frontera].sum(axis=0)).ravel()
        fuerza[dentro] = 0
        nuevos = np.argsort(fuerza)[::-1][:por_capa]
        nuevos = nuevos[fuerza[nuevos] > 0]
        dentro += list(nuevos)
        frontera = nuevos
    return np.array(dentro)


def main():
    import navis
    from navis.models import TraversalModel

    W, anot = rec.cargar()
    sem_global = rec.semillas_por_modalidad(anot)["gustativas"]
    nodos = subgrafo(W, sem_global)
    Ws = sparse.csr_matrix(W)[nodos][:, nodos]
    A = rec.normalizar(Ws)
    semillas = np.arange(len(sem_global))  # las semillas van primero en `nodos`
    print(f"Subgrafo: {len(nodos)} neuronas, {A.nnz} aristas, {len(semillas)} semillas")

    # --- 02_recorrido.py
    t = time.time()
    M = rec.preparar(A)
    capas = rec.recorrer_lote(M, semillas, CORRIDAS, np.random.default_rng(0), max_pasos=MAX_PASOS)
    cuenta = (capas > 0).sum(axis=1)
    propio = np.divide(capas.sum(axis=1), cuenta, out=np.full(len(nodos), np.nan), where=cuenta > 0)
    t_propio = time.time() - t

    # --- navis
    Ac = A.tocoo()
    aristas = pd.DataFrame({"source": Ac.row, "target": Ac.col, "weight": Ac.data})
    t = time.time()
    modelo = TraversalModel(aristas, seeds=semillas.tolist(), weights="weight", max_steps=MAX_PASOS)
    modelo.run(iterations=CORRIDAS)
    t_navis = time.time() - t
    resumen_navis = modelo.summary.set_index("node")["layer_mean"] if "node" in modelo.summary.columns \
        else modelo.summary["layer_mean"]
    de_navis = np.full(len(nodos), np.nan)
    de_navis[resumen_navis.index.to_numpy(int)] = resumen_navis.to_numpy()

    alc_propio = cuenta / CORRIDAS
    ok = ~np.isnan(propio) & ~np.isnan(de_navis)
    dif = np.abs(propio[ok] - de_navis[ok])
    rho = spearmanr(propio[ok], de_navis[ok]).statistic
    res = {
        "neuronas": int(len(nodos)), "aristas": int(A.nnz), "semillas": int(len(semillas)),
        "corridas": CORRIDAS, "max_pasos": MAX_PASOS,
        "comparadas": int(ok.sum()),
        "alcanzadas_solo_propio": int((~np.isnan(propio) & np.isnan(de_navis)).sum()),
        "alcanzadas_solo_navis": int((np.isnan(propio) & ~np.isnan(de_navis)).sum()),
        "diferencia_media": round(float(dif.mean()), 4),
        "diferencia_p99": round(float(np.quantile(dif, 0.99)), 4),
        "diferencia_max": round(float(dif.max()), 4),
        "spearman": round(float(rho), 5),
        "no_semillas_diferencia_media": round(float(np.abs(propio - de_navis)[ok & ~np.isin(np.arange(len(nodos)), semillas)].mean()), 4),
        "no_semillas_diferencia_max": round(float(np.abs(propio - de_navis)[ok & ~np.isin(np.arange(len(nodos)), semillas)].max()), 4),
        "nota": ("Las diferencias grandes son semillas sin aristas de salida en el subgrafo: navis las quita "
                 "de las semillas (TraversalModel.__init__) y las alcanza como neuronas normales."),
        "diferencia_p99_alcance_90": round(float(np.quantile(
            np.abs(propio - de_navis)[ok & (alc_propio >= 0.9)], 0.99)), 4),
        "neuronas_alcance_90": int((ok & (alc_propio >= 0.9)).sum()),
        "diferencia_media_con_signo": round(float((propio[ok] - de_navis[ok]).mean()), 4),
        "alcance_de_las_10_mas_distintas": [round(float(x), 3) for x in
                                             alc_propio[np.flatnonzero(ok)[np.argsort(dif)[::-1][:10]]]],
        "segundos_propio": round(t_propio, 1), "segundos_navis": round(t_navis, 1),
        "navis_version": navis.__version__,
    }
    (AQUI / "resultados").mkdir(exist_ok=True)
    Acsc = sparse.csc_matrix(A)
    entrada_max = np.asarray(Acsc.max(axis=0).todense()).ravel()
    grado_ent = np.diff(Acsc.indptr)
    desde_semillas = np.asarray(Acsc[semillas].sum(axis=0)).ravel()
    pd.DataFrame({"nodo": np.arange(len(nodos)), "root_id": anot["root_id"].to_numpy()[nodos],
                  "propio": propio, "navis": de_navis, "alcance": alc_propio,
                  "grado_entrada": grado_ent, "entrada_max": entrada_max,
                  "fraccion_desde_semillas": desde_semillas,
                  "es_semilla": np.isin(np.arange(len(nodos)), semillas)}
                 ).to_csv(AQUI / "resultados" / "02_validacion_navis.csv", index=False)
    for k, v in res.items():
        print(f"{k:>24}: {v}")
    (AQUI / "resultados" / "02_validacion_navis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
