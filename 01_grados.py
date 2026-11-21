# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "matplotlib", "pyarrow"]
# ///
"""Fase 0: cargar el conectoma de FlyWire (v783) y graficar la distribución de grados.

Uso:  uv run 01_grados.py        (antes: uv run 00_descargar.py)
Lee:  datos/proofread_connections_783.feather  y  datos/neuron_annotations.tsv
Deja: figuras/01_grados.png, resultados/01_resumen.json y un resumen en la terminal.

Sirve para comprobar que los datos se cargaron bien antes de analizar nada:
las cifras deben parecerse a las publicadas (139,255 neuronas; unos 2.7 millones
de conexiones con 5 sinapsis o más).
"""
from pathlib import Path
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import sparse

AQUI = Path(__file__).parent
DATOS = AQUI / "datos"
FIGURAS = AQUI / "figuras"
RESULTADOS = AQUI / "resultados"
UMBRAL = 5  # sinapsis mínimas por par de neuronas (el mismo que usa FlyWire)


def exigir(ruta):
    if not ruta.exists():
        sys.exit(f"Falta {ruta}. Corre primero: uv run 00_descargar.py")
    return ruta


def cargar_pares():
    """Una fila por par (pre, post) con el total de sinapsis de todos los neuropilos."""
    t = feather.read_table(exigir(DATOS / "proofread_connections_783.feather"),
                           columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    filas = t.num_rows
    df = t.to_pandas()
    pares = df.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False, sort=False)["syn_count"].sum()
    return pares.rename(columns={"pre_pt_root_id": "pre", "post_pt_root_id": "post"}), filas


def main():
    t0 = time.time()
    pares, filas = cargar_pares()
    anot = pd.read_csv(exigir(DATOS / "neuron_annotations.tsv"), sep="\t",
                       usecols=["root_id", "flow", "super_class"])

    fuertes = pares[pares["syn_count"] >= UMBRAL]
    ids = np.union1d(anot["root_id"].to_numpy(), np.union1d(pares["pre"], pares["post"]))
    indice = pd.Series(np.arange(len(ids)), index=ids)
    fila = indice[fuertes["pre"]].to_numpy()
    col = indice[fuertes["post"]].to_numpy()
    W = sparse.csr_matrix((fuertes["syn_count"].to_numpy(), (fila, col)), shape=(len(ids), len(ids)))

    sale = np.diff(W.indptr)            # a cuántas neuronas manda
    entra = np.diff(W.tocsc().indptr)   # de cuántas recibe
    sin_anotar = len(np.setdiff1d(ids, anot["root_id"].to_numpy()))

    resumen = {
        "filas_por_neuropilo": int(filas),
        "pares_sin_umbral": int(len(pares)),
        "sinapsis_sin_umbral": int(pares["syn_count"].sum()),
        "umbral": UMBRAL,
        "neuronas": int(len(ids)),
        "neuronas_anotadas": int(len(anot)),
        "neuronas_sin_anotacion": int(sin_anotar),
        "conexiones": int(W.nnz),
        "sinapsis": int(W.sum()),
        "grado_medio": round(W.nnz / len(ids), 2),
        "grado_max_salida": int(sale.max()),
        "grado_max_entrada": int(entra.max()),
        "neuronas_sin_conexion": int(((sale + entra) == 0).sum()),
        "segundos": round(time.time() - t0, 1),
    }
    for k, v in resumen.items():
        print(f"{k:>24}: {v:,}" if isinstance(v, int) else f"{k:>24}: {v}")
    print("\nNeuronas por superclase:")
    print(anot["super_class"].value_counts().to_string())

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "01_resumen.json").write_text(json.dumps(resumen, indent=1), encoding="utf-8")

    FIGURAS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for grados, nombre in ((entra, "entrada"), (sale, "salida")):
        g = np.sort(grados[grados > 0])
        ccdf = 1.0 - np.arange(len(g)) / len(g)   # fracción de neuronas con grado >= k
        ax.loglog(g, ccdf, label=f"grado de {nombre}")
    ax.set_xlabel("grado k (neuronas conectadas, ≥5 sinapsis)")
    ax.set_ylabel("fracción de neuronas con grado ≥ k")
    ax.set_title("FlyWire v783: distribución de grados")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURAS / "01_grados.png", dpi=200)
    print(f"\nFigura: {FIGURAS / '01_grados.png'}  ({time.time() - t0:.0f} s en total)")


if __name__ == "__main__":
    main()
