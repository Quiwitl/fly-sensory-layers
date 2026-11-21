# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow", "matplotlib"]
# ///
"""Fase 1: resumen y comprobaciones de las capas que calculó 02_recorrido.py.

Uso:  uv run 02_resumen.py
Deja: resultados/02_capas_por_modalidad.csv   neuronas por capa redondeada (1–10) y modalidad
      resultados/02_percentil_superclase.csv  percentil mediano por superclase y modalidad
      figuras/02_percentil_superclase.png

Comprobaciones contra lo publicado (Dorkenwald et al. 2024):
  1. Neuronas por capa desde las gustativas (su fig. 5 da, para la versión del preprint,
     147 / 498 / 1,205 / 2,439 / 7,149 en las capas 1–5).
  2. Células de Kenyon: un grupo lo alcanzan pronto las olfativas y otro las neuronas de
     proyección visual (los tipos visuales KCg-d y KCab-p).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
R = AQUI / "resultados"
ORDEN_SUPER = ["sensory", "ascending", "sensory_ascending", "visual_projection", "central",
               "visual_centrifugal", "descending", "motor", "endocrine"]


def main():
    capas = pd.read_parquet(R / "02_capas.parquet")
    perc = pd.read_parquet(R / "02_percentiles.parquet")
    mods = [c for c in perc.columns if c != "root_id"]

    # 1. neuronas por capa redondeada
    tabla = pd.DataFrame({m: capas[m].round().value_counts().reindex(range(1, 11), fill_value=0) for m in mods})
    tabla.index.name = "capa"
    tabla.to_csv(R / "02_capas_por_modalidad.csv")
    print("Neuronas por capa redondeada (columnas: modalidades)")
    print(tabla.to_string())
    print("\nGustativas, capas 1–5 · publicado (preprint): 147 / 498 / 1,205 / 2,439 / 7,149")
    print("                        · aquí:               " + " / ".join(f"{v:,}" for v in tabla['gustativas'].iloc[:5]))

    # 2. percentil mediano por superclase
    p = perc.merge(capas[["root_id", "super_class", "cell_class", "cell_type"]], on="root_id")
    sup = p.groupby("super_class")[mods].median().reindex([s for s in ORDEN_SUPER if s in set(p.super_class)])
    sup.to_csv(R / "02_percentil_superclase.csv")
    print("\nPercentil mediano por superclase (0 = alcanzada primero)")
    print(sup.round(2).to_string())

    # 3. células de Kenyon
    kc = p[p["cell_class"] == "Kenyon_Cell"].copy()
    if len(kc):
        kc["visual"] = kc["cell_type"].fillna("").str.match(r"KCg-d|KCab-p")
        g = kc.groupby("visual")[["olfativas", "proyeccion_visual"]].median()
        g.index = ["KC olfativas (resto)", "KC visuales (KCg-d, KCab-p)"][: len(g)]
        print("\nCélulas de Kenyon: percentil mediano")
        print(g.round(2).to_string())

    # 4. descendentes
    dn = capas[capas["super_class"] == "descending"][mods].mean()
    print("\nCapa media de las neuronas descendentes por modalidad")
    print(dn.round(2).to_string())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    im = ax.imshow(sup.to_numpy(), cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mods)), [m.replace("_", " ") for m in mods], rotation=40, ha="right")
    ax.set_yticks(range(len(sup)), [s.replace("_", " ") for s in sup.index])
    for i in range(sup.shape[0]):
        for j in range(sup.shape[1]):
            v = sup.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="percentil mediano (0 = primero)")
    ax.set_title("FlyWire v783, cerebro central: qué tan pronto llega cada modalidad a cada superclase")
    fig.tight_layout()
    (AQUI / "figuras").mkdir(exist_ok=True)
    fig.savefig(AQUI / "figuras" / "02_percentil_superclase.png", dpi=200)


if __name__ == "__main__":
    main()
