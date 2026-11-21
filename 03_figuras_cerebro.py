# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "pyarrow", "matplotlib"]
# ///
"""Fase 2: figuras del cerebro, propias al lado de las publicadas.

Uso:  uv run 03_figuras_cerebro.py [sufijo]      (sufijo de 02_capas, p. ej. _completo_p15)
Deja: figuras/03_superclases{sufijo}.png   percentil mediano por superclase y modalidad: propio | publicado
      figuras/03_descendentes{sufijo}.png  capa media de las neuronas descendentes por modalidad
      resultados/03_descendentes{sufijo}.csv
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
PUB = AQUI / "datos" / "rangos_publicados"
PARES = {
    "olfativas": "olfactory", "gustativas": "gustatory", "johnston": "mechanosensory-jo",
    "cerdas_cabeza": "mechanosensory-head-bristle", "cerdas_ojo": "mechanosensory-eye-bristle",
    "taste_peg": "mechanosensory-taste-peg", "termosensoriales": "thermosensory",
    "higrosensoriales": "hygrosensory", "proyeccion_visual": "visual_projection",
    "fotorreceptores": "visual", "ocelos": "ocellar", "ascendentes": "ascending", "todas": "allinputs",
}
SUPER = ["ascending", "visual_projection", "central", "visual_centrifugal", "descending", "motor", "endocrine"]
TINTA, ACENTO = "#1F3A5F", "#B5432A"


def percentil(serie):
    """Percentil del rango sin contar las semillas (capa 1)."""
    v = serie.where(serie > 1)
    return v.rank(pct=True)


def main():
    sufijo = sys.argv[1] if len(sys.argv) > 1 else ""
    propio = pd.read_parquet(AQUI / "resultados" / f"02_capas{sufijo}.parquet")
    propio["root_id"] = propio["root_id"].astype(np.int64)
    anot = pd.read_csv(AQUI / "datos" / "neuron_annotations.tsv", sep="\t", usecols=["root_id", "super_class"])
    anot["root_id"] = anot["root_id"].astype(np.int64)

    med_p, med_q, dn = {}, {}, []
    for mod, suf in PARES.items():
        pub = pd.read_feather(PUB / f"neuron_class_ranking_df_783-{suf}-10000.feather")[["root_id", "layer_mean"]]
        pub["root_id"] = pub["root_id"].astype(np.int64)
        m = anot.merge(propio[["root_id", mod]], on="root_id", how="left").merge(pub, on="root_id", how="left")
        m["p_propio"], m["p_pub"] = percentil(m[mod]), percentil(m["layer_mean"])
        g = m.groupby("super_class")[["p_propio", "p_pub"]].median()
        med_p[mod], med_q[mod] = g["p_propio"], g["p_pub"]
        d = m[m["super_class"] == "descending"]
        dn.append({"modalidad": mod, "propio": d[mod].mean(), "publicado": d["layer_mean"].mean()})

    P = pd.DataFrame(med_p).reindex(SUPER)
    Q = pd.DataFrame(med_q).reindex(SUPER)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
    for ax, T, titulo in ((axs[0], P, "propio"), (axs[1], Q, "publicado (Dorkenwald et al. 2024)")):
        im = ax.imshow(T.to_numpy(), cmap="viridis_r", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(T.shape[1]), [c.replace("_", " ") for c in T.columns], rotation=40, ha="right")
        ax.set_yticks(range(T.shape[0]), [s.replace("_", " ") for s in T.index])
        for i in range(T.shape[0]):
            for j in range(T.shape[1]):
                v = T.iat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v > 0.55 else "black")
        ax.set_title(titulo)
    fig.colorbar(im, ax=axs, label="percentil mediano (0 = alcanzada primero)", shrink=0.9)
    fig.suptitle("FlyWire v783: qué tan pronto llega cada modalidad a cada superclase", y=1.0)
    (AQUI / "figuras").mkdir(exist_ok=True)
    fig.savefig(AQUI / "figuras" / f"03_superclases{sufijo}.png", dpi=170, bbox_inches="tight")

    D = pd.DataFrame(dn).set_index("modalidad")
    D.round(3).to_csv(AQUI / "resultados" / f"03_descendentes{sufijo}.csv")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(D))
    ax.bar(x - 0.2, D["propio"], 0.4, color=TINTA, label="propio")
    ax.bar(x + 0.2, D["publicado"], 0.4, color=ACENTO, label="publicado")
    ax.set_xticks(x, [i.replace("_", " ") for i in D.index], rotation=40, ha="right")
    ax.set_ylabel("capa media de las descendentes")
    ax.set_title("Neuronas descendentes: en cuántas capas les llega cada modalidad")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(AQUI / "figuras" / f"03_descendentes{sufijo}.png", dpi=170)
    print(D.round(2).to_string())
    print("\ndiferencia media de percentil por superclase (propio - publicado):")
    print((P - Q).round(3).to_string())


if __name__ == "__main__":
    main()
