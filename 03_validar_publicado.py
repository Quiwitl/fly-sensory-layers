# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow", "matplotlib"]
# ///
"""Fase 2 (H1): comparar las capas de 02_recorrido.py con las publicadas por Dorkenwald et al. 2024.

Uso:  uv run 03_validar_publicado.py [sufijo]     (sufijo de 02_capas, p. ej. _completo_p15)
Lee:  resultados/02_capas.parquet  y  datos/rangos_publicados/*.feather (Zenodo 12588557)
Deja: resultados/03_validacion_publicado.csv   una fila por modalidad
      figuras/03_propio_vs_publicado.png         capa media propia contra publicada

Criterio de H1 (protocolo): Spearman >= 0.9 por modalidad, sin contar las semillas.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AQUI = Path(__file__).parent
PUB = AQUI / "datos" / "rangos_publicados"
# modalidad propia -> sufijo del archivo publicado
PARES = {
    "olfativas": "olfactory", "gustativas": "gustatory", "johnston": "mechanosensory-jo",
    "cerdas_cabeza": "mechanosensory-head-bristle", "cerdas_ojo": "mechanosensory-eye-bristle",
    "taste_peg": "mechanosensory-taste-peg", "termosensoriales": "thermosensory",
    "higrosensoriales": "hygrosensory", "proyeccion_visual": "visual_projection",
    "fotorreceptores": "visual", "ocelos": "ocellar", "ascendentes": "ascending", "todas": "allinputs",
}


def leer_publicado(sufijo):
    d = pd.read_feather(PUB / f"neuron_class_ranking_df_783-{sufijo}-10000.feather")
    if "root_id" not in d.columns:
        for c in ("node", "root_783", "id"):
            if c in d.columns:
                d = d.rename(columns={c: "root_id"})
                break
        else:
            d = d.reset_index().rename(columns={"index": "root_id"})
    d = d[["root_id", "layer_mean"]].copy()
    # Los root_id tienen 18 dígitos: si un lado es uint64 y el otro int64, pandas los cruza como
    # float64 y empareja neuronas distintas. Siempre int64 en los dos lados.
    d["root_id"] = d["root_id"].astype(np.int64)
    return d


def main():
    import sys
    sufijo = sys.argv[1] if len(sys.argv) > 1 else ""   # p. ej. "_completo_p15"
    propio = pd.read_parquet(AQUI / "resultados" / f"02_capas{sufijo}.parquet")
    propio["root_id"] = propio["root_id"].astype(np.int64)
    filas = []
    fig, axs = plt.subplots(3, 5, figsize=(15, 9))
    for ax, (mod, suf) in zip(axs.flat, PARES.items()):
        pub = leer_publicado(suf)
        m = propio[["root_id", mod]].merge(pub, on="root_id", how="outer", indicator=True)
        ambos = m[(m["_merge"] == "both")].dropna(subset=[mod, "layer_mean"])
        sin_semillas = ambos[(ambos[mod] > 1) & (ambos["layer_mean"] > 1)]
        rho = spearmanr(sin_semillas[mod], sin_semillas["layer_mean"]).statistic
        dif = sin_semillas[mod] - sin_semillas["layer_mean"]
        filas.append({
            "modalidad": mod, "archivo_publicado": suf,
            "neuronas_propias": int(propio[mod].notna().sum()), "neuronas_publicadas": int(len(pub)),
            "en_ambos": int(len(ambos)), "solo_propio": int((m["_merge"] == "left_only").sum()),
            "solo_publicado": int((m["_merge"] == "right_only").sum()),
            "semillas_propias": int((ambos[mod] == 1).sum()), "semillas_publicadas": int((ambos["layer_mean"] == 1).sum()),
            "spearman": round(float(rho), 4), "dif_media": round(float(dif.mean()), 3),
            "dif_abs_media": round(float(dif.abs().mean()), 3),
            "dif_abs_p95": round(float(dif.abs().quantile(0.95)), 3),
            "cumple_H1": bool(rho >= 0.9),
        })
        ax.hexbin(sin_semillas["layer_mean"], sin_semillas[mod], gridsize=40, mincnt=1, bins="log", cmap="viridis")
        lim = [1, max(sin_semillas["layer_mean"].max(), sin_semillas[mod].max())]
        ax.plot(lim, lim, color="#B5432A", lw=0.8)
        ax.set_title(f"{mod.replace('_', ' ')}  ρ={rho:.3f}", fontsize=10)
    for ax in axs.flat[len(PARES):]:
        ax.axis("off")
    fig.supxlabel("capa media publicada (Dorkenwald et al. 2024, 10,000 corridas)")
    fig.supylabel("capa media propia (02_recorrido.py)")
    fig.tight_layout()
    (AQUI / "figuras").mkdir(exist_ok=True)
    fig.savefig(AQUI / "figuras" / f"03_propio_vs_publicado{sufijo}.png", dpi=170)

    t = pd.DataFrame(filas)
    t.to_csv(AQUI / "resultados" / f"03_validacion_publicado{sufijo}.csv", index=False)
    print(t.to_string(index=False))
    print(f"\nH1 se cumple en {int(t.cumple_H1.sum())} de {len(t)} modalidades (Spearman >= 0.9)")


if __name__ == "__main__":
    main()
