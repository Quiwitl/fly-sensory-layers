# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow", "matplotlib"]
# ///
"""P4: sensibilidad de las capas en FlyWire al umbral de sinapsis (1 contra 5) y al tope de pasos (15 contra 30).

Uso:  uv run 06_sensibilidad.py
Lee:  resultados/02_capas<sufijo>.parquet de cada corrida que se compara; las tres de P4 se corrieron en la
      GPU con la misma regla de paro (02_recorrido.py --gpu --etiqueta _gpu):
      umbral: _completo_p15_gpu (5 sinapsis) contra _completo_p15_u1_gpu (1 sinapsis), las dos con tope 15
      tope:   _completo_p15_u1_gpu (15 pasos) contra _completo_u1_gpu (30 pasos, el tope por defecto), sin umbral
      ruido:  _completo_p15_u1 (CPU, 1,000 corridas fijas) contra _completo_p15_u1_gpu: misma configuración,
              otro motor y otras corridas; marca cuánto difieren dos corridas sin cambiar nada
Deja: resultados/06_sensibilidad.csv    una fila por comparación y modalidad
      resultados/06_sensibilidad.json   mediana y rango del Spearman por comparación
      figuras/06_sensibilidad.png       Spearman por modalidad en cada comparación

Criterio (PLAN, P4): sin hipótesis; se reporta la mediana y el rango del Spearman por modalidad.
Las semillas (capa 1 en alguna de las dos corridas) no cuentan, igual que en H1.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AQUI = Path(__file__).parent
RES = AQUI / "resultados"
FIG = AQUI / "figuras"
COMPARACIONES = [
    ("umbral", "5 sinapsis contra 1 (tope 15)", "_completo_p15_gpu", "_completo_p15_u1_gpu"),
    ("tope", "15 pasos contra 30 (sin umbral)", "_completo_p15_u1_gpu", "_completo_u1_gpu"),
    ("ruido", "misma configuración, CPU contra GPU", "_completo_p15_u1", "_completo_p15_u1_gpu"),
]
NO_MODALIDAD = {"root_id", "super_class", "cell_class", "cell_type"}


def cargar(sufijo):
    f = RES / f"02_capas{sufijo}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f)
    d["root_id"] = d["root_id"].astype("int64")  # uint64 contra int64 desalinea el merge
    return d


def comparar(a, b, nombre):
    mods = [c for c in a.columns if c not in NO_MODALIDAD and c in b.columns]
    m = a[["root_id"] + mods].merge(b[["root_id"] + mods], on="root_id", suffixes=("_a", "_b"))
    filas = []
    for mod in mods:
        x, y = m[f"{mod}_a"].to_numpy(), m[f"{mod}_b"].to_numpy()
        ambas = ~np.isnan(x) & ~np.isnan(y)
        ok = ambas & (x > 1) & (y > 1)
        rho = spearmanr(x[ok], y[ok]).statistic if ok.sum() > 2 else np.nan
        d = y[ok] - x[ok]
        filas.append({"comparacion": nombre, "modalidad": mod, "neuronas": int(ok.sum()),
                      "solo_en_a": int((~np.isnan(x) & np.isnan(y)).sum()),
                      "solo_en_b": int((np.isnan(x) & ~np.isnan(y)).sum()),
                      "spearman": float(rho), "dif_media_b_menos_a": float(d.mean()),
                      "dif_abs_mediana": float(np.median(np.abs(d)))})
    return pd.DataFrame(filas)


def main():
    tablas, resumen = [], {"criterio": "PLAN.md, P4: sin hipótesis; mediana y rango del Spearman por modalidad"}
    for clave, texto, sa, sb in COMPARACIONES:
        a, b = cargar(sa), cargar(sb)
        if a is None or b is None:
            print(f"{clave}: falta {'02_capas' + (sa if a is None else sb) + '.parquet'}; se salta")
            continue
        t = comparar(a, b, clave)
        tablas.append(t)
        sin_todas = t[t.modalidad != "todas"]
        resumen[clave] = {"que": texto, "a": sa, "b": sb, "modalidades": int(len(sin_todas)),
                          "spearman_mediana": round(float(sin_todas.spearman.median()), 4),
                          "spearman_min": round(float(sin_todas.spearman.min()), 4),
                          "spearman_max": round(float(sin_todas.spearman.max()), 4),
                          "modalidad_min": sin_todas.loc[sin_todas.spearman.idxmin(), "modalidad"],
                          "dif_media_rango": [round(float(sin_todas.dif_media_b_menos_a.min()), 3),
                                              round(float(sin_todas.dif_media_b_menos_a.max()), 3)]}
        print(f"\n--- {clave}: {texto}")
        print(t.round(4).to_string(index=False))
        r = resumen[clave]
        print(f"Spearman: mediana {r['spearman_mediana']}, rango {r['spearman_min']}–{r['spearman_max']} "
              f"(mínimo en {r['modalidad_min']})")
    if not tablas:
        return
    todo = pd.concat(tablas)
    todo.to_csv(RES / "06_sensibilidad.csv", index=False)
    (RES / "06_sensibilidad.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    FIG.mkdir(exist_ok=True)
    claves = list(todo.comparacion.unique())
    mods = list(todo[todo.comparacion == claves[0]].modalidad)
    fig, ax = plt.subplots(figsize=(7, 0.38 * len(mods) + 1.2))
    for k, (clave, marca) in enumerate(zip(claves, ["o", "s", "x"])):
        t = todo[todo.comparacion == clave].set_index("modalidad").reindex(mods)
        ax.scatter(t.spearman, np.arange(len(mods)) + (k - (len(claves) - 1) / 2) * 0.18, marker=marca, s=30,
                   label=resumen[clave]["que"])
    ax.set_yticks(np.arange(len(mods)), mods)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman de la capa media por neurona")
    ax.set_title("P4: sensibilidad de las capas en FlyWire")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=8, ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "06_sensibilidad.png", dpi=150)
    print("\n-> resultados/06_sensibilidad.csv, .json y figuras/06_sensibilidad.png")


if __name__ == "__main__":
    main()
