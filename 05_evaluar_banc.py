# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "pyarrow", "matplotlib"]
# ///
"""Fase 3: evalúa H2 y H3 contra los criterios fijados en PLAN.md (20 nov 2026), sin cambiarlos.

Uso:  uv run 05_evaluar_banc.py [--etiqueta _prueba] [--bootstrap 1000]
Lee:  resultados/02_capas_completo_p15_u1.parquet   FlyWire v783, cerebro completo, sin umbral, tope 15
      resultados/04_h2<etiqueta>_capas.parquet       BANC, cerebro sin cordón, tope 15
      resultados/04_h3<etiqueta>_blancos.npz         BANC completo: capa de cada motoneurona en cada corrida
      resultados/04_h3_azar<K>_blancos.npz           grafos al azar (control), si existen
Deja: resultados/05_h2_tipos.csv, 05_h2_spearman.csv, 05_h3_segmentos.csv, 05_h3_atajo.csv, 05_evaluacion.json
      figuras/05_h2_tipos.png, figuras/05_h3_motoneuronas.png

H2 (PLAN): Spearman entre la capa por tipo celular en FlyWire y en el BANC, por modalidad; la capa de un
tipo es el promedio de sus neuronas; se empareja con `fafb_cell_type`. Se cumple si la mediana entre
modalidades es >= 0.7 y al menos 3 de cada 4 modalidades llegan a 0.7. También solo con tipos de 5
neuronas o más (en las dos moscas).
H3 (PLAN): capa mediana de las motoneuronas de cada segmento desde cada modalidad. Atajo en una pata si
la mediana desde los mecanosensores de esa pata es al menos 2 capas menor que la menor de las medianas
de la cabeza y el IC 95 % (bootstrap sobre motoneuronas) de la diferencia no incluye 0. H3 si hay atajo
en las tres patas. Control: el atajo cuenta solo si la diferencia observada es mayor que en los 10
grafos al azar. La capa de una motoneurona es su capa media en las corridas que la alcanzan; también
se reporta la fracción de corridas que la alcanzan, y todo con tope 15.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

AQUI = Path(__file__).parent
RES = AQUI / "resultados"
FIG = AQUI / "figuras"
CABEZA = ["olfativas", "gustativas", "johnston", "proyeccion_visual"]
PATAS = {"MN_T1": "mecano_T1", "MN_T2": "mecano_T2", "MN_T3": "mecano_T3"}
UMBRAL_RHO, MIN_NEURONAS, DIF_MIN = 0.7, 5, 2.0


# ----------------------------------------------------------------- H2

def capa_por_tipo(df, col_tipo, modalidades):
    d = df[df[col_tipo].notna() & (df[col_tipo].astype(str) != "")]
    g = d.groupby(col_tipo)
    out = g[modalidades].mean()
    out["n"] = g.size()
    return out


def evaluar_h2(etiqueta):
    fw = pd.read_parquet(RES / "02_capas_completo_p15_u1.parquet")
    bc = pd.read_parquet(RES / f"04_h2{etiqueta}_capas.parquet")
    mods = [m for m in CABEZA if m in bc.columns and m in fw.columns]
    a = capa_por_tipo(fw, "cell_type", mods)
    b = capa_por_tipo(bc, "fafb_cell_type", mods)
    t = a.join(b, how="inner", lsuffix="_flywire", rsuffix="_banc")
    t.to_csv(RES / "05_h2_tipos.csv")
    filas = []
    for m in mods:
        for nombre, sub in (("todos", t), (f"n>={MIN_NEURONAS}", t[(t.n_flywire >= MIN_NEURONAS) & (t.n_banc >= MIN_NEURONAS)])):
            x = sub[[f"{m}_flywire", f"{m}_banc"]].dropna()
            rho = spearmanr(x.iloc[:, 0], x.iloc[:, 1]).statistic if len(x) > 2 else np.nan
            filas.append({"modalidad": m, "tipos": nombre, "n_tipos": len(x), "spearman": rho})
    s = pd.DataFrame(filas)
    s.to_csv(RES / "05_h2_spearman.csv", index=False)
    veredicto = {}
    for nombre, sub in s.groupby("tipos"):
        r = sub["spearman"].to_numpy()
        veredicto[nombre] = {"mediana": float(np.nanmedian(r)), "modalidades_>=0.7": int((r >= UMBRAL_RHO).sum()),
                             "modalidades": len(r),
                             "se_cumple": bool(np.nanmedian(r) >= UMBRAL_RHO and (r >= UMBRAL_RHO).sum() >= 0.75 * len(r))}
    return t, s, mods, veredicto


# ----------------------------------------------------------------- H3

def medias_mn(z, tope=None):
    """Capa media por motoneurona y modalidad (corridas que la alcanzan) y fracción de corridas que la alcanzan."""
    grupo = z["grupo"]
    capas, alc = {}, {}
    for k in z.files:
        if not k.startswith("capas_"):
            continue
        c = z[k].astype(float)
        if tope:
            c[c > tope] = 0
        ok = c > 0
        capas[k[6:]] = np.where(ok.any(1), np.where(ok, c, 0).sum(1) / np.maximum(ok.sum(1), 1), np.nan)
        alc[k[6:]] = ok.mean(1)
    return grupo, pd.DataFrame(capas), pd.DataFrame(alc)


def diferencia(capas, idx):
    med = {m: np.nanmedian(capas[m].to_numpy()[idx]) for m in capas.columns}
    return med


def evaluar_h3(z, tope, boot, rng):
    grupo, capas, alc = medias_mn(z, tope)
    seg_filas, atajo = [], {}
    for seg in pd.unique(grupo):
        idx = np.flatnonzero(grupo == seg)
        for m in capas.columns:
            seg_filas.append({"segmento": seg, "modalidad": m, "motoneuronas": len(idx),
                              "capa_mediana": float(np.nanmedian(capas[m].to_numpy()[idx])),
                              "alcanzadas_alguna_vez": int((alc[m].to_numpy()[idx] > 0).sum()),
                              "fraccion_alcance_media": float(alc[m].to_numpy()[idx].mean())})
        pata = PATAS.get(seg)
        cabeza = [m for m in CABEZA if m in capas.columns]
        if pata not in capas.columns or not cabeza:
            continue
        med = diferencia(capas, idx)
        dif = min(med[m] for m in cabeza) - med[pata]
        difs = []
        for _ in range(boot):
            r = rng.choice(idx, len(idx), replace=True)
            mb = diferencia(capas[cabeza + [pata]], r)
            difs.append(min(mb[m] for m in cabeza) - mb[pata])
        lo, hi = np.nanpercentile(difs, [2.5, 97.5]) if difs else (np.nan, np.nan)
        atajo[seg] = {"mediana_pata": float(med[pata]), "menor_mediana_cabeza": float(min(med[m] for m in cabeza)),
                      "modalidad_cabeza_mas_cercana": min(cabeza, key=lambda m: med[m]),
                      "diferencia": float(dif), "ic95": [float(lo), float(hi)],
                      "atajo_sin_control": bool(dif >= DIF_MIN and lo > 0)}
    return pd.DataFrame(seg_filas), atajo, capas, grupo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(888)
    FIG.mkdir(exist_ok=True)
    out = {"criterios": "PLAN.md, 20 nov 2026 (sin cambios)"}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # H2
    if (RES / f"04_h2{args.etiqueta}_capas.parquet").exists():
        t, s, mods, v = evaluar_h2(args.etiqueta)
        out["H2"] = {"spearman": s.to_dict("records"), "veredicto": v}
        print(s.to_string(index=False))
        print("H2:", v)
        fig, ax = plt.subplots(1, len(mods), figsize=(4 * len(mods), 4), squeeze=False)
        for a, m in zip(ax[0], mods):
            x = t[[f"{m}_flywire", f"{m}_banc"]].dropna()
            a.scatter(x.iloc[:, 0], x.iloc[:, 1], s=4, alpha=0.4)
            r = s[(s.modalidad == m) & (s.tipos == "todos")].spearman.iloc[0]
            a.set_title(f"{m}  ρ = {r:.2f}  ({len(x)} tipos)")
            a.set_xlabel("capa media del tipo, FlyWire")
            a.set_ylabel("capa media del tipo, BANC")
        fig.tight_layout()
        fig.savefig(FIG / "05_h2_tipos.png", dpi=150)
        plt.close(fig)

    # H3
    f = RES / f"04_h3{args.etiqueta}_blancos.npz"
    if f.exists():
        z = np.load(f)
        out["H3"] = {}
        for tope in (None, 15):
            nombre = "tope30" if tope is None else "tope15"
            seg, atajo, capas, grupo = evaluar_h3(z, tope, args.bootstrap, rng)
            seg.to_csv(RES / f"05_h3_segmentos_{nombre}.csv", index=False)
            # control: diferencia en los grafos al azar que existan
            azar = [evaluar_h3(np.load(fa), tope, 0, rng)[1] for fa in sorted(f for f in RES.glob("04_h3_azar*_blancos.npz") if "prueba" not in f.name)]
            for s_, d in atajo.items():
                difs_azar = [a_[s_]["diferencia"] for a_ in azar if s_ in a_]
                d["diferencias_azar"] = difs_azar
                d["grafos_azar"] = len(difs_azar)
                d["supera_los_disponibles"] = bool(d["diferencia"] > max(difs_azar)) if difs_azar else None
                # el PLAN pide 10 grafos al azar; con menos, el control queda incompleto (None)
                d["supera_azar"] = d["supera_los_disponibles"] if len(difs_azar) >= 10 else None
                d["atajo"] = bool(d["atajo_sin_control"] and d["supera_azar"]) if d["supera_azar"] is not None else None
            patas = [atajo.get(k, {}) for k in PATAS]
            out["H3"][nombre] = {"segmentos": seg.to_dict("records"), "atajo": atajo,
                                 "H3_sin_control": all(p.get("atajo_sin_control") for p in patas) if all(patas) else None,
                                 "H3": (all(p.get("atajo") for p in patas)
                                        if all(patas) and all(p.get("atajo") is not None for p in patas) else None)}
            print(f"--- H3 {nombre}")
            print(seg.pivot(index="modalidad", columns="segmento", values="capa_mediana").round(2).to_string())
            for k, v in atajo.items():
                print(k, {kk: vv for kk, vv in v.items() if kk != "diferencias_azar"})
            if tope is None:
                mods = list(capas.columns)
                segs = list(pd.unique(grupo))
                fig, ax = plt.subplots(1, len(segs), figsize=(4 * len(segs), 4.5), squeeze=False, sharey=True)
                for a, sg in zip(ax[0], segs):
                    idx = grupo == sg
                    datos = [capas[m].to_numpy()[idx] for m in mods]
                    a.boxplot([d[~np.isnan(d)] for d in datos], showfliers=False)
                    a.set_xticks(range(1, len(mods) + 1), mods, rotation=60, ha="right", fontsize=8)
                    a.set_title(f"{sg} ({idx.sum()} motoneuronas)")
                ax[0][0].set_ylabel("capa media de la motoneurona (tope 30)")
                fig.tight_layout()
                fig.savefig(FIG / "05_h3_motoneuronas.png", dpi=150)
                plt.close(fig)
    (RES / f"05_evaluacion{args.etiqueta}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float),
                                                            encoding="utf-8")
    print(f"-> resultados/05_evaluacion{args.etiqueta}.json")


if __name__ == "__main__":
    main()
