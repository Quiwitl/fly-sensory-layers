# -*- coding: utf-8 -*-
"""Datos del explorador interactivo (resultados/07_explorador.json): resume lo que ya calcularon las fases 1-4.

- FlyWire v783 (cerebro, 13 modalidades, sin umbral, 15 pasos): capa por superclase, por tipo celular y en
  las motoneuronas del cerebro (02_capas_completo_p15_u1_gpu.parquet).
- BANC v888 (cerebro y cordón, 7 modalidades, 15 pasos): capa por superclase y por tipo, y en las motoneuronas
  de cada pata y del ala (04_h3_capas_p15.parquet, 04_h3_blancos.npz).
- Dos moscas: capa media por tipo en FlyWire contra BANC (05_h2_tipos.csv, 05_h2_spearman.csv).
- Atajo de las patas y los 10 grafos al azar (misma cuenta que 05_evaluar_banc.py, tope de 15 pasos).
- Réplica de lo publicado (03_validacion_publicado_completo_p15_u1.csv).

Las semillas (capa 1) no cuentan en los resúmenes: se ve a dónde llega la señal, no de dónde sale.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
RES = AQUI / "resultados"
spec = importlib.util.spec_from_file_location("ev", AQUI / "05_evaluar_banc.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

FW_MODS = ["olfativas", "gustativas", "johnston", "cerdas_cabeza", "cerdas_ojo", "taste_peg", "termosensoriales",
           "higrosensoriales", "proyeccion_visual", "fotorreceptores", "ocelos", "ascendentes", "todas"]
BANC_MODS = ["olfativas", "gustativas", "johnston", "proyeccion_visual", "mecano_T1", "mecano_T2", "mecano_T3"]


def r1(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def por_superclase(df, mods, col="super_class"):
    out = {}
    for m in mods:
        v = df[[col, m]].copy()
        v = v[v[m] > 1.0]                                      # sin semillas ni no alcanzadas
        filas = []
        for sc, g in v.groupby(col):
            x = g[m].to_numpy()
            filas.append({"sc": sc, "n": int(len(x)), "q10": r1(np.percentile(x, 10)), "q25": r1(np.percentile(x, 25)),
                          "med": r1(np.median(x)), "q75": r1(np.percentile(x, 75)), "q90": r1(np.percentile(x, 90))})
        out[m] = filas
    return out


def por_tipo(df, mods, col_tipo="cell_type", col_sc="super_class"):
    d = df[df[col_tipo].notna() & (df[col_tipo] != "")]
    g = d.groupby(col_tipo)
    nombres = sorted(g.groups)
    sc = g[col_sc].agg(lambda s: s.mode().iat[0] if len(s.mode()) else "").reindex(nombres)
    n = g.size().reindex(nombres)
    L = {}
    for m in mods:
        x = d[[col_tipo, m]].copy()
        x.loc[x[m] <= 1.0, m] = np.nan                         # semillas fuera
        mean = x.groupby(col_tipo)[m].mean().reindex(nombres)
        L[m] = [None if not np.isfinite(v) else int(round(v * 10)) for v in mean.to_numpy()]
    return {"nombres": nombres, "sc": sc.tolist(), "n": [int(v) for v in n], "L10": L}


def main():
    out = {"generado": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}

    # FlyWire
    fw = pd.read_parquet(RES / "02_capas_completo_p15_u1_gpu.parquet")
    out["fw_neuronas"] = int(len(fw))
    out["fw_superclases"] = por_superclase(fw, FW_MODS)
    out["fw_tipos"] = por_tipo(fw, FW_MODS)
    mn = fw[fw["super_class"] == "motor"]
    out["fw_motor"] = {m: {"med": r1(np.median(mn.loc[mn[m] > 1, m])), "n": int((mn[m] > 1).sum())} for m in FW_MODS}

    pub = pd.read_csv(RES / "03_validacion_publicado_completo_p15_u1.csv")
    out["h1_columnas"] = pub.columns.tolist()
    out["h1"] = pub.to_dict("records")

    # BANC
    banc = pd.read_parquet(RES / "04_h3_capas_p15.parquet")
    out["banc_neuronas"] = int(len(banc))
    out["banc_superclases"] = por_superclase(banc, BANC_MODS)
    out["banc_tipos"] = por_tipo(banc, BANC_MODS)
    seg = pd.read_csv(RES / "05_h3_segmentos_tope15.csv")
    out["banc_mn"] = {s: {r.modalidad: r1(r.capa_mediana) for r in g.itertuples()} | {"n": int(g.motoneuronas.iat[0])}
                      for s, g in seg.groupby("segmento")}

    # atajo de las patas y control al azar (tope 15, como el informe)
    rng = np.random.default_rng(888)
    _, atajo, _, _ = ev.evaluar_h3(np.load(RES / "04_h3_blancos.npz"), 15, 1000, rng)
    azar = [ev.evaluar_h3(np.load(f), 15, 0, rng)[1] for f in sorted(RES.glob("04_h3_azar*_blancos.npz")) if "prueba" not in f.name]
    out["atajo"] = {str(s): {"pata": r1(a["mediana_pata"]), "cabeza": r1(a["menor_mediana_cabeza"]),
                        "cabeza_mod": a["modalidad_cabeza_mas_cercana"], "dif": r1(a["diferencia"]),
                        "ic95": [r1(x) for x in a["ic95"]], "azar": [r1(z[s]["diferencia"]) for z in azar if s in z]}
                    for s, a in atajo.items()}

    # dos moscas: un solo listado de tipos y una columna por modalidad y mosca (capa x100)
    t = pd.read_csv(RES / "05_h2_tipos.csv")
    sp = pd.read_csv(RES / "05_h2_spearman.csv")
    H2_MODS = ["olfativas", "gustativas", "johnston", "proyeccion_visual"]
    t = t.dropna(subset=[f"{m}_{f}" for m in H2_MODS for f in ("flywire", "banc")], how="all")
    a100 = lambda c: [None if not np.isfinite(v) else int(round(v * 100)) for v in t[c].to_numpy()]
    out["h2"] = {"tipos": t["cell_type"].tolist(),
                 "n": [int(min(a, b)) for a, b in t[["n_flywire", "n_banc"]].fillna(0).to_numpy()],
                 "fw": {m: a100(f"{m}_flywire") for m in H2_MODS},
                 "banc": {m: a100(f"{m}_banc") for m in H2_MODS},
                 "rho": {m: r1(sp[(sp.modalidad == m) & (sp.tipos == "todos")].spearman.iat[0], 3) for m in H2_MODS},
                 "rho5": {m: r1(sp[(sp.modalidad == m) & (sp.tipos == "n>=5")].spearman.iat[0], 3) for m in H2_MODS},
                 "n_tipos": {m: int(sp[(sp.modalidad == m) & (sp.tipos == "todos")].n_tipos.iat[0]) for m in H2_MODS}}
    res = json.loads((RES / "04_h3_resumen.json").read_text(encoding="utf-8"))
    out["banc_semillas"] = {m: v["semillas"] for m, v in res["modalidades"].items()}
    out["fw_semillas"] = {r["modalidad"]: int(r["semillas_propias"]) for r in out["h1"]}

    f = RES / "07_explorador.json"
    f.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f, f"{f.stat().st_size / 1e6:.2f} MB")
    print("FlyWire superclases:", sorted({r["sc"] for v in out["fw_superclases"].values() for r in v}))
    print("BANC superclases:", sorted({r["sc"] for v in out["banc_superclases"].values() for r in v}))
    print("tipos FlyWire / BANC:", len(out["fw_tipos"]["nombres"]), len(out["banc_tipos"]["nombres"]))
    print("motor FlyWire:", {k: v for k, v in list(out["fw_motor"].items())[:3]})
    print("atajo:", {k: (v["dif"], min(v["azar"]), max(v["azar"]), len(v["azar"])) for k, v in out["atajo"].items()})
    print("H1:", out["h1_columnas"], out["h1"][:2])


if __name__ == "__main__":
    main()
