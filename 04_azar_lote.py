# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Control contra el azar de la Fase 3: corre 04_banc.py --h3 en los grafos al azar K = 1..10, uno tras otro.

Uso:  uv run 04_azar_lote.py [--desde 1] [--hasta 10] [--procesos 4] [--esperar K] [--retraso S] [--gpu]
--gpu: cada grafo en la GPU con el Python que tenga CuPy (--python-gpu o FLY_GPU_PYTHON) (validado el 21 nov 2026 contra la CPU en el grafo 1).
Se pueden correr dos cadenas a la vez: cada K deja resultados/04_azar{K}.ocupado mientras corre
y la otra cadena se lo salta. --esperar K: no empezar hasta que exista el resumen de K.
--retraso S: esperar S segundos antes de empezar (para que el barajado de aristas, que usa
mucha memoria, no coincida con el de la otra cadena).
Cada grafo deja resultados/04_h3_azar{K}_* y su log en resultados/04_azar{K}.log.
Si un K ya tiene su resumen, se salta (se puede relanzar sin repetir trabajo).
Al terminar escribe resultados/04_azar_fin.txt con la hora y qué K quedaron.
Después: uv run 05_evaluar_banc.py
"""
from pathlib import Path
import argparse
import os
import sys
import subprocess
import time

AQUI = Path(__file__).parent
R = AQUI / "resultados"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", type=int, default=1)
    ap.add_argument("--hasta", type=int, default=10)
    ap.add_argument("--procesos", type=int, default=4)
    ap.add_argument("--esperar", type=int, default=0)
    ap.add_argument("--retraso", type=int, default=0)
    ap.add_argument("--gpu", action="store_true", help="correr cada grafo en la GPU (04_banc.py --gpu)")
    ap.add_argument("--python-gpu", default=os.environ.get("FLY_GPU_PYTHON", sys.executable),
                    help="Python con CuPy (o la variable FLY_GPU_PYTHON)")
    ap.add_argument("--lote-gpu", type=int, default=250)
    a = ap.parse_args()
    hechos, fallas = [], []
    t0 = time.time()
    if a.esperar:
        while not (R / f"04_h3_azar{a.esperar}_resumen.json").exists():
            time.sleep(30)
    time.sleep(a.retraso)
    for k in range(a.desde, a.hasta + 1):
        ocupado = R / f"04_azar{k}.ocupado"
        if (R / f"04_h3_azar{k}_resumen.json").exists() or ocupado.exists():
            hechos.append(k)
            continue
        ocupado.write_text(time.strftime("%H:%M"), encoding="utf-8")
        with open(R / f"04_azar{k}.log", "w", encoding="utf-8") as log:
            if a.gpu:
                cmd = [a.python_gpu, "-u", "-W", "ignore", "04_banc.py", "--h3", "--azar", str(k), "--gpu",
                       "--lote", str(a.lote_gpu)]
            else:
                cmd = ["uv", "run", "-q", "04_banc.py", "--h3", "--azar", str(k), "--procesos", str(a.procesos)]
            r = subprocess.run(cmd, cwd=AQUI, stdout=log, stderr=subprocess.STDOUT)
        ocupado.unlink(missing_ok=True)
        (hechos if r.returncode == 0 and (R / f"04_h3_azar{k}_resumen.json").exists() else fallas).append(k)
    (R / f"04_azar_fin_{a.desde}-{a.hasta}.txt").write_text(
        f"terminado {time.strftime('%Y-%m-%d %H:%M')} en {(time.time() - t0) / 60:.0f} min\n"
        f"hechos: {hechos}\nfallas: {fallas}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
