# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Descarga los datos de FlyWire v783 a datos/ sin iniciar sesión.

Uso:  uv run 00_descargar.py
Baja:
  - proofread_connections_783.feather (Zenodo 10676866, ~810 MB): conexiones
    entre neuronas revisadas, separadas por neuropilo, sin umbral.
  - neuron_annotations.tsv (GitHub flyconnectome/flywire_annotations, ~30 MB):
    clase de cada neurona (Schlegel et al. 2024).
  - rangos_publicados/*.feather (Zenodo 12588557, ~25 MB, CC BY 4.0): capas medias
    publicadas por Dorkenwald et al. 2024 (10,000 corridas por modalidad), para validar.

Zenodo limita la velocidad por conexión, así que el archivo grande se baja en
partes en paralelo (peticiones con Range) y al final se verifica el MD5.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import socket
import sys
import time
import urllib.request

AQUI = Path(__file__).parent
DATOS = AQUI / "datos"
ZENODO = "https://zenodo.org/api/records/10676866"
ANOTACIONES = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
               "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
PARTES = 24
socket.setdefaulttimeout(60)  # Zenodo a veces deja la conexión abierta sin mandar nada


def info_zenodo(nombre):
    with urllib.request.urlopen(ZENODO, timeout=60) as r:
        registro = json.load(r)
    for f in registro["files"]:
        if f["key"] == nombre:
            return f["links"]["self"], f["size"], f["checksum"].removeprefix("md5:")
    sys.exit(f"{nombre} no está en el registro de Zenodo")


def bajar_rango(url, destino, inicio, fin, intentos=5):
    """Baja los bytes [inicio, fin] a un archivo parcial; reanuda si ya hay algo."""
    hecho = destino.stat().st_size if destino.exists() else 0
    esperado = fin - inicio + 1
    for intento in range(intentos):
        if hecho >= esperado:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={inicio + hecho}-{fin}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(destino, "ab") as out:
                while bloque := r.read(1 << 20):
                    out.write(bloque)
            hecho = destino.stat().st_size
        except OSError as e:
            hecho = destino.stat().st_size if destino.exists() else 0
            print(f"  reintento {intento + 1} de {destino.name}: {e}", flush=True)
            time.sleep(3 * (intento + 1))
    if hecho < esperado:
        sys.exit(f"No se pudo completar {destino.name}")


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        while bloque := f.read(1 << 24):
            h.update(bloque)
    return h.hexdigest()


def bajar_zenodo(nombre):
    destino = DATOS / nombre
    url, tam, suma = info_zenodo(nombre)
    if destino.exists() and destino.stat().st_size == tam and md5(destino) == suma:
        print(f"{nombre}: ya está y el MD5 coincide")
        return
    partes_dir = DATOS / f".{nombre}.partes"
    partes_dir.mkdir(exist_ok=True)
    paso = -(-tam // PARTES)
    rangos = [(i, i * paso, min(tam, (i + 1) * paso) - 1) for i in range(PARTES)]
    print(f"{nombre}: {tam / 2**20:,.0f} MB en {PARTES} partes", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(PARTES) as ex:
        futuros = [ex.submit(bajar_rango, url, partes_dir / f"{i:03d}", a, b) for i, a, b in rangos]
        while not all(f.done() for f in futuros):
            bajado = sum(p.stat().st_size for p in partes_dir.iterdir())
            print(f"  {bajado / tam:6.1%}  {bajado / 2**20 / (time.time() - t0):.1f} MB/s", flush=True)
            time.sleep(15)
        for f in futuros:
            f.result()
    with open(destino, "wb") as out:
        for i, _, _ in rangos:
            out.write((partes_dir / f"{i:03d}").read_bytes())
    if md5(destino) != suma:
        destino.unlink()
        sys.exit(f"{nombre}: el MD5 no coincide; se borró. Vuelve a correr el script.")
    for p in partes_dir.iterdir():
        p.unlink()
    partes_dir.rmdir()
    print(f"{nombre}: listo en {time.time() - t0:.0f} s, MD5 verificado", flush=True)


def bajar_rangos():
    """Rangos de flujo de información publicados (Dorkenwald et al. 2024, fig. 6)."""
    destino = DATOS / "rangos_publicados"
    destino.mkdir(exist_ok=True)
    with urllib.request.urlopen("https://zenodo.org/api/records/12588557", timeout=60) as r:
        archivos = json.load(r)["files"]
    def uno(f):  # Zenodo limita cada conexión: un hilo por archivo
        ruta = destino / f["key"]
        for intento in range(5):
            if ruta.exists() and ruta.stat().st_size == f["size"] and md5(ruta) == f["checksum"].removeprefix("md5:"):
                return
            try:
                urllib.request.urlretrieve(f["links"]["self"], ruta)
            except OSError as e:
                print(f"  reintento {intento + 1} de {f['key']}: {e}", flush=True)
                time.sleep(3 * (intento + 1))
        raise RuntimeError(f"No se pudo bajar bien {f['key']}")
    with ThreadPoolExecutor(len(archivos)) as ex:
        list(ex.map(uno, archivos))
    print(f"rangos_publicados: {len(archivos)} archivos")


def main():
    DATOS.mkdir(exist_ok=True)
    bajar_zenodo("proofread_connections_783.feather")
    destino = DATOS / "neuron_annotations.tsv"
    if not destino.exists():
        urllib.request.urlretrieve(ANOTACIONES, destino)
    print(f"neuron_annotations.tsv: {destino.stat().st_size / 2**20:.1f} MB")
    bajar_rangos()


if __name__ == "__main__":
    main()
