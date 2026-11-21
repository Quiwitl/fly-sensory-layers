"""Modelo de recorrido en GPU (CuPy). Mismo modelo que 02_recorrido.py: regla por arista con
reintento, P_j = 1 - prod(1 - p_ij), paro cuando no quedan aristas candidatas o en el tope.

Lo usa 04_banc.py con --gpu. Necesita CuPy con las bibliotecas de CUDA (cuSPARSE, cuRAND).
Necesita CuPy con CUDA 12 en un entorno propio (p. ej. `pip install cupy-cuda12x numpy scipy pandas pyarrow`);
los encabezados de uv de los scripts no lo instalan.

La diferencia con la CPU es solo de escala: se corren `lote` corridas a la vez como columnas
de una matriz densa en la GPU, y cada paso es un producto disperso × denso (cuSPARSE).
Los números aleatorios salen de cuRAND, así que las corridas no son idénticas a las de la CPU,
pero siguen la misma distribución (se valida comparando un grafo corrido en las dos).
"""
import time

import numpy as np
import cupy as cp
import cupyx.scipy.sparse as cs


def preparar_gpu(M):
    """Copia a la GPU las matrices que arma 02_recorrido.preparar (regla por arista)."""
    if M["regla"] != "arista":
        raise ValueError("la versión GPU solo implementa la regla por arista")
    return {k: cs.csr_matrix(M[k].astype(np.float32)) for k in ("logqT", "segurasT", "candT")}


def recorrer_lote_gpu(G, semillas, corridas, rs, max_pasos):
    """Capas (int16, n × corridas) en la GPU; 0 = no alcanzada."""
    n = G["candT"].shape[0]
    sem = cp.asarray(semillas)
    capas = cp.zeros((n, corridas), dtype=cp.int16)
    capas[sem, :] = 1
    X = cp.zeros((n, corridas), dtype=cp.float32)
    X[sem, :] = 1.0
    for t in range(2, max_pasos + 1):
        pendientes = (capas == 0) & ((G["candT"] @ X) > 0)
        if not bool(pendientes.any()):
            break
        P = -cp.expm1(G["logqT"] @ X)
        P[(G["segurasT"] @ X) > 0] = 1.0
        nuevas = pendientes & (rs.random((n, corridas), dtype=cp.float32) < P)
        del P
        capas[nuevas] = t
        X[nuevas] = 1.0
    return capas


def trabajo_gpu(G, nombre, sem, blancos_idx, semilla, azar, lote, corridas, max_pasos, tope_corto):
    """Igual que 04_banc._trabajo, pero en la GPU. Devuelve lo mismo."""
    t0 = time.time()
    seed = int(np.random.SeedSequence([semilla, sum(map(ord, nombre)), azar]).generate_state(1)[0])
    rs = cp.random.default_rng(seed)
    n = G["candT"].shape[0]
    suma, cuenta = cp.zeros(n), cp.zeros(n)
    suma_c, cuenta_c = cp.zeros(n), cp.zeros(n)
    bl = cp.asarray(blancos_idx)
    capas_blancos = np.zeros((len(blancos_idx), corridas), dtype=np.int16)
    media_ant, cambio, hechas = None, np.inf, 0
    while hechas < corridas:
        k = min(lote, corridas - hechas)
        c = recorrer_lote_gpu(G, sem, k, rs, max_pasos)
        suma += c.sum(axis=1)
        cuenta += (c > 0).sum(axis=1)
        if tope_corto and tope_corto < max_pasos:
            cc = cp.where(c <= tope_corto, c, 0)
            suma_c += cc.sum(axis=1)
            cuenta_c += (cc > 0).sum(axis=1)
            del cc
        if len(blancos_idx):
            capas_blancos[:, hechas:hechas + k] = c[bl].get()
        del c
        hechas += k
        m = cp.where(cuenta > 0, suma / cp.maximum(cuenta, 1), cp.nan).get()
        if media_ant is not None:
            ok = ~np.isnan(m) & ~np.isnan(media_ant)
            cambio = float(np.quantile(np.abs(m[ok] - media_ant[ok]), 0.99)) if ok.any() else 0.0
        media_ant = m
    out = {"media": media_ant, "frac": (cuenta / hechas).get()}
    if tope_corto and tope_corto < max_pasos:
        out["media_c"] = cp.where(cuenta_c > 0, suma_c / cp.maximum(cuenta_c, 1), cp.nan).get()
        out["frac_c"] = (cuenta_c / hechas).get()
    cp.get_default_memory_pool().free_all_blocks()
    return nombre, out, capas_blancos, hechas, cambio, time.time() - t0


def correr_modalidad_gpu(G, semillas, semilla, lote=100, max_corridas=10_000, tol=0.01, cuantil=0.99,
                         max_pasos=30, al_lote=None):
    """Igual que 02_recorrido.correr_modalidad, en la GPU: lotes de `lote` corridas hasta que la capa
    media cambie menos de `tol` en el `cuantil` de las neuronas (y al menos 5 lotes).
    `semilla` es un entero (de SeedSequence); `al_lote(hechas, cambio)` se llama tras cada lote.
    Devuelve (media, fracción alcanzada, corridas hechas, cambio final) como arreglos de NumPy."""
    rs = cp.random.default_rng(semilla)
    n = G["candT"].shape[0]
    suma, cuenta = cp.zeros(n), cp.zeros(n)
    media_ant, hechas, cambio = None, 0, np.inf
    while hechas < max_corridas:
        c = recorrer_lote_gpu(G, semillas, lote, rs, max_pasos)
        suma += c.sum(axis=1)
        cuenta += (c > 0).sum(axis=1)
        del c
        hechas += lote
        media = cp.where(cuenta > 0, suma / cp.maximum(cuenta, 1), cp.nan).get()
        if media_ant is not None:
            ok = ~np.isnan(media) & ~np.isnan(media_ant)
            cambio = float(np.quantile(np.abs(media[ok] - media_ant[ok]), cuantil)) if ok.any() else 0.0
        if al_lote:
            al_lote(hechas, cambio)
        if media_ant is not None and hechas >= 5 * lote and cambio < tol:
            break
        media_ant = media
    frac = (cuenta / hechas).get()
    cp.get_default_memory_pool().free_all_blocks()
    return media, frac, hechas, cambio
