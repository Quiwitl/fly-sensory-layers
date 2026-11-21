# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pandas", "pyarrow", "hypothesis"]
# ///
"""Pruebas basadas en propiedades para 02_recorrido.py. Uso:  uv run propiedades_recorrido.py

pruebas_recorrido.py revisa casos escritos a mano. Aquí hypothesis genera cientos de
grafos al azar (tamaño, aristas, pesos, semillas, umbral y regla) y comprueba que el
modelo cumpla siempre lo mismo, sin importar el grafo:

  P1  normalizar: A >= 0, mismo patrón de aristas que W y cada columna suma 0 o 1.
  P2  las semillas quedan en la capa 1 y ninguna otra neurona queda en la capa 1.
  P3  toda capa está en {0} ∪ [1, max_pasos].
  P4  causalidad: una neurona en la capa t > 1 tiene al menos un predecesor (A > 0)
      que ya estaba recorrido en una capa < t. La señal no aparece de la nada.
  P5  certeza: si recibe >= umbral de su entrada desde una neurona en la capa t,
      su capa es a lo más t + 1 (con regla por arista, esa arista tiene p = 1).
  P6  reproducible: la misma semilla aleatoria da las mismas capas.
  P7  recorrer_lote: cada corrida de un lote cumple P2–P5 por separado.

Si una propiedad falla, hypothesis reduce el grafo al ejemplo más chico que la rompe
y lo imprime: ese ejemplo es el caso de prueba que hay que agregar a pruebas_recorrido.py.
"""
from pathlib import Path
import importlib.util
import sys

import numpy as np
from scipy import sparse
from hypothesis import given, settings, strategies as st, HealthCheck

ruta = Path(__file__).parent / "02_recorrido.py"
if not ruta.exists():
    sys.exit("Falta 02_recorrido.py")
spec = importlib.util.spec_from_file_location("recorrido", ruta)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

AJUSTES = settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def grafos(draw, max_n=25):
    """Un grafo dirigido con pesos (sinapsis), un conjunto de semillas y una semilla aleatoria."""
    n = draw(st.integers(2, max_n))
    m = draw(st.integers(0, 4 * n))
    o = draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m))
    d = draw(st.lists(st.integers(0, n - 1), min_size=m, max_size=m))
    w = draw(st.lists(st.integers(1, 60), min_size=m, max_size=m))
    W = sparse.csr_matrix((np.array(w, dtype=float), (np.array(o, dtype=int), np.array(d, dtype=int))),
                          shape=(n, n))
    semillas = np.array(sorted(draw(st.sets(st.integers(0, n - 1), min_size=1, max_size=max(1, n // 3)))))
    azar = draw(st.integers(0, 2**32 - 1))
    return W, semillas, azar


umbrales = st.sampled_from([0.1, 0.3, 0.5, 1.0])
reglas = st.sampled_from(["arista", "neurona"])
pasos = st.integers(2, 20)


def revisar_capas(A, semillas, capas, umbral, max_pasos):
    """P2–P5 sobre una corrida. A: matriz normalizada (fila = origen, columna = destino)."""
    A = sparse.csr_matrix(A)
    n = A.shape[0]
    capas = np.asarray(capas)
    es_semilla = np.zeros(n, dtype=bool)
    es_semilla[semillas] = True
    # P2
    assert (capas[es_semilla] == 1).all(), f"semillas fuera de la capa 1: {capas[es_semilla]}"
    assert not ((capas == 1) & ~es_semilla).any(), "una neurona que no es semilla quedó en la capa 1"
    # P3
    assert ((capas == 0) | ((capas >= 1) & (capas <= max_pasos))).all(), f"capa fuera de rango: {capas}"
    AT = sparse.csr_matrix(A.T)  # fila = destino
    for j in range(n):
        t = capas[j]
        fila = AT.getrow(j)
        pred, frac = fila.indices, fila.data
        cp = capas[pred]
        # P4
        if t > 1:
            assert ((cp >= 1) & (cp < t) & (frac > 0)).any(), \
                f"neurona {j} en capa {t} sin predecesor recorrido antes (predecesores {list(pred)} en capas {list(cp)})"
        # P5
        seguras = (frac >= umbral - 1e-9) & (cp >= 1)
        if seguras.any() and not es_semilla[j]:
            tope = cp[seguras].min() + 1
            if tope <= max_pasos:
                assert 1 < t <= tope, f"neurona {j}: recibe >= {umbral} desde la capa {tope - 1} pero quedó en {t}"


@given(grafos())
@AJUSTES
def p1_normalizar(g):
    W, _, _ = g
    copia = W.copy()
    A = sparse.csr_matrix(mod.normalizar(W))
    assert (A.data >= 0).all()
    sumas = np.asarray(A.sum(axis=0)).ravel()
    entradas = np.asarray(W.sum(axis=0)).ravel()
    assert np.allclose(sumas[entradas > 0], 1) and np.allclose(sumas[entradas == 0], 0), sumas
    assert ((A != 0) != (W != 0)).nnz == 0, "normalizar cambió qué aristas existen"
    assert (W != copia).nnz == 0, "normalizar modificó W"


@given(grafos(), umbrales, reglas, pasos)
@AJUSTES
def p2_p5_una_corrida(g, umbral, regla, max_pasos):
    W, semillas, azar = g
    A = mod.normalizar(W)
    capas = mod.recorrer(A, semillas, np.random.default_rng(azar), umbral=umbral, max_pasos=max_pasos, regla=regla)
    assert len(capas) == W.shape[0]
    revisar_capas(A, semillas, capas, umbral, max_pasos)


@given(grafos(), umbrales, reglas)
@AJUSTES
def p6_reproducible(g, umbral, regla):
    W, semillas, azar = g
    A = mod.normalizar(W)
    a = mod.recorrer(A, semillas, np.random.default_rng(azar), umbral=umbral, regla=regla)
    b = mod.recorrer(A, semillas, np.random.default_rng(azar), umbral=umbral, regla=regla)
    assert np.array_equal(a, b)


@given(grafos(max_n=15), umbrales, st.integers(2, 8))
@AJUSTES
def p7_lote(g, umbral, corridas):
    W, semillas, azar = g
    A = mod.normalizar(W)
    M = mod.preparar(A, umbral, "arista")
    capas = mod.recorrer_lote(M, semillas, corridas, np.random.default_rng(azar), max_pasos=12)
    assert capas.shape == (W.shape[0], corridas)
    for k in range(corridas):
        revisar_capas(A, semillas, capas[:, k], umbral, 12)


PROPIEDADES = [p1_normalizar, p2_p5_una_corrida, p6_reproducible, p7_lote]

if __name__ == "__main__":
    fallas = 0
    for p in PROPIEDADES:
        try:
            p()
            print(f"✓ {p.__name__}")
        except Exception as e:  # noqa: BLE001 — hypothesis ya imprimió el ejemplo mínimo
            fallas += 1
            print(f"✗ {p.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(PROPIEDADES) - fallas}/{len(PROPIEDADES)} propiedades se cumplen")
    sys.exit(1 if fallas else 0)
