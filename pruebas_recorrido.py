# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "pandas", "pyarrow"]
# ///
"""Pruebas para 02_recorrido.py (Fase 1). Uso:  uv run pruebas_recorrido.py

02_recorrido.py debe definir:

    normalizar(W) -> A
        W: matriz dispersa n×n, W[i, j] = sinapsis de i hacia j.
        A: matriz dispersa con A[i, j] = W[i, j] / (entradas totales de j).
        Cada columna suma 1 (o 0 si la neurona no recibe nada). No modifica W.

    recorrer(A, semillas, rng, umbral=0.3, max_pasos=50) -> capas
        semillas: arreglo de índices de las neuronas semilla.
        rng: np.random.Generator (todo el azar sale de aquí).
        capas: np.ndarray de enteros de largo n. Semillas = 1; una neurona
        recorrida por primera vez en el paso t tiene capa t; nunca alcanzada = 0.

Estas pruebas valen para cualquier forma razonable del modelo. La prueba que
distingue la regla «por neurona» de la regla «por arista» no está aquí: esa la
escribes tú cuando decidas cuál usar (ver fase1-guia.md).
"""
from pathlib import Path
import importlib.util
import sys
import time

import numpy as np
from scipy import sparse

ruta = Path(__file__).parent / "02_recorrido.py"
if not ruta.exists():
    sys.exit("Falta 02_recorrido.py")
spec = importlib.util.spec_from_file_location("recorrido", ruta)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def grafo(n, aristas):
    """aristas: lista de (origen, destino, sinapsis)."""
    o, d, w = zip(*aristas)
    return sparse.csr_matrix((w, (o, d)), shape=(n, n), dtype=float)


def prueba_normalizar():
    W = grafo(4, [(0, 2, 3), (1, 2, 7), (2, 3, 5)])
    copia = W.copy()
    A = sparse.csr_matrix(mod.normalizar(W))
    sumas = np.asarray(A.sum(axis=0)).ravel()
    assert np.allclose(sumas, [0, 0, 1, 1]), f"sumas por columna: {sumas}"
    assert np.isclose(A[0, 2], 0.3) and np.isclose(A[1, 2], 0.7)
    assert (W != copia).nnz == 0, "normalizar modificó W"


def prueba_semillas_capa_1():
    A = mod.normalizar(grafo(3, [(0, 1, 5), (1, 2, 5)]))
    capas = mod.recorrer(A, np.array([0]), np.random.default_rng(0))
    assert capas[0] == 1, f"la semilla debe tener capa 1, tiene {capas[0]}"


def prueba_cadena():
    # 0 -> 1 -> 2 -> 3 -> 4, cada una es el 100 % de la entrada de la siguiente
    A = mod.normalizar(grafo(5, [(i, i + 1, 10) for i in range(4)]))
    for s in range(20):
        capas = mod.recorrer(A, np.array([0]), np.random.default_rng(s))
        assert list(capas) == [1, 2, 3, 4, 5], f"cadena: {list(capas)}"


def prueba_treinta_por_ciento():
    # 2 recibe 30 % de la semilla 0 y 70 % de 1, que nadie alcanza: debe entrar en el paso 2
    A = mod.normalizar(grafo(3, [(0, 2, 3), (1, 2, 7)]))
    for s in range(500):
        capas = mod.recorrer(A, np.array([0]), np.random.default_rng(s))
        assert capas[2] == 2, f"con 30 % la probabilidad es 1; salió capa {capas[2]}"
        assert capas[1] == 0, "1 no tiene entradas: nunca se alcanza"


def prueba_direccion():
    # la información no viaja hacia atrás: 1 -> 0 no hace que 1 se recorra desde 0
    A = mod.normalizar(grafo(3, [(1, 0, 10), (0, 2, 10)]))
    capas = mod.recorrer(A, np.array([0]), np.random.default_rng(0))
    assert capas[1] == 0 and capas[2] == 2, f"dirección: {list(capas)}"


def prueba_reproducible():
    rng = np.random.default_rng(1)
    n = 300
    o, d = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    A = mod.normalizar(grafo(n, list(zip(o, d, rng.integers(1, 20, 3000)))))
    a = mod.recorrer(A, np.arange(5), np.random.default_rng(42))
    b = mod.recorrer(A, np.arange(5), np.random.default_rng(42))
    assert np.array_equal(a, b), "con la misma semilla aleatoria debe dar lo mismo"


def prueba_rapidez():
    # 20 mil neuronas y 200 mil aristas: una corrida debe tardar menos de 1 s.
    # Si falla, hay un ciclo de Python por arista o por neurona.
    rng = np.random.default_rng(2)
    n, m = 20_000, 200_000
    W = sparse.csr_matrix((rng.integers(1, 20, m).astype(float),
                           (rng.integers(0, n, m), rng.integers(0, n, m))), shape=(n, n))
    A = mod.normalizar(W)
    t = time.perf_counter()
    mod.recorrer(A, np.arange(100), np.random.default_rng(0))
    dt = time.perf_counter() - t
    assert dt < 1.0, f"una corrida tardó {dt:.2f} s"


def prueba_regla_por_arista():
    # Decisión (como navis): cada arista se sortea aparte. 3 recibe 10 % de cada semilla 0, 1, 2
    # y 70 % de 4, que nadie alcanza. En el paso 2: P = 1 - (1 - 1/3)^3 ≈ 0.704 (por neurona sería 1).
    A = mod.normalizar(grafo(5, [(0, 3, 1), (1, 3, 1), (2, 3, 1), (4, 3, 7)]))
    rng = np.random.default_rng(3)
    en_paso_2 = np.mean([mod.recorrer(A, np.array([0, 1, 2]), rng)[3] == 2 for _ in range(4000)])
    assert abs(en_paso_2 - (1 - (2 / 3) ** 3)) < 0.03, f"fracción en el paso 2: {en_paso_2:.3f}"


def prueba_reintento():
    # Decisión (como navis): quien falla se vuelve a sortear. Con 15 % de entrada (p = 0.5),
    # la capa es 1 + geométrica(0.5): media 3, y casi nunca queda sin alcanzar.
    A = mod.normalizar(grafo(3, [(0, 2, 15), (1, 2, 85)]))
    rng = np.random.default_rng(4)
    capas = np.array([mod.recorrer(A, np.array([0]), rng)[2] for _ in range(4000)])
    assert (capas > 0).mean() > 0.999, "con reintento casi siempre se alcanza"
    assert abs(capas[capas > 0].mean() - 3.0) < 0.1, f"capa media {capas[capas > 0].mean():.2f}, se esperaba 3"


PRUEBAS = [prueba_normalizar, prueba_semillas_capa_1, prueba_cadena,
           prueba_treinta_por_ciento, prueba_direccion, prueba_reproducible, prueba_rapidez,
           prueba_regla_por_arista, prueba_reintento]

if __name__ == "__main__":
    fallas = 0
    for p in PRUEBAS:
        try:
            p()
            print(f"✓ {p.__name__}")
        except Exception as e:  # noqa: BLE001 — se reporta cualquier falla
            fallas += 1
            print(f"✗ {p.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(PRUEBAS) - fallas}/{len(PRUEBAS)} pasan")
    sys.exit(1 if fallas else 0)
