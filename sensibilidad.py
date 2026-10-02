"""Sensibilidad a los parámetros que no salen de los datos.

Se perturban los cortes por señal, el solape y los pesos de REVELA con ruido
normal, y se compara el método de desfuzzificación.
"""
import numpy as np
import pandas as pd

import agregadores as A
import esquema as E
import fis
import metricas as M

SENALES = E.SENALES_FUSION


def _mide(motor_kwargs, tr, te, cortes=(25.0, 50.0, 75.0), revela=None):
    ag = A.MotorFIS(senales=SENALES, **motor_kwargs).ajusta(tr)
    r = ag.riesgo(te[SENALES].to_numpy(dtype=float))
    y = te[E.COL_PELIGROSO].to_numpy(dtype=int)
    ex, pr, _ = M.curva(r, y)
    nivel = M.a_nivel(r, cortes)
    return {
        "area_op": M.area_bajo_curva(ex, pr, ex_max=0.25),
        "exposicion": M.exposicion_ponderada(nivel, revela),
        "proteccion": float((nivel >= E.NIVEL_EXPONE)[y == 1].mean()) if y.sum() else np.nan,
    }


def sensibilidad(tr, te, repeticiones=40, semilla=0):
    rng = np.random.default_rng(semilla)
    base = _mide({}, tr, te)
    filas = []

    def anota(grupo, sigma, res):
        for k, v in res.items():
            filas.append({"grupo": grupo, "sigma": sigma, "metrica": k,
                          "valor": v, "desvio_rel": (v - base[k]) / abs(base[k])
                          if base[k] else np.nan})

    # 1. niveles de corte de cuantil
    for sigma in (0.02, 0.05, 0.10):
        for _ in range(repeticiones):
            probs = {}
            for s in SENALES:
                p1, p2 = fis.PROBS_POR_SENAL[s]
                p1n = float(np.clip(p1 + rng.normal(0, sigma), 0.05, 0.90))
                p2n = float(np.clip(p2 + rng.normal(0, sigma), p1n + 0.03, 0.99))
                probs[s] = (p1n, p2n)
            anota("cortes de cuantil", sigma, _mide({"probs": probs}, tr, te))

    # 2. solape
    for sigma in (0.02, 0.05, 0.10):
        for _ in range(repeticiones):
            sol = float(np.clip(0.15 + rng.normal(0, sigma), 0.01, 0.5))
            anota("solape", sigma, _mide({"solape": sol}, tr, te))

    # 3. pesos de revelacion por nivel
    for sigma in (0.05, 0.10):
        for _ in range(repeticiones):
            w = M.REVELA.copy()
            w[2] = float(np.clip(w[2] + rng.normal(0, sigma), 0.02, 0.9))
            anota("pesos de revelacion", sigma, _mide({}, tr, te, revela=w))

    # 4. metodo de desfuzzificacion (discreto)
    for met in ("altura", "centroide"):
        anota("desfuzzificacion", met, _mide({"metodo": met}, tr, te))

    df = pd.DataFrame(filas)
    resumen = (df.groupby(["grupo", "sigma", "metrica"])["desvio_rel"]
                 .agg(media="mean", p95=lambda x: np.percentile(np.abs(x), 95))
                 .reset_index())
    return base, resumen


if __name__ == "__main__":
    import sintetico

    df = sintetico.genera(6000, semilla=1000)
    tr = df[df[E.COL_SPLIT] == "train"]
    te = df[df[E.COL_SPLIT] == "test"]

    base, res = sensibilidad(tr, te, repeticiones=30)
    print("base:", {k: round(v, 4) for k, v in base.items()})
    print("\ndesvio relativo respecto a la base (media y percentil 95 del valor absoluto)\n")
    print(res.round(4).to_string(index=False))
