"""Experimento prerregistrado (ver PREREGISTRO.md).

    python3 decisivo.py

Ajusta el motor con NSGA-II sobre validación y cinco semillas, elige con el mismo
criterio los cortes de los demás agregadores, y mide en prueba una sola vez.
"""
import copy
import json
import os

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import agregadores as A
import esquema as E
import experimento as X
import fis
import metricas as M

S = E.SENALES_FUSION
EXPO_MAX = 0.10          # prerregistrado
SB_MAX = 0.30            # prerregistrado
MARGEN = 0.02            # prerregistrado
SEMILLAS = [11, 23, 37, 41, 53]
POBLACION, GENERACIONES = 60, 80
SALIDA = "salidas"


def mide(nivel, d):
    y = d[E.COL_PELIGROSO].to_numpy()
    noe = d[E.COL_EXPLETIVO].to_numpy() == 1
    expuesto = nivel >= E.NIVEL_EXPONE
    return {
        "exposicion": M.exposicion_ponderada(nivel),
        "proteccion": float(expuesto[y == 1].mean()),
        "exposicion_mensajes": float(expuesto.mean()),
        "sobreexp_noe": float(expuesto[noe].mean()),
        "llamadas": float(((nivel >= 1) & (nivel < 3)).mean()),
    }


# --- motor difuso -----------------------------------------------------------

def decodifica(x, n_reglas):
    esc, j = {}, 0
    for s in S:
        c1, dc, de = x[j], x[j + 1], x[j + 2]
        esc[s] = (float(c1), float(c1 + dc), float(de))
        j += 3
    w = np.asarray(x[j:j + n_reglas], dtype=float); j += n_reglas
    d1, d2, d3 = x[j], x[j + 1], x[j + 2]
    return esc, w, (float(d1), float(d1 + d2), float(d1 + d2 + d3))


def construye(x, reglas_base):
    esc, w, cortes = decodifica(x, len(reglas_base))
    reglas = copy.deepcopy(reglas_base)
    for r, wk in zip(reglas, w):
        r.peso = float(wk)
    return fis.MotorEscala(reglas=reglas, escala=esc, senales=S), cortes


class Problema(Problem):
    def __init__(self, va, reglas):
        self.va, self.reglas = va, reglas
        self.Xv = va[S].to_numpy(dtype=float)
        lo, hi = [], []
        for _ in S:
            lo += [0.10, 0.15, 0.02]; hi += [0.50, 0.60, 0.15]
        lo += [0.0] * len(reglas); hi += [1.0] * len(reglas)
        lo += [2.0, 2.0, 2.0]; hi += [45.0, 45.0, 45.0]
        super().__init__(n_var=len(lo), n_obj=3, n_ieq_constr=2,
                         xl=np.array(lo), xu=np.array(hi))

    def _evaluate(self, P, out, *a, **k):
        F, G = [], []
        for x in P:
            m, c = construye(x, self.reglas)
            niv = M.a_nivel(m.riesgo(self.Xv), c)
            r = mide(niv, self.va)
            F.append([r["exposicion"], -r["proteccion"], r["llamadas"]])
            G.append([r["exposicion"] - EXPO_MAX, r["sobreexp_noe"] - SB_MAX])
        out["F"], out["G"] = np.array(F), np.array(G)


def ajusta_fis(va, semilla):
    reglas = fis.reglamento_v2()
    prob = Problema(va, reglas)
    res = minimize(prob, NSGA2(pop_size=POBLACION, sampling=FloatRandomSampling(),
                               crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                               eliminate_duplicates=True),
                   get_termination("n_gen", GENERACIONES), seed=semilla, verbose=False)
    if res.X is None:
        return None
    Xs = np.atleast_2d(res.X)
    # elegir en validación: máxima protección entre los que cumplen
    mejor, mp = None, -1
    for x in Xs:
        m, c = construye(x, reglas)
        r = mide(M.a_nivel(m.riesgo(prob.Xv), c), va)
        if r["exposicion"] <= EXPO_MAX and r["sobreexp_noe"] <= SB_MAX and r["proteccion"] > mp:
            mejor, mp = x, r["proteccion"]
    return None if mejor is None else construye(mejor, reglas)


# --- agregadores continuos con dos cortes ------------------------------------

def elige_cortes(r, d):
    qs = np.unique(np.quantile(r, np.linspace(0.20, 0.999, 160)))
    mejor, mp = None, -1
    for t2 in qs:
        for t3 in qs[qs > t2]:
            niv = np.where(r >= t3, 3, np.where(r >= t2, 2, 0))
            m = mide(niv, d)
            if m["exposicion"] <= EXPO_MAX and m["sobreexp_noe"] <= SB_MAX and m["proteccion"] > mp:
                mejor, mp = (t2, t3), m["proteccion"]
    return mejor


def nivel_continuo(r, cortes):
    t2, t3 = cortes
    return np.where(r >= t3, 3, np.where(r >= t2, 2, 0))


# --- bootstrap ----------------------------------------------------------------

def ic_bootstrap(nivel, d, clave, n=2000, semilla=0):
    rng = np.random.default_rng(semilla)
    vals = []
    idx = np.arange(len(d))
    for _ in range(n):
        b = rng.integers(0, len(d), len(d))
        vals.append(mide(nivel[b], d.iloc[b])[clave])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def principal():
    df = X._offendes()
    tr, va, te = [df[df[E.COL_SPLIT] == s] for s in ("train", "validation", "test")]
    filas, niveles_test = [], {}

    # agregadores continuos
    for ag in [A.MaximoActual(), A.MediaPonderada(S), A.Supervisado(S)]:
        ag.ajusta(tr)
        cortes = elige_cortes(ag.riesgo_df(va), va)
        if cortes is None:
            filas.append({"metodo": ag.nombre, "factible": False}); continue
        niv = nivel_continuo(ag.riesgo_df(te), cortes)
        niveles_test[ag.nombre] = niv
        filas.append({"metodo": ag.nombre, "factible": True, **mide(niv, te)})

    # motor difuso, cinco semillas
    fis_filas = []
    for sem in SEMILLAS:
        r = ajusta_fis(va, sem)
        if r is None:
            fis_filas.append({"semilla": sem, "factible": False}); continue
        m, c = r
        niv = M.a_nivel(m.riesgo(te[S].to_numpy(dtype=float)), c)
        niveles_test[f"FIS s{sem}"] = niv
        fis_filas.append({"semilla": sem, "factible": True, **mide(niv, te),
                          "cortes": c, "escala": m.escala,
                          "pesos": {rg.id: round(rg.peso, 3) for rg in m.reglas}})
        print(f"  semilla {sem}: " + ", ".join(f"{k}={v:.3f}" for k, v in mide(niv, te).items()))

    ok = [f for f in fis_filas if f["factible"]]
    fis_med = {k: float(np.mean([f[k] for f in ok])) for k in
               ["exposicion", "proteccion", "exposicion_mensajes", "sobreexp_noe", "llamadas"]} if ok else {}
    fis_sd = {k: float(np.std([f[k] for f in ok])) for k in fis_med}
    filas.append({"metodo": f"FIS (media de {len(ok)} semillas)", "factible": bool(ok), **fis_med})

    tabla = pd.DataFrame(filas)

    # intervalos para la logística y la mejor semilla mediana del FIS
    ics = {}
    log = A.Supervisado.nombre
    if log in niveles_test:
        ics[log] = {k: ic_bootstrap(niveles_test[log], te, k) for k in ("proteccion", "sobreexp_noe")}

    # regla de decisión
    decision = "plan B"
    motivo = ""
    fila_log = tabla[tabla.metodo == log]
    if ok and len(fila_log) and bool(fila_log.iloc[0]["factible"]):
        pl, sl = float(fila_log.iloc[0]["proteccion"]), float(fila_log.iloc[0]["sobreexp_noe"])
        pf, sf = fis_med["proteccion"], fis_med["sobreexp_noe"]
        c1 = pf >= pl - MARGEN and sf <= sl
        c2 = pf > pl and sf <= sl + MARGEN
        if c1 or c2:
            decision = "plan A: seguimos con el motor difuso como aportación"
        motivo = (f"proteccion FIS {pf:.3f} vs logistica {pl:.3f}; "
                  f"sobreexp. NOE FIS {sf:.3f} vs logistica {sl:.3f}; "
                  f"no inferioridad={c1}, superioridad={c2}")
    else:
        motivo = "algún método no encontró punto factible en validación"

    os.makedirs(SALIDA, exist_ok=True)
    tabla.to_csv(f"{SALIDA}/decisivo_tabla.csv", index=False)
    with open(f"{SALIDA}/decisivo.json", "w") as fh:
        json.dump({"tabla": filas, "fis_semillas": fis_filas, "fis_sd": fis_sd,
                   "ic95_logistica": ics, "decision": decision, "motivo": motivo},
                  fh, indent=2, ensure_ascii=False, default=str)

    pd.set_option("display.width", 200)
    print("\n== PRUEBA (una sola ejecución) ==")
    print(tabla.round(3).to_string(index=False))
    if fis_sd:
        print("\ndesviación típica del FIS entre semillas:", {k: round(v, 3) for k, v in fis_sd.items()})
    if ics:
        print("IC 95 % bootstrap de la logística:", {k: tuple(round(x, 3) for x in v) for k, v in ics[log].items()})
    print(f"\nDECISIÓN: {decision}\n  {motivo}")


if __name__ == "__main__":
    principal()
