"""Curva completa, ablación, sensibilidad y transferencia entre plataformas.

    python3 experimentos_finales.py

Los puntos de operación se eligen en validación y se miden en prueba. Deja en
salidas/ la figura de la frontera y las cuatro tablas.
"""
import json
import os

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import agregadores as A
import decisivo as D
import esquema as E
import experimento as X
import fis
import metricas as M

S = E.SENALES_FUSION
SALIDA = "salidas"
SEMILLAS = [11, 23, 37]
PRESUPUESTOS = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25]


# --- frentes ------------------------------------------------------------------

def frente_fis(tr_va, va, semilla, senales=S):
    """Soluciones no dominadas en validación.

    Se relaja la restricción de exposición para recorrer la curva entera; la de
    sobreexposición benigna se mantiene.
    """
    reglas = fis.reglamento_v2()
    prob = D.Problema(va, reglas)
    prob.xl, prob.xu = prob.xl, prob.xu
    # relajar la restricción de exposición para barrer el rango entero
    class P2(D.Problema):
        def _evaluate(self, P, out, *a, **k):
            F, G = [], []
            for x in P:
                m, c = D.construye(x, self.reglas)
                r = D.mide(M.a_nivel(m.riesgo(self.Xv), c), self.va)
                F.append([r["exposicion"], -r["proteccion"], r["llamadas"]])
                G.append([r["exposicion"] - 0.30, r["sobreexp_noe"] - D.SB_MAX])
            out["F"], out["G"] = np.array(F), np.array(G)
    prob = P2(va, reglas)
    res = minimize(prob, NSGA2(pop_size=D.POBLACION, sampling=FloatRandomSampling(),
                               crossover=SBX(prob=0.9, eta=15), mutation=PM(eta=20),
                               eliminate_duplicates=True),
                   get_termination("n_gen", D.GENERACIONES), seed=semilla, verbose=False)
    return [] if res.X is None else list(np.atleast_2d(res.X))


def puntos_fis(soluciones, va, te):
    Xv, Xt = va[S].to_numpy(float), te[S].to_numpy(float)
    out = []
    for x in soluciones:
        m, c = D.construye(x, fis.reglamento_v2())
        v = D.mide(M.a_nivel(m.riesgo(Xv), c), va)
        t = D.mide(M.a_nivel(m.riesgo(Xt), c), te)
        out.append({"val": v, "test": t, "x": x})
    return out


def puntos_continuo(ag, tr, va, te, n=60):
    """Rejilla de cortes, quedándose con los no dominados en validación."""
    ag.ajusta(tr)
    rv, rt = ag.riesgo_df(va), ag.riesgo_df(te)
    qs = np.unique(np.quantile(rv, np.linspace(0.20, 0.999, n)))
    cand = []
    for t2 in qs:
        for t3 in qs[qs > t2]:
            v = D.mide(D.nivel_continuo(rv, (t2, t3)), va)
            if v["sobreexp_noe"] <= D.SB_MAX:
                cand.append(((t2, t3), v))
    # no dominados en validación: minimizar exposición, maximizar protección
    P = np.array([[c[1]["exposicion"], c[1]["proteccion"]] for c in cand])
    keep = []
    for i in range(len(P)):
        dom = (P[:, 0] <= P[i, 0]) & (P[:, 1] >= P[i, 1])
        dom[i] = False
        if not (dom & ((P[:, 0] < P[i, 0]) | (P[:, 1] > P[i, 1]))).any():
            keep.append(i)
    return [{"val": cand[i][1], "test": D.mide(D.nivel_continuo(rt, cand[i][0]), te),
             "cortes": cand[i][0]} for i in keep]


def en_presupuesto(puntos, b):
    """Mejor punto de validación dentro del presupuesto b."""
    ok = [p for p in puntos if p["val"]["exposicion"] <= b]
    if not ok:
        return None
    return max(ok, key=lambda p: p["val"]["proteccion"])


def principal():
    os.makedirs(SALIDA, exist_ok=True)
    df = X._offendes()
    tr, va, te = [df[df[E.COL_SPLIT] == s] for s in ("train", "validation", "test")]
    res = {}

    print("== frentes ==")
    fis_pts = []
    for sem in SEMILLAS:
        sols = frente_fis(tr, va, sem)
        fis_pts += puntos_fis(sols, va, te)
        print(f"  FIS semilla {sem}: {len(sols)} soluciones")
    metodos = {"FIS (propuesta)": fis_pts}
    for ag in [A.MaximoActual(), A.MediaPonderada(S), A.Supervisado(S)]:
        metodos[ag.nombre] = puntos_continuo(ag, tr, va, te)
        print(f"  {ag.nombre}: {len(metodos[ag.nombre])} puntos no dominados en validación")

    # --- tabla por presupuesto de exposición ---------------------------------
    filas = []
    for b in PRESUPUESTOS:
        for nom, pts in metodos.items():
            p = en_presupuesto(pts, b)
            filas.append({"presupuesto": b, "metodo": nom,
                          **({} if p is None else
                             {"exposicion": p["test"]["exposicion"],
                              "proteccion": p["test"]["proteccion"],
                              "sobreexp_noe": p["test"]["sobreexp_noe"],
                              "llamadas": p["test"]["llamadas"]})})
    tabla_b = pd.DataFrame(filas)
    tabla_b.to_csv(f"{SALIDA}/tabla_presupuestos.csv", index=False)
    print("\n== protección en prueba por presupuesto de exposición ==")
    piv = tabla_b.pivot(index="presupuesto", columns="metodo", values="proteccion")
    print(piv.round(3).to_string())
    res["presupuestos"] = piv.round(4).to_dict()

    # dominancia: en cuántos presupuestos gana el FIS
    gana = int((piv["FIS (propuesta)"] >= piv.drop(columns="FIS (propuesta)").max(axis=1)).sum())
    print(f"\nel FIS es el mejor en {gana} de {len(piv)} presupuestos")
    res["presupuestos_ganados"] = [gana, len(piv)]

    # --- figura ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    estilo = {"FIS (propuesta)": dict(color="black", marker="o", s=16, zorder=5),
              "max (sistema actual)": dict(color="firebrick", marker="s", s=12),
              "media ponderada": dict(color="tab:blue", marker="^", s=12),
              "supervisado (logistica)": dict(color="tab:green", marker="v", s=12)}
    for nom, pts in metodos.items():
        P = np.array([[p["test"]["exposicion"], p["test"]["proteccion"]] for p in pts])
        P = P[P[:, 0] <= 0.32]
        ax.scatter(P[:, 0], P[:, 1], label=nom, alpha=0.75, **estilo[nom])
    ax.set_xlabel("Exposición: fracción de contenido revelado al tutor")
    ax.set_ylabel("Protección: exhaustividad sobre mensajes peligrosos")
    ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{SALIDA}/curva_frontera.png", dpi=200); plt.close(fig)

    # --- ablación por señal ---------------------------------------------------
    print("\n== ablación (presupuesto 0.10, 2 semillas) ==")
    filas = []
    for quitar in [None] + S:
        vals = []
        for sem in SEMILLAS[:2]:
            va2, te2 = va.copy(), te.copy()
            if quitar:
                va2[quitar] = 0.0; te2[quitar] = 0.0
            pts = puntos_fis(frente_fis(tr, va2, sem), va2, te2)
            p = en_presupuesto(pts, 0.10)
            if p: vals.append(p["test"]["proteccion"])
        filas.append({"configuracion": "completo" if quitar is None else f"sin {quitar}",
                      "proteccion": float(np.mean(vals)) if vals else np.nan})
    tabla_a = pd.DataFrame(filas)
    tabla_a["caida"] = tabla_a["proteccion"].iloc[0] - tabla_a["proteccion"]
    tabla_a.to_csv(f"{SALIDA}/tabla_ablacion_final.csv", index=False)
    print(tabla_a.round(3).to_string(index=False))
    res["ablacion"] = tabla_a.round(4).to_dict("records")

    # --- sensibilidad ---------------------------------------------------------
    print("\n== sensibilidad del punto de operación ==")
    base = en_presupuesto(puntos_fis(frente_fis(tr, va, 37), va, te), 0.10)
    x0 = base["x"]
    m0, c0 = D.construye(x0, fis.reglamento_v2())
    Xt = te[S].to_numpy(float)
    ref = D.mide(M.a_nivel(m0.riesgo(Xt), c0), te)
    filas = [{"variacion": "referencia", **{k: ref[k] for k in ("exposicion", "proteccion", "sobreexp_noe")}}]
    rng = np.random.default_rng(0)
    for sigma in (0.02, 0.05, 0.10):
        vs = []
        for _ in range(20):
            x = np.clip(x0 + np.concatenate([rng.normal(0, sigma, 3 * len(S)),
                                             np.zeros(len(m0.reglas) + 3)]), m0 and 0.01, None)
            m, c = D.construye(x, fis.reglamento_v2())
            vs.append(D.mide(M.a_nivel(m.riesgo(Xt), c), te))
        filas.append({"variacion": f"cortes de pertenencia sigma={sigma}",
                      **{k: float(np.mean([v[k] for v in vs])) for k in ("exposicion", "proteccion", "sobreexp_noe")}})
    for w2 in (0.10, 0.40):
        rho = M.REVELA.copy(); rho[2] = w2
        niv = M.a_nivel(m0.riesgo(Xt), c0)
        filas.append({"variacion": f"peso del indicador = {w2}",
                      "exposicion": M.exposicion_ponderada(niv, rho),
                      "proteccion": ref["proteccion"], "sobreexp_noe": ref["sobreexp_noe"]})
    m1 = D.construye(x0, fis.reglamento_v2())[0]
    v = D.mide(M.a_nivel(m1.riesgo(Xt, metodo="centroide"), c0), te)
    filas.append({"variacion": "desfuzzificación por centroide",
                  **{k: v[k] for k in ("exposicion", "proteccion", "sobreexp_noe")}})
    tabla_s = pd.DataFrame(filas)
    tabla_s.to_csv(f"{SALIDA}/tabla_sensibilidad.csv", index=False)
    print(tabla_s.round(3).to_string(index=False))
    res["sensibilidad"] = tabla_s.round(4).to_dict("records")

    # --- transferencia entre plataformas --------------------------------------
    print("\n== transferencia: ajuste con YouTube, prueba en Instagram y Twitter ==")
    tr_y, va_y = tr[tr.media == "youtube"], va[va.media == "youtube"]
    te_o = te[te.media != "youtube"]
    filas = []
    vals = []
    for sem in SEMILLAS[:2]:
        pts = puntos_fis(frente_fis(tr_y, va_y, sem), va_y, te_o)
        p = en_presupuesto(pts, 0.10)
        if p: vals.append(p["test"])
    if vals:
        filas.append({"metodo": "FIS (propuesta)",
                      **{k: float(np.mean([v[k] for v in vals])) for k in
                         ("exposicion", "proteccion", "sobreexp_noe")}})
    for ag in [A.MaximoActual(), A.MediaPonderada(S), A.Supervisado(S)]:
        pts = puntos_continuo(ag, tr_y, va_y, te_o)
        p = en_presupuesto(pts, 0.10)
        if p:
            filas.append({"metodo": ag.nombre,
                          **{k: p["test"][k] for k in ("exposicion", "proteccion", "sobreexp_noe")}})
    tabla_t = pd.DataFrame(filas)
    tabla_t.to_csv(f"{SALIDA}/tabla_transferencia.csv", index=False)
    print(tabla_t.round(3).to_string(index=False))
    res["transferencia"] = tabla_t.round(4).to_dict("records")

    with open(f"{SALIDA}/finales.json", "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\nFicheros en {SALIDA}/")


if __name__ == "__main__":
    principal()
