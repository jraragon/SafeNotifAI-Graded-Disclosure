"""Chequeo post-hoc de alineación con el objetivo.

Política lineal r = w·x (4 señales) con dos cortes (t2, t3 = t2 + dt), niveles {0,2,3}.
Se ajusta en VALIDACIÓN con el mismo criterio que el FIS: máxima protección sujeta a
E <= 0.10 y B <= 0.30 (restricciones penalizadas), con evolución diferencial de scipy.
Se mide una sola vez en PRUEBA. Tres semillas del optimizador.
Guarda salidas/posthoc_lineal.json y salidas/posthoc_lineal_log.txt.
"""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import esquema as E, experimento as X, decisivo as D
from scipy.optimize import differential_evolution

S = E.SENALES_FUSION
df = X._offendes(); tr, va, te = [df[df.split == s] for s in ("train", "validation", "test")]
Xv, Xt = va[S].to_numpy(float), te[S].to_numpy(float)

def lv(Xm, p):
    w = np.array(p[:4]); r = Xm @ w; t2, t3 = p[4], p[4] + p[5]
    return np.where(r >= t3, 3, np.where(r >= t2, 2, 0))

def obj(p):
    m = D.mide(lv(Xv, p), va)
    pen = max(0, m["exposicion"] - 0.10) * 50 + max(0, m["sobreexp_noe"] - 0.30) * 50
    return -m["proteccion"] + pen

out, lineas = [], []
def log(s): print(s); lineas.append(s)
for name, lo in [("linear monotone (w>=0)", 0.0), ("linear free (w in [-1,1])", -1.0)]:
    prot = []
    for seed in [1, 2, 3]:
        b = [(lo, 1)] * 4 + [(-1, 2.5), (0, 2.5)]
        r = differential_evolution(obj, b, seed=seed, maxiter=150, popsize=30, tol=1e-8, polish=False)
        mv, mt = D.mide(lv(Xv, r.x), va), D.mide(lv(Xt, r.x), te)
        niv = lv(Xt, r.x); frac = (np.bincount(niv, minlength=4) / len(niv)).round(3).tolist()
        prot.append(mt["proteccion"])
        log(f"{name} seed={seed} weights {dict(zip(S, np.round(r.x[:4], 3).tolist()))} "
            f"cuts t2={r.x[4]:.3f} t3={r.x[4]+r.x[5]:.3f}")
        log(f"   val : P={mv['proteccion']:.3f} E={mv['exposicion']:.3f} B={mv['sobreexp_noe']:.3f} I_G={mv['exposicion_mensajes']:.3f}")
        log(f"   test: P={mt['proteccion']:.3f} E={mt['exposicion']:.3f} B={mt['sobreexp_noe']:.3f} I_G={mt['exposicion_mensajes']:.3f} levels(L0..L3)={frac}")
        out.append({"policy": name, "seed": seed, "weights": dict(zip(S, r.x[:4].tolist())),
                    "t2": float(r.x[4]), "t3": float(r.x[4] + r.x[5]), "val": mv, "test": mt, "test_levels": frac})
    log(f"{name}: mean test P = {np.mean(prot):.3f}  (range {min(prot):.3f}-{max(prot):.3f})")
json.dump(out, open("salidas/posthoc_lineal.json", "w"), indent=2)
open("salidas/posthoc_lineal_log.txt", "w").write("\n".join(lineas) + "\n")
