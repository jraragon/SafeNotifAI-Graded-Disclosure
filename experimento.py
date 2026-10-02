"""
Ejecuta el experimento completo de la fase 7 sobre el corpus sintético.

Cuando llegue el acceso a los corpus reales solo hay que cambiar `carga_datos`
para que lea los ficheros normalizados de la fase 1 en lugar de generarlos. Todo
lo demás queda igual.

    python3 experimento.py
"""
import json
import os
import warnings

import numpy as np
import pandas as pd

import agregadores as A
import esquema as E
import metricas as M
import sintetico

SENALES = E.SENALES_FUSION      # el experimento principal excluye el phishing

warnings.filterwarnings("ignore")

REPETICIONES = int(os.environ.get("REPS", 30))          # las mismas que el unico articulo del corpus con contraste completo
EXPOSICION_OBJETIVO = 0.10  # presupuesto de exposicion para la lectura operativa
SALIDA = "salidas"


RUTA_SENALES = os.path.join("datos", "senales_offendes.csv")
_CACHE = {}


def datos_reales():
    """True si ya existen las señales calculadas por sensores.py."""
    return os.path.exists(RUTA_SENALES)


def _offendes():
    if "df" not in _CACHE:
        import datos_offendes as D
        _CACHE["df"] = D.une_senales(D.carga(), RUTA_SENALES)
    return _CACHE["df"]


def carga_datos(semilla):
    """Devuelve (tr, te, tr_cruzado, te_cruzado).

    Con datos reales la prueba cruzada es entre plataformas: se ajusta con
    YouTube y se mide en Instagram y Twitter.

    Con el corpus fijo y métodos deterministas, repetir daría siempre el mismo
    número, así que cada repetición es un remuestreo bootstrap de la prueba.
    """
    if not datos_reales():
        df = sintetico.genera(n=6000, semilla=semilla)
        tr = df[df[E.COL_SPLIT] == "train"]
        te = df[df[E.COL_SPLIT] == "test"]
        cr = sintetico.genera_cruzado(n=3000, semilla=semilla + 500)
        return tr, te, tr, cr

    df = _offendes()
    tr = df[df[E.COL_SPLIT] == "train"]
    te = df[df[E.COL_SPLIT] == "test"]
    tr_c = tr[tr["media"] == "youtube"]
    te_c = te[te["media"] != "youtube"]
    rng = np.random.default_rng(semilla)
    te_b = te.iloc[rng.integers(0, len(te), len(te))]
    te_cb = te_c.iloc[rng.integers(0, len(te_c), len(te_c))]
    return tr, te_b, tr_c, te_cb


def evalua(ag, tr, te):
    """Ajusta en entrenamiento y mide en prueba."""
    ag.ajusta(tr)
    r = ag.riesgo_df(te)
    y = te[E.COL_PELIGROSO].to_numpy()
    ex, pr, _ = M.curva(r, y)
    prot, ex_real, corte = M.proteccion_a_exposicion(r, y, EXPOSICION_OBJETIVO)
    return {
        "auc_op": M.area_bajo_curva(ex, pr, ex_max=0.25),
        "auc_total": M.area_bajo_curva(ex, pr, ex_max=1.0),
        "hv": M.hipervolumen(np.column_stack([ex, pr])),
        "proteccion_al_10": prot,
        "exposicion_real": ex_real,
        "f1": M.f1(r, y, corte),
        "fpr": M.falsos_positivos(r, y, corte),
        "sobreexposicion_benigna": M.sobreexposicion_benigna(
            r, te[E.COL_EXPLETIVO].to_numpy(), corte),
        # punto real del sistema actual, con su umbral fijo de 50
        "exposicion_umbral_50": float((r >= 50).mean()) if isinstance(ag, A.MaximoActual) else np.nan,
        "proteccion_umbral_50": (float((r >= 50)[y == 1].mean())
                                 if isinstance(ag, A.MaximoActual) and y.sum() else np.nan),
        "_riesgo": r,
        "_corte": corte,
    }


def principal():
    print("DATOS:", "OffendES real" if datos_reales() else "SINTETICOS (no son resultados)")
    metodos = [a.nombre for a in A.todos()]
    acum = {m: {k: [] for k in
                ["auc_op", "auc_total", "hv", "proteccion_al_10", "exposicion_real",
                 "f1", "fpr", "sobreexposicion_benigna",
                 "exposicion_umbral_50", "proteccion_umbral_50"]}
            for m in metodos}
    cruzado = {m: [] for m in metodos}
    curvas_ultima = {}

    for rep in range(REPETICIONES):
        tr, te, tr_c, te_c = carga_datos(semilla=1000 + rep)
        for ag in A.todos():
            res = evalua(ag, tr, te)
            for k in acum[ag.nombre]:
                acum[ag.nombre][k].append(res[k])
            if rep == REPETICIONES - 1:
                ex, pr, _ = M.curva(res["_riesgo"], te[E.COL_PELIGROSO].to_numpy())
                curvas_ultima[ag.nombre] = (ex, pr)
        # cruzada: se reajusta en el origen y se mide en el destino
        for ag in A.todos():
            ag.ajusta(tr_c)
            rc = ag.riesgo_df(te_c)
            exc, prc, _ = M.curva(rc, te_c[E.COL_PELIGROSO].to_numpy())
            cruzado[ag.nombre].append(M.area_bajo_curva(exc, prc, ex_max=0.25))

    # --- tabla comparativa ---------------------------------------------------
    filas = []
    for m in metodos:
        f = {"metodo": m}
        for k, v in acum[m].items():
            f[k] = float(np.nanmean(v)) if not np.all(np.isnan(v)) else np.nan
            f[k + "_sd"] = float(np.nanstd(v)) if not np.all(np.isnan(v)) else np.nan
        f["auc_op_cruzado"] = float(np.mean(cruzado[m]))
        filas.append(f)
    tabla = pd.DataFrame(filas)

    # --- contrastes ----------------------------------------------------------
    ref = A.MotorFIS.nombre
    aucs = {m: acum[m]["auc_op"] for m in metodos}
    fried = M.friedman(aucs)
    wil = M.wilcoxon_contra(aucs, ref)

    # --- ablacion por sensor -------------------------------------------------
    abl = ablacion()

    # --- salidas -------------------------------------------------------------
    import os
    os.makedirs(SALIDA, exist_ok=True)
    tabla.to_csv(f"{SALIDA}/tabla_comparativa.csv", index=False)
    abl.to_csv(f"{SALIDA}/tabla_ablacion.csv", index=False)
    with open(f"{SALIDA}/contrastes.json", "w") as fh:
        json.dump({"friedman": fried, "wilcoxon_contra_fis": wil}, fh,
                  indent=2, ensure_ascii=False)
    figura(curvas_ultima)

    # --- informe por pantalla ------------------------------------------------
    cols = ["metodo", "auc_op", "auc_op_sd", "auc_total", "proteccion_al_10",
            "exposicion_real", "f1", "fpr", "sobreexposicion_benigna",
            "auc_op_cruzado"]
    print("\n== Comparativa de agregadores ==",
          f"({REPETICIONES} repeticiones, exposicion objetivo {EXPOSICION_OBJETIVO:.0%})\n")
    print(tabla[cols].round(3).to_string(index=False))

    fa = tabla[tabla.metodo == A.MaximoActual.nombre].iloc[0]
    print(f"\nSistema actual en su punto real (umbral 50): exposicion "
          f"{fa['exposicion_umbral_50']:.3f}, proteccion {fa['proteccion_umbral_50']:.3f}")

    print(f"\n== Friedman sobre el area operativa ==\n  chi2 = {fried['estadistico']:.2f}   p = {fried['p']:.2e}")
    print("  rangos medios (1 = mejor):")
    for k, v in sorted(fried["rangos"].items(), key=lambda kv: kv[1]):
        print(f"    {v:5.2f}  {k}")

    print(f"\n== Wilcoxon de '{ref}' contra el resto, con Holm ==")
    for w in wil:
        print(f"  {w['marca']}  {w['metodo']:<26} p_holm = {w['p_holm']:.2e}"
              f"   dif = {w['dif_media']:+.4f}")

    print("\n== Ablacion por sensor (area operativa) ==\n")
    print(abl.round(3).to_string(index=False))

    print(f"\nFicheros en {SALIDA}/")
    return tabla


def ablacion(repeticiones=10):
    """Quita cada señal y mide cuánto se pierde."""
    filas = []
    base = None
    for quitar in [None] + SENALES:
        vals = []
        for rep in range(repeticiones):
            tr, te, _, _ = carga_datos(semilla=2000 + rep)
            tr2, te2 = tr.copy(), te.copy()
            if quitar is not None:
                # anular la señal, no borrar la columna
                tr2[quitar] = 0.0
                te2[quitar] = 0.0
            ag = A.MotorFIS(senales=SENALES).ajusta(tr2)
            r = ag.riesgo(te2[SENALES].to_numpy())
            ex, pr, _ = M.curva(r, te2[E.COL_PELIGROSO].to_numpy())
            vals.append(M.area_bajo_curva(ex, pr, ex_max=0.25))
        media = float(np.mean(vals))
        if quitar is None:
            base = media
        filas.append({"configuracion": "completo" if quitar is None else f"sin {quitar}",
                      "auc": media,
                      "caida": 0.0 if quitar is None else base - media})
    return pd.DataFrame(filas)


def figura(curvas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    estilos = {A.MotorFIS.nombre: dict(lw=2.4, color="black", zorder=5),
               A.MaximoActual.nombre: dict(lw=1.8, ls="--", color="firebrick")}
    for nombre, (ex, pr) in curvas.items():
        ax.plot(ex, pr, label=nombre, **estilos.get(nombre, dict(lw=1.2, alpha=0.85)))
    ax.axvline(EXPOSICION_OBJETIVO, color="grey", lw=0.8, ls=":")
    ax.annotate("presupuesto de exposicion", (EXPOSICION_OBJETIVO, 0.04),
                xytext=(6, 0), textcoords="offset points", fontsize=8, color="grey")
    ax.set_xlabel("Exposicion: fraccion de mensajes que llegan al tutor")
    ax.set_ylabel("Proteccion: exhaustividad sobre mensajes peligrosos")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/exposicion_proteccion.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    principal()
