"""Métricas del trabajo.

  exposición  fracción de contenido del menor que llega al tutor
  protección  exhaustividad sobre los mensajes peligrosos
"""
import numpy as np
from scipy import stats

import esquema as E

# np.trapezoid solo existe desde numpy 2.0; antes se llamaba np.trapz
_trapecio = getattr(np, "trapezoid", None) or np.trapz


# --- Política de divulgación --------------------------------------------------

def a_nivel(riesgo, cortes=(25.0, 50.0, 75.0)):
    """Riesgo continuo a niveles de divulgación. El segundo corte decide qué ve el tutor."""
    r = np.asarray(riesgo, dtype=float)
    return np.digitize(r, np.asarray(cortes, dtype=float))


def expone(riesgo, corte):
    """Máscara booleana de mensajes que llegan al tutor (nivel >= 2)."""
    return np.asarray(riesgo, dtype=float) >= corte


# Cuánto revela cada nivel, en fracción del contenido del mensaje. Medir la
# exposición como binaria borra la diferencia entre el indicador agregado y el
# mensaje entero, y el optimizador acaba vaciando los niveles intermedios.
# El 0,25 es una decisión de diseño y entra en el análisis de sensibilidad.
REVELA = np.array([0.0, 0.0, 0.25, 1.0])


def exposicion_ponderada(nivel, revela=None):
    """Exposición como fracción de contenido revelado, no de mensajes marcados."""
    w = REVELA if revela is None else np.asarray(revela, dtype=float)
    return float(w[np.asarray(nivel, dtype=int)].mean())


# --- Métricas puntuales -------------------------------------------------------

def exposicion(riesgo, corte):
    return float(expone(riesgo, corte).mean())


def proteccion(riesgo, peligroso, corte):
    y = np.asarray(peligroso, dtype=int)
    if y.sum() == 0:
        return float("nan")
    return float(expone(riesgo, corte)[y == 1].mean())


def sobreexposicion_benigna(riesgo, expletivo, corte):
    """Fracción de mensajes malsonantes pero inocuos que llegan al tutor."""
    m = np.asarray(expletivo, dtype=int) == 1
    if m.sum() == 0:
        return float("nan")
    return float(expone(riesgo, corte)[m].mean())


def falsos_positivos(riesgo, peligroso, corte):
    y = np.asarray(peligroso, dtype=int)
    if (y == 0).sum() == 0:
        return float("nan")
    return float(expone(riesgo, corte)[y == 0].mean())


def f1(riesgo, peligroso, corte):
    y = np.asarray(peligroso, dtype=int)
    p = expone(riesgo, corte).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec))


# --- La curva ----------------------------------------------------------------

def curva(riesgo, peligroso, n=101):
    """Curva de exposición frente a protección, barriendo el corte."""
    r = np.asarray(riesgo, dtype=float)
    cortes = np.linspace(float(r.min()), float(r.max()) + 1e-9, n)
    ex = np.array([exposicion(r, c) for c in cortes])
    pr = np.array([proteccion(r, peligroso, c) for c in cortes])
    o = np.argsort(ex)
    return ex[o], pr[o], cortes[o]


def area_bajo_curva(ex, pr, ex_max=1.0, normaliza=True):
    """Área bajo la curva, limitada al tramo operativo.

    Sobre todo el rango daría peso a exposiciones del 60 u 80 %, que no son
    desplegables, y premiaría al método que ordena bien lo irrelevante.
    """
    ex = np.asarray(ex, dtype=float)
    pr = np.asarray(pr, dtype=float)
    o = np.argsort(ex)
    ex, pr = ex[o], pr[o]
    m = ex <= ex_max + 1e-12
    if m.sum() < 2:
        return 0.0
    a = float(_trapecio(pr[m], ex[m]))
    return a / ex_max if normaliza else a


def proteccion_a_exposicion(riesgo, peligroso, ex_objetivo, n=401):
    """Protección con un presupuesto de exposición dado.

    Cortar por el cuantil no garantiza la exposición pedida: con pocas reglas la
    puntuación tiene mesetas y el corte cae dentro. Se interpola sobre la curva y
    se devuelve también la exposición alcanzada.

    Devuelve (proteccion, exposicion_real, corte).
    """
    r = np.asarray(riesgo, dtype=float)
    ex, pr, cortes = curva(r, peligroso, n=n)

    # punto alcanzable más cercano por debajo del presupuesto
    viables = np.where(ex <= ex_objetivo + 1e-12)[0]
    if len(viables) == 0:
        return float(pr[0]), float(ex[0]), float(cortes[0])
    i = viables[np.argmax(ex[viables])]

    # interpolación con el siguiente punto
    if i + 1 < len(ex) and ex[i + 1] > ex[i]:
        t = (ex_objetivo - ex[i]) / (ex[i + 1] - ex[i])
        t = float(np.clip(t, 0.0, 1.0))
        pr_i = float(pr[i] + t * (pr[i + 1] - pr[i]))
    else:
        pr_i = float(pr[i])
    return pr_i, float(ex[i]), float(cortes[i])


# --- Pareto ------------------------------------------------------------------

def frente_pareto(puntos):
    """
    Puntos no dominados, con exposición a minimizar y protección a maximizar.
    `puntos` es un array (n, 2) de (exposicion, proteccion).
    """
    P = np.asarray(puntos, dtype=float)
    keep = np.ones(len(P), dtype=bool)
    for i in range(len(P)):
        if not keep[i]:
            continue
        dom = (P[:, 0] <= P[i, 0]) & (P[:, 1] >= P[i, 1])
        dom[i] = False
        estricto = dom & ((P[:, 0] < P[i, 0]) | (P[:, 1] > P[i, 1]))
        if estricto.any():
            keep[i] = False
    return P[keep]


def hipervolumen(puntos, ref=(1.0, 0.0)):
    """
    Hipervolumen en dos dimensiones respecto al punto de referencia
    (exposición máxima, protección nula).
    """
    F = frente_pareto(puntos)
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0])]
    hv, x_prev = 0.0, ref[0]
    for x, y in F[::-1]:
        hv += max(x_prev - x, 0.0) * max(y - ref[1], 0.0)
        x_prev = x
    return float(hv)


# --- Contrastes --------------------------------------------------------------

def holm(pvals, nombres=None):
    """Corrección de Holm. Devuelve lista de (nombre, p, p_ajustada)."""
    p = np.asarray(pvals, dtype=float)
    k = len(p)
    o = np.argsort(p)
    ajust = np.empty(k)
    acum = 0.0
    for rango, idx in enumerate(o):
        val = (k - rango) * p[idx]
        acum = max(acum, val)
        ajust[idx] = min(acum, 1.0)
    nombres = nombres if nombres is not None else [str(i) for i in range(k)]
    return list(zip(nombres, p.tolist(), ajust.tolist()))


def friedman(medidas_por_metodo):
    """Friedman sobre {metodo: [valor por repeticion]}, con los rangos medios."""
    nombres = list(medidas_por_metodo)
    M = np.array([medidas_por_metodo[n] for n in nombres], dtype=float)
    est, p = stats.friedmanchisquare(*M)
    # rangos medios, 1 al mejor
    rangos = np.apply_along_axis(stats.rankdata, 0, -M).mean(axis=1)
    return {"estadistico": float(est), "p": float(p),
            "rangos": dict(zip(nombres, rangos.round(3).tolist()))}


def wilcoxon_contra(medidas_por_metodo, referencia):
    """Wilcoxon pareado contra la referencia, con corrección de Holm."""
    ref = np.asarray(medidas_por_metodo[referencia], dtype=float)
    otros = [n for n in medidas_por_metodo if n != referencia]
    pvals, difs = [], []
    for n in otros:
        v = np.asarray(medidas_por_metodo[n], dtype=float)
        if np.allclose(ref, v):
            pvals.append(1.0)
        else:
            pvals.append(float(stats.wilcoxon(ref, v).pvalue))
        difs.append(float(np.mean(ref - v)))
    salida = []
    for (n, p, pa), d in zip(holm(pvals, otros), difs):
        if pa >= 0.05:
            marca = "="
        else:
            marca = "+" if d > 0 else "-"
        salida.append({"metodo": n, "p": p, "p_holm": pa,
                       "dif_media": d, "marca": marca})
    return salida
