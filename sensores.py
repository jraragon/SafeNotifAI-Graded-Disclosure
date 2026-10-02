"""Cálculo de las cuatro señales sobre OffendES.

Necesita PyTorch y descarga los modelos de Hugging Face, así que se ejecuta en
local. Guarda por bloques y se reanuda si se interrumpe.

    pip install pysentimiento pandas
    python3 sensores.py --glosario datos/lexico_malsonante.csv

Salida: datos/senales_offendes.csv con id, odio, emocion, sentimiento,
malsonante e ira. El resto de scripts solo lee ese fichero.

  odio         P(hateful)
  emocion      suma de anger, disgust, fear y sadness
  sentimiento  P(NEG)
  malsonante   noisy-OR sobre el léxico, ver grado_malsonante
  ira          P(anger), solo para la línea base
"""
import argparse
import os
import re
import sys
import time
import unicodedata

import pandas as pd

import datos_offendes as D
import esquema as E

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "senales_offendes.csv")
BLOQUE = 2000
MAX_CHARS = 1000        # los comentarios muy largos se recortan: el modelo trunca igualmente

EMOCIONES_NEGATIVAS = {"anger", "disgust", "fear", "sadness"}


# --- Jerga: de binario a grado ------------------------------------------------

def _normaliza(t):
    t = unicodedata.normalize("NFKD", str(t).lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()


def carga_glosario(ruta):
    """CSV con columnas `termino` y `peso`, con el peso en [0, 1]."""
    g = pd.read_csv(ruta)
    if not {"termino", "peso"} <= set(g.columns):
        raise ValueError("el glosario necesita las columnas 'termino' y 'peso'")
    g["termino"] = g["termino"].map(_normaliza)
    g = g[g["termino"].str.len() > 0].drop_duplicates("termino")
    patrones = [(re.compile(r"(?<!\w)" + re.escape(t) + r"(?!\w)"), float(p))
                for t, p in zip(g["termino"], g["peso"].clip(0, 1))]
    return patrones


def grado_malsonante(texto, patrones):
    """Grado de registro malsonante en [0, 1], por noisy-OR sobre el léxico.

    Un término de peso 0,5 da 0,5; dos, 0,75. No se normaliza por longitud.
    """
    t = _normaliza(texto)
    resto = 1.0
    for patron, peso in patrones:
        n = len(patron.findall(t))
        if n:
            resto *= (1.0 - peso) ** n
    return 1.0 - resto


# --- Modelos -----------------------------------------------------------------

def _proba(salida, clave):
    """Probabilidad de una etiqueta sin depender de mayúsculas."""
    m = {k.lower(): v for k, v in salida.probas.items()}
    return float(m.get(clave.lower(), 0.0))


def calcula(textos, analizadores, patrones):
    textos = [str(t)[:MAX_CHARS] for t in textos]
    odio = analizadores["hate_speech"].predict(textos)
    emo = analizadores["emotion"].predict(textos)
    sen = analizadores["sentiment"].predict(textos)
    filas = []
    for i, t in enumerate(textos):
        pe = {k.lower(): v for k, v in emo[i].probas.items()}
        filas.append({
            "odio": _proba(odio[i], "hateful"),
            "emocion": float(min(1.0, sum(pe.get(e, 0.0) for e in EMOCIONES_NEGATIVAS))),
            # solo para la línea base
            "ira": float(pe.get("anger", 0.0)),
            "sentimiento": _proba(sen[i], "NEG"),
            "malsonante": grado_malsonante(t, patrones),
        })
    return filas


def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glosario", required=True, help="CSV con columnas termino,peso")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    from pysentimiento import create_analyzer

    patrones = carga_glosario(args.glosario)
    print(f"glosario: {len(patrones)} términos")

    analizadores = {t: create_analyzer(task=t, lang="es", batch_size=args.batch)
                    for t in ("hate_speech", "emotion", "sentiment")}
    for t, a in analizadores.items():
        print(f"  {t}: etiquetas {list(a.id2label.values())}")
    # comprobación de que las etiquetas son las que el script espera
    et_emo = {v.lower() for v in analizadores["emotion"].id2label.values()}
    if not EMOCIONES_NEGATIVAS <= et_emo:
        sys.exit(f"etiquetas de emoción inesperadas: {et_emo}")

    df = D.carga()
    hechos = set()
    if os.path.exists(SALIDA):
        hechos = set(pd.read_csv(SALIDA, usecols=[E.COL_ID], dtype=str)[E.COL_ID])
        print(f"reanudando: {len(hechos)} ya calculados")
    pend = df[~df[E.COL_ID].isin(hechos)]

    t0 = time.time()
    for ini in range(0, len(pend), BLOQUE):
        b = pend.iloc[ini:ini + BLOQUE]
        filas = calcula(b[E.COL_TEXTO].tolist(), analizadores, patrones)
        out = pd.DataFrame(filas)
        out.insert(0, E.COL_ID, b[E.COL_ID].to_numpy())
        out.to_csv(SALIDA, mode="a", index=False, header=not os.path.exists(SALIDA))
        hecho = ini + len(b)
        ritmo = hecho / max(time.time() - t0, 1e-6)
        print(f"  {hecho}/{len(pend)}  ({ritmo:.1f} textos/s, "
              f"quedan ~{(len(pend) - hecho) / max(ritmo, 1e-6) / 60:.0f} min)")

    s = pd.read_csv(SALIDA, dtype={E.COL_ID: str})
    print(f"\nlisto: {len(s)} filas en {SALIDA}")
    print(s[E.SENALES_FUSION].describe().round(3).to_string())


if __name__ == "__main__":
    principal()
