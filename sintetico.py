"""Corpus sintético con el esquema definitivo, para depurar la tubería.

Las señales guardan una relación conocida con la etiqueta, incluido el caso NOE:
odio alto en mensajes que no son peligrosos.
"""
import numpy as np
import pandas as pd

import esquema as E


def _recorta(x):
    return np.clip(x, 0.0, 1.0)


def genera(n=6000, semilla=7, corpus="offendes"):
    """Devuelve un DataFrame con el esquema de esquema.COLUMNAS."""
    rng = np.random.default_rng(semilla)

    # Proporciones aproximadas al reparto real de OffendES: mayoría no ofensivo,
    # una fracción apreciable de malsonante benigno.
    etiquetas = rng.choice(["NO", "NOE", "OFG", "OFP"], size=n,
                           p=[0.55, 0.15, 0.15, 0.15])

    filas = []
    for i, et in enumerate(etiquetas):
        if et == "NO":
            odio = rng.beta(1.5, 12)
            emocion = rng.beta(2, 6)
            sentimiento = rng.beta(2, 4)
            malsonante = rng.beta(1.2, 12)
        elif et == "NOE":
            # malsonante sin intención de ofender: odio alto pero sin peligro
            odio = rng.beta(5, 4)
            emocion = rng.beta(2, 5)
            sentimiento = rng.beta(2, 5)
            malsonante = rng.beta(6, 3)
        elif et == "OFG":
            odio = rng.beta(5, 3)
            emocion = rng.beta(4, 4)
            sentimiento = rng.beta(5, 3)
            malsonante = rng.beta(3, 4)
        else:  # OFP
            # evidencia repartida y concordante
            odio = rng.beta(4, 4)
            emocion = rng.beta(5, 3)
            sentimiento = rng.beta(6, 3)
            malsonante = rng.beta(4, 4)

        # el phishing es independiente de la ofensa
        phishing = rng.beta(1.2, 20) if rng.random() > 0.05 else rng.beta(8, 2)

        norm = E.normaliza(et, "offendes")
        filas.append({
            E.COL_ID: f"{corpus}-{i:06d}",
            E.COL_TEXTO: f"[sintetico {et} {i}]",
            E.COL_CORPUS: corpus,
            E.COL_ORIGINAL: et,
            E.COL_PELIGROSO: norm[E.COL_PELIGROSO],
            E.COL_RIESGO_ORD: norm[E.COL_RIESGO_ORD],
            E.COL_EXPLETIVO: norm[E.COL_EXPLETIVO],
            "odio": _recorta(odio),
            "emocion": _recorta(emocion),
            "sentimiento": _recorta(sentimiento),
            "phishing": _recorta(phishing),
            "malsonante": _recorta(malsonante),
            # la ira es una parte de la emoción negativa
            "ira": _recorta(emocion * rng.uniform(0.3, 0.9)),
        })

    df = pd.DataFrame(filas)

    # partición 70/10/20; con datos reales se usan las oficiales
    u = rng.random(len(df))
    df[E.COL_SPLIT] = np.where(u < 0.70, "train",
                               np.where(u < 0.80, "validation", "test"))
    return df[E.COLUMNAS]


def genera_cruzado(n=4000, semilla=11):
    """Segundo corpus para la evaluación cruzada, con otra calibración."""
    df = genera(n=n, semilla=semilla, corpus="superset")
    rng = np.random.default_rng(semilla + 1)
    for s in E.SENALES:
        # desplazamiento y escala distintos, más ruido
        df[s] = _recorta(df[s] * rng.uniform(0.8, 1.2) + rng.normal(0, 0.05, len(df)))
    df[E.COL_RIESGO_ORD] = None
    df[E.COL_EXPLETIVO] = 0
    df[E.COL_ORIGINAL] = df[E.COL_PELIGROSO]
    df[E.COL_SPLIT] = "test"
    return df[E.COLUMNAS]


if __name__ == "__main__":
    d = genera()
    print(d.groupby(E.COL_ORIGINAL)[E.SENALES].mean().round(3))
    print()
    print(d[E.COL_ORIGINAL].value_counts())
    print()
    print(d[E.COL_SPLIT].value_counts())
