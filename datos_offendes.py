"""Carga y normalización de OffendES.

Origen: https://github.com/fmplaza/OffendES (Apache-2.0).

    git clone --depth 1 https://github.com/fmplaza/OffendES.git
    cp -r OffendES/split_MeOffendES datos/offendes

Se usan las particiones oficiales de MeOffendES con dos correcciones: se quitan
del entrenamiento los 39 comentarios que también aparecen en prueba, y la
validación se extrae del entrenamiento, porque la oficial solo tiene 100
comentarios. La prueba no se toca.
"""
import csv
import os

import numpy as np
import pandas as pd

import esquema as E

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos", "offendes")
FICHEROS = {"train": "training_set.tsv", "dev_oficial": "dev_set.tsv", "test": "test_set.tsv"}

FRACCION_VALIDACION = 0.15
SEMILLA_PARTICION = 20240601


def _lee(ruta):
    """Lee un TSV de MeOffendES.

    Hay una fila con un tabulador dentro del comentario, así que no vale el
    parser por defecto: los cuatro últimos campos son fijos y el resto es texto.
    """
    filas = []
    with open(ruta, encoding="utf-8") as fh:
        cab = fh.readline().rstrip("\n").split("\t")
        assert cab == ["comment_id", "comment", "influencer", "influencer_gender",
                       "media", "label"], cab
        for linea in fh:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 6:
                continue
            cid = partes[0]
            influencer, genero, media, etiqueta = partes[-4:]
            texto = "\t".join(partes[1:-4])
            filas.append((cid, texto, influencer, genero, media, etiqueta))
    return pd.DataFrame(filas, columns=["comment_id", "comment", "influencer",
                                        "influencer_gender", "media", "label"])


def carga(dir_=DIR):
    """Devuelve un DataFrame con el esquema común más los metadatos de OffendES."""
    partes = []
    for split, fichero in FICHEROS.items():
        d = _lee(os.path.join(dir_, fichero))
        d["split"] = split
        partes.append(d)
    df = pd.concat(partes, ignore_index=True)

    # fuga: fuera del entrenamiento todo texto que también esté en prueba
    normal = df["comment"].map(lambda t: " ".join(str(t).lower().split()))
    en_test = set(normal[df["split"] == "test"])
    fuga = (df["split"] != "test") & normal.isin(en_test)
    df = df[~fuga].reset_index(drop=True)
    carga.eliminados_por_fuga = int(fuga.sum())

    # etiquetas normalizadas
    norm = df["label"].map(lambda et: E.normaliza(et, "offendes"))
    df[E.COL_PELIGROSO] = norm.map(lambda d: d[E.COL_PELIGROSO])
    df[E.COL_RIESGO_ORD] = norm.map(lambda d: d[E.COL_RIESGO_ORD])
    df[E.COL_EXPLETIVO] = norm.map(lambda d: d[E.COL_EXPLETIVO])

    df = df.rename(columns={"comment_id": E.COL_ID, "comment": E.COL_TEXTO,
                            "label": E.COL_ORIGINAL})
    df[E.COL_ID] = "offendes-" + df[E.COL_ID].astype(str)
    df[E.COL_CORPUS] = "offendes"

    # validación extraída del entrenamiento, estratificada por etiqueta
    rng = np.random.default_rng(SEMILLA_PARTICION)
    for et, grupo in df[df[E.COL_SPLIT] == "train"].groupby(E.COL_ORIGINAL):
        n_val = int(round(len(grupo) * FRACCION_VALIDACION))
        idx = rng.choice(grupo.index.to_numpy(), size=n_val, replace=False)
        df.loc[idx, E.COL_SPLIT] = "validation"

    assert df[E.COL_ID].is_unique, "identificadores duplicados"
    return df


def une_senales(df, ruta_senales):
    """Añade las columnas de señales calculadas por sensores.py."""
    s = pd.read_csv(ruta_senales, dtype={E.COL_ID: str})
    faltan = set(E.SENALES_FUSION + E.SENALES_AUX) - set(s.columns)
    if faltan:
        raise ValueError(f"faltan señales en {ruta_senales}: {faltan}")
    out = df.merge(s, on=E.COL_ID, how="left", validate="one_to_one")
    sin = out[E.SENALES_FUSION + E.SENALES_AUX].isna().any(axis=1).sum()
    if sin:
        raise ValueError(f"{sin} comentarios sin señales: vuelve a ejecutar sensores.py")
    if "phishing" not in out.columns:
        out["phishing"] = 0.0
    return out


def tabla_descriptiva(df):
    """Tabla del corpus para el artículo: recuento por partición y etiqueta."""
    t = pd.crosstab(df[E.COL_ORIGINAL], df[E.COL_SPLIT], margins=True, margins_name="total")
    orden = [c for c in ["train", "validation", "dev_oficial", "test", "total"] if c in t.columns]
    return t[orden].reindex(["NO", "NOE", "OFG", "OFP", "total"])


if __name__ == "__main__":
    df = carga()
    print(f"{len(df)} comentarios ({carga.eliminados_por_fuga} eliminados por fuga)\n")
    print(tabla_descriptiva(df).to_string())
    print("\npor plataforma y partición:")
    print(pd.crosstab(df["media"], df[E.COL_SPLIT]).to_string())
    print("\nlongitud del texto en caracteres:",
          df[E.COL_TEXTO].str.len().describe()[["mean", "50%", "max"]].round(1).to_dict())
