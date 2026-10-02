"""Esquema común de los corpus y normalización de etiquetas."""

# Identificación y texto
COL_ID = "id"
COL_TEXTO = "texto"
COL_CORPUS = "corpus"
COL_SPLIT = "split"          # train / validation / test
COL_ORIGINAL = "etiqueta_original"

# Etiquetas normalizadas
COL_PELIGROSO = "peligroso"            # binaria: la usa la exhaustividad de protección
COL_RIESGO_ORD = "riesgo_ord"          # ordinal 0-2: se compara con los niveles de divulgación
COL_EXPLETIVO = "expletivo_benigno"    # marca el subconjunto de la prueba de sobreexposición

# Señales de entrada (las cinco, en [0, 1])
SENALES = ["odio", "emocion", "sentimiento", "phishing", "malsonante"]

# Señales del experimento principal. El phishing queda fuera: ningún corpus
# anota URLs, así que cada acierto del sensor cuenta como falso positivo.
SENALES_FUSION = ["odio", "emocion", "sentimiento", "malsonante"]

# Solo la usa la línea base: la regla actual toma el máximo de odio, ira y
# phishing, no de la emoción agregada.
SENALES_AUX = ["ira"]

# Niveles de divulgación
NIVEL_NADA = 0
NIVEL_AVISO_MENOR = 1
NIVEL_INDICADOR = 2
NIVEL_ALERTA = 3

# A partir de aquí el contenido llega al tutor.
NIVEL_EXPONE = NIVEL_INDICADOR


# OffendES. OFP: ofensivo hacia una persona; OFG: hacia un colectivo;
# NOE: no ofensivo con lenguaje malsonante; NO: no ofensivo.
OFFENDES = {
    "NO":  {COL_PELIGROSO: 0, COL_RIESGO_ORD: 0, COL_EXPLETIVO: 0},
    "NOE": {COL_PELIGROSO: 0, COL_RIESGO_ORD: 0, COL_EXPLETIVO: 1},
    "OFG": {COL_PELIGROSO: 1, COL_RIESGO_ORD: 1, COL_EXPLETIVO: 0},
    "OFP": {COL_PELIGROSO: 1, COL_RIESGO_ORD: 2, COL_EXPLETIVO: 0},
}

# Superset de odio. Etiqueta binaria: no distingue el objetivo de la ofensa.
SUPERSET = {
    0: {COL_PELIGROSO: 0, COL_RIESGO_ORD: None, COL_EXPLETIVO: 0},
    1: {COL_PELIGROSO: 1, COL_RIESGO_ORD: None, COL_EXPLETIVO: 0},
}

# HateCheck. Solo diagnóstico: no entra en el ajuste.
HATECHECK = {
    "non-hateful": {COL_PELIGROSO: 0, COL_RIESGO_ORD: None, COL_EXPLETIVO: 0},
    "hateful":     {COL_PELIGROSO: 1, COL_RIESGO_ORD: None, COL_EXPLETIVO: 0},
}


def normaliza(etiqueta, corpus):
    """Devuelve el dict de etiquetas normalizadas para una etiqueta original."""
    tabla = {"offendes": OFFENDES, "superset": SUPERSET, "hatecheck": HATECHECK}[corpus]
    if etiqueta not in tabla:
        raise KeyError(f"etiqueta {etiqueta!r} no prevista en {corpus}")
    return dict(tabla[etiqueta])


COLUMNAS = [COL_ID, COL_TEXTO, COL_CORPUS, COL_SPLIT, COL_ORIGINAL,
            COL_PELIGROSO, COL_RIESGO_ORD, COL_EXPLETIVO] + SENALES + SENALES_AUX
