"""Léxico de registro malsonante a partir de HurtLex (español, v1.2).

    git clone --depth 1 https://github.com/valeriobasile/hurtlex.git /tmp/hurtlex
    python3 construir_lexico.py /tmp/hurtlex/lexica/ES/1.2/hurtlex_ES.tsv

HurtLex: Bassignana, Basile y Patti (2018), CC BY-NC-SA 4.0.

El filtro se fija antes de mirar las etiquetas de OffendES, para no ajustar el
sensor al corpus de evaluación: nivel conservative y las cuatro categorías de
registro obsceno y despectivo (asm, asf, pr, cds). Peso 0,5 para todos, porque
HurtLex no da intensidad por término.
"""
import os
import sys

import pandas as pd

from sensores import _normaliza

CATEGORIAS = {"asm", "asf", "pr", "cds"}
PESO = 0.5

ruta = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hurtlex/lexica/ES/1.2/hurtlex_ES.tsv"
h = pd.read_csv(ruta, sep="\t")
sel = h[(h["level"] == "conservative") & (h["category"].isin(CATEGORIAS))]
terminos = sorted({_normaliza(t) for t in sel["lemma"].dropna()
                   if 2 < len(str(t)) < 30})
os.makedirs("datos", exist_ok=True)
pd.DataFrame({"termino": terminos, "peso": PESO}).to_csv("datos/lexico_malsonante.csv", index=False)
print(f"{len(terminos)} términos en datos/lexico_malsonante.csv")
