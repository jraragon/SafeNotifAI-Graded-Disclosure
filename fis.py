"""Motor de inferencia difusa tipo Mamdani.

Escrito a mano y no con una librería porque hacen falta la traza de reglas
activadas, que alimenta la explicación, y la fuente de cada regla.
"""
from dataclasses import dataclass, field

import numpy as np

import esquema as E


# --- Funciones de pertenencia -------------------------------------------------

@dataclass
class Trapecio:
    """Trapecio (a, b, c, d): sube en [a,b], vale 1 en [b,c], baja en [c,d]."""
    a: float
    b: float
    c: float
    d: float

    def grado(self, x):
        x = np.asarray(x, dtype=float)
        g = np.zeros_like(x)
        # rampa de subida
        if self.b > self.a:
            m = (x > self.a) & (x < self.b)
            g[m] = (x[m] - self.a) / (self.b - self.a)
        # meseta
        g[(x >= self.b) & (x <= self.c)] = 1.0
        # rampa de bajada
        if self.d > self.c:
            m = (x > self.c) & (x < self.d)
            g[m] = (self.d - x[m]) / (self.d - self.c)
        # bordes abiertos
        if self.a == self.b:
            g[x <= self.a] = 1.0 if self.b >= self.a else 0.0
        if self.c == self.d:
            g[x >= self.d] = 1.0
        return g


def desde_cuantiles(valores, etiquetas=("bajo", "medio", "alto"),
                    probs=(0.50, 0.85), solape=0.15):
    """Pertenencias de una señal a partir de sus cuantiles.

    Los niveles de `probs` no son terciles: en una señal asimétrica dejarían un
    tercio de los mensajes en "alto".
    """
    v = np.asarray(valores, dtype=float)
    if len(probs) != len(etiquetas) - 1:
        raise ValueError("hacen falta len(etiquetas)-1 niveles de probabilidad")

    lo, hi = float(np.min(v)), float(np.max(v))
    cortes = [lo] + [float(np.quantile(v, p)) for p in probs] + [hi]

    # con señales concentradas dos cortes pueden coincidir
    eps = max((hi - lo) * 1e-3, 1e-6)
    for i in range(1, len(cortes)):
        if cortes[i] <= cortes[i - 1]:
            cortes[i] = cortes[i - 1] + eps

    ancho = max(float(np.mean(np.diff(cortes))), eps)
    s = solape * ancho

    terminos = {}
    for i, et in enumerate(etiquetas):
        izq, der = cortes[i], cortes[i + 1]
        a = izq - s if i > 0 else izq
        b = izq + s if i > 0 else izq
        c = der - s if i < len(etiquetas) - 1 else der
        d = der + s if i < len(etiquetas) - 1 else der
        # el trapecio tiene que cumplir a <= b <= c <= d
        b = max(b, a)
        c = max(c, b)
        d = max(d, c)
        terminos[et] = Trapecio(float(a), float(b), float(c), float(d))
    return terminos


# --- Reglas -------------------------------------------------------------------

# Ninguna regla puede quedar sin fuente.
FUENTE_OFFENDES = "definiciones de anotacion de OffendES"
FUENTE_CUANTILES = "escala o distribucion de la senal"
FUENTE_MARCO = "marco sobre descortesia e insulto ritual"

FUENTES = {FUENTE_OFFENDES, FUENTE_CUANTILES, FUENTE_MARCO}


@dataclass
class Regla:
    id: str
    # {senal: [terminos]}, OR dentro de la señal y AND entre señales
    antecedente: dict
    consecuente: str          # término de la variable de salida
    fuente: str
    peso: float = 1.0
    nota: str = ""

    def __post_init__(self):
        if self.fuente not in FUENTES:
            raise ValueError(f"regla {self.id} sin fuente valida: {self.fuente!r}")


# Variable de salida: riesgo en [0, 100], cuatro términos que corresponden a los
# cuatro niveles de divulgación.
SALIDA = {
    "nada":       Trapecio(0, 0, 10, 25),
    "aviso":      Trapecio(15, 25, 40, 50),
    "indicador":  Trapecio(40, 50, 65, 78),
    "alerta":     Trapecio(70, 82, 100, 100),
}
SALIDA_A_NIVEL = {
    "nada": E.NIVEL_NADA,
    "aviso": E.NIVEL_AVISO_MENOR,
    "indicador": E.NIVEL_INDICADOR,
    "alerta": E.NIVEL_ALERTA,
}


def reglamento_inicial(con_phishing=True):
    """Reglamento derivado de tres fuentes, no elicitado de un panel."""
    R = []

    # -- Nada: ausencia de señal en todo --------------------------------------
    R.append(Regla("R01", {"odio": ["bajo"], "sentimiento": ["bajo"], "emocion": ["bajo"]},
                   "nada", FUENTE_OFFENDES,
                   nota="categoria NO: ni ofensivo ni malsonante"))

    # -- descortesía fingida: taco sin concordancia afectiva ----------------
    R.append(Regla("R02", {"odio": ["alto", "medio"], "malsonante": ["alto"],
                           "emocion": ["bajo"], "sentimiento": ["bajo", "medio"]},
                   "aviso", FUENTE_MARCO,
                   nota="malsonante sin intencion de ofender: se avisa al menor, no al tutor"))
    R.append(Regla("R03", {"odio": ["alto"], "malsonante": ["alto"], "emocion": ["medio"]},
                   "aviso", FUENTE_MARCO,
                   nota="idem con emocion intermedia"))

    # -- evidencia parcial concordante, que el máximo descarta ----------------
    R.append(Regla("R04", {"odio": ["medio"], "emocion": ["medio", "alto"],
                           "sentimiento": ["medio", "alto"], "malsonante": ["bajo", "medio"]},
                   "indicador", FUENTE_OFFENDES,
                   nota="acumulacion de indicios debiles concordantes"))
    R.append(Regla("R05", {"odio": ["medio", "alto"], "sentimiento": ["alto"],
                           "malsonante": ["bajo", "medio"]},
                   "indicador", FUENTE_OFFENDES,
                   nota="ofensa con sentimiento marcadamente negativo"))

    # -- Ofensa dirigida ------------------------------------------------------
    R.append(Regla("R06", {"odio": ["alto"], "emocion": ["alto"], "sentimiento": ["alto"],
                           "malsonante": ["bajo", "medio"]},
                   "alerta", FUENTE_OFFENDES,
                   nota="categoria OFP: concordancia plena sin atenuante de registro malsonante"))
    R.append(Regla("R07", {"odio": ["alto"], "emocion": ["medio", "alto"],
                           "sentimiento": ["alto"], "malsonante": ["medio"]},
                   "alerta", FUENTE_OFFENDES,
                   nota="ofensa dirigida con registro malsonante intermedio"))

    # -- Phishing: riesgo independiente --------------------------------------
    R.append(Regla("R08", {"phishing": ["alto"]},
                   "alerta", FUENTE_OFFENDES,
                   nota="enlace sospechoso: riesgo distinto del de hostilidad"))
    R.append(Regla("R09", {"phishing": ["medio"], "odio": ["bajo"]},
                   "indicador", FUENTE_OFFENDES,
                   nota="enlace dudoso en mensaje por lo demas inocuo"))

    # -- Casos intermedios ----------------------------------------------------
    R.append(Regla("R10", {"odio": ["medio"], "emocion": ["bajo"], "sentimiento": ["bajo"]},
                   "aviso", FUENTE_CUANTILES,
                   nota="senal unica en franja media sin concordancia"))
    R.append(Regla("R11", {"odio": ["bajo"], "sentimiento": ["alto"], "emocion": ["alto"]},
                   "aviso", FUENTE_MARCO,
                   nota="malestar sin hostilidad: puede ser la victima, no el agresor"))
    R.append(Regla("R12", {"odio": ["bajo"], "malsonante": ["alto"], "emocion": ["bajo"]},
                   "nada", FUENTE_MARCO,
                   nota="registro malsonante solo no es evidencia de nada"))

    if not con_phishing:
        R = [r for r in R if "phishing" not in r.antecedente]
    return R


# --- Inferencia ---------------------------------------------------------------

# Cortes por señal: "odio alto" y "phishing alto" no tienen la misma frecuencia
# base, así que el phishing corta en el percentil 97.
PROBS_POR_SENAL = {
    "odio":        (0.50, 0.85),
    "emocion":     (0.50, 0.85),
    "sentimiento": (0.50, 0.85),
    "phishing":    (0.90, 0.97),
    "malsonante":       (0.60, 0.88),
}


class MotorDifuso:
    def __init__(self, reglas=None, salida=None, solape=0.15, probs=None, senales=None):
        self.senales = list(senales) if senales is not None else list(E.SENALES)
        self.reglas = reglas if reglas is not None else reglamento_inicial(
            con_phishing="phishing" in self.senales)
        self.salida = salida if salida is not None else dict(SALIDA)
        self.solape = solape
        self.probs = dict(probs) if probs is not None else dict(PROBS_POR_SENAL)
        self.terminos = None          # {senal: {termino: Trapecio}}
        self._malla = np.linspace(0, 100, 201)

    def ajusta_pertenencias(self, df):
        """Construye las pertenencias de cada señal con los cuantiles del corpus."""
        self.terminos = {
            s: desde_cuantiles(df[s].to_numpy(), probs=self.probs[s], solape=self.solape)
            for s in self.senales
        }
        return self

    def diagnostico(self, X):
        """Integridad del reglamento: reglas muertas, dominantes y cobertura."""
        act = self.activaciones(X)
        n = act.shape[0]
        muertas = [r.id for k, r in enumerate(self.reglas) if act[:, k].max() <= 0]
        dominantes = [r.id for k, r in enumerate(self.reglas)
                      if (act[:, k] > 0).mean() > 0.9]
        sin_cobertura = int((act.sum(axis=1) <= 0).sum())
        return {
            "reglas": len(self.reglas),
            "reglas_muertas": muertas,
            "reglas_dominantes": dominantes,
            "sin_cobertura": sin_cobertura,
            "sin_cobertura_pct": round(100.0 * sin_cobertura / n, 2),
            "activaciones_por_mensaje": round(float((act > 0).sum(axis=1).mean()), 2),
        }

    def _grados(self, X):
        """{senal: {termino: array(n)}}"""
        return {
            s: {t: trap.grado(X[:, j]) for t, trap in self.terminos[s].items()}
            for j, s in enumerate(self.senales)
        }

    def activaciones(self, X):
        """Matriz (n, n_reglas) con el grado de activación de cada regla."""
        X = np.asarray(X, dtype=float)
        g = self._grados(X)
        act = np.zeros((X.shape[0], len(self.reglas)))
        for k, r in enumerate(self.reglas):
            fuerza = np.ones(X.shape[0])
            for senal, terms in r.antecedente.items():
                # OR dentro de la señal
                parcial = np.max(np.stack([g[senal][t] for t in terms]), axis=0)
                # AND entre señales
                fuerza = np.minimum(fuerza, parcial)
            act[:, k] = fuerza * r.peso
        return act

    def _centroides_salida(self):
        """Centroide de cada término de salida, sobre la malla."""
        cen = {}
        for t, trap in self.salida.items():
            g = trap.grado(self._malla)
            cen[t] = float((self._malla * g).sum() / g.sum())
        return cen

    def riesgo(self, X, metodo="altura"):
        """Riesgo desfuzzificado en [0, 100].

        Por defecto, media de los centroides ponderada por activación, que es
        monótona en el grado. El centroide de área clásico no lo es: con una sola
        regla activa devuelve el centro de su consecuente se active como se active.
        """
        act = self.activaciones(X)
        n = act.shape[0]

        if metodo == "altura":
            cen = self._centroides_salida()
            pesos_c = np.array([cen[r.consecuente] for r in self.reglas])
            total = act.sum(axis=1)
            out = np.zeros(n)
            m = total > 0
            out[m] = (act[m] @ pesos_c) / total[m]
            return out

        if metodo == "centroide":
            base = {t: trap.grado(self._malla) for t, trap in self.salida.items()}
            out = np.zeros(n)
            for i in range(n):
                agregado = np.zeros_like(self._malla)
                for k, r in enumerate(self.reglas):
                    if act[i, k] <= 0:
                        continue
                    agregado = np.maximum(agregado, np.minimum(base[r.consecuente], act[i, k]))
                s = agregado.sum()
                out[i] = float((self._malla * agregado).sum() / s) if s > 0 else 0.0
            return out

        raise ValueError(f"metodo desconocido: {metodo}")

    def traza(self, x):
        """Reglas activadas por un mensaje, con su grado. Alimenta la explicación."""
        act = self.activaciones(np.asarray(x, dtype=float).reshape(1, -1))[0]
        filas = [
            {"regla": r.id, "grado": round(float(act[k]), 4),
             "consecuente": r.consecuente, "fuente": r.fuente, "nota": r.nota}
            for k, r in enumerate(self.reglas) if act[k] > 0
        ]
        return sorted(filas, key=lambda f: -f["grado"])

    def resumen_fuentes(self):
        """Recuento de reglas por fuente."""
        from collections import Counter
        return Counter(r.fuente for r in self.reglas)


# --- Pertenencias sobre la escala de probabilidad y reglamento v2 ------------
#
# Los cuantiles no sirven aquí: emoción y sentimiento son casi binarias y el
# percentil 85 cae en 1, con lo que "alto" degenera en un punto. La escala del
# clasificador ya tiene significado propio, así que los cortes van sobre ella.

ESCALA_DEFECTO = {
    "odio":        (0.30, 0.70, 0.10),
    "emocion":     (0.30, 0.70, 0.10),
    "sentimiento": (0.30, 0.70, 0.10),
    # noisy-OR: un término da 0,5 y dos dan 0,75
    "malsonante":  (0.20, 0.62, 0.08),
}


def desde_escala(c1, c2, delta):
    """Tres términos en [0, 1] con cortes c1 < c2 y solape delta."""
    c1, c2, delta = float(c1), float(c2), float(delta)
    c2 = max(c2, c1 + 2 * delta + 1e-3)
    return {
        "bajo":  Trapecio(0.0, 0.0, max(c1 - delta, 0.0), c1 + delta),
        "medio": Trapecio(max(c1 - delta, 0.0), c1 + delta, c2 - delta, min(c2 + delta, 1.0)),
        "alto":  Trapecio(c2 - delta, min(c2 + delta, 1.0), 1.0, 1.0),
    }


class MotorEscala(MotorDifuso):
    """Motor con pertenencias en la escala de probabilidad. No usa los datos."""

    def __init__(self, reglas=None, escala=None, senales=None):
        senales = list(senales) if senales is not None else list(E.SENALES_FUSION)
        super().__init__(reglas=reglas if reglas is not None else reglamento_v2(),
                         senales=senales)
        self.escala = dict(ESCALA_DEFECTO) if escala is None else dict(escala)
        self.terminos = {s: desde_escala(*self.escala[s]) for s in self.senales}

    def ajusta_pertenencias(self, df=None):
        return self


def reglamento_v2():
    """Reglamento definitivo, con la fuente de cada regla.

    La hostilidad se juzga por la concordancia entre odio, emoción y sentimiento.
    El registro malsonante no suma evidencia: cambia la lectura. Sin afecto hostil
    concordante, el taco es marcador de insulto ritual y queda en aviso al menor.
    """
    R = []
    # -- sin evidencia --------------------------------------------------------
    R.append(Regla("V01", {"odio": ["bajo"], "emocion": ["bajo"], "sentimiento": ["bajo", "medio"]},
                   "nada", FUENTE_OFFENDES, nota="sin hostilidad ni afecto negativo"))
    R.append(Regla("V02", {"odio": ["bajo"], "emocion": ["bajo", "medio"], "sentimiento": ["alto"],
                           "malsonante": ["bajo"]},
                   "nada", FUENTE_OFFENDES, nota="critica o descontento sin hostilidad"))

    # -- descortesía fingida (insulto ritual) ---------------------------------
    R.append(Regla("V03", {"malsonante": ["medio", "alto"], "odio": ["bajo", "medio"],
                           "emocion": ["bajo", "medio"]},
                   "aviso", FUENTE_MARCO,
                   nota="taco sin afecto hostil concordante: insulto ritual, aviso al menor"))
    R.append(Regla("V04", {"malsonante": ["medio", "alto"], "odio": ["bajo"],
                           "emocion": ["alto"], "sentimiento": ["bajo", "medio"]},
                   "aviso", FUENTE_MARCO,
                   nota="taco con emocion intensa pero sin polaridad negativa: probable efusividad"))

    # -- afecto hostil sin marca de registro ----------------------------------
    R.append(Regla("V05", {"emocion": ["alto"], "sentimiento": ["alto"], "odio": ["bajo"],
                           "malsonante": ["bajo"]},
                   "indicador", FUENTE_OFFENDES,
                   nota="afecto hostil concordante sin taco ni odio: posible ataque personal"))
    R.append(Regla("V06", {"emocion": ["medio"], "sentimiento": ["alto"], "odio": ["medio"]},
                   "indicador", FUENTE_OFFENDES,
                   nota="indicios parciales concordantes"))

    # -- hostilidad concordante -----------------------------------------------
    R.append(Regla("V07", {"odio": ["medio", "alto"], "emocion": ["alto"], "sentimiento": ["alto"]},
                   "alerta", FUENTE_OFFENDES,
                   nota="concordancia plena: ofensa dirigida"))
    R.append(Regla("V08", {"odio": ["alto"], "emocion": ["medio", "alto"]},
                   "alerta", FUENTE_OFFENDES,
                   nota="odio alto con afecto negativo"))
    R.append(Regla("V09", {"emocion": ["alto"], "sentimiento": ["alto"], "odio": ["bajo"],
                           "malsonante": ["medio", "alto"]},
                   "indicador", FUENTE_MARCO,
                   nota="taco con afecto hostil: el registro ya no atenua, pero sin odio no llega a alerta"))
    R.append(Regla("V10", {"odio": ["alto"], "emocion": ["bajo"], "sentimiento": ["bajo", "medio"]},
                   "aviso", FUENTE_CUANTILES,
                   nota="odio alto aislado sin afecto: probable falso positivo del sensor"))
    return R
