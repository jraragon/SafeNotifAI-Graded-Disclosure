"""Agregadores comparados.

Todos reciben las mismas señales y devuelven un riesgo en [0, 100]; lo único que
cambia es cómo las combinan.

    ag = Agregador(); ag.ajusta(df_train); r = ag.riesgo(X)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import esquema as E
import fis


class Base:
    nombre = "base"
    ajustable = False        # ¿necesita etiquetas?

    def __init__(self, senales=None):
        self.senales = list(senales) if senales is not None else list(E.SENALES)

    def ajusta(self, df):
        return self

    def riesgo(self, X):
        raise NotImplementedError

    def riesgo_df(self, df):
        """Riesgo desde el DataFrame; cada agregador toma sus columnas."""
        return self.riesgo(df[self.senales].to_numpy(dtype=float))


class MaximoActual(Base):
    """La regla desplegada en SafeNotifAI (TFM, Figura 13):

        score_base = max(score_hate, score_anger, score_phishing_local)
        if label_sentiment == "NEG" and score_neg > 60:
            score_base = max(score_base, int(score_neg * 0.8))

    con alerta a partir de 50. La condición sobre la etiqueta equivale a
    P(NEG) > 0,6. El phishing queda fuera, como en el resto de agregadores.
    """
    nombre = "max (sistema actual)"

    UMBRAL_NEG = 0.60
    FACTOR_NEG = 0.80

    def __init__(self, senales=None):
        # usa las señales del sistema, no las del experimento
        super().__init__(["odio", "ira", "sentimiento"])

    def riesgo(self, X):
        X = np.asarray(X, dtype=float)
        odio, ira, neg = X[:, 0], X[:, 1], X[:, 2]
        base = np.maximum(odio, ira) * 100.0
        ajustado = np.floor(neg * 100.0 * self.FACTOR_NEG)   # el int() del original
        base = np.where(neg > self.UMBRAL_NEG, np.maximum(base, ajustado), base)
        return base


class MediaPonderada(Base):
    """Media ponderada, con pesos por correlación de cada señal con la etiqueta."""
    nombre = "media ponderada"
    ajustable = True

    def __init__(self, senales=None):
        super().__init__(senales)
        self.pesos = None

    def ajusta(self, df):
        y = df[E.COL_PELIGROSO].to_numpy(dtype=float)
        cors = []
        for s in self.senales:
            x = df[s].to_numpy(dtype=float)
            c = np.corrcoef(x, y)[0, 1] if x.std() > 0 else 0.0
            cors.append(max(float(c), 0.0))
        cors = np.array(cors)
        self.pesos = (cors / cors.sum() if cors.sum() > 0
                      else np.ones(len(self.senales)) / len(self.senales))
        return self

    def riesgo(self, X):
        return np.asarray(X, dtype=float) @ self.pesos * 100.0


class OWA(Base):
    """OWA: los pesos se aplican a las señales ordenadas de mayor a menor.

    Con orness 0,6 queda ligeramente compensatorio, que es lo que interesa aquí:
    varias señales medias pesan más que una alta aislada.
    """
    nombre = "OWA"

    def __init__(self, orness=0.6, senales=None):
        super().__init__(senales)
        self.orness = orness
        n = len(self.senales)
        # pesos exponenciales calibrados al orness pedido
        alfa = (1.0 - orness) / max(orness, 1e-6)
        w = np.array([(i + 1) ** alfa - i ** alfa for i in range(n)], dtype=float)
        self.pesos = w / w.sum()

    def riesgo(self, X):
        Xo = np.sort(np.asarray(X, dtype=float), axis=1)[:, ::-1]
        return Xo @ self.pesos * 100.0


class TopsisDifuso(Base):
    """TOPSIS: cercanía relativa al perfil ideal frente al anti-ideal."""
    nombre = "TOPSIS difuso"
    ajustable = True

    def __init__(self, senales=None):
        super().__init__(senales)
        self.pesos = None

    def ajusta(self, df):
        mp = MediaPonderada(self.senales).ajusta(df)
        self.pesos = mp.pesos
        return self

    def riesgo(self, X):
        X = np.asarray(X, dtype=float)
        w = self.pesos if self.pesos is not None else np.ones(X.shape[1]) / X.shape[1]
        Xp = X * w
        ideal = w * 1.0
        anti = w * 0.0
        d_mas = np.sqrt(((Xp - ideal) ** 2).sum(axis=1))
        d_menos = np.sqrt(((Xp - anti) ** 2).sum(axis=1))
        den = d_mas + d_menos
        return np.where(den > 0, d_menos / np.maximum(den, 1e-12), 0.0) * 100.0


class Supervisado(Base):
    """Regresión logística sobre las señales, con clases equilibradas."""
    nombre = "supervisado (logistica)"
    ajustable = True

    def __init__(self, senales=None):
        super().__init__(senales)
        self.esc = StandardScaler()
        self.mod = LogisticRegression(max_iter=1000, class_weight="balanced")

    def ajusta(self, df):
        X = df[self.senales].to_numpy(dtype=float)
        y = df[E.COL_PELIGROSO].to_numpy(dtype=int)
        self.mod.fit(self.esc.fit_transform(X), y)
        return self

    def riesgo(self, X):
        return self.mod.predict_proba(self.esc.transform(np.asarray(X, dtype=float)))[:, 1] * 100.0


class MotorFIS(Base):
    """El motor propuesto. Las pertenencias salen de los cuantiles de las señales."""
    nombre = "FIS (propuesta)"
    ajustable = False

    def __init__(self, reglas=None, solape=0.15, probs=None, metodo="altura", senales=None):
        super().__init__(senales)
        self.motor = fis.MotorDifuso(reglas=reglas, solape=solape, probs=probs,
                                     senales=self.senales)
        self.metodo = metodo

    def ajusta(self, df):
        self.motor.ajusta_pertenencias(df)
        return self

    def riesgo(self, X):
        return self.motor.riesgo(X, metodo=self.metodo)

    def traza(self, x):
        return self.motor.traza(x)


def todos(senales=None):
    """Los seis agregadores, sobre las señales de fusión."""
    senales = senales if senales is not None else E.SENALES_FUSION
    return [MaximoActual(senales=senales), MediaPonderada(senales), OWA(senales=senales),
            TopsisDifuso(senales), Supervisado(senales), MotorFIS(senales=senales)]
