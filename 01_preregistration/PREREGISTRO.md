# Prerregistro del experimento decisivo (fase 4, plan A)

Escrito el 22-09-2026, antes de rediseñar el reglamento y antes de ejecutar nada
del plan A. No se modifica después de ver resultados de prueba.

## Qué ha pasado hasta aquí (declarado)

Durante el diagnóstico del reglamento provisional se miraron resultados sobre el
conjunto de prueba varias veces. Desde este punto, todas las decisiones de diseño
(pertenencias, reglas, restricciones, selección del punto de operación) se toman
solo con entrenamiento y validación. La prueba se usa una única vez, al final.

## Comparación primaria

Todos los agregadores producen cuatro niveles de divulgación. Para cada uno, el
punto de operación se elige **en validación** con las mismas restricciones:

- exposición por contenido (pesos 0, 0, 0.25, 1) <= 0.10
- sobreexposición benigna (fracción de NOE en nivel >= 2) <= 0.30

y, entre los puntos que las cumplen, el de mayor protección (exhaustividad de
OFP+OFG en nivel >= 2).

Agregadores comparados con ese mismo procedimiento: regla actual (máximo),
media ponderada y regresión logística. El motor difuso se ajusta con NSGA-II en
validación, con esas dos restricciones dentro del problema, y se ejecuta con 5
semillas.

**Métrica primaria:** protección en prueba en el punto elegido.

## Regla de decisión

- Seguimos con el enfoque de ASOC (el motor difuso como aportación) si el motor
  cumple las dos cosas en prueba respecto a la regresión logística:
  1. protección no inferior: media de las 5 semillas >= protección de la
     logística - 0.02;
  2. sobreexposición benigna menor o igual que la de la logística.
- También si su protección es mayor que la de la logística sin que empeore la
  sobreexposición benigna en más de 0.02.
- En cualquier otro caso, plan B: la aportación es la divulgación graduada,
  independiente del agregador, y el motor queda como implementación interpretable.

## Cambios de diseño decididos a priori (sin mirar la prueba)

- Pertenencias sobre la escala de probabilidad, no por cuantiles, porque la
  emoción y el sentimiento son casi binarias y los cuantiles altos degeneran en
  un punto. Por defecto: bajo (0, 0, 0.2, 0.4), medio (0.2, 0.4, 0.6, 0.8),
  alto (0.6, 0.8, 1, 1). Registro malsonante: un término (0.5) es "medio" y dos
  (0.75) son "alto".
- Las cuatro señales de fusión no cambian (odio, emoción, sentimiento, registro
  malsonante), aunque en entrenamiento la ira separe algo mejor que la emoción
  agregada. Cambiarlas ahora sería elegir con los datos.
- Reglas rediseñadas desde el marco teórico y comprobadas solo con cobertura y
  diagnóstico en entrenamiento.
