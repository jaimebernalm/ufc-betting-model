# Candidata v2: integración y evaluación en sombra

La candidata estudiada en las ablaciones se ha integrado como una versión independiente. Los modelos son copias exactas de los ya evaluados. Kelly, sharpening 1,25, cutoff 2025-11-30, diez semillas, filtros y cuenta C mantienen su configuración. La integración no añade un nuevo ajuste ni una nueva optimización retrospectiva.

## Constructor que utiliza la versión

- **Tasas:** suma los totales UFC congelados y los nuevos combates UFC completados. Excluye siempre el día objetivo y cualquier fecha posterior.
- **Contadores:** utiliza el mismo acumulador cronológico, con los historiales externos previamente auditados para el conjunto de 796 peleadores. Esos registros externos no contribuyen a las tasas.
- **Perfiles:** conserva los datos estáticos originales; completa solamente los atributos que faltaban y que se verificaron en las ablaciones. No sustituye alturas, pesos o alcances existentes por otra fotografía actual.
- **Identidades nuevas:** si no están en los perfiles congelados, registra el problema y omite la predicción hasta revisar una nueva versión. No amplía silenciosamente la población ni el historial externo. El paquete contiene 4.615 perfiles; los homónimos sin resolver siguen bloqueados.
- **Evidencia futura:** archiva por hash las páginas utilizadas. Los registros de decisiones identifican el paquete, el historial y las respuestas oficiales observadas. Si un nuevo combate completado carece de identidad o totales verificables, no inventa estadísticas.

El entrenamiento histórico y la incorporación de combates futuros utilizan el mismo acumulador. También se ha corregido el parser general: una victoria por parada médica se identifica en el detalle del combate, aunque el resumen del perfil la agrupe como KO/TKO.

## Validación antes de activar

La reconstrucción reproduce exactamente las **7.425 filas** de características utilizadas por la candidata. La comprobación incluye tasas, contadores, perfiles, diferencias y rankings; las variables de habilidad se conservan. Se verificaron además **84 combates** mediante el constructor de filas de inferencia.

Las probabilidades de ambas variantes del ensemble en los **408 combates** de evaluación coinciden con las archivadas: **error absoluto máximo 0,0**. Por ello no ha sido necesario reentrenar ni repetir la búsqueda de estrategias; los resultados de las comparaciones anteriores siguen siendo los de esta candidata.

Se contrastó la corrección de paradas médicas con las páginas oficiales archivadas de Jean Silva, Curtis Blaydes y Arnold Allen. La incorporación de nuevos combates se comprueba retirando registros UFC del paquete a partir de una fecha anterior y reconstruyéndolos mediante esas páginas. El detalle de casos y discrepancias está en `artifacts/shadow_candidate_2026_09_14_v2/validation.json`.

La batería automatizada pasa **103 tests**. Incluye exclusión de resultados futuros y del mismo día, separación de tasas e historial externo, clasificación de paradas médicas, respuestas incompletas, identidades desconocidas, archivo de fuentes y bloqueo cuando cambian los hashes del paquete o del código.

## Qué resultados respalda esta activación

En los 311 combates elegibles del diagnóstico anterior:

| Medida | Primera corrección | Candidata v2 |
|---|---:|---:|
| Log loss A/B | 0,666933 | 0,646488 |
| Brier A/B | 0,230727 | 0,224110 |
| Log loss C | 0,656720 | 0,639423 |
| Brier C | 0,225307 | 0,219901 |

Esto justifica observar la candidata, pero no demuestra rentabilidad futura. Las simulaciones económicas anteriores no mejoraban en todos los periodos; el replay operativo de Kalshi seguía perdiendo dinero. Esta integración conserva esas conclusiones. Los metadatos estáticos tampoco tienen una fecha histórica de publicación demostrada.

## Registro y funcionamiento

Paquete: `artifacts/shadow_candidate_2026_09_14_v2/`.

Capturas, fuentes, saldo simulado y estado: `data/processed/shadow_candidate_2026_09_14_v2/`.

Referencia previa conservada: `artifacts/corrected_2026_09_14/`. Copia adicional del código, configuración, LaunchAgent y estado anterior: `artifacts/baselines/2026_09_14_before_candidate_v2/`.

La tarea local `com.ufcbet.watchdog` comprueba cada 60 segundos los disparadores ya existentes. Funciona con saldo de papel independiente, sin órdenes ni avisos de apuestas. El Mac debe estar despierto y conectado. Una ejecución correcta sin evento significa que el vigilante funciona; no equivale a una captura de mercado ni a una apuesta resuelta.

El protocolo conserva los saldos de referencia y las proporciones entre las cuentas, registrados únicamente en los artefactos locales. La evaluación formal exige **ambos mínimos: 12 jornadas completadas y 150 combates elegibles resueltos**. Se observarán ROI con importe fijo, calibración de las apuestas seleccionadas, crecimiento y drawdown del saldo simulado, liquidez y oportunidades omitidas. Cambiar parámetros o fuentes requiere una nueva versión; el resultado de cada combate no es motivo para reajustar la estrategia.

Reproducción: `scripts/research/package_shadow_candidate_v2.py`, `validate_shadow_candidate_v2.py` y `freeze_shadow_candidate_v2.py`. Los scripts de empaquetado y congelación se niegan a sobrescribir una versión ya congelada; las verificaciones posteriores deben mantener intactos sus archivos.


Nota de publicación: las referencias a `artifacts/`, datos, registros operativos y documentos privados corresponden a archivos locales excluidos de Git. Esta copia pública omite saldos personales y rutas específicas del equipo; los originales se conservan localmente.
