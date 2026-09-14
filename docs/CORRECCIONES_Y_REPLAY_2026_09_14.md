# Correcciones, reentrenamiento y comparación con simulaciones anteriores

Fecha: 14-09-2026. No se optimizó la estrategia a la vista de estos resultados.

## Resultado principal

Se reconstruyeron las variables y se reentrenaron las diez semillas. Las ventanas que anteriormente eran positivas siguen siendo positivas, pero el sistema corregido no mejora todas las métricas ni elimina las pérdidas posteriores de Kalshi. Estas ventanas ya se han reutilizado: son diagnóstico, no validación independiente de una ventaja nueva.

ROI con importe fijo de efectivo por apuesta, incluidas las comisiones modeladas. A y B usan la misma probabilidad; sus diferencias Kelly no intervienen en esta tabla. C conserva el comportamiento anterior de ocultar las columnas de habilidad en inferencia.

| Ventana | Modelo | Informe anterior guardado | Modelos desplegados congelados | Reentrenamiento corregido |
|---|---|---:|---:|---:|
| Polymarket, 06-12-2025 a 16-05-2026 | A/B | +22,46% (156 apuestas) | +17,77% (157) | +11,74% (158) |
| Polymarket, misma ventana | C | +10,81% (152) | +19,93% (160) | +16,71% (160) |
| Kalshi, 24-01 a 16-05-2026 | A/B | +38,99% (42) | +23,08% (43) | +30,57% (42) |
| Kalshi, misma ventana | C | +36,39% (38) | +28,33% (43) | +31,86% (43) |
| Kalshi, 30-05 a 01-08-2026 | A/B | −30,29% (60) | −34,87% (54) | −22,77% (55) |
| Kalshi, misma ventana | C | −37,03% (58) | −32,79% (52) | −23,90% (55) |

Las tres columnas comparten exactamente la población elegible del informe guardado (175, 53 y 63 combates, respectivamente), precios y reglas. Cambia qué apuestas selecciona cada predicción. Se reproducen las cifras del informe original a partir de sus probabilidades guardadas. Ese informe y los archivos de modelos desplegados no son la misma referencia; mezclarlos habría producido una comparación engañosa. La referencia desplegada usa sus variables históricas congeladas, no pretende reconstruir cada dato que recibió el bot en cada ejecución real.

Al utilizar la elegibilidad corregida por identidades —la que utiliza la nueva versión— cambian ligeramente las dos primeras poblaciones: Polymarket 176 y Kalshi inicial 54 combates elegibles. En esa comparación emparejada, modelo congelado → corregido: Polymarket A/B +17,92% → +11,04%, C +20,07% → +15,98%; Kalshi inicial A/B +20,28% → +27,54%, C +28,33% → +28,87%. El tramo posterior no cambia. Ambas comparaciones están guardadas separadamente.

**Limitación de ejecución fundamental:** las 224 observaciones de Polymarket son precios de cierre, sin profundidad histórica. Las 57 observaciones del periodo inicial de Kalshi tampoco tienen asks verdaderos archivados: usan el fallback histórico de cierre/negociación. Las 72 observaciones posteriores de Kalshi sí tienen asks archivados. Por tanto, la ventana positiva más llamativa no demuestra que todas esas compras hubieran sido ejecutables.

Intervalos del 95% por remuestreo de jornadas del reentrenamiento, sobre la población exacta del informe: Polymarket A/B [−3,08%, +26,32%], C [−0,91%, +35,75%]; Kalshi inicial A/B [−15,06%, +87,09%], C [−14,48%, +88,66%]. Todos incluyen cero. En el tramo posterior de Kalshi los intervalos son A/B [−38,36%, −8,25%] y C [−39,55%, −8,32%]. El remuestreo no corrige la selección retrospectiva de periodos o estrategias.

## Congelación y cambios objetivos

La referencia anterior se copió antes de modificar el sistema: `artifacts/baselines/2026_09_14_before_fixes/`, 432 archivos y 728.447.272 bytes, verificados con SHA256. Incluye modelos, datos, configuraciones, versiones de paquetes, estado git y parche de los cambios locales existentes. HEAD original: `56ae9c9`. No se copiaron credenciales. Los modelos originales siguen en su ubicación; los nuevos están en una carpeta independiente.

Se corrigieron:

- Identidades: alias explícitos y verificables, separación de los dos Bruno Silva y rechazo de homónimos ambiguos; se evita decidir una identidad mediante una coincidencia difusa automática. Se conservaron los alias previos del usuario.
- Rankings: se usa la última fotografía completa anterior al combate; quien desaparece de ella queda sin ranking. Se reconocen nombres de listas P4P masculinas/femeninas y cambios de divisiones. La fotografía oficial observada hoy nunca se retrofecha.
- Habilidad: se elimina la segunda inversión del signo al aumentar los datos; se construyen índices y divisiones únicamente con historia anterior al mes. Las claves del posterior incorporan datos previos, división, receta, código y versiones relevantes.
- Variables: entrenamiento y producción comparten el cálculo de estado previo. Se excluyó DWCS del proveedor en vivo porque el entrenamiento reconstruido utiliza historial UFC. La página oficial de Mike Davis confirmaba esa diferencia.
- Resultados: dos ganadores invertidos se verificaron contra fichas oficiales (Mallory Martin–Virna Jandiroba y Mike Davis–Mason Jones). Además, se reconciliaron once revisiones a no-contest. Dos filas sin combate completado coincidente se excluyeron: Hall–Souza (09-05-2020), Borella–Maverick (27-06-2020). Estas categorías están separadas en el registro; una revisión posterior a NC no se presenta como un error original de captura.
- Contabilidad: el ledger depende de fills identificables, cantidades reales, compras/ventas YES/NO, comisiones y liquidaciones de ambos lados. Una recomendación sin fill no genera PnL. Las marcas provisionales no liberan efectivo; la liquidación oficial puede corregirlas. Los fills problemáticos se aíslan y bloquean recomendaciones reales.
- Ejecución: se comprueba el edge y EV neto en todos los niveles consumidos, se respeta la profundidad y efectivo global disponible, y se reservan recomendaciones pendientes sin inventar beneficios. Se conserva el reparto alternativo 1:2:6 para fills sin recomendación vinculable, como política explícita.
- Operación: dry-run y sombra tienen registros separados, sin alertas; los reintentos conservan evidencia. Se corrigió el parser de IDs del calendario con sufijos repetidos. El historial para sombra se actualiza en una ruta independiente y sólo con jornadas anteriores al día UTC actual.

La migración del ledger real se ancló al saldo confirmado por la API, sin posiciones abiertas, conservando las proporciones entre las cuentas. Los importes personales se mantienen únicamente en el registro local. La diferencia y los valores originales están en `ledger_migration.json`. Repetir sync no volvió a descontar fills antiguos. No se enviaron órdenes ni alertas.

## Reconstrucción y receta conservada

Fuente bruta fijada: exportación de UFCStats, repositorio https://github.com/Greco1899/scrape_ufc_stats.git, commit `cb4ecb64dd62324a7a51a378f6b5bbb0fc99bc65`. Incluye 788 eventos, 8.912 resultados y 41.906 filas de estadísticas por asalto. Se guardaron además 16 perfiles oficiales verificados para resolver perfiles ausentes/obsoletos.

Del sobre original de 7.445 filas quedan 7.443 verificadas. Se reconstruyeron tasas, contadores y estados previos; se procesaron 198 meses de habilidad, incluidos los meses iniciales sin posterior utilizable. El entrenamiento contiene 7.017 combates decisivos anteriores al 30-11-2025.

Receta: semillas 0–9; 2.000 árboles CatBoost, profundidad 6, learning rate 0,05, L2=3, semivida de recencia 4 años, sin early stopping. Se guardan 20 bundles: diez reales y sus diez versiones C que ocultan habilidad en inferencia. Se mantiene simetrización por orientación, sharpening T=1,25, edge bruto mínimo 3%, Kelly A=0,10/B=C=0,25, cap A=10%, B/C sin cap, límite conjunto del 5% de profundidad en banda de 3 céntimos y apuesta mínima 0,50 dólares. No se buscaron nuevos hiperparámetros ni filtros de underdogs.

Cambiar solamente las variables conservando modelos viejos empeora aún más la log loss (aproximadamente 0,78/0,80); está guardado como diagnóstico de cambio de distribución, no como versión candidata.

## Calidad probabilística y ejecución acotada

En los 408 combates posteriores al cutoff:

| Métrica; menor es mejor | Anterior congelado A/B | Corregido A/B | Anterior C | Corregido C |
|---|---:|---:|---:|---:|
| Log loss | 0,6351 | 0,6960 | 0,6292 | 0,6882 |
| Brier | 0,2127 | 0,2441 | 0,2111 | 0,2400 |

Corregir bugs no garantiza mejorar estas métricas. El resultado tampoco identifica qué arreglo individual produce cada cambio: se compara el paquete conjunto y un diagnóstico de cambio de variables. Para atribuciones causales por corrección harían falta ablaciones adicionales, que no se han utilizado para seleccionar estrategia.

Cobertura del replay: 353 observaciones de precio para 296 combates distintos, de 408 posibles. Quedan **112 combates sin precio archivado**; no se inventan sus oportunidades ni resultados económicos. Las oportunidades conocidas sin apuesta se conservan con su motivo.

Se ejecutaron 18 escenarios sobre los 72 combates Kalshi con asks: sistema congelado/corregido × profundidad nocional 100/1.000/10.000 dólares × desplazamiento adverso 0/1/3 céntimos. Un 1% de la profundidad está al mejor precio y el resto al nivel adverso. Se conserva el cap del 5%; el efectivo queda bloqueado hasta fin de jornada. El orden usa el cierre archivado del mercado como aproximación: no hay timestamps completos de inicio real de cada combate ni libros históricos completos.

Ejemplo de sensibilidad, profundidad 1.000 dólares y 1 céntimo adverso:

| Cuenta | Capital inicial de referencia | Final anterior | Final corregido | Drawdown anterior | Drawdown corregido |
|---|---:|---:|---:|---:|---:|
| A | 50,75 | 30,45 | 35,76 | 41,49% | 31,51% |
| B | 84,55 | 20,33 | 32,13 | 77,41% | 64,53% |
| C | 215,15 | 37,20 | 98,83 | 84,42% | 57,15% |

La nueva versión pierde menos en este escenario, pero sigue perdiendo. Son trayectorias hipotéticas sujetas a esos supuestos, no fills reconstruidos. El saldo inicial histórico de la comparación se mantuvo igual en ambos sistemas; no se sustituyó por el nuevo ancla del ledger.

También se compararon 30 capturas reales guardadas, compatibles con T=1,25 y anteriores al inicio programado: A/B anterior +11,12% en 23 apuestas frente a nuevo −22,36% en 24; C anterior −19,87% en 28 frente a nuevo +16,90% en 28. Es una muestra pequeña seleccionada por disponibilidad de capturas y horario programado, no evidencia independiente ni una reconstrucción garantizada anterior al inicio real.

## Validación y límites que permanecen

- 97 pruebas pasan; Ruff sin errores. Cubren identidades, futuro, ranking desaparecido, signo, precio adverso, profundidad, efectivo, fills parciales/ambos lados, idempotencia, reversión de liquidación, dry-run, calendario, proveedor UFC-only y aislamiento del wrapper de sombra.
- Paridad de 101 filas históricas entre constructor batch y producción: cero discrepancias. Paridad independiente del proveedor bruto frente al scraper oficial para Bruno FLW, Bruno MW y Mike Davis: todos los campos coinciden.
- Posteriors: ningún mes esperado sin correspondencia con la caché persistida; Rhat máximo 1,0354, ESS mínimo 34,79, sin meses con divergencias. El ESS mínimo es bajo para el peor parámetro: estos diagnósticos no significan mezcla excelente en todos los parámetros.
- No existen fechas históricas de publicación para todos los atributos estáticos del perfil ni para todas las revisiones posteriores a NC. No puede acreditarse un historial perfectamente point-in-time para esos campos.
- La fuente histórica de rankings deja un hueco entre 16-06 y 14-09-2026. La foto oficial observada el 14-09 queda fechada ese día; no rellena retrospectivamente julio/agosto. Se utiliza la última información disponible, que puede estar atrasada.
- La reconstrucción no convierte los precios de cierre en precios ejecutables ni sustituye libros faltantes. No permite afirmar un replay operativo completo donde faltan datos.
- El trabajo reconstruye el modelo v3 desplegado. Los experimentos v3_1 no se han presentado como modelos nuevos validados.

## Sombra activada y pendiente futuro

Se sustituyó el comando del LaunchAgent existente `com.ufcbet.watchdog` por `python -m ufc_pred.cli.shadow_runner`, conservando el intervalo de 60 segundos. Se guardó el plist anterior. La comprobación de hashes, actualización independiente del historial y primera ejecución completaron correctamente: historial hasta 12-09-2026, hoy sin evento UFC.

Se verifican hashes de 37 archivos del bundle/configuración y 74 módulos de ejecución. Cualquier cambio del modelo, política congelada o runtime bloquea la captura hasta versionarlo explícitamente. El protocolo fija revisión tras al menos 12 jornadas completas y 150 combates elegibles resueltos, sin retocar la estrategia entre resultados.

Los libros, variables y recomendaciones nuevas se guardarán en `data/processed/shadow_2026_09_14/`, con identificadores del bundle/historial y fecha de captura. Las capturas manuales `--once` no cuentan como observaciones prospectivas del protocolo. Las recomendaciones se simulan en un ledger de papel separado; se informa expresamente que no son fills reales. Se conservan calendarios y errores para poder detectar oportunidades omitidas.

**La validación futura todavía no existe:** cero combates capturados/resueltos al cierre de este trabajo. El Mac debe estar despierto y conectado. La operación automática preparada es Kalshi, que era el watchdog del repo; Polymarket se ha comparado históricamente, pero no se ha añadido un segundo watchdog prospectivo para ese mercado.

## Reproducción y archivos

Desde la raíz, utilizando el entorno existente:

```sh
./.conda/bin/python scripts/research/validate_result_envelope.py
./.conda/bin/python scripts/research/rebuild_corrected_system.py
./.conda/bin/python scripts/research/evaluate_corrected_system.py
./.conda/bin/python scripts/research/compare_positive_windows.py
./.conda/bin/python scripts/research/compare_archived_live_captures.py
./.conda/bin/python scripts/research/check_corrected_posteriors.py
./.conda/bin/python -m pytest -q
./.conda/bin/python -m ufc_pred.cli.shadow_runner --check
```

Los targets `corrected-history`, `corrected-skill`, `corrected-train`, `corrected-evaluate` y `shadow-check` del Makefile describen las fases. Reentrenar/sobrescribir archivos congelados requiere una nueva versión para no contaminar el seguimiento prospectivo. No ejecutar reconstrucciones sobre el bundle observado mientras se utiliza como referencia.

Resultados en `artifacts/corrected_2026_09_14/`: `manifest.json`, `runtime_manifest.json`, `historical_report_window_comparison.json`, `positive_window_comparison.json`, `comparison_results.json`, `execution_replay.csv`, `paired_price_comparison.parquet`, `archived_live_comparison.json`, `feature_parity.json`, `raw_live_provider_parity.json`, `posterior_validation.json`, `result_corrections.json`, `ledger_migration.json` y `shadow_protocol.json`.


Nota de publicación: las referencias a `artifacts/`, datos, registros operativos y documentos privados corresponden a archivos locales excluidos de Git. Esta copia pública omite saldos personales y rutas específicas del equipo; los originales se conservan localmente.
