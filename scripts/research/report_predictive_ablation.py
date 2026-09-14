"""Render the diagnostic report from persisted experimental results."""

import json

from ablate_predictive_degradation import OUT

from ufc_pred.paths import ROOT

r = json.loads((OUT / "ablation_results.json").read_text())
assert len(r["completed_variants"]) == r["total_planned"] == 14
candidate = "profiles_plus_prior_external_counters"
labels = {
    "old_frozen": "Archivos antiguos + filas históricas antiguas",
    "corrected": "Primera reconstrucción corregida",
    candidate: "Candidata: perfiles completados + antecedentes externos previos",
}
text = """# Por qué cambiaron las predicciones: comparaciones por bloques

14 de septiembre de 2026. Estudio diagnóstico sobre periodos reutilizados, sin cambiar cutoff, semillas, hiperparámetros ni estrategia.

## Conclusión para decidir

La primera reconstrucción no fue una mejora uniforme: eliminó información posterior al combate presente en las tasas antiguas, pero también perdió atributos físicos disponibles y antecedentes previos válidos al restringir todo el historial a UFC. Son efectos distintos. Los rankings y la habilidad corregida no explican el deterioro principal.

Se evaluaron 14 configuraciones con diez semillas cada una, incluyendo controles de aumento de datos, intervenciones por bloques y restauraciones inversas. Una configuración es equivalente a otra por construcción y se reutiliza como comprobación de consistencia; no se cuenta como evidencia independiente. Los 20 bundles de la candidata están guardados aparte. No se ha sustituido el modelo del watchdog.

La candidata recupera buena parte de la calidad probabilística perdida, pero **no mejora de forma general la rentabilidad de las simulaciones**. No volvería a usar tasas históricas contaminadas para recuperar mejores métricas. Tampoco consideraría demostrada una ventaja rentable de la candidata.

## 1. La población sobre la que se medía el deterioro importa

De los 408 combates evaluados, 97 quedan excluidos por el filtro de debut y 311 son elegibles por experiencia. Los 97 excluidos concentran el 89,3% del aumento total de log loss A/B entre los archivos antiguos y la primera reconstrucción. Es una descomposición del error observado, no una atribución causal exclusiva al debut.

El error en los debutantes sube de 0,5602 a 0,7891; entre los elegibles sube de 0,6584 a 0,6669. Por tanto, describir únicamente la métrica global como una pérdida general de capacidad exageraba el deterioro en la población donde se apuesta.

| Población | Versión | Log loss A/B | Brier A/B | Log loss C | Brier C |
|---|---|---:|---:|---:|---:|
"""
for subset, label in [("all", "408 combates"), ("eligible", "311 elegibles")]:
    for name, display in labels.items():
        d = r["models"][name][subset]
        text += f"| {label} | {display} | {d['real']['log_loss']:.4f} | {d['real']['brier']:.4f} | {d['corrupted']['log_loss']:.4f} | {d['corrupted']['brier']:.4f} |\n"
text += """
Menor es mejor. La referencia antigua utiliza las filas antiguas, que contienen tiempos mezclados. No equivale a evaluar los archivos antiguos con información reconstruida anterior al combate.

El intervalo del 95% por remuestreo emparejado de jornadas para el deterioro de log loss en elegibles es [−0,0384, +0,0568] en A/B y [−0,0478, +0,0522] en C. Ambos incluyen cero. La candidata también tiene intervalos que incluyen cero al compararla con esa referencia antigua: A/B [−0,0616, +0,0379], C [−0,0683, +0,0368]. No hay fundamento para declarar superioridad definitiva frente a los archivos antiguos a partir de esta muestra.

Sí se observa recuperación frente a la primera reconstrucción: en los 408 combates, cambio de log loss A/B −0,0378, intervalo [−0,0615, −0,0134]; C −0,0354, intervalo [−0,0592, −0,0104]. Estos intervalos describen esta muestra y no descuentan la exploración de varias variantes sobre ella.

## 2. Qué muestran los reentrenamientos controlados

Cada fila incorpora el bloque indicado sobre la anterior. Se reconstruye/reentrena con las mismas semillas y receta, y se evalúa sobre los mismos combates. Los nombres de los bloques incluyen sus diferencias derivadas para no combinar, por ejemplo, tasas nuevas con diferencias antiguas.

| Paso acumulativo | Log loss global A/B | Log loss elegibles A/B | Log loss elegibles C |
|---|---:|---:|---:|
"""
sequence = [
    ("00_old_features_fixed_symmetry", "Datos antiguos, aumento de datos corregido"),
    ("01_verified_results", "+ resultados verificados y exclusiones"),
    ("02_add_rankings", "+ rankings corregidos"),
    ("03_add_skill", "+ habilidad corregida"),
    ("04_add_profiles", "+ perfiles de la exportación reconstruida"),
    ("05_add_counters", "+ contadores UFC-only"),
    ("06_add_rates", "+ tasas estrictamente previas, unidades homogéneas"),
]
for name, label in sequence:
    d = r["models"][name]
    text += f"| {label} | {d['all']['real']['log_loss']:.4f} | {d['eligible']['real']['log_loss']:.4f} | {d['eligible']['corrupted']['log_loss']:.4f} |\n"
text += """
La habilidad mejora claramente este bloque: log loss global 0,6512 → 0,6191. Los perfiles reconstruidos empeoran la métrica global pero mejoran ligeramente la de elegibles en ese punto. Sustituir las tasas antiguas al final aumenta el error global de 0,6458 a 0,6960. Ese salto no debe interpretarse como prueba de que las tasas posteriores al combate sean una opción válida para producción.

Las restauraciones inversas confirman interacciones: mantener todo corregido y restaurar solamente los contadores antiguos da 0,6630 global / 0,6442 elegibles; restaurar solamente tasas antiguas da 0,6458 / 0,6503; restaurar habilidad antigua empeora a 0,7080 / 0,6764. Los efectos dependen del resto de las variables. No es correcto asignar a cada arreglo un porcentaje causal único sumando estos cambios.

Los controles de aumento de datos antiguo se guardan separadamente. No reproducen exactamente los archivos entrenados en mayo con el dataset disponible en septiembre: en la semilla 0 el máximo cambio de probabilidad es 0,2487, media 0,0477. Por ello, la diferencia entre el artefacto de mayo y un nuevo ajuste no se atribuye exclusivamente al signo. En cambio, repetir el entrenamiento corregido con dos hilos reproduce exactamente las probabilidades del bundle corregido original: diferencia máxima 0,0. Las comparaciones entre bloques dentro del estudio mantienen los demás factores controlados.

## 3. Información futura y pérdida de información válida

### Tasas de tiempos mezclados

En 322 de las 816 filas de peleador del periodo evaluado, las cinco tasas antiguas son compatibles con el estado posterior al combate dentro de 0,011 de tolerancia de redondeo, y no con el estado anterior. Esto es un indicador conservador de compatibilidad temporal, no un censo exhaustivo de todas las fugas. El updater antiguo, además, tomaba tasas del perfil actual al añadir resultados históricos.

Un ejemplo claro es Mairon Santos ante Muhammad Naimov el 06-12-2025: la fila antigua registra 3,33 golpes significativos por minuto, mientras que antes del combate eran 3,53; 3,33 corresponde al estado posterior. En Fares Ziam de esa jornada, la tasa de intentos de sumisión antigua 0,5 coincide con la posterior; la previa era 0,2. No son valores que pudieran reconstruirse antes del combate usando esos resultados todavía desconocidos.

La escala tampoco era homogénea: la media de `R_avg_SIG_STR_landed` en entrenamiento era 19,89 y en evaluación 4,55; en la versión reconstruida son 3,44 y 3,89. Por tanto, intercambiar datos sin reentrenar produce además un cambio de distribución.

### Atributos físicos que se perdieron

La comparación encontró 289 celdas de atributos antes presentes que quedan vacías: 84 alturas, 82 alcances, 106 guardias y 17 edades. Afectan a 101 peleadores. En evaluación, 66 combates pierden algún atributo, incluidos 16 elegibles; ese subconjunto concentra el 62,9% del incremento de log loss A/B. Se solapa con los debutantes: no se pueden sumar ambos porcentajes.

Se verificaron los 101 perfiles contra sus páginas oficiales y se completaron los atributos faltantes que pudieron verificarse cuando la identidad coincidía. Se completaron 235 celdas de atributos del dataset, manteniendo vacíos los valores no verificables. Se dejó sin completar una discrepancia de nombre en el perfil de Jose Montanha, cuya cabecera muestra Henrique Da Silva Lopes: requiere confirmar si es un alias o una identidad incorrecta. No se decidió por similitud. Las páginas de Anthony Wint, Terrance Chatman y Guilherme Pat contienen altura/alcance/guardia que faltaban en la exportación fijada.

Completar esos campos y reentrenar mejora la log loss global A/B de 0,6960 a 0,6801. La fecha histórica de publicación de esos atributos está sin documentar: este resultado es una sensibilidad a calidad de fuente, no una garantía de disponibilidad histórica perfecta.

### Antecedentes previos fuera de UFC

Restringir todas las estadísticas a UFC eliminó información previa real. Por ejemplo, antes de Allen–Jean Silva, Silva tenía cinco victorias UFC y una victoria adicional en DWCS; Waldo Cortes Acosta tenía diez victorias UFC y una adicional. Excluir DWCS era una decisión de alcance para conseguir consistencia entre proveedores, no una corrección objetivamente obligatoria.

Se revisaron los 796 peleadores cuyos contadores cambiaban entre ambas fuentes. En 528 aparecieron antecedentes externos; se registraron 1.071 apariciones de peleador fechadas (no 1.071 jornadas ni necesariamente combates distintos). Se combinaron con los resultados UFC brutos, utilizando para cada predicción solamente registros de días anteriores. Se verificaron todas esas identidades; la ambigüedad de Bruno Silva se resolvió con sus IDs ya auditados.

Para los restantes peleadores se conserva el historial UFC de referencia. Esta reconstrucción está dirigida a la población con discrepancias conocidas; no se presenta como un censo completo de todas las promociones para los 2.268 peleadores del dataset. Las tasas permanecen UFC-only; se amplían exclusivamente los contadores, con el mismo significado en entrenamiento y evaluación. El filtro de debut sigue basándose en el historial UFC y no cambia.

La candidata combina estos contadores previos con la compleción de atributos faltantes. Recupera aproximadamente el 62% del incremento de log loss global A/B de la primera reconstrucción.

## 4. Comparación de modelos con exactamente las mismas entradas

Esta comprobación separa la elección de pesos del modelo de la ventaja de tener otras variables:

| Entradas de evaluación | Pesos | Log loss elegibles A/B | Log loss elegibles C |
|---|---|---:|---:|
"""
for name, input_label, model_label in [
    ("old_on_corrected_features", "Primeras variables corregidas", "Archivos antiguos"),
    ("corrected", "Primeras variables corregidas", "Reentrenamiento corregido"),
    ("old_on_candidate_features", "Variables de la candidata", "Archivos antiguos"),
    (candidate, "Variables de la candidata", "Reentrenamiento candidato"),
]:
    d = r["models"][name]["eligible"]
    text += f"| {input_label} | {model_label} | {d['real']['log_loss']:.4f} | {d['corrupted']['log_loss']:.4f} |\n"
text += """
Con las mismas entradas de la candidata, el reentrenamiento mejora respecto a los archivos antiguos. No recomendaría conectar esos archivos antiguos directamente al constructor nuevo.

## 5. Las simulaciones no mejoran de forma uniforme

Para mantener continuidad con el informe anterior, esta tabla usa exactamente su población elegible original y los mismos precios/comisiones. ROI con importe fijo de efectivo; no es crecimiento Kelly:

| Ventana | Cuenta/modelo | Primera reconstrucción | Candidata |
|---|---|---:|---:|
| Polymarket, 06-12-2025 a 16-05-2026 | A/B | +11,74% | +8,56% |
| Polymarket, misma ventana | C | +16,71% | +8,68% |
| Kalshi, 24-01 a 16-05-2026 | A/B | +30,57% | +22,74% |
| Kalshi, misma ventana | C | +31,86% | +25,71% |
| Kalshi, 30-05 a 01-08-2026 | A/B | −22,77% | −24,79% |
| Kalshi, misma ventana | C | −23,90% | −21,87% |

Las tablas completas del CSV utilizan la elegibilidad corregida por identidades, idéntica entre variantes: en esa población la candidata da Polymarket +7,86%/+7,99%, Kalshi inicial +19,95%/+22,78%, Kalshi posterior −24,79%/−21,87%. La diferencia entre estas dos tablas procede exclusivamente de la población elegible; no se han cambiado los filtros para favorecer una variante.

El replay hipotético de los 72 combates Kalshi con asks, profundidad 1.000 dólares, un céntimo adverso y efectivo bloqueado hasta fin de jornada da:

| Cuenta | Capital inicial | Final primera reconstrucción | Final candidata | Drawdown primera | Drawdown candidata |
|---|---:|---:|---:|---:|---:|
"""
for a in ["A", "B", "C"]:
    old = r["execution"]["corrected"][a]
    new = r["execution"][candidate][a]
    text += f"| {a} | {old['initial']:.2f} | {old['final_equity']:.2f} | {new['final_equity']:.2f} | {old['max_drawdown']:.2%} | {new['max_drawdown']:.2%} |\n"
text += """
La candidata pierde menos y tiene menor drawdown en este escenario concreto. No demuestra menor volatilidad general. Los periodos positivos anteriores carecen de asks/profundidad histórica completos; las oportunidades sin precio siguen sin inventarse. La rentabilidad realizada requiere fills reales.

También se midió la log loss antes de sharpening como diagnóstico, sin modificar T=1,25 en la estrategia. La candidata A/B elegible pasa de 0,6465 con T=1,25 a 0,6312 sin sharpening. Esto sugiere exceso de confianza residual, pero no se ha elegido otra temperatura sobre estos datos reutilizados.

## 6. Discrepancia adicional de producción: paradas médicas

La comprobación independiente de ocho perfiles detectó que el resumen del perfil oficial presenta algunas paradas médicas como KO/TKO, mientras el detalle del combate distingue `TKO - Doctor's Stoppage`. El parser actual usa el resumen. Afecta, por ejemplo, a Jean Silva y Curtis Blaydes: cambia la separación entre victorias KO y doctor aunque el total sea igual.

Esto no explica por sí mismo el deterioro de las métricas históricas: es una discrepancia adicional entre entrenamiento e inferencia que las pruebas anteriores con otros perfiles no detectaron. Se ha preparado un parche que lee el método detallado de los combates previos. Se verificó sobre HTML oficial archivado de Silva, Blaydes y Arnold Allen, con cero diferencias respecto al constructor histórico. Está guardado como parche revisable; aplicar cambios al runtime congelado requiere una nueva versión de sombra.

## 7. Decisión y estado de entrega

Mantendría las correcciones de identidad, rankings, habilidad, temporalidad, contabilidad y ejecución. Conservaría información externa previa válida en vez de excluirla por conveniencia del export, y completaría atributos verificados. No recuperaría tasas posteriores al combate ni seleccionaría el modelo por el mayor ROI retrospectivo.

La candidata merece ser la siguiente versión a validar, porque mejora sobre la primera reconstrucción con una razón verificable en los datos. Sus resultados de apuestas siguen siendo mixtos. Antes de usarla en el watchdog hay que integrar su contrato de entradas, corregir el parser de paradas médicas y congelar una nueva versión; los bundles están marcados `deployment_ready: false`. No se ha promovido un modelo por estas simulaciones ni se han tocado los parámetros de la estrategia.

La sombra existente continúa con su versión congelada. No se enviaron órdenes ni alertas. Todavía no hay validación prospectiva de la candidata.

## Reproducibilidad y comprobaciones

- Cutoff 30-11-2025; 7.017 filas de entrenamiento verificadas; diez semillas; 2.000 árboles, profundidad 6, learning rate 0,05, L2=3, recencia 4 años. La referencia sin corrección de resultados tiene 7.030 filas.
- Simetrización en inferencia, T=1,25, Kelly y caps A/B/C, edge de 3%, profundidad y mínimo de apuesta: conservados.
- 14 configuraciones completas, diez semillas por configuración y predicciones de cada tratamiento real/C; todas las evaluaciones usan 408 combates emparejados. Las semillas no se tratan como muestras temporales independientes.
- Igualdad comprobada entre el último bloque acumulativo y el bundle corregido original. Un ajuste de referencia con diferente número de hilos da diferencia de probabilidad 0,0. Empaquetar los 20 bundles de la candidata conserva exactamente sus probabilidades: error máximo 0,0.
- Se comprobaron independientemente 44.550 identidades aritméticas de victorias, derrotas y asaltos: valor UFC de referencia más antecedentes externos estrictamente anteriores, con cero discrepancias.
- Se conservan datasets, pesos, predicciones individuales, resultados por combate, intervalos por jornada, importancias, HTML de evidencia y manifiestos SHA256. Los manifests y pruebas del runtime congelado siguen verificando.
- Límites: ventanas reutilizadas, hueco histórico de rankings, fechas de publicación de perfiles/NC desconocidas, cobertura externa dirigida y libros históricos incompletos. Los intervalos reportados no corrigen búsquedas múltiples ni crean validación prospectiva.

Secuencia reproducible, desde la raíz del repo (los bundles de investigación están separados de producción):

```sh
./.conda/bin/python scripts/research/ablate_predictive_degradation.py --variants 06_add_rates
./.conda/bin/python scripts/research/diagnose_predictive_degradation.py
./.conda/bin/python scripts/research/complete_ablation_profiles.py
./.conda/bin/python scripts/research/rebuild_ablation_prior_counters.py
./.conda/bin/python scripts/research/ablate_predictive_degradation.py
./.conda/bin/python scripts/research/diagnose_predictive_degradation.py
./.conda/bin/python scripts/research/summarize_predictive_ablation.py
./.conda/bin/python scripts/research/package_ablation_candidate.py
./.conda/bin/python scripts/research/verify_proposed_doctor_fix.py
```

Las comprobaciones oficiales iniciales se reproducen con `verify_ablation_source_states.py`; su caché conserva la fecha/vintage de este estudio. No reemplazar esa evidencia por respuestas futuras sin versionarla.
"""
text += f"\n[Resultados completos]({OUT}/ablation_summary.csv) · [Resultados e intervalos]({OUT}/ablation_results.json) · [Candidata y contrato pendiente]({OUT}/candidate/manifest.json) · [Parche del parser]({OUT}/proposed_doctor_parser_fix.patch)\n"
path = ROOT / "docs/ABLACIONES_Y_CAUSAS_DEL_DETERIORO_2026_09_14.md"
path.write_text(text)
print(path)
