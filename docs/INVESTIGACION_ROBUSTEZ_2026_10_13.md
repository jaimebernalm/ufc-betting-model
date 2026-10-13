# Por qué el modelo perdía en vivo y qué podría sustituirlo

13 de octubre de 2026. Trabajo conjunto de Claude (Opus 5.5) y Codex (GPT-6 Astra) en la rama `research/robustez-modelo-2026-10-13`, con subagentes para el scraping y la ingeniería de variables. No se ha tocado el checkout principal, el watchdog, las configuraciones en vivo ni dinero real.

## Conclusión para decidir

1. **El sistema actual tiene un rendimiento histórico negativo consistente** en todos los periodos y cortes evaluados.
   - Reentrenando su receta año a año (walk-forward estricto), apostar contra la cuota de cierre de las casas da **−7,0% de ROI** en 4.385 apuestas (IC95 −10,3% a −3,6%).
   - En 2024-2026, la misma receta con la transformación desplegada (T=1,25) pierde **−5,2% por oportunidad**.
   - Con Kelly 25% sin topes, la simulación termina en **ruina** en ambos periodos.
2. **Errores verificados y mecanismos plausibles**, no una atribución causal exhaustiva:
   - El modelo predice peor que el mercado (log-loss 0,700 frente a 0,607).
   - Sus discrepancias con el mercado van sobre todo hacia los **underdogs**, que en UFC son el lado perdedor del sesgo favorito-longshot.
   - En las 62 ejecuciones reales, las apuestas a precio ≤0,55 perdieron ≈36% y las de >0,55 ganaron ≈8% (muestra pequeña).
3. **La alternativa estudiada** parte de la cuota de mercado y solo aprende correcciones pequeñas:
   - La familia tiene **estimaciones favorables** de calidad probabilística en desarrollo y en 2024-2026.
   - **El finalista congelado (F1) no pasó la regla de confirmación fijada de antemano.** Ninguna candidata quedó promovida.
   - Su rentabilidad es **incierta**: el PnL estimado es positivo, pero con intervalos que incluyen pérdidas.
4. **Recomendación:**
   - No usar dinero real con ningún modelo; esto incluye el sistema actual, cuyo uso con dinero conviene parar. La decisión es tuya.
   - Si se quiere seguir, abrir un **estudio nuevo y exploratorio** de registro prospectivo en papel, con un protocolo propio. No es el cumplimiento de la puerta original, que F1 no superó. El logger de F1 ya existe (`ufc_pred.research.anchored_shadow`).

## 1. Qué estaba mal

### 1.1 El modelo competía contra un mercado mucho mejor que él

Experimento E0 (`scripts/research/r2026_10/e0_walkforward_vs_market.py`):
- Reentrenamiento anual expansivo 2013→2026, solo con peleas anteriores al 1 de enero del año evaluado.
- Referencia: la cuota de cierre de las casas, sin vig.
- Cinco semillas.

| Modelo (2016-2026, 4.731 peleas) | Log-loss | Δ frente al mercado | ROI a la cuota ofrecida, con vig |
|---|---:|---:|---:|
| Mercado (cierre, sin vig) | 0,607 | — | — |
| Receta desplegada **reentrenada** (2000 árboles, prof. 6, sin early stopping) | 0,700 | +0,093 | −7,0% (4.385 apuestas) |
| La misma, muy regularizada | 0,655 | +0,048 | −8,2% |

- La receta sobreajusta y produce probabilidades extremas. Eso es coherente con los "edges" de 10 a 66 céntimos vistos en vivo.
- Que quitar el early stopping "mejorara el ROI" en una ventana pequeña y reutilizada es compatible con haber seleccionado ruido.
- Con un mercado claramente mejor en log-loss, las discrepancias grandes son más probablemente error del modelo que valor. No hemos clasificado cada apuesta.

### 1.2 Sesgo favorito-longshot y dirección de las apuestas

- **Casas (unas 6.900 peleas).** La pendiente de calibración de logit(q) pasa de 1,03 (2010-14) a 1,08 (2018-22) y a **1,22 (2022-26)**.
  - Favoritos de 0,75-0,85: ganan el 84,5% frente al 79,3% implícito.
  - Underdogs de 0,15-0,25: ganan el 15,5% frente al 20,7% implícito.
- **Kalshi.** El patrón coincide con la literatura (Bürgi, Deng y Whelan, *Makers and Takers: The Economics of the Kalshi Prediction Market*): los contratos baratos pierden incluso después de comisiones.
- **El modelo desplegado elegía favorito solo en el 35-40% de sus apuestas.**
- **Sharpening T=1,25.** Empeora la log-loss (0,702 → 0,734 en dev; 0,653 → 0,670 en 2024-26). En PnL, sin embargo, A0T no fue peor que A0: −5,2% frente a −6,9% por oportunidad en 2024-26. Con estos datos no podemos afirmar que amplificara las pérdidas.

### 1.3 Problemas de datos y contratos encontrados en la revisión cruzada

| Hallazgo | Quién lo vio | Estado |
|---|---|---|
| `empty_arena`: en vivo se fuerza a False y en el entrenamiento de 2022-23 es NaN | Codex | Eliminado de los modelos nuevos |
| El histórico "T-90" de Kalshi no tiene timestamp acreditado (mezcla último trade y cierre) | Codex | Solo diagnóstico |
| El replay de septiembre dimensiona con resultados de la misma cartelera (lookahead) | Codex | El estudio nuevo fija el capital por cartelera |
| La definición de debutante con contadores v2 falla (incluyen combates externos) | Codex | Se usan apariciones UFC previas |
| Columnas de metadatos (`debut`) entraban como features por la lista de exclusión | Codex | Corregido con test; ejecuciones afectadas repetidas |
| GLM con winsorización asimétrica, que rompía la simetría de esquinas | Claude, con test | Corregido |
| Relleno de cuotas con la mediana de cuotas americanas (precio inválido) | Codex | Un par real de una casa, por prioridad fija |
| Emparejamientos BFO por inicial: Junior y Justin Tafa confundidos | Codex | Solo se aceptan emparejamientos exactos |
| Orientación de los snapshots de Kalshi: siguen el ticker de Kalshi, no la esquina UFC | Claude | Corregido en el análisis; el runner en vivo era correcto |
| El "cierre" de Kalshi en BFO se registra **después** del resultado | Subagente BFO | Nunca se usa como precio |
| ~25% de las capturas archivadas tienen un timestamp en o tras el inicio *estimado* de la pelea. UFC.com solo publica el inicio por sección; el runner estima sección + 30 min × índice | Claude y Codex | No demuestra precios in-play, pero el disparador no se comporta como un "T-90"; conviene revisarlo |

## 2. Qué se probó

### Protocolo

- **Desarrollo:** años 2016-2023, con reentrenamiento anual.
  - La selección se hizo aquí, entre unos 15 candidatos. Por esa búsqueda múltiple, los intervalos de dev son **nominales**.
- **Congelación:** F1, controles y ablaciones se congelaron con hashes en `artifacts/r2026_10/manifest.json` **antes de mirar 2024-2026**. Algunas comparaciones de dev ya se habían hecho antes de esa congelación.
- **Confirmación:** 2024-2026, mirada una sola vez.
  - **No es un holdout virgen:** ese periodo ya se había usado en el test histórico, en las auditorías de septiembre y en E0.
  - Es una confirmación retrospectiva de una receta congelada hoy.
- **G1:** diferencia pareada de log-loss frente al mercado sin calibrar, con bootstrap por cartelera. Exige que el extremo superior del IC sea menor que 0.
- **G2:** PnL plano por oportunidad elegible (sin debutantes) a la cuota ofrecida con vig, apostando cuando EV > 2%.
- **Regla de promoción congelada:** recomendar F1 para una sombra nueva solo si pasaba G1 y tenía PnL positivo en 2024-2026.
- **Semillas:** 10 por candidata final y 5 en E0.

### Familias evaluadas

- **Controles:** la receta v2 reentrenada (A0) y la misma receta reentrenada con T=1,25 (A0T, también con la regla desplegada "p − precio ≥ 3c").
- **Corrección FLB pura:** σ(a·logit q), con *a* ajustado solo con años anteriores.
- **Residuo anclado al mercado:** logit p = a·logit q + f(x).
  - f es un CatBoost pequeño: 300 árboles, profundidad 3, l2 = 20.
  - Su línea base es el mercado ya corregido por FLB.
- **Variables nuevas:**
  - Props de método de cierre (KO, sumisión, decisión).
  - B1 (Codex): tasas suavizadas de ataque y defensa con exposición.
  - B2 (Codex): ajuste por rival con Poisson ridge, reajustado cada mes solo con el pasado.
  - Movimiento de línea entre apertura y cierre (BFO).

## 3. Resultados

### Desarrollo, 2018-2023 (IC nominales tras selección múltiple)

Soporte común: 2.812 peleas; 2.229 oportunidades sin debutantes.

| Candidato | Δ log-loss frente al mercado [IC95] | EV>2%: apuestas | ROI [IC95] | PnL/oportunidad |
|---|---|---:|---|---:|
| A0T (receta desplegada reentrenada, T=1,25) | +0,122 | 2.024 | −3,2% | −2,9% |
| Corrección FLB pura | −0,0012 [−0,0021; −0,0003] | 3 | — | ≈0 (el vig de casa lo absorbe) |
| R3: residuo anclado | −0,0026 [−0,0052; +0,00001] | 632 | +10,8% [+2,6; +18,9] | +3,1% |
| **F1 = R6_mkt** (R3 + props de método) | **−0,0028 [−0,0055; −0,0001]** | 662 | **+12,2% [+3,5; +20,6]** | **+3,6%** |
| F1 + B1 (Codex) | −0,0039 [−0,0067; −0,0010] | 771 | +10,3% [+2,2; +18,0] | +3,5% |
| F1 + B1 + B2 (Codex) | −0,0036 [−0,0067; −0,0006] | 867 | +8,4% [+1,5; +15,5] | +3,3% |

**Estabilidad**, calculada con los OOF limpios de 10 semillas (EV > 2%):

| | Años 2018-2023 | Semillas |
|---|---|---|
| R3 | positivo en los 6 (+1,5% a +17,9%) | +6,6% a +12,3% |
| F1 | positivo en 5 de 6 (2021: −0,7%) | +8,8% a +14,3% |

**Lo que no se cuenta como evidencia:**
- La antigua sensibilidad de hiperparámetros de R3 se ejecutó antes de corregir el contrato de `debut`. Solo se conserva como trazabilidad.
- B1 y B2 no demostraron mejora frente a F1. La diferencia pareada fue −0,0011 [−0,0024; +0,0003] y −0,0008 [−0,0028; +0,0011]. Por la regla fijada de antemano, F2 se omitió.
- El movimiento de línea entre apertura y cierre **no mostró mejora en este experimento** (coeficiente 0,035; Δ +0,001).

### Confirmación 2024-2026 (mirada única; 1.367 peleas, 1.115 oportunidades sin debutantes)

| Modelo | Δ log-loss [IC95] | EV>2%: PnL/oportunidad | ROI [IC95] |
|---|---|---:|---|
| **F1 (congelado)** | −0,0043 [−0,0087; **+0,0002**] | +0,6% | +2,1% [−6,8; +10,6] |
| R3 (ablación) | −0,0041 [−0,0086; +0,0004] | +0,7% | +2,5% |
| F1 + B1 (ablación) | −0,0054 [−0,0103; −0,0003] | +1,4% | +4,0% |
| F1 + B1 + B2 (ablación) | −0,0060 [−0,0110; −0,0007] | +1,8% | +4,6% [−2,9; +12,5] |
| A0 (receta desplegada reentrenada) | +0,074 | −6,9% | −7,9% |
| A0T (misma, T=1,25) | +0,091 | −5,2% | −5,9% |

**F1 no queda confirmado** y ninguna candidata fue promovida. El extremo superior de G1 es +0,0002. Su estimación de log-loss es favorable y su PnL estimado es positivo, pero el IC incluye pérdidas, y eso no acredita rentabilidad. Las ablaciones con B1/B2 pasan G1 en este periodo, pero no reemplazan a F1: su superioridad frente a F1 exige diferencias pareadas, y estas no son significativas (B1−F1: −0,0011 [−0,0028; +0,0007]; B1B2−F1: −0,0017 [−0,0040; +0,0006]).

Detalles de este bloque:
- Las 371 filas añadidas con cuotas BFO no tienen props de método, porque esa variable procede del export de Kaggle.
- Las tablas salen del archivo autocontenido `artifacts/r2026_10/claude/holdout_eval.parquet` (SHA256 `3a6bc351…`).
- El `bfo_matched.parquet` actual se regeneró después: tiene la misma q en esas 1.367 peleas y añadiría 7.

### Tamaño de apuesta (estudio separado)

¼-Kelly, tope del 2% por pelea y del 15% por cartelera, capital fijo por cartelera:

| | 2018-2023 (dev) | 2024-2026 |
|---|---|---|
| F1 | ×2,46 (DD máx. 14%) | ×1,03 (DD 14%) |
| F1 + B1 + B2 | — | ×1,15 (DD 22%) |
| A0T (receta desplegada reentrenada), mismos topes | ×1,09 (DD 34%) | ×0,58 (DD 53%) |
| A0T, ¼-Kelly sin topes | **ruina** | **ruina** |

### Diagnóstico en Kalshi (capturas en vivo jun-oct 2026: muestra pequeña, no es un backtest ejecutable)

Hay 13 carteleras. En el universo común sin debutantes (59 peleas), con EV > 2%:

| Modelo | Anclado a casas | Anclado al precio de Kalshi |
|---|---:|---:|
| Modelo antiguo | −3,1% por oportunidad | — |
| F1 | +4,2% | +9,1% |

Los IC son muy anchos. Lo único que se repite en todos los cortes es la dirección: el modelo antiguo apuesta más a underdogs y pierde, mientras el anclado apuesta mayoritariamente a favoritos.

## 4. Qué recomendamos y qué no

- **Recomendamos no usar dinero real** con ningún candidato y parar el uso del sistema actual con dinero. No se ha tocado nada; la decisión es tuya.
- **Estudio nuevo, exploratorio e independiente (opcional).** Es un registro prospectivo en papel con su propio protocolo, declarado ahora y antes del primer dato. No sustituye a la promoción fallida.
  - **Co-candidatas:** F1 y F1+B1+B2. Ninguna ablación se promoverá mirando resultados.
  - **Controles:** el precio contemporáneo sin calibrar y la corrección FLB (MF).
  - **Métrica primaria:** Δ log-loss frente al precio contemporáneo; las señales son hipotéticas. Para evaluar una política económica haría falta PnL con costes y **una recaptura de Kalshi en el momento de decidir**: un ask anterior no acredita la ejecución después de ver BFO.
  - **Registro:** todas las oportunidades y todas las abstenciones con su motivo, timestamps completos y la procedencia y frescura de la historia base y de B1/B2.
  - **Horizonte fijo:** primera revisión con 12 carteleras completas y 150 peleas elegibles, o el 2027-03-31, lo que ocurra antes. Es un suelo de revisión, no una garantía de potencia. Un resultado inconcluso sigue siendo inconcluso.
  - **Estado:** el **logger de F1 está disponible**; la **comparación prospectiva de las dos co-candidatas sigue pendiente**. F1+B1+B2 necesita las 34 columnas B1/B2 de cada pelea futura (`codex_future_performance.py --as-of <fecha>`), calculadas sobre un snapshot de UFCStats actualizado y versionado. El snapshot actual termina el 2026-09-12.
- **Revisar el disparador del runner:** una de cada cuatro capturas llega en o después del inicio estimado.
- **No recomendamos:**
  - Dinero real.
  - Kelly sin topes.
  - Escoger entre ablaciones a la vista de 2024-2026.
- **Nota de método (preferencia de Claude, no una objeción de Codex).** Para este logger de probabilidades, ambos aceptamos la Δ log-loss como métrica primaria. Para evaluar una política económica, PnL/ROI con costes y ejecución contemporánea, conforme al CLAUDE.md. Claude añade una preferencia propia: en futuras selecciones económicas, exigir además información incremental frente al mercado como puerta, porque el ROI sobre ventanas pequeñas y reutilizadas es muy ruidoso. No se propone cambiar el CLAUDE.md.

## 5. Herramientas entregadas

- **`src/ufc_pred/research/harness.py` y `anchored.py`:** harness walk-forward, puertas y modelos anclados, con tests en `tests/research/`.
- **`src/ufc_pred/research/bfo.py`:** scraper educado y con caché de BestFightOdds.
  - Cubre 804 eventos y el 96% de las peleas emparejadas, con aperturas desde 2016 y cierres por casa desde 2021.
  - **No** decodifica el endpoint ofuscado de gráficos.
- **`src/ufc_pred/research/perf_features.py`, `perf_adjusted.py` y `perf_future.py` (Codex):** B1/B2 históricos y para peleas futuras con una fecha as-of explícita; 112 tests y paridad exacta del replay.
- **`src/ufc_pred/research/anchored_shadow.py`:** **logger en papel de F1** (probabilidades contemporáneas y señales hipotéticas; nunca ejecuta órdenes).
  - **Entrada:** las capturas de la sombra v2 (variables y snapshot de Kalshi). Descarga BFO en el momento: cuotas de casas y props de método, solo de casas, con los códigos explícitos de `data-li`.
  - **Abstenciones:**
    - timestamps ausentes;
    - captura futura, o en o tras el inicio estimado;
    - captura antigua;
    - BFO no contemporáneo (desfase medido *después* de descargar; 10 min por defecto);
    - cartelera anterior a la fecha de ajuste del bundle;
    - debutante;
    - precio ausente o probabilidad no finita.
  - **Trazabilidad:** cada registro guarda los hashes de la captura, del runner y del manifiesto del bundle, más una copia inmutable del HTML de BFO usado.
  - **Tope por cartelera:** el 15% se aplica a posteriori en `report()` como escalado *descriptivo*, no como sizing online con efectivo reservado.
  - **Bundle:** `artifacts/r2026_10/anchored_shadow_bundle_v1/`, con todo lo anterior al 2026-10-13. Nunca se sobrescribe; al cargarlo se verifican modelos, datos de ajuste, política y código del modelo. El v0 se conserva solo como trazabilidad.
  - **Tests:** las puertas tienen tests sintéticos aislados en un directorio temporal (`tests/research/test_anchored_shadow_gates.py`).
  - **No está conectado a launchd.**

## Reproducción

Desde el worktree. La ruta del intérprete contiene espacios, así que va entre comillas.

```bash
cd /Users/jaimebernal/Desktop/pred_project/ufc-pred-research
PY="/Users/jaimebernal/Desktop/pred_project/UFC pred/.conda/bin/python"
export PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1
"$PY" scripts/research/r2026_10/e0_walkforward_vs_market.py --years 2013-2026 --seeds 5
"$PY" scripts/research/r2026_10/claude_dev_run.py --cands R3 --seeds 10
"$PY" scripts/research/r2026_10/claude_dev_run.py --cands R6 --seeds 10 --perf artifacts/r2026_10/claude/mkt_extra.parquet --tag _mkt
"$PY" scripts/research/r2026_10/claude_dev_report.py
"$PY" scripts/research/r2026_10/claude_holdout_from_frozen.py
"$PY" -m pytest -q tests/research
```

- **Tablas de 2024-2026:** se reproducen solo desde el archivo congelado (`claude_holdout_from_frozen.py` comprueba su SHA256 y no reescribe nada).
- **No volver a ejecutar** `claude_run_holdout.sh` ni `claude_holdout_report.py` como reproducción: recargan el BFO actual (7 peleas más) y sobrescribirían la evidencia congelada.
- **Manifiesto:** `artifacts/r2026_10/manifest.json` recoge la congelación con hashes, las revisiones posteriores y sus motivos.

Fuentes externas:
- [Bürgi, Deng y Whelan, UCD WP25/19](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)
- [CEPR DP20631](https://cepr.org/publications/dp20631)
- [Tarifas de Kalshi](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
- [Robots de BestFightOdds](https://www.bestfightodds.com/robots.txt)
