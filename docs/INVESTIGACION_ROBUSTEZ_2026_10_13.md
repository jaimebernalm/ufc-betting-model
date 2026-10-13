# Por qué el modelo perdía en vivo y qué lo sustituiría

13 de octubre de 2026. Trabajo conjunto de Claude (Opus 5.5) y Codex (GPT-6 Astra) en la rama `research/robustez-modelo-2026-10-13`, con subagentes para el scraping y la ingeniería de variables. No se ha tocado el checkout principal, el watchdog, las configuraciones en vivo ni dinero real.

## Conclusión para decidir

1. **El sistema actual tiene un rendimiento histórico negativo consistente.** Reentrenando su receta año a año (walk-forward estricto), apostar contra la cuota de cierre de las casas da −7,0% de ROI en 4.385 apuestas (IC95 −10,3% a −3,6%). En el periodo 2024-2026 pierde −5,2% por oportunidad con la transformación desplegada (T=1,25). Con su Kelly 25% sin topes, la simulación termina en **ruina** tanto en 2018-2023 como en 2024-2026. Esto no es mala suerte.
2. **El mecanismo está identificado.** El modelo predice peor que el mercado (log-loss 0,700 frente a 0,607). Trata ese error como "edge" y además lo dirige sistemáticamente hacia los **underdogs**, que es justo el lado perdedor del sesgo favorito-longshot del mercado de UFC. En las 62 ejecuciones reales, las apuestas a precio ≤0,55 perdieron ≈36%; las de >0,55 ganaron ≈8%.
3. **La alternativa construida** parte de la cuota de mercado y solo aprende correcciones pequeñas: la corrección del sesgo favorito-longshot y variables de rendimiento. Mejora la probabilidad del mercado tanto en desarrollo como en 2024-2026, y lleva el resultado económico de claramente negativo a aproximadamente neutro o algo positivo. **No obstante, el finalista congelado no pasó la regla de confirmación fijada de antemano.** Por eso no hay base para apostar dinero real con él.
4. **Recomendación:** dejar de usar el sistema actual con dinero real y observar la alternativa en sombra de papel, con un protocolo prospectivo nuevo. La herramienta está construida (`ufc_pred.research.anchored_shadow`). Activarla y decidir cuándo es cosa tuya.

## 1. Qué estaba mal

### 1.1 El modelo competía contra un mercado mucho mejor que él

Experimento E0 (`scripts/research/r2026_10/e0_walkforward_vs_market.py`): reentrenamiento anual expansivo 2013→2026, entrenando solo con peleas anteriores al 1 de enero del año evaluado. La referencia es la cuota de cierre de las casas, sin vig. Se usaron cinco semillas.

| Modelo (2016-2026, 4.731 peleas) | Log-loss | Δ frente al mercado | ROI a la cuota ofrecida, con vig |
|---|---:|---:|---:|
| Mercado (cierre, sin vig) | 0,607 | — | — |
| Receta desplegada (2000 árboles, profundidad 6, sin early stopping) | 0,700 | +0,093 | −7,0% (4.385 apuestas) |
| La misma, muy regularizada | 0,655 | +0,048 | −8,2% |

La receta sobreajusta y produce probabilidades extremas: de ahí los "edges" de 10 a 66 céntimos observados en vivo. Quitar el early stopping "mejoró el ROI" en una ventana pequeña y reutilizada. Eso era seleccionar ruido. Con un mercado tan superior, casi toda discrepancia de 3 céntimos o más es error del modelo.

### 1.2 Sesgo favorito-longshot: el error iba en la dirección equivocada

- **Casas, unas 6.900 peleas.** La pendiente de calibración de logit(q) crece de 1,03 (2010-14) a 1,08 (2018-22) y a **1,22 (2022-26)**. Los favoritos de 0,75-0,85 ganan el 84,5% de las veces frente al 79,3% implícito. Los underdogs de 0,15-0,25 ganan el 15,5% frente al 20,7% implícito.
- **Kalshi.** El patrón coincide con la literatura (Bürgi, Deng y Whelan, *Makers and Takers: The Economics of the Kalshi Prediction Market*): los contratos baratos pierden incluso después de comisiones.
- **El modelo desplegado elegía favorito solo en el 35-40% de sus apuestas.** Apostaba precisamente al lado sobrevalorado.
- **El sharpening T=1,25 amplificaba el problema**, porque agrandaba exactamente las discrepancias que eran error.

### 1.3 Problemas de datos y contratos encontrados en la revisión cruzada

| Hallazgo | Quién lo vio | Estado |
|---|---|---|
| `empty_arena`: en vivo se fuerza a False y en el entrenamiento de 2022-23 es NaN | Codex | Eliminado de todos los modelos nuevos |
| El histórico "T-90" de Kalshi no tiene timestamp acreditado (mezcla último trade y cierre) | Codex | Solo se usa como diagnóstico |
| El replay de septiembre dimensiona con resultados de la misma cartelera (lookahead) | Codex | El estudio nuevo fija el capital por cartelera |
| La definición de debutante con contadores v2 falla porque incluye combates externos | Codex | Se usan apariciones UFC previas |
| Columnas de metadatos entraban como features por la lista de exclusión | Codex | Corregido, con test |
| GLM con winsorización asimétrica, que rompía la simetría de esquinas | Claude, con test | Corregido |
| Relleno de cuotas con la mediana de cuotas americanas (precio inválido) | Codex | Se usa un par real de una casa, por prioridad fija |
| Emparejamientos BFO por inicial: Junior y Justin Tafa confundidos | Codex | Solo se aceptan emparejamientos exactos |
| Orientación de los snapshots de Kalshi: siguen el ticker de Kalshi, no la esquina UFC | Claude | Corregido; el runner en vivo era correcto |
| El "cierre" de Kalshi en BFO se registra **después** del resultado | Subagente BFO | Nunca se usa como precio |
| ~25% de las capturas archivadas tienen un timestamp en o tras el inicio *estimado* de la pelea. UFC.com solo publica el inicio por sección y el runner estima sección + 30 min × índice | Claude y Codex | No demuestra precios in-play (no hay hora real de inicio), pero el disparador no se comporta como un "T-90"; conviene revisarlo |

## 2. Qué se probó

### Protocolo (congelado en `artifacts/r2026_10/manifest.json` antes de comparar)

- **Desarrollo:** años 2016-2023, con reentrenamiento anual. Toda la selección se hizo aquí.
- **Confirmación:** 2024-2026, mirada una sola vez después de congelar el finalista. **No es un holdout virgen**: ese periodo ya se había usado en el test histórico, en las auditorías de septiembre y en E0. Es una confirmación retrospectiva de una receta congelada hoy.
- **G1:** diferencia pareada de log-loss frente al mercado sin calibrar, con bootstrap por cartelera. Exige que el extremo superior del IC sea menor que 0.
- **G2:** PnL plano por oportunidad elegible (sin debutantes) a la cuota ofrecida con vig, apostando cuando EV > 2%.
- Un candidato debe pasar las dos puertas. Se usaron 10 semillas por candidato y se guarda cada una.

### Familias evaluadas

- **Controles:** la receta v2 reentrenada (A0) y su versión con T=1,25 (A0T, también con la regla desplegada "p − precio ≥ 3c").
- **Corrección FLB pura:** σ(a·logit q), con *a* ajustado solo con años anteriores.
- **Modelo residual anclado al mercado:** logit p = a·logit q + f(x), con un CatBoost pequeño (300 árboles, profundidad 3, l2 = 20) cuya línea base es el mercado ya corregido por FLB.
- **Variables nuevas:**
  - Props de método de cierre (KO, sumisión, decisión).
  - B1 (Codex): tasas suavizadas de ataque y defensa con exposición.
  - B2 (Codex): ajuste por rival con Poisson ridge, reajustado cada mes solo con el pasado.
  - Movimiento de línea entre apertura y cierre (BFO).

## 3. Resultados

### Desarrollo, 2018-2023 (soporte común: 2.812 peleas; 2.229 oportunidades sin debutantes)

| Candidato | Δ log-loss frente al mercado [IC95] | EV>2%: apuestas | ROI [IC95] | PnL/oportunidad |
|---|---|---:|---|---:|
| A0T (control desplegado) | +0,122 | 2.024 | −3,2% | −2,9% |
| Corrección FLB pura | −0,0012 [−0,0021; −0,0003] | 3 | — | ≈0 (el vig de casa lo absorbe) |
| R3: residuo anclado | −0,0026 [−0,0052; +0,00001] | 632 | +10,8% [+2,6; +18,9] | +3,1% |
| **F1 = R6_mkt** (R3 + props de método) | **−0,0028 [−0,0055; −0,0001]** | 662 | **+12,2% [+3,5; +20,6]** | **+3,6%** |
| F1 + B1 (Codex) | −0,0039 [−0,0067; −0,0010] | 771 | +10,3% [+2,2; +18,0] | +3,5% |
| F1 + B1 + B2 (Codex) | −0,0036 [−0,0067; −0,0006] | 867 | +8,4% [+1,5; +15,5] | +3,3% |

- **Robustez de R3.** Fue positivo en los 6 años y en las 10 semillas. En los 6 vecinos de hiperparámetros probados (número de árboles, profundidad, l2), el Δll quedó entre −0,0022 y −0,0027 y el PnL fue positivo en todos.
- **B1 y B2 no demostraron mejora frente a F1.** La diferencia pareada fue de −0,0011 [−0,0024; +0,0003] y −0,0008 [−0,0028; +0,0011]. Por la regla fijada de antemano, F2 se omitió.
- **Movimiento de línea entre apertura y cierre:** no aporta nada sobre el cierre (coeficiente 0,035; Δ +0,001).

### Confirmación 2024-2026 (mirada única; 1.367 peleas, 1.115 oportunidades sin debutantes)

| Modelo | Δ log-loss [IC95] | EV>2%: PnL/oportunidad | ROI [IC95] |
|---|---|---:|---|
| **F1 (congelado)** | −0,0043 [−0,0087; **+0,0002**] | +0,6% | +2,1% [−6,8; +10,6] |
| R3 (ablación) | −0,0041 [−0,0086; +0,0004] | +0,7% | +2,5% |
| F1 + B1 (ablación) | −0,0054 [−0,0103; −0,0003] | +1,4% | +4,0% |
| F1 + B1 + B2 (ablación) | −0,0060 [−0,0110; −0,0007] | +1,8% | +4,6% [−2,9; +12,5] |
| A0 (control) | +0,074 | −6,9% | −7,9% |
| A0T (control desplegado) | +0,091 | −5,2% | −5,9% |

**F1 no queda confirmado.** El extremo superior de G1 es +0,0002. Su estimación de Δ log-loss es favorable y su PnL estimado es positivo, pero el IC incluye pérdidas. Eso **no** acredita rentabilidad. Las ablaciones con B1/B2 pasan G1 en este periodo, pero no reemplazan a F1: su superioridad frente a F1 exige diferencias pareadas, y estas no son significativas (B1−F1: −0,0011 [−0,0028; +0,0007]; B1B2−F1: −0,0017 [−0,0040; +0,0006]).

Detalles de este bloque:
- Las 371 filas añadidas con cuotas BFO no tienen props de método, porque esa variable procede del export de Kaggle.
- Los resultados se calcularon sobre el `bfo_matched.parquet` vigente en la congelación. El archivo regenerado después da una q idéntica en esas 1.367 peleas y solo añadiría 7.

### Tamaño de apuesta: ¼-Kelly, tope del 2% por pelea y 15% por cartelera, capital fijo por cartelera

| | 2018-2023 (dev) | 2024-2026 |
|---|---|---|
| F1 | ×2,46 (DD máx. 14%) | ×1,03 (DD 14%) |
| F1 + B1 + B2 | — | ×1,15 (DD 22%) |
| Sistema desplegado (A0T), mismos topes | ×1,09 (DD 34%) | ×0,58 (DD 53%) |
| Sistema desplegado, ¼-Kelly sin topes | **ruina** | **ruina** |

### Diagnóstico en Kalshi (capturas en vivo jun-oct 2026: muestra pequeña, no es un backtest ejecutable)

Hay 13 carteleras. En el universo común sin debutantes (59 peleas), con EV > 2%:

| Modelo | Anclado a casas | Anclado al precio de Kalshi |
|---|---:|---:|
| Modelo antiguo | −3,1% por oportunidad | — |
| F1 | +4,2% | +9,1% |

Los IC son muy anchos. Lo único que se repite en todos los cortes es la dirección: el modelo antiguo apuesta a underdogs y pierde, mientras el anclado apuesta mayoritariamente a favoritos.

## 4. Qué recomendamos y qué no

- **Recomendamos:** parar el uso del sistema actual con dinero real. No se ha tocado nada; la decisión es tuya.
- **Recomendamos:** una sombra de papel prospectiva con un protocolo nuevo, fijado **desde ahora** (antes del primer dato):
  - **Co-candidatas:** F1 y F1+B1+B2. Ninguna ablación se promoverá a posteriori mirando resultados.
  - **Controles:** el precio contemporáneo sin calibrar y la corrección FLB (MF).
  - **Registro:** todas las oportunidades y todas las abstenciones con su motivo, con timestamps completos y la procedencia y frescura de la historia base y de B1/B2.
  - **Métrica primaria:** Δ log-loss frente al precio contemporáneo. Las señales son hipotéticas. Si se quiere PnL de papel, hay que **recapturar Kalshi en el momento de decidir**: un ask anterior no acredita la ejecución después de ver BFO.
  - **Horizonte fijo:** primera revisión al alcanzar 12 carteleras completas y 150 peleas elegibles, o el 2027-03-31, lo que ocurra antes. Es un suelo de revisión, no una garantía de potencia. Un resultado inconcluso sigue siendo inconcluso.
  - **Estado:** el **logger de F1 ya está disponible**; la **comparación prospectiva de ambas sigue pendiente**. Para F1+B1+B2 hacen falta las 34 columnas B1/B2 de cada pelea futura (`codex_future_performance.py --as-of <fecha>`), calculadas sobre un snapshot de UFCStats actualizado y versionado. El snapshot actual termina el 2026-09-12.
- **Recomendamos:** revisar el disparador del runner. Una de cada cuatro capturas llega en o después del inicio estimado.
- **No recomendamos:**
  - Dinero real con ningún candidato.
  - Volver a T=1,25.
  - Kelly sin topes.
  - Escoger entre ablaciones a la vista de 2024-2026.
- **Desacuerdo de método, abierto para ti.** Claude cree que la métrica primaria de selección debería ser la información incremental sobre el mercado (Δ log-loss), con el ROI como puerta. Codex prefiere mantener el ROI como criterio económico, según el CLAUDE.md, y acepta ambas puertas. En este bloque se usaron las dos, así que el desacuerdo no cambió ningún resultado. Sí afecta a cómo se redacte la regla §2 del CLAUDE.md en el futuro.

## 5. Herramientas entregadas

- `src/ufc_pred/research/harness.py` y `anchored.py`: el harness walk-forward, las puertas y los modelos anclados (con tests en `tests/research/`).
- `src/ufc_pred/research/bfo.py`: scraper educado y con caché de BestFightOdds. Cubre 804 eventos, el 96% de las peleas emparejadas, con aperturas desde 2016 y cierres por casa desde 2021. **No** decodifica el endpoint ofuscado de gráficos.
- `src/ufc_pred/research/perf_features.py`, `perf_adjusted.py` y `perf_future.py` (Codex): variables B1/B2, históricas y para peleas futuras, con una fecha as-of explícita. Hay 112 tests y la paridad del replay es exacta.
- `src/ufc_pred/research/anchored_shadow.py`: **logger en papel de F1** (probabilidades contemporáneas y señales hipotéticas; nunca ejecuta órdenes).
  - **Entrada:** las capturas de la sombra v2 (variables y snapshot de Kalshi). Descarga la página de BFO en el momento: cuotas de casas y props de método, solo de casas, identificadas por los códigos explícitos de `data-li`.
  - **Abstenciones:** falta un timestamp; la captura es futura o está en o tras el inicio estimado; la captura es antigua; BFO no es contemporáneo (desfase medido *después* de la descarga, 10 min por defecto); la cartelera es anterior a la fecha de ajuste del bundle; hay un debutante; falta precio o una probabilidad no es finita.
  - **Trazabilidad:** cada registro guarda el hash de la captura, del runner y del manifiesto del bundle, más una copia inmutable del HTML de BFO usado.
  - **Tope por cartelera:** el 15% se aplica a posteriori en `report()` como escalado *descriptivo*. No es sizing online con efectivo reservado.
  - **Bundle:** `artifacts/r2026_10/anchored_shadow_bundle_v1/`, ajustado con todo lo anterior al 2026-10-13. Nunca se sobrescribe, y al cargarlo se verifican modelos, datos de ajuste, política y código del modelo. El v0 se conserva solo para trazabilidad.
  - **Tests:** las puertas tienen tests sintéticos en `tests/research/test_anchored_shadow_gates.py`.
  - **No está conectado a launchd.** Para usarlo hay que ejecutar `score` durante la cartelera, con capturas previas al inicio.

## Reproducción

```bash
PY="/Users/jaimebernal/Desktop/pred_project/UFC pred/.conda/bin/python"
PYTHONPATH=src $PY scripts/research/r2026_10/e0_walkforward_vs_market.py --years 2013-2026 --seeds 5
PYTHONPATH=src $PY scripts/research/r2026_10/claude_dev_run.py --cands R3 --seeds 10
PYTHONPATH=src $PY scripts/research/r2026_10/claude_dev_run.py --cands R6 --seeds 10 --perf artifacts/r2026_10/claude/mkt_extra.parquet --tag _mkt
PYTHONPATH=src $PY scripts/research/r2026_10/claude_dev_report.py
bash scripts/research/r2026_10/claude_run_holdout.sh   # confirmación (ya ejecutada una vez)
PYTHONPATH=src $PY scripts/research/r2026_10/claude_holdout_report.py --names R6_mkt,R3,MF,R6_mkt_B1v2,R6_mkt_B1v2B2
PYTHONPATH=src $PY -m pytest -q tests/research
```

El manifiesto con hashes, la congelación, las revisiones posteriores y sus motivos está en `artifacts/r2026_10/manifest.json`. Las salidas están en `artifacts/r2026_10/`.

Fuentes externas:
- [Bürgi, Deng y Whelan, UCD WP25/19](https://www.ucd.ie/economics/t4media/WP2025_19.pdf)
- [CEPR DP20631](https://cepr.org/publications/dp20631)
- [Tarifas de Kalshi](https://kalshi.com/docs/kalshi-fee-schedule.pdf)
- [Robots de BestFightOdds](https://www.bestfightodds.com/robots.txt)
