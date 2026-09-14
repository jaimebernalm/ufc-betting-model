# Auditoría de simulación frente a operación real

Fecha: 14 de septiembre de 2026. Revisión del checkout local basado en `56ae9c9`, incluyendo los cambios locales que ya existían al empezar.

## Conclusión

**Hay bugs reales, pero no hay evidencia de un único bug cuya corrección convierta las pérdidas en la rentabilidad prometida por los backtests.** Hay tres problemas distintos:

1. Los backtests, el entrenamiento actual y los artefactos servidos no representan siempre el mismo sistema. Cambian las variables, los modelos, el universo apostable, el umbral, las comisiones y la ejecución.
2. Persisten defectos de ingeniería: rankings obsoletos, identidades mezcladas, contabilidad de recomendaciones como si fueran ejecuciones, un doble cambio de signo en entrenamiento y fallos de captura. He reproducido también errores latentes en sizing y liquidación.
3. La evidencia favorable está seleccionada retrospectivamente y no acredita una ventaja estable. Incluso los resultados de la auditoría previa con variables reconstruidas y modelos reentrenados contienen un periodo posterior claramente perdedor.

La premisa «en teoría debería ser lo mismo» solo sería defendible para **el mismo combate, las mismas entradas disponibles entonces, el mismo artefacto, la misma política y una ejecución realizable equivalente**. Tampoco esa igualdad garantiza que dos periodos diferentes tengan el mismo ROI.

No he sustituido modelos, modificado la política de apuestas, actualizado saldos ni activado/desactivado el watchdog. Los únicos archivos nuevos de esta revisión son el informe, el script de auditoría y sus resultados. Las pruebas del ledger usan carpetas temporales y clientes ficticios.

## Alcance y evidencia

Inventarié 68 archivos Python bajo `src`, 61 bajo `scripts` —incluido el nuevo auditor—, 14 notebooks y 11 archivos Python de pruebas. Revisé las rutas de ingestión, variables, entrenamiento, calibración, evaluación, inferencia, sizing, notificaciones, fills y liquidación, además de las decisiones en README, STATUS, DEPLOY y auditorías anteriores. El inventario incluye funciones y títulos de los notebooks; no representa haber ejecutado todas las variantes históricas.

Trabajo ejecutado:

- Suite existente: **81 pruebas pasan**.
- Reproducciones aisladas de invariantes rotos con el código real.
- Inspección de la matriz completa de entrenamiento.
- Dos entrenamientos diagnósticos nuevos, misma semilla 0 y 2.000 árboles, cambiando únicamente el doble signo; modelos mantenidos en memoria.
- Comparaciones con los 20 artefactos desplegados sobre 224 combates posteriores al cutoff.
- Sensibilidad de rankings sobre 44 filas de oportunidades live archivadas.
- Recálculo de resultados de la auditoría estricta desde sus predicciones persistidas.
- Reconstrucción de las 66 ejecuciones archivadas, agrupadas en 62 apuestas, y comparación con recomendaciones.
- Prueba del parser actual sobre el HTML real archivado del 12 de septiembre.

**Limitación:** el directorio externo de CSV usado por la auditoría estricta de agosto (`/tmp/ufc-audit.DVTvfN/repo`) ya no existe. No he repetido ese reentrenamiento completo ni los centenares de ajustes NUTS históricos. Distingo sus resultados guardados de mis experimentos nuevos. Tampoco reconstruyo una profundidad histórica del order book que no fue archivada.

Resultados y reproducción: [script de auditoría][script], [resultados principales][core], [operación y ejecuciones][ops], [revisión del test histórico][headline], [inventario][inventory].

## 1. Rankings: tres defectos de producción confirmados

Código: [rankings_attach.py][ranks].

**Caché sin caducidad.** `load_rankings()` llama a `download_rankings()` con `force=False`; si el archivo existe, nunca lo renueva. El CSV local termina el **19 de mayo de 2026**: tiene **118 días de antigüedad** al hacer esta revisión. Actualizar combates no refresca ese CSV. La guardia de frescura del runner comprueba el historial de combates, no los rankings.

**Los peleadores nunca abandonan el ranking.** `_rank_at()` busca la última fecha en la que *ese peleador apareció*, no su presencia en la última foto de la división. Una exclusión posterior no elimina su puesto. Ejemplos reproducidos para el 20 de mayo de 2026:

| Peleador/división | Valor que devuelve el código | Fecha de su última aparición |
|---|---:|---|
| Tony Ferguson, Lightweight | 15 | 2022-12-13 |
| Conor McGregor, Featherweight | 0, campeón | 2016-11-21 |
| Dominick Cruz, Bantamweight | 14 | 2024-10-29 |
| Henry Cejudo, Flyweight | 0, campeón | 2020-02-24 |

**Pound-for-Pound usa un nombre de categoría antiguo.** El código pide `Pound-for-Pound`, cuya serie termina en **2020-01-20**. El archivo continúa con `Men's Pound-for-Pound` y `Women's Pound-for-Pound`. Se conservan puestos anteriores a 2020 o se producen ausencias incorrectas.

**¿Intencional?** No encontré documentación que defina estas variables como «último ranking alcanzado en toda la carrera». La documentación las presenta como rankings previos al combate y el runbook promete rankings actuales. La implementación incumple esa semántica. Conservar un campeón de 2016 no es simplemente aceptar el retraso de una actualización semanal.

**Impacto medido, no promesa de mejora.** Cambié solo las columnas de ranking en las 44 filas archivadas, usando la última foto disponible por división, separación P4P por género y normalización de espacios. Mantuve los artefactos antiguos, las demás variables, la simetrización, T=1,25 y las cuotas:

| Ensemble | Cambio absoluto medio de probabilidad | Máximo | ROI antes | ROI después |
|---|---:|---:|---:|---:|
| Real | 4,98 puntos | 41,20 puntos | −24,40% | −32,10% |
| Corrupted | 4,43 puntos | 41,06 puntos | −25,81% | −30,56% |

Es una **sensibilidad con modelos antiguos**, no una validación de modelos reentrenados ni una reconstrucción de rankings de junio/julio que faltan en el CSV. Demuestra que el defecto es material, pero corregirlo aisladamente **no rescata ese periodo**. El capital Kelly mejora en este ensayo mientras el ROI plano empeora: son métricas distintas y no deben intercambiarse.

Corrección propuesta: historial de snapshots completos con IDs de peleador/división, ausencia explícita de quien salió del ranking, actualización verificable y el mismo constructor para train y serve.

## 2. Dos Bruno Silva se convierten en una sola persona

Código: [índice de habilidad][skillindex], [resolución de perfiles][state].

El histórico contiene **23 filas** con el nombre `Bruno Silva`: 11 de peso medio, 11 de peso mosca y una de peso gallo. Las primeras usan altura 182,88 cm; las del peleador pequeño, 162,56 cm. Son dos carreras diferentes. UFC documenta tanto al [Bruno Silva de peso mosca](https://www.ufc.com/athlete/bruno-silva) como al [Bruno Silva que compitió en peso medio contra Alex Pereira](https://www.ufc.com/news/bruno-silva-fears-no-challenge-middleweight-ufc-san-diego).

`build_index()` crea **un solo ID** para ambos nombres, mezclando sus victorias, derrotas y rivales en el posterior. Los contadores de apariciones y el fallback de «último combate del peleador» usan también el nombre como identidad. No es un cambio de división de una sola persona.

En la ruta raw, `_index_for_char()` guarda `nombre_normalizado → URL`. Una segunda fila con el mismo nombre sobrescribe silenciosamente la primera. Lo reproduje con dos perfiles ficticios del mismo nombre; **no he afirmado cuál de los dos perfiles devuelve hoy el sitio real**, porque no inspeccioné ese orden en la web de UFCStats.

**¿Intencional?** No hay regla de desambiguación que justifique fusionarlos. Es un defecto de identidad. La normalización de `Su Mudaerji` a `Sumudaerji` que ya estaba en tus cambios locales corrige el problema inverso —una persona dividida en dos nombres— y no arregla esta colisión.

Afecta de forma confirmada a las variables de habilidad; puede seleccionar el perfil raw incorrecto. No he cuantificado su contribución causal al PnL global. Solución: usar el ID/URL estable del peleador como clave, resolver nombres solo en los límites con proveedores y rechazar ambigüedades.

## 3. El ledger sigue contabilizando recomendaciones, no fills

Código: [ops/bankroll.py][bankroll], [ops/fills.py][fills].

`sync()` recorre `bet_notifications`, toma las participaciones y el importe recomendados y aplica su PnL cuando identifica un ganador. **No consulta el archivo de ejecuciones.** `save_fills()` captura operaciones reales, pero ese dato no alimenta la contabilidad que usa Kelly.

Consecuencia: una apuesta omitida, parcial o ajustada manualmente modifica el capital virtual como si se hubiera ejecutado tal cual. El error se transmite al tamaño de las siguientes apuestas.

Esto **ya estaba identificado**, no es un descubrimiento nuevo: la nota de `configs/bankrolls.json` documenta el ajuste manual del 8 de agosto y cita expresamente este root cause. Hay tres recomendaciones del 25 de julio sin fills archivados. Encontré otra del 15 de agosto sin fill asociado; la ausencia en el archivo por sí sola no demuestra que jamás se ejecutara, pero sí impide contabilizarla como ejecución confirmada.

Prueba aislada: sin crear ningún fill, una recomendación ganadora incrementa la cuenta A en **16,38 dólares**. La prueba solo escribe un ledger temporal.

**¿Intencional?** Las cuentas virtuales A/B/C y el seguimiento provisional son deliberados. Pero el repo documenta su reconciliación con el dinero real, y el propio archivo de saldos reconoce el problema. No es correcto tratar ese saldo como capital ejecutable si no incorpora fills, fees efectivas y operaciones manuales.

### Tres errores adicionales del ledger

- **Dos lados:** el sizing permite que A/B apuesten por un peleador y C por el otro, pero `sync()` procesa solo `next(iter(orders.items()))`. Prueba: C debe bajar de 300 a **256,09**, pero permanece en **300**. No encontré notificaciones de dos lados entre las 87 archivadas: es un fallo latente, no una explicación demostrada de las pérdidas ya observadas.
- **Dry-run destructivo para la evidencia:** [write_record()][write_record] escribe siempre el mismo nombre. `process_one()` omite la deduplicación para dry-run, así que una simulación posterior puede sobrescribir el registro real. Lo reproduje en temporal. Hay tres mercados efectivamente comprados cuyo único registro guardado dice `dry_run=True`; eso es compatible con la sobrescritura, aunque el archivo aislado no prueba cómo ocurrió.
- **Cambio de jornada:** el runner sustituye `card_state` antes de reconciliar la jornada anterior. Si quedaron resoluciones pendientes cuando el ordenador dejó de ejecutar el proceso, el nuevo estado puede perderlas. Es un riesgo del flujo comprobado por lectura; no he atribuido una cantidad histórica concreta a este caso.

Corrección propuesta: un ledger de eventos de ejecución/liquidación, con IDs idempotentes por fill, atribución explícita entre cuentas y reconciliación independiente del día actual. Separar archivos reales y simulados.

## 4. Doble inversión de signo en el entrenamiento de la escalera

Código: [prepare / swap][swap], [harness][harness], [flip_signed_columns][flip].

`prepare(augment_symmetry=True)` ya niega `skill_diff_mean` en la mitad con esquinas intercambiadas. Después, `build_matrices()` llama a `flip_signed_columns()` y vuelve a negarla.

Ejemplo exacto: `+2` en el combate original pasa a `−2` tras el intercambio correcto y vuelve a `+2` al entrar en el harness. La etiqueta sí queda invertida.

En la matriz real de v3 hay **5.456 filas originales con skill no nulo ni cero**, y sus **5.456 copias intercambiadas tienen el signo incorrecto**. Afecta también a las recetas con medias de habilidad sufijadas y a varios scripts que conservan la inversión manual.

**¿Intencional?** No. La documentación dice que esas columnas deben negarse una vez. Además, [train_corrupted()][strategy] explica expresamente que se eliminó su inversión manual para evitar este mismo doble cambio desde junio. Esa eliminación no se aplicó al harness y otros consumidores. Las pruebas existentes verifican cada función por separado con una matriz artificial que aún no ha sido invertida; por eso pasan.

**No confundir con el modelo desplegado:** el runner carga los archivos `v3_real_2025_11_30_seed*.joblib` y `v3_corrupted_...`, creados el **25 de mayo de 2026**. No carga el artefacto `v7_1_catboost_ensemble_v3.joblib` de la escalera. El bug actual del harness invalida reproducciones/comparaciones y futuros entrenamientos por esa ruta; no demuestra que causara por sí mismo las pérdidas de los artefactos antiguos.

### A/B nuevo, aislando únicamente este signo

Misma semilla 0, mismos datos hasta 2022, misma validación 2023, 2.000 árboles y sin early stopping. ROI diagnóstico contra precios de sportsbook sin vig y comisión cuadrática 0,07; no son asks históricos reales.

| Matriz | Log loss validación | Apuestas | ROI plano | IC95 del ROI |
|---|---:|---:|---:|---|
| Doble inversión actual | 0,64203 | 356 | +1,27% | [−9,67%, +12,67%] |
| Una inversión correcta | 0,64291 | 345 | +0,001% | [−11,66%, +11,94%] |

**Es un bug aunque arreglarlo no mejore el ROI en esta semilla.** El efecto económico favorable no queda probado, y ambas bandas incluyen cero. No elegí la variante por estos resultados ni cambié el deployment.

## 5. El sizing no vuelve a comprobar la ventaja al consumir el libro

Código: [selección y Kelly][sizingpick], [construcción de recomendación][sizingfill].

La selección y Kelly usan el mejor ask. Después se recorre el libro y se obtiene un precio medio superior, pero no se recalculan Kelly, la elegibilidad ni el precio máximo rentable.

Reproducción con el código real:

- Modelo: probabilidad **53,1%**.
- Libro: una participación a **50¢**, el resto a **53¢**.
- Bankroll: 1.000 dólares; Kelly 25%.
- Devuelve `BET`, stake **6,99 dólares**, límite **53¢** y precio medio **52,77¢**.
- Sumando la comisión cuadrática, el ROI esperado según el propio modelo es **−2,59%**.

No es un debate sobre acertar el combate: la recomendación contradice su propio criterio de valor. El cap del 5% sobre profundidad a tres céntimos no lo evita cuando el primer nivel tiene poca cantidad.

**¿Intencional?** Caminar conservadoramente el libro está documentado; aceptar una ejecución con EV negativo no. La corrección debe limitar precios y cantidades usando el coste marginal completo y volver a decidir después del redondeo de cantidad/fees.

**Impacto observado:** no encontré EV negativo al precio medio recomendado entre las órdenes archivadas con probabilidades disponibles. Es un bug demostrado mediante contrajemplo, pero no está probado como causa del PnL pasado.

Además, una cuenta con bankroll cero y señal positiva no entra en `scaled_stakes`, pero posteriormente se accede a ella: **`KeyError(0)`**. Puede abortar las recomendaciones del combate para las otras cuentas. El runbook pide detener solo la cuenta arruinada; el código no implementa correctamente ese comportamiento.

## 6. El sistema puede no apostar combates que el replay sí incluye

Código: [parser de calendario][schedule], [replay de jornadas perdidas][replay].

El HTML archivado de **2026-09-12** contiene **13 combates**, bajo `main-card--2` y `prelims-card--2`. `fetch_card_schedule()` busca únicamente los IDs sin sufijo y devuelve **cero combates** al leer ese mismo HTML. Lo ejecuté con `_fetch_html` sustituido por la lectura local, sin red ni notificaciones.

Tu script local `replay_missed_cards_kalshi.py` ya admite los sufijos y describe el fallo; el parser usado por el runner todavía no. Por tanto, es **un problema previamente señalado en el trabajo local, confirmado y pendiente de trasladar a producción**.

Otros fallos de nombres/transitorios de agosto ya tienen correcciones locales o commits específicos. No los presento como bugs nuevos ni los borro. Sí implican que una simulación retrospectiva con aliases corregidos, historial ya actualizado y ejecución continua no reproduce la disponibilidad del sistema en aquella noche.

El replay de jornadas perdidas añade otras aproximaciones: profundidad fija de 100.000 participaciones; captura al cierre del mercado anterior sin replicar exactamente la detección live y su gracia; saldo liquidado inmediatamente; ausencia de fallos de red y disponibilidad del ordenador. Es un contrafactual útil **condicionado a esas hipótesis**, no una reproducción idéntica de la operación real.

## 7. Los protocolos llamados «deployment» no son intercambiables

### Artefactos y variables

Los 20 modelos activos se entrenaron en mayo, antes de las reparaciones de variables raw y de simetría. El constructor live sí fue actualizado. La auditoría estricta de agosto reentrenó modelos para investigar el problema, pero sus resultados no hacen que el loader actual use automáticamente esos modelos.

Esto ya está reconocido en [FEATURE_PARITY_AUDIT.md][prioraudit], incluida tu decisión explícita del 6 de agosto de continuar con las cuentas y artefactos existentes. **Respeto esa decisión; no considero accidental lo que aceptaste expresamente.** Sigue siendo una limitación para interpretar un backtest de otros modelos como evidencia del deployment activo.

### Dos significados diferentes de «3%»

[strategy_grid.simulate_kelly()][strategy] exige `p × cuota_efectiva − 1 > 0.03`: retorno esperado por dólar. Producción exige `p − ask >= 0.03`: tres puntos de probabilidad, antes de fees. Son reglas diferentes.

Por ejemplo, `p=0,22`, ask `0,20` y fee cuadrática 0,07 dan EV de aproximadamente **4,17%**, pero solo **2 puntos** de diferencia de probabilidad. Pasa el primer filtro y no el segundo. En mi comparación de 224 combates, cambiar únicamente el filtro con las mismas probabilidades y fees modifica 6 decisiones del ensemble real y una del corrupted.

El filtro live está documentado en la auditoría anterior, así que es una elección conocida. El error es afirmar que coincide con el EV threshold original; incluso un comentario en sizing lo llama equivalente.

### Una escalera de simulaciones sobre los mismos datos

Ensemble real antiguo, 224 combates posteriores a 2025-11-30 y hasta 2026-05-16, cuotas Polymarket archivadas, Kelly 25%, capital inicial 1. **Filas históricas registradas, sin corrección raw: estas cifras no validan rentabilidad operable.** La comisión Kalshi aplicada a cuotas Polymarket es una sensibilidad controlada, no una simulación de órdenes Kalshi reales.

| Cambio acumulativo | Apuestas | ROI plano | Capital final / inicial |
|---|---:|---:|---:|
| Orientación original, EV >3%, fee 2% sobre ganancias | 198 | +15,18% | 84,21× |
| Simetrizar las predicciones | 208 | +14,88% | 139,87× |
| Añadir sharpening T=1,25 | 210 | +17,09% | 403,96× |
| Cambiar a tres puntos de probabilidad | 200 | +21,37% | 415,99× |
| Fee cuadrática Kalshi 0,07 | 200 | +18,57% | 231,99× |
| Excluir debutantes | 157 | +15,28% | 21,99× |
| Dimensionar al inicio de cada jornada, exposición total ≤100% | 157 | +15,28% | 17,88× |

Se ve por qué pequeños cambios pueden multiplicar el resultado Kelly. La tabla no atribuye porcentajes independientes de causalidad: los efectos dependen del orden de los cambios. El resultado incluye ejecución ideal a una cuota y no acredita profundidad suficiente. El drawdown secuencial y el medido al cierre de cada jornada tampoco son directamente comparables.

### Comisiones y bugs en evaluadores secundarios

- El harness, `test_set_evaluation.py` y varios scripts usan por defecto `fee_model="winnings"`, aunque textos públicos llaman a sus números «real Kalshi fees».
- El evaluador [venue_priced_evaluation.py][venuefee] calcula el retorno ganador como `(1-price-fee)/price` y la pérdida como `−1`. Mezcla bases de capital y omite la fee en el lado perdedor si la base es el nominal. Para una victoria y una derrota a 50¢ devuelve ROI **−1,75%**, frente a **−3,38%** sobre coste total correcto.
- El `swap_corners()` de ese mismo script intercambia las variables laterales, pero no cambia `better_rank=Red` a `Blue`. No es equivalente al swap actual de producción.
- Estos errores del evaluador legacy **no invalidan automáticamente** el evaluador estricto multivenue, que tiene otra ruta y cuya aritmética recalcule por separado.

Verifiqué la comisión general contra el [documento oficial de Kalshi vigente desde julio de 2026](https://kalshi.com/docs/kalshi-fee-schedule.pdf): fórmula cuadrática, multiplicador por contrato y redondeo de coste más fee. La conversión de NO bids a YES asks del adaptador coincide con la [documentación oficial del order book](https://docs.kalshi.com/getting_started/orderbook_responses). **No encontré una inversión global YES/NO en ese adaptador.**

**Contraprueba importante:** corregir solamente la comisión del test antiguo casi no cambia su ROI. Sobre el mismo artefacto, el universo completo pasa de **+10,875% a +10,947%**; sin debutantes, de **+7,786% a +7,793%**, con IC95 que sigue incluyendo cero. Esa discrepancia de fees no explica el salto a pérdidas live.

## 8. El índice Bayesiano sí incorpora información posterior

Código: [build_index()][skillindex], [posterior/cache][skillcache].

El comentario dice que construir el índice con todo el histórico solo numera peleadores y no filtra información futura. Sin embargo, también calcula **la división modal de cada peleador usando todo el histórico**, y esa división determina su prior jerárquico.

Al comparar el índice con datos anteriores al cutoff 2025-11-30 frente al construido con todo el dataset actual, cambian **17 asignaciones de división modal**. Entre ellas están Chase Hooper, Chidi Njokuani y Michel Pereira; Bruno Silva aparece también por la colisión anterior.

No se están filtrando directamente ganadores futuros en la likelihood mensual. Pero sí hay **información futura en una entrada del modelo**. Por tanto, la afirmación de independencia temporal completa es falsa. Construir IDs globales puede ser razonable; inferir su prior con carreras futuras es otra operación.

Además, el hash de caché incluye fecha, nombres y ganador, pero **no incluye `weight_class`**, el índice completo, la configuración NUTS ni la versión de la receta. Cambiar una división conserva exactamente la misma clave. Un replay puede reutilizar un posterior viejo mientras un ajuste forzado cambia el resultado.

**¿Intencional?** El índice global fue una decisión explícita. El problema es la consecuencia temporal de calcular allí la moda, que la justificación escrita no contempla. No medí el cambio de ROI tras regenerar todo NUTS: no presento este defecto como explicación cuantificada de las pérdidas.

En la variante v3.1 hay otra limitación: `skill_diff_for_targets()` lee el estado del último combate conocido sin propagar la incertidumbre hasta la fecha objetivo. La documentación atribuye al random walk protección frente a un largo descanso, pero la predicción no incorpora ese último intervalo. Afecta a una iteración experimental, no al v3 escalar actualmente desplegado.

## 9. Reproducibilidad rota por la reorganización de scripts

Encontré **31 candidatos** bajo `scripts/research` y `scripts/tools` que conservan `Path(__file__).resolve().parents[1]` o `.parent.parent` como raíz. Tras moverlos a esas subcarpetas, eso apunta a `repo/scripts`, no a `repo`.

Casos concretos: [entrenamiento del deployment][trainpath], `strict_raw_retrain_audit.py`, `strict_multivenue_postcutoff.py`, `experiment_extremeness.py`, `all_accounts_seed_grid.py` y generadores de notebooks. El entrenamiento de deployment intenta guardar en `scripts/artifacts/models`; las auditorías buscan `scripts/data/...`. Algunos fallan pronto; otros podrían entrenar antes de fallar al guardar.

**¿Intencional?** Es un defecto de rutas coherente con una reorganización, no una política del modelo. Las bibliotecas de producción que importan `ufc_pred.paths` no tienen este mismo problema. No ejecuté indiscriminadamente los scripts afectados porque algunos entrenan y escriben al importarse; comprobé sus expresiones y las rutas que producen.

Impide tratar un JSON antiguo como resultado regenerado por el código actual. Los modelos necesitan un manifiesto con hash de datos, esquema y constructor de variables, protocolo de simetría, versiones de librerías y configuración de inferencia.

## 10. Qué explica el dinero perdido y qué no

### Ejecuciones archivadas

Reconstruí coste como `participaciones × precio + fee_cost`; PnL como payout menos coste. Para junio/julio, usé las liquidaciones Kalshi archivadas. Para las 18 ejecuciones de agosto, resultados de combates locales unidos por fecha, nombres normalizados sin fuzzy matching y la orientación Kalshi guardada. **No consulté liquidaciones Kalshi actuales para agosto.** No quedó ninguna de las 66 ejecuciones sin asignar.

| Jornada | Apuestas | Ganadas | Capital apostado, fees incluidas | PnL | ROI |
|---|---:|---:|---:|---:|---:|
| 2026-06-06 | 9 | 3 | $575,21 | −$90,97 | −15,81% |
| 2026-06-14 | 7 | 4 | $342,70 | −$113,45 | −33,10% |
| 2026-06-20 | 9 | 4 | $380,20 | +$111,33 | +29,28% |
| 2026-06-27 | 9 | 5 | $614,78 | −$134,04 | −21,80% |
| 2026-07-11 | 10 | 2 | $633,95 | −$371,48 | −58,60% |
| 2026-08-08 | 9 | 6 | $307,60 | +$19,14 | +6,22% |
| 2026-08-15 | 9 | 4 | $167,47 | +$24,69 | +14,74% |
| **Total** | **62** | **28** | **$3.021,90** | **−$554,77** | **−18,36%** |

El capital apostado suma importes reinvertidos; **no es el depósito inicial ni el drawdown de la cuenta**. La fila total describe exclusivamente el archivo local de fills, no garantiza que ese archivo contenga toda operación histórica de la cuenta. No hay fills posteriores al 15 de agosto en ese archivo.

En **58 mercados** comparables, la diferencia absoluta media entre precio recomendado y realmente pagado fue **0,001105 dólares por contrato: 0,11¢**; máximo **1¢**. A igual número de contratos, las compras reales costaron **$1,93 menos** que al precio medio recomendado, antes de comparar las fees. Esto descarta que un gran slippage adverso sistemático explique las pérdidas de esa muestra. No descarta órdenes omitidas, tamaños diferentes ni falta de liquidez en otros momentos.

### La auditoría estricta previa sigue mostrando fracaso posterior

Recalculé desde `strict_multivenue_postcutoff_fights.parquet`:

| Ventana, sin debutantes | Real: apuestas / ganadas / ROI | Corrupted: apuestas / ganadas / ROI |
|---|---|---|
| Polymarket hasta 2026-05-16 | 156 / 97 / **+22,46%** | 152 / 83 / **+10,81%** |
| Kalshi hasta 2026-05-16 | 42 / 26 / **+38,99%** | 38 / 23 / **+36,39%** |
| Kalshi después de 2026-05-16, hasta 2026-08-01 | 60 / 24 / **−30,29%** | 58 / 21 / **−37,03%** |

Coinciden con el resultado multivenue guardado. El anterior reentrenamiento completo sobre las mismas 44 oportunidades live devolvió **−18,95% y −20,80%**, según su artefacto. Esa segunda cifra la he leído, **no vuelto a producir mediante reentrenamiento raw completo**.

Esto mantiene la conclusión principal de agosto: la reconstrucción incorrecta de variables era real y favorecía comparaciones históricas, pero corregirla no bastó para que el modelo seleccionara buenos resultados en aquel periodo live.

### La certeza estadística de README/docs es excesiva

El test original ya está gastado. Las decisiones posteriores sobre cutoffs congelados, variantes corrupted, sharpening, caps y universos reutilizan periodos que incluyen 2024–2026. Eso sirve para investigación, pero no convierte cada comparación en validación independiente.

Probar diez semillas sobre los mismos ganadores mide sensibilidad del entrenamiento; **no equivale a diez periodos independientes de mercado**. Un intervalo sobre semillas tampoco cubre la incertidumbre temporal ni la selección entre muchas estrategias. A/B comparten modelo y C está muy correlacionada: tres cuentas no son tres fuentes independientes de evidencia.

«El intervalo del periodo negativo excluye cero» acredita evidencia de mal rendimiento bajo los supuestos de ese intervalo; **no prueba por sí solo un cambio estructural permanente ni su causa**. El README afirma «not noise» y sugiere maduración del mercado, pero la propia auditoría detallada es más prudente: el escaneo de cambios de régimen ajustado por buscar fechas no fue significativo; tiempo, universo y fuente de precios están confundidos. Aumentar volumen tampoco demuestra por sí solo que desapareció una ineficiencia.

Tampoco se deduce una ley general de que log loss y beneficio sean incompatibles, ni que eliminar early stopping quite el sesgo de selección. Elegir árboles, sharpening o políticas por ROI sobre un periodo reutilizado también ajusta decisiones a ese periodo. Un modelo puede tener peor log loss global y ganar en un subconjunto; lo que falta es acreditar esa ventaja **fuera del conjunto donde se eligió la política**.

Las dos jornadas positivas de agosto son evidencia nueva limitada, no rehabilitación estadística. Del mismo modo, no hay fundamento para prometer que una racha positiva deba compensar las pérdidas anteriores.

## Qué era deliberado y no he tratado como bug

| Decisión | Clasificación |
|---|---|
| Cuenta C fuerza skill a NaN | Deliberada, reproducida explícitamente por `CorruptedSkillModel`; nació de un accidente, pero su conservación y deployment fueron conscientes. |
| Cutoff congelado 2025-11-30 | Deliberado; no confundir antigüedad del modelo con historial de resultados sin actualizar. |
| Simetrización en inferencia | Corrección intencional para artefactos antiguos no perfectamente simétricos. |
| Sharpening T=1,25 | Deliberado; necesita evidencia prospectiva, no se elimina por parecer sobreconfianza. |
| Kelly 10% en A y 25% en B/C | Política explícita. «Sin cap» en B/C no significa full Kelly: sigue siendo Kelly fraccionado. |
| Cuentas virtuales y ejecución humana | Deliberado. La falta de conexión entre fills y saldo usado para sizing es un defecto separado. |
| Exclusión de debutantes | Política reconocida; su implementación indirecta vía nombre es frágil, pero no la relajé. |
| Continuar con los modelos antiguos tras agosto | Decisión explícita del propietario, documentada. No la revertí. |

## Orden recomendado para reparar y volver a comparar

1. **Identidad y datos:** ID estable de peleador; snapshots correctos de rankings; eliminar el doble signo; prior de habilidad construido solo con información temporalmente disponible y caché versionada. Construir train y serve con la misma función.
2. **Contabilidad y ejecución:** ledger basado en fills, ambos lados y fees reales; estado de liquidación que sobreviva entre jornadas; bankroll cero; separación de dry-run; reevaluación del EV al precio ejecutable. Reparar el parser de calendario.
3. **Una única especificación de estrategia:** artefacto, variables, orientación, transformación de probabilidades, fórmula del threshold, fee, universo, orden temporal, saldo disponible y reglas de ejecución. Tanto el simulador como el runner deben consumir esa especificación.
4. **Reproducibilidad:** corregir rutas; congelar datos y manifiestos; ejecutar replay sobre snapshots inmutables y conservar por qué cada oportunidad se apostó, se omitió o falló. En ausencia de profundidad histórica, presentar límites/sensibilidades, no liquidez inventada como hecho.
5. **Evaluación nueva:** separar experimentos de un tramo posterior que no vuelva a usarse para elegir estrategias. Comparar también contra el mercado en las apuestas seleccionadas; medir calibración de probabilidades en ese subconjunto, ROI plano y crecimiento Kelly por separado.

No conviene retocar ahora Kelly o excluir underdogs porque mejore retrospectivamente estas mismas jornadas. Primero hay que conseguir una comparación trazable; después se podrá medir si queda una ventaja económica.

## Reproducción

Desde la raíz de este repo, con el entorno existente:

```bash
.conda/bin/python -m pytest -q
.conda/bin/python scripts/research/audit_sim_live_2026_09_14.py --fit-ablation
.conda/bin/python scripts/research/audit_sim_live_2026_09_14.py --operational-only
.conda/bin/python scripts/research/audit_sim_live_2026_09_14.py --headline-only
```

El script usa datos locales. Los dos modelos de la ablación no se guardan sobre los de producción. Las salidas nuevas quedan en `artifacts/audit_2026_09_14/`:

- `audit_results.json`: invariantes, matriz, protocolos, sensibilidad de rankings, resultados previos recalculados y A/B.
- `operational_results.json`: identidad, parser, errores del evaluador y resumen de fills.
- `actual_fills_recheck.csv`: coste y resultado por ejecución, con fuente del ganador.
- `recommendation_fill_comparison.csv`: diferencias entre recomendación y ejecución por mercado.
- `test_headline_fee_recheck.json`: cambio aislado de comisión del test histórico.
- `inventory.json`: archivos, funciones, títulos de notebooks y hashes del inventario inicial.

Los resultados son diagnósticos, no una nueva estrategia seleccionada ni una previsión de beneficios.

[script]: <../scripts/research/audit_sim_live_2026_09_14.py>
[core]: <../artifacts/audit_2026_09_14/audit_results.json>
[ops]: <../artifacts/audit_2026_09_14/operational_results.json>
[headline]: <../artifacts/audit_2026_09_14/test_headline_fee_recheck.json>
[inventory]: <../artifacts/audit_2026_09_14/inventory.json>
[ranks]: <../src/ufc_pred/ingest/rankings_attach.py>
[skillindex]: <../src/ufc_pred/features/skill_v3.py>
[state]: <../src/ufc_pred/ingest/ufcstats_state.py>
[bankroll]: <../src/ufc_pred/ops/bankroll.py>
[fills]: <../src/ufc_pred/ops/fills.py>
[write_record]: <../src/ufc_pred/cli/bet_runner.py>
[swap]: <../src/ufc_pred/features/static_v1.py>
[harness]: <../src/ufc_pred/models/_harness.py>
[flip]: <../src/ufc_pred/features/joins.py>
[strategy]: <../src/ufc_pred/backtest/strategy_grid.py>
[sizingpick]: <../src/ufc_pred/inference/sizing.py>
[sizingfill]: <../src/ufc_pred/inference/sizing.py>
[schedule]: <../src/ufc_pred/ingest/ufc_schedule.py>
[replay]: <../scripts/research/replay_missed_cards_kalshi.py>
[prioraudit]: <../FEATURE_PARITY_AUDIT.md>
[venuefee]: <../scripts/research/venue_priced_evaluation.py>
[skillcache]: <../src/ufc_pred/inference/skill_for_upcoming.py>
[trainpath]: <../scripts/tools/train_deployment_ensemble.py>


Nota de publicación: las referencias a `artifacts/`, datos, registros operativos y documentos privados corresponden a archivos locales excluidos de Git. Esta copia pública omite saldos personales y rutas específicas del equipo; los originales se conservan localmente.
