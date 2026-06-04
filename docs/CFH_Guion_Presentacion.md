# GUIÓN — CFH: Protocolo Matemático, Indicadores y Experimentos
**Mireya Camacho Celis · Defensa agosto 2026**
*45–60 min con preguntas*

---

## SLIDE 1 — PORTADA

"Buenos días. Voy a presentar el protocolo matemático de la CFH — el framework computacional para medir la injusticia discursiva y epistémica en el archivo judicial colombiano de los falsos positivos.

El proyecto tiene cuatro pilares teóricos, no uno. Y eso es deliberado, porque estoy midiendo un fenómeno que tiene cuatro dimensiones distintas.

**Habermas** en la Teoría de la Acción Comunicativa me da los criterios para evaluar el discurso judicial: un acto de habla válido requiere sinceridad expresiva, verdad proposicional y corrección normativa. El DIS Score mide las violaciones a esas tres pretensiones.

**Galtung** en *Cultural Violence* me da el concepto de violencia cultural — los aspectos del lenguaje que hacen que la violencia directa parezca natural o inevitable. El léxico militar-burocrático de los falsos positivos ('bajas operacionales', 'resultados') es Galtung lexicalizado.

**Fricker** en *Epistemic Injustice* me da los dos tipos de injusticia epistémica que quiero medir: la testimonial — cuando se niega credibilidad al testimonio — y la hermenéutica — cuando no existen los recursos conceptuales para articular la experiencia. El IEI Score las operacionaliza.

**Fraser** en *Scales of Justice* me da el concepto de paridad participativa y reconocimiento epistémico como condición de justicia. El REP Score y el ICM tri-canal miden si el discurso produce ese reconocimiento.

La H₃ central conecta todo: mayor injusticia discursiva — en el sentido de Habermas y Galtung — predice menor transición epistémica — en el sentido de Fricker y Fraser."

---

## SLIDE 2 — MARCO TEÓRICO CUATRI-PILAR

"Este slide muestra cómo cada teoría se traduce en computación.

**Habermas** → el DIS Score. Sus tres pretensiones de validez del discurso se convierten en tres indicadores: SA mide la violación de la sinceridad expresiva, NV mide la violación de la corrección normativa, y la ausencia de REP mide la violación de la rectitud comunicativa. El ICM también es habermasiano: la incongruencia entre lo que se dice con palabras, voz y cara viola la sinceridad expresiva en su forma más profunda.

**Galtung** → EBI y NV. El vocabulario militar que presenta el homicidio como operación del Estado es exactamente la violencia cultural de Galtung: un aspecto de la cultura que legitima la violencia directa. La recategorización de civiles como combatientes es la misma operación — hace parecer legítimo lo que fue un crimen.

**Fricker** → IEI Score. La distancia al léxico de MAFAPO es la hermeneutical gap operacionalizada: el documento no comparte el espacio semántico de quienes sufrieron el crimen. El NV dentro del IEI es el credibility discount: negar la victimización es negar el testimonio.

**Fraser** → REP y ICM. El REP mide si el documento produce reconocimiento epistémico — si restaura a la víctima como interlocutora válida, condición de la paridad participativa. El ICM mide si ese reconocimiento es performativamente coherente — si se dice con palabras, voz y cara al mismo tiempo.

Lo importante es que cada decisión de diseño — qué medir, con qué peso, con qué método — tiene un anclaje teórico explícito. No hay números elegidos arbitrariamente."

---

## SLIDE 3 — ARQUITECTURA

"La arquitectura tiene tres capas con distintos tipos de señal.

La Capa 1 es textual — gramática y léxico. Aquí Habermas y Galtung son los marcos dominantes. La Capa 2 es semántica — el espacio de significado de los documentos. Aquí Fricker es el marco dominante: la distancia semántica operacionaliza la hermeneutical gap. La Capa 3 es multimodal — video y audio. Aquí Habermas y Fraser son los marcos: la congruencia inter-canal y el reconocimiento performativo.

Los colores: verde calculado, rojo o naranja pendiente. 8 de 13 calculados. Los tres pendientes críticos — EBI, Surprisal, Convergencia Restaurativa — dependen de CFH-BERT v3, que espera el κ>0.80 del IAA."

---

## SLIDE 4 — INDICADORES CAPA 1 (y₁–y₄)

### y₁ — EBI (Galtung)
"El EBI es el indicador más específico del proyecto y el más directamente galtungiano. El léxico de 'bajas en combate', 'neutralización', 'resultado operacional' es violencia cultural lexicalizada — hace parecer que el asesinato de un civil fue un hecho operacional legítimo. Lo detecto con CFH-BERT porque necesito identificar secuencias, no palabras sueltas — 'operacional' por sí solo no es EBI, 'resultado operacional' sí."

### y₂ — SA (Habermas)
"El SA mide la violación habermasiana de la sinceridad expresiva a nivel gramatical. La pasiva sin agente, el 'se' impersonal, la nominalización — todos borran al perpetrador como sujeto gramatical. Lo que Habermas llama 'distorsión sistemática de la comunicación' se manifiesta aquí en el árbol de dependencias sintácticas. El resultado no significativo entre A y B es un hallazgo teóricamente consistente: la supresión de agentividad es constitutiva del lenguaje jurídico colombiano. Viola Habermas en ambos sistemas."

### y₃ — Civil Distance (Fraser)
"La distancia al léxico civil operacionaliza a Fraser: la paridad participativa requiere que el discurso institucional reconozca a los ciudadanos como portadores de derechos. Un documento que usa vocabulario militar en lugar de vocabulario constitucional está construyendo un marco donde las víctimas no existen como ciudadanos. p<0.001."

### y₄ — NV (Fricker + Galtung)
"El NV tiene doble anclaje teórico. Para Fricker es el credibility discount — la víctima pierde credibilidad porque se le atribuye una identidad que invalida su testimonio. Para Galtung es violencia cultural — recategorizar civiles como combatientes legitima retroactivamente su eliminación. El mismo indicador opera en dos marcos teóricos distintos, y por eso aparece tanto en el DIS como en el IEI."

---

## SLIDE 5 — INDICADORES CAPA 2 (y₇–y₁₀)

### y₈ — Dist. MAFAPO (Fricker)
"Este es el indicador más directamente frickeriano del framework. La hermeneutical gap de Fricker — la brecha entre los recursos hermenéuticos de quien tiene la experiencia y los del marco institucional — se mide aquí como distancia coseno entre el embedding del documento y el centroide del léxico de las madres de los falsos positivos. Si el documento habla como MAFAPO, la distancia es baja; si habla como el Ejército, la distancia es alta."

### y₉ — Dist. CIDH (Fraser)
"La distancia al léxico de la Corte IDH operacionaliza a Fraser: el marco normativo internacional es la condición institucional de la paridad participativa. Un documento que no comparte ese léxico está excluyendo el marco que haría posible reconocer la injusticia. También hay un anclaje frickeriano: la ausencia del marco DDHH es una forma de hermeneutical injustice institucional."

### y₁₀ — REP (Fraser + Fricker)
"El REP tiene también doble anclaje. Para Fraser, el lenguaje reparador produce reconocimiento epistémico — restaura a la víctima como interlocutora válida, condición de la paridad participativa. Para Fricker, el REP representa la expansión de los recursos hermenéuticos disponibles para articular la experiencia. El REP alto en la JEP vs. el corpus ordinario es la evidencia más robusta de H₃."

---

## SLIDE 6 — DIS SCORE

"El DIS Score agrega tres indicadores en una medida de injusticia discursiva. Los tres términos corresponden a tres violaciones distintas en el marco teórico.

**y₂ SA con peso 0.35 → Habermas:** la gramática viola la sinceridad expresiva. Peso máximo junto con NV porque ambos operan directamente sobre la injusticia testimonial.

**y₄ NV con peso 0.35 → Fricker + Galtung:** el doble anclaje justifica el peso máximo. La negación de victimización es simultáneamente credibility discount (Fricker) y violencia cultural (Galtung) — las dos cosas al mismo tiempo.

**1 − y₁₀ con peso 0.30 → Fraser:** la ausencia de reconocimiento epistémico. Este término mide lo que el documento no hace, no solo lo que hace. Peso ligeramente menor porque opera desde el polo positivo ausente.

Los pesos no son estadísticos — son teóricos. Si los optimizara sobre los datos estaría haciendo un ajuste al corpus, no una operacionalización de las teorías. Esa es la diferencia entre un índice computacional con base teórica y un índice empírico."

---

## SLIDE 7 — IEI SCORE

"El IEI mide la injusticia epistémica — dimensión más profunda que la discursiva, en términos de Fricker. La injusticia epistémica no es solo que el testimonio sea silenciado gramaticalmente, sino que no existen los conceptos para articular la experiencia.

**y₈ MAFAPO con peso 0.35 → Fricker:** la hermeneutical gap directa. Mayor peso porque es el anclaje más directo con la teoría.

**y₉ CIDH con peso 0.20 → Fraser:** la exclusión del marco normativo. Peso menor porque es más mediado — opera a través de las instituciones, no del testimonio directo.

**y₄ NV con peso 0.25 → Fricker:** el mismo NV que en el DIS, pero ahora en el rol del credibility discount como causa de injusticia epistémica. El mismo indicador en dos índices con pesos distintos — porque el NV opera en ambas dimensiones.

**1 − y₁₀ con peso 0.20 → Fraser + Fricker:** la ausencia de expansión de recursos hermenéuticos. Peso menor porque es consecuencia de las otras tres dimensiones.

El IEI puede ser alto aunque el DIS sea bajo — ese es el caso Catatumbo. Y eso es exactamente lo que los cuatro marcos teóricos predicen: la corrección gramatical no implica la corrección epistémica."

---

## SLIDE 8 — MODELO SEM

"El SEM es el marco analítico confirmatorio. Cuatro variables latentes: ξ₁ Violencia Discursiva — los indicadores de Habermas y Galtung. ξ₂ Contexto Institucional. η₁ DIS Score. η₂ Transición Epistémica — los indicadores de Fricker y Fraser.

La H₃ — β₂₃ < 0 — conecta los dos marcos teóricos en una única hipótesis estadística: mayor violencia discursiva habermasiana-galtungiana predice menor transición epistémica frickeriana-fraseriana.

Por qué SEM: porque los indicadores tienen error de medición. El análisis de árboles sintácticos de spaCy comete errores de parsing; CFH-BERT comete errores de clasificación. El SEM modela ese error explícitamente y separa la varianza verdadera de la de error. Una regresión ordinaria ignoraría eso."

---

## SLIDE 9 — RESULTADOS A vs B

"5 de 6 indicadores con p<0.001. El hallazgo clave es la convergencia multi-teórica: indicadores de Habermas, Galtung, Fricker y Fraser señalan en la misma dirección con métodos independientes.

El SA no significativo refuerza el argumento habermasiano: la distorsión sistemática de la comunicación no es un problema de la justicia ordinaria versus la transicional — es constitutiva del lenguaje jurídico-institucional colombiano en general. Viola Habermas en ambos sistemas."

---

## SLIDE 10 — CORPUS C: DIS/IEI POR SUBCASO

"Los cinco subcasos tienen perfiles distintos que corresponden a diferentes combinaciones de los marcos teóricos.

Casanare: coherencia negativa — viola Habermas-Galtung en el DIS Y viola Fricker-Fraser en el IEI. El General Torres Escalante tiene SA=0.974 y REP=0.000.

Dabeiba: la materialidad del reconocimiento — 49 fosas identificadas, nombres, fechas — se refleja en índices moderados en ambas dimensiones. La concreción reduce tanto el DIS como el IEI.

Huila: el mejor perfil bi-índice del corpus. Mayor REP del corpus C (0.147) — el más cercano al reconocimiento fraseriano.

Catatumbo: el caso paradigmático que me lleva al siguiente slide."

---

## SLIDE 11 — DISOCIACIÓN DIS-IEI

"Catatumbo es el caso más importante del framework porque demuestra empíricamente que los cuatro marcos teóricos son necesarios — ninguno es suficiente.

DIS=0.110 — el más bajo. El Capitán Chaparro corrige las pretensiones habermasianas gramaticales: habla en primera persona, sin eufemismos, sin pasivas. El canal habermasiano está corregido.

IEI=0.624 — el más alto. Pero su léxico está alejado del espacio semántico de las madres de las víctimas. La hermeneutical gap de Fricker persiste. El canal frickeriano está intacto.

Desde Habermas: hay coherencia gramatical. Desde Fricker: hay brecha hermenéutica. Desde Fraser: el reconocimiento verbal no produce paridad participativa porque falta el reconocimiento semántico. Desde Galtung: la corrección de la violencia discursiva explícita no elimina la violencia cultural hermenéutica.

Cuatro marcos teóricos, cuatro diagnósticos distintos. Ninguno solo captura la injusticia completa. Eso valida la necesidad del framework multi-teórico.

Y por eso el ICM tri-canal es necesario: si la disociación existe entre canales textuales, puede existir entre el canal verbal y los canales vocal y facial."

---

## SLIDE 12 — ICM TRI-CANAL

"El ICM mide la congruencia entre los tres canales de comunicación. Su anclaje es habermasiano y fraseriano.

**Habermas:** la acción comunicativa requiere sinceridad en todos los canales simultáneamente. La incongruencia entre lo que se dice, cómo se dice con la voz, y qué se expresa con la cara es una violación de la sinceridad expresiva habermasiana en su forma más profunda.

**Fraser:** la paridad participativa no se garantiza con las palabras correctas si los otros canales dicen algo diferente. La congruencia tri-canal es el requisito performativo del reconocimiento genuino.

Los pesos: 0.40 facial, 0.40 vocal, 0.20 verbal. El verbal tiene el menor peso porque es el canal más susceptible de manipulación consciente — se puede aprender a decir las palabras correctas. Los canales vocal y facial son más difíciles de controlar simultáneamente.

La salvaguarda epistemológica es no negociable: el ICM mide congruencia inter-canal, no sinceridad. No es un polígrafo. Es una medida de consistencia discursiva — una propiedad del discurso, no del sujeto."

---

## SLIDE 13 — IAA Y CFH-BERT

"El κ>0.80 sobre 500 fragmentos desbloquea los tres indicadores pendientes: y₁ EBI (Galtung), y₇ Surprisal (Habermas) y y₁₁ Convergencia Restaurativa (Fricker). Con ellos el SEM completo puede estimarse y la H₃ puede confirmarse estadísticamente."

---

## SLIDE 14 — ESTADO Y PENDIENTES

"La integración final: Habermas+Galtung → DIS Score (η₁). Fricker+Fraser → IEI Score (η₂). Habermas+Fraser → ICM tri-canal (y₁₂). Los cuatro marcos teóricos están presentes en el modelo. La H₃ los conecta en una única hipótesis estadística.

El alcance H3 para agosto: framework + dataset κ>0.80 + paper en inglés. Los cinco papers derivados son trabajo futuro declarado."

---

## PREGUNTAS ANTICIPADAS

**"¿Por qué cuatro marcos teóricos? ¿No es demasiado?"**
No — es necesario. La disociación DIS-IEI en Catatumbo demuestra empíricamente que un solo marco no captura la injusticia completa. Si solo tuviera Fricker, el caso Catatumbo parecería el menos injusto (DIS bajo). Si solo tuviera Habermas, el caso Catatumbo parecería el más injusto (IEI alto). Necesito los cuatro para el diagnóstico correcto.

**"¿Por qué esos pesos y no otros?"**
Los pesos son teóricamente justificados, no estadísticamente optimizados. Si los optimizara sobre el corpus estaría ajustando el modelo a los datos y perdería la validez teórica. Los pesos reflejan la jerarquía del marco teórico: la violencia activa directa (NV, SA) tiene mayor peso que la ausencia del polo positivo (1−REP).

**"¿Por qué normalizar dentro del Corpus C y no sobre A+B?"**
Porque quiero comparar subcasos entre sí. Si normalizo sobre A+B, los cinco subcasos JEP quedarían comprimidos en el extremo bajo — todos son JEP. La normalización dentro del Corpus C revela la variación interna, que es donde está la información relevante.

**"¿El ICM no es un polígrafo?"**
No. El polígrafo asume que hay estados psicológicos internos que se manifiestan en señales fisiológicas, y que la "mentira" tiene una firma fisiológica específica. El ICM mide algo diferente: si los tres canales de comunicación son coherentes entre sí. Eso es una propiedad del discurso, no del sujeto. Habermas lo llamaría integridad performativa del acto de habla.

**"¿Por qué CFH-BERT y no GPT o Claude?"**
Por auditabilidad. En un sistema de análisis forense, las probabilidades de clasificación por token tienen que ser examinables, replicables e interpretables. Un encoder bidireccional como BERT produce representaciones que puedo examinar y validar con κ inter-anotador. Los modelos generativos producen texto — no me dan log-P(token|contexto) auditables. En el contexto jurídico colombiano, la auditabilidad no es una preferencia técnica, es un requisito metodológico.
