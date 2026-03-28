# PARTE III: Metodología cuantitativa para el análisis del discurso

## 3.1 Modelos de ecuaciones estructurales (SEM): fundamentos y tipología

Los Modelos de Ecuaciones Estructurales (SEM, *Structural Equation Modeling*) constituyen una familia de técnicas estadísticas multivariadas que combinan el análisis factorial confirmatorio (CFA) con la regresión múltiple para estimar simultáneamente relaciones entre variables observables y variables latentes (Bollen, 1989; Kline, 2023). Su incorporación al proyecto CFH resuelve el problema epistemológico más serio del diseño original: la imposibilidad de observar directamente constructos como "Injusticia Discursiva" o "Transición Epistémica", que son entidades teóricas sin correspondencia directa en el mundo observacional.

**La distinción modelo de medición / modelo estructural** es el núcleo conceptual del SEM. El **modelo de medición** especifica cómo los constructos latentes (variables teóricas, representadas con elipses en los diagramas de trayectorias) se relacionan con sus indicadores observables (variables medidas, representadas con rectángulos). Las cargas factoriales λ cuantifican la fuerza de esa relación: un indicador con λ = 0.85 explica el 72% de su varianza a partir del constructo latente, lo que implica alta validez de constructo. El **modelo estructural** especifica las relaciones causales entre los propios constructos latentes mediante coeficientes de regresión estructural β, análogos a los coeficientes en una regresión múltiple pero estimados con corrección por error de medición.

**CB-SEM versus PLS-SEM.** La literatura distingue dos tradiciones principales. El SEM basado en covarianzas (CB-SEM, *Covariance-Based SEM*), implementado en R mediante `lavaan` (Rosseel, 2012) o en Python mediante `semopy` (Igolkina & Meshcheryakov, 2020), estima los parámetros del modelo minimizando la discrepancia entre la matriz de covarianza observada y la matriz de covarianza implicada por el modelo. El SEM basado en varianza o de mínimos cuadrados parciales (PLS-SEM, Hair et al., 2022) maximiza la varianza explicada de las variables endógenas y es más apropiado para modelos explorativos con muestras pequeñas. Para el proyecto CFH, **CB-SEM es la elección correcta** por tres razones: (1) el proyecto es confirmatorio —las hipótesis teóricas de Fraser y Galtung guían la especificación del modelo—; (2) el corpus A cuenta con 280 documentos, suficiente para estimación por Máxima Verosimilitud (ML) con modelos de complejidad moderada (regla práctica: n ≥ 10 × número de parámetros libres); (3) el interés está en los índices de bondad de ajuste global del modelo, que CB-SEM proporciona y PLS-SEM no.

**Indicadores de bondad de ajuste.** Los criterios de evaluación del ajuste del modelo son un componente estándar del reporte de resultados SEM (Hu & Bentler, 1999):
- **RMSEA** (*Root Mean Square Error of Approximation*): mide el error por grado de libertad. Valores < 0.05 indican ajuste excelente, < 0.08 ajuste aceptable.
- **CFI** (*Comparative Fit Index*): compara el modelo propuesto con el modelo nulo. Valores > 0.95 indican ajuste excelente, > 0.90 aceptable.
- **SRMR** (*Standardized Root Mean Square Residual*): residuo promedio de covarianza. Valores < 0.08 indican ajuste aceptable.
- **χ²/df** (*Chi-square ratio*): sensible al tamaño de muestra. Valores < 3.0 generalmente aceptables.

La biblioteca `semopy` v2.x (Igolkina & Meshcheryakov, 2020) implementa CB-SEM en Python con una sintaxis inspirada en R/lavaan, soporte para estimación ML y GLS, y cálculo de índices de ajuste estándar. Su integración con el ecosistema Python del proyecto (pandas, numpy) la hace la elección natural para el módulo de análisis SEM.

**Referencias clave:** Bollen (1989), Kline (2023, 4ª ed.), Hu & Bentler (1999), Rosseel (2012, lavaan), Igolkina & Meshcheryakov (2020, semopy), Hair et al. (2022, PLS-SEM), MacKinnon (2008).

---

## 3.2 SEM en análisis computacional del discurso

La aplicación de SEM al análisis computacional del discurso es un campo emergente con antecedentes en lingüística de corpus, análisis crítico del discurso y ciencias políticas computacionales. Su revisión es necesaria para posicionar el proyecto CFH como extensión de trabajos existentes, no como propuesta aislada.

En **lingüística de corpus**, Biber (1988, 1995) introdujo el uso de análisis multivariado para identificar dimensiones latentes de variación entre registros lingüísticos (su modelo de 5 dimensiones, desarrollado sobre el British National Corpus, puede considerarse precursor metodológico del enfoque SEM). Trabajos más recientes han aplicado CFA para validar taxonomías de rasgos lingüísticos en corpora especializados (Egbert & Biber, 2019; Berber Sardinha & Pinto, 2019), estableciendo precedente para el uso de constructos latentes en lingüística computacional.

En **análisis del discurso político computacional**, Laver et al. (2003) aplicaron modelos de escala unidimensional (*Wordfish*, *Wordscores*) para estimar posiciones ideológicas de partidos a partir de manifiestos electorales. Slapin & Proksch (2008) refinaron estos métodos. La extensión a SEM —donde la posición ideológica sería un constructo latente con múltiples indicadores textuales— ha sido propuesta por Proksch et al. (2019) en el contexto del análisis de debates parlamentarios europeos, aunque con implementación metodológica más cercana a modelos de espacio vectorial que a CB-SEM formal.

En el dominio más cercano al proyecto CFH —análisis computacional de narrativas de trauma y justicia transicional— el trabajo de Cohn et al. (2004) en análisis computacional de narrativas de trauma mediante LIWC (*Linguistic Inquiry and Word Count*) es un antecedente relevante: aunque metodológicamente más simple que SEM, establece la viabilidad del análisis cuantitativo de constructos psicológicos latentes (elaboración cognitiva, expresión emocional) mediante indicadores textuales contados. Para el proyecto CFH, la innovación consiste en reemplazar los conteos de palabras de LIWC por embeddings de modelos de lenguaje fine-tuneados para el dominio del conflicto colombiano, y en usar CB-SEM para estimar formalmente las relaciones entre los constructos.

El trabajo más directamente relacionado con el proyecto CFH es el de **Hopp et al. (2021)** sobre *Computational Approaches to the Study of Political Communication*, que propone integrar indicadores de NLP (sentimiento, entidad nombrada, framing) como variables observables en modelos de mediación SEM para el análisis de efectos de priming mediático. Este trabajo provee el blueprint metodológico más preciso para la especificación del modelo CFH: los indicadores NLP son las variables observables, los constructos teóricos de la comunicación política son las variables latentes, y las relaciones entre constructos son las hipótesis del análisis.

**Brecha que cubre el proyecto CFH:** no existe ningún trabajo publicado que aplique CB-SEM con indicadores extraídos de modelos de lenguaje fine-tuneados sobre corpus de justicia transicional. La contribución metodológica del proyecto CFH es exactamente este puente entre (a) la riqueza representacional de los LLMs de dominio específico y (b) el rigor inferencial del SEM confirmatorio.

**Referencias clave:** Biber (1988, 1995), Laver et al. (2003, Wordscores), Proksch et al. (2019), Hopp et al. (2021), Cohn et al. (2004, LIWC narratives), Egbert & Biber (2019).

---

## 3.3 SEM multi-grupo: invarianza de medición entre corpus

El diseño quasi-experimental del proyecto CFH compara tres corpus que representan tres momentos y modalidades del discurso judicial colombiano. Para que esa comparación sea metodológicamente válida, es necesario establecer que el modelo de medición —las relaciones entre los indicadores observables y los constructos latentes— opera de forma equivalente en los tres grupos. Este requisito se denomina **invarianza de medición** (*measurement invariance*) y su verificación mediante SEM multi-grupo (MG-SEM) es el estándar metodológico para comparaciones entre grupos (Vandenberg & Lance, 2000; Putnick & Bornstein, 2016).

El protocolo de invarianza progresiva establece cuatro niveles (de menor a mayor restricción):

1. **Invarianza configural:** el mismo patrón de cargas (qué indicadores pertenecen a qué constructo) se aplica en todos los grupos. Es el nivel mínimo; si no se cumple, los constructos no son comparables.

2. **Invarianza métrica:** las cargas factoriales λ son iguales entre grupos. Permite comparar las relaciones entre constructos (coeficientes β). Si no se cumple, los indicadores "pesan" de forma diferente en cada corpus.

3. **Invarianza escalar:** además de las cargas, los interceptos de los indicadores son iguales. Permite comparar los valores medios de los constructos latentes entre grupos. Este es el nivel necesario para afirmar que el DIS Score (η₁) es directamente comparable entre corpus A, B y C.

4. **Invarianza estricta:** además de cargas e interceptos, las varianzas de los errores de medición son iguales. El nivel más restrictivo; raramente se requiere en la práctica.

Para el proyecto CFH, la hipótesis de invarianza escalar tiene implicaciones teóricas directas: si los indicadores del constructo ξ₁ (*Violencia Discursiva*) son escalármente invariantes entre corpus A y corpus C, significa que la misma escala de medición aplica para el lenguaje de la justicia ordinaria y el lenguaje de las audiencias JEP —es decir, que se puede hacer una comparación directa de los valores medios del constructo. Si la invarianza escalar falla (como es probable dado el contraste entre los dos sistemas), la invarianza métrica parcial permite identificar qué indicadores son comparables y cuáles reflejan diferencias genuinas en cómo el constructo se manifiesta en cada institución.

Esta distinción tiene relevancia sustantiva: si el indicador "supresión de agentividad" (λ de ξ₁) tiene una carga factorial significativamente diferente en corpus A versus corpus C, eso no es un problema estadístico sino un hallazgo: la supresión de agentividad funciona diferente —tiene diferente peso en la producción de violencia discursiva— en el lenguaje ordinario judicial y en el lenguaje de las audiencias de reconocimiento. Ese es precisamente el tipo de hallazgo que el proyecto CFH busca.

**Software:** `semopy` implementa MG-SEM mediante el parámetro `groups` en la especificación del modelo. Las pruebas de invarianza se realizan mediante diferencia de χ² entre modelos anidados (modelo configural vs. métrico vs. escalar), con corrección de Satorra-Bentler para distribuciones no normales.

**Referencias clave:** Vandenberg & Lance (2000), Putnick & Bornstein (2016), Millsap (2011), Byrne et al. (1989, invarianza parcial), Muthén & Muthén (2017).

---

## 3.4 SEM multimodal: integración de indicadores textuales y acústicos

La integración de múltiples modalidades de señal —texto, audio, potencialmente video— dentro de un único modelo de medición formalizado es uno de los desarrollos más recientes en metodología cuantitativa del discurso. El trabajo de **Zadeh et al. (2018)** sobre fusión multimodal tardía (*late fusion*) en análisis de sentimientos establece que la integración de representaciones textuales, acústicas y visuales mediante modelos de atención produce mejoras consistentes sobre modelos unimodales. Sin embargo, la fusión por atención opera como una caja negra: no explicita las relaciones entre modalidades en términos teóricamente interpretables.

El SEM multimodal (Bagozzi & Yi, 2012; Yoo & Alavi, 2001) ofrece una alternativa que preserva la interpretabilidad: si el constructo latente η₂ (*Transición Epistémica*) existe como entidad teórica, entonces sus manifestaciones en las señales textual y acústica son indicadores co-efectos del mismo constructo, no señales que se "fusionan" sino señales que covarian porque comparten una causa común latente. El modelo SEM formaliza esta relación y permite verificar empíricamente si la covarianza entre indicadores textuales y acústicos es consistente con la hipótesis de constructo común (mediante el índice de ajuste CFI) o si las dos modalidades miden constructos diferentes (índice de discrepancia RMSEA).

Para el corpus C del proyecto CFH, los indicadores del modelo multimodal de η₂ se organizan en dos grupos. Los indicadores textuales incluyen: el léxico restaurativo (proporción de tokens pertenecientes a un diccionario de términos asociados a reparación, responsabilidad y reconocimiento del daño); la convergencia semántica (distancia coseno entre el embedding del segmento y el centroide del corpus MAFAPO); y el surprisal en el modelo entrenado sobre corpus de paz (impredecibilidad semántica como señal de ruptura con el vocabulario de guerra). Los indicadores acústicos incluyen: la tasa de habla (reducción asociada a deliberación consciente y peso emocional); la energía promedio (proxy de activación afectiva); y el ratio de pausas largas (> 0.5 segundos, asociado en la literatura a carga cognitiva y procesamiento emocional de contenido traumático en contextos de testimonio).

La **especificación del modelo multimodal** requiere tratar la diferencia de escala entre indicadores textuales (típicamente en el rango [0, 1] normalizado) y acústicos (Hz para F0, dB para energía) mediante estandarización z-score antes de la estimación SEM. Los errores de medición de indicadores de la misma modalidad pueden correlacionarse libremente (errores correlacionados de método), lo que el modelo SEM maneja mediante la liberación de covarianzas entre errores del mismo bloque modal.

**Limitación y alcance para la tesis.** El modelo SEM multimodal se aplica únicamente al corpus C (audiencias JEP), que es el único con señal de audio disponible. Para los corpus A y B, el modelo opera solo con indicadores textuales. El MG-SEM de la sección 3.3 puede contrastar la estructura de η₂ entre corpus B (solo texto) y corpus C (texto + audio) como análisis de sensibilidad: si la estructura se mantiene con o sin los indicadores acústicos, la validación discriminante del constructo es más robusta.

**Referencias clave:** Zadeh et al. (2018, CMU-MOSI), Bagozzi & Yi (2012), Schuller et al. (2021, INTERSPEECH ComParE), Yoo & Alavi (2001), Baltrušaitis et al. (2019, multimodal ML survey).

---

## 3.5 Diseño quasi-experimental en NLP legal

El diseño quasi-experimental (Campbell & Stanley, 1966; Shadish et al., 2002) busca establecer inferencias causales cuando la asignación aleatoria a condiciones es imposible. En el proyecto CFH, la imposibilidad de asignación aleatoria es obvia: los documentos del corpus A fueron producidos por la justicia ordinaria y los del corpus C por la JEP, y no existe mecanismo aleatorio que hubiera asignado un caso a uno u otro sistema. Sin embargo, la estructura del Macrocaso 003 —donde los mismos hechos (las mismas muertes) son tratados por ambos sistemas— provee una lógica de comparación que se aproxima al contrafáctico experimental.

La estructura formal adoptada es un **diseño de regresión discontinua** (*regression discontinuity design*, RDD) aplicado a documentos: la variable de asignación es el año de producción del documento (pre-2016: corpus A, post-2016: corpus B y C), con el Acuerdo Final como umbral de discontinuidad. Los documentos no se distribuyen aleatoriamente a ambos lados del umbral —todos los documentos post-2016 son de la JEP— pero la comparación de las medias de η₁ y η₂ a ambos lados del umbral, bajo el supuesto de que las características observables del corpus (región, tipo de crimen, período 2002-2008) son comparables, permite la inferencia de un efecto de "tratamiento institucional".

La **amenaza de validez interna** más seria en este diseño es la historia: el cambio en el discurso podría deberse a transformaciones culturales y políticas del período 2016-2022, no al sistema de justicia per se. El proyecto CFH aborda esta amenaza mediante la inclusión del constructo latente ξ₂ (*Contexto Institucional*) en el modelo SEM, que actúa como covariable controlando las diferencias entre corpus que no se deben al sistema de justicia sino al período histórico.

En el campo del NLP legal, los antecedentes de diseños quasi-experimentales son escasos pero existentes. Ash et al. (2021) emplearon cambios exógenos en la composición de cortes judiciales como instrumentos para analizar el efecto de la ideología de los jueces sobre el lenguaje de las sentencias. Luo et al. (2017) utilizaron un diseño de differences-in-differences sobre corpus de textos legales para medir el impacto de reformas legislativas sobre el lenguaje contractual. Estos trabajos establecen precedente para la aplicación de metodología causal en análisis de texto jurídico.

**Referencias clave:** Shadish et al. (2002), Imbens & Lemieux (2008, RDD), Ash et al. (2021), Luo et al. (2017), Campbell & Stanley (1966).

---

## 3.6 Anotación de corpus y acuerdo inter-anotadores en textos de conflicto

La construcción del corpus anotado que alimenta el modelo SEM requiere un protocolo de anotación riguroso. La anotación de textos de violencia política y conflicto armado presenta desafíos específicos que superan los de la anotación de textos generales: la ambigüedad semántica es alta (¿cuándo un eufemismo es intencionalmente violento y cuándo es simplemente terminología técnica?), la carga emocional puede afectar las decisiones de los anotadores, y las categorías teóricas (violencia cultural, supresión de agentividad) no tienen correspondencias simples en unidades lingüísticas observables.

La **taxonomía de violencia discursiva** propuesta para el proyecto CFH tiene cuatro categorías primarias, operacionalizadas a nivel de sintagma (span de texto):

1. **Eufemismo bélico-institucional** (*EBI*): expresiones que renombran homicidios como acciones militares legítimas. Ejemplos prototípicos: "baja en combate", "resultado operacional", "dado de baja", "misión táctica". El criterio de anotación es la sustituibilidad: si el sintagma puede reemplazarse por "fue asesinado" sin pérdida de referencia, se anota como EBI.

2. **Supresión de agentividad** (*SA*): construcciones que eliminan o difuminan el agente responsable de la acción. Operacionalización: pasivas sin complemento agente ("se presentó la baja"), nominalizaciones de procesos dinámicos ("el deceso"), construcciones impersonales ("se produjo un intercambio de disparos").

3. **Negación de victimización** (*NV*): expresiones que recaracterizan a la víctima como combatiente o agresor. Ejemplos: "presunto guerrillero", "quien portaba armas", "en actitud hostil".

4. **Ruptura epistémica positiva** (*REP*): expresiones que refieren la perspectiva de la víctima como válida e irrefutable. Estas solo aparecen en corpus B y C. Ejemplos: "era un civil inocente", "la familia confirma que no tenía vínculos", "reconocemos el daño causado".

El **protocolo IAA** sigue el estándar de Artstein & Poesio (2008): dos anotadores independientes anotarán el 20% del corpus, con umbral mínimo de Cohen κ = 0.75 para cada categoría antes de expandir la anotación al corpus completo. Las discrepancias se resolverán mediante consenso con un tercer anotador experto en derecho internacional humanitario. El coeficiente de Krippendorff α (Krippendorff, 2018) se calculará adicionalmente como medida de acuerdo más conservadora para datos nominales con múltiples categorías.

La herramienta de anotación recomendada es **Label Studio** (HeadHunter LLC, 2020), de código abierto, con soporte para anotación de spans de texto, exportación en formato IOB2 compatible con HuggingFace Datasets, y gestión de múltiples anotadores con cálculo automático de IAA.

**Referencias clave:** Artstein & Poesio (2008), Cohen (1960), Krippendorff (2018), Pustejovsky & Stubbs (2012), Yimam et al. (2014, WebAnno), Bontcheva et al. (2013).

---

## 3.7 Ingeniería de la Paz: fundamentos y antecedentes empíricos

La Ingeniería de la Paz (*Peace Engineering*) es un campo interdisciplinar emergente que integra principios y métodos de la ingeniería y la tecnología con los objetivos de la construcción de paz y la resolución de conflictos. Su premisa central es que las herramientas tecnológicas no son neutrales: pueden diseñarse deliberadamente para contribuir a la justicia, la reparación y la no violencia. Esta premisa contrasta con la posición de neutralidad tecnológica que subyace a la mayoría de aplicaciones de IA en el sistema judicial.

El trabajo de Moro et al. (2022) del Kroc Institute for International Peace Studies (Universidad de Notre Dame) sobre análisis computacional de acuerdos de paz provee el antecedente empírico más directo: usando NLP sobre el corpus de 34 acuerdos de paz del PA-X Peace Agreement Database, estos autores identificaron patrones lingüísticos asociados a la durabilidad de los acuerdos, estableciendo que la especificidad del lenguaje en cláusulas de derechos humanos predice positivamente la implementación efectiva. Este trabajo demuestra la viabilidad del análisis computacional del lenguaje como herramienta de auditoría de compromisos de paz.

El proyecto **Colombia Peace Barometer** del Centro de Recursos para el Análisis de Conflictos (CERAC) ha aplicado técnicas de análisis de datos para monitorear el cumplimiento del Acuerdo Final, incluyendo análisis de texto de informes de verificación. Aunque metodológicamente menos sofisticado que el proyecto CFH, establece precedente institucional para el uso de análisis computacional en el contexto de la paz colombiana.

En el dominio de la IA para la justicia transicional, el trabajo del **Human Rights Data Analysis Group** (HRDAG) sobre análisis estadístico de violaciones de derechos humanos en Colombia (Ball et al., 2000; Silva & Ball, 2007) provee el antecedente metodológico más sólido: el uso de modelos estadísticos para establecer la sistematicidad y escala de crímenes de Estado a partir de registros documentales. El proyecto CFH extiende esta tradición hacia el dominio del lenguaje: mientras HRDAG analiza quién murió y cuándo, CFH analiza cómo se habló de esas muertes.

La conexión con el marco teórico de **Reparación Algorítmica** (sección 4.2) sitúa el proyecto CFH dentro de la corriente más radical de la Ingeniería de la Paz: no solo usar la IA para documentar el daño, sino diseñarla activamente para nombrarlo, desenmascarlo y contribuir a su reparación simbólica. Esta posición tiene antecedentes en el campo del diseño tecnológico para la justicia social (Costanza-Chock, 2020) y en la literatura sobre *value-sensitive design* (Friedman et al., 2019).

**Referencias clave:** Moro et al. (2022, Kroc Institute), Bell (2000, HRDAG), Costanza-Chock (2020), Friedman et al. (2019), PA-X Database, CERAC (2016-2022).

---

# PARTE IV: Marco original y posicionamiento del proyecto CFH

## 4.1 Hermenéutica Forense Computacional: posicionamiento en la literatura

La **Hermenéutica Forense Computacional** (HFC) es el marco epistemológico original de este proyecto. Su posicionamiento como contribución teórica nueva requiere demostrar que: (a) no existe un marco previo con el mismo alcance y objetivos, y (b) la HFC resuelve un problema que los marcos existentes dejan sin resolver.

La **hermenéutica jurídica** clásica (Gadamer, 1960/1989; Ricoeur, 1981) estudia los principios de interpretación de los textos normativos: la fusión de horizontes entre el texto y el intérprete, el círculo hermenéutico, la pre-comprensión del intérprete. Su fortaleza es la profundidad filosófica; su limitación para el proyecto CFH es que opera sobre textos individuales en manos de intérpretes individuales, sin capacidad de análisis sistemático a la escala de un corpus de 280 sentencias.

El **análisis crítico del discurso** (ACD, Fairclough 1992; Van Dijk 1993, 2008) estudia las relaciones entre el lenguaje, el poder y la ideología. Su fortaleza es exactamente la que le falta a la hermenéutica jurídica: una teoría explícita de cómo el lenguaje produce y reproduce estructuras de dominación. Su limitación para el proyecto CFH es metodológica: el ACD opera mediante análisis cualitativo intensivo de textos selectos, sin mecanismos para cuantificar la intensidad de los fenómenos discursivos sobre un corpus masivo.

Los **métodos computacionales de análisis del discurso** (Computational Discourse Analysis, Janier et al., 2021; Lippi & Torroni, 2016) aplican NLP al estudio de la argumentación, la coherencia y la estructura retórica. Su fortaleza es la escalabilidad; su limitación es que típicamente operan sobre categorías formales (tipos de acto de habla, relaciones de coherencia) sin un marco normativo explícito que defina qué discursos son justos o injustos.

La **Hermenéutica Forense Computacional** ocupa el espacio que ninguno de estos marcos ocupa: combina (1) la profundidad filosófica de la hermenéutica en la definición de constructos (¿qué es injusticia discursiva? ¿qué es transición epistémica?), (2) la teoría crítica del ACD sobre la relación entre lenguaje y poder, (3) la escalabilidad del análisis computacional para corpus masivos, y (4) el rigor inferencial del SEM para la estimación de constructos latentes y sus relaciones causales. La palabra "forense" señala el dominio de aplicación (archivos judiciales de crímenes de Estado) y la orientación hacia la producción de evidencia válida en contextos de justicia; "computacional" señala el método; "hermenéutica" señala el compromiso epistemológico con la interpretación significativa en lugar del procesamiento mecánico.

**Diferenciación de marcos relacionados.** La HFC difiere de la **Forensic Linguistics** (Coulthard & Johnson, 2010) en que esta última analiza texto para establecer autoría o relevancia probatoria en casos individuales, no para auditar sistemas institucionales. Difiere del **Computational Legal Analysis** (Surden, 2014) en que este se enfoca en la eficiencia procedimental, no en la justicia epistémica. Difiere de la **Computational Peace Science** (Hammond, 2018) en que esta opera principalmente sobre datos estructurados (eventos de conflicto, acuerdos), no sobre el lenguaje del archivo judicial.

---

## 4.2 Reparación Algorítmica: de la crítica a la técnica

El concepto de **Reparación Algorítmica** (*Algorithmic Reparation*) emerge de la confluencia entre la literatura sobre sesgos en IA y la teoría de la justicia reparativa. Su premisa es que si los algoritmos pueden producir daño —como documentan Angwin et al. (2016) con COMPAS, Obermeyer et al. (2019) con algoritmos de salud, y Buolamwini & Gebru (2018) con reconocimiento facial— entonces los algoritmos también pueden diseñarse para reparar daños históricos, no solo para evitar producir nuevos daños.

La crítica a la **neutralidad tecnológica** es el punto de partida. Winner (1980) demostró que los artefactos tecnológicos "tienen política": las elecciones de diseño incorporan valores que favorecen determinados grupos sobre otros. Benjamin (2019) y Noble (2018) extienden este argumento al dominio de los algoritmos de IA, demostrando que la discriminación puede estar codificada en los datos de entrenamiento, las funciones de pérdida y los criterios de evaluación. Binns (2018) problematiza la noción de "fairness" algorítmica, argumentando que las múltiples definiciones matemáticas de equidad son mutuamente incompatibles, y que la elección entre ellas es una decisión política, no técnica.

La **Reparación Algorítmica** radicaliza este argumento: no se trata solo de hacer algoritmos "menos sesgados" sino de reorientar el diseño tecnológico hacia la reparación activa de daños históricos. Costanza-Chock (2020) en *Design Justice* articula los principios de un diseño tecnológico centrado en las comunidades más afectadas por los daños del sistema. Para el proyecto CFH, esto se traduce en una decisión de diseño específica: el polo normativo del modelo de medición no es un promedio de textos judiciales ni una definición neutral de "lenguaje justo", sino los relatos de las víctimas (MAFAPO) y los estándares de la CIDH. La Reparación Algorítmica, en este contexto, es el acto de diseñar el modelo de tal manera que el criterio de verdad sea la voz de las personas más dañadas por el sistema que se está auditando.

**Limitaciones y controversias.** La noción de Reparación Algorítmica no está exenta de crítica. Hoffmann (2019) argumenta que el énfasis en soluciones técnicas puede desviar la atención de los cambios institucionales y políticos que la reparación genuina requiere. Para el proyecto CFH, esta crítica es bienvenida: el modelo SEM no repara nada por sí mismo. Lo que hace es producir evidencia empírica que puede ser usada en procesos de reparación institucional. Su contribución es epistémica, no jurídica ni política.

**Referencias clave:** Angwin et al. (2016, COMPAS), Obermeyer et al. (2019), Buolamwini & Gebru (2018), Winner (1980), Benjamin (2019), Noble (2018), Binns (2018), Costanza-Chock (2020), Hoffmann (2019).

---

## 4.3 Ética de investigación en corpus judiciales sensibles

El acceso, procesamiento y análisis de documentos judiciales relativos a crímenes de Estado plantea obligaciones éticas que deben abordarse explícitamente antes del inicio de la recolección de datos. Esta sección identifica los principales desafíos éticos del proyecto CFH y propone mitigaciones específicas.

**Acceso al corpus y condiciones legales.** Los documentos del corpus A (sentencias de justicia ordinaria) están disponibles en el Repositorio de Jurisprudencia del Consejo Superior de la Judicatura (https://jurisprudencia.ramajudicial.gov.co), bajo las condiciones de uso establecidas por la Ley 1712 de 2014 (transparencia y acceso a la información pública). Su procesamiento computacional para fines de investigación académica es compatible con estas condiciones, siempre que no se publiquen fragmentos que identifiquen a víctimas o procesados sin consentimiento. Los documentos del corpus B (Autos JEP) están disponibles en el repositorio público de la JEP (https://www.jep.gov.co). Las grabaciones del corpus C pueden ser de acceso restringido para algunas audiencias; se requiere verificar caso por caso el régimen de publicidad de cada sesión.

**Anonimización de partes procesales.** Los metadatos extraídos por el módulo de ingesta incluyen nombres de procesados, víctimas, fiscales y defensores. En el corpus final para entrenamiento de modelos y análisis estadístico, estos campos deben ser sustituidos por identificadores seudonimizados (*pseudonymization*), no simplemente eliminados (*anonymization*): la seudonimización permite la trazabilidad interna sin exposición de datos personales. El protocolo de seudonimización debe documentarse como parte del registro de auditoría del pipeline.

**Protocolo de revisión ética institucional.** Se recomienda someter el protocolo de investigación a revisión del Comité de Ética de Investigación de la institución académica (equivalente al IRB en contexto anglosajón), con especial atención a: (1) el manejo de relatos de trauma en el corpus C (transcripciones de testimonios de víctimas); (2) el riesgo de re-identificación a partir de combinaciones de metadatos; y (3) el uso de los resultados del modelo en contextos que puedan afectar a las personas mencionadas en el corpus.

**Posicionamiento del investigador.** El análisis computacional de violaciones de derechos humanos no es moralmente neutro. El proyecto CFH adopta una postura explícita de solidaridad epistémica con las víctimas, formalizada en la elección del polo normativo del modelo (MAFAPO + CIDH). Esta postura debe declararse en la sección metodológica de la tesis como posicionamiento reflexivo, siguiendo la tradición de la investigación crítica (Harding, 1991; Creswell & Poth, 2018).

**Referencias clave:** Ley 1712 de 2014 (Colombia), JEP Reglamento General (2018), CIOMS International Ethical Guidelines (2016), Harding (1991), Creswell & Poth (2018), Dencik et al. (2019, data ethics in human rights research).
