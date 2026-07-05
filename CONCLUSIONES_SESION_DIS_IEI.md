# Conclusiones de la sesión — DIS/IEI tri-corpus + IAA (para ajustar capítulos)

Fecha: 2026-07-05
Autora: Mireya Camacho Celis — Tesis CFH, defensa agosto 2026

---

## 1. Decisión de diseño: fórmulas definitivas de los índices (OPCIÓN A)

Los tres índices sintéticos del framework CFH quedan así (los tres NO comparten
ningún indicador → dimensiones conceptualmente separadas):

- **DIS** (Injusticia Discursiva — superficie del lenguaje, Galtung):
  `DIS = 0.40·EBI_z + 0.30·SA_z + 0.30·(1−REP_z)`
  Mecanismos discursivos: eufemismo bélico (y1), supresión de agentividad (y2),
  y ausencia de ruptura epistémica positiva (1−y10). REP se ubica aquí porque
  fue anotado como marcador discursivo léxico-textual.

- **IEI** (Injusticia Epistémica — daño al sujeto como conocedor, Fricker):
  `IEI = 0.40·MAFAPO_z + 0.30·CIDH_z + 0.30·NV_z`
  Distancia semántica al léxico de víctimas (y8), a estándares DDHH (y9), y
  negación de victimización (y4). NV se ubica aquí porque negar a la víctima su
  condición es injusticia testimonial (credibility deficit de Fricker).

- **ICM** (Congruencia Multimodal, solo Corpus C): sin cambios.

**Justificación de la Opción A (IEI sin REP) sobre la Opción B (IEI con REP):**
doble respaldo. (1) Teórico: DIS e IEI no comparten indicadores → dimensiones
limpias. (2) Empírico: la correlación DIS–IEI es más baja en la Opción A
(ρ=0.164) que en la B (ρ=0.327); incluir REP en ambos generaba correlación
artificial.

## 2. Hallazgo empírico: el IEI captura la brecha epistémica justicia ordinaria vs JEP

Medias por corpus (normalización z-score+sigmoide conjunta A+B+C, n=946):

| Índice | A (ordinaria) | B (JEP escrito) | C (JEP oral) |
|--------|---------------|-----------------|--------------|
| DIS    | 0.510         | 0.498           | 0.394        |
| IEI    | 0.513         | 0.353           | 0.370        |

- **IEI:** A (0.513) >> B (0.353) ≈ C (0.370). La justicia ordinaria presenta
  mayor injusticia epistémica; la JEP (escrita y oral) la reduce ~0.15 puntos.
  Esto confirma la hipótesis central: los mecanismos transicionales corrigen la
  injusticia epistémica del archivo judicial ordinario.
- **DIS:** transversal entre A y B (0.510 vs 0.498); solo el registro oral (C)
  la reduce. La injusticia discursiva de *superficie* (gramática, eufemismo) es
  una convención del género jurídico compartida por ambos sistemas escritos.

**Interpretación teórica:** la corrección de la injusticia epistémica (adoptar
el léxico de las víctimas, dejar de negar su condición) y la de la injusticia
discursiva superficial (gramática institucional) son fenómenos DISOCIADOS. La
JEP corrige la primera pero conserva convenciones de la segunda.

## 3. Hallazgo metodológico: los tres índices miden dimensiones distintas

Matriz de correlaciones (Spearman):

| Par            | ρ           |
|----------------|-------------|
| DIS – IEI      | 0.164       |
| DIS – ICM      | −0.19       |
| IEI – ICM      | −0.11       |

Ninguna correlación supera 0.20 en valor absoluto. Los tres índices capturan
fenómenos independientes: discursivo (DIS), epistémico (IEI) y multimodal (ICM).
Esto valida empíricamente la arquitectura multi-dimensional del framework: no
son medidas redundantes de una misma "injusticia", sino tres dimensiones
separables. Es la evidencia cuantitativa del argumento dimensional del framework.

## 4. Pesos, parsimonia y significancia — estatuto metodológico de los índices

Esta sección es CRÍTICA para la defensa: aclara qué se puede y qué NO se puede
afirmar sobre los índices, y blinda el trabajo ante preguntas del jurado.

### 4.1 Los pesos son teóricos, no optimizados a los datos

Los pesos de DIS e IEI se establecieron según la relevancia teórica de cada
indicador dentro de su dimensión —los mecanismos de violencia cultural
(Galtung, 1990) para el DIS y las formas de injusticia epistémica (Fricker,
2007) para el IEI—, y NO mediante optimización estadística sobre los datos.

Esto es una FORTALEZA, no una limitación, y así debe presentarse:
- Evita el sobreajuste (overfitting) de los pesos a la muestra concreta.
- Preserva la interpretabilidad teórica: cada peso refleja una decisión
  conceptual justificable, no un artefacto de maximización.
- Impide la objeción de "haber elegido los pesos que daban significativo".

Redacción sugerida para el documento:
> "Los pesos de los índices sintéticos se establecieron según la relevancia
> teórica de cada mecanismo dentro de su dimensión, no mediante optimización
> sobre los datos. Esto evita el sobreajuste y preserva la interpretabilidad
> del índice."

### 4.2 El análisis de parsimonia fue DIAGNÓSTICO, no criterio de selección

Se realizó una búsqueda exhaustiva de combinaciones de pesos (grid search) que
midió, para cada combinación, el tamaño del efecto (d de Cohen) y la
significancia (Mann-Whitney U) de la separación A vs B. PERO este análisis se
usó como herramienta DIAGNÓSTICA —para entender qué indicador aporta más
varianza inter-corpus—, NO como criterio para elegir los pesos finales.

Hallazgos del diagnóstico de parsimonia (útiles como evidencia, no como
justificación de pesos):
- El Eufemismo Bélico-Institucional (y1/EBI) es el mecanismo que MÁS separa
  A vs B (correlación peso→d de Cohen = +0.970).
- La Supresión de Agentividad (y2/SA) y la Negación de Victimización (y4/NV)
  son transversales al género jurídico (correlación peso→d NEGATIVA: restan
  separación).
- Esto es coherente con el hallazgo del Cap. 5 de que SA no difiere entre
  corpus (A=0.885, B=0.913, p=0.934 n.s.).

Redacción sugerida:
> "El análisis de parsimonia se empleó como herramienta diagnóstica para
> caracterizar la contribución de cada indicador a la varianza inter-corpus
> —confirmando que el Eufemismo Bélico-Institucional es el mecanismo más
> discriminante mientras que la Supresión de Agentividad es transversal al
> género jurídico—, no como criterio de selección de pesos."

### 4.3 Sobre la significancia: qué SÍ y qué NO se puede afirmar

LO QUE NO SE PUEDE AFIRMAR:
- Que el DIS separa "significativamente" los corpus (p<0.05). La parsimonia
  mostró que casi ninguna combinación de pesos alcanza significancia, debido
  al tamaño limitado del Corpus B.
- Que los pesos son "óptimos". No lo son ni pretenden serlo.

LO QUE SÍ SE PUEDE AFIRMAR (y es teóricamente más fuerte):
- Los índices son medidas DESCRIPTIVO-DIMENSIONALES, no discriminadores
  estadísticos. Su validez no depende de la significancia de la separación
  entre corpus, sino de:
  (a) su INDEPENDENCIA DIMENSIONAL (correlaciones inter-índice <0.33), y
  (b) su COHERENCIA TEÓRICA (anclaje en Galtung/Fricker/Zehr).
- El IEI muestra una "diferencia SUSTANCIAL en la DIRECCIÓN PREDICHA" entre
  justicia ordinaria (0.513) y JEP (0.353). Nótese: "sustancial en la dirección
  predicha", NO "significativa" — esta precisión de lenguaje es la que blinda
  el trabajo ante el tamaño de B.

PRECISIÓN DE LENGUAJE CLAVE (usar siempre):
- Correcto: "diferencia sustancial en la dirección predicha"
- Correcto: "coherente con la hipótesis de corrección epistémica"
- Evitar: "diferencia significativa" / "el índice separa significativamente"
- Evitar: "pesos óptimos"

### 4.4 Respuesta preparada para el jurado

Si preguntan "¿por qué el DIS no separa los corpus?":
> "Porque el DIS no está diseñado para discriminar corpus, sino para medir la
> injusticia discursiva de superficie —gramática y eufemismo—, que es
> transversal al género jurídico escrito. Lo que separa a la justicia ordinaria
> de la JEP es la dimensión epistémica (IEI), y esa disociación es precisamente
> el hallazgo: la JEP corrige el daño epistémico a las víctimas sin alterar las
> convenciones discursivas del lenguaje jurídico."

Si preguntan "¿cómo eligieron los pesos?":
> "Por coherencia teórica con los marcos de Galtung y Fricker, no por
> optimización sobre los datos. El análisis de parsimonia se usó solo para
> diagnosticar la contribución de cada indicador, no para fijar los pesos, lo
> que evita el sobreajuste y mantiene la interpretabilidad del índice."

### 4.5 Párrafo formal listo para insertar (Cap. 5 o Cap. 6)

> Los pesos de los índices sintéticos DIS e IEI se establecieron según la
> relevancia teórica de cada indicador dentro de su dimensión —los mecanismos
> de violencia cultural (Galtung, 1990) para el DIS y las formas de injusticia
> epistémica (Fricker, 2007) para el IEI—, y no mediante optimización
> estadística sobre los datos. El análisis de parsimonia realizado se empleó
> como herramienta diagnóstica para caracterizar la contribución de cada
> indicador a la varianza inter-corpus —confirmando que el Eufemismo
> Bélico-Institucional (y₁) es el mecanismo más discriminante, mientras que la
> Supresión de Agentividad (y₂) es transversal al género jurídico—, no como
> criterio de selección de pesos. Esta decisión evita el sobreajuste y preserva
> la interpretabilidad teórica de los índices.
>
> Es importante precisar el estatuto de los tres índices: DIS, IEI e ICM no
> fueron diseñados como discriminadores estadísticos entre corpus, sino como
> medidas descriptivas de tres dimensiones distintas de la injusticia. Su
> validez no depende de la significancia de la separación entre corpus
> —limitada por el tamaño del Corpus B—, sino de su independencia dimensional,
> evidenciada por correlaciones inter-índice inferiores a 0.33 (Spearman). El
> IEI muestra una diferencia sustancial en la dirección predicha entre la
> justicia ordinaria (0.513) y la JEP (0.353), coherente con la hipótesis de
> corrección epistémica; el DIS, en cambio, resulta transversal entre los
> sistemas escritos, lo que sugiere que la injusticia discursiva de superficie
> constituye una convención del género jurídico compartida.

## 5. Fiabilidad inter-anotador (IAA) — dos rondas

- **Ronda 1 (anotación independiente, 100 fragmentos):**
  EBI κ=0.722, SA κ=0.617, NV κ=0.709 (los tres sustanciales);
  REP κ=0.309 (aceptable, el más bajo). Global macro = 0.589 (moderado).
- **Ronda 2 (revisión de los 34 desacuerdos de REP):** tras calibrar la
  definición operativa del constructo de Ruptura Epistémica Positiva, ambos
  anotadores revisaron los fragmentos en disputa. 28/34 convergieron; 6/34
  siguen en desacuerdo (casos genuinamente ambiguos, no forzados).
  REP κ=0.841 (casi perfecto). **Global macro = 0.722 (sustancial).**

El REP tuvo la menor concordancia inicial porque la frontera entre
reconocimiento genuino y lenguaje jurídico formulaico (p. ej. "homicidio en
persona protegida" en secciones RESUELVE) es interpretativamente difusa. La
calibración del constructo resolvió la mayoría de los desacuerdos.

## 6. Fortalecimiento del Corpus B

El Corpus B pasó de 54 secciones (cuyo texto segmentado no estaba persistido y
salía en 0.000 en todos los indicadores) a **80 secciones reales** segmentadas
de 14 documentos únicos por secciones temáticas (RECONOCIMIENTO 14,
HECHOS_Y_CONDUCTAS 14, CALIFICACION_JURIDICA 13, PATRONES_MACROCRIMINALES 11,
SANCIONES_PROPIAS 11, CONSIDERACIONES 8, RESUELVE 6, CUERPO 3). Los 6
indicadores (y1, y2, y4, y8, y9, y10) fueron recalculados sobre texto real.

## 7. Ajustes pendientes en el documento de tesis

- **Cap. 3 (Marco Teórico):** documentar la ubicación de REP en DIS (marcador
  discursivo) y NV en IEI (injusticia testimonial de Fricker).
- **Cap. 5 (Resultados):** reemplazar la tabla DIS/IEI con las medias nuevas
  (n=946); añadir la matriz de correlaciones DIS/IEI/ICM como evidencia de las
  tres dimensiones; reportar IAA de dos rondas (global 0.722); insertar el
  párrafo formal de §4.5 sobre pesos/parsimonia/significancia.
- **Cap. 6 (Discusión):** integrar la disociación DIS–IEI (la JEP corrige lo
  epistémico pero no lo discursivo superficial) como hallazgo central; discutir
  que DIS es transversal al género jurídico escrito.

## 8. Pendiente técnico: SEM MG (multigrupo)

El DIS/IEI descriptivo está cerrado. Falta el **SEM multigrupo (MG-SEM)** que
modele formalmente la estructura entre corpus. Consideraciones para ese paso:
- Desbalance de unidades: A=819, B=80, C=47. Manejar con submuestreo de A o
  con estimador robusto.
- Definir la pregunta: (a) invarianza de medición (¿DIS/IEI significan lo mismo
  en A/B/C?) o (b) diferencias estructurales entre corpus.
- Con n por grupo tan dispar, evaluar si un enfoque no paramétrico es más
  defendible que el MG-SEM con latentes.

---

# ANEXO — Bloque estructural (SEM exploratorios) — sesión 2026-07-05 (cont.)

Los siguientes análisis SEM son EXPLORATORIOS y complementan la evidencia
principal (DIS/IEI descriptivo entre corpus). No la sustituyen. Se documentan
para los Cap. 5 y 6.

## A. SEM de C (n=47) — ruta injusticia → transición

Modelo con η1 medido por un solo indicador (y8, para evitar el colapso
y8~y9=0.91) y η2 = transición (y10+y11). Resultado: β(y8→transición) nominal
pero NO robusto; el CFI resulta imposible (>1), confirmando que con n=47 el
modelo no se identifica. **Conclusión: el SEM de C es exploratorio**; la
limitación es el tamaño muestral (requeriría n≥150 comparecientes, es decir,
más audiencias transcritas).

## B. MG-SEM (DIS → IEI por corpus) — dos versiones

- **v2a (todos):** A n=819 β=+0.130 (p<0.001); B n=80 β=+0.198 (p=0.071);
  C n=47 β=+0.182 (p=0.204). Rango entre grupos: 0.068.
- **v2b (muestra equilibrada A=80):** A β=+0.047 (p=0.67); B β=+0.198;
  C β=+0.182. Rango: 0.151.

**Lectura:** la relación DIS→IEI es DÉBIL en los tres corpus (β<0.2). Esto
refuerza la independencia dimensional: si DIS predijera fuertemente IEI, no
serían dimensiones distintas. Al equilibrar la muestra, el β de A cae a 0.047,
mostrando que el n grande le daba significancia a una relación trivial.
**El MG-SEM es exploratorio** (falta invarianza configural/métrica, n
balanceado y misma unidad de análisis: A/B por sección, C por compareciente).

## C. SEM de C con los tres índices (DIS, IEI, ICM) — tres estructuras

Se compararon tres estructuras teóricas:
- Opción 1 (cadena IEI~DIS; ICM~DIS+IEI): todos los β no significativos
  (p>0.20); CFI imposible → modelo no identificado.
- Opción 2 (ICM predictor): ídem, β no significativos, ajuste pobre.
- Opción 3 (dimensiones paralelas): DIS~~IEI r=+0.18; DIS~~ICM r=−0.20;
  IEI~~ICM r=−0.17 (todas n.s., |r|<0.21).

**Conclusión:** no hay relaciones causales entre los tres índices. La estructura
de tres dimensiones paralelas independientes es la que describe los datos.
Confirma el argumento dimensional del framework: DIS, IEI e ICM capturan tres
dimensiones empíricamente separables de la injusticia (discursiva, epistémica,
multimodal).

## D. HALLAZGO — indicadores individuales: y8 predice la transición (y10)

Chequeo complementario con las variables individuales (no los índices),
regresión `y10_rep ~ y1+y2+y4+y8+y9`, en dos universos:

- **Corpus C (n=47):** y8_mafapo→y10 β=−0.679 (p=0.047 *); y4_nv→y10
  β=+0.335 (p=0.024 *). Los demás n.s.
- **Tri-corpus (n=946):** y8_mafapo→y10 β=−0.597 (p<0.001 ***);
  y9_cidh→y10 β=+0.363 (p<0.001 ***).

**Hallazgo central:** la distancia semántica al lenguaje de las víctimas (y8,
centroide MAFAPO) predice NEGATIVAMENTE la transición epistémica (y10/REP), de
forma robusta y CONSISTENTE EN AMBOS UNIVERSOS (β≈−0.6, mismo signo). Cuanto más
lejos está el discurso del lenguaje de las víctimas, menor es la ruptura
epistémica positiva.

Esta relación queda ATENUADA al agregar los indicadores en el índice IEI
(donde y8 se promedia con y9 y NV), pero EMERGE nítida a nivel de indicador.
Interpretación: los índices son óptimos para el argumento dimensional (miden
dimensiones distintas); el indicador individual y8 es más sensible para
capturar la relación injusticia→transición.

Nota metodológica: el CFI sigue siendo imposible (modelo saturado); lo
interpretable aquí son los COEFICIENTES DE REGRESIÓN (β, p), no el ajuste
global. La colinealidad y8~y9 (r=0.91 en C, 0.86 en ABC) confirma la decisión
de usar solo y8 en el SEM de C.

Redacción sugerida para Cap. 5/6:
> El análisis por indicadores individuales revela que la distancia semántica al
> lenguaje de las víctimas (y₈, centroide MAFAPO) predice negativamente la
> transición epistémica (y₁₀/REP) de forma robusta y consistente en el Corpus C
> (β=−0.68, p=0.047) y en el tri-corpus (β=−0.60, p<0.001). Esta relación,
> atenuada al agregar los indicadores en el índice IEI, emerge nítidamente a
> nivel de indicador: a mayor distancia del discurso respecto al lenguaje de las
> víctimas, menor ruptura epistémica positiva. El hallazgo respalda la hipótesis
> central del framework —la injusticia epistémica se asocia inversamente con la
> capacidad de transición reparadora— y complementa el análisis dimensional.

## E. Síntesis del bloque estructural (para la defensa)

La evidencia estructural opera en DOS niveles complementarios:
1. **Nivel índice (dimensional):** DIS, IEI, ICM son tres dimensiones
   independientes (correlaciones <0.33). Los SEM causales fracasan porque no hay
   causalidad entre dimensiones: son paralelas.
2. **Nivel indicador (relacional):** la distancia a las víctimas (y8) predice la
   transición (y10) de forma robusta (β≈−0.6, p<0.05 en ambos universos).

Ambos niveles son EXPLORATORIOS (n=47 en C) y complementan —no sustituyen— la
evidencia principal: el contraste inter-corpus del IEI (justicia ordinaria 0.51
vs JEP 0.35). La fortaleza del CFH está en (a) medir tres dimensiones distintas,
(b) el contraste epistémico entre corpus, y (c) la relación y8→y10 a nivel de
indicador. NO está en un SEM confirmatorio con n pequeño.

## F. Jackknife de Dabeiba — chequeo de robustez (complementario)

El subcaso Dabeiba (n=12) mostró en el análisis por subcaso una correlación
fuerte injusticia~transición (r=−0.748), en la dirección predicha. Por el n
pequeño, se aplicó jackknife (recalcular quitando un compareciente a la vez).

Resultado:
- Correlación completa: −0.748; media jackknife: −0.726.
- 11 de 12 iteraciones en rango [−0.738, −0.812] → estable y mismo signo.
- Al excluir un caso (Carlos Andrés Caravalli), cae a −0.315 (delta +0.433).
- Veredicto: MODERADA. No cambia de signo (no es artefacto de un caso), pero
  un compareciente tiene peso desproporcionado.

**Cómo reportarlo (honesto):**
> En el subcaso Dabeiba (n=12), la relación entre injusticia epistémica y
> transición reparadora resultó fuerte (r=−0.748). El análisis de sensibilidad
> por jackknife confirmó que la dirección y magnitud se mantienen estables al
> excluir cualquiera de once comparecientes (rango −0.738 a −0.812), aunque la
> exclusión de un caso individual la reduce a −0.315. Dado el reducido tamaño
> muestral, el hallazgo se interpreta como sugerente y exploratorio, coherente
> con la hipótesis pero no confirmatorio.

**Estatuto:** Dabeiba es un complemento exploratorio, NO un pilar de la tesis.
La evidencia principal sigue siendo la relación y8→y10 (β≈−0.6, robusta en C y
tri-corpus) y el contraste inter-corpus del IEI (ordinaria 0.51 vs JEP 0.35).
El jackknife confirma que fue correcto no apoyar la tesis en Dabeiba.
