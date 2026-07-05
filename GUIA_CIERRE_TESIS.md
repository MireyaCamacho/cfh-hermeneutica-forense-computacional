# Guía de cierre de la tesis CFH — de "medir" a "concluir"

**Para:** Mireya Camacho · **De:** Julián Zuluaga (dirección) · **Fecha:** 2 de julio de 2026
**Horizonte:** defensa agosto 2026

Mireya, este documento es el mapa para llevar la tesis desde donde está hoy hasta una
versión completa y defendible. No es una lista de correcciones sueltas: es **un hilo**.
Léelo de corrido una vez y luego trabájalo por pasos. La idea que ordena todo es una sola:

> **La tesis ya sabe medir. Lo que falta es que concluya.**

---

## 1. Dónde estás hoy (con honestidad)

Diste un salto muy grande. El ICM pasó de calcularse "por subcaso" —mezclando canales de
personas distintas— a calcularse **por compareciente individual**, con la cadena de
enganche correcta (marcación × diarización × texto) y 47 comparecientes con unidad limpia.
Eso resolvió el problema estructural más serio que tenía el trabajo. Es trabajo de nivel alto.

Ahora bien, mirando la tesis como un todo, el estado es desparejo entre sus tres planos:

| Plano | Estado | Lectura |
|---|---|---|
| **Estructura** | ~85% | El esqueleto está y el arreglo grande (ICM por persona) ya se hizo. |
| **Metodología** | ~65% | El instrumento está bien diseñado, pero su **validación** tiene huecos: el clasificador sin κ, la Capa 2 semántica en centroides mezclados, y el verbal con un bug. |
| **Resultados** | ~55% | Los de texto son sólidos; **el resultado que responde la pregunta central aún no existe.** |

La trampa es que la tesis *parece* más terminada de lo que está, porque ya está escrita y
tiene tablas y números. Pero **tener números no es tener una conclusión.** Ahí es donde
vamos a trabajar.

---

## 2. La idea que ordena el cierre

Tu pregunta central es: **¿la justicia transicional (JEP) repara la violencia discursiva de
la justicia ordinaria, o la reproduce?**

Tienes las piezas para responderla, pero todavía no las conectaste:
- El plano del **texto** ya muestra algo fuerte: la JEP corrige la *gramática* de la
  violencia (DIS baja) pero la *brecha epistémica* con la voz de las víctimas persiste (IEI
  alta). Eso es la disociación de Catatumbo. **Esa es media respuesta: "repara la forma, no
  el fondo".**
- El plano **multimodal** (el ICM por compareciente) debería completar la respuesta
  mostrando *quién* reproduce esa injusticia y *cómo* — sobre todo si la **jerarquía militar**
  la modula. Pero ese análisis todavía no está hecho.

El cierre consiste en **convertir el instrumento en argumento.** Los pasos de abajo están
ordenados para eso.

---

## 3. El camino de cierre, paso a paso

### Paso 1 — Arreglar el canal verbal *(bloqueante, primero)*

**Qué encontramos.** Tu ICM da `y10_rep = 0.000` a comparecientes que sí reconocen. El caso
claro es **José David Restrepo Solarte** (Huila): su texto dice literalmente *"reconozco… mi
responsabilidad… pido perdón… las víctimas…"* (1.154 tokens), y el extractor le pone **cero**.
Le pasa a 7 comparecientes. Eso es un **falso negativo**: no es un score bajo, es una falla
de detección que corrompe su canal verbal y, por tanto, su ICM.

**Qué hacer.**
1. Entra al `REPExtractor` (`code/src/features/y10_rep_extractor.py`) y reproduce el caso
   Restrepo aislado. Revisa si falla por spaCy (`es_core_news_lg` no cargado), por
   codificación del texto, o porque el patrón de REP no captura la forma en que él lo dice.
2. Corrige y **re-verifica los 7 ceros**: cada uno debe quedar en un valor coherente con su
   texto, o confirmarse que el cero es real (alguien que de verdad no reconoció).
3. Recalcula `icm_tricanal_final.csv` con el verbal corregido.

**Por qué primero.** Todo análisis posterior usa el verbal. Si está roto, el resto se
contamina. Es días de trabajo, no semanas.

---

### Paso 2 — Recalcular la Capa 2 semántica (NLP)

**Qué encontramos.** El motor semántico (ConfliBERT-Spanish) quedó a medio migrar:
- Construiste el **centroide MAFAPO v5** (293 textos, ±5.8%) — muy bien, cerró el punto §3.2
  de la devolución (los polos frágiles de 25 textos).
- Pero **solo migraste el Corpus C a v5.** Los indicadores semánticos y₈ (distancia MAFAPO)
  e y₉ (distancia CIDH) de **Corpus A y Corpus B siguen calculados con el centroide anterior
  (v3b)**. Es decir: comparas la brecha A vs B con **dos reglas de medir distintas**.
- Peor: el **Corpus B semántico solo tiene 54 bloques**, no los ~214 que ya procesaste en la
  Capa 1. Tu hallazgo estrella —"la brecha semántica resiste el control temporal"— está hoy
  sobre **B submuestreado y centroides mezclados.**

**Qué hacer.**
1. **Recalcular y₈/y₉ con el centroide v5 para los tres corpora** (A, B y C), no solo C. Todos
   sobre la misma regla. Tienes el script (`cfh_verificar_y8y9_centroide_v5.py`,
   `cfh_migrar_centroide_v5.py`); extiéndelo a A y B.
2. **Procesar el Corpus B semántico completo** (54 → ~214 bloques), para que y₈/y₉ cubran lo
   mismo que la Capa 1.
3. **Re-correr DIS e IEI** con los y₈/y₉ nuevos (dependen de ellos), y actualizar las tablas
   de resultados y el apéndice.
4. **Surprisal (y₇):** o lo calculas de forma definitiva (BETO log-probs) o lo declaras como
   no incluido en esta versión. Hoy está a medias.
5. **y₁ EBI:** declararlo **no operativo** (F1=0.0 depende de CFH-BERT afinado, que a su vez
   depende de la IAA). No mostrar un 0.0 como resultado — sostener Galtung con y₃ (léxico
   civil) e y₄ (NV), como ya te sugerí en la devolución §2.

**Por qué.** Toda la Capa 2 y los índices DIS/IEI dependen de y₈/y₉. Si A y B están en
centroides distintos, la comparación —que es el corazón cuantitativo de la tesis— es
metodológicamente atacable. Unificar en v5 **blinda el hallazgo estrella.**

---

### Paso 3 — Completar `rango_militar` en los 47

**Qué encontramos.** El campo `rango_militar` está lleno en solo **13 de 47** comparecientes;
los 26 de Huila lo tienen vacío. Sin rango no se puede responder la parte de la pregunta que
habla de la **jerarquía** (¿la estructura militar reproduce la injusticia?).

**Qué hacer.**
1. Completa `rango_militar` para los 47 en las tablas de marcación. La mayoría se
   autoidentifica al hablar ("fui capitán del batallón…") — el rango suele estar en la
   propia transcripción y en los letreros de la JEP.
2. Normaliza a tres niveles para el análisis: **Alto** (General/Coronel/Tte. Coronel),
   **Medio** (Mayor/Capitán/Teniente), **Tropa/Suboficial** (Sargento/Cabo/Soldado).

**Por qué.** Es la variable que convierte 47 mediciones en una **respuesta sobre el poder y
el reconocimiento**. Sin ella, tienes un instrumento sin tesis.

---

### Paso 4 — Segundo calificador (IAA): validar la medición *(en paralelo, YA)*

**Qué es y por qué es distinto.** Esto no es un segundo *jurado* de la tesis: es un **segundo
anotador/calificador** que codifica de forma independiente una muestra de tus segmentos, para
poder calcular el **acuerdo inter-anotador (κ)**. Es lo que demuestra que tu taxonomía CFH es
**reproducible** y no solo tu interpretación personal. Ya venía como pendiente Alto desde la
guía de dirección ("Segundo anotador IAA, mín. 30 textos, kappa por clase") y desde la
devolución (§7.1).

**Qué encontramos.** La taxonomía la anotaste tú sola. Sin κ hay **riesgo de circularidad**:
el clasificador aprende tu criterio y luego "se valida" contra tu mismo criterio. Es el
flanco que un jurado técnico ataca primero.

**Qué hacer.**
1. Consigue el **segundo calificador** (otra persona) y dale una **muestra ciega** de mín. 30
   textos por clase, con las plantillas ya preparadas (`data/CFH_IAA_*.xlsx`).
2. Calcula el **κ de Cohen por clase** (`code/calcular_kappa_iaa.py`). Meta: κ ≥ 0.80.
3. Según el resultado:
   - κ ≥ 0.80 → el clasificador (REP, NV) queda validado; puedes defender resultados con él.
   - κ < 0.80 → se declara con transparencia como **anotación de anotador único, IAA
     limitada**, y se sostienen las conclusiones con los **extractores de reglas** (y₂, y₃,
     y₈, y₉), no con el clasificador. *(Esto también es un cierre válido — lo importante es
     no ocultar el estado.)*

**Por qué en paralelo y ya.** Es lo **único que no depende solo de ti** (necesita otra persona
y su tiempo). Es el cuello de botella de toda la tesis. Arráncalo esta semana y que corra de
fondo mientras haces los pasos 1, 2, 3 y 5.

---

### Paso 5 — El análisis por rango: aquí nace la conclusión

Con el verbal arreglado (paso 1), el NLP unificado (paso 2) y el rango completo (paso 3),
corre **el análisis que responde la pregunta**:

1. **ICM y cada canal por nivel de rango** (Alto / Medio / Tropa), con test (Kruskal-Wallis
   o correlación de Spearman rango↔ICM), sobre N=47.
2. **DIS/IEI vs ICM a nivel individual** — ¿la injusticia epistémica del texto se acompaña de
   incongruencia multimodal?
3. La hipótesis a probar (hoy es solo un indicio en el facial, n=9): **¿a mayor rango, menor
   congruencia / menor reconocimiento?** Si se sostiene, *esa es tu respuesta*: la jerarquía
   militar modula el reconocimiento — la estructura reproduce la injusticia.

**El resultado esperado, redactado como tesis:** *"La JEP repara la gramática de la violencia
(DIS baja) pero la brecha epistémica con la voz de las víctimas persiste (IEI alta), y esa
persistencia se concentra en los comparecientes de mayor rango."* Eso es una conclusión, no
una medición.

---

### Paso 6 — Blindar el canal facial (con/sin)

**Qué pasa.** El facial es tu canal más débil: video en 360p, distress escalado a ojo
(×8), y con frames por persona a veces escasos. No lo elimines —es parte del framework— pero
**no cuelgues la conclusión de él.**

**Qué hacer.** Corre el análisis del paso 5 **dos veces**: tri-canal (con facial) y bi-canal
(solo vocal + verbal). Si la conclusión se sostiene **sin** el facial, quedas blindada: el
canal débil deja de ser un flanco y pasa a ser una confirmación exploratoria.

---

### Paso 7 — Documentar atrición y reproducibilidad

**Qué pasa.** De 12 comparecientes marcados en Catatumbo, solo 3 entraron al ICM (los demás
cayeron por solapamiento colectivo o poca evidencia). Eso es correcto —solo entra atribución
limpia— pero **hoy es invisible**: el `icm_tricanal_final_excluidos.csv` que tu script promete
no existe.

**Qué hacer.**
1. Genera el archivo de excluidos y una tabla de atrición (cuántos comparecientes por
   subcaso: marcados → analizados, y por qué cayeron).
2. Deja la cadena reproducible: los videos no están en el repo; documenta de dónde salen
   (las URLs de YouTube en `corpus_c_videos.txt`) y en qué resolución.

**Por qué.** Un jurado que ve "3 comparecientes en Catatumbo" cuando hubo 12 va a preguntar.
La respuesta honesta y documentada te protege.

---

## 4. Cómo esto arma la tesis final

Cada paso alimenta un capítulo. Este es el hilo:

- **Cap. 4 (Metodología):** la cadena de enganche (marcación × diarización × texto), el
  centroide v5 unificado y el recálculo semántico (paso 2), las escalas absolutas del vocal,
  los pisos de robustez, el **segundo calificador / κ** (paso 4) y la atrición (paso 7).
  Aquí demuestras que **mides bien**.
- **Cap. 5 (Resultados):** la brecha A vs B (texto, ya sólida y ahora sobre centroide único)
  + **el análisis por rango** (pasos 5 y 6). Aquí está el resultado nuevo con N=47.
- **Cap. 6 (Discusión/Conclusión):** la respuesta a la pregunta — "repara la forma, no el
  fondo, y la jerarquía modula". Aquí **cierras el argumento normativo** que el jurado de
  derecho va a exigir.

---

## 5. La narrativa de defensa

Tu historia más fuerte no es "hice un índice perfecto". Es esta:

> *"Medí algo que nadie había medido —la injusticia discursiva y epistémica en el mismo
> universo de hechos entre justicia ordinaria y transicional—; encontré los límites de mi
> propio instrumento; y los corregí."*

La **autocorrección del ICM** (de subcaso a compareciente) no es una debilidad que esconder:
es la prueba de tu madurez metodológica. Cuéntala.

---

## 6. Checklist de cierre

- [ ] **Paso 1** — Verbal corregido + 7 ceros re-verificados + ICM recalculado
- [ ] **Paso 2** — y₈/y₉ recalculados con centroide v5 en A, B y C + Corpus B semántico completo (54→~214) + DIS/IEI re-corrido + y₇/y₁ declarados
- [ ] **Paso 3** — `rango_militar` completo en los 47 + normalizado a 3 niveles
- [ ] **Paso 4** — Segundo calificador (IAA) lanzado + κ por clase calculado *(arrancar ya, corre de fondo)*
- [ ] **Paso 5** — Análisis por rango (ICM y canales) + DIS/IEI vs ICM + la conclusión escrita
- [ ] **Paso 6** — Resultado replicado con/sin facial (bi vs tri-canal)
- [ ] **Paso 7** — Excluidos + tabla de atrición + nota de reproducibilidad
- [ ] Cap. 4/5/6 actualizados con lo anterior
- [ ] Simulacro de defensa con las 3 preguntas duras (fiabilidad, facial, conclusión)

---

**El orden que te recomiendo:** Paso 4 primero (lanzar el segundo calificador, para que corra
de fondo) → Paso 1 (verbal) → Paso 2 (recálculo NLP) → Paso 3 (rango) → Paso 5 (el análisis,
la conclusión) → Pasos 6 y 7 (blindaje) → escritura final de Cap 4/5/6. Con eso, agosto es
alcanzable.

Cualquier duda, la trabajamos en sesión. Vas muy bien.

— Julián
