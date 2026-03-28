# Sección 2.2 (VERSIÓN CORREGIDA)
## Modelos de lenguaje para conflicto armado: ConfliBERT-Spanish como modelo central del proyecto CFH

---

### Corrección de nomenclatura

Los documentos previos del proyecto usaban indistintamente los nombres "ConfliBERT" y "ConflictBERT-Spanish". La distinción es técnicamente importante: **ConfliBERT-Spanish** (Yang et al., 2023) existe como modelo publicado y verificado; "ConflictBERT" con *t* no corresponde a ningún modelo real. Esta sección emplea exclusivamente la nomenclatura correcta y la cita oficial.

---

### ConfliBERT-Spanish: descripción, corpus y capacidades

**ConfliBERT-Spanish** (Yang, Alsarra, Abdeljaber, Zawad, Delaram, Osorio et al., 2023) es un modelo de lenguaje BERT preentrenado específicamente para el análisis de conflicto político y violencia en español, desarrollado por el grupo *eventdata* de la Universidad de Texas en Dallas — el mismo equipo que publicó el ConfliBERT original en inglés (Hu et al., 2022, NAACL). Fue presentado en el *7th IEEE Congress on Information Science and Technology* (CiSt 2023, diciembre).

**Corpus de preentrenamiento.** El modelo fue entrenado sobre un corpus de 8.275.941 artículos en español extraídos de medios de comunicación, periódicos y fuentes gubernamentales de países hispanohablantes, con énfasis en contenido político y de conflicto de América Latina. El corpus totaliza 2.070.287.605 tokens y 52.485.055 oraciones (promedio de 39,45 palabras por oración). Esta escala supera en un orden de magnitud al corpus de entrenamiento de BETO (Cañete et al., 2020), aunque BETO fue entrenado sobre texto general más diverso.

**Rendimiento empírico.** En evaluación sobre tareas de dominio específico, ConfliBERT-Spanish supera consistentemente a mBERT (cased y uncased) y BETO (cased y uncased) en las tres categorías de tareas evaluadas:

| Dataset | Dominio | Tarea | ConfliBERT-ES | Mejor baseline |
|---|---|---|---|---|
| Huffingtonpost | Política | BC | **89.60** F1 | 88.16 (BETO cased) |
| Protest | Conflicto | BC | **87.25** F1 | 85.54 (BETO uncased) |
| Insight Crime | Crimen | MLC | **77.74** F1 | 75.78 (BETO cased) |
| Protest | Conflicto | MLC | **63.48** F1 | 59.73 (BETO cased) |
| Mx News | Política | NER | **83.96** F1 | 83.36 (BETO cased) |

*BC = clasificación binaria; MLC = clasificación multietiqueta; NER = reconocimiento de entidades nombradas.*

**Características lingüísticas abordadas.** A diferencia de BETO (entrenado sobre texto general), ConfliBERT-Spanish incorpora tratamiento específico de características del español político latinoamericano: la *pro-drop feature* (omisión del sujeto pronominal), la inflexión rica en persona, número, género y tiempo verbal, y el léxico específico de organizaciones armadas, actores estatales y vocabulario de operaciones militares en el contexto latinoamericano.

**Acceso y uso.** El repositorio oficial del equipo eventdata (github.com/eventdata/ConfliBERT-Manual) documenta el modelo y sus variantes. Los pesos del modelo ConfliBERT-Spanish están disponibles para uso académico bajo solicitud al grupo de investigación. Para el proyecto CFH, el acceso se gestiona a través de los canales del grupo eventdata-utd en HuggingFace.

---

### Justificación de ConfliBERT-Spanish como modelo base del proyecto CFH

**Adecuación de dominio.** El corpus de preentrenamiento de ConfliBERT-Spanish incluye fuentes latinoamericanas de política y conflicto — precisamente el dominio del Macrocaso 003. Esto es cualitativamente diferente a usar BETO o RoBERTa-bne: estos modelos fueron preentrenados sobre texto general (noticias, Wikipedia, web crawl), mientras que ConfliBERT-Spanish fue preentrenado sobre texto de conflicto. La distribución del vocabulario del corpus de preentrenamiento ya está alineada con el léxico de las sentencias de justicia ordinaria y los documentos JEP.

**Implicación para los indicadores del SEM.** Los embeddings de ConfliBERT-Spanish representan el espacio semántico del conflicto político latinoamericano, lo que significa que la distancia coseno entre un texto y el polo normativo (MAFAPO + CIDH) calculada con estos embeddings tiene una interpretación sustantiva directa: es la distancia semántica dentro del espacio de significados del conflicto político, no la distancia en un espacio semántico general. Esto fortalece la validez de constructo de los indicadores del modelo SEM.

**Limitación a documentar.** ConfliBERT-Spanish fue evaluado principalmente sobre tareas de clasificación de eventos (CAMEO, InsightCrime, noticias de protesta). No hay evaluaciones publicadas sobre análisis de lenguaje judicial colombiano ni sobre la detección de eufemismos jurídico-militares específicos. Esta limitación es el punto de partida del fine-tuning supervisado del proyecto CFH: el modelo base provee representaciones de dominio; el fine-tuning sobre la taxonomía CFH (EBI, SA, NV, REP) especializa esas representaciones hacia las categorías de violencia discursiva del Macrocaso 003.

---

### Estrategia de fine-tuning para el proyecto CFH

La cadena de entrenamiento queda definida así:

**Etapa 1 — Modelo base:** ConfliBERT-Spanish (Yang et al., 2023). Preentrenado sobre 2B tokens de conflicto político latinoamericano. Esta etapa ya está completada por el equipo eventdata-UTD.

**Etapa 2 — Domain-adaptive fine-tuning sobre corpus CFH no anotado:** Continuar el preentrenamiento con el objetivo MLM (*masked language modeling*) sobre el corpus completo de sentencias del Macrocaso 003 (corpus A), autos JEP (corpus B) y transcripciones de audiencias (corpus C), sin etiquetas. Objetivo: adaptar las representaciones al vocabulario jurídico colombiano específico del período 2002-2022.

**Etapa 3 — Fine-tuning supervisado sobre corpus anotado:** Entrenamiento con supervisión sobre los segmentos anotados con la taxonomía CFH (EBI, SA, NV, REP). Objetivo: capacitar al modelo para identificar los cuatro tipos de violencia discursiva como tarea de clasificación de spans (token classification en formato BIO).

**Denominación del modelo resultante:** CFH-BERT (v1.0) para diferenciarlo en publicaciones y facilitar la citación. La ficha técnica del modelo se publicará junto con el corpus anotado como contribución abierta del proyecto.

---

### Referencias

- Yang, W., Alsarra, S., Abdeljaber, L., Zawad, N., Delaram, Z., Osorio, J., Khan, L., Brandt, P. T., & D'Orazio, V. (2023, December). ConfliBERT-Spanish: A pre-trained Spanish language model for political conflict and violence. In *2023 7th IEEE Congress on Information Science and Technology (CiSt)* (pp. 287–292). IEEE.
- Hu, Y., Hosseini, M., Skorupa Parolin, E., Osorio, J., Khan, L., Brandt, P., & D'Orazio, V. (2022, July). ConfliBERT: A pre-trained language model for political conflict and violence. In *Proceedings of NAACL 2022*. Association for Computational Linguistics.
- Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). Spanish pre-trained BERT model and evaluation data. In *ICLR 2020 Workshop on Better Language Models and Their Implications*.
