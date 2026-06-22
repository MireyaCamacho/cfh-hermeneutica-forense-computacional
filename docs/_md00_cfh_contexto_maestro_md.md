# CFH — Documento maestro de contexto

> Documento ancla del Project. Síntesis ejecutiva de la tesis para que cualquier chat nuevo se oriente sin re-explicación. Actualizar cuando cambien decisiones metodológicas, resultados consolidados o el horizonte de defensa.

**Última actualización:** 20 de junio de 2026
**Fuentes:** Cap2-V7 + Cap3-V10 + Cap4-V10 + Cap5-V22 + Cap6-V17 + sesión de trabajo 20/06/2026.

---

## 1. Identidad del proyecto

- **Autora:** Mireya Camacho Celis — pregrado en Ciencia de Datos, Universidad Externado de Colombia.
- **Email:** mireyacamachocelis@gmail.com
- **Título definitivo:** *El lenguaje de los falsos positivos: medición computacional multimodal de la injusticia discursiva y epistémica en el archivo judicial colombiano.*
- **Director:** Julián Zuluaga. **Entrega revisada al director:** 1 de julio de 2026. **Defensa:** agosto 2026.
- **Repositorio:** `github.com/MireyaCamacho/cfh-hermeneutica-forense-computacional` (rama main)
- **Idioma de trabajo:** español (paper final en inglés).

## 2. Pregunta de investigación

¿Es posible medir, computacionalmente y de forma reproducible, la *injusticia discursiva* y la *injusticia epistémica* presentes en el archivo judicial colombiano sobre los falsos positivos, mediante un conjunto de indicadores lingüísticos y multimodales derivados de embeddings contextuales en español, prosodia y expresión facial?

## 3. Horizonte y alcance

- **Defensa:** agosto 2026 — alcance **H3** (framework CFH formalizado + dataset anotado + paper en inglés).
- **H4 (post-defensa, 24 meses):** 5 papers derivados. Decisión del 25/04/2026; H4 NO es alcance de defensa.

## 4. Marco teórico — 5 niveles articulados

1. **Filosófico-político:** Galtung (1990) violencia cultural; Fraser (1995, 2008) justicia como reconocimiento; Habermas (1987) colonización sistémica del lenguaje.
2. **Restaurativo:** Zehr (2002). Introduce la distinción alta/baja congruencia multimodal (clave para Capa 3).
3. **Epistémico:** Fricker (2007), Dotson (2011), Medina (2013), Pohlhaus (2012). Articulado con crítica decolonial (Santos 2009; Castillejo-Cuéllar 2016) y filosofía colombiana (Wolf 2022; Páez & Matida 2022). Rojas-Andrade et al. (2025) es el único antecedente que aplica Fricker al Macrocaso 003.
4. **Formal-computacional:** framework CFH de tres capas + DIS Score + IEI.
5. **Multimodal:** ICM tri-canal v2 (verbal + vocal + facial), extensión original a Fricker (que opera monomodal). Antecedente: Dryzek & Niemeyer (2024) DRI por encuesta deliberativa — metodológicamente próximo pero diferente en unidad de análisis, método y corpus.

**Soportes NLP del marco:** Sap et al. (2017) → y₂ SA. Mendelsohn et al. (2020) → y₄ NV. Beach et al. (2021) → y₁₁/y₁₂/y₁₃.

## 5. Framework CFH — 3 capas analíticas

| Capa | Dimensión | Pregunta | Indicadores |
|------|-----------|----------|-------------|
| **1 — Léxica** | ¿Cambian las palabras? | Mecanismos de violencia discursiva en texto | EBI, SA, NV, REP, persona gramatical, hedging, léxico emocional, y₁₁/y₁₂/y₁₃ Beach |
| **2 — Semántica** | ¿Cambia el significado? | Distancia semántica al polo de las víctimas | Dist. MAFAPO (y₈), Dist. CIDH (y₉), surprisal (y₇) |
| **3 — Multimodal** | ¿Es congruente? | Congruencia texto/voz/rostro en audiencias orales | OpenSMILE eGeMAPS + OpenFace 3.0 / MediaPipe FaceLandmarker + ICM tri-canal v2 |

**Nota herramientas faciales:** OpenFace 3.0 (Baltrusaitis et al., 2025) para Catatumbo, Casanare, Dabeiba, Huila. MediaPipe FaceLandmarker (Lugaresi et al., 2023) para Costa Caribe. Los blendshapes de MediaPipe se mapean a AUs equivalentes.

## 6. Índices sintéticos (3)

- **DIS Score:** `0.35×y₂_norm + 0.35×y₄_norm + 0.30×(1−y₁₀_norm)`. Mide injusticia discursiva (gramática + léxico + ausencia de reparación). **Menor es mejor.**
- **IEI:** `0.35×y₈_norm + 0.20×y₉_norm + 0.25×y₄_norm + 0.20×(1−y₁₀_norm)`. Mide el hermeneutical gap y el credibility deficit (Fricker). **Menor es mejor.**
- **ICM tri-canal v2:** `0.40×ICM_facial + 0.40×ICM_vocal + 0.20×ICM_verbal_v2`. donde `ICM_verbal_v2 = 0.60×ICM_verbal_v1(y₈) + 0.40×y₁₁_prop_MAFAPO`. **Mayor es mejor** (mayor congruencia multimodal).
- **Lenguaje canónico:** "alta/baja congruencia multimodal" — nunca "genuino/performativo" en tablas y resultados.

## 7. Decisiones metodológicas vigentes

- **Mann-Whitney U:** análisis **exploratorio**.
- **Path analysis (CFI=0.619, RMSEA=0.437, n=873):** exploratorio — reencuadrado así, no como SEM confirmatorio.
- **β=−5.337 para y₃:** coeficiente NO estandarizado. Varianza extremadamente baja del indicador (rango 0.985–0.992) explica la magnitud.
- **α de Cronbach negativo (DIS α=−0.015):** hallazgo real de multidimensionalidad, no error. DIS e IEI se reportan como perfiles compuestos de ponderación teórica, no escalas psicométricas.
- **Embedding model:** ConfliBERT-Spanish definitivo (no cambiar).
- **CFH-BERT v2:** F1 macro=0.58 (n=100 anotaciones). v3 pendiente con κ>0.80.
- **ICM:** medida de congruencia entre canales comunicativos, no de sinceridad individual (Barrett et al., 2019; Crivelli & Fridlund, 2018; Reisenzein et al., 2013; Sen et al., 2024).
- **Auditoría intersectional facial pendiente** (Buolamwini & Gebru, 2018).
- **Léxico civil** en CFH = léxico de las víctimas (MAFAPO), no del derecho civil.
- **"Léxico civil" en CFH** refiere al léxico de las víctimas (MAFAPO), no al derecho civil.
- **Costa Caribe ICM facial:** disponible con N=32 frames (MediaPipe). No excluido.
- **Guía de Muestreo Multimodal (Zuluaga, 2026):** 3 filtros implementados en `code/cfh_muestreo_capa3_v2.py` — F1 identidad facial (face_recognition, tolerancia 0.55), F2 calidad mínima (ancho ≥ 100px), F3 verificación manual de hablante.

## 8. Corpora — descripción consolidada

| Corpus | Docs | Bloques | Período | Fuente |
|--------|------|---------|---------|--------|
| A-CE | ~151 | 520 | 1994–2021 | Consejo de Estado — reparación directa |
| A-CSJ | ~86 | 299 | 2012–2020 | Corte Suprema Sala Penal — casación |
| B-JEP | 9 | 2.678 | 2021–2024 | Autos y resoluciones SRVR — Macrocaso 003 (N_B=214 secciones target) |
| C-JEP oral | 5 | 588 | 2022–2024 | Audiencias JEP — 38.1h efectivas, 1.17M chars |
| **Total** | **~246** | **4.085+** | **1994–2024** | A + B + C |

**Corpus C — 5 subcasos:** Catatumbo (Cap. Chaparro, rango/subcaso), Costa Caribe (12 comparecientes Batallón La Popa, ICM facial N=32), Casanare (Gral. rango/subcaso), Dabeiba (Oficial, rango/subcaso), Huila (soldados/suboficiales incl. rango/subcaso).
**Nota anonimización:** comparecientes identificados por rango/subcaso en los capítulos (29+10 sustituciones completadas).

**Centroides de referencia ("Memoria de Verdad Triangulada"):**
- **Centroide MAFAPO v4** (20/06/2026): 169 textos en 4 bloques (B1=25 MAFAPO escritos, B2=15 Dejusticia, B3=8 comunicado conjunto mayo 2024, B4=102 segmentos voz directa de víctimas en audiencias JEP transcritas con Whisper large-v3). Ponderación: w=1.8 voz directa, w=1.0 textos escritos. Margen ±7.7%. r(y₈/y₉)=0.638. Guardado: `data/referencias/centroide_mafapo_v4.npy`.
- **Centroide CIDH v3:** 25 fragmentos sentencia Villamizar Durán vs. Colombia (Corte IDH, 2018) + comunicados CIDH.

## 9. Resultados consolidados (snapshot canónico)

**Capas 1+2 — A vs B (Tabla 5.5, 9 indicadores):** 8 de 9 significativos en la dirección predicha. y₂ SA no significativo (transversal al género jurídico). Cohen's d: REP d=0.514***, DIS d=0.335***, SA d=0.257*, NV d=0.324 n.s.

**Indicadores estructurales (resisten control temporal A 2018-2023 vs B):**
- y₈ Dist. MAFAPO: 0.217 vs 0.191 (p=0.001 ***)
- y₉ Dist. CIDH: 0.261 vs 0.235 (p<0.001 ***)

**Path analysis exploratorio:** y₃ distancia léxico-civil es el predictor más fuerte de REP (β=−5.337, no estandarizado, p<0.001). La transición epistémica opera más por adopción del vocabulario de víctimas que por reducción del encubrimiento.

**CFH-BERT v2:** F1 macro=0.58 (vs v1 0.27, +0.31). REP F1=0.77. NV F1=0.32 (+0.28 por weighted loss).

**Parsimonia DIS:** 80/90 combinaciones de pesos ρ≥0.90 (89%). IEI: 259/288 (90%). El hallazgo A vs B no depende de la especificación exacta de pesos.

**ICM tri-canal v2 (Tabla 5.16):**
- Catatumbo: 0.295 — el más bajo
- Casanare: 0.355
- Huila: 0.421
- Dabeiba: 0.490 — el más alto
- Costa Caribe: ICM_vocal=0.465, ICM_facial=0.104 (N=32), ICM_tri=parcial

**Disociación DIS vs. IEI (Tabla 5.14) — valores canónicos:**
- Casanare: DIS=0.808, IEI=0.517 — alta en ambas
- **Catatumbo: DIS=0.110, IEI=0.624, Δ=0.514** — hallazgo paradigmático
- Dabeiba: DIS=0.490, IEI=0.299
- Huila: DIS=0.228, IEI=0.081 — mejor perfil bi-índice
- Costa Caribe: DIS=0.464, IEI=0.231

## 10. Versiones actuales de capítulos (20/06/2026)

| Capítulo | Versión | Contenido clave |
|----------|---------|-----------------|
| Cap 1 — Introducción | V3 | Título corregido con "multimodal y epistémica" |
| Cap 2 — Estado del Arte | V7 | Dryzek & Niemeyer (2024), LegalEye, Goupil corregido, fila Dryzek en Tabla 2.1, LLMs justificación ConfliBERT |
| Cap 3 — Marco Teórico | V10 | Fricker §3.5, centroide v4 en §3.8, OpenFace/MediaPipe diferenciados, ERLACS movida a Referencias |
| Cap 4 — Metodología | V10 | Guía muestreo 3 filtros en §4.5.4.1, herramientas faciales por subcaso |
| Cap 5 — Resultados | V22 | Cohen's d N_B=214, sensibilidad pesos, correlación y₄/y₁₀, Costa Caribe N=32 |
| Cap 6 — Discusión | V17 | Normalización percentil §6.3.1, citas movidas a Referencias |
| **Consolidado** | **v1** | **CFH_Tesis_Consolidada_v1.docx — 6 caps integrados** |

## 11. Estado de observaciones del director (20/06/2026)

**✓ Cerradas (25 de 27):**
1.1 Cifras inconsistentes ✓ | 1.2 Auto_128 + duplicados (OCR pendiente Colab) | 1.3 Subcaso Meta ✓ | 2.2 Normalización percentil ✓ | 3.1 N_B=214 ✓ | 3.2 Centroide v4 ✓ | 3.3 Cohen's d ✓ | 4.2 Correlación y₄/y₁₀ ✓ | 4.3 Sensibilidad pesos ✓ | 5.1 SEM→path analysis ✓ | 6.1 Guía muestreo 3 filtros ✓ | 6.2 OpenFace/MediaPipe ✓ | 6.3 ICM canónico ✓ | 6.4 Costa Caribe N=32 ✓ | 6.5 Congruencia multimodal ✓ | 6.6 Anonimización ✓ | 8.1 Duplicaciones Cap6 ✓ | 8.2 Citas a Referencias ✓ | 8.3 Números erróneos ✓ | 8.4 Fricker en §3.5 ✓ | 9.1 Dryzek+LegalEye ✓ | 10.1 Token revocado ✓ | 10.2 Parsimonia datos reales ✓ | 10.3 Ingesta Capa3 cfh.db ✓ | Goupil ✓ | ERLACS ✓ | Dryzek Tabla 2.1 ✓

**⏳ Pendientes GPU (Colab Pro, mañana):**
- 1.2 OCR `auto-RC-AI-016-2025-dabeiba.pdf`
- 3.1 Embeddings y₈/y₉ corpus B completo (199 bloques)
- 4.1 α Cronbach corpus completo

**⏳ Pendientes segundo anotador:**
- 2.1 y₁ EBI no operativo (esperar κ>0.80)
- 7.1 IAA κ>0.80 — `IAA_segundo_anotador.xlsx` entregado desde 18/05/2026

## 12. Entorno y stack tecnológico

**Entorno:** Windows 11 | conda env `cfh` | Python 3.11 | Ruta: `C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional`

**Bases de datos:** `data/cfh.db` (SQLite) — tablas: indicadores, capa3_facial, capa3_vocal, capa3_icm_tri, capa3_runs. Todos los resultados con `modelo_id` y `run_id`.

**NLP:** ConfliBERT-Spanish-Beto-Cased-v1 | CFH-BERT v2 (F1=0.58) | spaCy | BETO (surprisal)

**Multimodal Capa 3:** Whisper large-v3 (ASR) | pyannote-audio (diarización) | OpenFace 3.0 (Baltrusaitis et al., 2025) para 4 subcasos | MediaPipe FaceLandmarker (Lugaresi et al., 2023) para Costa Caribe | OpenSMILE eGeMAPS v02 (88 features, 10.897 ventanas)

**Corpus C en Drive:** G:\Mi unidad\CHF_Corpus\corpus_c\ (ID carpeta: 1IRoO1L-29Yvg3LvrUT0xGyfcsSmu6kXp) — 16 MP3 + 11 TXT transcritos con Whisper + candidatos_victimas_v4.json (102 segmentos)

**Scripts clave del repo:**
- `code/cfh_muestreo_capa3_v2.py` — 3 filtros + agregación por turno
- `code/cfh_ingesta_capa3.py` — ingesta Capa 3 a cfh.db
- `code/cfh_parsimonia_pesos_v2.py` — análisis parsimonia con datos reales
- `code/cfh_ocr_corpus_b.py` — OCR Dabeiba 2025 (ejecutar en Colab)

## 13. Próximos pasos críticos

**Mañana (GPU Colab Pro):**
1. OCR `auto-RC-AI-016-2025-dabeiba.pdf` — `python code/cfh_ocr_corpus_b.py`
2. Embeddings ConfliBERT corpus B completo → y₈/y₉ sobre 199 bloques
3. α Cronbach definitivo con corpus completo
4. Centroide v5 ±5% (refinamiento con ConfliBERT sobre 35.085 segmentos transcritos)

**Esta semana:**
5. Segundo anotador — seguimiento IAA κ>0.80
6. Filtro 3 Guía de Muestreo — verificación manual SPEAKER_XX por subcaso (requiere revisar TXT en Drive)
7. Emails académicos (Miranda Fricker NYU, Wooseong Yang UIC, Javier Osorio Arizona) — cuando lleguen resultados IAA

**Antes del 1 julio (entrega al director):**
8. Actualizar Cap 5/6 con estadísticos finales GPU
9. Entrega `CFH_Tesis_Consolidada_v2.docx` al director

## 14. Contactos académicos clave

- **Wooseong Yang** — `wyang73@uic.edu` (autor ConfliBERT-Spanish, UIC)
- **Prof. Javier Osorio** — `josorio1@arizona.edu` (University of Arizona)
- **Miranda Fricker** — NYU (outreach pendiente de envío)

## 15. Convenciones de trabajo con Claude

- Responder en **español** salvo que se pida lo contrario.
- Asumir el entorno técnico de la sección 12 — no preguntarlo cada vez.
- Scripts: archivos completos listos para ejecutar, no fragmentos sueltos.
- Mireya escribe con typos frecuentes — interpretar por contexto semántico.
- Para cambios metodológicos importantes: cuestionar antes de implementar.
- ICM o Capa 3 facial: **siempre incluir salvaguardas epistemológicas** (Barrett et al., 2019; Crivelli & Fridlund; Reisenzein et al.; Sen et al.; Buolamwini & Gebru). El ICM mide congruencia entre canales, no sinceridad individual.
- No actualizar estadísticos finales en capítulos hasta correr el pipeline completo en Colab Pro.

---

*Fin del documento maestro. Versiones canónicas: Cap5-V22 para resultados. Cap3-V10 para marco teórico. Cap6-V17 para discusión. CFH_Tesis_Consolidada_v1.docx para entrega.*
