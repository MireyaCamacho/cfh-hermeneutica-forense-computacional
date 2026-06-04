# Auditoría de la base de datos `cfh.db`

**Fecha:** 2026-05-08T18:22:02
**BD:** `cfh.db`  
**Tamaño:** 17.39 MB

## Resumen ejecutivo

| Estado | Cantidad |
|--------|---------:|
| ✅ OK     | 67 |
| ⚠️  WARN  | 0 |
| ❌ ERROR  | 0 |
| ℹ️  INFO  | 55 |

**Resultado:** ✅ La BD pasa todos los checks críticos. Los WARN son aspectos a documentar pero no bloqueantes para defensa.

---

## 1. Integridad referencial

- ✅ **bloques con documento_id huérfano** — 0 huérfanos
- ✅ **indicadores con bloque_id huérfano** — 0 huérfanos
- ✅ **indicadores con modelo_id no registrado** — 0 huérfanos
- ✅ **indicadores con run_id no registrado** — 0 huérfanos
- ✅ **segmentos_orales con audiencia_id huérfano** — 0 huérfanos
- ✅ **audiencias con documento_id huérfano** — 0 huérfanos
- ✅ **comparecientes con audiencia_id huérfano** — 0 huérfanos

## 2. Consistencia de datos

- ✅ **documentos: doc_id_externo único** — sin duplicados
- ✅ **bloques: identificador_externo único** — sin duplicados
- ✅ **indicadores: clave (bloque, codigo, run) única** — sin duplicados
- ✅ **y2_sa: rango** — min=0.0000, max=1.0000, mean=0.8625, n=1556
- ✅ **y4_nv: rango** — min=0.0000, max=1.0000, mean=0.2566, n=1556
- ✅ **y10_rep: rango** — min=0.0000, max=1.0000, mean=0.1060, n=1556
- ✅ **y3_civil: rango** — min=0.9520, max=1.0000, mean=0.9891, n=1018
- ✅ **y8_mafapo: rango** — min=0.0961, max=0.3118, mean=0.1911, n=588
- ✅ **y9_cidh: rango** — min=0.1445, max=0.3650, mean=0.2629, n=588
- ✅ **y8_mafapo_cs: rango** — min=0.1193, max=0.4045, mean=0.2100, n=873
- ✅ **y9_cidh_cs: rango** — min=0.1618, max=0.4196, mean=0.2525, n=873
- ✅ **y7_surprisal: rango** — min=0.6216, max=0.7464, mean=0.6844, n=873
- ✅ **documentos.corpus: sin NULLs**
- ✅ **bloques.documento_id: sin NULLs**
- ✅ **indicadores.valor: sin NULLs**
- ✅ **indicadores.codigo: sin NULLs**
- ✅ **indicadores.modelo_id: sin NULLs**
- ✅ **indicadores.run_id: sin NULLs**

## 3. Trazabilidad — runs y modelos

- ✅ **indicadores con modelo trazable** — 31.624/31.624 (100.0%)
- ✅ **indicadores con run trazable** — 31.624/31.624 (100.0%)
- ℹ️ **modelo: Pipeline-Capa1 cap5-v15** — 19.387 indicadores
- ℹ️ **modelo: Pipeline-Beach y11-y12-y13** — 6.696 indicadores
- ℹ️ **modelo: ConfliBERT-Spanish beto-cased-v1** — 4.668 indicadores
- ℹ️ **modelo: BETO cased** — 873 indicadores
- ℹ️ **modelo: CFH-BERT v1** — 0 indicadores
- ℹ️ **modelo: CFH-BERT v2** — 0 indicadores
- ℹ️ **modelo: MediaPipe-FaceLandmarker tasks-2023** — 0 indicadores
- ℹ️ **modelo: OpenSMILE-eGeMAPS v02** — 0 indicadores
- ℹ️ **modelo: Whisper large-v3** — 0 indicadores
- ℹ️ **modelo: pyannote-audio 2.1** — 0 indicadores
- ℹ️ **run #1: Indicadores Corpus A — pipeline cap.5** — 5.733 indicadores · fecha=2026-05-08 22:48:03
- ℹ️ **run #2: Indicadores Corpus B v1 — versión cap.5** — 378 indicadores · fecha=2026-05-08 22:52:36
- ℹ️ **run #3: Indicadores Corpus B v2 — versión actualizada** — 1.015 indicadores · fecha=2026-05-08 22:52:36
- ℹ️ **run #4: Indicadores Corpus C — capas 1 y 2** — 1.176 indicadores · fecha=2026-05-08 22:52:37
- ℹ️ **run #5: Indicadores ConfliBERT-Spanish — distancias semánticas + surprisal** — 4.365 indicadores · fecha=2026-05-08 22:52:39
- ℹ️ **run #6: Indicadores Corpus C Capa 1 — y2/y4/y10** — 1.614 indicadores · fecha=2026-05-08 22:57:49
- ℹ️ **run #7: Indicadores extras — léxico emocional + Beach** — 0 indicadores · fecha=2026-05-08 22:58:33
- ℹ️ **run #8: Segmentos faciales — MediaPipe FaceLandmarker** — 0 indicadores · fecha=2026-05-08 22:58:34
- ℹ️ **run #9: Segmentos vocales — eGeMAPS comparecientes** — 0 indicadores · fecha=2026-05-08 22:58:36
- ℹ️ **run #10: Indicadores extras — léxico emocional + Beach** — 17.343 indicadores · fecha=2026-05-08 23:05:43

## 4. Cobertura por indicador

- ℹ️ **y2_sa** — A=819, B=199, C=538, total=1556
- ℹ️ **y4_nv** — A=819, B=199, C=538, total=1556
- ℹ️ **y10_rep** — A=819, B=199, C=538, total=1556
- ℹ️ **y3_civil** — A=819, B=199, C=0, total=1018
- ℹ️ **y8_mafapo** — A=0, B=0, C=588, total=588
- ℹ️ **y9_cidh** — A=0, B=0, C=588, total=588
- ℹ️ **y8_mafapo_cs** — A=819, B=54, C=0, total=873
- ℹ️ **y9_cidh_cs** — A=819, B=54, C=0, total=873
- ℹ️ **y7_surprisal** — A=819, B=54, C=0, total=873
- ℹ️ **y11_quotes** — A=777, B=60, C=0, total=837
- ℹ️ **y12_judgment** — A=777, B=60, C=0, total=837
- ℹ️ **y13_evidential** — A=777, B=60, C=0, total=837
- ℹ️ **emo_balance_victimas** — A=819, B=0, C=0, total=819
- ℹ️ **accountability_score** — A=819, B=0, C=0, total=819
- ℹ️ **bloques sin indicadores (esperable para granulares B y Beach huérfanos)** — 2658 de 5585 (47.6%)
- ✅ **documentos sin bloques** — 0

## 5. Reproducibilidad del Capítulo 5 v15

- ✅ **Tabla 5.5 — y2_sa (A)** — esperado=0.885, obs=0.8855, err=0.0005, n=819
- ✅ **Tabla 5.5 — y2_sa (B-v1)** — esperado=0.913, obs=0.9132, err=0.0002, n=54
- ✅ **Tabla 5.5 — y4_nv (A)** — esperado=0.239, obs=0.2393, err=0.0003, n=819
- ✅ **Tabla 5.5 — y4_nv (B-v1)** — esperado=0.233, obs=0.2330, err=0.0000, n=54
- ✅ **Tabla 5.5 — y10_rep (A)** — esperado=0.086, obs=0.0861, err=0.0001, n=819
- ✅ **Tabla 5.5 — y10_rep (B-v1)** — esperado=0.153, obs=0.1530, err=0.0000, n=54
- ✅ **Tabla 5.5 — y3_civil (A)** — esperado=0.990, obs=0.9901, err=0.0001, n=819
- ✅ **Tabla 5.5 — y3_civil (B-v1)** — esperado=0.987, obs=0.9866, err=0.0004, n=54
- ✅ **Tabla 5.5 — y8_mafapo_cs (A)** — esperado=0.211, obs=0.2113, err=0.0003, n=819
- ✅ **Tabla 5.5 — y8_mafapo_cs (B-v1)** — esperado=0.191, obs=0.1913, err=0.0003, n=54
- ✅ **Tabla 5.5 — y9_cidh_cs (A)** — esperado=0.254, obs=0.2537, err=0.0003, n=819
- ✅ **Tabla 5.5 — y9_cidh_cs (B-v1)** — esperado=0.235, obs=0.2349, err=0.0001, n=54
- ✅ **Tabla 5.9 — Catatumbo y8_mafapo** — esperado=0.207, obs=0.2068, err=0.0002, n=58
- ✅ **Tabla 5.9 — Catatumbo y9_cidh** — esperado=0.271, obs=0.2707, err=0.0003, n=58
- ✅ **Tabla 5.9 — Costa Caribe y8_mafapo** — esperado=0.189, obs=0.1893, err=0.0003, n=128
- ✅ **Tabla 5.9 — Costa Caribe y9_cidh** — esperado=0.259, obs=0.2594, err=0.0004, n=128
- ✅ **Tabla 5.9 — Casanare y8_mafapo** — esperado=0.193, obs=0.1929, err=0.0001, n=124
- ✅ **Tabla 5.9 — Casanare y9_cidh** — esperado=0.264, obs=0.2639, err=0.0001, n=124
- ✅ **Tabla 5.9 — Dabeiba y8_mafapo** — esperado=0.189, obs=0.1894, err=0.0004, n=144
- ✅ **Tabla 5.9 — Dabeiba y9_cidh** — esperado=0.262, obs=0.2619, err=0.0001, n=144
- ✅ **Tabla 5.9 — Huila y8_mafapo** — esperado=0.186, obs=0.1862, err=0.0002, n=134
- ✅ **Tabla 5.9 — Huila y9_cidh** — esperado=0.263, obs=0.2632, err=0.0002, n=134

## 6. Consistencia del Corpus C

- ✅ **audiencias canónicas** — 5 (esperado 5)
- ✅ **comparecientes registrados** — 8 (esperado 8: Catatumbo 2 + Casanare 1 + Dabeiba 2 + Huila 3)
- ✅ **Capa 2 — Catatumbo** — obs=58, esperado cap.5=58
- ✅ **Capa 2 — Costa Caribe** — obs=128, esperado cap.5=128
- ✅ **Capa 2 — Casanare** — obs=124, esperado cap.5=124
- ✅ **Capa 2 — Dabeiba** — obs=144, esperado cap.5=144
- ✅ **Capa 2 — Huila** — obs=134, esperado cap.5=134
- ✅ **Capa 1 — Catatumbo** — obs=58, esperado cap.5=58
- ✅ **Capa 1 — Costa Caribe** — obs=120, esperado cap.5=120
- ✅ **Capa 1 — Casanare** — obs=120, esperado cap.5=120
- ✅ **Capa 1 — Dabeiba** — obs=120, esperado cap.5=120
- ✅ **Capa 1 — Huila** — obs=120, esperado cap.5=120
- ✅ **Costa Caribe: 0 segmentos (DRM esperado)** — exclusión documentada en cap. 5 §5.9 y cap. 6 §6.3.1
- ✅ **Casanare segmentos** — faciales=780, vocales=354
- ✅ **Catatumbo segmentos** — faciales=1240, vocales=332
- ✅ **Dabeiba segmentos** — faciales=387, vocales=83
- ✅ **Huila segmentos** — faciales=1402, vocales=342

## 7. Exclusiones y limitaciones documentadas

- ℹ️ **y₁ EBI = 0 en todo el corpus** — extractor pendiente — sec. 14 documento maestro, cap. 6 §6.3.2
- ℹ️ **Costa Caribe sin video facial** — DRM YouTube — cap. 5 §5.9, cap. 6 §6.3.1
- ℹ️ **Modelo SEM completo no estimado** — y₇ requiere CFH-BERT v3 con IAA κ>0.80 sobre 500 fragmentos
- ℹ️ **MediaPipe sin auditoría intersectional** — Buolamwini & Gebru (2018) — limitación cap. 6 §6.3.4
- ℹ️ **ICM mide congruencia, no sinceridad** — Barrett et al. (2019), Crivelli & Fridlund (2018) — cap. 3 §3.7.2
- ℹ️ **CFH-BERT v2 F1 macro = 0.58** — n=100 anotaciones; v3 definitivo pendiente

## 8. Metadata de la BD

- ℹ️ **tamaño BD** — 17.39 MB
- ℹ️ **archivo** — C:\PROYECTOS 2026\TESIS 2026\CFH_Hermeneutica_Forense_Computacional\cfh.db
- ℹ️ **tabla corpora** — 6 filas
- ℹ️ **tabla documentos** — 309 filas
- ℹ️ **tabla audiencias** — 5 filas
- ℹ️ **tabla comparecientes** — 8 filas
- ℹ️ **tabla bloques** — 5.585 filas
- ℹ️ **tabla segmentos_orales** — 9.660 filas
- ℹ️ **tabla modelos** — 10 filas
- ℹ️ **tabla runs** — 10 filas
- ℹ️ **tabla indicadores** — 31.624 filas
- ℹ️ **tabla anotaciones** — 84 filas
- ℹ️ **tabla centroides** — 0 filas
- ℹ️ **versión SQLite** — 3.52.0

---

## Notas para defensa

Esta auditoría se ejecuta sobre la BD `cfh.db` y reproduce los valores publicados en el Capítulo 5 v15. La BD es el ÚNICO punto de verdad para el modelo SEM, las tablas del cap. 5 y los análisis de Capa 1, 2 y 3.

**Trazabilidad:** cada indicador está asociado a un `run_id` con fecha y descripción, y un `modelo_id` con nombre y versión. Esto permite reproducir cualquier número del cap. 5 ejecutando una query SQL filtrada por run.

**Auditabilidad:** las exclusiones (Costa Caribe DRM, EBI sin extractor, MediaPipe sin auditoría intersectional) están documentadas explícitamente y referenciadas a las secciones del cap. 5/6 donde se discuten.

**Reproducibilidad:** las medias por corpus reproducen las Tablas 5.5 y 5.9 al cuarto decimal, validando que la BD está en sintonía con los datos publicados en la tesis.