# OBJETIVO — Sprint 1: Cimientos

**Prioridad:** critical
**Deadline:** Próxima sesión de tutoría
**Estimado:** 8-10h

## Contexto

El proyecto ha avanzado rápidamente en implementación (Capa 3 facial, ICM, visualizaciones) pero sin una estructura que garantice reproducibilidad, trazabilidad de experimentos, y consistencia metodológica. Antes de seguir implementando, necesitamos consolidar los cimientos: datos inventariados, pregunta definida, repo organizado, y entorno listo para experimentación sistemática.

## Entregables

- [ ] **E1. Pregunta e hipótesis finales** — Versión acordada con director, operacionalizable, cada término definido
- [ ] **E2. Inventario formal de datos** (`data/README.md`) — Cada corpus con: fuente, tamaño, formato, features extraídos, features pendientes, calidad, limitaciones
- [ ] **E3. Reorganización del repo** — Estructura `data/`, `experiments/`, `cfh/`, `docs/`, `tutoria/`. Scripts de raíz movidos o integrados
- [ ] **E4. Extracción de audio** (.wav) de las 5 audiencias del Corpus C — Prerequisito para prosodia (eGeMAPS)
- [ ] **E5. Tabla de features completa** — Qué se extrae, de qué corpus, con qué herramienta, estado actual (implementado/pendiente/placeholder)

## Criterios de aceptación

- `data/README.md` existe y documenta los 3 corpus con estado real (no planificado)
- Pregunta de investigación tiene máximo 50 palabras y cada concepto es operacionalizable
- No hay scripts .py sueltos en la raíz del repo
- Los 5 archivos .wav de Corpus C existen en `data/raw/corpus_c/`
- Tabla de features tiene columnas: feature, modalidad, corpus, herramienta, estado, referencia

## Lo que NO hacer en este sprint

- No implementar features nuevos
- No correr más experimentos
- No escribir más capítulos de la tesis
- No fine-tunear modelos

## Notas

Este sprint es de **orden y claridad**, no de código. Una semana de estructura vale más que tres semanas de implementación sin rumbo.

Papers de referencia a leer en profundidad durante este sprint:
1. Gutiérrez-Osorio et al. (2025) — JEP + NLP (arXiv:2504.04325)
2. LegalEye — Baldivas et al. (2025) — Multimodal español (MDPI)
3. Cann et al. (2025) — Eco semántico (EPJ Data Science)

---
*Generado por ORBIX — 2026-04-17*
