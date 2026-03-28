"# CFH Hermen�utica Forense Computacional" 
# cfh-hermeneutica-forense-computacional
Tesis: Hermenéutica Forense Computacional: SEM + ConfliBERT-Spanish para análisis de justicia transicional colombiana

# Hermenéutica Forense Computacional (CFH)
### Evaluando la justicia transicional colombiana mediante SEM, ConfliBERT-Spanish y reparación algorítmica

**Tesis de pregrado · Ciencia de Datos**
**Autora:** Mireya Camacho Celis
**Director:** [Nombre director]
**Institución:** [Nombre institución]
**Estado:** En desarrollo · Marzo 2026

---

## Resumen

Este proyecto propone el framework de **Hermenéutica Forense Computacional** para medir empíricamente la injusticia discursiva en los archivos judiciales colombianos del fenómeno de los *falsos positivos* (Macrocaso 003 de la JEP). Combina modelos de lenguaje de dominio específico (ConfliBERT-Spanish), modelado de ecuaciones estructurales (SEM) y análisis multimodal para cuantificar la distancia semántica entre el lenguaje de la justicia ordinaria y la voz de las víctimas.

---

## Estructura del repositorio

```
cfh/
├── docs/                      ← Documentación académica (trabajo en progreso)
│   ├── estado_del_arte/       ← Estado del arte por partes
│   ├── sem_model/             ← Especificación del modelo SEM
│   ├── marco_conceptual/      ← Framework teórico
│   └── metodologia/           ← Diseño metodológico
│
├── code/                      ← Código fuente (módulos validados)
│   ├── src/ingestion/         ← Pipeline de ingesta (✅ completo, 32 tests)
│   └── tests/
│
└── resources/                 ← Referencias y recursos del proyecto
```

---

## Componentes principales

### Framework teórico
- **Justicia Discursiva** — Habermas (1981/1987), Fraser (1995, 2008)
- **Violencia Cultural** — Galtung (1969, 1990)
- **Justicia Restaurativa** — Zehr (2002), Braithwaite (1989)
- **Ingeniería de la Paz** — Moro et al. (2022)

### Modelo SEM (4 variables latentes)

| Variable | Tipo | Descripción |
|---|---|---|
| ξ₁ Violencia Discursiva | Exógena | Eufemismos, supresión de agentividad, negación de victimización |
| ξ₂ Contexto Institucional | Exógena | Sistema de justicia (A/B/C) y período temporal |
| η₁ Injusticia Discursiva | Endógena | DIS Score — distancia al polo normativo MAFAPO+CIDH |
| η₂ Transición Epistémica | Endógena | Convergencia hacia semántica de paz |

### Corpus (3 tipos)
- **Corpus A** — 280 sentencias Justicia Ordinaria (2002–2008)
- **Corpus B** — Autos escritos JEP (Macrocaso 003)
- **Corpus C** — Audiencias de reconocimiento JEP (muestra, corpus oral)

### Stack tecnológico
- `ConfliBERT-Spanish` (Yang et al., 2023) → motor NLP principal
- `semopy` → estimación CB-SEM
- `Whisper large-v3` → ASR corpus C
- `OpenSMILE eGeMAPS` → indicadores acústicos
- `MLflow` → tracking de experimentos
- `DVC` → versionado de datos

---

## Estado del proyecto

| Componente | Estado |
|---|---|
| Marco teórico (Partes I–IV) | En revisión |
| Corrección ConfliBERT-Spanish | ✅ Completado |
| Modelo de medición SEM v2.0 | ✅ Completado |
| Módulo de ingesta (Corpus A/B/C) | ✅ Completado · 32 tests |
| Pipeline ASR Corpus C | Pendiente |
| Fine-tuning CFH-BERT | Pendiente |
| Extracción features y1..y12 | Pendiente |
| Estimación SEM completa | Pendiente |

---

## Hipótesis central

**H₃ (principal):** El coeficiente de ruta β₂₃ (η₁ → η₂) es estadísticamente significativo (p < .01), indicando que el grado de injusticia discursiva en el archivo judicial ordinario predice la magnitud de la transición epistémica requerida en los mecanismos transicionales.

---

## Referencias clave

- Yang, W. et al. (2023). ConfliBERT-Spanish. *IEEE CiSt 2023*.
- Hu, Y. et al. (2022). ConfliBERT. *NAACL 2022*.
- Kline, R.B. (2023). *Principles and Practice of SEM* (4th ed.).
- JEP (2021). Auto No. 019 — Macrocaso 003.
- Fraser, N. (1995). From redistribution to recognition. *NLR*.
- Galtung, J. (1990). Cultural violence. *JPR*, 27(3), 291–305.

---

## Contacto

**Mireya Camacho Celis**
mireyacamachocelis@gmail.com

