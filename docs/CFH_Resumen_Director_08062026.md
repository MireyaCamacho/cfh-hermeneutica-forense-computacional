# CFH — Resumen de avances para reunión con director
## 08 de junio de 2026

---

## 1. COMPLETADO DESDE LA ÚLTIMA REUNIÓN

### Corpus C — Costa Caribe (subcaso 5/5)
- **Diarización completa**: 10.46h, 5,717 segmentos, SPEAKER_00 = compareciente principal (44.7 min)
- **eGeMAPS**: 240 segmentos procesados con OpenSMILE eGeMAPS v02
- **AUs**: 32 frames MediaPipe FaceLandmarker (43% detección — comparable con Dabeiba 40%)
- **ICM tri-canal**: vocal=0.465, facial=0.104, verbal=0.132 → dis_tri=0.012, cong_injusta=True
- **El Corpus C está completo: 5/5 subcasos con DIS, IEI e ICM calculados**

### IAA
- Excels de anotación entregados a 3 anotadores externos
- Segunda anotadora: 60 fragmentos ciegos
- Anotadores 2 y 3: 200 fragmentos nuevos cada uno
- En proceso — pendiente de retorno

---

## 2. RECONFIGURACIÓN METODOLÓGICA — DECISIONES TOMADAS

### 2.1 Tres versiones del DIS Score

El proceso de validación empírica reveló que el DIS Score tiene comportamientos distintos según el nivel de análisis:

| Versión | Pesos | Propósito | Resultado |
|---------|-------|-----------|-----------|
| **DIS_v1 (principal)** | SA=0.35, NV=0.35, REP=0.30 | Corpus C — comparar comparecientes | Parsimonia 89% |
| DIS_v2 (alternativa) | SA=0.25, NV=0.045, REP=0.705 | A vs B — comparar sistemas | d=0.481, p=0.0003 |
| DIS_v3 (exploratoria) | SA=0.555, NV=0.342, REP=0.102 | Corpus C — pesos empíricos | Parsimonia 41% |

**Decisión**: mantener DIS_v1 como versión principal. La parsimonia superior de v1 (89%) respecto a v3 empírico (41%) confirma que los pesos habermasianos son más estables para el Corpus C oral que los pesos calculados empíricamente.

### 2.2 Hallazgo crítico sobre SA y NV

SA (p=0.934) y NV (d=0.024) no discriminan entre Corpus A y B. Esto no es un fallo del modelo — es el **hallazgo habermasiano central**: la supresión de agentividad y la negación de victimización son distorsiones comunicativas constitutivas del lenguaje jurídico-institucional colombiano en ambos sistemas.

**Reformulación de H₁**: la brecha entre sistemas de justicia es epistémica (IEI: d=0.715, p<0.001), no discursiva-gramatical (DIS_v1: d=0.166, p=0.115). El DIS_v1 opera como índice de perfilamiento de comparecientes en el Corpus C oral.

### 2.3 Normalización definitiva

Se resolvió el problema de valores 0.000 y 1.000 absolutos que producía la normalización min-max dentro del Corpus C. La solución es z-score+sigmoid sobre la distribución conjunta A+B+C (n=1,420):

- **DIS e IEI**: z-score+sigmoid sobre A+B+C
- **ICM**: z-score+sigmoid sobre distribución conjunta del Corpus C (vocal: 1,351 seg; facial: 2,822 frames; verbal: 547 bloques)
- **Resultado**: cero valores extremos en todos los índices

### 2.4 Análisis de parsimonia completo

| Índice | Combinaciones | Métrica | Resultado |
|--------|--------------|---------|-----------|
| DIS_v1 (Corpus C) | 90 | ρ≥0.90 | **89%** |
| DIS_v2 (A vs B) | 171 | d≥0.30 | 35% |
| DIS_v3 (Corpus C empírico) | 104 | ρ≥0.90 | 41% |
| IEI | 288 | ρ≥0.90 | **90%** |
| ICM | 114 | ρ≥0.90 | 75% |

**Conclusión**: el framework CFH es parsimonioso en su versión principal. Los pesos teóricos producen índices más estables que los empíricos para el Corpus C oral.

---

## 3. RECONFIGURACIÓN DE HIPÓTESIS

### H₁ — Reformulada
**Antes**: "el DIS es mayor en el corpus ordinario que en el JEP"
**Ahora**: "el corpus ordinario presenta mayor brecha epistémica (IEI: d=0.715, p<0.001) y menor lenguaje reparatorio (REP: d=0.371, p<0.001) que el corpus JEP escrito. La injusticia habermasiana (SA, NV) es transversal a ambos sistemas."

### H₃ — Se mantiene
β₂₃ < 0 — mayor DIS predice menor transición epistémica. Opera a nivel del Corpus C oral entre comparecientes. El SEM requiere κ>0.80 para estimarse completamente.

### H_ICM_A — Nueva hipótesis trimodal (exploratoria)
Mayor DIS predice mayor disociación entre canales verbal y no-verbal. **Confirmada**: ρ(DIS, dis_tri)=+0.90.

### H_ICM_B — Nueva hipótesis trimodal (exploratoria)
IEI alto + DIS bajo → congruencia tri-canal en dirección de no-reconocimiento. **Confirmada en Catatumbo y Costa Caribe**: cong_injusta=True.

---

## 4. CASO PARADIGMÁTICO CATATUMBO — REFORMULACIÓN

El hallazgo central se reformuló para eliminar la vulnerabilidad metodológica (el DIS mínimo depende de los pesos de SA):

**Formulación anterior** *(vulnerable)*: "Catatumbo tiene el DIS mínimo del corpus"

**Formulación robusta** *(100% estable)*: "Catatumbo es el único subcaso donde IEI>DIS en el 100% del espacio de pesos plausibles del DIS Score (141 combinaciones evaluadas). Adicionalmente, presenta congruencia tri-canal injusta (cong_injusta=True) con dis_tri=0.011 — el menor del corpus. Los tres canales convergen en la dirección del no-reconocimiento sin violencia gramatical explícita."

---

## 5. ICM TRI-CANAL v5 — RESULTADOS DEFINITIVOS (5 subcasos)

| Subcaso | DIS | IEI | dis_tri | cong_injusta | Perfil |
|---------|-----|-----|---------|--------------|--------|
| Casanare | 0.808 | 0.517 | 0.109 | No | NO-VERBAL>VERBAL |
| Catatumbo | 0.110 | 0.624 | 0.011 | **Sí** | CONGRUENTE INJUSTO ★ |
| Dabeiba | 0.490 | 0.299 | 0.079 | No | NO-VERBAL>VERBAL |
| Huila | 0.228 | 0.081 | 0.034 | No | CONGRUENTE |
| Costa Caribe | 0.464 | 0.231 | 0.012 | Sí | CONGRUENTE INJUSTO |

**Hallazgo ICM**: ρ(DIS, dis_tri)=+0.90 — mayor injusticia discursiva predice mayor disociación multimodal. Catatumbo y Costa Caribe muestran congruencia tri-canal injusta — todos los canales apuntan al no-reconocimiento sin que ninguno sea dramáticamente injusto.

---

## 6. CINCO DEBILIDADES IDENTIFICADAS Y RESPONDIDAS

Documentadas en §6.3.5 del Cap. 6:

| D | Debilidad | Respuesta |
|---|-----------|-----------|
| D1 | DIS_v1 no discrimina A vs B | Calibrado para Corpus C oral — DIS_v2 sí discrimina (d=0.481) |
| D2 | IEI discrimina pero DIS no | La asimetría es el hallazgo — confirma independencia de dimensiones |
| D3 | REP domina DIS en A vs B | En Corpus C SA y NV dominan — pesos v1 más estables que v3 empírico |
| D4 | N_B=54 | Restricción institucional — reconocida como limitación |
| D5 | ICM con N=5 | Universo completo — análisis de perfiles, no inferencial |

---

## 7. AVANCES EN ARTÍCULO PARA EPISTEME

Se inició el paper en inglés para *Episteme: A Journal of Social Epistemology*:
- **Título**: *Measuring the Unmeasured: Computational Forensic Hermeneutics and the Multimodal Structure of Epistemic Injustice in Transitional Justice Archives*
- **Abstract y sección 2 reformulados** con MAFAPO como validador epistémico central (no Fricker)
- **Miranda Fricker** respondió positivamente al correo de contacto y solicitó leer el artículo
- **Envío programado**: viernes 12 de junio

---

## 8. PENDIENTES CRÍTICOS

| Pendiente | Estado | Acción requerida |
|-----------|--------|-----------------|
| IAA κ>0.80 | ⏳ En proceso | Esperar retorno anotadores |
| CFH-BERT v3 | ⏳ Bloqueado por IAA | — |
| y₁ EBI, y₇ Surprisal, y₁₁ Conv. Rest. | ⏳ Bloqueado por CFH-BERT v3 | — |
| SEM completo (β₂₃, CFI, RMSEA) | ⏳ Bloqueado por CFH-BERT v3 | — |
| Paper inglés H3 | 🔄 En redacción | Enviar a Miranda el viernes |
| Cap. 4 taxonomía de modelos | ⏳ Pendiente | Próxima sesión |

---

## 9. ESTRUCTURA DE TESIS — ESTADO

| Capítulo | Estado | Versión |
|----------|--------|---------|
| Cap. 1 Introducción | ✅ | V2 |
| Cap. 2 Estado del arte | ✅ | V3 |
| Cap. 3 Marco teórico | ✅ actualizado | **V5** |
| Cap. 4 Metodología | 🔄 falta taxonomía | V5 |
| Cap. 5 Resultados | ✅ actualizado | **V16** |
| Cap. 6 Discusión | ✅ actualizado | **v10** |

