# CFH - Paquete reproducible

**Hermeneutica Forense Computacional (CFH)** | Mireya Camacho Celis
Tesis - Universidad Externado de Colombia | Macrocaso 003 JEP (falsos positivos)

Reproduce los resultados cuantitativos centrales: indices DIS/IEI/ICM,
fiabilidad inter-anotador (IAA) y analisis estructural exploratorio.

## Entorno
- Python 3.11 (conda env `cfh`), Windows 11
- `pip install -r requirements.txt`
- ConfliBERT-Spanish-Beto-Cased-v1 se descarga de HuggingFace en la 1a corrida
  (solo lo necesitan los scripts que recalculan embeddings; los outputs ya
  vienen calculados).

## Estructura
- `scripts/`            scripts de analisis
- `data/features/`      indicadores base
- `data/referencias/`   centroides, anotacion IAA, indicadores del SEM
- `outputs/`            resultados ya calculados (para verificar sin recomputar)

IMPORTANTE: correr los scripts DESDE la raiz de esta carpeta, p. ej.:
    python scripts\cfh_auditoria_profunda.py

## Orden de ejecucion
1. cfh_fortalecer_corpus_b.py       - regenera 80 secciones de Corpus B
2. cfh_completar_indicadores_B.py   - calcula los 6 indicadores de B
3. cfh_dis_iei_paso2.py             - ensambla A+B+C y calcula DIS/IEI (opcion A)
4. cfh_reporte_kappa_dos_rondas.py  - IAA dos rondas (kappa global 0.722)
5. cfh_sem_exploratorios.py         - SEM de C + MG-SEM (exploratorios)
6. cfh_sem_c_tres_indices.py        - integra DIS/IEI/ICM en C
7. cfh_sem_indicadores_individuales.py - regresion por indicadores (y8->y10)
8. cfh_jackknife_dabeiba.py         - robustez del subcaso Dabeiba
9. cfh_auditoria_profunda.py        - valida que todo reproduce (33 checks)

## Resultados clave (reproducibles)
- DIS por corpus: A=0.510, B=0.498, C=0.394
- IEI por corpus: A=0.513, B=0.353, C=0.370  (IEI separa ordinaria vs JEP)
- Correlaciones DIS/IEI/ICM: |r|<0.33 (tres dimensiones distintas)
- IAA global: 0.722 (sustancial); REP tras revision: 0.841
- Hallazgo: y8 (distancia a victimas) predice negativamente y10 (transicion):
  beta=-0.68 (C), beta=-0.60 (tri-corpus)

## Verificacion
Correr `python scripts\cfh_auditoria_profunda.py`: debe dar 33 OK, 0 ERROR.

## Notas metodologicas
- DIS, IEI e ICM son indices DESCRIPTIVO-DIMENSIONALES (3 dimensiones distintas),
  no discriminadores. Pesos teoricos (Galtung/Fricker), no optimizados.
- Los SEM (pasos 5-8) son EXPLORATORIOS (n=47 en Corpus C). La evidencia
  principal es el contraste inter-corpus del IEI y la relacion y8->y10.
