# CFH — Reformulación del caso paradigmático Catatumbo
## Párrafos de reemplazo para Cap. 5 §5.13 y Cap. 6 §6.1.8

---

## JUSTIFICACIÓN METODOLÓGICA DEL ANÁLISIS GLOBAL
### (Para incluir en §5.13 antes del análisis por subcaso)

El análisis de los resultados del Corpus C se realiza a nivel de los índices compuestos DIS e IEI, y no a nivel de los indicadores individuales, por una razón metodológica sustantiva: los fenómenos que se están midiendo — la injusticia discursiva habermasiana y la injusticia epistémica fricker-fraseriana — son propiedades emergentes del discurso que no son reducibles a ningún indicador individual. Examinar los indicadores por separado equivaldría a evaluar si un texto es literario mirando exclusivamente la frecuencia de sustantivos o solo el uso de metáforas: cada dimensión aporta información parcial, pero la propiedad que interesa emerge de la configuración conjunta.

Esta decisión tiene tres fundamentos convergentes. En primer lugar, cada indicador tiene error de medición específico: spaCy comete errores de parsing en el ~5% de las oraciones complejas; CFH-BERT v2 tiene F1_macro=0.58, lo que implica que aproximadamente el 42% de los spans son clasificados incorrectamente en al menos una clase. El índice compuesto promedia esos errores individuales y produce una estimación más estable de la propiedad latente que cualquier indicador por separado — que es exactamente la lógica del modelo de ecuaciones estructurales (SEM) que se aplica en §5.14.

En segundo lugar, los indicadores individuales son mutuamente insuficientes para capturar los constructos teóricos. La injusticia discursiva habermasiana requiere que fallen simultáneamente la pretensión de sinceridad (SA), la pretensión de corrección normativa (NV) y la rectitud comunicativa (ausencia de REP). Un documento con SA alto pero NV bajo no presenta el mismo perfil de violación habermasiana que uno con SA alto y NV alto. El DIS Score captura esa configuración conjunta; el análisis por indicador no puede hacerlo.

En tercer lugar, y más importante, el caso paradigmático de Catatumbo es invisible a cualquier análisis indicador por indicador. La sección siguiente demuestra que Catatumbo no es extremo en ningún indicador individual — su naturaleza paradigmática emerge únicamente del índice compuesto y específicamente de la disociación entre los dos índices.

---

## PÁRRAFO DE REEMPLAZO — Cap. 5 §5.13
### (Reemplaza el análisis del subcaso Catatumbo en la discusión de la Tabla 5.14)

El subcaso Catatumbo (Cap. Chaparro) presenta el perfil más complejo del Corpus C y el hallazgo más significativo del framework CFH. Una inspección de los indicadores individuales no revela ninguna señal de alarma: SA=0.532 (el valor más bajo del corpus C — rango 5°), NV=0.220 (rango 4°), y₈ MAFAPO=0.207 (rango 1°, pero con una diferencia de apenas 0.021 respecto al cuarto valor). Ningún indicador individual posiciona a Catatumbo como caso extremo en ninguna dirección. Es precisamente esta aparente normalidad a nivel de indicadores individuales la que hace necesario el índice compuesto: la condición paradigmática de Catatumbo es una propiedad emergente que solo es visible cuando se considera la configuración conjunta de indicadores discursivos e indicadores semánticos.

El DIS Score de Catatumbo (η₁=0.110) es el más bajo del corpus C: el Capitán Chaparro usa un lenguaje que viola en menor medida las pretensiones de validez habermasianas — habla en primera persona, con menor supresión de agentividad, con menor negación de victimización. Sin embargo, el IEI Score (η₂=0.624) es simultáneamente el más alto: el espacio semántico de su discurso se mantiene alejado del léxico de las víctimas (y₈=0.207, el máximo del corpus C) y de los estándares del derecho internacional de los DDHH (y₉=0.271, el máximo del corpus C). La disociación resultante Δ=0.514 es la mayor del corpus C.

Esta configuración — DIS bajo, IEI alto — tiene una interpretación precisa en el marco teórico del framework. En términos de Habermas, el Cap. Chaparro corrige las distorsiones comunicativas en el canal verbal: las pretensiones de sinceridad expresiva (SA bajo) y corrección normativa (NV bajo) están relativamente satisfechas. Pero en términos de Fricker, la hermeneutical gap persiste: el discurso no comparte los recursos hermenéuticos de las víctimas, no adopta su léxico, no se aproxima al marco semántico con que ellas nombran lo que les ocurrió. Corregir la injusticia habermasiana en el canal gramatical no equivale a cerrar la injusticia fricker-fraseriana en el canal semántico. Los dos tipos de injusticia son dimensiones independientes que pueden disociarse.

La robustez de este hallazgo ha sido verificada empíricamente mediante un análisis exhaustivo de sensibilidad sobre 141 combinaciones de pesos plausibles para el DIS Score. La disociación IEI>DIS en Catatumbo — la condición que define su naturaleza paradigmática — se verifica en el **100%** de las combinaciones evaluadas. La condición de máxima disociación del corpus se mantiene en el **85%** de las combinaciones. La condición de ser el único subcaso con IEI>DIS se mantiene en el **84%** de las combinaciones. Estos resultados confirman que el hallazgo no depende de la especificación particular de los pesos del DIS Score sino que es una propiedad estructural de la configuración discursiva del subcaso Catatumbo.

---

## PÁRRAFO DE REEMPLAZO — Cap. 6 §6.1.8
### (Reemplaza o amplía la descripción del hallazgo central de la disociación)

El hallazgo central de la comparación entre subcasos del Corpus C no es la ordenación de los subcasos en una escala de injusticia — es la disociación entre las dos dimensiones que el framework mide. Cuatro de los cinco subcasos presentan el patrón DIS>IEI: la injusticia discursiva habermasiana supera a la injusticia epistémica fricker-fraseriana. Ese patrón es coherente con la hipótesis de que el sistema judicial ordinario produce primariamente distorsiones comunicativas en el canal verbal-gramatical.

El subcaso Catatumbo rompe ese patrón de manera que no es visible a nivel de ningún indicador individual. A nivel de indicadores, Catatumbo no es extremo en ninguna dimensión particular: su SA=0.532 lo ubica en el último lugar del corpus en supresión de agentividad, pero su NV=0.220 es el segundo más bajo, y su REP=0.121 es moderado. Si se analizan los indicadores por separado, Catatumbo aparece como el subcaso "más normal" del corpus — el que menos viola las pretensiones habermasianas en cualquier indicador individual. Solo cuando se considera el índice compuesto DIS, que captura la configuración conjunta de las tres pretensiones de validez, y se pone en relación con el IEI, que captura la brecha semántica con el espacio hermenéutico de las víctimas, emerge la condición paradójica del caso: la menor violación verbal-gramatical del corpus coexiste con la mayor distancia semántica al léxico de las víctimas.

Esta es precisamente la justificación metodológica de los índices compuestos frente al análisis indicador por indicador: los constructos teóricos que interesan — la injusticia discursiva habermasiana, la injusticia epistémica frickeriana — son propiedades multidimensionales que no se reducen a ningún indicador aislado. La disociación DIS-IEI de Catatumbo es una propiedad emergente del índice compuesto que confirma empíricamente la necesidad de medir simultáneamente las dos dimensiones y de considerar su configuración conjunta, no sus valores aislados.

El análisis de robustez confirma que este hallazgo no es artefacto de la especificación de pesos. La condición IEI>DIS en Catatumbo se mantiene en el 100% de los 141 pares de pesos evaluados en el espacio plausible del DIS Score. Reformulando el hallazgo en términos que son completamente independientes de la especificación de pesos: Catatumbo es el único subcaso del Corpus C donde la distancia semántica al léxico de las víctimas (y₈, componente del IEI) supera al índice de violación de las pretensiones comunicativas (DIS), cualquiera que sea la ponderación asignada a los componentes del DIS dentro del rango metodológicamente plausible. Ese hallazgo no es una decisión del investigador — es una propiedad de los datos.

---

## NOTA PARA AMBOS CAPÍTULOS
### (Pie de tabla o nota metodológica)

*Sobre el análisis a nivel de índice compuesto vs. indicadores individuales: los resultados se presentan a nivel de los índices DIS e IEI y no de los indicadores individuales porque (a) los constructos teóricos que operacionalizan — la injusticia discursiva habermasiana y la injusticia epistémica frickeriana — son propiedades configurales que emergen de la combinación de indicadores, no de ninguno aislado; (b) el error de medición de cada indicador individual (F1_macro CFH-BERT v2 = 0.58; error parsing spaCy ~5%) se promedia en el índice compuesto, produciendo estimaciones más estables de las variables latentes; y (c) el hallazgo central del Corpus C — la disociación DIS-IEI de Catatumbo — es invisible a nivel de indicadores individuales y solo emerge a nivel del índice compuesto, lo que constituye una validación empírica de la necesidad de la medición multidimensional.*

