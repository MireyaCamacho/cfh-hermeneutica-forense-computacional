# Protocolo de Dirección Técnica — CFH

Este directorio gestiona la comunicación entre el director técnico (Julián Zuluaga) y la tesista (Mireya Camacho Celis) usando el protocolo HANDOFF de Orbital Lab.

## Flujo de trabajo

```
1. Director define OBJETIVO.md → sprint con entregables claros
2. Tesista trabaja → implementa, experimenta, documenta
3. Tesista entrega ACTA_ENTREGA.md → resumen de lo logrado
4. Director revisa → feedback, ajustes, siguiente sprint
5. Acta se archiva en historial/ → trazabilidad completa
```

## Estructura

```
.orbital/
├── config.yaml          ← Metadata del proyecto
├── OBJETIVO.md          ← Sprint actual (qué hacer)
├── README.md            ← Este archivo
└── historial/           ← Actas archivadas
    └── ACTA_SPRINT_XX.md
```

## Template de entrega (ACTA_ENTREGA.md)

Cuando completes un sprint, crea `.orbital/ACTA_ENTREGA.md` con este formato:

```markdown
---
sprint: "Sprint N — Nombre"
fecha: "YYYY-MM-DD"
actual_hours: X
---

# Acta de Entrega — Sprint N

## Resumen ejecutivo
[2-3 oraciones de qué se logró]

## Entregables completados
- [x] Entregable 1 — descripción
- [x] Entregable 2 — descripción
- [ ] Entregable 3 — no completado, razón: ...

## Hallazgos
[Descubrimientos inesperados, resultados de experimentos]

## Decisiones técnicas tomadas
[Qué decisiones se tomaron y por qué]

## Problemas encontrados
[Bloqueos, limitaciones, cosas que no funcionaron]

## Próximos pasos sugeridos
[Qué debería seguir según la tesista]
```

## Convenciones

- **Experimentos:** Directorio `experiments/EXP-XXX_nombre/` con `FINDINGS.md`
- **Datos:** Todo inventariado en `data/README.md`
- **Decisiones:** Log en `tutoria/decisiones.md`

---
*Protocolo HANDOFF v1.0 — Orbital Lab / Universidad Externado de Colombia*
